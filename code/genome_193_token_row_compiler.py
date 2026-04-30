"""
genome_193_token_row_compiler.py

Train a small MLP to predict Qwen3 trained embedding rows from token-level
features (byte histogram, length, log-frequency). Supervision: 42k exact
string-matched GPT-2/Qwen3 token pairs. Evaluation: train GPT-2-tokenizer
Qwen3-arch shell using ONLY compiler-generated rows (no copied target rows).

Codex design gate: codex_outputs/g193_token_row_compiler_design_gate.md
Prereg: research/prereg/genome_193_token_row_compiler_2026-04-30.md
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import genome_167_kd_canonical as g167
import genome_188_tokenizer_flow_bridge as g188

OUT_PATH = ROOT / "results" / "genome_193_token_row_compiler.json"

SEEDS = [42, 7, 13]
TRAIN_STEPS = 5000
EMBED_DIM = 1024
HOLDOUT_FRAC = 0.20
COMPILER_EPOCHS = 200
COMPILER_LR = 1e-3
COMPILER_BATCH = 512
COMPILER_HIDDEN = 1024
ANCHOR_LAMBDA = 0.01
DEVICE = g188.DEVICE
FORWARD_DTYPE = g188.FORWARD_DTYPE


def print_flush(msg: str) -> None:
    print(msg, flush=True)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------- Feature extraction ----------

def byte_histogram(token_str: str) -> np.ndarray:
    """256-dim histogram of UTF-8 byte values in token string."""
    hist = np.zeros(256, dtype=np.float32)
    for b in token_str.encode("utf-8", errors="replace"):
        hist[b] += 1.0
    return hist


def build_features(tok, token_counts: dict[int, int], vocab_size: int) -> np.ndarray:
    """Build feature matrix for all tokens. Shape: (vocab_size, 258)."""
    vocab = tok.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    features = np.zeros((vocab_size, 258), dtype=np.float32)

    for tid in range(vocab_size):
        token_str = inv_vocab.get(tid, "")
        features[tid, :256] = byte_histogram(token_str)
        features[tid, 256] = len(token_str.encode("utf-8", errors="replace"))
        features[tid, 257] = np.log1p(token_counts.get(tid, 0))

    return features


# ---------- Compiler model ----------

class TokenRowCompiler(nn.Module):
    def __init__(self, input_dim: int = 258, hidden_dim: int = COMPILER_HIDDEN, output_dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_compiler(
    features: np.ndarray,
    targets: np.ndarray,
    train_ids: np.ndarray,
    holdout_ids: np.ndarray,
    seed: int,
    n_epochs: int = COMPILER_EPOCHS,
) -> tuple[TokenRowCompiler, dict]:
    """Train the compiler on train_ids, evaluate on holdout_ids."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TokenRowCompiler().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=COMPILER_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    feat_train = torch.from_numpy(features[train_ids]).to(DEVICE)
    tgt_train = torch.from_numpy(targets[train_ids]).to(DEVICE)
    feat_hold = torch.from_numpy(features[holdout_ids]).to(DEVICE)
    tgt_hold = torch.from_numpy(targets[holdout_ids]).to(DEVICE)

    n_train = len(train_ids)
    best_hold_loss = float("inf")
    patience = 0
    max_patience = 30

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, COMPILER_BATCH):
            idx = perm[i:i + COMPILER_BATCH]
            pred = model(feat_train[idx])
            loss = F.mse_loss(pred, tgt_train[idx])

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite compiler loss at epoch {epoch}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            hold_pred = model(feat_hold)
            hold_loss = F.mse_loss(hold_pred, tgt_hold).item()
            cos_sim = F.cosine_similarity(hold_pred, tgt_hold, dim=1).mean().item()

        if hold_loss < best_hold_loss:
            best_hold_loss = hold_loss
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        if (epoch + 1) % 20 == 0:
            print_flush(f"    compiler epoch={epoch+1} train_mse={epoch_loss/n_batches:.6f} "
                        f"hold_mse={hold_loss:.6f} hold_cos={cos_sim:.4f}")

        if patience >= max_patience:
            print_flush(f"    early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_hold_pred = model(feat_hold)
        final_cos = F.cosine_similarity(final_hold_pred, tgt_hold, dim=1).mean().item()
        final_mse = F.mse_loss(final_hold_pred, tgt_hold).item()

    stats = {
        "best_holdout_mse": best_hold_loss,
        "final_holdout_mse": final_mse,
        "final_holdout_cosine": final_cos,
        "epochs_trained": epoch + 1,
        "n_train": n_train,
        "n_holdout": len(holdout_ids),
    }
    return model, stats


def generate_all_rows(
    compiler: TokenRowCompiler,
    features: np.ndarray,
    target_fro: float,
) -> np.ndarray:
    """Generate embedding rows for ALL tokens using the compiler."""
    compiler.eval()
    feat_t = torch.from_numpy(features).to(DEVICE)
    with torch.no_grad():
        pred = compiler(feat_t).cpu().numpy()
    return g188.normalize_to_fro_norm(pred, target_fro)


# ---------- Shell training (reuses g188 infrastructure) ----------

def train_shell_cell(
    arm_label: str,
    seed: int,
    tok_gpt2,
    custom_embed: np.ndarray | None,
    anchor_embed: np.ndarray | None,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
    *,
    n_steps: int = TRAIN_STEPS,
) -> dict[str, Any]:
    """Train one GPT-2-tokenizer Qwen3-arch shell cell."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = g188.make_gpt2_qwen3_model(tok_gpt2, seed)

    if custom_embed is not None:
        emb_t = torch.from_numpy(custom_embed).to(
            model.model.embed_tokens.weight.device,
            dtype=model.model.embed_tokens.weight.dtype,
        )
        with torch.no_grad():
            model.model.embed_tokens.weight.copy_(emb_t)

    anchor_target = None
    if anchor_embed is not None and ANCHOR_LAMBDA > 0.0:
        anchor_target = torch.from_numpy(anchor_embed).to(DEVICE, dtype=torch.float32)

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=g188.LR, betas=g188.BETAS, weight_decay=g188.WEIGHT_DECAY,
    )

    n_train = train_ids.shape[0]
    trajectory = {}
    t0 = time.time()

    for step in range(1, n_steps + 1):
        idx = torch.randint(0, n_train, (g188.BATCH_SIZE,))
        batch_ids = train_ids[idx].to(DEVICE)
        batch_mask = train_mask[idx].to(DEVICE)

        with torch.amp.autocast("cuda", dtype=FORWARD_DTYPE):
            out = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_ids)
            loss = out.loss

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step} arm={arm_label} seed={seed}")

        optimizer.zero_grad()
        loss.backward()

        if anchor_target is not None:
            with torch.no_grad():
                coeff = 2.0 * ANCHOR_LAMBDA
                param = model.model.embed_tokens.weight
                if param.grad is not None:
                    param.grad.add_((param.detach().to(anchor_target.dtype) - anchor_target) * coeff)

        torch.nn.utils.clip_grad_norm_(model.parameters(), g188.GRAD_CLIP)
        optimizer.step()

        if step % g188.EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                val_nll = g188._eval_nll(model, val_ids, val_mask)
            trajectory[str(step)] = float(val_nll)
            print_flush(f"    eval step={step} val_nll={val_nll:.4f}")
            model.train()

    model.eval()
    with torch.no_grad():
        final_nll = g188._eval_nll(model, val_ids, val_mask)

    result = {
        "arm_label": arm_label,
        "seed": seed,
        "final_val_nll": float(final_nll),
        "trajectory": trajectory,
        "wallclock_s": time.time() - t0,
    }
    del model, optimizer
    cleanup_cuda()
    return result


# ---------- Verdict ----------

def compute_verdict(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", {})
    required = ["scratch_ce", "compiled_init_anchor", "compiled_shuffled"]
    if not all(arm in results and len(results[arm]) >= len(SEEDS) for arm in required):
        return {"status": "incomplete"}

    scratch_nlls = {str(s): float(results["scratch_ce"][str(s)]["final_val_nll"]) for s in SEEDS}

    def arm_gaps(arm_name):
        nlls = {str(s): float(results[arm_name][str(s)]["final_val_nll"]) for s in SEEDS}
        gaps = [scratch_nlls[str(s)] - nlls[str(s)] for s in SEEDS]
        return float(np.mean(gaps)), gaps, sum(1 for g in gaps if g > 0)

    compiled_mean, compiled_gaps, compiled_pos = arm_gaps("compiled_init_anchor")
    shuffled_mean, _, _ = arm_gaps("compiled_shuffled")
    init_only_mean, _, _ = arm_gaps("compiled_init_only") if "compiled_init_only" in results else (0.0, [], 0)

    content_pass = (
        compiled_mean >= 0.30
        and compiled_pos >= 3
        and shuffled_mean <= 0.10
        and (compiled_mean - shuffled_mean) >= 0.20
    )

    if content_pass:
        verdict = "PASS"
    elif compiled_mean >= 0.15:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        "status": "complete",
        "verdict": verdict,
        "compiled_mean_gain": compiled_mean,
        "compiled_per_seed": compiled_gaps,
        "shuffled_mean_gain": shuffled_mean,
        "init_only_mean_gain": init_only_mean,
    }


# ---------- Main ----------

ARMS = [
    "scratch_ce",
    "compiled_init_anchor",
    "compiled_init_only",
    "compiled_shuffled",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    smoke = args.smoke
    n_steps = 50 if smoke else TRAIN_STEPS
    seeds = [42] if smoke else SEEDS
    compiler_epochs = 5 if smoke else COMPILER_EPOCHS
    run_out_path = OUT_PATH.with_name(OUT_PATH.stem + "_smoke.json") if smoke else OUT_PATH

    print_flush(f"=== g193 Token-Row Compiler ===")
    print_flush(f"  smoke={smoke}, steps={n_steps}, seeds={seeds}")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok_qwen = AutoTokenizer.from_pretrained(g188.QWEN_MODEL_ID)
    tok_gpt2 = AutoTokenizer.from_pretrained(g188.GPT2_MODEL_ID)
    if tok_gpt2.pad_token is None:
        tok_gpt2.pad_token = tok_gpt2.eos_token

    print_flush("\n--- Loading data ---")
    train_ids, train_mask, _ = g167.load_c4_windows(
        tok_gpt2, split="train", seed=g188.C4_TRAIN_SEED, n_windows=g188.N_TRAIN_WINDOWS,
    )
    train_hashes = g167.collect_13gram_hashes(train_ids, train_mask)
    val_ids, val_mask, _ = g167.load_c4_windows(
        tok_gpt2, split="train", seed=g188.C4_VAL_SEED, n_windows=g188.N_C4_VAL_WINDOWS,
        forbidden_hashes=train_hashes,
    )
    print_flush(f"  Train: {train_ids.shape}, Val: {val_ids.shape}")

    print_flush("\n--- Loading Qwen3 trained embeddings ---")
    qwen_model = AutoModelForCausalLM.from_pretrained(g188.QWEN_MODEL_ID, torch_dtype=torch.float32)
    trained_embed = qwen_model.model.embed_tokens.weight.detach().cpu().numpy()
    trained_fro = float(np.linalg.norm(trained_embed, "fro"))
    del qwen_model
    cleanup_cuda()
    print_flush(f"  Trained embed: {trained_embed.shape}, Fro={trained_fro:.1f}")

    print_flush("\n--- Building string-match pairs ---")
    src_vocab = tok_qwen.get_vocab()
    tgt_vocab = tok_gpt2.get_vocab()
    gpt2_vocab = len(tok_gpt2)

    matched_pairs = []
    for token_str, tgt_id in tgt_vocab.items():
        if tgt_id >= gpt2_vocab:
            continue
        if token_str in src_vocab:
            src_id = src_vocab[token_str]
            if src_id < trained_embed.shape[0]:
                matched_pairs.append((tgt_id, src_id))

    matched_tgt_ids = np.array([p[0] for p in matched_pairs])
    target_rows = np.array([trained_embed[p[1]] for p in matched_pairs], dtype=np.float32)
    n_matched = len(matched_pairs)
    print_flush(f"  Matched pairs: {n_matched}")

    print_flush("\n--- Building token features ---")
    token_counts = Counter(train_ids.reshape(-1).tolist())
    features = build_features(tok_gpt2, token_counts, gpt2_vocab)
    print_flush(f"  Features: {features.shape}")

    print_flush("\n--- Train/holdout split ---")
    rng = np.random.default_rng(193)
    n_holdout = int(n_matched * HOLDOUT_FRAC)
    perm = rng.permutation(n_matched)
    holdout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]
    compiler_train_tgt_ids = matched_tgt_ids[train_idx]
    compiler_holdout_tgt_ids = matched_tgt_ids[holdout_idx]
    print_flush(f"  Compiler train: {len(train_idx)}, holdout: {len(holdout_idx)}")

    features_matched = features[matched_tgt_ids]
    targets_matched = target_rows

    print_flush("\n--- Training compiler ---")
    compiler, compiler_stats = train_compiler(
        features_matched, targets_matched, train_idx, holdout_idx, seed=193,
        n_epochs=compiler_epochs,
    )
    print_flush(f"  Compiler done: holdout_mse={compiler_stats['final_holdout_mse']:.6f}, "
                f"holdout_cosine={compiler_stats['final_holdout_cosine']:.4f}")

    print_flush("\n--- Generating full embedding table ---")
    compiled_embed = generate_all_rows(compiler, features, trained_fro)
    print_flush(f"  Generated: {compiled_embed.shape}, Fro={np.linalg.norm(compiled_embed):.1f}")

    compiled_shuffled = compiled_embed.copy()
    shuffle_perm = rng.permutation(gpt2_vocab)
    compiled_shuffled = compiled_shuffled[shuffle_perm]
    compiled_shuffled = g188.normalize_to_fro_norm(compiled_shuffled, trained_fro)

    del compiler
    cleanup_cuda()

    arm_configs = {
        "scratch_ce":            {"custom_embed": None,              "anchor_embed": None},
        "compiled_init_anchor":  {"custom_embed": compiled_embed,    "anchor_embed": compiled_embed},
        "compiled_init_only":    {"custom_embed": compiled_embed,    "anchor_embed": None},
        "compiled_shuffled":     {"custom_embed": compiled_shuffled, "anchor_embed": compiled_shuffled},
    }

    embed_hash = hashlib.sha256(compiled_embed.tobytes()).hexdigest()[:16]

    payload: dict[str, Any] | None = None
    can_resume = False
    if not args.no_resume and run_out_path.exists():
        payload = json.loads(run_out_path.read_text(encoding="utf-8"))
        old_hash = payload.get("config", {}).get("embed_hash", "")
        if old_hash == embed_hash:
            can_resume = True
            print_flush(f"  Resume: embed_hash matches ({embed_hash})")
        else:
            print_flush(f"  Resume: embed_hash MISMATCH ({old_hash} vs {embed_hash}), starting fresh")
            payload = None

    if payload is None or not can_resume:
        payload = {
            "genome": 193,
            "name": "token_row_compiler",
            "timestamp_utc_started": now_utc(),
            "config": {
                "train_steps": n_steps,
                "seeds": seeds,
                "anchor_lambda": ANCHOR_LAMBDA,
                "n_matched": n_matched,
                "holdout_frac": HOLDOUT_FRAC,
                "compiler_epochs": compiler_epochs,
                "compiler_hidden": COMPILER_HIDDEN,
                "trained_fro": trained_fro,
                "embed_hash": embed_hash,
            },
            "compiler_stats": compiler_stats,
            "results": {},
            "summary": {},
            "verdict": "INCOMPLETE",
        }

    t_start = time.time()

    def save():
        payload["timestamp_utc_last_write"] = now_utc()
        payload["elapsed_s"] = time.time() - t_start
        tmp = run_out_path.with_suffix(run_out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False), encoding="utf-8")
        os.replace(tmp, run_out_path)

    for arm_label in ARMS:
        payload["results"].setdefault(arm_label, {})
        cfg = arm_configs[arm_label]

        for seed in seeds:
            key = str(seed)
            if key in payload["results"][arm_label] and not args.no_resume:
                cell = payload["results"][arm_label][key]
                if isinstance(cell, dict) and "final_val_nll" in cell:
                    print_flush(f"\n  Skipping {arm_label}/seed={seed} (done)")
                    continue

            print_flush(f"\n  === {arm_label} seed={seed} ===")
            result = train_shell_cell(
                arm_label=arm_label,
                seed=seed,
                tok_gpt2=tok_gpt2,
                custom_embed=cfg["custom_embed"],
                anchor_embed=cfg["anchor_embed"],
                train_ids=train_ids,
                train_mask=train_mask,
                val_ids=val_ids,
                val_mask=val_mask,
                n_steps=n_steps,
            )
            payload["results"][arm_label][key] = result
            save()
            print_flush(f"  {arm_label} seed={seed} nll={result['final_val_nll']:.4f} ({result['wallclock_s']:.0f}s)")

    summary = compute_verdict(payload)
    payload["summary"] = summary
    payload["verdict"] = summary.get("verdict", "INCOMPLETE")
    save()

    print_flush(f"\n*** g193 VERDICT: {summary.get('verdict', '?')} ***")
    for key in ["compiled_mean_gain", "shuffled_mean_gain", "init_only_mean_gain"]:
        if key in summary:
            print_flush(f"  {key}: {summary[key]:+.4f}")


if __name__ == "__main__":
    main()
