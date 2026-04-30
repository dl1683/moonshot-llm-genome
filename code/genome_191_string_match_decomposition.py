"""
genome_191_string_match_decomposition.py

Decomposes the g188 direct_string_match signal (+0.478 nats, 93% of g181b)
into content vs format components.

7 arms: scratch_ce, direct_init_only, direct_anchor_only, matched_rows_only,
unmatched_rows_only, row_shuffled_matched, frequency_bucket_shuffle.
All GPT-2-tokenizer 8-layer Qwen3-arch, 3 seeds, 5000 steps.

Codex design gate: codex_outputs/g191_string_match_decomposition_design_gate_20260430.md
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

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import genome_165_annealed_donor as g165
import genome_167_kd_canonical as g167
import genome_188_tokenizer_flow_bridge as g188

OUT_PATH = ROOT / "results" / "genome_191_string_match_decomposition.json"

SEEDS = [42, 7, 13]
TRAIN_STEPS = 5000
ANCHOR_LAMBDA = 0.01
LOG_EVERY = 100
EVAL_EVERY = 500
DEVICE = g165.DEVICE

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

ARMS = [
    "scratch_ce",
    "direct_init_only",
    "direct_anchor_only",
    "matched_rows_only",
    "unmatched_rows_only",
    "row_shuffled_matched",
    "frequency_bucket_shuffle",
]


def print_flush(msg: str) -> None:
    print(msg, flush=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------- Embedding construction ----------

def build_string_match_with_mask(
    tok_src, tok_tgt, src_embeddings: np.ndarray, tgt_vocab_size: int, tgt_embed_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Like g188's direct_string_match_embeddings but also returns matched_mask."""
    src_vocab = tok_src.get_vocab()
    tgt_vocab = tok_tgt.get_vocab()

    result = np.zeros((tgt_vocab_size, tgt_embed_dim), dtype=np.float32)
    matched = np.zeros(tgt_vocab_size, dtype=bool)

    for token_str, tgt_id in tgt_vocab.items():
        if tgt_id >= tgt_vocab_size:
            continue
        if token_str in src_vocab:
            src_id = src_vocab[token_str]
            if src_id < src_embeddings.shape[0]:
                result[tgt_id] = src_embeddings[src_id]
                matched[tgt_id] = True

    if matched.any():
        mean_emb = result[matched].mean(axis=0)
        result[~matched] = mean_emb

    n_matched = int(matched.sum())
    print_flush(f"  String match: {n_matched}/{tgt_vocab_size} tokens matched ({100*n_matched/tgt_vocab_size:.1f}%)")
    return result, matched


def build_matched_rows_only(full_embed: np.ndarray, matched_mask: np.ndarray) -> np.ndarray:
    """Keep matched rows from full_embed; zero-fill unmatched."""
    out = np.zeros_like(full_embed)
    out[matched_mask] = full_embed[matched_mask]
    return out


def build_unmatched_rows_only(full_embed: np.ndarray, matched_mask: np.ndarray) -> np.ndarray:
    """Keep unmatched rows (mean-filled in full_embed); zero matched."""
    out = np.zeros_like(full_embed)
    out[~matched_mask] = full_embed[~matched_mask]
    return out


def build_row_shuffled_matched(full_embed: np.ndarray, matched_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute rows among matched tokens only, preserving norms/spectrum."""
    out = full_embed.copy()
    matched_ids = np.where(matched_mask)[0]
    matched_rows = out[matched_ids].copy()
    perm = rng.permutation(len(matched_ids))
    out[matched_ids] = matched_rows[perm]
    return out


def build_frequency_bucket_shuffle(
    full_embed: np.ndarray, matched_mask: np.ndarray, tok_tgt, train_ids: torch.Tensor, rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle matched rows within frequency quintiles."""
    token_counts = Counter(train_ids.reshape(-1).tolist())
    matched_ids = np.where(matched_mask)[0]
    freqs = np.array([token_counts.get(int(tid), 0) for tid in matched_ids])
    quintiles = np.quantile(freqs, [0.2, 0.4, 0.6, 0.8])

    out = full_embed.copy()
    for lo, hi in zip(
        [0.0] + quintiles.tolist(),
        quintiles.tolist() + [float("inf")],
    ):
        bucket = matched_ids[(freqs >= lo) & (freqs < hi)]
        if len(bucket) > 1:
            rows = out[bucket].copy()
            perm = rng.permutation(len(bucket))
            out[bucket] = rows[perm]
    return out


# ---------- Training with row-wise anchor masking ----------

def train_cell(
    arm_label: str,
    seed: int,
    tok_gpt2,
    custom_embed: np.ndarray | None,
    anchor_embed: np.ndarray | None,
    anchor_mask: np.ndarray | None,
    anchor_lambda: float,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
    *,
    n_steps: int = TRAIN_STEPS,
    custom_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Train one cell with optional row-wise anchor masking."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = g188.make_gpt2_qwen3_model(tok_gpt2, seed)

    if custom_embed is not None:
        emb_t = torch.from_numpy(custom_embed).to(
            model.model.embed_tokens.weight.device,
            dtype=model.model.embed_tokens.weight.dtype,
        )
        with torch.no_grad():
            if custom_mask is None:
                model.model.embed_tokens.weight.copy_(emb_t)
            else:
                mask_t = torch.from_numpy(custom_mask).to(emb_t.device)
                model.model.embed_tokens.weight[mask_t] = emb_t[mask_t]
            if hasattr(model, "lm_head") and not model.config.tie_word_embeddings:
                if custom_mask is None:
                    model.lm_head.weight.copy_(emb_t)
                else:
                    model.lm_head.weight[mask_t] = emb_t[mask_t]

    anchor_target = None
    row_mask_t = None
    actual_lambda = 0.0
    if anchor_embed is not None and anchor_lambda > 0.0:
        anchor_target = torch.from_numpy(anchor_embed).to(DEVICE, dtype=torch.float32)
        actual_lambda = anchor_lambda
        if anchor_mask is not None:
            row_mask_t = torch.from_numpy(anchor_mask.astype(np.float32)).to(DEVICE).unsqueeze(1)

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

        with torch.amp.autocast("cuda", dtype=g188.FORWARD_DTYPE):
            out = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_ids)
            loss = out.loss

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step} arm={arm_label} seed={seed}")

        optimizer.zero_grad()
        loss.backward()

        if anchor_target is not None and actual_lambda > 0.0:
            with torch.no_grad():
                coeff = 2.0 * actual_lambda
                param = model.model.embed_tokens.weight
                if param.grad is not None:
                    grad_add = (param.detach().to(anchor_target.dtype) - anchor_target) * coeff
                    if row_mask_t is not None:
                        grad_add = grad_add * row_mask_t
                    param.grad.add_(grad_add)

        torch.nn.utils.clip_grad_norm_(model.parameters(), g188.GRAD_CLIP)
        optimizer.step()

        if step % LOG_EVERY == 0:
            print_flush(f"    step {step}/{n_steps} loss={loss.item():.4f}")

        if step % EVAL_EVERY == 0 or step == n_steps:
            model.eval()
            with torch.no_grad():
                val_nll = g188._eval_nll(model, val_ids, val_mask)
            trajectory[str(step)] = float(val_nll)
            if step % EVAL_EVERY == 0:
                print_flush(f"    eval step={step} val_nll={val_nll:.4f}")
            model.train()

    model.eval()
    with torch.no_grad():
        final_nll = g188._eval_nll(model, val_ids, val_mask)

    result = {
        "arm_label": arm_label,
        "seed": seed,
        "anchor_lambda": actual_lambda,
        "has_row_mask": anchor_mask is not None,
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
    required = ["scratch_ce", "matched_rows_only", "row_shuffled_matched",
                 "frequency_bucket_shuffle", "direct_init_only", "direct_anchor_only"]
    if not all(arm in results and len(results[arm]) >= len(SEEDS) for arm in required):
        return {"status": "incomplete"}

    scratch_nlls = {str(s): float(results["scratch_ce"][str(s)]["final_val_nll"]) for s in SEEDS}

    def arm_gaps(arm_name):
        nlls = {str(s): float(results[arm_name][str(s)]["final_val_nll"]) for s in SEEDS}
        gaps = [scratch_nlls[str(s)] - nlls[str(s)] for s in SEEDS]
        return float(np.mean(gaps)), gaps, sum(1 for g in gaps if g > 0)

    matched_mean, matched_gaps, matched_pos = arm_gaps("matched_rows_only")
    unmatched_mean, _, _ = arm_gaps("unmatched_rows_only") if "unmatched_rows_only" in results else (0.0, [], 0)
    shuffled_mean, _, _ = arm_gaps("row_shuffled_matched")
    freq_mean, _, _ = arm_gaps("frequency_bucket_shuffle")
    init_only_mean, _, _ = arm_gaps("direct_init_only")
    anchor_only_mean, _, _ = arm_gaps("direct_anchor_only")

    content_pass = (
        matched_mean >= 0.35
        and matched_pos >= 3
        and shuffled_mean <= 0.10
        and freq_mean <= 0.10
        and (matched_mean - shuffled_mean) >= 0.25
    )
    format_pass = shuffled_mean >= 0.25

    if content_pass:
        verdict = "PASS_CONTENT"
    elif format_pass:
        verdict = "PASS_FORMAT"
    elif matched_mean >= 0.15:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        "status": "complete",
        "verdict": verdict,
        "matched_rows_mean_gain": matched_mean,
        "matched_rows_per_seed": matched_gaps,
        "unmatched_rows_mean_gain": unmatched_mean,
        "row_shuffled_mean_gain": shuffled_mean,
        "freq_bucket_mean_gain": freq_mean,
        "init_only_mean_gain": init_only_mean,
        "anchor_only_mean_gain": anchor_only_mean,
    }


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    smoke = args.smoke
    n_steps = 50 if smoke else TRAIN_STEPS
    seeds = [42] if smoke else SEEDS
    run_out_path = OUT_PATH.with_name(OUT_PATH.stem + "_smoke.json") if smoke else OUT_PATH

    print_flush(f"=== g191 String-Match Decomposition ===")
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

    print_flush("\n--- Building string-match embeddings ---")
    gpt2_vocab = len(tok_gpt2)
    embed_dim = trained_embed.shape[1]
    full_embed, matched_mask = build_string_match_with_mask(
        tok_qwen, tok_gpt2, trained_embed, gpt2_vocab, embed_dim,
    )
    full_embed = g188.normalize_to_fro_norm(full_embed, trained_fro)

    rng = np.random.default_rng(191)
    matched_only = build_matched_rows_only(full_embed, matched_mask)
    unmatched_only = build_unmatched_rows_only(full_embed, matched_mask)
    shuffled = build_row_shuffled_matched(full_embed, matched_mask, rng)
    shuffled = g188.normalize_to_fro_norm(shuffled, trained_fro)
    freq_shuf = build_frequency_bucket_shuffle(full_embed, matched_mask, tok_gpt2, train_ids, rng)
    freq_shuf = g188.normalize_to_fro_norm(freq_shuf, trained_fro)

    n_matched = int(matched_mask.sum())
    n_unmatched = int((~matched_mask).sum())
    print_flush(f"  Matched: {n_matched}, Unmatched: {n_unmatched}")

    arm_configs = {
        "scratch_ce":              {"custom_embed": None,          "anchor_embed": None,          "anchor_mask": None},
        "direct_init_only":        {"custom_embed": full_embed,    "anchor_embed": None,          "anchor_mask": None},
        "direct_anchor_only":      {"custom_embed": None,          "anchor_embed": full_embed,    "anchor_mask": None},
        "matched_rows_only":       {"custom_embed": full_embed,    "anchor_embed": full_embed,    "anchor_mask": matched_mask, "custom_mask": matched_mask},
        "unmatched_rows_only":     {"custom_embed": None,          "anchor_embed": unmatched_only, "anchor_mask": ~matched_mask},
        "row_shuffled_matched":    {"custom_embed": shuffled,      "anchor_embed": shuffled,      "anchor_mask": matched_mask, "custom_mask": matched_mask},
        "frequency_bucket_shuffle":{"custom_embed": freq_shuf,     "anchor_embed": freq_shuf,     "anchor_mask": matched_mask, "custom_mask": matched_mask},
    }

    if not args.no_resume and run_out_path.exists():
        payload = json.loads(run_out_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "genome": 191,
            "name": "string_match_decomposition",
            "timestamp_utc_started": now_utc(),
            "config": {
                "train_steps": n_steps,
                "seeds": seeds,
                "anchor_lambda": ANCHOR_LAMBDA,
                "n_matched": n_matched,
                "n_unmatched": n_unmatched,
                "trained_fro": trained_fro,
            },
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
            result = train_cell(
                arm_label=arm_label,
                seed=seed,
                tok_gpt2=tok_gpt2,
                custom_embed=cfg["custom_embed"],
                anchor_embed=cfg["anchor_embed"],
                anchor_mask=cfg["anchor_mask"],
                anchor_lambda=ANCHOR_LAMBDA,
                train_ids=train_ids,
                train_mask=train_mask,
                val_ids=val_ids,
                val_mask=val_mask,
                n_steps=n_steps,
                custom_mask=cfg.get("custom_mask"),
            )
            payload["results"][arm_label][key] = result
            save()
            print_flush(f"  {arm_label} seed={seed} nll={result['final_val_nll']:.4f} ({result['wallclock_s']:.0f}s)")

    summary = compute_verdict(payload)
    payload["summary"] = summary
    payload["verdict"] = summary.get("verdict", "INCOMPLETE")
    save()

    print_flush(f"\n*** g191 VERDICT: {summary.get('verdict', '?')} ***")
    for key in ["matched_rows_mean_gain", "unmatched_rows_mean_gain",
                "row_shuffled_mean_gain", "freq_bucket_mean_gain",
                "init_only_mean_gain", "anchor_only_mean_gain"]:
        if key in summary:
            print_flush(f"  {key}: {summary[key]:+.4f}")


if __name__ == "__main__":
    main()
