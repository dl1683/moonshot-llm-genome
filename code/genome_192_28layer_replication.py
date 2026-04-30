"""
genome_192_28layer_replication.py

Tests whether the g191 matched_rows_only +0.465 nats signal persists
at full 28-layer depth (vs 8-layer shell). Resolves adversarial A16 #3.

3 arms x 3 seeds = 9 cells.
Gated on g194 PASS_DIRECTION.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
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
import genome_191_string_match_decomposition as g191

OUT_PATH = ROOT / "results" / "genome_192_28layer_replication.json"

SEEDS = [42, 7, 13]
TRAIN_STEPS = 5000
ANCHOR_LAMBDA = 0.01
LOG_EVERY = 100
EVAL_EVERY = 500
DEVICE = g165.DEVICE
NUM_LAYERS_28 = 28

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

ARMS = [
    "scratch_ce",
    "matched_rows_only",
    "row_shuffled",
]


def print_flush(msg: str) -> None:
    print(msg, flush=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_28layer_model(tok_gpt2, seed: int):
    from transformers import Qwen3ForCausalLM
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=len(tok_gpt2),
        hidden_size=1024,
        num_hidden_layers=NUM_LAYERS_28,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=2816,
        max_position_embeddings=g188.SEQ_LEN + 64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        head_dim=64,
        rope_theta=10000.0,
    )
    model = Qwen3ForCausalLM(cfg).to(dtype=g188.FORWARD_DTYPE, device=DEVICE)
    model.config.pad_token_id = tok_gpt2.pad_token_id
    return model


def train_cell_28layer(
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
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = make_28layer_model(tok_gpt2, seed)

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
        "num_layers": NUM_LAYERS_28,
        "final_val_nll": float(final_nll),
        "trajectory": trajectory,
        "wallclock_s": time.time() - t0,
    }
    del model, optimizer
    cleanup_cuda()
    return result


def compute_verdict(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", {})
    required = ["scratch_ce", "matched_rows_only", "row_shuffled"]
    if not all(arm in results and len(results[arm]) >= len(SEEDS) for arm in required):
        return {"status": "incomplete"}

    scratch_nlls = {str(s): float(results["scratch_ce"][str(s)]["final_val_nll"]) for s in SEEDS}

    def arm_stats(arm_name):
        nlls = {str(s): float(results[arm_name][str(s)]["final_val_nll"]) for s in SEEDS}
        gaps = [scratch_nlls[str(s)] - nlls[str(s)] for s in SEEDS]
        return float(np.mean(gaps)), gaps

    matched_mean, matched_gaps = arm_stats("matched_rows_only")
    shuffled_mean, shuffled_gaps = arm_stats("row_shuffled")

    matched_all_positive = all(g > 0 for g in matched_gaps)
    shuffled_harmful = shuffled_mean <= 0.0

    if matched_mean >= 0.20 and matched_all_positive and shuffled_harmful:
        verdict = "PASS_PERSISTENCE"
    elif matched_mean >= 0.10 and matched_all_positive:
        verdict = "PASS_ATTENUATION"
    else:
        verdict = "FAIL"

    return {
        "status": "complete",
        "verdict": verdict,
        "matched_mean_gain": matched_mean,
        "matched_per_seed": matched_gaps,
        "shuffled_mean_gain": shuffled_mean,
        "shuffled_per_seed": shuffled_gaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    smoke = args.smoke
    n_steps = 50 if smoke else TRAIN_STEPS
    seeds = [42] if smoke else SEEDS
    run_out_path = OUT_PATH.with_name(OUT_PATH.stem + "_smoke.json") if smoke else OUT_PATH

    print_flush(f"=== g192 28-Layer String-Match Replication ===")
    print_flush(f"  smoke={smoke}, steps={n_steps}, seeds={seeds}, layers={NUM_LAYERS_28}")

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
    full_embed, matched_mask = g191.build_string_match_with_mask(
        tok_qwen, tok_gpt2, trained_embed, gpt2_vocab, embed_dim,
    )
    full_embed = g188.normalize_to_fro_norm(full_embed, trained_fro)

    rng = np.random.default_rng(192)
    shuffled = g191.build_row_shuffled_matched(full_embed, matched_mask, rng)
    shuffled = g188.normalize_to_fro_norm(shuffled, trained_fro)

    n_matched = int(matched_mask.sum())
    print_flush(f"  Matched: {n_matched}")

    arm_configs = {
        "scratch_ce":       {"custom_embed": None,       "anchor_embed": None,       "anchor_mask": None},
        "matched_rows_only":{"custom_embed": full_embed,  "anchor_embed": full_embed,  "anchor_mask": matched_mask, "custom_mask": matched_mask},
        "row_shuffled":     {"custom_embed": shuffled,    "anchor_embed": shuffled,    "anchor_mask": matched_mask, "custom_mask": matched_mask},
    }

    if not args.no_resume and run_out_path.exists():
        payload = json.loads(run_out_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "genome": 192,
            "name": "28layer_replication",
            "timestamp_utc_started": now_utc(),
            "config": {
                "train_steps": n_steps,
                "seeds": seeds,
                "anchor_lambda": ANCHOR_LAMBDA,
                "num_layers": NUM_LAYERS_28,
                "n_matched": n_matched,
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
            result = train_cell_28layer(
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

    print_flush(f"\n*** g192 VERDICT: {summary.get('verdict', '?')} ***")
    for key, val in summary.items():
        if key.endswith("_mean_gain"):
            print_flush(f"  {key}: {val:+.4f}")


if __name__ == "__main__":
    main()
