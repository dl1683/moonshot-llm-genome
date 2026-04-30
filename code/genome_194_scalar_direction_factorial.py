"""
genome_194_scalar_direction_factorial.py

Resolves A17 (SEV-10): scalar-vs-direction confound in g191 row-identity signal.
Decomposes e_t = r_t * u_t and tests which component carries the +0.465 nats benefit.

6 arms x 3 seeds = 18 cells.
Codex direction review: codex_outputs/heartbeats/cycle162_direction_review_20260430.md
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
import genome_191_string_match_decomposition as g191

OUT_PATH = ROOT / "results" / "genome_194_scalar_direction_factorial.json"

SEEDS = [42, 7, 13]
TRAIN_STEPS = 5000
ANCHOR_LAMBDA = 0.01
LOG_EVERY = 100
EVAL_EVERY = 500
DEVICE = g165.DEVICE
PERM_SEED = 194

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

ARMS = [
    "scratch_ce",
    "full_match",
    "correct_dir_shuffled_norm",
    "shuffled_dir_correct_norm",
    "random_dir_correct_norm",
    "correct_dir_uniform_norm",
]


def print_flush(msg: str) -> None:
    print(msg, flush=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------- Decomposition helpers ----------

def decompose_rows(embed: np.ndarray, matched_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose matched rows into norms (r_t) and unit directions (u_t)."""
    norms = np.linalg.norm(embed, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    unit_dirs = embed / norms
    return norms.squeeze(1), unit_dirs


def build_correct_dir_shuffled_norm(
    unit_dirs: np.ndarray, norms: np.ndarray, matched_mask: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Correct directions, shuffled norms among matched tokens."""
    out = np.zeros_like(unit_dirs)
    matched_ids = np.where(matched_mask)[0]
    perm = rng.permutation(len(matched_ids))
    shuffled_norms = norms[matched_ids][perm]
    out[matched_ids] = unit_dirs[matched_ids] * shuffled_norms[:, None]
    return out


def build_shuffled_dir_correct_norm(
    unit_dirs: np.ndarray, norms: np.ndarray, matched_mask: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Shuffled directions, correct norms per token."""
    out = np.zeros_like(unit_dirs)
    matched_ids = np.where(matched_mask)[0]
    perm = rng.permutation(len(matched_ids))
    shuffled_dirs = unit_dirs[matched_ids][perm]
    out[matched_ids] = shuffled_dirs * norms[matched_ids, None]
    return out


def build_random_dir_correct_norm(
    norms: np.ndarray, matched_mask: np.ndarray, embed_dim: int, rng: np.random.Generator,
) -> np.ndarray:
    """Random unit directions, correct norms."""
    out = np.zeros((len(matched_mask), embed_dim), dtype=np.float32)
    matched_ids = np.where(matched_mask)[0]
    random_vecs = rng.standard_normal((len(matched_ids), embed_dim)).astype(np.float32)
    random_norms = np.linalg.norm(random_vecs, axis=1, keepdims=True)
    random_norms = np.maximum(random_norms, 1e-8)
    random_unit = random_vecs / random_norms
    out[matched_ids] = random_unit * norms[matched_ids, None]
    return out


def build_correct_dir_uniform_norm(
    unit_dirs: np.ndarray, norms: np.ndarray, matched_mask: np.ndarray,
) -> np.ndarray:
    """Correct directions, uniform (mean) norm."""
    out = np.zeros_like(unit_dirs)
    matched_ids = np.where(matched_mask)[0]
    mean_norm = float(norms[matched_ids].mean())
    out[matched_ids] = unit_dirs[matched_ids] * mean_norm
    return out


# ---------- Main ----------

def compute_verdict(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", {})
    required = ["scratch_ce", "full_match", "correct_dir_shuffled_norm",
                 "shuffled_dir_correct_norm", "random_dir_correct_norm",
                 "correct_dir_uniform_norm"]
    if not all(arm in results and len(results[arm]) >= len(SEEDS) for arm in required):
        return {"status": "incomplete"}

    scratch_nlls = {str(s): float(results["scratch_ce"][str(s)]["final_val_nll"]) for s in SEEDS}

    def arm_stats(arm_name):
        nlls = {str(s): float(results[arm_name][str(s)]["final_val_nll"]) for s in SEEDS}
        gaps = [scratch_nlls[str(s)] - nlls[str(s)] for s in SEEDS]
        return float(np.mean(gaps)), gaps

    full_mean, full_gaps = arm_stats("full_match")
    cd_sn_mean, cd_sn_gaps = arm_stats("correct_dir_shuffled_norm")
    sd_cn_mean, sd_cn_gaps = arm_stats("shuffled_dir_correct_norm")
    rd_cn_mean, rd_cn_gaps = arm_stats("random_dir_correct_norm")
    cd_un_mean, cd_un_gaps = arm_stats("correct_dir_uniform_norm")

    direction_pass = (
        cd_sn_mean >= 0.30
        and (sd_cn_mean < 0.15 or rd_cn_mean < 0.15)
    )
    scalar_pass = (
        sd_cn_mean >= 0.30
        and cd_un_mean < 0.15
    )
    both_pass = (
        cd_sn_mean >= 0.20
        and sd_cn_mean >= 0.20
        and not (cd_sn_mean > 0.80 * full_mean and sd_cn_mean < 0.20 * full_mean)
        and not (sd_cn_mean > 0.80 * full_mean and cd_sn_mean < 0.20 * full_mean)
    )

    if direction_pass:
        verdict = "PASS_DIRECTION"
    elif scalar_pass:
        verdict = "PASS_SCALAR"
    elif both_pass:
        verdict = "PASS_BOTH"
    else:
        verdict = "FAIL"

    return {
        "status": "complete",
        "verdict": verdict,
        "full_match_mean_gain": full_mean,
        "full_match_per_seed": full_gaps,
        "correct_dir_shuffled_norm_mean_gain": cd_sn_mean,
        "correct_dir_shuffled_norm_per_seed": cd_sn_gaps,
        "shuffled_dir_correct_norm_mean_gain": sd_cn_mean,
        "shuffled_dir_correct_norm_per_seed": sd_cn_gaps,
        "random_dir_correct_norm_mean_gain": rd_cn_mean,
        "random_dir_correct_norm_per_seed": rd_cn_gaps,
        "correct_dir_uniform_norm_mean_gain": cd_un_mean,
        "correct_dir_uniform_norm_per_seed": cd_un_gaps,
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

    print_flush(f"=== g194 Scalar/Direction Factorial ===")
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

    print_flush("\n--- Building string-match base embeddings ---")
    gpt2_vocab = len(tok_gpt2)
    embed_dim = trained_embed.shape[1]
    full_embed, matched_mask = g191.build_string_match_with_mask(
        tok_qwen, tok_gpt2, trained_embed, gpt2_vocab, embed_dim,
    )
    full_embed = g188.normalize_to_fro_norm(full_embed, trained_fro)

    norms, unit_dirs = decompose_rows(full_embed, matched_mask)
    n_matched = int(matched_mask.sum())
    matched_norms = norms[matched_mask]
    print_flush(f"  Matched: {n_matched}, norm range: [{matched_norms.min():.4f}, {matched_norms.max():.4f}], mean={matched_norms.mean():.4f}")

    rng = np.random.default_rng(PERM_SEED)

    matched_only = g191.build_matched_rows_only(full_embed, matched_mask)

    cd_sn = build_correct_dir_shuffled_norm(unit_dirs, norms, matched_mask, rng)
    cd_sn = g188.normalize_to_fro_norm(cd_sn, trained_fro)

    rng2 = np.random.default_rng(PERM_SEED + 1)
    sd_cn = build_shuffled_dir_correct_norm(unit_dirs, norms, matched_mask, rng2)
    sd_cn = g188.normalize_to_fro_norm(sd_cn, trained_fro)

    rng3 = np.random.default_rng(PERM_SEED + 2)
    rd_cn = build_random_dir_correct_norm(norms, matched_mask, embed_dim, rng3)
    rd_cn = g188.normalize_to_fro_norm(rd_cn, trained_fro)

    cd_un = build_correct_dir_uniform_norm(unit_dirs, norms, matched_mask)
    cd_un = g188.normalize_to_fro_norm(cd_un, trained_fro)

    arm_configs = {
        "scratch_ce":                {"custom_embed": None,       "anchor_embed": None,       "anchor_mask": None},
        "full_match":                {"custom_embed": full_embed, "anchor_embed": full_embed, "anchor_mask": matched_mask, "custom_mask": matched_mask},
        "correct_dir_shuffled_norm": {"custom_embed": cd_sn,      "anchor_embed": cd_sn,      "anchor_mask": matched_mask, "custom_mask": matched_mask},
        "shuffled_dir_correct_norm": {"custom_embed": sd_cn,      "anchor_embed": sd_cn,      "anchor_mask": matched_mask, "custom_mask": matched_mask},
        "random_dir_correct_norm":   {"custom_embed": rd_cn,      "anchor_embed": rd_cn,      "anchor_mask": matched_mask, "custom_mask": matched_mask},
        "correct_dir_uniform_norm":  {"custom_embed": cd_un,      "anchor_embed": cd_un,      "anchor_mask": matched_mask, "custom_mask": matched_mask},
    }

    if not args.no_resume and run_out_path.exists():
        payload = json.loads(run_out_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "genome": 194,
            "name": "scalar_direction_factorial",
            "timestamp_utc_started": now_utc(),
            "config": {
                "train_steps": n_steps,
                "seeds": seeds,
                "anchor_lambda": ANCHOR_LAMBDA,
                "perm_seed": PERM_SEED,
                "n_matched": n_matched,
                "trained_fro": trained_fro,
                "matched_norm_mean": float(matched_norms.mean()),
                "matched_norm_std": float(matched_norms.std()),
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
            result = g191.train_cell(
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

    print_flush(f"\n*** g194 VERDICT: {summary.get('verdict', '?')} ***")
    for key, val in summary.items():
        if key.endswith("_mean_gain"):
            print_flush(f"  {key}: {val:+.4f}")


if __name__ == "__main__":
    main()
