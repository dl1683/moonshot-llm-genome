"""
genome_189_c23_content_causality.py

Cycle 150 adversarial SEV-10: C23 content-transfer claim (+0.513 nats) may be
FORMAT (norm/spectrum/frequency structure) rather than CONTENT (actual learned
token relationships). This experiment resolves it with 5 matched controls.

Arms (7 total, Qwen3-0.6B, same tokenizer, 6 seeds, 5000 steps):
  1. scratch_ce — no anchor baseline
  2. true_trained_anchor — real Qwen3 trained embed/lm_head
  3. row_shuffled_anchor — globally permuted rows (destroys token identity)
  4. freq_bucket_shuffle_anchor — rows shuffled within frequency quantiles
  5. spectrum_preserving_random — random orthogonal x trained SVD spectrum
  6. same_frobenius_gaussian — iid Gaussian, same Frobenius norm
  7. anchor_to_initial — anchor toward independent random init

PASS_CONTENT: true_trained beats EVERY control by >=0.20 nats AND 5/6 seeds.
FORMAT_FAIL: any control within 0.20 nats of true or beats true in >=2/6 seeds.

Codex design gate: codex_outputs/g189_content_causality_design_gate_20260430.md
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import genome_165_annealed_donor as g165
import genome_167_kd_canonical as g167
import genome_181a_tokenizer_isolation as g181a

OUT_PATH = ROOT / "results" / "genome_189_c23_content_causality.json"

SEEDS = [42, 7, 13, 101, 202, 303]
SEQ_LEN = g165.SEQ_LEN
BATCH_SIZE = g165.BATCH_SIZE
EVAL_BATCH_SIZE = g165.BATCH_SIZE
TRAIN_STEPS = 5000
N_TRAIN_WINDOWS = 16384
N_C4_VAL_WINDOWS = 256
LR = g165.LR
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
LOG_EVERY = 100
EVAL_EVERY = 500

C4_TRAIN_SEED = g165.C4_TRAIN_SEED
C4_VAL_SEED = g165.C4_VAL_SEED

ANCHOR_LAMBDA_BASE = 0.01
N_BOOT = 10_000
N_FREQ_BUCKETS = 20

PASS_GAP_NATS = 0.20
PASS_SEED_FRACTION = 5  # out of 6

DEVICE = g165.DEVICE
FORWARD_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


ALL_ARMS = [
    "scratch_ce",
    "true_trained_anchor",
    "row_shuffled_anchor",
    "freq_bucket_shuffle_anchor",
    "spectrum_preserving_random",
    "same_frobenius_gaussian",
    "anchor_to_initial",
]


def print_flush(msg: str) -> None:
    print(msg, flush=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def autocast_context():
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


# ---------- Target construction ----------

def get_trained_embed(tok) -> np.ndarray:
    trained_model, _ = g165.load_trained_donor(tok)
    embed = trained_model.model.embed_tokens.weight.detach().float().cpu().numpy()
    del trained_model
    cleanup_cuda()
    return embed


def build_row_shuffled(trained_embed: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    perm = rng.permutation(trained_embed.shape[0])
    return trained_embed[perm].copy()


def build_freq_bucket_shuffled(
    trained_embed: np.ndarray, tok, rng: np.random.Generator,
) -> np.ndarray:
    from collections import Counter
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=True)
    counter: Counter = Counter()
    n_docs = 0
    for doc in ds:
        ids = tok(doc["text"], truncation=True, max_length=512, add_special_tokens=False)["input_ids"]
        counter.update(ids)
        n_docs += 1
        if n_docs >= 5000:
            break

    vocab_size = trained_embed.shape[0]
    freqs = np.zeros(vocab_size, dtype=np.float64)
    for tid, count in counter.items():
        if tid < vocab_size:
            freqs[tid] = count

    bucket_indices = np.argsort(freqs)
    bucket_size = vocab_size // N_FREQ_BUCKETS
    result = trained_embed.copy()
    for b in range(N_FREQ_BUCKETS):
        start = b * bucket_size
        end = start + bucket_size if b < N_FREQ_BUCKETS - 1 else vocab_size
        bucket_ids = bucket_indices[start:end]
        shuffled = rng.permutation(bucket_ids)
        result[bucket_ids] = trained_embed[shuffled]

    return result


def build_spectrum_preserving_random(
    trained_embed: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    U, S, Vt = np.linalg.svd(trained_embed, full_matrices=False)
    n_rows, n_cols = trained_embed.shape
    k = len(S)

    Q_left = np.linalg.qr(rng.standard_normal((n_rows, k)))[0]
    Q_right = np.linalg.qr(rng.standard_normal((n_cols, k)))[0]

    result = (Q_left * S[np.newaxis, :]) @ Q_right.T
    fro_target = np.linalg.norm(trained_embed, "fro")
    fro_actual = np.linalg.norm(result, "fro")
    if fro_actual > 0:
        result *= fro_target / fro_actual
    return result.astype(trained_embed.dtype)


def build_same_frobenius_gaussian(
    trained_embed: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    result = rng.standard_normal(trained_embed.shape).astype(trained_embed.dtype)
    fro_target = np.linalg.norm(trained_embed, "fro")
    fro_actual = np.linalg.norm(result, "fro")
    if fro_actual > 0:
        result *= fro_target / fro_actual
    return result


def build_anchor_to_initial(seed: int, vocab_size: int, embed_dim: int, trained_fro: float) -> np.ndarray:
    torch.manual_seed(seed + 999999)
    model = g165.load_random_init(seed + 999999)
    embed = model.model.embed_tokens.weight.detach().float().cpu().numpy()
    del model
    cleanup_cuda()

    fro_actual = np.linalg.norm(embed, "fro")
    if fro_actual > 0:
        embed *= trained_fro / fro_actual
    return embed


def compute_gradient_matched_lambda(
    init_embed: torch.Tensor,
    target_true: torch.Tensor,
    target_arm: torch.Tensor,
    lambda_base: float,
) -> float:
    d_true = torch.norm(init_embed.float() - target_true.float(), p="fro").item()
    g_ref = 2.0 * lambda_base * d_true

    d_arm = torch.norm(init_embed.float() - target_arm.float(), p="fro").item()
    if d_arm < 1e-12:
        return lambda_base
    return g_ref / (2.0 * d_arm)


# ---------- Training ----------

def train_cell(
    arm_label: str,
    seed: int,
    anchor_target: np.ndarray | None,
    anchor_lambda: float,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
    *,
    n_steps: int = TRAIN_STEPS,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    recipient = g165.load_random_init(seed)
    if hasattr(recipient.config, "use_cache"):
        recipient.config.use_cache = False

    actual_anchor_pairs = []
    if anchor_target is not None:
        target_tensor = torch.from_numpy(anchor_target).to(DEVICE, dtype=torch.float32)
        param_dict = dict(recipient.named_parameters())
        for pname in ["model.embed_tokens.weight"]:
            if pname in param_dict and param_dict[pname].shape == target_tensor.shape:
                actual_anchor_pairs.append((param_dict[pname], target_tensor))

    optimizer = torch.optim.AdamW(
        recipient.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY,
    )
    rng = np.random.default_rng(seed)
    train_schedule = rng.integers(0, int(train_ids.shape[0]), size=(n_steps, BATCH_SIZE), dtype=np.int64)

    trajectory = []
    initial_metrics = g181a.evaluate_nll(recipient, val_ids, val_mask)
    trajectory.append({"step": 0, **initial_metrics})
    print_flush(f"    {arm_label} seed={seed} lambda={anchor_lambda:.8g} step=0 nll={initial_metrics['nll']:.4f}")

    t0 = time.time()
    recipient.train()
    for step in range(1, n_steps + 1):
        batch_indices = train_schedule[step - 1]
        ids = train_ids[torch.as_tensor(batch_indices, dtype=torch.long)].to(DEVICE)
        mask = train_mask[torch.as_tensor(batch_indices, dtype=torch.long)].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            logits = recipient(input_ids=ids, attention_mask=mask, use_cache=False).logits
            ce_loss = g167.causal_ce_loss(logits, ids, mask)

        if not torch.isfinite(ce_loss):
            raise RuntimeError(f"non-finite CE loss at step {step} arm={arm_label} seed={seed}")

        ce_loss.backward()

        if actual_anchor_pairs and anchor_lambda > 0.0:
            with torch.no_grad():
                coeff = 2.0 * anchor_lambda
                for param, target in actual_anchor_pairs:
                    if param.grad is not None:
                        param.grad.add_(param.detach().to(target.dtype) - target, alpha=coeff)

        torch.nn.utils.clip_grad_norm_(recipient.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % LOG_EVERY == 0 or step == n_steps:
            row = {"step": step, "ce_loss": float(ce_loss.item()), "elapsed_s": time.time() - t0}
            if step % EVAL_EVERY == 0 or step == n_steps:
                row.update(g181a.evaluate_nll(recipient, val_ids, val_mask))
                print_flush(f"    {arm_label} seed={seed} step={step} ce={row['ce_loss']:.4f} val_nll={row['nll']:.4f} ({row['elapsed_s']:.0f}s)")
            elif step % (LOG_EVERY * 5) == 0:
                print_flush(f"    {arm_label} seed={seed} step={step} ce={row['ce_loss']:.4f} ({row['elapsed_s']:.0f}s)")
            trajectory.append(row)

    final_metrics = trajectory[-1]
    if "nll" not in final_metrics:
        final_metrics = {"step": n_steps, **g181a.evaluate_nll(recipient, val_ids, val_mask)}
        trajectory.append(final_metrics)

    result = {
        "seed": seed, "arm_label": arm_label,
        "anchor_lambda": anchor_lambda,
        "initial_metrics": initial_metrics,
        "final_nll": float(final_metrics["nll"]),
        "final_top1_acc": float(final_metrics["top1_acc"]),
        "trajectory": trajectory,
        "wallclock_s": time.time() - t0,
    }
    del recipient, optimizer
    cleanup_cuda()
    return result


# ---------- Verdict ----------

def compute_verdict(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload["results"]
    incomplete = [
        f"{arm}:{s}" for arm in ALL_ARMS for s in SEEDS
        if not cell_done(payload, arm, s)
    ]
    if incomplete:
        return {"status": "incomplete", "missing_cells": len(incomplete)}

    scratch_nlls = {str(s): float(results["scratch_ce"][str(s)]["final_nll"]) for s in SEEDS}
    true_nlls = {str(s): float(results["true_trained_anchor"][str(s)]["final_nll"]) for s in SEEDS}

    scratch_gap = np.mean([scratch_nlls[str(s)] - true_nlls[str(s)] for s in SEEDS])

    control_arms = [a for a in ALL_ARMS if a not in ("scratch_ce", "true_trained_anchor")]
    gaps = {}
    seed_wins = {}
    for ctrl in control_arms:
        ctrl_nlls = {str(s): float(results[ctrl][str(s)]["final_nll"]) for s in SEEDS}
        per_seed = [ctrl_nlls[str(s)] - true_nlls[str(s)] for s in SEEDS]
        gaps[ctrl] = float(np.mean(per_seed))
        seed_wins[ctrl] = sum(1 for g in per_seed if g > 0)

    pass_content = all(
        gaps[c] >= PASS_GAP_NATS and seed_wins[c] >= PASS_SEED_FRACTION
        for c in control_arms
    )
    scratch_pass = scratch_gap >= PASS_GAP_NATS

    format_fail_arms = [
        c for c in control_arms
        if gaps[c] < PASS_GAP_NATS or seed_wins[c] < (len(SEEDS) - 1)
    ]

    if pass_content and scratch_pass:
        verdict = "PASS_CONTENT"
    elif not scratch_pass:
        verdict = "REPRO_FAIL"
    else:
        verdict = "FORMAT_FAIL"

    return {
        "status": "complete",
        "verdict": verdict,
        "scratch_gap_mean": float(scratch_gap),
        "control_gaps": gaps,
        "control_seed_wins": seed_wins,
        "format_fail_arms": format_fail_arms,
        "pass_content": pass_content,
    }


def cell_done(payload: dict[str, Any], arm_label: str, seed: int) -> bool:
    cell = payload["results"].get(arm_label, {}).get(str(seed))
    if not isinstance(cell, dict):
        return False
    return "final_nll" in cell


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", type=str, default=None, help="comma-separated seed subset")
    parser.add_argument("--arms", type=str, default=None, help="comma-separated arm subset")
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--prepare-targets-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    seeds = [42] if args.smoke else (
        [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS
    )
    arms = args.arms.split(",") if args.arms else ALL_ARMS
    train_steps = 50 if args.smoke else TRAIN_STEPS

    print_flush(f"=== g189 C23 Content-Causality Controls ===")
    print_flush(f"  smoke={args.smoke}, seeds={seeds}, arms={arms}, steps={train_steps}")

    # Load tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(g165._MODEL_ID, trust_remote_code=True)

    # Build targets
    print_flush("\n--- Building anchor targets ---")
    trained_embed = get_trained_embed(tok)
    trained_fro = float(np.linalg.norm(trained_embed, "fro"))
    print_flush(f"  Trained embed: {trained_embed.shape}, Fro={trained_fro:.1f}")

    target_rng = np.random.default_rng(189)

    targets: dict[str, np.ndarray] = {}
    targets["true_trained_anchor"] = trained_embed

    print_flush("  Building row-shuffled target...")
    targets["row_shuffled_anchor"] = build_row_shuffled(trained_embed, target_rng)

    print_flush("  Building freq-bucket-shuffled target...")
    targets["freq_bucket_shuffle_anchor"] = build_freq_bucket_shuffled(trained_embed, tok, target_rng)

    print_flush("  Building spectrum-preserving random target...")
    targets["spectrum_preserving_random"] = build_spectrum_preserving_random(trained_embed, target_rng)

    print_flush("  Building same-Frobenius Gaussian target...")
    targets["same_frobenius_gaussian"] = build_same_frobenius_gaussian(trained_embed, target_rng)

    print_flush("  Building anchor-to-initial target...")
    targets["anchor_to_initial"] = build_anchor_to_initial(
        seed=189, vocab_size=trained_embed.shape[0], embed_dim=trained_embed.shape[1], trained_fro=trained_fro,
    )

    for name, t in targets.items():
        fro = float(np.linalg.norm(t, "fro"))
        print_flush(f"    {name}: shape={t.shape}, Fro={fro:.1f} (ratio={fro/trained_fro:.4f})")

    if args.prepare_targets_only:
        print_flush("\n  Targets prepared. Exiting (--prepare-targets-only).")
        return

    # Load data
    print_flush("\n--- Loading training data ---")
    train_ids, train_mask, _ = g167.load_c4_windows(
        tok, split="train", seed=C4_TRAIN_SEED, n_windows=N_TRAIN_WINDOWS,
    )
    train_hashes = g167.collect_13gram_hashes(train_ids, train_mask)
    val_ids, val_mask, _ = g167.load_c4_windows(
        tok, split="train", seed=C4_VAL_SEED, n_windows=N_C4_VAL_WINDOWS,
        forbidden_hashes=train_hashes,
    )
    print_flush(f"  Train: {train_ids.shape}, Val: {val_ids.shape}")

    # Resume
    if not args.no_resume and OUT_PATH.exists():
        payload = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    else:
        payload = {
            "genome": 189,
            "name": "c23_content_causality",
            "timestamp_utc_started": now_utc(),
            "model_id": g165._MODEL_ID,
            "device": str(DEVICE),
            "config": {
                "seeds": seeds,
                "train_steps": train_steps,
                "anchor_lambda_base": ANCHOR_LAMBDA_BASE,
                "n_freq_buckets": N_FREQ_BUCKETS,
                "pass_gap_nats": PASS_GAP_NATS,
                "pass_seed_fraction": PASS_SEED_FRACTION,
            },
            "target_diagnostics": {},
            "results": {},
            "summary": {},
            "verdict": "INCOMPLETE",
        }

    for arm in arms:
        payload["results"].setdefault(arm, {})

    t_start = time.time()

    def save():
        payload["timestamp_utc_last_write"] = now_utc()
        payload["elapsed_s"] = time.time() - t_start
        tmp = OUT_PATH.with_suffix(OUT_PATH.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, OUT_PATH)

    # Train cells
    print_flush("\n--- Training cells ---")
    n_cells = 0
    for arm_label in arms:
        for seed in seeds:
            key = str(seed)
            if key in payload["results"].get(arm_label, {}) and not args.no_resume:
                if cell_done(payload, arm_label, seed):
                    print_flush(f"  Skipping {arm_label}/seed={seed} (done)")
                    continue

            if args.max_cells and n_cells >= args.max_cells:
                print_flush(f"  max-cells={args.max_cells} reached, stopping")
                save()
                return

            print_flush(f"\n  === {arm_label} seed={seed} ===")

            if arm_label == "scratch_ce":
                anchor_target = None
                anchor_lambda = 0.0
            else:
                anchor_target = targets[arm_label]

                # Gradient matching: compute lambda so initial gradient L2 matches true_trained
                init_model = g165.load_random_init(seed)
                init_embed = init_model.model.embed_tokens.weight.detach().float().cpu()
                del init_model
                cleanup_cuda()

                target_true_t = torch.from_numpy(targets["true_trained_anchor"]).float()
                target_arm_t = torch.from_numpy(anchor_target).float()

                anchor_lambda = compute_gradient_matched_lambda(
                    init_embed, target_true_t, target_arm_t, ANCHOR_LAMBDA_BASE,
                )

                d_arm = torch.norm(init_embed - target_arm_t, p="fro").item()
                g_arm = 2.0 * anchor_lambda * d_arm
                print_flush(f"    gradient-matched lambda={anchor_lambda:.6f} d={d_arm:.1f} G={g_arm:.1f}")

                diag_key = f"{arm_label}_{seed}"
                payload["target_diagnostics"][diag_key] = {
                    "lambda": anchor_lambda,
                    "distance_to_target": d_arm,
                    "initial_gradient_l2": g_arm,
                    "target_fro": float(np.linalg.norm(anchor_target, "fro")),
                }

            result = train_cell(
                arm_label=arm_label,
                seed=seed,
                anchor_target=anchor_target,
                anchor_lambda=anchor_lambda,
                train_ids=train_ids,
                train_mask=train_mask,
                val_ids=val_ids,
                val_mask=val_mask,
                n_steps=train_steps,
            )
            payload["results"][arm_label][key] = result
            save()
            n_cells += 1

            print_flush(f"  {arm_label} seed={seed} final_nll={result['final_nll']:.4f} ({result['wallclock_s']:.0f}s)")

    summary = compute_verdict(payload)
    payload["summary"] = summary
    payload["verdict"] = summary.get("verdict", "INCOMPLETE")
    save()

    print_flush(f"\n*** g189 VERDICT: {summary.get('verdict', '?')} ***")
    print_flush(f"  scratch_gap={summary.get('scratch_gap_mean', 0):+.4f}")
    for ctrl, gap in summary.get("control_gaps", {}).items():
        wins = summary.get("control_seed_wins", {}).get(ctrl, 0)
        print_flush(f"  {ctrl}: gap={gap:+.4f} wins={wins}/{len(SEEDS)}")


if __name__ == "__main__":
    main()
