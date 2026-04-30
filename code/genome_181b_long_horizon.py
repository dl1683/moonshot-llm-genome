"""
genome_181b_long_horizon.py

Cycle 65 A8 long-horizon attenuation control.

Question: Does the g181a embed-lm-head-only +0.483 nat gain persist at 5000 steps
(2.5x the g181a 2000-step horizon), or does scratch CE close the gap?

Arms (2 only --no_embed_lm_head already proven to HARM in g181a):
  1. scratch_ce --baseline, no anchor
  2. embed_lm_head_only_anchor --lambda gradient-matched to g181a full-anchor at step 0

PASS: embed_lm_head_only gap at step 5000 >= +0.5 nats vs scratch
FAIL: gap < +0.3 nats at step 5000

Outputs: results/genome_181b_long_horizon.json
"""
# ENVELOPE COMPLIANCE (cycle 73 g181b, 2026-04-29)
# 2 arms x 3 seeds x 5000 steps x ~0.4s/step = ~12000s = 3.3h. Setup/eval
# overhead ~600s. Total ~12600s <= 14400s COMPUTE.md hard envelope.
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

OUT_PATH = ROOT / "results" / "genome_181b_long_horizon.json"

SEEDS = [42, 7, 13]
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

ANCHOR_LAMBDA_FULL = 0.01
ANCHOR_SCHEDULE = "constant"
N_BOOT = 10_000

PASS_EMBED_GAIN_NATS = 0.50
FAIL_EMBED_GAIN_MAX_NATS = 0.30

DEVICE = g165.DEVICE
FORWARD_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


@dataclass(frozen=True)
class ArmSpec:
    label: str
    anchor_subset: str

    @property
    def use_anchor(self) -> bool:
        return self.anchor_subset != "none"


ARMS = [
    ArmSpec(label="scratch_ce", anchor_subset="none"),
    ArmSpec(label="embed_lm_head_only_anchor", anchor_subset="embed_lm_head"),
]

ARM_BY_LABEL = {arm.label: arm for arm in ARMS}


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def autocast_context():
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def base_payload() -> dict[str, Any]:
    return {
        "genome": "181b",
        "name": "long_horizon_attenuation",
        "timestamp_utc_started": now_utc(),
        "model_id": g165._MODEL_ID,
        "device": DEVICE,
        "forward_dtype": str(FORWARD_DTYPE),
        "config": {
            "seeds": SEEDS,
            "arms": [{"label": arm.label, "anchor_subset": arm.anchor_subset} for arm in ARMS],
            "train_steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "lr": LR,
            "betas": list(BETAS),
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "n_train_windows": N_TRAIN_WINDOWS,
            "n_c4_val_windows": N_C4_VAL_WINDOWS,
            "anchor_lambda_full": ANCHOR_LAMBDA_FULL,
            "anchor_schedule": ANCHOR_SCHEDULE,
            "pass_criteria": {
                "PASS": "embed_lm_head_only gap at step 5000 >= +0.5 nats vs scratch",
                "FAIL": "gap < +0.3 nats at step 5000",
                "INTERMEDIATE": "gap in [+0.3, +0.5)",
            },
            "envelope_estimate_s": {
                "train": 2 * 3 * TRAIN_STEPS * 0.4,
                "overhead": 600,
                "total": 2 * 3 * TRAIN_STEPS * 0.4 + 600,
                "limit": 14_400,
            },
        },
        "data": {},
        "donor": {},
        "anchor_diagnostics": {},
        "results": {arm.label: {} for arm in ARMS},
        "summary": {},
        "verdict": "INCOMPLETE",
    }


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("data", {})
    payload.setdefault("donor", {})
    payload.setdefault("anchor_diagnostics", {})
    payload.setdefault("results", {})
    for arm in ARMS:
        payload["results"].setdefault(arm.label, {})
    payload.setdefault("summary", {})
    payload.setdefault("verdict", "INCOMPLETE")
    return payload


def load_or_create_payload(resume: bool) -> dict[str, Any]:
    if resume and OUT_PATH.exists():
        try:
            return normalize_payload(json.loads(OUT_PATH.read_text(encoding="utf-8")))
        except Exception as exc:
            raise RuntimeError(f"failed to read existing {OUT_PATH}: {exc}") from exc
    return base_payload()


def write_payload(payload: dict[str, Any], *, t_start: float, incremental: bool) -> None:
    payload["timestamp_utc_last_write"] = now_utc()
    payload["elapsed_s"] = time.time() - t_start
    payload["incremental"] = bool(incremental)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = OUT_PATH.with_suffix(OUT_PATH.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    os.replace(tmp_path, OUT_PATH)


def cell_done(payload: dict[str, Any], arm_label: str, seed: int) -> bool:
    cell = payload["results"].get(arm_label, {}).get(str(seed))
    if not isinstance(cell, dict):
        return False
    traj = cell.get("trajectory", [])
    if not traj:
        return False
    return traj[-1].get("step", 0) >= TRAIN_STEPS


def all_cells_done(payload: dict[str, Any]) -> bool:
    return all(cell_done(payload, arm.label, seed) for arm in ARMS for seed in SEEDS)


def load_main_data(tok):
    train_ids, train_mask, train_meta = g167.load_c4_windows(
        tok, split="train", seed=C4_TRAIN_SEED, n_windows=N_TRAIN_WINDOWS,
    )
    train_hashes = g167.collect_13gram_hashes(train_ids, train_mask)
    val_ids, val_mask, val_meta = g167.load_c4_windows(
        tok, split="train", seed=C4_VAL_SEED, n_windows=N_C4_VAL_WINDOWS,
        forbidden_hashes=train_hashes,
    )
    val_hashes = g167.collect_13gram_hashes(val_ids, val_mask)
    overlap = len(train_hashes.intersection(val_hashes))
    if overlap != 0:
        raise RuntimeError(f"C4 train/val token 13-gram overlap: {overlap}")
    meta = {
        "train": train_meta, "c4_val": val_meta,
        "train_seed": C4_TRAIN_SEED, "val_seed": C4_VAL_SEED,
        "train_shape": list(train_ids.shape), "val_shape": list(val_ids.shape),
        "n_train_windows": N_TRAIN_WINDOWS,
    }
    return train_ids, train_mask, val_ids, val_mask, meta


def train_cell(
    arm: ArmSpec, *, seed: int, actual_lambda_0: float,
    donor_params_device: dict[str, torch.Tensor] | None,
    train_ids: torch.Tensor, train_mask: torch.Tensor,
    val_ids: torch.Tensor, val_mask: torch.Tensor,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    recipient = g165.load_random_init(seed)
    if hasattr(recipient.config, "use_cache"):
        recipient.config.use_cache = False

    optimizer = torch.optim.AdamW(
        recipient.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY,
    )
    rng = np.random.default_rng(seed)
    train_schedule = rng.integers(0, int(train_ids.shape[0]), size=(TRAIN_STEPS, BATCH_SIZE), dtype=np.int64)

    anchor_pairs = []
    if arm.use_anchor:
        if donor_params_device is None:
            raise RuntimeError(f"{arm.label} requires donor params")
        anchor_pairs = g181a.build_anchor_pairs(recipient, donor_params_device, arm.anchor_subset)

    trajectory = []
    initial_metrics = g181a.evaluate_nll(recipient, val_ids, val_mask)
    trajectory.append({"step": 0, **initial_metrics})
    print(f"    {arm.label} seed={seed} lambda={actual_lambda_0:.8g} step=0 nll={initial_metrics['nll']:.4f}")

    t0 = time.time()
    recipient.train()
    for step in range(1, TRAIN_STEPS + 1):
        batch_indices = train_schedule[step - 1]
        ids = train_ids[torch.as_tensor(batch_indices, dtype=torch.long)].to(DEVICE)
        mask = train_mask[torch.as_tensor(batch_indices, dtype=torch.long)].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            logits = recipient(input_ids=ids, attention_mask=mask, use_cache=False).logits
            ce_loss = g167.causal_ce_loss(logits, ids, mask)

        if not torch.isfinite(ce_loss):
            raise RuntimeError(f"non-finite CE loss at step {step} arm={arm.label} seed={seed}")

        ce_loss.backward()

        if anchor_pairs and actual_lambda_0 > 0.0:
            with torch.no_grad():
                coeff = 2.0 * actual_lambda_0
                for _, param, donor_tensor in anchor_pairs:
                    if param.grad is None:
                        continue
                    param.grad.add_(param.detach().to(donor_tensor.dtype) - donor_tensor, alpha=coeff)

        torch.nn.utils.clip_grad_norm_(recipient.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % LOG_EVERY == 0 or step == TRAIN_STEPS:
            row = {"step": step, "ce_loss": float(ce_loss.item()), "elapsed_s": time.time() - t0}
            if step % EVAL_EVERY == 0 or step == TRAIN_STEPS:
                row.update(g181a.evaluate_nll(recipient, val_ids, val_mask))
                print(f"    {arm.label} seed={seed} step={step} ce={row['ce_loss']:.4f} val_nll={row['nll']:.4f} ({row['elapsed_s']:.0f}s)")
            elif step % (LOG_EVERY * 5) == 0:
                print(f"    {arm.label} seed={seed} step={step} ce={row['ce_loss']:.4f} ({row['elapsed_s']:.0f}s)")
            trajectory.append(row)

    final_metrics = trajectory[-1]
    if "nll" not in final_metrics:
        final_metrics = {"step": TRAIN_STEPS, **g181a.evaluate_nll(recipient, val_ids, val_mask)}
        trajectory.append(final_metrics)

    result = {
        "seed": seed, "arm_label": arm.label,
        "lambda_0": actual_lambda_0,
        "initial_metrics": initial_metrics,
        "final_nll": float(final_metrics["nll"]),
        "final_top1_acc": float(final_metrics["top1_acc"]),
        "trajectory": trajectory,
        "wallclock_s": time.time() - t0,
    }
    del recipient, optimizer
    cleanup_cuda()
    return result


def build_summary(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload["results"]
    incomplete = [
        f"{arm.label}:{seed}" for arm in ARMS for seed in SEEDS
        if not cell_done(payload, arm.label, seed)
    ]
    if incomplete:
        return {"status": "incomplete", "missing_cells": incomplete}

    scratch_nlls = {str(s): float(results["scratch_ce"][str(s)]["final_nll"]) for s in SEEDS}
    embed_nlls = {str(s): float(results["embed_lm_head_only_anchor"][str(s)]["final_nll"]) for s in SEEDS}

    gains = {str(s): scratch_nlls[str(s)] - embed_nlls[str(s)] for s in SEEDS}
    gain_values = [gains[str(s)] for s in SEEDS]
    mean_gain = float(np.mean(gain_values))
    std_gain = float(np.std(gain_values))

    rng = np.random.default_rng(181300)
    boot_means = np.empty(N_BOOT, dtype=np.float64)
    arr = np.asarray(gain_values, dtype=np.float64)
    for i in range(N_BOOT):
        boot_means[i] = arr[rng.integers(0, len(arr), size=len(arr))].mean()
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    if mean_gain >= PASS_EMBED_GAIN_NATS and ci_lo > 0:
        status = "PASS"
        verdict = f"PASS: embed_lm_head_only gap = {mean_gain:+.3f} nats at step {TRAIN_STEPS}, CI [{ci_lo:+.3f}, {ci_hi:+.3f}]. Persists beyond g181a's 2000-step horizon."
    elif mean_gain < FAIL_EMBED_GAIN_MAX_NATS:
        status = "FAIL"
        verdict = f"FAIL: embed_lm_head_only gap = {mean_gain:+.3f} nats at step {TRAIN_STEPS} (< +0.3 threshold). Scratch CE catches up --effect is short-horizon only."
    else:
        status = "INTERMEDIATE"
        verdict = f"INTERMEDIATE: embed_lm_head_only gap = {mean_gain:+.3f} nats at step {TRAIN_STEPS}, CI [{ci_lo:+.3f}, {ci_hi:+.3f}]."

    trajectory_summary = {}
    for arm in ARMS:
        eval_steps = []
        for s in SEEDS:
            traj = results[arm.label][str(s)]["trajectory"]
            for row in traj:
                if "nll" in row and row["step"] not in [r["step"] for r in eval_steps]:
                    eval_steps.append({"step": row["step"]})
        for step_row in eval_steps:
            step = step_row["step"]
            nlls = []
            for s in SEEDS:
                traj = results[arm.label][str(s)]["trajectory"]
                for row in traj:
                    if row["step"] == step and "nll" in row:
                        nlls.append(row["nll"])
            if nlls:
                step_row[f"{arm.label}_mean_nll"] = float(np.mean(nlls))
        trajectory_summary[arm.label] = eval_steps

    gap_over_time = []
    for step in sorted(set(r["step"] for r in trajectory_summary.get("scratch_ce", []))):
        scratch_at_step = []
        embed_at_step = []
        for s in SEEDS:
            for row in results["scratch_ce"][str(s)]["trajectory"]:
                if row["step"] == step and "nll" in row:
                    scratch_at_step.append(row["nll"])
            for row in results["embed_lm_head_only_anchor"][str(s)]["trajectory"]:
                if row["step"] == step and "nll" in row:
                    embed_at_step.append(row["nll"])
        if scratch_at_step and embed_at_step and len(scratch_at_step) == len(embed_at_step):
            gap = float(np.mean(scratch_at_step)) - float(np.mean(embed_at_step))
            gap_over_time.append({"step": step, "mean_gap_nats": gap})

    return {
        "status": status,
        "verdict": verdict,
        "scratch_final_nll": {"per_seed": scratch_nlls, "mean": float(np.mean(list(scratch_nlls.values())))},
        "embed_final_nll": {"per_seed": embed_nlls, "mean": float(np.mean(list(embed_nlls.values())))},
        "embed_gain_vs_scratch": {
            "per_seed": gains, "mean": mean_gain, "std": std_gain,
            "ci_95_lo": ci_lo, "ci_95_hi": ci_hi,
        },
        "gap_over_time": gap_over_time,
        "reference_g181a_embed_gain_at_2000": 0.483,
        "criteria": {
            "pass_threshold_nats": PASS_EMBED_GAIN_NATS,
            "fail_threshold_nats": FAIL_EMBED_GAIN_MAX_NATS,
            "train_steps": TRAIN_STEPS,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Genome 181b long-horizon attenuation control.")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    print("genome_181b: long-horizon attenuation control")
    print(f"  model={g165._MODEL_ID}")
    print(f"  device={DEVICE} forward_dtype={FORWARD_DTYPE}")
    print(f"  arms={[arm.label for arm in ARMS]}")
    print(f"  seeds={SEEDS} steps={TRAIN_STEPS}")
    print(f"  output={OUT_PATH}")

    t_start = time.time()
    payload = load_or_create_payload(resume=not args.no_resume)

    donor_model, tok = g165.load_trained_donor()
    if hasattr(donor_model.config, "use_cache"):
        donor_model.config.use_cache = False
    donor_params_cpu = g181a.snapshot_params_cpu(donor_model)
    del donor_model
    cleanup_cuda()

    donor_params_device = g181a.stage_params_to_device(donor_params_cpu)
    del donor_params_cpu
    cleanup_cuda()

    if "lambda_and_frobenius" not in payload.get("anchor_diagnostics", {}):
        payload["anchor_diagnostics"]["lambda_and_frobenius"] = g181a.compute_lambda_diagnostics(donor_params_device)

    train_ids, train_mask, val_ids, val_mask, data_meta = load_main_data(tok)
    payload["data"] = data_meta
    write_payload(payload, t_start=t_start, incremental=True)

    if not all_cells_done(payload):
        for seed in SEEDS:
            for arm in ARMS:
                if cell_done(payload, arm.label, seed):
                    print(f"  skip complete cell {arm.label} seed={seed}")
                    continue

                actual_lambda = 0.0
                if arm.use_anchor:
                    diag = payload["anchor_diagnostics"]["lambda_and_frobenius"]["by_seed"][str(seed)]
                    actual_lambda = float(diag["lambda_embed_lm_head_only_anchor"])

                print(f"\n--- arm={arm.label} seed={seed} lambda={actual_lambda:.8g} ---")
                payload["results"][arm.label][str(seed)] = train_cell(
                    arm, seed=seed, actual_lambda_0=actual_lambda,
                    donor_params_device=donor_params_device if arm.use_anchor else None,
                    train_ids=train_ids, train_mask=train_mask,
                    val_ids=val_ids, val_mask=val_mask,
                )
                summary = build_summary(payload)
                payload["summary"] = summary
                payload["verdict"] = summary.get("verdict", "INCOMPLETE")
                write_payload(payload, t_start=t_start, incremental=True)

    summary = build_summary(payload)
    payload["summary"] = summary
    payload["verdict"] = summary.get("verdict", "INCOMPLETE")
    write_payload(payload, t_start=t_start, incremental=False)
    print(f"\n=== verdict: {payload['verdict']} ===")
    print(f"Saved: {OUT_PATH} ({payload['elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()
