"""
genome_174_donor_specificity_control.py

Matched-null donor-specificity control from cycle 45 adversarial review and
g172 advisor prioritization.

Question
--------
Do the g165/g167 gains require trained donor information in the continuous
loss, or can matched nulls explain the effect as generic regularization or
generic dense supervision?

Part A: weight-anchor matched null extending g165
  - scratch_baseline
  - anchor_trained_donor: lambda_0=0.01 constant Frobenius anchor to trained Qwen3-0.6B
  - anchor_random_donor: lambda_0=0.01 constant Frobenius anchor to random-init Qwen3-0.6B
  - anchor_permuted_donor: lambda_0=0.01 constant Frobenius anchor to layerwise-permuted trained Qwen3-0.6B
  - 3 seeds, 500 steps, g165 data/eval protocol

Part B: KD matched null extending g167
  - scratch_ce
  - kd_trained_teacher: top-k=64 KD with trained Qwen3-0.6B teacher, reusing g167 cache
  - kd_uniform_target: same top-k KD math with uniform probabilities on random supports
  - kd_random_teacher: same top-k KD math with fresh random-init Qwen3-0.6B teacher inference
  - 3 seeds, 6000 steps, g167 minimal_3L student/data/eval protocol

Outputs
-------
  - results/genome_174_donor_specificity_control.json
  - results/genome_174_targets/genome_174_anchor_random_donor_seed*.npz
  - results/genome_174_targets/genome_174_anchor_permuted_donor_seed*.npz
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass
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


OUT_PATH = ROOT / "results" / "genome_174_donor_specificity_control.json"
TARGET_DIR = ROOT / "results" / "genome_174_targets"
CACHE_DIR = ROOT / "results" / "cache"

SEEDS = [42, 7, 13]

# Part A is locked to the g165 strong constant-anchor condition.
ANCHOR_LAMBDA_0 = 0.01
ANCHOR_SCHEDULE = "constant"
RANDOM_DONOR_SEED = 174165001
PERMUTED_DONOR_SEED = 174165002

# Part B is locked to the g167 canonical KD setup.
RANDOM_TEACHER_SEED = 174167001
UNIFORM_TARGET_SEED = 174167002

REFERENCE_G165_TRAINED_GAIN_NATS = 1.088
REFERENCE_G167_TRAINED_GAIN_PP = 1.014
NULL_80_FRACTION = 0.80
PASS_PART_A_MARGIN_NATS = 0.30
PASS_PART_B_MARGIN_PP = 0.30
N_BOOT = 10_000

DEVICE = g167.DEVICE
FORWARD_DTYPE = g167.FORWARD_DTYPE

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


@dataclass(frozen=True)
class AnchorArmSpec:
    label: str
    target_kind: str
    lambda_0: float
    schedule: str
    description: str

    @property
    def use_anchor(self) -> bool:
        return self.target_kind != "none" and self.lambda_0 > 0.0


@dataclass(frozen=True)
class KDArmSpec:
    label: str
    target_kind: str
    description: str

    @property
    def use_kd(self) -> bool:
        return self.target_kind != "none"


ANCHOR_ARMS = [
    AnchorArmSpec(
        label="scratch_baseline",
        target_kind="none",
        lambda_0=0.0,
        schedule=ANCHOR_SCHEDULE,
        description="No anchor; g165 scratch baseline.",
    ),
    AnchorArmSpec(
        label="anchor_trained_donor",
        target_kind="trained",
        lambda_0=ANCHOR_LAMBDA_0,
        schedule=ANCHOR_SCHEDULE,
        description="Constant Frobenius anchor to trained Qwen3-0.6B weights.",
    ),
    AnchorArmSpec(
        label="anchor_random_donor",
        target_kind="random",
        lambda_0=ANCHOR_LAMBDA_0,
        schedule=ANCHOR_SCHEDULE,
        description="Constant Frobenius anchor to random-init Qwen3-0.6B weights.",
    ),
    AnchorArmSpec(
        label="anchor_permuted_donor",
        target_kind="permuted",
        lambda_0=ANCHOR_LAMBDA_0,
        schedule=ANCHOR_SCHEDULE,
        description="Constant Frobenius anchor to layerwise hidden-dim-permuted trained Qwen3-0.6B weights.",
    ),
]

KD_ARMS = [
    KDArmSpec(
        label="scratch_ce",
        target_kind="none",
        description="No KD; g167 scratch CE baseline.",
    ),
    KDArmSpec(
        label="kd_trained_teacher",
        target_kind="trained",
        description="CE plus top-k=64 KD with trained Qwen3-0.6B teacher.",
    ),
    KDArmSpec(
        label="kd_uniform_target",
        target_kind="uniform",
        description="CE plus top-k=64 KD with uniform target probabilities on teacher-free random supports.",
    ),
    KDArmSpec(
        label="kd_random_teacher",
        target_kind="random",
        description="CE plus top-k=64 KD with random-init Qwen3-0.6B teacher logits.",
    ),
]

ANCHOR_TARGET_LABELS = ["anchor_trained_donor", "anchor_random_donor", "anchor_permuted_donor"]
ANCHOR_NULL_LABELS = ["scratch_baseline", "anchor_random_donor", "anchor_permuted_donor"]
KD_TARGET_LABELS = ["kd_trained_teacher", "kd_uniform_target", "kd_random_teacher"]
KD_NULL_LABELS = ["scratch_ce", "kd_uniform_target", "kd_random_teacher"]


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def assert_locked_protocol() -> None:
    checks = {
        "device agreement": (g165.DEVICE, g167.DEVICE),
        "g165 seeds": (list(g165.SEEDS), SEEDS),
        "g165 seq_len": (g165.SEQ_LEN, 256),
        "g165 batch_size": (g165.BATCH_SIZE, 8),
        "g165 n_steps": (g165.N_STEPS, 500),
        "g165 lr": (g165.LR, 3e-4),
        "g167 seeds": (list(g167.SEEDS), SEEDS),
        "g167 seq_len": (g167.SEQ_LEN, 256),
        "g167 train_batch_size": (g167.TRAIN_BATCH_SIZE, 8),
        "g167 train_steps": (g167.TRAIN_STEPS, 6000),
        "g167 lr": (g167.LR, 3e-4),
        "g167 kd_topk": (g167.KD_TOPK, 64),
        "g167 kd_temp": (g167.KD_TEMP, 2.0),
        "g167 kd_gamma": (g167.KD_GAMMA, 0.5),
    }
    mismatches = [f"{name}: observed={a!r} expected={b!r}" for name, (a, b) in checks.items() if a != b]
    if mismatches:
        raise RuntimeError("locked protocol drift detected:\n  " + "\n  ".join(mismatches))


def base_payload() -> dict[str, Any]:
    return {
        "genome": 174,
        "name": "donor_specificity_control",
        "timestamp_utc_started": now_utc(),
        "model_id": g165._MODEL_ID,
        "teacher_model_id": g167.TEACHER_HF,
        "device": DEVICE,
        "forward_dtype": str(FORWARD_DTYPE),
        "config": {
            "seeds": SEEDS,
            "part_a": {
                "protocol_source": "genome_165_annealed_donor.py",
                "steps": g165.N_STEPS,
                "batch_size": g165.BATCH_SIZE,
                "seq_len": g165.SEQ_LEN,
                "lr": g165.LR,
                "lambda_0": ANCHOR_LAMBDA_0,
                "schedule": ANCHOR_SCHEDULE,
                "reference_trained_gain_nats": REFERENCE_G165_TRAINED_GAIN_NATS,
                "pass_margin_nats": PASS_PART_A_MARGIN_NATS,
            },
            "part_b": {
                "protocol_source": "genome_167_kd_canonical.py",
                "student_family": "minimal_3L_noMLP_llama_qwen_vocab",
                "steps": g167.TRAIN_STEPS,
                "train_batch_size": g167.TRAIN_BATCH_SIZE,
                "eval_batch_size": g167.EVAL_BATCH_SIZE,
                "seq_len": g167.SEQ_LEN,
                "lr": g167.LR,
                "lr_warmup_steps": g167.LR_WARMUP_STEPS,
                "betas": list(g167.BETAS),
                "weight_decay": g167.WEIGHT_DECAY,
                "grad_clip": g167.GRAD_CLIP,
                "n_train_windows": g167.N_TRAIN_WINDOWS,
                "n_c4_val_windows": g167.N_C4_VAL_WINDOWS,
                "n_wikitext_val_windows": g167.N_WIKITEXT_VAL_WINDOWS,
                "kd_top_k": g167.KD_TOPK,
                "kd_temp": g167.KD_TEMP,
                "kd_gamma": g167.KD_GAMMA,
                "reference_trained_gain_pp": REFERENCE_G167_TRAINED_GAIN_PP,
                "pass_margin_pp": PASS_PART_B_MARGIN_PP,
            },
            "null_80_fraction": NULL_80_FRACTION,
            "random_donor_seed": RANDOM_DONOR_SEED,
            "permuted_donor_seed": PERMUTED_DONOR_SEED,
            "random_teacher_seed": RANDOM_TEACHER_SEED,
            "uniform_target_seed": UNIFORM_TARGET_SEED,
        },
        "part_a": {
            "arms": [arm.__dict__ for arm in ANCHOR_ARMS],
            "results": {arm.label: {} for arm in ANCHOR_ARMS},
            "target_npz": {},
            "anchor_diagnostics": {},
        },
        "part_b": {
            "arms": [arm.__dict__ for arm in KD_ARMS],
            "results": {arm.label: {} for arm in KD_ARMS},
            "data": {},
            "teacher_reference": {},
            "cache_meta": {},
        },
        "summary": {},
        "verdict": "INCOMPLETE",
    }


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "part_a" not in payload:
        payload["part_a"] = {}
    if "part_b" not in payload:
        payload["part_b"] = {}
    payload["part_a"].setdefault("results", {})
    payload["part_a"].setdefault("target_npz", {})
    payload["part_a"].setdefault("anchor_diagnostics", {})
    payload["part_b"].setdefault("results", {})
    payload["part_b"].setdefault("data", {})
    payload["part_b"].setdefault("teacher_reference", {})
    payload["part_b"].setdefault("cache_meta", {})
    for arm in ANCHOR_ARMS:
        payload["part_a"]["results"].setdefault(arm.label, {})
    for arm in KD_ARMS:
        payload["part_b"]["results"].setdefault(arm.label, {})
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


def cell_done(payload: dict[str, Any], part_key: str, arm_label: str, seed: int) -> bool:
    return str(seed) in payload[part_key]["results"].get(arm_label, {})


def all_cells_done(payload: dict[str, Any], part_key: str, arms: list[Any]) -> bool:
    return all(cell_done(payload, part_key, arm.label, seed) for arm in arms for seed in SEEDS)


def paired_bootstrap_ci(values: list[float], *, seed: int, n_boot: int = N_BOOT) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        boot_means[idx] = arr[rng.integers(0, len(arr), size=len(arr))].mean()
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def metric_summary(values_by_seed: dict[str, float], *, seed: int) -> dict[str, Any]:
    values = [float(values_by_seed[str(s)]) for s in SEEDS]
    ci_lo, ci_hi = paired_bootstrap_ci(values, seed=seed)
    return {
        "per_seed": values_by_seed,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "ci_95_lo": ci_lo,
        "ci_95_hi": ci_hi,
    }


def _write_npy_to_zip(zf: zipfile.ZipFile, key: str, array: np.ndarray) -> None:
    with zf.open(f"{key}.npy", "w", force_zip64=True) as handle:
        np.lib.format.write_array(handle, array, allow_pickle=False)


def save_state_npz(
    path: Path,
    params: dict[str, torch.Tensor],
    *,
    metadata: dict[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return {
            "path": str(path),
            "exists": True,
            "skipped_existing": True,
            "size_bytes": path.stat().st_size,
            "sha1": None,
            "metadata": metadata,
        }

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha1()
    names: list[str] = []
    shapes: list[str] = []
    dtypes: list[str] = []
    n_params = 0

    with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for idx, (name, tensor) in enumerate(params.items()):
            arr = np.ascontiguousarray(tensor.detach().cpu().numpy())
            names.append(name)
            shapes.append("x".join(str(dim) for dim in arr.shape))
            dtypes.append(arr.dtype.str)
            n_params += int(arr.size)

            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(arr.shape).encode("utf-8"))
            digest.update(b"\0")
            digest.update(arr.dtype.str.encode("utf-8"))
            digest.update(b"\0")
            digest.update(memoryview(arr).cast("B"))

            _write_npy_to_zip(zf, f"arr_{idx:05d}", arr)

        state_sha1 = digest.hexdigest()
        full_metadata = dict(metadata)
        full_metadata.update(
            {
                "format": "npz_state_arrays_v1",
                "array_key_pattern": "arr_%05d",
                "n_tensors": len(names),
                "n_params": n_params,
                "state_sha1": state_sha1,
                "saved_utc": now_utc(),
            }
        )
        _write_npy_to_zip(zf, "names", np.asarray(names, dtype=np.str_))
        _write_npy_to_zip(zf, "shapes", np.asarray(shapes, dtype=np.str_))
        _write_npy_to_zip(zf, "dtypes", np.asarray(dtypes, dtype=np.str_))
        _write_npy_to_zip(zf, "metadata_json", np.asarray(json.dumps(full_metadata, sort_keys=True), dtype=np.str_))

    os.replace(tmp_path, path)
    return {
        "path": str(path),
        "exists": True,
        "skipped_existing": False,
        "size_bytes": path.stat().st_size,
        "sha1": state_sha1,
        "metadata": full_metadata,
    }


def snapshot_params_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    raw = g165.snapshot_donor_params(model)
    params = {name: tensor.detach().cpu().contiguous() for name, tensor in raw.items()}
    del raw
    cleanup_cuda()
    return params


def load_trained_anchor_params() -> tuple[dict[str, torch.Tensor], Any]:
    donor, tok = g165.load_trained_donor()
    params = snapshot_params_cpu(donor)
    del donor
    cleanup_cuda()
    return params, tok


def load_random_anchor_params(seed: int) -> dict[str, torch.Tensor]:
    random_model = g165.load_random_init(seed)
    random_model.eval()
    for param in random_model.parameters():
        param.requires_grad_(False)
    params = snapshot_params_cpu(random_model)
    del random_model
    cleanup_cuda()
    return params


def stage_params_to_device(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.to(g165.DEVICE) for name, tensor in params.items()}


def infer_hidden_size(params: dict[str, torch.Tensor]) -> int:
    emb = params.get("model.embed_tokens.weight")
    if emb is not None and emb.ndim == 2:
        return int(emb.shape[1])

    dim_counts: dict[int, int] = {}
    for tensor in params.values():
        for dim in tensor.shape:
            if dim >= 128:
                dim_counts[int(dim)] = dim_counts.get(int(dim), 0) + 1
    if not dim_counts:
        raise RuntimeError("could not infer hidden size from donor parameters")
    return max(dim_counts.items(), key=lambda item: item[1])[0]


_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def layer_key_for_param(name: str) -> str:
    match = _LAYER_RE.search(name)
    if match:
        return f"layer_{match.group(1)}"
    return "global"


def build_permuted_anchor_params(
    trained_params: dict[str, torch.Tensor],
    *,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    hidden_size = infer_hidden_size(trained_params)
    rng = np.random.default_rng(seed)
    perms: dict[str, torch.Tensor] = {}
    permuted: dict[str, torch.Tensor] = {}
    applied_axes: dict[str, list[int]] = {}

    def get_perm(key: str) -> torch.Tensor:
        if key not in perms:
            perms[key] = torch.as_tensor(rng.permutation(hidden_size).astype(np.int64), dtype=torch.long)
        return perms[key]

    for name, tensor in trained_params.items():
        key = layer_key_for_param(name)
        perm = get_perm(key)
        out = tensor
        axes: list[int] = []
        for dim, size in enumerate(tensor.shape):
            if int(size) == hidden_size:
                out = out.index_select(dim, perm)
                axes.append(dim)
        permuted[name] = out.contiguous().clone()
        if axes:
            applied_axes[name] = axes

    metadata = {
        "seed": seed,
        "hidden_size": hidden_size,
        "permutation_keys": sorted(perms.keys()),
        "n_permutation_keys": len(perms),
        "n_tensors_with_hidden_axis_permuted": len(applied_axes),
        "permutation_policy": (
            "independent random permutation for each model.layers.N group; "
            "one global permutation for embeddings, lm_head, and non-layer tensors; "
            "applied to every parameter axis whose length equals hidden_size"
        ),
        "applied_axes_by_param": applied_axes,
    }
    return permuted, metadata


def anchor_state_norm_sq(params: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for tensor in params.values():
        total += float((tensor.float() ** 2).sum().item())
    return total


def anchor_param_count(params: dict[str, torch.Tensor]) -> int:
    return int(sum(tensor.numel() for tensor in params.values()))


def compute_anchor_diagnostics(
    *,
    target_label: str,
    target_params_device: dict[str, torch.Tensor],
) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    n_matched = 0
    for seed in SEEDS:
        recipient = g165.load_random_init(seed)
        with torch.no_grad():
            f2 = float(g165.anchor_loss(recipient, target_params_device, submanifold="all").item())
        n_params = int(sum(p.numel() for p in recipient.parameters()))
        by_seed[str(seed)] = {
            "frobenius_sq_init_to_target": f2,
            "frobenius_init_to_target": math.sqrt(f2),
            "anchor_grad_l2_at_lambda_0": 2.0 * ANCHOR_LAMBDA_0 * math.sqrt(f2),
            "mean_sq_gap_per_recipient_param": f2 / max(n_params, 1),
        }
        n_matched = n_params
        del recipient
        cleanup_cuda()
    values = [float(by_seed[str(seed)]["frobenius_sq_init_to_target"]) for seed in SEEDS]
    return {
        "target_label": target_label,
        "lambda_0": ANCHOR_LAMBDA_0,
        "n_recipient_params": n_matched,
        "by_seed": by_seed,
        "mean_frobenius_sq": float(np.mean(values)),
        "std_frobenius_sq": float(np.std(values)),
    }


def load_part_a_data(tok) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    train_texts = g165.load_c4_texts(g165.C4_TRAIN_SEED, g165.N_TRAIN_TOKENS)
    val_texts = g165.load_c4_texts(g165.C4_VAL_SEED, g165.N_VAL_TOKENS)
    train_ids, train_mask = g165.tokenize_block(tok, train_texts, g165.SEQ_LEN)
    val_ids, val_mask = g165.tokenize_block(tok, val_texts, g165.SEQ_LEN)
    meta = {
        "train_seed": g165.C4_TRAIN_SEED,
        "val_seed": g165.C4_VAL_SEED,
        "n_train_texts": len(train_texts),
        "n_val_texts": len(val_texts),
        "train_shape": list(train_ids.shape),
        "val_shape": list(val_ids.shape),
        "seq_len": g165.SEQ_LEN,
    }
    return train_ids, train_mask, val_ids, val_mask, meta


def run_anchor_cell(
    arm: AnchorArmSpec,
    *,
    seed: int,
    target_params_device: dict[str, torch.Tensor] | None,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
    tok,
) -> dict[str, Any]:
    trajectory = g165.train_one_arm(
        arm_label=arm.label,
        lam0=arm.lambda_0,
        schedule_name=arm.schedule,
        seed=seed,
        donor_params=target_params_device if arm.use_anchor else None,
        train_ids=train_ids,
        train_mask=train_mask,
        val_ids=val_ids,
        val_mask=val_mask,
        tok=tok,
        submanifold="all",
    )
    return {
        "seed": seed,
        "arm_label": arm.label,
        "target_kind": arm.target_kind,
        "lambda_0": arm.lambda_0,
        "schedule": arm.schedule,
        "description": arm.description,
        "trajectory": trajectory,
        "final_nll": float(trajectory[-1]["nll"]),
        "initial_nll": float(trajectory[0]["nll"]),
    }


def part_a_final_nll(results: dict[str, Any], arm_label: str, seed: int) -> float:
    return float(results[arm_label][str(seed)]["final_nll"])


def build_part_a_summary(part_a: dict[str, Any]) -> dict[str, Any]:
    results = part_a["results"]
    if not all(str(seed) in results["scratch_baseline"] for seed in SEEDS):
        return {"status": "incomplete"}
    if not all(str(seed) in results[label] for label in ANCHOR_TARGET_LABELS for seed in SEEDS):
        return {"status": "incomplete"}

    arm_final_nll = {
        arm.label: metric_summary(
            {str(seed): part_a_final_nll(results, arm.label, seed) for seed in SEEDS},
            seed=174100 + idx * 10,
        )
        for idx, arm in enumerate(ANCHOR_ARMS)
    }

    vs_scratch: dict[str, Any] = {}
    for idx, label in enumerate(ANCHOR_TARGET_LABELS):
        values = {
            str(seed): part_a_final_nll(results, "scratch_baseline", seed) - part_a_final_nll(results, label, seed)
            for seed in SEEDS
        }
        vs_scratch[label] = metric_summary(values, seed=174200 + idx * 10)

    null_mean_nll = {
        label: float(arm_final_nll[label]["mean"])
        for label in ANCHOR_NULL_LABELS
    }
    best_null_label = min(null_mean_nll, key=null_mean_nll.get)
    trained_vs_best_null_values = {
        str(seed): part_a_final_nll(results, best_null_label, seed)
        - part_a_final_nll(results, "anchor_trained_donor", seed)
        for seed in SEEDS
    }
    trained_vs_best_null = metric_summary(trained_vs_best_null_values, seed=174300)

    per_seed_best_null_values = {}
    for seed in SEEDS:
        best_nll = min(part_a_final_nll(results, label, seed) for label in ANCHOR_NULL_LABELS)
        per_seed_best_null_values[str(seed)] = best_nll - part_a_final_nll(results, "anchor_trained_donor", seed)
    per_seed_best_null_positive_count = int(sum(value > 0.0 for value in per_seed_best_null_values.values()))

    trained_effect_observed = float(vs_scratch["anchor_trained_donor"]["mean"])
    observed_null_threshold = NULL_80_FRACTION * trained_effect_observed
    reference_null_threshold = NULL_80_FRACTION * REFERENCE_G165_TRAINED_GAIN_NATS
    null_80_flags = {}
    for label in ["anchor_random_donor", "anchor_permuted_donor"]:
        null_effect = float(vs_scratch[label]["mean"])
        null_80_flags[label] = {
            "effect_vs_scratch_nats": null_effect,
            "ge_80pct_observed_trained_effect": bool(
                trained_effect_observed > 0.0 and null_effect >= observed_null_threshold
            ),
            "ge_80pct_reference_g165_effect": bool(null_effect >= reference_null_threshold),
        }

    null_80_fail = any(
        row["ge_80pct_observed_trained_effect"] or row["ge_80pct_reference_g165_effect"]
        for row in null_80_flags.values()
    )
    margin_mean = float(trained_vs_best_null["mean"])
    margin_lo = float(trained_vs_best_null["ci_95_lo"])
    all_seeds_positive = per_seed_best_null_positive_count == len(SEEDS)
    pass_criteria = {
        "trained_beats_best_nontrained_by_ge_0p30_nats": margin_mean >= PASS_PART_A_MARGIN_NATS,
        "paired_bootstrap_ci_excludes_zero": margin_lo > 0.0,
        "all_three_seeds_positive_vs_per_seed_best_nontrained": all_seeds_positive,
        "no_matched_null_reaches_80pct_trained_effect": not null_80_fail,
        "trained_effect_vs_scratch_positive": trained_effect_observed > 0.0,
    }
    status = "pass" if all(pass_criteria.values()) else "fail"
    verdict = (
        "PASS: trained donor anchor beats best non-trained anchor/null "
        f"({best_null_label}) by {margin_mean:+.3f} nats with 95% CI "
        f"[{margin_lo:+.3f}, {float(trained_vs_best_null['ci_95_hi']):+.3f}]."
        if status == "pass"
        else "FAIL: anchor donor-specificity criteria not met. "
        f"trained-best_null ({best_null_label})={margin_mean:+.3f} nats, "
        f"95% CI [{margin_lo:+.3f}, {float(trained_vs_best_null['ci_95_hi']):+.3f}], "
        f"positive_seeds={per_seed_best_null_positive_count}/3, null_80_fail={null_80_fail}."
    )

    return {
        "status": status,
        "verdict": verdict,
        "arm_final_nll": arm_final_nll,
        "vs_scratch_final_nll_gain": vs_scratch,
        "best_nontrained_label": best_null_label,
        "trained_vs_best_nontrained_nll_gain": trained_vs_best_null,
        "per_seed_best_nontrained_margin_nats": per_seed_best_null_values,
        "positive_direction_seeds_vs_per_seed_best_nontrained": per_seed_best_null_positive_count,
        "null_80_flags": null_80_flags,
        "observed_trained_effect_vs_scratch_nats": trained_effect_observed,
        "observed_null_80_threshold_nats": observed_null_threshold,
        "reference_null_80_threshold_nats": reference_null_threshold,
        "criteria": pass_criteria,
    }


def run_part_a(payload: dict[str, Any], *, t_start: float, force_target_npz: bool) -> None:
    print("\n=== PART A: weight-anchor matched null ===")
    trained_params, tok = load_trained_anchor_params()
    payload["part_a"]["trained_anchor_param_count"] = anchor_param_count(trained_params)
    payload["part_a"]["trained_anchor_norm_sq"] = anchor_state_norm_sq(trained_params)

    train_ids, train_mask, val_ids, val_mask, data_meta = load_part_a_data(tok)
    payload["part_a"]["data"] = data_meta
    write_payload(payload, t_start=t_start, incremental=True)

    scratch_arm = ANCHOR_ARMS[0]
    for seed in SEEDS:
        if cell_done(payload, "part_a", scratch_arm.label, seed):
            print(f"  skip complete cell part_a {scratch_arm.label} seed={seed}")
            continue
        print(f"\n--- part_a arm={scratch_arm.label} seed={seed} ---")
        payload["part_a"]["results"][scratch_arm.label][str(seed)] = run_anchor_cell(
            scratch_arm,
            seed=seed,
            target_params_device=None,
            train_ids=train_ids,
            train_mask=train_mask,
            val_ids=val_ids,
            val_mask=val_mask,
            tok=tok,
        )
        write_payload(payload, t_start=t_start, incremental=True)

    for arm in ANCHOR_ARMS[1:]:
        if all(cell_done(payload, "part_a", arm.label, seed) for seed in SEEDS):
            print(f"  skip complete arm part_a {arm.label}")
            continue

        if arm.target_kind == "trained":
            target_params_cpu = trained_params
            target_metadata = {
                "target_kind": "trained",
                "source_model": g165._MODEL_ID,
                "construction": "trained donor snapshot via genome_165.load_trained_donor",
            }
        elif arm.target_kind == "random":
            target_params_cpu = load_random_anchor_params(RANDOM_DONOR_SEED)
            target_metadata = {
                "target_kind": "random",
                "source_model_arch": g165._MODEL_ID,
                "random_seed": RANDOM_DONOR_SEED,
                "construction": "AutoModelForCausalLM.from_config random init via genome_165.load_random_init",
            }
            npz_path = TARGET_DIR / f"genome_174_anchor_random_donor_seed{RANDOM_DONOR_SEED}.npz"
            payload["part_a"]["target_npz"][arm.label] = save_state_npz(
                npz_path,
                target_params_cpu,
                metadata=target_metadata,
                force=force_target_npz,
            )
            write_payload(payload, t_start=t_start, incremental=True)
        elif arm.target_kind == "permuted":
            target_params_cpu, perm_metadata = build_permuted_anchor_params(
                trained_params,
                seed=PERMUTED_DONOR_SEED,
            )
            target_metadata = {
                "target_kind": "permuted",
                "source_model": g165._MODEL_ID,
                "source": "trained donor snapshot",
                **perm_metadata,
            }
            npz_path = TARGET_DIR / f"genome_174_anchor_permuted_donor_seed{PERMUTED_DONOR_SEED}.npz"
            payload["part_a"]["target_npz"][arm.label] = save_state_npz(
                npz_path,
                target_params_cpu,
                metadata=target_metadata,
                force=force_target_npz,
            )
            write_payload(payload, t_start=t_start, incremental=True)
        else:
            raise ValueError(f"unknown anchor target kind: {arm.target_kind}")

        target_params_device = stage_params_to_device(target_params_cpu)
        if arm.label not in payload["part_a"]["anchor_diagnostics"]:
            payload["part_a"]["anchor_diagnostics"][arm.label] = compute_anchor_diagnostics(
                target_label=arm.label,
                target_params_device=target_params_device,
            )
            write_payload(payload, t_start=t_start, incremental=True)

        for seed in SEEDS:
            if cell_done(payload, "part_a", arm.label, seed):
                print(f"  skip complete cell part_a {arm.label} seed={seed}")
                continue
            print(f"\n--- part_a arm={arm.label} seed={seed} ---")
            payload["part_a"]["results"][arm.label][str(seed)] = run_anchor_cell(
                arm,
                seed=seed,
                target_params_device=target_params_device,
                train_ids=train_ids,
                train_mask=train_mask,
                val_ids=val_ids,
                val_mask=val_mask,
                tok=tok,
            )
            payload["summary"]["part_a"] = build_part_a_summary(payload["part_a"])
            write_payload(payload, t_start=t_start, incremental=True)

        del target_params_device
        if arm.target_kind in {"random", "permuted"}:
            del target_params_cpu
        cleanup_cuda()

    del trained_params
    cleanup_cuda()
    payload["summary"]["part_a"] = build_part_a_summary(payload["part_a"])
    write_payload(payload, t_start=t_start, incremental=True)


def estimate_topk_cache_bytes(n_windows: int, seq_len: int, top_k: int) -> int:
    return g167.estimate_teacher_cache_bytes(n_windows, seq_len, top_k)


def precompute_random_teacher_topk_cache(
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    train_signature: str,
) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / (
        f"genome_174_random_teacher_topk_cache_{train_signature[:16]}_seed{RANDOM_TEACHER_SEED}.pt"
    )
    print(
        f"  precomputing fresh random-teacher top-{g167.KD_TOPK} cache over "
        f"{train_ids.shape[0]} windows"
    )

    teacher = g165.load_random_init(RANDOM_TEACHER_SEED)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    n_windows = int(train_ids.shape[0])
    n_pos = int(train_ids.shape[1]) - 1
    topk_idx = torch.empty((n_windows, n_pos, g167.KD_TOPK), dtype=torch.int32)
    topk_logits = torch.empty((n_windows, n_pos, g167.KD_TOPK), dtype=torch.bfloat16)
    t0 = time.time()

    with torch.inference_mode():
        for start in range(0, n_windows, g167.TRAIN_BATCH_SIZE):
            ids = train_ids[start : start + g167.TRAIN_BATCH_SIZE].to(DEVICE)
            mask = train_mask[start : start + g167.TRAIN_BATCH_SIZE].to(DEVICE)
            with g167.autocast_context():
                logits = teacher(input_ids=ids, attention_mask=mask, use_cache=False).logits
            values, indices = logits[:, :-1].contiguous().float().topk(g167.KD_TOPK, dim=-1)
            batch_n = ids.shape[0]
            topk_idx[start : start + batch_n] = indices.to(torch.int32).cpu()
            topk_logits[start : start + batch_n] = values.to(torch.bfloat16).cpu()
            if start == 0 or (start // g167.TRAIN_BATCH_SIZE) % 200 == 0:
                print(f"    random-teacher cache {start:5d}/{n_windows} windows ({time.time() - t0:.0f}s)")

    meta = {
        "teacher_kind": "random_init_qwen3_arch",
        "teacher_hf": g167.TEACHER_HF,
        "random_teacher_seed": RANDOM_TEACHER_SEED,
        "train_signature": train_signature,
        "n_train_windows": int(train_ids.shape[0]),
        "seq_len": int(train_ids.shape[1]),
        "top_k": g167.KD_TOPK,
        "dtype_logits": "bfloat16",
        "dtype_indices": "int32",
        "fresh_inference": True,
    }
    torch.save({"meta": meta, "topk_idx": topk_idx, "topk_logits": topk_logits}, cache_path)

    del teacher
    cleanup_cuda()
    return {
        "topk_idx": topk_idx,
        "topk_logits": topk_logits,
        "path": str(cache_path),
        "cache_hit": False,
        "meta": meta,
        "estimated_bytes": estimate_topk_cache_bytes(
            int(train_ids.shape[0]),
            int(train_ids.shape[1]),
            g167.KD_TOPK,
        ),
    }


def build_uniform_topk_indices(
    *,
    n_windows: int,
    n_pos: int,
    vocab_size: int,
    top_k: int,
    seed: int,
) -> torch.Tensor:
    print(
        f"  building teacher-free uniform top-{top_k} supports "
        f"shape=({n_windows}, {n_pos}, {top_k}) seed={seed}"
    )
    rng = np.random.default_rng(seed)
    out = torch.empty((n_windows, n_pos, top_k), dtype=torch.int32)
    offsets = np.arange(top_k, dtype=np.int64).reshape(1, 1, top_k)
    chunk = 256
    for start in range(0, n_windows, chunk):
        end = min(start + chunk, n_windows)
        bases = rng.integers(0, vocab_size, size=(end - start, n_pos, 1), dtype=np.int64)
        idx = ((bases + offsets) % vocab_size).astype(np.int32, copy=False)
        out[start:end] = torch.from_numpy(idx)
    return out


def load_part_b_data(tok) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    train_ids, train_mask, train_meta = g167.load_c4_windows(
        tok,
        split="train",
        seed=g167.C4_TRAIN_SEED,
        n_windows=g167.N_TRAIN_WINDOWS,
    )
    train_hashes = g167.collect_13gram_hashes(train_ids, train_mask)
    c4_val_ids, c4_val_mask, c4_val_meta = g167.load_c4_windows(
        tok,
        split="validation",
        seed=g167.C4_VAL_SEED,
        n_windows=g167.N_C4_VAL_WINDOWS,
        forbidden_hashes=train_hashes,
    )
    wiki_val_ids, wiki_val_mask, wiki_val_meta = g167.load_wikitext_windows(
        tok,
        split="validation",
        seed=g167.WIKITEXT_VAL_SEED,
        n_windows=g167.N_WIKITEXT_VAL_WINDOWS,
        forbidden_hashes=train_hashes,
    )
    data_meta = {
        "train": train_meta,
        "c4_val": c4_val_meta,
        "wikitext_val": wiki_val_meta,
        "train_13gram_hash_count": len(train_hashes),
    }
    return train_ids, train_mask, c4_val_ids, c4_val_mask, wiki_val_ids, wiki_val_mask, data_meta


def kd_loss_for_target(
    *,
    arm: KDArmSpec,
    student_shift_logits: torch.Tensor,
    batch_index_tensor: torch.Tensor,
    target_caches: dict[str, Any],
) -> torch.Tensor:
    if arm.target_kind == "uniform":
        uniform_idx = target_caches["uniform_topk_idx"][batch_index_tensor].to(DEVICE, dtype=torch.long)
        uniform_logits = torch.zeros(uniform_idx.shape, device=DEVICE, dtype=torch.float32)
        return g167.topk_kd_loss(student_shift_logits, uniform_idx, uniform_logits)

    cache = target_caches[arm.target_kind]
    batch_topk_idx = cache["topk_idx"][batch_index_tensor].to(DEVICE, dtype=torch.long)
    batch_topk_logits = cache["topk_logits"][batch_index_tensor].to(DEVICE, dtype=torch.float32)
    return g167.topk_kd_loss(student_shift_logits, batch_topk_idx, batch_topk_logits)


def run_kd_cell(
    arm: KDArmSpec,
    *,
    seed: int,
    vocab_size: int,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    c4_val_ids: torch.Tensor,
    c4_val_mask: torch.Tensor,
    wiki_val_ids: torch.Tensor,
    wiki_val_mask: torch.Tensor,
    train_schedule: np.ndarray,
    target_caches: dict[str, Any],
) -> dict[str, Any]:
    g167.set_seed(seed)
    model = g167.make_minimal_student(vocab_size=vocab_size, seed=seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=g167.LR,
        betas=g167.BETAS,
        weight_decay=g167.WEIGHT_DECAY,
    )
    n_total_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_log: list[dict[str, Any]] = []
    observed_kd_steps = 0
    t0 = time.time()

    print(
        f"  {arm.label} seed={seed} params={n_total_params / 1e6:.2f}M "
        f"target={arm.target_kind}"
    )

    for step in range(1, g167.TRAIN_STEPS + 1):
        current_lr = g167.warmup_lr(step - 1, g167.LR, g167.LR_WARMUP_STEPS)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        batch_indices = train_schedule[step - 1]
        batch_index_tensor = torch.as_tensor(batch_indices, dtype=torch.long)
        ids = train_ids[batch_index_tensor].to(DEVICE)
        mask = train_mask[batch_index_tensor].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with g167.autocast_context():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            ce_loss = g167.causal_ce_loss(logits, ids, mask)
            kd_loss_value = None
            if arm.use_kd:
                kd_loss_value = kd_loss_for_target(
                    arm=arm,
                    student_shift_logits=logits[:, :-1].contiguous().float(),
                    batch_index_tensor=batch_index_tensor,
                    target_caches=target_caches,
                )
                total_loss = (1.0 - g167.KD_GAMMA) * ce_loss + g167.KD_GAMMA * kd_loss_value
                observed_kd_steps += 1
            else:
                total_loss = ce_loss

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"non-finite loss at step {step} arm={arm.label} seed={seed}")

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), g167.GRAD_CLIP)
        optimizer.step()

        if step % g167.LOG_EVERY == 0 or step == g167.TRAIN_STEPS:
            row = {
                "step": step,
                "lr": float(current_lr),
                "ce_loss": float(ce_loss.item()),
                "kd_loss": float(kd_loss_value.item()) if kd_loss_value is not None else None,
                "total_loss": float(total_loss.item()),
                "elapsed_s": time.time() - t0,
            }
            train_log.append(row)
            kd_text = "na" if row["kd_loss"] is None else f"{row['kd_loss']:.4f}"
            print(
                f"    step={step:5d} ce={row['ce_loss']:.4f} "
                f"kd={kd_text} total={row['total_loss']:.4f} ({row['elapsed_s']:.0f}s)"
            )

    expected_kd_steps = g167.TRAIN_STEPS if arm.use_kd else 0
    if observed_kd_steps != expected_kd_steps:
        raise RuntimeError(
            f"KD step mismatch arm={arm.label}: observed {observed_kd_steps}, expected {expected_kd_steps}"
        )

    final_metrics = {
        "c4_val": g167.evaluate_model(model, c4_val_ids, c4_val_mask),
        "wikitext_val": g167.evaluate_model(model, wiki_val_ids, wiki_val_mask),
    }
    wallclock_s = time.time() - t0
    print(
        f"    final c4_val: nll={final_metrics['c4_val']['nll']:.4f} "
        f"top1={100.0 * final_metrics['c4_val']['top1_acc']:.2f}%"
    )
    print(
        f"    final wikitext_val: nll={final_metrics['wikitext_val']['nll']:.4f} "
        f"top1={100.0 * final_metrics['wikitext_val']['top1_acc']:.2f}%"
    )

    payload = {
        "seed": seed,
        "arm_label": arm.label,
        "target_kind": arm.target_kind,
        "description": arm.description,
        "use_kd": arm.use_kd,
        "kd_step_count": observed_kd_steps,
        "n_total_params": int(n_total_params),
        "n_trainable_params": int(n_trainable_params),
        "train_log": train_log,
        "final_metrics": final_metrics,
        "wallclock_s": wallclock_s,
    }

    del model, optimizer
    cleanup_cuda()
    return payload


def part_b_metric(results: dict[str, Any], arm_label: str, seed: int, dataset: str, metric: str) -> float:
    return float(results[arm_label][str(seed)]["final_metrics"][dataset][metric])


def summarize_kd_pairwise(
    results: dict[str, Any],
    better_arm: str,
    worse_arm: str,
    *,
    seed_base: int,
) -> dict[str, Any]:
    c4_top1_delta_pp = {}
    c4_nll_gain = {}
    wiki_top1_delta_pp = {}
    wiki_nll_gain = {}
    for seed in SEEDS:
        seed_key = str(seed)
        c4_top1_delta_pp[seed_key] = (
            part_b_metric(results, better_arm, seed, "c4_val", "top1_acc")
            - part_b_metric(results, worse_arm, seed, "c4_val", "top1_acc")
        ) * 100.0
        c4_nll_gain[seed_key] = (
            part_b_metric(results, worse_arm, seed, "c4_val", "nll")
            - part_b_metric(results, better_arm, seed, "c4_val", "nll")
        )
        wiki_top1_delta_pp[seed_key] = (
            part_b_metric(results, better_arm, seed, "wikitext_val", "top1_acc")
            - part_b_metric(results, worse_arm, seed, "wikitext_val", "top1_acc")
        ) * 100.0
        wiki_nll_gain[seed_key] = (
            part_b_metric(results, worse_arm, seed, "wikitext_val", "nll")
            - part_b_metric(results, better_arm, seed, "wikitext_val", "nll")
        )
    return {
        "better_arm": better_arm,
        "worse_arm": worse_arm,
        "c4_val_top1_delta_pp": metric_summary(c4_top1_delta_pp, seed=seed_base),
        "c4_val_nll_gain": metric_summary(c4_nll_gain, seed=seed_base + 1),
        "wikitext_val_top1_delta_pp": metric_summary(wiki_top1_delta_pp, seed=seed_base + 2),
        "wikitext_val_nll_gain": metric_summary(wiki_nll_gain, seed=seed_base + 3),
        "positive_primary_effect_seeds": int(sum(float(c4_top1_delta_pp[str(s)]) > 0.0 for s in SEEDS)),
    }


def build_part_b_summary(part_b: dict[str, Any]) -> dict[str, Any]:
    results = part_b["results"]
    if not all(str(seed) in results["scratch_ce"] for seed in SEEDS):
        return {"status": "incomplete"}
    if not all(str(seed) in results[label] for label in KD_TARGET_LABELS for seed in SEEDS):
        return {"status": "incomplete"}

    arm_final_metrics = {}
    for idx, arm in enumerate(KD_ARMS):
        arm_final_metrics[arm.label] = {
            "c4_val_top1_acc": metric_summary(
                {str(seed): part_b_metric(results, arm.label, seed, "c4_val", "top1_acc") for seed in SEEDS},
                seed=174400 + idx * 10,
            ),
            "c4_val_nll": metric_summary(
                {str(seed): part_b_metric(results, arm.label, seed, "c4_val", "nll") for seed in SEEDS},
                seed=174401 + idx * 10,
            ),
            "wikitext_val_top1_acc": metric_summary(
                {
                    str(seed): part_b_metric(results, arm.label, seed, "wikitext_val", "top1_acc")
                    for seed in SEEDS
                },
                seed=174402 + idx * 10,
            ),
            "wikitext_val_nll": metric_summary(
                {str(seed): part_b_metric(results, arm.label, seed, "wikitext_val", "nll") for seed in SEEDS},
                seed=174403 + idx * 10,
            ),
        }

    vs_scratch = {
        label: summarize_kd_pairwise(results, label, "scratch_ce", seed_base=174500 + idx * 10)
        for idx, label in enumerate(KD_TARGET_LABELS)
    }

    null_mean_top1 = {
        label: float(arm_final_metrics[label]["c4_val_top1_acc"]["mean"])
        for label in KD_NULL_LABELS
    }
    best_null_label = max(null_mean_top1, key=null_mean_top1.get)
    trained_vs_best_null = summarize_kd_pairwise(
        results,
        "kd_trained_teacher",
        best_null_label,
        seed_base=174600,
    )

    per_seed_best_null_values = {}
    for seed in SEEDS:
        best_top1 = max(part_b_metric(results, label, seed, "c4_val", "top1_acc") for label in KD_NULL_LABELS)
        per_seed_best_null_values[str(seed)] = (
            part_b_metric(results, "kd_trained_teacher", seed, "c4_val", "top1_acc") - best_top1
        ) * 100.0
    per_seed_best_null_positive_count = int(sum(value > 0.0 for value in per_seed_best_null_values.values()))

    trained_effect_observed = float(vs_scratch["kd_trained_teacher"]["c4_val_top1_delta_pp"]["mean"])
    observed_null_threshold = NULL_80_FRACTION * trained_effect_observed
    reference_null_threshold = NULL_80_FRACTION * REFERENCE_G167_TRAINED_GAIN_PP
    null_80_flags = {}
    for label in ["kd_uniform_target", "kd_random_teacher"]:
        null_effect = float(vs_scratch[label]["c4_val_top1_delta_pp"]["mean"])
        null_80_flags[label] = {
            "effect_vs_scratch_pp": null_effect,
            "ge_80pct_observed_trained_effect": bool(
                trained_effect_observed > 0.0 and null_effect >= observed_null_threshold
            ),
            "ge_80pct_reference_g167_effect": bool(null_effect >= reference_null_threshold),
        }

    null_80_fail = any(
        row["ge_80pct_observed_trained_effect"] or row["ge_80pct_reference_g167_effect"]
        for row in null_80_flags.values()
    )
    primary = trained_vs_best_null["c4_val_top1_delta_pp"]
    margin_mean = float(primary["mean"])
    margin_lo = float(primary["ci_95_lo"])
    all_seeds_positive = per_seed_best_null_positive_count == len(SEEDS)
    pass_criteria = {
        "trained_beats_best_nontrained_by_ge_0p30_pp": margin_mean >= PASS_PART_B_MARGIN_PP,
        "paired_bootstrap_ci_excludes_zero": margin_lo > 0.0,
        "all_three_seeds_positive_vs_per_seed_best_nontrained": all_seeds_positive,
        "no_matched_null_reaches_80pct_trained_effect": not null_80_fail,
        "trained_effect_vs_scratch_positive": trained_effect_observed > 0.0,
    }
    status = "pass" if all(pass_criteria.values()) else "fail"
    verdict = (
        "PASS: trained-teacher KD beats best non-trained KD/null "
        f"({best_null_label}) by {margin_mean:+.3f} pp C4 top1 with 95% CI "
        f"[{margin_lo:+.3f}, {float(primary['ci_95_hi']):+.3f}]."
        if status == "pass"
        else "FAIL: KD donor-specificity criteria not met. "
        f"trained-best_null ({best_null_label})={margin_mean:+.3f} pp, "
        f"95% CI [{margin_lo:+.3f}, {float(primary['ci_95_hi']):+.3f}], "
        f"positive_seeds={per_seed_best_null_positive_count}/3, null_80_fail={null_80_fail}."
    )

    return {
        "status": status,
        "verdict": verdict,
        "arm_final_metrics": arm_final_metrics,
        "vs_scratch": vs_scratch,
        "best_nontrained_label": best_null_label,
        "trained_vs_best_nontrained": trained_vs_best_null,
        "per_seed_best_nontrained_margin_pp": per_seed_best_null_values,
        "positive_direction_seeds_vs_per_seed_best_nontrained": per_seed_best_null_positive_count,
        "null_80_flags": null_80_flags,
        "observed_trained_effect_vs_scratch_pp": trained_effect_observed,
        "observed_null_80_threshold_pp": observed_null_threshold,
        "reference_null_80_threshold_pp": reference_null_threshold,
        "criteria": pass_criteria,
    }


def run_part_b(payload: dict[str, Any], *, t_start: float) -> None:
    print("\n=== PART B: KD matched null ===")
    tok = g167.load_tokenizer()
    vocab_size = len(tok)
    payload["part_b"]["vocab_size"] = vocab_size

    train_ids, train_mask, c4_val_ids, c4_val_mask, wiki_val_ids, wiki_val_mask, data_meta = load_part_b_data(tok)
    payload["part_b"]["data"] = data_meta
    train_signature = g167.tensor_sha1(train_ids)
    payload["part_b"]["train_signature"] = train_signature
    write_payload(payload, t_start=t_start, incremental=True)

    teacher, _ = g167.load_trained_teacher(tok)
    payload["part_b"]["teacher_reference"] = {
        "c4_val": g167.evaluate_model(teacher, c4_val_ids, c4_val_mask),
        "wikitext_val": g167.evaluate_model(teacher, wiki_val_ids, wiki_val_mask),
    }
    trained_cache = g167.precompute_teacher_topk_cache(
        teacher,
        train_ids,
        train_mask,
        train_signature=train_signature,
    )
    payload["part_b"]["cache_meta"]["trained_teacher"] = {
        "path": trained_cache["path"],
        "cache_hit": trained_cache["cache_hit"],
        "estimated_bytes": trained_cache["estimated_bytes"],
        "estimated_gib": trained_cache["estimated_bytes"] / (1024**3),
        "train_signature": train_signature,
        "shared_with_genome_167": True,
    }
    del teacher
    cleanup_cuda()
    write_payload(payload, t_start=t_start, incremental=True)

    random_cache = precompute_random_teacher_topk_cache(
        train_ids,
        train_mask,
        train_signature=train_signature,
    )
    payload["part_b"]["cache_meta"]["random_teacher"] = {
        "path": random_cache["path"],
        "cache_hit": random_cache["cache_hit"],
        "estimated_bytes": random_cache["estimated_bytes"],
        "estimated_gib": random_cache["estimated_bytes"] / (1024**3),
        "train_signature": train_signature,
        "random_teacher_seed": RANDOM_TEACHER_SEED,
        "fresh_inference": True,
    }
    write_payload(payload, t_start=t_start, incremental=True)

    uniform_topk_idx = build_uniform_topk_indices(
        n_windows=int(train_ids.shape[0]),
        n_pos=int(train_ids.shape[1]) - 1,
        vocab_size=vocab_size,
        top_k=g167.KD_TOPK,
        seed=UNIFORM_TARGET_SEED,
    )
    payload["part_b"]["cache_meta"]["uniform_target"] = {
        "target_kind": "uniform",
        "support_policy": "teacher-free contiguous random support per train window and shifted position",
        "uniform_target_seed": UNIFORM_TARGET_SEED,
        "top_k": g167.KD_TOPK,
        "logits": "zeros generated on device at training time",
        "topk_idx_shape": list(uniform_topk_idx.shape),
        "estimated_idx_bytes": int(uniform_topk_idx.numel() * uniform_topk_idx.element_size()),
    }
    write_payload(payload, t_start=t_start, incremental=True)

    target_caches = {
        "trained": trained_cache,
        "random": random_cache,
        "uniform_topk_idx": uniform_topk_idx,
    }

    print(f"\n=== Running Part B {len(KD_ARMS)} arms x {len(SEEDS)} seeds ===")
    print("=== Pairing rule: same seed -> same student init and same batch schedule across all KD arms ===")
    for seed in SEEDS:
        train_schedule = g167.build_train_schedule(seed, n_examples=int(train_ids.shape[0]))
        for arm in KD_ARMS:
            if cell_done(payload, "part_b", arm.label, seed):
                print(f"  skip complete cell part_b {arm.label} seed={seed}")
                continue
            print(f"\n--- part_b arm={arm.label} seed={seed} ---")
            payload["part_b"]["results"][arm.label][str(seed)] = run_kd_cell(
                arm,
                seed=seed,
                vocab_size=vocab_size,
                train_ids=train_ids,
                train_mask=train_mask,
                c4_val_ids=c4_val_ids,
                c4_val_mask=c4_val_mask,
                wiki_val_ids=wiki_val_ids,
                wiki_val_mask=wiki_val_mask,
                train_schedule=train_schedule,
                target_caches=target_caches,
            )
            payload["summary"]["part_b"] = build_part_b_summary(payload["part_b"])
            write_payload(payload, t_start=t_start, incremental=True)

    payload["summary"]["part_b"] = build_part_b_summary(payload["part_b"])
    write_payload(payload, t_start=t_start, incremental=True)


def build_overall_summary(payload: dict[str, Any]) -> dict[str, Any]:
    part_a = build_part_a_summary(payload["part_a"])
    part_b = build_part_b_summary(payload["part_b"])
    status_a = part_a.get("status")
    status_b = part_b.get("status")

    if status_a == "pass" and status_b == "pass":
        verdict = (
            "PASS: donor-specificity locked in both matched-null factorials. "
            f"Part A: {part_a['verdict']} Part B: {part_b['verdict']}"
        )
        status = "pass"
    elif status_a == "incomplete" or status_b == "incomplete":
        verdict = "INCOMPLETE: at least one part is not fully run."
        status = "incomplete"
    else:
        verdict = (
            "FAIL: donor-specificity not locked under matched-null criteria. "
            f"Part A status={status_a}; Part B status={status_b}."
        )
        status = "fail"

    return {
        "status": status,
        "verdict": verdict,
        "part_a": part_a,
        "part_b": part_b,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genome 174 donor-specificity matched-null control.")
    parser.add_argument(
        "--part",
        choices=["all", "anchor", "kd"],
        default="all",
        help="Run both parts, only Part A anchor controls, or only Part B KD controls.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing result JSON and start a fresh payload.",
    )
    parser.add_argument(
        "--force-target-npz",
        action="store_true",
        help="Rewrite random/permuted donor target NPZ files even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_locked_protocol()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("genome_174: donor-specificity matched-null control")
    print(f"  model={g165._MODEL_ID}")
    print(f"  device={DEVICE} forward_dtype={FORWARD_DTYPE}")
    print(f"  seeds={SEEDS}")
    print(f"  Part A: lambda_0={ANCHOR_LAMBDA_0} schedule={ANCHOR_SCHEDULE} steps={g165.N_STEPS}")
    print(f"  Part B: topk={g167.KD_TOPK} T={g167.KD_TEMP} gamma={g167.KD_GAMMA} steps={g167.TRAIN_STEPS}")
    print(f"  output={OUT_PATH}")

    t_start = time.time()
    payload = load_or_create_payload(resume=not args.no_resume)

    if args.part in {"all", "anchor"}:
        if all_cells_done(payload, "part_a", ANCHOR_ARMS):
            print("  Part A already complete in result JSON; skipping.")
        else:
            run_part_a(payload, t_start=t_start, force_target_npz=args.force_target_npz)

    if args.part in {"all", "kd"}:
        if all_cells_done(payload, "part_b", KD_ARMS):
            print("  Part B already complete in result JSON; skipping.")
        else:
            run_part_b(payload, t_start=t_start)

    overall = build_overall_summary(payload)
    payload["summary"] = overall
    payload["verdict"] = overall["verdict"]
    write_payload(payload, t_start=t_start, incremental=False)
    print(f"\n=== verdict: {payload['verdict']} ===")
    print(f"Saved: {OUT_PATH} ({payload['elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()
