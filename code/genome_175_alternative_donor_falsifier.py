"""
genome_175_alternative_donor_falsifier.py

Donor-identity specificity falsifier for the g174 Part A weight-anchor result.

Question
--------
g174 Part A locked trained-structure specificity against random/permuted nulls,
but it did not prove Qwen3 donor-identity specificity. This experiment anchors
the same random-init Qwen3-0.6B-shape recipient to a different trained model of
the same architecture. If that alternative trained donor gets most of the true
Qwen3 donor's +1.087 nats, the active ingredient is a generic trained-like
weight basin rather than Qwen3-specific identity.

Design
------
Alternative donor implementation: Option A.
  - Initialize a Qwen3-0.6B-architecture model from scratch with seed 1234.
  - Train it for 3000 steps on Wikitext-103 train.
  - Save the resulting FP32 parameter snapshot to NPZ for reproducibility.

Main arms, matched to g165/g174 Part A:
  1. scratch_baseline
  2. anchor_qwen3_donor
  3. anchor_alternative_trained_donor

All main cells use seeds [42, 7, 13], 500 steps, batch 8, seq_len 256,
lr 3e-4, BF16 model forward, and lambda=0.01 constant Frobenius anchor.

Outputs
-------
  - results/genome_175_alternative_donor_falsifier.json
  - results/genome_175_targets/genome_175_alternative_trained_donor_optionA_*.npz
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
import zipfile
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer


CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import genome_165_annealed_donor as g165
import genome_167_kd_canonical as g167


OUT_PATH = ROOT / "results" / "genome_175_alternative_donor_falsifier.json"
TARGET_DIR = ROOT / "results" / "genome_175_targets"

SEEDS = [42, 7, 13]

ANCHOR_LAMBDA_0 = 0.01
ANCHOR_SCHEDULE = "constant"

ALT_DONOR_OPTION = "option_a_wikitext103_scratch_pretrain"
ALT_DONOR_SEED = 1234
ALT_DONOR_STEPS = 3000
ALT_DONOR_BATCH_SIZE = g165.BATCH_SIZE
ALT_DONOR_LR = g165.LR
ALT_DONOR_BETAS = (0.9, 0.95)
ALT_DONOR_WEIGHT_DECAY = 0.01
ALT_DONOR_GRAD_CLIP = 1.0
ALT_DONOR_LOG_EVERY = 250
ALT_DONOR_DATASET_NAME = "wikitext"
ALT_DONOR_DATASET_FALLBACK = "Salesforce/wikitext"
ALT_DONOR_DATASET_CONFIG = "wikitext-103-raw-v1"
ALT_DONOR_SPLIT = "train"

REFERENCE_QWEN3_DONOR_GAIN_NATS = 1.087
NULL_80_FRACTION = 0.80
PASS_IDENTITY_MARGIN_NATS = 0.50
FAIL_IDENTITY_MARGIN_NATS = 0.20
N_BOOT = 10_000

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
class AnchorArmSpec:
    label: str
    target_kind: str
    lambda_0: float
    schedule: str
    description: str

    @property
    def use_anchor(self) -> bool:
        return self.target_kind != "none" and self.lambda_0 > 0.0


ARMS = [
    AnchorArmSpec(
        label="scratch_baseline",
        target_kind="none",
        lambda_0=0.0,
        schedule=ANCHOR_SCHEDULE,
        description="No anchor; exact g174 Part A scratch baseline protocol.",
    ),
    AnchorArmSpec(
        label="anchor_qwen3_donor",
        target_kind="qwen3_trained",
        lambda_0=ANCHOR_LAMBDA_0,
        schedule=ANCHOR_SCHEDULE,
        description="Constant Frobenius anchor to the original trained Qwen3-0.6B donor.",
    ),
    AnchorArmSpec(
        label="anchor_alternative_trained_donor",
        target_kind="alternative_trained",
        lambda_0=ANCHOR_LAMBDA_0,
        schedule=ANCHOR_SCHEDULE,
        description=(
            "Constant Frobenius anchor to a separately trained Qwen3-0.6B-shape "
            "model pretrained from scratch on Wikitext-103 train."
        ),
    ),
]

ARM_LABELS = [arm.label for arm in ARMS]
ANCHOR_LABELS = ["anchor_qwen3_donor", "anchor_alternative_trained_donor"]


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context():
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def assert_locked_protocol() -> None:
    checks = {
        "g165 seeds": (list(g165.SEEDS), SEEDS),
        "g165 seq_len": (g165.SEQ_LEN, 256),
        "g165 batch_size": (g165.BATCH_SIZE, 8),
        "g165 n_steps": (g165.N_STEPS, 500),
        "g165 lr": (g165.LR, 3e-4),
        "g165 eval_every": (g165.EVAL_EVERY, 25),
        "g167 seq_len": (g167.SEQ_LEN, g165.SEQ_LEN),
        "alt donor batch": (ALT_DONOR_BATCH_SIZE, g165.BATCH_SIZE),
        "alt donor lr": (ALT_DONOR_LR, g165.LR),
        "anchor lambda": (ANCHOR_LAMBDA_0, 0.01),
        "anchor schedule": (ANCHOR_SCHEDULE, "constant"),
    }
    mismatches = [
        f"{name}: observed={observed!r} expected={expected!r}"
        for name, (observed, expected) in checks.items()
        if observed != expected
    ]
    if mismatches:
        raise RuntimeError("locked protocol drift detected:\n  " + "\n  ".join(mismatches))


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(g165._MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def base_payload() -> dict[str, Any]:
    return {
        "genome": 175,
        "name": "alternative_donor_identity_falsifier",
        "timestamp_utc_started": now_utc(),
        "model_id": g165._MODEL_ID,
        "device": DEVICE,
        "forward_dtype": str(FORWARD_DTYPE),
        "config": {
            "seeds": SEEDS,
            "protocol_source": "g165/g174 Part A",
            "main": {
                "steps": g165.N_STEPS,
                "batch_size": g165.BATCH_SIZE,
                "seq_len": g165.SEQ_LEN,
                "lr": g165.LR,
                "eval_every": g165.EVAL_EVERY,
                "lambda_0": ANCHOR_LAMBDA_0,
                "schedule": ANCHOR_SCHEDULE,
                "reference_qwen3_donor_gain_nats": REFERENCE_QWEN3_DONOR_GAIN_NATS,
                "null_80_fraction": NULL_80_FRACTION,
            },
            "alternative_donor": {
                "implementation_option": ALT_DONOR_OPTION,
                "seed": ALT_DONOR_SEED,
                "steps": ALT_DONOR_STEPS,
                "batch_size": ALT_DONOR_BATCH_SIZE,
                "seq_len": g165.SEQ_LEN,
                "lr": ALT_DONOR_LR,
                "betas": list(ALT_DONOR_BETAS),
                "weight_decay": ALT_DONOR_WEIGHT_DECAY,
                "grad_clip": ALT_DONOR_GRAD_CLIP,
                "dataset_candidates": [ALT_DONOR_DATASET_NAME, ALT_DONOR_DATASET_FALLBACK],
                "dataset_config": ALT_DONOR_DATASET_CONFIG,
                "split": ALT_DONOR_SPLIT,
                "n_train_windows": ALT_DONOR_STEPS * ALT_DONOR_BATCH_SIZE,
            },
            "criteria": {
                "identity_specificity_locked": (
                    "mean(qwen3_gain - alt_gain) >= +0.5 nats and paired bootstrap "
                    "95% CI excludes zero"
                ),
                "identity_specificity_rejected": (
                    "mean(qwen3_gain - alt_gain) <= +0.2 nats or paired bootstrap "
                    "95% CI crosses zero"
                ),
                "intermediate": "mean margin in (+0.2, +0.5) nats with CI excluding zero",
            },
        },
        "arms": [arm.__dict__ for arm in ARMS],
        "alternative_donor": {},
        "data": {},
        "target_npz": {},
        "anchor_diagnostics": {},
        "results": {arm.label: {} for arm in ARMS},
        "summary": {},
        "verdict": "INCOMPLETE",
    }


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("alternative_donor", {})
    payload.setdefault("data", {})
    payload.setdefault("target_npz", {})
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
    return str(seed) in payload["results"].get(arm_label, {})


def all_cells_done(payload: dict[str, Any], arm_label: str) -> bool:
    return all(cell_done(payload, arm_label, seed) for seed in SEEDS)


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


def load_state_npz(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    params: dict[str, torch.Tensor] = {}
    with np.load(path, allow_pickle=False) as data:
        names = [str(x) for x in data["names"].tolist()]
        metadata_raw = data["metadata_json"]
        metadata_json = str(metadata_raw.tolist())
        metadata = json.loads(metadata_json)
        for idx, name in enumerate(names):
            arr = data[f"arr_{idx:05d}"]
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)
            params[name] = torch.from_numpy(np.ascontiguousarray(arr)).clone()
    return params, metadata


def snapshot_params_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    raw = g165.snapshot_donor_params(model)
    params = {name: tensor.detach().cpu().contiguous() for name, tensor in raw.items()}
    del raw
    cleanup_cuda()
    return params


def stage_params_to_device(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.to(DEVICE) for name, tensor in params.items()}


def anchor_state_norm_sq(params: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for tensor in params.values():
        total += float((tensor.float() ** 2).sum().item())
    return total


def anchor_param_count(params: dict[str, torch.Tensor]) -> int:
    return int(sum(tensor.numel() for tensor in params.values()))


def validate_target_params(
    params: dict[str, torch.Tensor],
    *,
    label: str,
    reference_seed: int = SEEDS[0],
) -> dict[str, Any]:
    recipient = g165.load_random_init(reference_seed)
    model_shapes = {name: tuple(param.shape) for name, param in recipient.named_parameters()}
    target_shapes = {name: tuple(tensor.shape) for name, tensor in params.items()}

    missing = sorted(set(model_shapes) - set(target_shapes))
    extra = sorted(set(target_shapes) - set(model_shapes))
    shape_mismatch = sorted(
        {
            name: {"model": list(model_shapes[name]), "target": list(target_shapes[name])}
            for name in set(model_shapes).intersection(target_shapes)
            if model_shapes[name] != target_shapes[name]
        }.items()
    )

    n_model_params = int(sum(param.numel() for param in recipient.parameters()))
    n_target_matched_params = int(
        sum(params[name].numel() for name in model_shapes if name in params and tuple(params[name].shape) == model_shapes[name])
    )
    del recipient
    cleanup_cuda()

    if missing or shape_mismatch:
        preview = {
            "missing_count": len(missing),
            "missing_first10": missing[:10],
            "shape_mismatch_count": len(shape_mismatch),
            "shape_mismatch_first10": shape_mismatch[:10],
            "extra_count": len(extra),
            "extra_first10": extra[:10],
        }
        raise RuntimeError(f"{label}: target params do not match Qwen3 recipient architecture: {preview}")

    return {
        "label": label,
        "reference_seed": reference_seed,
        "n_model_tensors": len(model_shapes),
        "n_target_tensors": len(target_shapes),
        "n_extra_tensors": len(extra),
        "extra_tensors_first10": extra[:10],
        "n_model_params": n_model_params,
        "n_target_matched_params": n_target_matched_params,
        "all_model_params_matched": n_target_matched_params == n_model_params,
    }


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
        n_params = int(sum(param.numel() for param in recipient.parameters()))
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


def load_main_data(tok) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    train_texts = g165.load_c4_texts(g165.C4_TRAIN_SEED, g165.N_TRAIN_TOKENS)
    val_texts = g165.load_c4_texts(g165.C4_VAL_SEED, g165.N_VAL_TOKENS)
    train_ids, train_mask = g165.tokenize_block(tok, train_texts, g165.SEQ_LEN)
    val_ids, val_mask = g165.tokenize_block(tok, val_texts, g165.SEQ_LEN)
    meta = {
        "train_dataset": "allenai/c4:en:train",
        "val_dataset": "allenai/c4:en:train",
        "train_seed": g165.C4_TRAIN_SEED,
        "val_seed": g165.C4_VAL_SEED,
        "n_train_texts": len(train_texts),
        "n_val_texts": len(val_texts),
        "train_shape": list(train_ids.shape),
        "val_shape": list(val_ids.shape),
        "seq_len": g165.SEQ_LEN,
        "protocol_note": "Exact g165/g174 Part A C4 text loader and tokenizer block.",
    }
    return train_ids, train_mask, val_ids, val_mask, meta


def alternative_npz_path() -> Path:
    return (
        TARGET_DIR
        / (
            "genome_175_alternative_trained_donor_optionA_"
            f"wikitext103_seed{ALT_DONOR_SEED}_steps{ALT_DONOR_STEPS}.npz"
        )
    )


def load_alt_wikitext_train_windows(tok) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    input_ids, attention_mask, meta = g167.load_wikitext_windows(
        tok,
        split=ALT_DONOR_SPLIT,
        seed=ALT_DONOR_SEED,
        n_windows=ALT_DONOR_STEPS * ALT_DONOR_BATCH_SIZE,
    )
    meta.update(
        {
            "purpose": "alternative donor pretraining",
            "implementation_option": ALT_DONOR_OPTION,
            "expected_dataset_candidates": [ALT_DONOR_DATASET_NAME, ALT_DONOR_DATASET_FALLBACK],
        }
    )
    return input_ids, attention_mask, meta


def pretrain_alternative_donor(tok, *, payload: dict[str, Any], t_start: float) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    print("\n=== Alternative donor pretraining: Option A Wikitext-103 ===")
    train_ids, train_mask, data_meta = load_alt_wikitext_train_windows(tok)
    payload["alternative_donor"]["data"] = data_meta
    write_payload(payload, t_start=t_start, incremental=True)

    set_seed(ALT_DONOR_SEED)
    model = g165.load_random_init(ALT_DONOR_SEED)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ALT_DONOR_LR,
        betas=ALT_DONOR_BETAS,
        weight_decay=ALT_DONOR_WEIGHT_DECAY,
    )
    n_total_params = int(sum(param.numel() for param in model.parameters()))
    n_train = int(train_ids.shape[0])
    if n_train < ALT_DONOR_STEPS * ALT_DONOR_BATCH_SIZE:
        raise RuntimeError(
            f"alternative donor train window shortfall: got {n_train}, "
            f"need {ALT_DONOR_STEPS * ALT_DONOR_BATCH_SIZE}"
        )
    perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(ALT_DONOR_SEED)).numpy()

    train_log: list[dict[str, Any]] = []
    t0 = time.time()
    print(
        f"  init_seed={ALT_DONOR_SEED} steps={ALT_DONOR_STEPS} "
        f"batch={ALT_DONOR_BATCH_SIZE} params={n_total_params / 1e6:.1f}M"
    )
    for step in range(1, ALT_DONOR_STEPS + 1):
        start = (step - 1) * ALT_DONOR_BATCH_SIZE
        end = start + ALT_DONOR_BATCH_SIZE
        idx = perm[start:end]
        ids = train_ids[idx].to(DEVICE)
        mask = train_mask[idx].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            out = model(input_ids=ids, attention_mask=mask, labels=ids, use_cache=False)
            loss = out.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite alternative donor loss at step {step}: {float(loss.item())}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), ALT_DONOR_GRAD_CLIP)
        optimizer.step()

        if step % ALT_DONOR_LOG_EVERY == 0 or step == 1 or step == ALT_DONOR_STEPS:
            row = {
                "step": step,
                "train_loss": float(loss.item()),
                "grad_norm": float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm),
                "elapsed_s": time.time() - t0,
            }
            train_log.append(row)
            print(
                f"    alt_pretrain step={step:4d} loss={row['train_loss']:.4f} "
                f"grad_norm={row['grad_norm']:.3f} ({row['elapsed_s']:.0f}s)"
            )

    params = snapshot_params_cpu(model)
    del model, optimizer
    cleanup_cuda()

    metadata = {
        "target_kind": "alternative_trained",
        "implementation_option": ALT_DONOR_OPTION,
        "source_model_arch": g165._MODEL_ID,
        "init_seed": ALT_DONOR_SEED,
        "data_seed": ALT_DONOR_SEED,
        "steps": ALT_DONOR_STEPS,
        "batch_size": ALT_DONOR_BATCH_SIZE,
        "seq_len": g165.SEQ_LEN,
        "lr": ALT_DONOR_LR,
        "betas": list(ALT_DONOR_BETAS),
        "weight_decay": ALT_DONOR_WEIGHT_DECAY,
        "grad_clip": ALT_DONOR_GRAD_CLIP,
        "dataset": data_meta,
        "train_log": train_log,
        "n_total_params": n_total_params,
        "wallclock_s": time.time() - t0,
        "construction": "AutoModelForCausalLM.from_config Qwen3-0.6B architecture trained from scratch on Wikitext-103 train.",
    }
    return params, metadata


def ensure_alternative_donor_params(
    tok,
    *,
    payload: dict[str, Any],
    t_start: float,
    force: bool,
) -> dict[str, torch.Tensor]:
    path = alternative_npz_path()
    if path.exists() and not force:
        print(f"\n=== Loading existing alternative donor NPZ: {path} ===")
        params, metadata = load_state_npz(path)
        payload["target_npz"]["anchor_alternative_trained_donor"] = {
            "path": str(path),
            "exists": True,
            "loaded_existing": True,
            "size_bytes": path.stat().st_size,
            "metadata": metadata,
        }
        payload["alternative_donor"].update(
            {
                "status": "loaded_existing_npz",
                "npz_path": str(path),
                "metadata": metadata,
            }
        )
        write_payload(payload, t_start=t_start, incremental=True)
        return params

    params, metadata = pretrain_alternative_donor(tok, payload=payload, t_start=t_start)
    validation = validate_target_params(params, label="anchor_alternative_trained_donor")
    metadata["shape_validation"] = validation
    save_meta = save_state_npz(path, params, metadata=metadata, force=True)
    payload["target_npz"]["anchor_alternative_trained_donor"] = save_meta
    payload["alternative_donor"].update(
        {
            "status": "pretrained_and_saved",
            "npz_path": str(path),
            "metadata": metadata,
            "param_count": anchor_param_count(params),
            "norm_sq": anchor_state_norm_sq(params),
        }
    )
    write_payload(payload, t_start=t_start, incremental=True)
    return params


def load_qwen3_anchor_params() -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    donor, _tok = g165.load_trained_donor()
    params = snapshot_params_cpu(donor)
    del donor, _tok
    cleanup_cuda()
    metadata = {
        "target_kind": "qwen3_trained",
        "source_model": g165._MODEL_ID,
        "construction": "Original trained Qwen3-0.6B donor snapshot via genome_165.load_trained_donor.",
    }
    return params, metadata


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


def final_nll(results: dict[str, Any], arm_label: str, seed: int) -> float:
    return float(results[arm_label][str(seed)]["final_nll"])


def build_summary(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload["results"]
    if not all(str(seed) in results["scratch_baseline"] for seed in SEEDS):
        return {"status": "incomplete", "verdict": "INCOMPLETE: scratch baseline cells are not complete."}
    if not all(str(seed) in results[label] for label in ANCHOR_LABELS for seed in SEEDS):
        return {"status": "incomplete", "verdict": "INCOMPLETE: anchored cells are not complete."}

    arm_final_nll = {
        arm.label: metric_summary(
            {str(seed): final_nll(results, arm.label, seed) for seed in SEEDS},
            seed=175100 + idx * 10,
        )
        for idx, arm in enumerate(ARMS)
    }

    vs_scratch: dict[str, Any] = {}
    for idx, label in enumerate(ANCHOR_LABELS):
        values = {
            str(seed): final_nll(results, "scratch_baseline", seed) - final_nll(results, label, seed)
            for seed in SEEDS
        }
        vs_scratch[label] = metric_summary(values, seed=175200 + idx * 10)

    qwen_minus_alt_values = {
        str(seed): (
            final_nll(results, "anchor_alternative_trained_donor", seed)
            - final_nll(results, "anchor_qwen3_donor", seed)
        )
        for seed in SEEDS
    }
    qwen_minus_alt = metric_summary(qwen_minus_alt_values, seed=175300)

    qwen_gain = float(vs_scratch["anchor_qwen3_donor"]["mean"])
    alt_gain = float(vs_scratch["anchor_alternative_trained_donor"]["mean"])
    qwen_reference_delta = qwen_gain - REFERENCE_QWEN3_DONOR_GAIN_NATS
    observed_80_threshold = NULL_80_FRACTION * qwen_gain
    reference_80_threshold = NULL_80_FRACTION * REFERENCE_QWEN3_DONOR_GAIN_NATS
    alt_ratio_observed = alt_gain / qwen_gain if qwen_gain > 0.0 else float("nan")
    alt_ratio_reference = alt_gain / REFERENCE_QWEN3_DONOR_GAIN_NATS

    margin_mean = float(qwen_minus_alt["mean"])
    margin_lo = float(qwen_minus_alt["ci_95_lo"])
    margin_hi = float(qwen_minus_alt["ci_95_hi"])
    ci_excludes_zero = bool(margin_lo > 0.0 or margin_hi < 0.0)
    ci_crosses_zero = bool(margin_lo <= 0.0 <= margin_hi)

    criteria = {
        "mean_qwen3_minus_alt_ge_0p5_nats": margin_mean >= PASS_IDENTITY_MARGIN_NATS,
        "paired_bootstrap_ci_excludes_zero": ci_excludes_zero,
        "mean_qwen3_minus_alt_le_0p2_nats": margin_mean <= FAIL_IDENTITY_MARGIN_NATS,
        "paired_bootstrap_ci_crosses_zero": ci_crosses_zero,
        "alt_gain_ge_80pct_observed_qwen3_gain": bool(qwen_gain > 0.0 and alt_gain >= observed_80_threshold),
        "alt_gain_ge_80pct_reference_qwen3_gain": bool(alt_gain >= reference_80_threshold),
        "qwen3_sanity_gain_within_0p10_nats_of_reference": abs(qwen_reference_delta) <= 0.10,
    }
    positive_direction_seeds = int(sum(float(value) > 0.0 for value in qwen_minus_alt_values.values()))

    if criteria["mean_qwen3_minus_alt_ge_0p5_nats"] and margin_lo > 0.0:
        status = "identity_specificity_locked"
        verdict = (
            "PASS: donor-identity specificity LOCKED. "
            f"Qwen3 beats alternative trained donor by {margin_mean:+.3f} nats "
            f"with 95% CI [{margin_lo:+.3f}, {margin_hi:+.3f}]."
        )
    elif criteria["mean_qwen3_minus_alt_le_0p2_nats"] or ci_crosses_zero:
        status = "identity_specificity_rejected"
        verdict = (
            "FAIL: donor-identity specificity REJECTED / trained-like-basin explanation survives. "
            f"Qwen3-minus-alt margin={margin_mean:+.3f} nats, "
            f"95% CI [{margin_lo:+.3f}, {margin_hi:+.3f}], "
            f"alt_gain={alt_gain:+.3f} nats ({100.0 * alt_ratio_reference:.1f}% of +1.087 reference)."
        )
    else:
        status = "intermediate"
        verdict = (
            "INTERMEDIATE: partial donor-identity specificity. "
            f"Qwen3-minus-alt margin={margin_mean:+.3f} nats with 95% CI "
            f"[{margin_lo:+.3f}, {margin_hi:+.3f}]."
        )

    return {
        "status": status,
        "verdict": verdict,
        "arm_final_nll": arm_final_nll,
        "vs_scratch_final_nll_gain": vs_scratch,
        "qwen3_minus_alt_trained_gain_nats": qwen_minus_alt,
        "positive_direction_seeds_qwen3_better_than_alt": positive_direction_seeds,
        "observed_qwen3_effect_vs_scratch_nats": qwen_gain,
        "observed_alt_trained_effect_vs_scratch_nats": alt_gain,
        "qwen3_reference_gain_nats": REFERENCE_QWEN3_DONOR_GAIN_NATS,
        "qwen3_observed_minus_reference_nats": qwen_reference_delta,
        "observed_alt_fraction_of_qwen3_effect": alt_ratio_observed,
        "reference_alt_fraction_of_qwen3_effect": alt_ratio_reference,
        "observed_80pct_qwen3_threshold_nats": observed_80_threshold,
        "reference_80pct_qwen3_threshold_nats": reference_80_threshold,
        "criteria": criteria,
    }


def run_scratch_cells(
    payload: dict[str, Any],
    *,
    t_start: float,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
    tok,
) -> None:
    arm = ARMS[0]
    for seed in SEEDS:
        if cell_done(payload, arm.label, seed):
            print(f"  skip complete cell {arm.label} seed={seed}")
            continue
        print(f"\n--- main arm={arm.label} seed={seed} ---")
        payload["results"][arm.label][str(seed)] = run_anchor_cell(
            arm,
            seed=seed,
            target_params_device=None,
            train_ids=train_ids,
            train_mask=train_mask,
            val_ids=val_ids,
            val_mask=val_mask,
            tok=tok,
        )
        payload["summary"] = build_summary(payload)
        payload["verdict"] = payload["summary"]["verdict"]
        write_payload(payload, t_start=t_start, incremental=True)


def run_anchor_arm(
    payload: dict[str, Any],
    *,
    t_start: float,
    arm: AnchorArmSpec,
    target_params_cpu: dict[str, torch.Tensor],
    target_metadata: dict[str, Any],
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
    tok,
) -> None:
    if all_cells_done(payload, arm.label):
        print(f"  skip complete arm {arm.label}")
        return

    validation = validate_target_params(target_params_cpu, label=arm.label)
    payload["target_npz"].setdefault(arm.label, {})
    payload["target_npz"][arm.label].setdefault("metadata", target_metadata)
    payload["target_npz"][arm.label]["shape_validation"] = validation
    payload["target_npz"][arm.label]["param_count"] = anchor_param_count(target_params_cpu)
    payload["target_npz"][arm.label]["norm_sq"] = anchor_state_norm_sq(target_params_cpu)
    write_payload(payload, t_start=t_start, incremental=True)

    target_params_device = stage_params_to_device(target_params_cpu)
    if arm.label not in payload["anchor_diagnostics"]:
        payload["anchor_diagnostics"][arm.label] = compute_anchor_diagnostics(
            target_label=arm.label,
            target_params_device=target_params_device,
        )
        write_payload(payload, t_start=t_start, incremental=True)

    for seed in SEEDS:
        if cell_done(payload, arm.label, seed):
            print(f"  skip complete cell {arm.label} seed={seed}")
            continue
        print(f"\n--- main arm={arm.label} seed={seed} ---")
        payload["results"][arm.label][str(seed)] = run_anchor_cell(
            arm,
            seed=seed,
            target_params_device=target_params_device,
            train_ids=train_ids,
            train_mask=train_mask,
            val_ids=val_ids,
            val_mask=val_mask,
            tok=tok,
        )
        payload["summary"] = build_summary(payload)
        payload["verdict"] = payload["summary"]["verdict"]
        write_payload(payload, t_start=t_start, incremental=True)

    del target_params_device
    cleanup_cuda()


def run_main_cells(
    payload: dict[str, Any],
    *,
    t_start: float,
    tok,
    alt_params_cpu: dict[str, torch.Tensor] | None,
) -> None:
    print("\n=== Main identity falsifier: 3 arms x 3 seeds ===")
    train_ids, train_mask, val_ids, val_mask, data_meta = load_main_data(tok)
    payload["data"] = data_meta
    write_payload(payload, t_start=t_start, incremental=True)

    run_scratch_cells(
        payload,
        t_start=t_start,
        train_ids=train_ids,
        train_mask=train_mask,
        val_ids=val_ids,
        val_mask=val_mask,
        tok=tok,
    )

    qwen_arm = ARMS[1]
    if not all_cells_done(payload, qwen_arm.label):
        qwen_params_cpu, qwen_metadata = load_qwen3_anchor_params()
        run_anchor_arm(
            payload,
            t_start=t_start,
            arm=qwen_arm,
            target_params_cpu=qwen_params_cpu,
            target_metadata=qwen_metadata,
            train_ids=train_ids,
            train_mask=train_mask,
            val_ids=val_ids,
            val_mask=val_mask,
            tok=tok,
        )
        del qwen_params_cpu
        cleanup_cuda()
    else:
        print(f"  skip complete arm {qwen_arm.label}")

    alt_arm = ARMS[2]
    if not all_cells_done(payload, alt_arm.label):
        if alt_params_cpu is None:
            alt_params_cpu, alt_metadata = load_state_npz(alternative_npz_path())
        else:
            alt_metadata = payload.get("alternative_donor", {}).get("metadata", {})
        run_anchor_arm(
            payload,
            t_start=t_start,
            arm=alt_arm,
            target_params_cpu=alt_params_cpu,
            target_metadata=alt_metadata,
            train_ids=train_ids,
            train_mask=train_mask,
            val_ids=val_ids,
            val_mask=val_mask,
            tok=tok,
        )
    else:
        print(f"  skip complete arm {alt_arm.label}")

    payload["summary"] = build_summary(payload)
    payload["verdict"] = payload["summary"]["verdict"]
    write_payload(payload, t_start=t_start, incremental=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genome 175 alternative trained donor identity falsifier.")
    parser.add_argument(
        "--stage",
        choices=["all", "alternative-donor", "main"],
        default="all",
        help="Run the full experiment, only build/load the alternative donor, or only run main cells.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing result JSON and start a fresh payload.",
    )
    parser.add_argument(
        "--force-alternative-donor",
        action="store_true",
        help="Retrain and rewrite the alternative donor NPZ even if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_locked_protocol()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print("genome_175: alternative trained donor identity falsifier")
    print(f"  model_arch={g165._MODEL_ID}")
    print(f"  device={DEVICE} forward_dtype={FORWARD_DTYPE}")
    print(f"  main seeds={SEEDS} steps={g165.N_STEPS} batch={g165.BATCH_SIZE}")
    print(
        f"  anchor lambda_0={ANCHOR_LAMBDA_0} schedule={ANCHOR_SCHEDULE} "
        f"reference_qwen3_gain={REFERENCE_QWEN3_DONOR_GAIN_NATS:+.3f} nats"
    )
    print(
        f"  alternative donor={ALT_DONOR_OPTION} seed={ALT_DONOR_SEED} "
        f"steps={ALT_DONOR_STEPS} corpus=Wikitext-103 train"
    )
    print(f"  output={OUT_PATH}")

    t_start = time.time()
    payload = load_or_create_payload(resume=not args.no_resume)
    tok = load_tokenizer()
    alt_params_cpu: dict[str, torch.Tensor] | None = None

    if args.stage in {"all", "alternative-donor"}:
        alt_params_cpu = ensure_alternative_donor_params(
            tok,
            payload=payload,
            t_start=t_start,
            force=args.force_alternative_donor,
        )
        if args.stage == "alternative-donor":
            payload["summary"] = build_summary(payload)
            payload["verdict"] = payload["summary"]["verdict"]
            write_payload(payload, t_start=t_start, incremental=False)
            print(f"Saved alternative donor state: {alternative_npz_path()}")
            return

    if args.stage == "main" and not alternative_npz_path().exists():
        alt_params_cpu = ensure_alternative_donor_params(
            tok,
            payload=payload,
            t_start=t_start,
            force=False,
        )

    if args.stage in {"all", "main"}:
        run_main_cells(payload, t_start=t_start, tok=tok, alt_params_cpu=alt_params_cpu)

    payload["summary"] = build_summary(payload)
    payload["verdict"] = payload["summary"]["verdict"]
    write_payload(payload, t_start=t_start, incremental=False)
    print(f"\n=== verdict: {payload['verdict']} ===")
    print(f"Saved: {OUT_PATH} ({payload['elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()
