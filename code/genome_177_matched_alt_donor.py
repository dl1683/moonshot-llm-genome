"""
genome_177_matched_alt_donor.py

Matched alternative-donor falsifier for the C22 donor-identity claim.

Cycle 55 adversarial review found genome_175 confounded by a weak Wikitext
alternative donor, under-training, and unmatched anchor force. This experiment
trains three same-architecture alternative donors from scratch on C4 until their
held-out C4 NLL matches the Qwen3 donor target, then reruns the g165/g174
500-step anchor protocol with per-seed anchor-gradient normalization.

Design
------
Alternative donors:
  - Qwen3-0.6B architecture from random init.
  - Init/data seeds 1234, 5678, 9999.
  - C4 train stream, held-out C4 validation stopping metric.
  - Stop when held-out C4 NLL <= min(3.60, Qwen3_C4_NLL + 0.05), unless
    --alt-target-nll overrides the threshold.
  - Save FP32 parameter snapshots as NPZ.

Main arms:
  1. scratch_baseline
  2. anchor_qwen3_donor, lambda=0.01
  3. anchor_alt_donor_seed_1234, lambda normalized to Qwen3 initial grad L2
  4. anchor_alt_donor_seed_5678, lambda normalized to Qwen3 initial grad L2
  5. anchor_alt_donor_seed_9999, lambda normalized to Qwen3 initial grad L2

All main cells use the locked g165/g174 Part A protocol:
  seeds [42, 7, 13], 500 steps, batch 8, seq_len 256, lr 3e-4,
  BF16 recipient forward, FP32 donor anchor state.

Outputs
-------
  - results/genome_177_matched_alt_donor.json
  - results/genome_177_targets/genome_177_alt_donor_c4_seed*.npz
"""
# ENVELOPE COMPLIANCE (g177v2, 2026-04-28)
# Target alt-donor steps: 10,000 per donor, strict stop NLL remains
# min(3.60, Qwen3_heldout_C4_NLL + 0.05); unmatched donors are saved but are
# not valid for main matched-condition cells unless --allow-unmatched-donors is
# explicitly set.
# Envelope budget: pretrain ~2,600s/donor x 3 = 7,800s; recipient
# ~200s/cell x 15 = 3,000s; setup/eval/checkpoint overhead <=1,200s; total
# ~12,000s (<=14,400s).
# Justification: at the observed ~0.25s/step, 30,000 steps/donor is
# out-of-envelope before main cells. There is no defensible evidence that
# same-architecture Qwen3 scratch donors reach 3.60 NLL by 12k-18k steps, so
# the stop rule stays strict and the cap/budget enforce an auditable partial
# result instead of silently relaxing the matched-NLL condition.
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
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import genome_165_annealed_donor as g165
import genome_167_kd_canonical as g167


OUT_PATH = ROOT / "results" / "genome_177_matched_alt_donor.json"
TARGET_DIR = ROOT / "results" / "genome_177_targets"

SEEDS = [42, 7, 13]
ALT_DONOR_SEEDS = [1234, 5678, 9999]

ANCHOR_LAMBDA_QWEN3 = 0.01
ANCHOR_SCHEDULE = "constant"

ALT_DONOR_MAX_STEPS = 10_000
ALT_PRETRAIN_TOTAL_BUDGET_SEC = 8_000
ALT_DONOR_EST_SEC_PER_STEP = 0.25
ALT_DONOR_EST_EVAL_OVERHEAD_SEC = 100.0
ALT_DONOR_TARGET_NLL = 3.60
ALT_DONOR_QWEN3_MATCH_TOLERANCE_NATS = 0.05
ALT_DONOR_BATCH_SIZE = g165.BATCH_SIZE
ALT_DONOR_LR = g165.LR
ALT_DONOR_BETAS = (0.9, 0.95)
ALT_DONOR_WEIGHT_DECAY = 0.01
ALT_DONOR_GRAD_CLIP = 1.0
ALT_DONOR_LOG_EVERY = 100
ALT_DONOR_EVAL_EVERY = 500
ALT_DONOR_STOP_RULE_LAST_K = 3
ALT_DONOR_EVAL_WINDOWS = 256
ALT_DONOR_TRAIN_SPLIT = "train"
ALT_DONOR_EVAL_SPLIT = "validation"
ALT_DONOR_C4_TRAIN_BUFFER_SIZE = 10_000
ALT_DONOR_C4_EVAL_SEED = 177101

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
    donor_seed: int | None
    description: str

    @property
    def use_anchor(self) -> bool:
        return self.target_kind != "none"


ARMS = [
    AnchorArmSpec(
        label="scratch_baseline",
        target_kind="none",
        donor_seed=None,
        description="No anchor; exact g165/g174 Part A scratch baseline protocol.",
    ),
    AnchorArmSpec(
        label="anchor_qwen3_donor",
        target_kind="qwen3_trained",
        donor_seed=None,
        description="Constant Frobenius anchor to the original trained Qwen3-0.6B donor.",
    ),
    *[
        AnchorArmSpec(
            label=f"anchor_alt_donor_seed_{seed}",
            target_kind="alternative_c4_trained",
            donor_seed=seed,
            description=(
                "Constant Frobenius anchor to a same-architecture alternative "
                f"donor trained from scratch on C4 with seed {seed}; lambda is "
                "normalized per recipient seed to match Qwen3's initial anchor-gradient L2."
            ),
        )
        for seed in ALT_DONOR_SEEDS
    ],
]

ARM_BY_LABEL = {arm.label: arm for arm in ARMS}
ANCHOR_LABELS = [arm.label for arm in ARMS if arm.use_anchor]
ALT_ARM_LABELS = [arm.label for arm in ARMS if arm.target_kind == "alternative_c4_trained"]


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
        "anchor lambda qwen3": (ANCHOR_LAMBDA_QWEN3, 0.01),
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
        "genome": 177,
        "name": "matched_alt_donor_identity_falsifier",
        "timestamp_utc_started": now_utc(),
        "model_id": g165._MODEL_ID,
        "device": DEVICE,
        "forward_dtype": str(FORWARD_DTYPE),
        "config": {
            "seeds": SEEDS,
            "protocol_source": "g165/g174 Part A",
            "main": {
                "arms": [arm.__dict__ for arm in ARMS],
                "steps": g165.N_STEPS,
                "batch_size": g165.BATCH_SIZE,
                "seq_len": g165.SEQ_LEN,
                "lr": g165.LR,
                "eval_every": g165.EVAL_EVERY,
                "qwen3_lambda_0": ANCHOR_LAMBDA_QWEN3,
                "schedule": ANCHOR_SCHEDULE,
                "reference_qwen3_donor_gain_nats": REFERENCE_QWEN3_DONOR_GAIN_NATS,
                "null_80_fraction": NULL_80_FRACTION,
            },
            "alternative_donors": {
                "seeds": ALT_DONOR_SEEDS,
                "corpus": "allenai/c4:en",
                "train_split": ALT_DONOR_TRAIN_SPLIT,
                "heldout_split": ALT_DONOR_EVAL_SPLIT,
                "heldout_seed": ALT_DONOR_C4_EVAL_SEED,
                "heldout_windows": ALT_DONOR_EVAL_WINDOWS,
                "target_nll_nominal": ALT_DONOR_TARGET_NLL,
                "qwen3_match_tolerance_nats": ALT_DONOR_QWEN3_MATCH_TOLERANCE_NATS,
                "max_steps": ALT_DONOR_MAX_STEPS,
                "total_budget_sec": ALT_PRETRAIN_TOTAL_BUDGET_SEC,
                "estimated_sec_per_step": ALT_DONOR_EST_SEC_PER_STEP,
                "estimated_eval_overhead_sec_per_donor": ALT_DONOR_EST_EVAL_OVERHEAD_SEC,
                "batch_size": ALT_DONOR_BATCH_SIZE,
                "seq_len": g165.SEQ_LEN,
                "lr": ALT_DONOR_LR,
                "betas": list(ALT_DONOR_BETAS),
                "weight_decay": ALT_DONOR_WEIGHT_DECAY,
                "grad_clip": ALT_DONOR_GRAD_CLIP,
                "eval_every": ALT_DONOR_EVAL_EVERY,
                "stop_rule_last_k": ALT_DONOR_STOP_RULE_LAST_K,
                "log_every": ALT_DONOR_LOG_EVERY,
                "overlap_filter": "reject alt train/eval windows sharing any token 13-gram with recipient main train or eval windows",
            },
            "lambda_normalization": {
                "qwen3_lambda": ANCHOR_LAMBDA_QWEN3,
                "actual_alt_lambda_formula": (
                    "lambda_alt = lambda_qwen3 * sqrt(qwen3_frobenius_sq / alt_frobenius_sq)"
                ),
                "note": (
                    "The squared-distance ratio requested by the adversarial note is "
                    "recorded as lambda_f2_ratio_formula. Exact initial anchor-gradient "
                    "matching for lambda*||theta-target||^2 requires the square-root ratio."
                ),
            },
            "criteria": {
                "identity_specificity_locked": (
                    "mean(qwen3_gain - best_alt_gain) >= +0.5 nats and paired "
                    "95% CI excludes zero"
                ),
                "intermediate": (
                    "mean margin in [+0.2, +0.5) nats with paired 95% CI excluding zero"
                ),
                "identity_specificity_rejected": (
                    "mean margin <= +0.2 nats, CI crosses zero, or matched-alt reaches "
                    ">=80% of the +1.087 nats Qwen3 reference gain"
                ),
            },
        },
        "data": {},
        "qwen3_reference": {},
        "alternative_donors": {},
        "target_npz": {},
        "anchor_diagnostics": {},
        "results": {arm.label: {} for arm in ARMS},
        "summary": {},
        "verdict": "INCOMPLETE",
    }


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.setdefault("config", {})
    alt_config = config.setdefault("alternative_donors", {})
    alt_config.setdefault("max_steps", ALT_DONOR_MAX_STEPS)
    alt_config.setdefault("total_budget_sec", ALT_PRETRAIN_TOTAL_BUDGET_SEC)
    alt_config.setdefault("estimated_sec_per_step", ALT_DONOR_EST_SEC_PER_STEP)
    alt_config.setdefault("estimated_eval_overhead_sec_per_donor", ALT_DONOR_EST_EVAL_OVERHEAD_SEC)
    alt_config.setdefault("stop_rule_last_k", ALT_DONOR_STOP_RULE_LAST_K)
    alt_config.setdefault(
        "overlap_filter",
        "reject alt train/eval windows sharing any token 13-gram with recipient main train or eval windows",
    )
    payload.setdefault("data", {})
    payload.setdefault("qwen3_reference", {})
    payload.setdefault("alternative_donors", {})
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
    cell = payload["results"].get(arm_label, {}).get(str(seed))
    return isinstance(cell, dict) and "final_nll" in cell


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
        metadata_json = str(data["metadata_json"].tolist())
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
        sum(
            params[name].numel()
            for name in model_shapes
            if name in params and tuple(params[name].shape) == model_shapes[name]
        )
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


@torch.no_grad()
def evaluate_nll(
    model: torch.nn.Module,
    eval_ids: torch.Tensor,
    eval_mask: torch.Tensor,
    *,
    batch_size: int = g165.BATCH_SIZE,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, eval_ids.shape[0], batch_size):
        ids = eval_ids[start : start + batch_size].to(DEVICE)
        mask = eval_mask[start : start + batch_size].to(DEVICE)
        with autocast_context():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        shift_logits = logits[:, :-1].contiguous().float()
        shift_labels = ids[:, 1:].contiguous().clone()
        shift_mask = mask[:, 1:].contiguous().bool()
        labels_for_loss = shift_labels.clone()
        labels_for_loss[~shift_mask] = -100
        loss_sum = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            labels_for_loss.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n_tokens = int(shift_mask.sum().item())
        total_loss += float(loss_sum.item())
        total_tokens += n_tokens
    model.train()
    return total_loss / max(total_tokens, 1)


class StreamingC4Batcher:
    def __init__(
        self,
        tok,
        *,
        split: str,
        seed: int,
        batch_size: int,
        seq_len: int,
        buffer_size: int,
        forbidden_hashes: set[int] | None = None,
    ) -> None:
        self.tok = tok
        self.split = split
        self.seed = seed
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.buffer_size = buffer_size
        self.forbidden_hashes = forbidden_hashes
        self.epoch = 0
        self.records_seen = 0
        self.windows_yielded = 0
        self.overlap_rejects = 0
        self.chosen_dataset = ""
        self._token_buffer: list[int] = []
        self._buffer_cursor = 0
        self._iterator = self._new_iterator()

    def _new_iterator(self) -> Iterator[dict[str, Any]]:
        ds, chosen_name = g167._load_streaming_dataset(
            ["allenai/c4"],
            "en",
            split=self.split,
            seed=self.seed + self.epoch,
        )
        self.chosen_dataset = chosen_name
        return iter(ds)

    def _next_record(self) -> dict[str, Any]:
        while True:
            try:
                return next(self._iterator)
            except StopIteration:
                self.epoch += 1
                self._token_buffer = []
                self._buffer_cursor = 0
                self._iterator = self._new_iterator()

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        windows: list[np.ndarray] = []
        while len(windows) < self.batch_size:
            while (len(self._token_buffer) - self._buffer_cursor) >= self.seq_len and len(windows) < self.batch_size:
                window = np.asarray(
                    self._token_buffer[self._buffer_cursor : self._buffer_cursor + self.seq_len],
                    dtype=np.int64,
                )
                self._buffer_cursor += self.seq_len
                if self.forbidden_hashes is not None:
                    row_hashes = g167.rolling_13gram_hashes(window)
                    if any(int(h) in self.forbidden_hashes for h in row_hashes.tolist()):
                        self.overlap_rejects += 1
                        continue
                windows.append(window)

            if len(windows) >= self.batch_size:
                break

            record = self._next_record()
            text = record.get("text", "")
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue
            self.records_seen += 1
            token_ids = self.tok(
                text,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]
            if not token_ids:
                continue
            self._token_buffer.extend(token_ids)
            if self._buffer_cursor >= 8192:
                self._token_buffer = self._token_buffer[self._buffer_cursor :]
                self._buffer_cursor = 0

        self.windows_yielded += len(windows)
        ids = torch.tensor(np.stack(windows, axis=0), dtype=torch.long)
        mask = torch.ones_like(ids, dtype=torch.long)
        return ids, mask

    def metadata(self) -> dict[str, Any]:
        return {
            "dataset_name": self.chosen_dataset,
            "config_name": "en",
            "split": self.split,
            "seed": self.seed,
            "shuffle_buffer_size": self.buffer_size,
            "seq_len": self.seq_len,
            "sampling": "streamed_concatenated_nonoverlapping_windows",
            "epochs_started": self.epoch + 1,
            "records_seen": self.records_seen,
            "windows_yielded": self.windows_yielded,
            "overlap_filter": (
                "recipient_main_train_plus_eval_token_13gram"
                if self.forbidden_hashes is not None
                else None
            ),
            "forbidden_hash_count": len(self.forbidden_hashes) if self.forbidden_hashes is not None else 0,
            "overlap_rejects": self.overlap_rejects,
        }


def load_alt_eval_data(
    tok,
    *,
    n_windows: int,
    forbidden_hashes: set[int],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    ids, mask, meta = g167.load_c4_windows(
        tok,
        split=ALT_DONOR_EVAL_SPLIT,
        seed=ALT_DONOR_C4_EVAL_SEED,
        n_windows=n_windows,
        forbidden_hashes=forbidden_hashes,
    )
    meta.update(
        {
            "purpose": "alternative donor held-out C4 stopping metric",
            "nll_target_nominal": ALT_DONOR_TARGET_NLL,
            "overlap_filter": "recipient_main_train_plus_eval_token_13gram",
            "forbidden_hash_count": len(forbidden_hashes),
        }
    )
    return ids, mask, meta


def resolve_stop_nll(qwen3_heldout_nll: float, explicit_target_nll: float | None) -> float:
    if explicit_target_nll is not None:
        return float(explicit_target_nll)
    return min(ALT_DONOR_TARGET_NLL, qwen3_heldout_nll + ALT_DONOR_QWEN3_MATCH_TOLERANCE_NATS)


def alt_npz_path(seed: int) -> Path:
    return TARGET_DIR / f"genome_177_alt_donor_c4_seed{seed}.npz"


def final_eval_nll_from_metadata(metadata: dict[str, Any]) -> float | None:
    if "final_heldout_c4_nll" in metadata:
        return float(metadata["final_heldout_c4_nll"])
    eval_log = metadata.get("eval_log")
    if isinstance(eval_log, list) and eval_log:
        last = eval_log[-1]
        if isinstance(last, dict) and "heldout_c4_nll" in last:
            return float(last["heldout_c4_nll"])
    return None


def final_stop_rule_nll_from_metadata(metadata: dict[str, Any]) -> float | None:
    if "final_stop_rule_mean_heldout_c4_nll" in metadata:
        return float(metadata["final_stop_rule_mean_heldout_c4_nll"])
    return final_eval_nll_from_metadata(metadata)


def donor_metadata_is_matched(metadata: dict[str, Any], stop_nll: float) -> bool:
    final_nll = final_stop_rule_nll_from_metadata(metadata)
    own_stop_nll = float(metadata.get("heldout_c4_stop_nll", stop_nll))
    effective_stop_nll = min(own_stop_nll, stop_nll)
    if metadata.get("matched_target") is True:
        return final_nll is not None and final_nll <= effective_stop_nll
    return final_nll is not None and final_nll <= effective_stop_nll


def evaluate_qwen3_on_alt_heldout(
    tok,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
    val_meta: dict[str, Any],
    forbidden_meta: dict[str, Any],
) -> dict[str, Any]:
    donor, _ = g165.load_trained_donor(tok)
    if hasattr(donor.config, "use_cache"):
        donor.config.use_cache = False
    nll = evaluate_nll(donor, val_ids, val_mask, batch_size=g165.BATCH_SIZE)
    del donor
    cleanup_cuda()
    return {
        "heldout_c4_nll": float(nll),
        "heldout_windows": int(val_ids.shape[0]),
        "heldout_tokens": int(val_mask[:, 1:].sum().item()),
        "evaluated_utc": now_utc(),
        "model": g165._MODEL_ID,
        "dataset_eval": val_meta,
        "forbidden_hash_sha1": forbidden_meta["combined_forbidden_hash_sha1"],
    }


def pretrain_alternative_donor(
    tok,
    *,
    donor_seed: int,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
    val_meta: dict[str, Any],
    stop_nll: float,
    max_steps: int,
    eval_every: int,
    forbidden_hashes: set[int],
    forbidden_meta: dict[str, Any],
    payload: dict[str, Any],
    t_start: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    print(f"\n=== Alternative donor pretraining: C4 seed={donor_seed} ===")
    set_seed(donor_seed)
    model = g165.load_random_init(donor_seed)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ALT_DONOR_LR,
        betas=ALT_DONOR_BETAS,
        weight_decay=ALT_DONOR_WEIGHT_DECAY,
    )
    batcher = StreamingC4Batcher(
        tok,
        split=ALT_DONOR_TRAIN_SPLIT,
        seed=donor_seed,
        batch_size=ALT_DONOR_BATCH_SIZE,
        seq_len=g165.SEQ_LEN,
        buffer_size=ALT_DONOR_C4_TRAIN_BUFFER_SIZE,
        forbidden_hashes=forbidden_hashes,
    )

    n_total_params = int(sum(param.numel() for param in model.parameters()))
    train_log: list[dict[str, Any]] = []
    eval_log: list[dict[str, Any]] = []
    t0 = time.time()

    initial_nll = evaluate_nll(model, val_ids, val_mask, batch_size=g165.BATCH_SIZE)
    eval_row = {
        "step": 0,
        "heldout_c4_nll": float(initial_nll),
        "stop_rule_mean_heldout_c4_nll": float(initial_nll),
        "stop_rule_last_k": ALT_DONOR_STOP_RULE_LAST_K,
        "elapsed_s": time.time() - t0,
    }
    eval_log.append(eval_row)
    print(
        f"  init_seed={donor_seed} max_steps={max_steps} batch={ALT_DONOR_BATCH_SIZE} "
        f"params={n_total_params / 1e6:.1f}M stop_nll={stop_nll:.4f}"
    )
    print(f"    alt_pretrain seed={donor_seed} step=0 heldout_c4_nll={initial_nll:.4f}")

    matched = initial_nll <= stop_nll
    final_step = 0
    final_loss = float("nan")

    for step in range(1, max_steps + 1):
        final_step = step
        ids_cpu, mask_cpu = batcher.next_batch()
        ids = ids_cpu.to(DEVICE)
        mask = mask_cpu.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            out = model(input_ids=ids, attention_mask=mask, labels=ids, use_cache=False)
            loss = out.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite alternative donor loss at step {step}: {float(loss.item())}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), ALT_DONOR_GRAD_CLIP)
        optimizer.step()
        final_loss = float(loss.item())

        should_log = step == 1 or step % ALT_DONOR_LOG_EVERY == 0 or step == max_steps
        should_eval = step % eval_every == 0 or step == max_steps

        if should_log:
            row = {
                "step": step,
                "train_loss": final_loss,
                "grad_norm": float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm),
                "elapsed_s": time.time() - t0,
                "stream_records_seen": batcher.records_seen,
                "stream_windows_yielded": batcher.windows_yielded,
                "stream_overlap_rejects": batcher.overlap_rejects,
            }
            train_log.append(row)
            print(
                f"    alt_pretrain seed={donor_seed} step={step:5d} "
                f"loss={row['train_loss']:.4f} grad_norm={row['grad_norm']:.3f} "
                f"({row['elapsed_s']:.0f}s)"
            )

        if should_eval:
            heldout_nll = evaluate_nll(model, val_ids, val_mask, batch_size=g165.BATCH_SIZE)
            stop_window = eval_log[-(ALT_DONOR_STOP_RULE_LAST_K - 1) :] if ALT_DONOR_STOP_RULE_LAST_K > 1 else []
            stop_values = [float(row["heldout_c4_nll"]) for row in stop_window]
            stop_values.append(float(heldout_nll))
            stop_rule_mean = float(np.mean(stop_values))
            has_full_stop_window = len(stop_values) >= ALT_DONOR_STOP_RULE_LAST_K
            matched_now = bool(has_full_stop_window and stop_rule_mean <= stop_nll)
            eval_row = {
                "step": step,
                "heldout_c4_nll": float(heldout_nll),
                "stop_rule_mean_heldout_c4_nll": stop_rule_mean,
                "stop_rule_last_k": ALT_DONOR_STOP_RULE_LAST_K,
                "stop_rule_full_window": has_full_stop_window,
                "elapsed_s": time.time() - t0,
                "train_loss_last": final_loss,
            }
            eval_log.append(eval_row)
            payload["alternative_donors"][str(donor_seed)] = {
                "status": "pretraining",
                "current_step": step,
                "latest_heldout_c4_nll": float(heldout_nll),
                "latest_stop_rule_mean_heldout_c4_nll": stop_rule_mean,
                "stop_nll": stop_nll,
                "matched": matched_now,
                "eval_log": eval_log,
                "train_log_tail": train_log[-10:],
                "stream": batcher.metadata(),
                "overlap_rejects": batcher.overlap_rejects,
            }
            write_payload(payload, t_start=t_start, incremental=True)
            print(
                f"    alt_pretrain seed={donor_seed} step={step:5d} "
                f"heldout_c4_nll={heldout_nll:.4f} "
                f"stop_mean_k{ALT_DONOR_STOP_RULE_LAST_K}={stop_rule_mean:.4f} "
                f"target<={stop_nll:.4f}"
            )
            if matched_now:
                matched = True
                break
            model.train()

    params = snapshot_params_cpu(model)
    del model, optimizer
    cleanup_cuda()

    final_eval_nll = float(eval_log[-1]["heldout_c4_nll"])
    final_stop_rule_nll = float(eval_log[-1]["stop_rule_mean_heldout_c4_nll"])
    status = "matched_target" if matched else "max_steps_reached_without_target"
    metadata = {
        "target_kind": "alternative_c4_trained",
        "source_model_arch": g165._MODEL_ID,
        "init_seed": donor_seed,
        "data_seed": donor_seed,
        "steps": final_step,
        "max_steps": max_steps,
        "batch_size": ALT_DONOR_BATCH_SIZE,
        "seq_len": g165.SEQ_LEN,
        "lr": ALT_DONOR_LR,
        "betas": list(ALT_DONOR_BETAS),
        "weight_decay": ALT_DONOR_WEIGHT_DECAY,
        "grad_clip": ALT_DONOR_GRAD_CLIP,
        "heldout_c4_stop_nll": stop_nll,
        "stop_rule_last_k": ALT_DONOR_STOP_RULE_LAST_K,
        "final_heldout_c4_nll": final_eval_nll,
        "final_stop_rule_mean_heldout_c4_nll": final_stop_rule_nll,
        "matched_target": bool(matched),
        "status": status,
        "dataset_train": batcher.metadata(),
        "dataset_eval": val_meta,
        "overlap_filter": forbidden_meta,
        "forbidden_hash_sha1": forbidden_meta["combined_forbidden_hash_sha1"],
        "overlap_rejects": {
            "train": batcher.overlap_rejects,
            "eval": int(val_meta.get("forbidden_overlap_rejects", 0)),
        },
        "train_log": train_log,
        "eval_log": eval_log,
        "n_total_params": n_total_params,
        "wallclock_s": time.time() - t0,
        "construction": (
            "AutoModelForCausalLM.from_config Qwen3-0.6B architecture trained "
            "from scratch on streamed C4 train."
        ),
    }
    return params, metadata


def ensure_alternative_donor(
    tok,
    *,
    donor_seed: int,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
    val_meta: dict[str, Any],
    stop_nll: float,
    max_steps: int,
    eval_every: int,
    forbidden_hashes: set[int],
    forbidden_meta: dict[str, Any],
    payload: dict[str, Any],
    t_start: float,
    force: bool,
    allow_unmatched: bool,
) -> None:
    path = alt_npz_path(donor_seed)
    key = str(donor_seed)
    arm_label = f"anchor_alt_donor_seed_{donor_seed}"

    if path.exists() and not force:
        print(f"\n=== Loading existing matched alternative donor NPZ: {path} ===")
        params, metadata = load_state_npz(path)
        validation = validate_target_params(params, label=arm_label)
        final_nll = final_eval_nll_from_metadata(metadata)
        matched = donor_metadata_is_matched(metadata, stop_nll)
        overlap_ok = overlap_filter_matches(metadata, forbidden_meta)
        if not overlap_ok:
            raise RuntimeError(
                f"existing donor {path} was not trained with the recipient 13-gram "
                "overlap filter required by g177v2. Use --force-alt-donors to retrain."
            )
        if not matched and not allow_unmatched:
            print(
                f"existing donor {path} is not matched to held-out C4 target: "
                f"final_nll={final_nll} stop_nll={stop_nll:.4f}. "
                "It remains saved for audit but will block main cells unless "
                "--allow-unmatched-donors is set."
            )
        payload["target_npz"][arm_label] = {
            "path": str(path),
            "exists": True,
            "loaded_existing": True,
            "size_bytes": path.stat().st_size,
            "metadata": metadata,
            "shape_validation": validation,
        }
        payload["alternative_donors"][key] = {
            "status": "loaded_existing_npz",
            "npz_path": str(path),
            "metadata": metadata,
            "matched_target": bool(matched),
            "final_heldout_c4_nll": final_nll,
            "final_stop_rule_mean_heldout_c4_nll": final_stop_rule_nll_from_metadata(metadata),
            "stop_nll": stop_nll,
            "overlap_rejects": metadata.get("overlap_rejects", {}),
            "param_count": anchor_param_count(params),
            "norm_sq": anchor_state_norm_sq(params),
        }
        del params
        cleanup_cuda()
        write_payload(payload, t_start=t_start, incremental=True)
        return

    params, metadata = pretrain_alternative_donor(
        tok,
        donor_seed=donor_seed,
        val_ids=val_ids,
        val_mask=val_mask,
        val_meta=val_meta,
        stop_nll=stop_nll,
        max_steps=max_steps,
        eval_every=eval_every,
        forbidden_hashes=forbidden_hashes,
        forbidden_meta=forbidden_meta,
        payload=payload,
        t_start=t_start,
    )
    validation = validate_target_params(params, label=arm_label)
    metadata["shape_validation"] = validation
    matched = bool(metadata["matched_target"])
    if not matched and not allow_unmatched:
        save_meta = save_state_npz(path, params, metadata=metadata, force=True)
        payload["target_npz"][arm_label] = save_meta
        payload["alternative_donors"][key] = {
            "status": "saved_unmatched_requires_explicit_allow_for_main",
            "npz_path": str(path),
            "metadata": metadata,
            "matched_target": False,
            "final_heldout_c4_nll": metadata["final_heldout_c4_nll"],
            "final_stop_rule_mean_heldout_c4_nll": metadata["final_stop_rule_mean_heldout_c4_nll"],
            "stop_nll": stop_nll,
            "overlap_rejects": metadata.get("overlap_rejects", {}),
            "param_count": anchor_param_count(params),
            "norm_sq": anchor_state_norm_sq(params),
        }
        write_payload(payload, t_start=t_start, incremental=True)
        print(
            f"alternative donor seed={donor_seed} did not reach held-out C4 target: "
            f"final_nll={metadata['final_heldout_c4_nll']:.4f} stop_nll={stop_nll:.4f}. "
            "Saved the unmatched checkpoint for audit; main cells will not run with it "
            "unless --allow-unmatched-donors is set."
        )
        del params
        cleanup_cuda()
        return

    save_meta = save_state_npz(path, params, metadata=metadata, force=True)
    payload["target_npz"][arm_label] = save_meta
    payload["alternative_donors"][key] = {
        "status": "pretrained_and_saved",
        "npz_path": str(path),
        "metadata": metadata,
        "matched_target": bool(matched),
        "final_heldout_c4_nll": metadata["final_heldout_c4_nll"],
        "final_stop_rule_mean_heldout_c4_nll": metadata["final_stop_rule_mean_heldout_c4_nll"],
        "stop_nll": stop_nll,
        "overlap_rejects": metadata.get("overlap_rejects", {}),
        "param_count": anchor_param_count(params),
        "norm_sq": anchor_state_norm_sq(params),
    }
    del params
    cleanup_cuda()
    write_payload(payload, t_start=t_start, incremental=True)


def ensure_alternative_donors(
    tok,
    *,
    payload: dict[str, Any],
    t_start: float,
    force: bool,
    allow_unmatched: bool,
    max_steps: int,
    eval_every: int,
    eval_windows: int,
    explicit_target_nll: float | None,
    forbidden_hashes: set[int],
    forbidden_meta: dict[str, Any],
) -> None:
    val_ids, val_mask, val_meta = load_alt_eval_data(
        tok,
        n_windows=eval_windows,
        forbidden_hashes=forbidden_hashes,
    )
    payload["data"]["alternative_donor_heldout_c4"] = val_meta
    payload["data"]["recipient_overlap_forbidden_hashes"] = forbidden_meta

    qwen3_eval = payload.get("qwen3_reference", {}).get("alternative_donor_heldout_c4")
    qwen3_eval_current = (
        isinstance(qwen3_eval, dict)
        and "heldout_c4_nll" in qwen3_eval
        and qwen3_eval.get("forbidden_hash_sha1") == forbidden_meta["combined_forbidden_hash_sha1"]
    )
    if not qwen3_eval_current:
        print("\n=== Qwen3 reference eval on alternative-donor held-out C4 ===")
        qwen3_eval = evaluate_qwen3_on_alt_heldout(tok, val_ids, val_mask, val_meta, forbidden_meta)
        payload["qwen3_reference"]["alternative_donor_heldout_c4"] = qwen3_eval
        write_payload(payload, t_start=t_start, incremental=True)

    stop_nll = resolve_stop_nll(float(qwen3_eval["heldout_c4_nll"]), explicit_target_nll)
    payload["config"]["alternative_donors"]["effective_stop_nll"] = stop_nll
    payload["config"]["alternative_donors"]["effective_max_steps"] = max_steps
    payload["config"]["alternative_donors"]["effective_eval_every"] = eval_every
    payload["config"]["alternative_donors"]["effective_eval_windows"] = eval_windows
    payload["config"]["alternative_donors"]["effective_total_budget_sec"] = ALT_PRETRAIN_TOTAL_BUDGET_SEC
    payload["config"]["alternative_donors"]["estimated_sec_per_donor"] = estimate_alt_donor_wallclock_s(max_steps)
    write_payload(payload, t_start=t_start, incremental=True)

    budget_start = time.time()
    budget_exhausted = False
    for donor_seed in ALT_DONOR_SEEDS:
        elapsed_budget_s = time.time() - budget_start
        next_est_s = 0.0 if alt_npz_path(donor_seed).exists() and not force else estimate_alt_donor_wallclock_s(max_steps)
        remaining_budget_s = ALT_PRETRAIN_TOTAL_BUDGET_SEC - elapsed_budget_s
        if elapsed_budget_s > 0.0 and next_est_s > remaining_budget_s:
            payload["alternative_donors"][str(donor_seed)] = {
                "status": "skipped_pretrain_budget_exhausted",
                "matched_target": False,
                "budget_elapsed_s": elapsed_budget_s,
                "budget_remaining_s": remaining_budget_s,
                "estimated_next_donor_s": next_est_s,
                "total_budget_sec": ALT_PRETRAIN_TOTAL_BUDGET_SEC,
            }
            budget_exhausted = True
            payload["config"]["alternative_donors"]["pretrain_budget_elapsed_s"] = elapsed_budget_s
            payload["config"]["alternative_donors"]["pretrain_budget_exhausted"] = True
            print(
                f"  skipping alternative donor seed={donor_seed}: "
                f"elapsed_pretrain_budget={elapsed_budget_s:.0f}s "
                f"remaining={remaining_budget_s:.0f}s estimated_next={next_est_s:.0f}s"
            )
            write_payload(payload, t_start=t_start, incremental=True)
            break
        ensure_alternative_donor(
            tok,
            donor_seed=donor_seed,
            val_ids=val_ids,
            val_mask=val_mask,
            val_meta=val_meta,
            stop_nll=stop_nll,
            max_steps=max_steps,
            eval_every=eval_every,
            forbidden_hashes=forbidden_hashes,
            forbidden_meta=forbidden_meta,
            payload=payload,
            t_start=t_start,
            force=force,
            allow_unmatched=allow_unmatched,
        )
        payload["config"]["alternative_donors"]["pretrain_budget_elapsed_s"] = time.time() - budget_start
        payload["config"]["alternative_donors"]["pretrain_budget_exhausted"] = budget_exhausted
        write_payload(payload, t_start=t_start, incremental=True)


def load_qwen3_anchor_params(tok) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    donor, _ = g165.load_trained_donor(tok)
    if hasattr(donor.config, "use_cache"):
        donor.config.use_cache = False
    params = snapshot_params_cpu(donor)
    del donor
    cleanup_cuda()
    metadata = {
        "target_kind": "qwen3_trained",
        "source_model": g165._MODEL_ID,
        "construction": "Original trained Qwen3-0.6B donor snapshot via genome_165.load_trained_donor.",
    }
    return params, metadata


def compute_frobenius_by_seed(
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
            "mean_sq_gap_per_recipient_param": f2 / max(n_params, 1),
        }
        n_matched = n_params
        del recipient
        cleanup_cuda()

    values = [float(by_seed[str(seed)]["frobenius_sq_init_to_target"]) for seed in SEEDS]
    return {
        "target_label": target_label,
        "n_recipient_params": n_matched,
        "by_seed": by_seed,
        "mean_frobenius_sq": float(np.mean(values)),
        "std_frobenius_sq": float(np.std(values)),
    }


def build_qwen3_anchor_diagnostics(fro: dict[str, Any]) -> dict[str, Any]:
    diag = dict(fro)
    diag["lambda_0"] = ANCHOR_LAMBDA_QWEN3
    diag["lambda_policy"] = "fixed_qwen3_reference"
    for seed in SEEDS:
        row = diag["by_seed"][str(seed)]
        f2 = float(row["frobenius_sq_init_to_target"])
        row["actual_lambda_0"] = ANCHOR_LAMBDA_QWEN3
        row["anchor_grad_l2_at_actual_lambda"] = 2.0 * ANCHOR_LAMBDA_QWEN3 * math.sqrt(f2)
    return diag


def build_alt_anchor_diagnostics(
    *,
    alt_fro: dict[str, Any],
    qwen3_diag: dict[str, Any],
) -> dict[str, Any]:
    diag = dict(alt_fro)
    diag["lambda_policy"] = "per_seed_initial_anchor_gradient_l2_match_to_qwen3"
    diag["qwen3_reference_label"] = "anchor_qwen3_donor"
    diag["qwen3_lambda_0"] = ANCHOR_LAMBDA_QWEN3
    actual_lambdas = []
    f2_ratio_lambdas = []

    for seed in SEEDS:
        seed_key = str(seed)
        row = diag["by_seed"][seed_key]
        alt_f2 = float(row["frobenius_sq_init_to_target"])
        qwen3_f2 = float(qwen3_diag["by_seed"][seed_key]["frobenius_sq_init_to_target"])
        qwen3_grad_l2 = float(qwen3_diag["by_seed"][seed_key]["anchor_grad_l2_at_actual_lambda"])

        lambda_f2_ratio = ANCHOR_LAMBDA_QWEN3 * (qwen3_f2 / alt_f2)
        lambda_grad_matched = ANCHOR_LAMBDA_QWEN3 * math.sqrt(qwen3_f2 / alt_f2)
        alt_grad_l2_actual = 2.0 * lambda_grad_matched * math.sqrt(alt_f2)
        alt_grad_l2_f2_ratio = 2.0 * lambda_f2_ratio * math.sqrt(alt_f2)

        row.update(
            {
                "qwen3_frobenius_sq_same_recipient_seed": qwen3_f2,
                "qwen3_anchor_grad_l2_reference": qwen3_grad_l2,
                "lambda_f2_ratio_formula": lambda_f2_ratio,
                "lambda_grad_norm_matched": lambda_grad_matched,
                "actual_lambda_0": lambda_grad_matched,
                "lambda_normalization_vs_qwen3": lambda_grad_matched / ANCHOR_LAMBDA_QWEN3,
                "anchor_grad_l2_at_actual_lambda": alt_grad_l2_actual,
                "anchor_grad_l2_at_f2_ratio_lambda": alt_grad_l2_f2_ratio,
                "actual_minus_qwen3_anchor_grad_l2": alt_grad_l2_actual - qwen3_grad_l2,
            }
        )
        actual_lambdas.append(lambda_grad_matched)
        f2_ratio_lambdas.append(lambda_f2_ratio)

    diag["mean_actual_lambda_0"] = float(np.mean(actual_lambdas))
    diag["std_actual_lambda_0"] = float(np.std(actual_lambdas))
    diag["mean_lambda_f2_ratio_formula"] = float(np.mean(f2_ratio_lambdas))
    diag["std_lambda_f2_ratio_formula"] = float(np.std(f2_ratio_lambdas))
    return diag


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


def hash_set_sha1(hashes: set[int]) -> str:
    if not hashes:
        return hashlib.sha1(b"").hexdigest()
    arr = np.fromiter(hashes, dtype=np.uint64, count=len(hashes))
    arr.sort()
    return hashlib.sha1(arr.tobytes()).hexdigest()


def build_recipient_forbidden_hashes(
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
) -> tuple[set[int], dict[str, Any]]:
    train_hashes = g167.collect_13gram_hashes(train_ids, train_mask)
    heldout_hashes = g167.collect_13gram_hashes(val_ids, val_mask)
    forbidden_hashes = set(train_hashes)
    forbidden_hashes.update(heldout_hashes)
    meta = {
        "source": "recipient_main_train_plus_eval_token_13gram",
        "hash_len": g167.HASH_LEN,
        "main_train_hash_count": len(train_hashes),
        "main_eval_hash_count": len(heldout_hashes),
        "combined_forbidden_hash_count": len(forbidden_hashes),
        "combined_forbidden_hash_sha1": hash_set_sha1(forbidden_hashes),
        "main_train_seed": g165.C4_TRAIN_SEED,
        "main_eval_seed": g165.C4_VAL_SEED,
        "main_train_shape": list(train_ids.shape),
        "main_eval_shape": list(val_ids.shape),
    }
    return forbidden_hashes, meta


def estimate_alt_donor_wallclock_s(max_steps: int) -> float:
    return float(max_steps) * ALT_DONOR_EST_SEC_PER_STEP + ALT_DONOR_EST_EVAL_OVERHEAD_SEC


def overlap_filter_matches(metadata: dict[str, Any], forbidden_meta: dict[str, Any]) -> bool:
    dataset_train = metadata.get("dataset_train", {})
    dataset_eval = metadata.get("dataset_eval", {})
    expected_count = int(forbidden_meta["combined_forbidden_hash_count"])
    expected_sha1 = str(forbidden_meta["combined_forbidden_hash_sha1"])
    return (
        isinstance(dataset_train, dict)
        and isinstance(dataset_eval, dict)
        and dataset_train.get("overlap_filter") == "recipient_main_train_plus_eval_token_13gram"
        and dataset_eval.get("overlap_filter") == "recipient_main_train_plus_eval_token_13gram"
        and int(dataset_train.get("forbidden_hash_count", -1)) == expected_count
        and int(dataset_eval.get("forbidden_hash_count", -1)) == expected_count
        and metadata.get("forbidden_hash_sha1") == expected_sha1
    )


def run_anchor_cell(
    arm: AnchorArmSpec,
    *,
    seed: int,
    actual_lambda_0: float,
    target_params_device: dict[str, torch.Tensor] | None,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
    tok,
) -> dict[str, Any]:
    trajectory = g165.train_one_arm(
        arm_label=arm.label,
        lam0=actual_lambda_0,
        schedule_name=ANCHOR_SCHEDULE,
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
        "donor_seed": arm.donor_seed,
        "lambda_0": actual_lambda_0,
        "schedule": ANCHOR_SCHEDULE,
        "description": arm.description,
        "trajectory": trajectory,
        "final_nll": float(trajectory[-1]["nll"]),
        "initial_nll": float(trajectory[0]["nll"]),
    }


def final_nll(results: dict[str, Any], arm_label: str, seed: int) -> float:
    return float(results[arm_label][str(seed)]["final_nll"])


def build_summary(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload["results"]
    incomplete = [
        f"{arm.label}:{seed}"
        for arm in ARMS
        for seed in SEEDS
        if not cell_done(payload, arm.label, seed)
    ]
    if incomplete:
        return {
            "status": "incomplete",
            "missing_cells": incomplete,
            "verdict": f"INCOMPLETE: {len(incomplete)} / {len(ARMS) * len(SEEDS)} main cells missing.",
        }

    arm_final_nll = {
        arm.label: metric_summary(
            {str(seed): final_nll(results, arm.label, seed) for seed in SEEDS},
            seed=177100 + idx * 10,
        )
        for idx, arm in enumerate(ARMS)
    }

    vs_scratch: dict[str, Any] = {}
    for idx, label in enumerate(ANCHOR_LABELS):
        values = {
            str(seed): final_nll(results, "scratch_baseline", seed) - final_nll(results, label, seed)
            for seed in SEEDS
        }
        vs_scratch[label] = metric_summary(values, seed=177200 + idx * 10)

    qwen_gain = float(vs_scratch["anchor_qwen3_donor"]["mean"])
    alt_gain_by_label = {label: float(vs_scratch[label]["mean"]) for label in ALT_ARM_LABELS}
    best_alt_label = max(alt_gain_by_label, key=alt_gain_by_label.get)
    best_alt_gain = alt_gain_by_label[best_alt_label]

    qwen_minus_alt: dict[str, Any] = {}
    for idx, label in enumerate(ALT_ARM_LABELS):
        values = {
            str(seed): final_nll(results, label, seed) - final_nll(results, "anchor_qwen3_donor", seed)
            for seed in SEEDS
        }
        qwen_minus_alt[label] = metric_summary(values, seed=177300 + idx * 10)

    best_values = {
        str(seed): final_nll(results, best_alt_label, seed) - final_nll(results, "anchor_qwen3_donor", seed)
        for seed in SEEDS
    }
    qwen_minus_best_alt = metric_summary(best_values, seed=177400)

    qwen_reference_delta = qwen_gain - REFERENCE_QWEN3_DONOR_GAIN_NATS
    observed_80_threshold = NULL_80_FRACTION * qwen_gain
    reference_80_threshold = NULL_80_FRACTION * REFERENCE_QWEN3_DONOR_GAIN_NATS
    best_alt_ratio_observed = best_alt_gain / qwen_gain if qwen_gain > 0.0 else float("nan")
    best_alt_ratio_reference = best_alt_gain / REFERENCE_QWEN3_DONOR_GAIN_NATS

    margin_mean = float(qwen_minus_best_alt["mean"])
    margin_lo = float(qwen_minus_best_alt["ci_95_lo"])
    margin_hi = float(qwen_minus_best_alt["ci_95_hi"])
    ci_crosses_zero = bool(margin_lo <= 0.0 <= margin_hi)
    ci_excludes_zero_positive = bool(margin_lo > 0.0)
    matched_alt_within_80pct_observed = bool(qwen_gain > 0.0 and best_alt_gain >= observed_80_threshold)
    matched_alt_within_80pct_reference = bool(best_alt_gain >= reference_80_threshold)

    criteria = {
        "mean_qwen3_minus_best_alt_ge_0p5_nats": margin_mean >= PASS_IDENTITY_MARGIN_NATS,
        "paired_bootstrap_ci_excludes_zero_positive": ci_excludes_zero_positive,
        "mean_qwen3_minus_best_alt_in_intermediate_band": (
            FAIL_IDENTITY_MARGIN_NATS < margin_mean < PASS_IDENTITY_MARGIN_NATS
        ),
        "mean_qwen3_minus_best_alt_le_0p2_nats": margin_mean <= FAIL_IDENTITY_MARGIN_NATS,
        "paired_bootstrap_ci_crosses_zero": ci_crosses_zero,
        "best_alt_gain_ge_80pct_observed_qwen3_gain": matched_alt_within_80pct_observed,
        "best_alt_gain_ge_80pct_reference_qwen3_gain": matched_alt_within_80pct_reference,
        "qwen3_sanity_gain_within_0p10_nats_of_reference": abs(qwen_reference_delta) <= 0.10,
    }

    positive_direction_seeds = int(sum(float(value) > 0.0 for value in best_values.values()))

    if (
        criteria["mean_qwen3_minus_best_alt_le_0p2_nats"]
        or criteria["paired_bootstrap_ci_crosses_zero"]
        or criteria["best_alt_gain_ge_80pct_reference_qwen3_gain"]
    ):
        status = "identity_specificity_rejected"
        verdict = (
            "FAIL: donor-identity specificity REJECTED. "
            f"Best matched alt ({best_alt_label}) gain={best_alt_gain:+.3f} nats "
            f"({100.0 * best_alt_ratio_reference:.1f}% of +1.087 reference); "
            f"Qwen3-minus-best-alt margin={margin_mean:+.3f} nats, "
            f"95% CI [{margin_lo:+.3f}, {margin_hi:+.3f}]."
        )
    elif criteria["mean_qwen3_minus_best_alt_ge_0p5_nats"] and ci_excludes_zero_positive:
        status = "identity_specificity_locked"
        verdict = (
            "PASS: donor-identity specificity LOCKED. "
            f"Qwen3 beats best matched alt ({best_alt_label}) by {margin_mean:+.3f} nats "
            f"with 95% CI [{margin_lo:+.3f}, {margin_hi:+.3f}]."
        )
    else:
        status = "intermediate"
        verdict = (
            "INTERMEDIATE: partial donor-identity specificity. "
            f"Qwen3-minus-best-alt margin={margin_mean:+.3f} nats with 95% CI "
            f"[{margin_lo:+.3f}, {margin_hi:+.3f}]; best alt={best_alt_label}."
        )

    return {
        "status": status,
        "verdict": verdict,
        "arm_final_nll": arm_final_nll,
        "vs_scratch_final_nll_gain": vs_scratch,
        "qwen3_minus_each_alt_gain_nats": qwen_minus_alt,
        "qwen3_minus_best_alt_gain_nats": qwen_minus_best_alt,
        "best_alt_label_by_mean_gain": best_alt_label,
        "best_alt_gain_nats": best_alt_gain,
        "positive_direction_seeds_qwen3_better_than_best_alt": positive_direction_seeds,
        "observed_qwen3_effect_vs_scratch_nats": qwen_gain,
        "qwen3_reference_gain_nats": REFERENCE_QWEN3_DONOR_GAIN_NATS,
        "qwen3_observed_minus_reference_nats": qwen_reference_delta,
        "best_alt_fraction_of_observed_qwen3_effect": best_alt_ratio_observed,
        "best_alt_fraction_of_reference_qwen3_effect": best_alt_ratio_reference,
        "observed_80pct_qwen3_threshold_nats": observed_80_threshold,
        "reference_80pct_qwen3_threshold_nats": reference_80_threshold,
        "matched_alt_within_80pct_observed_qwen3": matched_alt_within_80pct_observed,
        "matched_alt_within_80pct_reference_qwen3": matched_alt_within_80pct_reference,
        "criteria": criteria,
    }


def lambda_for_cell(payload: dict[str, Any], arm: AnchorArmSpec, seed: int) -> float:
    if arm.target_kind == "none":
        return 0.0
    if arm.target_kind == "qwen3_trained":
        return ANCHOR_LAMBDA_QWEN3
    diag = payload["anchor_diagnostics"].get(arm.label)
    if not isinstance(diag, dict):
        raise RuntimeError(f"missing anchor diagnostics for {arm.label}")
    return float(diag["by_seed"][str(seed)]["actual_lambda_0"])


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
    arm = ARM_BY_LABEL["scratch_baseline"]
    for seed in SEEDS:
        if cell_done(payload, arm.label, seed):
            print(f"  skip complete cell {arm.label} seed={seed}")
            continue
        print(f"\n--- main arm={arm.label} seed={seed} ---")
        payload["results"][arm.label][str(seed)] = run_anchor_cell(
            arm,
            seed=seed,
            actual_lambda_0=0.0,
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


def run_qwen3_anchor_arm(
    payload: dict[str, Any],
    *,
    t_start: float,
    tok,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
) -> None:
    arm = ARM_BY_LABEL["anchor_qwen3_donor"]
    if "anchor_qwen3_donor" not in payload["anchor_diagnostics"] or not all_cells_done(payload, arm.label):
        qwen_params_cpu, qwen_metadata = load_qwen3_anchor_params(tok)
        validation = validate_target_params(qwen_params_cpu, label=arm.label)
        payload["target_npz"].setdefault(arm.label, {})
        payload["target_npz"][arm.label].setdefault("metadata", qwen_metadata)
        payload["target_npz"][arm.label]["shape_validation"] = validation
        payload["target_npz"][arm.label]["param_count"] = anchor_param_count(qwen_params_cpu)
        payload["target_npz"][arm.label]["norm_sq"] = anchor_state_norm_sq(qwen_params_cpu)
        write_payload(payload, t_start=t_start, incremental=True)

        qwen_params_device = stage_params_to_device(qwen_params_cpu)
        payload["anchor_diagnostics"][arm.label] = build_qwen3_anchor_diagnostics(
            compute_frobenius_by_seed(
                target_label=arm.label,
                target_params_device=qwen_params_device,
            )
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
                actual_lambda_0=ANCHOR_LAMBDA_QWEN3,
                target_params_device=qwen_params_device,
                train_ids=train_ids,
                train_mask=train_mask,
                val_ids=val_ids,
                val_mask=val_mask,
                tok=tok,
            )
            payload["summary"] = build_summary(payload)
            payload["verdict"] = payload["summary"]["verdict"]
            write_payload(payload, t_start=t_start, incremental=True)

        del qwen_params_cpu, qwen_params_device
        cleanup_cuda()
    else:
        print(f"  skip complete arm {arm.label}")


def run_alt_anchor_arm(
    payload: dict[str, Any],
    *,
    t_start: float,
    arm: AnchorArmSpec,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
    tok,
    allow_unmatched: bool,
    forbidden_meta: dict[str, Any],
) -> None:
    if arm.donor_seed is None:
        raise RuntimeError(f"{arm.label}: donor_seed missing")
    path = alt_npz_path(arm.donor_seed)
    if not path.exists():
        raise FileNotFoundError(
            f"missing alternative donor NPZ for seed={arm.donor_seed}: {path}. "
            "Run --stage alt-donors first."
        )
    if all_cells_done(payload, arm.label) and arm.label in payload["anchor_diagnostics"]:
        print(f"  skip complete arm {arm.label}")
        return
    if "anchor_qwen3_donor" not in payload["anchor_diagnostics"]:
        raise RuntimeError("qwen3 anchor diagnostics must be computed before alternative anchor arms")

    print(f"\n=== Loading alternative donor for main arm: {arm.label} ===")
    params_cpu, metadata = load_state_npz(path)
    matched = donor_metadata_is_matched(
        metadata,
        float(payload["config"]["alternative_donors"].get("effective_stop_nll", ALT_DONOR_TARGET_NLL)),
    )
    if not overlap_filter_matches(metadata, forbidden_meta):
        raise RuntimeError(
            f"{arm.label}: donor checkpoint was not trained with the recipient "
            "13-gram overlap filter required by g177v2. Rebuild with --force-alt-donors."
        )
    if not matched and not allow_unmatched:
        raise RuntimeError(
            f"{arm.label}: donor checkpoint is not matched to held-out C4 target. "
            "Use --allow-unmatched-donors only for diagnostic runs."
        )

    validation = validate_target_params(params_cpu, label=arm.label)
    payload["target_npz"].setdefault(arm.label, {})
    payload["target_npz"][arm.label].update(
        {
            "path": str(path),
            "exists": True,
            "loaded_existing": True,
            "size_bytes": path.stat().st_size,
            "metadata": metadata,
            "shape_validation": validation,
            "param_count": anchor_param_count(params_cpu),
            "norm_sq": anchor_state_norm_sq(params_cpu),
        }
    )
    write_payload(payload, t_start=t_start, incremental=True)

    params_device = stage_params_to_device(params_cpu)
    payload["anchor_diagnostics"][arm.label] = build_alt_anchor_diagnostics(
        alt_fro=compute_frobenius_by_seed(
            target_label=arm.label,
            target_params_device=params_device,
        ),
        qwen3_diag=payload["anchor_diagnostics"]["anchor_qwen3_donor"],
    )
    write_payload(payload, t_start=t_start, incremental=True)

    for seed in SEEDS:
        if cell_done(payload, arm.label, seed):
            print(f"  skip complete cell {arm.label} seed={seed}")
            continue
        actual_lambda = lambda_for_cell(payload, arm, seed)
        diag_row = payload["anchor_diagnostics"][arm.label]["by_seed"][str(seed)]
        print(
            f"\n--- main arm={arm.label} seed={seed} "
            f"lambda={actual_lambda:.6g} grad_l2={diag_row['anchor_grad_l2_at_actual_lambda']:.4f} ---"
        )
        payload["results"][arm.label][str(seed)] = run_anchor_cell(
            arm,
            seed=seed,
            actual_lambda_0=actual_lambda,
            target_params_device=params_device,
            train_ids=train_ids,
            train_mask=train_mask,
            val_ids=val_ids,
            val_mask=val_mask,
            tok=tok,
        )
        payload["results"][arm.label][str(seed)]["lambda_diagnostics"] = diag_row
        payload["summary"] = build_summary(payload)
        payload["verdict"] = payload["summary"]["verdict"]
        write_payload(payload, t_start=t_start, incremental=True)

    del params_cpu, params_device
    cleanup_cuda()


def run_main_cells(
    payload: dict[str, Any],
    *,
    t_start: float,
    tok,
    allow_unmatched: bool,
    forbidden_meta: dict[str, Any],
    main_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]] | None = None,
) -> None:
    print("\n=== Main matched-alt identity falsifier: 5 arms x 3 seeds ===")
    if main_data is None:
        train_ids, train_mask, val_ids, val_mask, data_meta = load_main_data(tok)
    else:
        train_ids, train_mask, val_ids, val_mask, data_meta = main_data
    payload["data"]["main_g165_c4"] = data_meta
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

    run_qwen3_anchor_arm(
        payload,
        t_start=t_start,
        tok=tok,
        train_ids=train_ids,
        train_mask=train_mask,
        val_ids=val_ids,
        val_mask=val_mask,
    )

    for label in ALT_ARM_LABELS:
        run_alt_anchor_arm(
            payload,
            t_start=t_start,
            arm=ARM_BY_LABEL[label],
            train_ids=train_ids,
            train_mask=train_mask,
            val_ids=val_ids,
            val_mask=val_mask,
            tok=tok,
            allow_unmatched=allow_unmatched,
            forbidden_meta=forbidden_meta,
        )

    payload["summary"] = build_summary(payload)
    payload["verdict"] = payload["summary"]["verdict"]
    write_payload(payload, t_start=t_start, incremental=True)


def print_active_ingredient_summary(summary: dict[str, Any]) -> None:
    if summary.get("status") == "incomplete":
        print(f"\n=== verdict: {summary.get('verdict', 'INCOMPLETE')} ===")
        return

    best_alt = summary["best_alt_label_by_mean_gain"]
    margin = summary["qwen3_minus_best_alt_gain_nats"]
    print("\n=== Active-ingredient analysis ===")
    print(f"  best_alt={best_alt}")
    for seed in SEEDS:
        value = float(margin["per_seed"][str(seed)])
        print(f"  seed={seed}: delta_qwen3_minus_best_alt={value:+.4f} nats")
    print(
        f"  mean_delta={float(margin['mean']):+.4f} nats "
        f"95%CI=[{float(margin['ci_95_lo']):+.4f}, {float(margin['ci_95_hi']):+.4f}]"
    )
    print(
        "  best_alt_within_80pct_reference_qwen3="
        f"{summary['matched_alt_within_80pct_reference_qwen3']} "
        f"(best_alt_gain={summary['best_alt_gain_nats']:+.4f}, "
        f"threshold={summary['reference_80pct_qwen3_threshold_nats']:+.4f})"
    )
    print(f"  verdict={summary['verdict']}")


def all_alt_npzs_exist() -> bool:
    return all(alt_npz_path(seed).exists() for seed in ALT_DONOR_SEEDS)


def alt_donors_blocking_main(payload: dict[str, Any]) -> list[int]:
    stop_nll = float(payload["config"]["alternative_donors"].get("effective_stop_nll", ALT_DONOR_TARGET_NLL))
    blocking = []
    for seed in ALT_DONOR_SEEDS:
        row = payload.get("alternative_donors", {}).get(str(seed), {})
        metadata = row.get("metadata") if isinstance(row, dict) else None
        if not isinstance(metadata, dict) or not donor_metadata_is_matched(metadata, stop_nll):
            blocking.append(seed)
    return blocking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genome 177 matched alternative donor identity falsifier.")
    parser.add_argument(
        "--stage",
        choices=["all", "alt-donors", "main"],
        default="all",
        help="Run the full experiment, only build/load alt donors, or only run main cells.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing result JSON and start a fresh payload.",
    )
    parser.add_argument(
        "--force-alt-donors",
        action="store_true",
        help="Retrain and rewrite alternative donor NPZ files even if they already exist.",
    )
    parser.add_argument(
        "--allow-unmatched-donors",
        action="store_true",
        help="Allow main cells to run if an alt donor did not reach the held-out C4 NLL target.",
    )
    parser.add_argument(
        "--alt-max-steps",
        type=int,
        default=ALT_DONOR_MAX_STEPS,
        help="Maximum C4 pretraining steps per alternative donor.",
    )
    parser.add_argument(
        "--alt-eval-every",
        type=int,
        default=ALT_DONOR_EVAL_EVERY,
        help="Held-out C4 eval cadence during alternative donor pretraining.",
    )
    parser.add_argument(
        "--alt-eval-windows",
        type=int,
        default=ALT_DONOR_EVAL_WINDOWS,
        help="Held-out C4 windows for alternative donor stopping eval.",
    )
    parser.add_argument(
        "--alt-target-nll",
        type=float,
        default=None,
        help="Explicit held-out C4 NLL stop threshold. Defaults to min(3.60, Qwen3_NLL + 0.05).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_locked_protocol()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print("genome_177: matched alternative donor identity falsifier")
    print(f"  model_arch={g165._MODEL_ID}")
    print(f"  device={DEVICE} forward_dtype={FORWARD_DTYPE}")
    print(f"  main seeds={SEEDS} steps={g165.N_STEPS} batch={g165.BATCH_SIZE}")
    print(
        f"  alt donor seeds={ALT_DONOR_SEEDS} max_steps={args.alt_max_steps} "
        f"target_nll={'auto' if args.alt_target_nll is None else args.alt_target_nll}"
    )
    print(
        f"  qwen3_lambda={ANCHOR_LAMBDA_QWEN3} schedule={ANCHOR_SCHEDULE} "
        f"reference_qwen3_gain={REFERENCE_QWEN3_DONOR_GAIN_NATS:+.3f} nats"
    )
    print(f"  output={OUT_PATH}")

    t_start = time.time()
    payload = load_or_create_payload(resume=not args.no_resume)
    tok = load_tokenizer()
    main_data = load_main_data(tok)
    train_ids, train_mask, val_ids, val_mask, data_meta = main_data
    forbidden_hashes, forbidden_meta = build_recipient_forbidden_hashes(
        train_ids,
        train_mask,
        val_ids,
        val_mask,
    )
    payload["data"]["main_g165_c4"] = data_meta
    payload["data"]["recipient_overlap_forbidden_hashes"] = forbidden_meta
    write_payload(payload, t_start=t_start, incremental=True)
    print(
        "  recipient overlap filter: "
        f"{forbidden_meta['combined_forbidden_hash_count']} token 13-grams "
        f"sha1={forbidden_meta['combined_forbidden_hash_sha1'][:12]}"
    )

    need_alt_donors = args.stage in {"all", "alt-donors"} or not all_alt_npzs_exist()
    if need_alt_donors:
        ensure_alternative_donors(
            tok,
            payload=payload,
            t_start=t_start,
            force=args.force_alt_donors,
            allow_unmatched=args.allow_unmatched_donors,
            max_steps=args.alt_max_steps,
            eval_every=args.alt_eval_every,
            eval_windows=args.alt_eval_windows,
            explicit_target_nll=args.alt_target_nll,
            forbidden_hashes=forbidden_hashes,
            forbidden_meta=forbidden_meta,
        )
        if args.stage == "alt-donors":
            payload["summary"] = build_summary(payload)
            payload["verdict"] = payload["summary"]["verdict"]
            write_payload(payload, t_start=t_start, incremental=False)
            for seed in ALT_DONOR_SEEDS:
                print(f"Saved/loaded alternative donor seed={seed}: {alt_npz_path(seed)}")
            print(f"Saved: {OUT_PATH} ({payload['elapsed_s']:.1f}s)")
            return
    elif args.stage == "alt-donors":
        print("  all alternative donor NPZ files already exist")
        return

    if args.stage == "all" and need_alt_donors and not args.allow_unmatched_donors:
        blocking = alt_donors_blocking_main(payload)
        if blocking:
            payload["summary"] = build_summary(payload)
            payload["verdict"] = (
                "INCOMPLETE: alternative donors not matched within the envelope; "
                f"blocking seeds={blocking}."
            )
            write_payload(payload, t_start=t_start, incremental=False)
            print(
                "  not launching main cells: unmatched or skipped alternative donors "
                f"block matched-condition inference (seeds={blocking})"
            )
            print(f"Saved: {OUT_PATH} ({payload['elapsed_s']:.1f}s)")
            return

    if args.stage in {"all", "main"}:
        run_main_cells(
            payload,
            t_start=t_start,
            tok=tok,
            allow_unmatched=args.allow_unmatched_donors,
            forbidden_meta=forbidden_meta,
            main_data=main_data,
        )

    payload["summary"] = build_summary(payload)
    payload["verdict"] = payload["summary"]["verdict"]
    write_payload(payload, t_start=t_start, incremental=False)
    print_active_ingredient_summary(payload["summary"])
    print(f"\nSaved: {OUT_PATH} ({payload['elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()
