"""
genome_181a_tokenizer_isolation.py

Cycle 65 tokenizer-isolation control for the g165 +1 nat anchor effect.

Question
--------
Is the continuous Qwen3 donor-anchor gain evidence of trained-structure transfer,
or is it mostly the Qwen3 tokenizer / embedding / lm-head surface acting as a
trained lexical prior?

Arms
----
  1. scratch_ce
  2. full_anchor, lambda=0.01 on all Qwen3 donor weights
  3. embed_lm_head_only_anchor, lambda gradient-matched to arm 2 at step 0
  4. no_embed_lm_head_anchor, lambda gradient-matched to arm 2 at step 0

Outputs
-------
  - results/genome_181a_tokenizer_isolation.json
"""
# ENVELOPE COMPLIANCE (cycle 65 g181a, 2026-04-28)
# 4 arms x 3 seeds x 2000 steps x ~0.4s/step = ~9600s. Setup, validation,
# donor snapshot, and incremental JSON writes budgeted at ~600s. Total expected
# wall time ~10200s, <= 14400s COMPUTE.md section 6 hard envelope. Max
# concurrent GPU state is one Qwen3-0.6B recipient plus one FP32 donor snapshot
# and batch activations, staying below the 22 GB VRAM ceiling.
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


OUT_PATH = ROOT / "results" / "genome_181a_tokenizer_isolation.json"

SEEDS = [42, 7, 13]
SEQ_LEN = g165.SEQ_LEN
BATCH_SIZE = g165.BATCH_SIZE
EVAL_BATCH_SIZE = g165.BATCH_SIZE
TRAIN_STEPS = 2000
N_TRAIN_WINDOWS = 8192
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
REFERENCE_G165_GAIN_NATS = 1.087
N_BOOT = 10_000

PASS_NO_EMBED_GAIN_NATS = 0.50
PASS_NO_EMBED_BEATS_EMBED_NATS = 0.30
FAIL_EMBED_GAIN_NATS = 0.50
FAIL_NO_EMBED_GAIN_MAX_NATS = 0.30

DEVICE = g165.DEVICE
FORWARD_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

EMBED_LM_HEAD_PARAM_NAMES = {
    "model.embed_tokens.weight",
    "lm_head.weight",
    "model.lm_head.weight",
}

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
    anchor_subset: str
    description: str

    @property
    def use_anchor(self) -> bool:
        return self.anchor_subset != "none"


ARMS = [
    AnchorArmSpec(
        label="scratch_ce",
        anchor_subset="none",
        description="No anchor; plain C4 CE from random Qwen3-0.6B init.",
    ),
    AnchorArmSpec(
        label="full_anchor",
        anchor_subset="all",
        description="lambda=0.01 Frobenius anchor on all matchable Qwen3 donor weights.",
    ),
    AnchorArmSpec(
        label="embed_lm_head_only_anchor",
        anchor_subset="embed_lm_head",
        description=(
            "Anchor only model.embed_tokens.weight and lm_head.weight, with lambda "
            "normalized per seed to match full-anchor initial gradient L2."
        ),
    ),
    AnchorArmSpec(
        label="no_embed_lm_head_anchor",
        anchor_subset="no_embed_lm_head",
        description=(
            "Anchor all matchable Qwen3 donor weights except embed_tokens and lm_head, "
            "with lambda normalized per seed to match full-anchor initial gradient L2."
        ),
    ),
]

ARM_BY_LABEL = {arm.label: arm for arm in ARMS}
ANCHOR_LABELS = [arm.label for arm in ARMS if arm.use_anchor]


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
        "genome": "181a",
        "name": "tokenizer_isolation",
        "timestamp_utc_started": now_utc(),
        "model_id": g165._MODEL_ID,
        "device": DEVICE,
        "forward_dtype": str(FORWARD_DTYPE),
        "config": {
            "seeds": SEEDS,
            "arms": [arm.__dict__ for arm in ARMS],
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
            "lambda_normalization": {
                "formula_embed_lm_head": "lambda_3 = lambda_full * sqrt(F2_full / F2_embed)",
                "formula_no_embed_lm_head": "lambda_4 = lambda_full * sqrt(F2_full / F2_no_embed)",
                "purpose": "equalize initial anchor-gradient L2 across anchored arms per recipient seed",
            },
            "pass_criteria": {
                "PASS": (
                    "no_embed_lm_head gain >= +0.5 nats vs scratch; no_embed_lm_head beats "
                    "embed_lm_head_only by >= +0.3 nats; 3/3 seeds positive; paired CI for "
                    "no_embed_lm_head minus embed_lm_head_only excludes zero"
                ),
                "FAIL_TOKENIZER": (
                    "embed_lm_head_only gain >= +0.5 nats and no_embed_lm_head gain < +0.3 nats"
                ),
                "INTERMEDIATE": "anything else",
            },
            "envelope_estimate_s": {
                "train": 4 * 3 * TRAIN_STEPS * 0.4,
                "setup_eval_overhead": 600,
                "total": 4 * 3 * TRAIN_STEPS * 0.4 + 600,
                "limit": 14_400,
            },
        },
        "data": {},
        "donor": {},
        "anchor_diagnostics": {},
        "results": {arm.label: {} for arm in ARMS},
        "summary": {},
        "build_summary": {},
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
    payload.setdefault("build_summary", {})
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


def all_cells_done(payload: dict[str, Any]) -> bool:
    return all(cell_done(payload, arm.label, seed) for arm in ARMS for seed in SEEDS)


def is_embed_lm_head_name(name: str) -> bool:
    return name in EMBED_LM_HEAD_PARAM_NAMES


def param_in_subset(name: str, subset: str) -> bool:
    if subset == "all":
        return True
    if subset == "embed_lm_head":
        return is_embed_lm_head_name(name)
    if subset == "no_embed_lm_head":
        return not is_embed_lm_head_name(name)
    if subset == "none":
        return False
    raise ValueError(f"unknown anchor subset {subset!r}")


def snapshot_params_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    raw = g165.snapshot_donor_params(model)
    params = {name: tensor.detach().cpu().contiguous() for name, tensor in raw.items()}
    del raw
    cleanup_cuda()
    return params


def stage_params_to_device(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.to(DEVICE) for name, tensor in params.items()}


def anchor_param_count(params: dict[str, torch.Tensor], subset: str) -> int:
    return int(sum(tensor.numel() for name, tensor in params.items() if param_in_subset(name, subset)))


def anchor_state_norm_sq(params: dict[str, torch.Tensor], subset: str) -> float:
    total = 0.0
    for name, tensor in params.items():
        if param_in_subset(name, subset):
            total += float((tensor.float() ** 2).sum().item())
    return total


def load_trained_anchor_params(tok) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    donor, _ = g165.load_trained_donor(tok)
    if hasattr(donor.config, "use_cache"):
        donor.config.use_cache = False
    params = snapshot_params_cpu(donor)
    metadata = {
        "target_kind": "qwen3_trained",
        "source_model": g165._MODEL_ID,
        "construction": "trained donor snapshot via genome_165.load_trained_donor",
        "param_count_all": anchor_param_count(params, "all"),
        "param_count_embed_lm_head": anchor_param_count(params, "embed_lm_head"),
        "param_count_no_embed_lm_head": anchor_param_count(params, "no_embed_lm_head"),
        "norm_sq_all": anchor_state_norm_sq(params, "all"),
        "norm_sq_embed_lm_head": anchor_state_norm_sq(params, "embed_lm_head"),
        "norm_sq_no_embed_lm_head": anchor_state_norm_sq(params, "no_embed_lm_head"),
    }
    del donor
    cleanup_cuda()
    return params, metadata


def validate_anchor_params(params: dict[str, torch.Tensor], *, reference_seed: int = SEEDS[0]) -> dict[str, Any]:
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
    tied_input_output = False
    try:
        tied_input_output = (
            recipient.get_input_embeddings().weight is recipient.get_output_embeddings().weight
        )
    except Exception:
        tied_input_output = False

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
        raise RuntimeError(f"trained donor params do not match recipient architecture: {preview}")

    return {
        "reference_seed": reference_seed,
        "n_model_tensors": len(model_shapes),
        "n_target_tensors": len(target_shapes),
        "n_extra_tensors": len(extra),
        "extra_tensors_first10": extra[:10],
        "n_model_params": n_model_params,
        "n_target_matched_params": n_target_matched_params,
        "all_model_params_matched": n_target_matched_params == n_model_params,
        "embed_lm_head_param_names_requested": sorted(EMBED_LM_HEAD_PARAM_NAMES),
        "embed_lm_head_param_names_present": sorted(
            name for name in EMBED_LM_HEAD_PARAM_NAMES if name in model_shapes
        ),
        "input_output_embeddings_tied_in_recipient": tied_input_output,
    }


def frobenius_sq_for_subset(
    recipient: torch.nn.Module,
    donor_params_device: dict[str, torch.Tensor],
    subset: str,
) -> tuple[float, int, int]:
    total = torch.zeros((), device=DEVICE, dtype=torch.float32)
    matched_tensors = 0
    matched_params = 0
    with torch.no_grad():
        for name, param in recipient.named_parameters():
            if name not in donor_params_device:
                continue
            donor_tensor = donor_params_device[name]
            if param.shape != donor_tensor.shape:
                continue
            if not param_in_subset(name, subset):
                continue
            total = total + ((param.detach().to(torch.float32) - donor_tensor) ** 2).sum()
            matched_tensors += 1
            matched_params += int(param.numel())
    if matched_tensors == 0 and subset != "none":
        raise RuntimeError(f"no matched tensors for anchor subset {subset!r}")
    return float(total.item()), matched_tensors, matched_params


def build_anchor_pairs(
    recipient: torch.nn.Module,
    donor_params_device: dict[str, torch.Tensor],
    subset: str,
) -> list[tuple[str, torch.nn.Parameter, torch.Tensor]]:
    pairs = []
    for name, param in recipient.named_parameters():
        if name not in donor_params_device:
            continue
        donor_tensor = donor_params_device[name]
        if param.shape != donor_tensor.shape:
            continue
        if not param_in_subset(name, subset):
            continue
        pairs.append((name, param, donor_tensor))
    if not pairs and subset != "none":
        raise RuntimeError(f"no anchor pairs for subset {subset!r}")
    return pairs


def compute_lambda_diagnostics(
    donor_params_device: dict[str, torch.Tensor],
) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}

    for seed in SEEDS:
        recipient = g165.load_random_init(seed)
        f2_embed, n_embed_tensors, n_embed_params = frobenius_sq_for_subset(
            recipient,
            donor_params_device,
            "embed_lm_head",
        )
        f2_no_embed, n_no_embed_tensors, n_no_embed_params = frobenius_sq_for_subset(
            recipient,
            donor_params_device,
            "no_embed_lm_head",
        )
        # Cycle 66 SEV-6: independent partition-consistency check via direct full computation.
        f2_full_direct, n_full_tensors, n_full_params = frobenius_sq_for_subset(
            recipient,
            donor_params_device,
            "all",
        )
        f2_full = f2_embed + f2_no_embed
        rel_err = abs(f2_full - f2_full_direct) / max(f2_full_direct, 1e-30)
        if rel_err > 1e-6:
            raise RuntimeError(
                f"Frobenius^2 partition mismatch seed={seed}: "
                f"f2_embed+f2_no_embed={f2_full:.6f} vs f2_full_direct={f2_full_direct:.6f} "
                f"rel_err={rel_err:.2e} > 1e-6 -- masking has overlap or gap"
            )
        if f2_embed <= 0.0 or f2_no_embed <= 0.0:
            raise RuntimeError(
                f"non-positive Frobenius component seed={seed}: embed={f2_embed}, no_embed={f2_no_embed}"
            )

        lambda_embed = ANCHOR_LAMBDA_FULL * math.sqrt(f2_full / f2_embed)
        lambda_no_embed = ANCHOR_LAMBDA_FULL * math.sqrt(f2_full / f2_no_embed)
        full_grad_l2 = 2.0 * ANCHOR_LAMBDA_FULL * math.sqrt(f2_full)
        embed_grad_l2 = 2.0 * lambda_embed * math.sqrt(f2_embed)
        no_embed_grad_l2 = 2.0 * lambda_no_embed * math.sqrt(f2_no_embed)

        by_seed[str(seed)] = {
            "seed": seed,
            "frobenius_sq_full": f2_full,
            "frobenius_sq_embed_lm_head": f2_embed,
            "frobenius_sq_no_embed_lm_head": f2_no_embed,
            "frobenius_sq_no_embed_lm_head_by_difference": f2_full - f2_embed,
            "frobenius_full": math.sqrt(f2_full),
            "frobenius_embed_lm_head": math.sqrt(f2_embed),
            "frobenius_no_embed_lm_head": math.sqrt(f2_no_embed),
            "lambda_full_anchor": ANCHOR_LAMBDA_FULL,
            "lambda_embed_lm_head_only_anchor": lambda_embed,
            "lambda_no_embed_lm_head_anchor": lambda_no_embed,
            "lambda_embed_lm_head_multiplier_vs_full": lambda_embed / ANCHOR_LAMBDA_FULL,
            "lambda_no_embed_lm_head_multiplier_vs_full": lambda_no_embed / ANCHOR_LAMBDA_FULL,
            "anchor_grad_l2_full_anchor": full_grad_l2,
            "anchor_grad_l2_embed_lm_head_only_anchor": embed_grad_l2,
            "anchor_grad_l2_no_embed_lm_head_anchor": no_embed_grad_l2,
            "embed_minus_full_anchor_grad_l2": embed_grad_l2 - full_grad_l2,
            "no_embed_minus_full_anchor_grad_l2": no_embed_grad_l2 - full_grad_l2,
            "matched_tensors_embed_lm_head": n_embed_tensors,
            "matched_tensors_no_embed_lm_head": n_no_embed_tensors,
            "matched_params_embed_lm_head": n_embed_params,
            "matched_params_no_embed_lm_head": n_no_embed_params,
            "matched_params_full": n_embed_params + n_no_embed_params,
            "frobenius_sq_full_direct": f2_full_direct,
            "matched_tensors_full_direct": n_full_tensors,
            "matched_params_full_direct": n_full_params,
            "frobenius_sq_partition_relative_error": rel_err,
        }
        del recipient
        cleanup_cuda()

    embed_lambdas = [by_seed[str(seed)]["lambda_embed_lm_head_only_anchor"] for seed in SEEDS]
    no_embed_lambdas = [by_seed[str(seed)]["lambda_no_embed_lm_head_anchor"] for seed in SEEDS]
    f2_full_values = [by_seed[str(seed)]["frobenius_sq_full"] for seed in SEEDS]
    f2_embed_values = [by_seed[str(seed)]["frobenius_sq_embed_lm_head"] for seed in SEEDS]
    f2_no_embed_values = [by_seed[str(seed)]["frobenius_sq_no_embed_lm_head"] for seed in SEEDS]

    return {
        "lambda_policy": "per_seed_initial_anchor_gradient_l2_match_to_full_anchor",
        "lambda_full_anchor": ANCHOR_LAMBDA_FULL,
        "formula": {
            "embed_lm_head_only_anchor": "lambda_full * sqrt(F2_full / F2_embed)",
            "no_embed_lm_head_anchor": "lambda_full * sqrt(F2_full / F2_no_embed)",
        },
        "by_seed": by_seed,
        "means": {
            "frobenius_sq_full": float(np.mean(f2_full_values)),
            "frobenius_sq_embed_lm_head": float(np.mean(f2_embed_values)),
            "frobenius_sq_no_embed_lm_head": float(np.mean(f2_no_embed_values)),
            "lambda_embed_lm_head_only_anchor": float(np.mean(embed_lambdas)),
            "lambda_no_embed_lm_head_anchor": float(np.mean(no_embed_lambdas)),
        },
        "stds": {
            "frobenius_sq_full": float(np.std(f2_full_values)),
            "frobenius_sq_embed_lm_head": float(np.std(f2_embed_values)),
            "frobenius_sq_no_embed_lm_head": float(np.std(f2_no_embed_values)),
            "lambda_embed_lm_head_only_anchor": float(np.std(embed_lambdas)),
            "lambda_no_embed_lm_head_anchor": float(np.std(no_embed_lambdas)),
        },
    }


def lambda_for_cell(payload: dict[str, Any], arm: AnchorArmSpec, seed: int) -> float:
    if arm.anchor_subset == "none":
        return 0.0
    if arm.anchor_subset == "all":
        return ANCHOR_LAMBDA_FULL
    diag = payload.get("anchor_diagnostics", {}).get("lambda_and_frobenius")
    if not isinstance(diag, dict):
        raise RuntimeError("missing lambda_and_frobenius diagnostics")
    row = diag["by_seed"][str(seed)]
    if arm.anchor_subset == "embed_lm_head":
        return float(row["lambda_embed_lm_head_only_anchor"])
    if arm.anchor_subset == "no_embed_lm_head":
        return float(row["lambda_no_embed_lm_head_anchor"])
    raise ValueError(f"unknown anchor subset {arm.anchor_subset!r}")


def load_main_data(tok) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    train_ids, train_mask, train_meta = g167.load_c4_windows(
        tok,
        split="train",
        seed=C4_TRAIN_SEED,
        n_windows=N_TRAIN_WINDOWS,
    )
    train_hashes = g167.collect_13gram_hashes(train_ids, train_mask)
    val_ids, val_mask, val_meta = g167.load_c4_windows(
        tok,
        split="train",
        seed=C4_VAL_SEED,
        n_windows=N_C4_VAL_WINDOWS,
        forbidden_hashes=train_hashes,
    )
    val_hashes = g167.collect_13gram_hashes(val_ids, val_mask)
    overlap = len(train_hashes.intersection(val_hashes))
    meta = {
        "protocol_source": "g165 Qwen3 C4 train-stream protocol with explicit 8192/256 window counts",
        "train": train_meta,
        "c4_val": val_meta,
        "train_seed": C4_TRAIN_SEED,
        "val_seed": C4_VAL_SEED,
        "train_shape": list(train_ids.shape),
        "val_shape": list(val_ids.shape),
        "train_13gram_hash_count": len(train_hashes),
        "val_13gram_hash_count": len(val_hashes),
        "train_val_13gram_overlap_count": overlap,
        "val_forbidden_hash_source": "train windows",
    }
    if overlap != 0:
        raise RuntimeError(f"C4 train/val token 13-gram overlap detected: {overlap}")
    return train_ids, train_mask, val_ids, val_mask, meta


@torch.no_grad()
def evaluate_nll(model: torch.nn.Module, eval_ids: torch.Tensor, eval_mask: torch.Tensor) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_top1 = 0
    for start in range(0, eval_ids.shape[0], EVAL_BATCH_SIZE):
        ids = eval_ids[start : start + EVAL_BATCH_SIZE].to(DEVICE)
        mask = eval_mask[start : start + EVAL_BATCH_SIZE].to(DEVICE)
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
        preds = shift_logits.argmax(dim=-1)
        total_top1 += int(((preds == shift_labels) & shift_mask).sum().item())
    model.train()
    return {
        "nll": total_loss / max(total_tokens, 1),
        "top1_acc": total_top1 / max(total_tokens, 1),
        "n_tokens": float(total_tokens),
    }


def train_anchor_cell(
    arm: AnchorArmSpec,
    *,
    seed: int,
    actual_lambda_0: float,
    donor_params_device: dict[str, torch.Tensor] | None,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    recipient = g165.load_random_init(seed)
    if hasattr(recipient.config, "use_cache"):
        recipient.config.use_cache = False

    optimizer = torch.optim.AdamW(
        recipient.parameters(),
        lr=LR,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    train_schedule = g167.build_train_schedule(seed, n_examples=int(train_ids.shape[0]))
    if train_schedule.shape[0] < TRAIN_STEPS:
        rng = np.random.default_rng(seed)
        train_schedule = rng.integers(
            0,
            int(train_ids.shape[0]),
            size=(TRAIN_STEPS, BATCH_SIZE),
            dtype=np.int64,
        )
    else:
        train_schedule = train_schedule[:TRAIN_STEPS]

    anchor_pairs: list[tuple[str, torch.nn.Parameter, torch.Tensor]] = []
    if arm.use_anchor:
        if donor_params_device is None:
            raise RuntimeError(f"{arm.label} requires donor params")
        anchor_pairs = build_anchor_pairs(recipient, donor_params_device, arm.anchor_subset)

    trajectory: list[dict[str, Any]] = []
    initial_metrics = evaluate_nll(recipient, val_ids, val_mask)
    trajectory.append({"step": 0, **initial_metrics})
    print(
        f"    {arm.label} seed={seed} lambda={actual_lambda_0:.8g} "
        f"step=0 nll={initial_metrics['nll']:.4f}"
    )

    t0 = time.time()
    recipient.train()
    for step in range(1, TRAIN_STEPS + 1):
        batch_indices = train_schedule[step - 1]
        batch_index_tensor = torch.as_tensor(batch_indices, dtype=torch.long)
        ids = train_ids[batch_index_tensor].to(DEVICE)
        mask = train_mask[batch_index_tensor].to(DEVICE)

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
            row = {
                "step": step,
                "ce_loss": float(ce_loss.item()),
                "elapsed_s": time.time() - t0,
            }
            if step % EVAL_EVERY == 0 or step == TRAIN_STEPS:
                row.update(evaluate_nll(recipient, val_ids, val_mask))
                print(
                    f"    {arm.label} seed={seed} step={step} "
                    f"ce={row['ce_loss']:.4f} val_nll={row['nll']:.4f} "
                    f"({row['elapsed_s']:.0f}s)"
                )
            elif step % (LOG_EVERY * 5) == 0:
                print(
                    f"    {arm.label} seed={seed} step={step} "
                    f"ce={row['ce_loss']:.4f} ({row['elapsed_s']:.0f}s)"
                )
            trajectory.append(row)

    final_metrics = trajectory[-1]
    if "nll" not in final_metrics:
        final_metrics = {"step": TRAIN_STEPS, **evaluate_nll(recipient, val_ids, val_mask)}
        trajectory.append(final_metrics)

    wallclock_s = time.time() - t0
    result = {
        "seed": seed,
        "arm_label": arm.label,
        "anchor_subset": arm.anchor_subset,
        "description": arm.description,
        "lambda_0": actual_lambda_0,
        "schedule": ANCHOR_SCHEDULE,
        "anchor_pair_count": len(anchor_pairs),
        "anchor_param_count": int(sum(param.numel() for _, param, _ in anchor_pairs)),
        "initial_metrics": initial_metrics,
        "final_nll": float(final_metrics["nll"]),
        "final_top1_acc": float(final_metrics["top1_acc"]),
        "trajectory": trajectory,
        "wallclock_s": wallclock_s,
    }

    del recipient, optimizer
    cleanup_cuda()
    return result


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
            "verdict": f"INCOMPLETE: {len(incomplete)} / {len(ARMS) * len(SEEDS)} cells missing.",
        }

    arm_final_nll = {
        arm.label: metric_summary(
            {str(seed): final_nll(results, arm.label, seed) for seed in SEEDS},
            seed=181100 + idx * 10,
        )
        for idx, arm in enumerate(ARMS)
    }

    per_arm_per_seed = {
        arm.label: {
            str(seed): {
                "final_c4_nll": final_nll(results, arm.label, seed),
                "gain_vs_scratch_nats": (
                    0.0
                    if arm.label == "scratch_ce"
                    else final_nll(results, "scratch_ce", seed) - final_nll(results, arm.label, seed)
                ),
            }
            for seed in SEEDS
        }
        for arm in ARMS
    }

    vs_scratch: dict[str, Any] = {}
    for idx, label in enumerate(ANCHOR_LABELS):
        values = {
            str(seed): final_nll(results, "scratch_ce", seed) - final_nll(results, label, seed)
            for seed in SEEDS
        }
        vs_scratch[label] = metric_summary(values, seed=181200 + idx * 10)

    no_embed_minus_embed_values = {
        str(seed): final_nll(results, "embed_lm_head_only_anchor", seed)
        - final_nll(results, "no_embed_lm_head_anchor", seed)
        for seed in SEEDS
    }
    no_embed_minus_embed = metric_summary(no_embed_minus_embed_values, seed=181300)

    full_minus_embed_values = {
        str(seed): final_nll(results, "embed_lm_head_only_anchor", seed)
        - final_nll(results, "full_anchor", seed)
        for seed in SEEDS
    }
    full_minus_no_embed_values = {
        str(seed): final_nll(results, "no_embed_lm_head_anchor", seed)
        - final_nll(results, "full_anchor", seed)
        for seed in SEEDS
    }

    full_minus_embed = metric_summary(full_minus_embed_values, seed=181310)
    full_minus_no_embed = metric_summary(full_minus_no_embed_values, seed=181320)

    no_embed_gain = float(vs_scratch["no_embed_lm_head_anchor"]["mean"])
    embed_gain = float(vs_scratch["embed_lm_head_only_anchor"]["mean"])
    no_embed_beats_embed = float(no_embed_minus_embed["mean"])
    no_embed_embed_ci_lo = float(no_embed_minus_embed["ci_95_lo"])
    no_embed_embed_ci_hi = float(no_embed_minus_embed["ci_95_hi"])
    positive_direction_seeds = int(sum(float(v) > 0.0 for v in no_embed_minus_embed_values.values()))

    criteria = {
        "no_embed_gain_ge_0p5_nats_vs_scratch": no_embed_gain >= PASS_NO_EMBED_GAIN_NATS,
        "no_embed_beats_embed_by_ge_0p3_nats": no_embed_beats_embed >= PASS_NO_EMBED_BEATS_EMBED_NATS,
        "positive_direction_no_embed_gt_embed_3_of_3": positive_direction_seeds == len(SEEDS),
        "ci_no_embed_minus_embed_excludes_zero_positive": no_embed_embed_ci_lo > 0.0,
        "embed_gain_ge_0p5_nats_vs_scratch": embed_gain >= FAIL_EMBED_GAIN_NATS,
        "no_embed_gain_lt_0p3_nats_vs_scratch": no_embed_gain < FAIL_NO_EMBED_GAIN_MAX_NATS,
    }

    if (
        criteria["no_embed_gain_ge_0p5_nats_vs_scratch"]
        and criteria["no_embed_beats_embed_by_ge_0p3_nats"]
        and criteria["positive_direction_no_embed_gt_embed_3_of_3"]
        and criteria["ci_no_embed_minus_embed_excludes_zero_positive"]
    ):
        status = "PASS"
        verdict = (
            "PASS: non-embedding/non-lm-head donor weights preserve the anchor effect. "
            f"no_embed_lm_head gain={no_embed_gain:+.3f} nats vs scratch; "
            f"no_embed-minus-embed={no_embed_beats_embed:+.3f} nats with 95% CI "
            f"[{no_embed_embed_ci_lo:+.3f}, {no_embed_embed_ci_hi:+.3f}]."
        )
    elif (
        criteria["embed_gain_ge_0p5_nats_vs_scratch"]
        and criteria["no_embed_gain_lt_0p3_nats_vs_scratch"]
    ):
        status = "FAIL_TOKENIZER"
        verdict = (
            "FAIL_TOKENIZER: embedding/lm-head anchor carries the effect while "
            f"no_embed_lm_head does not. embed gain={embed_gain:+.3f} nats; "
            f"no_embed gain={no_embed_gain:+.3f} nats."
        )
    else:
        status = "INTERMEDIATE"
        verdict = (
            "INTERMEDIATE: tokenizer-isolation criteria did not cleanly pass or fail. "
            f"embed gain={embed_gain:+.3f} nats; no_embed gain={no_embed_gain:+.3f} nats; "
            f"no_embed-minus-embed={no_embed_beats_embed:+.3f} nats with 95% CI "
            f"[{no_embed_embed_ci_lo:+.3f}, {no_embed_embed_ci_hi:+.3f}]."
        )

    return {
        "status": status,
        "verdict": verdict,
        "per_arm_per_seed": per_arm_per_seed,
        "arm_final_nll": arm_final_nll,
        "vs_scratch_final_nll_gain": vs_scratch,
        "no_embed_minus_embed_nll_gain": no_embed_minus_embed,
        "full_minus_embed_nll_gain": full_minus_embed,
        "full_minus_no_embed_nll_gain": full_minus_no_embed,
        "positive_direction_seeds_no_embed_gt_embed": positive_direction_seeds,
        "criteria": criteria,
        "reference_g165_full_anchor_gain_nats": REFERENCE_G165_GAIN_NATS,
    }


def build_codex_sev7_gate(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("status") == "incomplete":
        return {
            "gate": "SEV-7_INCOMPLETE",
            "blocking": True,
            "reason": "All 12 cells must finish before tokenizer-artifact verdict can be interpreted.",
        }

    status = summary.get("status")
    criteria = summary.get("criteria", {})
    no_embed_minus_embed = summary.get("no_embed_minus_embed_nll_gain", {})
    return {
        "gate": f"SEV-7_{status}",
        "blocking": status == "FAIL_TOKENIZER",
        "status": status,
        "criteria": criteria,
        "primary_margin_no_embed_minus_embed_nats": no_embed_minus_embed,
        "interpretation": {
            "PASS": "Transfer story survives the tokenizer-prior attack.",
            "FAIL_TOKENIZER": "C18/C19/C21 collapse to tokenizer-init/lexical-prior artifact until rescued.",
            "INTERMEDIATE": "Effect is not cleanly localized; requires follow-up before strong transfer framing.",
        }.get(str(status), "Run incomplete."),
    }


def refresh_build_summary(payload: dict[str, Any]) -> None:
    summary = build_summary(payload)
    payload["summary"] = summary
    payload["build_summary"] = {
        "design": (
            "Four-arm tokenizer isolation of g165 full-anchor effect: scratch CE, full donor anchor, "
            "embed/lm-head-only donor anchor, and all-except-embed/lm-head donor anchor."
        ),
        "lambda_values_and_frobenius_breakdown": payload.get("anchor_diagnostics", {}).get(
            "lambda_and_frobenius",
            {},
        ),
        "pass_criteria_text": payload["config"]["pass_criteria"],
        "codex_sev7_verdict_gate": build_codex_sev7_gate(summary),
        "envelope_compliance": payload["config"]["envelope_estimate_s"],
    }
    payload["verdict"] = summary.get("verdict", "INCOMPLETE")


def run_cells(
    payload: dict[str, Any],
    *,
    t_start: float,
    donor_params_device: dict[str, torch.Tensor],
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
) -> None:
    for seed in SEEDS:
        for arm in ARMS:
            if cell_done(payload, arm.label, seed):
                print(f"  skip complete cell {arm.label} seed={seed}")
                continue
            actual_lambda = lambda_for_cell(payload, arm, seed)
            diag = {}
            if arm.use_anchor:
                diag = payload["anchor_diagnostics"]["lambda_and_frobenius"]["by_seed"][str(seed)]
            print(
                f"\n--- arm={arm.label} seed={seed} subset={arm.anchor_subset} "
                f"lambda={actual_lambda:.8g} ---"
            )
            payload["results"][arm.label][str(seed)] = train_anchor_cell(
                arm,
                seed=seed,
                actual_lambda_0=actual_lambda,
                donor_params_device=donor_params_device if arm.use_anchor else None,
                train_ids=train_ids,
                train_mask=train_mask,
                val_ids=val_ids,
                val_mask=val_mask,
            )
            if diag:
                payload["results"][arm.label][str(seed)]["lambda_diagnostics"] = diag
            refresh_build_summary(payload)
            write_payload(payload, t_start=t_start, incremental=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genome 181a tokenizer-isolation control.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing result JSON and start a fresh payload.",
    )
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Load donor/data and write lambda diagnostics, but do not train cells.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("genome_181a: tokenizer-isolation control")
    print(f"  model={g165._MODEL_ID}")
    print(f"  device={DEVICE} forward_dtype={FORWARD_DTYPE}")
    print(f"  arms={[arm.label for arm in ARMS]}")
    print(f"  seeds={SEEDS} steps={TRAIN_STEPS} batch={BATCH_SIZE} seq_len={SEQ_LEN}")
    print(f"  output={OUT_PATH}")

    t_start = time.time()
    payload = load_or_create_payload(resume=not args.no_resume)

    donor_model, tok = g165.load_trained_donor()
    if hasattr(donor_model.config, "use_cache"):
        donor_model.config.use_cache = False
    donor_params_cpu = snapshot_params_cpu(donor_model)
    del donor_model
    cleanup_cuda()

    validation = validate_anchor_params(donor_params_cpu)
    payload["donor"] = {
        "metadata": {
            "target_kind": "qwen3_trained",
            "source_model": g165._MODEL_ID,
            "construction": "trained donor snapshot via genome_165.load_trained_donor",
        },
        "validation": validation,
        "param_count_all": anchor_param_count(donor_params_cpu, "all"),
        "param_count_embed_lm_head": anchor_param_count(donor_params_cpu, "embed_lm_head"),
        "param_count_no_embed_lm_head": anchor_param_count(donor_params_cpu, "no_embed_lm_head"),
        "norm_sq_all": anchor_state_norm_sq(donor_params_cpu, "all"),
        "norm_sq_embed_lm_head": anchor_state_norm_sq(donor_params_cpu, "embed_lm_head"),
        "norm_sq_no_embed_lm_head": anchor_state_norm_sq(donor_params_cpu, "no_embed_lm_head"),
    }

    donor_params_device = stage_params_to_device(donor_params_cpu)
    del donor_params_cpu
    cleanup_cuda()

    if "lambda_and_frobenius" not in payload.get("anchor_diagnostics", {}):
        payload["anchor_diagnostics"]["lambda_and_frobenius"] = compute_lambda_diagnostics(donor_params_device)

    train_ids, train_mask, val_ids, val_mask, data_meta = load_main_data(tok)
    payload["data"] = data_meta
    refresh_build_summary(payload)
    write_payload(payload, t_start=t_start, incremental=True)

    if args.diagnostics_only:
        print("  diagnostics-only requested; not training cells.")
        return

    if all_cells_done(payload):
        print("  all cells already complete in result JSON; recomputing summary only.")
    else:
        run_cells(
            payload,
            t_start=t_start,
            donor_params_device=donor_params_device,
            train_ids=train_ids,
            train_mask=train_mask,
            val_ids=val_ids,
            val_mask=val_mask,
        )

    refresh_build_summary(payload)
    write_payload(payload, t_start=t_start, incremental=False)
    print(f"\n=== verdict: {payload['verdict']} ===")
    print(f"Saved: {OUT_PATH} ({payload['elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()
