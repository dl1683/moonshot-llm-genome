"""
genome_169_scaffold_swap_distillation.py

Functional scaffold distillation ("ScaffoldSwap") for the post-g165/g168
transfer-axis rethink.

Design
------
Donor:
  - Qwen3-0.6B, pretrained, frozen, BF16 on GPU

Recipient:
  - Random-init Qwen3-0.6B-architecture model
  - FP32 parameters optimized by AdamW
  - BF16 forward under autocast on GPU

Training:
  - At every transformer-block boundary, run donor and recipient blocks in
    parallel, then inject donor computation directly into the shared residual
    stream:

        h_mix = alpha(t) * h_donor + (1 - alpha(t)) * h_recipient

    The donor path is wrapped in no-grad, so gradients flow only through the
    recipient contribution while the donor acts as a decaying activation
    teacher.
  - The mixed residual stream is then fed into BOTH the next donor block and
    the next recipient block.
  - Loss is standard next-token CE on the recipient logits.

Arms:
  - step           : alpha(t) = 1      if t < 50   else 0
  - linear         : alpha(t) = max(0, 1 - t / 250)
  - exponential    : alpha(t) = exp(-t / 50)
  - constant_zero  : alpha(t) = 0                    (scratch control)
  - constant_full  : donor-only upper bound control  (recipient not updated)

Eval:
  - Pure recipient forward only (no scaffold) at steps 0 / 50 / 250 / 500
  - Metric: mean C4-validation NLL on 200 windows of length 256

PASS (locked):
  - At least one decay schedule (step / linear / exponential) achieves
    recipient-alone final C4 NLL advantage >= +0.5 nats vs scratch with paired
    bootstrap 95% CI excluding 0.

FAIL (locked):
  - All 3 decay schedules wash out below +0.2 nats OR CI crosses 0.
"""
from __future__ import annotations

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

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "models"))
try:
    from registry import resolve as _resolve_model  # type: ignore

    _MODEL_ID = _resolve_model("qwen3-0.6b").get("hf_id", "Qwen/Qwen3-0.6B")
except Exception:
    _MODEL_ID = "Qwen/Qwen3-0.6B"

OUT_PATH = ROOT / "results" / "genome_169_scaffold_swap_distillation.json"

SEEDS = [42, 7, 13]
SEQ_LEN = 256
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
N_STEPS = 500
EVAL_STEPS = [0, 50, 250, 500]
TRAIN_WINDOWS = N_STEPS * TRAIN_BATCH_SIZE
N_C4_VAL_WINDOWS = 200
LR = 3e-4
BETAS = (0.9, 0.95)
N_BOOT = 10_000

C4_TRAIN_SEED = 165
C4_VAL_SEED = 1650

PASS_ADVANTAGE_NATS = 0.5
FAIL_WASHOUT_NATS = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FORWARD_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


@dataclass(frozen=True)
class ArmSpec:
    label: str
    schedule: str
    counts_for_verdict: bool
    description: str


ARM_SPECS = [
    ArmSpec(
        label="scaffold_step",
        schedule="step",
        counts_for_verdict=True,
        description="Hard donor scaffold for first 50 updates, then recipient alone.",
    ),
    ArmSpec(
        label="scaffold_linear",
        schedule="linear",
        counts_for_verdict=True,
        description="Linearly annealed donor scaffold to zero by ~250 updates.",
    ),
    ArmSpec(
        label="scaffold_exponential",
        schedule="exponential",
        counts_for_verdict=True,
        description="Exponentially decayed donor scaffold with tau=50 updates.",
    ),
    ArmSpec(
        label="scratch_baseline",
        schedule="constant_zero",
        counts_for_verdict=False,
        description="Recipient-only scratch training control.",
    ),
    ArmSpec(
        label="constant_full_upper_bound",
        schedule="constant_full",
        counts_for_verdict=False,
        description="Frozen donor-only upper bound; recipient never updated.",
    ),
]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context():
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def alpha_schedule(name: str, t: int) -> float:
    """Return alpha(t) for zero-based update index t.

    We use zero-based t so the step schedule applies donor scaffolding to the
    first 50 optimizer updates exactly.
    """
    if name == "step":
        return 1.0 if t < 50 else 0.0
    if name == "linear":
        return max(0.0, 1.0 - t / 250.0)
    if name == "exponential":
        return math.exp(-t / 50.0)
    if name == "constant_zero":
        return 0.0
    if name == "constant_full":
        return 1.0
    raise ValueError(f"unknown schedule {name}")


def _load_config():
    cfg = AutoConfig.from_pretrained(
        _MODEL_ID,
        local_files_only=True,
        trust_remote_code=False,
    )
    if hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = "eager"
    return cfg


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(
        _MODEL_ID,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def load_trained_donor(tok=None):
    if tok is None:
        tok = load_tokenizer()
    cfg = _load_config()
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID,
        config=cfg,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=FORWARD_DTYPE,
    ).to(DEVICE).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tok


def load_random_init(seed: int):
    cfg = _load_config()
    set_seed(seed)
    model = AutoModelForCausalLM.from_config(cfg)
    model = model.to(DEVICE)
    model.train()
    return model


def load_c4_windows(tok, *, split: str, seed: int, n_windows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Stream deterministic fixed-length C4 windows.

    This preserves the g165 data-loading spirit:
      - stream `allenai/c4` English
      - shuffle with fixed seed and 10k buffer
      - reject short texts
      - cheap prefix dedup via `text[:200]`

    Unlike g165, we materialize exact token windows so eval is the locked
    200-window C4-validation measurement.
    """
    print(f"  loading C4 {split} windows (seed={seed}, n={n_windows}, len={SEQ_LEN})")
    ds = load_dataset(
        "allenai/c4",
        "en",
        split=split,
        streaming=True,
        trust_remote_code=False,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    rng = np.random.default_rng(seed)
    windows: list[np.ndarray] = []
    seen_prefixes: set[str] = set()

    for record in ds:
        text = record.get("text", "")
        if not isinstance(text, str) or len(text) < 100:
            continue
        prefix = text[:200]
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)

        token_ids = tok(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        if len(token_ids) < SEQ_LEN:
            continue

        if len(token_ids) == SEQ_LEN:
            start = 0
        else:
            start = int(rng.integers(0, len(token_ids) - SEQ_LEN + 1))
        window = np.asarray(token_ids[start:start + SEQ_LEN], dtype=np.int64)
        if window.shape[0] != SEQ_LEN:
            continue
        windows.append(window)
        if len(windows) >= n_windows:
            break

    if len(windows) < n_windows:
        raise RuntimeError(
            f"only sampled {len(windows)} / {n_windows} windows from "
            f"allenai/c4:{split} seed={seed}"
        )

    input_ids = torch.tensor(np.stack(windows, axis=0), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    print(f"  got {input_ids.shape[0]} windows")
    return input_ids, attention_mask


def build_batch_indices(perm: np.ndarray, start: int, batch_size: int) -> np.ndarray:
    end = start + batch_size
    if end <= perm.shape[0]:
        return perm[start:end]
    overflow = end - perm.shape[0]
    return np.concatenate([perm[start:], perm[:overflow]], axis=0)


def causal_ce_loss(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous().clone()
    shift_mask = attention_mask[:, 1:].contiguous()
    shift_labels[shift_mask == 0] = -100
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def eval_nll(model, val_ids: torch.Tensor, val_mask: torch.Tensor) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, val_ids.shape[0], EVAL_BATCH_SIZE):
        ids = val_ids[start:start + EVAL_BATCH_SIZE].to(DEVICE)
        mask = val_mask[start:start + EVAL_BATCH_SIZE].to(DEVICE)
        with autocast_context():
            out = model(input_ids=ids, attention_mask=mask, labels=ids, use_cache=False)
        n_tokens = int((mask[:, 1:] != 0).sum().item())
        total_loss += float(out.loss.item()) * n_tokens
        total_tokens += n_tokens
    model.train()
    return total_loss / max(total_tokens, 1)


def prepare_qwen_shared_inputs(recipient, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    backbone = recipient.model
    inputs_embeds = backbone.embed_tokens(input_ids)
    cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
    position_ids = cache_position.unsqueeze(0)
    mask_kwargs = {
        "config": backbone.config,
        "input_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    mask_mapping = {
        "full_attention": create_causal_mask(**mask_kwargs),
    }
    if getattr(backbone, "has_sliding_layers", False):
        mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    position_embeddings = backbone.rotary_emb(inputs_embeds, position_ids)
    return cache_position, position_ids, mask_mapping, position_embeddings


def scaffold_forward_logits(
    donor,
    recipient,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    alpha: float,
    *,
    donor_only: bool,
) -> tuple[torch.Tensor, str]:
    """Return logits for the requested forward mode.

    `donor_only=True` is the constant-full control: use the frozen donor as an
    online upper bound and do not update the recipient.
    """
    if donor_only:
        with torch.no_grad():
            logits = donor(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        return logits, "donor_only"

    if alpha <= 0.0:
        logits = recipient(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        return logits, "recipient_only"

    donor_backbone = donor.model
    recipient_backbone = recipient.model

    donor_hidden = donor_backbone.embed_tokens(input_ids)
    recipient_hidden = recipient_backbone.embed_tokens(input_ids)
    cache_position, position_ids, mask_mapping, position_embeddings = prepare_qwen_shared_inputs(
        recipient,
        input_ids,
        attention_mask,
    )

    for donor_layer, recipient_layer in zip(donor_backbone.layers, recipient_backbone.layers):
        attention_key = getattr(recipient_layer, "attention_type", "full_attention")
        attention_tensor = mask_mapping[attention_key]

        with torch.no_grad():
            donor_block_out = donor_layer(
                donor_hidden,
                attention_mask=attention_tensor,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        recipient_block_out = recipient_layer(
            recipient_hidden,
            attention_mask=attention_tensor,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        # ScaffoldSwap mixes the donor's live computation directly into the
        # residual stream. Codex cycle 36 SEV9 fix 2026-04-27: original mix
        # `alpha * donor + (1-alpha) * recipient` multiplied recipient gradient
        # by (1-alpha) at every block boundary -> with 28 layers and alpha=0.5,
        # earliest-layer recipient gradient scaled by 0.5^28 ~ 3.7e-9 (gradients
        # essentially zero). Equivalent forward value but full recipient gradient:
        #   r + alpha * (d - r).detach() = alpha*d + (1-alpha)*r at the value level,
        #   but the gradient w.r.t. recipient params flows fully through r.
        mixed_hidden = recipient_block_out + alpha * (donor_block_out - recipient_block_out).detach()

        donor_hidden = mixed_hidden
        recipient_hidden = mixed_hidden

    hidden_states = recipient_backbone.norm(recipient_hidden)
    logits = recipient.lm_head(hidden_states)
    return logits, "scaffold_swap"


def paired_bootstrap_ci(deltas: list[float], *, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    if len(deltas) < 2:
        return float("nan"), float("nan")
    arr = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        sample = arr[rng.integers(0, len(arr), size=len(arr))]
        boot_means[idx] = sample.mean()
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def train_one_arm(
    arm: ArmSpec,
    *,
    seed: int,
    donor,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
):
    set_seed(seed)
    recipient = load_random_init(seed)
    optimizer = torch.optim.AdamW(recipient.parameters(), lr=LR, betas=BETAS)
    n_trainable = sum(p.numel() for p in recipient.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in recipient.parameters())

    perm = torch.randperm(
        train_ids.shape[0],
        generator=torch.Generator().manual_seed(seed),
    ).cpu().numpy()

    trajectory = []
    nll0 = eval_nll(recipient, val_ids, val_mask)
    trajectory.append(
        {
            "step": 0,
            "recipient_only_nll": nll0,
            "alpha_last_update": None,
            "mean_train_ce": None,
        }
    )
    print(f"    {arm.label} seed={seed} step=0 recipient_nll={nll0:.4f}")

    mode_counts = {"recipient_only": 0, "scaffold_swap": 0, "donor_only": 0}
    running_loss = 0.0
    running_steps = 0
    t_arm = time.time()

    recipient.train()
    for step in range(1, N_STEPS + 1):
        cursor = ((step - 1) * TRAIN_BATCH_SIZE) % train_ids.shape[0]
        idx = build_batch_indices(perm, cursor, TRAIN_BATCH_SIZE)
        ids = train_ids[idx].to(DEVICE)
        mask = train_mask[idx].to(DEVICE)

        t_update = step - 1
        alpha = alpha_schedule(arm.schedule, t_update)
        donor_only = arm.schedule == "constant_full"

        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            logits, mode = scaffold_forward_logits(
                donor,
                recipient,
                ids,
                mask,
                alpha,
                donor_only=donor_only,
            )
            loss = causal_ce_loss(logits, ids, mask)

        mode_counts[mode] += 1
        running_loss += float(loss.item())
        running_steps += 1

        if not donor_only:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(recipient.parameters(), 1.0)
            optimizer.step()

        if step in EVAL_STEPS[1:]:
            recipient_nll = eval_nll(recipient, val_ids, val_mask)
            mean_train_ce = running_loss / max(running_steps, 1)
            elapsed = time.time() - t_arm
            trajectory.append(
                {
                    "step": step,
                    "recipient_only_nll": recipient_nll,
                    "alpha_last_update": alpha,
                    "mean_train_ce": mean_train_ce,
                }
            )
            print(
                f"    {arm.label} seed={seed} step={step} "
                f"recipient_nll={recipient_nll:.4f} alpha={alpha:.4f} "
                f"train_ce={mean_train_ce:.4f} ({elapsed:.0f}s)"
            )
            running_loss = 0.0
            running_steps = 0

    payload = {
        "trajectory": trajectory,
        "mode_counts": mode_counts,
        "n_trainable_params": n_trainable,
        "n_total_params": n_total,
        "note": (
            "constant_full_upper_bound uses donor-only scaffold CE and skips optimizer "
            "updates; recipient-only eval is reported for comparability."
            if arm.schedule == "constant_full"
            else None
        ),
    }

    del recipient, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def step_to_nll(payload: dict, step: int) -> float:
    for row in payload["trajectory"]:
        if int(row["step"]) == step:
            return float(row["recipient_only_nll"])
    raise KeyError(f"step {step} not found in trajectory")


def summarize_results(results: dict, donor_reference_nll: float) -> dict:
    if "scratch_baseline" not in results:
        return {"verdict": "ERROR: scratch baseline missing"}

    scratch = results["scratch_baseline"]
    per_arm = {}
    pass_arms = []
    weak_arms = []

    best_decay_final = -float("inf")
    best_decay_step50 = -float("inf")
    best_decay_step250 = -float("inf")
    best_decay_label = None

    for arm in ARM_SPECS:
        if arm.label == "scratch_baseline":
            continue

        arm_results = results.get(arm.label, {})
        advantages_by_step = {}
        for step in EVAL_STEPS:
            deltas = []
            for seed in SEEDS:
                seed_key = str(seed)
                if seed_key not in scratch or seed_key not in arm_results:
                    continue
                scratch_nll = step_to_nll(scratch[seed_key], step)
                arm_nll = step_to_nll(arm_results[seed_key], step)
                deltas.append(scratch_nll - arm_nll)
            if deltas:
                ci_lo, ci_hi = paired_bootstrap_ci(deltas, seed=42 + step)
                advantages_by_step[str(step)] = {
                    "mean_advantage_nats": float(np.mean(deltas)),
                    "ci_95_lo": ci_lo,
                    "ci_95_hi": ci_hi,
                    "seed_deltas": deltas,
                    "n_seeds": len(deltas),
                }

        final_stats = advantages_by_step.get(str(N_STEPS))
        if final_stats is None:
            continue

        per_arm[arm.label] = {
            "schedule": arm.schedule,
            "counts_for_verdict": arm.counts_for_verdict,
            "description": arm.description,
            "final_recipient_advantage_nats": final_stats,
            "advantage_by_step": advantages_by_step,
        }

        final_mean = float(final_stats["mean_advantage_nats"])
        final_ci_lo = float(final_stats["ci_95_lo"])
        if arm.counts_for_verdict:
            if final_mean >= PASS_ADVANTAGE_NATS and final_ci_lo > 0:
                pass_arms.append(arm.label)
            elif final_mean >= FAIL_WASHOUT_NATS and final_ci_lo > 0:
                weak_arms.append(arm.label)

            if final_mean > best_decay_final:
                best_decay_final = final_mean
                best_decay_label = arm.label
            step50_stats = advantages_by_step.get("50")
            step250_stats = advantages_by_step.get("250")
            if step50_stats is not None:
                best_decay_step50 = max(best_decay_step50, float(step50_stats["mean_advantage_nats"]))
            if step250_stats is not None:
                best_decay_step250 = max(best_decay_step250, float(step250_stats["mean_advantage_nats"]))

    if pass_arms:
        verdict = (
            "PASS: at least one decay schedule achieves recipient-alone final "
            f"C4 advantage >= +{PASS_ADVANTAGE_NATS:.1f} nats with CI>0 after scaffold removal: "
            f"{pass_arms}"
        )
        active_ingredient = (
            "Temporary donor computation leaves a persistent residue in the "
            "recipient after the scaffold is gone; function transfer works "
            "without a continuous weight anchor."
        )
    elif weak_arms:
        verdict = (
            "WEAK_SIGNAL: no decay schedule reaches the prereg +0.5 nat PASS bar, "
            f"but {weak_arms} remain above +{FAIL_WASHOUT_NATS:.1f} nats with CI>0."
        )
        active_ingredient = (
            "ScaffoldSwap improves recipient-alone convergence, but the retained "
            "benefit is below the locked PASS magnitude."
        )
    else:
        verdict = (
            "FAIL: all 3 decay schedules wash out below +0.2 nats or their 95% CI "
            "crosses zero after scaffold removal."
        )
        if max(best_decay_step50, best_decay_step250) >= FAIL_WASHOUT_NATS:
            active_ingredient = (
                "The scaffold produces at most transient help during training, but "
                "the advantage washes out once donor compute is removed."
            )
        else:
            active_ingredient = (
                "Even live donor scaffolding fails to leave a material recipient-only "
                "residue; scratch matches the scaffolded runs."
            )

    return {
        "verdict": verdict,
        "pass_arms": pass_arms,
        "weak_arms": weak_arms,
        "best_decay_arm": best_decay_label,
        "best_decay_final_advantage_nats": best_decay_final,
        "best_decay_step50_advantage_nats": best_decay_step50,
        "best_decay_step250_advantage_nats": best_decay_step250,
        "donor_reference_c4_nll": donor_reference_nll,
        "constant_full_excluded_from_pass": True,
        "per_arm": per_arm,
        "active_ingredient_analysis": active_ingredient,
    }


def save_incremental(results: dict, *, donor_reference_nll: float, t_start: float) -> None:
    payload = {
        "genome": 169,
        "name": "scaffold_swap_distillation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "donor_reference_c4_nll": donor_reference_nll,
        "elapsed_s": time.time() - t_start,
        "incremental": True,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("genome_169: ScaffoldSwap functional scaffold distillation")
    print(f"  donor/recipient model: {_MODEL_ID}")
    print(f"  device={DEVICE} forward_dtype={FORWARD_DTYPE}")
    print(f"  seeds={SEEDS} steps={N_STEPS} batch={TRAIN_BATCH_SIZE} eval={EVAL_STEPS}")

    t_start = time.time()
    tok = load_tokenizer()
    donor, _ = load_trained_donor(tok)

    train_ids, train_mask = load_c4_windows(
        tok,
        split="train",
        seed=C4_TRAIN_SEED,
        n_windows=TRAIN_WINDOWS,
    )
    val_ids, val_mask = load_c4_windows(
        tok,
        split="validation",
        seed=C4_VAL_SEED,
        n_windows=N_C4_VAL_WINDOWS,
    )
    print(f"  train={tuple(train_ids.shape)}  val={tuple(val_ids.shape)}")

    donor_reference_nll = eval_nll(donor, val_ids, val_mask)
    print(f"  donor reference C4-val NLL={donor_reference_nll:.4f}")

    results = {arm.label: {} for arm in ARM_SPECS}
    print(f"\n=== Running {len(ARM_SPECS)} arms x {len(SEEDS)} seeds = {len(ARM_SPECS) * len(SEEDS)} cells ===")
    print("=== Iteration: seed-major, scratch-matched recipients across arms ===")

    for seed in SEEDS:
        for arm in ARM_SPECS:
            print(f"\n--- arm={arm.label} seed={seed} ---")
            payload = train_one_arm(
                arm,
                seed=seed,
                donor=donor,
                train_ids=train_ids,
                train_mask=train_mask,
                val_ids=val_ids,
                val_mask=val_mask,
            )
            results[arm.label][str(seed)] = payload
            save_incremental(results, donor_reference_nll=donor_reference_nll, t_start=t_start)

    summary = summarize_results(results, donor_reference_nll)
    out = {
        "genome": 169,
        "name": "scaffold_swap_distillation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": _MODEL_ID,
        "device": DEVICE,
        "forward_dtype": str(FORWARD_DTYPE),
        "recipient_param_dtype": "torch.float32",
        "config": {
            "seeds": SEEDS,
            "seq_len": SEQ_LEN,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "train_steps": N_STEPS,
            "eval_steps": EVAL_STEPS,
            "train_windows": TRAIN_WINDOWS,
            "n_c4_val_windows": N_C4_VAL_WINDOWS,
            "lr": LR,
            "betas": list(BETAS),
            "c4_train_seed": C4_TRAIN_SEED,
            "c4_val_seed": C4_VAL_SEED,
            "arm_specs": [
                {
                    "label": arm.label,
                    "schedule": arm.schedule,
                    "counts_for_verdict": arm.counts_for_verdict,
                    "description": arm.description,
                }
                for arm in ARM_SPECS
            ],
            "pass_advantage_nats": PASS_ADVANTAGE_NATS,
            "fail_washout_nats": FAIL_WASHOUT_NATS,
        },
        "results": results,
        "summary": summary,
        "verdict": summary["verdict"],
        "elapsed_s": time.time() - t_start,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n=== verdict: {summary['verdict']} ===")
    print(f"=== active ingredient: {summary['active_ingredient_analysis']} ===")
    print(f"Saved {OUT_PATH} ({out['elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()
