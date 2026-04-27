"""
genome_165_annealed_donor.py — annealed donor / decaying-anchor washout test.

Per cycle 24 strategic pivot in research/programs/post_g158c_decision_tree.md
+ research/EARLY_HELP_META_AUDIT_2026-04-27.md, this experiment tests whether a
DECAYING anchor schedule (rather than the static-anchor regime g008 already
tested) can produce persistent positive NLL advantage at convergence.

Design (per research/prereg/genome_165_annealed_donor_2026-04-27.md):
  Donor:   Qwen3-0.6B (trained, frozen)
  Recipient: random-init Qwen3-0.6B-architecture model
  Loss:    L_CE(recipient) + lambda(t) * ||theta_recipient - theta_donor||_F^2
  Schedules:
    - constant:        lambda(t) = lambda_0                    [washout-replication control]
    - step:            lambda(t) = lambda_0 if t < 25 else 0   [hard cutoff at 25]
    - linear:          lambda(t) = lambda_0 * max(0, 1 - t/50) [decay to 0 by step 50]
    - exponential:     lambda(t) = lambda_0 * exp(-t/10)       [tau=10 decay]
    - hard_cut_step1:  lambda(t) = lambda_0 if t==1 else 0     [Codex cycle 30: early-only]
  lambda_0 in {1.3e-4, 1.3e-3, 1.0e-2}
  Seeds:   [42, 7, 13]
  + 1 attention-only anchor (Codex cycle 30: tests g125 boundary on attn submanifold)
  + 1 scratch baseline (no anchor; lambda=0)

  14 arms x 3 seeds = 42 cells.
  500 train steps, batch=8, lr=3e-4, BF16 forward, FP32 anchor regularizer.
  C4 val NLL every 25 steps.

PASS: at least one (lambda_0, schedule) combination produces final-step C4 NLL
advantage over scratch >= +0.5 nats with paired bootstrap 95% CI excluding 0.
"""
from __future__ import annotations
import gc
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "models"))
try:
    from registry import resolve as _resolve_model  # type: ignore
    _MODEL_ID = _resolve_model("qwen3-0.6b").get("hf_id", "Qwen/Qwen3-0.6B")
except Exception:
    _MODEL_ID = "Qwen/Qwen3-0.6B"

OUT_PATH = ROOT / "results" / "genome_165_annealed_donor.json"

# --- Hyperparameters (locked per prereg) ---
SEEDS = [42, 7, 13]
SEQ_LEN = 256
N_TRAIN_TOKENS = 500 * 8 * SEQ_LEN  # 500 steps * batch 8 * seq_len
N_VAL_TOKENS = 64 * SEQ_LEN
LR = 3e-4
BATCH_SIZE = 8
N_STEPS = 500
EVAL_EVERY = 25

# Codex lean pre-flight 2026-04-27 (codex_outputs/g165_lambda_grid_check_20260427T114500.md):
# Frobenius F^2 = ||theta_init - theta_donor||^2 ≈ 2.03e6 over 596M params (per-param
# mean squared gap ~3.4e-3). At lambda_0=1.0, anchor gradient dominates CE by ~759x;
# at 0.01 still 7.6x. The original grid {1.0, 0.1, 0.01} would just collapse the
# recipient to a donor clone in all 3 strengths. Codex-recommended grid below
# spans weak/balanced/strong without freezing the recipient:
#   1.3e-4: weak anchor, CE dominates 10x
#   1.3e-3: balanced (CE and anchor comparable)
#   1.0e-2: strong anchor (7.6x dominant), recipient still trains
LAMBDA_0_GRID = [1.3e-4, 1.3e-3, 1.0e-2]
SCHEDULES = ["constant", "step", "linear", "exponential"]

C4_TRAIN_SEED = 165
C4_VAL_SEED = 1650

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def lambda_schedule(name: str, lam0: float, t: int) -> float:
    if name == "constant":
        return lam0
    if name == "step":
        return lam0 if t < 25 else 0.0
    if name == "linear":
        return lam0 * max(0.0, 1.0 - t / 50.0)
    if name == "exponential":
        return lam0 * math.exp(-t / 10.0)
    if name == "hard_cut_step1":
        # Active at step 1 only; zero from step 2 onward.
        # Per Codex cycle 30 direction review: tests "early-help only, no
        # continued anchor" — the corrected g125 boundary inside g165.
        return lam0 if t == 1 else 0.0
    raise ValueError(f"unknown schedule {name}")


def load_c4_texts(seed: int, n_target_tokens: int):
    """Stream C4-en train, dedup by 13-gram, return texts until token budget reached."""
    print(f"  loading C4 texts (seed={seed}, target={n_target_tokens} tokens)")
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)
    texts = []
    seen_13gram = set()
    approx_tokens = 0
    for example in ds:
        text = example["text"]
        if len(text) < 100:
            continue
        # cheap 13-gram dedup over chars (proxy for token-level)
        g = text[:200]
        if g in seen_13gram:
            continue
        seen_13gram.add(g)
        texts.append(text)
        approx_tokens += len(text) // 4
        if approx_tokens >= n_target_tokens:
            break
    print(f"  got {len(texts)} texts (~{approx_tokens} tokens)")
    return texts


def tokenize_block(tok, texts, seq_len):
    enc = tok(texts, return_tensors="pt", padding="max_length",
              truncation=True, max_length=seq_len)
    return enc["input_ids"], enc["attention_mask"]


def load_trained_donor(tok=None):
    if tok is None:
        tok = AutoTokenizer.from_pretrained(_MODEL_ID)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(_MODEL_ID, dtype=torch.bfloat16).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def load_random_init(seed: int):
    cfg = AutoConfig.from_pretrained(_MODEL_ID)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).to(DEVICE)
    return model


def snapshot_donor_params(donor) -> dict:
    """Return {name: tensor.detach().clone()} of donor params, in FP32 for stable
    anchor distance. Only matchable params (same shape as recipient)."""
    out = {}
    for name, p in donor.named_parameters():
        out[name] = p.detach().to(torch.float32).clone()
    return out


def anchor_loss(recipient, donor_params: dict, submanifold: str = "all") -> torch.Tensor:
    """Compute Frobenius ||theta_r - theta_d||^2 in FP32.

    submanifold:
      "all"  - anchor every parameter (default; full-weight anchor).
      "attn" - anchor only attention parameters (per Codex cycle 30 direction
               review: g125's persistence used attention-only anchor; this
               restores apples-to-apples comparison inside g165).
    """
    total = torch.zeros((), device=DEVICE, dtype=torch.float32)
    matched = 0
    for name, p in recipient.named_parameters():
        if name not in donor_params:
            continue
        if p.shape != donor_params[name].shape:
            continue
        if submanifold == "attn" and ".self_attn." not in name:
            continue
        total = total + ((p.to(torch.float32) - donor_params[name]) ** 2).sum()
        matched += 1
    if matched == 0:
        raise RuntimeError(f"anchor_loss({submanifold=}): no parameters matched between recipient and donor")
    return total


@torch.no_grad()
def eval_nll(model, val_ids, val_mask) -> float:
    """Batched NLL over val tokens. Returns mean NLL per non-pad token."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i in range(0, val_ids.shape[0], BATCH_SIZE):
        ids = val_ids[i:i + BATCH_SIZE].to(DEVICE)
        mask = val_mask[i:i + BATCH_SIZE].to(DEVICE)
        out = model(input_ids=ids, attention_mask=mask, labels=ids)
        # HF mean-loss over non-ignored tokens; reweight by token count
        n_tokens = (mask[:, 1:] != 0).sum().item()
        total_loss += out.loss.item() * n_tokens
        total_tokens += n_tokens
    model.train()
    return total_loss / max(total_tokens, 1)


def train_one_arm(
    arm_label: str, lam0: float, schedule_name: str,
    seed: int, donor_params: dict | None,
    train_ids, train_mask, val_ids, val_mask, tok,
    submanifold: str = "all",
):
    """Train one recipient arm; return list of (step, val_nll)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    recipient = load_random_init(seed)
    optimizer = torch.optim.AdamW(recipient.parameters(), lr=LR, betas=(0.9, 0.95))

    n_train = train_ids.shape[0]
    perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(seed)).numpy()

    trajectory = []
    # Eval at step 0
    nll0 = eval_nll(recipient, val_ids, val_mask)
    trajectory.append({"step": 0, "nll": nll0})
    print(f"    {arm_label} seed={seed} step=0 nll={nll0:.4f}")

    t0 = time.time()
    recipient.train()
    for step in range(1, N_STEPS + 1):
        # batch indices (with wrap)
        idx = perm[(step * BATCH_SIZE) % n_train : (step * BATCH_SIZE) % n_train + BATCH_SIZE]
        if len(idx) < BATCH_SIZE:
            idx = perm[:BATCH_SIZE]
        ids = train_ids[idx].to(DEVICE)
        mask = train_mask[idx].to(DEVICE)

        out = recipient(input_ids=ids, attention_mask=mask, labels=ids)
        loss_ce = out.loss

        # Anchor regularizer
        if donor_params is not None and lam0 > 0:
            lam_t = lambda_schedule(schedule_name, lam0, step)
            if lam_t > 0:
                loss_anchor = anchor_loss(recipient, donor_params, submanifold=submanifold)
                loss = loss_ce + lam_t * loss_anchor
            else:
                loss = loss_ce
        else:
            loss = loss_ce

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(recipient.parameters(), 1.0)
        optimizer.step()

        if step % EVAL_EVERY == 0:
            nll = eval_nll(recipient, val_ids, val_mask)
            trajectory.append({"step": step, "nll": nll})
            elapsed = time.time() - t0
            print(f"    {arm_label} seed={seed} step={step} nll={nll:.4f} ({elapsed:.0f}s)")

    # Cleanup
    del recipient, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return trajectory


def paired_bootstrap_ci(deltas_per_seed: list[float], n_boot: int = 10000, seed: int = 0):
    """95% paired bootstrap CI on the mean of deltas (one delta per seed)."""
    rng = np.random.default_rng(seed)
    n = len(deltas_per_seed)
    arr = np.array(deltas_per_seed, dtype=float)
    if n < 2:
        return (float("nan"), float("nan"))
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        boot_means[i] = arr[rng.integers(0, n, n)].mean()
    return (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))


def main():
    print(f"genome_165: annealed-donor / decaying-anchor washout test")
    print(f"  donor: {_MODEL_ID}  device: {DEVICE}")
    print(f"  seeds: {SEEDS}  steps: {N_STEPS}  batch: {BATCH_SIZE}")
    print(f"  schedules: {SCHEDULES}  lambda_0_grid: {LAMBDA_0_GRID}")

    t_start = time.time()
    donor, tok = load_trained_donor()
    donor_params = snapshot_donor_params(donor)
    # Free donor model from GPU (we keep params snapshot in FP32 RAM/GPU)
    del donor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Keep donor_params on GPU FP32 for fast anchor compute
    for k in list(donor_params.keys()):
        donor_params[k] = donor_params[k].to(DEVICE)

    # Token data: train + val (shared across arms within run)
    train_texts = load_c4_texts(C4_TRAIN_SEED, N_TRAIN_TOKENS)
    val_texts = load_c4_texts(C4_VAL_SEED, N_VAL_TOKENS)
    train_ids, train_mask = tokenize_block(tok, train_texts, SEQ_LEN)
    val_ids, val_mask = tokenize_block(tok, val_texts, SEQ_LEN)
    print(f"  train: {train_ids.shape}  val: {val_ids.shape}")

    # Build arm specifications
    arms = []
    for lam0 in LAMBDA_0_GRID:
        for sched in SCHEDULES:
            arms.append({
                "label": f"anchor_lam{lam0}_{sched}",
                "lam0": lam0,
                "schedule": sched,
                "submanifold": "all",
            })
    # Codex cycle 30 direction review: attention-only hard-cut anchor.
    # Tests "early-help only, no continued anchor" on the SAME submanifold (attention-
    # only) where g125 saw +0.07 nats persistence. Restores apples-to-apples comparison.
    arms.append({
        "label": "anchor_attn_only_lam1.3e-3_hardcut",
        "lam0": 1.3e-3,
        "schedule": "hard_cut_step1",
        "submanifold": "attn",
    })
    arms.append({"label": "scratch_baseline", "lam0": 0.0, "schedule": "constant", "submanifold": "all"})  # no donor

    print(f"\n=== Running {len(arms)} arms x {len(SEEDS)} seeds = {len(arms)*len(SEEDS)} cells ===")

    results = {}  # arm_label -> {seed: trajectory}
    for arm_spec in arms:
        results[arm_spec["label"]] = {}
        for seed in SEEDS:
            print(f"\n--- arm={arm_spec['label']} seed={seed} ---")
            traj = train_one_arm(
                arm_label=arm_spec["label"],
                lam0=arm_spec["lam0"],
                schedule_name=arm_spec["schedule"],
                seed=seed,
                donor_params=donor_params if arm_spec["lam0"] > 0 else None,
                train_ids=train_ids, train_mask=train_mask,
                val_ids=val_ids, val_mask=val_mask, tok=tok,
                submanifold=arm_spec.get("submanifold", "all"),
            )
            results[arm_spec["label"]][seed] = traj
            # incremental save after each cell
            _save_results(results, arms, t_start)

    # Final analysis: PASS / WEAK / FAIL
    summary = compute_verdict(results, arms)
    out = {
        "genome": 165,
        "name": "annealed_donor",
        "config": {
            "model_id": _MODEL_ID,
            "seeds": SEEDS,
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "lambda_0_grid": LAMBDA_0_GRID,
            "schedules": SCHEDULES,
        },
        "results": results,
        "summary": summary,
        "verdict": summary["verdict"],
        "elapsed_s": time.time() - t_start,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n=== verdict: {summary['verdict']} ===")
    print(f"Saved: {OUT_PATH}  ({out['elapsed_s']:.1f}s)")


def _save_results(results, arms, t_start):
    out = {
        "genome": 165,
        "name": "annealed_donor",
        "results": results,
        "elapsed_s": time.time() - t_start,
        "incremental": True,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")


def compute_verdict(results, arms):
    """For each anchored arm, compute paired delta vs scratch at final step.
    PASS if any arm has mean_delta >= +0.5 nats AND CI excludes 0."""
    if "scratch_baseline" not in results:
        return {"verdict": "ERROR: no scratch baseline"}
    scratch = results["scratch_baseline"]
    final_scratch = {seed: traj[-1]["nll"] for seed, traj in scratch.items()}

    per_arm = {}
    pass_arms = []
    weak_arms = []
    for arm_spec in arms:
        label = arm_spec["label"]
        if label == "scratch_baseline":
            continue
        if label not in results:
            continue
        deltas = []
        for seed in SEEDS:
            if seed in results[label] and seed in final_scratch:
                # delta = scratch_nll - anchor_nll (positive = anchor helps at final step)
                d = final_scratch[seed] - results[label][seed][-1]["nll"]
                deltas.append(d)
        if not deltas:
            continue
        mean_d = float(np.mean(deltas))
        ci_lo, ci_hi = paired_bootstrap_ci(deltas, seed=42)
        per_arm[label] = {
            "mean_final_advantage_nats": mean_d,
            "ci_95_lo": ci_lo, "ci_95_hi": ci_hi,
            "n_seeds": len(deltas),
        }
        if mean_d >= 0.5 and ci_lo > 0:
            pass_arms.append(label)
        elif mean_d >= 0.2 and ci_lo > 0:
            weak_arms.append(label)

    if pass_arms:
        verdict = f"PASS: {len(pass_arms)} arm(s) produce >=+0.5 nats persistent advantage with CI>0: {pass_arms}"
    elif weak_arms:
        verdict = f"WEAK_PASS: {len(weak_arms)} arm(s) produce +0.2..+0.5 nats with CI>0: {weak_arms}"
    else:
        verdict = "FAIL: 0 arms produce persistent positive advantage; washout pattern replicates at canonical scale"

    return {
        "verdict": verdict,
        "per_arm": per_arm,
        "pass_arms": pass_arms,
        "weak_arms": weak_arms,
    }


if __name__ == "__main__":
    main()
