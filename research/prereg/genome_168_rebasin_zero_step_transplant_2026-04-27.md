# genome_168 — re-basin + norm-refit zero-step transplant

**Status.** DRAFT 2026-04-27 — pending Codex pre-flight before LOCK.
**Trigger.** Codex transfer-axis rethink consult 2026-04-27 ranked this **#1 technique at 8.3/10 PASS** — fires immediately in parallel with g165 (or right after).
**Source.** `codex_outputs/transfer_axis_rethink_20260427T200000.md` proposal #1.

## Hypothesis

Zero-step capability transfer fails not because donor weights "don't carry capability" but because **basis + norm mismatch** between donor and random-init recipient corrupts the donor signal at injection. Fix the alignment: fit per-layer permutation/orthogonal transform + RMSNorm refit on a calibration set, copy donor blocks into the aligned coordinates, and the transferred capability survives without any gradient steps.

The exact failure mode in g121-g124 was: PC1 alignment was killed (g124), but full-stack permutation re-basin + norm-refit was never tested. This experiment closes that loophole.

## Why this beats anchor-decay (g165)

- g165's hypothesis: donor signal helps then washes out under SGD; decay schedule rescues.
- This experiment's hypothesis: donor signal works at zero steps **if you align coordinates first**. No SGD involved. No washout problem.
- If PASS: this is a literal §0 zero-step capability transfer demonstration — exactly the project end-goal phrasing.
- If FAIL: alignment is not the loophole; the surgery story is genuinely closed.

## Setup

- **Donor:** Qwen3-0.6B (trained, frozen).
- **Recipient:** Qwen3-0.6B-architecture random-init.
- **Calibration set:** ~10k C4 tokens, used to fit alignment transforms per layer.
- **Per-layer alignment** (factorial dimension):
  - **identity** (control / baseline): no alignment, raw donor copy
  - **permutation**: per-layer orthogonal/permutation matrix found by minimizing activation distance on calibration set
  - **norm_refit**: RMSNorm scales refit to recipient's activation distribution
  - **permutation + norm_refit**: combined
- **Recipient blocks transplanted:** all transformer blocks (attention + MLP).
- **Steps:** 0 (zero-step transfer); also evaluate at step=10 and step=50 with continued training to measure persistence under modest SGD.
- **Eval:** C4 NLL + top-1 (200 windows, len 256).
- **Seeds:** [42, 7, 13] (canonical 3-seed for the alignment transforms — re-fit per seed since calibration sample varies).
- **Comparators:**
  - `random_init_only` (no donor copy at all)
  - `donor_full` (donor model itself; upper bound)
  - `raw_copy` (donor weights copied into recipient with NO alignment; the "naive" transfer that's been killed historically)

## Cells

7 arms × 3 seeds × 3 step-points (0, 10, 50) = 63 evals, but only 4 alignment arms × 3 seeds = 12 cells of actual training (the others are just transplant + eval, no training).

## Locked PASS / FAIL criteria (from Codex)

**PASS:**
- Across 3 seeds, mean **zero-step C4 NLL gain ≥ +0.8 nats** vs random-init recipient
- 95% bootstrap CI excludes zero
- Beats the historical `all_attn` zero-step effect (~+0.05-0.10 nats per WIKI bestiary) by **≥5×** = ≥0.5 nats minimum
- At step=50 with continued training, gain still ≥ +0.4 nats (persists under modest SGD)

**FAIL:**
- Mean zero-step gain < +0.3 nats OR CI crosses zero OR step=50 advantage washes out below +0.1 nats

## Compute envelope (COMPUTE.md §9 compliance)

- Wall-clock estimate: ≤ 2 hr (Codex estimate)
- Peak VRAM: < 20 GB (donor + recipient + alignment fit; NO AdamW until step=50 phase)
- Peak RAM: < 24 GB
- Disk: < 200 MB

## Why this is the highest-leverage post-g158c move

Codex transfer-axis rethink 2026-04-27 ranked this **8.3/10** — higher than:
- g165 (7.3) — ANY anchor-decay variant
- g162 (6.8) — capacity sweep
- g166 (6.4) — optimizer-state combined
- ScaffoldSwap (8.0) — close runner-up

It is **fully orthogonal** to the anchor-decay axis (no SGD interaction; alignment is purely geometric).

§0 link: this directly tests the project end goal (capability transfer without retraining the recipient). PASS would be the canonical §0 cash-out at single-architecture scale; the next experiment would extend to cross-architecture alignment.

## Files (to be created when LOCKED)

- `code/genome_168_rebasin_zero_step_transplant.py`
- `results/genome_168_rebasin_zero_step_transplant.json`
- `code/integrate_g168.py`

## Pre-flight protocol

Before LOCK, fire Codex consult to audit:
1. Permutation fit: which permutation algorithm? Hungarian on activation correlation per layer? Is it cheap enough?
2. Norm refit: how exactly is RMSNorm scale derived from recipient activations? Match donor's mean activation magnitude per layer?
3. Cross-tied weights: Qwen3-0.6B has tied embedding and lm_head; how does that affect transplant?
4. Layer-norm placement: input_layernorm vs post_attention_layernorm — which gets refit, both, neither?
5. Calibration set leakage: same C4 stream as eval? Need separate split.

## Provenance

- Codex transfer-axis rethink: `codex_outputs/transfer_axis_rethink_20260427T200000.md` proposal #1
- Historical surgery dead-ends: g117-g124 (single-direction PC1 surgery), specifically g124 (PC1 alignment killed)
- All_attn residue (mildly positive): WIKI bestiary "all-attn surgery"
- Re-basin literature: Ainsworth et al. 2022 (https://arxiv.org/abs/2209.04836) — Git Re-Basin
