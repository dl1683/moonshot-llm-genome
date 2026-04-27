# genome_166 — optimizer-state + decay-anchor combined transfer

**Status.** DRAFT 2026-04-27 — pending g165 FAIL verdict + Codex pre-flight sign-off before LOCK.
**Trigger.** Activates iff g165 returns FAIL (no anchored arm produces final-step C4 NLL advantage ≥ +0.5 nats with CI > 0).
**Source.** Codex cycle 27 direction review Q2 (`codex_outputs/heartbeats/cycle27_direction_review_20260427T110500.md`) + g137 decay-shape extraction in `research/EARLY_HELP_META_AUDIT_2026-04-27.md`.

## Hypothesis

The right transfer signal is not weight-init AND not optimizer-state alone — it's the COMBINATION: transfer optimizer state (Adam m, v moments) AND apply a decay-anchored weight regularizer at the appropriate timescale (~1420-step half-life, derived from g137).

The empirical motivation:
- 7 weight-init donor mechanisms wash out by step 4-2000.
- g137 (optimizer-state) shows the same washout pattern with proper comparator (1064: +0.046 → 4000: -0.0004) — half-life ~1420 steps.
- BUT the DECAY constants are DIFFERENT: weight-init mechanisms wash out at ~25 steps; optimizer-state at ~1420 steps. The two signals compound or interfere depending on timing.

The null is well-defined: if g165's weight-init decay schedules (25-step timescale) all FAIL AND g137's optimizer-state-alone shows wash-out, then either the recipient cannot persistently use donor signal under SGD (in which case g166 also fails and we should pivot to g155 hardware unblock), or the right signal is at a different abstraction level entirely.

## Setup

- **Donor:** Qwen3-0.6B (trained, Adam state preserved from a checkpoint near training completion).
- **Recipient:** random-init Qwen3-0.6B-architecture model.
- **Loss:** L_CE(recipient) + λ(t) · ||θ_recipient - θ_donor||_F²
- **Optimizer state initialization** (factorial dimension 1):
  - **none**: standard Adam from zero (control)
  - **resume_true**: load donor's Adam m, v moments at recipient init
- **Anchor schedules** (factorial dimension 2 — long-timescale, NOT g165's short-timescale):
  - **constant** (control): λ(t) = λ_0
  - **slow_step**: λ(t) = λ_0 if t < 1000 else 0
  - **slow_linear**: λ(t) = λ_0 · max(0, 1 - t / 2840) [decay to 0 by 2× the g137 half-life]
  - **slow_exponential**: λ(t) = λ_0 · exp(-t / 1420) [τ = g137 empirical half-life]
- **λ_0:** 1.3e-3 (the balanced strength from g165's revised grid; only 1 strength to keep wall-clock tractable)
- **Submanifold:** "all" (full-weight Frobenius, matching g165's main arms)
- **Seeds:** [42, 7, 13]
- **Training:** 4000 steps (long enough to observe full g137-scale washout), batch_size=8, lr=3e-4, eval every 250 steps.

## Cells

(2 optimizer states × 4 anchor schedules) + 1 scratch baseline = 9 arms × 3 seeds = **27 cells**.

## Locked PASS / FAIL criteria

**PASS:**
- At least one (optimizer_state, schedule) combination produces final-step C4 NLL advantage over scratch ≥ +0.2 nats (lower than g165's +0.5 because the longer training horizon means smaller absolute differences are still meaningful) with bootstrap 95% CI excluding zero.
- Ideally: the (resume_true, slow_decay) combination beats both (resume_true, constant) [pure long-anchor] and (none, slow_decay) [weight-anchor only] — i.e., the combination is super-additive.

**WEAK_PASS:**
- One combination produces ≥ +0.1 nats with CI > 0.

**FAIL:**
- All 8 anchored combinations wash out at the long-timescale.
- This would close the donor/transfer chain entirely. Pivot to g155 procurement (wall-power meter unblocks 8.2/10 path) as the only remaining manifesto-grade direction.

## Compute envelope (COMPUTE.md §9 compliance)

- Wall-clock estimate: ~4-5h (4000 steps × 27 cells × ~0.4 s/step ≈ 12 hrs CPU-time but parallelizable to per-arm; serial run is ~3.5-4hr)
- Peak VRAM: < 12 GB (one model live + Adam state at FP32 on GPU adds ~1GB)
- Peak RAM: < 24 GB
- **Envelope risk:** at the upper edge. If pre-flight shows >4hr likely, drop to 2 schedules × 2 optimizer states = 4 anchored arms + 1 scratch = 5 arms × 3 seeds = 15 cells.

## Why this is the right FAIL pivot (Codex rationale)

Per Codex cycle 27 direction review Q2: the g137 decay-shape extraction shows the optimizer-state advantage has a half-life of ~1420 steps, NOT the 25-step timescale of weight-init mechanisms. g165 only tests short-timescale schedules (matching weight-init). g166 extends to long-timescale schedules (matching optimizer-state).

This separates two independent decay processes. If g165 FAILs and g166 PASSes: the right transfer signal IS at the optimizer-state level under appropriate decay. If both FAIL: weight-space transfer is not the right abstraction; pivot to the §0 alternative path (g155 procurement OR fundamentally different transfer mechanism).

## What FAIL would mean for the moonshot

If g166 FAILs after g165 FAILs:
- The "annealed donor rescues SGD washout" hypothesis is empirically dead at canonical scale.
- The §0 zero-step transfer goal can be achieved (immediate +10 nats peak across mechanisms) but cannot be made persistent under continued training via any donor-signal mechanism we've tested.
- Implication: capability transfer either (a) requires continued frozen donor signal (g125-style anchor-rate-zero), or (b) requires a fundamentally different mechanism (e.g., gradient-direction transfer, distillation-style soft labels, modular composition).
- Project pivot: wall-power meter procurement becomes the only path to a flagship-grade finding.

## Files (to be created when LOCKED)

- `code/genome_166_optimizer_state_decay_anchor.py` — fork of `genome_165_annealed_donor.py`, add optimizer-state copy at recipient init, switch to long-timescale schedules.
- `results/genome_166_optimizer_state_decay_anchor.json`
- `code/integrate_g166.py` — verdict integration with active-ingredient analysis: distinguishes "optimizer-state alone wins", "decay-anchor alone wins", "combination wins", "neither wins".

## Pre-flight protocol

Before LOCK, fire Codex consult to audit:
1. Is the Adam m, v transfer correctly preserving donor optimizer state? Need to checkpoint donor's optimizer state from a real Qwen3-0.6B training run (currently we don't have one — would need to either train a Qwen3-0.6B for some steps from scratch and save state, OR find a teacher checkpoint that preserves optimizer state alongside weights).
2. Does the long-timescale schedule (4000 steps) provide enough resolution to distinguish persistence vs. delayed washout?
3. λ_0=1.3e-3 fixed at the g165 balanced strength — should this be re-derived for the longer horizon?

## Provenance

- Audit motivation: `research/EARLY_HELP_META_AUDIT_2026-04-27.md` g137 decay-shape extraction
- Codex cycle 27 direction review Q2: `codex_outputs/heartbeats/cycle27_direction_review_20260427T110500.md`
- g165 prereg (predecessor): `research/prereg/genome_165_annealed_donor_2026-04-27.md`
- g137 source: `results/genome_137_optimizer_state_transfer.json`
