# genome_165 — annealed donor / decaying-anchor washout test

**Status.** DRAFT 2026-04-27 — pending Codex pre-flight sign-off before LOCK.
**Trigger.** Locked as first post-g158c GPU slot regardless of g158c verdict (per cycle 24 strategic pivot in `research/programs/post_g158c_decision_tree.md`).
**Source.** Codex cycle 24 direction review (`codex_outputs/heartbeats/cycle24_direction_review_20260427T101600.md`) + early-help meta-audit (`research/EARLY_HELP_META_AUDIT_2026-04-27.md`).

## Hypothesis

A decaying-anchor / annealed donor schedule produces persistent positive NLL advantage at convergence, beating both fixed-persistence donor (currently empirically dominated, 0/7 mechanisms persist) and scratch (current dominant arm at convergence).

The motivation is empirical, not post-hoc:
- 7 distinct donor mechanisms (ridge-grafted init, mean-shift init, trainable mean-shift, weight-space seed, rank30 adapter, frozen-attn glue, optimizer-state) all wash out by step 4-1128.
- 0/7 produce persistent advantage; mean final advantage = -0.43 nats.
- g008 already tested STATIC anchoring (fixed-strength regularization to donor) and still washed out.
- The hypothesis is that DECAYING the anchor — strong early, zero by some τ — lets the donor's information transfer without preventing the recipient from adapting beyond the donor's distribution.

## Setup

- **Donor:** Qwen3-0.6B (trained checkpoint), pulled from `../models/registry.py`.
- **Recipient:** random-init Qwen3-0.6B-architecture model (same head dim, same depth).
- **Loss:** L_total = L_CE(recipient) + λ(t) · ‖θ_recipient - θ_donor‖² (Frobenius regularizer to donor weights, with anneal schedule).
- **Schedules** (factorial with λ_0):
  - **constant** (control, washout-replication): λ(t) = λ_0
  - **step**: λ(t) = λ_0 if t < 25 else 0
  - **linear**: λ(t) = λ_0 · max(0, 1 - t/50)
  - **exponential**: λ(t) = λ_0 · exp(-t/10)
  - (additional standalone arm) **hard_cut_step1** at λ_0=1.3e-3 with attention-only submanifold — see "Cycle 30 added arm" below
- **λ_0 ∈ {1.3e-4, 1.3e-3, 1.0e-2}** (3 strengths × 4 schedules = 12 anchored arms + 1 scratch baseline = 13 arms total). **REVISED 2026-04-27 per Codex lean pre-flight**: Frobenius F² ≈ 2.03e6 over 596M params; the original grid {1.0, 0.1, 0.01} would collapse all three strengths to "donor clone" because anchor gradient dominates CE by ≥7.6× across the entire grid. The revised grid spans weak (CE 10× dominates), balanced (comparable gradients), and strong (anchor 7.6× dominates without being effectively frozen). See `codex_outputs/g165_lambda_grid_check_20260427T114500.md`.
- **Seeds:** [42, 7, 13] (3 seeds for canonical scope).
- **Training:** 500 steps, batch_size=8, lr=3e-4 (matched to grafting series).
- **Eval:** C4 val every 25 steps; full trajectory recorded for trajectory comparison.

## Cells

42 train cells: 14 arms × 3 seeds. (12 anchored full-weight + 1 attention-only-hardcut + 1 scratch baseline.)

## Cycle 30 added arm (Codex direction review 2026-04-27)

Codex cycle 30 direction review identified a blind spot: g165 anchors ALL recipient weights, but g125's persistence (the empirical motivation, +0.07 nats) used **attention-only** anchor. Without an attention-only arm in g165, results aren't directly comparable to g125's boundary condition. Added one arm:

- Label: `anchor_attn_only_lam1.3e-3_hardcut`
- Submanifold: attention parameters only (`.self_attn.` in name match)
- λ_0: 1.3e-3 (the balanced strength from the Codex-revised grid)
- Schedule: `hard_cut_step1` (λ active only at training step 1, zero from step 2 onward)
- Tests: "early-help only on attention submanifold, no continued anchor" — directly probes the g125-boundary inside g165's framework.

**Compute estimate:** ~3-4hr wall (matches Codex cycle 24 estimate). VRAM peak <12 GB (one model live at a time).

## Locked PASS / FAIL criteria

**PASS:**
- At least one (λ_0, schedule) combination produces final-step C4 NLL advantage over scratch ≥ +0.5 nats with 95% bootstrap CI excluding zero (paired across 3 seeds).
- Ideally: the annealed schedules (step / linear / exponential) beat the constant (washout) schedule, which validates the "decay is the active ingredient" claim.

**WEAK_PASS:**
- At least one combination produces final-step advantage ≥ +0.2 nats with CI excluding zero, but not strong enough to lock the headline finding.

**FAIL:**
- All 12 anchored configurations wash out (final-step advantage < +0.2 nats OR CI crosses zero).
- This would replicate the n=7 audit finding at canonical scale and would close the annealed-donor avenue as a flagship-grade contribution. Pivot to either (a) optimizer-state + decay anchored combined experiment (the g137 outlier follow-up) or (b) procurement of wall-power meter to unblock g155.

## Compute envelope (COMPUTE.md §9 compliance)

- Wall-clock estimate: 3-4h
- Peak VRAM: < 12 GB (one Qwen3-0.6B-architecture model live; donor weights memory-mapped)
- Peak RAM: < 24 GB
- Disk: < 200 MB results JSON
- Quantization: BF16 forward, FP32 lr probes if needed

## Why this is the primary §0-axis move (Codex rationale)

Per cycle 24 direction review:
- §0 axis (capability transfer without retraining) should replace architecture-prior axis as primary line.
- Donor/transfer chain was retired prematurely; today's audit is the strongest fresh signal in the repo because it directly reframes the moonshot bottleneck.
- Anchored/annealed donor scores 7.3/10 on PASS — higher than Path A (g162 capacity sweep, 6.8) and B (g158e endpoint expansion, 6.4). Unlike Path C (g155, 8.2), it is NOT hardware-blocked.
- The framing is mechanistically motivated, not post-hoc: 7 mechanisms wash out → decay schedule is the natural intervention.

## What FAIL means for §0

Codex frames this carefully: the headline is NOT "§0 is solved" but "zero-step donor signal is real and large, but unstable under SGD." If g165 FAILs:
- The "active ingredient" of donor mechanisms is still the early signal (which is real and large).
- But under continued training, the recipient's gradient dynamics consistently destroy the donor's contribution.
- Pivot direction: the right transfer signal may be at the optimizer-state / update-direction level, not parameter-value level (g137 outlier hypothesis).

## Files (to be created when LOCKED)

- `code/genome_165_annealed_donor.py` — main runner
- `results/genome_165_annealed_donor.json`
- `code/integrate_g165.py` — verdict integration helper

## Pre-flight protocol

Before LOCK, fire Codex consult: "Read `code/genome_165_annealed_donor.py` (when written) and this prereg. Audit:
1. Donor weights loading: are we honestly using the trained Qwen3-0.6B and not accidentally training the donor too?
2. Anchor regularizer: is the Frobenius distance computed on the right submanifold (full weights vs attention-only vs MLP-only)? Decision: full weights for first audit; can refine if needed.
3. λ scaling: at λ_0=1.0 the regularizer might dominate the CE loss. Verify the gradient ratio at step 0 is sane.
4. Seed independence + LR robustness.
5. Bootstrap CI methodology on paired N=3 seed deltas at step 500."

## Provenance

- Audit motivation: `research/EARLY_HELP_META_AUDIT_2026-04-27.md`
- Codex cycle 24 direction review: `codex_outputs/heartbeats/cycle24_direction_review_20260427T101600.md`
- Decision tree override: `research/programs/post_g158c_decision_tree.md` § STRATEGIC OVERRIDE
- g008 trainable mean-shift PILOT (showed static anchor washout): `grafting/results/grafting_008_trainable_meanshift_persistence.json`
