# genome_162 — transport-arm capacity sweep (Path A: PASS_canonical)

**Status.** DRAFT 2026-04-27 — pending g158c PASS_canonical verdict + Codex pre-flight sign-off before LOCK.
**Trigger.** Activates iff g158c returns PASS_canonical (mean rho >= +0.8, mean Delta_256 >= +2.0pp with 95% CI excluding zero, mean Delta_32 <= 0.0pp).
**Source.** Codex direction consult 2026-04-27 (`codex_outputs/heartbeats/post_g158c_design_20260427T090500.md`) Path A.

## Hypothesis

The architecture-prior advantage from removing MLP layers is monotone in transport-arm capacity (number of transport-only layers k). Doubling transport capacity from k=3 to k=7 produces a smooth dose-response in the long-context regime (L=256), and a mostly-flat or slightly-negative slope in the short-context regime (L=32) where the inversion was originally observed.

This turns the binary 3L-vs-6L effect of g158 into a CURVE — a context-conditional dose-response architecture intervention.

## Setup

- **Arms:** noMLP at layers k ∈ {3, 4, 5, 7}, all hidden=384. Single baseline arm `baseline_6L+MLP` (the same as g158c).
- **Contexts:** L ∈ {32, 256} only (drop intermediate L=64, 128 to keep wall-clock tractable).
- **Seeds:** [42] (single seed pilot, parallel to g158 PILOT scope; canonical extension if PASS).
- **Per-cell FLOP budget:** matched at 193.27 TFLOP (same as g158c).
- **LR selection:** per-arm at L=32 and L=256 separately, take min (same policy as g158c per Codex pre-flight Sev7 fix).
- **Eval:** C4 + Wikitext val (top-1 acc, NLL).

## Cells

10 train cells total: 2 contexts × (1 baseline + 4 noMLP depths) × 1 seed.

## Locked PASS / FAIL criteria (from Codex direction consult)

**PASS:**
- At L=256: `rho(k, Delta_256) >= +0.8` (rank correlation between k and Delta_256, n=4 transport depths)
- At L=256: `Delta_256(7L) - Delta_256(3L) >= +0.8pp` (k=7 wins more than k=3 by margin >= 0.8pp)
- At L=32: `Delta_32(7L) <= Delta_32(3L) + 0.1pp` (no positive slope of advantage with k at short L; should be flat or slightly negative)

**FAIL:**
- `rho(k, Delta_256) < +0.3` OR
- `Delta_256(7L) - Delta_256(3L) < +0.3pp`

## Compute envelope (COMPUTE.md §9 compliance)

- Wall-clock estimate: 2.5–3.0h (Codex direction consult estimate)
- Peak VRAM: < 4 GB (one ~30M BF16 model at a time)
- Peak RAM: < 16 GB
- Disk: < 100 MB results JSON
- Quantization: BF16 forward, FP32 LR-selection probes

## Why this is highest-leverage Path A move (Codex rationale)

Per Codex direction consult: "this turns g158 from a binary pairwise effect into a dose-response architecture intervention. If that curve is smooth, you have the closest thing left to a design law on this branch. If it is not smooth, the whole transport story stays a brittle 3L vs 6L curiosity. Head ablation is lower leverage first because it can localize a toy effect that may not generalize at all."

## Codex score

- PASS=6.8/10 (Codex audit). Same-family same-method confirmation; still doesn't break §0.1 ceiling.
- FAIL=6.1/10. Honest constraint that g158/g156 story does not scale into a transport-budget law.

## Files (to be created when LOCKED)

- `code/genome_162_transport_arm_capacity_sweep.py` — fork of `code/genome_158c_3seed_canonical.py`, swap arm config to {3L, 4L, 5L, 7L noMLP} + 6L+MLP baseline, restrict L to {32, 256}, single seed.
- `results/genome_162_transport_arm_capacity_sweep.json`
- `code/integrate_g162.py` — verdict integration helper (decision rule above, mechanical PASS/FAIL emit).

## Pre-flight protocol

Before LOCK, fire Codex consult: "Read `code/genome_162_transport_arm_capacity_sweep.py` (when written) and the prereg. Audit for correctness (FLOP-matching at 4 different k, no LR/eval leakage), performance (peak VRAM at k=7), and any hidden assumption that biases the rho calculation."

## Provenance

- g158 PILOT (single-seed, DIRECTIONAL_SUPPORT): `results/genome_158_context_length_inversion.json`
- g158c canonical (running): `code/genome_158c_3seed_canonical.py`
- Codex direction consult: `codex_outputs/heartbeats/post_g158c_design_20260427T090500.md`
- Decision tree: `research/programs/post_g158c_decision_tree.md` Path A
