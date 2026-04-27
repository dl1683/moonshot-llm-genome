# Pre-registration: genome_159b cross-class lesion RANK SWEEP

**Date:** 2026-04-27
**Status:** LOCKED at first commit. CONDITIONAL: launches only if g159 returns INCOMPLETE or KILL with the diagnostic note "ratio NaN due to non-positive local lesion delta" (i.e., rank-32 lesion too small to bite).
**Author:** Devansh / Neural Genome
**Predecessor:** `research/prereg/genome_159_cross_class_lesion_2026-04-26.md`

## 0. Why this prereg

g159 mid-run observation: Qwen3 at all 3 depths produces non-positive local lesion delta (d_l ≤ 0.03 nats), causing the per-depth ratio R = ΔNLL_transport / ΔNLL_local to be undefined. This indicates the rank-32 PCA lesion is removing too little of the local sublayer's contribution to the residual stream — either because the local sublayer is genuinely low-impact at these depths in trained Qwen3, OR because top-32 captures only a fraction of variance (we measured: local top-32 var-explained = 0.23-0.28 on Qwen3, while transport top-32 = 0.50).

Rank-32 was the locked Codex-identified primary spec but admittedly unfair across architectures (3.1% vs 4.2% of residual width). At ~25% var-explained on local, rank-32 isn't biting.

g159b: rank sweep across {32, 64, 128, 256} on the same 3 architectures. If at any rank ≥ 64 the local sublayer becomes load-bearing (d_l > 0.02 nats) AND the ratio R discriminates, the locked g159 prereg is salvageable at higher rank. If even rank-256 doesn't bite, the local sublayer truly isn't load-bearing at these depths — different theoretical conclusion.

## 1. Hypothesis

For ranks ≥ 64 PCA components projected out of the local sublayer's residual contribution, the lesion bites (d_l > 0.02 nats on natural data) AND the natural-vs-shuffled R ratio differential is observable per the locked g159 PASS criterion.

## 2. System

- Same 3 architectures as g159: Qwen3-0.6B, RWKV-4-169M, Falcon-H1-0.5B
- Same 3 depths: {0.25, 0.5, 0.75}
- Rank sweep: {32, 64, 128, 256} (4 rank values)
- Same calibration set (2048 c4-val) and eval (1024 nat + 1024 shuf)
- Reuse fitted PCA bases from g159 if available (top-256 components are a superset of top-32, so we just slice)

## 3. Pre-stated criteria

- **PASS_159b:** at SOME rank in {32, 64, 128, 256}, all 3 classes show median R_nat ≥ 1.5 AND ≥2/3 classes show R_shuf collapse ≥40%. Plus ALL d_l values positive at that rank. Theory's class-extension claim survives at properly-sized lesion.

- **PARTIAL_159b:** 2/3 classes pass at SOME rank.

- **KILL_159b:** at NO rank does d_l consistently bite AND ratio discriminate. Theory's class-extension prediction is genuinely empirically false.

## 4. Compute envelope

- Reuse fitted PCA bases from g159 (already cached?) → just slicing rows
- 4 ranks × 3 models × 3 depths × 2 lesion sites × 2 conditions × 1024 windows = 73,728 lesion-eval forwards. At ~3 forward/s on RTX 5090 (mixed batch sizes), ~7 hr.

That's over envelope. Reduce: skip rank=128 (interpolate), skip depth=0.75 (use only midpoints {0.25, 0.5}). 2 ranks × 3 models × 2 depths × 4 conditions × 1024 ≈ 49k forwards = ~5 hr.

Still over envelope. **Cut to 2 architectures (Qwen3 + RWKV-4) and 2 ranks {64, 128}**: 2 × 2 × 2 × 4 × 1024 ≈ 33k forwards = ~3 hr. Within envelope.

If even reduced scope is over envelope at runtime, kill and split sessions.

## 5. Conditional launch

Launches if g159 returns INCOMPLETE or KILL with the diagnostic non-positive-local-delta cause. If g159 returns PASS or PARTIAL, this prereg is archived.

## 6. Locking

LOCKED upon commit.
