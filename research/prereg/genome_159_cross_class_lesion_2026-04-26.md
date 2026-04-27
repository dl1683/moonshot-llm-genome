# Pre-registration: genome_159 cross-class causal transport-vs-local lesion

**Date:** 2026-04-26
**Status:** LOCKED at first commit. CONDITIONAL: launches only if g157 returns PASS_G157 or PARTIAL_G157.
**Author:** Devansh / Neural Genome
**Theory ref:** `research/derivations/prefix_information_transport.md`
**Program ref:** `research/programs/post_g156_pass_program.md` §g159

## 0. End-goal alignment

g159 is the architecture-class extension test. The transport theory should NOT be a Llama-only effect. By lesioning transport-output vs local-output subspaces in *pretrained* models from three different architecture classes, we can test whether the asymmetry replicates without any new training. If yes, the theory is class-general; if not, it shrinks back to a Llama-family story.

## 1. Hypothesis

In distinct text architecture classes, transport-sublayer lesions hurt natural-text NLL more than equal-rank local-sublayer lesions, AND that gap shrinks on order-destroyed (token-shuffled) controls.

## 2. Systems (canonical registry: `Projects/models/MODEL_DIRECTORY.md`)

- **Qwen3-0.6B** (transformer; attention + MLP)
- **RWKV-4-169M** (linear-recurrent; time-mix + channel-mix)
- **Falcon-H1-0.5B** (hybrid transformer + Mamba)

For each model, define "transport sublayer" = the cross-token mixing module:
- Qwen3-0.6B: self-attention output (per layer)
- RWKV-4-169M: time-mix output (per layer)
- Falcon-H1-0.5B: attention OR Mamba output, whichever is the cross-token mixer in that block

"Local sublayer" = the per-token nonlinear module:
- Qwen3-0.6B: SwiGLU MLP output
- RWKV-4-169M: channel-mix output
- Falcon-H1-0.5B: MLP output

## 3. Lesion protocol

At each functional depth d ∈ {0.25, 0.50, 0.75}:

1. Calibration pass: run model on 2048 c4 windows; collect transport-sublayer outputs T_l(x) and local-sublayer outputs L_l(x) (residual-stream contributions only — i.e. the per-block delta).
2. Fit top-32 PCA basis for T_l and a separate top-32 PCA basis for L_l on the calibration set.
3. Project-out at eval time: subtract the top-32 PCA component of the corresponding output before adding to the residual stream.

Two lesion conditions per (model, depth):
- **lesion_transport**: project out top-32 of T_l only
- **lesion_local**: project out top-32 of L_l only

Eval on:
- **natural**: 1024 windows from `c4_val_dedup_v2`
- **token_shuffled**: same 1024 windows after per-row token permutation (matched control, shuffle_seed=42)

Metric: NLL increase from baseline (no lesion) on each eval set.

## 4. Pre-stated criteria (per locked program §g159)

For each (model, depth, eval), compute ratio:
  R_nat = ΔNLL_lesion_transport / max(ΔNLL_lesion_local, ε)
  R_shuf = same on shuffled

Aggregate across depths (median of three depths):
- **PASS**: in 3/3 classes, median R_nat ≥ 1.5; on shuffled controls, R_shuf falls by ≥ 40% (i.e. R_shuf ≤ 0.6 × R_nat) in ≥ 2/3 classes.
- **PARTIAL**: 2/3 classes pass on R_nat ≥ 1.5; OR 3/3 with median R_nat only ≥ 1.25.
- **KILL**: local lesions match or exceed transport lesions (R_nat < 1.0) in ≥ 2/3 classes.

## 5. Universality level claimed

If PASS: candidate Level-2 (family-local universality across 3 trained-NN classes; not yet biology-validated, but biology is per §0.05 scope-locked OUT of this moonshot, so Level-2 is the ceiling for this primitive).

## 6. Compute envelope (COMPUTE.md §9)

- VRAM: largest model is Qwen3-0.6B BF16 ≈ 1.4 GB + activations. Peak < 6 GB. ✓
- RAM: PCA bases (3 models × 3 depths × 2 sublayers × 32 components × hidden ≤ 4096) trivially small. ✓
- Wall-clock: 3 models × 3 depths × 2 lesions × 2 conditions × 1024 windows ≈ 36 forward passes. Each ~5-10s. Total ~6 min eval + ~10 min calibration + ~10 min PCA fits = ~30-40 min. ✓
- Disk: PCA artifacts < 100MB.
- Quantization: Qwen3-0.6B BF16, RWKV-4-169M FP16, Falcon-H1-0.5B BF16. All < 1B per ladder.

## 7. Conditional launch

Launches only if g157 PASS or PARTIAL. If g157 KILLs, theory is dead and g159 is moot.

Also gated on g156 saved checkpoints being preserved (yes — under `results/genome_156_checkpoints/`, gitignored).

## 8. Audit-hard protocol (per program shared protocol)

- 13-token rolling-hash dedup on c4_val_dedup_v2 against any training data the pretrained models saw (best-effort; we cannot fully audit Qwen3/RWKV/Falcon training corpora, but we can document the dedup we DID do).
- Wikitext-103 VAL split, not train.
- Three-seed bootstrap CIs on each ratio.

## 9. What a null result means

KILL = transport-vs-local asymmetry is Llama-only. The principle does not generalize across architecture classes. Theory becomes a Llama-family-specific story. Manifesto-aligned re-pivot would be: ride the empirical Llama result into the distillation product (g154+g155 path) without further architectural derivation.

## 10. Artifacts

- `code/genome_159_cross_class_lesion.py`
- `results/genome_159_cross_class_lesion.json`
- `results/genome_159_run.log`
- `cache/g159_pca_bases/` (gitignored)
- Ledger entry per CLAUDE.md §4.6

## 11. Locking

LOCKED upon commit.
