# Pre-registration: Candidate-8 spectral bridge universality (P1)

**Date:** 2026-04-22
**Status:** LOCKED at commit where this file is first added.
**Author:** session agent (Claude Opus 4.7), per strategic-verdict 2026-04-22-0047 P-M-I mandate (derivation slot extension).

## Hypothesis

**H8.1 (ratio universality):** On every trained system in the candidate-5 scorecard, measured at C4-clean mid-depth `n=1000`, the quantity

```
ratio(X) := eff_rank(X) / d_rd(X)
```

will be within **15 % relative error** of the measured `c(X) = p · d_rd(X)`.

`eff_rank(X) = (sum σ²)² / sum(σ⁴)` where `σ_i` are singular values of the centered activation matrix. `d_rd(X)` from `code/genome_rate_distortion_probe.py`. `p` from `code/genome_primitives.knn_clustering_coefficient` kNN power-law fit.

## Pass / fail criteria

- **PASS:** ≥ 9 of 11 systems with `rel_err(ratio, c) < 0.15`.
- **PARTIAL:** 5–8 of 11 pass. Bridge is real but narrower-scope.
- **FAIL:** < 5 of 11 pass. Qwen3 at 2.06 was a coincidence; candidate-8 is falsified.

If PASS: candidate-8 graduates from hypothesis to empirical identity, and the derivation pathway `(α, spectrum shape) → (eff_rank, d_rd) → c` becomes the primary derivation target.

If PARTIAL: isolate which systems pass (likely: same-modality clusters) and re-prereg a narrower version.

If FAIL: close candidate-8. Return to drawing board on what *else* produces the Qwen3 eff_rank/d_rd = 2.06 coincidence.

## Systems

Pulled from `../../models/registry.py`:

| Class | Model | Modality | c_C4 observed |
|---|---|---|---|
| 1 | Qwen3-0.6B | text | 1.89 |
| 2 | DeepSeek-R1-Distill-Qwen-1.5B | text | 2.41 |
| 3 | RWKV-4-169M | text | 1.95 |
| 4 | Falcon-H1-0.5B | text | 2.15 |
| 6 | DINOv2-small | vision | 2.96 |
| 7 | BERT-base | text-MLM | 2.65 |
| 7 | RoBERTa-base | text-MLM | 2.25 |
| 8 | MiniLM-L6 | text-contrastive | 2.03 |
| 10 | CLIP-text-B/32 | text + 1 align | 3.14 |
| 10 | CLIP-vision-B/32 | vision + 1 align | 3.95 |
| 11 | DiT-XL/2-256 | image-diffusion | 2.33 |

(11 systems.)

## Method

For each system, load at FP16, extract mid-depth (`n_hidden_layers() // 2`), pool (seq_mean text / cls_or_mean vision), 1000 C4 stimuli (seed 42) for text or 500 imagenet_val (seed 42) for vision. Compute:

1. `p` via kNN-clustering k-grid `[3,5,8,12,18,27,40,60,90,130]` + power-law fit on `C(k)`.
2. `d_rd` via rate-distortion probe (`genome_rate_distortion_probe.rate_distortion_dim`).
3. `eff_rank` via centered-covariance singular-spectrum participation ratio.
4. `c = p · d_rd` and `ratio = eff_rank / d_rd`.
5. `rel_err = |ratio - c| / c`.

Output: one row per system, stored in `results/gate2/svd_bridge_multimodel.json`.

## What a null result means

If candidate-8 FAILS:

- The Qwen3 `eff_rank/d_rd = 2.06 ≈ c = 1.89` match at 9 % rel_err is a COINCIDENCE (one data point is insufficient).
- The SVD alpha=0.861 signature remains real but is not the correct derivation path.
- Must investigate alternative spectral-to-c bridges (e.g., `alpha` alone, Rényi entropy of spectrum, or specific kNN-spectral moment).

## What a positive result means

If candidate-8 PASSES:

- `c` is derivable from the activation covariance spectrum via a two-quantity ratio.
- The P2 derivation (`eff_rank(α)` and `d_rd(α)` from pure power-law spectra) becomes a tractable random-matrix theory problem and is the next Nature-grade target.
- Different `c` values across (model, stimulus) pairs (e.g., Qwen3 C4 vs wikitext) should match their respective ratios point-by-point.

## Compute envelope (COMPUTE.md §9 checklist)

- Per-system: model load (~1-5 GB VRAM), extract 1000 stim, SVD on 1000×h matrix. Peak ~8 GB VRAM.
- Total wall: ~30-45 min for 11 systems.
- No training involved. Single-GPU RTX 5090 laptop, 22 GB budget, well within envelope.

## Locking

This prereg is **LOCKED** at the first commit adding this file. No modification until the experiment fires. Result commit must reference this file.
