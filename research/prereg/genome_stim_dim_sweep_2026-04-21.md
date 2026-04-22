# Pre-registration: Stimulus intrinsic-dimension sweep — does `c = p·d_rd` depend on stimulus dim?

**Status:** LOCKED at commit that adds this file.

**Date:** 2026-04-21.

**Authors:** Dev (CMC / AI Moonshots).

---

## Hypothesis

The modality-stratified invariant `c = p · d_rd` takes value ≈ 2 on text and ≈ 3 on vision (genome_036/039). The integers are suggestive: they might come from stimulus intrinsic dimensionality (text ≈ 1D sequence, vision ≈ 2D image) via a theoretical constant related to rate-distortion asymptotics.

**Pre-registered prediction:** If we feed a VISION architecture (DINOv2-small) stimuli that have been artificially collapsed to approximate 1D structure, its measured `c = p · d_rd` will **shift toward 2** (the text value). Conversely, if we feed a TEXT architecture stimuli with richer-than-1D structure, its `c` will shift upward toward 3.

## Specific decision rules

**Vision shift test (primary):**

On DINOv2-small mid-depth, compare `c` measured on:

- `c_natural` — 1000 natural ImageNet-val images (seed 42). Baseline.
- `c_1d_stripes` — 1000 images where only a single row is a natural ImageNet row, repeated vertically to fill the 224×224 canvas (artificial 1D structure).
- `c_iid_noise` — 1000 images drawn IID from uniform noise (no spatial structure at all).

**Rules:**

1. `c_natural` should reproduce the prior vision value (2.63-3.95 band, mean 3.18).
2. **Shift prediction:** `c_1d_stripes < c_natural` by at least 0.3 (i.e., forcing 1D structure drops c substantially).
3. **Floor prediction:** `c_iid_noise < c_1d_stripes` OR `c_1d_stripes` breaks the power-law form entirely (R² < 0.90).
4. **Optional tight prediction:** `c_1d_stripes` lands within 0.3 of text `c ≈ 2` (i.e., `c_1d_stripes ∈ [1.7, 2.3]`). Not required for hypothesis support but would be strong evidence.

Pre-registered outcome categories:

- **STRONGLY SUPPORTED** — rules 1, 2, 3, and 4 all hold.
- **SUPPORTED** — rules 1, 2, 3 hold; 4 does not hold.
- **PARTIAL** — rule 2 holds (direction correct) but magnitude < 0.3.
- **FALSIFIED** — rule 2 fails (c_1d_stripes ≥ c_natural).

**Text shift test (secondary, fires if vision primary is suggestive):**

On Qwen3-0.6B mid-depth, compare `c` measured on:

- `c_c4_natural` — 1000 C4-clean sentences seed 42. Baseline.
- `c_c4_shuffled` — same sentences with tokens shuffled within each sequence (destroys syntactic dependencies).
- `c_c4_markov1` — 1000 synthetic sequences from a uniform-bigram Markov chain on the C4 vocab (simple 1st-order dependencies only).

Pre-registered prediction: `c_shuffled > c_c4_natural` (destroying structure → richer-effective-dim → higher c)? Or the opposite? Not locking a direction here — exploratory.

## Held-out systems

If the vision primary test gives `c_1d_stripes < 2.5` (i.e., shifted toward text value), repeat on a second vision architecture (I-JEPA-ViT-H/14) to confirm it's not DINOv2-specific. Pre-registered pass: second system shifts in the same direction by ≥ 0.3.

## Sample sizes + seeds

`n = 1000` stimuli per condition, seed = 42 for all stimulus generation. Single-pass measurement; no cross-seed averaging (if effects are real at this scale, cross-seed tightens them; if not, no averaging will save them).

## Reporting commitment

Ledger entry `genome_040_stim_dim_sweep_vision` + (if fires) `genome_041_stim_dim_sweep_text`. Report raw `p`, `d_rd`, `c`, `R²` per cell; pre-registered verdict category; paper §5.2 updated with outcome regardless of sign.

## Compute envelope

4 vision conditions × extract + d_rd + kNN fit ≈ 10 min total on RTX 5090. Well under envelope.

## LOCK

This file locked at the commit that adds it to the repository.

**Pre-registered by:** Dev (devansh@svam.com), 2026-04-21.
