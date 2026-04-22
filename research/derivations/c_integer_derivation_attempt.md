# Derivation attempt: why `c_text ≈ 2` and `c_vision ≈ 3`?

*Draft. Not locked. First-principles attempt at the integer gap in the modality-stratified training invariant `c = p · d_rd`.*

---

## The target

Empirically, across 4 text architectures (Qwen3-0.6B, Qwen3-1.7B, RWKV-4-169M, DeepSeek-R1-Distill-Qwen-1.5B) and 3 vision architectures (DINOv2-small, I-JEPA-ViT-H/14, CLIP-ViT-B/32) at mid-depth on their natural stimulus banks:

- `c_text = p · d_rd ≈ 2.07 ± 0.23` (n=4)
- `c_vision = p · d_rd ≈ 3.18 ± 0.69` (n=3)

Two suggestively-integer values separated by ~1.

And from genome_040 / genome_041: forcing vision stimuli to ~1D structure (single-row-tiled images) shifts DINOv2 `c` from 2.96 → 2.51 (Δ = -0.45) and I-JEPA from 3.22 → 2.93 (Δ = -0.29). Both shift toward the text value. Direction matches stimulus intrinsic-dim.

## First-principles sketch

### Setup

Pooled hidden-state point cloud `X ⊂ R^h` of size `n`. Training minimizes a next-token (text) or reconstruction/contrastive (vision) loss. Call the informative features of the stimulus distribution `F`; call its intrinsic dim `d_stim`.

### Candidate 1: `c` ≈ `d_stim + 1`?

For text, pooled-sentence features roughly track 1-D sequential information — tokens unfold on a 1D axis. If the relevant intrinsic dim of a pooled-text feature vector is approximately 1, then `c_text` ≈ `1 + 1 = 2`.

For vision, pooled-image features track 2-D spatial structure — pixels unfold on a 2D grid. If the relevant intrinsic dim is approximately 2, then `c_vision` ≈ `2 + 1 = 3`.

The "+1" would have to come from a dimensional offset in the clustering-coefficient scaling law. Empirical check: forcing vision stimuli to ~1D structure should shift `c_vision` toward `1 + 1 = 2`. Genome_040/041 observe exactly that direction, partially.

### Candidate 2: `c` ≈ `2 · d_stim`?

Text `c = 2 = 2·1`, vision `c = 3 ≈ 2·1.5`. Doesn't fit as cleanly since `d_stim_vision = 2` would give `c = 4`, which is too high. Unless the effective `d_stim` for a pooled ViT at mid-depth is closer to 1.5 (because pooling degrades 2D grid structure). Less clean.

### Candidate 3: Rate-distortion argument

From rate-distortion theory for a `d_stim`-dim Gaussian source: `R(D) ≈ (d_stim/2) · log(σ²/D)`. The rate-distortion dimension `d_rd` we measure via k-means scaling recovers `d_stim` when the point cloud is close to Gaussian on the underlying manifold.

The clustering coefficient `C(k)` for a kNN graph on `n` points drawn from a `d_stim`-dim distribution has been studied (Steinwart et al. 2022, Aamari et al. 2019 on kNN-graph asymptotics on manifolds). The leading-order scaling is roughly `C(k) ~ (k/n)^β` where `β` depends on `d_stim`.

For `β = 2/d_stim`: `log C / log k = 2/d_stim - 2/d_stim log n / log k` — hard to pin down a clean integer constant.

### Candidate 4 (speculative): stimulus dim + token dim

A trained text pooled hidden state compresses both:
- The 1D *position* of the token being summarized
- The 1D *identity* of the token content

Two 1-D axes → text `c = 2`.

A trained vision pooled hidden state compresses:
- The 2D *spatial position* of the patch being summarized
- The 1D *identity* of the patch content (or some channel-mixing axis)

Three axes (2 spatial + 1 identity) → vision `c = 3`.

This would predict:
- Audio pooled features: 1D time + 1D frequency + 1D identity → `c = 3`? or 2D+1D → `c = 3`?
- Diffusion (DiT, where each token is a 2D-patch at a specific noise level): 2D spatial + 1D temporal/noise → `c = 3`? — DiT gave `c ≈ 0.21·k_dim ≈ ?` (we have DiT data at `c = 0.18-0.21 * d_rd` from genome_022 but the mapping to this derivation is rough).

If this derivation is correct, the general prediction is **`c = d_stim_axes_count + d_identity_axes_count = d_stim_structural + 1`** (identity always contributes 1 axis for token-class or patch-class).

### Test the candidate-4 prediction

Qwen3 text 1D+1D → `c = 2`. ✓ (observed 2.07)
DINOv2 vision 2D+1D → `c = 3`. ✓ (observed 2.96)
DiT diffusion 2D+1D (+ noise time axis?) → `c = 3` or 4. (observed 2.5× the expected — but DiT's hidden state has unusual structure due to VAE latent encoding and AdaLN conditioning; harder to interpret.)

Forced 1D-stripes vision 1D+1D → `c` should shift toward 2. Empirically does: genome_040 Δc = -0.45.

IID-noise images (no structure) → `c` should be... this breaks the derivation (no effective stimulus dim). Observed `c_noise ≈ 8` on DINOv2 in genome_040 — extreme outlier. The derivation only applies to structured stimuli.

### What's missing from candidate 4

- A clean proof of why the count of stimulus-dim axes enters the `C(k)` power-law exponent as a LINEAR SUM with a +1 offset.
- Derivation of *which* `C(k)` scaling law regime we're in (large-n, fixed-k; fixed-n, k/n→0; etc.).
- Why training specifically finds the `c = d_stim + 1` fixed point.

## Honest status

This is a hand-waved candidate, not a derivation. But it makes concrete predictions:

1. **Audio/speech representations should land at `c ≈ 3`** (1D time + 1D frequency + 1D phoneme identity), same as vision.
2. **Multi-modal models (CLIP, Perceiver) should land at `c` = max(modality_c)** — whichever modality has more structural axes dominates.
3. **Video models should land at `c ≈ 4`** (2D spatial + 1D temporal + 1D identity).
4. **1D-structured vision stimuli should shift `c_vision` toward 2**, which genome_040/041 partially confirms.

## Next fires that would sharpen this

- Audio-modality system (whisper / wav2vec) measured for `c`. If `c ≈ 3`, candidate 4 strengthens.
- Video system (VideoMAE, ViVit) measured for `c`. If `c ≈ 4`, candidate 4 strengthens further.
- Pure multi-scale time-series model — should land at different `c`.

## Why this (if it holds) matters

If `c = d_stim_axes + 1` is true, then:

- The integer gap 2 vs 3 is NOT a free parameter — it is a direct reading of stimulus dimensionality.
- Scale doesn't determine `c`; stimulus structure does. Scale just ensures training converges there.
- A model's `c` value becomes a readout of what DIMENSIONAL STRUCTURE the model has learned to represent — and can be compared directly to the dimensional structure of biological cortical areas processing similar stimuli.
- The Genome Equation achieves its "derive-not-fit" criterion: given only the modality/stimulus dim, predict `c` without any curve-fitting.

That would be a Nature-grade claim grounded on this session's data.

## This is a DRAFT

Not claiming any of the above is proven. The hand-wave in candidate 4 is too loose. A proper derivation would need to rigorously connect:
1. Continuum `C(k)` scaling for kNN graphs on `d_stim`-dim manifolds
2. Rate-distortion `d_rd` for text vs vision manifolds
3. The integer offset `+1`

Future sessions should either (a) rigorously derive candidate 4, or (b) falsify it by measuring an audio or video system.
