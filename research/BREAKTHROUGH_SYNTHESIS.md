# Breakthrough synthesis — 2026-04-21 session

*Draft integration of genome_036 through genome_047. Read this before firing the next compiler-sprint experiment. Not paper polish — this is the framing the paper will need if the claims hold.*

---

## The single clean claim

**Training produces a weight-activation co-evolution characterized by a modality-stratified geometric invariant `c = p · d_rd`. The invariant is necessary for capability but not sufficient; capability is carried by the specific trained feature-directions occupying the geometric envelope the invariant describes.**

Five supporting sub-claims:

1. **The invariant exists and is training-specific** (genome_036, 038, 039).
   Across 4 text + 3 vision trained systems, `c = p · d_rd ≈ 2.07` (text) or `3.18` (vision). Random-init twins break the relation (DeepSeek untrained `c ≈ 14`). Training compresses `d_rd` ~3× and simultaneously establishes the identity.

2. **The geometric envelope is stimulus-intrinsic-dim-sensitive** (genome_040, 041).
   Forcing vision stimuli to approximate 1D structure (single-row-tiled images) shifts vision `c` *toward* text `c = 2` on 2/2 vision systems (Δc = -0.45 DINOv2, -0.29 I-JEPA). The integer gap 2 vs 3 is partly driven by stimulus dimensionality.

3. **No activation-space transfusion produces capability** (genome_042, 043, 046).
   Covariance whiten+recolor, k-means codebook snap, and trained-basis projection each move the untrained geometric summary substantially toward trained but leave NLL at untrained baseline (0% drop). 2nd-moment, piecewise-constant, and basis-only transfer all fail.

4. **PCA destruction of trained geometry monotonely destroys capability** (genome_044).
   Projecting trained activations onto top-`k` PCA (for `k ∈ {4, ..., 512}`) traces a clean monotone joint curve: `d_rd` drops, `c` drops, NLL rises, across 2 orders of magnitude. Geometry is *necessary* for capability.

5. **Random subspaces at same rank preserve geometry but destroy capability** (genome_045).
   At `k = 256`, a random orthonormal subspace produces `d_rd = 10.42, c = 1.95` (essentially at full-trained `10.69, 1.90`) yet NLL is 11.58 (untrained-level). PCA at same rank produces less-matched geometry but near-baseline NLL. **At matched envelope, direction identity is what carries capability.**

6. **Weight interpolation is non-linear** (genome_047).
   `W(α) = α · W_trained + (1-α) · W_untrained` produces a non-monotone geometric excursion (`c` rises from 4.04 to 4.84 at α=0.25 before falling to 1.90) and a threshold+acceleration NLL (almost flat until α ≈ 0.5, then exponential descent). Mode-connectivity-like; intermediate weights are a distinct regime.

---

## Why this is bigger than the paper currently claims

The original paper claim: "9 architectures share a kNN clustering coefficient power-law." That's a respectable correlation on top of the Platonic Representation Hypothesis literature. Workshop-grade.

The synthesis claim: "Training produces a geometric invariant that is causally necessary for capability but not sufficient — capability is carried by specific trained feature-directions within a geometric envelope that random directions at the same rank also satisfy." That is a **principled causal characterization of what training is**, with:

- A falsifiable invariant (`c = p · d_rd`, modality-stratified)
- A necessity claim (PCA compression destroys)
- A non-sufficiency claim (transfusion fails, random same-rank destroys at preserved envelope)
- An emergent-structure claim (weight interpolation is non-linear)
- A mechanism hint (weight-activation co-evolution, not either alone)

No big lab publishes this because it reframes scaling: the compute is not paying for "more geometry" (random directions satisfy the envelope for free); it is paying for *specific learned directions* that the rest of the trained weights can interpret. This is the manifesto claim — *Intelligence = Geometry, not Scale* — in its sharpest form yet: the geometry is free, the directions within it are what scale produces.

---

## The three paradigm-shift claims this can become, ranked by feasibility

1. **A derivation of `c = 2` (text) and `c = 3` (vision) from stimulus intrinsic-dim + rate-distortion theory.**
   Partial evidence from genome_040/041 (1D-stimulus shifts vision toward text). A clean first-principles argument that connects `c` to the ratio of stimulus intrinsic dim to effective coding dim would be Nature-grade.
   *Status: hint-level; derivation attempt pending.*

2. **An electricity-grade efficiency demo.**
   Train a small model from scratch with `c = p · d_rd` as auxiliary regularizer. If convergence is ≥20% faster (at matched validation loss) than baseline, geometry is a training-compute surrogate. That is the Compiler.
   *Status: not attempted this session; ~1 day of compute to do a first pass.*

3. **Geometry transfusion that preserves directions AND weight-interaction machinery.**
   Full mid-layer trained-weight transplant into otherwise-untrained network, with gradient-free loss measurement, to test whether a single trained layer carries partial capability. If yes, capability is layer-localized; if no, it is distributed.
   *Status: not attempted; ~1 hour of compute.*

---

## What the next experiment should be

Per strategic verdict 2026-04-21-2329 (MINOR-ADJUSTMENT, Compiler-first sprint), the three questions were (1) is the invariant causal? (yes, genome_044), (2) what higher-order invariant is capability-linked? (direction identity, genome_045), (3) can we get a tiny compute-efficiency win?

**Question 3 is the only open rung-3 question.** The single experiment that would answer it in the affirmative is #2 above: small-model training with `c = p · d_rd` aux regularizer. That is the next concrete breakthrough-direction fire.

---

## What this does not address

- No biological derivation yet (why does mouse V1 hit the same band as DINOv2?).
- No out-of-distribution test (would this break at inference-time shift?).
- No evidence that the direction-identity claim generalizes beyond DeepSeek to vision / diffusion / biology.
- Single model (DeepSeek) for the rung-3 experiments — need replication on Qwen3 + RWKV + DINOv2 once the story is locked.

These are second-order. The first-order move is #2 electricity-demo.
