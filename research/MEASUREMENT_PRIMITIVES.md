# MEASUREMENT PRIMITIVES

*The toolkit for mapping representational geometry. Each primitive is labeled either **coordinate** (has passed the architecture-agnosticism gate — works on ≥3 system classes) or **diagnostic** (useful but architecture-specific). Coordinates enter the atlas. Diagnostics are annotations only.*

This file is a living catalog. Add a primitive only after defining:
- what it measures,
- what its derivation basis is (or "empirical only"),
- which systems it currently works on,
- its status (coordinate / diagnostic / untested),
- what its candidate universality level is.

---

## Status legend

- 🟢 **Coordinate** — validated on ≥3 distinct system classes; eligible for atlas entries.
- 🟡 **Candidate coordinate** — works on 1–2 classes; promotion test pending.
- ⚪ **Diagnostic** — confirmed architecture-specific; useful as annotation, not atlas coordinate.
- ⚫ **Untested** — proposed but not yet run.

System classes for agnosticism: autoregressive LLM · SSM · hybrid · diffusion · JEPA · vision encoder · world model · biological recording · untrained network.

---

## 1. Information-geometric primitives

### 1.1 Intrinsic dimension (ID) ⚫
**Measures.** Effective dimensionality of the manifold on which activations lie.
**Methods.** TwoNN (Facco et al. 2017), MLE (Levina & Bickel 2004), persistence-based ID.
**Derivation basis.** Well-founded in statistics; distribution-free estimators exist.
**Agnosticism expectation.** Should work on any activation vector. High prior for promotion to coordinate.
**Candidate universality level.** Level 1 or 2. CTI-adjacent: prior work in `LLM exploration/` found SSM intrinsic dim ≈ 5.3× Transformer at comparable scale — a candidate Level-2 family constant.
**Script stub.** `code/genome_intrinsic_dim.py`

### 1.2 Participation ratio (PR) ⚫
**Measures.** Effective number of active dimensions in activations.
**Methods.** `PR = (Σ λ_i)² / Σ λ_i²` from activation covariance eigenvalues.
**Derivation basis.** Canonical in random matrix theory and neuroscience (Gao & Ganguli 2015 for neural recordings).
**Agnosticism expectation.** High — pure covariance measure.
**Candidate universality level.** Level 1 for scaling with depth/width; Level 2 family constants.
**Script stub.** `code/genome_participation_ratio.py`

### 1.3 Fisher information matrix trace ⚫
**Measures.** Local curvature of the loss landscape around trained weights.
**Methods.** Empirical FIM on a calibration set; eigenspectrum of FIM.
**Derivation basis.** Amari, information geometry.
**Agnosticism expectation.** Works on any differentiable network with a probability output. Harder on JEPAs and self-supervised encoders (needs proxy loss).
**Candidate universality level.** Level 2 family constants likely.

---

## 2. Manifold-geometric primitives

### 2.1 Persistent homology of activation point clouds ⚫
**Measures.** Topological structure (connected components, loops, voids) of activation clouds across training/inference.
**Methods.** Ripser / GUDI; compute birth-death diagrams for H_0, H_1, H_2.
**Derivation basis.** Topological data analysis literature (Carlsson 2009); applied to NN by Naitzat et al. 2020 ("Topology of Deep Neural Networks").
**Agnosticism expectation.** Topology is modality-agnostic by construction. High prior.
**Candidate universality level.** Level 1 if birth-death spectra share a shape across classes.

### 2.2 Ricci curvature of neighborhood graphs ⚫
**Measures.** Local geometry of activation space — positive curvature = clustering, negative = dispersion.
**Methods.** Ollivier-Ricci on k-NN graphs of activation embeddings.
**Derivation basis.** Discrete geometric analysis.
**Agnosticism expectation.** High — pure graph property.
**Prior work signal.** `llm-platonic-geometry/` reports positive Ricci curvature in LLM embedding spaces (Finite Geometric Types hypothesis). Candidate Level-2 or Level-1.

### 2.3 Lyapunov spectrum (dynamical) ⚫
**Measures.** Stability of forward-pass dynamics, treating generation as a dynamical system.
**Methods.** Jacobian power iteration; finite-time Lyapunov exponents.
**Derivation basis.** Dynamical systems theory.
**Agnosticism expectation.** Applies naturally to autoregressive, recurrent, and state-space models; needs adaptation for feedforward vision encoders (layer-wise instead of time-wise).
**Prior signal.** `llm-platonic-geometry/` — Lyapunov ≈ 0 in LLMs (near-critical dynamics).

---

## 3. Representation-similarity primitives

### 3.1 Centered kernel alignment (CKA) 🟡
**Measures.** Similarity between two representation spaces.
**Methods.** Linear or RBF-kernel CKA (Kornblith et al. 2019).
**Derivation basis.** Well-established.
**Agnosticism expectation.** Cross-system by construction. Current state: widely used on LLMs + vision; untested on diffusion/JEPA/world models at scale.
**Promotion test.** Run CKA between Qwen3-0.6B, Mamba2-370M, Falcon-H1-0.5B, DINOv2-small, V-JEPA-base, stable-diffusion-v1-5-text-encoder. If CKA patterns are stable across runs, promote to coordinate.

### 3.2 Procrustes / CCA / SVCCA ⚫
**Measures.** Linear alignability of representations across models.
**Methods.** Orthogonal Procrustes; canonical correlation; SVCCA (Raghu 2017).
**Derivation basis.** Classical multivariate statistics.
**Prior signal.** `llm-rosetta-stone/` found that Procrustes alignment across architectures is weak (p=0.82) — captures text statistics not capabilities. A **negative** result that is itself informative: it bounds where linear alignment works.
**Expected level.** Level-3 scope limitation; useful for falsifying "linear cross-arch structure exists."

### 3.3 RSA (representational similarity analysis) ⚫
**Measures.** Pairwise similarity of stimulus representations, compared across systems.
**Methods.** Compute RDM (representation dissimilarity matrix); correlate across systems.
**Derivation basis.** Kriegeskorte 2008 (neuroscience origin); established ML extension.
**Agnosticism expectation.** Highest of all similarity primitives — only requires matched inputs and any embedding function. Canonical bridge to biology.
**Priority.** First primitive to port to Allen Neuropixels validation.

---

## 4. Decomposition primitives

### 4.1 Sparse autoencoders (SAE) ⚫
**Measures.** Decomposes activations into monosemantic features.
**Methods.** Train SAE on layer activations; analyze feature dictionary.
**Derivation basis.** Bricken et al. 2023; Templeton et al. 2024 (Anthropic).
**Agnosticism expectation.** The method is activation-agnostic, but interpretation is modality-specific. Features from an LLM SAE are words; features from a diffusion SAE are image structures; features from a world-model SAE are action-state abstractions.
**Atlas role.** Cross-class structural comparison of SAE feature geometry — *not* the features themselves — is the genome-level question.

### 4.2 PCA/SVD spectral analysis 🟡
**Measures.** Principal directions of variance in activations.
**Methods.** SVD of activation matrices layer-by-layer.
**Derivation basis.** Classical.
**Agnosticism expectation.** Trivial to apply anywhere; already promoted by common practice. Main question is what invariants to extract (power-law slope, spectral gap, rank).
**Candidate universality level.** Spectral decay exponent is a strong Level-2 family constant candidate.

---

## 5. Causal / intervention primitives

### 5.1 Activation ablation ⚫
**Measures.** Behavioral consequences of zeroing or replacing activations.
**Methods.** Zero-ablate / mean-ablate / resample-ablate at specified layer, component, or direction. Measure loss or task-accuracy delta.
**Derivation basis.** Standard.
**Agnosticism expectation.** Applies to any forward pass.
**Atlas role.** Every Level-1 claim gets an ablation test (per CLAUDE.md §4.4).

### 5.2 Activation patching / path patching ⚫
**Measures.** Causal path of information through a network.
**Methods.** Run clean & corrupted inputs; patch activations between them; measure logit delta.
**Derivation basis.** Causal mediation analysis; canonical in Anthropic / Redwood circuits work.
**Agnosticism expectation.** Native to transformers; adaptation needed for SSMs and diffusion.
**Status.** 🟡 in autoregressive models, ⚫ elsewhere. Develop the SSM/diffusion generalization as a research task.

### 5.3 Direction-level steering (CAA, DIM) ⚪
**Measures.** Whether a direction carries a causally effective concept.
**Methods.** CAA (contrastive activation addition); direction-indirect-mediation.
**Prior signal.** `llm-rosetta-stone/` — CAA works on instruction-tuned gemma-3-1b-it (p=0.008, d=0.800); fails on Hybrids (Falcon-H1 collapses) and is underpowered on SSMs. **Diagnostic**, not coordinate.
**Atlas role.** Use as an annotation of "is the model steerable linearly at this layer?"

---

## 6. Probing primitives

### 6.1 Linear probes ⚪
**Measures.** Linear decodability of a concept from activations.
**Status.** Classic but modality-tied. For the atlas, useful as a *diagnostic* of "is concept X linearly represented."
**Caveat.** Linear decodability does not imply linear usability (see `OPEN_MYSTERIES.md` — the reading/writing asymmetry).

### 6.2 MDL probes ⚫
**Measures.** Minimum description length of the probe itself — corrects for probe overfitting.
**Methods.** Voita & Titov 2020 (MDL probing).
**Agnosticism expectation.** Pure information-theoretic — works anywhere.

### 6.3 Non-linear / MLP probes ⚫
**Measures.** Non-linear decodability.
**Atlas role.** Paired with linear probes to test the "concepts on non-linear manifolds" hypothesis (OPEN_MYSTERIES.md §2).

---

## 7. Compression / rate-distortion primitives

### 7.1 Task-conditional compression ⚫
**Measures.** How much activation dimensionality can be compressed before task performance degrades — per task.
**Methods.** PCA truncation + downstream eval. Follows `LLM Genome Project` prior work (the "99.7% compression preserves fluency" finding).
**Agnosticism expectation.** Should generalize to any net producing task outputs. Higher-level generalization to JEPA/world models requires proxy "tasks" (e.g., prediction quality).
**Candidate universality level.** Level 1 if the compressibility-vs-task-diversity function has a universal form.

### 7.2 Successive refinement spectrum ⚫
**Measures.** Distortion as a function of rate (bits retained).
**Derivation basis.** Equitz-Cover 1991; direct CTI lineage.
**Atlas role.** Cross-system D(R) curves are a strong Level-1 candidate.

---

## 8. Primitives specific to non-autoregressive systems

These must be developed explicitly — do not assume standard tools port.

### 8.1 Diffusion noise-schedule representations ⚫
**Measures.** How a diffusion model's internal state evolves across noise levels.
**Why it matters.** Diffusion does not have "layers" in the transformer sense; the geometric story is across denoising steps, not depth.

### 8.2 JEPA predictor-encoder alignment ⚫
**Measures.** How the predictor's internal state aligns with the encoder's latent across prediction targets.
**Why it matters.** JEPAs are natively dual-network; single-stream primitives miss the geometry.

### 8.3 World-model latent-rollout geometry ⚫
**Measures.** Trajectories in the latent state space over rollouts.
**Why it matters.** Dreamer-style world models encode dynamics; static activation analysis misses dynamics.

---

## 9. The agnosticism gate

A primitive is promoted from 🟡 to 🟢 only when the Cross-System Auditor (Codex persona, CLAUDE.md §7.4) certifies that:

1. The primitive has been run on ≥3 distinct system classes.
2. The output is interpretable at the same semantic level across classes (not just numerically computable).
3. A negative result in any class is explainable by a principled reason, not "the tool doesn't work there."
4. At least one potential universality claim has been formulated using it.

---

## 10. What belongs here vs. what belongs elsewhere

| Belongs in this file | Belongs in `SYSTEM_BESTIARY.md` | Belongs in `OPEN_MYSTERIES.md` | Belongs in ledger |
|---|---|---|---|
| Named measurement method | Specific model or biological dataset | Unresolved cross-system phenomenon | One-off experiment record |
| Theoretical basis | Quantization/VRAM notes | Failed mechanism hypotheses | Concrete numbers |
| Agnosticism status | Why each system is in scope | Hypothesis landscape | Results |

Every primitive added here should list which ledger entries first used it. Entries without active usage after 90 days are archived into a "retired primitives" appendix — we do not silently let unused tools bloat this file.
