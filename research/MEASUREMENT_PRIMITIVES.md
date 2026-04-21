# MEASUREMENT PRIMITIVES

*The toolkit for mapping representational geometry. Each primitive is labeled either **coordinate** (has passed the architecture-agnosticism gate — works on ≥3 system classes) or **diagnostic** (useful but architecture-specific). Coordinates enter the atlas. Diagnostics are annotations only.*

This file is a living catalog. Add a primitive only after defining:
- what it measures,
- what its derivation basis is (or "empirical only"),
- which systems it currently works on,
- its status (coordinate / diagnostic / untested),
- what its candidate universality level is.

---

## Status legend (four-tier per `research/atlas_tl_session.md` §2.5)

- 🟢¹ **Level-1 universal coordinate** — Gate 2 passed: ≥5 classes with derivation-first functional form, causal test, biology instantiation (§2.5.2 G2.1-G2.5).
- 🟢² **Level-2 family-local coordinate** — Gate 1 passed on ≥5 classes AND joint fit shows family-specific constants (no single universal form).
- 🟡 **Coordinate / Gate-1 portability passed** — ≥3 classes pass G1.1-G1.5 + negative control (§2.5.1). No universality claim.
- ⚪ **Diagnostic (Level-0)** — class-local, or fails semantic comparability, or fails G1.x on some classes. Useful as annotation.
- ⚫ **Untested** — proposed but not yet run.

Promotion semantics are locked in `research/atlas_tl_session.md §2.5` (two-gate spec + pre-reg template §2.5.5). This file is the catalog; promotion verdicts come from pre-registered probe results logged in `experiments/ledger.jsonl`.

System classes for agnosticism (from `SYSTEM_BESTIARY.md`): autoregressive LLM · reasoning · SSM · hybrid · diffusion · vision encoder · JEPA · world model · biological recording · untrained network.

---

## 1. Information-geometric primitives

### 1.1 Intrinsic dimension (ID) ⚫
**Measures.** Effective dimensionality of the manifold on which activations lie.
**Methods.** TwoNN (Facco et al. 2017), MLE (Levina & Bickel 2004), persistence-based ID. **For Gate-1 G1.4 (estimator-variant stability) TwoNN + MLE are the pre-registered pair.**
**Derivation basis.** Well-founded in statistics; distribution-free estimators exist.
**Agnosticism expectation.** Should work on any activation vector. High prior for promotion to coordinate.
**Candidate universality level.** Level 1 or 2. CTI-adjacent: prior work in `LLM exploration/` found SSM intrinsic dim ≈ 5.3× Transformer at comparable scale — a candidate Level-2 family constant.
**Biology instantiation (§2.5 G2.5 required declaration).** Given neural population activity `N_neurons × T_timepoints` under stimulus `s`, compute ID on columns (time-slices as points in N-dim neural space) via TwoNN. Directly applicable to Allen Neuropixels and fMRI BOLD series.
**Active prereg.** `research/atlas_tl_session.md §3.7 (strawman; genome_id_portability_2026-04-20, not yet locked)`
**Script stub.** `code/genome_intrinsic_dim.py`

### 1.2 Participation ratio (PR) ⚫
**Measures.** Effective number of active dimensions in activations.
**Methods.** `PR = (Σ λ_i)² / Σ λ_i²` from activation covariance eigenvalues. **For Gate-1 G1.4 the centered vs uncentered PR forms are the pre-registered estimator pair.**
**Derivation basis.** Canonical in random matrix theory and neuroscience (Gao & Ganguli 2015 for neural recordings).
**Agnosticism expectation.** High — pure covariance measure.
**Candidate universality level.** Level 1 for scaling with depth/width; Level 2 family constants.
**Biology instantiation (§2.5 G2.5 required declaration).** Directly from Gao & Ganguli 2015: PR on the neuron-neuron covariance matrix of population activity under fixed stimulus. Directly applicable to Allen Neuropixels.
**Active prereg.** P1.2 under `research/atlas_tl_session.md §3c`, batched with ID.
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

### 2.4 Koopman spectrum ⚫
**Measures.** Spectral signature of the (approximately linear) operator mapping observables forward along the layer/time/step axis of a trajectory.
**Methods.** Dynamic-mode decomposition (DMD, Schmid 2010) on per-input layer/time/step matrices; aggregate eigenvalue ECDFs across the stimulus bank; pseudo-resolvent Koopman for spectral-pollution control (2026 lit). Observable family must be pre-registered.
**Derivation basis.** Koopman 1931 — linear operator theory on observables of a dynamical system. Natural invariant under time-reparameterization when observables are chosen correctly.
**Agnosticism expectation.** Very high. Published applications in 2025-2026 cover transformers, Mamba SSMs, and diffusion simultaneously (Koopman-enhanced transformers; Residual Koopman Spectral Profiling for Mamba; Hierarchical Koopman Diffusion; Koopman-Wasserstein generative). Strongest cross-class candidate per `atlas_tl_session.md §B8b` and H11.
**Candidate universality level.** Level 1 candidate. Eigenvalue ECDF shape (log-log slope) is the candidate universal; scales per-system are the Level-2 constants.
**Biology instantiation (§2.5 G2.5).** Given `N_neurons × T_timepoints`, apply DMD on the time axis to estimate the Koopman operator on population activity. Eigenvalue ECDF is directly comparable to model Koopman spectra. Standard in computational neuroscience dynamical-systems analyses.
**Risk.** Estimator-choice sensitivity is severe — practical choices (observable family, Hankel depth, rank truncation) dominate the spectrum. Pre-reg must LOCK observable family, estimator rank, and DMD variant before any run.
**Active prereg.** Deferred to Batch 2 per `atlas_tl_session.md §3e`.

### 2.5 Local-connectivity statistics ⚫
**Measures.** Per-point stability of the local neighborhood (kNN set) under resampling — captures "how locally-stable is the representation manifold" independent of global structure.
**Methods.** Candidate specs under consideration:
  - **Mean kNN Jaccard self-stability.** Resample the point cloud, measure Jaccard overlap of each point's k-neighbor set across resamples; average across points. Cheap (kNN in O(n log n)).
  - **kNN-graph clustering coefficient.** Per-node fraction of neighbors-of-neighbors that are also direct neighbors.
  - **Diffusion entropy at step t.** Entropy of random-walk distribution on the kNN graph after t steps.
**Derivation basis.** Local-manifold learning (Neighborhood Preserving Embedding He et al. 2005; NNGS, Fornasier et al. 2024 as cross-system variant). Codex Round 1 Intuition 2 (medium-high conviction): global similarity collapses under scale correction, only local neighborhood structure survives cross-architecture.
**Agnosticism expectation.** High — pure graph-theoretic properties of point-cloud neighborhood structure. Works on any point cloud.
**Candidate universality level.** Level-1 candidate. Intuitively: local neighborhood preservation = intrinsic manifold property, architecture-agnostic.
**Distinction from similarity primitives.** NNGS (Jaccard of kNN graphs between TWO embeddings) is a cross-system diagnostic (Level-0) — it measures how similar two systems are, not a per-system coordinate. Local-connectivity statistics are per-system.
**Active prereg.** Deferred; Codex Round 2 to rule whether it joins Batch 1 as a 4th primitive.

---

## 3. Representation-similarity primitives

### 3.1 Centered kernel alignment (CKA) ⚪ (DEMOTED — per Round 1)
**Status note (2026-04-20):** Demoted from 🟡 to ⚪ diagnostic. Round 1 Codex review: CKA is scale-confounded (Feb 2026 "Aristotelian View" paper shows apparent convergence largely disappears under scale correction). PC-dominance makes it sensitive to leading principal components only. Do NOT treat as coordinate. Use at most as a cross-check alongside local-neighborhood primitives.
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

## 9. The agnosticism gate — two-gate spec

**LOCKED IN `research/atlas_tl_session.md §2.5`** (produced as Round-1 priority-directive deliverable). This section now cross-references the locked spec.

**Gate 1 — PORTABILITY (promotion from ⚫/⚪ to 🟡).** Five criteria, all with pre-registered tolerances:
- G1.1 Computability within COMPUTE.md envelope
- G1.2 Invariance under the primitive's declared invariance group
- G1.3 Stability under stimulus resampling (H12)
- G1.4 Stability under estimator variant
- G1.5 Stability under quantization ladder (H13)
Plus the negative-control rule (primitive must distinguish trained vs untrained or it's Level-0).

**Promotion threshold:** Gate 1 passed on ≥3 distinct system classes → 🟡 (coordinate, portability gate passed). **No universality claim.**

**Gate 2 — UNIVERSALITY (promotion from 🟡 to 🟢¹ or 🟢²).** Five additional criteria:
- G2.1 Gate-1 portability on ≥5 classes (Level-1 threshold per `UNIVERSALITY_LEVELS.md`)
- G2.2 Derivation-first functional form from first principles (LOCKED before fitting)
- G2.3 Joint-fit residual < α_universal × per-class independent fit residual
- G2.4 Causal test: ablation of coordinate-defined subspace produces predicted behavior change
- G2.5 Biology instantiation specified (may be deferred in execution, NOT in declaration)

**Promotion threshold:** all five → 🟢¹ (Level-1 universal). If G2.2/G2.3 show family-local constants → 🟢² (Level-2 family-local). If G2.2 derivation-first fails → remains 🟡 as "Phase-2 atlas observation" pending derivation.

See `atlas_tl_session.md §2.5.5` for the LOCKED-at-commit pre-registration template.

---

## 10. What belongs here vs. what belongs elsewhere

| Belongs in this file | Belongs in `SYSTEM_BESTIARY.md` | Belongs in `OPEN_MYSTERIES.md` | Belongs in ledger |
|---|---|---|---|
| Named measurement method | Specific model or biological dataset | Unresolved cross-system phenomenon | One-off experiment record |
| Theoretical basis | Quantization/VRAM notes | Failed mechanism hypotheses | Concrete numbers |
| Agnosticism status | Why each system is in scope | Hypothesis landscape | Results |

Every primitive added here should list which ledger entries first used it. Entries without active usage after 90 days are archived into a "retired primitives" appendix — we do not silently let unused tools bloat this file.
