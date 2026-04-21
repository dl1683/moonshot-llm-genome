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

Promotion semantics are locked in `research/atlas_tl_session.md §2.5` (two-gate spec + pre-reg template §2.5.9). This file is the catalog; promotion verdicts come from pre-registered probe results logged in `experiments/ledger.jsonl`.

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

### 2.5 kNN-5 clustering coefficient ⚫ (Batch-1 P1.3; single-cloud local-neighborhood coordinate)
**Measures.** Per-point local manifold density — for each point, the fraction of pairs among its k-nearest neighbors that are themselves neighbors of each other in the kNN graph. Averaged over all points in the cloud to produce a single scalar `C(X)`.
**Methods.** Build kNN graph (k=5, Euclidean, with weighted and unweighted estimator variants for Gate-1 G1.4); compute `C(i) = (# edges among kNN_5(i)) / C(k, 2)` per point; average. `C(X) = mean_i C(i)`.
**Derivation basis.** Graph theory; local-manifold learning lineage (NPE He et al. 2005; local-neighborhood invariants). Addresses Codex Round 1 Intuition 2 (medium-high): "global similarity collapses, only local neighborhood survives cross-architecture."
**Invariance group G_f.** Orthogonal rotations + global isotropic rescaling (kNN sets are rotation/scale-invariant). NOT invariant to non-isotropic scaling.
**Agnosticism expectation.** High — pure graph-theoretic property of a point cloud. Works on any point cloud with a defined distance.
**Candidate universality level.** Level-1 candidate pending Gate-1 pass + subsequent Gate-2 derivation.
**Analytical SE.** Under independence of per-point clustering values (approximation), `SE(C(X)) = std(C(i)) / √n`. O(1/n). No bootstrap needed.
**Distinction from related methods.**
- **kNN Jaccard self-stability across resampled clouds** (earlier P1.3 draft): conflated coordinate with stability diagnostic; retired per Codex Round 3 NEW kill shot #3.
- **NNGS (cross-system Jaccard)**: Level-0 diagnostic, not a per-system coordinate. See §3.2 below.
**Biology instantiation (§2.5 G2.5 required declaration).** On neural population data: build kNN graph on stimulus-indexed response vectors `x_i ∈ R^{N_neurons}` (one point per stimulus condition, trial-averaged). Compute mean clustering coefficient over stimuli. Standard application of graph-theoretic neuroscience tools.
**Active prereg.** P1.3 within Batch 1 under `atlas_tl_session.md §3c`.

### 2.6 Local-neighborhood alternative specs ⚫ (Batch-2 options if P1.3 fails Gate 1)
Codex Round 3 flagged cleaner alternatives. Kept as fallbacks:
  - **Local reachability density (LOF-style).** Per-point density of the local neighborhood; sensitive to outliers/boundary points.
  - **Heat-kernel trace / diffusion entropy at fixed t.** `tr(exp(-tL))` or entropy of the random-walk distribution after t steps on the kNN-graph Laplacian. Single-cloud scalar; t pre-registered.

If P1.3 clustering-coefficient fails Gate 1 on any system in Batch 1, these are the first-backup primitives before falling back to heavier Koopman/Ricci/PH probes in Batch 2.

---

## 3. Representation-similarity primitives

### 3.1 Centered kernel alignment (CKA) ⚪ (DEMOTED — per Round 1; revival path closed per R5)
**Status note (2026-04-21):** Demoted from 🟡 to ⚪ diagnostic. Round 1 Codex review: CKA is scale-confounded (Feb 2026 "Aristotelian View" paper shows apparent convergence largely disappears under scale correction). PC-dominance makes it sensitive to leading principal components only. **Not a coordinate candidate.** Codex R5 adversarial audit flagged the prior "Promotion test" paragraph as a silent revival vector — removed.
**Measures.** Similarity between two representation spaces.
**Methods.** Linear or RBF-kernel CKA (Kornblith et al. 2019).
**Derivation basis.** Well-established.
**Agnosticism expectation.** Cross-system by construction. Current state: widely used on LLMs + vision; untested on diffusion/JEPA/world models at scale.
**Role in atlas.** Level-0 cross-system diagnostic only — may appear in figures or cross-references but NEVER as a Gate-1 coordinate. If promoted back, it requires a fresh design round addressing the scale-confound in its definition (e.g., local-subspace CKA per ICLR 2025) — not the current formulation.

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

**Gate 1 — PORTABILITY (promotion from ⚫/⚪ to 🟡).** Seven criteria, all under the §2.5.6 equivalence/precision criterion `|Δ| + c·SE(Δ) < δ` (Bonferroni-corrected):
- G1.1 Computability within COMPUTE.md envelope (declarative)
- G1.2 Invariance under declared invariance group (tested via random transformation + equivalence check)
- G1.3 Stimulus-resample stability (H12) — 3 seed-disjoint resamples, pairwise equivalence
- G1.4 Estimator-variant stability — two estimators (e.g., TwoNN/MLE, weighted/unweighted clustering) agree within δ
- G1.5 Quantization stability (H13) — FP16 vs Q8 agree within δ
- G1.6 Subsample asymptote (H14) — n-sweep, slope within 1 SE of zero at n_max/2 vs n_max
- G1.7 Preprocessing / metric declaration (primitive identity)
Plus the negative-control rule (trained vs untrained differ by ≥ δ_neg-control, i.e., primitive measures learned geometry, not just architecture).

K enumeration (Batch-1 prereg §3.7 example): 3 systems × 6 decisions (G1.2..G1.6 + negative-control aggregated per-system) = K=18. `c = z_{1 − 0.05/18} ≈ 2.77` one-sided. Equivalence margin δ_relative = 0.10 default, with mandatory sensitivity sweep at δ ∈ {0.05, 0.10, 0.20}.

**Promotion threshold:** Gate 1 passed on ≥3 distinct system classes → 🟡 (coordinate, portability gate passed). **No universality claim.**

**Gate 2 — UNIVERSALITY (promotion from 🟡 to 🟢¹ or 🟢²).** Five additional criteria:
- G2.1 Gate-1 portability on ≥5 classes (Level-1 threshold per `UNIVERSALITY_LEVELS.md`)
- G2.2 Derivation-first functional form from first principles (LOCKED before fitting)
- G2.3 Joint-fit residual < α_universal × per-class independent fit residual
- G2.4 Causal test: ablation of coordinate-defined subspace produces predicted behavior change
- G2.5 Biology instantiation specified (may be deferred in execution, NOT in declaration)

**Promotion threshold:** all five → 🟢¹ (Level-1 universal). If G2.2/G2.3 show family-local constants → 🟢² (Level-2 family-local). If G2.2 derivation-first fails → remains 🟡 as "Phase-2 atlas observation" pending derivation.

See `atlas_tl_session.md §2.5.9` for the LOCKED-at-commit pre-registration template.

---

## 10. What belongs here vs. what belongs elsewhere

| Belongs in this file | Belongs in `SYSTEM_BESTIARY.md` | Belongs in `OPEN_MYSTERIES.md` | Belongs in ledger |
|---|---|---|---|
| Named measurement method | Specific model or biological dataset | Unresolved cross-system phenomenon | One-off experiment record |
| Theoretical basis | Quantization/VRAM notes | Failed mechanism hypotheses | Concrete numbers |
| Agnosticism status | Why each system is in scope | Hypothesis landscape | Results |

Every primitive added here should list which ledger entries first used it. Entries without active usage after 90 days are archived into a "retired primitives" appendix — we do not silently let unused tools bloat this file.
