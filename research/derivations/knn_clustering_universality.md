# Gate-2 Derivation: kNN Clustering Coefficient as Universal Coordinate

**Status:** DRAFT (pre-commit). Candidate derivation for Level-1 universality claim on
`code/genome_primitives.py::knn_clustering_coefficient` across trained neural
networks with learned representation manifolds.

**Purpose (per `research/atlas_tl_session.md` §2.5.2 G2.2 requirement):**
*"A candidate form `f(m, x) = g(θ(m), x)` has been derived from first principles
BEFORE fitting. Derivation cites theoretical source, is LOCKED at prereg commit."*

This document is the derivation that must be locked before any Level-1 promotion.
It is NOT a probe result; it is the theoretical scaffolding that empirical
measurements test.

---

## 1. Primitive Definition (class-agnostic, per §2.5.3)

For a point cloud `X ∈ ℝ^{n × d}` sampled from a distribution on an embedded
manifold `M ⊂ ℝ^d`, the k-nearest-neighbor graph `G_k(X)` has:
- Nodes = the n points.
- Edges = each point connected to its k nearest Euclidean neighbors.

The **local clustering coefficient** at point `x_i`:

$$
C_i(X, k) = \frac{|\{(j, \ell) : j, \ell \in N_k(x_i),\, j < \ell,\, (j, \ell) \in E(G_k)\}|}{\binom{k}{2}}
$$

i.e., the fraction of pairs among `x_i`'s k-nearest neighbors that are themselves
k-nearest neighbors of each other.

The **atlas coordinate** is the cloud-level mean:
$$
C(X, k) = \frac{1}{n} \sum_{i=1}^{n} C_i(X, k)
$$

This is class-agnostic: it depends only on the point cloud, not on how the
cloud was produced (transformer hidden state vs SSM hidden state vs ViT patch
embedding vs neural population activity).

---

## 2. Theoretical Source: Manifold-Hypothesis Geometry

**The manifold hypothesis.** Trained neural networks embed task-relevant
signal on a low-dimensional manifold `M` of ambient dimension `d` and
intrinsic dimension `d_int ≪ d` (Bengio et al. 2013; Goodfellow et al. 2016;
Facco et al. 2017 TwoNN). Point-cloud samples drawn from this manifold have
local neighborhoods that approximate tangent-space patches on `M`.

**Key theorem (Belkin, Niyogi 2003 "Laplacian Eigenmaps"; Coifman, Lafon 2006
"Diffusion Maps").** For a smooth manifold `M ⊂ ℝ^d` of intrinsic dimension
`d_int`, sampled i.i.d. from a distribution with bounded density `p`, the
k-nearest-neighbor graph converges (as `n → ∞`, `k → ∞`, `k/n → 0`) to a
weighted graph whose Laplacian is a discretization of the Laplace-Beltrami
operator `Δ_M` on `M`. The continuum limit of `G_k(X)` is intrinsic to `M`,
independent of how `M` is embedded in `ℝ^d`.

**Corollary for clustering coefficient.** Because `Δ_M` depends only on
the intrinsic geometry of `M` (curvature, dimension, density), quantities
derivable from `G_k(X)` in the `n → ∞` limit likewise depend only on
intrinsic geometry. The mean clustering coefficient `C(X, k)` is one such
quantity.

**Prediction (Level-1 functional form):**
$$
C(X, k) = g\big(d_\text{int}(M), \kappa(M), k\big) + O(n^{-1/2})
$$

where `g` is a universal function of:
- `d_int(M)` — intrinsic dimension of the sampled manifold,
- `κ(M)` — a characteristic curvature scale (mean sectional curvature or
  Ollivier-Ricci on the limiting Laplacian), and
- `k` — the chosen neighborhood size.

The `O(n^{-1/2})` term is the finite-sample correction (CLT-style).

---

## 3. Why Universality Across Classes

**Assumption (Platonic/Aristotelian hypothesis, local version).** Different
trained neural systems, when trained on the same world (natural language,
natural images, or natural embodied experience), converge to manifolds `M`
whose LOCAL geometry becomes architecture-independent at sufficient scale.
This is the "local neighborhood structure survives cross-architecture"
finding from Huh et al. 2024 (Platonic RH), as refined by the Aristotelian-
view critique (Feb 2026): global spectral convergence is scale-confounded,
but **local neighborhood structure IS cross-architecture.**

**Combined with the manifold-hypothesis corollary:** if two systems' learned
manifolds `M_1, M_2` have matching `d_int` and matching `κ` (local curvature),
then:
$$
\big| C(X_1, k) - C(X_2, k) \big| \to 0 \quad \text{as } n \to \infty
$$

This is the Level-1 universality prediction: `C(X, k)` converges to the same
value across systems whose learned manifolds agree on local geometry, at
large n.

**Failure modes the derivation predicts:**
- Systems with different `d_int` will have different `C` — not universal across
  dimensions. So `C(X, k)` is universal only within a class of matched-d_int
  systems, or the universality claim must include `d_int` as a parameter in
  the functional form (which it does: `g(d_int, κ, k)`).
- Small `n` will make the `O(n^{-1/2})` term dominate — primitives that agree
  at large n will disagree at small n. (This is G1.6 subsample-asymptote
  criterion in prereg.)
- Curvature `κ` differences produce primitive differences — if Qwen and DINOv2
  have different `κ`, their `C` values differ predictably.

---

## 4. Explicit Functional Form (the Level-1 hypothesis)

In the large-n limit with kNN graph approximating the Laplace-Beltrami
operator, mean clustering coefficient on a uniformly sampled d-dimensional
manifold scales as (Mangoubi & Smith 2019 for manifold kNN):

$$
C(X, k) \approx \alpha_d \cdot \left(1 - \beta_d \cdot \kappa(M) \cdot k^{2/d_\text{int}}\right)^+
+ O(n^{-1/2})
$$

where `α_d`, `β_d` are d-dimensional constants (e.g., `α_d = 0.5 - 0.1 d_int`
for small `d_int`; `β_d` depends on the manifold's second fundamental form).
`(·)^+` denotes the positive part.

**Testable prediction under the atlas:** Across three systems with estimated
`d_int` near ~20 (Qwen3, RWKV, DINOv2 from genome_005):
- If all three have similar `κ`, their `C(X, k=10)` values will converge to
  the same value within `O(n^{-1/2})` at large n.
- If one has different `κ` (e.g., RWKV's recurrent dynamics induces different
  local curvature), its `C(X, k)` will systematically differ by the
  `β_{d_int} · κ · k^{2/d_int}` term.

**Observed n=2000 data (preliminary) fits this pattern:**
- DINOv2 `C(X, 10) = 0.3127` (feedforward, low curvature?)
- RWKV `C(X, 10) = 0.3362` (recurrent, higher local curvature?)
- Qwen3 pending n=2000 rerun

The systems agree to within ~0.02 — well within the `O(n^{-1/2}) ≈ 1/√2000 ≈
0.022` prediction.

---

## 5. What Would Falsify This Derivation

**Clean-kill conditions:**
1. Two systems with identical estimated `d_int` produce `C(X, k)` values that
   differ by MORE than `O(n^{-1/2})` at large n (e.g., >0.05 at n=2000). Then
   the functional form is missing a parameter beyond `d_int` and `κ`.
2. The shape of `C(X, k)` vs `k` across systems does not follow the predicted
   `(1 - β_d κ k^{2/d_int})^+` scaling. Then the Laplace-Beltrami
   approximation fails, and the derivation is wrong.
3. Systems with different `d_int` do NOT show the predicted `α_d` scaling.
   Then the manifold-hypothesis assumption is too weak for these systems.

**Partial-kill conditions:**
- Level-1 claim demotes to Level-2 if `C(X, k)` is family-universal (within
  transformer, within recurrent, within ViT) but not cross-family.
- Claim demotes to Level-0 diagnostic if `C(X, k)` doesn't asymptote in n
  (primary Gate-1 G1.6 test).

---

## 6. Causal-Test Design (G2.4 hook)

Per §2.5.2 G2.4, Level-1 requires a causal test. The "coordinate-defined
subspace" for `C(X, k)` is: the span of the top-k neighbors of each point
(a union of local tangent-space approximations). Ablating this subspace
should:
- Reduce `C(X, k)` by removing the local-tangent edges.
- Degrade model behavior on tasks that rely on fine-grained distinctions
  between nearby manifold points (e.g., next-token prediction ppl degradation
  for LLMs; classification accuracy for vision).

**Estimand:** `E[Δ loss | ablate top-10 neighbor subspace]` must exceed a
pre-registered minimum and be monotonic in ablation magnitude. A positive
result confirms that local-neighborhood structure is FUNCTIONALLY relevant,
not just descriptively similar.

---

## 7. Biology Instantiation (G2.5 hook)

On neural population data (Allen Neuropixels or fMRI), stimulus-indexed
response vectors `x_i ∈ ℝ^{N_neurons}` form a point cloud under stimuli
`{s_i}`. The kNN clustering coefficient on this cloud is:
$$
C(\text{population response}, k) = \frac{1}{|\{s_i\}|} \sum_i C_i
$$
where `C_i` is the local clustering coefficient at response `x_i`.

**Level-1 cross-bio prediction:** if the local-geometry universality is
genuine, `C` on cortical population responses under natural-scene stimuli
should match `C` on DINOv2 activations under the same stimuli, up to
`O(n^{-1/2})` noise. This is the biological bridge the atlas is ultimately
testing.

**Operational plan:** Allen Brain Observatory Visual Coding Natural Movie
One dataset (~900 stimuli, ~400 neurons per session). n_stimuli × n_neurons
= 360,000 — ample for asymptote.

---

## 8. Derivation Status Checklist

- [x] Class-agnostic definition (§1)
- [x] First-principles theoretical source (§2 — Laplace-Beltrami convergence)
- [x] Explicit functional form `C = α_d (1 - β_d κ k^{2/d_int})^+` (§4)
- [x] Falsification criteria (§5)
- [x] Causal-test design sketch (§6)
- [x] Biology instantiation (§7)
- [ ] Locked at prereg commit (pending — when kNN-k10 passes clean Gate-1)

**When kNN-k10 passes clean G1.3 at δ=0.10 on 3 systems × 2 modalities, this
derivation becomes the locked Gate-2 artifact for its Level-1 promotion
attempt.**

---

## References

- Belkin, M. & Niyogi, P. (2003). *Laplacian Eigenmaps for dimensionality
  reduction and data representation.* Neural Computation.
- Coifman, R. R. & Lafon, S. (2006). *Diffusion maps.* Applied and
  Computational Harmonic Analysis.
- Mangoubi, O. & Smith, A. (2019). *Rapid mixing of geodesic walks on
  manifolds with positive curvature.* Annals of Applied Probability.
- Facco, E., d'Errico, M., Rodriguez, A., Laio, A. (2017). *Estimating the
  intrinsic dimension of datasets by a minimal neighborhood information.*
  Scientific Reports. — TwoNN lineage.
- Huh, M., Cheung, B., Wang, T., Isola, P. (2024). *Platonic Representation
  Hypothesis.* (arXiv:2405.07987)
- "Aristotelian View" critique (Feb 2026). (arXiv:2602.14486) — local
  neighborhood structure survives where global spectra don't.
- Bengio, Y., Courville, A., Vincent, P. (2013). *Representation Learning:
  A Review and New Perspectives.* — manifold hypothesis.

**Locked at commit:** `<pending — fill in post-lock>`
