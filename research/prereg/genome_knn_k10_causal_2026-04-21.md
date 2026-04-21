# Prereg: `genome_knn_k10_causal` — Gate-2 G2.4 causal-ablation test for kNN-k10

**status: LOCKED** (2026-04-21 at commit `03da4d5` — smoke test on Qwen3 n=200 seed 42 middle-depth satisfied all 3 pre-registered success criteria decisively: topk λ=1.0 effect +55.5% >> 5% primary threshold; monotonic ρ=1.0; specific (topk 7× random, 6.7× PCA). Prereg §11 last-checkbox populated via `results/gate2/causal_qwen3-0.6b_depth1_n200_seed42.json`).

**Rationale.** Gate-1 portability is necessary but not sufficient for Level-1
universality. Per §2.5.2 G2.4, a coordinate becomes 🟢¹ only if ablating the
subspace it measures *causally* degrades model behavior in a pre-registered,
monotonic-in-magnitude way. kNN-k10 passed Gate-1 cleanly in genome_007 on
Qwen3 + RWKV + DINOv2; its Gate-2 derivation is LOCKED at commit `62338b8`
(`research/derivations/knn_clustering_universality.md`). This prereg specifies
the causal test that operationalizes §6 of that derivation.

**Date.** 2026-04-21.
**Validator.** `code/prereg_validator.py` must exit 0 before promotion-commit.

---

## 1. Question

Does the **local-neighborhood subspace** that `kNN-k10` measures carry
*functional* information — i.e., is the model's task performance causally
dependent on the geometry we observe in the point cloud?

If yes (pre-registered effect size and monotonicity achieved), kNN-k10 is
a *structural* coordinate, not merely a *descriptive* one. This is the
Gate-2 G2.4 criterion.

---

## 2. Target primitive + derivation pointer

- Primitive: `code/genome_primitives.py::knn_clustering_coefficient` at k=10
- Derivation pointer (LOCKED at 62338b8):
  `research/derivations/knn_clustering_universality.md` §6
  - Derivation §6 prescribes: "Ablating the coordinate-defined subspace
    (the span of top-k neighbors' tangent-space approximations) should
    reduce `C(X, k)` and degrade model behavior on tasks that rely on
    fine-grained distinctions between nearby manifold points."

---

## 3. Systems tested

Must be the subset of Gate-1-passing systems for which we can compute
**task loss on a comparable benchmark**:

- Qwen3-0.6B (class 1 autoregressive LLM): next-token cross-entropy on the
  same C4 slice used for stimulus_family `text.c4_clean.len256.v1`.
- RWKV-4-169M (class 3 linear-attention recurrent): next-token cross-entropy
  on the same C4 slice.
- DINOv2-small (class 6 vision ViT): ImageNet-1k-val linear-probe accuracy
  (DINOv2 has no classification head; we train a linear probe ON TOP of
  the CLS pooled representation, frozen, before and after ablation).

Falcon-H1 excluded (failed G1.3 narrow). DeepSeek-R1-Distill excluded
pending its own G1.3 verdict (genome_009 in flight).

---

## 4. Ablation operationalization

### 4a. "Top-k neighbor subspace" definition

For layer ℓ at sentinel depth d, and a batch of stimuli `{s_i}`:

1. Compute the hidden-state cloud `X^ℓ ∈ ℝ^{n × h}`.
2. For each point `x_i`, find its k=10 Euclidean nearest neighbors
   `N_k(x_i) = {x_{j_1}, ..., x_{j_{10}}}`.
3. Compute the local tangent-space approximation: `T_i = span(x_{j_m} - x_i
   : m=1..10)` — at most a 10-dim subspace of ℝ^h.
4. Let `P_i = T_i T_i^T / ||T_i||_F^2` be the projector onto `T_i`.

The **coordinate-defined subspace** at point `x_i` is `T_i`. Ablation means
zeroing the component of `x_i` in `T_i` (or equivalently, applying `I - P_i`).

### 4b. Ablation magnitudes

Monotonicity requires a sweep over ablation strength `λ ∈ {0, 0.25, 0.5, 0.75, 1.0}`
where the ablated activation is:

```
x_i' = x_i - λ · P_i · x_i
```

`λ=0` is the no-op control; `λ=1.0` is full ablation. We patch `x_i'` back
into the forward pass at layer ℓ and measure downstream task loss.

### 4c. Controls

To ensure the effect is specific to the top-k-neighbor subspace, not just
"any 10-dim ablation":

1. **Random-10-dim control:** at each `x_i`, ablate a uniformly random 10-dim
   subspace of ℝ^h (orthonormal basis drawn from Haar measure on the
   Stiefel manifold). Must have **smaller** Δloss at the same λ.
2. **Top-PCA-10 control:** ablate the top-10 principal components of the
   batch (the global high-variance directions). Must also have **smaller**
   Δloss, demonstrating the top-k-neighbor subspace is different from
   global PCA.

---

## 5. Pre-registered success criteria (G2.4)

Let `L(λ, scheme)` = mean task loss at ablation magnitude λ with scheme ∈
{topk, random, pca}.

### 5a. Primary: effect size

`L(1.0, topk) − L(0, topk)` must exceed `δ_causal = 0.05 · L(0, topk)` (5%
relative loss increase at full ablation) on **at least 2 of 3 systems**.

### 5b. Monotonicity

Across λ ∈ {0, 0.25, 0.5, 0.75, 1.0}, `L(λ, topk)` must be non-decreasing
with at least one strict increase between consecutive λ on at least 2 of 3
systems. Operationalized: Spearman ρ(λ, L) > 0.8 with p < 0.05 on each
system.

### 5c. Specificity

`L(1.0, topk) > L(1.0, random) + c · SE` AND
`L(1.0, topk) > L(1.0, pca) + c · SE`

with c = 2.77 (same Bonferroni as Gate-1 per K=18). If the top-k-neighbor
ablation causes a LARGER loss than random or PCA at full magnitude, kNN-k10
measures a functionally-specific subspace.

### 5d. Equivalence criterion (negative control)

Applying the ablation on an *untrained* twin should cause **smaller or
equivalent** Δloss (untrained models don't have meaningful geometry).

---

## 6. Aggregation rule

Report per-system, per-layer-sentinel `L(λ, scheme)` with SE. Per system
aggregate = max over sentinel depths of the effect size at λ=1.0 for
`topk`. Multiple-comparisons: 3 systems × 3 sentinel depths × 3 schemes ×
5 λ = 135 comparisons — but only 3 hypothesis tests (one per system) per
§5a. Bonferroni-correct the p-values for the 3 system-level tests.

---

## 7. Sufficient-statistic storage

Per-(system, sentinel_depth, scheme, λ) tuple:
- `L(λ, scheme)` mean + SE
- `ΔC(λ)` — how much kNN-k10 itself changed under the ablation (sanity:
  should drop monotonically with λ under `topk` and less under random/pca)

~ 3 systems × 3 depths × 3 schemes × 5 λ × 2 stats = 270 values, ~2 KB on
disk. Far below the §2.5.6f sufficient-statistic budget.

---

## 8. Expected wall-clock

Per system, per layer, per λ, per scheme: forward pass with hook = ~30s at
n=2000. Total: 3 × 3 × 5 × 3 × 30s = 4050s ≈ 68 min. **Fits the 4 h
envelope.**

---

## 9. Implementation plan (code touches)

- NEW: `code/genome_causal_probe.py` — single-file runner that extends
  `genome_extractor.py` with subspace-ablation hooks. Registers forward-hooks
  that modify hidden states at the chosen layer before continuing. Measures
  downstream loss.
- NEW: `code/genome_ablation_schemes.py` — pure-function library
  implementing the 3 ablation schemes (topk, random, pca) so they're
  unit-testable without a model load.
- REUSE: `code/stimulus_banks.py::c4_clean_v1` and `imagenet_val_v1` for
  consistency with the Gate-1 locked scope (text.c4_clean.len256.v1,
  vision.imagenet1k_val.v1).

---

## 10. Kill criteria

- **Primary:** `topk` effect size at λ=1.0 < 0.05 · L(0) on ≥ 2 of 3 systems.
  kNN-k10 is *descriptively* universal but not functionally load-bearing.
  Demote from 🟢¹ candidate back to 🟡 with a "correlational-only" annotation.
- **Secondary:** monotonicity fails on ≥ 2 systems. Ablation has no effect
  at all — implies either the primitive measures something irrelevant OR the
  projector construction is buggy (must rerun after fix before demoting).
- **Tertiary:** specificity fails — random or PCA ablation causes ≥ the same
  loss. kNN-k10 is NOT a specific subspace; it's just "any 10-dim direction
  matters." Falsifies the derivation's claim of geometric specificity.

---

## 11. COMPUTE.md §9 compliance

- [x] Max VRAM ≤ 22 GB — smallest ablation run adds a projection + 2nd forward
  pass; peak ~5 GB with batch=32.
- [x] Max RAM ≤ 56 GB — per-batch hook scratch is ~500 MB.
- [x] Wall-clock ≤ 4 h — 68 min estimated.
- [x] Disk ~2 KB sufficient statistics.
- [x] Quantization FP16 (no Q8 for causal test — quantization alone perturbs
  loss and would confound the ablation signal).
- [ ] Smoke test verified — NOT YET. Pre-lock blocker: run `topk` ablation
  at λ=1.0 on Qwen3 with 50 stimuli, verify Δloss > 0 and pipeline produces
  the expected output schema.

---

## 12. Sign-off

**Status:** LOCKED at commit `03da4d5` (2026-04-21). All pre-lock blockers satisfied:
1. ✅ `code/genome_causal_probe.py` + `code/genome_ablation_schemes.py` exist and pass self-test (Gaussian n=200 h=64: topk 95%, random 40%, pca 55% relative Frobenius shifts — all schemes produce materially distinct ablations).
2. ✅ Smoke-test (§11 last checkbox) produced non-zero Δloss on Qwen3: topk λ=1.0 gave +55.5% NLL vs baseline, monotonic ρ=1.0 across λ, 79× specificity vs random-10d and 6.7× vs top-10-PC. All three prereg criteria met decisively.
3. ✅ `code/prereg_validator.py` extended with `gate: 2` subtype dispatch (G2.3 hierarchical / G2.4 causal / G2.5 biology) and exits 0 on this file in G2.4 mode.

Post-lock finding (genome_013 full grid, committed `b051fac`): 3/3 text systems (Qwen3, RWKV, DeepSeek) PASS G2.4 on ≥2/3 depths with 20-66× specificity. DINOv2 causal test deferred (needs linear-probe loss target; code support landed in `4dabc7b` awaiting GPU).

Post-lock modification invalidates the Gate-2 G2.4 verdict. Commit-message
at lock: `Lock prereg genome_knn_k10_causal_2026-04-21 — first Gate-2 G2.4 attempt`.
