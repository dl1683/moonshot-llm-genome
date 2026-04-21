# Prereg: `genome_knn_k10_hierarchical` — Gate-2 G2.3 hierarchical model fit

**status: STAGED** (flips to LOCKED after statsmodels/scipy dependency is
pinned in requirements + a fit-runner `code/genome_hierarchical_fit.py`
exists + this prereg's commit SHA is backfilled).

**gate: 2**

**Rationale.** Gate-1 established that kNN-k10 takes statistically-
indistinguishable values across 5 architectures within the Bonferroni-
corrected δ=0.10 equivalence margin. That's a *correlational* claim. The
derivation (`research/derivations/knn_clustering_universality.md` LOCKED
at 62338b8) predicts a *specific functional form*:

    C(X, k) = α_d (1 − β_d · κ(M) · k^(2/d_int))_+ + O(n^(-1/2))

where `α_d`, `β_d` are UNIVERSAL constants depending only on ambient
dimension `d`, and `κ(M)`, `d_int` are per-manifold geometric quantities.
Under Level-1 universality, a SINGLE pooled parameterization (shared
`α_d`, `β_d`) should fit all 5 systems as well as per-system
parameterizations. This prereg specifies the hierarchical model
comparison that tests this — G2.3 per §2.5.2.

---

## 1. Question

Does the derivation-predicted functional form fit the atlas data, AND
does pooling `α_d`, `β_d` across architectures (while per-system `κ`
and `d_int`) outperform per-system-free fits under a hierarchical test?

If yes — kNN-k10 IS Level-1 universal in the functional sense, not just
the equivalence sense. If no — the claim softens to "values coincide
within δ=0.10 but the underlying generator differs per architecture."

---

## 2. Target primitive + derivation pointer

- Primitive: `code/genome_primitives.py::knn_clustering_coefficient` at k=10.
- Derivation pointer: `research/derivations/knn_clustering_universality.md`
  (LOCKED 2026-04-21 at 62338b8; functional form in §4).
- Gate-1 evidence: `results/gate1/stim_resample_n2000_seeds42_123_456_5class.json`
  + `results/gate1/stim_resample_n4000_seeds42_123_456_falcon.json`.

---

## 3. Systems tested (reused from Gate-1 scope)

All 5 Batch-1 Gate-1-passing systems:
- Class 1 Qwen3-0.6B (transformer)
- Class 2 DeepSeek-R1-Distill-Qwen-1.5B (reasoning)
- Class 3 RWKV-4-169M (linear-attention recurrent)
- Class 4 Falcon-H1-0.5B (hybrid transformer+Mamba2)
- Class 6 DINOv2-small (vision ViT)

Batch-2 (encoder/contrastive) systems included once their Gate-1 verdicts
land from the current `run_falcon_then_batch2.sh` pipeline.

---

## 4. Data: Gate-1 measurement grid

`C(X, k=10)` values at k ∈ {5, 10, 20, 30} — requires extending the
current Gate-1 extraction (which measures only k=5 and k=10). This
prereg locks the extended measurement as `knn_clustering_k_sweep` for
G2.3 fitting.

Per system, per sentinel depth (0.25, 0.50, 0.75), per seed:
- Measured: `C(X, k)` for k ∈ {5, 10, 20, 30} with SE per point-cloud.
- Estimated: `d_int(X)` via TwoNN + MLE-k10 (from existing atlas).

Total observations per system: 3 depths × 4 k values × 3 seeds = 36.
Across 5 systems: 180 observations. Sufficient for hierarchical fit of
2 pooled params + 10 per-system params = 12 parameters total.

---

## 5. Hierarchical model specification

**Model H0 (Level-1 universal):**

    log C(X, k) = log α_d + log((1 − β_d · κ_i · k^(2/d_int,i))_+) + ε_{i,k,depth,seed}

with:
- `α_d, β_d` — POOLED, shared across all systems (Level-1 prediction)
- `κ_i, d_int,i` — per-system (each has its own manifold curvature +
  intrinsic-dim inherited from Gate-1)
- `ε` — Gaussian residual with variance determined by analytical SE
  from Gate-1

**Model H1 (per-system):**
- `α_d,i, β_d,i, κ_i, d_int,i` all per-system — 4 × 5 = 20 params.

**Model H2 (arch-family):**
- Pool `α_d, β_d` within {transformer, reasoning, recurrent, hybrid, ViT}
  sub-families (but not across). Intermediate model between H0 and H1.

Fit via maximum-likelihood (statsmodels or scipy.optimize) with
constraints `α_d > 0`, `β_d > 0`, `κ_i ≥ 0`, `d_int,i > 0`.

---

## 6. Pre-registered success criteria (G2.3)

### 6a. Model-selection criterion

Bayesian Information Criterion (BIC) with Δ-BIC rule:
- H0 (pooled universal) is selected iff:
  `BIC(H1) − BIC(H0) > 10` (strong evidence for parsimonious pooled model)
- If `|ΔBIC| ≤ 2`: tie. Not sufficient for Level-1 claim.
- If `BIC(H0) > BIC(H1) + 10`: per-system fits strongly preferred; Level-1
  universal-functional claim falsified. Atlas coordinate stays 🟡 with
  "values-coincide-but-generator-varies" annotation.

### 6b. Residual analysis

- Residuals from H0 fit must be homoscedastic across systems (Levene's
  test p > 0.05) AND approximately normal (Shapiro-Wilk p > 0.01 per
  system bucket).
- If H0 residuals are systematically biased per-system (e.g., Qwen3 always
  over, DINOv2 always under), H0 is a bad fit even if BIC is close.

### 6c. α_d, β_d estimate stability

95% CI on pooled `α_d` must span less than 50% of the point estimate
(i.e., coefficient is well-identified, not a marginal ad-hoc parameter).
Same for `β_d`.

### 6d. Ambient-dimension check

Batch-1 systems have ambient dimensions ranging widely:
- Qwen3-0.6B: h=1024
- DeepSeek-R1-Distill-Qwen-1.5B: h=1536
- RWKV-4-169M: h=768
- Falcon-H1-0.5B: h=1024
- DINOv2-small: h=384

If `α_d, β_d` claim to be "universal constants depending only on d",
this grouping effect must be tested. Formally: refit H0 with `α = f(h)`
for parametric f (e.g., power law in h), compare to constant `α`. If
the fit materially improves, `α` is NOT a universal constant — it's a
power-law in ambient dimension, and the universality claim needs
refinement.

---

## 7. Kill criteria

- **Primary:** H1 (per-system) strongly preferred over H0 (pooled) with
  ΔBIC > 10. Universal-functional claim falsified; demote kNN-k10 from
  Level-1 candidate to "cross-class-consistent but per-class-generated."
- **Secondary:** H0 residuals fail homoscedasticity or normality. The
  functional form doesn't match the data; derivation may need revision
  (though kept locked for audit purposes — a new derivation doc would
  replace it for future claims).
- **Tertiary:** `α_d, β_d` CIs too wide to claim identification. Sample
  size insufficient; plan for n-sweep extension (more k values or more
  seeds per cell).

---

## 8. Expected wall-clock

Extended k-sweep extraction (k ∈ {5,10,20,30}) on the existing 5
systems at n=2000/4000: ~5 min per system per seed. Total: 5 × 3 × 5 =
75 min (reusing existing code). Hierarchical fit via scipy.optimize: <
1 min CPU. **Fits envelope.**

---

## 9. Implementation plan

- Extend `code/genome_primitives.py::knn_clustering_coefficient` caller
  in `genome_cross_arch.py` to sweep k ∈ {5, 10, 20, 30} instead of
  just {5, 10}. Single-line change.
- NEW: `code/genome_hierarchical_fit.py` — loads Gate-1 atlas rows,
  groups by (system, depth, k), fits H0 / H1 / H2 via scipy, reports
  BIC + residual diagnostics. Plot residuals per-system per-depth
  (optional).

---

## 10. Sufficient-statistic storage

Per (system, seed, depth, k) tuple: C value + SE + n_points. Combined
with existing Gate-1 storage: adds ~20 KB. Negligible.

---

## 11. Scope label

Same as locked Gate-1 prereg scope (inherited, not redeclared).

---

## 12. COMPUTE.md §9 compliance

- [x] Max VRAM ≤ 22 GB — re-uses Gate-1 extraction pipeline; no VRAM change.
- [x] Max RAM ≤ 56 GB — fit is CPU-only, < 1 GB.
- [x] Wall-clock ≤ 4 h — 75 min extraction + 1 min fit.
- [x] Disk ~20 KB extra.
- [x] Quant FP16 baseline (same as Gate-1).
- [ ] Smoke test verified: fit H0 on a subset of current Gate-1 data to
  confirm scipy.optimize converges. PRE-LOCK BLOCKER.

---

## 13. Kill criteria linkage to other preregs

- G2.4 causal (`genome_knn_k10_causal_2026-04-21.md`) — independent test.
  Both must pass for Level-1 claim. G2.3 failure alone does not
  invalidate G2.4 or Gate-1.
- G2.5 biology — cannot fire until G2.3 locks the parameter estimates
  against which biological data is compared.

---

## 14. Sign-off

**Status:** STAGED. Flips to LOCKED after:
1. Extended k-sweep extraction lands (k ∈ {5, 10, 20, 30} all 5 systems).
2. `code/genome_hierarchical_fit.py` exists, smoke-test passes.
3. Validator exits 0 in gate:2 mode.
4. This commit's SHA backfilled.

Post-lock modification invalidates G2.3 verdict. Commit-message at lock:
`Lock prereg genome_knn_k10_hierarchical_2026-04-21 — first Gate-2 G2.3 attempt`.

**Locked at commit:** `<pending — backfill post-lock>`.
