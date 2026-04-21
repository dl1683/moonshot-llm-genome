# Power-Law v2 — derivation notes (scientific follow-up)

**Status:** EXPLORATORY. Not a paper claim. Drafted 2026-04-21 after the locked v1 (Laplace-Beltrami convergence) was falsified at the G2.3 gate. These are candidate frameworks for a first-principles derivation of the observed `C(X, k) = c_0 · k^p` with `p ≈ 0.17 ± 0.022` (CV 12.5%) and `c_0 ≈ 0.21` across 18 (system × depth) cells on 6 trained architectures.

The paper (§4.5) honestly reports the v1 falsification and the empirical power-law replacement without a re-derivation. This note sketches where that re-derivation should start.

---

## What needs to be explained

Empirical regularities holding across 6 architectures × 3 depths at n=2000, 3 stimulus-resample seeds, k-grid `{3, 5, 8, 12, 18, 27, 40, 60, 90, 130}`:

1. **Monotonic increase in k.** `C(X, k)` grows with `k` on every system and every mid-depth sentinel. v1 predicted the opposite sign.
2. **Power-law fit.** `log C = p · log k + log c_0` fits with `R² > 0.994` on every cell (mean 0.9974, min 0.9928).
3. **Narrow p band.** Across the full atlas, `p` takes values in `[0.133, 0.209]`. Coefficient of variation 12.5%, tighter than any previous cross-architecture measurement in this project.
4. **Narrow c_0 band.** `c_0 ≈ 0.21` across systems with comparable tightness.
5. **Above random baseline.** On iid Gaussian clouds matched for (n, h), `C` is 4–7× smaller and shows a different `k`-scaling (near-flat). So the power law is a property of *learned* geometry, not a random-geometry artifact.

Any v2 derivation must predict (1) sign, (2) shape (power law, not some other monotone form), and (3) quantitatively the band `p ∈ [0.13, 0.21]` — without free parameters that could have been fit to match.

---

## Candidate framework A — fractal correlation dimension

If the pooled hidden-state point cloud lies on a set with fractal correlation dimension `d_2` (Grassberger-Procaccia), then:

- Number of neighbor pairs within radius `r` scales as `N_pair(r) ~ r^{d_2}`.
- The `k`-th nearest neighbor distance scales as `r_k ~ k^{1/d_int}`, where `d_int` is the intrinsic (pointwise) dimension.
- For an edge `(i, j)` to contribute to clustering, both `i → j` and a common neighbor `m` with `m ∈ kNN(i)`, `m ∈ kNN(j)` must lie within radius `~ r_k` of `i`.
- In a fractal regime, the probability a given pair of `kNN(i)` points are also mutual neighbors scales like a power of `r_k` with exponent governed by `d_2 / d_int`.

Schematically (needs rigorous development):

```
C(k) ~ k^{d_2 / d_int - 1}
```

Our observed `p ≈ 0.17` would imply `d_2 / d_int ≈ 1.17`. This is non-trivial — for uniform-on-manifold data `d_2 = d_int` and `p = 0` (no scaling), whereas our trained networks robustly show `p > 0`. The excess `d_2 - d_int > 0` would quantify how strongly the learned representation *concentrates* mass beyond what a uniform manifold would produce.

**Test:** estimate `d_2` (GP correlation integral) and `d_int` (TwoNN or MLE) independently on the same point clouds; check whether `p ≈ d_2/d_int - 1` holds per cell. Our existing `genome_primitives.py` already has TwoNN / MLE; GP needs adding.

**Caveat:** this sketch assumes a single-scale fractal. Real point clouds likely have a scale-dependent correlation dimension (multifractal). The observed power law with narrow p across systems might reflect a *shared* slope in the log-log plot of `N_pair(r)` — i.e., a common *multifractal regime* in the range of `r` that kNN with `k ∈ [3, 130]` probes.

---

## Candidate framework B — doubling-dimension ratio

The doubling dimension `d_db` of a metric space is the smallest `d` such that every ball of radius `2r` can be covered by `2^d` balls of radius `r`. For data drawn iid from a distribution with doubling dim `d_db` embedded in ambient dim `h`:

- kNN graph clustering at fixed `k` depends on the ratio of "local" to "ambient" volumes.
- As `k` grows, the effective probe radius grows; clustering responds to the doubling structure at larger scales.

A schematic prediction:

```
C(k) ~ k^{(h - d_db) / d_db}
```

This is appealing because it connects `p` to the gap between ambient dim and doubling dim — the exact quantity that the manifold hypothesis says neural networks exploit. Trained networks have small `d_db` embedded in large `h`, so `(h - d_db)/d_db` is the natural compressibility exponent.

**Test:** measure doubling dim directly (net-growth scaling on the point cloud), predict `p` from `(h - d_db)/d_db`, compare to empirical `p`.

---

## Candidate framework C — heavy-tailed neighbor-count distribution — **FALSIFIED 2026-04-21 (genome_nn_degree_pilot_qwen3)**

**Pilot result** (`results/gate2/nn_degree_pilot_qwen3.json`, Qwen3-0.6B mid-depth `ℓ/L = 0.52`, n=1000 C4-clean stimuli, seed 42):

| k | mean in-deg | max in-deg | `α` tail | R² log-log | `p_pred = (3-α)/(α-1)` |
|---|---|---|---|---|---|
| 10 | 10.0 | 72 | **3.80** | 0.961 | **-0.285** |
| 30 | 30.0 | 149 | 3.92 | 0.931 | -0.315 |
| 60 | 60.0 | 266 | 3.78 | 0.909 | -0.282 |

Empirical Qwen3 mid `p = +0.156` (from `results/gate2/ck_power_fit.json`). Framework C predicts `p` in `[-0.32, -0.28]`. **Sign is wrong, magnitude is off by 3×.**

**Interpretation.** The NN-in-degree distribution *is* a power law (log-log tail `R² > 0.9`) but with `α ≈ 3.8` — a sub-Zipf, only-moderately-heavy tail. In Dorogovtsev-Mendes scale-free-graph theory, `α > 3` regimes predict clustering that is constant or decreasing in `k`; only `α < 2` would produce the observed growth. Trained networks' degree distributions land in the wrong regime for this framework to explain `C(k) ~ k^p`.

**Kill decision.** Drop framework C from the candidate list. The NN-degree distribution is still an interesting *descriptor* of learned geometry — `α ≈ 3.8` across depths 10/30/60-neighborhoods at k=1000 is itself non-trivial and worth checking cross-architecture as a separate diagnostic — but it does not *cause* the observed power law in clustering.

**Reduced candidate set after this pilot: A, B, D (three remaining).**

---

## Candidate framework C — heavy-tailed neighbor-count distribution (original text, retained)

In trained networks, the `k`-NN degree distribution (how many points claim `x` as one of their `k` nearest) is likely heavy-tailed — a few "hub" points are many others' neighbors, mirroring the power-law degree distribution seen in many learned graph structures (word embeddings, protein interaction networks).

If the NN-in-degree has tail exponent `α`, then clustering coefficient inherits a related power-law in `k` from the standard relation between degree distribution and clustering in scale-free graphs:

```
C(k) ~ k^{(3 - α)/(α - 1)}    (rough, borrowed from Dorogovtsev-Mendes for scale-free networks)
```

**Test:** measure the empirical NN-in-degree distribution on our existing point clouds. If it's approximately power-law with `α ≈ 2.6` (one self-consistent value for `p ≈ 0.17`), this framework is supported.

---

## Candidate framework D — rate-distortion / successive refinement

The manifesto frames intelligence as multi-resolution coding (Equitz-Cover, Rimoldi). In that view:

- The point cloud at layer `ℓ` is an encoder output.
- `k` indexes resolution: small `k` = fine-grained, large `k` = coarse-grained.
- Clustering coefficient captures the local-coherence-vs-scale trade-off the encoder has learned.

The Shannon lower bound predicts how rate decays with distortion for a given source. Trained networks approximate source-optimal codes (cf. Tishby's information bottleneck). The power-law exponent `p` then measures how gracefully the code degrades with resolution — which would be a scale-free property of the source distribution in the rate-distortion sense.

**Test:** compute the rate-distortion function `R(D)` on the same point clouds (via universal bounds like GRP); check whether the log-log slope of `R(D)` equals `p` or `1 - p` or has some other clean relation.

This is the most speculative framework — but potentially the most manifesto-aligned.

---

## Framework ranking (subjective, pending Codex review)

| Framework | Cleanness | Testability | Manifesto alignment | Priority |
|---|---|---|---|---|
| A — fractal d_2/d_int | High | High (both estimators are off-the-shelf) | Medium | 1 |
| B — doubling-dim ratio | Medium | Medium (doubling-dim needs implementation) | High | 2 |
| D — rate-distortion | Low | Low (R(D) is hard to estimate directly) | Very high | 3 |
| ~~C — heavy-tailed NN-degree~~ | ~~Medium~~ | ~~High~~ | ~~Medium~~ | **FALSIFIED** (2026-04-21 pilot: α ≈ 3.8, predicts wrong sign) |

---

## Next concrete steps

1. **Pilot A** — compute GP correlation dimension `d_2` on existing 18 cells; plot `p` vs `(d_2/d_int - 1)`. Takes ~30 min CPU on existing point clouds.
2. **Pilot C** — compute NN-in-degree histogram per cell; check for power-law tail. ~10 min.
3. **Pilot B** — implement net-growth doubling-dim estimator; ~1 day.
4. **Codex design gate** — once one framework shows a quantitative match to within `|observed - predicted| / observed < 20%` across ≥ 3 systems, write it up as v2 derivation and lock it *before* fitting additional systems.

---

## Scope this note does NOT close

- Does **not** explain narrow `c_0 ≈ 0.21`. That's a separate fit constant; may come out of the same framework or may need a second argument (e.g., a normalized-volume constant).
- Does **not** address biology equivalence. If v2 holds, biology should produce `p` in the same band — our 10-session run is the first empirical check.
- Does **not** claim a preferred framework. Paper remains at "empirical power-law, no new derivation locked" per §4.5.

This note is a scaffold for the Gate-2 G2.3 *re-*derivation effort. Lock only the framework that survives pilots 1–3 and Codex review.
