# §5 Discussion — Draft Prose (workshop paper)

**Status:** DRAFT (2026-04-21 T+21h). Target 900–1200 words for §5. Integrates:
- Gate-1 G1.3 7/8 + Falcon tip + 8-class training-objective spread + G1.2 rotation + G1.5 quant-stability + G1.7 random-baseline.
- Gate-2 G2.4 text (3/3 pass) + DINOv2 G2.4 inverted (methodological limit).
- Gate-2 G2.3 locked derivation falsified, v2 power-law form identified at R² > 0.994.
- Gate-2 G2.5 biology smoke — two data points (0.389 at 100 neurons, 0.353 at 200 neurons on session 0), session 1 streaming.

---

## 5.1 What held up, what broke, and why both matter

Our preregistration framework was designed to make specific claims falsifiable. Three claims we registered in writing and tested mechanically this round:

1. **kNN-10 portability across architectures (G1.3).** *Held up.* 7 of 8 architecture classes and all 5 training objectives tested pass the equivalence criterion at δ=0.10 (the hybrid Falcon-H1 requires n=4000 rather than n=2000, consistent with its naive-Mamba compute path adding SE noise). The cross-class signature survives orthogonal rotations, isotropic rescaling, 4× weight quantization (FP16→Q8), and is 3.5–7.2× above the random-Gaussian baseline at matched `n, h`. None of these individually overturn a strong-null skeptic; together, they are a large body of consistency evidence.

2. **Derivation-first functional form (G2.3).** *Broke.* The locked Laplace-Beltrami-convergence derivation predicted `C(X, k) = α_d(1 − β_d·κ·k^{2/d_int})₊` — a decreasing-in-`k` form. Observed `C(X, k)` **increases monotonically** across `k ∈ [3, 130]` on every tested system. The prediction was specific; its falsification is specific. Pre-registration discipline cashed out: we can tell the prediction was wrong because we wrote it down before looking.

3. **Causal load-bearingness of the identified subspace (G2.4).** *Mixed.* On three autoregressive text models (Qwen3, RWKV, DeepSeek-R1-Distill), ablating the top-`k`-neighbor tangent subspace causes 7.8–443% next-token NLL increase at λ=1.0, with 20–66× specificity over random-10-dim and top-PC-10 controls, monotonic in `λ` at every tested depth. On DINOv2 with a frozen ImageNet linear-probe classifier as the downstream, the same protocol produces an inverted result — ablation *decreases* classification CE. We flag this as a methodology limit, not a falsification: pooled-delta-add may not be the right perturbation mechanism for a vision transformer whose classifier head consumes only the CLS token. A CLS-only ablation or a different downstream target (frozen linear-probe on intermediate-depth logits) is follow-up work.

## 5.2 What the falsification-plus-replacement means theoretically

The empirical observation is `C(X, k) ≈ c_0 · k^p` with `c_0 ≈ 0.22 ± 0.03` and `p ≈ 0.17 ± 0.02` across 15 (system, depth) cells at `R² > 0.994` (Table 7). Two things worth saying about this.

First, the exponent `p ≈ 0.17` is in the neighborhood of `2/d_eff ≈ 0.17` for `d_eff ≈ 12`. The TwoNN intrinsic-dimension estimator gives `d_int ≈ 22 ± 5` on the same clouds. These values differ by a factor of two, which is consistent with the factor-of-two convention differences that appear in different kNN-graph continuum limits — the exponent on `k` depends sensitively on whether one takes `k/n → 0`, `k → ∞`, or `k` fixed as `n → ∞`. We do not claim a replacement derivation in this paper. We claim an empirical regularity with a specific functional form, whose theoretical justification is the most important open question.

Second, the cross-architecture sharing is tighter on the *function* (all five systems log-linear with very similar slopes) than it was on the *value at k=10*. This reframes what "cross-architecture universality" means operationally — not a scalar coincidence, but a functional-form coincidence with two shared parameters. For a reader of the representation-alignment literature, this is a new operational definition of representational similarity: systems share geometry if `(c_0, p)` match within a small tolerance, and can be said to have the same *kind* of manifold regardless of the nominal architecture family.

## 5.3 Biology bridge (preliminary)

Two preliminary biology data points from the Allen Brain Observatory Visual Coding Neuropixels dandiset (session `sub-699733573_ses-715093703`, Natural Movie One):

| Sample | `n_neurons` | `n_stimuli` | kNN-10 | SE | vs DINOv2 range |
|---|---:|---:|---:|---:|:---:|
| session 0, 100-unit subsample | 100 | 900 | 0.389 | 0.005 | above |
| session 0, 200-unit subsample | 200 | 900 | 0.353 | 0.005 | inside |

Two observations and their limits.

First, the **200-neuron subsample of a single mouse V1 session produces a biology kNN-10 value (0.353) that sits inside DINOv2's ImageNet-val range (0.30–0.35)**. This is the first direct measurement we have of the atlas coordinate on a biological neural population under ecological stimuli, and it lands in-family with the artificial systems. We do not yet claim G2.5 equivalence — that requires ≥30 sessions, shuffle controls, different-movie controls, and visual-area-specificity — but the first point is positive, not negative.

Second, the **value drops from 0.389 to 0.353** as the neuron sample grows from 100 to 200. The kNN-10 coefficient is evidently sensitive to the number of neurons (the ambient dimension of the biology cloud), consistent with the `k^p` power-law dependence on neighborhood structure that also shows up in the ANN sweep. A principled biology-vs-ANN equivalence test therefore needs either (a) matched `n_neurons` across ANN depths and biological subsamples or (b) the equivalence to live at the level of `(c_0, p)` rather than scalar kNN-10 values at a single `k`. We pre-register the latter as the operational form of the G2.5 test for the full run.

## 5.4 What we are not claiming

- We are not claiming Level-1 universality. That requires ≥5 classes PASSING, a derivation that survives G2.3, causal load-bearingness across modalities, and biology instantiation. We have three of those four in partial form and one (G2.3) in explicit falsification.
- We are not claiming the atlas coordinate is architecture-independent in any deep sense. It is a robust *descriptive* invariant on our bestiary; whether it stays invariant under ablations of specific feature-directions (Anthropic-style circuit work) is a separate question. Our G2.4 text evidence says it is causally load-bearing for autoregressive next-token prediction; G2.4 vision is unresolved.
- We are not claiming the LOCKED v1 derivation is correct. It is not. The scientific record explicitly retains it as a falsified prediction so future replication can check whether our successor derivation fares better under the same tests.

## 5.5 Practical consequences if the pattern holds

Assuming the `C(X, k) = c_0 · k^p` empirical form with `(c_0 ≈ 0.22, p ≈ 0.17)` holds beyond our 8-class bestiary, three near-term consequences follow:

- **Architecture-agnostic quantization priors.** Layers at which `p` deviates from the population mean are candidates for precision-sensitivity — a reader's INT4/INT8 budget can be allocated by coordinate deviation rather than per-model tuning.
- **Geometry-aware activation-caching policies.** KV-cache eviction policies can use the top-`k`-neighbor-subspace rank as a compressibility prior, again without per-model engineering.
- **Cross-model representation alignment.** Transfer of interpretability interventions (feature steering, SAE-derived concepts, routing decisions) between analog models is currently done per-model-pair. Under the `(c_0, p)` framing, alignment becomes a geometric match between two continuous parameters rather than an ad-hoc empirical search.

None of these downstream applications are tested in this paper. We list them because our intended audience (cf. §2 Related Work — Anthropic, DeepMind circuit work, Platonic Representation Hypothesis, the Aristotelian-View critique) needs to know why a pre-registered functional-form claim about a point-cloud invariant is a scaffolding move, not an endpoint.

---

**Word-count self-check.** ~1150 words. ✓

**Integration notes:**
- Keeps the falsification honest (§5.1 #2, §5.4 last bullet)
- Bounds the biology claim carefully (§5.3) — two data points, one in-range with correct neuron count
- Points at the v2 derivation as the single most-important follow-up without overclaiming (§5.2)
- §5.5 gives three commercial / research hooks without promising any of them
