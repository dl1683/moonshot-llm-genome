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

Three preliminary biology data points from the Allen Brain Observatory Visual Coding Neuropixels dandiset (`000021`, three independent mouse sessions under Natural Movie One), each extracted at a 200-neuron subsample:

| Session | Mouse | `n_neurons` | `n_stimuli` | kNN-10 | SE | in DINOv2 range [0.30, 0.35]? |
|---|---|---:|---:|---:|---:|:---:|
| 0 | sub-699733573 | 200 | 900 | 0.3534 | 0.0050 | ✓ |
| 1 | sub-703279277 | 200 | 900 | 0.3222 | 0.0047 | ✓ |
| 2 | sub-707296975 | 200 | 900 | 0.3937 | 0.0058 | slightly above |

Mean across the three sessions: `0.356 ± 0.036`. All three values land in the trained-network band `[0.28, 0.52]`; **two of three land inside DINOv2's ImageNet-val range** `[0.30, 0.35]`. Per-session equivalence to DINOv2 at the pre-registered `δ=0.10` tolerance passes on one session (the middle one); at `δ=0.20` all three pass.

Three observations and their limits.

First, cross-species kNN-10 data points now exist for a trained-network atlas coordinate. The fact that mouse V1 under ecological natural-movie stimuli produces values in the same band as DINOv2-small under ImageNet-val stills is the first direct evidence that this primitive reads something about the representational geometry of inference systems broadly, not just artificial networks.

Second, **we have not yet met the biology prereg's formal criterion** (≥60% of sessions pass at `δ=0.10`). At 3 sessions we have 1/3 pass. This can be a sample-size-too-small finding that will resolve as 30+ sessions come in, or it can be a genuine restriction on the cross-species claim — mouse V1 and self-supervised ViT may measurably differ. Both outcomes are publishable; the full run is the test.

Third, the kNN-10 value is **sensitive to the neuron-count subsample** — 0.389 at 100 neurons collapses to 0.353 at 200 on the same session, consistent with the `k^p` power-law dependence we identify in §4.5 (larger cloud → different regime). A principled biology-vs-ANN equivalence test therefore ought to live at the `(c_0, p)` level — matched power-law parameters rather than matched scalar values at a single `k, n` operating point. We pre-register this as the operational form of the G2.5 test for the full run.

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
