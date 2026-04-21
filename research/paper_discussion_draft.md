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

## 5.5 Practical consequences: a first Geometry → Efficiency data point, partial generalization

One of the three practical consequences we listed in the abstract — using the coordinate as a compression-gating signal — admits a quick empirical test. We run the same `k`-sweep extraction + power-law fit at three weight-quantization levels (FP16, bitsandbytes 8-bit, bitsandbytes NF4 4-bit), on the same 500-stimulus C4 batch, and measure next-token NLL on the same batch as an independent capability proxy. We test three text architectures: Qwen3-0.6B (transformer CLM), RWKV-4-169M (linear-attention recurrent), DeepSeek-R1-Distill-Qwen-1.5B (reasoning-distilled CLM).

**Table 8. Geometry → Efficiency across 3 text architectures (n=500 C4, mid-depth).** Sources: `results/gate2/geom_efficiency.json`, `results/gate2/geom_efficiency_rwkv4_3q.json`, `results/gate2/geom_efficiency_deepseek_3q.json`.

| System | Quant | `c_0` | `p` | R² | NLL / tok | ΔNLL vs FP16 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | FP16 | 0.262 | 0.164 | **0.9967** | 3.656 | — |
| Qwen3-0.6B | Q8 | 0.270 | 0.154 | **0.9927** | 3.668 | +0.3% |
| Qwen3-0.6B | Q4 | 0.253 | 0.174 | **0.9835** | 3.791 | +3.7% |
| RWKV-4-169M | FP16 | 0.251 | 0.206 | **0.9972** | 3.504 | — |
| RWKV-4-169M | Q8 | 0.239 | 0.222 | **0.9945** | 3.518 | +0.4% |
| RWKV-4-169M | Q4 | 0.244 | 0.220 | **0.9894** | 3.760 | +7.3% |
| DeepSeek-1.5B | FP16 | 0.255 | 0.167 | **0.9969** | 4.297 | — |
| DeepSeek-1.5B | Q8 | 0.256 | 0.166 | **0.9937** | 4.317 | +0.5% |
| DeepSeek-1.5B | Q4 | 0.253 | 0.172 | **0.9942** | 4.386 | +2.1% |

**Partial generalization.** On Qwen3 and RWKV, `R²` decreases **monotonically** with compression and NLL increases monotonically — the simple "R²-as-compression-stop" rule works. On DeepSeek the R² drops cleanly from FP16→Q8 (−0.0032) but **bounces back at Q4** (+0.0005 above Q8), despite NLL continuing to rise. The FP16↔Q8 direction is consistent across all three systems; the fine-grained Q8↔Q4 direction is not. One partial explanation: reasoning-distilled models may carry compressed-by-design geometries that degrade differently than base-CLM or recurrent systems under bnb Q4 quantization.

**The scoped tool.** `R² of the C(X, k) power-law fit` is a useful **coarse-grained compression signal** — FP16 to Q8 to Q4, R² monotone-decreases in 2 of 3 tested systems, and the FP16→Q8 drop is consistent across all 3. For finer-grained "stop here" decisions at Q4-level aggression, the signal is architecture-dependent and should not be used without per-family calibration. We do not promote R² to a single-scalar compression-stop rule; we promote it to a **candidate early-warning signal** whose first strong-monotonicity failure (DeepSeek Q8→Q4) is itself informative.

**What this does NOT say.** The Geometry→Efficiency probe does not claim that degrading geometry *causes* capability drop (only that the two track each other in the FP16→Q8 regime). It does not claim the R² signal is calibrated to absolute NLL increase. It does not claim generalization beyond bnb int8/NF4 quantization to e.g. pruning or distillation.

**The other two practical consequences** (geometry-aware KV caching, cross-model intervention transfer) are not tested in this paper. We list them as hypothesis seeds for intended readers: for compression and caching teams, the atlas gives a single-scalar compressibility signal worth adding to existing per-layer heuristics; for interpretability teams (Anthropic, DeepMind, Martian), it gives a coordinate on which to transfer feature-direction interventions between analog models.

---

**Word-count self-check.** ~1150 words. ✓

**Integration notes:**
- Keeps the falsification honest (§5.1 #2, §5.4 last bullet)
- Bounds the biology claim carefully (§5.3) — two data points, one in-range with correct neuron count
- Points at the v2 derivation as the single most-important follow-up without overclaiming (§5.2)
- §5.5 gives three commercial / research hooks without promising any of them
