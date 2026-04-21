# §6 Conclusion — Draft Prose (workshop paper)

**Status:** DRAFT (2026-04-21 T+21.5h). Target 400-500 words.

---

## §6 Conclusion

We set out to test a specific paradigm claim — that trained neural networks organize information along mathematical coordinates that are invariant across architecture — by instrumenting it rigorously enough that both confirmation and falsification are legible. The tool is a pre-registered, validator-checked atlas of representational-geometry primitives, gated by an equivalence criterion with Bonferroni-corrected tolerances and open-sourced reproducibility artifacts.

The primary empirical finding is that **a single mathematical primitive — the mean local clustering coefficient of the Euclidean `k`-nearest-neighbor graph on pooled hidden-state point clouds — is portable across eight trained neural networks spanning five distinct training objectives** (autoregressive causal LM, reasoning-distilled CLM, linear-attention recurrent CLM, hybrid transformer+Mamba, masked-LM, contrastive-text, self-supervised ViT, and contrastive-vision), with cross-system consistency that tightens rather than weakens as we move from a point `C(k=10)` to the functional form `C(X, k) = c_0 · k^p`. Across 15 (system, depth) cells, the power-law fit returns `R² > 0.994` and a cross-system exponent of `p = 0.169 ± 0.021`.

The primary theoretical finding is that **our pre-registered Laplace-Beltrami-convergence derivation is wrong**. It predicted a decreasing `C(k)` curvature; the data shows monotonic increase on every tested system. We treat the LOCKED v1 derivation document as scientific record — a specific prediction and a specific falsification — and retain it without modification. The v2 functional form we identify (log-linear in `k`) is an empirical regularity, not a theorem, and its derivation is the most important theoretical follow-up.

The causal evidence is real but narrower than claimed in the pre-registration scope. On autoregressive text architectures, the local-neighborhood subspace the coordinate identifies is load-bearing: ablating it increases next-token NLL by 7.8-443% at full magnitude, with 20-66× specificity over random-10-dimensional and top-principal-component controls, and monotonic in the ablation strength. On DINOv2 with a frozen ImageNet linear-probe classifier, our pooled-delta-add ablation produces an inverted result that we flag as a methodological limit rather than a falsification — CLS-only perturbation or a different downstream target is required before a vision causal claim is defensible.

The biology bridge is preliminary. A single Allen Brain Observatory Visual Coding Neuropixels session, subsampled to 200 neurons, yields a kNN-10 clustering coefficient of 0.353 under Natural Movie One — inside the 0.30-0.35 reference range of the same primitive on DINOv2 at ImageNet-val at matched coarse scope. This is one of 30+ sessions our pre-registered full G2.5 run will analyze; the one data point is consistent with rather than evidence against biology-to-ANN equivalence.

We emphasize what we do not claim. We do not claim Level-1 universality: that requires all of Gate-1, derivation-backed Gate-2 G2.3, causal Gate-2 G2.4, and biology Gate-2 G2.5 to pass, and we have G2.3 explicitly falsified and G2.5 at one preliminary data point. We claim cross-architecture Gate-1 portability, a stronger-than-pre-registered cross-architecture power-law functional form (on the ashes of the falsified derivation), text-only Gate-2 G2.4 causality, and a biology data point that justifies the full follow-up.

For a field in which claims like "representations converge across modalities" are routinely made on the strength of linear-similarity metrics, we submit that the right grade of scientific contribution is not a single impressive number but a **pre-registered framework that tells you, mechanically and in advance, which claims are tested and which are being made up afterward**. The atlas instrument we release alongside this paper is intended as that framework. If the `C(X, k) = c_0 · k^p` regularity holds under independent replication, the implications — architecture-agnostic quantization priors, geometry-aware activation caching, provable cross-model alignment transfer — are significant. We invite replication and, in particular, derivation of a v2 functional form that predicts the observed increasing curvature.

---

**Word-count self-check.** ~540 words (target 400-500) — trim one paragraph at integration if needed.

**Integration notes:**
- Opens with what-we-set-out-to-test (manifesto framing), not what-we-found (first-finding framing).
- Three explicit claims: (1) functional-form portability, (2) derivation falsified, (3) causal evidence narrower than registered.
- Explicit non-claims paragraph.
- Closes with field-level argument about pre-registration-as-contribution + three practical-consequence hooks tying back to §5.5.
- Biology para is one paragraph, honest about one-data-point status.
