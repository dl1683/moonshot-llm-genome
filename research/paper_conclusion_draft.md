# §6 Conclusion — Draft Prose (workshop paper)

**Status:** DRAFT (2026-04-21 T+21.5h). Target 400-500 words.

---

## §6 Conclusion

We set out to test a specific paradigm claim — that trained neural networks organize information along mathematical coordinates that are invariant across architecture — by instrumenting it rigorously enough that both confirmation and falsification are legible. The tool is a pre-registered, validator-checked atlas of representational-geometry primitives, gated by an equivalence criterion with Bonferroni-corrected tolerances and open-sourced reproducibility artifacts.

The primary empirical finding is that **a single mathematical primitive — the mean local clustering coefficient of the Euclidean `k`-nearest-neighbor graph on pooled hidden-state point clouds — is portable across nine trained neural networks spanning seven distinct training objectives** (autoregressive causal LM, reasoning-distilled CLM, linear-attention recurrent CLM, hybrid transformer+Mamba, masked-LM, contrastive-text, self-supervised ViT, contrastive-vision, predictive-masked ViT, and class-conditional diffusion transformer), with cross-system consistency that tightens rather than weakens as we move from a point `C(k=10)` to the functional form `C(X, k) = c_0 · k^p`. Across **27 (system, depth, seed) cells** — including a class-conditional diffusion transformer as a genuinely non-next-token-time generative-prediction system — the power-law fit returns `R² > 0.989` (mean 0.997) and a cross-system exponent of `p = 0.179 ± 0.021` (CV 12.0%).

The primary theoretical finding is that **our pre-registered Laplace-Beltrami-convergence derivation is wrong**. It predicted a decreasing `C(k)` curvature; the data shows monotonic increase on every tested system. We treat the LOCKED v1 derivation document as scientific record — a specific prediction and a specific falsification — and retain it without modification. The v2 functional form we identify (log-linear in `k`) is an empirical regularity, not a theorem, and its derivation is the most important theoretical follow-up.

The causal evidence is real but narrower than claimed in the pre-registration scope. On autoregressive text architectures, the local-neighborhood subspace the coordinate identifies is load-bearing: ablating it increases next-token NLL by 7.8-443% at full magnitude, with 20-66× specificity over random-10-dimensional and top-principal-component controls, and monotonic in the ablation strength. On DINOv2 with a frozen ImageNet linear-probe classifier, our pooled-delta-add ablation produces an inverted result that we flag as a methodological limit rather than a falsification — CLS-only perturbation or a different downstream target is required before a vision causal claim is defensible.

The biology bridge passes at the pre-registered scope. Across 10 Allen Brain Observatory Visual Coding Neuropixels sessions under Natural Movie One (200 cortical units each, 50 ms integration window), the biological kNN-10 clustering coefficient lands inside the DINOv2 ImageNet-val reference band at the pre-registered δ=0.10 tolerance in 10 of 10 sessions (100%), and at the tighter δ=0.05 tolerance in 8 of 10 (80%). The 60% pre-registered threshold is cleared by 40 points at δ=0.10 and by 20 points at δ=0.05. A full-scale G2.5 — shuffle control, different-movie control, 30+ sessions, area-specificity — remains follow-up work; the 10-session run passes the pre-registered criterion as written.

We emphasize what we do not claim. We do not claim Level-1 universality: that requires all of Gate-1, **derivation-backed** Gate-2 G2.3, causal Gate-2 G2.4, and biology Gate-2 G2.5 to pass. We have Gate-1 + G2.4 text + G2.5 10-session biology passing. G2.3 is partial: the pre-registered v1 Laplace-Beltrami-convergence derivation is falsified; the empirical replacement `C(X, k) = c_0 · k^p` holds at `R² > 0.989` across 27 cells; three of four candidate v2 derivation frameworks (fractal `d_2/d_int`, doubling-dim ratio, heavy-tailed NN-degree) are themselves falsified by structural sign or magnitude errors; rate-distortion is untested. Level-1 is gated on G2.3 re-derivation.

For a field in which claims like "representations converge across modalities" are routinely made on the strength of linear-similarity metrics, we submit that the right grade of scientific contribution is not a single impressive number but a **pre-registered framework that tells you, mechanically and in advance, which claims are tested and which are being made up afterward**. The atlas instrument we release alongside this paper is intended as that framework. If the `C(X, k) = c_0 · k^p` regularity holds under independent replication, the implications — architecture-agnostic quantization priors, geometry-aware activation caching, provable cross-model alignment transfer — are significant. We invite replication and, in particular, derivation of a v2 functional form that predicts the observed increasing curvature.

---

**Word-count self-check.** ~540 words (target 400-500) — trim one paragraph at integration if needed.

**Integration notes:**
- Opens with what-we-set-out-to-test (manifesto framing), not what-we-found (first-finding framing).
- Three explicit claims: (1) functional-form portability, (2) derivation falsified, (3) causal evidence narrower than registered.
- Explicit non-claims paragraph.
- Closes with field-level argument about pre-registration-as-contribution + three practical-consequence hooks tying back to §5.5.
- Biology para is one paragraph, honest about one-data-point status.
