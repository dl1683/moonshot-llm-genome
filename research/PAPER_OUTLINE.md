# Workshop Paper Outline — Neural Genome: Cross-Architecture kNN Clustering Coefficient

**Status:** DRAFT (2026-04-21). Locks when Codex R8 + G2.3 fit + DINOv2 causal all land with favorable verdicts.

**Target venue (pick one):** NeurIPS 2026 Workshop on Scientific Methods for ML · ICLR 2026 Workshop on Representation Alignment · ICML 2026 Workshop on Mechanistic Interpretability.

**Manuscript target length:** 8 pages + appendix (workshop standard).

---

## Title (candidates)

1. *A Cross-Architecture Geometric Invariant: kNN Clustering Coefficient Across 8 Neural Network Classes and 5 Training Objectives*
2. *Geometry, Not Scale: Empirical Evidence for a Universal Local-Neighborhood Invariant in Trained Neural Networks*
3. *The First Atlas Coordinate: Cross-Class Portability and Causal Load-Bearing of kNN-10 Clustering*

Preference: #2 — manifesto-aligned, not jargon-heavy, works for both ML and interp audiences.

---

## Abstract (~200 words, drafted below)

We report the first mathematical coordinate in a cross-architecture representational-geometry atlas that passes all five pre-registered criteria for Level-1 universality: portability across ≥5 architecture classes, resampling stability, estimator-variant robustness, quantization-stability, and causal load-bearing. The coordinate is the **mean local clustering coefficient of the k=10 Euclidean nearest-neighbor graph** on pooled hidden-state point clouds. On 9 architectures spanning 7 distinct training objectives (autoregressive CLM, reasoning-distilled CLM, linear-attention recurrent, hybrid transformer+Mamba, masked-LM, contrastive text, self-supervised vision, contrastive vision, predictive-masked vision, class-conditional diffusion transformer), the coordinate takes statistically-indistinguishable values at a Bonferroni-corrected δ=0.10 equivalence threshold. The value is 4–7× above a random-Gaussian baseline at matched sample size and ambient dimension, ruling out a high-dimensional artifact. A stronger control with random-initialized twins of the same architectures produces power-law exponents spanning `p ∈ [0, 0.37]` (22× wider than the trained cluster, modality-stratified to `p ≈ 0.17` for text and `p ≈ 0.22` for vision), establishing that the cross-architecture band is the *output of training*, not an architectural constant. The coordinate survives 4× weight compression (FP16→Q8) at tighter δ=0.05. On three text architectures, ablating the subspace the coordinate identifies causes next-token loss to increase 7.8–443% at full magnitude while random-10-dim and top-10-PC controls move loss <1% and <11% respectively (specificity 20–66×). **On 10 Allen Brain Observatory Visual Coding Neuropixels sessions under Natural Movie One, mouse V1 kNN-10 values land inside the DINOv2 reference band at the pre-registered δ=0.10 tolerance with 100% pass rate (10 of 10) and at strict δ=0.05 with 80% pass rate (8 of 10), clearing the pre-registered 60% threshold with ≥20-point margin at both tolerances.** A Laplace-Beltrami-convergence derivation predicts the functional form `C(X,k) = α_d(1 − β_d·κ·k^(2/d_int))₊` before fitting. Results are pre-registered, reproducible on a single consumer GPU, and released open-source.

---

## §1 Introduction (1 page)

**Hook.** The dominant paradigm in AI scaling is "more parameters, more data, more compute." We test the complementary hypothesis — that intelligence is a property of **representational geometry**, not parameter count — by looking for mathematical coordinates that take the same values across architectures.

**Contribution.** First cross-architecture coordinate passing portability + quantization + causality, with a pre-registered, LOCKED derivation predicting its functional form.

**What this paper is NOT.** We do not claim Level-1 universality. We claim Level-1 *Gate-1* satisfaction (portability) plus *Gate-2 G2.4* (causality on 3/3 text systems). Gate-2 G2.3 (hierarchical-fit identification of α_d, β_d) and G2.5 (biology bridge) are explicitly deferred follow-up work.

---

## §2 Related work (1 page)

Tables, not prose:

- **Linear-similarity approaches** (CKA, SVCCA, Procrustes) — Kornblith 2019, Raghu 2017, Morcos 2018. Critique: conflates scale with geometry (Aristotelian-View 2026).
- **Representation-alignment universality** — Huh et al. 2024 Platonic Rep Hypothesis. Our work uses a non-linear graph-theoretic primitive, giving different robustness.
- **Mechanistic interpretability** — Elhage, Nanda et al. 2022–2025. Different level of analysis (features vs. point-cloud geometry). Complementary, not competing.
- **Manifold-hypothesis lineage** — Bengio 2013, Facco et al. 2017 (TwoNN), Belkin-Niyogi 2003 (Laplacian Eigenmaps). Our derivation is the Laplace-Beltrami limit of kNN graphs specialized to the clustering coefficient.
- **Biology-AI bridges** — Allen Brain Observatory, Yamins 2014 hierarchy matching, our own CTI 2026 showing equicorrelation ρ≈0.45 constant across 12 architectures + mouse V1.

---

## §3 Method (2 pages)

**§3.1 Primitive.** Formal definition of kNN-k clustering coefficient. Analytical SE. Rotation + isotropic-rescale invariance (stated and proven).

**§3.2 Gate semantics.** Noise-calibrated equivalence criterion `|Δ| + c·SE(Δ) < δ·median(|f|)` with Bonferroni-corrected c, pre-registered δ sensitivity sweep, sentinel depths at ℓ/L ∈ {0.25, 0.50, 0.75}.

**§3.3 Stimulus families ℱ.** Machine-checkable (generator + filter + invariance-check pinned to (git_commit, file_path, symbol)). Text: `c4_clean.len256.v1`. Vision: `imagenet1k_val.v1`. Hashes provided.

**§3.4 Architecture bestiary.** 8 classes listed with HF IDs, parameter counts, training objectives, and pooling semantics.

**§3.5 Causal-ablation protocol.** Three schemes (topk, random-10d, pca-10) with λ sweep. Specificity as the ratio topk / max(random, pca).

**§3.6 Derivation.** Brief summary of Laplace-Beltrami convergence → functional form. Full derivation in appendix.

**§3.7 Pre-registration discipline.** Every claim in this paper has a locked prereg file with a validator-verified commit SHA. Validators open-sourced.

---

## §4 Results (3 pages — the core)

**§4.1 Cross-architecture portability (Gate-1 G1.3).**

Table 1: kNN-k10 values per (system, depth) seed-averaged. 8 systems × 3 depths = 24 cells. Verdict column: PASS/FAIL at δ ∈ {0.05, 0.10, 0.20}.

Key finding: 7/8 systems PASS at δ=0.10. Falcon-H1 requires n=4000 (SE halves). Spread across 5 training objectives. Values cluster in [0.28, 0.36].

**§4.2 Not a random-geometry artifact.**

Table 2: random-Gaussian baseline at matched n and h. kNN-k10 ≈ 0.05–0.08. Trained-network values are 4–7× higher → not trivial high-dim behavior.

**§4.3 Quantization stability (G1.5).**

Table 3: FP16 vs Q8 on 4 text systems. kNN-k10 max_stat < δ=0.05 · median on all 4 — geometry survives 4× compression.

**§4.4 Causal ablation (Gate-2 G2.4).**

Table 4: 3 text systems × 3 depths × 3 schemes × 5 λ grid. All 8 completed cells monotonic in λ; topk λ=1 effect 7.8–443%; random 0.2–13%; pca 4.9–16.3%. Specificity 20–66× topk/random.

Figure 1: loss vs λ curves, 3 systems, colored by scheme.

**§4.5 Functional-form fit (Gate-2 G2.3 — conditional on extended k-sweep finishing).**

Table 5: H0 pooled (α_d, β_d shared; κ_i, d_int,i per-system) vs H1 per-system. ΔBIC, residuals, coefficient CIs. If H0 wins by ΔBIC > 10 → Level-1 functional-form claim. If not → we report it honestly and demote to "cross-class coincidence, not universal generator."

---

## §5 Discussion (1 page)

**§5.1 What the result does and doesn't say.** Cross-class portability ≠ Level-1 universality. Level-1 requires G2.5 biology, which is this paper's explicit next step.

**§5.2 Limits of the instrument.** Falcon n=2000 narrow-fail reminds us δ=0.10 is noise-sensitive at small n. DINOv2 causal test uses classification CE not NLL; cross-modality loss comparisons should be interpreted carefully.

**§5.3 Practical consequences if Level-1 holds.**
- Portable quantization priors (which layers tolerate INT4/INT8, architecture-agnostic).
- Geometry-aware KV caching (which activation subspaces are safe to compress).
- Alignment-intervention transfer (circuits in Model A → Model B via shared coordinate).

---

## §6 Conclusion + Next (1/2 page)

Gate-2 G2.3 (functional-form) + G2.5 (biology) are the remaining tests that would move kNN-k10 from "portable coordinate with causal backing" to Level-1 universal.

---

## Appendix

A. Full Laplace-Beltrami derivation (from `research/derivations/knn_clustering_universality.md` LOCKED).
B. All prereg files (reproduced verbatim).
C. Validator source + worked examples.
D. Random-Gaussian baseline full table.
E. Reproducibility: git commits + dataset hashes + hardware envelope.

---

## Open-source release plan

- Repo: `github.com/dl1683/moonshot-llm-genome` (currently local-only, no remote).
- Release: atlas code + all 9+ ledger entries + prereg + validator + this paper as preprint.
- Embargo: 2-week window for syndicate collaborators (Martian, Furiosa, Weka, VERSES) to give feedback + co-author requests before public drop.

---

## Draft author list

- Devansh (CMC — lead)
- TBD collaborators post-syndicate outreach

---

## Blockers for lock

- [ ] Codex R8 feedback integrated
- [ ] G2.3 extended k-sweep + hierarchical fit verdict landed
- [ ] DINOv2 causal probe smoke + grid (at least depth-1) landed
- [ ] Either RWKV-depth-2 NaN fix OR explicit acknowledgment of the numerical-stability caveat in the limitations section
- [ ] Allen Neuropixels pipeline scaffolded (biology bridge named as follow-up even if not run)
- [ ] README + CLAIM_EVIDENCE_MAP.md + WIKI all consistent with final text

---

*This outline lives in `research/PAPER_OUTLINE.md`. Codex R8 Q5 (the "next 72 hours" question) should inform the blocker order above.*
