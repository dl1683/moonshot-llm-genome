# §1 Introduction + §2 Related Work — Draft Prose (workshop paper)

**Status:** DRAFT (2026-04-21 T+21.5h). Target §1 ≈ 600-700 words, §2 ≈ 600-800 words.

---

## §1 Introduction

A useful way to state the current AI paradigm is that it treats scale as the primary lever: more parameters, more data, more compute, more energy. The complement paradigm — that *intelligence is a property of representational geometry* and that scale matters only insofar as it lets that geometry emerge — is older (Bengio, Courville & Vincent 2013 on the manifold hypothesis) but has lacked tools crisp enough to make it mechanically testable on modern networks.

This paper contributes one such tool: a pre-registered, mechanically-validated atlas coordinate — the mean local clustering coefficient of the `k`-nearest-neighbor graph on pooled hidden-state point clouds — whose *functional form* `C(X, k) ≈ c_0 · k^p` is shared across eight trained neural networks spanning five distinct training objectives, with cross-architecture consistency tighter than any value-level similarity metric in the literature we are aware of. Crucially, we report **both** a confirmation and a pre-registered falsification: the coordinate's cross-architecture portability holds at the functional-form level; the specific Laplace-Beltrami-convergence derivation we locked in writing before running the experiments is falsified by the data (it predicted decreasing-in-`k`; reality is increasing-in-`k`). Pre-registration discipline made the falsification clean. The universality survives the falsification because the empirical functional form — a simple power law with shared `(c_0, p)` — is a strictly stronger claim than the point-portability we originally registered.

**What we report**, organized by the pre-registered Gate structure (`research/prereg/genome_knn_k10_portability_2026-04-21.md` LOCKED at commit `62338b8`):

| Gate | Criterion | This paper's status |
|---|---|---|
| G1.2 | Rotation / isotropic-scale invariance | PASS (by construction + empirical verification) |
| G1.3 | Stimulus-resample stability across architectures | **7/8 PASS at δ=0.10 + Falcon tip at n=4000** |
| G1.5 | Quantization stability FP16↔Q8 | PASS 4/4 text at δ=0.05 |
| G1.7 | Not a random-geometry artifact | PASS (4-7× above Gaussian baseline) |
| G2.3 | Functional-form identification (derivation-backed) | v1 FALSIFIED; v2 power-law with R² > 0.994 |
| G2.4 | Causal-ablation load-bearingness | PASS 3/3 text; DINOv2 method-limit |
| G2.5 | Biological instantiation | Preliminary (Allen V1, 200 neurons in DINOv2 range) |

**What we do not claim.** We do not claim Level-1 universality (the formal standard is Gate-1 + Gate-2 all PASS). We claim a strong G1 portability result, a narrower-than-pre-registered but stronger-than-expected functional-form universality (G2.3 replacement), a text-only G2.4 causal claim, and a preliminary G2.5 data point that lands in-range.

**Why a reader of the representation-alignment literature should care.** Existing cross-architecture similarity claims (CKA, SVCCA, the Platonic Representation Hypothesis) are mostly linear-similarity claims on activation spaces, and have been criticized for conflating scale with geometry (the "Aristotelian-View" critique, 2026). Our primitive is a rank-based graph invariant that escapes the PC-dominance pathology by construction. Existing mechanistic-interpretability work (Anthropic circuits, SAE feature decomposition) operates at the feature-direction level rather than the point-cloud level. Our result complements, rather than competes with, this line: we show that a point-cloud-level invariant with specific functional form is preserved across architectures even when feature-directional circuits clearly differ.

**Contributions.**

1. A pre-registered, validator-checked Gate-1 / Gate-2 framework for atlas-coordinate claims, with LOCKED artifacts at specific git commits.
2. A ≥3.5× separation of the kNN-10 clustering coefficient over its random-Gaussian null baseline on eight architectures across five training objectives.
3. A pre-registered falsification of a specific Laplace-Beltrami-convergence derivation — and its replacement by an empirical power-law form with R² > 0.994 across 15 (system, depth) cells.
4. A text-architecture causal-ablation result showing the local-neighborhood subspace is load-bearing (7.8–443% loss increase, 20–66× specificity).
5. A first biological data point (mouse V1 Neuropixels under natural movies) that lands inside the trained-network reference range at matched neuron count.

All artifacts — code, atlas rows, pre-registrations, validator, paper drafts — are released open-source at the time of submission.

---

## §2 Related Work

This section groups the prior work that shapes our interpretation into four threads.

**Cross-architecture representational similarity (linear-metric lineage).** Kornblith et al. (2019, arXiv:1905.00414) established CKA as the dominant cross-architecture similarity metric. Raghu et al. (2017, arXiv:1706.05806) contributed SVCCA. Morcos et al. (2018) contributed projection-weighted CCA. Huh, Cheung, Wang, & Isola (2024, arXiv:2405.07987, *Platonic Representation Hypothesis*) argued cross-architecture / cross-modality convergence is large using these metrics. The *Aristotelian View* critique (Feb 2026, arXiv:2602.14486) flagged that linear-similarity metrics conflate scale with geometry — top-PC dominance makes networks look more similar than their underlying manifolds justify, and the signal that genuinely survives cross-architecture comparison is *local-neighborhood* structure rather than global alignment. Our result is consistent with the Aristotelian-View direction: a rank-based graph primitive on local neighborhoods passes where linear-alignment fails.

**Manifold-hypothesis and kNN-graph geometry.** Bengio, Courville & Vincent (2013) articulated the manifold hypothesis. Facco et al. (2017, *TwoNN*) gave the intrinsic-dimension estimator we use for independent validation. Belkin & Niyogi (2003, *Laplacian Eigenmaps*) and Coifman & Lafon (2006, *Diffusion Maps*) established that kNN graphs converge to the Laplace-Beltrami operator on the underlying manifold under standard i.i.d.-sampling and bounded-density assumptions. Our LOCKED v1 derivation specialized these results to the clustering coefficient and predicted a *decreasing*-in-`k` functional form; the data falsifies that sign (§4.5). A successor derivation that recovers `k^p` with `p ≈ 0.17` — plausibly from an alternate `k/n → 0` vs `k → ∞` scaling limit — is the single most important follow-up.

**Mechanistic interpretability (feature-direction lineage).** Anthropic's feature-and-circuit program (Elhage et al. 2022; Olsson et al. 2022, on induction heads; subsequent SAE work 2023-2025) studies interpretable causal directions within single models. DeepMind's circuit-analysis and causal-intervention work (2023-2025) similarly targets model-internal structure. These literatures are not directly comparable to a cross-architecture point-cloud invariant — they answer "what is Model X doing?" not "what is shared across Models X, Y, Z?" The two lines complement: if our functional form `C(X, k) ≈ c_0 · k^p` transfers across model families, it gives a coordinate on which feature-direction interventions (routing, steering, alignment patches) can be *compared* between analog models.

**Biological-neural comparison.** Yamins et al. (2014) established hierarchy-matching between deep networks and macaque IT. The Allen Brain Observatory Visual Coding Neuropixels project (dandiset `000021`) provides the open Neuropixels recordings we use for G2.5. More recent comparisons (Schrimpf et al. 2018 Brain-Score; Conwell et al. 2024; our own CTI 2026, github.com/dl1683/moonshot-cti-universal-law) use regression-based or representational-similarity-analysis frameworks. We are not aware of a prior publication that measures a kNN-graph-level invariant on mouse V1 under Natural Movie One and compares it to a self-supervised vision transformer at matched neuron count. Our preliminary value (0.353 at 200 neurons, inside DINOv2's 0.30–0.35 range) is a single data point that motivates the full 30-session run pre-registered in `research/prereg/genome_knn_k10_biology_2026-04-21.md`.

**Pre-registration in ML research.** Pre-registration is rare in ML. Cohn et al. (2022) and the ML-Reproducibility-Challenge workflow are the main adjacent efforts. Our framework adds machine-validated pre-reg artifacts — the validator verifies code identities, scope IDs, equivalence-criterion parameters, and LOCKED-status discipline programmatically. We believe this is the first ML paper we are aware of in which the primary theoretical prediction was *falsified by its own pre-registered data* and where the falsification is itself the intended scientific record.

---

**Word-count self-check.** §1 ~720 words. §2 ~800 words. Combined 1520 words target was 1200-1500 — slight overshoot, trim §2 Platonic/Aristotelian para if needed at paper-assembly time.

**Integration notes:**
- §1 leads with the scale-vs-geometry contrast (manifesto hook), summarizes the gate structure in a table, states contributions as 5 bullets.
- §2 splits into 4 threads: linear-similarity lineage, manifold+kNN lineage, mechanistic interp, biology comparison — each with a honest one-line position of our work relative to the thread.
- Last paragraph on pre-registration-in-ML is important methodological contribution to flag.
