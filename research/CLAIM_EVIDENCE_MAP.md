# CLAIM ↔ EVIDENCE MAP

**Purpose (per CLAUDE.md §9 success heuristic).** Every claim anywhere in the
repo (README, MANIFESTO, WIKI, blog-style copy) maps to exactly one ledger
entry + exactly one locked prereg. If a claim isn't in this table, it should
either be in the table (add it + map it) or deleted (unsubstantiated).

**Last updated.** 2026-04-26 (architecture-prior thread C10–C13 added).

---

## 1. Core claims with evidence trail

| # | Claim (verbatim or paraphrased) | Strength | Ledger entry | Prereg | Key result |
|---|---|---|---|---|---|
| C1 | kNN-10 clustering coefficient passes Gate-1 portability on 3 Batch-1 systems (Qwen3, RWKV, DINOv2) at strict δ=0.10 | 🟡 portability (scope: text.c4_clean.len256.v1 + vision.imagenet1k_val.v1) | `genome_007_stim_resample_n2000` | `genome_knn_k10_portability_2026-04-21.md` LOCKED at 62338b8 | `results/gate1/stim_resample_n2000_seeds42_123_456.json` — 3 systems max_stat={0.0253, 0.0239, 0.0188} vs margins={0.0330, 0.0336, 0.0313} all PASS |
| C2 | kNN-10 is FP16↔Q8 quantization-stable at δ=0.05 on all 4 text classes (Qwen3, DeepSeek, RWKV, Falcon-H1) | 🟡 G1.5 quant-stable at tight δ | `genome_008_quant_stability_g15` | `genome_knn_k10_portability_2026-04-21.md` §7 | `results/gate1/quant_stability_n2000_seed42.json` — max_stats 0.0136-0.0147 vs margin 0.016-0.017 all PASS |
| C3 | Geometry survives 4× weight compression regardless of architectural class (manifesto efficiency-hook confirmed) | interpretation of C2 | `genome_008_quant_stability_g15` | (inherited C2) | Same as C2 — extension across 4 classes means survival is cross-class |
| C4 | kNN-10 passes Gate-1 portability at δ=0.10 on 5 Batch-1 classes (adds DeepSeek reasoning + Falcon hybrid) — Level-1 threshold satisfied at the Gate-1 portion | 🟡 5-class Level-1 threshold (Gate-1 portion only) | `genome_009_stim_resample_5class` + `genome_010_falcon_n4000_tips_level1` | `genome_knn_k10_portability_2026-04-21.md` (within-scope; Falcon required n=4000 SE-halving) | `results/gate1/stim_resample_n2000_seeds42_123_456_5class.json` + `results/gate1/stim_resample_n4000_seeds42_123_456_falcon.json` |
| C5 | PR_uncentered's 5/5 G1.3 pass is a DC-activation-mean artifact, not a substantive geometric claim | demoted ⚪ diagnostic | (self-analysis committed in `0c505dc`) | n/a | Per-system values 1.01-1.60 vs PR_centered 13-44 → ratio 13-39× means DC eigenvector captures ≥95% of uncentered covariance |
| C6 | ID (TwoNN, MLE-k10) is architecture-dominated, not learned-geometry: fails Gate-1 G1.3 on all systems AND fails neg-control | ⚪ diagnostic | `genome_004_neg_control` + `genome_006_stim_resample_g13` + `genome_007` + `genome_009` | `genome_id_portability_2026-04-21.md` LOCKED (now superseded by focused kNN prereg) | 6% trained-vs-untrained gap on Qwen3 at depth 0.25; fails G1.3 all δ |
| C7 | Gate-2 derivation predicts functional form C(X,k) = α_d (1 − β_d·κ·k^(2/d_int))₊ + O(n^(-1/2)) from Laplace-Beltrami convergence | theoretical framework, not empirical claim | n/a (derivation) | `research/derivations/knn_clustering_universality.md` LOCKED at 62338b8 | Laplacian Eigenmaps (Belkin-Niyogi 2003) + Diffusion Maps (Coifman-Lafon 2006) |
| C8 | kNN-10 passes Gate-1 portability at δ=0.10 on 7/8 classes across 5 training objectives (CLM + reasoning + linear-attn + MLM + contrastive-text + self-sup-ViT + contrastive-vision; Falcon narrow-fail at n=2000 tips at n=4000) | 🟡 extended portability (8 classes, 5 objectives) | `genome_011_8class_batch2` + `genome_010_falcon_n4000_tips_level1` | `genome_knn_k10_batch2_2026-04-21.md` **LOCKED at 3e8d395** (2026-04-21) | `results/gate1/stim_resample_n2000_8class_full.json` |
| C9 | kNN-10 local-neighborhood subspace is CAUSALLY load-bearing on Qwen3 mid-depth (ablation smoke) | Gate-2 G2.4 smoke evidence | `genome_012_g24_causal_smoke_qwen3` | `genome_knn_k10_causal_2026-04-21.md` LOCKED at 03da4d5 | topk λ=1 +55% NLL vs random 0.7% / pca 8.3% — specificity ratios 79× / 6.7× |
| C10 | The architecture-prior carrying capability in random-init Llama-3 is LOCALIZED to attention + width + residuals; MLP and excess depth contribute negligibly | 🟡 single-family decomposition (Llama-3 derivatives only) | `genome_138_arch_prior_decomposition` | (no LOCKED prereg — predates §4.1 enforcement on this thread) | PASS verdict in ledger; quantified gaps per ablation |
| C11 | A 3-layer Llama-3 with hidden=384 and ZERO MLP (minimal_3L) matches the MLP-equipped baseline of equal training FLOPs at 30M params, within seed-noise on C4 + WikiText-103 | 🟡 same-architecture-family efficiency match | `genome_141_minimal_prior_capability` | (no LOCKED prereg) | C4 gap +0.12pp (within std 0.08pp), OOD gap −0.05pp (within std 0.05pp) |
| C12 | Architecture-prior win is SCALE-MONOTONIC: at 100M and 200M params, minimal beats baseline by ~0.8pp top-1 on C4 + OOD, and by +0.73pp on HellaSwag (capability-grade) | 🟡 single-family scale-monotonic | `genome_146_matched_flops_bigdata_100m` + `genome_147_matched_flops_200m` + `genome_148_hellaswag_capability` | (no LOCKED prereg) | C4 gaps +0.82/+0.79pp, OOD +0.77/+0.78pp, HellaSwag +0.73pp — all PASS verdicts. Statistical strength ~2σ per-scale; replication strength is the load-bearing argument. |
| C13 | Architecture-prior win is NOT a hyperparameter-tuning artifact: when each arm tunes its own LR (best-vs-best across 4-LR sweep), minimal still beats baseline +0.65pp C4 / +0.52pp OOD | 🟡 HP-robust within well-behaved LR basin | `genome_151_arm_specific_lr` | `research/prereg/genome_153_mlp_depth_factorial_2026-04-26.md` (mechanism follow-up) | baseline_best=lr2e-4@18.34%, minimal_best=lr3e-4@18.99% on C4 |

---

## 2. Provisional claims pending evidence

These claims appear in session docs / roadmap text but are NOT yet backed by
a LOCKED prereg + passing ledger entry. Treat as aspirational until promoted.

| # | Provisional claim | Missing piece | Prereg status |
|---|---|---|---|
| P1 | kNN-10 causally load-bearing on ≥2/3 systems at ≥2/3 depths (2-of-3: Qwen3, RWKV, DINOv2) | G2.4 FULL GRID running now; smoke on Qwen3 passed decisively (C9) | `genome_knn_k10_causal_2026-04-21.md` LOCKED at 03da4d5 — awaits full grid to produce final verdict |
| P2 | kNN-10 functional form is pooled-universal (ΔBIC(per-system − pooled) > 10) | Extended k-sweep {5, 10, 20, 30} + hierarchical fit | `genome_knn_k10_hierarchical_2026-04-21.md` STAGED |
| P3 | kNN-10 extends to encoder / contrastive training objectives (BERT MLM + MiniLM contrastive + CLIP vision contrastive) | Batch-2 G1.3 sweep | `genome_knn_k10_batch2_2026-04-21.md` STAGED |
| P4 | kNN-10 on biological neural populations matches kNN-10 on DINOv2 under same stimulus set | G2.5 Allen Neuropixels pipeline | Not yet drafted |
| P5 | kNN-10 is language-invariant (English ↔ multilingual C4) | Multilingual extraction + G1.3 | Not yet drafted |
| P6 | Architecture-prior win persists at long horizon (50k vs 25k steps at 200M) — i.e., is not a short-horizon compute-optimality artifact | g152 long-horizon crossover (running 2026-04-26) | (no LOCKED prereg yet — pre-staged code, no claim file) |
| P7 | The architecture-prior win is mechanistically driven by residual-branch density, not by MLP-as-special | g153 2×2 mlp×depth factorial | `research/prereg/genome_153_mlp_depth_factorial_2026-04-26.md` LOCKED |
| P8 | A frozen-teacher logit-distillation signal (top-k=64 KL) lifts minimal_3L_30M student top-1 by ≥0.30pp over CE-only training (smoke test of g155 production pipeline) | g154 distillation smoke | `research/prereg/genome_154_distillation_smoke_2026-04-26.md` LOCKED |
| P9 | A first-principles derivation exists for WHY attention + width + residuals beat MLP at matched compute (axis being explored: information-theoretic, statistical-mechanics, rate-distortion, spectral) | Codex first-principles derivation consult fired 2026-04-26 (`codex_outputs/first_principles_derivation.md`); experiment to follow | Not yet drafted |

---

## 3. Explicitly rejected / demoted claims

| # | Claim that was tested and FAILED | Demotion | Evidence |
|---|---|---|---|
| R1 | ID (TwoNN, MLE) is a coordinate | ⚪ diagnostic | C6 above |
| R2 | kNN-k5 is Gate-1 portable | ⚪ diagnostic | Fails G1.3 on 3/4 text at n=2000 |
| R3 | PR_centered is a coordinate | ⚪ diagnostic | Opposite-sign trajectory across feedforward vs recurrent (genome_003); fails G1.3 |
| R4 | PR_uncentered is a coordinate | ⚪ diagnostic | DC-artifact per C5 |
| R5 | CKA is a cross-class primitive | ⚪ diagnostic (pre-atlas decision) | PC-dominance + scale-confound per Aristotelian-View 2026 |
| R6 | Linear alignment (SVCCA, Procrustes) is cross-arch | ⚪ diagnostic (prior lit) | Fails cross-arch (p=0.82 in published work) |

---

## 4. Audit rule

Every commit that adds a claim to README / MANIFESTO / WIKI MUST update §1 or
§2 of this table in the same commit. Codex's Cross-System Auditor at the PR
gate checks for orphaned claims.

A claim is "orphaned" if it appears in user-facing copy but not in §1 or §2.
