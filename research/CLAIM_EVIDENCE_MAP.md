# CLAIM ↔ EVIDENCE MAP

**Purpose (per CLAUDE.md §9 success heuristic).** Every claim anywhere in the
repo (README, MANIFESTO, WIKI, blog-style copy) maps to exactly one ledger
entry + exactly one locked prereg. If a claim isn't in this table, it should
either be in the table (add it + map it) or deleted (unsubstantiated).

**Last updated.** 2026-04-21 (session T+12h).

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

---

## 2. Provisional claims pending evidence

These claims appear in session docs / roadmap text but are NOT yet backed by
a LOCKED prereg + passing ledger entry. Treat as aspirational until promoted.

| # | Provisional claim | Missing piece | Prereg status |
|---|---|---|---|
| P1 | kNN-10 causally load-bearing (ablating the local-neighborhood subspace degrades downstream loss ≥5%, monotonic in λ, specific vs random-10d and PCA-10 controls) | G2.4 smoke test result + full-grid run | `genome_knn_k10_causal_2026-04-21.md` STAGED |
| P2 | kNN-10 functional form is pooled-universal (ΔBIC(per-system − pooled) > 10) | Extended k-sweep {5, 10, 20, 30} + hierarchical fit | `genome_knn_k10_hierarchical_2026-04-21.md` STAGED |
| P3 | kNN-10 extends to encoder / contrastive training objectives (BERT MLM + MiniLM contrastive + CLIP vision contrastive) | Batch-2 G1.3 sweep | `genome_knn_k10_batch2_2026-04-21.md` STAGED |
| P4 | kNN-10 on biological neural populations matches kNN-10 on DINOv2 under same stimulus set | G2.5 Allen Neuropixels pipeline | Not yet drafted |
| P5 | kNN-10 is language-invariant (English ↔ multilingual C4) | Multilingual extraction + G1.3 | Not yet drafted |

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
