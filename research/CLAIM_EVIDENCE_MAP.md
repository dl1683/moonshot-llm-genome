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
| C11 | At 30M params with **matched STEPS** (4000 each), a 3-layer Llama-3 minimal_3L (hidden=384, no MLP, ~21M params) matches the MLP-equipped baseline (~30M params) within seed-noise on C4 + WikiText-103. Note: this is matched-steps, NOT matched-FLOPs — minimal does FEWER FLOPs per step (smaller model), so the match-without-loss is actually a stronger efficiency claim than the original "matched-FLOPs" framing implied. | 🟡 single-family same-step efficiency match (NOT matched-FLOPs at 30M; this row was corrected per Codex 2026-04-26 adversarial audit) | `genome_141_minimal_prior_capability` | (no LOCKED prereg) | C4 gap +0.12pp (within std 0.08pp), OOD gap −0.05pp (within std 0.05pp) |
| C12 | Architecture-prior win is SCALE-MONOTONIC at 100M and 200M params with matched-step protocols (baseline=4000, minimal=8000), minimal beats baseline by ~0.8pp top-1 on C4 + WikiText-103. **HellaSwag claim demoted:** g148 ran at N=500 with both arms near random chance (25%); per-seed top-1 was 26.0/23.4/25.6 (baseline) vs 25.2/26.8/25.2 (minimal) — the +0.73pp mean gap is within noise. HellaSwag here is directional only, NOT capability-grade. **g152 long-horizon (2026-04-26) attenuates the magnitude:** at 200M with N_TRAIN=131072 and 3 seeds, minimal wins at every checkpoint (4k/8k → 25k/50k) but the gap shrinks monotonically from a +1.60pp C4 / +1.70pp OOD peak (at base=8k, min=16k) to +0.27pp C4 / +0.45pp OOD at final (base=25k, min=50k). **HONEST CAVEAT (Codex 2026-04-26):** final-checkpoint 3-seed paired-gap 95% CIs *include zero* (C4: [-0.42, +0.95]pp; OOD: [-0.06, +0.97]pp). Power-law extrapolation projects gap → 0 by 1B-7B scale. Direction survives long horizon at 200M; **magnitude is regime-dependent and the long-run gap is statistically indistinguishable from zero**. | 🟡 narrow regime-specific empirical effect (Codex §0.1 score: 4/10 — not breakthrough); attenuation is *consistency* evidence not *discrimination* | `genome_146_matched_flops_bigdata_100m` + `genome_147_matched_flops_200m` + `genome_148_hellaswag_capability` + `genome_152_long_horizon_crossover` | (no LOCKED prereg) | C4 gaps +0.82/+0.79pp at short horizon, attenuates to +0.27pp at long horizon (3-seed CIs include zero); replication strength is the load-bearing argument; magnitude framing must explicitly acknowledge attenuation AND non-discriminating CIs. |
| C13 | Architecture-prior win is NOT a hyperparameter-tuning artifact: when each arm tunes its own LR (best-vs-best across 4-LR sweep), minimal still beats baseline +0.65pp C4 / +0.52pp OOD. Caveat: **single-seed best-vs-best on the same eval set** — selecting per-arm best LR introduces a multiple-comparison effect that is not corrected. C13 is robust enough to motivate g153/g156 follow-up but is itself not bulletproof. | 🟡 HP-robust within well-behaved LR basin (single-seed; multiple-comparison uncorrected) | `genome_151_arm_specific_lr` | `research/prereg/genome_153_mlp_depth_factorial_2026-04-26.md` (mechanism follow-up — note: NOT a prereg of g151 itself) | baseline_best=lr2e-4@18.34%, minimal_best=lr3e-4@18.99% on C4 |

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
| ~~P6~~ | ~~Architecture-prior win persists at long horizon (50k vs 25k steps at 200M)~~ — **PROMOTED to C12 row** with attenuating-magnitude finding integrated. Direction survives long horizon; magnitude attenuates 6× from peak to final. | g152 complete 2026-04-26 | (resolved into C12) |
| P7 | The architecture-prior win is mechanistically driven by residual-branch density, not by MLP-as-special | g153 2×2 mlp×depth factorial | `research/prereg/genome_153_mlp_depth_factorial_2026-04-26.md` LOCKED |
| P8 | A frozen-teacher logit-distillation signal (top-k=64 KL) lifts minimal_3L_30M student top-1 by ≥0.30pp over CE-only training (smoke test of g155 production pipeline) | g154 distillation smoke | `research/prereg/genome_154_distillation_smoke_2026-04-26.md` LOCKED |
| P9 | A first-principles derivation exists for WHY attention + width + residuals beat MLP at matched compute (axis being explored: information-theoretic, statistical-mechanics, rate-distortion, spectral) | Codex first-principles derivation consult fired 2026-04-26 (`codex_outputs/first_principles_derivation.md`); experiment to follow | Not yet drafted |
| P10 | Distilled MLP-free student matches >=90% of Qwen3-8B on C3_macro (HellaSwag + PIQA + Winogrande) AND beats teacher TEI/kJ by >=4x AND beats best non-distilled sub-2B baseline by >=1.25x — manifesto's electricity-grade efficiency demo | g155 production distillation + benchmark execution (gated on g154 PASS + wall-power meter acquisition) | `research/prereg/genome_155_edge_benchmark_c3_energy_2026-04-26.md` **LOCKED at 1a00ee1** |
| ~~P11~~ → **C14** | **PROMOTED 2026-04-26 from P11.** g156 PASS_TRANSPORT: Δ_nat=+0.56pp, Δ_shuf=−0.197pp (minimal LOSES on shuffled — sharp inversion), C=+0.757pp. All three pre-stated thresholds cleared cleanly. Theory's predicted inversion observed at 200M with 3 seeds. Codex §0.1 score: 4/10 → 6/10. | 🟡 single-family but cross-axis falsifiable evidence (compute attenuation + data-order inversion) | `genome_156_prefix_destruction_200m` | `research/prereg/genome_156_prefix_destruction_200m_2026-04-26.md` **LOCKED at 848affe**, executed 2026-04-26 with PASS verdict |
| P12 | At g156-PASS checkpoints, mid-layer transport surplus G_l = η̂_l − δ̂_l^mlp > 0 on natural arms AND collapses to ≤ 0 on shuffled arms — direct measurement of the transport budget criterion | g157 layerwise η/δ probe (gated on g156 PASS + saved checkpoints) | `research/programs/post_g156_pass_program.md` §g157 |
| P13 | Architecture advantage is monotone in transport demand: as context length shrinks, the minimal-arm win shrinks then inverts (Spearman ρ(context, Δ_L) ≥ 0.8, Δ_32 ≤ −0.2pp, Δ_256 ≥ +0.5pp) | g158 context-length inversion sweep (gated on g156 PASS) | `research/programs/post_g156_pass_program.md` §g158 |
| P14 | Transport-vs-local causal asymmetry replicates across architecture classes: in pretrained Qwen3, RWKV, Falcon-H1, transport-sublayer lesions hurt natural text more than local-sublayer lesions; gap shrinks on shuffled controls | g159 cross-class causal lesion (gated on g156 PASS) | `research/programs/post_g156_pass_program.md` §g159 |
| P15 | The same natural-vs-shuffled contrast holds at training time in a non-transformer (RWKV): transport-heavy variant beats channel-mix-heavy on natural but not shuffled c4 | g161 RWKV training extension (gated on g156 PASS + g159) | `research/programs/post_g156_pass_program.md` §g161 |
| P16 | The transport principle is a model-selection rule: at matched inference FLOPs and matched distillation budget, a transport-heavy student beats a local-heavy student on C3_macro and CtQ_90 (manifesto end-goal — capability transfer + efficiency) | g160 transport-guided student comparison (gated on g156-g159 PASS) | `research/programs/post_g156_pass_program.md` §g160 |

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

## 3.5. Audit findings carried forward (open issues)

These are weaknesses in the C10-C13 chain identified by the Codex 2026-04-26 adversarial review (`codex_outputs/adversarial_kill_arch_prior.md`) that have NOT yet been patched but are tracked here so future work can address them:

- **A1.** OOD eval in g141..g151 uses Wikitext-103 **train** split, not validation/test. Not leakage against C4 training, but the wording "held-out OOD" oversells the discipline. Future runs must use Wikitext val/test.
- **A2.** C4 eval slice is the next chunk of the same shuffled `allenai/c4` train stream as training data. No dedup audit. Could not verify offline; flag for hash-based dedup in g153/g156.
- **A3.** C12's HellaSwag scoring uses separate context+ending tokenization then concatenation, which can distort boundary tokenization. Future capability-grade tests must use full-string tokenization on the entire HellaSwag validation set (not N=500).
- **A4.** Verdict on the thesis as a flagship breakthrough claim per Codex (2026-04-26 adversarial audit): "**not worth publishing as a flagship claim now**" — at the time, was 4/10 §0.1. **UPDATED 2026-04-26 evening:** g152 returned AMBIGUOUS/attenuating (no crossover; gap shrinks 6× from peak to final, 95% CIs at final include zero). g156 returned PASS_TRANSPORT (Δ_nat=+0.56pp, Δ_shuf=−0.20pp, C=+0.76pp; sharp inversion; signal dominates noise). Combined: §0.1 score moves to 6/10 — serious theory lead with cross-axis falsifiable evidence (compute attenuation + data-order inversion). **Closing the gap to "measured design law" requires g157 (η/δ probe) PASS — gated on next experiment.**

Path forward (status 2026-04-26 evening): g156 PASSed; g157 prereg+impl LOCKED, ready to launch when GPU frees. g158/g159/g160/g161 preregs all LOCKED, conditional on their gating. The empirical chain has a derivation backbone candidate; g157+ tests whether that backbone has an internal measurable quantity (η > δ^mlp). If g157 PASSes, §0.1 → 7-8/10 (theory has internal-quantity validation). If g157 KILLs, the mechanism is wrong but g156 PASS still stands as cross-axis empirical evidence.

---

## 4. Audit rule

Every commit that adds a claim to README / MANIFESTO / WIKI MUST update §1 or
§2 of this table in the same commit. Codex's Cross-System Auditor at the PR
gate checks for orphaned claims.

A claim is "orphaned" if it appears in user-facing copy but not in §1 or §2.
