# Post-g156-PASS Experimental Program

**Status:** LOCKED 2026-04-26 by Codex Architecture-Theorist + Scaling-Expert consult.
**Source:** `codex_outputs/post_g156_experimental_program.md`
**Triggers if:** g156 returns PASS_TRANSPORT.
**Goal:** turn the prefix-information transport derivation from a single-test surprise into a measured design law.

## Shared audit-hard protocol (all 5)

- **Splits:** new locked splits — `c4_train_dedup_v2` (allenai/c4 train), `c4_val_dedup_v2` (allenai/c4 validation), `wikitext103_val_v2` (Wikitext-103 *validation*, not train as in g141..g151).
- **Dedup rule:** tokenizer-specific exact 13-token rolling-hash overlap audit; drop any eval window whose 13-gram hash appears in train.
- **Matched-FLOPs:** any train-vs-train comparison uses a frozen analytic FLOP counter and matches total train FLOPs within ±2%.
- **Seeds:** training {42, 7, 13}; inference {42, 123, 456} with bootstrap CIs.
- **Capability claims:** full validation sets (HellaSwag, PIQA, Winogrande), full-string tokenization.

These are the protocols that survive the Codex 2026-04-26 adversarial audit (CLAIM_EVIDENCE_MAP §3.5).

## Sequencing

1. **g157** — η/δ probe on g156 checkpoints. v1 was 91-hr; relocked as PILOT (1-seed, 4 ckpts, ~30 min projected). Running 2026-04-26 21:35.
   - Backup: **g157b** — embedding-layer prefix probe variant (LOCKED; conditional on g157 PILOT KILL/WEAK due to suspected probe-design issue at same-layer prefix).
2. **g158** — context-length inversion sweep (~1.6-2.0 GPU-hr post-fixes). Pre-flight blockers patched (exact-FLOP match, dedup, NaN guard).
3. **g159** — cross-class causal lesion on Qwen3 + RWKV + Falcon-H1 (~0.9-1.6 GPU-hr per Codex). Pre-flight blockers patched (val data, exact streaming PCA).
4. **g161** — RWKV training extension (~2-3 GPU-hr). Stub written; awaiting Codex implementation design (codex_outputs/g161_rwkv_implementation.md, firing).
5. **g160** — transport-guided student comparison (~3+ GPU-hr; pre-flight pending).

g152 and g153 are NOT in this stack (they patch the old empirical chain rather than add new evidence types). g154 (distillation smoke) PASSed and unblocks g160.

## Status as of 2026-04-26 evening

- g154: **PASS** (KD beats scratch +0.59pp top-1; pipeline validated)
- g156: **PASS_TRANSPORT** (Δ_nat=+0.56pp, Δ_shuf=−0.20pp, C=+0.76pp; theory's predicted inversion observed)
- g152: AMBIGUOUS/PARTIAL (long-horizon attenuation observed; final-checkpoint CIs include zero)
- g157 PILOT: running, early data shows G_l < 0 on natural arms (probe-design issue suspected)
- g157b: LOCKED, ready to launch if g157 PILOT confirms probe-design issue
- g158/g159/g160/g161: locked + implemented + Codex-pre-flighted + patched (g158/g159) or awaiting Codex design (g161); ready to launch when GPU frees

## g157 — Layerwise η/δ probe

**Hypothesis:** on natural-text g156 checkpoints, mid-layer transport surplus is positive (η̂_l > δ̂_l^mlp); the surplus collapses on shuffled checkpoints.

**System:** the 12 final g156 checkpoints (now saved per the hygiene patch). Probe at functional depths {0.2, 0.35, 0.5, 0.65, 0.8}; probe data 4096/512/512 windows from `c4_val_dedup_v2`, plus 512 Wikitext-val windows for OOD.

**PASS:** in the minimal-natural arm, mean mid-band G := η̂_l − δ̂_l^mlp ≥ +0.02 nats in ≥2/3 seeds, AND minimal-shuffled mean ≤ 0.00, AND pooled contrast G_nat − G_shuf ≥ +0.03 nats.
**PARTIAL:** same direction, contrast only ≥ +0.015 nats.
**KILL:** no positive natural surplus, OR shuffle leaves surplus unchanged within 0.01 nats.

**Falsifies broader theory:** g156 would remain an input-perturbation phenomenon, not evidence for the budget criterion η_l > δ_l^mlp.

## g158 — Context-length inversion sweep

**Hypothesis:** the architecture advantage is monotone in transport demand. As context shrinks, the minimal win shrinks then inverts.

**System:** exact-FLOP-matched 30M family (`baseline_6L+MLP` vs `minimal_3L_noMLP`), context lengths {32, 64, 128, 256}, seeds {42, 7, 13}, arm-specific LR chosen on a separate validation bank then frozen.

**PASS:** with Δ_L := top1_minimal − top1_baseline, require Spearman ρ(context, Δ_L) ≥ 0.8 AND Δ_32 ≤ −0.2pp AND Δ_256 ≥ +0.5pp AND sign pattern agrees on both C4-val and Wikitext-val.
**PARTIAL:** monotone increase with Δ_256 ≥ +0.3pp but no clean sign flip.
**KILL:** no monotone increase OR minimal wins at all context lengths.

**Falsifies broader theory:** loses the clean inversion axis. Cannot say transport demand is the control variable.

## g159 — Cross-class causal lesion

**Hypothesis:** in distinct text architecture classes, transport-sublayer lesions hurt natural text more than equal-rank local-sublayer lesions; the gap shrinks on order-destroyed controls.

**System:** pretrained Qwen3-0.6B (transformer), RWKV-4-169M (linear-recurrent), Falcon-H1-0.5B (hybrid). At depths {0.25, 0.5, 0.75}, fit top-32 PCA subspaces separately for transport outputs vs local outputs on calibration; project them out on eval. Eval on natural `c4_val_dedup_v2` and token-shuffled matched controls.

**PASS:** in 3/3 classes, natural ΔNLL_transport / ΔNLL_local ≥ 1.5; on shuffled controls that ratio falls by ≥40% in ≥2/3 classes.
**PARTIAL:** 2/3 classes pass, OR natural ratio only ≥ 1.25.
**KILL:** local lesions match or exceed transport lesions in ≥2/3 classes.

**Falsifies broader theory:** principle is not class-general; shrinks back to a Llama-family story.

## g161 — RWKV training extension

**Hypothesis:** the natural-vs-shuffled contrast appears in a transport architecture outside standard self-attention. In small RWKV, removing channel-mix and spending budget on more time-mix/depth helps natural but not shuffled text.

**System:** custom small RWKV family with exact-FLOP matching: baseline `12L h512 + channel-mix` vs transport-heavy `18L h512 no-channel-mix`. Seeds {42, 7, 13}. Natural vs token-shuffled C4. Eval on C4-val and Wikitext-val.

**PASS:** Δ_nat ≥ +0.3pp AND Δ_shuf ≤ +0.1pp AND contrast Δ_nat − Δ_shuf ≥ +0.3pp on both eval sets.
**PARTIAL:** Δ_nat ≥ +0.2pp AND contrast ≥ +0.2pp.
**KILL:** no contrast or reversed contrast.

**Falsifies broader theory:** derivation does not survive beyond transformer-like decoders, even if pretrained-lesion result in g159 does.

## g160 — Transport-guided student comparison

**Hypothesis:** under matched student inference cost AND matched distillation budget, a transport-heavy student beats a local-heavy student on held-out capability AND step-to-target efficiency.

**System:** teacher Qwen3-0.6B; two students at matched inference FLOPs within ±2% — transport-heavy `6L_noMLP_wide` vs local-heavy `4L_MLP` at ~50-70M; seeds {42, 7, 13}; 8192 C4 train windows; full HellaSwag, PIQA, Winogrande validation at end.

**PASS:** C3_macro_transport − C3_macro_local ≥ +1.0pp AND CtQ_90 ≥ 20% lower for transport-heavy in 2/3 seeds. If wall-power meter available, also TEI/kJ_transport / TEI/kJ_local ≥ 1.25.
**PARTIAL:** C3 gain ≥ +0.5pp OR only CtQ_90 lands.
**KILL:** local-heavy ties or wins on C3_macro and convergence.

**Falsifies broader theory:** explains small-budget pretraining but doesn't guide capability-transfer / efficiency design (which is the manifesto end-goal).

## η/δ Probe Primitive (for g157)

**What it measures:** for each layer, how much next-token information is still missing because prefix info hasn't been transported into the current-token state, vs how much extra info can still be unlocked by a token-local nonlinear decoder.

**Estimator:**

1. Collect `(h_t^l, prefix_{<t}, x_{t+1})` tuples from a frozen checkpoint on deduped text.
2. Split by sequence into probe-train, probe-val, probe-test.
3. Train three equal-budget probes per layer:
   - `q_lin(y|h)`: linear softmax probe
   - `q_local(y|h)`: 2-layer token-local MLP probe
   - `q_prefix(y|h, prefix)`: one-head cross-attention probe over prefix tokens + h
4. Compute held-out cross-entropies `CE_lin`, `CE_local`, `CE_prefix`.
5. Define:
   - `δ̂_l^mlp = CE_lin − CE_local`
   - `η̂_l = CE_local − CE_prefix`
6. Report `G_l = η̂_l − δ̂_l^mlp` and `R_l = η̂_l / max(δ̂_l^mlp, eps)`.

**Falsifies transport theory if:** in the very regime where g156 PASSes, mid-layer natural checkpoints don't show G_l > 0; OR shuffled checkpoints fail to push G_l back to ≤ 0; OR R_l doesn't predict the sign change in g158.

**Cost:** activation dump + tiny probe training, ~2 GPU-hr at the proposed layer subsampling.

## Claim available after all 5 PASS

> Within the tested scope of 256-token English autoregressive language modeling on consumer-scale budgets, prefix-information transport is a measured design law, not a post hoc story. A preregistered layerwise probe shows η̂_l > δ̂_l^mlp exactly in the regimes where transport-heavy architectures win; reducing transport demand by prefix destruction or short context produces the predicted inversion; the same transport-vs-local causal asymmetry appears across transformer, recurrent, and hybrid text architectures; and the law selects the better matched-cost distilled student.

This is the claim that survives an adversarial Codex audit and that the architecture-prior chain alone (C10-C13) cannot make.
