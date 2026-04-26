# Post-g156-PASS Program

Assuming `g156` is `PASS_TRANSPORT`, the next program should stop trying to add one more Llama ablation row and instead do five things: measure the proposed quantity directly, verify a unique prediction of the theory, extend it beyond one family, then show it selects a better transfer/efficiency design.

Immediate hygiene, not counted as a new experiment: if `g156` PASSes without final checkpoints saved, rerun the identical locked script once to emit final weights for all 12 cells. Experiment 1 needs those checkpoints.

Two things I would **not** count in the top-5 theory stack: `g152` and `g153`. They still matter, but they mostly patch the old empirical chain rather than add a new evidence type.

## Shared audit-hard protocol for all 5

- Use new locked splits: `c4_train_dedup_v2` from `allenai/c4` train, `c4_val_dedup_v2` from `allenai/c4` validation, `wikitext103_val_v2` from Wikitext-103 validation.
- Dedup rule: tokenizer-specific exact 13-token rolling-hash overlap audit; drop any eval window whose 13-gram hash appears in train.
- Any train-vs-train comparison must use a frozen analytic FLOP counter and match total train FLOPs within `±2%`.
- Training seeds default to `{42, 7, 13}`.
- Inference-only experiments use stimulus seeds `{42, 123, 456}` plus bootstrap CIs over sequences.
- Any capability claim must use full validation sets with full-string tokenization: HellaSwag, PIQA, Winogrande.

## A. Ranked follow-ups

1. **g157: Layerwise `η/δ` probe on g156 checkpoints**

Hypothesis: on natural-text `g156` checkpoints, the mid-layer transport surplus is positive, `η̂_l > δ̂_l^mlp`, and this surplus collapses on shuffled checkpoints.

System: the 12 final `g156` checkpoints; probe at functional depths `{0.2, 0.35, 0.5, 0.65, 0.8}`; probe data `4096/512/512` windows from `c4_val_dedup_v2`, plus `512` Wikitext-val windows for OOD confirmation.

PASS: in the minimal natural arm, mean mid-band `G := η̂_l - δ̂_l^mlp >= +0.02` nats in at least `2/3` seeds, and the corresponding minimal shuffled mean is `<= 0.00`, with pooled contrast `G_nat - G_shuf >= +0.03` nats.  
PARTIAL: same direction, but contrast only `>= +0.015` nats.  
KILL: no positive natural surplus, or shuffle leaves the surplus unchanged within `0.01` nats.

Falsifies the broader theory if it fails: `g156` would remain an input-perturbation phenomenon, not evidence for the budget criterion `η_l > δ_l^mlp`.

Compute: `1.5-2.0` GPU-hours.

Why this advances theory: this is the missing measurement primitive. Without it, you still have a good killer test; with it, you have a measurable internal quantity.

2. **g158: Context-length inversion sweep**

Hypothesis: the architecture advantage is monotone in transport demand. As context length shrinks, the minimal win should shrink and then invert.

System: exact-FLOP-matched 30M family rerun of `g141`-style pair, `baseline_6L+MLP` vs `minimal_3L_noMLP`, context lengths `{32, 64, 128, 256}`, seeds `{42,7,13}`, one arm-specific LR choice on a separate validation bank, then frozen for the full sweep.

PASS: letting `Δ_L := top1_minimal - top1_baseline`, require `Spearman rho(context, Δ_L) >= 0.8`, `Δ_32 <= -0.2pp`, `Δ_256 >= +0.5pp`, and the sign pattern agrees on both C4-val and Wikitext-val means.  
PARTIAL: monotone increase with `Δ_256 >= +0.3pp`, but no clean sign flip.  
KILL: no monotone increase, or minimal wins at all context lengths.

Falsifies the broader theory if it fails: the theory loses its clean inversion axis. You could no longer say transport demand is the control variable.

Compute: `1.0-1.5` GPU-hours.

Why this advances theory: this is the sharpest prediction that `g156` alone does not buy you. A real theory predicts where it should stop working.

3. **g159: Cross-class causal transport-vs-local lesion**

Hypothesis: in distinct text architecture classes, transport sublayer lesions should hurt natural text more than equal-rank local sublayer lesions, and that gap should shrink on order-destroyed controls.

System: pretrained `Qwen3-0.6B` (transformer), `RWKV-4-169M` (linear-recurrent), `Falcon-H1-0.5B` (hybrid). At functional depths `{0.25, 0.5, 0.75}`, fit top-32 PCA subspaces separately for transport outputs and local outputs on a calibration split, then project them out on eval. Eval on natural `c4_val_dedup_v2` and token-shuffled matched controls.

PASS: in all `3/3` classes, natural-text `ΔNLL_transport / ΔNLL_local >= 1.5`, and on shuffled controls that ratio falls by at least `40%` in at least `2/3` classes.  
PARTIAL: `2/3` classes pass, or the natural ratio is only `>= 1.25`.  
KILL: local lesions match or exceed transport lesions in `>=2/3` classes.

Falsifies the broader theory if it fails: the principle is probably not class-general; it shrinks back toward a Llama-family story.

Compute: `0.5-0.8` GPU-hours.

Why this advances theory: it extends the claim across architecture classes without new training machinery and does so causally, not correlationally.

4. **g160: Transport-guided student comparison**

Hypothesis: under matched student inference cost and matched distillation budget, a transport-heavy student should beat a local-heavy student on held-out capability and step-to-target efficiency.

System: teacher `Qwen3-0.6B`; two students at matched inference FLOPs within `±2%`: a transport-heavy `6L_noMLP_wide` student and a local-heavy `4L_MLP` student at roughly `50M-70M`; seeds `{42,7,13}`; `8192` C4 train windows; full HellaSwag, PIQA, Winogrande validation at the end.

PASS: `C3_macro_transport - C3_macro_local >= +1.0pp` and `CtQ_90` is at least `20%` lower for the transport-heavy student in `2/3` seeds. If a wall-power meter is already available, also require `TEI/kJ_transport / TEI/kJ_local >= 1.25`.  
PARTIAL: `C3` gain `>= +0.5pp` or only the `CtQ_90` criterion lands.  
KILL: local-heavy ties or wins on `C3_macro` and convergence.

Falsifies the broader theory if it fails: the theory may explain small-budget pretraining, but it does not yet guide capability-transfer / efficiency design.

Compute: `2.5-3.5` GPU-hours.

Why this advances theory: it cashes the law out into a model-selection rule for the manifesto end-goal instead of leaving it as explanation-only.

5. **g161: Direct non-transformer training extension in RWKV**

Hypothesis: the same natural-vs-shuffled contrast should appear in a transport architecture outside standard self-attention. In small RWKV, removing channel-mix and spending budget on more time-mix/depth should help on natural text but not on shuffled text.

System: custom small RWKV family with exact-FLOP matching, baseline `12L h512 + channel-mix`, transport-heavy variant `18L h512 no-channel-mix`, seeds `{42,7,13}`, natural vs token-shuffled C4, eval on C4-val and Wikitext-val.

PASS: `Δ_nat >= +0.3pp`, `Δ_shuf <= +0.1pp`, and contrast `Δ_nat - Δ_shuf >= +0.3pp` on both eval sets.  
PARTIAL: natural gain `>= +0.2pp` and contrast `>= +0.2pp`.  
KILL: no contrast or reversed contrast.

Falsifies the broader theory if it fails: the derivation probably does not survive beyond transformer-like decoders, even if the pretrained lesion result in `g159` does.

Compute: `2.0-3.0` GPU-hours.

Why this advances theory: this is the direct training-time architecture-class extension a hostile auditor will eventually demand. I rank it fifth only because it has the highest engineering risk.

## B. Sequencing

1. Run `g157` first. If you cannot measure `η̂_l` and `δ̂_l^mlp` directly after a `g156` PASS, you still do not have the load-bearing object of the theory.

2. Run `g158` second. It is cheap, clean, and uniquely diagnostic. A theory that only explains one ablation but cannot predict an inversion regime is still fragile.

3. Run `g159` third. After the probe and inversion land in Llama, extend the causal asymmetry across classes using pretrained models before you spend time on new training code.

4. Run `g161` fourth. Only after `g159` do the RWKV engineering costs make sense; by then you know the cross-class direction is probably real.

5. Run `g160` fifth. This is the cash-out experiment. Once the theory has survived 1-4, ask whether it selects the better student for transfer/efficiency.

## C. First new measurement primitive: the `η/δ` probe

What it measures: for each layer, how much next-token information is still missing because prefix information has not yet been transported into the current-token state, versus how much extra next-token information can still be unlocked by a token-local nonlinear decoder.

Concrete estimator:

- Collect tuples `(h_t^l, prefix_{<t}, x_{t+1})` from a frozen checkpoint on deduped text.
- Split by sequence into probe-train, probe-val, probe-test.
- Train three equal-budget probes per layer:
- `q_lin(y|h)`: linear softmax probe.
- `q_local(y|h)`: 2-layer token-local MLP probe.
- `q_prefix(y|h,prefix)`: one-head cross-attention probe that sees the actual prefix tokens or their embeddings plus `h`.
- Compute held-out cross-entropies `CE_lin`, `CE_local`, `CE_prefix`.
- Define `δ̂_l^mlp = CE_lin - CE_local`.
- Define `η̂_l = CE_local - CE_prefix`.
- Report `G_l = η̂_l - δ̂_l^mlp` and `R_l = η̂_l / max(δ̂_l^mlp, eps)`.

What would falsify the transport theory: if, in the very regime where `g156` PASSes, the mid-layer natural checkpoints do not show `G_l > 0`, or shuffled checkpoints fail to push `G_l` back to `<= 0`, or `R_l` does not correctly predict the sign change in `g158`.

Compute cost: one activation dump plus tiny probe training. At the proposed layer subsampling, `1.5-2.0` GPU-hours for the full `g157` grid.

## D. Claim you could make after all 5 PASS

You could say:

**Within the tested scope of 256-token English autoregressive language modeling on consumer-scale budgets, prefix-information transport is a measured design law, not a post hoc story. A preregistered layerwise probe shows `η̂_l > δ̂_l^mlp` exactly in the regimes where transport-heavy architectures win; reducing transport demand by prefix destruction or short context produces the predicted inversion; the same transport-vs-local causal asymmetry appears across transformer, recurrent, and hybrid text architectures; and the law selects the better matched-cost distilled student.**

That is a stronger claim than today’s. Today you have “one Llama-family matched-budget win plus one killer ablation if `g156` passes.” After these five, you have a cross-condition, cross-class, internally measured, inversion-predictive theory.