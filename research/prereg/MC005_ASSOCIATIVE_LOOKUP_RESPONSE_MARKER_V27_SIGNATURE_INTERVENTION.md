# MC005 Associative Lookup Response-Marker V27 Signature Intervention Preregistration

Date: 2026-06-30

## Purpose

V26 found an internal row signature for MC005 row-level all-three structure:
the residual stream at layer 20 on the target source-value token
(`l20_target_value`) predicted all-three rows on a held-out seed with AUC
0.7933, beating output/layout baselines and a shuffled-selection null.

V27 asks whether that signature is causal:

> If we move the `l20_target_value` signature score while preserving prompt
> text, does the layers-24-26 parent/pair row structure move in the predicted
> direction?

## Source Artifacts

V27 uses:

- V25 row artifact:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v25_factorial_row_heterogeneity_20260630T205454.json`
- V26 signature artifact:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v26_internal_row_signature_20260630T210810.json`

V27 recomputes the V26 direction from V25 discovery rows, rather than reading a
stored direction vector.

## Split

Direction training split:

- V25 lookup seed 233

Causal evaluation split:

- V25 lookup seed 239

Only parent-effect lookup rows are eligible.

## Signature Direction

V27 collects baseline residual-stream hidden states at:

- layer 20 output;
- token position `target_value`.

It trains the same discovery-only mean-difference direction as V26:

1. z-score hidden dimensions using discovery rows only;
2. compute all-three positive mean minus non-all-three negative mean;
3. L2-normalize that z-space direction.

To intervene, V27 converts the z-space direction into the minimal raw hidden
delta that changes the V26 signature score by a fixed amount. The preregistered
score shift is:

- `+4.0` for `plus_target`;
- `-4.0` for `minus_target`.

The raw delta is inserted at the layer-20 output on the requested token
position during the prompt forward pass. No prompt text changes.

## Intervention Arms

Primary arms:

- `none`: no residual intervention;
- `plus_target`: add the +4.0 signature-score delta at `target_value`;
- `minus_target`: add the -4.0 signature-score delta at `target_value`.

Controls:

- `random_target`: add a random raw direction with the same L2 norm as the
  +4.0 delta at `target_value`;
- `plus_final_colon`: add the +4.0 signature delta at `final_colon` instead
  of `target_value`;
- `plus_distractor`: add the +4.0 signature delta at `distractor_value`
  instead of `target_value`.

For each lookup arm, V27 scores:

- baseline logits with the residual intervention;
- parent `slice_l24_26_all` target-value source mask;
- three leave-one-layer-out pair masks;
- three single-layer masks.

For the primary `plus_target` arm, V27 also scores parent source controls:

- target-value source mask;
- distractor-value source mask;
- random-value source mask.

## Prediction

The V26 positive signature was oriented toward all-three rows. Therefore:

- `plus_target` should make rows more all-three-like;
- `minus_target` should make rows less all-three-like or at least move
  opposite `plus_target`;
- random and wrong-position controls should be smaller than `plus_target`.

The primary row-structure metrics are:

- all-three fraction among parent-effect rows;
- median best leave-one-layer-out pair share.

## Null Holdout

V27 also applies residual interventions to V25 answer-absent null rows from
seeds 251 and 257.

Null arms:

- `plus_source_value`: +4.0 signature delta at `source_value`;
- `plus_final_colon`: +4.0 signature delta at `final_colon`;
- `random_source_value`: random same-norm delta at `source_value`.

Nulls are clean only if every null arm has:

- baseline target wins at least 75 percent;
- target-win loss exactly 0;
- absolute mean margin delta at most 0.25.

## Success Criteria

V27 supports causal row-signature control only if all criteria pass:

1. V26 source artifacts validate and the recomputed selected direction is
   `l20_target_value`.
2. Holdout `none` baseline is valid: target wins at least 75 percent and at
   least 80 parent-effect rows.
3. `plus_target` increases all-three fraction by at least +0.10 versus `none`.
4. `plus_target` lowers median best-pair share by at least 0.05 versus `none`.
5. `minus_target` opposes `plus_target`: its all-three fraction is no higher
   than `none` plus 0.03, or its median best-pair share is at least 0.03 above
   `plus_target`.
6. Every control arm has less than half the `plus_target` all-three-fraction
   gain, or absolute all-three-fraction gain at most 0.03.
7. `plus_target` parent source controls remain target-specific: the
   target-value source-mask mean delta is at least 0.50 more negative than the
   distractor-value and random-value source-mask mean deltas.
8. Residual answer-absent nulls are clean.

## Diagnostic Labels

- `signature_intervention_supported`: all criteria pass.
- `source_signature_invalid`: V25/V26 validation or direction reconstruction
  fails.
- `base_parent_invalid`: the no-intervention holdout surface is invalid.
- `plus_target_no_row_effect`: `plus_target` does not increase all-three
  fraction enough.
- `plus_target_no_pair_share_effect`: `plus_target` does not reduce best-pair
  share enough.
- `minus_not_opposed`: the negative intervention does not oppose the positive
  intervention.
- `control_matched_effect`: random or wrong-position control matches the
  positive effect.
- `source_control_failed`: parent source controls fail under `plus_target`.
- `null_failed`: answer-absent residual nulls fail.

## Allowed Interpretation

If V27 passes, the current claim becomes:

> MC005 has a causal internal row-signature control: moving the layer-20
> target-source residual signature changes the row-level all-three/pair
> structure inside the reliable layers-24-26 parent surface.

If V27 fails, the current claim remains:

> MC005 has a reliable aggregate layers-24-26 control surface and a predictive
> internal row signature, but the tested signature intervention did not prove
> causal control under the required controls.

