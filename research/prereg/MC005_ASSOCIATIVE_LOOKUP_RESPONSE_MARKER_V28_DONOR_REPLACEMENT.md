# MC005 Associative Lookup Response-Marker V28 Donor Replacement Preregistration

Date: 2026-06-30

## Purpose

V26 found a predictive internal row signature for MC005 row-level all-three
structure: `l20_target_value` reached 0.7933 holdout AUC. V27 then tested a
simple additive residual intervention on that signature and failed: the
`plus_target` arm did not improve all-three row structure, while controls
matched or exceeded the primary row change.

V28 asks whether the failure is specific to additive steering:

> If we replace the layer-20 target source-value activation with an actual
> discovery-row donor activation from all-three rows, does the held-out
> layers-24-26 parent/pair row structure move in the predicted direction?

## Source Artifacts

V28 uses:

- V25 row artifact:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v25_factorial_row_heterogeneity_20260630T205454.json`
- V26 signature artifact:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v26_internal_row_signature_20260630T210810.json`
- V27 negative additive-intervention artifact:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v27_signature_intervention_20260630T212004.json`

V28 recomputes the V26 target-value signature from V25 discovery rows only, to
verify that the source signature remains the same before donor replacement.

## Split

Donor pool:

- V25 lookup seed 233;
- parent-effect rows only;
- labels from V25 `all_three_margin_row`.

Evaluation split:

- V25 lookup seed 239;
- parent-effect rows only.

The answer-absent null holdout remains V25 null seeds 251 and 257.

## Donor Matching

For lookup rows, donor replacement uses discovery rows only. Positive donors
come from discovery all-three rows. Negative donors come from discovery
non-all-three rows. For each held-out row, V28 selects the closest donor within
the requested donor label using only non-hidden layout and margin fields:

- target position group;
- distractor relation;
- baseline margin;
- target slot;
- distractor slot;
- target/distractor token distance.

The label is used only to choose the donor pool. The matching score does not
use held-out all-three labels or held-out hidden states.

## Intervention Arms

Lookup arms:

- `none`: no activation replacement;
- `positive_target`: replace layer-20 `target_value` activation with a matched
  discovery positive `target_value` donor;
- `negative_target`: replace layer-20 `target_value` activation with a matched
  discovery negative `target_value` donor;
- `random_target`: replace layer-20 `target_value` activation with a discovery
  donor sampled without regard to label;
- `positive_final_colon`: replace layer-20 `final_colon` activation with a
  matched discovery positive `final_colon` donor;
- `positive_distractor`: replace layer-20 `distractor_value` activation with a
  matched discovery positive `distractor_value` donor.

For every lookup arm, V28 scores:

- baseline logits under the activation replacement;
- parent `slice_l24_26_all` target-value source mask;
- three leave-one-layer-out pair masks;
- three single-layer masks.

For `positive_target`, V28 also scores parent source controls:

- target-value source mask;
- distractor-value source mask;
- random-value source mask.

## Null Holdout

V28 applies donor replacement to V25 answer-absent null rows from seeds 251 and
257.

Null arms:

- `positive_source_value`: replace layer-20 `source_value` with a positive
  discovery `target_value` donor;
- `positive_final_colon`: replace layer-20 `final_colon` with a positive
  discovery `final_colon` donor;
- `random_source_value`: replace layer-20 `source_value` with a random
  discovery `target_value` donor.

Nulls are clean only if every null arm has:

- baseline target wins at least 75 percent;
- target-win loss exactly 0;
- absolute mean margin delta at most 0.25.

## Success Criteria

V28 supports donor-replacement row control only if all criteria pass:

1. V25, V26, and V27 artifacts validate, and V27 remains a failed additive
   negative control with diagnostic class `plus_target_no_row_effect`.
2. The recomputed V26 source signature remains `l20_target_value` with holdout
   AUC at least 0.70.
3. Holdout `none` baseline is valid: target wins at least 75 percent and at
   least 80 parent-effect rows.
4. `positive_target` increases all-three fraction by at least +0.10 versus
   `none`.
5. `positive_target` lowers median best-pair share by at least 0.05 versus
   `none`.
6. `negative_target` opposes `positive_target`: its all-three fraction is no
   higher than `none` plus 0.03, or its median best-pair share is at least 0.03
   above `positive_target`.
7. Every control arm has less than half the `positive_target` all-three
   fraction gain, or absolute all-three fraction gain at most 0.03.
8. `positive_target` parent source controls remain target-specific: target
   source-mask mean delta is at least 0.50 more negative than distractor and
   random source masks.
9. Donor replacement answer-absent nulls are clean.

## Diagnostic Labels

- `donor_replacement_supported`: all criteria pass.
- `source_artifact_invalid`: V25/V26/V27 validation fails.
- `source_signature_invalid`: source signature does not reproduce.
- `base_parent_invalid`: no-intervention holdout surface is invalid.
- `positive_target_no_row_effect`: positive donor replacement does not increase
  all-three fraction enough.
- `positive_target_no_pair_share_effect`: positive donor replacement does not
  reduce best-pair share enough.
- `negative_not_opposed`: negative donor replacement does not oppose positive
  donor replacement.
- `control_matched_effect`: random or wrong-position replacement matches the
  positive target effect.
- `source_control_failed`: parent source controls fail under
  `positive_target`.
- `null_failed`: answer-absent donor-replacement nulls fail.

## Allowed Interpretation

If V28 passes:

> The V26 row signature was not controllable by additive steering, but actual
> layer-20 donor-state replacement causally changes MC005 row-level structure
> under the required controls.

If V28 fails:

> The V26 signature remains predictive only under the tested intervention
> families. Neither additive residual steering nor matched donor activation
> replacement has proven reliable causal row control.
