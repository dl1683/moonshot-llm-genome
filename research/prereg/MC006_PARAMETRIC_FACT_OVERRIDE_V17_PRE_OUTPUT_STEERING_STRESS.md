# MC006 Parametric Fact Override V17 Pre-Output Steering Stress Preregistration

Date: 2026-07-01

## Purpose

MC006 V16 found a pre-output hidden signal on the V14 matched generated table:
`after_mapping_line/layer_4` reached 1.000 discovery and holdout AUC, beat the
same-position output margin and shuffled-label p95, but remained globally
output-confounded because candidate-score margin and final next-token output
margin also reached 1.000 holdout AUC.

V17 asks a deliberately narrower causal question:

> If we add or subtract the V16 pre-output direction at its selected token and
> layer, does the final answer behavior or final true-minus-override margin move
> in the predicted direction, and do simple controls match the effect?

V17 is a known-confounded causal stress test. It cannot by itself promote MC006
to a mechanism card unless global output-interface controls are also beaten,
which V16 did not support.

## Source Artifacts

V17 uses:

- V14 behavior artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v14_parser_normalized_20260701T000538.json`
- V16 lead-time signature artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v16_pre_output_position_signature_20260701T002113.json`

Required V14 source conditions:

- `run_type == parametric_fact_override_v14_parser_normalized`;
- `diagnostic_class == parser_normalized_generated_substrate_passed`;
- `passed == true`.

Required V16 source conditions:

- `run_type == parametric_fact_override_v16_pre_output_position_signature`;
- `diagnostic_class == leadtime_signal_supported_but_output_global_confounded`;
- `leadtime_supported == true`;
- selected pre-output candidate is `after_mapping_line/layer_4`;
- V16 source artifact hash matches the V14 artifact hash.

## Rows

V17 uses the V14 selected template rows:

- 30 primary binary rows: `true_answer` or `override_answer`;
- 10 side rows: `lure_answer` or `unparsed`.

The direction is fit only on non-holdout primary rows. The primary causal
stress verdict is evaluated on holdout primary rows. Discovery/calibration rows
and side rows are reported as diagnostics and side-effect surfaces.

## Direction Construction

Recompute V16 features from the V14 prompts:

- selected position: `after_mapping_line`;
- selected layer: 4;
- labels: `true_answer` = 1 and `override_answer` = 0;
- training rows: discovery plus calibration primary rows.

Fit the same standardized mean-difference direction used by V16. Convert it
into a raw residual delta calibrated to shift the standardized signature score
by the requested dose.

Default dose sweep:

- 1.0;
- 2.0;
- 4.0.

## Intervention Arms

For each selected dose, run:

- `plus_selected`: add the direction at `after_mapping_line/layer_4`;
- `minus_selected`: subtract the direction at `after_mapping_line/layer_4`;
- `random_selected`: add a deterministic random vector with matched delta norm
  at `after_mapping_line/layer_4`;
- `plus_wrong_position`: add the direction at `after_return_line/layer_4`;
- `plus_wrong_layer`: add the direction at `after_mapping_line/layer_16`.

Also run `baseline` with no hook.

The hook applies only when the selected prompt token exists in the current
forward pass. During cached generation steps, the sequence length no longer
contains that prompt token, so the hook must not inject into every generated
token.

## Measurements

For each row and arm:

- greedy generated completion;
- NFKD strict first-line parsed label;
- final-prompt true first-token minus override first-token logit margin from
  the first generated-token score;
- margin delta versus the current-run `baseline` arm;
- label change versus the current-run `baseline` arm;
- label change versus the frozen V14 source label.

## Causal Stress Criteria

V17 supports a causal stress effect only if all criteria pass on holdout primary
rows for at least one dose:

1. V14 and V16 source checks pass.
2. Baseline current-run holdout labels match the frozen V14 binary labels on at
   least 6/8 holdout primary rows.
3. `plus_selected` increases mean true-minus-override margin by at least 0.25.
4. `minus_selected` decreases mean true-minus-override margin by at least 0.25.
5. The average selected-arm absolute effect beats `random_selected`,
   `plus_wrong_position`, and `plus_wrong_layer` by at least 0.25.
6. At least one generated-label change on holdout primary rows is in the
   predicted direction, or the run is explicitly classified as margin-only.

## Mechanism Promotion Criteria

V17 supports a mechanism-grade control surface only if the causal stress
criteria pass and:

7. generated-label changes are directional, not margin-only;
8. side rows do not show broad parse/label corruption: under the selected dose,
   no more than 3/10 side rows may change parse label under either
   `plus_selected` or `minus_selected`;
9. the effect is not matched by final output-interface controls.

Given V16, criterion 9 is expected to fail. This is intentional: V17 is meant
to document whether the V16 lead-time signature is causally active despite the
known final-output confound.

## Diagnostic Labels

- `pre_output_steering_mechanism_supported`: causal stress and mechanism
  criteria pass.
- `causal_stress_positive_global_output_confounded`: margin and generated-label
  effects are directional, controls do not match, but global output confounding
  blocks mechanism promotion.
- `margin_only_causal_effect_global_output_confounded`: final margins move
  directionally and controls do not match, but generated labels do not.
- `source_artifact_invalid`: source checks fail.
- `baseline_reproduction_failed`: current-run baseline does not reproduce the
  frozen V14 holdout labels sufficiently.
- `intervention_failed`: selected plus/minus arms do not move margins in the
  predicted directions.
- `control_matched`: random, wrong-position, or wrong-layer controls match the
  selected-arm effect.
- `side_effect_failed`: side rows show broad parse/label corruption.
- `mixed_causal_stress_failure`: any other mixed failure.

## Allowed Interpretation

If V17 finds a causal stress effect:

> The V16 early hidden signature is not merely passive: additive residual
> perturbation at the selected early position/layer moves the final answer
> margin and possibly generated labels in the predicted direction. Because V16
> is globally output-confounded, this remains a known-confounded causal stress
> result, not a mechanism card.

If V17 fails:

> The V16 early hidden signature is predictive/monitoring-only under the tested
> additive residual intervention. MC006 remains behavior-supported and
> lead-time-signature-supported, but not causally controllable on this route.
