# MC006 Parametric Fact Override V17 Pre-Output Steering Stress Status

Status: additive steering on the V16 early direction failed the causal stress
gate.

Date: 2026-07-01

## Artifact

- preregistration:
  `research/prereg/MC006_PARAMETRIC_FACT_OVERRIDE_V17_PRE_OUTPUT_STEERING_STRESS.md`
- runner:
  `code/mc006_parametric_fact_override_v17_pre_output_steering_stress.py`
- source V14 behavior artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v14_parser_normalized_20260701T000538.json`
- source V14 SHA256:
  `7596e8c6869cfcb7413e9bbe8d9b6324b2ed3115021bbb1bf7091059cbac54c5`
- source V16 signature artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v16_pre_output_position_signature_20260701T002113.json`
- source V16 SHA256:
  `6c99826028de85d9b03b22cd8c467ccc42483e1b80ca4f1bbcab6fc67074eb31`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v17_pre_output_steering_stress_20260701T003620.json`
- result SHA256:
  `d6033ff8a685292341818a95b33c1c19ae3f799afdb013fee8ade43a910e2437`

## Verdict

MC006 V17 does not support causal control of the V16 lead-time signature by
simple additive residual steering.

The diagnostic class is:

```text
intervention_failed
```

V17 is a negative causal stress test. It does not erase V16's predictive
lead-time signal, but it blocks the simple additive steering route from that
signal to behavior control.

## Source Checks

Source checks passed:

- V14 was the parser-normalized generated behavior substrate;
- V16 was the lead-time signature artifact;
- V16 selected `after_mapping_line/layer_4`;
- V16's source hash matched V14;
- V17 structural checks matched V15/V16:
  - 30 binary rows;
  - 10 side rows;
  - 21 true-answer rows;
  - 9 override-answer rows;
  - holdout: 6 true and 2 override.

Current-run baseline generation reproduced the frozen V14 holdout labels:

- 8/8 holdout primary rows matched;
- mismatches: 0.

## Intervention

V17 fit the V16-style direction on non-holdout primary rows at:

- position: `after_mapping_line`;
- layer: 4.

It converted that direction into residual deltas calibrated to shift the
standardized signature score by:

- 1.0;
- 2.0;
- 4.0.

For each dose, V17 ran:

- `plus_selected`;
- `minus_selected`;
- `random_selected`;
- `plus_wrong_position`;
- `plus_wrong_layer`;
- `baseline`.

The hook injected only when the selected prompt token existed, so cached
generation steps were not blanket-steered.

## Holdout Results

Dose summary on holdout primary rows:

| Dose | Plus Delta | Minus Delta | Selected Abs Effect | Max Control Abs Effect | Control Gap | Predicted Label Changes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | +0.0469 | +0.0625 | 0.0547 | 0.0469 | +0.0078 | 0 |
| 2.0 | +0.0625 | -0.0156 | 0.0391 | 0.0625 | -0.0234 | 0 |
| 4.0 | +0.1250 | -0.1406 | 0.1328 | 0.1875 | -0.0547 | 0 |

The strongest directional plus/minus dose was 4.0, but it still failed:

- `plus_selected` moved the margin only +0.1250, below the +0.25 floor;
- `minus_selected` moved the margin only -0.1406, above the -0.25 floor;
- max control absolute effect was 0.1875, larger than the selected-arm average
  absolute effect of 0.1328;
- no holdout generated label changed in the predicted direction.

The preregistered best dose by selected/control gap was 1.0, but it also
failed because `minus_selected` moved in the wrong direction on average.

## Side Rows

Side rows did not show broad parse corruption:

- max side-label changes under selected plus/minus arms at any dose: 1/10;
- preregistered side-effect ceiling: 3/10.

This is useful but not enough to rescue the intervention. The primary holdout
causal criteria failed.

## Criteria

| Criterion | Result |
| --- | --- |
| source checks pass | pass |
| baseline reproduces at least 6/8 holdout labels | pass: 8/8 |
| `plus_selected` increases margin by at least 0.25 | fail: max +0.1250 |
| `minus_selected` decreases margin by at least 0.25 | fail: max -0.1406 |
| selected absolute effect beats controls by at least 0.25 | fail |
| generated labels change in predicted direction or margin-only class applies | margin-only possible, but margin criteria failed |
| side rows avoid broad parse corruption | pass |
| not globally output-confounded | fail by V16 premise |

## Interpretation

V17 turns V16 from "early hidden signal" into a sharper map:

- V16: the model carries a pre-output hidden trace of the eventual
  true-versus-override behavior.
- V17: simply adding/subtracting that direction at the selected early
  position/layer does not reliably control the final answer on holdout.

The likely interpretations are:

- the V16 direction is predictive but not a simple additive control vector;
- the causal route from the fake-mapping boundary to the final answer is
  distributed across later positions/layers;
- the final output interface is strong enough that small early residual
  perturbations do not cross behavioral margins;
- stronger or non-additive interventions would need new preregistration and
  controls.

MC006 remains:

- behavior-supported by V14;
- lead-time-signature-supported by V16;
- additive-steering-failed by V17;
- not intervention-ready.
