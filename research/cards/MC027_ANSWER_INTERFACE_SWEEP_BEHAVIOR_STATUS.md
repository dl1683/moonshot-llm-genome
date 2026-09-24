# MC027 Answer-Interface Sweep Behavior Status

Status: smoke_only.

Observed pattern: `answer_interface_substrate_candidate`.

Date: 2026-07-01

## Artifact

- runner:
  `code/mc027_answer_interface_sweep.py`
- result:
  `results/cards/MC027/mc027_answer_interface_sweep_behavior_20260701T154841.json`

## Verdict

MC027 is an answer-interface diagnostic. It measures whether output
format changes the bridge behavior before any hidden-state probe is
allowed.

## Selected Interface Config

- selected config: `bare_integer`
- selection key: `[1.0, 0.95, 0.9625, 1.0, 0]`

## Gate Criteria

| Criterion | Value |
| --- | --- |
| `smoke_mode` | `true` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `false` |
| `source_disjoint_holdout` | `true` |
| `selected_prompt_audit_passed` | `true` |
| `prompts_have_no_status_lexemes` | `true` |
| `any_interface_gate_passed` | `true` |
| `interface_gate_by_variant` | `{"answer_prefix": false, "bare_integer": true, "json_answer": false, "letter_choices": false, "numeric_options": false}` |
| `interface_atomic_control_range` | `1.000` |
| `interface_null_range` | `0.925` |
| `interface_operation_atomic_range` | `0.950` |
| `interface_atomic_control_range_at_least_50p` | `true` |
| `interface_null_range_at_least_50p` | `true` |
| `interface_operation_atomic_range_at_least_30p` | `true` |
| `candidate_and_output_margins_reported` | `false` |
| `selected_familiar_lookup_at_least_90p` | `true` |
| `selected_atomic_control_at_least_85p` | `true` |
| `selected_answer_absent_unknown_at_least_90p` | `true` |
| `selected_rule_absent_unknown_at_least_90p` | `true` |
| `selected_operation_local_at_least_85p` | `true` |
| `selected_operation_atomic_at_least_85p` | `true` |
| `selected_all_panels_parseable_at_least_95p` | `true` |
| `all_controls_passed` | `true` |

## Selected Panels

| Panel | Rows | Parseable | Local Rate | Atomic Rate | Unknown Rate | Other Rate | Expected Correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `familiar_interface_lookup` | 40 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| `atomic_interface_control` | 40 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| `operation_local_interface_conflict` | 40 | 1.000 | 0.975 | 0.025 | 0.000 | 0.000 | 0.975 |
| `operation_atomic_interface_conflict` | 40 | 1.000 | 0.025 | 0.950 | 0.000 | 0.025 | 0.950 |
| `operation_rule_absent_interface_null` | 40 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| `answer_absent_interface_null` | 40 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |

## Interface Comparison

| Interface | Atomic Control Atomic | Answer-Absent UNKNOWN | Operation-Local Local | Operation-Atomic Atomic | Gate Passed |
| --- | ---: | ---: | ---: | ---: | --- |
| `bare_integer` | 1.000 | 1.000 | 0.975 | 0.950 | `true` |
| `answer_prefix` | 0.000 | 1.000 | 0.850 | 0.275 | `false` |
| `json_answer` | 0.000 | 0.075 | 0.000 | 0.000 | `false` |
| `letter_choices` | 0.367 | 0.717 | 0.833 | 0.158 | `false` |
| `numeric_options` | 0.000 | 1.000 | 0.867 | 0.175 | `false` |

## Claim Boundary

MC027 does not establish a signature, intervention, or mechanism card.
It is a behavior-substrate and interface-law diagnostic.
