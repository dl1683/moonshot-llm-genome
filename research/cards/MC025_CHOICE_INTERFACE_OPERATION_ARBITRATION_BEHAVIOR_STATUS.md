# MC025 Choice-Interface Operation Arbitration Behavior Status

Status: smoke_only.

Observed pattern: `choice_interface_smoke_failed`.

Date: 2026-07-01

## Artifact

- runner:
  `code/mc025_choice_interface_operation_arbitration.py`
- result:
  `results/cards/MC025/mc025_choice_interface_operation_arbitration_behavior_20260701T145555.json`

## Verdict

This is a smoke or partial run. It diagnoses whether constraining
the answer interface repairs operation routing.

## Selected Template

- selected template: `compact_choice_examples`
- selection key: `[0.36666666666666664, 0.15, 0.5291666666666667, 0.975, 0]`

## Gate Criteria

| Criterion | Passed |
| --- | --- |
| `smoke_mode` | `true` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `false` |
| `source_disjoint_holdout` | `true` |
| `selected_prompt_audit_passed` | `true` |
| `prompts_have_no_status_lexemes` | `true` |
| `atomic_number_visible_in_conflict_options` | `true` |
| `synthetic_control_at_least_90p` | `true` |
| `familiar_control_at_least_90p` | `true` |
| `atomic_control_at_least_85p` | `false` |
| `answer_absent_unknown_at_least_90p` | `false` |
| `operation_rule_absent_unknown_at_least_90p` | `true` |
| `operation_local_choice_conflict_at_least_85p` | `true` |
| `operation_atomic_choice_conflict_at_least_85p` | `false` |
| `all_panels_parseable_at_least_95p` | `true` |
| `candidate_and_output_margins_reported` | `true` |
| `all_controls_passed` | `false` |
| `choice_operation_behavior_gate_passed` | `false` |

## Selected Panels

| Panel | Rows | Parseable | Local Rate | Atomic Rate | Unknown Rate | Expected Correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `synthetic_choice_lookup` | 120 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| `familiar_choice_lookup` | 120 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| `atomic_choice_control` | 120 | 1.000 | 0.000 | 0.367 | 0.633 | 0.367 |
| `operation_local_choice_conflict` | 120 | 0.992 | 0.908 | 0.000 | 0.083 | 0.908 |
| `operation_atomic_choice_conflict` | 120 | 0.975 | 0.717 | 0.150 | 0.108 | 0.150 |
| `operation_rule_absent_choice_null` | 120 | 1.000 | 0.083 | 0.000 | 0.917 | 0.917 |
| `answer_absent_choice_null` | 120 | 1.000 | 0.267 | 0.042 | 0.692 | 0.692 |

## Aggregate Conflict

- primary conflict expected-correct rate: 0.529
- operation-local choice local rate: 0.908
- operation-atomic choice atomic rate: 0.150

## Claim Boundary

MC025 is a prompt-visible answer-interface diagnostic. The atomic
number is present in the choices on conflict rows, so this can only
test whether constrained choices repair routing/output behavior. It
does not establish a learned-memory hidden signature, intervention,
or mechanism card.
