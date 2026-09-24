# MC024 Few-Shot Operation Numeric Arbitration Behavior Status

Status: query_operation_atomic_branch_failed.

Observed pattern: `operation_atomic_branch_failed`.

Date: 2026-07-01

## Artifact

- runner:
  `code/mc024_fewshot_operation_numeric_arbitration.py`
- result:
  `results/cards/MC024/mc024_fewshot_operation_numeric_arbitration_behavior_20260701T144033.json`

## Verdict

The few-shot operation behavior gate failed.
Hidden-state work remains forbidden for this route.

## Selected Template

- selected template: `compact_worked_examples`
- selection key: `[1.0, 0.8125, 0.903125, 1.0, -2]`

## Gate Criteria

| Criterion | Passed |
| --- | --- |
| `smoke_mode` | `false` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `true` |
| `source_disjoint_holdout` | `true` |
| `selected_prompt_audit_passed` | `true` |
| `prompts_have_no_status_lexemes` | `true` |
| `synthetic_control_at_least_90p` | `true` |
| `familiar_control_at_least_90p` | `true` |
| `atomic_control_at_least_85p` | `true` |
| `answer_absent_unknown_at_least_90p` | `true` |
| `operation_local_conflict_at_least_85p` | `true` |
| `operation_atomic_conflict_at_least_85p` | `false` |
| `operation_rule_absent_unknown_at_least_90p` | `true` |
| `all_panels_parseable_at_least_95p` | `true` |
| `candidate_and_output_margins_reported` | `true` |
| `all_controls_passed` | `true` |

## Selected Panels

| Panel | Rows | Parseable | Local Rate | Atomic Rate | Unknown Rate | Other Rate | Expected Correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `synthetic_numeric_lookup` | 160 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| `familiar_entity_numeric_lookup` | 160 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| `real_world_atomic_number_control` | 160 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| `operation_local_conflict` | 160 | 1.000 | 0.994 | 0.006 | 0.000 | 0.000 | 0.994 |
| `operation_atomic_conflict` | 160 | 1.000 | 0.019 | 0.812 | 0.000 | 0.169 | 0.812 |
| `operation_rule_absent_conflict` | 160 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| `answer_absent_null` | 160 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |

## Aggregate Conflict

- primary conflict expected-correct rate: 0.903
- operation-local local rate: 0.994
- operation-atomic atomic rate: 0.812

## Claim Boundary

MC024 is a behavior diagnostic for worked-example query operation
handles over prompt-local versus learned atomic-number branches. It
does not establish an internal signature, intervention, or mechanism
card.
