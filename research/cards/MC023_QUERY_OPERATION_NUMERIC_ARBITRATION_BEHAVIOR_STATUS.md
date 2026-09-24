# MC023 Query-Operation Numeric Arbitration Behavior Status

Status: smoke_only.

Observed pattern: `operation_local_branch_failed`.

Date: 2026-07-01

## Artifact

- runner:
  `code/mc023_query_operation_numeric_arbitration.py`
- result:
  `results/cards/MC023/mc023_query_operation_numeric_arbitration_behavior_20260701T135542.json`

## Verdict

This is a smoke or partial run. It diagnoses whether a
query-level operation handle can repair local-versus-learned
arbitration without row-level source-status text.

## Selected Template

- selected template: `query_first_operation`
- selection key: `[1.0, 0.7, 0.7375, 1.0, -2]`

## Gate Criteria

| Criterion | Passed |
| --- | --- |
| `smoke_mode` | `true` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `false` |
| `source_disjoint_holdout` | `true` |
| `selected_prompt_audit_passed` | `true` |
| `prompts_have_no_status_lexemes` | `true` |
| `synthetic_control_at_least_90p` | `true` |
| `familiar_control_at_least_90p` | `true` |
| `atomic_control_at_least_85p` | `true` |
| `answer_absent_unknown_at_least_90p` | `true` |
| `operation_local_conflict_at_least_85p` | `false` |
| `operation_atomic_conflict_at_least_85p` | `false` |
| `operation_rule_absent_unknown_at_least_90p` | `true` |
| `all_panels_parseable_at_least_95p` | `true` |
| `candidate_and_output_margins_reported` | `true` |
| `all_controls_passed` | `true` |

## Selected Panels

| Panel | Rows | Parseable | Local Rate | Atomic Rate | Unknown Rate | Other Rate | Expected Correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `synthetic_numeric_lookup` | 40 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| `familiar_entity_numeric_lookup` | 40 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| `real_world_atomic_number_control` | 40 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| `operation_local_conflict` | 40 | 1.000 | 0.775 | 0.125 | 0.000 | 0.100 | 0.775 |
| `operation_atomic_conflict` | 40 | 1.000 | 0.000 | 0.700 | 0.000 | 0.300 | 0.700 |
| `operation_rule_absent_conflict` | 40 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| `answer_absent_null` | 40 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |

## Aggregate Conflict

- primary conflict expected-correct rate: 0.738
- operation-local local rate: 0.775
- operation-atomic atomic rate: 0.700

## Claim Boundary

MC023 is a behavior diagnostic for query-level operation handles over
prompt-local versus learned atomic-number branches. It does not
establish an internal signature, intervention, or mechanism card.
