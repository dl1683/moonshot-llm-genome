# MC026 Numeric-Option Operation Arbitration Behavior Status

Status: smoke_only.

Observed pattern: `numeric_option_smoke_failed`.

Date: 2026-07-01

## Artifact

- runner:
  `code/mc026_numeric_option_operation_arbitration.py`
- result:
  `results/cards/MC026/mc026_numeric_option_operation_arbitration_behavior_20260701T151215.json`

## Verdict

This is a smoke or partial run. It diagnoses whether returning
the listed number/UNKNOWN repairs operation routing better than
A/B/C choice labels.

## Selected Template

- selected template: `compact_numeric_options`
- selection key: `[0.0, 0.15833333333333333, 0.5416666666666666, 0.9666666666666667, 0]`

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
| `answer_absent_unknown_at_least_90p` | `true` |
| `operation_rule_absent_unknown_at_least_90p` | `true` |
| `operation_local_option_conflict_at_least_85p` | `true` |
| `operation_atomic_option_conflict_at_least_85p` | `false` |
| `all_panels_parseable_at_least_95p` | `true` |
| `candidate_and_output_margins_reported` | `true` |
| `all_controls_passed` | `false` |
| `numeric_option_behavior_gate_passed` | `false` |

## Selected Panels

| Panel | Rows | Parseable | Local Rate | Atomic Rate | Unknown Rate | Other Rate | Expected Correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `synthetic_option_lookup` | 120 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| `familiar_option_lookup` | 120 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| `atomic_option_control` | 120 | 0.967 | 0.000 | 0.000 | 0.967 | 0.000 | 0.000 |
| `operation_local_option_conflict` | 120 | 1.000 | 0.925 | 0.000 | 0.075 | 0.000 | 0.925 |
| `operation_atomic_option_conflict` | 120 | 1.000 | 0.783 | 0.158 | 0.058 | 0.000 | 0.158 |
| `operation_rule_absent_option_null` | 120 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| `answer_absent_option_null` | 120 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |

## Aggregate Conflict

- primary conflict expected-correct rate: 0.542
- operation-local option local rate: 0.925
- operation-atomic option atomic rate: 0.158

## Claim Boundary

MC026 is a prompt-visible answer-interface diagnostic. The atomic
number is present in the answer list on conflict rows, so this can
only test whether numeric options repair routing/output behavior.
It does not establish a learned-memory hidden signature,
intervention, or mechanism card.
