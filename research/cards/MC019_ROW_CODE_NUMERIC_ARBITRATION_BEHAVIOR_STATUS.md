# MC019 Row-Code Numeric Arbitration Behavior Status

Status: smoke_only.

Observed pattern: `mixed_or_unresolved_row_code_failure`.

Date: 2026-07-01

## Artifact

- runner:
  `code/mc019_row_code_numeric_arbitration.py`
- result:
  `results/cards/MC019/mc019_row_code_numeric_behavior_20260701T122807.json`

## Verdict

This is a smoke or partial run. It is not a full-run verdict,
but the selected-template pattern is usable as a bridge
diagnostic.

## Selected Template

- selected template: `compact_route_column`
- selection key: `[1.0, 0.4, 0.525, 12, 1, -1]`

## Gate Criteria

| Criterion | Passed |
| --- | --- |
| `smoke_mode` | `true` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `false` |
| `source_disjoint_holdout` | `true` |
| `primary_expected_labels_balanced` | `true` |
| `primary_route_labels_balanced` | `true` |
| `primary_expected_labels_balanced_by_split` | `true` |
| `primary_route_labels_balanced_by_split` | `true` |
| `selected_prompt_audit_passed` | `true` |
| `primary_prompts_have_no_status_lexemes` | `true` |
| `synthetic_panel_local_at_least_90p` | `true` |
| `synthetic_panel_parseable_at_least_95p` | `true` |
| `familiar_panel_local_at_least_90p` | `true` |
| `familiar_panel_parseable_at_least_95p` | `true` |
| `real_world_panel_atomic_at_least_85p` | `true` |
| `real_world_panel_parseable_at_least_95p` | `true` |
| `route_rule_absent_unknown_at_least_90p` | `true` |
| `route_rule_absent_parseable_at_least_95p` | `true` |
| `answer_absent_unknown_at_least_90p` | `true` |
| `answer_absent_parseable_at_least_95p` | `true` |
| `expected_local_conflict_local_at_least_85p` | `false` |
| `expected_atomic_conflict_atomic_at_least_85p` | `false` |
| `primary_conflict_expected_correct_at_least_85p` | `false` |
| `primary_conflict_binary_rows_at_least_40` | `false` |
| `non_holdout_conflict_balance_passed` | `true` |
| `holdout_conflict_balance_passed` | `false` |
| `primary_conflict_parseability_at_least_90p` | `true` |
| `candidate_and_output_margins_reported` | `false` |
| `visible_status_label_absent_by_design` | `true` |

## Selected Controls

| Panel | Rows | Parseable | Local Rate | Atomic Rate | Unknown Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `synthetic_numeric_lookup` | 20 | 1.000 | 1.000 | 0.000 | 0.000 |
| `familiar_entity_numeric_lookup` | 20 | 1.000 | 1.000 | 0.000 | 0.000 |
| `real_world_atomic_number_control` | 20 | 1.000 | 0.000 | 1.000 | 0.000 |
| `route_rule_absent_conflict` | 20 | 1.000 | 0.000 | 0.000 | 1.000 |
| `answer_absent_null` | 20 | 1.000 | 0.000 | 0.000 | 1.000 |

## Primary Conflict

- rows: 40
- parseable rate: 1.000
- local-number rows: 19
- atomic-number rows: 13
- expected-correct rows: 21
- expected-correct rate: 0.525

## Smoke Interpretation

MC019 is a useful negative bridge diagnostic. Unlike MC017/MC018, the selected
row-code template kept all direct controls and nulls clean in the 10-source
smoke: synthetic local lookup, familiar-entity local lookup, real-world atomic
control, route-rule-absent UNKNOWN, and answer-absent UNKNOWN were all 20/20.
It also removed source-status labels, hid target and lure atomic numbers, and
balanced expected local/atomic labels plus route labels by split.

The failure is the primary route-code conflict. The selected
`compact_route_column` template produced 40/40 parseable primary conflict rows
but only 21/40 expected-correct rows. Expected-local rows reached 13/20 local,
and expected-atomic rows reached only 8/20 atomic. This is not a clean local
collapse, but it is not rule following either.

The repair template `query_row_repeated` tested whether the issue was merely
finding the queried row in the table. It improved expected correctness to 27/40
and expected-local rows to 19/20, but expected-atomic rows stayed weak at 8/20
and the answer-absent null slipped to 17/20. Repeating the query row helps row
lookup, but it does not make learned atomic answers reliably win under
prompt-local table pressure.

The next bridge cannot rely on neutral row codes alone. It needs a stronger
source contract or a different behavior family if expected-atomic rows are to
survive without visible trusted/untrusted status text.

## Claim Boundary

MC019 is a behavior-substrate test. Passing it would show a
prompt-visible row-code contract can create local-versus-learned
contrast under order controls. It would not by itself establish an
internal signature, intervention, or mechanism card.
