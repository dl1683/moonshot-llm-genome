# MC017 Selector-Token Numeric Arbitration Behavior Status

Status: smoke_only.

Date: 2026-07-01

## Artifact

- runner:
  `code/mc017_selector_token_numeric_arbitration.py`
- result:
  `results/cards/MC017/mc017_selector_token_numeric_behavior_20260701T115850.json`

## Verdict

This is a smoke or partial run. It validates runner plumbing only;
it is not a full-run verdict for or against the full MC017 behavior substrate.
It is enough to motivate an answer-interface diagnostic, because the selected
template returned `LOCAL` on 10/10 atomic-selector control rows even when the
prompt explicitly said no local lab table was active and standard chemistry was
the active source.

## Selected Template

- selected template: `selector_rule`
- selection key: `[0.0, 1.0, 0.5, 0, 0, 0]`

## Gate Criteria

| Criterion | Passed |
| --- | --- |
| `smoke_mode` | `true` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `false` |
| `source_disjoint_holdout` | `true` |
| `primary_expected_labels_balanced` | `true` |
| `primary_expected_labels_balanced_by_split` | `true` |
| `selected_prompt_audit_passed` | `true` |
| `primary_prompts_have_no_status_lexemes` | `true` |
| `synthetic_panel_a_local_at_least_90p` | `true` |
| `synthetic_panel_a_parseable_at_least_95p` | `true` |
| `familiar_panel_b_local_at_least_90p` | `true` |
| `familiar_panel_b_parseable_at_least_95p` | `true` |
| `real_world_panel_c_atomic_at_least_85p` | `true` |
| `real_world_panel_c_parseable_at_least_95p` | `true` |
| `local_selector_control_local_at_least_90p` | `true` |
| `local_selector_control_parseable_at_least_95p` | `true` |
| `atomic_selector_control_atomic_at_least_90p` | `false` |
| `atomic_selector_control_parseable_at_least_95p` | `true` |
| `answer_absent_panel_unknown_at_least_90p` | `false` |
| `answer_absent_panel_parseable_at_least_95p` | `true` |
| `expected_local_conflict_local_selector_at_least_85p` | `true` |
| `expected_atomic_conflict_atomic_selector_at_least_85p` | `false` |
| `primary_conflict_expected_correct_at_least_85p` | `false` |
| `primary_conflict_binary_rows_at_least_40` | `false` |
| `non_holdout_conflict_local_at_least_10` | `true` |
| `non_holdout_conflict_atomic_at_least_10` | `false` |
| `holdout_conflict_local_at_least_4` | `true` |
| `holdout_conflict_atomic_at_least_4` | `false` |
| `primary_conflict_parseability_at_least_90p` | `true` |
| `candidate_and_output_margins_reported` | `true` |
| `visible_status_label_absent_by_design` | `true` |
| `non_holdout_conflict_label_balance_passed` | `false` |
| `holdout_conflict_label_balance_passed` | `false` |

## Selected Controls

| Panel | Rows | Parseable | Key Label Rate |
| --- | ---: | ---: | ---: |
| `synthetic_numeric_lookup` | 10 | 1.000 | 1.000 |
| `familiar_entity_numeric_lookup` | 10 | 1.000 | 1.000 |
| `real_world_atomic_number_control` | 10 | 1.000 | 1.000 |
| `local_selector_control` | 10 | 1.000 | 1.000 |
| `atomic_selector_control` | 10 | 1.000 | 0.000 |
| `selector_rule_absent_conflict` | 10 | 1.000 | 1.000 |
| `answer_absent_null` | 10 | 1.000 | 0.100 |

## Primary Conflict

- rows: 20
- parseable rate: 1.000
- LOCAL rows: 20
- ATOMIC rows: 0
- binary selector rows: 20
- expected-correct rows: 10
- expected-correct rate: 0.500

## Smoke Interpretation

MC017 did not just fail expected-atomic conflict rows. It also failed the
atomic-only selector control, while numeric direct controls stayed clean. That
separates the failure from ordinary atomic-number recall and points to the
source-token answer interface itself. The immediate next diagnostic is a
neutral, counterbalanced selector label test rather than hidden-state probing.

## Forbidden Claims

- MC017 is a mechanism card.
- MC017 supports intervention.
- MC017 found an internal knowledge-control surface.
- A selector-token behavior pass would by itself prove a numeric control surface.

