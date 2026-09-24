# MC015 Parity-Gated Numeric Arbitration Behavior Status

Status: parity_expected_local_conflict_failed.

Date: 2026-07-01

## Artifact

- runner:
  `code/mc015_parity_gated_numeric_arbitration.py`
- result:
  `results/cards/MC015/mc015_parity_gated_numeric_behavior_20260701T110321.json`

## Verdict

The parity-gated behavior gate failed. The target atomic number
is hidden and the visible status label channel is absent, but
the learned parity condition did not produce a clean gated
local-versus-learned conflict table. Hidden-state work remains
forbidden for this route.

## Selected Template

- selected template: `parity_rule`
- selection key: `[0.3, 1.0, 0.4875, 20, 4, 0]`

## Gate Criteria

| Criterion | Passed |
| --- | --- |
| `smoke_mode` | `false` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `true` |
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
| `answer_absent_panel_g_unknown_at_least_90p` | `true` |
| `answer_absent_panel_g_parseable_at_least_95p` | `true` |
| `expected_local_conflict_local_at_least_85p` | `false` |
| `expected_local_conflict_parseable_at_least_90p` | `true` |
| `expected_atomic_conflict_atomic_at_least_85p` | `false` |
| `expected_atomic_conflict_parseable_at_least_90p` | `true` |
| `primary_conflict_expected_correct_at_least_85p` | `false` |
| `primary_conflict_binary_rows_at_least_40` | `true` |
| `non_holdout_conflict_local_at_least_10` | `true` |
| `non_holdout_conflict_atomic_or_lure_at_least_10` | `true` |
| `holdout_conflict_local_at_least_4` | `true` |
| `holdout_conflict_atomic_or_lure_at_least_4` | `true` |
| `primary_conflict_parseability_at_least_90p` | `true` |
| `candidate_and_output_margins_reported` | `true` |
| `visible_status_label_absent_by_design` | `true` |
| `non_holdout_conflict_label_balance_passed` | `true` |
| `holdout_conflict_label_balance_passed` | `true` |

## Selected Controls

| Panel | Rows | Parseable | Key Label Rate |
| --- | ---: | ---: | ---: |
| `synthetic_numeric_lookup` | 40 | 1.000 | 1.000 |
| `familiar_entity_numeric_lookup` | 40 | 1.000 | 1.000 |
| `real_world_atomic_number_control` | 40 | 1.000 | 1.000 |
| `local_if_even_conflict` | 40 | 1.000 | 0.450 |
| `local_if_odd_conflict` | 40 | 1.000 | 0.525 |
| `parity_rule_absent_conflict` | 40 | 1.000 | 1.000 |
| `answer_absent_null` | 40 | 1.000 | 1.000 |

## Primary Conflict

- rows: 80
- parseable rate: 1.000
- local-number rows: 54
- atomic/lure-number rows: 24
- binary conflict rows: 78
- expected-correct rows: 39
- expected-correct rate: 0.487

## Forbidden Claims

- MC015 is a mechanism card.
- MC015 supports intervention.
- MC015 found an internal knowledge-control surface.
- Any hidden-state or causal claim follows from this behavior run alone.

