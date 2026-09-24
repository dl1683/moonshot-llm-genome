# MC011 Atomic-Number Code Arbitration Behavior Status

Status: numeric_conflict_contrast_absent.

Date: 2026-07-01

## Artifact

- runner:
  `code/mc011_atomic_number_code_arbitration.py`
- result:
  `results/cards/MC011/mc011_atomic_number_code_behavior_20260701T092147.json`

## Verdict

The generated-answer behavior gate did not pass. The result is a
behavior diagnostic only, and hidden-state work remains forbidden for
this route.

## Selected Template

- selected template: `neutral_numeric`
- selection key: `[1.0, 1.0, 240.0, 0, 0, -1]`

## Gate Criteria

| Criterion | Passed |
| --- | --- |
| `smoke_mode` | `false` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `true` |
| `source_disjoint_holdout` | `true` |
| `selected_prompt_audit_passed` | `true` |
| `synthetic_panel_a_local_at_least_90p` | `true` |
| `synthetic_panel_a_parseable_at_least_95p` | `true` |
| `familiar_panel_b_local_at_least_90p` | `true` |
| `familiar_panel_b_parseable_at_least_95p` | `true` |
| `real_world_panel_c_atomic_at_least_85p` | `true` |
| `real_world_panel_c_parseable_at_least_95p` | `true` |
| `answer_absent_panel_f_unknown_at_least_90p` | `true` |
| `answer_absent_panel_f_parseable_at_least_95p` | `true` |
| `primary_conflict_binary_rows_at_least_40` | `true` |
| `non_holdout_conflict_local_at_least_10` | `true` |
| `non_holdout_conflict_atomic_or_lure_at_least_10` | `false` |
| `holdout_conflict_local_at_least_4` | `true` |
| `holdout_conflict_atomic_or_lure_at_least_4` | `false` |
| `primary_conflict_parseability_at_least_90p` | `true` |
| `candidate_and_output_margins_reported` | `true` |
| `non_holdout_conflict_label_balance_passed` | `false` |
| `holdout_conflict_label_balance_passed` | `false` |

## Selected Controls

| Panel | Rows | Parseable | Key Label Rate |
| --- | ---: | ---: | ---: |
| `synthetic_numeric_lookup` | 40 | 1.000 | 1.000 |
| `familiar_entity_numeric_lookup` | 40 | 1.000 | 1.000 |
| `real_world_atomic_number_control` | 40 | 1.000 | 1.000 |
| `answer_absent_null` | 40 | 1.000 | 1.000 |

## Primary Conflict

- rows: 240
- parseable rate: 1.000
- local-number rows: 240
- atomic/lure-number rows: 0
- binary conflict rows: 240

## Forbidden Claims

- MC011 is a mechanism card.
- MC011 supports intervention.
- MC011 found a knowledge-control surface.
- Any hidden-state or causal claim follows from this behavior run alone.

