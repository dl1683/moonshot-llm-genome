# MC010 Two-Hop Fact-Code Arbitration Behavior Status

Status: two_hop_synthetic_lookup_failed.

Date: 2026-07-01

## Artifact

- runner:
  `code/mc010_two_hop_fact_code_arbitration.py`
- result:
  `results/cards/MC010/mc010_two_hop_fact_code_behavior_20260701T085414.json`

## Verdict

The generated-answer behavior gate did not pass. The result is a
behavior diagnostic only, and hidden-state work remains forbidden for
this route.

## Selected Template

- selected template: `neutral_contract`
- selection key: `[0.3, 1.0, 229.0, 0, 0, -1]`

## Gate Criteria

| Criterion | Passed |
| --- | --- |
| `smoke_mode` | `false` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `true` |
| `source_disjoint_holdout` | `true` |
| `selected_prompt_audit_passed` | `true` |
| `synthetic_panel_a_task_at_least_90p` | `false` |
| `synthetic_panel_a_parseable_at_least_95p` | `true` |
| `familiar_panel_b_task_at_least_90p` | `true` |
| `familiar_panel_b_parseable_at_least_95p` | `true` |
| `real_world_panel_c_real_at_least_85p` | `false` |
| `real_world_panel_c_parseable_at_least_95p` | `false` |
| `answer_absent_panel_f_unknown_at_least_90p` | `true` |
| `answer_absent_panel_f_parseable_at_least_95p` | `true` |
| `primary_conflict_binary_rows_at_least_40` | `true` |
| `non_holdout_conflict_task_at_least_10` | `true` |
| `non_holdout_conflict_real_or_lure_at_least_10` | `false` |
| `holdout_conflict_task_at_least_4` | `true` |
| `holdout_conflict_real_or_lure_at_least_4` | `false` |
| `primary_conflict_parseability_at_least_90p` | `true` |
| `candidate_and_output_margins_reported` | `true` |
| `non_holdout_conflict_label_balance_passed` | `false` |
| `holdout_conflict_label_balance_passed` | `false` |

## Selected Controls

| Panel | Rows | Parseable | Key Label Rate |
| --- | ---: | ---: | ---: |
| `synthetic_two_hop_lookup` | 40 | 1.000 | 0.575 |
| `familiar_entity_two_hop_lookup` | 40 | 1.000 | 0.950 |
| `real_world_memory_control` | 40 | 0.300 | 0.300 |
| `answer_absent_null` | 40 | 1.000 | 0.900 |

## Primary Conflict

- rows: 240
- parseable rate: 1.000
- task-code rows: 229
- real/lure-symbol rows: 0
- binary conflict rows: 229

## Allowed Claims

- This run documents the exact MC010 behavior boundary reached by
  the selected prompt contract.

## Forbidden Claims

- MC010 is a mechanism card.
- MC010 supports intervention.
- MC010 found a truth vector or general factual-recall mechanism.
- Any hidden-state or causal claim follows from this behavior run alone.
- Two-hop fact-code behavior is a valid hidden control surface.

