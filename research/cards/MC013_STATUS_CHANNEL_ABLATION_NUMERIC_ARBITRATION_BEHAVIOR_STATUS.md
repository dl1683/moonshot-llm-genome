# MC013 Status-Channel Ablation Numeric Arbitration Behavior Status

Status: status_channel_ablation_collapsed_contrast.

Date: 2026-07-01

## Artifact

- runner:
  `code/mc013_status_channel_ablation_numeric_arbitration.py`
- result:
  `results/cards/MC013/mc013_status_channel_ablation_numeric_behavior_20260701T100644.json`

## Verdict

The matched status-channel ablation gate failed. The statused positive
control may still reproduce MC012, but the local-versus-learned split
does not survive the text-identical ablation condition. Hidden-state
work remains forbidden for this route.

## Selected Template

- selected template: `compact_status_ablation`
- selection key: `[0.975, 1.0, 80.0, 0, 0, 0, -1]`

## Gate Criteria

| Criterion | Passed |
| --- | --- |
| `smoke_mode` | `false` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `true` |
| `source_disjoint_holdout` | `true` |
| `selected_prompt_audit_passed` | `true` |
| `ablation_prompt_pairs_identical` | `true` |
| `synthetic_panel_a_local_at_least_90p` | `true` |
| `synthetic_panel_a_parseable_at_least_95p` | `true` |
| `familiar_panel_b_local_at_least_90p` | `true` |
| `familiar_panel_b_parseable_at_least_95p` | `true` |
| `real_world_panel_c_atomic_at_least_85p` | `true` |
| `real_world_panel_c_parseable_at_least_95p` | `true` |
| `answer_absent_panel_f_unknown_at_least_90p` | `true` |
| `answer_absent_panel_f_parseable_at_least_95p` | `true` |
| `statused_trusted_conflict_local_at_least_85p` | `true` |
| `statused_trusted_conflict_parseable_at_least_90p` | `true` |
| `statused_untrusted_conflict_atomic_at_least_85p` | `true` |
| `statused_untrusted_conflict_parseable_at_least_90p` | `true` |
| `ablation_trusted_local_at_least_85p` | `true` |
| `ablation_trusted_parseable_at_least_90p` | `true` |
| `ablation_untrusted_atomic_at_least_85p` | `false` |
| `ablation_untrusted_parseable_at_least_90p` | `true` |
| `primary_ablation_parseability_at_least_90p` | `true` |
| `primary_ablation_binary_rows_at_least_40` | `true` |
| `non_holdout_ablation_local_at_least_10` | `true` |
| `non_holdout_ablation_atomic_or_lure_at_least_10` | `false` |
| `holdout_ablation_local_at_least_4` | `true` |
| `holdout_ablation_atomic_or_lure_at_least_4` | `false` |
| `candidate_and_output_margins_reported` | `true` |
| `non_holdout_ablation_label_balance_passed` | `false` |
| `holdout_ablation_label_balance_passed` | `false` |

## Selected Controls

| Panel | Rows | Parseable | Key Label Rate |
| --- | ---: | ---: | ---: |
| `synthetic_numeric_lookup` | 40 | 1.000 | 1.000 |
| `familiar_entity_numeric_lookup` | 40 | 1.000 | 1.000 |
| `real_world_atomic_number_control` | 40 | 1.000 | 1.000 |
| `statused_trusted_conflict` | 40 | 1.000 | 1.000 |
| `statused_untrusted_conflict` | 40 | 1.000 | 0.975 |
| `matched_ablation_trusted_conflict` | 40 | 1.000 | 1.000 |
| `matched_ablation_untrusted_conflict` | 40 | 1.000 | 0.000 |
| `answer_absent_null` | 40 | 1.000 | 1.000 |

## Statused Positive Control

- rows: 80
- parseable rate: 1.000
- local-number rows: 40
- atomic/lure-number rows: 39

## Matched Ablation Conflict

- rows: 80
- parseable rate: 1.000
- local-number rows: 80
- atomic/lure-number rows: 0
- binary conflict rows: 80

## Forbidden Claims

- MC013 is a mechanism card.
- MC013 supports intervention.
- MC013 found an internal knowledge-control surface.
- Any hidden-state or causal claim follows from this behavior run alone.

