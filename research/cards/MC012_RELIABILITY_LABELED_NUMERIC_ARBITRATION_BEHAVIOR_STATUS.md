# MC012 Reliability-Labeled Numeric Arbitration Behavior Status

Status: reliability_prompt_channel_visible.

Date: 2026-07-01

## Artifact

- runner:
  `code/mc012_reliability_labeled_numeric_arbitration.py`
- result:
  `results/cards/MC012/mc012_reliability_labeled_numeric_behavior_20260701T094345.json`

## Verdict

The generated-answer behavior contrast passed, but it is prompt-channel
visible by construction. The result is behavior-ready as a diagnostic
table, not signature-ready. Hidden-state work remains forbidden until a
materially different prompt-channel locality control passes.

## Selected Template

- selected template: `compact_reliability`
- selection key: `[1.0, 1.0, 79.0, 31, 8, -1]`

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
| `trusted_conflict_local_at_least_85p` | `true` |
| `trusted_conflict_parseable_at_least_90p` | `true` |
| `untrusted_conflict_atomic_at_least_85p` | `true` |
| `untrusted_conflict_parseable_at_least_90p` | `true` |
| `primary_conflict_binary_rows_at_least_40` | `true` |
| `non_holdout_conflict_local_at_least_10` | `true` |
| `non_holdout_conflict_atomic_or_lure_at_least_10` | `true` |
| `holdout_conflict_local_at_least_4` | `true` |
| `holdout_conflict_atomic_or_lure_at_least_4` | `true` |
| `primary_conflict_parseability_at_least_90p` | `true` |
| `candidate_and_output_margins_reported` | `true` |
| `prompt_channel_contrast_visible_by_design` | `true` |
| `prompt_channel_locality_gate_passed` | `false` |
| `non_holdout_conflict_label_balance_passed` | `true` |
| `holdout_conflict_label_balance_passed` | `true` |

## Selected Controls

| Panel | Rows | Parseable | Key Label Rate |
| --- | ---: | ---: | ---: |
| `synthetic_numeric_lookup` | 40 | 1.000 | 1.000 |
| `familiar_entity_numeric_lookup` | 40 | 1.000 | 1.000 |
| `real_world_atomic_number_control` | 40 | 1.000 | 1.000 |
| `trusted_source_conflict` | 40 | 1.000 | 1.000 |
| `untrusted_source_conflict` | 40 | 1.000 | 0.975 |
| `answer_absent_null` | 40 | 1.000 | 1.000 |

## Primary Conflict

- rows: 80
- parseable rate: 1.000
- local-number rows: 40
- atomic/lure-number rows: 39
- binary conflict rows: 79

## Forbidden Claims

- MC012 is a mechanism card.
- MC012 supports intervention.
- MC012 found a knowledge-control surface.
- Any hidden-state or causal claim follows from this behavior run alone.

