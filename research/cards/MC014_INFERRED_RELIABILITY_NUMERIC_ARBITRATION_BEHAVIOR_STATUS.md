# MC014 Inferred-Reliability Numeric Arbitration Behavior Status

Status: calibration_inference_conflict_collapsed.

Date: 2026-07-01

## Artifact

- runner:
  `code/mc014_inferred_reliability_numeric_arbitration.py`
- result:
  `results/cards/MC014/mc014_inferred_reliability_numeric_behavior_20260701T103426.json`

## Verdict

The inferred-reliability behavior gate failed. The visible status
label is absent, but the calibration-consistency prompt did not
produce a clean local-versus-learned conflict table. Hidden-state
work remains forbidden for this route.

## Selected Template

- selected template: `calibration_rule`
- selection key: `[0.0, 1.0, 80.0, 0, 0, 0]`

## Gate Criteria

| Criterion | Passed |
| --- | --- |
| `smoke_mode` | `false` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `true` |
| `source_disjoint_holdout` | `true` |
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
| `consistent_conflict_local_at_least_85p` | `true` |
| `consistent_conflict_parseable_at_least_90p` | `true` |
| `inconsistent_conflict_atomic_at_least_85p` | `false` |
| `inconsistent_conflict_parseable_at_least_90p` | `true` |
| `primary_conflict_binary_rows_at_least_40` | `true` |
| `non_holdout_conflict_local_at_least_10` | `true` |
| `non_holdout_conflict_atomic_or_lure_at_least_10` | `false` |
| `holdout_conflict_local_at_least_4` | `true` |
| `holdout_conflict_atomic_or_lure_at_least_4` | `false` |
| `primary_conflict_parseability_at_least_90p` | `true` |
| `candidate_and_output_margins_reported` | `true` |
| `visible_status_label_absent_by_design` | `true` |
| `non_holdout_conflict_label_balance_passed` | `false` |
| `holdout_conflict_label_balance_passed` | `false` |

## Selected Controls

| Panel | Rows | Parseable | Key Label Rate |
| --- | ---: | ---: | ---: |
| `synthetic_numeric_lookup` | 40 | 1.000 | 1.000 |
| `familiar_entity_numeric_lookup` | 40 | 1.000 | 1.000 |
| `real_world_atomic_number_control` | 40 | 1.000 | 1.000 |
| `calibration_consistent_conflict` | 40 | 1.000 | 1.000 |
| `calibration_inconsistent_conflict` | 40 | 1.000 | 0.000 |
| `calibration_absent_conflict` | 40 | 1.000 | 1.000 |
| `answer_absent_null` | 40 | 1.000 | 1.000 |

## Primary Conflict

- rows: 80
- parseable rate: 1.000
- local-number rows: 80
- atomic/lure-number rows: 0
- binary conflict rows: 80

## Forbidden Claims

- MC014 is a mechanism card.
- MC014 supports intervention.
- MC014 found an internal knowledge-control surface.
- Any hidden-state or causal claim follows from this behavior run alone.

