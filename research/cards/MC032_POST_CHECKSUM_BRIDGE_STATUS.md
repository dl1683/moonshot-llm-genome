# MC032 Post-Checksum Bridge Status

Status: smoke_only.

Date: 2026-07-01

## Artifact

- runner: `code/mc032_post_checksum_bridge.py`
- result: `results/cards/MC032/mc032_post_checksum_bridge_behavior_smoke_20260701T192315.json`

## Verdict

This is a smoke or partial run. It is not a full behavior-gate
verdict. Hidden-state work remains forbidden.

## Selected Template

- selected template: `mirror_registry`
- selection key: `[0.0, 1.0, -0.0, 0, 0, -1]`

## Gate Criteria

| Criterion | Passed |
| --- | --- |
| `smoke_mode` | `true` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `false` |
| `source_disjoint_holdout` | `true` |
| `selected_prompt_audit_passed` | `true` |
| `primary_prompts_have_no_status_or_checksum_route_lexemes` | `true` |
| `synthetic_panel_local_at_least_90p` | `true` |
| `familiar_panel_local_at_least_90p` | `true` |
| `real_world_panel_atomic_at_least_85p` | `true` |
| `answer_absent_unknown_at_least_90p` | `true` |
| `match_conflict_local_at_least_85p` | `false` |
| `mismatch_conflict_atomic_at_least_85p` | `false` |
| `mismatch_side_number_below_10p` | `true` |
| `primary_conflict_binary_rows_at_least_40` | `false` |
| `primary_conflict_expected_correct_at_least_85p` | `false` |
| `non_holdout_conflict_local_at_least_10` | `true` |
| `non_holdout_conflict_atomic_or_lure_at_least_10` | `false` |
| `holdout_conflict_local_at_least_4` | `false` |
| `holdout_conflict_atomic_or_lure_at_least_4` | `false` |
| `primary_conflict_parseability_at_least_90p` | `true` |
| `candidate_and_output_margins_reported` | `true` |
| `non_holdout_conflict_label_balance_passed` | `false` |
| `holdout_conflict_label_balance_passed` | `false` |

## Selected Controls

| Panel | Rows | Parseable | Key Label Rate |
| --- | ---: | ---: | ---: |
| `synthetic_numeric_lookup` | 10 | 1.000 | 1.000 |
| `familiar_entity_numeric_lookup` | 10 | 1.000 | 1.000 |
| `real_world_atomic_number_control` | 10 | 1.000 | 1.000 |
| `crosscheck_match_conflict` | 10 | 1.000 | 0.800 |
| `crosscheck_mismatch_conflict` | 10 | 1.000 | 0.000 |
| `crosscheck_absent_conflict` | 10 | 1.000 | 1.000 |
| `answer_absent_null` | 10 | 1.000 | 1.000 |

## Primary Conflict

- rows: 20
- parseable rate: 1.000
- local-number rows: 15
- atomic/lure-number rows: 0
- side-number rows: 0
- binary conflict rows: 15

## Mismatch Branch

- atomic rate: 0.000
- local rate: 0.700
- side-number rate: 0.000

## Forbidden Claims

- MC032 is a mechanism card.
- MC032 supports intervention.
- MC032 found an internal knowledge-control surface.
- Any hidden-state or causal claim follows from this behavior run alone.
