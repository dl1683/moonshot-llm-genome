# MC033 Fact-Claim Bridge Closeout Status

Status: smoke_only.

Date: 2026-07-01

## Artifact

- runner: `code/mc033_fact_claim_bridge_closeout.py`
- result: `results/cards/MC033/mc033_fact_claim_bridge_behavior_smoke_20260701T194546.json`

## Verdict

This is a smoke or partial run. Hidden-state work remains forbidden.

## Selected Template

- selected template: `memory_comparison`
- selection key: `[0.1, 1.0, -0.4, 7, 2, -2]`

## Gate Criteria

| Criterion | Passed |
| --- | --- |
| `smoke_mode` | `true` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `false` |
| `source_disjoint_holdout` | `true` |
| `selected_prompt_audit_passed` | `true` |
| `primary_prompts_have_no_status_or_closed_route_lexemes` | `true` |
| `synthetic_panel_local_at_least_90p` | `true` |
| `familiar_panel_local_at_least_90p` | `true` |
| `real_world_panel_atomic_at_least_85p` | `true` |
| `answer_absent_unknown_at_least_90p` | `true` |
| `match_conflict_local_at_least_85p` | `false` |
| `mismatch_conflict_atomic_at_least_85p` | `false` |
| `mismatch_claimed_number_below_10p` | `false` |
| `primary_conflict_binary_rows_at_least_40` | `false` |
| `primary_conflict_expected_correct_at_least_85p` | `false` |
| `non_holdout_conflict_local_at_least_10` | `false` |
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
| `fact_claim_match_conflict` | 10 | 1.000 | 0.400 |
| `fact_claim_mismatch_conflict` | 10 | 1.000 | 0.100 |
| `fact_claim_absent_conflict` | 10 | 1.000 | 1.000 |
| `answer_absent_null` | 10 | 1.000 | 1.000 |

## Primary Conflict

- rows: 20
- parseable rate: 1.000
- local-number rows: 9
- atomic/lure-number rows: 10
- binary conflict rows: 19

## Mismatch Branch

- atomic rate: 0.100
- local rate: 0.500
- claimed-number/lure rate: 0.400

## Forbidden Claims

- MC033 is a mechanism card.
- MC033 supports intervention.
- MC033 found an internal knowledge-control surface.
- Any hidden-state or causal claim follows from this behavior run alone.
