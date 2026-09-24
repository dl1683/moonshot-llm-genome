# MC031 Statusless Reliability Bridge Status

Status: smoke_only.

Date: 2026-07-01

## Artifact

- runner:
  `code/mc031_statusless_reliability_bridge.py`
- result:
  `results/cards/MC031/mc031_statusless_reliability_bridge_smoke_20260701T185520.json`

## Verdict

This is a smoke or partial run. It validates runner plumbing only;
it is not a full behavior-gate verdict. The selected smoke output
still exposes the immediate failure mode: invalid-checksum rows
do not produce the atomic branch, so hidden-state work remains
forbidden unless a full run reverses that pattern.

## Selected Template

- selected template: `arithmetic_checksum`
- selection key: `[0.0, 1.0, 20.0, 0, 0, 0]`

## Gate Criteria

| Criterion | Passed |
| --- | --- |
| `smoke_mode` | `true` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `false` |
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
| `valid_conflict_local_at_least_85p` | `true` |
| `valid_conflict_parseable_at_least_90p` | `true` |
| `invalid_conflict_atomic_at_least_85p` | `false` |
| `invalid_conflict_parseable_at_least_90p` | `true` |
| `primary_conflict_binary_rows_at_least_40` | `false` |
| `primary_conflict_expected_correct_at_least_85p` | `false` |
| `non_holdout_conflict_local_at_least_10` | `true` |
| `non_holdout_conflict_atomic_or_lure_at_least_10` | `false` |
| `holdout_conflict_local_at_least_4` | `true` |
| `holdout_conflict_atomic_or_lure_at_least_4` | `false` |
| `primary_conflict_parseability_at_least_90p` | `true` |
| `candidate_and_output_margins_reported` | `false` |
| `visible_status_label_absent_by_design` | `true` |
| `non_holdout_conflict_label_balance_passed` | `false` |
| `holdout_conflict_label_balance_passed` | `false` |

## Selected Controls

| Panel | Rows | Parseable | Key Label Rate |
| --- | ---: | ---: | ---: |
| `synthetic_numeric_lookup` | 10 | 1.000 | 1.000 |
| `familiar_entity_numeric_lookup` | 10 | 1.000 | 1.000 |
| `real_world_atomic_number_control` | 10 | 1.000 | 1.000 |
| `checksum_valid_conflict` | 10 | 1.000 | 1.000 |
| `checksum_invalid_conflict` | 10 | 1.000 | 0.000 |
| `checksum_absent_conflict` | 10 | 1.000 | 1.000 |
| `answer_absent_null` | 10 | 1.000 | 1.000 |

## Primary Conflict

- rows: 20
- parseable rate: 1.000
- local-number rows: 20
- atomic/lure-number rows: 0
- binary conflict rows: 20

## Forbidden Claims

- MC031 is a mechanism card.
- MC031 supports intervention.
- MC031 found an internal knowledge-control surface.
- Any hidden-state or causal claim follows from this behavior run alone.
