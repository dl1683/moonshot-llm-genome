# MC020 Atomic Recall Table Pressure Behavior Status

Status: smoke_only.

Observed pattern: `route_rule_expected_atomic_failed`.

Date: 2026-07-01

## Artifact

- runner:
  `code/mc020_atomic_recall_table_pressure.py`
- result:
  `results/cards/MC020/mc020_atomic_recall_table_pressure_behavior_20260701T123741.json`

## Verdict

This is a smoke or partial run. It is a diagnostic of the
MC019 expected-atomic failure mode, not a full-run verdict.

## Selected Template

- selected template: `plain_table`
- selection key: `[0.9, 0.9, 0.1, 0.1, 1.0, 0]`

## Gate Criteria

| Criterion | Passed |
| --- | --- |
| `smoke_mode` | `true` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `false` |
| `source_disjoint_holdout` | `true` |
| `selected_prompt_audit_passed` | `true` |
| `prompts_have_no_status_lexemes` | `true` |
| `no_table_atomic_at_least_85p` | `true` |
| `distractor_table_atomic_at_least_85p` | `true` |
| `query_row_local_at_least_90p` | `true` |
| `answer_absent_unknown_at_least_90p` | `true` |
| `query_row_atomic_at_least_85p` | `true` |
| `query_row_repeated_atomic_at_least_85p` | `true` |
| `route_local_at_least_85p` | `true` |
| `route_atomic_at_least_85p` | `false` |
| `all_panels_parseable_at_least_95p` | `true` |
| `candidate_and_output_margins_reported` | `false` |
| `all_baseline_controls_passed` | `true` |

## Selected Panels

| Panel | Rows | Parseable | Local Rate | Atomic Rate | Unknown Rate | Other Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_table_atomic_control` | 20 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| `distractor_table_atomic_control` | 20 | 1.000 | 0.000 | 0.900 | 0.000 | 0.100 |
| `query_row_atomic_control` | 20 | 1.000 | 0.000 | 0.900 | 0.000 | 0.100 |
| `query_row_repeated_atomic_control` | 20 | 1.000 | 0.000 | 0.900 | 0.000 | 0.100 |
| `query_row_local_control` | 20 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| `route_local_conflict` | 20 | 1.000 | 0.850 | 0.000 | 0.000 | 0.150 |
| `route_atomic_conflict` | 20 | 1.000 | 0.750 | 0.100 | 0.000 | 0.150 |
| `answer_absent_null` | 20 | 1.000 | 0.000 | 0.000 | 0.900 | 0.100 |

## Smoke Interpretation

MC020 isolates the MC019 expected-atomic failure. Atomic recall itself remains
strong under table pressure: no-table atomic recall was 20/20, distractor-table
atomic recall was 18/20, query-row atomic recall was 18/20, and repeated-query
row atomic recall was also 18/20. The queried local number being present is
therefore not enough to suppress learned atomic recall when the instruction
directly asks for the standard atomic number.

The failure appears when the model must apply a conditional route rule. The
route-local panel reached 17/20 local, but the route-atomic panel collapsed to
15/20 local and only 2/20 atomic. That means the bridge failure is not
ordinary fact recall and not simple local-number interference. It is asymmetric
conditional source arbitration: the model follows the route rule when the rule
points to the prompt-local source, but usually ignores or overrides it when the
same kind of rule points to learned atomic memory.

The next bridge should not spend effort on making atomic recall more available.
It should test a materially different arbitration contract, because direct
atomic instructions already work with the local row present.

## Claim Boundary

MC020 is a diagnostic for local-table pressure on atomic recall.
It does not establish an internal signature, intervention, or
mechanism card.
