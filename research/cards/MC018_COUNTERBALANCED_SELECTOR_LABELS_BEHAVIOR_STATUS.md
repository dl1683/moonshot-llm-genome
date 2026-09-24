# MC018 Counterbalanced Selector Labels Behavior Status

Status: smoke_only.

Observed pattern: `mixed_or_unresolved_selector_failure`.

Date: 2026-07-01

## Artifact

- runner:
  `code/mc018_counterbalanced_selector_labels.py`
- result:
  `results/cards/MC018/mc018_counterbalanced_selector_labels_behavior_20260701T121143.json`

## Verdict

This is a smoke or partial run. Use the observed pattern only
as a diagnostic for whether a full run or prompt repair is worth
doing.

## Selected Template

- selected template: `explicit_source_labels`
- selection key: `[0.325, 0.3875, 0.525, 72, 12, 0.7375, 0]`

## Gate Criteria

| Criterion | Passed |
| --- | --- |
| `smoke_mode` | `true` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `false` |
| `source_disjoint_holdout` | `true` |
| `primary_expected_sources_balanced` | `true` |
| `primary_expected_choices_balanced` | `true` |
| `primary_expected_sources_balanced_by_split` | `true` |
| `primary_expected_choices_balanced_by_mapping` | `true` |
| `selected_prompt_audit_passed` | `true` |
| `primary_prompts_have_no_status_lexemes` | `true` |
| `local_selector_control_local_at_least_90p` | `false` |
| `local_selector_control_parseable_at_least_95p` | `true` |
| `atomic_selector_control_atomic_at_least_90p` | `false` |
| `atomic_selector_control_parseable_at_least_95p` | `true` |
| `answer_absent_panel_unknown_at_least_90p` | `false` |
| `answer_absent_panel_parseable_at_least_95p` | `true` |
| `expected_local_conflict_local_at_least_85p` | `false` |
| `expected_atomic_conflict_atomic_at_least_85p` | `false` |
| `primary_conflict_expected_correct_at_least_85p` | `false` |
| `primary_conflict_binary_rows_at_least_40` | `true` |
| `non_holdout_conflict_choice_balance_passed` | `true` |
| `holdout_conflict_choice_balance_passed` | `true` |
| `primary_conflict_choice_balance_passed` | `true` |
| `primary_conflict_parseability_at_least_90p` | `true` |
| `candidate_and_output_margins_reported` | `false` |
| `visible_status_label_absent_by_design` | `true` |

## Selected Controls

| Panel | Rows | Parseable | Local Rate | Atomic Rate | Unknown Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `local_selector_control` | 80 | 1.000 | 0.800 | 0.200 | 0.000 |
| `atomic_selector_control` | 80 | 1.000 | 0.175 | 0.825 | 0.000 |
| `selector_rule_absent_conflict` | 80 | 1.000 | 0.575 | 0.425 | 0.000 |
| `answer_absent_null` | 80 | 1.000 | 0.537 | 0.138 | 0.325 |

## Primary Conflict

- rows: 160
- parseable rate: 1.000
- A rows: 88
- B rows: 72
- first-listed choice rate: 0.762
- local-source rows: 102
- atomic-source rows: 58
- expected-correct rows: 84
- expected-correct rate: 0.525

## Smoke Interpretation

The expanded smoke does not support hidden-state probing. It does, however,
separate MC017's total `LOCAL` collapse into narrower components. Removing
`LOCAL`/`ATOMIC` as answer tokens partially repairs the atomic-only control
from 0/10 atomic in MC017 to 66/80 atomic in MC018, so the MC017 failure was
partly an answer-token artifact. But the primary conflict remains near chance
for the intended source rule: 84/160 source-correct despite balanced expected
local/atomic sources and balanced expected A/B choices. The dominant pressures
are presentation-order and local-source salience: first-listed choices account
for 122/160 primary conflict rows, and local-source selections account for
102/160 rows.

The next source-selector bridge must therefore beat both order controls and
local-source bias before any signature search is allowed.

## Claim Boundary

MC018 is an answer-interface diagnostic. Passing it would only
show that a neutral, counterbalanced selector behavior substrate
exists. Failing it names the remaining output-interface bias before
any hidden-state claim is attempted.
