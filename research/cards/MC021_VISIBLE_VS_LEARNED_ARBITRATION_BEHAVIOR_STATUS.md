# MC021 Visible-Versus-Learned Arbitration Behavior Status

Status: smoke_only.

Observed pattern: `generic_visible_route_failed`.

Date: 2026-07-01

## Artifact

- runner:
  `code/mc021_visible_vs_learned_arbitration.py`
- result:
  `results/cards/MC021/mc021_visible_vs_learned_arbitration_behavior_20260701T124555.json`

## Verdict

This is a smoke or partial run. It diagnoses whether conditional
routing works for prompt-visible branches while failing for
learned-memory branches.

## Selected Template

- selected template: `plain_branch_table`
- selection key: `[0.9, 0.5, 0.3, 1.0, 0]`

## Gate Criteria

| Criterion | Passed |
| --- | --- |
| `smoke_mode` | `true` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `false` |
| `source_disjoint_holdout` | `true` |
| `selected_prompt_audit_passed` | `true` |
| `prompts_have_no_status_lexemes` | `true` |
| `local_control_at_least_90p` | `true` |
| `visible_control_at_least_90p` | `true` |
| `atomic_control_at_least_85p` | `true` |
| `answer_absent_unknown_at_least_90p` | `true` |
| `visible_conflict_expected_correct_at_least_85p` | `false` |
| `learned_conflict_expected_correct_at_least_85p` | `false` |
| `learned_atomic_route_at_least_85p` | `false` |
| `all_panels_parseable_at_least_95p` | `true` |
| `candidate_and_output_margins_reported` | `false` |
| `all_controls_passed` | `true` |

## Selected Panels

| Panel | Rows | Parseable | Local Rate | Visible Rate | Atomic Rate | Unknown Rate | Other Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `local_number_control` | 20 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `visible_number_control` | 20 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| `atomic_number_control` | 20 | 1.000 | 0.000 | 0.000 | 0.900 | 0.000 | 0.100 |
| `route_visible_p_local_conflict` | 20 | 1.000 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 |
| `route_visible_p_reference_conflict` | 20 | 1.000 | 0.450 | 0.550 | 0.000 | 0.000 | 0.000 |
| `route_learned_p_local_conflict` | 20 | 1.000 | 0.700 | 0.050 | 0.150 | 0.000 | 0.100 |
| `route_learned_p_atomic_conflict` | 20 | 1.000 | 0.600 | 0.050 | 0.300 | 0.000 | 0.050 |
| `answer_absent_null` | 20 | 1.000 | 0.000 | 0.000 | 0.000 | 0.900 | 0.100 |

## Aggregate Conflicts

- visible conflict expected-correct rate: 0.725
- learned conflict expected-correct rate: 0.500

## Smoke Interpretation

MC021 rules out a too-narrow reading of MC020. The problem is not only routing
into learned memory. The selected template passed all direct controls and nulls:
local number 20/20, visible reference number 20/20, atomic number 18/20, and
answer-absent UNKNOWN 18/20. But conditional routing over two prompt-visible
branches was still only 29/40 expected-correct. The same route grammar over a
prompt-visible local branch and learned atomic branch was worse at 20/40.

This means the bridge has two layers of failure. First, conditional route
following is unstable even when both branches are visible in the prompt.
Second, when one branch is learned memory, the instability is amplified toward
prompt-local outputs. The next bridge should not assume that a route code is a
clean arbitration substrate; it needs either a different conditional format or
a different behavior family.

## Claim Boundary

MC021 is a behavior diagnostic for conditional routing over
prompt-visible versus learned-memory branches. It does not
establish an internal signature, intervention, or mechanism card.
