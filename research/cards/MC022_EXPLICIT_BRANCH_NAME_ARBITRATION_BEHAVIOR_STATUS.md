# MC022 Explicit Branch-Name Arbitration Behavior Status

Status: smoke_only.

Observed pattern: `explicit_visible_source_routing_failed`.

Date: 2026-07-01

## Artifact

- runner:
  `code/mc022_explicit_branch_name_arbitration.py`
- result:
  `results/cards/MC022/mc022_explicit_branch_name_arbitration_behavior_20260701T125634.json`

## Verdict

This is a smoke or partial run. It diagnoses whether replacing
opaque route codes with semantic answer-source labels repairs
visible-visible or visible-learned branch arbitration.

## Selected Template

- selected template: `explicit_source_column`
- selection key: `[0.9, 0.5, 0.25, 1.0, 0]`

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
| `learned_atomic_source_at_least_85p` | `false` |
| `all_panels_parseable_at_least_95p` | `true` |
| `candidate_and_output_margins_reported` | `false` |
| `all_controls_passed` | `true` |

## Selected Panels

| Panel | Rows | Parseable | Local Rate | Visible Rate | Atomic Rate | Unknown Rate | Other Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `local_number_control` | 20 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `visible_number_control` | 20 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| `atomic_number_control` | 20 | 1.000 | 0.000 | 0.000 | 0.900 | 0.000 | 0.100 |
| `source_visible_local_conflict` | 20 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `source_visible_reference_conflict` | 20 | 1.000 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 |
| `source_learned_local_conflict` | 20 | 1.000 | 0.950 | 0.000 | 0.050 | 0.000 | 0.000 |
| `source_learned_atomic_conflict` | 20 | 1.000 | 0.750 | 0.000 | 0.250 | 0.000 | 0.000 |
| `answer_absent_null` | 20 | 1.000 | 0.000 | 0.000 | 0.000 | 0.950 | 0.050 |

## Aggregate Conflicts

- visible conflict expected-correct rate: 0.750
- learned conflict expected-correct rate: 0.600

## Smoke Interpretation

MC022 rejects the simple explanation that MC021 only failed because opaque
`P`/`Q` route codes were too arbitrary. Semantic answer-source labels repaired
the direct controls and nulls, but not the conflict substrate.

The selected explicit-source template had clean controls: local 20/20, visible
reference 20/20, atomic 18/20, and answer-absent UNKNOWN 19/20. Conflict rows
were fully parseable and balanced by expected label/source label. Yet visible
conflicts reached only 30/40 expected-correct, and learned conflicts reached
only 24/40. The ATOMIC branch was the sharp failure: 5/20 ATOMIC-source rows
returned the learned atomic number, while 15/20 returned the prompt-local lab
number.

The rule-order split matters. When the nonlocal branch was defined first, the
visible-visible rows were 20/20 expected-correct; when LOCAL was defined first,
the same visible-visible contract collapsed to 20/20 local outputs. Defining
ATOMIC first helped the learned branch only partially: learned conflicts were
14/20 expected-correct, with 6/10 ATOMIC-source rows selecting atomic. The
failure is therefore not just arbitrary code opacity. It is an interaction
between local-source salience, rule-definition order, and learned-memory branch
selection.

## Claim Boundary

MC022 is a behavior diagnostic for semantic answer-source labels over
prompt-visible versus learned-memory branches. It does not establish
an internal signature, intervention, or mechanism card.
