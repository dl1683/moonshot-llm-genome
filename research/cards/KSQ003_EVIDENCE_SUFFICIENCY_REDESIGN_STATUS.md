# KSQ003 Evidence Sufficiency Redesign Status

Date: 2026-07-02

Runner:

> `code/ksq003_evidence_sufficiency_redesign.py`

Result:

> `results/cards/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN/ksq003_evidence_sufficiency_redesign_full_behavior.json`

Status: evidence_sufficiency_learned_branch_failed.

## Verdict

The evidence-sufficiency redesign did not pass behavior
admission. Hidden-state work remains forbidden.

## Route Decision

- route decision: `bound_evidence_sufficiency_without_hidden_state`
- exported diagnostic class: `STATUSLESS_EVIDENCE_SUFFICIENCY_BOUNDARY`
- behavior ready: `false`
- signature screen allowed: `false`

## Gate Criteria

| Criterion | Passed |
| --- | --- |
| `smoke_mode` | `false` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `true` |
| `source_disjoint_holdout` | `true` |
| `selected_prompt_audit_passed` | `true` |
| `primary_prompts_have_no_status_or_closed_route_lexemes` | `true` |
| `source_local_direct_control_passed` | `true` |
| `learned_fact_direct_control_passed` | `true` |
| `answer_absent_null_passed` | `true` |
| `complete_identity_conflict_atomic_passed` | `false` |
| `contradictory_identity_unknown_passed` | `false` |
| `single_feature_ablation_unknown_passed` | `false` |
| `holdout_complete_identity_atomic_passed` | `false` |
| `candidate_and_output_margins_reported` | `true` |

## Selected Template

- selected template: `compact_identity`
- selection key: `[0.2, 0.325, 0.55, 0.125, 1.0, -2]`

## Selected Controls

| Panel | Rows | Parseable | Key Label Rate |
| --- | ---: | ---: | ---: |
| `source_local_direct_control` | 40 | 1.000 | 1.000 |
| `learned_fact_direct_control` | 40 | 1.000 | 1.000 |
| `complete_identity_conflict` | 40 | 1.000 | 0.325 |
| `contradictory_identity_null` | 40 | 0.925 | 0.650 |
| `symbol_only_ablation` | 40 | 1.000 | 0.800 |
| `initial_only_ablation` | 40 | 1.000 | 0.200 |
| `answer_absent_and_side_null` | 40 | 1.000 | 0.950 |

## Primary Conflict

- rows: `40`
- atomic-number rows: `13`
- local-number rows: `0`
- parseable rate: `1.000`

## Null Stress

- rows: `120`
- unknown rows: `66`
- local-number rows: `5`

## Forbidden Claims

- KSQ003 is a mechanism card.
- KSQ003 supports intervention.
- KSQ003 found an internal knowledge-control surface.
- Any hidden-state or causal claim follows from this redesign alone.
