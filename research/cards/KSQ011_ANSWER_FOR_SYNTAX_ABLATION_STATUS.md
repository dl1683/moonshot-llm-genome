# KSQ011 Answer_For Syntax Ablation Status

Date: 2026-07-02

Runner:

> `code/ksq011_answer_for_syntax_ablation.py`

Result:

> `results/cards/KSQ011_ANSWER_FOR_SYNTAX_ABLATION/ksq011_answer_for_syntax_ablation_smoke_limit10.json`

Status: function_assignment_answer_channel_dominance.

## Route Decision

- route decision: `answer_for_syntax_ablation_diagnostic`
- exported diagnostic class: `FUNCTION_ASSIGNMENT_ANSWER_CHANNEL_DOMINANCE`
- behavior ready: `false`
- signature screen allowed: `false`

## Gate Criteria

| Criterion | Passed |
| --- | --- |
| `smoke_mode` | `true` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `false` |
| `source_disjoint_holdout` | `true` |
| `selected_prompt_audit_passed` | `true` |
| `exact_bridge_passed` | `true` |
| `exact_answer_for_variants_passed` | `false` |
| `function_variants_passed` | `false` |
| `nonfunction_variants_passed` | `false` |
| `query_only_control_passed` | `true` |
| `all_panels_passed` | `false` |
| `source_disjoint_holdout_passed` | `false` |
| `candidate_and_output_margins_reported` | `true` |

## Selected Template

- selected template: `tag_rows`
- selection key: `[0.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.1, 0.8, 0.2, 0.0, 0.8, 0.0, 1.0, -0.45, -0.9, 0.8, 0]`
- failed panels: `["other_exact_answer_for_alt", "other_spaced_answer_for_alt", "other_colon_answer_for_alt", "other_answer_to_alt", "other_value_for_alt", "other_prose_value_alt", "other_quoted_answer_for_alt", "other_bare_alt_mention", "adversary_only_exact_answer_for"]`

## Selected Panels

| Panel | Rows | Label Counts | Key Rate |
| --- | ---: | --- | ---: |
| `exact_bridge` | 10 | `{"evidence_answer": 9, "unparsed": 1}` | 0.900 |
| `other_exact_answer_for_alt` | 10 | `{"adversary_alt_overrode_bridge": 9, "unparsed": 1}` | 0.000 |
| `other_spaced_answer_for_alt` | 10 | `{"adversary_alt_overrode_bridge": 10}` | 0.000 |
| `other_colon_answer_for_alt` | 10 | `{"adversary_alt_overrode_bridge": 9, "evidence_answer": 1}` | 0.100 |
| `other_answer_to_alt` | 10 | `{"adversary_alt_overrode_bridge": 9, "unparsed": 1}` | 0.000 |
| `other_value_for_alt` | 10 | `{"adversary_alt_overrode_bridge": 8, "evidence_answer": 1, "unparsed": 1}` | 0.100 |
| `other_entity_equals_alt` | 10 | `{"abstain": 1, "evidence_answer": 8, "unparsed": 1}` | 0.800 |
| `other_prose_value_alt` | 10 | `{"adversary_alt_overrode_bridge": 7, "evidence_answer": 2, "unparsed": 1}` | 0.200 |
| `other_quoted_answer_for_alt` | 10 | `{"adversary_alt_overrode_bridge": 9, "unparsed": 1}` | 0.000 |
| `other_bare_alt_mention` | 10 | `{"evidence_answer": 8, "unparsed": 2}` | 0.800 |
| `adversary_only_exact_answer_for` | 10 | `{"adversary_only_answer_for_reproduced": 9, "unparsed": 1}` | 0.000 |
| `query_only_control` | 10 | `{"control_abstain": 10}` | 1.000 |

## Forbidden Claims

- KSQ011 is a mechanism card.
- KSQ011 proves real uncertainty, refusal, factual correction, or knowledge control.
- KSQ011 licenses hidden-state intervention or mechanism claims.
