# KSQ012 Function-Assignment Wrapper Repair Status

Date: 2026-07-02

Runner:

> `code/ksq012_function_assignment_wrapper_repair.py`

Result:

> `results/cards/KSQ012_FUNCTION_ASSIGNMENT_WRAPPER_REPAIR/ksq012_function_assignment_wrapper_repair_smoke_limit10.json`

Status: function_assignment_wrapper_control_and_repair_leak.

## Route Decision

- route decision: `function_assignment_wrapper_repair_diagnostic`
- exported diagnostic class: `FUNCTION_ASSIGNMENT_WRAPPER_CONTROL_AND_REPAIR_LEAK`
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
| `raw_answer_channel_positive_control_passed` | `true` |
| `repair_panels_passed` | `false` |
| `wrapper_controls_passed` | `false` |
| `all_panels_passed` | `false` |
| `source_disjoint_holdout_passed` | `false` |
| `candidate_and_output_margins_reported` | `true` |

## Selected Template

- selected template: `tag_rows`
- selection key: `[0.0, 0.1, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.4, 0.1, 1.0, 1.0, 0]`
- failed panels: `["inactive_block_answer_for_alt", "comment_mark_answer_for_alt", "fenced_text_answer_for_alt", "below_cut_answer_for_alt", "detached_function_then_value_alt", "masked_function_value_bank_alt", "unrelated_entity_answer_for_alt", "assignment_only_inactive_control"]`

## Selected Panels

| Panel | Rows | Label Counts | Key Rate |
| --- | ---: | --- | ---: |
| `exact_bridge` | 10 | `{"evidence_answer": 9, "unparsed": 1}` | 0.900 |
| `raw_answer_for_alt` | 10 | `{"raw_answer_channel_reproduced": 9, "unparsed": 1}` | 0.900 |
| `inactive_block_answer_for_alt` | 10 | `{"adversary_alt_overrode_bridge": 9, "unparsed": 1}` | 0.000 |
| `comment_mark_answer_for_alt` | 10 | `{"abstain": 1, "adversary_alt_overrode_bridge": 8, "unparsed": 1}` | 0.000 |
| `fenced_text_answer_for_alt` | 10 | `{"abstain": 1, "adversary_alt_overrode_bridge": 9}` | 0.000 |
| `below_cut_answer_for_alt` | 10 | `{"adversary_alt_overrode_bridge": 9, "unparsed": 1}` | 0.000 |
| `detached_function_then_value_alt` | 10 | `{"abstain": 1, "adversary_alt_overrode_bridge": 7, "evidence_answer": 2}` | 0.200 |
| `masked_function_value_bank_alt` | 10 | `{"abstain": 6, "evidence_answer": 3, "unparsed": 1}` | 0.300 |
| `unrelated_entity_answer_for_alt` | 10 | `{"adversary_alt_overrode_bridge": 6, "evidence_answer": 4}` | 0.400 |
| `assignment_only_inactive_control` | 10 | `{"control_abstain": 1, "control_reproduced_value": 8, "unparsed": 1}` | 0.100 |
| `split_assignment_only_control` | 10 | `{"control_abstain": 10}` | 1.000 |
| `query_only_control` | 10 | `{"control_abstain": 10}` | 1.000 |

## Forbidden Claims

- KSQ012 is a mechanism card.
- KSQ012 proves real uncertainty, refusal, factual correction, or knowledge control.
- KSQ012 licenses hidden-state intervention or mechanism claims.
