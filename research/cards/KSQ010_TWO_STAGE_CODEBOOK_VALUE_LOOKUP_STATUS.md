# KSQ010 Two-Stage Codebook Value Lookup Status

Date: 2026-07-02

Runner:

> `code/ksq010_two_stage_codebook_value_lookup.py`

Result:

> `results/cards/KSQ010_TWO_STAGE_CODEBOOK_VALUE_LOOKUP/ksq010_two_stage_codebook_value_lookup_smoke_limit10.json`

Status: codebook_positive_failed.

## Route Decision

- route decision: `two_stage_codebook_diagnostic`
- exported diagnostic class: `CODEBOOK_POSITIVE_FAILED`
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
| `exact_codebook_bridge_passed` | `false` |
| `bridge_vs_answer_for_alt_passed` | `false` |
| `answer_for_controls_passed` | `false` |
| `codebook_locality_controls_passed` | `false` |
| `abstain_controls_passed` | `false` |
| `all_panels_passed` | `false` |
| `source_disjoint_holdout_passed` | `false` |
| `candidate_and_output_margins_reported` | `true` |

## Selected Template

- selected template: `tag_rows`
- selection key: `[0.0, 0.8, 0.9, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -0.37, -0.9, 0.0, -2]`
- failed panels: `["exact_codebook_bridge", "missing_code_value", "conflicting_entity_codes", "conflicting_code_values", "uncounted_entity_code", "uncounted_code_value", "quoted_bridge_rows", "counted_answer_for_only", "bridge_vs_answer_for_alt"]`

## Selected Panels

| Panel | Rows | Label Counts | Key Rate |
| --- | ---: | --- | ---: |
| `exact_codebook_bridge` | 10 | `{"evidence_answer": 8, "unparsed": 2}` | 0.800 |
| `missing_entity_code` | 10 | `{"control_abstain": 9, "unparsed": 1}` | 0.900 |
| `missing_code_value` | 10 | `{"unparsed": 10}` | 0.000 |
| `unrelated_entity_code` | 10 | `{"control_abstain": 10}` | 1.000 |
| `conflicting_entity_codes` | 10 | `{"conflict_value_selected": 10}` | 0.000 |
| `conflicting_code_values` | 10 | `{"conflict_value_selected": 10}` | 0.000 |
| `uncounted_entity_code` | 10 | `{"unparsed": 10}` | 0.000 |
| `uncounted_code_value` | 10 | `{"uncounted_code_value_reproduced": 4, "unparsed": 6}` | 0.000 |
| `quoted_bridge_rows` | 10 | `{"quoted_bridge_reproduced": 3, "unparsed": 7}` | 0.000 |
| `counted_answer_for_only` | 10 | `{"counted_answer_for_reproduced": 10}` | 0.000 |
| `bridge_vs_answer_for_alt` | 10 | `{"answer_for_alt_overrode_bridge": 9, "unparsed": 1}` | 0.000 |
| `query_only_control` | 10 | `{"control_abstain": 10}` | 1.000 |

## Forbidden Claims

- KSQ010 is a mechanism card.
- KSQ010 proves real uncertainty, refusal, factual correction, or knowledge control.
- KSQ010 licenses hidden-state intervention or mechanism claims.
