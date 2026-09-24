# KSQ009 Schema-Specific Value Lookup Status

Date: 2026-07-02

Runner:

> `code/ksq009_schema_specific_value_lookup.py`

Result:

> `results/cards/KSQ009_SCHEMA_SPECIFIC_VALUE_LOOKUP/ksq009_schema_specific_value_lookup_smoke_limit10.json`

Status: schema_specific_positive_failed.

## Route Decision

- route decision: `schema_specific_lookup_diagnostic`
- exported diagnostic class: `SCHEMA_SPECIFIC_POSITIVE_FAILED`
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
| `exact_allowed_value_passed` | `false` |
| `allowed_value_vs_uncounted_answer_for_alt_passed` | `false` |
| `answer_for_schema_controls_passed` | `false` |
| `allowed_row_locality_controls_passed` | `false` |
| `abstain_controls_passed` | `false` |
| `all_panels_passed` | `false` |
| `source_disjoint_holdout_passed` | `false` |
| `candidate_and_output_margins_reported` | `true` |

## Selected Template

- selected template: `kv_lines`
- selection key: `[0.0, 0.2, 1.0, 1.0, 0.5, 0.1, 0.3, 0.8, 0.6, 0.0, 1.0, -0.3375, -0.9, 0.9, -1]`
- failed panels: `["exact_allowed_value", "conflicting_allowed_values", "counted_wrong_schema_answer_for", "uncounted_wrong_schema_answer_for", "uncounted_allowed_value", "quoted_allowed_value", "allowed_value_vs_uncounted_answer_for_alt"]`

## Selected Panels

| Panel | Rows | Label Counts | Key Rate |
| --- | ---: | --- | ---: |
| `exact_allowed_value` | 10 | `{"abstain": 8, "evidence_answer": 2}` | 0.200 |
| `absent_allowed_value` | 10 | `{"control_abstain": 10}` | 1.000 |
| `unrelated_entity_allowed_value` | 10 | `{"control_abstain": 10}` | 1.000 |
| `conflicting_allowed_values` | 10 | `{"conflict_value_selected": 5, "control_abstain": 5}` | 0.500 |
| `counted_wrong_schema_answer_for` | 10 | `{"control_abstain": 1, "counted_answer_for_reproduced": 9}` | 0.100 |
| `uncounted_wrong_schema_answer_for` | 10 | `{"control_abstain": 3, "uncounted_answer_for_reproduced": 7}` | 0.300 |
| `uncounted_allowed_value` | 10 | `{"control_abstain": 8, "uncounted_allowed_reproduced": 2}` | 0.800 |
| `quoted_allowed_value` | 10 | `{"control_abstain": 6, "quoted_allowed_reproduced": 4}` | 0.600 |
| `allowed_value_vs_uncounted_answer_for_alt` | 10 | `{"answer_for_alt_overrode_allowed_value": 9, "unparsed": 1}` | 0.000 |
| `query_only_control` | 10 | `{"control_abstain": 10}` | 1.000 |

## Forbidden Claims

- KSQ009 is a mechanism card.
- KSQ009 proves real uncertainty, refusal, factual correction, or knowledge control.
- KSQ009 licenses hidden-state intervention or mechanism claims.
