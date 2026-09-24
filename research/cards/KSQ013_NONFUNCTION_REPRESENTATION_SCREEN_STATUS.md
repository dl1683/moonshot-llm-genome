# KSQ013 Nonfunction Representation Screen Status

Date: 2026-07-02

Runner:

> `code/ksq013_nonfunction_representation_screen.py`

Result:

> `results/cards/KSQ013_NONFUNCTION_REPRESENTATION_SCREEN/ksq013_nonfunction_representation_screen_smoke_limit10.json`

Status: nonfunction_representation_control_and_repair_leak.

## Route Decision

- route decision: `nonfunction_representation_diagnostic`
- exported diagnostic class: `NONFUNCTION_REPRESENTATION_CONTROL_AND_REPAIR_LEAK`
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
| `nonfunction_panels_passed` | `false` |
| `nonfunction_controls_passed` | `false` |
| `all_panels_passed` | `false` |
| `source_disjoint_holdout_passed` | `false` |
| `candidate_and_output_margins_reported` | `true` |

## Selected Template

- selected template: `tag_rows`
- selection key: `[0.6, 0.6, 0.9, 0.9, 0.8, 0.7, 0.8, 0.6, 0.9, 0.8, 1.0, 1.0, 0.6, 1.0, 0]`
- failed panels: `["value_bank_alt", "metadata_value_alt", "decoy_entity_pair_alt", "entity_value_separate_alt", "bare_alt_mention", "entity_value_separate_only_control"]`

## Selected Panels

| Panel | Rows | Label Counts | Key Rate |
| --- | ---: | --- | ---: |
| `exact_bridge` | 10 | `{"evidence_answer": 9, "unparsed": 1}` | 0.900 |
| `raw_answer_for_alt` | 10 | `{"raw_answer_channel_reproduced": 9, "unparsed": 1}` | 0.900 |
| `value_bank_alt` | 10 | `{"evidence_answer": 8, "unparsed": 2}` | 0.800 |
| `metadata_value_alt` | 10 | `{"abstain": 1, "evidence_answer": 7, "unparsed": 2}` | 0.700 |
| `decoy_entity_pair_alt` | 10 | `{"evidence_answer": 8, "unparsed": 2}` | 0.800 |
| `entity_value_separate_alt` | 10 | `{"abstain": 1, "evidence_answer": 6, "unparsed": 3}` | 0.600 |
| `entity_value_slash_alt` | 10 | `{"evidence_answer": 9, "unparsed": 1}` | 0.900 |
| `bare_alt_mention` | 10 | `{"evidence_answer": 8, "unparsed": 2}` | 0.800 |
| `value_bank_only_control` | 10 | `{"control_abstain": 10}` | 1.000 |
| `decoy_pair_only_control` | 10 | `{"control_abstain": 10}` | 1.000 |
| `entity_value_separate_only_control` | 10 | `{"control_abstain": 6, "control_reproduced_value": 4}` | 0.600 |
| `query_only_control` | 10 | `{"control_abstain": 10}` | 1.000 |

## Forbidden Claims

- KSQ013 is a mechanism card.
- KSQ013 proves real uncertainty, refusal, factual correction, or knowledge control.
- KSQ013 licenses hidden-state intervention or mechanism claims.
