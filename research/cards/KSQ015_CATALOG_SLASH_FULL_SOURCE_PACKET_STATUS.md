# KSQ015 Catalog Slash Full-Source Packet Status

Date: 2026-07-02

Runner:

> `code/ksq015_catalog_slash_full_source_packet.py`

Result:

> `results/cards/KSQ015_CATALOG_SLASH_FULL_SOURCE_PACKET/ksq015_catalog_slash_full_source_full_behavior.json`

Status: catalog_slash_bridge_positive_failed.

## Route Decision

- route decision: `catalog_slash_full_source_diagnostic`
- exported diagnostic class: `CATALOG_SLASH_BRIDGE_POSITIVE_FAILED`
- behavior ready: `false`
- signature screen allowed: `false`

## Gate Criteria

| Criterion | Passed |
| --- | --- |
| `full_run_mode` | `true` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `true` |
| `source_disjoint_holdout` | `true` |
| `selected_prompt_audit_passed` | `true` |
| `exact_bridge_passed` | `false` |
| `raw_answer_channel_positive_control_passed` | `true` |
| `catalog_slash_bridge_panels_passed` | `false` |
| `catalog_slash_controls_passed` | `true` |
| `all_panels_passed` | `false` |
| `source_disjoint_holdout_passed` | `false` |
| `candidate_and_output_margins_reported` | `true` |

## Selected Template

- selected template: `tag_rows`
- selection key: `[0.875, 1.0, 0.875, 0.975, 0.9, 0.875, 0.875, 1.0, 1.0, 1.0, 1.0, 0]`
- failed panels: `["exact_bridge", "catalog_slash_reversed_entity_alt"]`

## Selected Panels

| Panel | Rows | Label Counts | Key Rate |
| --- | ---: | --- | ---: |
| `exact_bridge` | 40 | `{"evidence_answer": 35, "unparsed": 5}` | 0.875 |
| `raw_answer_for_alt` | 40 | `{"raw_answer_channel_reproduced": 39, "unparsed": 1}` | 0.975 |
| `catalog_slash_entity_alt` | 40 | `{"abstain": 1, "evidence_answer": 36, "unparsed": 3}` | 0.900 |
| `catalog_slash_decoy_alt` | 40 | `{"abstain": 3, "evidence_answer": 35, "unparsed": 2}` | 0.875 |
| `catalog_slash_reversed_entity_alt` | 40 | `{"evidence_answer": 35, "unparsed": 5}` | 0.875 |
| `catalog_slash_entity_only_control` | 40 | `{"control_abstain": 40}` | 1.000 |
| `catalog_slash_decoy_only_control` | 40 | `{"control_abstain": 40}` | 1.000 |
| `catalog_slash_reversed_entity_only_control` | 40 | `{"control_abstain": 40}` | 1.000 |
| `query_only_control` | 40 | `{"control_abstain": 40}` | 1.000 |

## Forbidden Claims

- KSQ015 is a mechanism card.
- KSQ015 proves real uncertainty, refusal, factual correction, or knowledge control.
- KSQ015 licenses hidden-state intervention or mechanism claims.
