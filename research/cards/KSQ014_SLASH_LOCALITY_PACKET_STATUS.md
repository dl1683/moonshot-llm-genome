# KSQ014 Slash Locality Packet Status

Date: 2026-07-02

Runner:

> `code/ksq014_slash_locality_packet.py`

Result:

> `results/cards/KSQ014_SLASH_LOCALITY_PACKET/ksq014_slash_locality_packet_smoke_limit10.json`

Status: slash_locality_bridge_loss.

## Route Decision

- route decision: `slash_locality_diagnostic`
- exported diagnostic class: `SLASH_LOCALITY_BRIDGE_LOSS`
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
| `slash_bridge_panels_passed` | `false` |
| `slash_controls_passed` | `true` |
| `all_panels_passed` | `false` |
| `source_disjoint_holdout_passed` | `false` |
| `candidate_and_output_margins_reported` | `true` |

## Selected Template

- selected template: `tag_rows`
- selection key: `[0.6, 1.0, 0.9, 0.9, 0.9, 0.9, 0.6, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 0]`
- failed panels: `["bare_slash_entity_alt"]`

## Selected Panels

| Panel | Rows | Label Counts | Key Rate |
| --- | ---: | --- | ---: |
| `exact_bridge` | 10 | `{"evidence_answer": 9, "unparsed": 1}` | 0.900 |
| `raw_answer_for_alt` | 10 | `{"raw_answer_channel_reproduced": 9, "unparsed": 1}` | 0.900 |
| `catalog_slash_entity_alt` | 10 | `{"evidence_answer": 9, "unparsed": 1}` | 0.900 |
| `catalog_slash_decoy_alt` | 10 | `{"abstain": 1, "evidence_answer": 9}` | 0.900 |
| `bare_slash_entity_alt` | 10 | `{"evidence_answer": 6, "unparsed": 4}` | 0.600 |
| `catalog_slash_reversed_entity_alt` | 10 | `{"evidence_answer": 9, "unparsed": 1}` | 0.900 |
| `catalog_slash_entity_only_control` | 10 | `{"control_abstain": 10}` | 1.000 |
| `catalog_slash_decoy_only_control` | 10 | `{"control_abstain": 10}` | 1.000 |
| `bare_slash_entity_only_control` | 10 | `{"control_abstain": 10}` | 1.000 |
| `catalog_slash_reversed_entity_only_control` | 10 | `{"control_abstain": 10}` | 1.000 |
| `query_only_control` | 10 | `{"control_abstain": 10}` | 1.000 |

## Forbidden Claims

- KSQ014 is a mechanism card.
- KSQ014 proves real uncertainty, refusal, factual correction, or knowledge control.
- KSQ014 licenses hidden-state intervention or mechanism claims.
