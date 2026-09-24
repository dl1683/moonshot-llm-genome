# KSQ008 Neutral-Evidence Channel Repair Status

Date: 2026-07-02

Runner:

> `code/ksq008_neutral_evidence_channel_repair.py`

Result:

> `results/cards/KSQ008_NEUTRAL_EVIDENCE_CHANNEL_REPAIR/ksq008_neutral_evidence_channel_repair_smoke_limit10.json`

Status: neutral_evidence_positive_failed.

## Route Decision

- route decision: `neutral_evidence_repair_diagnostic`
- exported diagnostic class: `NEUTRAL_EVIDENCE_POSITIVE_FAILED`
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
| `exact_neutral_evidence_passed` | `false` |
| `neutral_evidence_vs_forbidden_bare_alt_passed` | `false` |
| `answer_for_forbidden_controls_passed` | `false` |
| `neutral_channel_locality_controls_passed` | `true` |
| `abstain_controls_passed` | `false` |
| `all_panels_passed` | `false` |
| `source_disjoint_holdout_passed` | `false` |
| `candidate_and_output_margins_reported` | `true` |

## Selected Template

- selected template: `field_registry`
- selection key: `[0.2, 0.5, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.2, 1.0, -0.07, -0.0, 1.0, -1]`
- failed panels: `["exact_neutral_evidence", "conflicting_neutral_evidence", "counted_wrong_schema_answer_for", "neutral_evidence_vs_forbidden_bare_alt"]`

## Selected Panels

| Panel | Rows | Label Counts | Key Rate |
| --- | ---: | --- | ---: |
| `exact_neutral_evidence` | 10 | `{"abstain": 5, "evidence_answer": 5}` | 0.500 |
| `absent_neutral_evidence` | 10 | `{"control_abstain": 10}` | 1.000 |
| `unrelated_entity_neutral` | 10 | `{"control_abstain": 10}` | 1.000 |
| `conflicting_neutral_evidence` | 10 | `{"conflict_value_selected": 2, "control_abstain": 8}` | 0.800 |
| `forbidden_bare_answer_for` | 10 | `{"control_abstain": 10}` | 1.000 |
| `forbidden_claim_answer_for` | 10 | `{"control_abstain": 10}` | 1.000 |
| `forbidden_prose_claim` | 10 | `{"control_abstain": 10}` | 1.000 |
| `uncounted_neutral_evidence` | 10 | `{"control_abstain": 10}` | 1.000 |
| `quoted_neutral_evidence` | 10 | `{"control_abstain": 10}` | 1.000 |
| `counted_wrong_schema_answer_for` | 10 | `{"control_abstain": 5, "counted_wrong_schema_reproduced": 5}` | 0.500 |
| `neutral_evidence_vs_forbidden_bare_alt` | 10 | `{"abstain": 8, "evidence_answer": 2}` | 0.200 |
| `query_only_control` | 10 | `{"control_abstain": 10}` | 1.000 |

## Forbidden Claims

- KSQ008 is a mechanism card.
- KSQ008 proves real uncertainty, refusal, factual correction, or knowledge control.
- KSQ008 licenses hidden-state intervention or mechanism claims.
