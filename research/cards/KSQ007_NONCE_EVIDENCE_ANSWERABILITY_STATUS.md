# KSQ007 Nonce-Evidence Answerability Status

Date: 2026-07-02

Runner:

> `code/ksq007_nonce_evidence_answerability_calibrator.py`

Result:

> `results/cards/KSQ007_NONCE_EVIDENCE_ANSWERABILITY/ksq007_nonce_evidence_answerability_smoke_limit10.json`

Status: nonce_evidence_claim_or_mention_control_failed.

## Route Decision

- route decision: `nonce_evidence_answerability_failed`
- exported diagnostic class: `NONCE_EVIDENCE_ANSWERABILITY_BOUNDARY`
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
| `exact_evidence_rows_passed` | `true` |
| `absent_evidence_rows_passed` | `true` |
| `unrelated_entity_rows_passed` | `true` |
| `conflicting_evidence_rows_passed` | `true` |
| `claim_only_and_mention_only_controls_passed` | `false` |
| `query_only_control_passed` | `true` |
| `source_disjoint_answerability_holdout_passed` | `false` |
| `candidate_and_output_margins_reported` | `true` |

## Selected Template

- selected template: `evidence_rows`
- selection key: `[0.4, 0.9, 1.0, 1.0, 0.9, 0.4, 1.0, 1.0, -0.15, -0.1, 1.0, 0]`

## Selected Panels

| Panel | Rows | Label Counts | Key Rate |
| --- | ---: | --- | ---: |
| `exact_evidence_rows` | 10 | `{"abstain": 1, "evidence_answer": 9}` | 0.900 |
| `absent_evidence_rows` | 10 | `{"abstain": 10}` | 1.000 |
| `unrelated_entity_rows` | 10 | `{"control_abstain": 10}` | 1.000 |
| `conflicting_evidence_rows` | 10 | `{"abstain": 9, "conflict_value_selected": 1}` | 0.900 |
| `claim_only_control` | 10 | `{"claim_only_reproduced": 6, "control_abstain": 4}` | 0.400 |
| `mention_only_control` | 10 | `{"control_abstain": 10}` | 1.000 |
| `query_only_control` | 10 | `{"control_abstain": 10}` | 1.000 |

## Forbidden Claims

- KSQ007 is a mechanism card.
- KSQ007 proves real uncertainty, refusal, factual correction, or knowledge control.
- KSQ007 licenses hidden-state probing or intervention.
