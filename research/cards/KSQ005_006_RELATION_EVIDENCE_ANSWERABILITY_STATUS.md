# KSQ005/KSQ006 Relation-Evidence Answerability Status

Date: 2026-07-02

Runner:

> `code/ksq005_006_relation_evidence_answerability_redesign.py`

Result:

> `results/cards/KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY/ksq005_006_relation_evidence_answerability_full_behavior.json`

Status: relation_evidence_unknown_nonce_failed.

## Route Decision

- route decision: `kill_current_uncertainty_route`
- exported diagnostic class: `RELATION_EVIDENCE_ANSWERABILITY_BOUNDARY`
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
| `known_factual_direct_passed` | `true` |
| `supported_relation_rows_passed` | `true` |
| `unknown_nonce_absent_rows_passed` | `false` |
| `unsupported_relation_rows_passed` | `false` |
| `contradictory_relation_rows_passed` | `false` |
| `claim_only_and_mention_only_controls_passed` | `false` |
| `source_disjoint_answerability_holdout_passed` | `false` |
| `candidate_and_output_margins_reported` | `true` |

## Selected Template

- selected template: `compact_relation`
- selection key: `[0.975, 1.0, 0.675, 0.7, 0.05, 0.0125, -0.9875, 0.675, -2]`

## Selected Panels

| Panel | Rows | Label Counts | Key Rate |
| --- | ---: | --- | ---: |
| `known_factual_direct` | 40 | `{"known_correct": 39, "known_wrong_candidate": 1}` | 0.975 |
| `supported_relation_rows` | 40 | `{"supported_answer": 40}` | 1.000 |
| `unknown_nonce_absent_rows` | 40 | `{"abstain": 27, "unparsed": 13}` | 0.675 |
| `unsupported_relation_rows` | 40 | `{"abstain": 28, "unparsed": 2, "unsupported_answer": 10}` | 0.700 |
| `contradictory_relation_rows` | 40 | `{"abstain": 2, "prior_or_true_answer": 38}` | 0.050 |
| `claim_only_control` | 40 | `{"claim_only_reproduced_supported": 40}` | 0.000 |
| `mention_only_control` | 40 | `{"control_abstain": 1, "mention_only_reproduced_supported": 39}` | 0.025 |

## Forbidden Claims

- KSQ005/KSQ006 is a mechanism card.
- KSQ005/KSQ006 supports intervention.
- KSQ005/KSQ006 found an internal uncertainty, refusal, correction, context-support, or knowledge-control surface.
- Any hidden-state or causal claim follows from this redesign alone.
