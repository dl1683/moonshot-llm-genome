# KSQ007B Claim-Channel Boundary Status

Date: 2026-07-02

Runner:

> `code/ksq007_claim_channel_boundary_audit.py`

Result:

> `results/cards/KSQ007_CLAIM_CHANNEL_BOUNDARY/ksq007_claim_channel_boundary_smoke_limit10.json`

Status: answer_for_syntax_claim_leak.

## Route Decision

- route decision: `claim_channel_boundary_diagnostic`
- exported diagnostic class: `ANSWER_FOR_SYNTAX_CLAIM_LEAK`
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
| `exact_evidence_positive_passed` | `true` |
| `claim_controls_passed` | `false` |
| `syntax_controls_passed` | `false` |
| `section_boundary_controls_passed` | `false` |
| `mention_and_query_baselines_passed` | `true` |
| `all_control_panels_passed` | `false` |
| `source_disjoint_claim_boundary_holdout_passed` | `false` |
| `candidate_and_output_margins_reported` | `true` |

## Selected Template

- selected template: `counted_uncounted_sections`
- selection key: `[0.3, 0.9, 0.9, 0.6, 0.3, 1.0, 0.8, 1.0, 1.0, 1.0, 1.0, -0.15555555555555556, 1.0, -2]`
- failed control panels: `["claim_prose", "bare_same_syntax", "other_block_evidence_row"]`

## Selected Panels

| Panel | Rows | Label Counts | Key Rate |
| --- | ---: | --- | ---: |
| `exact_evidence_positive` | 10 | `{"abstain": 1, "evidence_answer": 9}` | 0.900 |
| `claim_same_syntax` | 10 | `{"claim_same_syntax_reproduced": 1, "control_abstain": 9}` | 0.900 |
| `claim_prose` | 10 | `{"claim_prose_reproduced": 4, "control_abstain": 6}` | 0.600 |
| `bare_same_syntax` | 10 | `{"bare_same_syntax_reproduced": 7, "control_abstain": 3}` | 0.300 |
| `not_evidence_prefix` | 10 | `{"control_abstain": 10}` | 1.000 |
| `other_block_evidence_row` | 10 | `{"control_abstain": 8, "other_block_evidence_reproduced": 2}` | 0.800 |
| `quoted_evidence_syntax` | 10 | `{"control_abstain": 10}` | 1.000 |
| `wrong_predicate_claim` | 10 | `{"control_abstain": 10}` | 1.000 |
| `mention_only` | 10 | `{"control_abstain": 10}` | 1.000 |
| `query_only` | 10 | `{"control_abstain": 10}` | 1.000 |

## Forbidden Claims

- KSQ007B is a mechanism card.
- KSQ007B proves real uncertainty, refusal, factual correction, or knowledge control.
- KSQ007B licenses hidden-state probing or intervention.
