# KSQ005 Grounded Answerability First-Run Status

Date: 2026-07-01

Runner:

> `code/ksq005_uncertainty_grounded_answerability_first_run.py`

Result:

> `results/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY/ksq005_uncertainty_grounded_answerability_first_run.json`

Status: structural_passed_with_behavior_smoke.

## Verdict

The structural gate passed and the 10-source behavior smoke exists. This is not a hidden-state license.

## Structural Criteria

| Criterion | Passed |
| --- | --- |
| `expected_row_count` | `true` |
| `all_panels_present` | `true` |
| `all_templates_present` | `true` |
| `no_duplicate_record_ids` | `true` |
| `source_split_disjoint` | `true` |
| `holdout_sources_present` | `true` |
| `calibration_sources_present` | `true` |
| `candidate_answers_parseable` | `true` |
| `no_candidate_collisions` | `true` |
| `prompt_audit_passed` | `true` |
| `label_grounding_separates_real_and_nonce_entities` | `true` |
| `shared_requested_mode_suffix` | `true` |

## Counts

- records: `480`
- sources: `40`
- panels: `{"contradicted_context_rows": 120, "known_factual_direct": 120, "unknown_nonce_rows": 120, "unsupported_context_rows": 120}`
- templates: `{"compact_field": 160, "plain_question": 160, "reference_note": 160}`
- split source counts: `{"calibration": 8, "discovery": 24, "holdout": 8}`

## Behavior Result

- result: `results/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY/ksq005_uncertainty_grounded_answerability_smoke_limit10.json`
- diagnostic class: `unknown_nonce_abstention_failed`
- behavior ready: `false`
- behavior candidate: `false`
- selected template: `reference_note`
- candidate/output margins reported: `true`

| Panel | Label counts |
| --- | --- |
| `known_factual_direct` | `{"known_correct": 8, "known_wrong_candidate": 1, "unparsed": 1}` |
| `unknown_nonce_rows` | `{"abstain": 4, "unparsed": 6}` |
| `unsupported_context_rows` | `{"abstain": 4, "unparsed": 3, "unsupported_answer": 3}` |
| `contradicted_context_rows` | `{"abstain": 3, "corrected": 5, "false_accept": 1, "wrong_answer": 1}` |

Interpretation: KSQ005 only becomes behavior-ready if the same
answer schema supports ordinary known facts, abstention on nonce
unknowns, abstention on unsupported context, and rejection of
prompt-listed false capitals. A prompt/schema-only abstention
pattern is a diagnostic, not uncertainty control.

## Forbidden Claims

- KSQ005 is a mechanism card.
- KSQ005 supports intervention.
- KSQ005 found an internal uncertainty, refusal, correction, or knowledge-control surface.
- Any hidden-state or causal claim follows from this first run alone.
