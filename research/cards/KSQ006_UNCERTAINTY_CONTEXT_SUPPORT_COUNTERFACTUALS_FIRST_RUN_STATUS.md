# KSQ006 Context-Support Counterfactuals First-Run Status

Date: 2026-07-01

Runner:

> `code/ksq006_uncertainty_context_support_counterfactuals_first_run.py`

Result:

> `results/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS/ksq006_uncertainty_context_support_counterfactuals_first_run.json`

Status: structural_passed_with_behavior_smoke.

## Verdict

The structural gate passed and the 10-source behavior smoke exists. This is not a hidden-state license.

## Structural Criteria

| Criterion | Passed |
| --- | --- |
| `expected_row_count` | `true` |
| `all_panels_present` | `true` |
| `all_templates_present` | `true` |
| `control_subtypes_present` | `true` |
| `no_duplicate_record_ids` | `true` |
| `source_split_disjoint` | `true` |
| `holdout_sources_present` | `true` |
| `calibration_sources_present` | `true` |
| `candidate_answers_parseable` | `true` |
| `no_candidate_collisions` | `true` |
| `prompt_audit_passed` | `true` |
| `no_support_word_prompt_channel` | `true` |
| `shared_requested_mode_suffix` | `true` |

## Counts

- records: `720`
- sources: `40`
- panels: `{"claim_only_and_context_only_controls": 240, "contradicting_context_rows": 120, "insufficient_context_rows": 120, "irrelevant_context_rows": 120, "supported_context_rows": 120}`
- templates: `{"compact_record": 240, "field_form": 240, "note_question": 240}`
- control subtypes: `{"claim_only_control": 120, "context_only_control": 120}`
- split source counts: `{"calibration": 8, "discovery": 24, "holdout": 8}`

## Behavior Result

- result: `results/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS/ksq006_uncertainty_context_support_counterfactuals_smoke_limit10.json`
- diagnostic class: `insufficient_context_abstention_failed`
- behavior ready: `false`
- behavior candidate: `false`
- selected template: `field_form`
- candidate/output margins reported: `true`

| Panel | Label counts |
| --- | --- |
| `supported_context_rows` | `{"abstain": 1, "supported_answer": 9}` |
| `irrelevant_context_rows` | `{"abstain": 10}` |
| `contradicting_context_rows` | `{"contradiction_detected": 8, "false_accept": 1, "true_answer_despite_contradiction": 1}` |
| `insufficient_context_rows` | `{"abstain": 5, "unsupported_answer": 5}` |
| `claim_only_and_context_only_controls` | `{"claim_only_reproduced_supported": 8, "context_only_reproduced_supported": 10, "control_abstain": 2}` |

Control subtype counts:

| Control subtype | Label counts |
| --- | --- |
| `claim_only_control` | `{"claim_only_reproduced_supported": 8, "control_abstain": 2}` |
| `context_only_control` | `{"context_only_reproduced_supported": 10}` |

Interpretation: KSQ006 only becomes behavior-ready if support
sensitivity cannot be reproduced by claim-only text, context-only
city mention, support-word prompts, answer schema, or output
geometry. A supported-row win with failed counterfactual controls
is a diagnostic, not uncertainty control.

## Forbidden Claims

- KSQ006 is a mechanism card.
- KSQ006 supports intervention.
- KSQ006 found an internal uncertainty, refusal, correction, context-support, or knowledge-control surface.
- Any hidden-state or causal claim follows from this first run alone.
