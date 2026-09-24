# KSQ001 Familiar Entity Prior Counterbalance First-Run Status

Date: 2026-07-01

Runner:

> `code/ksq001_familiar_entity_prior_counterbalance_first_run.py`

Result:

> `results/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE/ksq001_familiar_entity_prior_counterbalance_first_run.json`

Status: structural_passed_with_full_behavior.

## Verdict

The structural gate passed, but the full 40-source behavior run failed admission. Direct local lookup, direct semantic prior recall, nulls, holdout, and margin reporting survived; the selected conflict template failed parseability. Hidden-state work remains forbidden.

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
| `primary_expected_label_ambiguous` | `true` |

## Counts

- records: `480`
- sources: `40`
- panels: `{"answer_absent_irrelevant_nulls": 120, "semantic_prior_direct_control": 120, "semantic_prior_lure": 120, "source_local_artificial_lookup": 120}`
- templates: `{"association_question": 160, "compact_question": 160, "registry_question": 160}`
- split source counts: `{"calibration": 8, "discovery": 24, "holdout": 8}`

## Behavior Result

- result: `results/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE/ksq001_familiar_entity_prior_counterbalance_smoke_limit10.json`
- diagnostic class: `smoke_behavior_contrast_candidate`
- behavior ready: `false`
- behavior candidate: `true`
- selected template: `compact_question`
- candidate/output margins reported: `true`

| Panel | Label counts |
| --- | --- |
| `source_local_artificial_lookup` | `{"artificial_value": 10}` |
| `semantic_prior_direct_control` | `{"real_prior": 10}` |
| `semantic_prior_lure` | `{"artificial_value": 6, "real_prior": 1, "unknown": 2, "unparsed": 1}` |
| `answer_absent_irrelevant_nulls` | `{"unknown": 10}` |

Interpretation: KSQ001 only becomes behavior-ready if the familiar
entity conflict panel contains both prompt-local artificial values
and learned-prior answers while direct controls and nulls remain
clean. A one-branch outcome is a diagnostic, not a mechanism.

## Behavior Result

- result: `results/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE/ksq001_familiar_entity_prior_counterbalance_full_behavior.json`
- diagnostic class: `familiar_entity_conflict_parseability_failed`
- behavior ready: `false`
- behavior candidate: `false`
- selected template: `compact_question`
- candidate/output margins reported: `true`

| Panel | Label counts |
| --- | --- |
| `source_local_artificial_lookup` | `{"artificial_value": 40}` |
| `semantic_prior_direct_control` | `{"lure_value": 1, "real_prior": 38, "unparsed": 1}` |
| `semantic_prior_lure` | `{"artificial_value": 18, "real_prior": 2, "unknown": 4, "unparsed": 16}` |
| `answer_absent_irrelevant_nulls` | `{"unknown": 38, "unparsed": 2}` |

Interpretation: KSQ001 only becomes behavior-ready if the familiar
entity conflict panel contains both prompt-local artificial values
and learned-prior answers while direct controls and nulls remain
clean. A one-branch outcome is a diagnostic, not a mechanism.

## Forbidden Claims

- KSQ001 is a mechanism card.
- KSQ001 supports intervention.
- KSQ001 found an internal knowledge-control surface.
- Any hidden-state or causal claim follows from this first run alone.
