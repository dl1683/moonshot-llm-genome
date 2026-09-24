# KSQ002 Familiar Entity Source-Rewrite Equivalence First-Run Status

Date: 2026-07-01

Runner:

> `code/ksq002_familiar_entity_source_rewrite_equivalence_first_run.py`

Result:

> `results/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE/ksq002_familiar_entity_source_rewrite_equivalence_full_behavior.json`

Status: source_rewrite_holdout_failed.

## Verdict

The behavior gate failed. Hidden-state work remains forbidden.

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

## Counts

- records: `720`
- sources: `40`
- panels: `{"baseline_source_value_lookup": 120, "neutral_rewrite_lookup": 120, "query_only_control": 120, "rewrite_output_geometry_audit": 120, "source_deletion": 120, "source_disjoint_rewrite_holdout": 120}`
- templates: `{"compact_rewrite": 240, "registry_question": 240, "sentence_rewrite": 240}`
- split source counts: `{"calibration": 8, "discovery": 24, "holdout": 8}`

## Behavior Result

- result: `results/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE/ksq002_familiar_entity_source_rewrite_equivalence_full_behavior.json`
- diagnostic class: `source_rewrite_holdout_failed`
- behavior ready: `false`
- behavior candidate: `false`
- selected template: `sentence_rewrite`
- rewrite delta from baseline: `0.075`
- candidate/output margins reported: `true`

| Panel | Label counts |
| --- | --- |
| `baseline_source_value_lookup` | `{"artificial_value": 39, "unknown": 1}` |
| `neutral_rewrite_lookup` | `{"artificial_value": 36, "unparsed": 4}` |
| `source_deletion` | `{"unknown": 40}` |
| `query_only_control` | `{"unknown": 40}` |
| `source_disjoint_rewrite_holdout` | `{"artificial_value": 36, "unparsed": 4}` |
| `rewrite_output_geometry_audit` | `{"artificial_value": 36, "unparsed": 4}` |

## Source-Disjoint Holdout Boundary

- rewrite holdout rows: `16`
- artificial-value rows: `14`
- unparsed rows: `2`
- parseability rate: `0.875`
- artificial-value rate: `0.875`
- predeclared gate: `0.900`

Interpretation: KSQ002 only becomes behavior-ready if source-local
artificial lookup is robust to neutral rewrite and local to the
source row. The full run is close but still fails exactly at the
source-disjoint rewrite holdout parseability boundary. Deletion and query-only
controls are clean, so the current diagnostic is not generic source-presence
leakage; it is `SOURCE_REWRITE_HOLDOUT_PARSEABILITY_FAILURE`.

## Forbidden Claims

- KSQ002 is a mechanism card.
- KSQ002 supports intervention.
- KSQ002 found an internal source-channel or knowledge-control surface.
- Any hidden-state or causal claim follows from this first run alone.
