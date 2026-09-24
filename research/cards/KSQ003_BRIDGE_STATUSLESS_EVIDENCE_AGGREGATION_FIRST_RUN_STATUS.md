# KSQ003 Bridge Statusless Evidence Aggregation First-Run Status

Date: 2026-07-01

Runner:

> `code/ksq003_bridge_statusless_evidence_aggregation_first_run.py`

Result:

> `results/cards/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION/ksq003_bridge_statusless_evidence_aggregation_first_run.json`

Status: structural_passed_behavior_smoke_failed.

## Verdict

The structural gate passed, but the 10-source model smoke
failed behavior admission: the model followed the local
table even when evidence mismatched or was incomplete.
Hidden-state work remains forbidden.

## Structural Criteria

| Criterion | Passed |
| --- | --- |
| `expected_row_count` | `true` |
| `all_panels_present` | `true` |
| `all_templates_present` | `true` |
| `source_split_disjoint` | `true` |
| `holdout_sources_present` | `true` |
| `calibration_sources_present` | `true` |
| `real_atomic_number_hidden_in_conflicts` | `true` |
| `lure_atomic_number_hidden_in_conflicts` | `true` |
| `local_number_not_used_as_evidence` | `true` |
| `answer_absent_omits_query_local_number` | `true` |
| `candidate_answers_parseable` | `true` |
| `no_candidate_collisions` | `true` |
| `single_answer_suffix` | `true` |
| `primary_prompts_have_no_status_lexemes` | `true` |
| `primary_prompts_avoid_closed_route_lexemes` | `true` |
| `primary_expected_labels_balanced` | `true` |

## Counts

- records: `840`
- sources: `40`
- panels: `{"all_evidence_fit_conflict": 120, "answer_absent_and_side_null": 120, "learned_fact_direct_control": 120, "one_evidence_mismatch_conflict": 120, "parity_only_ablation": 120, "source_local_direct_control": 120, "symbol_only_ablation": 120}`
- templates: `{"compact_fit": 280, "evidence_packet": 280, "field_observations": 280}`
- split source counts: `{"calibration": 8, "discovery": 24, "holdout": 8}`

## Ten-Source Behavior Smoke

- result: `results/cards/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION/ksq003_bridge_statusless_evidence_aggregation_smoke_limit10.json`
- diagnostic class: `smoke_only`
- behavior ready: `false`
- selected template: `compact_fit`
- records: `210`
- sources: `10`
- candidate/output margins reported: `true`

| Panel | Label counts |
| --- | --- |
| `source_local_direct_control` | `{"local_number": 10}` |
| `learned_fact_direct_control` | `{"atomic_number": 10}` |
| `all_evidence_fit_conflict` | `{"local_number": 10}` |
| `one_evidence_mismatch_conflict` | `{"local_number": 10}` |
| `symbol_only_ablation` | `{"local_number": 10}` |
| `parity_only_ablation` | `{"local_number": 10}` |
| `answer_absent_and_side_null` | `{"unknown": 10}` |

Smoke interpretation: direct local-number control, learned
atomic-number control, all-evidence-fit local routing, and
answer-absent nulls worked in the selected template. The
one-evidence-mismatch branch and both single-feature
ablations collapsed to local-number outputs, so KSQ003 is
not a behavior-ready substrate.

## Forbidden Claims

- KSQ003 is a mechanism card.
- KSQ003 supports intervention.
- KSQ003 found an internal knowledge-control surface.
- Any hidden-state or causal claim follows from this first run alone.
