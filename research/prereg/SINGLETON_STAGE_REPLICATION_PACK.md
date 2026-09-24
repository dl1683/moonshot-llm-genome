# Singleton Stage Replication Pack

Date: 2026-07-01

Status: preregistered law-replication packet; no law promotion claimed.

Machine-readable artifact:

> `data/singleton_stage_replication_pack.json`

Builder:

> `code/singleton_stage_replication_pack.py`

Commands:

```powershell
python code\singleton_stage_replication_pack.py --write
python code\singleton_stage_replication_pack.py
python code\validate_control_surface_atlas.py
```

## Target Gap

- work order: `replicate_singleton_stage_laws`;
- urgency: `high`;
- track: `law_replication`;
- target gaps: `["axis_rules_sparse_or_singleton_heavy", "singleton_terminal_stage_evidence", "output_geometry_pressure_mixed", "domain_coverage_knowledge_bridge_dominates_failures", "model_family_coverage_qwen_dominant"]`;
- iteration budget: Three targeted row additions before new law language is allowed..

## Current Singleton Shape

- singleton terminal stages: `["intervention_failed", "pre_signature_prompt_channel_locality", "reliability_null_boundary"]`;
- singleton terminal rows: `["mc001g_gemma_truth_agreement", "mc012_reliability_labeled_numeric_arbitration", "mc005_associative_lookup"]`;
- singleton+sparse feature summaries: 82;
- axis evidence levels: `{"broad": 29, "singleton": 73, "sparse": 9, "supported": 12}`.

## Stage Targets

| Stage | Anchor Row | Current Singleton Feature Keys | Required New Rows | Stage Kill Rule |
| --- | --- | --- | ---: | --- |
| `intervention_failed` | `mc001g_gemma_truth_agreement` | `["route_disposition:disposition=failed_intervention_or_mechanism_route", "reliability_class:reliability_class=not_reliable_failed_intervention_route"]` | 2 | Kill the broad law if two new rows under the same claimed predictor land in different terminal stages. |
| `pre_signature_prompt_channel_locality` | `mc012_reliability_labeled_numeric_arbitration` | `["route_disposition:disposition=prompt_visible_positive_control", "primary_blocker:primary_blocker=prompt_visible_positive_control", "reliability_class:reliability_class=not_reliable_prompt_visible_positive_control", "diagnostic:diagnostic=MC012_BEHAVIOR_READY_PROMPT_VISIBLE_BRIDGE", "diagnostic:diagnostic=RELIABILITY_BEHAVIOR_CONTRAST_PASSED", "diagnostic:diagnostic=RELIABILITY_PROMPT_CHANNEL_VISIBLE"]` | 2 | Kill the broad law if two new rows under the same claimed predictor land in different terminal stages. |
| `reliability_null_boundary` | `mc005_associative_lookup` | `["route_disposition:disposition=bounded_mechanism_frozen", "primary_blocker:primary_blocker=bounded_internal_causal_with_null_boundary", "frontier_class:frontier_class=causal_surface_not_timing_frontier", "reliability_class:reliability_class=bounded_reliability_reference", "mixture_axis:causal_control=bounded", "mixture_axis:local_internal_path=high", "mixture_axis:output_geometry=medium", "mixture_axis:source_token_dependence=dominant", "pressure_class:pressure_class=internal_causal_surface", "diagnostic:diagnostic=MC005_BOUNDED_ATTENTION_WRITE_MEDIATION", "diagnostic:diagnostic=MODEL_SIZE_NULL_FRAGILITY", "diagnostic:diagnostic=NULL_ROW_LOW_MARGIN_FLIP"]` | 2 | Kill the broad law if two new rows under the same claimed predictor land in different terminal stages. |

## Replication Proposals


### `intervention_failed`

- proposal: `leadtime_signature_intervention_failure`;
- material difference: A predecision monitor task outside truth/agreement where the signature beats output, shuffle, and subgroup controls before an intervention is attempted.
- minimum evidence: `["behavior substrate passes", "hidden signature survives holdout, shuffle, subgroup, same-stage output, and final-stage output controls", "predeclared additive or patch intervention is attempted", "intervention fails to move behavior in the predicted direction or creates side effects", "failure is not reclassified as output-shadow or monitor-only"]`;
- promotion rule: Count as intervention_failed only if the signature gate passes first and the predicted intervention then fails.
- kill rule: Do not count output-confounded or shuffle-fragile signals as intervention failures; those remain signature-stage failures.
- proposal: `source_path_signature_intervention_failure`;
- material difference: A source-path or lookup-like task outside MC005 where a local path signature passes but the first causal intervention fails.
- minimum evidence: `["source-disjoint behavior substrate passes", "source-path signature beats deletion, neutral rewrite, query-only, and output/candidate controls", "local intervention is predeclared before observing effect size", "primary direction fails or side rows/null rows fail", "failed route is not patched repeatedly before classification"]`;
- promotion rule: Count as a replication only if a real signature reaches the intervention gate and dies there.
- kill rule: Kill the broad intervention-failed law if new rows die at behavior substrate, prompt channel, or output geometry instead.

### `pre_signature_prompt_channel_locality`

- proposal: `prompt_channel_locality_non_numeric_authority`;
- material difference: Non-numeric source-authority task with the visible channel carried by source wording rather than trusted/untrusted labels.
- minimum evidence: `["behavior contrast passes on source-disjoint holdout", "direct local and learned controls pass", "null rows pass", "visible channel ablation or matching collapses the contrast", "no hidden-state work is run after prompt-channel locality is shown"]`;
- promotion rule: Count as a replication only if the behavior contrast exists and the prompt-channel locality control explains it.
- kill rule: Do not count it for this stage if the contrast fails before the prompt-channel control or if hidden-state work is needed to diagnose the failure.
- proposal: `prompt_channel_locality_answer_schema_authority`;
- material difference: Answer-schema or instruction-channel authority task where the rule is visible through formatting rather than source status.
- minimum evidence: `["same semantic task under at least two answer schemas", "schema-visible positive control passes", "schema-neutral or schema-matched control collapses the contrast", "output/candidate baselines are reported", "source-disjoint holdout remains balanced"]`;
- promotion rule: Count as a replication only if visible schema authority, not hidden state, carries the behavior.
- kill rule: Kill the prompt-channel-locality law if a schema-neutral version preserves the contrast and licenses hidden-state work.

### `reliability_null_boundary`

- proposal: `mc005_width_null_boundary_transfer`;
- material difference: Non-Qwen or prompt-family transfer of the MC005 bounded reference, evaluated with null locality as a primary gate.
- minimum evidence: `["answer-present primary effect reproduces", "local or homologous intervention moves primary behavior", "answer-absent null rows are stratified by margin", "null flips or side effects persist after the planned transfer panel", "transfer is bounded rather than promoted"]`;
- promotion rule: Count as a reliability-null-boundary replication only if primary intervention works but null locality blocks promotion.
- kill rule: Do not count failed primary transfer as a reliability boundary; that is transfer-primary failure.
- proposal: `new_synthetic_lookup_null_boundary`;
- material difference: A second synthetic or semi-synthetic source-value task with a different answer interface and a predeclared local intervention.
- minimum evidence: `["behavior table passes answer-present and answer-absent panels", "internal/local path signature passes holdout and source controls", "intervention moves answer-present rows in the predicted direction", "answer-absent, side-row, robustness, or fluency gates remain bounded", "ordinary repair budget is exhausted before classification"]`;
- promotion rule: Count as a replication only if the route reaches reliability after a successful primary intervention.
- kill rule: Kill the reliability-boundary law for this proposal if it dies at signature, intervention, transfer, or output geometry instead.

## Decision Rules

- promotion rule: Promote a law only when a feature remains pure or nearly pure across at least three materially distinct rows.
- bound rule: Bound if the feature is predictive inside one behavior family but mixed across families.
- kill rule: Kill a proposed law if two new rows land in different terminal stages under the same claimed predictor.
- containment rule: Singleton features stay calibration labels, not predictive laws.
- export rule: Export the surviving or killed feature into the axis interaction map with evidence_level preserved.

## Claim Boundary

This packet makes singleton terminal-stage law replication testable by naming anchor rows, required new rows, minimum evidence, and kill rules for each target stage.

This packet does not reduce the singleton count, does not promote any predictive law, and does not add mechanism evidence.

## What This Proves

It proves that singleton-stage law replication is now a concrete
test plan rather than loose language. Each target stage has two
materially distinct proposed rows and a predeclared failure mode.

## What It Does Not Prove

It does not promote a law, close the singleton gap, or add a mechanism
claim. The current singleton count remains the current singleton count
until new rows land in the generated atlas.
