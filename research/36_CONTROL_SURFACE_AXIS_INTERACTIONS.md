# Control-Surface Axis Interactions

Date: 2026-07-01

Status: generated predictive axis-interaction layer implemented and validated.

Machine-readable artifact:

> `data/control_surface_axis_interactions.json`

Builder:

> `code/control_surface_axis_interactions.py`

Commands:

```powershell
python code\control_surface_axis_interactions.py --write
python code\control_surface_axis_interactions.py
python code\validate_control_surface_atlas.py
```

## Purpose

This layer asks which observed features predict where a control-surface
claim dies. It is intentionally small-n aware: every feature has a row
count, evidence level, dominant terminal stage, and purity.

## Generated Facts

- rows: 19;
- feature summaries: 123;
- pure predictive rules: 14;
- broad or supported pure rules: 10;
- mixed predictors: 31;
- feature source counts: `{"diagnostic": 60, "frontier_class": 4, "mixture_axis": 33, "pressure_class": 9, "primary_blocker": 4, "reliability_class": 6, "route_disposition": 7}`;
- evidence-level counts: `{"broad": 29, "singleton": 73, "sparse": 9, "supported": 12}`.

## Strong Pure Rules

| Feature | Predicts | Rows | Evidence | Purity |
| --- | --- | ---: | --- | ---: |
| `route_disposition:disposition=closed_before_hidden_state` | `pre_signature_behavior_substrate` | 11 | `broad` | 1.000 |
| `route_disposition:disposition=output_shadow_diagnostic_baseline` | `signature_output_geometry_shadow` | 3 | `supported` | 1.000 |
| `primary_blocker:primary_blocker=behavior_substrate_or_bridge_blocked` | `pre_signature_behavior_substrate` | 11 | `broad` | 1.000 |
| `reliability_class:reliability_class=not_reliable_behavior_or_bridge_blocked` | `pre_signature_behavior_substrate` | 11 | `broad` | 1.000 |
| `reliability_class:reliability_class=not_reliable_output_shadow_diagnostic` | `signature_output_geometry_shadow` | 3 | `supported` | 1.000 |
| `mixture_axis:output_geometry=untested` | `pre_signature_behavior_substrate` | 4 | `supported` | 1.000 |
| `pressure_class:pressure_class=behavior_or_bridge_substrate_blocked` | `pre_signature_behavior_substrate` | 11 | `broad` | 1.000 |
| `diagnostic:diagnostic=BEHAVIOR_SUBSTRATE_FAILED` | `pre_signature_behavior_substrate` | 7 | `broad` | 1.000 |
| `diagnostic:diagnostic=NUMERIC_DIRECT_CONTROLS_PASSED` | `pre_signature_behavior_substrate` | 4 | `supported` | 1.000 |
| `diagnostic:diagnostic=PROMPT_CONTRACT_PARSEABILITY` | `pre_signature_behavior_substrate` | 3 | `supported` | 1.000 |

## Mixed Predictors

| Feature | Rows | Dominant Stage | Purity | Stage Counts |
| --- | ---: | --- | ---: | --- |
| `pressure_class:pressure_class=prompt_contract_visible` | 19 | `pre_signature_behavior_substrate` | 0.579 | `{"intervention_failed": 1, "pre_signature_behavior_substrate": 11, "pre_signature_prompt_channel_locality": 1, "reliability_null_boundary": 1, "signature_monitor_no_lever": 2, "signature_output_geometry_shadow": 3}` |
| `pressure_class:pressure_class=output_geometry_visible` | 14 | `pre_signature_behavior_substrate` | 0.500 | `{"intervention_failed": 1, "pre_signature_behavior_substrate": 7, "pre_signature_prompt_channel_locality": 1, "signature_monitor_no_lever": 2, "signature_output_geometry_shadow": 3}` |
| `mixture_axis:transfer=untested` | 14 | `pre_signature_behavior_substrate` | 0.714 | `{"pre_signature_behavior_substrate": 10, "pre_signature_prompt_channel_locality": 1, "signature_monitor_no_lever": 1, "signature_output_geometry_shadow": 2}` |
| `mixture_axis:causal_control=untested` | 14 | `pre_signature_behavior_substrate` | 0.786 | `{"pre_signature_behavior_substrate": 11, "pre_signature_prompt_channel_locality": 1, "signature_monitor_no_lever": 1, "signature_output_geometry_shadow": 1}` |
| `pressure_class:pressure_class=source_or_prompt_token_dependent` | 13 | `pre_signature_behavior_substrate` | 0.692 | `{"pre_signature_behavior_substrate": 9, "pre_signature_prompt_channel_locality": 1, "reliability_null_boundary": 1, "signature_monitor_no_lever": 1, "signature_output_geometry_shadow": 1}` |
| `mixture_axis:lead_time_internal_signal=none` | 13 | `pre_signature_behavior_substrate` | 0.846 | `{"pre_signature_behavior_substrate": 11, "pre_signature_prompt_channel_locality": 1, "signature_output_geometry_shadow": 1}` |
| `mixture_axis:output_geometry=high` | 12 | `pre_signature_behavior_substrate` | 0.583 | `{"intervention_failed": 1, "pre_signature_behavior_substrate": 7, "pre_signature_prompt_channel_locality": 1, "signature_monitor_no_lever": 1, "signature_output_geometry_shadow": 2}` |
| `mixture_axis:prompt_format=high` | 12 | `pre_signature_behavior_substrate` | 0.750 | `{"intervention_failed": 1, "pre_signature_behavior_substrate": 9, "pre_signature_prompt_channel_locality": 1, "signature_output_geometry_shadow": 1}` |
| `frontier_class:frontier_class=frontier_not_reached` | 12 | `pre_signature_behavior_substrate` | 0.917 | `{"pre_signature_behavior_substrate": 11, "pre_signature_prompt_channel_locality": 1}` |
| `mixture_axis:source_token_dependence=high` | 11 | `pre_signature_behavior_substrate` | 0.818 | `{"pre_signature_behavior_substrate": 9, "pre_signature_prompt_channel_locality": 1, "signature_output_geometry_shadow": 1}` |
| `mixture_axis:prompt_authority=high` | 10 | `signature_output_geometry_shadow` | 0.300 | `{"intervention_failed": 1, "pre_signature_behavior_substrate": 3, "reliability_null_boundary": 1, "signature_monitor_no_lever": 2, "signature_output_geometry_shadow": 3}` |
| `diagnostic:diagnostic=PROMPT_AUTHORITY_DIAL` | 10 | `pre_signature_behavior_substrate` | 0.900 | `{"pre_signature_behavior_substrate": 9, "pre_signature_prompt_channel_locality": 1}` |
| `mixture_axis:local_internal_path=untested` | 10 | `pre_signature_behavior_substrate` | 0.900 | `{"pre_signature_behavior_substrate": 9, "pre_signature_prompt_channel_locality": 1}` |
| `mixture_axis:null_locality=behavior_only` | 9 | `pre_signature_behavior_substrate` | 0.889 | `{"pre_signature_behavior_substrate": 8, "pre_signature_prompt_channel_locality": 1}` |
| `mixture_axis:prompt_authority=dominant` | 9 | `pre_signature_behavior_substrate` | 0.889 | `{"pre_signature_behavior_substrate": 8, "pre_signature_prompt_channel_locality": 1}` |
| `mixture_axis:prompt_format=medium` | 6 | `signature_output_geometry_shadow` | 0.333 | `{"pre_signature_behavior_substrate": 1, "reliability_null_boundary": 1, "signature_monitor_no_lever": 2, "signature_output_geometry_shadow": 2}` |
| `pressure_class:pressure_class=null_boundary_or_locality_limited` | 6 | `signature_output_geometry_shadow` | 0.333 | `{"intervention_failed": 1, "pre_signature_behavior_substrate": 1, "reliability_null_boundary": 1, "signature_monitor_no_lever": 1, "signature_output_geometry_shadow": 2}` |
| `pressure_class:pressure_class=signature_or_intervention_failed` | 6 | `signature_output_geometry_shadow` | 0.333 | `{"intervention_failed": 1, "reliability_null_boundary": 1, "signature_monitor_no_lever": 2, "signature_output_geometry_shadow": 2}` |
| `diagnostic:diagnostic=OUTPUT_MARGIN_CONFUND` | 6 | `signature_output_geometry_shadow` | 0.500 | `{"intervention_failed": 1, "signature_monitor_no_lever": 2, "signature_output_geometry_shadow": 3}` |
| `primary_blocker:primary_blocker=output_geometry_shadow` | 6 | `signature_output_geometry_shadow` | 0.500 | `{"intervention_failed": 1, "signature_monitor_no_lever": 2, "signature_output_geometry_shadow": 3}` |
| `mixture_axis:local_internal_path=low` | 5 | `signature_output_geometry_shadow` | 0.400 | `{"intervention_failed": 1, "signature_monitor_no_lever": 2, "signature_output_geometry_shadow": 2}` |
| `mixture_axis:null_locality=bounded` | 5 | `signature_output_geometry_shadow` | 0.400 | `{"intervention_failed": 1, "reliability_null_boundary": 1, "signature_monitor_no_lever": 1, "signature_output_geometry_shadow": 2}` |
| `mixture_axis:source_token_dependence=medium` | 5 | `signature_output_geometry_shadow` | 0.400 | `{"intervention_failed": 1, "signature_monitor_no_lever": 2, "signature_output_geometry_shadow": 2}` |
| `pressure_class:pressure_class=internal_monitor_present` | 5 | `signature_output_geometry_shadow` | 0.400 | `{"intervention_failed": 1, "signature_monitor_no_lever": 2, "signature_output_geometry_shadow": 2}` |
| `diagnostic:diagnostic=SIGNATURE_NOT_CAUSAL` | 4 | `signature_output_geometry_shadow` | 0.250 | `{"intervention_failed": 1, "reliability_null_boundary": 1, "signature_monitor_no_lever": 1, "signature_output_geometry_shadow": 1}` |
| `mixture_axis:lead_time_internal_signal=low` | 4 | `signature_output_geometry_shadow` | 0.500 | `{"intervention_failed": 1, "reliability_null_boundary": 1, "signature_output_geometry_shadow": 2}` |
| `mixture_axis:null_locality=untested` | 4 | `pre_signature_behavior_substrate` | 0.500 | `{"pre_signature_behavior_substrate": 2, "signature_monitor_no_lever": 1, "signature_output_geometry_shadow": 1}` |
| `frontier_class:frontier_class=output_visible_at_or_before_frontier` | 4 | `signature_output_geometry_shadow` | 0.750 | `{"intervention_failed": 1, "signature_output_geometry_shadow": 3}` |
| `diagnostic:diagnostic=SHUFFLED_SELECTION_OVERFIT` | 3 | `signature_output_geometry_shadow` | 0.333 | `{"intervention_failed": 1, "signature_monitor_no_lever": 1, "signature_output_geometry_shadow": 1}` |
| `pressure_class:pressure_class=transfer_unproven_or_failed` | 3 | `signature_output_geometry_shadow` | 0.333 | `{"pre_signature_behavior_substrate": 1, "signature_monitor_no_lever": 1, "signature_output_geometry_shadow": 1}` |
| `mixture_axis:local_internal_path=none` | 3 | `pre_signature_behavior_substrate` | 0.667 | `{"pre_signature_behavior_substrate": 2, "signature_output_geometry_shadow": 1}` |

## Stage Feature Profiles

### `intervention_failed`

Rows: 1

| Feature | Stage Coverage | Global Rows | Purity | Evidence |
| --- | ---: | ---: | ---: | --- |
| `route_disposition:disposition=failed_intervention_or_mechanism_route` | 1.000 | 1 | 1.000 | `singleton` |
| `primary_blocker:primary_blocker=output_geometry_shadow` | 1.000 | 6 | 0.500 | `broad` |
| `frontier_class:frontier_class=output_visible_at_or_before_frontier` | 1.000 | 4 | 0.750 | `supported` |
| `reliability_class:reliability_class=not_reliable_failed_intervention_route` | 1.000 | 1 | 1.000 | `singleton` |
| `mixture_axis:causal_control=failed` | 1.000 | 2 | 0.500 | `sparse` |
| `mixture_axis:lead_time_internal_signal=low` | 1.000 | 4 | 0.500 | `supported` |

### `pre_signature_behavior_substrate`

Rows: 11

| Feature | Stage Coverage | Global Rows | Purity | Evidence |
| --- | ---: | ---: | ---: | --- |
| `route_disposition:disposition=closed_before_hidden_state` | 1.000 | 11 | 1.000 | `broad` |
| `primary_blocker:primary_blocker=behavior_substrate_or_bridge_blocked` | 1.000 | 11 | 1.000 | `broad` |
| `frontier_class:frontier_class=frontier_not_reached` | 1.000 | 12 | 0.917 | `broad` |
| `reliability_class:reliability_class=not_reliable_behavior_or_bridge_blocked` | 1.000 | 11 | 1.000 | `broad` |
| `mixture_axis:causal_control=untested` | 1.000 | 14 | 0.786 | `broad` |
| `mixture_axis:lead_time_internal_signal=none` | 1.000 | 13 | 0.846 | `broad` |

### `pre_signature_prompt_channel_locality`

Rows: 1

| Feature | Stage Coverage | Global Rows | Purity | Evidence |
| --- | ---: | ---: | ---: | --- |
| `route_disposition:disposition=prompt_visible_positive_control` | 1.000 | 1 | 1.000 | `singleton` |
| `primary_blocker:primary_blocker=prompt_visible_positive_control` | 1.000 | 1 | 1.000 | `singleton` |
| `frontier_class:frontier_class=frontier_not_reached` | 1.000 | 12 | 0.917 | `broad` |
| `reliability_class:reliability_class=not_reliable_prompt_visible_positive_control` | 1.000 | 1 | 1.000 | `singleton` |
| `mixture_axis:causal_control=untested` | 1.000 | 14 | 0.786 | `broad` |
| `mixture_axis:lead_time_internal_signal=none` | 1.000 | 13 | 0.846 | `broad` |

### `reliability_null_boundary`

Rows: 1

| Feature | Stage Coverage | Global Rows | Purity | Evidence |
| --- | ---: | ---: | ---: | --- |
| `route_disposition:disposition=bounded_mechanism_frozen` | 1.000 | 1 | 1.000 | `singleton` |
| `primary_blocker:primary_blocker=bounded_internal_causal_with_null_boundary` | 1.000 | 1 | 1.000 | `singleton` |
| `frontier_class:frontier_class=causal_surface_not_timing_frontier` | 1.000 | 1 | 1.000 | `singleton` |
| `reliability_class:reliability_class=bounded_reliability_reference` | 1.000 | 1 | 1.000 | `singleton` |
| `mixture_axis:causal_control=bounded` | 1.000 | 1 | 1.000 | `singleton` |
| `mixture_axis:lead_time_internal_signal=low` | 1.000 | 4 | 0.500 | `supported` |

### `signature_monitor_no_lever`

Rows: 2

| Feature | Stage Coverage | Global Rows | Purity | Evidence |
| --- | ---: | ---: | ---: | --- |
| `primary_blocker:primary_blocker=output_geometry_shadow` | 1.000 | 6 | 0.500 | `broad` |
| `frontier_class:frontier_class=predecision_monitor_no_lever` | 1.000 | 2 | 1.000 | `sparse` |
| `reliability_class:reliability_class=not_reliable_monitor_only_no_lever` | 1.000 | 2 | 1.000 | `sparse` |
| `mixture_axis:local_internal_path=low` | 1.000 | 5 | 0.400 | `broad` |
| `mixture_axis:prompt_authority=high` | 1.000 | 10 | 0.300 | `broad` |
| `mixture_axis:prompt_format=medium` | 1.000 | 6 | 0.333 | `broad` |

### `signature_output_geometry_shadow`

Rows: 3

| Feature | Stage Coverage | Global Rows | Purity | Evidence |
| --- | ---: | ---: | ---: | --- |
| `route_disposition:disposition=output_shadow_diagnostic_baseline` | 1.000 | 3 | 1.000 | `supported` |
| `primary_blocker:primary_blocker=output_geometry_shadow` | 1.000 | 6 | 0.500 | `broad` |
| `frontier_class:frontier_class=output_visible_at_or_before_frontier` | 1.000 | 4 | 0.750 | `supported` |
| `reliability_class:reliability_class=not_reliable_output_shadow_diagnostic` | 1.000 | 3 | 1.000 | `supported` |
| `mixture_axis:prompt_authority=high` | 1.000 | 10 | 0.300 | `broad` |
| `pressure_class:pressure_class=prompt_contract_visible` | 1.000 | 19 | 0.579 | `broad` |

## Interpretation

The largest current predictive rule is not a hidden mechanism rule.
It is route-level closure: `closed_before_hidden_state` predicts
`pre_signature_behavior_substrate` across 11 rows. That is the
project's strongest current law about where claims die.

Output geometry is different. It is broad, but mixed: it appears
across output-shadow signatures, monitor-only rows, and the failed
intervention route. So output geometry is a family-level pressure,
not a single terminal-stage rule.

MC005 remains a singleton internal-causal reliability boundary.
The layer marks singleton purity as singleton evidence, not as a
general law.

## What This Proves

It proves that the current genome map has predictive regularities, and
that their evidence strength can be made explicit. The strongest
broad rule is pre-signature route closure, not a truth vector or
knowledge vector.

## What It Does Not Prove

It does not prove universal laws of small LLMs. It proves current-contract
regularities over the validated 19-row atlas and bridge-derived stack.
