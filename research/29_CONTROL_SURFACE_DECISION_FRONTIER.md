# Control-Surface Decision Frontier

Date: 2026-07-01

Status: generated decision-frontier layer implemented and validated.

Machine-readable artifact:

> `data/control_surface_decision_frontier.json`

Builder:

> `code/control_surface_decision_frontier.py`

Commands:

```powershell
python code\control_surface_decision_frontier.py --write
python code\control_surface_decision_frontier.py
python code\validate_control_surface_atlas.py
```

## Purpose

The decision frontier makes lead-time a measured axis. It asks whether a
behavior family has no valid timing substrate, is already output-visible,
has a predecision monitor without a lever, or has a causal result that is
not actually a timing-frontier claim.

## Generated Facts

- atlas rows: 19;
- predecision causal candidates: 0;
- monitor-only rows: 2;
- output-visible frontier rows: 4;
- causal-not-timing rows: 1.

## Frontier Classes

| Frontier Class | Count | Ratio |
| --- | ---: | ---: |
| `causal_surface_not_timing_frontier` | 1 | 0.053 |
| `frontier_not_reached` | 12 | 0.632 |
| `output_visible_at_or_before_frontier` | 4 | 0.211 |
| `predecision_monitor_no_lever` | 2 | 0.105 |

## Row Map

| Row | Frontier | Lead-Time State | Primary Blocker | Next Action |
| --- | --- | --- | --- | --- |
| `mc001_qwen3_0p6b_truth_agreement` | `output_visible_at_or_before_frontier` | `zero_lead_time` | `output_geometry_shadow` | Change the answer interface or row geometry before treating hidden probes as upstream mechanisms. |
| `mc001b_qwen3_1p7b_truth_agreement` | `output_visible_at_or_before_frontier` | `lead_time_output_shadow` | `output_geometry_shadow` | Change the answer interface or row geometry before treating hidden probes as upstream mechanisms. |
| `mc001g_gemma_truth_agreement` | `output_visible_at_or_before_frontier` | `lead_time_output_shadow` | `output_geometry_shadow` | Change the answer interface or row geometry before treating hidden probes as upstream mechanisms. |
| `mc002_known_unknown` | `frontier_not_reached` | `not_reached` | `behavior_substrate_or_bridge_blocked` | Repair behavior substrate or close the route; do not probe hidden states. |
| `mc002b_context_support` | `frontier_not_reached` | `not_reached` | `behavior_substrate_or_bridge_blocked` | Repair behavior substrate or close the route; do not probe hidden states. |
| `mc003_delayed_copy` | `output_visible_at_or_before_frontier` | `lead_time_output_shadow` | `output_geometry_shadow` | Change the answer interface or row geometry before treating hidden probes as upstream mechanisms. |
| `mc004_in_context_binding` | `predecision_monitor_no_lever` | `lead_time_monitor_only` | `output_geometry_shadow` | Either run a materially different causal stress test or record the row as monitor-only. |
| `mc005_associative_lookup` | `causal_surface_not_timing_frontier` | `not_primary_axis` | `bounded_internal_causal_with_null_boundary` | Keep timing claims separate from the bounded causal surface; finish null/locality boundaries. |
| `mc006_parametric_fact_override` | `predecision_monitor_no_lever` | `lead_time_monitor_only` | `output_geometry_shadow` | Either run a materially different causal stress test or record the row as monitor-only. |
| `mc007_semi_synthetic_familiar_entity_lookup` | `frontier_not_reached` | `not_reached` | `behavior_substrate_or_bridge_blocked` | Repair behavior substrate or close the route; do not probe hidden states. |
| `mc008_symbolic_fact_code_arbitration` | `frontier_not_reached` | `not_reached` | `behavior_substrate_or_bridge_blocked` | Repair behavior substrate or close the route; do not probe hidden states. |
| `mc009_derived_code_arbitration` | `frontier_not_reached` | `not_reached` | `behavior_substrate_or_bridge_blocked` | Repair behavior substrate or close the route; do not probe hidden states. |
| `mc010_two_hop_fact_code_arbitration` | `frontier_not_reached` | `not_reached` | `behavior_substrate_or_bridge_blocked` | Repair behavior substrate or close the route; do not probe hidden states. |
| `mc011_atomic_number_code_arbitration` | `frontier_not_reached` | `not_reached` | `behavior_substrate_or_bridge_blocked` | Repair behavior substrate or close the route; do not probe hidden states. |
| `mc012_reliability_labeled_numeric_arbitration` | `frontier_not_reached` | `not_reached` | `prompt_visible_positive_control` | Repair behavior substrate or close the route; do not probe hidden states. |
| `mc013_status_channel_ablation_numeric_arbitration` | `frontier_not_reached` | `not_reached` | `behavior_substrate_or_bridge_blocked` | Repair behavior substrate or close the route; do not probe hidden states. |
| `mc014_inferred_reliability_numeric_arbitration` | `frontier_not_reached` | `not_reached` | `behavior_substrate_or_bridge_blocked` | Repair behavior substrate or close the route; do not probe hidden states. |
| `mc015_parity_gated_numeric_arbitration` | `frontier_not_reached` | `not_reached` | `behavior_substrate_or_bridge_blocked` | Repair behavior substrate or close the route; do not probe hidden states. |
| `mc016_alphabet_gated_numeric_arbitration` | `frontier_not_reached` | `not_reached` | `behavior_substrate_or_bridge_blocked` | Repair behavior substrate or close the route; do not probe hidden states. |

## Validation Checks

| Check | Passed | Actual |
| --- | --- | --- |
| `frontier_rows_cover_each_atlas_row_once` | `true` | `["mc001_qwen3_0p6b_truth_agreement", "mc001b_qwen3_1p7b_truth_agreement", "mc001g_gemma_truth_agreement", "mc002_known_unknown", "mc002b_context_support", "mc003_delayed_copy", "mc004_in_context_binding", "mc005_associative_lookup", "mc006_parametric_fact_override", "mc007_semi_synthetic_familiar_entity_lookup", "mc008_symbolic_fact_code_arbitration", "mc009_derived_code_arbitration", "mc010_two_hop_fact_code_arbitration", "mc011_atomic_number_code_arbitration", "mc012_reliability_labeled_numeric_arbitration", "mc013_status_channel_ablation_numeric_arbitration", "mc014_inferred_reliability_numeric_arbitration", "mc015_parity_gated_numeric_arbitration", "mc016_alphabet_gated_numeric_arbitration"]` |
| `frontier_counts_match_comparison_lead_time_counts` | `true` | `{"frontier_not_reached": 12, "lead_time_monitor_only": 2, "monitor_only": 2, "not_reached": 12, "output_visible": 4, "zero_plus_shadow": 4}` |
| `no_predecision_causal_candidate_yet` | `true` | `0` |
| `monitor_only_rows_are_mc004_and_mc006` | `true` | `["mc004_in_context_binding", "mc006_parametric_fact_override"]` |
| `output_visible_frontier_exceeds_monitor_only_frontier` | `true` | `{"output_visible_at_or_before_frontier": 4, "predecision_monitor_no_lever": 2}` |
| `mc005_is_not_counted_as_lead_time_promotion` | `true` | `["causal_surface_not_timing_frontier"]` |

## Interpretation

The current frontier map says there is no lead-time causal lever in the
atlas. MC004 and MC006 have predecision monitors, but they are not
intervention-ready. MC001/MC001B/MC001G/MC003-style timing evidence is
output-visible or zero-lead under current controls. MC005 remains the
bounded causal result, but its claim is source-value attention/write
mediation, not commitment timing.

## Claim Boundary

In the current atlas, lead-time evidence is mostly absent, output-visible, or monitor-only. MC004 and MC006 are predecision monitor rows; no row is a promoted or bounded predecision causal lever.

This artifact does not show that any current lead-time signal is a causal control vector, nor that MC005's bounded causal lookup surface is a general commitment-timing mechanism.
