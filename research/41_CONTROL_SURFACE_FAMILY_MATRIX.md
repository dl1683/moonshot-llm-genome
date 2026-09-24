# Control-Surface Family Matrix

Source updated_at: 2026-07-01

Status: generated cross-family matrix implemented and validated.

Machine-readable artifact:

> `data/control_surface_family_matrix.json`

Builder:

> `code/control_surface_family_matrix.py`

Commands:

```powershell
python code\control_surface_family_matrix.py --write
python code\control_surface_family_matrix.py
python code\validate_control_surface_atlas.py
```

## Purpose

This is the compact cross-family table for the current atlas. It joins
each behavior family to the mixture-law, decision-frontier, route,
reliability, transfer, and gate-geometry cells that define its current
claim boundary.

## Summary

- rows: 19;
- columns: 25;
- promotion-ready rows: 0;
- bounded reference rows: ["mc005_associative_lookup"];
- monitor-only rows: ["mc004_in_context_binding", "mc006_parametric_fact_override"].

Boolean axis counts:

- `prompt_contract_visible`: 19/19 (1.000);
- `output_geometry_visible`: 14/19 (0.737);
- `source_or_prompt_token_dependent`: 13/19 (0.684);
- `internal_monitor_present`: 5/19 (0.263);
- `internal_causal_surface`: 1/19 (0.053);
- `null_boundary_or_locality_limited`: 6/19 (0.316);
- `transfer_unproven_or_failed`: 3/19 (0.158);

## Matrix

| Row | Domain | Verdict | Blocker | Stage | Frontier | Reliability | Transfer | Axes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mc001_qwen3_0p6b_truth_agreement` | `truth_agreement` | `diagnostic_note` | `output_geometry_shadow` | `signature_output_geometry_shadow` | `output_visible_at_or_before_frontier` | `not_reliable_output_shadow_diagnostic` | `output_shadow_widening_baseline` | `prompt_contract_visible`<br>`output_geometry_visible`<br>`source_or_prompt_token_dependent`<br>`null_boundary_or_locality_limited` |
| `mc001b_qwen3_1p7b_truth_agreement` | `truth_agreement` | `diagnostic_note` | `output_geometry_shadow` | `signature_output_geometry_shadow` | `output_visible_at_or_before_frontier` | `not_reliable_output_shadow_diagnostic` | `diagnostic_cross_model_evidence_only` | `prompt_contract_visible`<br>`output_geometry_visible`<br>`internal_monitor_present`<br>`null_boundary_or_locality_limited`<br>`transfer_unproven_or_failed` |
| `mc001g_gemma_truth_agreement` | `truth_agreement` | `failed_mechanism_card` | `output_geometry_shadow` | `intervention_failed` | `output_visible_at_or_before_frontier` | `not_reliable_failed_intervention_route` | `diagnostic_cross_model_evidence_only` | `prompt_contract_visible`<br>`output_geometry_visible`<br>`internal_monitor_present`<br>`null_boundary_or_locality_limited` |
| `mc002_known_unknown` | `known_unknown_or_context_support` | `diagnostic_note` | `behavior_substrate_or_bridge_blocked` | `pre_signature_behavior_substrate` | `frontier_not_reached` | `not_reliable_behavior_or_bridge_blocked` | `diagnostic_cross_model_evidence_only` | `prompt_contract_visible`<br>`transfer_unproven_or_failed` |
| `mc002b_context_support` | `known_unknown_or_context_support` | `diagnostic_note` | `behavior_substrate_or_bridge_blocked` | `pre_signature_behavior_substrate` | `frontier_not_reached` | `not_reliable_behavior_or_bridge_blocked` | `closed_route_no_widening` | `prompt_contract_visible` |
| `mc003_delayed_copy` | `delayed_copy` | `diagnostic_note` | `output_geometry_shadow` | `signature_output_geometry_shadow` | `output_visible_at_or_before_frontier` | `not_reliable_output_shadow_diagnostic` | `output_shadow_widening_baseline` | `prompt_contract_visible`<br>`output_geometry_visible`<br>`internal_monitor_present` |
| `mc004_in_context_binding` | `in_context_binding` | `diagnostic_note` | `output_geometry_shadow` | `signature_monitor_no_lever` | `predecision_monitor_no_lever` | `not_reliable_monitor_only_no_lever` | `conditional_widening_requires_new_controls` | `prompt_contract_visible`<br>`output_geometry_visible`<br>`internal_monitor_present` |
| `mc005_associative_lookup` | `synthetic_lookup` | `bounded_mechanism_card` | `bounded_internal_causal_with_null_boundary` | `reliability_null_boundary` | `causal_surface_not_timing_frontier` | `bounded_reliability_reference` | `bounded_transfer_fragile_reference` | `prompt_contract_visible`<br>`source_or_prompt_token_dependent`<br>`internal_causal_surface`<br>`null_boundary_or_locality_limited` |
| `mc006_parametric_fact_override` | `parametric_fact_override` | `diagnostic_note` | `output_geometry_shadow` | `signature_monitor_no_lever` | `predecision_monitor_no_lever` | `not_reliable_monitor_only_no_lever` | `transfer_failed_or_bank_insufficient` | `prompt_contract_visible`<br>`output_geometry_visible`<br>`source_or_prompt_token_dependent`<br>`internal_monitor_present`<br>`null_boundary_or_locality_limited`<br>`transfer_unproven_or_failed` |
| `mc007_semi_synthetic_familiar_entity_lookup` | `semi_synthetic_familiar_entity` | `diagnostic_note` | `behavior_substrate_or_bridge_blocked` | `pre_signature_behavior_substrate` | `frontier_not_reached` | `not_reliable_behavior_or_bridge_blocked` | `closed_route_no_widening` | `prompt_contract_visible`<br>`output_geometry_visible`<br>`source_or_prompt_token_dependent` |
| `mc008_symbolic_fact_code_arbitration` | `symbolic_or_derived_code_arbitration` | `diagnostic_note` | `behavior_substrate_or_bridge_blocked` | `pre_signature_behavior_substrate` | `frontier_not_reached` | `not_reliable_behavior_or_bridge_blocked` | `closed_route_no_widening` | `prompt_contract_visible`<br>`source_or_prompt_token_dependent`<br>`null_boundary_or_locality_limited` |
| `mc009_derived_code_arbitration` | `symbolic_or_derived_code_arbitration` | `diagnostic_note` | `behavior_substrate_or_bridge_blocked` | `pre_signature_behavior_substrate` | `frontier_not_reached` | `not_reliable_behavior_or_bridge_blocked` | `closed_route_no_widening` | `prompt_contract_visible`<br>`source_or_prompt_token_dependent` |
| `mc010_two_hop_fact_code_arbitration` | `two_hop_arbitration` | `diagnostic_note` | `behavior_substrate_or_bridge_blocked` | `pre_signature_behavior_substrate` | `frontier_not_reached` | `not_reliable_behavior_or_bridge_blocked` | `closed_route_no_widening` | `prompt_contract_visible`<br>`output_geometry_visible`<br>`source_or_prompt_token_dependent` |
| `mc011_atomic_number_code_arbitration` | `numeric_or_status_arbitration` | `diagnostic_note` | `behavior_substrate_or_bridge_blocked` | `pre_signature_behavior_substrate` | `frontier_not_reached` | `not_reliable_behavior_or_bridge_blocked` | `closed_route_no_widening` | `prompt_contract_visible`<br>`output_geometry_visible`<br>`source_or_prompt_token_dependent` |
| `mc012_reliability_labeled_numeric_arbitration` | `numeric_or_status_arbitration` | `diagnostic_note` | `prompt_visible_positive_control` | `pre_signature_prompt_channel_locality` | `frontier_not_reached` | `not_reliable_prompt_visible_positive_control` | `prompt_visible_no_transfer_claim` | `prompt_contract_visible`<br>`output_geometry_visible`<br>`source_or_prompt_token_dependent` |
| `mc013_status_channel_ablation_numeric_arbitration` | `numeric_or_status_arbitration` | `diagnostic_note` | `behavior_substrate_or_bridge_blocked` | `pre_signature_behavior_substrate` | `frontier_not_reached` | `not_reliable_behavior_or_bridge_blocked` | `closed_route_no_widening` | `prompt_contract_visible`<br>`output_geometry_visible`<br>`source_or_prompt_token_dependent` |
| `mc014_inferred_reliability_numeric_arbitration` | `numeric_or_status_arbitration` | `diagnostic_note` | `behavior_substrate_or_bridge_blocked` | `pre_signature_behavior_substrate` | `frontier_not_reached` | `not_reliable_behavior_or_bridge_blocked` | `closed_route_no_widening` | `prompt_contract_visible`<br>`output_geometry_visible`<br>`source_or_prompt_token_dependent` |
| `mc015_parity_gated_numeric_arbitration` | `numeric_or_status_arbitration` | `diagnostic_note` | `behavior_substrate_or_bridge_blocked` | `pre_signature_behavior_substrate` | `frontier_not_reached` | `not_reliable_behavior_or_bridge_blocked` | `closed_route_no_widening` | `prompt_contract_visible`<br>`output_geometry_visible`<br>`source_or_prompt_token_dependent` |
| `mc016_alphabet_gated_numeric_arbitration` | `numeric_or_status_arbitration` | `diagnostic_note` | `behavior_substrate_or_bridge_blocked` | `pre_signature_behavior_substrate` | `frontier_not_reached` | `not_reliable_behavior_or_bridge_blocked` | `closed_route_no_widening` | `prompt_contract_visible`<br>`output_geometry_visible`<br>`source_or_prompt_token_dependent` |

## Validation Checks

| Check | Passed | Actual |
| --- | --- | --- |
| `matrix_has_one_row_per_atlas_family` | `true` | `{"row_count": 19, "unique_row_count": 19}` |
| `matrix_columns_are_declared` | `true` | `{"declared_column_count": 25, "matrix_column_count": 25}` |
| `primary_blockers_match_mixture_law` | `true` | `{"matrix": {"behavior_substrate_or_bridge_blocked": 11, "bounded_internal_causal_with_null_boundary": 1, "output_geometry_shadow": 6, "prompt_visible_positive_control": 1}, "mixture_law": {"behavior_substrate_or_bridge_blocked": 11, "bounded_internal_causal_with_null_boundary": 1, "output_geometry_shadow": 6, "prompt_visible_positive_control": 1}}` |
| `frontier_counts_match_decision_frontier` | `true` | `{"decision_frontier": {"causal_surface_not_timing_frontier": 1, "frontier_not_reached": 12, "output_visible_at_or_before_frontier": 4, "predecision_monitor_no_lever": 2}, "matrix": {"causal_surface_not_timing_frontier": 1, "frontier_not_reached": 12, "output_visible_at_or_before_frontier": 4, "predecision_monitor_no_lever": 2}}` |
| `reliability_counts_match_reliability_matrix` | `true` | `{"matrix": {"bounded_reliability_reference": 1, "not_reliable_behavior_or_bridge_blocked": 11, "not_reliable_failed_intervention_route": 1, "not_reliable_monitor_only_no_lever": 2, "not_reliable_output_shadow_diagnostic": 3, "not_reliable_prompt_visible_positive_control": 1}, "reliability_matrix": {"bounded_reliability_reference": 1, "not_reliable_behavior_or_bridge_blocked": 11, "not_reliable_failed_intervention_route": 1, "not_reliable_monitor_only_no_lever": 2, "not_reliable_output_shadow_diagnostic": 3, "not_reliable_prompt_visible_positive_control": 1}}` |
| `transfer_counts_match_transfer_matrix` | `true` | `{"matrix": {"bounded_transfer_fragile_reference": 1, "closed_route_no_widening": 10, "conditional_widening_requires_new_controls": 1, "diagnostic_cross_model_evidence_only": 3, "output_shadow_widening_baseline": 2, "prompt_visible_no_transfer_claim": 1, "transfer_failed_or_bank_insufficient": 1}, "transfer_matrix": {"bounded_transfer_fragile_reference": 1, "closed_route_no_widening": 10, "conditional_widening_requires_new_controls": 1, "diagnostic_cross_model_evidence_only": 3, "output_shadow_widening_baseline": 2, "prompt_visible_no_transfer_claim": 1, "transfer_failed_or_bank_insufficient": 1}}` |
| `terminal_stage_counts_match_gate_geometry` | `true` | `{"gate_geometry": {"intervention_failed": 1, "pre_signature_behavior_substrate": 11, "pre_signature_prompt_channel_locality": 1, "reliability_null_boundary": 1, "signature_monitor_no_lever": 2, "signature_output_geometry_shadow": 3}, "matrix": {"intervention_failed": 1, "pre_signature_behavior_substrate": 11, "pre_signature_prompt_channel_locality": 1, "reliability_null_boundary": 1, "signature_monitor_no_lever": 2, "signature_output_geometry_shadow": 3}}` |
| `boolean_axis_counts_match_mixture_pressure_counts` | `true` | `{"matrix": {"internal_causal_surface": 1, "internal_monitor_present": 5, "null_boundary_or_locality_limited": 6, "output_geometry_visible": 14, "prompt_contract_visible": 19, "source_or_prompt_token_dependent": 13, "transfer_unproven_or_failed": 3}, "mixture_law": {"internal_causal_surface": 1, "internal_monitor_present": 5, "null_boundary_or_locality_limited": 6, "output_geometry_visible": 14, "prompt_contract_visible": 19, "source_or_prompt_token_dependent": 13, "transfer_unproven_or_failed": 3}}` |
| `mc005_is_sole_bounded_internal_reference` | `true` | `["mc005_associative_lookup"]` |
| `mc006_and_mc004_are_monitor_only_rows` | `true` | `["mc004_in_context_binding", "mc006_parametric_fact_override"]` |
| `no_promotion_ready_rows` | `true` | `0` |
| `each_row_has_claim_boundaries_and_evidence` | `true` | `[]` |

## Claim Boundary

This matrix is the current cross-family atlas table: it shows how each behavior family distributes across visible prompt/output/source surfaces, internal monitors, bounded causal evidence, reliability gates, transfer status, and terminal claim boundary.

This matrix is not a promotion artifact and does not claim a general truth vector, general knowledge vector, full-reliability mechanism, or transfer-ready control surface.
