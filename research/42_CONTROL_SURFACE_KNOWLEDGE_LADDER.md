# Control-Surface Knowledge Ladder

Source updated_at: 2026-07-01

Status: generated knowledge-ladder coverage map implemented and validated.

Machine-readable artifact:

> `data/control_surface_knowledge_ladder.json`

Builder:

> `code/control_surface_knowledge_ladder.py`

Commands:

```powershell
python code\control_surface_knowledge_ladder.py --write
python code\control_surface_knowledge_ladder.py
python code\validate_control_surface_atlas.py
```

## Purpose

This map separates the knowledge-specific ladder from auxiliary
diagnostic rows. It shows how far the current atlas gets from pure
synthetic lookup toward real factual correction, refusal, and
uncertainty behavior.

## Summary

- levels: 5;
- ladder rows: 14;
- auxiliary diagnostic rows: 5;
- bounded-reference levels: 1;
- monitor-only levels: 1;
- promoted levels: 0;
- bridge hidden-state-allowed rungs: 0;
- bridge clean unconfounded rungs: 0;
- real abstention/uncertainty mechanism-ready levels: 0.

## Ladder

| Level | Status | Rank | Rows | Main Boundary |
| --- | --- | ---: | --- | --- |
| `level_1_synthetic_lookup` | `bounded_reference` | 4 | `mc005_associative_lookup` | blocker: {"bounded_internal_causal_with_null_boundary": 1}<br>stage: {"reliability_null_boundary": 1}<br>frontier: {"causal_surface_not_timing_frontier": 1}<br>reliability: {"bounded_reliability_reference": 1} |
| `level_2_semi_synthetic_familiar_entity` | `behavior_substrate_blocked` | 1 | `mc007_semi_synthetic_familiar_entity_lookup` | blocker: {"behavior_substrate_or_bridge_blocked": 1}<br>stage: {"pre_signature_behavior_substrate": 1}<br>frontier: {"frontier_not_reached": 1}<br>reliability: {"not_reliable_behavior_or_bridge_blocked": 1} |
| `level_3_symbolic_or_learned_memory_bridge` | `prompt_visible_or_behavior_blocked` | 2 | `mc008_symbolic_fact_code_arbitration`<br>`mc009_derived_code_arbitration`<br>`mc010_two_hop_fact_code_arbitration`<br>`mc011_atomic_number_code_arbitration`<br>`mc012_reliability_labeled_numeric_arbitration`<br>`mc013_status_channel_ablation_numeric_arbitration`<br>`mc014_inferred_reliability_numeric_arbitration`<br>`mc015_parity_gated_numeric_arbitration`<br>`mc016_alphabet_gated_numeric_arbitration` | blocker: {"behavior_substrate_or_bridge_blocked": 8, "prompt_visible_positive_control": 1}<br>stage: {"pre_signature_behavior_substrate": 8, "pre_signature_prompt_channel_locality": 1}<br>frontier: {"frontier_not_reached": 9}<br>reliability: {"not_reliable_behavior_or_bridge_blocked": 8, "not_reliable_prompt_visible_positive_control": 1} |
| `level_4_parametric_fact_override` | `monitor_only_no_lever` | 3 | `mc006_parametric_fact_override` | blocker: {"output_geometry_shadow": 1}<br>stage: {"signature_monitor_no_lever": 1}<br>frontier: {"predecision_monitor_no_lever": 1}<br>reliability: {"not_reliable_monitor_only_no_lever": 1} |
| `level_5_real_abstention_uncertainty` | `behavior_substrate_blocked` | 1 | `mc002_known_unknown`<br>`mc002b_context_support` | blocker: {"behavior_substrate_or_bridge_blocked": 2}<br>stage: {"pre_signature_behavior_substrate": 2}<br>frontier: {"frontier_not_reached": 2}<br>reliability: {"not_reliable_behavior_or_bridge_blocked": 2} |

## Auxiliary Diagnostics

| Row | Role | Domain | Frontier | Stage |
| --- | --- | --- | --- | --- |
| `mc001_qwen3_0p6b_truth_agreement` | `diagnostic_support_not_ladder_level` | `truth_agreement` | `output_visible_at_or_before_frontier` | `signature_output_geometry_shadow` |
| `mc001b_qwen3_1p7b_truth_agreement` | `diagnostic_support_not_ladder_level` | `truth_agreement` | `output_visible_at_or_before_frontier` | `signature_output_geometry_shadow` |
| `mc001g_gemma_truth_agreement` | `diagnostic_support_not_ladder_level` | `truth_agreement` | `output_visible_at_or_before_frontier` | `intervention_failed` |
| `mc003_delayed_copy` | `diagnostic_support_not_ladder_level` | `delayed_copy` | `output_visible_at_or_before_frontier` | `signature_output_geometry_shadow` |
| `mc004_in_context_binding` | `diagnostic_support_not_ladder_level` | `in_context_binding` | `predecision_monitor_no_lever` | `signature_monitor_no_lever` |

## Validation Checks

| Check | Passed | Actual |
| --- | --- | --- |
| `five_ladder_levels_declared` | `true` | `5` |
| `ladder_and_auxiliary_rows_partition_family_matrix` | `true` | `{"family_matrix_rows": ["mc001_qwen3_0p6b_truth_agreement", "mc001b_qwen3_1p7b_truth_agreement", "mc001g_gemma_truth_agreement", "mc002_known_unknown", "mc002b_context_support", "mc003_delayed_copy", "mc004_in_context_binding", "mc005_associative_lookup", "mc006_parametric_fact_override", "mc007_semi_synthetic_familiar_entity_lookup", "mc008_symbolic_fact_code_arbitration", "mc009_derived_code_arbitration", "mc010_two_hop_fact_code_arbitration", "mc011_atomic_number_code_arbitration", "mc012_reliability_labeled_numeric_arbitration", "mc013_status_channel_ablation_numeric_arbitration", "mc014_inferred_reliability_numeric_arbitration", "mc015_parity_gated_numeric_arbitration", "mc016_alphabet_gated_numeric_arbitration"], "ladder_plus_auxiliary": ["mc001_qwen3_0p6b_truth_agreement", "mc001b_qwen3_1p7b_truth_agreement", "mc001g_gemma_truth_agreement", "mc002_known_unknown", "mc002b_context_support", "mc003_delayed_copy", "mc004_in_context_binding", "mc005_associative_lookup", "mc006_parametric_fact_override", "mc007_semi_synthetic_familiar_entity_lookup", "mc008_symbolic_fact_code_arbitration", "mc009_derived_code_arbitration", "mc010_two_hop_fact_code_arbitration", "mc011_atomic_number_code_arbitration", "mc012_reliability_labeled_numeric_arbitration", "mc013_status_channel_ablation_numeric_arbitration", "mc014_inferred_reliability_numeric_arbitration", "mc015_parity_gated_numeric_arbitration", "mc016_alphabet_gated_numeric_arbitration"]}` |
| `synthetic_lookup_level_is_mc005_bounded_reference` | `true` | `{"mc005_route_status": "bounded_frozen_not_promoted", "row_ids": ["mc005_associative_lookup"], "status": "bounded_reference"}` |
| `semi_synthetic_level_is_blocked_mc007` | `true` | `{"row_ids": ["mc007_semi_synthetic_familiar_entity_lookup"], "status": "behavior_substrate_blocked"}` |
| `bridge_level_preserves_post_mc033_closeout` | `true` | `{"bridge_rung_count": 24, "clean_unconfounded": 0, "hidden_state_allowed": 0, "level_row_count": 9, "post_mc033_status": "killed_after_mc033"}` |
| `parametric_fact_level_is_mc006_monitor_only` | `true` | `{"mc006_route_status": "monitor_only_closed", "row_ids": ["mc006_parametric_fact_override"], "status": "monitor_only_no_lever"}` |
| `real_uncertainty_level_not_mechanism_ready` | `true` | `{"ready_count": 0, "row_ids": ["mc002_known_unknown", "mc002b_context_support"], "stage_rank": 1}` |
| `no_ladder_level_is_promoted` | `true` | `0` |

## Claim Boundary

The project currently has a partial knowledge ladder: synthetic lookup reaches a bounded causal reference, capital-fact override reaches monitor-only timing evidence, bridge rows mostly fail before hidden-state work, and real abstention/uncertainty is not mechanism-ready.

This ladder does not establish a broad knowledge mechanism, a truth or knowledge vector, a promoted mechanism card, a reliable MC006 steering surface, or a real-world factual correction/refusal mechanism.
