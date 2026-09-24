# Control-Surface Reliability Matrix

Date: 2026-07-01

Status: generated reliability matrix implemented and validated.

Machine-readable artifact:

> `data/control_surface_reliability_matrix.json`

Builder:

> `code/control_surface_reliability_matrix.py`

Commands:

```powershell
python code\control_surface_reliability_matrix.py --write
python code\control_surface_reliability_matrix.py
python code\validate_control_surface_atlas.py
```

## Purpose

This matrix makes reliability explicit. Each row is scored across the
mechanism-card gates: behavior, signature, intervention, null/locality,
local internal path, robustness/side effects, and transfer. The point is
to make failed claims die cleanly and make bounded claims stay bounded.

## Generated Facts

- atlas rows: 19;
- full reliability mechanisms: 0;
- bounded reliability references: 1;
- monitor-only rows: 2;
- behavior/bridge blocked rows: 11;
- output-shadow diagnostic rows: 3;
- transfer-ready mechanisms: 0.

## Reliability Classes

| Reliability Class | Count | Ratio |
| --- | ---: | ---: |
| `bounded_reliability_reference` | 1 | 0.053 |
| `not_reliable_behavior_or_bridge_blocked` | 11 | 0.579 |
| `not_reliable_failed_intervention_route` | 1 | 0.053 |
| `not_reliable_monitor_only_no_lever` | 2 | 0.105 |
| `not_reliable_output_shadow_diagnostic` | 3 | 0.158 |
| `not_reliable_prompt_visible_positive_control` | 1 | 0.053 |

## Missing Gates

| Missing or Failed Gate | Count |
| --- | ---: |
| `behavior_substrate` | 11 |
| `clean_predicted_intervention` | 19 |
| `control-surviving_signature` | 18 |
| `intervention_test` | 14 |
| `local_internal_path` | 18 |
| `null_and_locality_cleanliness` | 19 |
| `output_or_candidate_margin_separation` | 6 |
| `prompt_channel_locality` | 1 |
| `robustness_and_side_effects` | 19 |
| `transfer_or_widening` | 19 |

## Row Matrix

| Row | Reliability Class | Behavior | Signature | Intervention | Null/Locality | Transfer | Missing Gates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mc001_qwen3_0p6b_truth_agreement` | `not_reliable_output_shadow_diagnostic` | `passed_or_partial` | `output_visible_or_zero_lead` | `behavior_control_only_not_mechanism` | `bounded` | `output_shadow_widening_baseline` | `clean_predicted_intervention, control-surviving_signature, local_internal_path, null_and_locality_cleanliness, output_or_candidate_margin_separation, robustness_and_side_effects, transfer_or_widening` |
| `mc001b_qwen3_1p7b_truth_agreement` | `not_reliable_output_shadow_diagnostic` | `passed_or_partial` | `output_visible_or_zero_lead` | `behavior_control_only_not_mechanism` | `bounded` | `diagnostic_cross_model_evidence_only` | `clean_predicted_intervention, control-surviving_signature, local_internal_path, null_and_locality_cleanliness, output_or_candidate_margin_separation, robustness_and_side_effects, transfer_or_widening` |
| `mc001g_gemma_truth_agreement` | `not_reliable_failed_intervention_route` | `passed_or_partial` | `output_visible_or_zero_lead` | `failed` | `bounded` | `diagnostic_cross_model_evidence_only` | `clean_predicted_intervention, control-surviving_signature, local_internal_path, null_and_locality_cleanliness, output_or_candidate_margin_separation, robustness_and_side_effects, transfer_or_widening` |
| `mc002_known_unknown` | `not_reliable_behavior_or_bridge_blocked` | `failed_or_closed_before_hidden_state` | `not_allowed_or_not_reached` | `not_allowed` | `untested` | `diagnostic_cross_model_evidence_only` | `behavior_substrate, clean_predicted_intervention, control-surviving_signature, intervention_test, local_internal_path, null_and_locality_cleanliness, robustness_and_side_effects, transfer_or_widening` |
| `mc002b_context_support` | `not_reliable_behavior_or_bridge_blocked` | `failed_or_closed_before_hidden_state` | `not_allowed_or_not_reached` | `not_allowed` | `untested` | `closed_route_no_widening` | `behavior_substrate, clean_predicted_intervention, control-surviving_signature, intervention_test, local_internal_path, null_and_locality_cleanliness, robustness_and_side_effects, transfer_or_widening` |
| `mc003_delayed_copy` | `not_reliable_output_shadow_diagnostic` | `passed_or_partial` | `output_visible_or_zero_lead` | `not_tested` | `untested` | `output_shadow_widening_baseline` | `clean_predicted_intervention, control-surviving_signature, intervention_test, local_internal_path, null_and_locality_cleanliness, output_or_candidate_margin_separation, robustness_and_side_effects, transfer_or_widening` |
| `mc004_in_context_binding` | `not_reliable_monitor_only_no_lever` | `passed_or_partial` | `monitor_only_output_or_candidate_confounded` | `not_tested` | `untested` | `conditional_widening_requires_new_controls` | `clean_predicted_intervention, control-surviving_signature, intervention_test, local_internal_path, null_and_locality_cleanliness, output_or_candidate_margin_separation, robustness_and_side_effects, transfer_or_widening` |
| `mc005_associative_lookup` | `bounded_reliability_reference` | `passed_or_partial` | `bounded_internal_signature` | `bounded_causal_dirty` | `bounded_low_margin_and_model_size_fragile` | `bounded_transfer_fragile_reference` | `clean_predicted_intervention, null_and_locality_cleanliness, robustness_and_side_effects, transfer_or_widening` |
| `mc006_parametric_fact_override` | `not_reliable_monitor_only_no_lever` | `passed_or_partial` | `monitor_only_output_or_candidate_confounded` | `failed` | `bounded` | `transfer_failed_or_bank_insufficient` | `clean_predicted_intervention, control-surviving_signature, local_internal_path, null_and_locality_cleanliness, output_or_candidate_margin_separation, robustness_and_side_effects, transfer_or_widening` |
| `mc007_semi_synthetic_familiar_entity_lookup` | `not_reliable_behavior_or_bridge_blocked` | `failed_or_closed_before_hidden_state` | `not_allowed_or_not_reached` | `not_allowed` | `behavior_only` | `closed_route_no_widening` | `behavior_substrate, clean_predicted_intervention, control-surviving_signature, intervention_test, local_internal_path, null_and_locality_cleanliness, robustness_and_side_effects, transfer_or_widening` |
| `mc008_symbolic_fact_code_arbitration` | `not_reliable_behavior_or_bridge_blocked` | `failed_or_closed_before_hidden_state` | `not_allowed_or_not_reached` | `not_allowed` | `failed_or_fragile` | `closed_route_no_widening` | `behavior_substrate, clean_predicted_intervention, control-surviving_signature, intervention_test, local_internal_path, null_and_locality_cleanliness, robustness_and_side_effects, transfer_or_widening` |
| `mc009_derived_code_arbitration` | `not_reliable_behavior_or_bridge_blocked` | `failed_or_closed_before_hidden_state` | `not_allowed_or_not_reached` | `not_allowed` | `behavior_only` | `closed_route_no_widening` | `behavior_substrate, clean_predicted_intervention, control-surviving_signature, intervention_test, local_internal_path, null_and_locality_cleanliness, robustness_and_side_effects, transfer_or_widening` |
| `mc010_two_hop_fact_code_arbitration` | `not_reliable_behavior_or_bridge_blocked` | `failed_or_closed_before_hidden_state` | `not_allowed_or_not_reached` | `not_allowed` | `behavior_only` | `closed_route_no_widening` | `behavior_substrate, clean_predicted_intervention, control-surviving_signature, intervention_test, local_internal_path, null_and_locality_cleanliness, robustness_and_side_effects, transfer_or_widening` |
| `mc011_atomic_number_code_arbitration` | `not_reliable_behavior_or_bridge_blocked` | `failed_or_closed_before_hidden_state` | `not_allowed_or_not_reached` | `not_allowed` | `behavior_only` | `closed_route_no_widening` | `behavior_substrate, clean_predicted_intervention, control-surviving_signature, intervention_test, local_internal_path, null_and_locality_cleanliness, robustness_and_side_effects, transfer_or_widening` |
| `mc012_reliability_labeled_numeric_arbitration` | `not_reliable_prompt_visible_positive_control` | `passed_but_prompt_visible` | `not_allowed_or_not_reached` | `not_allowed` | `behavior_only` | `prompt_visible_no_transfer_claim` | `clean_predicted_intervention, control-surviving_signature, intervention_test, local_internal_path, null_and_locality_cleanliness, prompt_channel_locality, robustness_and_side_effects, transfer_or_widening` |
| `mc013_status_channel_ablation_numeric_arbitration` | `not_reliable_behavior_or_bridge_blocked` | `failed_or_closed_before_hidden_state` | `not_allowed_or_not_reached` | `not_allowed` | `behavior_only` | `closed_route_no_widening` | `behavior_substrate, clean_predicted_intervention, control-surviving_signature, intervention_test, local_internal_path, null_and_locality_cleanliness, robustness_and_side_effects, transfer_or_widening` |
| `mc014_inferred_reliability_numeric_arbitration` | `not_reliable_behavior_or_bridge_blocked` | `failed_or_closed_before_hidden_state` | `not_allowed_or_not_reached` | `not_allowed` | `behavior_only` | `closed_route_no_widening` | `behavior_substrate, clean_predicted_intervention, control-surviving_signature, intervention_test, local_internal_path, null_and_locality_cleanliness, robustness_and_side_effects, transfer_or_widening` |
| `mc015_parity_gated_numeric_arbitration` | `not_reliable_behavior_or_bridge_blocked` | `failed_or_closed_before_hidden_state` | `not_allowed_or_not_reached` | `not_allowed` | `behavior_only` | `closed_route_no_widening` | `behavior_substrate, clean_predicted_intervention, control-surviving_signature, intervention_test, local_internal_path, null_and_locality_cleanliness, robustness_and_side_effects, transfer_or_widening` |
| `mc016_alphabet_gated_numeric_arbitration` | `not_reliable_behavior_or_bridge_blocked` | `failed_or_closed_before_hidden_state` | `not_allowed_or_not_reached` | `not_allowed` | `behavior_only` | `closed_route_no_widening` | `behavior_substrate, clean_predicted_intervention, control-surviving_signature, intervention_test, local_internal_path, null_and_locality_cleanliness, robustness_and_side_effects, transfer_or_widening` |

## Validation Checks

| Check | Passed | Actual |
| --- | --- | --- |
| `reliability_entries_cover_each_atlas_row_once` | `true` | `["mc001_qwen3_0p6b_truth_agreement", "mc001b_qwen3_1p7b_truth_agreement", "mc001g_gemma_truth_agreement", "mc002_known_unknown", "mc002b_context_support", "mc003_delayed_copy", "mc004_in_context_binding", "mc005_associative_lookup", "mc006_parametric_fact_override", "mc007_semi_synthetic_familiar_entity_lookup", "mc008_symbolic_fact_code_arbitration", "mc009_derived_code_arbitration", "mc010_two_hop_fact_code_arbitration", "mc011_atomic_number_code_arbitration", "mc012_reliability_labeled_numeric_arbitration", "mc013_status_channel_ablation_numeric_arbitration", "mc014_inferred_reliability_numeric_arbitration", "mc015_parity_gated_numeric_arbitration", "mc016_alphabet_gated_numeric_arbitration"]` |
| `reliability_classes_match_route_dispositions` | `true` | `{"expected_from_routes": {"bounded_reliability_reference": 1, "not_reliable_behavior_or_bridge_blocked": 11, "not_reliable_failed_intervention_route": 1, "not_reliable_monitor_only_no_lever": 2, "not_reliable_output_shadow_diagnostic": 3, "not_reliable_prompt_visible_positive_control": 1}, "matrix": {"bounded_reliability_reference": 1, "not_reliable_behavior_or_bridge_blocked": 11, "not_reliable_failed_intervention_route": 1, "not_reliable_monitor_only_no_lever": 2, "not_reliable_output_shadow_diagnostic": 3, "not_reliable_prompt_visible_positive_control": 1}}` |
| `no_full_reliability_mechanism` | `true` | `0` |
| `mc005_is_only_bounded_reliability_reference` | `true` | `["mc005_associative_lookup"]` |
| `monitor_only_rows_are_mc004_and_mc006` | `true` | `["mc004_in_context_binding", "mc006_parametric_fact_override"]` |
| `mc012_is_prompt_visible_positive_control` | `true` | `["mc012_reliability_labeled_numeric_arbitration"]` |
| `every_row_has_failed_or_missing_gates` | `true` | `{}` |
| `no_transfer_ready_mechanism_crosscheck` | `true` | `0` |
| `clean_predicted_intervention_missing_is_majority` | `true` | `19` |
| `signature_gate_has_output_confounded_rows` | `true` | `{"bounded_internal_signature": 1, "monitor_only_output_or_candidate_confounded": 2, "not_allowed_or_not_reached": 12, "output_visible_or_zero_lead": 4}` |

## Interpretation

The most important result is not that the current atlas lacks a clean
promoted mechanism. The important result is that the failures are typed.
MC005 is the bounded reference: behavior, source-value signal, localized
attention/write mediation, and causal effect exist, but null locality and
model-size robustness prevent full reliability. MC004 and MC006 show
predecision monitors without reliable levers. MC001, MC001B, and MC003
are output-shadow diagnostics. MC012 is a prompt-visible positive control.
The remaining bridge rows mostly die before hidden-state work is justified.

This is the claim-killing engine in machine-readable form. Future work
should improve the distribution by changing rows between classes, not by
softening the gates.

## Claim Boundary

The current atlas has one bounded reliability reference and zero full reliability mechanisms. Typed failures are not bookkeeping; they are the measured shape of where control-surface claims break.

No row may be described as reliable mechanism control unless all signature, intervention, null/locality, robustness, side-effect, and transfer gates are clean under the recorded controls.
