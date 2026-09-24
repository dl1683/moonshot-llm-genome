# Control-Surface Coverage Gaps

Date: 2026-07-01

Status: generated coverage-gap layer implemented and validated.

Machine-readable artifact:

> `data/control_surface_coverage_gaps.json`

Builder:

> `code/control_surface_coverage_gaps.py`

Commands:

```powershell
python code\control_surface_coverage_gaps.py --write
python code\control_surface_coverage_gaps.py
python code\validate_control_surface_atlas.py
```

## Purpose

This layer is the negative-space map. It records which claims the
current atlas still cannot support, which apparent laws are too sparse,
and which next pressures are attached to those gaps.

## Generated Facts

- rows: 19;
- coverage gaps: 12;
- severity counts: `{"critical": 4, "high": 3, "medium": 5}`;
- gap type counts: `{"behavior_domain_coverage": 1, "bridge_substrate": 1, "intervention": 1, "knowledge_mechanism": 1, "mechanism_promotion": 1, "mechanism_reference": 1, "model_coverage": 1, "predictive_law_support": 2, "reliability": 1, "sample_size": 1, "transfer": 1}`;
- transfer-untested rows: 14;
- hidden-state-allowed bridge rungs: 0;
- clean unconfounded bridge rungs: 0;
- singleton+sparse feature summaries: 82.

## Gap Ledger

| Gap | Severity | Type | Evidence | Exit Condition |
| --- | --- | --- | --- | --- |
| `clean_intervention_absent` | `critical` | `intervention` | `{"clean_predicted_intervention_missing": 19, "intervention_test_missing": 14, "row_count": 19}` | A row passes a predicted intervention with documented nulls, locality, side effects, and holdout behavior. |
| `promoted_mechanism_absent` | `critical` | `mechanism_promotion` | `{"promoted_mechanism_count": 0, "row_count": 19}` | At least one row clears signature, intervention, reliability, null/locality, robustness, and transfer gates. |
| `full_reliability_absent` | `critical` | `reliability` | `{"full_reliability_count": 0, "missing_gate_counts": {"behavior_substrate": 11, "clean_predicted_intervention": 19, "control-surviving_signature": 18, "intervention_test": 14, "local_internal_path": 18, "null_and_locality_cleanliness": 19, "output_or_candidate_margin_separation": 6, "prompt_channel_locality": 1, "robustness_and_side_effects": 19, "transfer_or_widening": 19}}` | Full reliability count becomes nonzero without contradicting missing-gate validation. |
| `transfer_ready_mechanism_absent` | `critical` | `transfer` | `{"multi_model_row_count": 3, "transfer_ready_mechanism_count": 0, "transfer_value_counts": {"failed": 1, "low": 2, "medium": 2, "untested": 14}}` | A mechanism-like row transfers with matched null panels, side rows, and reliability fields beyond bounded. |
| `bridge_hidden_state_not_licensed` | `high` | `bridge_substrate` | `{"clean_unconfounded_bridge_count": 0, "hidden_state_allowed_count": 0, "recent_closed_rungs": ["MC030", "MC031", "MC032", "MC033"], "rung_count": 24, "smoke_rung_count": 17}` | A materially different bridge passes branch, null, local, side-number, prompt-channel, output/candidate, and source-disjoint gates together. |
| `axis_rules_sparse_or_singleton_heavy` | `high` | `predictive_law_support` | `{"evidence_level_counts": {"broad": 29, "singleton": 73, "sparse": 9, "supported": 12}, "feature_count": 123, "singleton_plus_sparse_count": 82}` | Broad-or-supported pure rules increase while singleton/sparse evidence stops dominating feature summaries. |
| `singleton_terminal_stage_evidence` | `high` | `sample_size` | `{"singleton_terminal_rows": ["mc001g_gemma_truth_agreement", "mc012_reliability_labeled_numeric_arbitration", "mc005_associative_lookup"], "singleton_terminal_stages": ["intervention_failed", "pre_signature_prompt_channel_locality", "reliability_null_boundary"]}` | At least three materially distinct rows occupy prompt-channel locality, failed-intervention, and reliability-boundary stages. |
| `domain_coverage_knowledge_bridge_dominates_failures` | `medium` | `behavior_domain_coverage` | `{"domain_row_counts": {"abstention_known_unknown": 2, "delayed_copy": 1, "in_context_binding": 1, "knowledge_bridge": 10, "parametric_fact_override": 1, "synthetic_lookup": 1, "truth_agreement": 3}, "domain_rows": {"abstention_known_unknown": ["mc002_known_unknown", "mc002b_context_support"], "delayed_copy": ["mc003_delayed_copy"], "in_context_binding": ["mc004_in_context_binding"], "knowledge_bridge": ["mc007_semi_synthetic_familiar_entity_lookup", "mc008_symbolic_fact_code_arbitration", "mc009_derived_code_arbitration", "mc010_two_hop_fact_code_arbitration", "mc011_atomic_number_code_arbitration", "mc012_reliability_labeled_numeric_arbitration", "mc013_status_channel_ablation_numeric_arbitration", "mc014_inferred_reliability_numeric_arbitration", "mc015_parity_gated_numeric_arbitration", "mc016_alphabet_gated_numeric_arbitration"], "parametric_fact_override": ["mc006_parametric_fact_override"], "synthetic_lookup": ["mc005_associative_lookup"], "truth_agreement": ["mc001_qwen3_0p6b_truth_agreement", "mc001b_qwen3_1p7b_truth_agreement", "mc001g_gemma_truth_agreement"]}, "knowledge_bridge_rows": ["mc007_semi_synthetic_familiar_entity_lookup", "mc008_symbolic_fact_code_arbitration", "mc009_derived_code_arbitration", "mc010_two_hop_fact_code_arbitration", "mc011_atomic_number_code_arbitration", "mc012_reliability_labeled_numeric_arbitration", "mc013_status_channel_ablation_numeric_arbitration", "mc014_inferred_reliability_numeric_arbitration", "mc015_parity_gated_numeric_arbitration", "mc016_alphabet_gated_numeric_arbitration"]}` | At least two non-bridge domains advance beyond singleton evidence at signature, intervention, or reliability stages. |
| `mc006_knowledge_route_monitor_only` | `medium` | `knowledge_mechanism` | `{"mc006_terminal_stage": ["mc004_in_context_binding", "mc006_parametric_fact_override"], "predecision_causal_candidate_count": 0}` | A knowledge-like row beats output/candidate baselines and supports a clean intervention. |
| `mc005_singleton_internal_causal_reference` | `medium` | `mechanism_reference` | `{"bounded_mechanism_count": 1, "bounded_rows": ["mc005_associative_lookup"]}` | A second materially distinct internal-causal bounded or promoted row appears outside MC005. |
| `model_family_coverage_qwen_dominant` | `medium` | `model_coverage` | `{"model_row_counts": {"Qwen/Qwen3-0.6B": 6, "Qwen/Qwen3-1.7B": 13, "google/gemma-2-2b": 1, "google/gemma-2-2b-it": 2}, "model_rows": {"Qwen/Qwen3-0.6B": ["mc001_qwen3_0p6b_truth_agreement", "mc002_known_unknown", "mc002b_context_support", "mc003_delayed_copy", "mc004_in_context_binding", "mc005_associative_lookup"], "Qwen/Qwen3-1.7B": ["mc001b_qwen3_1p7b_truth_agreement", "mc005_associative_lookup", "mc006_parametric_fact_override", "mc007_semi_synthetic_familiar_entity_lookup", "mc008_symbolic_fact_code_arbitration", "mc009_derived_code_arbitration", "mc010_two_hop_fact_code_arbitration", "mc011_atomic_number_code_arbitration", "mc012_reliability_labeled_numeric_arbitration", "mc013_status_channel_ablation_numeric_arbitration", "mc014_inferred_reliability_numeric_arbitration", "mc015_parity_gated_numeric_arbitration", "mc016_alphabet_gated_numeric_arbitration"], "google/gemma-2-2b": ["mc001g_gemma_truth_agreement"], "google/gemma-2-2b-it": ["mc001g_gemma_truth_agreement", "mc002_known_unknown"]}}` | Each major terminal-stage or mechanism-like claim has at least one materially comparable non-Qwen replication or failure. |
| `output_geometry_pressure_mixed` | `medium` | `predictive_law_support` | `{"mixed_predictor": [{"dominant_terminal_stage": "signature_output_geometry_shadow", "dominant_terminal_stage_count": 3, "evidence_level": "broad", "feature_key": "primary_blocker:primary_blocker=output_geometry_shadow", "name": "primary_blocker", "primary_blocker_counts": {"output_geometry_shadow": 6}, "route_disposition_counts": {"failed_intervention_or_mechanism_route": 1, "monitor_only_closed": 1, "monitor_only_conditional_revisit": 1, "output_shadow_diagnostic_baseline": 3}, "row_count": 6, "rows": ["mc001_qwen3_0p6b_truth_agreement", "mc001b_qwen3_1p7b_truth_agreement", "mc001g_gemma_truth_agreement", "mc003_delayed_copy", "mc004_in_context_binding", "mc006_parametric_fact_override"], "source": "primary_blocker", "terminal_stage_counts": {"intervention_failed": 1, "signature_monitor_no_lever": 2, "signature_output_geometry_shadow": 3}, "terminal_stage_purity": 0.5, "value": "output_geometry_shadow"}], "output_geometry_visible_count": 14}` | A future layer separates output-shadow, monitor-only, and failed-intervention cases with broader supported rules. |

## Interpretation

The atlas is not merely incomplete in an abstract sense. Its missing
coverage is structured: no promoted mechanisms, no full reliability,
no transfer-ready mechanisms, no hidden-state-ready bridge, thin
model-family coverage, singleton terminal-stage boundaries, and many
singleton or sparse feature regularities.

The immediate scientific pressure is therefore not to make a stronger
claim from the current map. It is to reduce one of these gaps in a
way that changes the generated snapshot, gate geometry, or axis
interaction artifact.

## What This Proves

It proves that the current negative space is now explicit and tied to
validated artifacts. Future experiments can be judged by whether
they reduce a named gap rather than merely adding another row.

## What It Does Not Prove

It does not prove that these gaps are exhaustive. It proves that these
are the current validated gaps implied by the checked-in atlas stack.
