# Control-Surface Transfer Matrix

Date: 2026-07-01

Status: generated transfer/widening matrix implemented and validated.

Machine-readable artifact:

> `data/control_surface_transfer_matrix.json`

Builder:

> `code/control_surface_transfer_matrix.py`

Commands:

```powershell
python code\control_surface_transfer_matrix.py --write
python code\control_surface_transfer_matrix.py
python code\validate_control_surface_atlas.py
```

## Purpose

The transfer matrix records which control surfaces are local, fragile,
failed under transfer, or not yet eligible for widening. It prevents
cross-model anecdotes from becoming transfer claims.

## Generated Facts

- atlas rows: 19;
- transfer-ready mechanisms: 0;
- multi-model rows: 3;
- transfer-gap queue items: 2.

## Transfer Classes

| Transfer Class | Count | Ratio |
| --- | ---: | ---: |
| `bounded_transfer_fragile_reference` | 1 | 0.053 |
| `closed_route_no_widening` | 10 | 0.526 |
| `conditional_widening_requires_new_controls` | 1 | 0.053 |
| `diagnostic_cross_model_evidence_only` | 3 | 0.158 |
| `output_shadow_widening_baseline` | 2 | 0.105 |
| `prompt_visible_no_transfer_claim` | 1 | 0.053 |
| `transfer_failed_or_bank_insufficient` | 1 | 0.053 |

## Row Matrix

| Row | Transfer Value | Transfer Class | Route Disposition | Required Gate |
| --- | --- | --- | --- | --- |
| `mc001_qwen3_0p6b_truth_agreement` | `untested` | `output_shadow_widening_baseline` | `output_shadow_diagnostic_baseline` | Hidden signature and intervention beat matched output/candidate baselines on holdout. |
| `mc001b_qwen3_1p7b_truth_agreement` | `low` | `diagnostic_cross_model_evidence_only` | `output_shadow_diagnostic_baseline` | A new substrate must beat output/prompt/shuffle controls before transfer matters. |
| `mc001g_gemma_truth_agreement` | `medium` | `diagnostic_cross_model_evidence_only` | `failed_intervention_or_mechanism_route` | A new substrate must beat output/prompt/shuffle controls before transfer matters. |
| `mc002_known_unknown` | `low` | `diagnostic_cross_model_evidence_only` | `closed_before_hidden_state` | A new substrate must beat output/prompt/shuffle controls before transfer matters. |
| `mc002b_context_support` | `untested` | `closed_route_no_widening` | `closed_before_hidden_state` | Behavior, parseability, nulls, source-disjoint holdout, and baseline gates. |
| `mc003_delayed_copy` | `untested` | `output_shadow_widening_baseline` | `output_shadow_diagnostic_baseline` | Hidden signature and intervention beat matched output/candidate baselines on holdout. |
| `mc004_in_context_binding` | `untested` | `conditional_widening_requires_new_controls` | `monitor_only_conditional_revisit` | Lead-time monitor must survive subgroup, shuffle, output/candidate, and intervention checks. |
| `mc005_associative_lookup` | `medium` | `bounded_transfer_fragile_reference` | `bounded_mechanism_frozen` | Primary effect plus null locality, source-disjoint holdout, side effects, and model-size robustness. |
| `mc006_parametric_fact_override` | `failed` | `transfer_failed_or_bank_insufficient` | `monitor_only_closed` | At least two transfer-ready templates or cross-model rows with candidate/output controls and shuffle/null gates. |
| `mc007_semi_synthetic_familiar_entity_lookup` | `untested` | `closed_route_no_widening` | `closed_before_hidden_state` | Behavior, parseability, nulls, source-disjoint holdout, and baseline gates. |
| `mc008_symbolic_fact_code_arbitration` | `untested` | `closed_route_no_widening` | `closed_before_hidden_state` | Behavior, parseability, nulls, source-disjoint holdout, and baseline gates. |
| `mc009_derived_code_arbitration` | `untested` | `closed_route_no_widening` | `closed_before_hidden_state` | Behavior, parseability, nulls, source-disjoint holdout, and baseline gates. |
| `mc010_two_hop_fact_code_arbitration` | `untested` | `closed_route_no_widening` | `closed_before_hidden_state` | Behavior, parseability, nulls, source-disjoint holdout, and baseline gates. |
| `mc011_atomic_number_code_arbitration` | `untested` | `closed_route_no_widening` | `closed_before_hidden_state` | Behavior, parseability, nulls, source-disjoint holdout, and baseline gates. |
| `mc012_reliability_labeled_numeric_arbitration` | `untested` | `prompt_visible_no_transfer_claim` | `prompt_visible_positive_control` | Prompt-channel locality before transfer. |
| `mc013_status_channel_ablation_numeric_arbitration` | `untested` | `closed_route_no_widening` | `closed_before_hidden_state` | Behavior, parseability, nulls, source-disjoint holdout, and baseline gates. |
| `mc014_inferred_reliability_numeric_arbitration` | `untested` | `closed_route_no_widening` | `closed_before_hidden_state` | Behavior, parseability, nulls, source-disjoint holdout, and baseline gates. |
| `mc015_parity_gated_numeric_arbitration` | `untested` | `closed_route_no_widening` | `closed_before_hidden_state` | Behavior, parseability, nulls, source-disjoint holdout, and baseline gates. |
| `mc016_alphabet_gated_numeric_arbitration` | `untested` | `closed_route_no_widening` | `closed_before_hidden_state` | Behavior, parseability, nulls, source-disjoint holdout, and baseline gates. |

## Transfer-Gap Queue Items

| Queue Item | Priority | Hypothesis | Next Test |
| --- | --- | --- | --- |
| `transfer_fails_at_reliability_before_primary_effect__next_test_1` | `immediate` | `transfer_fails_at_reliability_before_primary_effect` | Run any future transfer test with matched null panels and side rows from the start. |
| `transfer_fails_at_reliability_before_primary_effect__next_test_2` | `immediate` | `transfer_fails_at_reliability_before_primary_effect` | Do not mark a surface as transferred unless the atlas row's null_locality and transfer fields both move beyond bounded. |

## Validation Checks

| Check | Passed | Actual |
| --- | --- | --- |
| `transfer_entries_cover_each_atlas_row_once` | `true` | `["mc001_qwen3_0p6b_truth_agreement", "mc001b_qwen3_1p7b_truth_agreement", "mc001g_gemma_truth_agreement", "mc002_known_unknown", "mc002b_context_support", "mc003_delayed_copy", "mc004_in_context_binding", "mc005_associative_lookup", "mc006_parametric_fact_override", "mc007_semi_synthetic_familiar_entity_lookup", "mc008_symbolic_fact_code_arbitration", "mc009_derived_code_arbitration", "mc010_two_hop_fact_code_arbitration", "mc011_atomic_number_code_arbitration", "mc012_reliability_labeled_numeric_arbitration", "mc013_status_channel_ablation_numeric_arbitration", "mc014_inferred_reliability_numeric_arbitration", "mc015_parity_gated_numeric_arbitration", "mc016_alphabet_gated_numeric_arbitration"]` |
| `transfer_values_match_comparison_axis_counts` | `true` | `{"comparison": {"failed": 1, "low": 2, "medium": 2, "untested": 14}, "matrix": {"failed": 1, "low": 2, "medium": 2, "untested": 14}}` |
| `no_transfer_ready_mechanism` | `true` | `0` |
| `mc005_is_bounded_transfer_fragile_reference` | `true` | `["bounded_transfer_fragile_reference"]` |
| `mc006_transfer_route_failed` | `true` | `["transfer_failed_or_bank_insufficient"]` |
| `untested_transfer_is_majority` | `true` | `14` |
| `transfer_gap_queue_items_exist` | `true` | `2` |

## Interpretation

The current transfer picture is mostly absence, fragility, or failure.
MC005 is useful as the bounded reference specimen, but its transfer
story is explicitly null/model-size fragile. MC006 is the negative
transfer exemplar: locked-coordinate, expanded-bank, and transfer-role
repair routes are not enough. Most bridge rows are closed before
transfer is even a meaningful question.

## Claim Boundary

The current atlas has no transfer-ready mechanism. MC005 is a bounded transfer-fragile reference specimen, MC006 transfer is failed or bank-insufficient, and most rows are untested or closed before widening.

Medium or cross-model evidence in this matrix is not a transfer success unless signature, intervention, null, locality, side-effect, and output/candidate controls survive on the widened target.
