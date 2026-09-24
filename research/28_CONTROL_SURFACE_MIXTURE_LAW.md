# Control-Surface Mixture Law

Date: 2026-07-01

Status: generated mixture-law layer implemented and validated.

Machine-readable artifact:

> `data/control_surface_mixture_law.json`

Builder:

> `code/control_surface_mixture_law.py`

Commands:

```powershell
python code\control_surface_mixture_law.py --write
python code\control_surface_mixture_law.py
python code\validate_control_surface_atlas.py
```

## Purpose

The mixture law turns the atlas from a list of mechanism-card attempts
into a measured distribution of where behavior currently lives. Rows can
belong to multiple pressure classes, but each row also gets one primary
blocker so the project can track whether future experiments change the
shape of the map.

## Generated Facts

- atlas rows: 19;
- promoted mechanism cards: 0;
- bounded internal-causal rows: 1;
- bridge ladder rungs: 24;
- clean unconfounded bridge rungs: 0;
- smoke hidden-state-allowed cards: 0.

Strong-or-failed axis ratios:

- `prompt_authority`: 1.000;
- `output_geometry`: 0.737;
- `source_token_dependence`: 0.632;
- `local_internal_path`: 0.053;
- `causal_control`: 0.263;
- `transfer`: 0.053;

## Primary Blockers

| Primary Blocker | Count | Ratio | Rows |
| --- | ---: | ---: | --- |
| `behavior_substrate_or_bridge_blocked` | 11 | 0.579 | `mc002_known_unknown`<br>`mc002b_context_support`<br>`mc007_semi_synthetic_familiar_entity_lookup`<br>`mc008_symbolic_fact_code_arbitration`<br>`mc009_derived_code_arbitration`<br>`mc010_two_hop_fact_code_arbitration`<br>`mc011_atomic_number_code_arbitration`<br>`mc013_status_channel_ablation_numeric_arbitration`<br>`mc014_inferred_reliability_numeric_arbitration`<br>`mc015_parity_gated_numeric_arbitration`<br>`mc016_alphabet_gated_numeric_arbitration` |
| `bounded_internal_causal_with_null_boundary` | 1 | 0.053 | `mc005_associative_lookup` |
| `output_geometry_shadow` | 6 | 0.316 | `mc001_qwen3_0p6b_truth_agreement`<br>`mc001b_qwen3_1p7b_truth_agreement`<br>`mc001g_gemma_truth_agreement`<br>`mc003_delayed_copy`<br>`mc004_in_context_binding`<br>`mc006_parametric_fact_override` |
| `prompt_visible_positive_control` | 1 | 0.053 | `mc012_reliability_labeled_numeric_arbitration` |

## Pressure Classes

| Pressure Class | Count | Ratio |
| --- | ---: | ---: |
| `behavior_or_bridge_substrate_blocked` | 11 | 0.579 |
| `internal_causal_surface` | 1 | 0.053 |
| `internal_monitor_present` | 5 | 0.263 |
| `null_boundary_or_locality_limited` | 6 | 0.316 |
| `output_geometry_visible` | 14 | 0.737 |
| `prompt_contract_visible` | 19 | 1.000 |
| `signature_or_intervention_failed` | 6 | 0.316 |
| `source_or_prompt_token_dependent` | 13 | 0.684 |
| `transfer_unproven_or_failed` | 3 | 0.158 |

## Validation Checks

| Check | Passed | Actual |
| --- | --- | --- |
| `primary_blockers_cover_each_atlas_row_once` | `true` | `["mc001_qwen3_0p6b_truth_agreement", "mc001b_qwen3_1p7b_truth_agreement", "mc001g_gemma_truth_agreement", "mc002_known_unknown", "mc002b_context_support", "mc003_delayed_copy", "mc004_in_context_binding", "mc005_associative_lookup", "mc006_parametric_fact_override", "mc007_semi_synthetic_familiar_entity_lookup", "mc008_symbolic_fact_code_arbitration", "mc009_derived_code_arbitration", "mc010_two_hop_fact_code_arbitration", "mc011_atomic_number_code_arbitration", "mc012_reliability_labeled_numeric_arbitration", "mc013_status_channel_ablation_numeric_arbitration", "mc014_inferred_reliability_numeric_arbitration", "mc015_parity_gated_numeric_arbitration", "mc016_alphabet_gated_numeric_arbitration"]` |
| `prompt_contract_is_universal_current_pressure` | `true` | `19` |
| `zero_promoted_mechanisms_preserved` | `true` | `0.000` |
| `exactly_one_bounded_internal_causal_surface` | `true` | `["mc005_associative_lookup"]` |
| `output_visibility_exceeds_internal_causal_surfaces` | `true` | `{"internal_causal_surface": 1, "output_geometry_visible": 14}` |
| `bridge_ladder_has_no_unconfounded_hidden_state_candidate` | `true` | `{"clean_unconfounded": 0, "hidden_state_allowed": 0}` |

## Interpretation

The central result is not that the project has found a broad knowledge
mechanism. It has not. The central result is that the current behavior
families distribute heavily over prompt contracts, output geometry, and
source-token/path surfaces, while bounded internal causality is currently
represented by one narrow MC005 row.

That makes typed failure a primary datum. A result killed by output
geometry, prompt visibility, bridge-substrate collapse, null locality,
or transfer fragility is not just a dead mechanism claim; it is one more
measurement of the mixture.

## Claim Boundary

The current small-LLM genome map is a measured mixture distribution: prompt-contract and output/source-visible pressures dominate, MC005 is the only bounded internal-causal row, and the MC010-MC022 bridge has not produced a hidden-state-ready unconfounded knowledge substrate.

This artifact does not prove a general truth vector, a broad knowledge mechanism, or a deployable factual-control intervention.
