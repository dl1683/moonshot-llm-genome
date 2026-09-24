# Control-Surface Compositional Genome Audit

Source updated_at: 2026-07-01

Status: generated compositional-genome audit implemented and validated.

Machine-readable artifact:

> `data/control_surface_compositional_genome_audit.json`

Builder:

> `code/control_surface_compositional_genome_audit.py`

Commands:

```powershell
python code\control_surface_compositional_genome_audit.py --write
python code\control_surface_compositional_genome_audit.py
python code\validate_control_surface_atlas.py
```

## Purpose

This audit makes the project-level insight explicit: the current genome
object is not a single truth vector, knowledge vector, or mechanism card.
It is the measured distribution of where behavior lives across prompt,
output, source-token, internal-monitor, causal, reliability, and transfer
axes.

## Generated Facts

- atlas rows: 19;
- promoted mechanism cards: 0;
- prompt-contract-visible rows: 19;
- output-geometry-visible rows: 14;
- source-or-prompt-token-dependent rows: 13;
- behavior-or-bridge-substrate-blocked rows: 11;
- internal-monitor-present rows: 5;
- predecision-monitor-with-no-lever rows: 2;
- internal-causal-surface rows: 1;
- full-reliability mechanisms: 0;
- transfer-ready mechanisms: 0;
- clean unconfounded bridge substrates: 0.

## Distribution Axes

| Axis | Count | Denominator | Ratio | Interpretation |
| --- | ---: | --- | ---: | --- |
| `prompt_contract_visible` | 19 | 19 `atlas_rows` | 1.000 | Prompt wording, authority, format, or contract remains visible in every current atlas row. |
| `output_geometry_visible` | 14 | 19 `atlas_rows` | 0.737 | Most rows expose the behavior through answer/candidate/output geometry before a clean internal mechanism claim can be made. |
| `source_or_prompt_token_dependent` | 13 | 19 `atlas_rows` | 0.684 | Source tokens, local prompt values, or prompt-side routing pressure explain a majority of current behavior surfaces. |
| `behavior_or_bridge_substrate_blocked` | 11 | 19 `atlas_rows` | 0.579 | The largest primary failure mode is still getting a behavior substrate that justifies hidden-state work. |
| `internal_monitor_present` | 5 | 19 `atlas_rows` | 0.263 | Internal monitors exist, but most are output shadows, non-causal signatures, or monitor-only rows. |
| `predecision_monitor_no_lever` | 2 | 19 `atlas_rows` | 0.105 | Only MC004 and MC006 currently show predecision monitor evidence, and neither supplies a reliable lever. |
| `internal_causal_surface` | 1 | 19 `atlas_rows` | 0.053 | Only MC005 is currently counted as a bounded internal-causal surface. |
| `bounded_reliability_reference` | 1 | 19 `atlas_rows` | 0.053 | The reliability matrix has one bounded reference and no full-reliability mechanism. |
| `full_reliability_mechanism` | 0 | 19 `atlas_rows` | 0.000 | No row currently passes all behavior, signature, intervention, null/locality, robustness, and transfer gates. |
| `transfer_ready_mechanism` | 0 | 19 `atlas_rows` | 0.000 | No mechanism is currently transfer-ready. |
| `transfer_untested_or_missing` | 14 | 19 `atlas_rows` | 0.737 | Transfer remains mostly untested or unavailable, so generality claims are still barred. |
| `clean_unconfounded_bridge_substrate` | 0 | 24 `bridge_rungs` | 0.000 | The bridge ladder has not produced a clean unconfounded knowledge-like substrate for hidden-state work. |

## Anchor Findings

| Anchor | Role | Status | Insight |
| --- | --- | --- | --- |
| `mc005_reference_specimen` | `bounded_internal_causal_reference` | `{"reliability_class": "bounded_reliability_reference", "route_status": "bounded_frozen_not_promoted", "row_id": "mc005_associative_lookup", "terminal_stage": "reliability_null_boundary", "transfer_class": "bounded_transfer_fragile_reference", "verdict": "bounded_mechanism_card"}` | MC005 shows that a narrow internal causal surface can be real while still failing full reliability through null-locality and model-size transfer boundaries. |
| `mc006_predecision_monitor` | `knowledge_like_monitor_without_lever` | `{"frontier_class": "predecision_monitor_no_lever", "reliability_class": "not_reliable_monitor_only_no_lever", "route_status": "monitor_only_closed", "row_id": "mc006_parametric_fact_override", "terminal_stage": "signature_monitor_no_lever", "transfer_class": "transfer_failed_or_bank_insufficient", "verdict": "diagnostic_note"}` | MC006 shows why lead-time must be measured separately from causality: a matched generated behavior substrate and early monitor can coexist with failed steering, output/candidate visibility, shuffled-label failure, and transfer-bank closure. |
| `post_mc033_bridge_closeout` | `bridge_substrate_death_condition` | `{"clean_unconfounded_bridge_count": 0, "hidden_state_allowed_count": 0}` | The bridge sequence shows that clean direct controls and nulls can coexist with failed learned/local arbitration, so direct control success cannot license hidden-state work. |
| `monitor_only_frontier` | `lead_time_boundary` | `{"monitor_only_rows": ["mc004_in_context_binding", "mc006_parametric_fact_override"], "predecision_causal_candidate_count": 0}` | The current lead-time frontier has monitors, not levers: MC004 and MC006 are informative timing rows, but no atlas row is a predecision causal candidate. |

## Insight Claims

| Claim | Status | Interpretation |
| --- | --- | --- |
| `mixture_is_current_genome_object` | `supported_current_map` | The central present result is the distribution of behavior across visible, source-dependent, monitor-only, and bounded causal surfaces. |
| `typed_failures_are_primary_data` | `supported_current_map` | Output shadows, prompt-visible positives, bridge blocks, null boundaries, and transfer failures are not just dead ends; they are reusable diagnostic classes. |
| `lead_time_is_boundary_not_promotion` | `supported_current_map` | A signal can appear before the local output readout and still fail as a causal control surface once global output, shuffle, intervention, and transfer controls are applied. |
| `reliability_and_transfer_are_current_bottlenecks` | `supported_current_map` | The project can discuss control-surface structure, but broad mechanism and generality claims remain barred. |

## Validation Checks

| Check | Passed | Actual |
| --- | --- | --- |
| `source_row_counts_align` | `true` | `{"atlas": 19, "frontier": 19, "mixture": 19, "reliability": 19, "transfer": 19}` |
| `prompt_contract_pressure_is_universal` | `true` | `{"axis_id": "prompt_contract_visible", "count": 19, "denominator": 19, "denominator_label": "atlas_rows", "interpretation": "Prompt wording, authority, format, or contract remains visible in every current atlas row.", "ratio": 1.0}` |
| `visible_surfaces_dominate_internal_causal_surface` | `true` | `{"internal_causal_surface": 1, "output_geometry_visible": 14, "source_or_prompt_token_dependent": 13}` |
| `no_promoted_full_or_transfer_ready_mechanism` | `true` | `{"full_reliability": 0, "promoted": 0, "transfer_ready": 0}` |
| `mc005_is_only_bounded_internal_reference` | `true` | `{"bounded_reliability_count": 1, "internal_causal_rows": ["mc005_associative_lookup"], "mc005_summary": {"combined_null_flip_count": 5, "lookup_write_effect_exact": true, "reliability_class": "bounded_reliability_reference", "route_status": "bounded_frozen_not_promoted", "row_id": "mc005_associative_lookup", "same_route_repair_allowed": false, "strict_answer_absent_nulls_clean": false, "terminal_stage": "reliability_null_boundary", "transfer_class": "bounded_transfer_fragile_reference", "verdict": "bounded_mechanism_card"}}` |
| `mc006_is_monitor_only_closed` | `true` | `{"behavior_substrate_passed": true, "final_or_candidate_geometry_blocks_promotion": true, "frontier_class": "predecision_monitor_no_lever", "monitor_only_closeout_gate_passed": true, "predecision_monitor_supported": true, "promotion_gate_passed": false, "reliability_class": "not_reliable_monitor_only_no_lever", "route_status": "monitor_only_closed", "row_id": "mc006_parametric_fact_override", "terminal_stage": "signature_monitor_no_lever", "transfer_class": "transfer_failed_or_bank_insufficient", "verdict": "diagnostic_note"}` |
| `monitor_only_rows_are_not_causal_candidates` | `true` | `{"monitor_only_rows": ["mc004_in_context_binding", "mc006_parametric_fact_override"], "predecision_causal_candidate_count": 0}` |
| `bridge_closeout_keeps_hidden_state_disallowed` | `true` | `{"bridge_rung_count": 24, "clean_unconfounded_bridge_count": 0, "hidden_state_allowed_count": 0, "recent_closed_rung_ids": ["MC030", "MC031", "MC032", "MC033"], "same_family_route_status": "killed_after_mc033", "same_family_sequence_count": 3, "same_family_sequence_ids": ["MC031", "MC032", "MC033"], "smoke_rung_count": 17}` |
| `primary_blockers_partition_atlas` | `true` | `{"behavior_substrate_or_bridge_blocked": 11, "bounded_internal_causal_with_null_boundary": 1, "output_geometry_shadow": 6, "prompt_visible_positive_control": 1}` |
| `claim_boundary_forbids_general_vector_language` | `true` | `{"bounded_internal_reference_exists": true, "full_reliability_mechanism_exists": false, "general_knowledge_control_surface_found": false, "general_truth_vector_found": false, "lead_time_axis_first_class": true, "mixture_distribution_measured": true, "promoted_mechanism_exists": false, "transfer_ready_mechanism_exists": false, "typed_failure_taxonomy_supported": true}` |

## Claim Boundary

The project has a validator-backed compositional map of the current small-model control-surface mixture: prompt-contract, output-geometry, source/token, monitor-only, bounded-causal, null-locality, and transfer axes explain why rows pass, fail, or remain bounded.

This audit does not promote any mechanism card, does not find a general truth vector, does not establish a general knowledge control surface, and does not license hidden-state work on bridge routes closed before behavior substrate readiness.

## Interpretation

The audit converts negative results into measured structure. MC005 is the
calibration object for bounded internal causality; MC006 is the
knowledge-like timing boundary; the post-MC033 bridge closeout is the
substrate death condition; and the distribution axes are the current
genome-level object. The next useful experiments should change these
ratios or explain why they are stable.
