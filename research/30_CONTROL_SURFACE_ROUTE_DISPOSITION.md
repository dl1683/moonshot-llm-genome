# Control-Surface Route Disposition

Date: 2026-07-01

Status: generated route-disposition layer implemented and validated.

Machine-readable artifact:

> `data/control_surface_route_disposition.json`

Builder:

> `code/control_surface_route_disposition.py`

Commands:

```powershell
python code\control_surface_route_disposition.py --write
python code\control_surface_route_disposition.py
python code\validate_control_surface_atlas.py
```

## Purpose

The route-disposition ledger is the claim-killing layer. It says which
routes are closed, bounded, monitor-only, prompt-visible positive
controls, output-shadow diagnostics, or failed interventions. It also
states what would be required to reopen or promote a route.

## Generated Facts

- atlas rows: 19;
- bridge rungs: 24;
- hidden-state-ready atlas routes: 0;
- bridge rungs allowing hidden-state work: 0;
- immediate/high next-queue items: 9.

## Atlas Dispositions

| Disposition | Count | Ratio |
| --- | ---: | ---: |
| `bounded_mechanism_frozen` | 1 | 0.053 |
| `closed_before_hidden_state` | 11 | 0.579 |
| `failed_intervention_or_mechanism_route` | 1 | 0.053 |
| `monitor_only_closed` | 1 | 0.053 |
| `monitor_only_conditional_revisit` | 1 | 0.053 |
| `output_shadow_diagnostic_baseline` | 3 | 0.158 |
| `prompt_visible_positive_control` | 1 | 0.053 |

## Row Ledger

| Row | Disposition | Primary Blocker | Frontier | Allowed Action |
| --- | --- | --- | --- | --- |
| `mc001_qwen3_0p6b_truth_agreement` | `output_shadow_diagnostic_baseline` | `output_geometry_shadow` | `output_visible_at_or_before_frontier` | Use as output-geometry diagnostic; change interface or row geometry before probing again. |
| `mc001b_qwen3_1p7b_truth_agreement` | `output_shadow_diagnostic_baseline` | `output_geometry_shadow` | `output_visible_at_or_before_frontier` | Use as output-geometry diagnostic; change interface or row geometry before probing again. |
| `mc001g_gemma_truth_agreement` | `failed_intervention_or_mechanism_route` | `output_geometry_shadow` | `output_visible_at_or_before_frontier` | Treat as a failed mechanism route unless a new prompt contract is preregistered. |
| `mc002_known_unknown` | `closed_before_hidden_state` | `behavior_substrate_or_bridge_blocked` | `frontier_not_reached` | Do not probe hidden states; rebuild the behavior substrate or close the branch. |
| `mc002b_context_support` | `closed_before_hidden_state` | `behavior_substrate_or_bridge_blocked` | `frontier_not_reached` | Do not probe hidden states; rebuild the behavior substrate or close the branch. |
| `mc003_delayed_copy` | `output_shadow_diagnostic_baseline` | `output_geometry_shadow` | `output_visible_at_or_before_frontier` | Use as output-geometry diagnostic; change interface or row geometry before probing again. |
| `mc004_in_context_binding` | `monitor_only_conditional_revisit` | `output_geometry_shadow` | `predecision_monitor_no_lever` | Revisit only with larger source-disjoint controls or a materially different causal stress test. |
| `mc005_associative_lookup` | `bounded_mechanism_frozen` | `bounded_internal_causal_with_null_boundary` | `causal_surface_not_timing_frontier` | Use as reference specimen; do not continue ordinary repair of the same route. |
| `mc006_parametric_fact_override` | `monitor_only_closed` | `output_geometry_shadow` | `predecision_monitor_no_lever` | Record as monitor-only; do not continue the current prompt family as a promotion route. |
| `mc007_semi_synthetic_familiar_entity_lookup` | `closed_before_hidden_state` | `behavior_substrate_or_bridge_blocked` | `frontier_not_reached` | Do not probe hidden states; rebuild the behavior substrate or close the branch. |
| `mc008_symbolic_fact_code_arbitration` | `closed_before_hidden_state` | `behavior_substrate_or_bridge_blocked` | `frontier_not_reached` | Do not probe hidden states; rebuild the behavior substrate or close the branch. |
| `mc009_derived_code_arbitration` | `closed_before_hidden_state` | `behavior_substrate_or_bridge_blocked` | `frontier_not_reached` | Do not probe hidden states; rebuild the behavior substrate or close the branch. |
| `mc010_two_hop_fact_code_arbitration` | `closed_before_hidden_state` | `behavior_substrate_or_bridge_blocked` | `frontier_not_reached` | Do not probe hidden states; rebuild the behavior substrate or close the branch. |
| `mc011_atomic_number_code_arbitration` | `closed_before_hidden_state` | `behavior_substrate_or_bridge_blocked` | `frontier_not_reached` | Do not probe hidden states; rebuild the behavior substrate or close the branch. |
| `mc012_reliability_labeled_numeric_arbitration` | `prompt_visible_positive_control` | `prompt_visible_positive_control` | `frontier_not_reached` | Use as positive control; do not run hidden-state mechanism work. |
| `mc013_status_channel_ablation_numeric_arbitration` | `closed_before_hidden_state` | `behavior_substrate_or_bridge_blocked` | `frontier_not_reached` | Do not probe hidden states; rebuild the behavior substrate or close the branch. |
| `mc014_inferred_reliability_numeric_arbitration` | `closed_before_hidden_state` | `behavior_substrate_or_bridge_blocked` | `frontier_not_reached` | Do not probe hidden states; rebuild the behavior substrate or close the branch. |
| `mc015_parity_gated_numeric_arbitration` | `closed_before_hidden_state` | `behavior_substrate_or_bridge_blocked` | `frontier_not_reached` | Do not probe hidden states; rebuild the behavior substrate or close the branch. |
| `mc016_alphabet_gated_numeric_arbitration` | `closed_before_hidden_state` | `behavior_substrate_or_bridge_blocked` | `frontier_not_reached` | Do not probe hidden states; rebuild the behavior substrate or close the branch. |

## Bridge Rungs

| Card | Source | Disposition | Mixture | Boundary |
| --- | --- | --- | --- | --- |
| `MC010` | `atlas` | `bridge_rung_closed_before_hidden_state` | `absent` | Two-hop indirection did not create a behavior substrate. |
| `MC011` | `atlas` | `bridge_rung_closed_before_hidden_state` | `absent` | Answer-format repair is insufficient under prompt-local table authority. |
| `MC012` | `atlas` | `prompt_visible_positive_control` | `clean_prompt_visible` | Positive control only; source-status text carries the answer rule. |
| `MC013` | `atlas` | `bridge_rung_closed_before_hidden_state` | `statused_only` | The contrast does not survive status-channel removal. |
| `MC014` | `atlas` | `bridge_rung_closed_before_hidden_state` | `absent` | Calibration evidence did not overcome local table authority. |
| `MC015` | `atlas` | `bridge_rung_closed_before_hidden_state` | `mixed_wrong_rule` | Mixed outputs are not enough when expected labels are not followed. |
| `MC016` | `atlas` | `bridge_rung_closed_before_hidden_state` | `absent` | A visible non-status gate still collapses expected-atomic rows. |
| `MC017` | `smoke` | `bridge_rung_closed_before_hidden_state` | `absent` | Source-token answers are themselves a confounded behavior surface. |
| `MC018` | `smoke` | `bridge_rung_closed_before_hidden_state` | `mixed_wrong_rule` | Neutral labels expose first-listed-choice and local-source salience. |
| `MC019` | `smoke` | `bridge_rung_closed_before_hidden_state` | `mixed_wrong_rule` | Clean controls and nulls do not imply expected-atomic routing. |
| `MC020` | `smoke` | `bridge_rung_closed_before_hidden_state` | `asymmetric_route_failure` | The bottleneck is conditional arbitration, not basic atomic recall. |
| `MC021` | `smoke` | `bridge_rung_closed_before_hidden_state` | `asymmetric_route_failure` | Opaque route codes are not a clean arbitration substrate. |
| `MC022` | `smoke` | `bridge_rung_closed_before_hidden_state` | `asymmetric_route_failure` | Semantic labels do not solve learned-memory branch arbitration. |
| `MC023` | `smoke` | `bridge_rung_closed_before_hidden_state` | `mixed_below_gate` | Query-level operation handles preserve controls and nulls but do not clear both conflict branches. |
| `MC024` | `smoke` | `bridge_rung_closed_before_hidden_state` | `asymmetric_route_failure` | Worked examples repair local routing but not full-source learned atomic routing. |
| `MC025` | `smoke` | `bridge_rung_closed_before_hidden_state` | `choice_interface_confounded` | Prompt-visible candidate choices break atomic controls and nulls while leaving learned conflict routing weak. |
| `MC026` | `smoke` | `bridge_rung_closed_before_hidden_state` | `numeric_option_confounded` | Numeric options preserve nulls but turn direct atomic control into UNKNOWN and leave learned conflict routing weak. |
| `MC027` | `smoke` | `bridge_rung_closed_before_hidden_state` | `interface_dependent_smoke_candidate` | Bare integer clears the 10-source smoke, but structured interfaces distort controls, nulls, or learned routing; MC024 remains the full-source boundary. |
| `MC028` | `smoke` | `bridge_rung_closed_before_hidden_state` | `full_source_boundary_failed` | The bare-integer smoke survivor fails full-source promotion: controls and nulls stay clean, but operation-atomic rows fall below gate with other-number leakage. |
| `MC029` | `smoke` | `bridge_rung_closed_before_hidden_state` | `factorial_branch_null_tradeoff` | No variant passes the bridge gate. Rules-only improves operation-atomic routing to 0.875 and removes worked-example outputs, but answer-absent nulls fall to 0.806; query-before-examples amplifies other-number leakage. |
| `MC030` | `smoke` | `bridge_rung_closed_before_hidden_state` | `guarded_branch_null_tradeoff` | No guard variant passes. The unguarded rules-only baseline remains the best branch/null compromise; guards either worsen nulls or collapse learned atomic routing. |
| `MC031` | `smoke` | `bridge_rung_closed_before_hidden_state` | `statusless_reliability_local_collapse` | Direct controls, valid-checksum local rows, and answer-absent nulls are clean in smoke, but invalid-checksum rows select local numbers on every selected conflict. |
| `MC032` | `smoke` | `bridge_rung_closed_before_hidden_state` | `statusless_reliability_local_collapse` | Direct controls, answer-absent nulls, and side-number locality are clean in smoke, but mismatch rows still avoid the learned atomic branch. |
| `MC033` | `smoke` | `bridge_rung_closed_before_hidden_state` | `statusless_reliability_local_and_claim_leak` | Direct controls and answer-absent nulls are clean in smoke, but the match branch returns atomic answers too often and mismatch rows split between local and the wrong claimed number. |

## Validation Checks

| Check | Passed | Actual |
| --- | --- | --- |
| `route_entries_cover_each_atlas_row_once` | `true` | `["mc001_qwen3_0p6b_truth_agreement", "mc001b_qwen3_1p7b_truth_agreement", "mc001g_gemma_truth_agreement", "mc002_known_unknown", "mc002b_context_support", "mc003_delayed_copy", "mc004_in_context_binding", "mc005_associative_lookup", "mc006_parametric_fact_override", "mc007_semi_synthetic_familiar_entity_lookup", "mc008_symbolic_fact_code_arbitration", "mc009_derived_code_arbitration", "mc010_two_hop_fact_code_arbitration", "mc011_atomic_number_code_arbitration", "mc012_reliability_labeled_numeric_arbitration", "mc013_status_channel_ablation_numeric_arbitration", "mc014_inferred_reliability_numeric_arbitration", "mc015_parity_gated_numeric_arbitration", "mc016_alphabet_gated_numeric_arbitration"]` |
| `mc005_is_bounded_frozen` | `true` | `["bounded_mechanism_frozen"]` |
| `mc006_is_monitor_only_closed` | `true` | `["monitor_only_closed"]` |
| `mc012_is_prompt_visible_positive_control` | `true` | `["prompt_visible_positive_control"]` |
| `no_active_hidden_state_route` | `true` | `{"bounded_mechanism_frozen": 1, "closed_before_hidden_state": 11, "failed_intervention_or_mechanism_route": 1, "monitor_only_closed": 1, "monitor_only_conditional_revisit": 1, "output_shadow_diagnostic_baseline": 3, "prompt_visible_positive_control": 1}` |
| `bridge_rungs_have_no_hidden_state_allowed_candidate` | `true` | `{"bridge_entries": 24, "hidden_state_allowed_bridge_candidate": 0}` |
| `closed_or_bounded_routes_dominate` | `true` | `{"bounded_mechanism_frozen": 1, "closed_before_hidden_state": 11, "row_count": 19}` |

## Interpretation

This ledger makes negative results operational. A closed route is not a
loose invitation to keep trying adjacent prompt tweaks; it is a branch
that can only be reopened by satisfying the listed promotion rule.
MC005 remains the bounded reference specimen, MC006 is closed
monitor-only under the current prompt families, and MC012 remains a
prompt-visible positive control rather than a mechanism candidate.

## Claim Boundary

The current route ledger has no active hidden-state-ready route: MC005 is bounded and frozen, MC006 is monitor-only closed, MC012 is a prompt-visible positive control, and most bridge routes are closed before hidden-state work.

This ledger does not promote a new mechanism. It prevents closed, prompt-visible, output-shadowed, or monitor-only routes from being treated as active mechanism candidates.
