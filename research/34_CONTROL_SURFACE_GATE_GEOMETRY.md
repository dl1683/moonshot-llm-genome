# Control-Surface Gate Geometry

Date: 2026-07-01

Status: generated gate-geometry layer implemented and validated.

Machine-readable artifact:

> `data/control_surface_gate_geometry.json`

Builder:

> `code/control_surface_gate_geometry.py`

Commands:

```powershell
python code\control_surface_gate_geometry.py --write
python code\control_surface_gate_geometry.py
python code\validate_control_surface_atlas.py
```

## Purpose

The gate-geometry layer makes the shape of the mechanism-card bar a
first-class artifact. A failed control is not just a tombstone; it is
evidence about where current behavior families stop under the current
controls.

The ordered funnel is:

1. behavior substrate;
2. prompt-channel locality;
3. signature not explained by output/candidate geometry;
4. monitor-to-lever transition;
5. predicted intervention;
6. reliability, nulls, locality, side effects, and transfer;
7. promoted mechanism.

## Generated Facts

- atlas rows: 19;
- bridge rungs: 24;
- promoted mechanisms: 0;
- bridge rungs allowing hidden-state work: 0;
- clean unconfounded bridge rungs: 0;
- MC028 other-number rows in taxonomy: 28.

## Row Funnel

| Terminal Stage | Count | Ratio | Meaning |
| --- | ---: | ---: | --- |
| `pre_signature_behavior_substrate` | 11 | 0.579 | The behavior table, bridge substrate, nulls, parseability, or conflict mixture failed before hidden-state work is allowed. |
| `pre_signature_prompt_channel_locality` | 1 | 0.053 | The behavior contrast exists, but a visible prompt channel carries the rule, so hidden-state mechanism work is not licensed. |
| `signature_output_geometry_shadow` | 3 | 0.158 | The behavior table exists, but the hidden signal is output/candidate geometry or zero-lead shadow under current controls. |
| `signature_monitor_no_lever` | 2 | 0.105 | A predecision or monitor-like signal exists, but it has no clean causal lever or is still candidate/output confounded. |
| `intervention_failed` | 1 | 0.053 | A plausible signal or control route reached intervention work, but the intervention failed controls, locality, or causal criteria. |
| `reliability_null_boundary` | 1 | 0.053 | The internal causal surface is bounded by null, locality, side-effect, or transfer fragility. |
| `promoted_mechanism` | 0 | 0.000 | The surface clears signature, intervention, and reliability gates. |

## Ordered Buckets

| Bucket | Count | Ratio |
| --- | ---: | ---: |
| `pre_signature_blocked` | 12 | 0.632 |
| `signature_stage_blocked` | 5 | 0.263 |
| `intervention_stage_blocked` | 1 | 0.053 |
| `reliability_stage_bounded` | 1 | 0.053 |
| `promoted` | 0 | 0.000 |

## Row Ledger

| Row | Terminal Stage | Route | Frontier | Action |
| --- | --- | --- | --- | --- |
| `mc001_qwen3_0p6b_truth_agreement` | `signature_output_geometry_shadow` | `output_shadow_diagnostic_baseline` | `output_visible_at_or_before_frontier` | Treat hidden AUCs as dashboard readings until output/candidate geometry is matched or beaten. |
| `mc001b_qwen3_1p7b_truth_agreement` | `signature_output_geometry_shadow` | `output_shadow_diagnostic_baseline` | `output_visible_at_or_before_frontier` | Treat hidden AUCs as dashboard readings until output/candidate geometry is matched or beaten. |
| `mc001g_gemma_truth_agreement` | `intervention_failed` | `failed_intervention_or_mechanism_route` | `output_visible_at_or_before_frontier` | Kill or redesign the intervention family; do not promote behavior control as mechanism control. |
| `mc002_known_unknown` | `pre_signature_behavior_substrate` | `closed_before_hidden_state` | `frontier_not_reached` | Do not probe hidden states; rebuild the behavior table or record a diagnostic failure. |
| `mc002b_context_support` | `pre_signature_behavior_substrate` | `closed_before_hidden_state` | `frontier_not_reached` | Do not probe hidden states; rebuild the behavior table or record a diagnostic failure. |
| `mc003_delayed_copy` | `signature_output_geometry_shadow` | `output_shadow_diagnostic_baseline` | `output_visible_at_or_before_frontier` | Treat hidden AUCs as dashboard readings until output/candidate geometry is matched or beaten. |
| `mc004_in_context_binding` | `signature_monitor_no_lever` | `monitor_only_conditional_revisit` | `predecision_monitor_no_lever` | Preserve as monitor-only evidence until an intervention changes behavior without collateral damage. |
| `mc005_associative_lookup` | `reliability_null_boundary` | `bounded_mechanism_frozen` | `causal_surface_not_timing_frontier` | Freeze as bounded reference unless a materially different route repairs nulls, locality, robustness, and transfer. |
| `mc006_parametric_fact_override` | `signature_monitor_no_lever` | `monitor_only_closed` | `predecision_monitor_no_lever` | Preserve as monitor-only evidence until an intervention changes behavior without collateral damage. |
| `mc007_semi_synthetic_familiar_entity_lookup` | `pre_signature_behavior_substrate` | `closed_before_hidden_state` | `frontier_not_reached` | Do not probe hidden states; rebuild the behavior table or record a diagnostic failure. |
| `mc008_symbolic_fact_code_arbitration` | `pre_signature_behavior_substrate` | `closed_before_hidden_state` | `frontier_not_reached` | Do not probe hidden states; rebuild the behavior table or record a diagnostic failure. |
| `mc009_derived_code_arbitration` | `pre_signature_behavior_substrate` | `closed_before_hidden_state` | `frontier_not_reached` | Do not probe hidden states; rebuild the behavior table or record a diagnostic failure. |
| `mc010_two_hop_fact_code_arbitration` | `pre_signature_behavior_substrate` | `closed_before_hidden_state` | `frontier_not_reached` | Do not probe hidden states; rebuild the behavior table or record a diagnostic failure. |
| `mc011_atomic_number_code_arbitration` | `pre_signature_behavior_substrate` | `closed_before_hidden_state` | `frontier_not_reached` | Do not probe hidden states; rebuild the behavior table or record a diagnostic failure. |
| `mc012_reliability_labeled_numeric_arbitration` | `pre_signature_prompt_channel_locality` | `prompt_visible_positive_control` | `frontier_not_reached` | Use as positive control only; remove or match the visible prompt channel before claiming a hidden surface. |
| `mc013_status_channel_ablation_numeric_arbitration` | `pre_signature_behavior_substrate` | `closed_before_hidden_state` | `frontier_not_reached` | Do not probe hidden states; rebuild the behavior table or record a diagnostic failure. |
| `mc014_inferred_reliability_numeric_arbitration` | `pre_signature_behavior_substrate` | `closed_before_hidden_state` | `frontier_not_reached` | Do not probe hidden states; rebuild the behavior table or record a diagnostic failure. |
| `mc015_parity_gated_numeric_arbitration` | `pre_signature_behavior_substrate` | `closed_before_hidden_state` | `frontier_not_reached` | Do not probe hidden states; rebuild the behavior table or record a diagnostic failure. |
| `mc016_alphabet_gated_numeric_arbitration` | `pre_signature_behavior_substrate` | `closed_before_hidden_state` | `frontier_not_reached` | Do not probe hidden states; rebuild the behavior table or record a diagnostic failure. |

## Bridge Funnel

| Terminal Stage | Count | Ratio |
| --- | ---: | ---: |
| `pre_signature_behavior_substrate` | 23 | 0.958 |
| `pre_signature_prompt_channel_locality` | 1 | 0.042 |

## Bridge Ledger

| Card | Terminal Stage | Mixture | Dominant Failure |
| --- | --- | --- | --- |
| `MC010` | `pre_signature_behavior_substrate` | `absent` | synthetic and learned controls plus conflict contrast failed |
| `MC011` | `pre_signature_behavior_substrate` | `absent` | numeric conflict collapsed to prompt-local values |
| `MC012` | `pre_signature_prompt_channel_locality` | `clean_prompt_visible` | prompt-channel locality failed |
| `MC013` | `pre_signature_behavior_substrate` | `statused_only` | matched ablation collapsed to prompt-local values |
| `MC014` | `pre_signature_behavior_substrate` | `absent` | calibration-consistent and inconsistent rows both local |
| `MC015` | `pre_signature_behavior_substrate` | `mixed_wrong_rule` | rule-aligned expected correctness failed |
| `MC016` | `pre_signature_behavior_substrate` | `absent` | expected-atomic rows selected local |
| `MC017` | `pre_signature_behavior_substrate` | `absent` | ATOMIC selector control collapsed to LOCAL |
| `MC018` | `pre_signature_behavior_substrate` | `mixed_wrong_rule` | source-rule correctness near chance with order bias |
| `MC019` | `pre_signature_behavior_substrate` | `mixed_wrong_rule` | expected-atomic route weak despite clean nulls |
| `MC020` | `pre_signature_behavior_substrate` | `asymmetric_route_failure` | route-atomic branch collapsed toward local |
| `MC021` | `pre_signature_behavior_substrate` | `asymmetric_route_failure` | visible-visible route weak, learned route weaker |
| `MC022` | `pre_signature_behavior_substrate` | `asymmetric_route_failure` | local-source salience and rule-order sensitivity remain |
| `MC023` | `pre_signature_behavior_substrate` | `mixed_below_gate` | operation-local and operation-atomic conflict branches below gate |
| `MC024` | `pre_signature_behavior_substrate` | `asymmetric_route_failure` | operation-atomic branch below gate with other-number leakage |
| `MC025` | `pre_signature_behavior_substrate` | `choice_interface_confounded` | direct atomic control and answer-absent null failed under choice interface |
| `MC026` | `pre_signature_behavior_substrate` | `numeric_option_confounded` | direct atomic control abstains and operation-atomic branch remains below gate |
| `MC027` | `pre_signature_behavior_substrate` | `interface_dependent_smoke_candidate` | answer interfaces change behavior strongly and only the inherited bare-integer route survives smoke |
| `MC028` | `pre_signature_behavior_substrate` | `full_source_boundary_failed` | full-source learned atomic branch below gate with other-number leakage |
| `MC029` | `pre_signature_behavior_substrate` | `factorial_branch_null_tradeoff` | branch/null tradeoff after worked-example removal |
| `MC030` | `pre_signature_behavior_substrate` | `guarded_branch_null_tradeoff` | branch/null tradeoff persists under absence guards |
| `MC031` | `pre_signature_behavior_substrate` | `statusless_reliability_local_collapse` | statusless invalid-source branch collapses to local despite available atomic recall |
| `MC032` | `pre_signature_behavior_substrate` | `statusless_reliability_local_collapse` | statusless cross-table mismatch collapses toward the primary local number |
| `MC033` | `pre_signature_behavior_substrate` | `statusless_reliability_local_and_claim_leak` | fact-claim validity does not produce stable local-versus-learned routing |

## Interpretation

The current project does not merely have many failed mechanism-card
routes. It has a measured distribution of where those routes fail.
That distribution is now part of the genome map.

The dominant fact is pre-signature closure. Twelve of 19 atlas rows
stop before hidden-state mechanism work is licensed: 11 at behavior
or bridge-substrate quality and MC012 at visible prompt-channel
locality. Five rows stop at the signature stage: three are
output-geometry shadows and two are monitor-only no-lever rows.
One route has a failed intervention, and MC005 is the only bounded
internal-causal reliability specimen.

This is the concrete answer to asymptotic conservatism: the bar is
not merely getting stricter in prose. Its failure geometry is now
measured, versioned, and validated.

## What This Proves

It proves that the current atlas has an ordered claim-bar geometry
derived from validated route, reliability, decision-frontier, bridge,
and taxonomy artifacts.

It proves that killed controls are structured data: most deaths are
pre-signature, the signature-stage residue is output-shadow or
monitor-only, intervention failure is currently one row, and the
sole internal-causal result is bounded rather than promoted.

## What It Does Not Prove

It does not prove that the current bar is final, optimal, or fair for
every behavior family. It documents the bar that current claims have
actually faced.

It does not prove that a route killed under one prompt contract is
impossible under a materially different contract. It does require
that such a route be reopened explicitly rather than smuggled back in
as a near-duplicate repair.

## Next Use

After each future result, update this layer and ask:

> Did the new result move a row to a later terminal stage, or did it
> only add another example to an already measured death bucket?

That question keeps the project aimed at the genome law rather than
at isolated attractive results.
