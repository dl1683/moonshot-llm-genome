# Control-Surface Knowledge Gap Plan

Source updated_at: 2026-07-01

Status: generated missing-evidence plan implemented and validated.

Machine-readable artifact:

> `data/control_surface_knowledge_gap_plan.json`

Builder:

> `code/control_surface_knowledge_gap_plan.py`

Commands:

```powershell
python code\control_surface_knowledge_gap_plan.py --write
python code\control_surface_knowledge_gap_plan.py
python code\validate_control_surface_atlas.py
```

## Purpose

This layer converts the knowledge ladder into explicit missing-evidence
decisions. It treats typed failures as evidence about where the behavior
currently lives instead of treating them as failed prose claims.

## Generated Facts

- levels: 5;
- missing-evidence items: 15;
- linked existing work orders: 4;
- levels needing new behavior substrates: 3;
- levels needing predecision/reliability closure: 2;
- levels requiring future work orders: 3;
- new hidden-state search allowed levels: 0;
- same-family route-killed levels: 2;
- promoted levels: 0;
- bounded-reference levels: 1;
- monitor-only levels: 1;
- real uncertainty mechanism-ready levels: 0.

## Level Plan

| Level | Status | Decision State | Missing Items | Existing Work Orders | New Work Order? |
| --- | --- | --- | ---: | --- | --- |
| Pure synthetic lookup | `bounded_reference` | `bounded_reference_needs_reliability_and_width` | 3 | `close_mc005_reference_specimen`, `run_width_transfer_probe` | `False` |
| Semi-synthetic familiar entities | `behavior_substrate_blocked` | `material_new_substrate_required` | 3 | `close_post_mc033_bridge_substrate_family` | `True` |
| Symbolic / learned-memory bridge | `prompt_visible_or_behavior_blocked` | `same_family_bridge_route_killed` | 3 | `close_post_mc033_bridge_substrate_family` | `True` |
| Strong parametric fact override | `monitor_only_no_lever` | `monitor_only_predecision_frontier` | 3 | `close_mc006_predecision_frontier` | `False` |
| Real abstention / uncertainty | `behavior_substrate_blocked` | `real_uncertainty_behavior_substrate_absent` | 3 | none | `True` |

## Pure synthetic lookup

- level id: `level_1_synthetic_lookup`;
- row ids: `["mc005_associative_lookup"]`;
- decision state: `bounded_reference_needs_reliability_and_width`;
- next allowed action: Use the MC005 reference closeout and transfer-width probe. Do not keep adding same-route repairs unless the preregistered repair budget changes the reliability boundary.
- insight: The best current mechanism-like object is also the best calibration object for how reliability boundaries break.

Missing evidence:
- `answer_absent_null_boundary` (reliability): Repair or explicitly bound the answer-absent low-margin null flips without weakening the high-margin lookup mediation claim.
- `transfer_panel` (widening): Measure whether primary mediation, null locality, side effects, and prompt robustness transfer together beyond the current Qwen-dominant setting.
- `intervention_family_verdict` (route_disposition): Decide whether write replacement is promoted, bounded, or killed after a fixed repair budget.

Decision rules:
- `promotion_rule`: Promote only if high-margin lookup mediation stays strong and the answer-absent null boundary closes under locality, side-row, fluency, and holdout panels.
- `death_rule`: Kill the write-replacement route if null flips or collateral side effects persist across the fixed repair variants.
- `containment_rule`: If the primary effect survives but nulls remain fragile, keep MC005 as a bounded synthetic lookup reference, not a general knowledge mechanism.
- `export_rule`: Export NULL_ROW_LOW_MARGIN_FLIP, TRANSFER_PRIMARY_BEFORE_RELIABILITY, or TRANSFER_PRIMARY_FAILED as reusable diagnostics.

Forbidden moves:
- Do not describe MC005 as full reliability while answer-absent null flips remain.
- Do not call a primary-effect replication transfer unless null locality and side-effect panels also transfer.
- Do not broaden the claim to knowledge control.

Linked work orders:
- `close_mc005_reference_specimen` (deepening_closeout, immediate): Force a verdict on the MC005 internal-causal reference specimen.
- `run_width_transfer_probe` (widening_probe, immediate): Make transfer failure or success measured instead of assumed.

## Semi-synthetic familiar entities

- level id: `level_2_semi_synthetic_familiar_entity`;
- row ids: `["mc007_semi_synthetic_familiar_entity_lookup"]`;
- decision state: `material_new_substrate_required`;
- next allowed action: Treat MC007 as blocked under the current substrate. A future attempt needs a materially new familiar-entity contract before any signature search.
- insight: This level is where semantic familiarity first enters, and current evidence says the behavior contract fails before mechanism work.

Missing evidence:
- `clean_familiar_entity_behavior_substrate` (behavior): A familiar-entity task where local artificial values are behaviorally stable before hidden-state work.
- `prompt_authority_controls` (control): Prompt-authority, format, source-disjoint, null, and output/candidate baselines that do not explain the behavior.
- `semantic_prior_interference_panel` (robustness): A panel that measures whether familiar entity priors help, conflict with, or swamp task-local lookup.

Decision rules:
- `promotion_rule`: Promote this level only when a familiar-entity behavior substrate passes direct, null, prompt-channel, holdout, and output/candidate gates before probing.
- `death_rule`: Kill same-family MC007-style repairs if the behavior remains parse-, authority-, or prompt-contract-dominated.
- `containment_rule`: Until then, the level only says familiar entities are a bridge stressor, not a localizable control surface.
- `export_rule`: Export PROMPT_AUTHORITY_CONFUND or BEHAVIOR_TABLE_CONDITION_CONFOUND if the next substrate fails for visible reasons.

Forbidden moves:
- Do not run hidden-state probes on another MC007-style behavior failure.
- Do not treat real entity names as knowledge evidence when the artificial value contract is unstable.
- Do not reuse the same authority-dial route as a new substrate.

Linked work orders:
- `close_post_mc033_bridge_substrate_family` (bridge_substrate, immediate): Close the post-MC033 bridge substrate family before any hidden-state work.

## Symbolic / learned-memory bridge

- level id: `level_3_symbolic_or_learned_memory_bridge`;
- row ids: `["mc008_symbolic_fact_code_arbitration", "mc009_derived_code_arbitration", "mc010_two_hop_fact_code_arbitration", "mc011_atomic_number_code_arbitration", "mc012_reliability_labeled_numeric_arbitration", "mc013_status_channel_ablation_numeric_arbitration", "mc014_inferred_reliability_numeric_arbitration", "mc015_parity_gated_numeric_arbitration", "mc016_alphabet_gated_numeric_arbitration"]`;
- decision state: `same_family_bridge_route_killed`;
- next allowed action: Close the same-family bridge route after MC033. A future bridge must be a new substrate class, not another source-validity cue.
- insight: The bridge ladder is not an embarrassing absence of mechanisms; it is a map of which bridge substrates die under controls.

Missing evidence:
- `clean_unconfounded_bridge_substrate` (behavior): A bridge substrate outside MC007-MC033 that passes branch, null, local, side-number, parseability, prompt-channel, source-disjoint, and output/candidate controls.
- `hidden_state_permission_gate` (permission): A documented gate allowing hidden-state work only after the bridge behavior substrate is clean.
- `learned_local_arbitration_stability` (robustness): Evidence that learned/local arbitration remains stable across conflict mixtures instead of collapsing to prompt-visible or output-visible cues.

Decision rules:
- `promotion_rule`: Promote a bridge level to hidden-state work only if a materially new substrate clears all behavior, null, locality, holdout, and output/candidate gates together.
- `death_rule`: Keep the MC007-MC033 same-family route killed unless a new proposal changes the substrate class, not merely the cue wording.
- `containment_rule`: The current bridge evidence supports a substrate-failure taxonomy, not a symbolic or learned-memory mechanism.
- `export_rule`: Export POST_MC032_BRIDGE_ROUTE_CLOSED, FACT_CLAIM_MISMATCH_LOCAL_AND_CLAIM_LEAK, and STATUSLESS_SOURCE_VALIDITY_LOCAL_DOMINANCE.

Forbidden moves:
- Do not start probes on MC031-MC033-style rows.
- Do not reopen status labels, row codes, operation handles, worked examples, answer schemas, absence guards, checksum, consistency, or row-local fact-claim cues.
- Do not call a prompt-visible positive control a knowledge mechanism.

Linked work orders:
- `close_post_mc033_bridge_substrate_family` (bridge_substrate, immediate): Close the post-MC033 bridge substrate family before any hidden-state work.

## Strong parametric fact override

- level id: `level_4_parametric_fact_override`;
- row ids: `["mc006_parametric_fact_override"]`;
- decision state: `monitor_only_predecision_frontier`;
- next allowed action: Use the MC006 predecision frontier closeout. Probe earlier positions only under same-stage and final-margin controls; do not steer final-position high-AUC signatures.
- insight: MC006 is valuable because it measures where knowledge-like behavior becomes output-visible, even without a usable lever.

Missing evidence:
- `candidate_output_decoupled_signature` (signature): An earlier signal that survives same-stage output geometry, final candidate margin, shuffle, and source-disjoint holdout controls.
- `causal_predecision_lever` (intervention): A steering or editing route that changes generated capital-fact override behavior before answer commitment without broad collateral effects.
- `decision_timing_boundary` (lead_time): A measured boundary for whether MC006 has usable lead time or is already output-visible by the final decision state.

Decision rules:
- `promotion_rule`: Promote to intervention only if an earlier source/path signal predicts on holdout while beating same-stage output geometry and final candidate margin.
- `death_rule`: Kill final-token MC006 probing if another margin-matched pass is explained by candidate score or final output margin.
- `containment_rule`: If the route fails, keep MC006 as a decision-timing and output-visibility result, not a truth or knowledge vector.
- `export_rule`: Export FINAL_STATE_OUTPUT_VISIBLE or PREDECISION_MONITOR_NO_LEVER depending on where the controlled failure lands.

Forbidden moves:
- Do not steer V15/V16-style final-token perfect AUCs.
- Do not treat final margin as a nuisance if it is the faithful downstream decision state.
- Do not use lonely AUCs without same-stage and final-stage baselines.

Linked work orders:
- `close_mc006_predecision_frontier` (knowledge_frontier_closeout, high): Turn MC006 final-margin failure into a decision-timing result.

## Real abstention / uncertainty

- level id: `level_5_real_abstention_uncertainty`;
- row ids: `["mc002_known_unknown", "mc002b_context_support"]`;
- decision state: `real_uncertainty_behavior_substrate_absent`;
- next allowed action: Open a new behavior-family work order only after the behavior table is clean. Hidden-state work is not licensed for the current MC002/MC002B substrates.
- insight: The real safety-relevant level is still below the behavior gate; that is a result about the current map, not a reason to pretend the ladder is higher.

Missing evidence:
- `real_uncertainty_behavior_table` (behavior): A balanced generated-answer table for factual correction, refusal, or uncertainty behavior that passes prompt, label, and output-margin controls.
- `abstention_null_and_side_effect_gates` (reliability): Null rows, side-effect checks, fluency, locality, and holdouts that distinguish abstention from formatting or refusal-template artifacts.
- `known_unknown_label_grounding` (label_quality): A label source for known, unknown, corrected, unsupported, and abstain cases that does not leak through prompt text or answer interface.

Decision rules:
- `promotion_rule`: Promote this level to signature search only after real uncertainty behavior passes behavior, prompt, output, label-balance, null, and holdout gates.
- `death_rule`: Kill any uncertainty route whose labels or refusal behavior are explained by prompt wording, answer schema, or output margin.
- `containment_rule`: Until then, this level remains a future target and cannot support factual correction, refusal, or uncertainty mechanism claims.
- `export_rule`: Export BEHAVIOR_TABLE_CONDITION_CONFOUND, REQUESTED_MODE_CONFOUND, or OUTPUT_MARGIN_CONFUND if the substrate fails.

Forbidden moves:
- Do not probe hidden states on MC002/MC002B as if they were mechanism-ready.
- Do not collapse factual correction, refusal, and uncertainty into one label before the behavior table passes.
- Do not publish a real-world abstention control claim from synthetic or prompt-visible behavior.

## What This Proves

It proves that the knowledge ambition has been converted into a
level-by-level evidence ledger. The current map has one bounded
synthetic reference, one monitor-only parametric-fact frontier,
three levels that need new behavior substrates or fresh work orders,
and zero promoted knowledge mechanisms.

## What It Does Not Prove

It does not prove a broad knowledge mechanism, a truth vector, a
reliable steering route for MC006, or real-world factual correction
or abstention control.
