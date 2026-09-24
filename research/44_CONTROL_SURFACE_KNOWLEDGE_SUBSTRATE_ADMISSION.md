# Control-Surface Knowledge Substrate Admission

Source updated_at: 2026-07-01

Status: generated admission protocol implemented and validated.

Machine-readable artifact:

> `data/control_surface_knowledge_substrate_admission.json`

Builder:

> `code/control_surface_knowledge_substrate_admission.py`

Commands:

```powershell
python code\control_surface_knowledge_substrate_admission.py --write
python code\control_surface_knowledge_substrate_admission.py
python code\validate_control_surface_atlas.py
```

## Purpose

This protocol is the front door for future knowledge-family behavior
substrates. It is deliberately prior to hidden-state work. A candidate
substrate has to pass these gates before signature, steering, editing,
or surgery is licensed.

## Generated Facts

- admission packets: 3;
- admission gates per packet: 11;
- total gate entries: 33;
- levels requiring new behavior substrate: 3;
- future-work-order levels: 3;
- new hidden-state search allowed levels: 0;
- same-family route-killed levels: 2;
- bridge rungs: 24;
- bridge hidden-state-allowed rungs: 0;
- bridge clean unconfounded rungs: 0;
- smoke cards: 17;
- smoke hidden-state-allowed cards: 0;
- pre-signature-blocked atlas rows: 12;
- promoted mechanisms: 0.

## Admission Packets

| Level | Admission Class | Current Status | Hidden-State License |
| --- | --- | --- | --- |
| Semi-synthetic familiar entities | `new_familiar_entity_substrate` | `behavior_substrate_blocked` | `forbidden_until_all_admission_gates_pass` |
| Symbolic / learned-memory bridge | `new_bridge_substrate_class` | `prompt_visible_or_behavior_blocked` | `forbidden_until_all_admission_gates_pass` |
| Real abstention / uncertainty | `new_real_uncertainty_substrate` | `behavior_substrate_blocked` | `forbidden_until_all_admission_gates_pass` |

## Gate Order

1. `material_novelty`
2. `behavior_contract`
3. `parseability_and_label_balance`
4. `direct_controls`
5. `conflict_mixture`
6. `null_rows`
7. `source_disjoint_holdout`
8. `prompt_channel_locality`
9. `output_candidate_baselines`
10. `side_effect_and_leakage`
11. `split_freeze`

## Semi-synthetic familiar entities

- level id: `level_2_semi_synthetic_familiar_entity`;
- row ids: `["mc007_semi_synthetic_familiar_entity_lookup"]`;
- admission decision: No hidden-state work until a fresh familiar-entity table passes all admission gates.
- candidate substrate: Familiar entity names with artificial task-local values under a contract that separates source lookup from semantic prior pressure.
- material novelty: Must change the current MC007 route by changing the behavior family or prompt contract, not just another authority dial or parser repair.

Decision rules:
- `promotion_rule`: Admit to signature search only if source-local artificial values and semantic-prior controls both pass on source-disjoint holdout while prompt-authority and output/candidate baselines fail to explain the behavior.
- `death_rule`: Kill the substrate if behavior still depends on authority wording, parser choices, or semantic-prior leakage after one preregistered repair.
- `containment_rule`: If only source-local lookup works, classify the level as a source-value diagnostic, not a familiar-entity knowledge bridge.
- `export_rule`: Export PROMPT_AUTHORITY_CONFUND, SEMANTIC_PRIOR_INTERFERENCE, or BEHAVIOR_TABLE_CONDITION_CONFOUND.

Known failure modes:
- `PROMPT_AUTHORITY_CONFUND`
- `PROMPT_CONTRACT_PARSEABILITY`
- `BEHAVIOR_TABLE_CONDITION_CONFOUND`
- `semantic prior swamps artificial values`

Admission gates:
- `material_novelty`: Show that the proposal is materially outside MC007 V1-V4 ordinary prompt/authority/parser repairs.
- `behavior_contract`: Predeclare the exact prompt contract, answer interface, labels, parser, and allowed claim before any hidden-state collection.
- `parseability_and_label_balance`: Show parseability, label balance, and split balance on discovery, calibration, and holdout rows.
- `direct_controls`: Pass direct controls that isolate ordinary task competence from the claimed arbitration behavior.
- `conflict_mixture`: Produce a nontrivial conflict mixture in the primary behavior instead of all-local, all-learned, all-null, or all-format behavior.
- `null_rows`: Pass answer-absent, irrelevant-source, lure, or unsupported-context null rows appropriate to the level.
- `source_disjoint_holdout`: Preserve the behavior on source-disjoint holdout sources, not just template-disjoint or row-disjoint variants.
- `prompt_channel_locality`: Remove, match, or ablate visible prompt channels that can explain the label or requested mode.
- `output_candidate_baselines`: Report output margin, candidate-score margin, next-token margin, and simple prompt baselines beside any future signature.
- `side_effect_and_leakage`: Audit side answers, other-number leakage, fluency, answer shape, and unrelated rows before intervention work.
- `split_freeze`: Freeze discovery, calibration, holdout, null, and side-effect panels before hidden-state discovery.

## Symbolic / learned-memory bridge

- level id: `level_3_symbolic_or_learned_memory_bridge`;
- row ids: `["mc008_symbolic_fact_code_arbitration", "mc009_derived_code_arbitration", "mc010_two_hop_fact_code_arbitration", "mc011_atomic_number_code_arbitration", "mc012_reliability_labeled_numeric_arbitration", "mc013_status_channel_ablation_numeric_arbitration", "mc014_inferred_reliability_numeric_arbitration", "mc015_parity_gated_numeric_arbitration", "mc016_alphabet_gated_numeric_arbitration"]`;
- admission decision: Same-family bridge route remains killed; only a new substrate class can request admission.
- candidate substrate: A local-versus-learned arbitration task outside MC007-MC033 that creates stable branch behavior without visible status labels or answer-interface shortcuts.
- material novelty: Must be materially outside MC007-MC033: source labels, row codes, operation handles, worked examples, answer schemas, absence guards, checksum cues, cross-table consistency, and row-local fact claims.

Decision rules:
- `promotion_rule`: Admit to hidden-state search only if branch, null, local, side-number, parseability, prompt-channel, source-disjoint, and output/candidate controls pass together.
- `death_rule`: Kill the proposed bridge if it repeats a closed MC007-MC033 failure class or passes only as a prompt-visible positive control.
- `containment_rule`: If it fails, add it to the bridge failure taxonomy; do not call the failure a mechanism absence in learned facts generally.
- `export_rule`: Export the typed bridge failure into the bridge ladder, smoke diagnostics, error taxonomy, and gap plan.

Known failure modes:
- `POST_MC032_BRIDGE_ROUTE_CLOSED`
- `LOCAL_SOURCE_SALIENCE`
- `FACT_CLAIM_MISMATCH_LOCAL_AND_CLAIM_LEAK`
- `STATUSLESS_SOURCE_VALIDITY_LOCAL_DOMINANCE`
- `prompt-visible positive control`

Admission gates:
- `material_novelty`: Demonstrate the proposal is outside the MC007-MC033 bridge route family and not just a new source-validity cue.
- `behavior_contract`: Predeclare the exact prompt contract, answer interface, labels, parser, and allowed claim before any hidden-state collection.
- `parseability_and_label_balance`: Show parseability, label balance, and split balance on discovery, calibration, and holdout rows.
- `direct_controls`: Pass direct controls that isolate ordinary task competence from the claimed arbitration behavior.
- `conflict_mixture`: Produce a nontrivial conflict mixture in the primary behavior instead of all-local, all-learned, all-null, or all-format behavior.
- `null_rows`: Pass answer-absent, irrelevant-source, lure, or unsupported-context null rows appropriate to the level.
- `source_disjoint_holdout`: Preserve the behavior on source-disjoint holdout sources, not just template-disjoint or row-disjoint variants.
- `prompt_channel_locality`: Remove, match, or ablate visible prompt channels that can explain the label or requested mode.
- `output_candidate_baselines`: Report output margin, candidate-score margin, next-token margin, and simple prompt baselines beside any future signature.
- `side_effect_and_leakage`: Audit side answers, other-number leakage, fluency, answer shape, and unrelated rows before intervention work.
- `split_freeze`: Freeze discovery, calibration, holdout, null, and side-effect panels before hidden-state discovery.

## Real abstention / uncertainty

- level id: `level_5_real_abstention_uncertainty`;
- row ids: `["mc002_known_unknown", "mc002b_context_support"]`;
- admission decision: No existing real-uncertainty substrate is admissible; a new behavior family and work order are required.
- candidate substrate: Generated-answer factual correction, refusal, abstention, or uncertainty behavior with grounded known/unknown/support labels and no refusal-template leakage.
- material novelty: Must change task construction beyond MC002 and MC002B pressure prompts; labels and abstention behavior must be grounded before prompting.

Decision rules:
- `promotion_rule`: Admit to signature search only after generated factual correction/refusal/uncertainty behavior passes prompt, output, label-balance, null, and source-disjoint holdout gates.
- `death_rule`: Kill the substrate if abstention, refusal, or correction labels are explained by prompt wording, answer schema, output margin, or label leakage.
- `containment_rule`: If only a prompt-visible refusal behavior appears, classify it as requested-mode behavior control, not uncertainty control.
- `export_rule`: Export BEHAVIOR_TABLE_CONDITION_CONFOUND, REQUESTED_MODE_CONFOUND, OUTPUT_MARGIN_CONFUND, or LABEL_GROUNDING_FAILURE.

Known failure modes:
- `REQUESTED_MODE_CONFOUND`
- `OUTPUT_MARGIN_CONFUND`
- `BEHAVIOR_TABLE_CONDITION_CONFOUND`
- `refusal-template leakage`
- `known/unknown label leakage`

Admission gates:
- `material_novelty`: Show that the proposal is not another MC002/MC002B prompt-pressure repair.
- `behavior_contract`: Predeclare the exact prompt contract, answer interface, labels, parser, and allowed claim before any hidden-state collection.
- `parseability_and_label_balance`: Show parseability, label balance, and split balance on discovery, calibration, and holdout rows.
- `direct_controls`: Pass direct controls that isolate ordinary task competence from the claimed arbitration behavior.
- `conflict_mixture`: Produce a nontrivial conflict mixture in the primary behavior instead of all-local, all-learned, all-null, or all-format behavior.
- `null_rows`: Pass answer-absent, irrelevant-source, lure, or unsupported-context null rows appropriate to the level.
- `source_disjoint_holdout`: Preserve the behavior on source-disjoint holdout sources, not just template-disjoint or row-disjoint variants.
- `prompt_channel_locality`: Remove, match, or ablate visible prompt channels that can explain the label or requested mode.
- `output_candidate_baselines`: Report output margin, candidate-score margin, next-token margin, and simple prompt baselines beside any future signature.
- `side_effect_and_leakage`: Audit side answers, other-number leakage, fluency, answer shape, and unrelated rows before intervention work.
- `split_freeze`: Freeze discovery, calibration, holdout, null, and side-effect panels before hidden-state discovery.

## What This Proves

It proves that the project has a generated front-door rule for
future knowledge substrate proposals. The rule is tied to the
current gap plan, bridge ladder, smoke diagnostics, and gate geometry
rather than to review prose.

## What It Does Not Prove

It does not prove that a new substrate exists. It does not permit
hidden-state search. It does not upgrade any existing knowledge
level to a mechanism claim.
