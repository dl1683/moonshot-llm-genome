# Control-Surface Knowledge First-Run Pack

Source updated_at: 2026-07-01

Status: generated behavior-substrate first-run pack implemented and validated.

Machine-readable artifact:

> `data/control_surface_knowledge_first_run_pack.json`

Builder:

> `code/control_surface_knowledge_first_run_pack.py`

Commands:

```powershell
python code\control_surface_knowledge_first_run_pack.py --write
python code\control_surface_knowledge_first_run_pack.py
python code\validate_control_surface_atlas.py
```

## Purpose

This pack converts the knowledge-candidate queue into run-ready
behavior-substrate preregistrations. It deliberately stops before
hidden-state signatures. The output of a first run can only be a
behavior-substrate status card or a diagnostic note.

## Generated Facts

- first-run packets: 6;
- total panels: 37;
- total baselines: 24;
- total admission-gate bindings: 66;
- hidden-state packets: 0;
- primary model: `Qwen/Qwen3-1.7B`;
- secondary width-check model: `google/gemma-2-2b-it`;
- promoted mechanisms: 0.

## Packet Table

| Packet | Candidate | Level | Priority | Panels | Hidden-State License |
| --- | --- | --- | --- | ---: | --- |
| `ksq001_familiar_entity_prior_counterbalance_first_run_v1` | `ksq001_familiar_entity_prior_counterbalance` | `level_2_semi_synthetic_familiar_entity` | `high` | 5 | `forbidden_until_first_run_passes_admission` |
| `ksq003_bridge_statusless_evidence_aggregation_first_run_v1` | `ksq003_bridge_statusless_evidence_aggregation` | `level_3_symbolic_or_learned_memory_bridge` | `immediate` | 7 | `forbidden_until_first_run_passes_admission` |
| `ksq005_uncertainty_grounded_answerability_first_run_v1` | `ksq005_uncertainty_grounded_answerability` | `level_5_real_abstention_uncertainty` | `high` | 6 | `forbidden_until_first_run_passes_admission` |
| `ksq002_familiar_entity_source_rewrite_equivalence_first_run_v1` | `ksq002_familiar_entity_source_rewrite_equivalence` | `level_2_semi_synthetic_familiar_entity` | `medium` | 6 | `forbidden_until_first_run_passes_admission` |
| `ksq004_bridge_answer_interface_minimal_pairs_first_run_v1` | `ksq004_bridge_answer_interface_minimal_pairs` | `level_3_symbolic_or_learned_memory_bridge` | `high` | 6 | `forbidden_until_first_run_passes_admission` |
| `ksq006_uncertainty_context_support_counterfactuals_first_run_v1` | `ksq006_uncertainty_context_support_counterfactuals` | `level_5_real_abstention_uncertainty` | `medium` | 7 | `forbidden_until_first_run_passes_admission` |

## Counterbalance familiar-entity priors against artificial values.

- first run id: `ksq001_familiar_entity_prior_counterbalance_first_run_v1`;
- candidate id: `ksq001_familiar_entity_prior_counterbalance`;
- run scope: `behavior_substrate_admission_only`;
- primary model: `Qwen/Qwen3-1.7B`;
- secondary model: `google/gemma-2-2b-it`;
- artifact paths: `{"prereg": "research/prereg/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE_FIRST_RUN.md", "result_json": "results/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE/ksq001_familiar_entity_prior_counterbalance_first_run.json", "runner": "code/ksq001_familiar_entity_prior_counterbalance_first_run.py", "status_card": "research/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE_FIRST_RUN_STATUS.md"}`;

Freeze before run:
- prompt templates
- parser/scorer
- row generation seed
- source-disjoint split manifest
- panel thresholds
- baseline interfaces
- allowed and forbidden claims

Panels:
- `source_local_artificial_lookup` (direct_control, n >= 40): Pass if parseable_rate >= 0.90 and artificial_value_rate >= 0.85 on non-holdout and >= 0.75 on source-disjoint holdout.
- `semantic_prior_lure` (semantic_prior_control, n >= 40): Pass as an admission signal only if both branches appear in matched conflict rows and neither branch exceeds 0.90 globally.
- `answer_absent_irrelevant_nulls` (null, n >= 30): Pass if unknown_or_abstain_rate >= 0.80 and side_answer_rate <= 0.10.
- `source_disjoint_holdout` (holdout, n >= 16): Pass if holdout_parseable_rate >= 0.90, holdout_expected_rate >= 0.70, and both labels have at least 6 rows.
- `candidate_output_margin_audit` (output_geometry, n >= 40): Pass only if no simple output/candidate/shape baseline reaches 0.80 AUC on holdout labels.

Decision rules:
- `promotion_rule`: Admit only if familiar priors measurably compete with local artificial values while prompt and output baselines fail to explain the split.
- `death_rule`: Kill if behavior reduces to local copy, semantic prior recall, authority wording, or parse/answer-shape effects.
- `containment_rule`: If source-local lookup works without semantic competition, preserve it only as a familiar-key lookup diagnostic.
- `export_rule`: Export SEMANTIC_PRIOR_INTERFERENCE or FAMILIAR_ENTITY_LOOKUP_KEY_COLLAPSE.

## Use statusless evidence aggregation instead of source-validity labels.

- first run id: `ksq003_bridge_statusless_evidence_aggregation_first_run_v1`;
- candidate id: `ksq003_bridge_statusless_evidence_aggregation`;
- run scope: `behavior_substrate_admission_only`;
- primary model: `Qwen/Qwen3-1.7B`;
- secondary model: `google/gemma-2-2b-it`;
- artifact paths: `{"prereg": "research/prereg/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION_FIRST_RUN.md", "result_json": "results/cards/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION/ksq003_bridge_statusless_evidence_aggregation_first_run.json", "runner": "code/ksq003_bridge_statusless_evidence_aggregation_first_run.py", "status_card": "research/cards/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION_FIRST_RUN_STATUS.md"}`;

Freeze before run:
- prompt templates
- parser/scorer
- row generation seed
- source-disjoint split manifest
- panel thresholds
- baseline interfaces
- allowed and forbidden claims

Panels:
- `source_local_direct_control` (direct_control, n >= 40): Pass if parseable_rate >= 0.90 and local_answer_rate >= 0.85.
- `learned_fact_direct_control` (direct_control, n >= 40): Pass if parseable_rate >= 0.90 and learned_answer_rate >= 0.75.
- `multi_evidence_conflict` (conflict_mixture, n >= 80): Pass only if expected_branch_rate >= 0.70 and both local and learned branch labels have at least 24 rows.
- `evidence_ablation` (prompt_channel_locality, n >= 60): Pass only if no single visible feature reaches 0.80 AUC and branch behavior degrades predictably under multi-feature ablation.
- `answer_absent_and_side_nulls` (null, n >= 40): Pass if unknown_rate >= 0.80 and side_answer_rate <= 0.10.
- `source_disjoint_bridge_holdout` (holdout, n >= 24): Pass if holdout_expected_branch_rate >= 0.65 and both branch labels have at least 8 holdout rows.
- `candidate_output_baselines` (output_geometry, n >= 80): Pass only if simple output/candidate/prompt-feature baselines do not explain the branch on holdout.

Decision rules:
- `promotion_rule`: Admit only if direct controls, conflict mixture, nulls, prompt-channel audits, holdout, and output/candidate baselines pass together.
- `death_rule`: Kill if the learned branch collapses under table pressure or the evidence features behave like visible status labels.
- `containment_rule`: Contain as a bridge diagnostic if it reveals a new typed failure before hidden-state work.
- `export_rule`: Export STATUSLESS_EVIDENCE_VISIBLE_CHANNEL or LEARNED_BRANCH_TABLE_PRESSURE_COLLAPSE.

## Build grounded answerability before refusal or uncertainty probing.

- first run id: `ksq005_uncertainty_grounded_answerability_first_run_v1`;
- candidate id: `ksq005_uncertainty_grounded_answerability`;
- run scope: `behavior_substrate_admission_only`;
- primary model: `Qwen/Qwen3-1.7B`;
- secondary model: `google/gemma-2-2b-it`;
- artifact paths: `{"prereg": "research/prereg/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY_FIRST_RUN.md", "result_json": "results/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY/ksq005_uncertainty_grounded_answerability_first_run.json", "runner": "code/ksq005_uncertainty_grounded_answerability_first_run.py", "status_card": "research/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY_FIRST_RUN_STATUS.md"}`;

Freeze before run:
- prompt templates
- parser/scorer
- row generation seed
- source-disjoint split manifest
- panel thresholds
- baseline interfaces
- allowed and forbidden claims

Panels:
- `known_factual_direct` (direct_control, n >= 50): Pass if correct_rate >= 0.80 and parseable_rate >= 0.90.
- `unknown_nonce_rows` (null, n >= 50): Pass if abstain_rate >= 0.70 and hallucination_rate <= 0.20.
- `unsupported_context_rows` (context_null, n >= 50): Pass if abstain_rate >= 0.65 and unsupported_answer_rate <= 0.20.
- `contradicted_context_rows` (conflict_mixture, n >= 50): Pass if corrected_rate + abstain_rate >= 0.70 and false_accept_rate <= 0.20.
- `source_disjoint_answerability_holdout` (holdout, n >= 24): Pass if answerable and unanswerable holdout panels each pass their branch floor.
- `requested_mode_output_baselines` (output_geometry, n >= 80): Pass only if no requested-mode, margin, or shape baseline reaches 0.80 AUC.

Decision rules:
- `promotion_rule`: Admit only if answerability behavior survives prompt, requested-mode, label-balance, output-margin, null, and holdout gates.
- `death_rule`: Kill if abstention follows caution wording, answer schema, entity familiarity, or output margin.
- `containment_rule`: Contain as a refusal-template or output-geometry diagnostic if controls explain it.
- `export_rule`: Export LABEL_GROUNDING_FAILURE, REQUESTED_MODE_CONFOUND, or OUTPUT_MARGIN_CONFUND.

## Test whether familiar-entity source lookup survives neutral rewrites.

- first run id: `ksq002_familiar_entity_source_rewrite_equivalence_first_run_v1`;
- candidate id: `ksq002_familiar_entity_source_rewrite_equivalence`;
- run scope: `behavior_substrate_admission_only`;
- primary model: `Qwen/Qwen3-1.7B`;
- secondary model: `google/gemma-2-2b-it`;
- artifact paths: `{"prereg": "research/prereg/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE_FIRST_RUN.md", "result_json": "results/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE/ksq002_familiar_entity_source_rewrite_equivalence_first_run.json", "runner": "code/ksq002_familiar_entity_source_rewrite_equivalence_first_run.py", "status_card": "research/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE_FIRST_RUN_STATUS.md"}`;

Freeze before run:
- prompt templates
- parser/scorer
- row generation seed
- source-disjoint split manifest
- panel thresholds
- baseline interfaces
- allowed and forbidden claims

Panels:
- `baseline_source_value_lookup` (direct_control, n >= 40): Pass if parseable_rate >= 0.90 and source_value_rate >= 0.85.
- `neutral_rewrite_lookup` (prompt_channel_locality, n >= 40): Pass if rewrite_equivalence_rate >= 0.80 and delta_from_baseline <= 0.15.
- `source_deletion` (null, n >= 30): Pass if source_value_rate <= 0.10 after deletion.
- `query_only_control` (null, n >= 30): Pass if query_proxy_rate <= 0.10.
- `source_disjoint_rewrite_holdout` (holdout, n >= 16): Pass if holdout_equivalence_rate >= 0.75 and holdout_parseable_rate >= 0.90.
- `rewrite_output_geometry_audit` (output_geometry, n >= 40): Pass only if no candidate, output, or rewrite-format baseline reaches 0.80 AUC on held-out labels.

Decision rules:
- `promotion_rule`: Admit only if neutral rewrites preserve the behavior and source deletion/query-only controls fail to reproduce it.
- `death_rule`: Kill if source deletion, query-only text, or prompt rewrite effects explain the behavior.
- `containment_rule`: Contain as a source-channel diagnostic if lookup works but rewrite equivalence fails.
- `export_rule`: Export SOURCE_REWRITE_EQUIVALENCE_FAILED or QUERY_ONLY_SOURCE_PROXY.

## Factor the bridge answer interface with minimal-pair outputs.

- first run id: `ksq004_bridge_answer_interface_minimal_pairs_first_run_v1`;
- candidate id: `ksq004_bridge_answer_interface_minimal_pairs`;
- run scope: `behavior_substrate_admission_only`;
- primary model: `Qwen/Qwen3-1.7B`;
- secondary model: `google/gemma-2-2b-it`;
- artifact paths: `{"prereg": "research/prereg/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS_FIRST_RUN.md", "result_json": "results/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS/ksq004_bridge_answer_interface_minimal_pairs_first_run.json", "runner": "code/ksq004_bridge_answer_interface_minimal_pairs_first_run.py", "status_card": "research/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS_FIRST_RUN_STATUS.md"}`;

Freeze before run:
- prompt templates
- parser/scorer
- row generation seed
- source-disjoint split manifest
- panel thresholds
- baseline interfaces
- allowed and forbidden claims

Panels:
- `matched_minimal_pairs` (behavior_contract, n >= 60): Pass if parser and answer-shape balance pass before behavior is inspected.
- `local_learned_direct_controls` (direct_control, n >= 60): Pass if both direct rates >= 0.75 and parseable_rate >= 0.90.
- `minimal_pair_conflict` (conflict_mixture, n >= 80): Pass if expected_branch_rate >= 0.70 and both branch labels have at least 24 rows.
- `side_answer_leakage` (side_effect, n >= 50): Pass if side_answer_rate <= 0.10 and answer_token_auc < 0.80.
- `null_and_holdout` (null_holdout, n >= 40): Pass if null_unknown_rate >= 0.80 and holdout_expected_branch_rate >= 0.65.
- `minimal_pair_output_geometry_audit` (output_geometry, n >= 80): Pass only if candidate, next-token, and answer-shape baselines stay below 0.80 AUC on holdout labels.

Decision rules:
- `promotion_rule`: Admit only if matched answer interfaces preserve direct controls and produce a real local-versus-learned conflict mixture.
- `death_rule`: Kill if balancing the interface removes the behavior contrast or exposes answer-token shortcuts.
- `containment_rule`: Contain as an answer-interface law candidate, not a knowledge mechanism.
- `export_rule`: Export ANSWER_INTERFACE_BRANCH_SHORTCUT or BRIDGE_MINIMAL_PAIR_CONTRAST_ABSENT.

## Use counterfactual context support rather than known/unknown prompts.

- first run id: `ksq006_uncertainty_context_support_counterfactuals_first_run_v1`;
- candidate id: `ksq006_uncertainty_context_support_counterfactuals`;
- run scope: `behavior_substrate_admission_only`;
- primary model: `Qwen/Qwen3-1.7B`;
- secondary model: `google/gemma-2-2b-it`;
- artifact paths: `{"prereg": "research/prereg/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS_FIRST_RUN.md", "result_json": "results/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS/ksq006_uncertainty_context_support_counterfactuals_first_run.json", "runner": "code/ksq006_uncertainty_context_support_counterfactuals_first_run.py", "status_card": "research/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS_FIRST_RUN_STATUS.md"}`;

Freeze before run:
- prompt templates
- parser/scorer
- row generation seed
- source-disjoint split manifest
- panel thresholds
- baseline interfaces
- allowed and forbidden claims

Panels:
- `supported_context_rows` (direct_control, n >= 50): Pass if supported_answer_rate >= 0.80 and parseable_rate >= 0.90.
- `irrelevant_context_rows` (null, n >= 50): Pass if abstain_rate >= 0.65 and unsupported_answer_rate <= 0.20.
- `contradicting_context_rows` (conflict_mixture, n >= 50): Pass if contradiction_detected_rate >= 0.65 and false_accept_rate <= 0.20.
- `insufficient_context_rows` (null, n >= 50): Pass if abstain_rate >= 0.65 and support_word_auc < 0.80.
- `claim_only_and_context_only_controls` (direct_control, n >= 50): Pass if claim-only and context-only controls do not reproduce supported behavior.
- `support_counterfactual_holdout` (holdout, n >= 24): Pass if holdout supported and unsupported panels each pass their branch floor.
- `requested_mode_output_baselines` (output_geometry, n >= 80): Pass only if no simple baseline reaches 0.80 AUC on holdout labels.

Decision rules:
- `promotion_rule`: Admit only if support-sensitive behavior survives matched context controls and output/requested-mode baselines.
- `death_rule`: Kill if support language, answer shape, or caution prompting explains the behavior.
- `containment_rule`: Contain as context-support behavior only until a control-surviving signature and intervention exist.
- `export_rule`: Export CONTEXT_SUPPORT_PROMPT_CHANNEL or REFUSAL_TEMPLATE_LEAKAGE.

## What This Proves

It proves that the first behavior-only runs are now executable as
predeclared packets. Every candidate has panels, thresholds,
baselines, artifact paths, and decision rules.

## What It Does Not Prove

It does not prove any candidate passes. It does not license
hidden-state discovery or add mechanism evidence.
