# Control-Surface Knowledge First-Run Outcomes

Source updated_at: 2026-07-01

Status: generated executed-outcome matrix implemented and validated.

Machine-readable artifact:

> `data/control_surface_knowledge_first_run_outcomes.json`

Builder:

> `code/control_surface_knowledge_first_run_outcomes.py`

Commands:

```powershell
python code\control_surface_knowledge_first_run_outcomes.py --write
python code\control_surface_knowledge_first_run_outcomes.py
python code\validate_control_surface_atlas.py
```

## Summary

- outcomes: `6`
- structural passes: `6`
- full-behavior terminal gates: `3`
- smoke-behavior terminal gates: `3`
- behavior-ready rows: `0`
- hidden-state-allowed rows: `0`
- exported diagnostics: `{"ANSWER_INTERFACE_TEMPLATE_FRAGILITY": 1, "CONTEXT_SUPPORT_CLAIM_OR_MENTION_CONTROL_FAILURE": 1, "FAMILIAR_PRIOR_MIXTURE_PARSEABILITY_TRADEOFF": 1, "GROUNDED_ANSWERABILITY_UNKNOWN_UNSUPPORTED_FAILURE": 1, "SOURCE_REWRITE_HOLDOUT_PARSEABILITY_FAILURE": 1, "STATUSLESS_EVIDENCE_LOCAL_TABLE_DOMINANCE": 1}`

## Outcome Matrix

| Candidate | Level | Terminal Gate | Diagnostic | Selected Template | Failed Axes | Hidden State |
| --- | --- | --- | --- | --- | --- | --- |
| `ksq001_familiar_entity_prior_counterbalance` | `level_2_semi_synthetic_familiar_entity` | `full_behavior_gate` | `FAMILIAR_PRIOR_MIXTURE_PARSEABILITY_TRADEOFF` | `compact_question` | ["conflict_mixture_parseability"] | no |
| `ksq002_familiar_entity_source_rewrite_equivalence` | `level_2_semi_synthetic_familiar_entity` | `full_behavior_gate` | `SOURCE_REWRITE_HOLDOUT_PARSEABILITY_FAILURE` | `sentence_rewrite` | ["source_disjoint_rewrite_parseability"] | no |
| `ksq003_bridge_statusless_evidence_aggregation` | `level_3_symbolic_or_learned_memory_bridge` | `smoke_behavior_gate` | `STATUSLESS_EVIDENCE_LOCAL_TABLE_DOMINANCE` | `compact_fit` | ["one_evidence_mismatch_local_table_dominance", "single_feature_ablation_local_table_dominance"] | no |
| `ksq004_bridge_answer_interface_minimal_pairs` | `level_3_symbolic_or_learned_memory_bridge` | `full_behavior_gate` | `ANSWER_INTERFACE_TEMPLATE_FRAGILITY` | `compact_form` | ["compact_template_learned_branch_collapse", "answer_interface_not_sufficient_explanation"] | no |
| `ksq005_uncertainty_grounded_answerability` | `level_5_real_abstention_uncertainty` | `smoke_behavior_gate` | `GROUNDED_ANSWERABILITY_UNKNOWN_UNSUPPORTED_FAILURE` | `reference_note` | ["unknown_nonce_abstention_parseability", "unsupported_context_abstention_parseability"] | no |
| `ksq006_uncertainty_context_support_counterfactuals` | `level_5_real_abstention_uncertainty` | `smoke_behavior_gate` | `CONTEXT_SUPPORT_CLAIM_OR_MENTION_CONTROL_FAILURE` | `field_form` | ["insufficient_context_abstention", "claim_or_context_only_answer_channel"] | no |

## Per-Row Boundaries

### `ksq001_familiar_entity_prior_counterbalance`

- final diagnostic: `familiar_entity_conflict_parseability_failed`
- exported diagnostic: `FAMILIAR_PRIOR_MIXTURE_PARSEABILITY_TRADEOFF`
- terminal gate: `full_behavior_gate`
- final selected template: `compact_question`
- survived controls: `["source-local artificial lookup", "direct real-capital prior recall", "answer-absent nulls", "source-disjoint mixture", "candidate/output margin reporting"]`
- failed gates: `["full conflict parseability"]`
- final failed criteria: `["primary_parseability_at_least_90p", "smoke_mode"]`
- allowed claim: Familiar priors and prompt-local artificial values can both be separately controlled, with weak conflict mixture.
- forbidden claim: KSQ001 is not a behavior-ready knowledge substrate and licenses no hidden-state work.

### `ksq002_familiar_entity_source_rewrite_equivalence`

- final diagnostic: `source_rewrite_holdout_failed`
- exported diagnostic: `SOURCE_REWRITE_HOLDOUT_PARSEABILITY_FAILURE`
- terminal gate: `full_behavior_gate`
- final selected template: `sentence_rewrite`
- survived controls: `["baseline source-value lookup", "neutral rewrite lookup", "source deletion null", "query-only null", "candidate/output margin reporting"]`
- failed gates: `["full source-disjoint rewrite holdout parseability"]`
- final failed criteria: `["smoke_mode", "source_disjoint_rewrite_holdout_passed"]`
- allowed claim: Source rewrite is mostly robust and source-local under the selected full template, with clean deletion/query-only controls.
- forbidden claim: KSQ002 is not source-rewrite invariant and licenses no internal source-channel mechanism claim.

### `ksq003_bridge_statusless_evidence_aggregation`

- final diagnostic: `smoke_only`
- exported diagnostic: `STATUSLESS_EVIDENCE_LOCAL_TABLE_DOMINANCE`
- terminal gate: `smoke_behavior_gate`
- final selected template: `compact_fit`
- survived controls: `["local-number direct control", "learned atomic-number direct control", "all-evidence-fit local routing", "answer-absent nulls"]`
- failed gates: `["mismatch evidence should route to atomic branch", "symbol/parity-only ablations should become unknown"]`
- final failed criteria: `["full_source_count_is_40", "holdout_conflict_atomic_or_lure_at_least_8", "holdout_conflict_label_balance_passed", "holdout_conflict_local_at_least_8", "non_holdout_conflict_atomic_or_lure_at_least_24", "non_holdout_conflict_label_balance_passed", "non_holdout_conflict_local_at_least_24", "one_evidence_mismatch_atomic_passed", "primary_conflict_binary_rows_at_least_40", "primary_conflict_expected_correct_at_least_70p", "single_feature_ablation_unknown_passed"]`
- allowed claim: Statusless symbol/parity evidence is insufficient under this contract even though direct learned recall and nulls are available.
- forbidden claim: KSQ003 is not a behavior-ready learned/local bridge and licenses no hidden-state work.

### `ksq004_bridge_answer_interface_minimal_pairs`

- final diagnostic: `bridge_minimal_pair_contrast_absent`
- exported diagnostic: `ANSWER_INTERFACE_TEMPLATE_FRAGILITY`
- terminal gate: `full_behavior_gate`
- final selected template: `compact_form`
- survived controls: `["matched answer interface smoke", "direct local and atomic controls", "side-answer leakage smoke", "answer-absent null smoke", "candidate/output margin reporting"]`
- failed gates: `["full compact-form minimal-pair conflict robustness"]`
- final failed criteria: `["minimal_pair_conflict_passed", "null_and_holdout_passed", "smoke_mode"]`
- allowed claim: Answer-interface matching alone does not explain away the bridge, because question_form mostly retains local-vs-learned contrast.
- forbidden claim: KSQ004 is not a behavior-ready answer-interface or learned-memory bridge substrate.

### `ksq005_uncertainty_grounded_answerability`

- final diagnostic: `unknown_nonce_abstention_failed`
- exported diagnostic: `GROUNDED_ANSWERABILITY_UNKNOWN_UNSUPPORTED_FAILURE`
- terminal gate: `smoke_behavior_gate`
- final selected template: `reference_note`
- survived controls: `["known factual direct control", "contradicted familiar-context correction/abstention", "candidate/output margin reporting"]`
- failed gates: `["unknown nonce abstention", "unsupported context abstention"]`
- final failed criteria: `["full_source_count_is_40", "source_disjoint_answerability_holdout_passed", "unknown_nonce_rows_passed", "unsupported_context_rows_passed"]`
- allowed claim: Correction of false familiar-context claims is easier than grounded abstention on unknown or unsupported entities.
- forbidden claim: KSQ005 is not an uncertainty, refusal, or answerability-control surface.

### `ksq006_uncertainty_context_support_counterfactuals`

- final diagnostic: `insufficient_context_abstention_failed`
- exported diagnostic: `CONTEXT_SUPPORT_CLAIM_OR_MENTION_CONTROL_FAILURE`
- terminal gate: `smoke_behavior_gate`
- final selected template: `field_form`
- survived controls: `["supported context", "irrelevant context abstention", "contradicting context detection", "candidate/output margin reporting"]`
- failed gates: `["insufficient context abstention", "claim/context-only control locality"]`
- final failed criteria: `["claim_only_and_context_only_controls_passed", "full_source_count_is_40", "insufficient_context_rows_passed", "support_counterfactual_holdout_passed"]`
- allowed claim: Context support has a partial substrate, but relation-free claim or city mention controls still carry the answer channel.
- forbidden claim: KSQ006 is not a context-support, uncertainty, or correction-control surface.

## Validation Checks

| Check | Passed |
| --- | --- |
| `covers_all_first_run_packets` | `true` |
| `all_artifact_paths_exist` | `true` |
| `all_structural_gates_passed` | `true` |
| `no_behavior_ready_or_hidden_state_allowed_rows` | `true` |
| `expected_diagnostics_preserved` | `true` |
| `three_full_and_three_smoke_terminal_gates` | `true` |
| `two_outcomes_per_knowledge_level` | `true` |
| `diagnostic_boundaries_are_named` | `true` |

## Claim Boundary

All six knowledge first-run packets have been executed through at least smoke behavior; all six passed structural construction; none became behavior-ready or licensed hidden-state work. The reusable finding is the distribution of typed behavior-gate failures.

The KSQ outcome matrix does not promote a mechanism, does not claim a knowledge vector, and does not license probing, steering, editing, or surgery on any KSQ row.
