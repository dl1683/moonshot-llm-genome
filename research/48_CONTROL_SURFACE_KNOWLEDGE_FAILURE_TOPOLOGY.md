# Control-Surface Knowledge Failure Topology

Source updated_at: 2026-07-01

Status: generated KSQ failure topology implemented and validated.

Machine-readable artifact:

> `data/control_surface_knowledge_failure_topology.json`

Builder:

> `code/control_surface_knowledge_failure_topology.py`

Commands:

```powershell
python code\control_surface_knowledge_failure_topology.py --write
python code\control_surface_knowledge_failure_topology.py
python code\validate_control_surface_atlas.py
```

## Summary

- topology nodes: `4`
- second-wave work orders: `5`
- candidate coverage: `6/6`
- behavior-ready rows: `0`
- hidden-state-allowed rows: `0`
- node types: `{"bridge_boundary": 1, "material_redesign_required": 1, "narrow_repair_candidate": 1, "positive_infrastructure_result": 1}`
- work-order priorities: `{"high": 1, "immediate": 1, "medium": 3}`

## Topology Nodes

| Node | Type | Candidates | Decision |
| --- | --- | --- | --- |
| `structural_substrate_construction_solved` | `positive_infrastructure_result` | ["ksq001_familiar_entity_prior_counterbalance", "ksq002_familiar_entity_source_rewrite_equivalence", "ksq003_bridge_statusless_evidence_aggregation", "ksq004_bridge_answer_interface_minimal_pairs", "ksq005_uncertainty_grounded_answerability", "ksq006_uncertainty_context_support_counterfactuals"] | `reuse_harness` |
| `full_behavior_parseability_near_misses` | `narrow_repair_candidate` | ["ksq001_familiar_entity_prior_counterbalance", "ksq002_familiar_entity_source_rewrite_equivalence"] | `allow_one_repair_iteration_each` |
| `bridge_template_and_local_table_boundary` | `bridge_boundary` | ["ksq003_bridge_statusless_evidence_aggregation", "ksq004_bridge_answer_interface_minimal_pairs"] | `adjudicate_template_fragility_before_new_bridge_probe` |
| `real_uncertainty_answer_channel_failures` | `material_redesign_required` | ["ksq005_uncertainty_grounded_answerability", "ksq006_uncertainty_context_support_counterfactuals"] | `redesign_before_more_runs` |

## Second-Wave Work Orders

### `repair_ksq002_source_rewrite_holdout`

- priority: `immediate`
- track: `narrow_repair`
- candidates: `["ksq002_familiar_entity_source_rewrite_equivalence"]`
- target question: Can the source-disjoint rewrite holdout cross the predeclared parseability gate without weakening deletion/query-only locality?
- required evidence: `["Full 40-source rerun with source-disjoint rewrite holdout parseability >= 0.90.", "Neutral rewrite artificial-value rate remains within 0.15 of baseline.", "Source deletion and query-only controls remain UNKNOWN-dominant.", "Candidate/output margins are reported on the selected template.", "Status card states whether hidden-state work remains forbidden or is newly admitted."]`
- promotion rule: Admit only to behavior-ready status if rewrite, deletion, query-only, holdout, prompt audit, and margin-reporting gates all pass.
- kill rule: Kill ordinary source-rewrite repair if a single predeclared repair rerun still misses source-disjoint holdout parseability or breaks deletion/query-only controls.
- containment rule: If repaired, the claim is source-rewrite behavior admission only; no mechanism claim exists until a signature beats prompt/output controls and an intervention is tested.
- export rule: Export SOURCE_REWRITE_HOLDOUT_PARSEABILITY_FAILURE if the boundary persists, or SOURCE_REWRITE_BEHAVIOR_ADMITTED if the full gate passes.
- forbidden moves: `["Do not drop the source-disjoint holdout to improve pass rate.", "Do not remove deletion or query-only controls.", "Do not start hidden-state probing from the existing failed full run."]`

### `adjudicate_ksq004_template_invariance`

- priority: `high`
- track: `template_boundary`
- candidates: `["ksq004_bridge_answer_interface_minimal_pairs"]`
- target question: Is the bridge behavior genuinely template-local to question_form, or can it survive a second compact/neutral surface?
- required evidence: `["Predeclare template families before scoring.", "Report conflict, side-answer leakage, null/holdout, and candidate margins per template.", "Show whether question_form full behavior passes when selected directly.", "Show whether any compact/neutral relation-key template reaches the same branch balance.", "Preserve matched local/atomic direct controls under the same answer interface."]`
- promotion rule: Admit bridge behavior only if at least two predeclared templates pass conflict, side-leakage, null, holdout, and margin-reporting gates.
- kill rule: Kill same-family answer-interface repair if question_form is the only passing surface or compact/neutral templates keep collapsing expected-atomic rows to local answers.
- containment rule: A single passing template is a template-specific behavior note, not a learned-memory bridge substrate.
- export rule: Export ANSWER_INTERFACE_TEMPLATE_FRAGILITY if the split persists, or TEMPLATE_INVARIANT_BRIDGE_BEHAVIOR if two templates pass.
- forbidden moves: `["Do not treat first-token numeric margins as meaningful when sequence candidate scoring is the real output baseline.", "Do not call question_form alone a robust bridge.", "Do not start hidden-state probing while compact/neutral templates fail."]`

### `bound_ksq001_familiar_prior_parseability`

- priority: `medium`
- track: `bounded_repair_or_closeout`
- candidates: `["ksq001_familiar_entity_prior_counterbalance"]`
- target question: Can familiar-prior conflict parseability be repaired without removing the weak local-versus-prior mixture?
- required evidence: `["Full 40-source conflict parseability >= 0.90.", "Both artificial and real-prior/lure outcomes remain present.", "Direct local lookup, direct real-prior recall, nulls, and holdout survive.", "Answer-shape and candidate/output baselines are reported."]`
- promotion rule: Admit only if parseability repair preserves mixture and all direct/null/holdout gates.
- kill rule: Kill ordinary KSQ001 repair if parseability improves only by collapsing to local lookup, prior recall, or UNKNOWN.
- containment rule: The surviving claim is familiar-prior behavior pressure, not a knowledge substrate.
- export rule: Export FAMILIAR_PRIOR_MIXTURE_PARSEABILITY_TRADEOFF unless a clean full behavior substrate appears.
- forbidden moves: `["Do not use a parser repair that changes the behavior branch distribution without reporting it.", "Do not count direct controls as conflict success.", "Do not run hidden-state probes on the current failed full behavior table."]`

### `redesign_statusless_bridge_substrate`

- priority: `medium`
- track: `material_redesign`
- candidates: `["ksq003_bridge_statusless_evidence_aggregation"]`
- target question: What bridge substrate can route learned/local branches without visible status cues and without local-table dominance?
- required evidence: `["A materially new branch-selection family, not another symbol/parity ablation.", "Direct learned recall and local lookup remain clean.", "Mismatch/conflict rows produce the intended learned branch above gate.", "Single-feature ablations go UNKNOWN instead of local.", "Source-disjoint holdout and candidate/output baselines are reported."]`
- promotion rule: Admit only if conflict and ablation behavior both pass while direct controls and nulls stay clean.
- kill rule: Kill same-family evidence aggregation if the next material design again sends mismatch or ablation rows to local answers.
- containment rule: Until then, KSQ003 remains a local-table-dominance diagnostic.
- export rule: Export STATUSLESS_EVIDENCE_LOCAL_TABLE_DOMINANCE as the default bridge failure unless a new substrate passes.
- forbidden moves: `["Do not add visible trusted/untrusted labels.", "Do not reuse checksum, row-code, or fact-claim validity cues from the killed bridge family.", "Do not start hidden-state work before conflict and ablation gates pass."]`

### `redesign_real_uncertainty_answerability`

- priority: `medium`
- track: `material_redesign`
- candidates: `["ksq005_uncertainty_grounded_answerability", "ksq006_uncertainty_context_support_counterfactuals"]`
- target question: Can real uncertainty be made behavior-ready when answerability and relation support are tested against claim-only and mention-only controls?
- required evidence: `["Unknown and unsupported rows abstain or say UNKNOWN with >= 0.80 stability.", "Supported rows still answer correctly.", "Contradicting rows reject false context without relying on answer-shape artifacts.", "Claim-only and mention-only controls do not reproduce supported answers.", "Parser, answer schema, and output/candidate baselines are reported."]`
- promotion rule: Admit only if abstention, support, contradiction, and claim/mention controls all pass together.
- kill rule: Kill the current uncertainty route if the next redesign again fixes correction-looking rows but fails unknown/unsupported or claim/mention controls.
- containment rule: Correction of false familiar facts remains a behavior diagnostic until grounded abstention and relation support pass.
- export rule: Export GROUNDED_ANSWERABILITY_UNKNOWN_UNSUPPORTED_FAILURE and CONTEXT_SUPPORT_CLAIM_OR_MENTION_CONTROL_FAILURE unless both routes pass.
- forbidden moves: `["Do not call contradicted-context correction uncertainty control.", "Do not hide claim-only or city-mention controls.", "Do not start hidden-state probing from a smoke-stage abstention failure."]`

## Validation Checks

| Check | Passed |
| --- | --- |
| `topology_covers_all_ksq_outcomes` | `true` |
| `second_wave_covers_all_ksq_outcomes` | `true` |
| `no_hidden_state_license_created` | `true` |
| `work_orders_bind_existing_nodes` | `true` |
| `work_orders_have_decision_rules` | `true` |
| `work_orders_preserve_hidden_state_bar` | `true` |
| `near_miss_and_redesign_are_separated` | `true` |

## Claim Boundary

The first KSQ wave now has an explicit failure topology: source-rewrite and familiar-prior parseability are bounded repair candidates; bridge behavior is split between template fragility and local-table dominance; real uncertainty needs material redesign before more probing.

This topology does not make any KSQ row behavior-ready, does not license hidden-state search, and does not claim a knowledge-control surface.
