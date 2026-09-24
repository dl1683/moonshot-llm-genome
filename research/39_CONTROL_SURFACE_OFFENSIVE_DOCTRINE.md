# Control-Surface Offensive Doctrine

Date: 2026-07-01

Status: generated offensive-doctrine harness implemented and validated.

Machine-readable artifact:

> `data/control_surface_offensive_doctrine.json`

Builder:

> `code/control_surface_offensive_doctrine.py`

Commands:

```powershell
python code\control_surface_offensive_doctrine.py --write
python code\control_surface_offensive_doctrine.py
python code\validate_control_surface_atlas.py
```

## Purpose

This layer is the operational complement to the defensive audit stack.
It makes future branch quality machine-checkable before a run begins:
the branch must name the gap it targets, the generated layer it can
change, the evidence packet it owes, and the exact rule that kills it.

## Generated Facts

- branch contracts: 6;
- source work orders: 6;
- critical gaps covered by contracts: 4;
- urgency counts: `{"high": 3, "immediate": 3}`;
- track type counts: `{"bridge_substrate": 1, "deepening_closeout": 1, "knowledge_frontier_closeout": 1, "law_replication": 1, "process_harness": 1, "widening_probe": 1}`;
- top queue coverage: 5.

## Global Rules

- Start no branch without a named coverage gap.
- Start no branch without promote, bound, kill, containment, and export rules.
- Treat a typed failure as a result only if it changes the atlas, taxonomy, gap map, closure plan, or future branch contract.
- Require generated map movement before claiming progress beyond a local story.
- Treat hidden-state work as forbidden until behavior, null, holdout, prompt-channel, and output/candidate controls license it.

## Required Branch Fields

- `target_gap_ids`
- `expected_generated_layer_change`
- `promotion_rule`
- `bound_rule`
- `kill_rule`
- `containment_rule`
- `export_rule`
- `first_artifact`
- `iteration_budget`
- `forbidden_moves`
- `minimum_evidence_packet`
- `allowed_outputs`

## Branch Contracts

| Work Order | Urgency | Track | Target Gaps | Expected Layer Change | Kill Rule |
| --- | --- | --- | --- | --- | --- |
| `close_mc005_reference_specimen` | `immediate` | `deepening_closeout` | `["promoted_mechanism_absent", "full_reliability_absent", "clean_intervention_absent", "mc005_singleton_internal_causal_reference", "singleton_terminal_stage_evidence", "transfer_ready_mechanism_absent"]` | `["data/control_surface_atlas.json", "data/control_surface_reliability_matrix.json", "data/control_surface_route_disposition.json", "data/control_surface_gate_geometry.json"]` | Kill the write-replacement route if null flips or side effects persist across two materially different local intervention variants. |
| `close_post_mc033_bridge_substrate_family` | `immediate` | `bridge_substrate` | `["bridge_hidden_state_not_licensed", "clean_intervention_absent", "promoted_mechanism_absent", "domain_coverage_knowledge_bridge_dominates_failures", "full_reliability_absent", "output_geometry_pressure_mixed"]` | `["data/control_surface_bridge_ladder.json", "data/control_surface_smoke_diagnostics.json", "data/control_surface_gate_geometry.json", "data/control_surface_coverage_gaps.json"]` | Kill same-family repairs after MC033; do not add another source-validity cue unless it changes the substrate class, not just the wording of the reliability cue. |
| `close_mc006_predecision_frontier` | `high` | `knowledge_frontier_closeout` | `["clean_intervention_absent", "mc006_knowledge_route_monitor_only", "output_geometry_pressure_mixed", "promoted_mechanism_absent", "axis_rules_sparse_or_singleton_heavy"]` | `["data/control_surface_decision_frontier.json", "data/control_surface_route_disposition.json", "data/control_surface_coverage_gaps.json", "data/control_surface_error_taxonomy.json"]` | Kill final-prompt-token probing for MC006 if another margin-matched pass remains candidate-score or final-margin explained. |
| `run_width_transfer_probe` | `immediate` | `widening_probe` | `["transfer_ready_mechanism_absent", "model_family_coverage_qwen_dominant", "full_reliability_absent", "mc005_singleton_internal_causal_reference", "axis_rules_sparse_or_singleton_heavy"]` | `["data/control_surface_transfer_matrix.json", "data/control_surface_reliability_matrix.json", "data/control_surface_coverage_gaps.json", "data/control_surface_genome_snapshot.json"]` | Kill the transfer route for a surface if two model-family or prompt-family tests reproduce the same reliability failure. |
| `replicate_singleton_stage_laws` | `high` | `law_replication` | `["axis_rules_sparse_or_singleton_heavy", "singleton_terminal_stage_evidence", "output_geometry_pressure_mixed", "domain_coverage_knowledge_bridge_dominates_failures", "model_family_coverage_qwen_dominant"]` | `["data/control_surface_axis_interactions.json", "data/control_surface_law_hypotheses.json", "data/control_surface_law_audit.json", "data/control_surface_coverage_gaps.json"]` | Kill a proposed law if two new rows land in different terminal stages under the same claimed predictor. |
| `enforce_offensive_doctrine_harness` | `high` | `process_harness` | `["promoted_mechanism_absent", "clean_intervention_absent", "full_reliability_absent", "transfer_ready_mechanism_absent", "axis_rules_sparse_or_singleton_heavy", "bridge_hidden_state_not_licensed"]` | `["data/control_surface_offensive_doctrine.json", "research/39_CONTROL_SURFACE_OFFENSIVE_DOCTRINE.md", "code/validate_control_surface_atlas.py"]` | Kill any branch that repeats the same diagnostic class after the predeclared repair budget. |

## Allowed Outputs

- `promoted_mechanism_card`
- `bounded_mechanism_card`
- `failed_mechanism_card`
- `diagnostic_note`

## What This Proves

It proves that the project now has a generated intake harness for new
branches. A future experiment is not just a promising idea; it is a
contract against a named gap, expected generated-layer movement, and
a predeclared death rule.

## What It Does Not Prove

It does not prove that a control surface has been found, transferred,
or made reliable. It only prevents future evidence from entering the
atlas as an attractive but unbounded story.
