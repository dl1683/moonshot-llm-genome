"""Build the bridge-ladder map for MC010-MC033.

The atlas and smoke ledger already validate individual artifacts. This module
answers a different question: what did the bridge program learn cumulatively as
it moved from symbolic lookup, to numeric lookup, to prompt-visible reliability
labels, to channel ablations, learned-fact gates, branch arbitration, and
query-level operation handles, few-shot operation examples, constrained choice
interfaces, numeric option lists, answer-interface sweeps, full-source boundary
tests, factorized leak probes, and null-preserving repair attempts?
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_INDEX_PATH = ROOT / "data" / "control_surface_artifact_index.json"
COMPARISON_PATH = ROOT / "data" / "control_surface_comparison.json"
SMOKE_DIAGNOSTICS_PATH = ROOT / "data" / "control_surface_smoke_diagnostics.json"
BRIDGE_LADDER_PATH = ROOT / "data" / "control_surface_bridge_ladder.json"
BRIDGE_LADDER_REPORT_PATH = ROOT / "research" / "27_CONTROL_SURFACE_BRIDGE_LADDER.md"


RUNG_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "card_id": "MC010",
        "source": "atlas",
        "row_id": "mc010_two_hop_fact_code_arbitration",
        "title": "Two-hop symbolic indirection",
        "contract_axis": "source_path",
        "repair_attempt": "Remove direct entity-to-code rows with entity -> handle -> task-code indirection.",
        "behavior_outcome": "failed_controls_and_conflict",
        "claim_boundary": "Two-hop indirection did not create a behavior substrate.",
        "dominant_failure": "synthetic and learned controls plus conflict contrast failed",
        "prompt_channel": "no_status_channel",
        "answer_interface": "symbolic_code",
        "local_vs_learned_mixture": "absent",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("synthetic task-code rate", "artifact.metrics.panel.synthetic_two_hop_lookup.task_code_rate"),
            ("real memory symbol rate", "artifact.metrics.panel.real_world_memory_control.real_symbol_rate"),
            ("conflict task-code rate", "artifact.metrics.primary_conflict.task_code_rate"),
            ("conflict learned-symbol rate", "artifact.metrics.primary_conflict.real_or_lure_symbol_rate"),
        ],
    },
    {
        "card_id": "MC011",
        "source": "atlas",
        "row_id": "mc011_atomic_number_code_arbitration",
        "title": "Same-format numeric answer interface",
        "contract_axis": "answer_interface",
        "repair_attempt": "Use integers for both prompt-local values and learned atomic facts.",
        "behavior_outcome": "controls_clean_conflict_local_collapse",
        "claim_boundary": "Answer-format repair is insufficient under prompt-local table authority.",
        "dominant_failure": "numeric conflict collapsed to prompt-local values",
        "prompt_channel": "no_status_channel",
        "answer_interface": "integer",
        "local_vs_learned_mixture": "absent",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("synthetic local rate", "artifact.metrics.panel.synthetic_numeric_lookup.local_number_rate"),
            ("real atomic control rate", "artifact.metrics.panel.real_world_atomic_number_control.atomic_number_rate"),
            ("null UNKNOWN rate", "artifact.metrics.panel.answer_absent_null.unknown_rate"),
            ("conflict local rate", "artifact.metrics.primary_conflict.local_number_rate"),
            ("conflict atomic rate", "artifact.metrics.primary_conflict.atomic_number_rate"),
        ],
    },
    {
        "card_id": "MC012",
        "source": "atlas",
        "row_id": "mc012_reliability_labeled_numeric_arbitration",
        "title": "Explicit reliability labels",
        "contract_axis": "source_contract",
        "repair_attempt": "Use visible trusted/untrusted source-status labels.",
        "behavior_outcome": "behavior_ready_but_prompt_visible",
        "claim_boundary": "Positive control only; source-status text carries the answer rule.",
        "dominant_failure": "prompt-channel locality failed",
        "prompt_channel": "visible_status_channel",
        "answer_interface": "integer",
        "local_vs_learned_mixture": "clean_prompt_visible",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("trusted local rate", "artifact.metrics.panel.trusted_source_conflict.local_number_rate"),
            ("untrusted atomic rate", "artifact.metrics.panel.untrusted_source_conflict.atomic_number_rate"),
            ("primary local rate", "artifact.metrics.primary_conflict.local_number_rate"),
            ("primary atomic rate", "artifact.metrics.primary_conflict.atomic_number_rate"),
            ("prompt-channel locality gate", "artifact.metrics.criteria.prompt_channel_locality_gate_passed"),
        ],
    },
    {
        "card_id": "MC013",
        "source": "atlas",
        "row_id": "mc013_status_channel_ablation_numeric_arbitration",
        "title": "Text-identical status-channel ablation",
        "contract_axis": "prompt_channel_locality",
        "repair_attempt": "Reproduce MC012 with status text, then ablate the visible status channel.",
        "behavior_outcome": "positive_control_reproduced_ablation_collapsed",
        "claim_boundary": "The contrast does not survive status-channel removal.",
        "dominant_failure": "matched ablation collapsed to prompt-local values",
        "prompt_channel": "status_channel_removed",
        "answer_interface": "integer",
        "local_vs_learned_mixture": "statused_only",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("statused trusted local rate", "artifact.metrics.panel.statused_trusted_conflict.local_number_rate"),
            ("statused untrusted atomic rate", "artifact.metrics.panel.statused_untrusted_conflict.atomic_number_rate"),
            ("ablation local rate", "artifact.metrics.primary_ablation_conflict.local_number_rate"),
            ("ablation atomic rate", "artifact.metrics.primary_ablation_conflict.atomic_number_rate"),
            ("ablation prompts identical", "artifact.metrics.criteria.ablation_prompt_pairs_identical"),
        ],
    },
    {
        "card_id": "MC014",
        "source": "atlas",
        "row_id": "mc014_inferred_reliability_numeric_arbitration",
        "title": "Calibration-inferred reliability",
        "contract_axis": "inferred_source_validity",
        "repair_attempt": "Remove status labels and infer source validity from calibration rows.",
        "behavior_outcome": "calibration_inference_collapsed",
        "claim_boundary": "Calibration evidence did not overcome local table authority.",
        "dominant_failure": "calibration-consistent and inconsistent rows both local",
        "prompt_channel": "status_absent_calibration_visible",
        "answer_interface": "integer",
        "local_vs_learned_mixture": "absent",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("status lexemes absent", "artifact.metrics.criteria.visible_status_label_absent_by_design"),
            ("consistent local rate", "artifact.metrics.panel.calibration_consistent_conflict.local_number_rate"),
            ("inconsistent atomic rate", "artifact.metrics.panel.calibration_inconsistent_conflict.atomic_number_rate"),
            ("primary local rate", "artifact.metrics.primary_conflict.local_number_rate"),
        ],
    },
    {
        "card_id": "MC015",
        "source": "atlas",
        "row_id": "mc015_parity_gated_numeric_arbitration",
        "title": "Learned parity gate",
        "contract_axis": "learned_fact_gate",
        "repair_attempt": "Make the source rule depend on hidden learned atomic-number parity.",
        "behavior_outcome": "mixed_outputs_wrong_rule",
        "claim_boundary": "Mixed outputs are not enough when expected labels are not followed.",
        "dominant_failure": "rule-aligned expected correctness failed",
        "prompt_channel": "status_absent_learned_gate",
        "answer_interface": "integer",
        "local_vs_learned_mixture": "mixed_wrong_rule",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("expected labels balanced", "artifact.metrics.criteria.primary_expected_labels_balanced_by_split"),
            ("primary expected-correct rate", "artifact.metrics.primary_conflict.expected_correct_rate"),
            ("expected-local local rate", "artifact.metrics.primary_conflict_expected_local.local_number_rate"),
            ("expected-atomic atomic rate", "artifact.metrics.primary_conflict_expected_atomic.atomic_number_rate"),
        ],
    },
    {
        "card_id": "MC016",
        "source": "atlas",
        "row_id": "mc016_alphabet_gated_numeric_arbitration",
        "title": "Visible non-status alphabet gate",
        "contract_axis": "visible_nonstatus_gate",
        "repair_attempt": "Use a visible first-letter rule instead of source-status labels.",
        "behavior_outcome": "visible_gate_local_collapse",
        "claim_boundary": "A visible non-status gate still collapses expected-atomic rows.",
        "dominant_failure": "expected-atomic rows selected local",
        "prompt_channel": "status_absent_visible_feature",
        "answer_interface": "integer",
        "local_vs_learned_mixture": "absent",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("expected labels balanced", "artifact.metrics.criteria.primary_expected_labels_balanced_by_split"),
            ("primary local rate", "artifact.metrics.primary_conflict.local_number_rate"),
            ("expected-local local rate", "artifact.metrics.primary_conflict_expected_local.local_number_rate"),
            ("expected-atomic atomic rate", "artifact.metrics.primary_conflict_expected_atomic.atomic_number_rate"),
        ],
    },
    {
        "card_id": "MC017",
        "source": "smoke",
        "title": "LOCAL/ATOMIC source-token answers",
        "contract_axis": "source_answer_interface",
        "repair_attempt": "Ask for source tokens before any numeric answer.",
        "behavior_outcome": "answer_token_local_collapse",
        "claim_boundary": "Source-token answers are themselves a confounded behavior surface.",
        "dominant_failure": "ATOMIC selector control collapsed to LOCAL",
        "prompt_channel": "source_token_answer",
        "answer_interface": "source_token",
        "local_vs_learned_mixture": "absent",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("atomic selector atomic rate", "smoke.selected_metrics.panel_metrics.atomic_selector_control.atomic_selector_rate"),
            ("answer-absent UNKNOWN rate", "smoke.selected_metrics.panel_metrics.answer_absent_null.unknown_rate"),
            ("primary expected-correct rate", "smoke.selected_metrics.named_blocks.primary_conflict.expected_correct_rate"),
        ],
    },
    {
        "card_id": "MC018",
        "source": "smoke",
        "title": "Neutral counterbalanced selector labels",
        "contract_axis": "neutral_source_answer_interface",
        "repair_attempt": "Replace LOCAL/ATOMIC answers with counterbalanced A/B labels.",
        "behavior_outcome": "neutral_selector_rule_failed",
        "claim_boundary": "Neutral labels expose first-listed-choice and local-source salience.",
        "dominant_failure": "source-rule correctness near chance with order bias",
        "prompt_channel": "neutral_label_answer",
        "answer_interface": "choice_label",
        "local_vs_learned_mixture": "mixed_wrong_rule",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("primary expected-correct rate", "smoke.selected_metrics.named_blocks.primary_conflict.expected_correct_rate"),
            ("first-listed-choice rate", "smoke.selected_metrics.named_blocks.primary_conflict.first_listed_choice_rate"),
            ("local-selector rate", "smoke.selected_metrics.named_blocks.primary_conflict.local_selector_rate"),
        ],
    },
    {
        "card_id": "MC019",
        "source": "smoke",
        "title": "Neutral row-code numeric answers",
        "contract_axis": "row_local_route_code",
        "repair_attempt": "Keep integer answers but attach neutral route codes to rows.",
        "behavior_outcome": "row_code_conflict_failed",
        "claim_boundary": "Clean controls and nulls do not imply expected-atomic routing.",
        "dominant_failure": "expected-atomic route weak despite clean nulls",
        "prompt_channel": "row_code",
        "answer_interface": "integer",
        "local_vs_learned_mixture": "mixed_wrong_rule",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("primary expected-correct rate", "smoke.selected_metrics.named_blocks.primary_conflict.expected_correct_rate"),
            ("expected-atomic atomic rate", "smoke.selected_metrics.named_blocks.primary_conflict_expected_atomic.atomic_number_rate"),
            ("answer-absent UNKNOWN rate", "smoke.selected_metrics.panel_metrics.answer_absent_null.unknown_rate"),
        ],
    },
    {
        "card_id": "MC020",
        "source": "smoke",
        "title": "Atomic recall table-pressure isolation",
        "contract_axis": "atomic_recall_vs_routing",
        "repair_attempt": "Separate direct atomic recall under local-row pressure from route-rule arbitration.",
        "behavior_outcome": "direct_recall_passes_route_atomic_fails",
        "claim_boundary": "The bottleneck is conditional arbitration, not basic atomic recall.",
        "dominant_failure": "route-atomic branch collapsed toward local",
        "prompt_channel": "route_code",
        "answer_interface": "integer",
        "local_vs_learned_mixture": "asymmetric_route_failure",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("query-row atomic control rate", "smoke.selected_metrics.panel_metrics.query_row_atomic_control.atomic_number_rate"),
            ("route-local local rate", "smoke.selected_metrics.panel_metrics.route_local_conflict.local_number_rate"),
            ("route-atomic atomic rate", "smoke.selected_metrics.panel_metrics.route_atomic_conflict.atomic_number_rate"),
        ],
    },
    {
        "card_id": "MC021",
        "source": "smoke",
        "title": "Visible-versus-learned route-code arbitration",
        "contract_axis": "visible_vs_learned_branch_routing",
        "repair_attempt": "Test the same route-code grammar on visible-visible and visible-learned branches.",
        "behavior_outcome": "route_code_unstable_even_visible",
        "claim_boundary": "Opaque route codes are not a clean arbitration substrate.",
        "dominant_failure": "visible-visible route weak, learned route weaker",
        "prompt_channel": "route_code",
        "answer_interface": "integer",
        "local_vs_learned_mixture": "asymmetric_route_failure",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("visible conflict expected-correct rate", "smoke.selected_metrics.named_blocks.visible_conflict.expected_correct_rate"),
            ("learned conflict expected-correct rate", "smoke.selected_metrics.named_blocks.learned_conflict.expected_correct_rate"),
            ("learned atomic-branch atomic rate", "smoke.selected_metrics.panel_metrics.route_learned_p_atomic_conflict.atomic_number_rate"),
        ],
    },
    {
        "card_id": "MC022",
        "source": "smoke",
        "title": "Semantic branch-name arbitration",
        "contract_axis": "semantic_branch_label_routing",
        "repair_attempt": "Replace opaque route codes with LOCAL/REFERENCE/ATOMIC source labels.",
        "behavior_outcome": "semantic_labels_rule_order_sensitive",
        "claim_boundary": "Semantic labels do not solve learned-memory branch arbitration.",
        "dominant_failure": "local-source salience and rule-order sensitivity remain",
        "prompt_channel": "semantic_source_label",
        "answer_interface": "integer",
        "local_vs_learned_mixture": "asymmetric_route_failure",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("visible conflict expected-correct rate", "smoke.selected_metrics.named_blocks.visible_conflict.expected_correct_rate"),
            ("learned conflict expected-correct rate", "smoke.selected_metrics.named_blocks.learned_conflict.expected_correct_rate"),
            ("ATOMIC-source atomic rate", "smoke.selected_metrics.panel_metrics.source_learned_atomic_conflict.atomic_number_rate"),
            ("nonlocal-first visible expected-correct rate", "smoke.selected_metrics.named_blocks.visible_conflict_by_rule_order.alternate_first.expected_correct_rate"),
        ],
    },
    {
        "card_id": "MC023",
        "source": "smoke",
        "title": "Query-level operation arbitration",
        "contract_axis": "query_operation_routing",
        "repair_attempt": "Remove row-level source labels and use counterbalanced query operation handles.",
        "behavior_outcome": "query_operation_conflict_below_gate",
        "claim_boundary": "Query-level operation handles preserve controls and nulls but do not clear both conflict branches.",
        "dominant_failure": "operation-local and operation-atomic conflict branches below gate",
        "prompt_channel": "query_operation_code",
        "answer_interface": "integer",
        "local_vs_learned_mixture": "mixed_below_gate",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("synthetic local control rate", "smoke.selected_metrics.panel_metrics.synthetic_numeric_lookup.local_number_rate"),
            ("atomic control atomic rate", "smoke.selected_metrics.panel_metrics.real_world_atomic_number_control.atomic_number_rate"),
            ("answer-absent UNKNOWN rate", "smoke.selected_metrics.panel_metrics.answer_absent_null.unknown_rate"),
            ("operation-local local rate", "smoke.selected_metrics.panel_metrics.operation_local_conflict.local_number_rate"),
            ("operation-atomic atomic rate", "smoke.selected_metrics.panel_metrics.operation_atomic_conflict.atomic_number_rate"),
        ],
    },
    {
        "card_id": "MC024",
        "source": "smoke",
        "title": "Few-shot query-operation arbitration",
        "contract_axis": "fewshot_query_operation_routing",
        "repair_attempt": "Add balanced worked examples to query operation handles.",
        "behavior_outcome": "fewshot_operation_atomic_branch_failed",
        "claim_boundary": "Worked examples repair local routing but not full-source learned atomic routing.",
        "dominant_failure": "operation-atomic branch below gate with other-number leakage",
        "prompt_channel": "query_operation_code_with_worked_examples",
        "answer_interface": "integer",
        "local_vs_learned_mixture": "asymmetric_route_failure",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("synthetic local control rate", "smoke.selected_metrics.panel_metrics.synthetic_numeric_lookup.local_number_rate"),
            ("atomic control atomic rate", "smoke.selected_metrics.panel_metrics.real_world_atomic_number_control.atomic_number_rate"),
            ("answer-absent UNKNOWN rate", "smoke.selected_metrics.panel_metrics.answer_absent_null.unknown_rate"),
            ("operation-local local rate", "smoke.selected_metrics.panel_metrics.operation_local_conflict.local_number_rate"),
            ("operation-atomic atomic rate", "smoke.selected_metrics.panel_metrics.operation_atomic_conflict.atomic_number_rate"),
            ("operation-atomic other-number rate", "smoke.selected_metrics.panel_metrics.operation_atomic_conflict.other_number_rate"),
        ],
    },
    {
        "card_id": "MC025",
        "source": "smoke",
        "title": "Constrained choice operation arbitration",
        "contract_axis": "choice_answer_interface",
        "repair_attempt": "Constrain final answers to A/B/C options containing local number, atomic number, and UNKNOWN.",
        "behavior_outcome": "choice_interface_controls_failed",
        "claim_boundary": "Prompt-visible candidate choices break atomic controls and nulls while leaving learned conflict routing weak.",
        "dominant_failure": "direct atomic control and answer-absent null failed under choice interface",
        "prompt_channel": "prompt_visible_choice_options",
        "answer_interface": "choice_label",
        "local_vs_learned_mixture": "choice_interface_confounded",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("synthetic choice local rate", "smoke.selected_metrics.panel_metrics.synthetic_choice_lookup.local_number_rate"),
            ("atomic choice control atomic rate", "smoke.selected_metrics.panel_metrics.atomic_choice_control.atomic_number_rate"),
            ("answer-absent choice UNKNOWN rate", "smoke.selected_metrics.panel_metrics.answer_absent_choice_null.unknown_rate"),
            ("operation-local choice local rate", "smoke.selected_metrics.panel_metrics.operation_local_choice_conflict.local_number_rate"),
            ("operation-atomic choice atomic rate", "smoke.selected_metrics.panel_metrics.operation_atomic_choice_conflict.atomic_number_rate"),
        ],
    },
    {
        "card_id": "MC026",
        "source": "smoke",
        "title": "Prompt-visible numeric-option arbitration",
        "contract_axis": "numeric_option_answer_interface",
        "repair_attempt": "Show local number, atomic number, and UNKNOWN as allowed answers, but require returning the actual number or UNKNOWN.",
        "behavior_outcome": "numeric_option_atomic_abstention_failed",
        "claim_boundary": "Numeric options preserve nulls but turn direct atomic control into UNKNOWN and leave learned conflict routing weak.",
        "dominant_failure": "direct atomic control abstains and operation-atomic branch remains below gate",
        "prompt_channel": "prompt_visible_numeric_options",
        "answer_interface": "integer_or_unknown_from_options",
        "local_vs_learned_mixture": "numeric_option_confounded",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("synthetic option local rate", "smoke.selected_metrics.panel_metrics.synthetic_option_lookup.local_number_rate"),
            ("atomic option control atomic rate", "smoke.selected_metrics.panel_metrics.atomic_option_control.atomic_number_rate"),
            ("atomic option control UNKNOWN rate", "smoke.selected_metrics.panel_metrics.atomic_option_control.unknown_rate"),
            ("answer-absent option UNKNOWN rate", "smoke.selected_metrics.panel_metrics.answer_absent_option_null.unknown_rate"),
            ("operation-local option local rate", "smoke.selected_metrics.panel_metrics.operation_local_option_conflict.local_number_rate"),
            ("operation-atomic option atomic rate", "smoke.selected_metrics.panel_metrics.operation_atomic_option_conflict.atomic_number_rate"),
        ],
    },
    {
        "card_id": "MC027",
        "source": "smoke",
        "title": "Answer-interface sweep",
        "contract_axis": "answer_interface_law",
        "repair_attempt": "Sweep bare integer, prefix, JSON, A/B/C, and numeric-option answer interfaces on one operation substrate.",
        "behavior_outcome": "bare_integer_smoke_only_structured_interfaces_failed",
        "claim_boundary": "Bare integer clears the 10-source smoke, but structured interfaces distort controls, nulls, or learned routing; MC024 remains the full-source boundary.",
        "dominant_failure": "answer interfaces change behavior strongly and only the inherited bare-integer route survives smoke",
        "prompt_channel": "answer_interface_sweep",
        "answer_interface": "multi_interface",
        "local_vs_learned_mixture": "interface_dependent_smoke_candidate",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("bare integer atomic control rate", "smoke.selected_metrics.named_blocks.interface_variant_atomic_control.bare_integer.atomic_number_rate"),
            ("bare integer operation-atomic rate", "smoke.selected_metrics.named_blocks.interface_variant_operation_atomic.bare_integer.atomic_number_rate"),
            ("answer-prefix operation-atomic rate", "smoke.selected_metrics.named_blocks.interface_variant_operation_atomic.answer_prefix.atomic_number_rate"),
            ("JSON answer-absent UNKNOWN rate", "smoke.selected_metrics.named_blocks.interface_variant_answer_absent.json_answer.unknown_rate"),
            ("letter-choice operation-atomic rate", "smoke.selected_metrics.named_blocks.interface_variant_operation_atomic.letter_choices.atomic_number_rate"),
            ("numeric-option atomic control rate", "smoke.selected_metrics.named_blocks.interface_variant_atomic_control.numeric_options.atomic_number_rate"),
            ("numeric-option operation-atomic rate", "smoke.selected_metrics.named_blocks.interface_variant_operation_atomic.numeric_options.atomic_number_rate"),
        ],
    },
    {
        "card_id": "MC028",
        "source": "smoke",
        "title": "Bare-integer full-source boundary",
        "contract_axis": "full_source_boundary",
        "repair_attempt": "Run MC027's winning bare-integer answer interface on the full 40-source set.",
        "behavior_outcome": "bare_integer_full_source_atomic_other_number_leak",
        "claim_boundary": "The bare-integer smoke survivor fails full-source promotion: controls and nulls stay clean, but operation-atomic rows fall below gate with other-number leakage.",
        "dominant_failure": "full-source learned atomic branch below gate with other-number leakage",
        "prompt_channel": "bare_integer_full_source",
        "answer_interface": "bare_integer",
        "local_vs_learned_mixture": "full_source_boundary_failed",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("familiar lookup local rate", "smoke.selected_metrics.panel_metrics.familiar_interface_lookup.local_number_rate"),
            ("atomic control atomic rate", "smoke.selected_metrics.panel_metrics.atomic_interface_control.atomic_number_rate"),
            ("answer-absent UNKNOWN rate", "smoke.selected_metrics.panel_metrics.answer_absent_interface_null.unknown_rate"),
            ("operation-local local rate", "smoke.selected_metrics.panel_metrics.operation_local_interface_conflict.local_number_rate"),
            ("operation-atomic atomic rate", "smoke.selected_metrics.panel_metrics.operation_atomic_interface_conflict.atomic_number_rate"),
            ("operation-atomic other-number rate", "smoke.selected_metrics.panel_metrics.operation_atomic_interface_conflict.other_number_rate"),
        ],
    },
    {
        "card_id": "MC029",
        "source": "smoke",
        "title": "Operation leak factorial",
        "contract_axis": "factorized_leak_boundary",
        "repair_attempt": "Factor MC028's other-number leak across numeric examples, label-only examples, rules-only, query-before-example ordering, and query-row-last table ordering.",
        "behavior_outcome": "rules_only_improves_atomic_branch_but_nulls_fail",
        "claim_boundary": "No variant passes the bridge gate. Rules-only improves operation-atomic routing to 0.875 and removes worked-example outputs, but answer-absent nulls fall to 0.806; query-before-examples amplifies other-number leakage.",
        "dominant_failure": "branch/null tradeoff after worked-example removal",
        "prompt_channel": "operation_factorial",
        "answer_interface": "bare_integer",
        "local_vs_learned_mixture": "factorial_branch_null_tradeoff",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("baseline operation-atomic rate", "smoke.selected_metrics.named_blocks.variant_operation_atomic.baseline_numeric_examples.atomic_number_rate"),
            ("baseline other-number rate", "smoke.selected_metrics.named_blocks.variant_operation_atomic.baseline_numeric_examples.other_number_rate"),
            ("rules-only operation-atomic rate", "smoke.selected_metrics.named_blocks.variant_operation_atomic.rules_only.atomic_number_rate"),
            ("rules-only answer-absent UNKNOWN rate", "smoke.selected_metrics.named_blocks.variant_answer_absent.rules_only.unknown_rate"),
            ("query-before operation-atomic rate", "smoke.selected_metrics.named_blocks.variant_operation_atomic.query_before_examples.atomic_number_rate"),
            ("query-before other-number rate", "smoke.selected_metrics.named_blocks.variant_operation_atomic.query_before_examples.other_number_rate"),
            ("query-row-last other-number rate", "smoke.selected_metrics.named_blocks.variant_operation_atomic.query_row_last.other_number_rate"),
        ],
    },
    {
        "card_id": "MC030",
        "source": "smoke",
        "title": "Null-preserving rules repair",
        "contract_axis": "null_preserving_guard_repair",
        "repair_attempt": "Add explicit absence guards to MC029's rules-only prompt while preserving full-source panels.",
        "behavior_outcome": "absence_guards_worsen_nulls",
        "claim_boundary": "No guard variant passes. The unguarded rules-only baseline remains the best branch/null compromise; guards either worsen nulls or collapse learned atomic routing.",
        "dominant_failure": "branch/null tradeoff persists under absence guards",
        "prompt_channel": "operation_absence_guard",
        "answer_interface": "bare_integer",
        "local_vs_learned_mixture": "guarded_branch_null_tradeoff",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("baseline operation-atomic rate", "smoke.selected_metrics.named_blocks.variant_operation_atomic.rules_only_baseline.atomic_number_rate"),
            ("baseline answer-absent UNKNOWN rate", "smoke.selected_metrics.named_blocks.variant_answer_absent.rules_only_baseline.unknown_rate"),
            ("row-guard operation-atomic rate", "smoke.selected_metrics.named_blocks.variant_operation_atomic.row_absence_guard_before_rules.atomic_number_rate"),
            ("row-guard answer-absent UNKNOWN rate", "smoke.selected_metrics.named_blocks.variant_answer_absent.row_absence_guard_before_rules.unknown_rate"),
            ("decision-order answer-absent UNKNOWN rate", "smoke.selected_metrics.named_blocks.variant_answer_absent.decision_order_guard_after_query.unknown_rate"),
            ("query-last operation-atomic rate", "smoke.selected_metrics.named_blocks.variant_operation_atomic.decision_order_guard_query_last.atomic_number_rate"),
            ("query-last other-number rate", "smoke.selected_metrics.named_blocks.variant_operation_atomic.decision_order_guard_query_last.other_number_rate"),
        ],
    },
    {
        "card_id": "MC031",
        "source": "smoke",
        "title": "Statusless reliability bridge",
        "contract_axis": "statusless_checksum_reliability",
        "repair_attempt": "Replace visible status labels and operation examples with arithmetic checksum validity as a source-reliability cue.",
        "behavior_outcome": "statusless_checksum_invalid_branch_local_collapse",
        "claim_boundary": "Direct controls, valid-checksum local rows, and answer-absent nulls are clean in smoke, but invalid-checksum rows select local numbers on every selected conflict.",
        "dominant_failure": "statusless invalid-source branch collapses to local despite available atomic recall",
        "prompt_channel": "statusless_checksum_rule",
        "answer_interface": "bare_integer",
        "local_vs_learned_mixture": "statusless_reliability_local_collapse",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("synthetic lookup local rate", "smoke.selected_metrics.panel_metrics.synthetic_numeric_lookup.local_number_rate"),
            ("real atomic control rate", "smoke.selected_metrics.panel_metrics.real_world_atomic_number_control.atomic_number_rate"),
            ("answer-absent UNKNOWN rate", "smoke.selected_metrics.panel_metrics.answer_absent_null.unknown_rate"),
            ("valid-checksum local rate", "smoke.selected_metrics.panel_metrics.checksum_valid_conflict.local_number_rate"),
            ("invalid-checksum local rate", "smoke.selected_metrics.panel_metrics.checksum_invalid_conflict.local_number_rate"),
            ("invalid-checksum atomic/lure rate", "smoke.selected_metrics.panel_metrics.checksum_invalid_conflict.atomic_or_lure_number_rate"),
        ],
    },
    {
        "card_id": "MC032",
        "source": "smoke",
        "title": "Post-checksum cross-table consistency",
        "contract_axis": "statusless_cross_table_consistency",
        "repair_attempt": "Replace arithmetic checksum validity with agreement between two neutral local tables.",
        "behavior_outcome": "cross_table_mismatch_local_collapse",
        "claim_boundary": "Direct controls, answer-absent nulls, and side-number locality are clean in smoke, but mismatch rows still avoid the learned atomic branch.",
        "dominant_failure": "statusless cross-table mismatch collapses toward the primary local number",
        "prompt_channel": "statusless_cross_table_rule",
        "answer_interface": "integer",
        "local_vs_learned_mixture": "statusless_reliability_local_collapse",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("synthetic lookup local rate", "smoke.selected_metrics.panel_metrics.synthetic_numeric_lookup.local_number_rate"),
            ("real atomic control rate", "smoke.selected_metrics.panel_metrics.real_world_atomic_number_control.atomic_number_rate"),
            ("answer-absent UNKNOWN rate", "smoke.selected_metrics.panel_metrics.answer_absent_null.unknown_rate"),
            ("match-conflict local rate", "smoke.selected_metrics.panel_metrics.crosscheck_match_conflict.local_number_rate"),
            ("mismatch-conflict local rate", "smoke.selected_metrics.panel_metrics.crosscheck_mismatch_conflict.local_number_rate"),
            ("mismatch-conflict atomic/lure rate", "smoke.selected_metrics.panel_metrics.crosscheck_mismatch_conflict.atomic_or_lure_number_rate"),
            ("mismatch side-number rate", "smoke.selected_metrics.panel_metrics.crosscheck_mismatch_conflict.side_number_rate"),
        ],
    },
    {
        "card_id": "MC033",
        "source": "smoke",
        "title": "Row-local fact-claim comparison",
        "contract_axis": "learned_fact_claim_validity",
        "repair_attempt": "Replace checksum and cross-table cues with a row-local standard-number claim checked against learned atomic memory.",
        "behavior_outcome": "fact_claim_match_and_mismatch_failed",
        "claim_boundary": "Direct controls and answer-absent nulls are clean in smoke, but the match branch returns atomic answers too often and mismatch rows split between local and the wrong claimed number.",
        "dominant_failure": "fact-claim validity does not produce stable local-versus-learned routing",
        "prompt_channel": "statusless_fact_claim_rule",
        "answer_interface": "integer",
        "local_vs_learned_mixture": "statusless_reliability_local_and_claim_leak",
        "hidden_state_allowed": False,
        "headline_metrics": [
            ("synthetic lookup local rate", "smoke.selected_metrics.panel_metrics.synthetic_numeric_lookup.local_number_rate"),
            ("real atomic control rate", "smoke.selected_metrics.panel_metrics.real_world_atomic_number_control.atomic_number_rate"),
            ("answer-absent UNKNOWN rate", "smoke.selected_metrics.panel_metrics.answer_absent_null.unknown_rate"),
            ("match-conflict local rate", "smoke.selected_metrics.panel_metrics.fact_claim_match_conflict.local_number_rate"),
            ("match-conflict atomic rate", "smoke.selected_metrics.panel_metrics.fact_claim_match_conflict.atomic_number_rate"),
            ("mismatch-conflict atomic rate", "smoke.selected_metrics.panel_metrics.fact_claim_mismatch_conflict.atomic_number_rate"),
            ("mismatch-conflict local rate", "smoke.selected_metrics.panel_metrics.fact_claim_mismatch_conflict.local_number_rate"),
            ("mismatch claimed-number/lure rate", "smoke.selected_metrics.panel_metrics.fact_claim_mismatch_conflict.lure_atomic_number_rate"),
        ],
    },
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def dotted_get(payload: Any, dotted_path: str) -> Any | None:
    current = payload
    for part in dotted_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def primitive(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return repr(value)


def load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        load_json(ARTIFACT_INDEX_PATH),
        load_json(COMPARISON_PATH),
        load_json(SMOKE_DIAGNOSTICS_PATH),
    )


def artifact_by_row(artifact_index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for artifact in artifact_index.get("artifacts", []):
        row_id = artifact.get("row_id")
        if row_id:
            result[row_id] = artifact
    return result


def row_summary_by_id(comparison: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in comparison.get("row_summaries", [])}


def smoke_by_card(smoke: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {card["card_id"]: card for card in smoke.get("cards", [])}


def resolve_metric(
    metric_path: str,
    artifact: dict[str, Any] | None,
    smoke_card: dict[str, Any] | None,
) -> Any:
    if metric_path.startswith("artifact.metrics."):
        if artifact is None:
            return None
        metric_key = metric_path.removeprefix("artifact.metrics.")
        return artifact.get("metrics", {}).get(metric_key)
    if metric_path.startswith("artifact."):
        if artifact is None:
            return None
        return dotted_get(artifact, metric_path.removeprefix("artifact."))
    if metric_path.startswith("smoke."):
        if smoke_card is None:
            return None
        return dotted_get(smoke_card, metric_path.removeprefix("smoke."))
    raise ValueError(f"unknown metric path {metric_path}")


def build_rung(
    config: dict[str, Any],
    artifacts_by_row: dict[str, dict[str, Any]],
    rows_by_id: dict[str, dict[str, Any]],
    smoke_cards: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    artifact = artifacts_by_row.get(config.get("row_id", ""))
    row_summary = rows_by_id.get(config.get("row_id", ""))
    smoke_card = smoke_cards.get(config["card_id"])
    if config["source"] == "atlas" and (artifact is None or row_summary is None):
        raise AssertionError(f"{config['card_id']}: missing atlas artifact or row summary")
    if config["source"] == "smoke" and smoke_card is None:
        raise AssertionError(f"{config['card_id']}: missing smoke card")

    headline_metrics = []
    for label, metric_path in config["headline_metrics"]:
        headline_metrics.append(
            {
                "label": label,
                "path": metric_path,
                "value": primitive(resolve_metric(metric_path, artifact, smoke_card)),
            }
        )

    diagnostics = []
    if row_summary:
        diagnostics.extend(row_summary.get("diagnostics", []))
    if smoke_card:
        diagnostics.extend(smoke_card.get("exported_diagnostics", []))

    entry = {
        "card_id": config["card_id"],
        "source": config["source"],
        "row_id": config.get("row_id"),
        "title": config["title"],
        "contract_axis": config["contract_axis"],
        "repair_attempt": config["repair_attempt"],
        "behavior_outcome": config["behavior_outcome"],
        "claim_boundary": config["claim_boundary"],
        "dominant_failure": config["dominant_failure"],
        "prompt_channel": config["prompt_channel"],
        "answer_interface": config["answer_interface"],
        "local_vs_learned_mixture": config["local_vs_learned_mixture"],
        "hidden_state_allowed": config["hidden_state_allowed"],
        "behavior_ready": bool(
            (artifact or {}).get("behavior_ready")
            if config["source"] == "atlas"
            else (smoke_card or {}).get("behavior_ready")
        ),
        "signature_ready": bool(
            (artifact or {}).get("signature_ready")
            if config["source"] == "atlas"
            else (smoke_card or {}).get("signature_ready")
        ),
        "diagnostic_class": (
            (artifact or {}).get("diagnostic_class")
            if config["source"] == "atlas"
            else (smoke_card or {}).get("observed_failure_pattern") or (smoke_card or {}).get("diagnostic_class")
        ),
        "evidence_path": (
            (artifact or {}).get("path")
            if config["source"] == "atlas"
            else (smoke_card or {}).get("artifact_path")
        ),
        "status_card_path": (smoke_card or {}).get("status_card_path"),
        "headline_metrics": headline_metrics,
        "diagnostics": sorted(set(diagnostics)),
    }
    return entry


def build_validation_checks(rungs: list[dict[str, Any]], summary: dict[str, Any]) -> list[dict[str, Any]]:
    by_card = {rung["card_id"]: rung for rung in rungs}

    def metric(card_id: str, label: str) -> Any:
        for item in by_card[card_id]["headline_metrics"]:
            if item["label"] == label:
                return item["value"]
        return None

    checks = [
        {
            "id": "no_bridge_rung_allows_hidden_state",
            "card_id": "ALL",
            "actual": summary["hidden_state_allowed_count"],
            "predicate": "== 0",
            "passed": summary["hidden_state_allowed_count"] == 0,
            "why": "The ladder is a behavior-substrate map, not a hidden-state promotion.",
        },
        {
            "id": "exactly_one_prompt_visible_positive_control",
            "card_id": "ALL",
            "actual": summary["behavior_ready_count"],
            "predicate": "== 1",
            "passed": summary["behavior_ready_count"] == 1 and by_card["MC012"]["behavior_ready"],
            "why": "MC012 is the only bridge behavior-ready rung, and it is prompt-visible.",
        },
        {
            "id": "mc011_numeric_controls_but_conflict_local",
            "card_id": "MC011",
            "actual": {
                "real_atomic_control": metric("MC011", "real atomic control rate"),
                "conflict_local": metric("MC011", "conflict local rate"),
                "conflict_atomic": metric("MC011", "conflict atomic rate"),
            },
            "predicate": "real_atomic_control >= 0.85 and conflict_local == 1.0 and conflict_atomic == 0.0",
            "passed": metric("MC011", "real atomic control rate") >= 0.85
            and metric("MC011", "conflict local rate") == 1.0
            and metric("MC011", "conflict atomic rate") == 0.0,
            "why": "Same-format numeric answers repair controls while conflict remains prompt-local.",
        },
        {
            "id": "mc012_prompt_visible_contrast_only",
            "card_id": "MC012",
            "actual": {
                "trusted_local": metric("MC012", "trusted local rate"),
                "untrusted_atomic": metric("MC012", "untrusted atomic rate"),
                "prompt_channel_locality_gate": metric("MC012", "prompt-channel locality gate"),
            },
            "predicate": "trusted_local >= 0.85 and untrusted_atomic >= 0.85 and prompt_channel_locality_gate is false",
            "passed": metric("MC012", "trusted local rate") >= 0.85
            and metric("MC012", "untrusted atomic rate") >= 0.85
            and metric("MC012", "prompt-channel locality gate") is False,
            "why": "The one clean bridge table is carried by visible source-status text.",
        },
        {
            "id": "mc013_status_ablation_kills_contrast",
            "card_id": "MC013",
            "actual": {
                "statused_untrusted_atomic": metric("MC013", "statused untrusted atomic rate"),
                "ablation_local": metric("MC013", "ablation local rate"),
                "ablation_atomic": metric("MC013", "ablation atomic rate"),
            },
            "predicate": "statused_untrusted_atomic >= 0.85 and ablation_local == 1.0 and ablation_atomic == 0.0",
            "passed": metric("MC013", "statused untrusted atomic rate") >= 0.85
            and metric("MC013", "ablation local rate") == 1.0
            and metric("MC013", "ablation atomic rate") == 0.0,
            "why": "Removing the visible status channel collapses the learned branch.",
        },
        {
            "id": "mc016_visible_nonstatus_gate_collapses_atomic",
            "card_id": "MC016",
            "actual": {
                "primary_local": metric("MC016", "primary local rate"),
                "expected_atomic_atomic": metric("MC016", "expected-atomic atomic rate"),
            },
            "predicate": "primary_local == 1.0 and expected_atomic_atomic == 0.0",
            "passed": metric("MC016", "primary local rate") == 1.0
            and metric("MC016", "expected-atomic atomic rate") == 0.0,
            "why": "A visible non-status gate still does not make expected-atomic rows work.",
        },
        {
            "id": "mc020_atomic_recall_not_bottleneck",
            "card_id": "MC020",
            "actual": {
                "query_row_atomic": metric("MC020", "query-row atomic control rate"),
                "route_atomic": metric("MC020", "route-atomic atomic rate"),
            },
            "predicate": "query_row_atomic >= 0.85 and route_atomic < 0.50",
            "passed": metric("MC020", "query-row atomic control rate") >= 0.85
            and metric("MC020", "route-atomic atomic rate") < 0.50,
            "why": "The bridge bottleneck is arbitration, not recall availability.",
        },
        {
            "id": "mc022_semantic_labels_not_enough",
            "card_id": "MC022",
            "actual": {
                "nonlocal_first_visible": metric("MC022", "nonlocal-first visible expected-correct rate"),
                "atomic_source_atomic": metric("MC022", "ATOMIC-source atomic rate"),
            },
            "predicate": "nonlocal_first_visible == 1.0 and atomic_source_atomic < 0.50",
            "passed": metric("MC022", "nonlocal-first visible expected-correct rate") == 1.0
            and metric("MC022", "ATOMIC-source atomic rate") < 0.50,
            "why": "Semantic labels reveal rule-order sensitivity and do not solve learned-branch routing.",
        },
        {
            "id": "mc023_query_operation_controls_clean_but_conflict_below_gate",
            "card_id": "MC023",
            "actual": {
                "synthetic_local": metric("MC023", "synthetic local control rate"),
                "atomic_control": metric("MC023", "atomic control atomic rate"),
                "answer_absent_unknown": metric("MC023", "answer-absent UNKNOWN rate"),
                "operation_local": metric("MC023", "operation-local local rate"),
                "operation_atomic": metric("MC023", "operation-atomic atomic rate"),
            },
            "predicate": "controls/nulls == 1.0 and both operation branches < 0.85",
            "passed": metric("MC023", "synthetic local control rate") == 1.0
            and metric("MC023", "atomic control atomic rate") == 1.0
            and metric("MC023", "answer-absent UNKNOWN rate") == 1.0
            and metric("MC023", "operation-local local rate") < 0.85
            and metric("MC023", "operation-atomic atomic rate") < 0.85,
            "why": "Query-level operation handles clean the controls but do not yet create a behavior-ready bridge.",
        },
        {
            "id": "mc024_fewshot_operations_repair_local_not_atomic",
            "card_id": "MC024",
            "actual": {
                "synthetic_local": metric("MC024", "synthetic local control rate"),
                "atomic_control": metric("MC024", "atomic control atomic rate"),
                "answer_absent_unknown": metric("MC024", "answer-absent UNKNOWN rate"),
                "operation_local": metric("MC024", "operation-local local rate"),
                "operation_atomic": metric("MC024", "operation-atomic atomic rate"),
                "operation_atomic_other": metric("MC024", "operation-atomic other-number rate"),
            },
            "predicate": "controls/nulls == 1.0, local branch >= 0.85, atomic branch < 0.85, other-number leak >= 0.10",
            "passed": metric("MC024", "synthetic local control rate") == 1.0
            and metric("MC024", "atomic control atomic rate") == 1.0
            and metric("MC024", "answer-absent UNKNOWN rate") == 1.0
            and metric("MC024", "operation-local local rate") >= 0.85
            and metric("MC024", "operation-atomic atomic rate") < 0.85
            and metric("MC024", "operation-atomic other-number rate") >= 0.10,
            "why": "Balanced examples repair prompt-local routing but leave learned atomic routing below gate.",
        },
        {
            "id": "mc025_choice_interface_breaks_controls_and_atomic_branch",
            "card_id": "MC025",
            "actual": {
                "synthetic_local": metric("MC025", "synthetic choice local rate"),
                "atomic_control": metric("MC025", "atomic choice control atomic rate"),
                "answer_absent_unknown": metric("MC025", "answer-absent choice UNKNOWN rate"),
                "operation_local": metric("MC025", "operation-local choice local rate"),
                "operation_atomic": metric("MC025", "operation-atomic choice atomic rate"),
            },
            "predicate": "synthetic local == 1.0, atomic control < 0.85, answer-absent unknown < 0.90, local branch >= 0.85, atomic branch < 0.85",
            "passed": metric("MC025", "synthetic choice local rate") == 1.0
            and metric("MC025", "atomic choice control atomic rate") < 0.85
            and metric("MC025", "answer-absent choice UNKNOWN rate") < 0.90
            and metric("MC025", "operation-local choice local rate") >= 0.85
            and metric("MC025", "operation-atomic choice atomic rate") < 0.85,
            "why": "Constraining the output to prompt-visible choices introduces new control/null failures and does not rescue learned routing.",
        },
        {
            "id": "mc026_numeric_options_preserve_nulls_but_break_atomic_control",
            "card_id": "MC026",
            "actual": {
                "synthetic_local": metric("MC026", "synthetic option local rate"),
                "atomic_control": metric("MC026", "atomic option control atomic rate"),
                "atomic_control_unknown": metric("MC026", "atomic option control UNKNOWN rate"),
                "answer_absent_unknown": metric("MC026", "answer-absent option UNKNOWN rate"),
                "operation_local": metric("MC026", "operation-local option local rate"),
                "operation_atomic": metric("MC026", "operation-atomic option atomic rate"),
            },
            "predicate": "synthetic local == 1.0, atomic control == 0.0, atomic-control UNKNOWN >= 0.90, nulls == 1.0, local branch >= 0.85, atomic branch < 0.85",
            "passed": metric("MC026", "synthetic option local rate") == 1.0
            and metric("MC026", "atomic option control atomic rate") == 0.0
            and metric("MC026", "atomic option control UNKNOWN rate") >= 0.90
            and metric("MC026", "answer-absent option UNKNOWN rate") == 1.0
            and metric("MC026", "operation-local option local rate") >= 0.85
            and metric("MC026", "operation-atomic option atomic rate") < 0.85,
            "why": "Numeric options avoid MC025's null failure but create atomic abstention and do not rescue learned routing.",
        },
        {
            "id": "mc027_answer_interface_sweep_bare_integer_only_smoke_survivor",
            "card_id": "MC027",
            "actual": {
                "bare_atomic_control": metric("MC027", "bare integer atomic control rate"),
                "bare_operation_atomic": metric("MC027", "bare integer operation-atomic rate"),
                "prefix_operation_atomic": metric("MC027", "answer-prefix operation-atomic rate"),
                "json_answer_absent_unknown": metric("MC027", "JSON answer-absent UNKNOWN rate"),
                "choice_operation_atomic": metric("MC027", "letter-choice operation-atomic rate"),
                "numeric_atomic_control": metric("MC027", "numeric-option atomic control rate"),
                "numeric_operation_atomic": metric("MC027", "numeric-option operation-atomic rate"),
            },
            "predicate": "bare integer passes smoke while structured/option interfaces fail controls, nulls, or learned branch",
            "passed": metric("MC027", "bare integer atomic control rate") >= 0.85
            and metric("MC027", "bare integer operation-atomic rate") >= 0.85
            and metric("MC027", "answer-prefix operation-atomic rate") < 0.50
            and metric("MC027", "JSON answer-absent UNKNOWN rate") < 0.10
            and metric("MC027", "letter-choice operation-atomic rate") < 0.25
            and metric("MC027", "numeric-option atomic control rate") == 0.0
            and metric("MC027", "numeric-option operation-atomic rate") < 0.25,
            "why": "Answer format itself is a behavior surface; structured schemas are not neutral output repairs.",
        },
        {
            "id": "mc028_bare_integer_full_source_boundary_failed",
            "card_id": "MC028",
            "actual": {
                "familiar_lookup": metric("MC028", "familiar lookup local rate"),
                "atomic_control": metric("MC028", "atomic control atomic rate"),
                "answer_absent_unknown": metric("MC028", "answer-absent UNKNOWN rate"),
                "operation_local": metric("MC028", "operation-local local rate"),
                "operation_atomic": metric("MC028", "operation-atomic atomic rate"),
                "operation_atomic_other": metric("MC028", "operation-atomic other-number rate"),
            },
            "predicate": "controls/nulls == 1.0, local branch >= 0.85, atomic branch < 0.85, other-number leak >= 0.10",
            "passed": metric("MC028", "familiar lookup local rate") == 1.0
            and metric("MC028", "atomic control atomic rate") == 1.0
            and metric("MC028", "answer-absent UNKNOWN rate") == 1.0
            and metric("MC028", "operation-local local rate") >= 0.85
            and metric("MC028", "operation-atomic atomic rate") < 0.85
            and metric("MC028", "operation-atomic other-number rate") >= 0.10,
            "why": "The bare-integer smoke survivor fails full-source promotion at the learned atomic branch.",
        },
        {
            "id": "mc029_factorial_branch_null_tradeoff",
            "card_id": "MC029",
            "actual": {
                "baseline_operation_atomic": metric("MC029", "baseline operation-atomic rate"),
                "baseline_other": metric("MC029", "baseline other-number rate"),
                "rules_only_operation_atomic": metric("MC029", "rules-only operation-atomic rate"),
                "rules_only_answer_absent_unknown": metric("MC029", "rules-only answer-absent UNKNOWN rate"),
                "query_before_operation_atomic": metric("MC029", "query-before operation-atomic rate"),
                "query_before_other": metric("MC029", "query-before other-number rate"),
                "query_row_last_other": metric("MC029", "query-row-last other-number rate"),
            },
            "predicate": "rules-only improves atomic branch but fails nulls; query-before amplifies leak; query-row-last reduces other numbers without solving routing",
            "passed": metric("MC029", "baseline operation-atomic rate") < 0.85
            and metric("MC029", "baseline other-number rate") >= 0.20
            and metric("MC029", "rules-only operation-atomic rate") >= 0.85
            and metric("MC029", "rules-only answer-absent UNKNOWN rate") < 0.90
            and metric("MC029", "query-before operation-atomic rate") < 0.50
            and metric("MC029", "query-before other-number rate") >= 0.40
            and metric("MC029", "query-row-last other-number rate") < 0.10,
            "why": "Factorizing MC028 moves individual failure axes but does not create a complete bridge substrate.",
        },
        {
            "id": "mc030_absence_guard_repair_fails",
            "card_id": "MC030",
            "actual": {
                "baseline_operation_atomic": metric("MC030", "baseline operation-atomic rate"),
                "baseline_answer_absent_unknown": metric("MC030", "baseline answer-absent UNKNOWN rate"),
                "row_guard_operation_atomic": metric("MC030", "row-guard operation-atomic rate"),
                "row_guard_answer_absent_unknown": metric("MC030", "row-guard answer-absent UNKNOWN rate"),
                "decision_order_answer_absent_unknown": metric("MC030", "decision-order answer-absent UNKNOWN rate"),
                "query_last_operation_atomic": metric("MC030", "query-last operation-atomic rate"),
                "query_last_other": metric("MC030", "query-last other-number rate"),
            },
            "predicate": "baseline branch holds but nulls fail; guards do not restore nulls; query-last reduces other leakage by collapsing branch",
            "passed": metric("MC030", "baseline operation-atomic rate") >= 0.85
            and metric("MC030", "baseline answer-absent UNKNOWN rate") < 0.90
            and metric("MC030", "row-guard operation-atomic rate") >= 0.85
            and metric("MC030", "row-guard answer-absent UNKNOWN rate") < 0.50
            and metric("MC030", "decision-order answer-absent UNKNOWN rate") < 0.50
            and metric("MC030", "query-last other-number rate") < 0.10
            and metric("MC030", "query-last operation-atomic rate") < 0.50,
            "why": "MC030 closes simple absence-guard repair for the MC029 rules-only tradeoff.",
        },
        {
            "id": "mc031_statusless_checksum_invalid_branch_collapses",
            "card_id": "MC031",
            "actual": {
                "synthetic_lookup": metric("MC031", "synthetic lookup local rate"),
                "atomic_control": metric("MC031", "real atomic control rate"),
                "answer_absent_unknown": metric("MC031", "answer-absent UNKNOWN rate"),
                "valid_checksum_local": metric("MC031", "valid-checksum local rate"),
                "invalid_checksum_local": metric("MC031", "invalid-checksum local rate"),
                "invalid_checksum_atomic_lure": metric("MC031", "invalid-checksum atomic/lure rate"),
            },
            "predicate": "controls/nulls == 1.0, valid branch local >= 0.85, invalid branch local == 1.0 and atomic/lure == 0.0",
            "passed": metric("MC031", "synthetic lookup local rate") == 1.0
            and metric("MC031", "real atomic control rate") == 1.0
            and metric("MC031", "answer-absent UNKNOWN rate") == 1.0
            and metric("MC031", "valid-checksum local rate") >= 0.85
            and metric("MC031", "invalid-checksum local rate") == 1.0
            and metric("MC031", "invalid-checksum atomic/lure rate") == 0.0,
            "why": "MC031 shows statusless checksum reliability preserves controls but does not create learned-branch routing.",
        },
        {
            "id": "mc032_crosstable_mismatch_collapses_toward_local",
            "card_id": "MC032",
            "actual": {
                "synthetic_lookup": metric("MC032", "synthetic lookup local rate"),
                "atomic_control": metric("MC032", "real atomic control rate"),
                "answer_absent_unknown": metric("MC032", "answer-absent UNKNOWN rate"),
                "mismatch_local": metric("MC032", "mismatch-conflict local rate"),
                "mismatch_atomic_lure": metric("MC032", "mismatch-conflict atomic/lure rate"),
                "mismatch_side": metric("MC032", "mismatch side-number rate"),
            },
            "predicate": "controls/nulls == 1.0, mismatch local >= 0.70, atomic/lure == 0.0, side-number == 0.0",
            "passed": metric("MC032", "synthetic lookup local rate") == 1.0
            and metric("MC032", "real atomic control rate") == 1.0
            and metric("MC032", "answer-absent UNKNOWN rate") == 1.0
            and metric("MC032", "mismatch-conflict local rate") >= 0.70
            and metric("MC032", "mismatch-conflict atomic/lure rate") == 0.0
            and metric("MC032", "mismatch side-number rate") == 0.0,
            "why": "MC032 shows that replacing checksum validity with cross-table consistency still does not create learned-branch routing.",
        },
        {
            "id": "mc033_fact_claim_route_fails_both_branches",
            "card_id": "MC033",
            "actual": {
                "synthetic_lookup": metric("MC033", "synthetic lookup local rate"),
                "atomic_control": metric("MC033", "real atomic control rate"),
                "answer_absent_unknown": metric("MC033", "answer-absent UNKNOWN rate"),
                "match_local": metric("MC033", "match-conflict local rate"),
                "match_atomic": metric("MC033", "match-conflict atomic rate"),
                "mismatch_atomic": metric("MC033", "mismatch-conflict atomic rate"),
                "mismatch_local": metric("MC033", "mismatch-conflict local rate"),
                "mismatch_lure": metric("MC033", "mismatch claimed-number/lure rate"),
            },
            "predicate": "controls/nulls == 1.0, match local < 0.85, mismatch atomic < 0.85, mismatch lure >= 0.40",
            "passed": metric("MC033", "synthetic lookup local rate") == 1.0
            and metric("MC033", "real atomic control rate") == 1.0
            and metric("MC033", "answer-absent UNKNOWN rate") == 1.0
            and metric("MC033", "match-conflict local rate") < 0.85
            and metric("MC033", "mismatch-conflict atomic rate") < 0.85
            and metric("MC033", "mismatch claimed-number/lure rate") >= 0.40,
            "why": "MC033 closes the post-MC032 fact-claim repair: controls and nulls survive, but the routing rule does not.",
        },
    ]
    return checks


def build_control_surface_bridge_ladder() -> dict[str, Any]:
    artifact_index, comparison, smoke = load_inputs()
    artifacts_by_row = artifact_by_row(artifact_index)
    rows_by_id = row_summary_by_id(comparison)
    smoke_cards = smoke_by_card(smoke)
    rungs = [
        build_rung(config, artifacts_by_row, rows_by_id, smoke_cards)
        for config in RUNG_CONFIGS
    ]
    outcome_counts = Counter(rung["behavior_outcome"] for rung in rungs)
    axis_counts = Counter(rung["contract_axis"] for rung in rungs)
    mixture_counts = Counter(rung["local_vs_learned_mixture"] for rung in rungs)
    summary = {
        "rung_count": len(rungs),
        "atlas_rung_count": sum(1 for rung in rungs if rung["source"] == "atlas"),
        "smoke_rung_count": sum(1 for rung in rungs if rung["source"] == "smoke"),
        "behavior_ready_count": sum(1 for rung in rungs if rung["behavior_ready"]),
        "signature_ready_count": sum(1 for rung in rungs if rung["signature_ready"]),
        "hidden_state_allowed_count": sum(1 for rung in rungs if rung["hidden_state_allowed"]),
        "clean_unconfounded_bridge_count": sum(
            1 for rung in rungs if rung["local_vs_learned_mixture"] == "clean_unconfounded"
        ),
        "prompt_visible_positive_control_count": sum(
            1 for rung in rungs if rung["local_vs_learned_mixture"] == "clean_prompt_visible"
        ),
        "behavior_outcome_counts": dict(sorted(outcome_counts.items())),
        "contract_axis_counts": dict(sorted(axis_counts.items())),
        "local_vs_learned_mixture_counts": dict(sorted(mixture_counts.items())),
    }
    checks = build_validation_checks(rungs, summary)
    return {
        "schema_version": 1,
        "purpose": (
            "Cumulative bridge ladder for MC010-MC033, combining validated atlas "
            "rows and smoke diagnostics without promoting smoke results to atlas rows."
        ),
        "sources": {
            "artifact_index": rel(ARTIFACT_INDEX_PATH),
            "comparison": rel(COMPARISON_PATH),
            "smoke_diagnostics": rel(SMOKE_DIAGNOSTICS_PATH),
        },
        "rungs": rungs,
        "summary": summary,
        "validation_checks": checks,
        "allowed_claim": (
            "Across MC010-MC033, the only clean local-versus-learned bridge contrast "
            "is MC012's prompt-visible status-label positive control; every attempted "
            "non-status or ablated repair remains blocked before hidden-state work. "
            "MC024 shows that balanced worked examples can repair the prompt-local "
            "branch while the learned atomic branch remains the limiting failure. "
            "MC025 shows that prompt-visible choice constraints add control and null "
            "failures rather than repairing that branch. MC026 shows that numeric "
            "option lists preserve nulls but turn direct atomic recall into UNKNOWN. "
            "MC027 shows that answer-interface format is itself a strong behavior "
            "surface: bare integer answers survive 10-source smoke, while prefix, "
            "JSON, choice, and numeric-option interfaces fail different gates. "
            "MC028 closes that survivor at full-source scale: controls and nulls "
            "stay clean, but the learned atomic branch falls below gate with "
            "other-number leakage. MC029 shows that removing numeric examples can "
            "improve the learned branch, but the resulting rules-only variant breaks "
            "answer-absent nulls; factor movement is not bridge repair. MC030 shows "
            "that explicit absence guards do not repair that tradeoff and can make "
            "answer-absent rows substantially worse. MC031 tests a materially "
            "different statusless checksum cue: direct controls and nulls stay clean, "
            "but invalid-checksum conflict rows still collapse to local answers. "
            "MC032 then replaces checksum validity with cross-table consistency and "
            "finds the same broader boundary: mismatch rows still avoid the learned "
            "atomic branch while side-number leakage stays absent. MC033 then tests "
            "row-local fact claims against learned atomic memory and closes the "
            "post-MC032 repair route: direct controls and nulls stay clean, but the "
            "match and mismatch branches do not implement stable routing."
        ),
        "forbidden_claim": (
            "This ladder does not establish a hidden signature, causal intervention, "
            "or deployable knowledge-control mechanism."
        ),
    }


def validate_bridge_ladder(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("bridge ladder schema_version must be 1")
    expected_ids = [config["card_id"] for config in RUNG_CONFIGS]
    actual_ids = [rung.get("card_id") for rung in payload.get("rungs", [])]
    if actual_ids != expected_ids:
        raise AssertionError(f"bridge ladder ids expected {expected_ids}, got {actual_ids}")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"bridge ladder source missing: {rel_path}")
    for rung in payload["rungs"]:
        if rung["hidden_state_allowed"]:
            raise AssertionError(f"{rung['card_id']}: hidden_state_allowed must be false")
        if rung["source"] == "atlas" and not rung.get("row_id"):
            raise AssertionError(f"{rung['card_id']}: atlas rung missing row_id")
        evidence_path = rung.get("evidence_path")
        if not evidence_path or not (ROOT / evidence_path).exists():
            raise AssertionError(f"{rung['card_id']}: missing evidence path {evidence_path}")
        missing_metrics = [m for m in rung["headline_metrics"] if m["value"] is None]
        if missing_metrics:
            raise AssertionError(f"{rung['card_id']}: missing headline metrics {missing_metrics}")
    failed_checks = [check for check in payload.get("validation_checks", []) if not check.get("passed")]
    if failed_checks:
        raise AssertionError(f"bridge ladder checks failed: {failed_checks}")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Control-Surface Bridge Ladder",
        "",
        "Date: 2026-07-01",
        "",
        "Status: generated bridge ladder implemented and validated.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/control_surface_bridge_ladder.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_bridge_ladder.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_bridge_ladder.py --write",
        "python code\\control_surface_bridge_ladder.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Purpose",
        "",
        "The bridge ladder turns MC010-MC033 into one cumulative object. It asks",
        "what each repair changed, which behavior gate failed, and what the next",
        "bridge must beat before hidden-state work is allowed.",
        "",
        "## Generated Facts",
        "",
        f"- rungs: {summary['rung_count']};",
        f"- atlas rungs: {summary['atlas_rung_count']};",
        f"- smoke rungs: {summary['smoke_rung_count']};",
        f"- behavior-ready rungs: {summary['behavior_ready_count']};",
        f"- signature-ready rungs: {summary['signature_ready_count']};",
        f"- hidden-state-allowed rungs: {summary['hidden_state_allowed_count']};",
        f"- clean unconfounded bridge rungs: {summary['clean_unconfounded_bridge_count']};",
        f"- prompt-visible positive controls: {summary['prompt_visible_positive_control_count']}.",
        "",
        "## Ladder",
        "",
        "| Card | Axis | Outcome | Mixture | Headline Metrics | Boundary |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rung in payload["rungs"]:
        metrics = "<br>".join(
            f"{item['label']}: `{format_value(item['value'])}`"
            for item in rung["headline_metrics"]
        )
        lines.append(
            f"| `{rung['card_id']}` | `{rung['contract_axis']}` | `{rung['behavior_outcome']}` | "
            f"`{rung['local_vs_learned_mixture']}` | {metrics} | {rung['claim_boundary']} |"
        )
    lines.extend(
        [
            "",
            "## Validation Checks",
            "",
            "| Check | Card | Passed | Actual |",
            "| --- | --- | --- | --- |",
        ]
    )
    for check in payload["validation_checks"]:
        lines.append(
            f"| `{check['id']}` | `{check['card_id']}` | `{str(check['passed']).lower()}` | "
            f"`{format_value(check['actual'])}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The ladder's central finding is narrower and more useful than another",
            "single failed prompt: only MC012 makes the desired local-versus-learned",
            "contrast cleanly, and MC012 does it through visible trusted/untrusted",
            "source text. Removing that text, replacing it with calibration, using",
            "learned parity, using visible non-status features, changing answer",
            "interfaces, adding row codes, or using semantic source labels has not",
            "created an unconfounded behavior substrate. Query-level operation handles",
            "preserve direct controls and nulls in MC023, but still leave both local",
            "and atomic operation-conflict branches below gate quality. Balanced",
            "worked examples in MC024 repair the local operation branch but leave",
            "the learned atomic branch below gate on the full source-disjoint run.",
            "Constrained choices in MC025 fail even earlier: direct atomic control",
            "and answer-absent nulls break while the atomic branch stays weak.",
            "Numeric options in MC026 restore nulls but turn direct atomic control",
            "into UNKNOWN and still leave the atomic branch weak.",
            "MC027 turns the interface problem into a direct sweep: the bare-integer",
            "format is the only 10-source smoke survivor, while answer-prefix, JSON,",
            "A/B/C choice, and numeric-option interfaces each break controls, nulls,",
            "or learned routing.",
            "MC028 then tests the survivor at full-source scale and closes it before",
            "hidden-state work: controls and nulls remain clean, but the learned",
            "atomic operation branch falls below gate with other-number leakage.",
            "",
            "So the current bridge law is:",
            "",
            "> Prompt-local table authority is easy to make clean; learned-memory",
            "> branch selection is easy only when the prompt visibly names the source",
            "> status. Non-status repairs so far either collapse to local outputs,",
            "> become order-sensitive, or produce mixed outputs that do not follow",
            "> the intended rule. Query-level operations improve controls but do not",
            "> yet cross the behavior-gate threshold because the learned branch remains",
            "> the brittle edge under full-source evaluation. Prompt-visible choice",
            "> constraints are not a neutral fix; they introduce their own control",
            "> and null failures. Numeric option lists are not neutral either; they",
            "> can convert learned recall into abstention. More generally, answer",
            "> schemas are behavior surfaces, not passive wrappers around the same",
            "> internal computation. A smoke-passing answer interface is still not a",
            "> substrate until it survives full-source side-effect checks.",
            "",
            "## Claim Boundary",
            "",
            payload["allowed_claim"],
            "",
            payload["forbidden_claim"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write JSON and markdown artifacts")
    args = parser.parse_args()

    payload = build_control_surface_bridge_ladder()
    validate_bridge_ladder(payload)
    if args.write:
        write_json(BRIDGE_LADDER_PATH, payload)
        BRIDGE_LADDER_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        BRIDGE_LADDER_REPORT_PATH.write_text(render_markdown(payload), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "passed": True,
                "rung_count": payload["summary"]["rung_count"],
                "behavior_ready_count": payload["summary"]["behavior_ready_count"],
                "signature_ready_count": payload["summary"]["signature_ready_count"],
                "hidden_state_allowed_count": payload["summary"]["hidden_state_allowed_count"],
                "clean_unconfounded_bridge_count": payload["summary"]["clean_unconfounded_bridge_count"],
                "validation_check_count": len(payload["validation_checks"]),
                "output_path": rel(BRIDGE_LADDER_PATH),
                "report_path": rel(BRIDGE_LADDER_REPORT_PATH),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
