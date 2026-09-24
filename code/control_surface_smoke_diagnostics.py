"""Build the smoke-diagnostic ledger for post-atlas bridge probes.

MC017-MC033 are intentionally not promoted into the main mechanism-card atlas:
they are reduced smoke diagnostics that keep killing bridge substrates before
hidden-state work is allowed. This module makes those typed failures
machine-readable so they can be validated and compared instead of living only
in prose.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SMOKE_DIAGNOSTICS_PATH = ROOT / "data" / "control_surface_smoke_diagnostics.json"
SMOKE_DIAGNOSTICS_REPORT_PATH = ROOT / "research" / "26_CONTROL_SURFACE_SMOKE_DIAGNOSTICS.md"

BOOKKEEPING_FAILED_CRITERIA = {
    "smoke_mode",
    "full_source_count_is_40",
    "candidate_and_output_margins_reported",
}

COMPACT_SUMMARY_KEYS = (
    "rows",
    "parseable",
    "parseable_rate",
    "expected_correct",
    "expected_correct_rate",
    "expected_choice_correct_rate",
    "local_number",
    "local_number_rate",
    "visible_number",
    "visible_number_rate",
    "atomic_number",
    "atomic_number_rate",
    "lure_atomic_number",
    "lure_atomic_number_rate",
    "atomic_or_lure_number",
    "atomic_or_lure_number_rate",
    "side_number",
    "side_number_rate",
    "local_selector",
    "local_selector_rate",
    "atomic_selector",
    "atomic_selector_rate",
    "unknown",
    "unknown_rate",
    "other_number",
    "other_number_rate",
    "unparsed",
    "unparsed_rate",
    "first_listed_choice",
    "first_listed_choice_rate",
    "choice_a",
    "choice_a_rate",
    "choice_b",
    "choice_b_rate",
    "binary_selector_conflict",
    "binary_choice_conflict",
)

COUNT_FIELDS = (
    "label_counts",
    "choice_label_counts",
    "source_label_counts",
    "selected_label_counts",
    "selected_choice_counts",
    "selected_source_counts",
    "expected_label_counts",
    "expected_choice_counts",
    "expected_source_counts",
    "answer_source_label_counts",
    "route_label_counts",
)

SMOKE_CARD_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "card_id": "MC017",
        "title": "Selector-token numeric arbitration",
        "runner_path": "code/mc017_selector_token_numeric_arbitration.py",
        "status_card_path": "research/cards/MC017_SELECTOR_TOKEN_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md",
        "artifact_glob": "results/cards/MC017/*behavior*.json",
        "claim_status": "smoke_blocked_not_atlas_row",
        "failure_axes": [
            "answer_token_collapse",
            "atomic_selector_control_failed",
            "answer_absent_null_failed",
            "local_selector_prior",
        ],
        "exported_diagnostics": [
            "ANSWER_TOKEN_LOCAL_COLLAPSE",
            "ATOMIC_SELECTOR_CONTROL_FAILED",
            "NULL_SELECTOR_LOCAL_COLLAPSE",
        ],
        "lesson": (
            "Asking for LOCAL/ATOMIC source tokens creates its own behavior surface; "
            "the atomic-only selector control collapsed to LOCAL."
        ),
        "next_constraint": "Do not treat source-token answers as cleaner than numeric answers without neutral labels and null controls.",
        "headline_metric_paths": [
            ("atomic selector control atomic rate", "selected_metrics.panel_metrics.atomic_selector_control.atomic_selector_rate"),
            ("answer-absent UNKNOWN rate", "selected_metrics.panel_metrics.answer_absent_null.unknown_rate"),
            ("primary expected-correct rate", "selected_metrics.named_blocks.primary_conflict.expected_correct_rate"),
        ],
    },
    {
        "card_id": "MC018",
        "title": "Counterbalanced selector labels",
        "runner_path": "code/mc018_counterbalanced_selector_labels.py",
        "status_card_path": "research/cards/MC018_COUNTERBALANCED_SELECTOR_LABELS_BEHAVIOR_STATUS.md",
        "artifact_glob": "results/cards/MC018/*behavior*.json",
        "claim_status": "smoke_blocked_not_atlas_row",
        "failure_axes": [
            "neutral_label_rule_following_failed",
            "answer_option_order_bias",
            "definition_order_bias",
            "local_source_salience",
            "answer_absent_null_failed",
        ],
        "exported_diagnostics": [
            "NEUTRAL_SELECTOR_RULE_FAILED",
            "FIRST_LISTED_CHOICE_BIAS",
            "LOCAL_SOURCE_SALIENCE",
            "ANSWER_ABSENT_SELECTOR_NULL_FAILED",
        ],
        "lesson": (
            "Neutral A/B labels repaired the worst LOCAL token collapse but left "
            "source-rule correctness near chance and exposed strong first-listed/local-source pressure."
        ),
        "next_constraint": "Any source-selector bridge must beat neutral label, definition-order, answer-option-order, and null controls.",
        "headline_metric_paths": [
            ("primary expected-correct rate", "selected_metrics.named_blocks.primary_conflict.expected_correct_rate"),
            ("primary first-listed-choice rate", "selected_metrics.named_blocks.primary_conflict.first_listed_choice_rate"),
            ("primary local-selector rate", "selected_metrics.named_blocks.primary_conflict.local_selector_rate"),
            ("answer-absent UNKNOWN rate", "selected_metrics.panel_metrics.answer_absent_null.unknown_rate"),
        ],
    },
    {
        "card_id": "MC019",
        "title": "Row-code numeric arbitration",
        "runner_path": "code/mc019_row_code_numeric_arbitration.py",
        "status_card_path": "research/cards/MC019_ROW_CODE_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md",
        "artifact_glob": "results/cards/MC019/*behavior*.json",
        "claim_status": "smoke_blocked_not_atlas_row",
        "failure_axes": [
            "row_code_rule_following_failed",
            "expected_atomic_route_weak",
            "holdout_conflict_balance_failed",
        ],
        "exported_diagnostics": [
            "ROW_CODE_RULE_FAILED",
            "EXPECTED_ATOMIC_ROUTE_WEAK",
            "ROW_CODE_CONTROLS_CLEAN_CONFLICT_FAILED",
        ],
        "lesson": (
            "Neutral row-local route codes repaired direct controls and nulls, "
            "but expected-atomic rows did not reliably follow the route rule."
        ),
        "next_constraint": "Do not accept clean route-code controls unless expected-atomic rows also follow the route rule under prompt-local table pressure.",
        "headline_metric_paths": [
            ("primary expected-correct rate", "selected_metrics.named_blocks.primary_conflict.expected_correct_rate"),
            ("expected-atomic atomic rate", "selected_metrics.named_blocks.primary_conflict_expected_atomic.atomic_number_rate"),
            ("route-rule-absent UNKNOWN rate", "selected_metrics.panel_metrics.route_rule_absent_conflict.unknown_rate"),
            ("answer-absent UNKNOWN rate", "selected_metrics.panel_metrics.answer_absent_null.unknown_rate"),
        ],
    },
    {
        "card_id": "MC020",
        "title": "Atomic recall under table pressure",
        "runner_path": "code/mc020_atomic_recall_table_pressure.py",
        "status_card_path": "research/cards/MC020_ATOMIC_RECALL_TABLE_PRESSURE_BEHAVIOR_STATUS.md",
        "artifact_glob": "results/cards/MC020/*behavior*.json",
        "claim_status": "smoke_blocked_not_atlas_row",
        "failure_axes": [
            "direct_atomic_recall_available",
            "conditional_route_atomic_failed",
            "prompt_local_branch_dominance",
        ],
        "exported_diagnostics": [
            "DIRECT_ATOMIC_RECALL_SURVIVES_TABLE_PRESSURE",
            "CONDITIONAL_ROUTE_ATOMIC_COLLAPSE",
            "PROMPT_LOCAL_BRANCH_DOMINANCE",
        ],
        "lesson": (
            "Atomic recall survives local-row pressure under direct instructions; "
            "the failure appears when a conditional rule must route away from the local row."
        ),
        "next_constraint": "Optimize the arbitration contract, not basic atomic recall availability.",
        "headline_metric_paths": [
            ("query-row atomic control atomic rate", "selected_metrics.panel_metrics.query_row_atomic_control.atomic_number_rate"),
            ("repeated-query atomic control atomic rate", "selected_metrics.panel_metrics.query_row_repeated_atomic_control.atomic_number_rate"),
            ("route-local local rate", "selected_metrics.panel_metrics.route_local_conflict.local_number_rate"),
            ("route-atomic atomic rate", "selected_metrics.panel_metrics.route_atomic_conflict.atomic_number_rate"),
        ],
    },
    {
        "card_id": "MC021",
        "title": "Visible-versus-learned arbitration",
        "runner_path": "code/mc021_visible_vs_learned_arbitration.py",
        "status_card_path": "research/cards/MC021_VISIBLE_VS_LEARNED_ARBITRATION_BEHAVIOR_STATUS.md",
        "artifact_glob": "results/cards/MC021/*behavior*.json",
        "claim_status": "smoke_blocked_not_atlas_row",
        "failure_axes": [
            "generic_visible_route_instability",
            "learned_branch_worse_than_visible",
            "route_code_substrate_failed",
        ],
        "exported_diagnostics": [
            "GENERIC_VISIBLE_ROUTE_FAILED",
            "LEARNED_BRANCH_ARBITRATION_WEAKER",
            "ROUTE_CODE_SUBSTRATE_FAILED",
        ],
        "lesson": (
            "Route-code arbitration is already weak when both branches are prompt-visible; "
            "learned-memory branches amplify the weakness."
        ),
        "next_constraint": "Do not rely on opaque route codes as a clean arbitration substrate.",
        "headline_metric_paths": [
            ("visible conflict expected-correct rate", "selected_metrics.named_blocks.visible_conflict.expected_correct_rate"),
            ("learned conflict expected-correct rate", "selected_metrics.named_blocks.learned_conflict.expected_correct_rate"),
            ("learned atomic-branch atomic rate", "selected_metrics.panel_metrics.route_learned_p_atomic_conflict.atomic_number_rate"),
        ],
    },
    {
        "card_id": "MC022",
        "title": "Explicit branch-name arbitration",
        "runner_path": "code/mc022_explicit_branch_name_arbitration.py",
        "status_card_path": "research/cards/MC022_EXPLICIT_BRANCH_NAME_ARBITRATION_BEHAVIOR_STATUS.md",
        "artifact_glob": "results/cards/MC022/*behavior*.json",
        "claim_status": "smoke_blocked_not_atlas_row",
        "failure_axes": [
            "semantic_source_label_substrate_failed",
            "rule_order_sensitivity",
            "local_source_salience",
            "learned_memory_branch_collapse",
        ],
        "exported_diagnostics": [
            "SEMANTIC_SOURCE_LABEL_SUBSTRATE_FAILED",
            "RULE_ORDER_SENSITIVITY",
            "LOCAL_SOURCE_SALIENCE",
            "LEARNED_MEMORY_BRANCH_COLLAPSE",
        ],
        "lesson": (
            "Semantic answer-source names do not solve the route problem; nonlocal-first order can "
            "rescue visible-visible routing, but learned-memory routing remains weak."
        ),
        "next_constraint": "Any next bridge must beat semantic source-label, rule-order, local-salience, and learned-branch controls.",
        "headline_metric_paths": [
            ("visible conflict expected-correct rate", "selected_metrics.named_blocks.visible_conflict.expected_correct_rate"),
            ("learned conflict expected-correct rate", "selected_metrics.named_blocks.learned_conflict.expected_correct_rate"),
            ("ATOMIC-source atomic rate", "selected_metrics.panel_metrics.source_learned_atomic_conflict.atomic_number_rate"),
            ("nonlocal-first visible expected-correct rate", "selected_metrics.named_blocks.visible_conflict_by_rule_order.alternate_first.expected_correct_rate"),
        ],
    },
    {
        "card_id": "MC023",
        "title": "Query-operation numeric arbitration",
        "runner_path": "code/mc023_query_operation_numeric_arbitration.py",
        "status_card_path": "research/cards/MC023_QUERY_OPERATION_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md",
        "artifact_glob": "results/cards/MC023/*behavior*.json",
        "claim_status": "smoke_blocked_not_atlas_row",
        "failure_axes": [
            "query_operation_substrate_failed",
            "operation_local_branch_below_gate",
            "operation_atomic_branch_below_gate",
            "direct_controls_clean_conflict_failed",
        ],
        "exported_diagnostics": [
            "QUERY_OPERATION_SUBSTRATE_FAILED",
            "OPERATION_LOCAL_BRANCH_BELOW_GATE",
            "OPERATION_ATOMIC_BRANCH_BELOW_GATE",
            "QUERY_OPERATION_CONTROLS_CLEAN_CONFLICT_FAILED",
        ],
        "lesson": (
            "Query-level operation handles preserve direct controls and nulls, "
            "but both local and atomic operation-conflict branches remain below "
            "behavior-gate quality in the smoke run."
        ),
        "next_constraint": "Do not treat query-level operation handles as a clean bridge unless both local and atomic conflict branches exceed gate thresholds on full source-disjoint runs.",
        "headline_metric_paths": [
            ("synthetic local control rate", "selected_metrics.panel_metrics.synthetic_numeric_lookup.local_number_rate"),
            ("atomic control atomic rate", "selected_metrics.panel_metrics.real_world_atomic_number_control.atomic_number_rate"),
            ("answer-absent UNKNOWN rate", "selected_metrics.panel_metrics.answer_absent_null.unknown_rate"),
            ("operation-local local rate", "selected_metrics.panel_metrics.operation_local_conflict.local_number_rate"),
            ("operation-atomic atomic rate", "selected_metrics.panel_metrics.operation_atomic_conflict.atomic_number_rate"),
        ],
    },
    {
        "card_id": "MC024",
        "title": "Few-shot query-operation numeric arbitration",
        "runner_path": "code/mc024_fewshot_operation_numeric_arbitration.py",
        "status_card_path": "research/cards/MC024_FEWSHOT_OPERATION_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md",
        "artifact_glob": "results/cards/MC024/*behavior*.json",
        "claim_status": "smoke_blocked_not_atlas_row",
        "failure_axes": [
            "fewshot_operation_atomic_branch_failed",
            "worked_examples_controls_clean",
            "worked_examples_local_branch_passed",
            "operation_atomic_other_number_leak",
        ],
        "exported_diagnostics": [
            "FEWSHOT_OPERATION_ATOMIC_BRANCH_FAILED",
            "WORKED_EXAMPLES_CONTROLS_CLEAN",
            "WORKED_EXAMPLES_LOCAL_BRANCH_PASSED",
            "OPERATION_ATOMIC_OTHER_NUMBER_LEAK",
        ],
        "lesson": (
            "Balanced worked examples repair the local operation branch and preserve "
            "all direct controls and nulls, but the learned atomic operation branch "
            "falls below gate on the full source-disjoint run."
        ),
        "next_constraint": "Do not advance query-operation arbitration to hidden-state work until the learned atomic branch clears full-source gates without other-number leakage.",
        "headline_metric_paths": [
            ("synthetic local control rate", "selected_metrics.panel_metrics.synthetic_numeric_lookup.local_number_rate"),
            ("atomic control atomic rate", "selected_metrics.panel_metrics.real_world_atomic_number_control.atomic_number_rate"),
            ("answer-absent UNKNOWN rate", "selected_metrics.panel_metrics.answer_absent_null.unknown_rate"),
            ("operation-local local rate", "selected_metrics.panel_metrics.operation_local_conflict.local_number_rate"),
            ("operation-atomic atomic rate", "selected_metrics.panel_metrics.operation_atomic_conflict.atomic_number_rate"),
            ("operation-atomic other-number rate", "selected_metrics.panel_metrics.operation_atomic_conflict.other_number_rate"),
        ],
    },
    {
        "card_id": "MC025",
        "title": "Choice-interface operation arbitration",
        "runner_path": "code/mc025_choice_interface_operation_arbitration.py",
        "status_card_path": "research/cards/MC025_CHOICE_INTERFACE_OPERATION_ARBITRATION_BEHAVIOR_STATUS.md",
        "artifact_glob": "results/cards/MC025/*behavior*.json",
        "claim_status": "smoke_blocked_not_atlas_row",
        "failure_axes": [
            "choice_interface_atomic_control_failed",
            "choice_interface_answer_absent_null_failed",
            "choice_interface_atomic_branch_failed",
            "prompt_visible_choice_options_not_substrate",
        ],
        "exported_diagnostics": [
            "CHOICE_INTERFACE_ATOMIC_CONTROL_FAILED",
            "CHOICE_INTERFACE_ANSWER_ABSENT_NULL_FAILED",
            "CHOICE_INTERFACE_ATOMIC_BRANCH_FAILED",
            "PROMPT_VISIBLE_CHOICE_OPTIONS_NOT_SUBSTRATE",
        ],
        "lesson": (
            "Constrained A/B/C choices do not repair MC024. They preserve local "
            "lookup and local operation routing, but direct atomic control and "
            "answer-absent nulls fail while the atomic conflict branch collapses "
            "mostly to local choices."
        ),
        "next_constraint": "Do not treat prompt-visible candidate choices as a bridge repair unless direct atomic controls, nulls, and learned-branch conflict rows all pass together.",
        "headline_metric_paths": [
            ("synthetic choice local rate", "selected_metrics.panel_metrics.synthetic_choice_lookup.local_number_rate"),
            ("atomic choice control atomic rate", "selected_metrics.panel_metrics.atomic_choice_control.atomic_number_rate"),
            ("answer-absent choice UNKNOWN rate", "selected_metrics.panel_metrics.answer_absent_choice_null.unknown_rate"),
            ("operation-local choice local rate", "selected_metrics.panel_metrics.operation_local_choice_conflict.local_number_rate"),
            ("operation-atomic choice atomic rate", "selected_metrics.panel_metrics.operation_atomic_choice_conflict.atomic_number_rate"),
        ],
    },
    {
        "card_id": "MC026",
        "title": "Numeric-option operation arbitration",
        "runner_path": "code/mc026_numeric_option_operation_arbitration.py",
        "status_card_path": "research/cards/MC026_NUMERIC_OPTION_OPERATION_ARBITRATION_BEHAVIOR_STATUS.md",
        "artifact_glob": "results/cards/MC026/*behavior*.json",
        "claim_status": "smoke_blocked_not_atlas_row",
        "failure_axes": [
            "numeric_option_atomic_control_abstention",
            "numeric_option_atomic_branch_failed",
            "numeric_options_nulls_clean",
            "prompt_visible_numeric_options_not_substrate",
        ],
        "exported_diagnostics": [
            "NUMERIC_OPTION_ATOMIC_CONTROL_ABSTENTION",
            "NUMERIC_OPTION_ATOMIC_BRANCH_FAILED",
            "NUMERIC_OPTIONS_NULLS_CLEAN",
            "PROMPT_VISIBLE_NUMERIC_OPTIONS_NOT_SUBSTRATE",
        ],
        "lesson": (
            "Returning the listed number rather than an A/B/C label restores nulls "
            "but does not repair the bridge: direct atomic control collapses to "
            "UNKNOWN and the learned atomic conflict branch remains weak."
        ),
        "next_constraint": "Do not treat prompt-visible numeric options as a bridge repair unless direct atomic controls and learned-branch conflict rows pass together.",
        "headline_metric_paths": [
            ("synthetic option local rate", "selected_metrics.panel_metrics.synthetic_option_lookup.local_number_rate"),
            ("atomic option control atomic rate", "selected_metrics.panel_metrics.atomic_option_control.atomic_number_rate"),
            ("atomic option control UNKNOWN rate", "selected_metrics.panel_metrics.atomic_option_control.unknown_rate"),
            ("answer-absent option UNKNOWN rate", "selected_metrics.panel_metrics.answer_absent_option_null.unknown_rate"),
            ("operation-local option local rate", "selected_metrics.panel_metrics.operation_local_option_conflict.local_number_rate"),
            ("operation-atomic option atomic rate", "selected_metrics.panel_metrics.operation_atomic_option_conflict.atomic_number_rate"),
        ],
    },
    {
        "card_id": "MC027",
        "title": "Answer-interface sweep",
        "runner_path": "code/mc027_answer_interface_sweep.py",
        "status_card_path": "research/cards/MC027_ANSWER_INTERFACE_SWEEP_BEHAVIOR_STATUS.md",
        "artifact_glob": "results/cards/MC027/*behavior*.json",
        "claim_status": "smoke_only_not_atlas_row",
        "failure_axes": [
            "answer_interface_dispersion",
            "bare_integer_smoke_only_candidate",
            "structured_interfaces_break_behavior",
            "full_source_boundary_inherited_from_mc024",
        ],
        "exported_diagnostics": [
            "ANSWER_INTERFACE_DISPERSION",
            "BARE_INTEGER_SMOKE_ONLY_CANDIDATE",
            "STRUCTURED_INTERFACES_BREAK_BEHAVIOR",
            "FULL_SOURCE_BOUNDARY_INHERITED_FROM_MC024",
        ],
        "lesson": (
            "Sweeping answer interfaces on the same operation substrate shows that "
            "bare integer answers are the only 10-source smoke survivor; prefix, "
            "JSON, A/B/C choice, and numeric-option interfaces each distort controls "
            "or learned-branch routing."
        ),
        "next_constraint": "Do not treat a structured output schema as neutral; any future interface change must be tested as its own behavior surface before hidden-state work.",
        "headline_metric_paths": [
            ("selected interface", "selected_metrics.named_blocks.interface_variant_overall.bare_integer.rows"),
            ("bare integer atomic control rate", "selected_metrics.named_blocks.interface_variant_atomic_control.bare_integer.atomic_number_rate"),
            ("bare integer operation-atomic rate", "selected_metrics.named_blocks.interface_variant_operation_atomic.bare_integer.atomic_number_rate"),
            ("answer-prefix operation-atomic rate", "selected_metrics.named_blocks.interface_variant_operation_atomic.answer_prefix.atomic_number_rate"),
            ("JSON answer-absent UNKNOWN rate", "selected_metrics.named_blocks.interface_variant_answer_absent.json_answer.unknown_rate"),
            ("letter-choice operation-atomic rate", "selected_metrics.named_blocks.interface_variant_operation_atomic.letter_choices.atomic_number_rate"),
            ("numeric-option atomic control rate", "selected_metrics.named_blocks.interface_variant_atomic_control.numeric_options.atomic_number_rate"),
            ("numeric-option operation-atomic rate", "selected_metrics.named_blocks.interface_variant_operation_atomic.numeric_options.atomic_number_rate"),
        ],
    },
    {
        "card_id": "MC028",
        "title": "Bare-integer full-source boundary",
        "runner_path": "code/mc028_bare_integer_full_source_boundary.py",
        "status_card_path": "research/cards/MC028_BARE_INTEGER_FULL_SOURCE_BOUNDARY_STATUS.md",
        "artifact_glob": "results/cards/MC028/*behavior*.json",
        "claim_status": "full_source_boundary_failed_not_atlas_row",
        "failure_axes": [
            "bare_integer_full_source_atomic_branch_failed",
            "full_source_other_number_leak",
            "full_source_controls_clean",
            "smoke_to_full_source_boundary",
        ],
        "exported_diagnostics": [
            "BARE_INTEGER_FULL_SOURCE_ATOMIC_BRANCH_FAILED",
            "FULL_SOURCE_OTHER_NUMBER_LEAK",
            "FULL_SOURCE_CONTROLS_CLEAN",
            "SMOKE_TO_FULL_SOURCE_BOUNDARY",
        ],
        "lesson": (
            "MC027's bare-integer smoke survivor does not promote to a full-source "
            "bridge: controls and nulls remain clean, but operation-atomic rows fall "
            "below gate with substantial other-number leakage."
        ),
        "next_constraint": "Do not promote a 10-source bridge smoke to hidden-state work without full-source stress and other-number side-effect accounting.",
        "headline_metric_paths": [
            ("familiar lookup local rate", "selected_metrics.panel_metrics.familiar_interface_lookup.local_number_rate"),
            ("atomic control atomic rate", "selected_metrics.panel_metrics.atomic_interface_control.atomic_number_rate"),
            ("answer-absent UNKNOWN rate", "selected_metrics.panel_metrics.answer_absent_interface_null.unknown_rate"),
            ("operation-local local rate", "selected_metrics.panel_metrics.operation_local_interface_conflict.local_number_rate"),
            ("operation-atomic atomic rate", "selected_metrics.panel_metrics.operation_atomic_interface_conflict.atomic_number_rate"),
            ("operation-atomic other-number rate", "selected_metrics.panel_metrics.operation_atomic_interface_conflict.other_number_rate"),
        ],
    },
    {
        "card_id": "MC029",
        "title": "Operation leak factorial",
        "runner_path": "code/mc029_operation_leak_factorial.py",
        "status_card_path": "research/cards/MC029_OPERATION_LEAK_FACTORIAL_STATUS.md",
        "artifact_glob": "results/cards/MC029/*behavior*.json",
        "claim_status": "full_source_factorial_failed_not_atlas_row",
        "failure_axes": [
            "worked_example_output_leak",
            "rules_only_atomic_branch_improves",
            "rules_only_answer_absent_null_fails",
            "query_before_examples_amplifies_example_leak",
            "query_row_last_reduces_other_numbers_but_local_collapses",
            "factorial_branch_null_tradeoff",
        ],
        "exported_diagnostics": [
            "WORKED_EXAMPLE_OUTPUT_LEAK",
            "RULES_ONLY_ATOMIC_BRANCH_IMPROVES",
            "RULES_ONLY_ANSWER_ABSENT_NULL_FAILS",
            "QUERY_BEFORE_EXAMPLES_AMPLIFIES_EXAMPLE_LEAK",
            "QUERY_ROW_LAST_REDUCES_OTHER_NUMBERS_BUT_LOCAL_COLLAPSES",
            "FACTORIAL_BRANCH_NULL_TRADEOFF",
        ],
        "lesson": (
            "Factorizing MC028's leak shows the branch can move without producing a "
            "bridge: removing numeric examples improves operation-atomic routing, "
            "but answer-absent nulls fail; putting the query before examples "
            "amplifies worked-example copying."
        ),
        "next_constraint": "Do not treat example removal, query reordering, or row-order changes as a bridge repair unless learned-branch, local-branch, and null gates all pass together.",
        "headline_metric_paths": [
            ("baseline operation-atomic rate", "selected_metrics.named_blocks.variant_operation_atomic.baseline_numeric_examples.atomic_number_rate"),
            ("baseline other-number rate", "selected_metrics.named_blocks.variant_operation_atomic.baseline_numeric_examples.other_number_rate"),
            ("rules-only operation-atomic rate", "selected_metrics.named_blocks.variant_operation_atomic.rules_only.atomic_number_rate"),
            ("rules-only answer-absent UNKNOWN rate", "selected_metrics.named_blocks.variant_answer_absent.rules_only.unknown_rate"),
            ("query-before operation-atomic rate", "selected_metrics.named_blocks.variant_operation_atomic.query_before_examples.atomic_number_rate"),
            ("query-before other-number rate", "selected_metrics.named_blocks.variant_operation_atomic.query_before_examples.other_number_rate"),
            ("query-row-last other-number rate", "selected_metrics.named_blocks.variant_operation_atomic.query_row_last.other_number_rate"),
        ],
    },
    {
        "card_id": "MC030",
        "title": "Null-preserving rules repair",
        "runner_path": "code/mc030_null_preserving_rules_repair.py",
        "status_card_path": "research/cards/MC030_NULL_PRESERVING_RULES_REPAIR_STATUS.md",
        "artifact_glob": "results/cards/MC030/*behavior*.json",
        "claim_status": "full_source_guard_repair_failed_not_atlas_row",
        "failure_axes": [
            "rules_only_branch_null_tradeoff_persists",
            "absence_guards_worsen_answer_absent_nulls",
            "guarded_query_last_reduces_other_numbers_but_branch_collapses",
            "numeric_list_marker_structural_confounds_avoided",
        ],
        "exported_diagnostics": [
            "RULES_ONLY_BRANCH_NULL_TRADEOFF_PERSISTS",
            "ABSENCE_GUARDS_WORSEN_ANSWER_ABSENT_NULLS",
            "GUARDED_QUERY_LAST_REDUCES_OTHER_NUMBERS_BUT_BRANCH_COLLAPSES",
            "NUMERIC_LIST_MARKERS_CAN_LEAK_ATOMIC_TARGETS",
        ],
        "lesson": (
            "Explicit absence guards do not repair MC029's rules-only tradeoff. "
            "The unguarded baseline remains the best branch/null compromise; "
            "guards reduce some other-number leakage only by damaging nulls or "
            "collapsing learned atomic routing."
        ),
        "next_constraint": "Treat simple absence-guard repair as closed unless a materially different contract preserves operation-atomic routing and answer-absent nulls together.",
        "headline_metric_paths": [
            ("baseline operation-atomic rate", "selected_metrics.named_blocks.variant_operation_atomic.rules_only_baseline.atomic_number_rate"),
            ("baseline answer-absent UNKNOWN rate", "selected_metrics.named_blocks.variant_answer_absent.rules_only_baseline.unknown_rate"),
            ("row-guard operation-atomic rate", "selected_metrics.named_blocks.variant_operation_atomic.row_absence_guard_before_rules.atomic_number_rate"),
            ("row-guard answer-absent UNKNOWN rate", "selected_metrics.named_blocks.variant_answer_absent.row_absence_guard_before_rules.unknown_rate"),
            ("decision-order answer-absent UNKNOWN rate", "selected_metrics.named_blocks.variant_answer_absent.decision_order_guard_after_query.unknown_rate"),
            ("query-last operation-atomic rate", "selected_metrics.named_blocks.variant_operation_atomic.decision_order_guard_query_last.atomic_number_rate"),
            ("query-last other-number rate", "selected_metrics.named_blocks.variant_operation_atomic.decision_order_guard_query_last.other_number_rate"),
        ],
    },
    {
        "card_id": "MC031",
        "title": "Statusless reliability bridge",
        "runner_path": "code/mc031_statusless_reliability_bridge.py",
        "status_card_path": "research/cards/MC031_STATUSLESS_RELIABILITY_BRIDGE_STATUS.md",
        "artifact_glob": "results/cards/MC031/*smoke*.json",
        "claim_status": "statusless_checksum_bridge_smoke_failed_not_atlas_row",
        "failure_axes": [
            "statusless_checksum_invalid_branch_collapses_to_local",
            "checksum_valid_local_branch_clean",
            "direct_controls_and_nulls_clean",
            "atomic_or_lure_conflict_selection_absent",
            "smoke_only_no_candidate_output_baselines",
        ],
        "exported_diagnostics": [
            "STATUSLESS_CHECKSUM_INVALID_BRANCH_LOCAL_COLLAPSE",
            "CHECKSUM_VALID_LOCAL_BRANCH_CLEAN",
            "DIRECT_CONTROLS_AND_NULLS_CLEAN",
            "ATOMIC_OR_LURE_CONFLICT_SELECTION_ABSENT",
            "SMOKE_ONLY_NO_CANDIDATE_OUTPUT_BASELINES",
        ],
        "lesson": (
            "A statusless arithmetic-checksum rule can preserve direct local, "
            "atomic, valid-checksum, and answer-absent controls, but it does not "
            "route invalid-checksum rows to the learned atomic branch. The model "
            "selects local numbers on every selected invalid-checksum conflict."
        ),
        "next_constraint": (
            "Do not treat statusless reliability cues as bridge repair unless "
            "invalid-source rows produce atomic/lure selections on source-disjoint "
            "holdout while local controls and answer-absent nulls stay clean."
        ),
        "headline_metric_paths": [
            ("synthetic lookup local rate", "selected_metrics.panel_metrics.synthetic_numeric_lookup.local_number_rate"),
            ("real atomic control rate", "selected_metrics.panel_metrics.real_world_atomic_number_control.atomic_number_rate"),
            ("answer-absent UNKNOWN rate", "selected_metrics.panel_metrics.answer_absent_null.unknown_rate"),
            ("valid-checksum local rate", "selected_metrics.panel_metrics.checksum_valid_conflict.local_number_rate"),
            ("invalid-checksum local rate", "selected_metrics.panel_metrics.checksum_invalid_conflict.local_number_rate"),
            ("invalid-checksum atomic/lure rate", "selected_metrics.panel_metrics.checksum_invalid_conflict.atomic_or_lure_number_rate"),
        ],
    },
    {
        "card_id": "MC032",
        "title": "Post-checksum cross-table bridge",
        "runner_path": "code/mc032_post_checksum_bridge.py",
        "status_card_path": "research/cards/MC032_POST_CHECKSUM_BRIDGE_STATUS.md",
        "artifact_glob": "results/cards/MC032/*behavior_smoke*.json",
        "claim_status": "post_checksum_crosstable_bridge_smoke_failed_not_atlas_row",
        "failure_axes": [
            "post_checksum_crosstable_mismatch_local_collapse",
            "cross_table_match_branch_below_gate",
            "direct_controls_and_nulls_clean",
            "side_number_leak_absent",
            "candidate_output_baselines_reported",
        ],
        "exported_diagnostics": [
            "POST_CHECKSUM_CROSSTABLE_LOCAL_COLLAPSE",
            "CROSSTABLE_MATCH_BRANCH_BELOW_GATE",
            "DIRECT_CONTROLS_AND_NULLS_CLEAN",
            "CROSSTABLE_SIDE_NUMBER_LEAK_ABSENT",
            "CANDIDATE_OUTPUT_BASELINES_REPORTED",
        ],
        "lesson": (
            "Replacing arithmetic checksum validity with cross-table consistency "
            "does not repair the bridge. Direct local lookup, direct atomic recall, "
            "and answer-absent nulls stay clean, but mismatch rows still avoid the "
            "learned atomic branch and mostly select primary local numbers."
        ),
        "next_constraint": (
            "Treat the post-checksum bridge as failed unless one preregistered repair "
            "makes mismatch rows select learned atomic values while preserving local "
            "controls, nulls, side-number locality, source-disjoint holdout, and "
            "candidate/output margin reporting."
        ),
        "headline_metric_paths": [
            ("synthetic lookup local rate", "selected_metrics.panel_metrics.synthetic_numeric_lookup.local_number_rate"),
            ("real atomic control rate", "selected_metrics.panel_metrics.real_world_atomic_number_control.atomic_number_rate"),
            ("answer-absent UNKNOWN rate", "selected_metrics.panel_metrics.answer_absent_null.unknown_rate"),
            ("match-conflict local rate", "selected_metrics.panel_metrics.crosscheck_match_conflict.local_number_rate"),
            ("mismatch-conflict local rate", "selected_metrics.panel_metrics.crosscheck_mismatch_conflict.local_number_rate"),
            ("mismatch-conflict atomic/lure rate", "selected_metrics.panel_metrics.crosscheck_mismatch_conflict.atomic_or_lure_number_rate"),
            ("mismatch side-number rate", "selected_metrics.panel_metrics.crosscheck_mismatch_conflict.side_number_rate"),
        ],
    },
    {
        "card_id": "MC033",
        "title": "Fact-claim bridge closeout",
        "runner_path": "code/mc033_fact_claim_bridge_closeout.py",
        "status_card_path": "research/cards/MC033_FACT_CLAIM_BRIDGE_CLOSEOUT_STATUS.md",
        "artifact_glob": "results/cards/MC033/*behavior_smoke*.json",
        "claim_status": "post_mc032_fact_claim_bridge_smoke_failed_not_atlas_row",
        "failure_axes": [
            "fact_claim_match_branch_below_gate",
            "fact_claim_mismatch_local_and_claim_leak",
            "direct_controls_and_nulls_clean",
            "candidate_output_baselines_reported",
            "post_mc032_bridge_route_closed",
        ],
        "exported_diagnostics": [
            "FACT_CLAIM_MATCH_BRANCH_FAILED",
            "FACT_CLAIM_MISMATCH_LOCAL_AND_CLAIM_LEAK",
            "DIRECT_CONTROLS_AND_NULLS_CLEAN",
            "CANDIDATE_OUTPUT_BASELINES_REPORTED",
            "POST_MC032_BRIDGE_ROUTE_CLOSED",
        ],
        "lesson": (
            "A row-local fact claim checked against learned atomic memory does not "
            "repair the bridge. Direct controls and nulls stay clean, but the "
            "match branch often returns the atomic number and the mismatch branch "
            "mostly selects local or the wrong claimed number."
        ),
        "next_constraint": (
            "Treat the post-MC032 bridge repair route as closed. Do not add another "
            "same-family reliability cue unless it is materially outside source "
            "labels, row codes, operations, examples, answer schemas, absence "
            "guards, checksum, cross-table consistency, and row-local fact claims."
        ),
        "headline_metric_paths": [
            ("synthetic lookup local rate", "selected_metrics.panel_metrics.synthetic_numeric_lookup.local_number_rate"),
            ("real atomic control rate", "selected_metrics.panel_metrics.real_world_atomic_number_control.atomic_number_rate"),
            ("answer-absent UNKNOWN rate", "selected_metrics.panel_metrics.answer_absent_null.unknown_rate"),
            ("match-conflict local rate", "selected_metrics.panel_metrics.fact_claim_match_conflict.local_number_rate"),
            ("match-conflict atomic rate", "selected_metrics.panel_metrics.fact_claim_match_conflict.atomic_number_rate"),
            ("mismatch-conflict atomic rate", "selected_metrics.panel_metrics.fact_claim_mismatch_conflict.atomic_number_rate"),
            ("mismatch-conflict local rate", "selected_metrics.panel_metrics.fact_claim_mismatch_conflict.local_number_rate"),
            ("mismatch claimed-number/lure rate", "selected_metrics.panel_metrics.fact_claim_mismatch_conflict.lure_atomic_number_rate"),
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


def compact_block(block: Any) -> dict[str, Any]:
    if not isinstance(block, dict):
        return {}
    compact: dict[str, Any] = {}
    for key in COMPACT_SUMMARY_KEYS:
        value = block.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[key] = value
    for key in COUNT_FIELDS:
        value = block.get(key)
        if isinstance(value, dict):
            compact[key] = {
                str(label): count
                for label, count in sorted(value.items())
                if isinstance(count, (int, float, bool))
            }
    return compact


def compact_named_blocks(selected: dict[str, Any]) -> dict[str, Any]:
    named: dict[str, Any] = {}
    for key, value in selected.items():
        if key in {"panels", "rows", "split_row_counts", *COUNT_FIELDS}:
            continue
        if not isinstance(value, dict):
            continue
        if "rows" in value or "expected_correct_rate" in value:
            named[key] = compact_block(value)
            continue
        if value and all(isinstance(item, dict) for item in value.values()):
            nested: dict[str, Any] = {}
            for nested_key, nested_value in sorted(value.items()):
                nested[nested_key] = compact_block(nested_value)
            named[key] = nested
    return named


def latest_artifact(glob_pattern: str) -> Path:
    matches = sorted(ROOT.glob(glob_pattern))
    if not matches:
        raise AssertionError(f"no artifact matches {glob_pattern}")
    return matches[-1]


def failed_criteria(criteria: dict[str, Any]) -> list[str]:
    return sorted(key for key, value in criteria.items() if value is False)


def substantive_failed_criteria(criteria: dict[str, Any]) -> list[str]:
    return [
        key
        for key in failed_criteria(criteria)
        if key not in BOOKKEEPING_FAILED_CRITERIA
    ]


def build_card_entry(config: dict[str, Any]) -> dict[str, Any]:
    artifact_path = latest_artifact(config["artifact_glob"])
    payload = load_json(artifact_path)
    summary = payload.get("summary", {})
    selected = summary.get("selected_template_summary", {})
    criteria = summary.get("criteria", {})
    panel_metrics = {
        panel_name: compact_block(panel_summary)
        for panel_name, panel_summary in sorted(selected.get("panels", {}).items())
    }
    named_blocks = compact_named_blocks(selected)
    entry = {
        "card_id": config["card_id"],
        "title": config["title"],
        "claim_status": config["claim_status"],
        "artifact_path": rel(artifact_path),
        "runner_path": config["runner_path"],
        "status_card_path": config["status_card_path"],
        "run_type": payload.get("run_type"),
        "model_id": payload.get("model_id"),
        "limit_sources": payload.get("limit_sources"),
        "record_count": len(payload.get("records", [])),
        "selected_template": summary.get("selection", {}).get("selected_template"),
        "selection": summary.get("selection", {}),
        "diagnostic_class": summary.get("diagnostic_class"),
        "observed_failure_pattern": summary.get("observed_failure_pattern"),
        "smoke_only": bool(criteria.get("smoke_mode")),
        "structural_passed": bool(criteria.get("structural_passed")),
        "all_controls_passed": criteria.get("all_controls_passed"),
        "behavior_ready": bool(summary.get("behavior_ready")),
        "signature_ready": bool(summary.get("signature_ready")),
        "intervention_ready": bool(summary.get("intervention_ready")),
        "hidden_state_allowed": False,
        "criteria_failed": failed_criteria(criteria),
        "substantive_failed_criteria": substantive_failed_criteria(criteria),
        "baseline_missing_due_to_smoke": criteria.get("candidate_and_output_margins_reported") is False,
        "selected_metrics": {
            "split_row_counts": selected.get("split_row_counts", {}),
            "panel_metrics": panel_metrics,
            "named_blocks": named_blocks,
        },
        "failure_axes": config["failure_axes"],
        "exported_diagnostics": config["exported_diagnostics"],
        "lesson": config["lesson"],
        "next_constraint": config["next_constraint"],
        "headline_metrics": [
            {
                "label": label,
                "path": path,
                "value": dotted_get({"selected_metrics": {"panel_metrics": panel_metrics, "named_blocks": named_blocks}}, path),
            }
            for label, path in config["headline_metric_paths"]
        ],
    }
    return entry


def check_metric(
    entries_by_card: dict[str, dict[str, Any]],
    card_id: str,
    metric_path: str,
    op: str,
    threshold: float,
    check_id: str,
    explanation: str,
) -> dict[str, Any]:
    value = dotted_get(entries_by_card[card_id], metric_path)
    passed = False
    if isinstance(value, (int, float)):
        if op == "<":
            passed = value < threshold
        elif op == "<=":
            passed = value <= threshold
        elif op == ">":
            passed = value > threshold
        elif op == ">=":
            passed = value >= threshold
        elif op == "==":
            passed = value == threshold
        else:
            raise ValueError(f"unknown op {op}")
    return {
        "id": check_id,
        "card_id": card_id,
        "metric_path": metric_path,
        "actual": value,
        "op": op,
        "threshold": threshold,
        "passed": passed,
        "explanation": explanation,
    }


def build_validation_checks(entries: list[dict[str, Any]], summary: dict[str, Any]) -> list[dict[str, Any]]:
    by_card = {entry["card_id"]: entry for entry in entries}
    checks = [
        {
            "id": "all_cards_smoke_blocked",
            "card_id": "ALL",
            "metric_path": "summary.behavior_ready_count/signature_ready_count",
            "actual": {
                "behavior_ready_count": summary["behavior_ready_count"],
                "signature_ready_count": summary["signature_ready_count"],
            },
            "op": "==",
            "threshold": {"behavior_ready_count": 0, "signature_ready_count": 0},
            "passed": summary["behavior_ready_count"] == 0 and summary["signature_ready_count"] == 0,
            "explanation": "The smoke layer must not permit hidden-state work.",
        },
        check_metric(
            by_card,
            "MC017",
            "selected_metrics.panel_metrics.atomic_selector_control.atomic_selector_rate",
            "==",
            0.0,
            "mc017_atomic_selector_collapsed_to_local",
            "The ATOMIC selector control did not select ATOMIC at all.",
        ),
        check_metric(
            by_card,
            "MC018",
            "selected_metrics.named_blocks.primary_conflict.first_listed_choice_rate",
            ">=",
            0.75,
            "mc018_first_listed_choice_bias",
            "Neutral selector labels still show strong first-listed choice pressure.",
        ),
        check_metric(
            by_card,
            "MC018",
            "selected_metrics.named_blocks.primary_conflict.expected_correct_rate",
            "<",
            0.85,
            "mc018_selector_rule_not_followed",
            "Neutral labels do not produce behavior-gate quality rule following.",
        ),
        check_metric(
            by_card,
            "MC019",
            "selected_metrics.named_blocks.primary_conflict_expected_atomic.atomic_number_rate",
            "<",
            0.85,
            "mc019_expected_atomic_route_weak",
            "Expected-atomic route-code rows fail the atomic branch.",
        ),
        check_metric(
            by_card,
            "MC019",
            "selected_metrics.panel_metrics.answer_absent_null.unknown_rate",
            ">=",
            0.90,
            "mc019_nulls_clean_despite_conflict_failure",
            "Row-code nulls are clean, so the route failure is not just parse/null collapse.",
        ),
        check_metric(
            by_card,
            "MC020",
            "selected_metrics.panel_metrics.query_row_atomic_control.atomic_number_rate",
            ">=",
            0.85,
            "mc020_direct_atomic_recall_survives_query_row",
            "Direct atomic recall survives the queried local row.",
        ),
        check_metric(
            by_card,
            "MC020",
            "selected_metrics.panel_metrics.route_atomic_conflict.atomic_number_rate",
            "<",
            0.50,
            "mc020_route_atomic_branch_failed",
            "The atomic branch fails specifically under conditional routing.",
        ),
        check_metric(
            by_card,
            "MC021",
            "selected_metrics.named_blocks.visible_conflict.expected_correct_rate",
            "<",
            0.85,
            "mc021_visible_route_unstable",
            "Opaque route codes fail even when both branches are prompt-visible.",
        ),
        check_metric(
            by_card,
            "MC021",
            "selected_metrics.named_blocks.learned_conflict.expected_correct_rate",
            "<",
            0.85,
            "mc021_learned_route_unstable",
            "The visible-learned route also fails.",
        ),
        check_metric(
            by_card,
            "MC022",
            "selected_metrics.named_blocks.visible_conflict.expected_correct_rate",
            "<",
            0.85,
            "mc022_semantic_visible_route_unstable",
            "Semantic source labels still fail visible-visible arbitration overall.",
        ),
        check_metric(
            by_card,
            "MC022",
            "selected_metrics.panel_metrics.source_learned_atomic_conflict.atomic_number_rate",
            "<",
            0.50,
            "mc022_atomic_source_branch_collapsed",
            "Semantic ATOMIC-source rows still mostly resolve to prompt-local answers.",
        ),
        check_metric(
            by_card,
            "MC022",
            "selected_metrics.named_blocks.visible_conflict_by_rule_order.alternate_first.expected_correct_rate",
            "==",
            1.0,
            "mc022_nonlocal_first_visible_rescue",
            "Putting the nonlocal visible branch first rescues visible-visible rows.",
        ),
        check_metric(
            by_card,
            "MC022",
            "selected_metrics.named_blocks.visible_conflict_by_rule_order.local_first.expected_correct_rate",
            "<=",
            0.50,
            "mc022_local_first_visible_collapse",
            "Putting LOCAL first collapses the same visible-visible contract toward local outputs.",
        ),
        check_metric(
            by_card,
            "MC023",
            "all_controls_passed",
            "==",
            True,
            "mc023_controls_clean_before_conflict_failure",
            "Query-operation direct controls and nulls pass in the smoke run.",
        ),
        check_metric(
            by_card,
            "MC023",
            "selected_metrics.panel_metrics.operation_local_conflict.local_number_rate",
            "<",
            0.85,
            "mc023_operation_local_branch_below_gate",
            "The local operation-conflict branch is below behavior-gate quality.",
        ),
        check_metric(
            by_card,
            "MC023",
            "selected_metrics.panel_metrics.operation_atomic_conflict.atomic_number_rate",
            "<",
            0.85,
            "mc023_operation_atomic_branch_below_gate",
            "The atomic operation-conflict branch is below behavior-gate quality.",
        ),
        check_metric(
            by_card,
            "MC024",
            "all_controls_passed",
            "==",
            True,
            "mc024_controls_clean_before_atomic_branch_failure",
            "Few-shot operation direct controls and nulls pass on the full run.",
        ),
        check_metric(
            by_card,
            "MC024",
            "selected_metrics.panel_metrics.operation_local_conflict.local_number_rate",
            ">=",
            0.85,
            "mc024_fewshot_local_branch_passed",
            "Worked examples repair the local operation-conflict branch.",
        ),
        check_metric(
            by_card,
            "MC024",
            "selected_metrics.panel_metrics.operation_atomic_conflict.atomic_number_rate",
            "<",
            0.85,
            "mc024_fewshot_atomic_branch_below_gate",
            "The learned atomic operation-conflict branch remains below gate on the full run.",
        ),
        check_metric(
            by_card,
            "MC024",
            "selected_metrics.panel_metrics.operation_atomic_conflict.other_number_rate",
            ">=",
            0.10,
            "mc024_atomic_branch_other_number_leak",
            "The failed atomic branch leaks non-candidate numeric answers rather than only collapsing to local.",
        ),
        check_metric(
            by_card,
            "MC025",
            "all_controls_passed",
            "==",
            False,
            "mc025_choice_controls_failed",
            "The constrained-choice interface fails before conflict promotion because controls are not clean.",
        ),
        check_metric(
            by_card,
            "MC025",
            "selected_metrics.panel_metrics.atomic_choice_control.atomic_number_rate",
            "<",
            0.85,
            "mc025_choice_atomic_control_failed",
            "Direct atomic recall collapses under the A/B/C choice interface.",
        ),
        check_metric(
            by_card,
            "MC025",
            "selected_metrics.panel_metrics.answer_absent_choice_null.unknown_rate",
            "<",
            0.90,
            "mc025_choice_answer_absent_null_failed",
            "Answer-absent nulls are not clean under the A/B/C choice interface.",
        ),
        check_metric(
            by_card,
            "MC025",
            "selected_metrics.panel_metrics.operation_local_choice_conflict.local_number_rate",
            ">=",
            0.85,
            "mc025_choice_local_branch_passed",
            "The choice interface preserves the prompt-local operation branch.",
        ),
        check_metric(
            by_card,
            "MC025",
            "selected_metrics.panel_metrics.operation_atomic_choice_conflict.atomic_number_rate",
            "<",
            0.85,
            "mc025_choice_atomic_branch_failed",
            "The choice interface does not repair the learned atomic conflict branch.",
        ),
        check_metric(
            by_card,
            "MC026",
            "all_controls_passed",
            "==",
            False,
            "mc026_numeric_option_controls_failed",
            "The numeric-option interface fails before conflict promotion because direct atomic control is not clean.",
        ),
        check_metric(
            by_card,
            "MC026",
            "selected_metrics.panel_metrics.atomic_option_control.atomic_number_rate",
            "==",
            0.0,
            "mc026_numeric_option_atomic_control_zero",
            "Direct atomic recall selects no atomic answers under numeric options.",
        ),
        check_metric(
            by_card,
            "MC026",
            "selected_metrics.panel_metrics.atomic_option_control.unknown_rate",
            ">=",
            0.90,
            "mc026_numeric_option_atomic_control_abstains",
            "Direct atomic recall mostly becomes UNKNOWN under numeric options.",
        ),
        check_metric(
            by_card,
            "MC026",
            "selected_metrics.panel_metrics.answer_absent_option_null.unknown_rate",
            "==",
            1.0,
            "mc026_numeric_option_nulls_clean",
            "Unlike A/B/C choices, numeric options preserve answer-absent nulls.",
        ),
        check_metric(
            by_card,
            "MC026",
            "selected_metrics.panel_metrics.operation_local_option_conflict.local_number_rate",
            ">=",
            0.85,
            "mc026_numeric_option_local_branch_passed",
            "Numeric options preserve the prompt-local operation branch.",
        ),
        check_metric(
            by_card,
            "MC026",
            "selected_metrics.panel_metrics.operation_atomic_option_conflict.atomic_number_rate",
            "<",
            0.85,
            "mc026_numeric_option_atomic_branch_failed",
            "Numeric options do not repair the learned atomic conflict branch.",
        ),
        check_metric(
            by_card,
            "MC027",
            "selected_metrics.panel_metrics.atomic_interface_control.atomic_number_rate",
            ">=",
            0.85,
            "mc027_selected_bare_integer_atomic_control_passed",
            "The selected bare-integer interface preserves direct atomic control in smoke.",
        ),
        check_metric(
            by_card,
            "MC027",
            "selected_metrics.panel_metrics.operation_atomic_interface_conflict.atomic_number_rate",
            ">=",
            0.85,
            "mc027_selected_bare_integer_atomic_branch_passed_smoke",
            "The selected bare-integer interface clears the operation-atomic branch in smoke.",
        ),
        check_metric(
            by_card,
            "MC027",
            "selected_metrics.named_blocks.interface_variant_atomic_control.numeric_options.atomic_number_rate",
            "==",
            0.0,
            "mc027_numeric_options_zero_atomic_control",
            "Numeric options reproduce the direct atomic abstention failure inside the interface sweep.",
        ),
        check_metric(
            by_card,
            "MC027",
            "selected_metrics.named_blocks.interface_variant_answer_absent.json_answer.unknown_rate",
            "<",
            0.10,
            "mc027_json_null_collapsed",
            "The JSON schema turns answer-absent nulls into wrong numeric outputs.",
        ),
        check_metric(
            by_card,
            "MC027",
            "selected_metrics.named_blocks.interface_variant_operation_atomic.letter_choices.atomic_number_rate",
            "<",
            0.25,
            "mc027_letter_choice_atomic_branch_failed",
            "The A/B/C interface remains weak on learned atomic routing in the sweep.",
        ),
        check_metric(
            by_card,
            "MC027",
            "selected_metrics.named_blocks.interface_variant_operation_atomic.answer_prefix.atomic_number_rate",
            "<",
            0.50,
            "mc027_answer_prefix_atomic_branch_failed",
            "The ANSWER= prefix schema does not preserve learned atomic routing.",
        ),
        check_metric(
            by_card,
            "MC028",
            "all_controls_passed",
            "==",
            True,
            "mc028_full_source_controls_clean",
            "The full-source bare-integer boundary preserves direct controls and nulls.",
        ),
        check_metric(
            by_card,
            "MC028",
            "selected_metrics.panel_metrics.operation_local_interface_conflict.local_number_rate",
            ">=",
            0.85,
            "mc028_full_source_local_branch_passed",
            "The prompt-local operation branch survives full-source scale.",
        ),
        check_metric(
            by_card,
            "MC028",
            "selected_metrics.panel_metrics.operation_atomic_interface_conflict.atomic_number_rate",
            "<",
            0.85,
            "mc028_full_source_atomic_branch_failed",
            "The learned atomic operation branch falls below gate at full-source scale.",
        ),
        check_metric(
            by_card,
            "MC028",
            "selected_metrics.panel_metrics.operation_atomic_interface_conflict.other_number_rate",
            ">=",
            0.10,
            "mc028_full_source_other_number_leak",
            "The full-source failure is an other-number leakage boundary, not just local collapse.",
        ),
        check_metric(
            by_card,
            "MC029",
            "selected_metrics.named_blocks.variant_operation_atomic.rules_only.atomic_number_rate",
            ">=",
            0.85,
            "mc029_rules_only_atomic_branch_improves",
            "Removing worked examples pushes the learned atomic branch above the atomic-rate threshold.",
        ),
        check_metric(
            by_card,
            "MC029",
            "selected_metrics.named_blocks.variant_answer_absent.rules_only.unknown_rate",
            "<",
            0.90,
            "mc029_rules_only_nulls_fail",
            "The same rules-only variant breaks answer-absent nulls, so it is not a behavior substrate.",
        ),
        check_metric(
            by_card,
            "MC029",
            "selected_metrics.named_blocks.variant_operation_atomic.query_before_examples.atomic_number_rate",
            "<",
            0.50,
            "mc029_query_before_examples_atomic_branch_collapses",
            "Moving the query before numeric examples makes the learned atomic branch collapse.",
        ),
        check_metric(
            by_card,
            "MC029",
            "selected_metrics.named_blocks.variant_operation_atomic.query_before_examples.other_number_rate",
            ">=",
            0.40,
            "mc029_query_before_examples_amplifies_other_numbers",
            "Query-before-examples strongly amplifies other-number leakage.",
        ),
        check_metric(
            by_card,
            "MC029",
            "selected_metrics.named_blocks.variant_operation_atomic.query_row_last.other_number_rate",
            "<",
            0.10,
            "mc029_query_row_last_reduces_other_numbers",
            "Putting the query row last reduces other-number leakage.",
        ),
        check_metric(
            by_card,
            "MC029",
            "selected_metrics.named_blocks.variant_operation_atomic.query_row_last.atomic_number_rate",
            "<",
            0.85,
            "mc029_query_row_last_still_not_behavior_gate",
            "That other-number reduction does not rescue learned atomic routing.",
        ),
        check_metric(
            by_card,
            "MC030",
            "selected_metrics.named_blocks.variant_operation_atomic.rules_only_baseline.atomic_number_rate",
            ">=",
            0.85,
            "mc030_baseline_preserves_rules_only_branch",
            "The MC029 rules-only branch gain reproduces inside the MC030 repair attempt.",
        ),
        check_metric(
            by_card,
            "MC030",
            "selected_metrics.named_blocks.variant_answer_absent.rules_only_baseline.unknown_rate",
            "<",
            0.90,
            "mc030_baseline_null_still_fails",
            "The reproduced rules-only baseline still fails answer-absent null reliability.",
        ),
        check_metric(
            by_card,
            "MC030",
            "selected_metrics.named_blocks.variant_answer_absent.row_absence_guard_before_rules.unknown_rate",
            "<",
            0.50,
            "mc030_row_absence_guard_worsens_nulls",
            "The direct row-absence guard worsens answer-absent null behavior rather than repairing it.",
        ),
        check_metric(
            by_card,
            "MC030",
            "selected_metrics.named_blocks.variant_answer_absent.decision_order_guard_after_query.unknown_rate",
            "<",
            0.50,
            "mc030_decision_guard_worsens_nulls",
            "The decision-order guard also fails to restore answer-absent nulls.",
        ),
        check_metric(
            by_card,
            "MC030",
            "selected_metrics.named_blocks.variant_operation_atomic.decision_order_guard_query_last.other_number_rate",
            "<",
            0.10,
            "mc030_query_last_reduces_other_numbers",
            "The guarded query-last variant reduces other-number leakage.",
        ),
        check_metric(
            by_card,
            "MC030",
            "selected_metrics.named_blocks.variant_operation_atomic.decision_order_guard_query_last.atomic_number_rate",
            "<",
            0.50,
            "mc030_query_last_collapses_atomic_branch",
            "That other-number reduction comes with learned atomic branch collapse.",
        ),
        check_metric(
            by_card,
            "MC031",
            "selected_metrics.panel_metrics.synthetic_numeric_lookup.local_number_rate",
            "==",
            1.0,
            "mc031_synthetic_lookup_control_clean",
            "The statusless checksum smoke keeps direct synthetic local lookup clean.",
        ),
        check_metric(
            by_card,
            "MC031",
            "selected_metrics.panel_metrics.real_world_atomic_number_control.atomic_number_rate",
            "==",
            1.0,
            "mc031_real_atomic_control_clean",
            "The direct atomic-number control is clean, so learned atomic recall is available.",
        ),
        check_metric(
            by_card,
            "MC031",
            "selected_metrics.panel_metrics.answer_absent_null.unknown_rate",
            "==",
            1.0,
            "mc031_answer_absent_null_clean",
            "The answer-absent null remains clean in the selected checksum template.",
        ),
        check_metric(
            by_card,
            "MC031",
            "selected_metrics.panel_metrics.checksum_valid_conflict.local_number_rate",
            ">=",
            0.85,
            "mc031_valid_checksum_local_branch_clean",
            "Valid checksum rows follow the local branch.",
        ),
        check_metric(
            by_card,
            "MC031",
            "selected_metrics.panel_metrics.checksum_invalid_conflict.local_number_rate",
            "==",
            1.0,
            "mc031_invalid_checksum_collapses_to_local",
            "Invalid checksum rows collapse to local numbers rather than learned atomic numbers.",
        ),
        check_metric(
            by_card,
            "MC031",
            "selected_metrics.panel_metrics.checksum_invalid_conflict.atomic_or_lure_number_rate",
            "==",
            0.0,
            "mc031_invalid_checksum_atomic_lure_absent",
            "The invalid checksum branch selects no atomic or lure numbers.",
        ),
        check_metric(
            by_card,
            "MC032",
            "selected_metrics.panel_metrics.synthetic_numeric_lookup.local_number_rate",
            "==",
            1.0,
            "mc032_synthetic_lookup_control_clean",
            "The cross-table smoke keeps direct synthetic local lookup clean.",
        ),
        check_metric(
            by_card,
            "MC032",
            "selected_metrics.panel_metrics.real_world_atomic_number_control.atomic_number_rate",
            "==",
            1.0,
            "mc032_real_atomic_control_clean",
            "The direct atomic-number control is clean, so learned atomic recall is available.",
        ),
        check_metric(
            by_card,
            "MC032",
            "selected_metrics.panel_metrics.answer_absent_null.unknown_rate",
            "==",
            1.0,
            "mc032_answer_absent_null_clean",
            "The answer-absent null remains clean in the selected cross-table template.",
        ),
        check_metric(
            by_card,
            "MC032",
            "selected_metrics.panel_metrics.crosscheck_mismatch_conflict.atomic_or_lure_number_rate",
            "==",
            0.0,
            "mc032_mismatch_atomic_lure_absent",
            "Cross-table mismatch rows select no atomic or lure numbers.",
        ),
        check_metric(
            by_card,
            "MC032",
            "selected_metrics.panel_metrics.crosscheck_mismatch_conflict.local_number_rate",
            ">=",
            0.70,
            "mc032_mismatch_collapses_toward_local",
            "Cross-table mismatch rows mostly collapse toward the primary local number.",
        ),
        check_metric(
            by_card,
            "MC032",
            "selected_metrics.panel_metrics.crosscheck_mismatch_conflict.side_number_rate",
            "==",
            0.0,
            "mc032_side_number_leak_absent",
            "The mismatch failure is not caused by copying the second-table side number.",
        ),
        check_metric(
            by_card,
            "MC033",
            "selected_metrics.panel_metrics.synthetic_numeric_lookup.local_number_rate",
            "==",
            1.0,
            "mc033_synthetic_lookup_control_clean",
            "The fact-claim smoke keeps direct synthetic local lookup clean.",
        ),
        check_metric(
            by_card,
            "MC033",
            "selected_metrics.panel_metrics.real_world_atomic_number_control.atomic_number_rate",
            "==",
            1.0,
            "mc033_real_atomic_control_clean",
            "The direct atomic-number control is clean, so learned atomic recall is available.",
        ),
        check_metric(
            by_card,
            "MC033",
            "selected_metrics.panel_metrics.answer_absent_null.unknown_rate",
            "==",
            1.0,
            "mc033_answer_absent_null_clean",
            "The answer-absent null remains clean in the selected fact-claim template.",
        ),
        check_metric(
            by_card,
            "MC033",
            "selected_metrics.panel_metrics.fact_claim_match_conflict.local_number_rate",
            "<",
            0.85,
            "mc033_match_branch_below_gate",
            "Fact-claim match rows do not reliably return the local number.",
        ),
        check_metric(
            by_card,
            "MC033",
            "selected_metrics.panel_metrics.fact_claim_mismatch_conflict.atomic_number_rate",
            "<",
            0.85,
            "mc033_mismatch_atomic_branch_below_gate",
            "Fact-claim mismatch rows do not reliably return the learned atomic number.",
        ),
        check_metric(
            by_card,
            "MC033",
            "selected_metrics.panel_metrics.fact_claim_mismatch_conflict.lure_atomic_number_rate",
            ">=",
            0.40,
            "mc033_mismatch_claimed_number_leak",
            "Fact-claim mismatch rows copy the wrong claimed number at material rate.",
        ),
    ]
    return checks


def build_control_surface_smoke_diagnostics() -> dict[str, Any]:
    entries = [build_card_entry(config) for config in SMOKE_CARD_CONFIGS]
    failure_axes = Counter(axis for entry in entries for axis in entry["failure_axes"])
    exported = Counter(diagnostic for entry in entries for diagnostic in entry["exported_diagnostics"])
    diagnostic_classes = Counter(str(entry["diagnostic_class"]) for entry in entries)
    observed_patterns = Counter(
        str(entry["observed_failure_pattern"])
        for entry in entries
        if entry.get("observed_failure_pattern")
    )
    summary = {
        "card_count": len(entries),
        "expected_card_count": len(SMOKE_CARD_CONFIGS),
        "card_ids": [entry["card_id"] for entry in entries],
        "smoke_only_count": sum(1 for entry in entries if entry["smoke_only"]),
        "structural_passed_count": sum(1 for entry in entries if entry["structural_passed"]),
        "behavior_ready_count": sum(1 for entry in entries if entry["behavior_ready"]),
        "signature_ready_count": sum(1 for entry in entries if entry["signature_ready"]),
        "intervention_ready_count": sum(1 for entry in entries if entry["intervention_ready"]),
        "hidden_state_allowed_count": sum(1 for entry in entries if entry["hidden_state_allowed"]),
        "diagnostic_class_counts": dict(sorted(diagnostic_classes.items())),
        "observed_failure_pattern_counts": dict(sorted(observed_patterns.items())),
        "failure_axis_counts": dict(sorted(failure_axes.items())),
        "exported_diagnostic_counts": dict(sorted(exported.items())),
    }
    validation_checks = build_validation_checks(entries, summary)
    payload = {
        "schema_version": 1,
        "purpose": (
            "Structured smoke-diagnostic ledger for MC017-MC033. These runs are "
            "behavior-substrate diagnostics, not mechanism-card atlas rows."
        ),
        "atlas_ref": "data/control_surface_atlas.json",
        "cards": entries,
        "summary": summary,
        "validation_checks": validation_checks,
    }
    return payload


def validate_smoke_diagnostics(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("smoke diagnostics schema_version must be 1")
    cards = payload.get("cards", [])
    expected_ids = [config["card_id"] for config in SMOKE_CARD_CONFIGS]
    actual_ids = [entry.get("card_id") for entry in cards]
    if actual_ids != expected_ids:
        raise AssertionError(f"smoke card ids expected {expected_ids}, got {actual_ids}")
    for entry in cards:
        for path_field in ["artifact_path", "runner_path", "status_card_path"]:
            path = ROOT / entry[path_field]
            if not path.exists():
                raise AssertionError(f"{entry['card_id']}: missing {path_field} {entry[path_field]}")
        if entry["behavior_ready"] or entry["signature_ready"] or entry["intervention_ready"]:
            raise AssertionError(f"{entry['card_id']}: smoke diagnostic unexpectedly ready for hidden-state work")
        if entry["hidden_state_allowed"]:
            raise AssertionError(f"{entry['card_id']}: hidden_state_allowed must remain false")
        if not entry["failure_axes"] or not entry["exported_diagnostics"]:
            raise AssertionError(f"{entry['card_id']}: missing typed failure metadata")
    failed_checks = [check for check in payload.get("validation_checks", []) if not check.get("passed")]
    if failed_checks:
        raise AssertionError(f"smoke diagnostic checks failed: {failed_checks}")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if value is None:
        return "n/a"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Control-Surface Smoke Diagnostics",
        "",
        "Date: 2026-07-01",
        "",
        "Status: generated smoke-diagnostic ledger implemented and validated.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/control_surface_smoke_diagnostics.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_smoke_diagnostics.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_smoke_diagnostics.py --write",
        "python code\\control_surface_smoke_diagnostics.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Purpose",
        "",
            "MC017-MC033 are not promoted atlas rows. They are bridge",
        "diagnostics that test whether the bridge from prompt-local numeric",
        "tables to learned atomic facts has a behavior substrate worth probing.",
        "",
        "This ledger exists so typed failures remain auditable data. It keeps the",
        "project from treating these runs as anecdotes or accidentally upgrading",
        "them into mechanism claims.",
        "",
        "## Current Generated Facts",
        "",
        f"- smoke cards: {summary['card_count']};",
        f"- smoke-only cards: {summary['smoke_only_count']};",
        f"- structural-passed cards: {summary['structural_passed_count']};",
        f"- behavior-ready cards: {summary['behavior_ready_count']};",
        f"- signature-ready cards: {summary['signature_ready_count']};",
        f"- hidden-state-allowed cards: {summary['hidden_state_allowed_count']}.",
        "",
        "Failure-axis counts:",
        "",
    ]
    for axis, count in summary["failure_axis_counts"].items():
        lines.append(f"- `{axis}`: {count};")
    lines.extend(
        [
            "",
            "## Ledger",
            "",
            "| Card | Selected Template | Pattern | Headline Metrics | Exported Diagnostics |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for entry in payload["cards"]:
        headlines = "<br>".join(
            f"{metric['label']}: `{format_value(metric['value'])}`"
            for metric in entry["headline_metrics"]
        )
        exported = "<br>".join(f"`{diagnostic}`" for diagnostic in entry["exported_diagnostics"])
        pattern = entry.get("observed_failure_pattern") or entry["diagnostic_class"]
        lines.append(
            f"| `{entry['card_id']}` | `{entry['selected_template']}` | `{pattern}` | {headlines} | {exported} |"
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
            f"| `{check['id']}` | `{check['card_id']}` | `{str(check['passed']).lower()}` | `{format_value(check['actual'])}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The current smoke chain says the bridge failure is layered:",
            "",
            "1. `LOCAL`/`ATOMIC` answer tokens can create a local-selector collapse.",
            "2. Neutral source labels expose first-listed-choice and local-source salience.",
            "3. Row-local route codes can clean up controls and nulls while still failing expected-atomic routing.",
            "4. Direct atomic recall survives local-table pressure, so recall availability is not the bottleneck.",
            "5. Opaque route codes fail even between prompt-visible branches.",
            "6. Semantic answer-source labels remain sensitive to rule order and still fail learned-memory branch selection.",
            "7. Query-level operation handles preserve direct controls and nulls, but both conflict branches remain below gate quality.",
            "8. Balanced worked examples repair prompt-local operation routing, but the learned atomic branch remains below full-source gate quality.",
            "9. Prompt-visible A/B/C choices are not a neutral answer-interface fix; they break atomic controls and answer-absent nulls.",
            "10. Prompt-visible numeric options preserve nulls but turn direct atomic recall into UNKNOWN and leave the learned branch weak.",
            "11. Sweeping answer interfaces shows that bare integer is the only 10-source smoke survivor; structured schemas and option interfaces break controls, nulls, or learned routing.",
            "12. The bare-integer smoke survivor fails full-source promotion: controls and nulls stay clean, but operation-atomic rows leak other numbers below gate.",
            "13. Factorizing the leak and then adding absence guards moves the error axes but does not repair the rules-only branch/null tradeoff.",
            "",
            "The next bridge cannot just rename labels, repeat rows, or optimize direct",
            "atomic recall. It also cannot rely on query-level operation handles unless",
            "both local and learned branches exceed gate thresholds, and it cannot treat",
            "candidate choices, numeric option lists, JSON schemas, or answer-prefix",
            "schemas as free output repairs. It must create a behavior substrate where the same rule",
            "selects prompt-local, prompt-visible nonlocal, and learned-memory branches",
            "under balanced rule-order, local-salience, full-source, and",
            "other-number side-effect controls.",
            "",
            "## Claim Boundary",
            "",
            "This ledger does not establish a signature, intervention, or mechanism card.",
            "It is a generated diagnostic layer for behavior-substrate failures that",
            "should constrain future mechanism-card attempts.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write JSON and markdown artifacts")
    args = parser.parse_args()

    payload = build_control_surface_smoke_diagnostics()
    validate_smoke_diagnostics(payload)
    if args.write:
        write_json(SMOKE_DIAGNOSTICS_PATH, payload)
        SMOKE_DIAGNOSTICS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SMOKE_DIAGNOSTICS_REPORT_PATH.write_text(render_markdown(payload), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "passed": True,
                "card_count": payload["summary"]["card_count"],
                "behavior_ready_count": payload["summary"]["behavior_ready_count"],
                "signature_ready_count": payload["summary"]["signature_ready_count"],
                "hidden_state_allowed_count": payload["summary"]["hidden_state_allowed_count"],
                "validation_check_count": len(payload["validation_checks"]),
                "output_path": rel(SMOKE_DIAGNOSTICS_PATH),
                "report_path": rel(SMOKE_DIAGNOSTICS_REPORT_PATH),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
