"""Extract and validate control-surface result artifacts.

The atlas should not rely only on hand-written summaries. This module provides
a small parser registry that normalizes the result JSONs linked from atlas rows
into the fields the project repeatedly audits: pass/fail, diagnostic class,
readiness flags, selected coordinates/templates, criteria failures, and the
presence of baseline/null/intervention controls.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
ATLAS_PATH = ROOT / "data" / "control_surface_atlas.json"
ARTIFACT_INDEX_PATH = ROOT / "data" / "control_surface_artifact_index.json"

CONTROL_KEYWORDS = {
    "baseline",
    "candidate",
    "control",
    "final",
    "margin",
    "output",
    "shuffle",
    "shuffled",
}
INTERVENTION_KEYWORDS = {
    "causal",
    "intervention",
    "locality",
    "patch",
    "replacement",
    "side",
    "steer",
    "steering",
    "write",
}
NULL_KEYWORDS = {
    "absent",
    "null",
    "unknown",
}
FAMILY_CLAIM_CHECK_ROWS = {
    "mc001_source_mask_and_rewrite_equivalence_evidence": "mc001_qwen3_0p6b_truth_agreement",
    "mc001b_raw_signal_stronger_than_residual_signal": "mc001b_qwen3_1p7b_truth_agreement",
    "mc001g_layer14_intervention_no_holdout_class_movement": "mc001g_gemma_truth_agreement",
    "mc002_behavior_substrate_failure": "mc002_known_unknown",
    "mc002b_context_support_pressure_failure": "mc002b_context_support",
    "mc003_shuffle_and_condition_trace_signature_failure": "mc003_delayed_copy",
    "mc004_leadtime_shuffle_and_subgroup_failure": "mc004_in_context_binding",
    "mc005_l24_26_attention_write_boundary": "mc005_associative_lookup",
    "mc005_lookup_effect_with_nonzero_null_flips": "mc005_associative_lookup",
    "mc005_null_boundary_not_simple_margin_cutoff": "mc005_associative_lookup",
    "mc006_additive_steering_failure": "mc006_parametric_fact_override",
    "mc006_v16_output_candidate_confounded": "mc006_parametric_fact_override",
    "mc006_v21_pair_matching_margin_baseline_failure": "mc006_parametric_fact_override",
    "mc006_v22_source_path_final_margin_shadow": "mc006_parametric_fact_override",
    "mc006_v24_delayed_city_monitor_only": "mc006_parametric_fact_override",
    "mc006_v25_candidate_decoupled_shuffle_overfit": "mc006_parametric_fact_override",
    "mc006_v28_one_transfer_template_not_bank": "mc006_parametric_fact_override",
    "mc006_shuffle_null_and_transfer_role_failure": "mc006_parametric_fact_override",
    "mc007_contrast_absent_and_authority_controls_failed": "mc007_semi_synthetic_familiar_entity_lookup",
    "mc007_v1_source_lookup_without_conflict": "mc007_semi_synthetic_familiar_entity_lookup",
    "mc007_v2_authority_dial_parseability_failure": "mc007_semi_synthetic_familiar_entity_lookup",
    "mc007_v3_parseability_repair_failure": "mc007_semi_synthetic_familiar_entity_lookup",
    "mc007_v4_source_declaration_control_failure": "mc007_semi_synthetic_familiar_entity_lookup",
    "mc008_direct_controls_clean_before_null_failure": "mc008_symbolic_fact_code_arbitration",
    "mc008_null_repair_conflict_contrast_absent": "mc008_symbolic_fact_code_arbitration",
    "mc008_v2_null_repaired_conflict_absent": "mc008_symbolic_fact_code_arbitration",
    "mc009_membership_controls_clean_conflict_failed": "mc009_derived_code_arbitration",
    "mc009_membership_tradeoff_and_typed_slot_control_failure": "mc009_derived_code_arbitration",
    "mc009_typed_slot_balance_breaks_controls": "mc009_derived_code_arbitration",
    "mc010_two_hop_direct_controls_failed": "mc010_two_hop_fact_code_arbitration",
    "mc010_two_hop_conflict_table_dominant": "mc010_two_hop_fact_code_arbitration",
    "mc011_numeric_direct_controls_clean": "mc011_atomic_number_code_arbitration",
    "mc011_numeric_conflict_table_dominant": "mc011_atomic_number_code_arbitration",
    "mc012_reliability_behavior_contrast_passed": "mc012_reliability_labeled_numeric_arbitration",
    "mc012_reliability_prompt_channel_blocks_signature": "mc012_reliability_labeled_numeric_arbitration",
    "mc013_statused_positive_control_reproduced": "mc013_status_channel_ablation_numeric_arbitration",
    "mc013_status_channel_ablation_collapses_contrast": "mc013_status_channel_ablation_numeric_arbitration",
    "mc014_inferred_reliability_direct_controls_clean": "mc014_inferred_reliability_numeric_arbitration",
    "mc014_inferred_reliability_conflict_collapsed": "mc014_inferred_reliability_numeric_arbitration",
    "mc015_parity_gate_direct_controls_clean": "mc015_parity_gated_numeric_arbitration",
    "mc015_parity_gate_mixed_outputs_wrong_rule": "mc015_parity_gated_numeric_arbitration",
    "mc016_alphabet_gate_direct_controls_clean": "mc016_alphabet_gated_numeric_arbitration",
    "mc016_alphabet_gate_local_collapse": "mc016_alphabet_gated_numeric_arbitration",
}
FAMILY_CLAIM_CHECKS = tuple(FAMILY_CLAIM_CHECK_ROWS)


@dataclass
class ArtifactSummary:
    path: str
    row_id: str | None
    parser: str
    card_id: str | None
    run_type: str | None
    model_id: str | None
    record_count: int | None
    diagnostic_class: str | None
    passed: bool | None
    behavior_ready: bool | None
    signature_ready: bool | None
    intervention_ready: bool | None
    selected: dict[str, Any]
    failed_criteria: list[str]
    null_criteria: list[str]
    baseline_fields: list[str]
    intervention_fields: list[str]
    metrics: dict[str, Any]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dotted_get(payload: Any, dotted_path: str) -> Any | None:
    current = payload
    for part in dotted_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def first_present(payload: Any, dotted_paths: Iterable[str]) -> Any | None:
    for dotted_path in dotted_paths:
        value = dotted_get(payload, dotted_path)
        if value is not None:
            return value
    return None


def flatten_paths(payload: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_paths(value, child)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            child = f"{prefix}.{index}" if prefix else str(index)
            yield from flatten_paths(value, child)
    else:
        yield prefix, payload


def path_matches(path: str, keywords: set[str]) -> bool:
    lowered = path.lower()
    return any(keyword in lowered for keyword in keywords)


def primitive(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        if len(value) <= 8 and all(isinstance(item, (str, int, float, bool)) for item in value):
            return value
        return f"<list:{len(value)}>"
    if isinstance(value, dict):
        return f"<dict:{len(value)}>"
    return repr(value)


def set_metric(metrics: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        metrics[key] = primitive(value)


def add_dotted_metric(
    metrics: dict[str, Any],
    payload: Any,
    metric_key: str,
    dotted_path: str,
) -> None:
    set_metric(metrics, metric_key, dotted_get(payload, dotted_path))


def add_summary_count_metrics(
    metrics: dict[str, Any],
    prefix: str,
    summary_block: Any,
) -> None:
    if not isinstance(summary_block, dict):
        return
    for key, value in summary_block.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            metrics[f"{prefix}.{key}"] = value


def add_criteria_metrics(
    metrics: dict[str, Any],
    prefix: str,
    criteria: Any,
) -> None:
    if not isinstance(criteria, dict):
        return
    metrics[f"{prefix}_failed_count"] = sum(1 for value in criteria.values() if value is False)
    for key, value in criteria.items():
        if isinstance(value, bool):
            metrics[f"{prefix}.{key}"] = value


def add_panel_metrics(metrics: dict[str, Any], summary: dict[str, Any]) -> None:
    panels = dotted_get(summary, "selected_template_summary.panels")
    if not isinstance(panels, dict):
        return
    for panel_name, panel_summary in panels.items():
        add_summary_count_metrics(metrics, f"panel.{panel_name}", panel_summary)


def add_label_count_metrics(
    metrics: dict[str, Any],
    prefix: str,
    summary_block: Any,
) -> None:
    if not isinstance(summary_block, dict):
        return
    add_summary_count_metrics(metrics, prefix, summary_block)
    for label_field in ["label_counts", "labels", "parsed_answers"]:
        labels = summary_block.get(label_field)
        if isinstance(labels, dict):
            for label, count in labels.items():
                if isinstance(count, (int, float, bool)):
                    metrics[f"{prefix}.{label_field}.{label}"] = count


def add_named_block_metrics(
    metrics: dict[str, Any],
    prefix: str,
    blocks: Any,
    limit: int = 32,
) -> None:
    if not isinstance(blocks, dict):
        return
    for index, (name, block) in enumerate(sorted(blocks.items())):
        if index >= limit:
            metrics[f"{prefix}._truncated_after"] = limit
            break
        add_label_count_metrics(metrics, f"{prefix}.{name}", block)


def add_probe_score_metrics(
    metrics: dict[str, Any],
    prefix: str,
    scores: Any,
) -> None:
    if not isinstance(scores, dict):
        return
    for direction_name, direction_payload in scores.items():
        if not isinstance(direction_payload, dict):
            continue
        for score_name, score_payload in direction_payload.items():
            if not isinstance(score_payload, dict):
                continue
            for split_name, split_payload in score_payload.items():
                if not isinstance(split_payload, dict):
                    continue
                for metric_name in ["auc", "accuracy", "n", "positive_rate"]:
                    value = split_payload.get(metric_name)
                    if isinstance(value, (int, float, bool)):
                        metrics[
                            f"{prefix}.{direction_name}.{score_name}.{split_name}.{metric_name}"
                        ] = value


def add_legacy_behavior_metrics(metrics: dict[str, Any], payload: dict[str, Any]) -> None:
    add_label_count_metrics(metrics, "baseline.overall", dotted_get(payload, "baseline_summary.overall"))
    add_label_count_metrics(
        metrics,
        "baseline.wrong_hint_conditions",
        dotted_get(payload, "baseline_summary.wrong_hint_conditions"),
    )
    add_label_count_metrics(
        metrics,
        "baseline.no_and_correct_hint_conditions",
        dotted_get(payload, "baseline_summary.no_and_correct_hint_conditions"),
    )
    add_named_block_metrics(
        metrics,
        "baseline.by_condition",
        dotted_get(payload, "baseline_summary.by_condition"),
    )

    generation_arms = dotted_get(payload, "generation_validation.summary_by_arm")
    if isinstance(generation_arms, dict):
        for arm_name, arm_summary in sorted(generation_arms.items()):
            add_label_count_metrics(
                metrics,
                f"generation.{arm_name}.overall",
                dotted_get(arm_summary, "overall"),
            )
            add_label_count_metrics(
                metrics,
                f"generation.{arm_name}.wrong_hint_conditions",
                dotted_get(arm_summary, "wrong_hint_conditions"),
            )
            add_label_count_metrics(
                metrics,
                f"generation.{arm_name}.no_and_correct_hint_conditions",
                dotted_get(arm_summary, "no_and_correct_hint_conditions"),
            )

    add_probe_score_metrics(metrics, "probe.named_direction", payload.get("named_direction_probe_scores"))
    add_probe_score_metrics(metrics, "probe.residual", payload.get("residual_probe_scores"))

    rewrite_summary = payload.get("rewrite_parity_summary")
    if isinstance(rewrite_summary, dict):
        for rewrite_name, rewrite_payload in sorted(rewrite_summary.items()):
            add_summary_count_metrics(metrics, f"rewrite.{rewrite_name}", rewrite_payload)

    position_summary = payload.get("position_count_summary")
    if isinstance(position_summary, dict):
        for position_name, position_payload in sorted(position_summary.items()):
            add_summary_count_metrics(metrics, f"position.{position_name}", position_payload)

    arms = payload.get("arms")
    if isinstance(arms, dict):
        for arm_name, arm_payload in sorted(arms.items()):
            arm_summary = dotted_get(arm_payload, "summary")
            if isinstance(arm_summary, dict):
                for group_name, group_payload in sorted(arm_summary.items()):
                    add_label_count_metrics(
                        metrics,
                        f"arm.{arm_name}.{group_name}",
                        group_payload,
                    )

    structural_check = payload.get("structural_check")
    if isinstance(structural_check, dict):
        set_metric(metrics, "structural_check.passed", structural_check.get("passed"))
        checks = structural_check.get("checks")
        if isinstance(checks, dict):
            for check_name, value in sorted(checks.items()):
                if isinstance(value, bool):
                    metrics[f"structural_check.checks.{check_name}"] = value

    for field in ["bin_width", "eval_row_count", "layer", "limit", "records_n", "wrong_layer"]:
        set_metric(metrics, field, payload.get(field))
    discovery_rows = payload.get("discovery_rows")
    if isinstance(discovery_rows, list):
        metrics["discovery_row_count"] = len(discovery_rows)


def extract_legacy_family_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    add_legacy_behavior_metrics(metrics, payload)
    return metrics


def extract_family_metrics(payload: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    criteria = summary.get("criteria")
    if isinstance(criteria, dict):
        add_criteria_metrics(metrics, "criteria", criteria)
    add_criteria_metrics(metrics, "causal_criteria", summary.get("causal_criteria"))
    add_criteria_metrics(metrics, "mechanism_criteria", summary.get("mechanism_criteria"))

    add_dotted_metric(metrics, summary, "selected_template", "selection.selected_template")
    add_dotted_metric(metrics, summary, "behavior_gate_passed", "behavior_gate_passed")
    add_dotted_metric(metrics, summary, "behavior_ready", "behavior_ready")
    add_dotted_metric(metrics, summary, "signature_ready", "signature_ready")
    add_dotted_metric(metrics, summary, "intervention_ready", "intervention_ready")
    add_summary_count_metrics(
        metrics,
        "primary_conflict",
        dotted_get(summary, "selected_template_summary.primary_conflict"),
    )
    add_summary_count_metrics(
        metrics,
        "primary_conflict_expected_local",
        dotted_get(summary, "selected_template_summary.primary_conflict_expected_local"),
    )
    add_summary_count_metrics(
        metrics,
        "primary_conflict_expected_atomic",
        dotted_get(summary, "selected_template_summary.primary_conflict_expected_atomic"),
    )
    add_summary_count_metrics(
        metrics,
        "primary_ablation_conflict",
        dotted_get(summary, "selected_template_summary.primary_ablation_conflict"),
    )
    add_summary_count_metrics(
        metrics,
        "statused_conflict",
        dotted_get(summary, "selected_template_summary.statused_conflict"),
    )
    add_summary_count_metrics(
        metrics,
        "primary",
        dotted_get(summary, "selected_template_summary.primary"),
    )
    add_label_count_metrics(metrics, "overall", summary.get("overall"))
    add_label_count_metrics(metrics, "audit", summary.get("audit"))
    add_named_block_metrics(metrics, "by_condition", summary.get("by_condition"))
    add_named_block_metrics(metrics, "by_entity_condition", summary.get("by_entity_condition"))
    add_panel_metrics(metrics, summary)

    card_id = payload.get("card_id")
    if card_id == "MC005":
        add_dotted_metric(
            metrics,
            summary,
            "mc005.lookup_target_effect_reproduced",
            "criteria.v31_lookup_target_effect_reproduced",
        )
        add_dotted_metric(
            metrics,
            summary,
            "mc005.combined_null_flip_count",
            "combined_null_flip_count",
        )
        add_dotted_metric(metrics, summary, "mc005.fresh_row_count", "fresh_row_count")
        add_dotted_metric(
            metrics,
            summary,
            "mc005.lookup_changed_rows",
            "lookup_target_write.changed_rows",
        )
        add_dotted_metric(
            metrics,
            summary,
            "mc005.lookup_target_win_loss_rows",
            "lookup_target_write.target_win_loss_rows",
        )
        add_dotted_metric(
            metrics,
            summary,
            "mc005.write_layer_count",
            "write_layers",
        )

    if card_id == "MC006":
        add_dotted_metric(
            metrics,
            summary,
            "mc006.selected_pre_output_name",
            "selected_pre_output_candidate.name",
        )
        add_dotted_metric(
            metrics,
            summary,
            "mc006.selected_pre_output_holdout_auc",
            "selected_pre_output_candidate.holdout_auc",
        )
        add_dotted_metric(
            metrics,
            summary,
            "mc006.candidate_score_holdout_auc",
            "global_baseline_results.candidate_score_margin.holdout_auc",
        )
        add_dotted_metric(
            metrics,
            summary,
            "mc006.final_output_holdout_auc",
            "global_baseline_results.final_next_token_output_margin.holdout_auc",
        )
        add_dotted_metric(
            metrics,
            summary,
            "mc006.shuffle_holdout_auc_p95",
            "shuffle_selection_null.holdout_auc_p95",
        )
        add_dotted_metric(
            metrics,
            summary,
            "mc006.hidden_beats_shuffle_null",
            "criteria.hidden_beats_shuffle_null",
        )
        add_dotted_metric(
            metrics,
            summary,
            "mc006.transfer_ready_count",
            "transfer_ready_summary.transfer_ready_count",
        )
        add_dotted_metric(
            metrics,
            summary,
            "mc006.combined_ready_template_count",
            "combined_ready_summary.ready_template_count",
        )
        add_dotted_metric(
            metrics,
            summary,
            "mc006.combined_pooled_binary_rows",
            "combined_ready_summary.pooled_binary_rows",
        )
        add_dotted_metric(
            metrics,
            summary,
            "mc006.source_checks_passed",
            "source_checks_passed",
        )
        add_dotted_metric(
            metrics,
            summary,
            "mc006.causal_stress_supported",
            "causal_stress_supported",
        )
        add_dotted_metric(
            metrics,
            summary,
            "mc006.selected_layer",
            "selected_layer",
        )
        add_dotted_metric(
            metrics,
            summary,
            "mc006.selected_position",
            "selected_position",
        )
        add_dotted_metric(
            metrics,
            summary,
            "mc006.baseline_reproduction_matches",
            "baseline_reproduction.matches",
        )
        add_dotted_metric(
            metrics,
            summary,
            "mc006.baseline_reproduction_total",
            "baseline_reproduction.total",
        )
        for field in [
            "dose",
            "selected_abs_mean_effect",
            "max_control_abs_mean_effect",
            "control_gap",
            "plus_mean_margin_delta",
            "minus_mean_margin_delta",
            "plus_predicted_direction_changes",
            "minus_predicted_direction_changes",
            "total_predicted_direction_changes",
            "max_side_label_changes",
            "plus_side_label_changes",
            "minus_side_label_changes",
        ]:
            add_dotted_metric(metrics, summary, f"mc006.best_dose.{field}", f"best_dose.{field}")

    return metrics


def infer_record_count(payload: dict[str, Any]) -> int | None:
    for dotted_path in [
        "record_count",
        "records_n",
        "n_records",
        "summary.record_count",
        "summary.records_n",
        "summary.selected_template_rows",
        "summary.binary_row_count",
    ]:
        value = dotted_get(payload, dotted_path)
        if isinstance(value, int):
            return value
    for field in ["records", "rows", "outputs", "baseline_records"]:
        value = payload.get(field)
        if isinstance(value, list):
            return len(value)
    return None


def extract_selected(payload: dict[str, Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for dotted_path, value in flatten_paths(payload):
        leaf = dotted_path.rsplit(".", 1)[-1].lower()
        if "selected" not in leaf and leaf not in {
            "best_layer",
            "best_dose",
            "locked_coordinate",
            "position",
            "layer",
            "template",
        }:
            continue
        if len(selected) >= 40:
            break
        selected[dotted_path] = primitive(value)
    return selected


def extract_criteria(summary: dict[str, Any]) -> tuple[list[str], list[str]]:
    criteria_blocks: list[tuple[str, Any]] = [("", summary.get("criteria"))]
    for block_name in ["causal_criteria", "mechanism_criteria"]:
        criteria_blocks.append((block_name, summary.get(block_name)))
    failed: list[str] = []
    null_related: list[str] = []
    for block_name, criteria in criteria_blocks:
        if not isinstance(criteria, dict):
            continue
        for key, value in criteria.items():
            criterion_name = f"{block_name}.{key}" if block_name else key
            if value is False:
                failed.append(criterion_name)
            if path_matches(criterion_name, NULL_KEYWORDS):
                null_related.append(criterion_name)
    return sorted(failed), sorted(null_related)


class ArtifactParser:
    name = "base"

    def matches(self, payload: dict[str, Any]) -> bool:
        raise NotImplementedError

    def parse(self, path: str, row_id: str | None, payload: dict[str, Any]) -> ArtifactSummary:
        raise NotImplementedError


class SummarySchemaParser(ArtifactParser):
    name = "summary_schema"

    def matches(self, payload: dict[str, Any]) -> bool:
        return isinstance(payload.get("summary"), dict)

    def parse(self, path: str, row_id: str | None, payload: dict[str, Any]) -> ArtifactSummary:
        summary = payload["summary"]
        failed_criteria, null_criteria = extract_criteria(summary)
        baseline_fields = [
            dotted_path
            for dotted_path, _ in flatten_paths(summary)
            if path_matches(dotted_path, CONTROL_KEYWORDS)
        ][:80]
        intervention_fields = [
            dotted_path
            for dotted_path, _ in flatten_paths(summary)
            if path_matches(dotted_path, INTERVENTION_KEYWORDS)
        ][:80]
        return ArtifactSummary(
            path=path,
            row_id=row_id,
            parser=self.name,
            card_id=payload.get("card_id"),
            run_type=payload.get("run_type"),
            model_id=payload.get("model_id") or summary.get("model_id"),
            record_count=infer_record_count(payload),
            diagnostic_class=summary.get("diagnostic_class"),
            passed=first_present(payload, ["summary.passed", "passed"]),
            behavior_ready=first_present(
                payload,
                [
                    "summary.behavior_ready",
                    "summary.behavior_gate_passed",
                    "summary.criteria.behavior_ready",
                ],
            ),
            signature_ready=first_present(payload, ["summary.signature_ready"]),
            intervention_ready=first_present(payload, ["summary.intervention_ready"]),
            selected=extract_selected({"summary": summary}),
            failed_criteria=failed_criteria,
            null_criteria=null_criteria,
            baseline_fields=baseline_fields,
            intervention_fields=intervention_fields,
            metrics=extract_family_metrics(payload, summary),
        )


class LegacySchemaParser(ArtifactParser):
    name = "legacy_schema"

    def matches(self, payload: dict[str, Any]) -> bool:
        return True

    def parse(self, path: str, row_id: str | None, payload: dict[str, Any]) -> ArtifactSummary:
        baseline_fields = [
            dotted_path
            for dotted_path, _ in flatten_paths(payload)
            if path_matches(dotted_path, CONTROL_KEYWORDS)
        ][:80]
        intervention_fields = [
            dotted_path
            for dotted_path, _ in flatten_paths(payload)
            if path_matches(dotted_path, INTERVENTION_KEYWORDS)
        ][:80]
        return ArtifactSummary(
            path=path,
            row_id=row_id,
            parser=self.name,
            card_id=payload.get("card_id"),
            run_type=payload.get("run_type"),
            model_id=payload.get("model_id"),
            record_count=infer_record_count(payload),
            diagnostic_class=payload.get("diagnostic_class"),
            passed=payload.get("passed"),
            behavior_ready=payload.get("behavior_ready"),
            signature_ready=payload.get("signature_ready"),
            intervention_ready=payload.get("intervention_ready"),
            selected=extract_selected(payload),
            failed_criteria=[],
            null_criteria=[],
            baseline_fields=baseline_fields,
            intervention_fields=intervention_fields,
            metrics=extract_legacy_family_metrics(payload),
        )


PARSERS: tuple[ArtifactParser, ...] = (
    SummarySchemaParser(),
    LegacySchemaParser(),
)


def parse_artifact(path: Path, row_id: str | None = None) -> ArtifactSummary:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise AssertionError(f"{path}: artifact root must be an object")
    rel_path = path.relative_to(ROOT).as_posix()
    for parser in PARSERS:
        if parser.matches(payload):
            return parser.parse(rel_path, row_id, payload)
    raise AssertionError(f"{rel_path}: no parser matched")


def linked_artifact_paths(atlas: dict[str, Any]) -> list[tuple[str | None, str]]:
    paths: list[tuple[str | None, str]] = []
    seen_paths: set[str] = set()

    for row in atlas["rows"]:
        for rel_path in row["evidence"]:
            if rel_path.endswith(".json"):
                if rel_path not in seen_paths:
                    paths.append((row["id"], rel_path))
                    seen_paths.add(rel_path)

    for assertion in atlas.get("artifact_assertions", []):
        rel_path = assertion["path"]
        if rel_path not in seen_paths:
            paths.append((None, rel_path))
            seen_paths.add(rel_path)

    return paths


def extract_atlas_artifacts(atlas: dict[str, Any]) -> list[ArtifactSummary]:
    summaries: list[ArtifactSummary] = []
    for row_id, rel_path in linked_artifact_paths(atlas):
        path = ROOT / rel_path
        if not path.exists():
            raise AssertionError(f"missing linked artifact {rel_path}")
        summaries.append(parse_artifact(path, row_id=row_id))
    return summaries


def validate_row_artifact_consistency(
    atlas: dict[str, Any],
    artifact_summaries: list[ArtifactSummary],
) -> list[str]:
    """Return row/artifact contradictions.

    These checks are intentionally conservative. They only fail on direct
    contradictions between atlas readiness claims and linked artifact flags,
    not on missing fields or merely weak evidence.
    """

    rows = {row["id"]: row for row in atlas["rows"]}
    artifacts_by_row: dict[str, list[ArtifactSummary]] = {}
    for artifact in artifact_summaries:
        if artifact.row_id is not None:
            artifacts_by_row.setdefault(artifact.row_id, []).append(artifact)

    def row_has(row_id: str, predicate: Any) -> bool:
        return any(predicate(artifact) for artifact in artifacts_by_row.get(row_id, []))

    contradictions: list[str] = []
    for artifact in artifact_summaries:
        if artifact.row_id is None:
            continue
        row = rows[artifact.row_id]
        intervention_state = row["intervention"]["state"]
        lead_state = row["lead_time"]["state"]
        verdict_class = row["verdict"]["class"]

        if intervention_state == "not_allowed" and artifact.signature_ready is True:
            contradictions.append(
                f"{artifact.row_id}: {artifact.path} says signature_ready true "
                "while row intervention is not_allowed"
            )
        if intervention_state == "not_allowed" and artifact.intervention_ready is True:
            contradictions.append(
                f"{artifact.row_id}: {artifact.path} says intervention_ready true "
                "while row intervention is not_allowed"
            )
        if lead_state == "not_reached" and artifact.signature_ready is True:
            contradictions.append(
                f"{artifact.row_id}: {artifact.path} says signature_ready true "
                "while row lead_time is not_reached"
            )
        if verdict_class == "promoted_mechanism_card" and artifact.intervention_ready is False:
            contradictions.append(
                f"{artifact.row_id}: promoted row links {artifact.path} with "
                "intervention_ready false"
            )

    if "mc001_qwen3_0p6b_truth_agreement" in rows:
        if not row_has(
            "mc001_qwen3_0p6b_truth_agreement",
            lambda artifact: artifact.metrics.get(
                "generation.baseline.wrong_hint_conditions.truth_following_rate_parseable"
            )
            == 0.0
            and artifact.metrics.get(
                "generation.input_mask_hint_line.wrong_hint_conditions.truth_following_rate_parseable",
                0.0,
            )
            >= 0.60
            and artifact.metrics.get(
                "generation.rewrite_token_matched_neutral_hint_line.wrong_hint_conditions.truth_following_rate_parseable",
                0.0,
            )
            >= 0.50
            and artifact.metrics.get(
                "rewrite.rewrite_token_matched_neutral_hint_line.rendered_token_delta_mean"
            )
            == 0.0,
        ):
            contradictions.append(
                "mc001_qwen3_0p6b_truth_agreement: no linked artifact proves "
                "the source-mask effect and token-matched rewrite-equivalence boundary"
            )

    if "mc001b_qwen3_1p7b_truth_agreement" in rows:
        if not row_has(
            "mc001b_qwen3_1p7b_truth_agreement",
            lambda artifact: artifact.metrics.get(
                "probe.named_direction.raw_h21.direction_scalar.holdout.auc",
                0.0,
            )
            >= 0.95
            and artifact.metrics.get(
                "probe.named_direction.residual_h21.direction_scalar.holdout.auc",
                1.0,
            )
            <= 0.85,
        ):
            contradictions.append(
                "mc001b_qwen3_1p7b_truth_agreement: no linked artifact proves "
                "the raw h21 signal was strong while the residualized signal weakened"
            )

    if "mc001g_gemma_truth_agreement" in rows:
        if not row_has(
            "mc001g_gemma_truth_agreement",
            lambda artifact: artifact.metrics.get(
                "arm.baseline.matched_holdout.labels.truth_following"
            )
            == artifact.metrics.get("arm.layer14_alpha1.matched_holdout.labels.truth_following")
            == 7
            and artifact.metrics.get(
                "arm.baseline.matched_holdout.labels.user_agreement_error"
            )
            == artifact.metrics.get(
                "arm.layer14_alpha1.matched_holdout.labels.user_agreement_error"
            )
            == 7,
        ):
            contradictions.append(
                "mc001g_gemma_truth_agreement: no linked artifact proves the "
                "layer-14 intervention left matched holdout class counts unmoved"
            )

    if "mc002_known_unknown" in rows:
        if not row_has(
            "mc002_known_unknown",
            lambda artifact: artifact.metrics.get("audit.passed") is False
            and artifact.metrics.get("audit.fake_clean_source_count", 999) <= 1
            and artifact.metrics.get("audit.fake_pressure_hallucination_source_count", 0)
            >= 24,
        ):
            contradictions.append(
                "mc002_known_unknown: no linked artifact proves the base "
                "known/unknown behavior substrate failed with nonce hallucination pressure"
            )
        if not row_has(
            "mc002_known_unknown",
            lambda artifact: artifact.metrics.get("audit.passed") is False
            and artifact.metrics.get("audit.baseline_fake_clean_source_count") == 40
            and artifact.metrics.get("audit.baseline_real_clean_source_count", 0) >= 36,
        ):
            contradictions.append(
                "mc002_known_unknown: no linked artifact proves the calibrated "
                "IT repair still failed its behavior gate"
            )

    if "mc002b_context_support" in rows:
        if not row_has(
            "mc002b_context_support",
            lambda artifact: artifact.metrics.get("audit.passed") is False
            and artifact.metrics.get("audit.baseline_supported_clean_source_count", 0)
            >= 39
            and artifact.metrics.get("audit.baseline_unsupported_clean_source_count", 0)
            >= 35
            and artifact.metrics.get("overall.label_counts.near_neighbor_hallucination", 0)
            >= 16,
        ):
            contradictions.append(
                "mc002b_context_support: no linked artifact proves the support "
                "baseline improved while pressure behavior still failed"
            )

    if "mc003_delayed_copy" in rows:
        if not row_has(
            "mc003_delayed_copy",
            lambda artifact: artifact.metrics.get(
                "criteria.selected_hidden_holdout_auc_at_least_0_85"
            )
            is True
            and artifact.metrics.get("criteria.selected_hidden_above_shuffle_p95_by_0_05")
            is False
            and artifact.metrics.get("criteria.hidden_auc_beats_condition_trace_by_0_02")
            is False,
        ):
            contradictions.append(
                "mc003_delayed_copy: no linked artifact proves the early "
                "hidden signal failed shuffle and condition-trace controls"
            )

    if "mc004_in_context_binding" in rows:
        if not row_has(
            "mc004_in_context_binding",
            lambda artifact: artifact.metrics.get("criteria.selected_stage_before_answer_instruction")
            is True
            and artifact.metrics.get("criteria.selected_hidden_holdout_auc_at_least_0_85")
            is True
            and artifact.metrics.get(
                "criteria.hidden_auc_beats_same_stage_output_margin_by_0_02"
            )
            is True
            and artifact.metrics.get("criteria.selected_hidden_above_shuffle_p95_by_0_05")
            is False
            and artifact.metrics.get("criteria.target_order_subgroup_aucs_at_least_0_75")
            is False,
        ):
            contradictions.append(
                "mc004_in_context_binding: no linked artifact proves the "
                "lead-time signal passed same-stage output control but failed "
                "shuffle/subgroup reliability"
            )

    if "mc005_associative_lookup" in rows:
        if not row_has(
            "mc005_associative_lookup",
            lambda artifact: artifact.metrics.get("mc005.lookup_target_effect_reproduced")
            is True
            and artifact.metrics.get("mc005.write_layer_count") == [24, 25, 26]
            and artifact.metrics.get("criteria.lookup_loss_rows_baseline_margin_ge_2")
            is True,
        ):
            contradictions.append(
                "mc005_associative_lookup: no linked artifact proves the "
                "layers-24-26 attention-write boundary with high-margin lookup "
                "mediation"
            )
        if not row_has(
            "mc005_associative_lookup",
            lambda artifact: artifact.metrics.get("mc005.lookup_target_effect_reproduced")
            is True
            and isinstance(artifact.metrics.get("mc005.combined_null_flip_count"), int)
            and artifact.metrics["mc005.combined_null_flip_count"] > 0,
        ):
            contradictions.append(
                "mc005_associative_lookup: no linked artifact proves lookup "
                "effect reproduced with nonzero null flips"
            )
        if not row_has(
            "mc005_associative_lookup",
            lambda artifact: artifact.metrics.get(
                "criteria.v30_imported_flips_abs_margin_le_0p5"
            )
            is False
            and artifact.metrics.get("criteria.v31_fresh_null_flips_abs_margin_le_0p5")
            is True
            and artifact.metrics.get("criteria.v31_fresh_null_mean_deltas_within_0p25")
            is True
            and isinstance(artifact.metrics.get("mc005.combined_null_flip_count"), int)
            and artifact.metrics["mc005.combined_null_flip_count"] > 0,
        ):
            contradictions.append(
                "mc005_associative_lookup: no linked artifact proves the null "
                "boundary is not fully explained by a simple absolute 0.5 "
                "margin cutoff"
            )

    if "mc006_parametric_fact_override" in rows:
        if not row_has(
            "mc006_parametric_fact_override",
            lambda artifact: artifact.metrics.get("mc006.selected_pre_output_holdout_auc")
            == 1.0
            and artifact.metrics.get("mc006.candidate_score_holdout_auc") == 1.0
            and artifact.metrics.get("mc006.final_output_holdout_auc") == 1.0
            and artifact.metrics.get(
                "mechanism_criteria.selected_hidden_beats_candidate_score_by_0p02"
            )
            is False
            and artifact.metrics.get(
                "mechanism_criteria.selected_hidden_beats_final_next_token_output_by_0p02"
            )
            is False,
        ):
            contradictions.append(
                "mc006_parametric_fact_override: no linked artifact proves V16 "
                "is candidate-score and final-output confounded despite a "
                "strong pre-output hidden monitor"
            )
        if not row_has(
            "mc006_parametric_fact_override",
            lambda artifact: artifact.metrics.get("mc006.causal_stress_supported")
            is False
            and artifact.metrics.get(
                "causal_criteria.plus_selected_increases_margin_by_0p25"
            )
            is False
            and artifact.metrics.get(
                "causal_criteria.minus_selected_decreases_margin_by_0p25"
            )
            is False
            and artifact.metrics.get(
                "causal_criteria.selected_abs_effect_beats_controls_by_0p25"
            )
            is False
            and artifact.metrics.get(
                "mechanism_criteria.has_predicted_direction_generation_change"
            )
            is False
            and artifact.metrics.get("mc006.best_dose.total_predicted_direction_changes")
            == 0,
        ):
            contradictions.append(
                "mc006_parametric_fact_override: no linked artifact proves V17 "
                "additive steering failed causal/mechanism criteria with zero "
                "predicted-direction changes"
            )
        if not row_has(
            "mc006_parametric_fact_override",
            lambda artifact: artifact.metrics.get("criteria.pair_matched_diagnostic_ready")
            is True
            and artifact.metrics.get(
                "criteria.candidate_and_final_margin_holdout_pair_accuracy_at_most_0p65"
            )
            is False
            and artifact.metrics.get(
                "criteria.selected_hidden_holdout_pair_accuracy_at_least_0p65"
            )
            is False
            and artifact.metrics.get(
                "criteria.selected_hidden_beats_candidate_and_final_margins_by_0p05"
            )
            is False
            and artifact.metrics.get("intervention_ready") is False,
        ):
            contradictions.append(
                "mc006_parametric_fact_override: no linked artifact proves V21 "
                "pair matching remained margin-baseline dominated"
            )
        if not row_has(
            "mc006_parametric_fact_override",
            lambda artifact: artifact.metrics.get("criteria.curve_any_hidden_holdout_supported")
            is True
            and artifact.metrics.get(
                "criteria.curve_any_hidden_beats_candidate_and_final_margins_by_0p05"
            )
            is False
            and artifact.metrics.get(
                "criteria.selected_auc_hidden_holdout_auc_at_least_0p65"
            )
            is True
            and artifact.metrics.get(
                "criteria.selected_auc_hidden_beats_candidate_and_final_margins_by_0p05"
            )
            is False
            and artifact.metrics.get("intervention_ready") is False,
        ):
            contradictions.append(
                "mc006_parametric_fact_override: no linked artifact proves V22 "
                "source/path lead-time monitors stayed final-margin shadowed"
            )
        if not row_has(
            "mc006_parametric_fact_override",
            lambda artifact: artifact.metrics.get(
                "criteria.selected_final_city_margin_not_sign_barrier"
            )
            is True
            and artifact.metrics.get("criteria.selected_final_city_margin_overlap_exists")
            is True
            and artifact.metrics.get("criteria.selected_first_token_city_rate_at_most_0p1")
            is True
            and artifact.metrics.get("behavior_ready") is True
            and artifact.metrics.get("signature_ready") is False
            and artifact.metrics.get("intervention_ready") is False,
        ):
            contradictions.append(
                "mc006_parametric_fact_override: no linked artifact proves V24 "
                "broke the first-token city barrier while remaining monitor-only"
            )
        if not row_has(
            "mc006_parametric_fact_override",
            lambda artifact: artifact.metrics.get("criteria.candidate_decoupled_template_found")
            is True
            and artifact.metrics.get("criteria.selected_json_candidate_holdout_auc_not_perfect")
            is True
            and artifact.metrics.get("criteria.hidden_beats_candidate_controls")
            is True
            and artifact.metrics.get("criteria.hidden_beats_shuffle_null") is False
            and artifact.metrics.get("criteria.signature_gate") is False
            and artifact.metrics.get("intervention_ready") is False,
        ):
            contradictions.append(
                "mc006_parametric_fact_override: no linked artifact proves V25 "
                "candidate decoupling still failed the shuffled-label hidden "
                "selection null"
            )
        if not row_has(
            "mc006_parametric_fact_override",
            lambda artifact: artifact.metrics.get("mc006.hidden_beats_shuffle_null")
            is False,
        ):
            contradictions.append(
                "mc006_parametric_fact_override: no linked artifact proves the "
                "candidate-decoupled hidden selector failed shuffle nulls"
            )
        if not row_has(
            "mc006_parametric_fact_override",
            lambda artifact: artifact.metrics.get("mc006.transfer_ready_count") == 1
            and artifact.metrics.get("criteria.transfer_ready_templates_at_least_2")
            is False
            and artifact.metrics.get("criteria.transfer_repair_bank_ready") is False
            and artifact.metrics.get("mc006.combined_ready_template_count", 0) < 5
            and artifact.metrics.get("mc006.combined_pooled_binary_rows", 999) < 160,
        ):
            contradictions.append(
                "mc006_parametric_fact_override: no linked artifact proves V28 "
                "found one transfer-ready template but not a repair bank"
            )
        if not row_has(
            "mc006_parametric_fact_override",
            lambda artifact: artifact.metrics.get("mc006.transfer_ready_count") == 1,
        ):
            contradictions.append(
                "mc006_parametric_fact_override: no linked artifact proves the "
                "transfer-role repair found only one ready template"
            )

    if "mc007_semi_synthetic_familiar_entity_lookup" in rows:
        if not row_has(
            "mc007_semi_synthetic_familiar_entity_lookup",
            lambda artifact: artifact.metrics.get("criteria.contrast_present")
            is False
            and artifact.metrics.get("primary.artificial_value", 0) >= 75,
        ):
            contradictions.append(
                "mc007_semi_synthetic_familiar_entity_lookup: no linked artifact "
                "proves clean source-value lookup with contrast absent"
            )
        if not row_has(
            "mc007_semi_synthetic_familiar_entity_lookup",
            lambda artifact: artifact.metrics.get("selected_template")
            == "lookup_only"
            and artifact.metrics.get("criteria.contrast_present") is False
            and artifact.metrics.get("criteria.primary_parseability_at_least_90p")
            is True
            and artifact.metrics.get("primary.artificial_value", 0) >= 75
            and artifact.metrics.get("primary.real_prior", 999) == 0
            and artifact.metrics.get("panel.answer_absent_null.unknown", 0) >= 40
            and artifact.signature_ready is False,
        ):
            contradictions.append(
                "mc007_semi_synthetic_familiar_entity_lookup: no linked artifact "
                "proves V1 was source lookup without usable conflict"
            )
        if not row_has(
            "mc007_semi_synthetic_familiar_entity_lookup",
            lambda artifact: artifact.metrics.get("criteria.cross_dial_contrast_present")
            is True
            and artifact.metrics.get("criteria.cross_dial_contrast_balance_passed")
            is False
            and artifact.metrics.get("criteria.primary_parseability_at_least_90p")
            is False
            and artifact.metrics.get("criteria.authority_transition_observed")
            is False
            and artifact.metrics.get("primary.prior_or_lure", 0) >= 30
            and artifact.metrics.get("panel.answer_absent_null.unknown", 0) >= 40
            and artifact.signature_ready is False,
        ):
            contradictions.append(
                "mc007_semi_synthetic_familiar_entity_lookup: no linked artifact "
                "proves V2 authority-dial contrast failed parseability/balance"
            )
        if not row_has(
            "mc007_semi_synthetic_familiar_entity_lookup",
            lambda artifact: artifact.metrics.get("criteria.cross_panel_contrast_present")
            is True
            and artifact.metrics.get("criteria.cross_panel_behavior_gate_passed")
            is False
            and artifact.metrics.get("criteria.within_panel_behavior_gate_passed")
            is False
            and artifact.metrics.get("criteria.primary_parseability_at_least_90p")
            is False
            and artifact.metrics.get("panel.real_world_control.real_prior", 0) >= 34
            and artifact.metrics.get("panel.answer_absent_null.unknown", 0) >= 40
            and artifact.signature_ready is False,
        ):
            contradictions.append(
                "mc007_semi_synthetic_familiar_entity_lookup: no linked artifact "
                "proves V3 parseability repair still failed the behavior gate"
            )
        if not row_has(
            "mc007_semi_synthetic_familiar_entity_lookup",
            lambda artifact: artifact.metrics.get("criteria.controls_clean") is False,
        ):
            contradictions.append(
                "mc007_semi_synthetic_familiar_entity_lookup: no linked artifact "
                "proves the authority-interface controls failed"
            )
        if not row_has(
            "mc007_semi_synthetic_familiar_entity_lookup",
            lambda artifact: artifact.diagnostic_class == "control_panel_failed"
            and artifact.metrics.get("criteria.controls_clean") is False
            and artifact.signature_ready is False,
        ):
            contradictions.append(
                "mc007_semi_synthetic_familiar_entity_lookup: no linked artifact "
                "proves V4 source-declaration controls failed"
            )

    if "mc008_symbolic_fact_code_arbitration" in rows:
        if not row_has(
            "mc008_symbolic_fact_code_arbitration",
            lambda artifact: artifact.metrics.get("selected_template") == "symbol_field"
            and artifact.metrics.get("panel.synthetic_code_lookup.artificial_code", 0)
            >= 39
            and artifact.metrics.get(
                "panel.real_world_memory_control.real_or_lure_symbol",
                0,
            )
            >= 39
            and artifact.metrics.get("panel.answer_absent_null.unknown", 999) == 30
            and artifact.metrics.get("criteria.answer_absent_panel_f_unknown_at_least_90p")
            is False
            and artifact.metrics.get("criteria.primary_conflict_parseability_at_least_90p")
            is False
            and artifact.behavior_ready is False
            and artifact.signature_ready is False,
        ):
            contradictions.append(
                "mc008_symbolic_fact_code_arbitration: no linked artifact proves "
                "direct symbolic controls were clean before the null failure"
            )
        if not row_has(
            "mc008_symbolic_fact_code_arbitration",
            lambda artifact: artifact.metrics.get("selected_template")
            == "membership_authority_split"
            and artifact.metrics.get("panel.answer_absent_null.unknown", 0) >= 40
            and artifact.metrics.get("primary_conflict.real_or_lure_symbol", 999) <= 2
            and artifact.metrics.get("criteria.holdout_conflict_label_balance_passed")
            is False,
        ):
            contradictions.append(
                "mc008_symbolic_fact_code_arbitration: no linked artifact proves "
                "null repair with conflict contrast still absent"
            )
        if not row_has(
            "mc008_symbolic_fact_code_arbitration",
            lambda artifact: artifact.metrics.get("selected_template")
            == "membership_authority_split"
            and artifact.metrics.get("panel.synthetic_code_lookup.artificial_code", 0)
            >= 39
            and artifact.metrics.get("panel.real_world_memory_control.real_symbol", 0)
            >= 40
            and artifact.metrics.get("panel.answer_absent_null.unknown", 0) >= 40
            and artifact.metrics.get("primary_conflict.artificial_code", 0) >= 220
            and artifact.metrics.get("primary_conflict.real_or_lure_symbol", 999) <= 2
            and artifact.metrics.get("criteria.holdout_conflict_label_balance_passed")
            is False
            and artifact.signature_ready is False,
        ):
            contradictions.append(
                "mc008_symbolic_fact_code_arbitration: no linked artifact proves "
                "V2 repaired nulls while leaving conflict contrast absent"
            )

    if "mc009_derived_code_arbitration" in rows:
        if not row_has(
            "mc009_derived_code_arbitration",
            lambda artifact: artifact.metrics.get("selected_template")
            == "membership_authority_split"
            and artifact.metrics.get(
                "panel.synthetic_ordinal_code_lookup.derived_code",
                0,
            )
            >= 10
            and artifact.metrics.get(
                "panel.real_world_memory_control.real_or_lure_symbol",
                0,
            )
            >= 10
            and artifact.metrics.get("panel.answer_absent_null.unknown", 0) >= 10
            and artifact.metrics.get("criteria.synthetic_panel_a_derived_at_least_90p")
            is True
            and artifact.metrics.get("criteria.primary_conflict_parseability_at_least_90p")
            is False
            and artifact.metrics.get("criteria.holdout_conflict_label_balance_passed")
            is False
            and artifact.behavior_ready is False
            and artifact.intervention_ready is False,
        ):
            contradictions.append(
                "mc009_derived_code_arbitration: no linked artifact proves "
                "membership controls were clean while conflict rows failed"
            )
        if not row_has(
            "mc009_derived_code_arbitration",
            lambda artifact: artifact.metrics.get("selected_template")
            == "membership_authority_split"
            and artifact.metrics.get(
                "panel.synthetic_ordinal_code_lookup.derived_code",
                0,
            )
            >= 10
            and artifact.metrics.get("panel.answer_absent_null.unknown", 0) >= 10
            and artifact.metrics.get("primary_conflict.parseable_rate", 1.0) < 0.9,
        ):
            contradictions.append(
                "mc009_derived_code_arbitration: no linked artifact proves the "
                "membership-template control/conflict tradeoff"
            )
        if not row_has(
            "mc009_derived_code_arbitration",
            lambda artifact: artifact.metrics.get("selected_template")
            == "typed_slot_v2"
            and artifact.metrics.get("primary_conflict.parseable_rate", 0.0) >= 0.9
            and artifact.metrics.get("primary_conflict.real_or_lure_symbol", 0) >= 19
            and artifact.metrics.get(
                "panel.synthetic_ordinal_code_lookup.derived_code",
                999,
            )
            == 0
            and artifact.metrics.get("panel.answer_absent_null.unknown", 999) == 0,
        ):
            contradictions.append(
                "mc009_derived_code_arbitration: no linked artifact proves the "
                "typed-slot balance/control failure"
            )
        if not row_has(
            "mc009_derived_code_arbitration",
            lambda artifact: artifact.metrics.get("selected_template")
            == "typed_slot_v2"
            and artifact.metrics.get("primary_conflict.parseable_rate", 0.0) >= 0.9
            and artifact.metrics.get("primary_conflict.real_or_lure_symbol", 0) >= 19
            and artifact.metrics.get(
                "panel.synthetic_ordinal_code_lookup.derived_code",
                999,
            )
            == 0
            and artifact.metrics.get("panel.answer_absent_null.unknown", 999) == 0
            and artifact.metrics.get("criteria.synthetic_panel_a_derived_at_least_90p")
            is False
            and artifact.metrics.get("criteria.answer_absent_panel_f_unknown_at_least_90p")
            is False
            and artifact.behavior_ready is False,
        ):
            contradictions.append(
                "mc009_derived_code_arbitration: no linked artifact proves "
                "typed-slot balance came by breaking direct controls"
            )

    if "mc010_two_hop_fact_code_arbitration" in rows:
        if not row_has(
            "mc010_two_hop_fact_code_arbitration",
            lambda artifact: artifact.metrics.get("selected_template")
            == "neutral_contract"
            and artifact.metrics.get("criteria.synthetic_panel_a_task_at_least_90p")
            is False
            and artifact.metrics.get("criteria.familiar_panel_b_task_at_least_90p")
            is True
            and artifact.metrics.get("criteria.real_world_panel_c_real_at_least_85p")
            is False
            and artifact.metrics.get("criteria.answer_absent_panel_f_unknown_at_least_90p")
            is True
            and artifact.metrics.get("panel.synthetic_two_hop_lookup.task_code", 0)
            == 23
            and artifact.metrics.get("panel.familiar_entity_two_hop_lookup.task_code", 0)
            == 38
            and artifact.metrics.get("panel.real_world_memory_control.real_symbol", 0)
            == 12
            and artifact.metrics.get("panel.answer_absent_null.unknown", 0) == 36
            and artifact.behavior_ready is False
            and artifact.signature_ready is False,
        ):
            contradictions.append(
                "mc010_two_hop_fact_code_arbitration: no linked artifact proves "
                "the full behavior gate failed on synthetic lookup and real "
                "memory controls while familiar lookup and null controls held"
            )
        if not row_has(
            "mc010_two_hop_fact_code_arbitration",
            lambda artifact: artifact.metrics.get("selected_template")
            == "neutral_contract"
            and artifact.metrics.get("primary_conflict.parseable_rate", 0.0) == 1.0
            and artifact.metrics.get("primary_conflict.task_code", 0) >= 229
            and artifact.metrics.get("primary_conflict.real_or_lure_symbol", 999) == 0
            and artifact.metrics.get("criteria.candidate_and_output_margins_reported")
            is True
            and artifact.metrics.get("criteria.holdout_conflict_label_balance_passed")
            is False
            and artifact.intervention_ready is False,
        ):
            contradictions.append(
                "mc010_two_hop_fact_code_arbitration: no linked artifact proves "
                "the two-hop primary conflict stayed table-code dominant with "
                "candidate/output margins reported"
            )

    if "mc011_atomic_number_code_arbitration" in rows:
        if not row_has(
            "mc011_atomic_number_code_arbitration",
            lambda artifact: artifact.metrics.get("selected_template")
            == "neutral_numeric"
            and artifact.metrics.get("criteria.synthetic_panel_a_local_at_least_90p")
            is True
            and artifact.metrics.get("criteria.familiar_panel_b_local_at_least_90p")
            is True
            and artifact.metrics.get("criteria.real_world_panel_c_atomic_at_least_85p")
            is True
            and artifact.metrics.get("criteria.answer_absent_panel_f_unknown_at_least_90p")
            is True
            and artifact.metrics.get("panel.synthetic_numeric_lookup.local_number", 0)
            == 40
            and artifact.metrics.get("panel.familiar_entity_numeric_lookup.local_number", 0)
            == 40
            and artifact.metrics.get("panel.real_world_atomic_number_control.atomic_number", 0)
            == 40
            and artifact.metrics.get("panel.answer_absent_null.unknown", 0) == 40
            and artifact.behavior_ready is False
            and artifact.signature_ready is False,
        ):
            contradictions.append(
                "mc011_atomic_number_code_arbitration: no linked artifact proves "
                "the numeric bridge direct controls all passed while the behavior "
                "gate still blocked hidden-state work"
            )
        if not row_has(
            "mc011_atomic_number_code_arbitration",
            lambda artifact: artifact.metrics.get("selected_template")
            == "neutral_numeric"
            and artifact.metrics.get("primary_conflict.parseable_rate", 0.0) == 1.0
            and artifact.metrics.get("primary_conflict.local_number", 0) == 240
            and artifact.metrics.get("primary_conflict.atomic_or_lure_number", 999)
            == 0
            and artifact.metrics.get("criteria.candidate_and_output_margins_reported")
            is True
            and artifact.metrics.get("criteria.holdout_conflict_label_balance_passed")
            is False
            and artifact.intervention_ready is False,
        ):
            contradictions.append(
                "mc011_atomic_number_code_arbitration: no linked artifact proves "
                "the numeric primary conflict stayed fully table-local with "
                "candidate/output margins reported"
            )

    if "mc012_reliability_labeled_numeric_arbitration" in rows:
        if not row_has(
            "mc012_reliability_labeled_numeric_arbitration",
            lambda artifact: artifact.metrics.get("selected_template")
            == "compact_reliability"
            and artifact.metrics.get("criteria.synthetic_panel_a_local_at_least_90p")
            is True
            and artifact.metrics.get("criteria.familiar_panel_b_local_at_least_90p")
            is True
            and artifact.metrics.get("criteria.real_world_panel_c_atomic_at_least_85p")
            is True
            and artifact.metrics.get("criteria.answer_absent_panel_f_unknown_at_least_90p")
            is True
            and artifact.metrics.get("criteria.trusted_conflict_local_at_least_85p")
            is True
            and artifact.metrics.get("criteria.untrusted_conflict_atomic_at_least_85p")
            is True
            and artifact.metrics.get("criteria.non_holdout_conflict_label_balance_passed")
            is True
            and artifact.metrics.get("criteria.holdout_conflict_label_balance_passed")
            is True
            and artifact.metrics.get("panel.synthetic_numeric_lookup.local_number", 0)
            == 40
            and artifact.metrics.get("panel.familiar_entity_numeric_lookup.local_number", 0)
            == 40
            and artifact.metrics.get("panel.real_world_atomic_number_control.atomic_number", 0)
            == 40
            and artifact.metrics.get("panel.trusted_source_conflict.local_number", 0)
            == 40
            and artifact.metrics.get("panel.untrusted_source_conflict.atomic_number", 0)
            == 39
            and artifact.metrics.get("panel.answer_absent_null.unknown", 0) == 40
            and artifact.behavior_ready is True
            and artifact.signature_ready is False,
        ):
            contradictions.append(
                "mc012_reliability_labeled_numeric_arbitration: no linked "
                "artifact proves the reliability-labeled bridge passed direct "
                "controls, nulls, and local-versus-atomic conflict balance"
            )
        if not row_has(
            "mc012_reliability_labeled_numeric_arbitration",
            lambda artifact: artifact.metrics.get("selected_template")
            == "compact_reliability"
            and artifact.metrics.get("primary_conflict.parseable_rate", 0.0) == 1.0
            and artifact.metrics.get("primary_conflict.local_number", 0) == 40
            and artifact.metrics.get("primary_conflict.atomic_or_lure_number", 0)
            == 39
            and artifact.metrics.get("primary_conflict.binary_conflict", 0) == 79
            and artifact.metrics.get("criteria.candidate_and_output_margins_reported")
            is True
            and artifact.metrics.get("criteria.prompt_channel_contrast_visible_by_design")
            is True
            and artifact.metrics.get("criteria.prompt_channel_locality_gate_passed")
            is False
            and artifact.intervention_ready is False,
        ):
            contradictions.append(
                "mc012_reliability_labeled_numeric_arbitration: no linked "
                "artifact proves the mixed behavior table is prompt-channel "
                "visible and not signature-ready"
            )

    if "mc013_status_channel_ablation_numeric_arbitration" in rows:
        if not row_has(
            "mc013_status_channel_ablation_numeric_arbitration",
            lambda artifact: artifact.metrics.get("selected_template")
            == "compact_status_ablation"
            and artifact.metrics.get("criteria.synthetic_panel_a_local_at_least_90p")
            is True
            and artifact.metrics.get("criteria.familiar_panel_b_local_at_least_90p")
            is True
            and artifact.metrics.get("criteria.real_world_panel_c_atomic_at_least_85p")
            is True
            and artifact.metrics.get("criteria.answer_absent_panel_f_unknown_at_least_90p")
            is True
            and artifact.metrics.get("criteria.statused_trusted_conflict_local_at_least_85p")
            is True
            and artifact.metrics.get("criteria.statused_untrusted_conflict_atomic_at_least_85p")
            is True
            and artifact.metrics.get("panel.statused_trusted_conflict.local_number", 0)
            == 40
            and artifact.metrics.get("panel.statused_untrusted_conflict.atomic_number", 0)
            == 39
            and artifact.metrics.get("statused_conflict.local_number", 0) == 40
            and artifact.metrics.get("statused_conflict.atomic_or_lure_number", 0)
            == 39
            and artifact.signature_ready is False,
        ):
            contradictions.append(
                "mc013_status_channel_ablation_numeric_arbitration: no linked "
                "artifact proves the statused positive control reproduced the "
                "MC012 local-versus-atomic contrast"
            )
        if not row_has(
            "mc013_status_channel_ablation_numeric_arbitration",
            lambda artifact: artifact.metrics.get("selected_template")
            == "compact_status_ablation"
            and artifact.metrics.get("criteria.ablation_prompt_pairs_identical")
            is True
            and artifact.metrics.get("criteria.primary_ablation_parseability_at_least_90p")
            is True
            and artifact.metrics.get("criteria.primary_ablation_binary_rows_at_least_40")
            is True
            and artifact.metrics.get("criteria.ablation_untrusted_atomic_at_least_85p")
            is False
            and artifact.metrics.get("criteria.non_holdout_ablation_label_balance_passed")
            is False
            and artifact.metrics.get("criteria.holdout_ablation_label_balance_passed")
            is False
            and artifact.metrics.get("criteria.candidate_and_output_margins_reported")
            is True
            and artifact.metrics.get("primary_ablation_conflict.local_number", 0)
            == 80
            and artifact.metrics.get("primary_ablation_conflict.atomic_or_lure_number", 999)
            == 0
            and artifact.metrics.get("primary_ablation_conflict.binary_conflict", 0)
            == 80
            and artifact.behavior_ready is False
            and artifact.signature_ready is False
            and artifact.intervention_ready is False,
        ):
            contradictions.append(
                "mc013_status_channel_ablation_numeric_arbitration: no linked "
                "artifact proves matched source-status ablation collapsed the "
                "contrast to local-number answers"
            )

    if "mc014_inferred_reliability_numeric_arbitration" in rows:
        if not row_has(
            "mc014_inferred_reliability_numeric_arbitration",
            lambda artifact: artifact.metrics.get("selected_template")
            == "calibration_rule"
            and artifact.metrics.get("criteria.synthetic_panel_a_local_at_least_90p")
            is True
            and artifact.metrics.get("criteria.familiar_panel_b_local_at_least_90p")
            is True
            and artifact.metrics.get("criteria.real_world_panel_c_atomic_at_least_85p")
            is True
            and artifact.metrics.get("criteria.answer_absent_panel_g_unknown_at_least_90p")
            is True
            and artifact.metrics.get("criteria.primary_prompts_have_no_status_lexemes")
            is True
            and artifact.metrics.get("panel.synthetic_numeric_lookup.local_number", 0)
            == 40
            and artifact.metrics.get("panel.familiar_entity_numeric_lookup.local_number", 0)
            == 40
            and artifact.metrics.get("panel.real_world_atomic_number_control.atomic_number", 0)
            == 40
            and artifact.metrics.get("panel.answer_absent_null.unknown", 0) == 40
            and artifact.behavior_ready is False
            and artifact.signature_ready is False,
        ):
            contradictions.append(
                "mc014_inferred_reliability_numeric_arbitration: no linked "
                "artifact proves direct controls and nulls passed while primary "
                "prompts had no explicit status labels"
            )
        if not row_has(
            "mc014_inferred_reliability_numeric_arbitration",
            lambda artifact: artifact.metrics.get("selected_template")
            == "calibration_rule"
            and artifact.metrics.get("criteria.consistent_conflict_local_at_least_85p")
            is True
            and artifact.metrics.get("criteria.inconsistent_conflict_atomic_at_least_85p")
            is False
            and artifact.metrics.get("criteria.non_holdout_conflict_label_balance_passed")
            is False
            and artifact.metrics.get("criteria.holdout_conflict_label_balance_passed")
            is False
            and artifact.metrics.get("criteria.candidate_and_output_margins_reported")
            is True
            and artifact.metrics.get("primary_conflict.parseable_rate", 0.0) == 1.0
            and artifact.metrics.get("primary_conflict.local_number", 0) == 80
            and artifact.metrics.get("primary_conflict.atomic_or_lure_number", 999)
            == 0
            and artifact.metrics.get("primary_conflict.binary_conflict", 0) == 80
            and artifact.behavior_ready is False
            and artifact.signature_ready is False
            and artifact.intervention_ready is False,
        ):
            contradictions.append(
                "mc014_inferred_reliability_numeric_arbitration: no linked "
                "artifact proves inferred calibration collapsed to local-number "
                "answers despite candidate/output margin reporting"
            )

    if "mc015_parity_gated_numeric_arbitration" in rows:
        if not row_has(
            "mc015_parity_gated_numeric_arbitration",
            lambda artifact: artifact.metrics.get("selected_template") == "parity_rule"
            and artifact.metrics.get("criteria.synthetic_panel_a_local_at_least_90p")
            is True
            and artifact.metrics.get("criteria.familiar_panel_b_local_at_least_90p")
            is True
            and artifact.metrics.get("criteria.real_world_panel_c_atomic_at_least_85p")
            is True
            and artifact.metrics.get("criteria.answer_absent_panel_g_unknown_at_least_90p")
            is True
            and artifact.metrics.get("criteria.primary_prompts_have_no_status_lexemes")
            is True
            and artifact.metrics.get("criteria.primary_expected_labels_balanced")
            is True
            and artifact.metrics.get("criteria.primary_expected_labels_balanced_by_split")
            is True
            and artifact.metrics.get("panel.synthetic_numeric_lookup.local_number", 0)
            == 40
            and artifact.metrics.get("panel.familiar_entity_numeric_lookup.local_number", 0)
            == 40
            and artifact.metrics.get("panel.real_world_atomic_number_control.atomic_number", 0)
            == 40
            and artifact.metrics.get("panel.answer_absent_null.unknown", 0) == 40
            and artifact.behavior_ready is False
            and artifact.signature_ready is False,
        ):
            contradictions.append(
                "mc015_parity_gated_numeric_arbitration: no linked artifact "
                "proves direct controls, nulls, hidden target/lure numbers, "
                "status-label absence, and balanced expected labels"
            )
        if not row_has(
            "mc015_parity_gated_numeric_arbitration",
            lambda artifact: artifact.metrics.get("selected_template") == "parity_rule"
            and artifact.metrics.get("criteria.expected_local_conflict_local_at_least_85p")
            is False
            and artifact.metrics.get("criteria.expected_atomic_conflict_atomic_at_least_85p")
            is False
            and artifact.metrics.get("criteria.primary_conflict_expected_correct_at_least_85p")
            is False
            and artifact.metrics.get("criteria.non_holdout_conflict_label_balance_passed")
            is True
            and artifact.metrics.get("criteria.holdout_conflict_label_balance_passed")
            is True
            and artifact.metrics.get("criteria.candidate_and_output_margins_reported")
            is True
            and artifact.metrics.get("primary_conflict.parseable_rate", 0.0) == 1.0
            and artifact.metrics.get("primary_conflict.local_number", 0) == 54
            and artifact.metrics.get("primary_conflict.atomic_number", 0) == 24
            and artifact.metrics.get("primary_conflict.other_number", 0) == 2
            and artifact.metrics.get("primary_conflict.binary_conflict", 0) == 78
            and artifact.metrics.get("primary_conflict.expected_correct", 0) == 39
            and artifact.metrics.get("primary_conflict.expected_correct_rate", 0.0)
            == 0.4875
            and artifact.metrics.get("primary_conflict_expected_local.local_number", 0)
            == 27
            and artifact.metrics.get("primary_conflict_expected_atomic.atomic_number", 0)
            == 12
            and artifact.behavior_ready is False
            and artifact.signature_ready is False
            and artifact.intervention_ready is False,
        ):
            contradictions.append(
                "mc015_parity_gated_numeric_arbitration: no linked artifact "
                "proves mixed local/atomic outputs failed to follow the "
                "learned parity gate"
            )

    if "mc016_alphabet_gated_numeric_arbitration" in rows:
        if not row_has(
            "mc016_alphabet_gated_numeric_arbitration",
            lambda artifact: artifact.metrics.get("selected_template") == "alphabet_rule"
            and artifact.metrics.get("criteria.synthetic_panel_a_local_at_least_90p")
            is True
            and artifact.metrics.get("criteria.familiar_panel_b_local_at_least_90p")
            is True
            and artifact.metrics.get("criteria.real_world_panel_c_atomic_at_least_85p")
            is True
            and artifact.metrics.get("criteria.answer_absent_panel_g_unknown_at_least_90p")
            is True
            and artifact.metrics.get("criteria.primary_prompts_have_no_status_lexemes")
            is True
            and artifact.metrics.get("criteria.primary_expected_labels_balanced")
            is True
            and artifact.metrics.get("criteria.primary_expected_labels_balanced_by_split")
            is True
            and artifact.metrics.get("panel.synthetic_numeric_lookup.local_number", 0)
            == 40
            and artifact.metrics.get("panel.familiar_entity_numeric_lookup.local_number", 0)
            == 40
            and artifact.metrics.get("panel.real_world_atomic_number_control.atomic_number", 0)
            == 40
            and artifact.metrics.get("panel.answer_absent_null.unknown", 0) == 40
            and artifact.behavior_ready is False
            and artifact.signature_ready is False,
        ):
            contradictions.append(
                "mc016_alphabet_gated_numeric_arbitration: no linked artifact "
                "proves direct controls, nulls, hidden target/lure numbers, "
                "status-label absence, and balanced expected labels"
            )
        if not row_has(
            "mc016_alphabet_gated_numeric_arbitration",
            lambda artifact: artifact.metrics.get("selected_template") == "alphabet_rule"
            and artifact.metrics.get("criteria.expected_local_conflict_local_at_least_85p")
            is True
            and artifact.metrics.get("criteria.expected_atomic_conflict_atomic_at_least_85p")
            is False
            and artifact.metrics.get("criteria.primary_conflict_expected_correct_at_least_85p")
            is False
            and artifact.metrics.get("criteria.non_holdout_conflict_label_balance_passed")
            is False
            and artifact.metrics.get("criteria.holdout_conflict_label_balance_passed")
            is False
            and artifact.metrics.get("criteria.candidate_and_output_margins_reported")
            is True
            and artifact.metrics.get("primary_conflict.parseable_rate", 0.0) == 1.0
            and artifact.metrics.get("primary_conflict.local_number", 0) == 80
            and artifact.metrics.get("primary_conflict.atomic_number", 0) == 0
            and artifact.metrics.get("primary_conflict.other_number", 0) == 0
            and artifact.metrics.get("primary_conflict.atomic_or_lure_number", 0)
            == 0
            and artifact.metrics.get("primary_conflict.binary_conflict", 0) == 80
            and artifact.metrics.get("primary_conflict.expected_correct", 0) == 40
            and artifact.metrics.get("primary_conflict.expected_correct_rate", 0.0)
            == 0.5
            and artifact.metrics.get("primary_conflict_expected_local.local_number", 0)
            == 40
            and artifact.metrics.get("primary_conflict_expected_local.expected_correct", 0)
            == 40
            and artifact.metrics.get("primary_conflict_expected_atomic.local_number", 0)
            == 40
            and artifact.metrics.get("primary_conflict_expected_atomic.atomic_number", 0)
            == 0
            and artifact.metrics.get("primary_conflict_expected_atomic.expected_correct", 0)
            == 0
            and artifact.behavior_ready is False
            and artifact.signature_ready is False
            and artifact.intervention_ready is False,
        ):
            contradictions.append(
                "mc016_alphabet_gated_numeric_arbitration: no linked artifact "
                "proves visible non-status alphabet gates collapsed to local "
                "numbers on expected-atomic rows"
            )

    return contradictions


def validate_atlas_artifact_registry(atlas: dict[str, Any]) -> list[ArtifactSummary]:
    summaries = extract_atlas_artifacts(atlas)
    contradictions = validate_row_artifact_consistency(atlas, summaries)
    if contradictions:
        raise AssertionError(
            "artifact registry contradictions:\n" + "\n".join(contradictions)
        )
    return summaries


def registry_report(summaries: list[ArtifactSummary]) -> dict[str, Any]:
    parser_counts: dict[str, int] = {}
    row_counts: dict[str, int] = {}
    diagnostic_counts: dict[str, int] = {}
    missing_summary_ready_flags = 0
    artifacts_with_metrics = 0
    for summary in summaries:
        parser_counts[summary.parser] = parser_counts.get(summary.parser, 0) + 1
        row_key = summary.row_id or "<assertion_only>"
        row_counts[row_key] = row_counts.get(row_key, 0) + 1
        if summary.diagnostic_class:
            diagnostic_counts[summary.diagnostic_class] = (
                diagnostic_counts.get(summary.diagnostic_class, 0) + 1
            )
        if summary.signature_ready is None and summary.intervention_ready is None:
            missing_summary_ready_flags += 1
        if summary.metrics:
            artifacts_with_metrics += 1
    return {
        "artifact_count": len(summaries),
        "parser_counts": dict(sorted(parser_counts.items())),
        "row_counts": dict(sorted(row_counts.items())),
        "diagnostic_counts": dict(sorted(diagnostic_counts.items())),
        "missing_signature_and_intervention_flags": missing_summary_ready_flags,
        "artifacts_with_metrics": artifacts_with_metrics,
        "family_claim_checks": list(FAMILY_CLAIM_CHECKS),
        "family_claim_check_rows": dict(sorted(FAMILY_CLAIM_CHECK_ROWS.items())),
    }


def compact_artifact_entry(summary: ArtifactSummary) -> dict[str, Any]:
    return {
        "path": summary.path,
        "row_id": summary.row_id,
        "parser": summary.parser,
        "card_id": summary.card_id,
        "run_type": summary.run_type,
        "model_id": summary.model_id,
        "record_count": summary.record_count,
        "diagnostic_class": summary.diagnostic_class,
        "passed": summary.passed,
        "behavior_ready": summary.behavior_ready,
        "signature_ready": summary.signature_ready,
        "intervention_ready": summary.intervention_ready,
        "failed_criteria": summary.failed_criteria,
        "null_criteria": summary.null_criteria,
        "selected_field_count": len(summary.selected),
        "baseline_field_count": len(summary.baseline_fields),
        "intervention_field_count": len(summary.intervention_fields),
        "metrics": dict(sorted(summary.metrics.items())),
    }


def build_artifact_index(atlas: dict[str, Any]) -> dict[str, Any]:
    summaries = validate_atlas_artifact_registry(atlas)
    report = registry_report(summaries)
    return {
        "schema_version": 1,
        "updated_at": atlas.get("updated_at"),
        "atlas_ref": str(ATLAS_PATH.relative_to(ROOT)).replace("\\", "/"),
        "source": "code/control_surface_artifacts.py",
        "purpose": (
            "Compact normalized facts extracted from atlas-linked control-surface "
            "result artifacts. This is an audit index, not a replacement for raw "
            "artifacts or status cards."
        ),
        "report": report,
        "artifacts": [compact_artifact_entry(summary) for summary in summaries],
    }


def write_artifact_index(atlas: dict[str, Any], output_path: Path = ARTIFACT_INDEX_PATH) -> dict[str, Any]:
    artifact_index = build_artifact_index(atlas)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(artifact_index, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return artifact_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="print normalized artifacts as JSON")
    parser.add_argument(
        "--write-index",
        action="store_true",
        help="write compact normalized artifact index to data/control_surface_artifact_index.json",
    )
    args = parser.parse_args()

    atlas = load_json(ATLAS_PATH)
    summaries = validate_atlas_artifact_registry(atlas)
    report = registry_report(summaries)
    if args.write_index:
        artifact_index = write_artifact_index(atlas)
        print(
            f"wrote {ARTIFACT_INDEX_PATH.relative_to(ROOT).as_posix()} "
            f"with {artifact_index['report']['artifact_count']} artifacts"
        )
        return
    if args.json:
        print(
            json.dumps(
                {
                    "report": report,
                    "artifacts": [asdict(summary) for summary in summaries],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"artifact registry ok: {report['artifact_count']} linked artifacts")
        print("parser_counts:", json.dumps(report["parser_counts"], sort_keys=True))
        print("row_counts:", json.dumps(report["row_counts"], sort_keys=True))
        print(
            "diagnostic_counts:",
            json.dumps(report["diagnostic_counts"], sort_keys=True),
        )
        print(
            "missing_signature_and_intervention_flags:",
            report["missing_signature_and_intervention_flags"],
        )
        print("artifacts_with_metrics:", report["artifacts_with_metrics"])
        print("family_claim_checks:", json.dumps(report["family_claim_checks"]))


if __name__ == "__main__":
    main()
