"""Build a typed failure taxonomy from smoke and bridge diagnostics.

The atlas rows say which mechanism-card routes survived. The smoke and bridge
layers say why most routes died before hidden-state work. This module makes
those deaths a first-class artifact, with detailed audits for the current
MC028 other-number leak boundary, the MC029 factorized leak tradeoff, the
MC030 absence-guard repair failure, the MC031 statusless checksum collapse,
the MC032 cross-table consistency collapse, and the MC033 fact-claim closeout.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from control_surface_artifacts import ROOT, load_json
from control_surface_bridge_ladder import BRIDGE_LADDER_PATH
from control_surface_route_disposition import ROUTE_DISPOSITION_PATH
from control_surface_smoke_diagnostics import SMOKE_DIAGNOSTICS_PATH


ERROR_TAXONOMY_PATH = ROOT / "data" / "control_surface_error_taxonomy.json"
ERROR_TAXONOMY_REPORT_PATH = ROOT / "research" / "33_CONTROL_SURFACE_ERROR_TAXONOMY.md"

ROW_RE = re.compile(r"^(.+?) has local lab number (\d+)\.$")
EXAMPLE_RE = re.compile(r"^Example: (.+?) with operation (\w+) -> (\d+)\.$")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")


def ratio(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(count / total, 6)


def count_values(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def smoke_card_by_id(smoke: dict[str, Any], card_id: str) -> dict[str, Any]:
    matches = [card for card in smoke["cards"] if card["card_id"] == card_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one smoke card {card_id}, found {len(matches)}")
    return matches[0]


def smoke_panel(smoke_card: dict[str, Any], panel_name: str) -> dict[str, Any]:
    return smoke_card["selected_metrics"]["panel_metrics"][panel_name]


def prompt_number_sets(prompt: str) -> tuple[set[str], set[str]]:
    prompt_local_numbers: set[str] = set()
    example_outputs: set[str] = set()
    for line in prompt.splitlines():
        row_match = ROW_RE.match(line)
        if row_match:
            prompt_local_numbers.add(row_match.group(2))
            continue
        example_match = EXAMPLE_RE.match(line)
        if example_match:
            example_outputs.add(example_match.group(3))
    return prompt_local_numbers, example_outputs


def mc028_other_number_audit(mc028_artifact: dict[str, Any]) -> dict[str, Any]:
    rows = mc028_artifact["summary"]["selected_template_rows"]
    all_atomic = {
        row["atomic_number"]: row["element"]
        for row in rows
        if "atomic_number" in row and row.get("atomic_number") is not None
    }
    all_local = {
        row["local_number"]: row["element"]
        for row in rows
        if "local_number" in row and row.get("local_number") is not None
    }
    op_atomic_rows = [
        row for row in rows if row["panel"] == "operation_atomic_interface_conflict"
    ]
    other_rows = [
        row for row in op_atomic_rows if row.get("selected_label") == "other_number"
    ]

    exclusive_bucket_counts: Counter[str] = Counter()
    overlap_counts: Counter[str] = Counter()
    rows_by_bucket: dict[str, list[dict[str, Any]]] = {}

    for row in other_rows:
        selected = str(row["selected_answer"])
        prompt_local_numbers, example_outputs = prompt_number_sets(row["prompt"])
        labels: list[str] = []
        if selected in example_outputs:
            labels.append("worked_example_output")
        if selected in prompt_local_numbers:
            labels.append("prompt_local_row_number")
        if selected in all_local:
            labels.append("bank_local_number")
        if selected in all_atomic:
            labels.append("bank_atomic_number")
        atomic_value = int(row["atomic_number"])
        if selected == str(atomic_value * 2):
            labels.append("double_expected_atomic")
        if selected == str(atomic_value + 10):
            labels.append("expected_atomic_plus_10")
        if not labels:
            labels.append("off_bank_other_number")

        for label in labels:
            overlap_counts[label] += 1

        if "worked_example_output" in labels:
            bucket = "worked_example_output"
        elif "prompt_local_row_number" in labels:
            bucket = "prompt_local_row_number"
        elif "bank_local_number" in labels:
            bucket = "bank_local_number"
        elif "bank_atomic_number" in labels:
            bucket = "bank_atomic_number"
        else:
            bucket = "off_bank_other_number"

        exclusive_bucket_counts[bucket] += 1
        rows_by_bucket.setdefault(bucket, []).append(
            {
                "id": row["id"],
                "source_id": row["source_id"],
                "element": row["element"],
                "expected_atomic_number": row["atomic_number"],
                "local_number": row["local_number"],
                "selected_answer": selected,
                "operation_assignment": row["operation_assignment"],
                "rule_order": row["rule_order"],
                "overlap_labels": labels,
            }
        )

    panels = mc028_artifact["summary"]["selected_template_summary"]["panels"]
    return {
        "artifact_path": rel(ROOT / smoke_card_by_id(load_json(SMOKE_DIAGNOSTICS_PATH), "MC028")["artifact_path"]),
        "operation_atomic_rows": len(op_atomic_rows),
        "operation_atomic_atomic_rate": panels["operation_atomic_interface_conflict"]["atomic_number_rate"],
        "operation_atomic_other_number_count": len(other_rows),
        "operation_atomic_other_number_rate": ratio(len(other_rows), len(op_atomic_rows)),
        "operation_local_local_rate": panels["operation_local_interface_conflict"]["local_number_rate"],
        "direct_atomic_control_atomic_rate": panels["atomic_interface_control"]["atomic_number_rate"],
        "answer_absent_unknown_rate": panels["answer_absent_interface_null"]["unknown_rate"],
        "rule_absent_unknown_rate": panels["operation_rule_absent_interface_null"]["unknown_rate"],
        "exclusive_bucket_counts": dict(sorted(exclusive_bucket_counts.items())),
        "overlap_counts": dict(sorted(overlap_counts.items())),
        "rows_by_exclusive_bucket": {
            bucket: rows_by_bucket[bucket] for bucket in sorted(rows_by_bucket)
        },
        "interpretation": (
            "The MC028 failure is not one generic local-copy error. Wrong answers "
            "split across worked-example outputs, other bank atomic facts, bank "
            "local numbers, prompt-local numbers, and off-bank periodic-like "
            "numbers while direct atomic recall and null controls remain clean."
        ),
    }


def mc029_factorial_audit(mc029_artifact: dict[str, Any], artifact_path: str) -> dict[str, Any]:
    summary = mc029_artifact["summary"]
    criteria = summary["criteria"]
    by_template = summary["by_template"]
    template_order = [
        "baseline_numeric_examples",
        "label_examples_no_numbers",
        "rules_only",
        "query_before_examples",
        "query_row_last",
    ]

    variant_rows = []
    for template in template_order:
        template_summary = by_template[template]
        panels = template_summary["panels"]
        op_atomic = panels["operation_atomic_conflict"]
        op_local = panels["operation_local_conflict"]
        answer_absent = panels["answer_absent_null"]
        other_audit = template_summary.get("other_number_audit", {})
        bucket_counts = other_audit.get("exclusive_bucket_counts", {})
        variant_rows.append(
            {
                "template": template,
                "variant_gate_passed": criteria["variant_gate_by_template"][template],
                "operation_atomic_atomic_rate": op_atomic["atomic_number_rate"],
                "operation_atomic_local_number_rate": op_atomic["local_number_rate"],
                "operation_atomic_other_number_rate": op_atomic["other_number_rate"],
                "operation_local_local_rate": op_local["local_number_rate"],
                "answer_absent_unknown_rate": answer_absent["unknown_rate"],
                "worked_example_other_count": bucket_counts.get(
                    "worked_example_output",
                    0,
                ),
                "other_number_bucket_counts": bucket_counts,
            }
        )

    return {
        "artifact_path": artifact_path,
        "structural_passed": criteria["structural_passed"],
        "variant_gate_by_template": criteria["variant_gate_by_template"],
        "any_variant_behavior_gate_passed": criteria["any_variant_behavior_gate_passed"],
        "variant_rows": variant_rows,
        "baseline_operation_atomic_rate": criteria["baseline_operation_atomic_rate"],
        "baseline_other_number_rate": criteria["baseline_other_number_rate"],
        "max_operation_atomic_template": criteria["max_operation_atomic_template"],
        "max_operation_atomic_rate": criteria["max_operation_atomic_rate"],
        "min_other_number_template": criteria["min_other_number_template"],
        "min_other_number_rate": criteria["min_other_number_rate"],
        "rules_only_operation_atomic_rate": criteria["rules_only_operation_atomic_rate"],
        "rules_only_other_number_rate": criteria["rules_only_other_number_rate"],
        "rules_only_answer_absent_unknown_rate": criteria[
            "rules_only_answer_absent_unknown_rate"
        ],
        "baseline_worked_example_other_count": criteria[
            "baseline_worked_example_other_count"
        ],
        "label_examples_worked_example_other_count": criteria[
            "label_examples_worked_example_other_count"
        ],
        "query_before_worked_example_other_count": criteria[
            "query_before_worked_example_other_count"
        ],
        "interpretation": (
            "MC029 shows factor movement, not substrate repair. Removing numeric "
            "examples improves the learned atomic branch enough to cross the "
            "atomic-rate threshold, but answer-absent null reliability collapses. "
            "Moving the query before examples preserves nulls while amplifying "
            "worked-example copying. Putting the query row last reduces other-number "
            "leakage but shifts errors into local-row selection."
        ),
    }


def mc030_guard_audit(mc030_artifact: dict[str, Any], artifact_path: str) -> dict[str, Any]:
    summary = mc030_artifact["summary"]
    criteria = summary["criteria"]
    by_template = summary["by_template"]
    template_order = [
        "rules_only_baseline",
        "row_absence_guard_before_rules",
        "no_memory_for_absent_guard",
        "decision_order_guard_after_query",
        "decision_order_guard_query_last",
    ]

    variant_rows = []
    for template in template_order:
        template_summary = by_template[template]
        panels = template_summary["panels"]
        op_atomic = panels["operation_atomic_conflict"]
        answer_absent = panels["answer_absent_null"]
        op_rule_absent = panels["operation_rule_absent_null"]
        op_local = panels["operation_local_conflict"]
        variant_rows.append(
            {
                "template": template,
                "variant_gate_passed": criteria["variant_gate_by_template"][template],
                "operation_atomic_atomic_rate": op_atomic["atomic_number_rate"],
                "operation_atomic_other_number_rate": op_atomic["other_number_rate"],
                "operation_local_local_rate": op_local["local_number_rate"],
                "operation_rule_absent_unknown_rate": op_rule_absent["unknown_rate"],
                "answer_absent_unknown_rate": answer_absent["unknown_rate"],
                "answer_absent_other_number_rate": answer_absent["other_number_rate"],
                "parseable_floor": min(float(panel["parseable_rate"]) for panel in panels.values()),
            }
        )

    return {
        "artifact_path": artifact_path,
        "structural_passed": criteria["structural_passed"],
        "variant_gate_by_template": criteria["variant_gate_by_template"],
        "any_variant_behavior_gate_passed": criteria["any_variant_behavior_gate_passed"],
        "variant_rows": variant_rows,
        "baseline_operation_atomic_rate": criteria["baseline_operation_atomic_rate"],
        "baseline_other_number_rate": criteria["baseline_other_number_rate"],
        "baseline_answer_absent_unknown_rate": criteria[
            "baseline_answer_absent_unknown_rate"
        ],
        "max_operation_atomic_template": criteria["max_operation_atomic_template"],
        "max_operation_atomic_rate": criteria["max_operation_atomic_rate"],
        "max_answer_absent_unknown_template": criteria[
            "max_answer_absent_unknown_template"
        ],
        "max_answer_absent_unknown_rate": criteria["max_answer_absent_unknown_rate"],
        "min_other_number_template": criteria["min_other_number_template"],
        "min_other_number_rate": criteria["min_other_number_rate"],
        "min_other_number_template_operation_atomic_rate": criteria[
            "min_other_number_template_operation_atomic_rate"
        ],
        "interpretation": (
            "MC030 shows that simple absence guards do not repair the MC029 "
            "rules-only tradeoff. The baseline remains the best branch/null "
            "compromise. Row-absence and decision-order guards worsen "
            "answer-absent nulls, and the query-last guard reduces other-number "
            "leakage only by collapsing learned atomic routing."
        ),
    }


def mc031_checksum_audit(mc031_card: dict[str, Any]) -> dict[str, Any]:
    synthetic = smoke_panel(mc031_card, "synthetic_numeric_lookup")
    familiar = smoke_panel(mc031_card, "familiar_entity_numeric_lookup")
    atomic = smoke_panel(mc031_card, "real_world_atomic_number_control")
    answer_absent = smoke_panel(mc031_card, "answer_absent_null")
    valid = smoke_panel(mc031_card, "checksum_valid_conflict")
    invalid = smoke_panel(mc031_card, "checksum_invalid_conflict")
    absent = smoke_panel(mc031_card, "checksum_absent_conflict")
    return {
        "artifact_path": mc031_card["artifact_path"],
        "selected_template": mc031_card["selected_template"],
        "record_count": mc031_card["record_count"],
        "smoke_only": mc031_card["smoke_only"],
        "structural_passed": mc031_card["structural_passed"],
        "synthetic_lookup_local_rate": synthetic["local_number_rate"],
        "familiar_lookup_local_rate": familiar["local_number_rate"],
        "real_atomic_control_atomic_rate": atomic["atomic_number_rate"],
        "answer_absent_unknown_rate": answer_absent["unknown_rate"],
        "valid_checksum_local_rate": valid["local_number_rate"],
        "invalid_checksum_local_rate": invalid["local_number_rate"],
        "invalid_checksum_atomic_rate": invalid["atomic_number_rate"],
        "invalid_checksum_lure_rate": invalid["lure_atomic_number_rate"],
        "invalid_checksum_atomic_or_lure_rate": invalid["atomic_or_lure_number_rate"],
        "checksum_absent_local_rate": absent["local_number_rate"],
        "substantive_failed_criteria": mc031_card["substantive_failed_criteria"],
        "interpretation": (
            "MC031 isolates a statusless reliability-cue failure. Direct local "
            "lookup, familiar local lookup, direct atomic recall, valid-checksum "
            "local routing, and answer-absent nulls all survive smoke. The invalid "
            "checksum branch still selects local numbers on every selected row and "
            "never selects the atomic or lure value, so the bridge fails at source "
            "validity arbitration rather than recall or null behavior."
        ),
    }


def mc032_crosstable_audit(mc032_card: dict[str, Any]) -> dict[str, Any]:
    synthetic = smoke_panel(mc032_card, "synthetic_numeric_lookup")
    familiar = smoke_panel(mc032_card, "familiar_entity_numeric_lookup")
    atomic = smoke_panel(mc032_card, "real_world_atomic_number_control")
    answer_absent = smoke_panel(mc032_card, "answer_absent_null")
    match = smoke_panel(mc032_card, "crosscheck_match_conflict")
    mismatch = smoke_panel(mc032_card, "crosscheck_mismatch_conflict")
    absent = smoke_panel(mc032_card, "crosscheck_absent_conflict")
    return {
        "artifact_path": mc032_card["artifact_path"],
        "selected_template": mc032_card["selected_template"],
        "record_count": mc032_card["record_count"],
        "smoke_only": mc032_card["smoke_only"],
        "structural_passed": mc032_card["structural_passed"],
        "synthetic_lookup_local_rate": synthetic["local_number_rate"],
        "familiar_lookup_local_rate": familiar["local_number_rate"],
        "real_atomic_control_atomic_rate": atomic["atomic_number_rate"],
        "answer_absent_unknown_rate": answer_absent["unknown_rate"],
        "match_conflict_local_rate": match["local_number_rate"],
        "match_conflict_other_number_rate": match["other_number_rate"],
        "mismatch_conflict_local_rate": mismatch["local_number_rate"],
        "mismatch_conflict_atomic_rate": mismatch["atomic_number_rate"],
        "mismatch_conflict_lure_rate": mismatch["lure_atomic_number_rate"],
        "mismatch_conflict_atomic_or_lure_rate": mismatch[
            "atomic_or_lure_number_rate"
        ],
        "mismatch_conflict_unknown_rate": mismatch["unknown_rate"],
        "mismatch_conflict_other_number_rate": mismatch["other_number_rate"],
        "mismatch_conflict_side_number_rate": mismatch["side_number_rate"],
        "crosscheck_absent_local_rate": absent["local_number_rate"],
        "substantive_failed_criteria": mc032_card["substantive_failed_criteria"],
        "interpretation": (
            "MC032 removes the checksum-specific explanation for MC031. The cue is "
            "cross-table consistency rather than arithmetic validity, and direct "
            "local lookup, familiar local lookup, direct atomic recall, and "
            "answer-absent nulls all survive smoke. The mismatch branch still "
            "never selects the atomic or lure value; it mostly selects the primary "
            "local number, with no side-number copying. The failure is therefore "
            "broader local-table dominance under statusless source-validity pressure."
        ),
    }


def mc033_fact_claim_audit(mc033_card: dict[str, Any]) -> dict[str, Any]:
    synthetic = smoke_panel(mc033_card, "synthetic_numeric_lookup")
    familiar = smoke_panel(mc033_card, "familiar_entity_numeric_lookup")
    atomic = smoke_panel(mc033_card, "real_world_atomic_number_control")
    answer_absent = smoke_panel(mc033_card, "answer_absent_null")
    match = smoke_panel(mc033_card, "fact_claim_match_conflict")
    mismatch = smoke_panel(mc033_card, "fact_claim_mismatch_conflict")
    absent = smoke_panel(mc033_card, "fact_claim_absent_conflict")
    return {
        "artifact_path": mc033_card["artifact_path"],
        "selected_template": mc033_card["selected_template"],
        "record_count": mc033_card["record_count"],
        "smoke_only": mc033_card["smoke_only"],
        "structural_passed": mc033_card["structural_passed"],
        "synthetic_lookup_local_rate": synthetic["local_number_rate"],
        "familiar_lookup_local_rate": familiar["local_number_rate"],
        "real_atomic_control_atomic_rate": atomic["atomic_number_rate"],
        "answer_absent_unknown_rate": answer_absent["unknown_rate"],
        "match_conflict_local_rate": match["local_number_rate"],
        "match_conflict_atomic_rate": match["atomic_number_rate"],
        "match_conflict_other_number_rate": match["other_number_rate"],
        "mismatch_conflict_local_rate": mismatch["local_number_rate"],
        "mismatch_conflict_atomic_rate": mismatch["atomic_number_rate"],
        "mismatch_conflict_lure_rate": mismatch["lure_atomic_number_rate"],
        "mismatch_conflict_atomic_or_lure_rate": mismatch[
            "atomic_or_lure_number_rate"
        ],
        "fact_claim_absent_local_rate": absent["local_number_rate"],
        "substantive_failed_criteria": mc033_card["substantive_failed_criteria"],
        "interpretation": (
            "MC033 closes the post-MC032 repair path. It replaces checksum and "
            "cross-table cues with a row-local standard-number claim checked "
            "against learned atomic memory. Direct local lookup, familiar local "
            "lookup, direct atomic recall, and answer-absent nulls all survive "
            "smoke. The rule does not: match rows often return the atomic number "
            "instead of the local number, while mismatch rows mostly split between "
            "the local number and the wrong claimed number. The bridge failure is "
            "therefore not repaired by making the validity cue a learned-fact "
            "comparison."
        ),
    }


def build_card_failure_rows(smoke: dict[str, Any], route: dict[str, Any]) -> list[dict[str, Any]]:
    bridge_by_card = {
        entry["card_id"]: entry
        for entry in route.get("bridge_rung_entries", route.get("bridge_entries", []))
        if entry.get("card_id")
    }
    rows: list[dict[str, Any]] = []
    for card in smoke["cards"]:
        bridge_entry = bridge_by_card.get(card["card_id"], {})
        rows.append(
            {
                "card_id": card["card_id"],
                "title": card["title"],
                "claim_status": card["claim_status"],
                "diagnostic_class": card["diagnostic_class"],
                "observed_failure_pattern": card["observed_failure_pattern"],
                "all_controls_passed": card["all_controls_passed"],
                "behavior_ready": card["behavior_ready"],
                "signature_ready": card["signature_ready"],
                "hidden_state_allowed": card["hidden_state_allowed"],
                "failure_axes": card["failure_axes"],
                "exported_diagnostics": card["exported_diagnostics"],
                "substantive_failed_criteria": card["substantive_failed_criteria"],
                "lesson": card["lesson"],
                "next_constraint": card["next_constraint"],
                "bridge_disposition": bridge_entry.get("disposition"),
                "dominant_failure": bridge_entry.get("dominant_failure"),
                "artifact_path": card["artifact_path"],
                "status_card_path": card["status_card_path"],
            }
        )
    return rows


def build_control_surface_error_taxonomy() -> dict[str, Any]:
    smoke = load_json(SMOKE_DIAGNOSTICS_PATH)
    bridge = load_json(BRIDGE_LADDER_PATH)
    route = load_json(ROUTE_DISPOSITION_PATH)
    mc028_card = smoke_card_by_id(smoke, "MC028")
    mc028_artifact = load_json(ROOT / mc028_card["artifact_path"])
    mc029_card = smoke_card_by_id(smoke, "MC029")
    mc029_artifact = load_json(ROOT / mc029_card["artifact_path"])
    mc030_card = smoke_card_by_id(smoke, "MC030")
    mc030_artifact = load_json(ROOT / mc030_card["artifact_path"])
    mc031_card = smoke_card_by_id(smoke, "MC031")
    mc032_card = smoke_card_by_id(smoke, "MC032")
    mc033_card = smoke_card_by_id(smoke, "MC033")
    card_rows = build_card_failure_rows(smoke, route)

    failure_axis_counts = Counter()
    exported_diagnostic_counts = Counter()
    substantive_criteria_counts = Counter()
    for card in smoke["cards"]:
        failure_axis_counts.update(card["failure_axes"])
        exported_diagnostic_counts.update(card["exported_diagnostics"])
        substantive_criteria_counts.update(card["substantive_failed_criteria"])

    mc028_audit = mc028_other_number_audit(mc028_artifact)
    mc029_audit = mc029_factorial_audit(mc029_artifact, mc029_card["artifact_path"])
    mc030_audit = mc030_guard_audit(mc030_artifact, mc030_card["artifact_path"])
    mc031_audit = mc031_checksum_audit(mc031_card)
    mc032_audit = mc032_crosstable_audit(mc032_card)
    mc033_audit = mc033_fact_claim_audit(mc033_card)
    bridge_disposition_counts = route["summary"]["bridge_disposition_counts"]
    summary = {
        "smoke_card_count": smoke["summary"]["card_count"],
        "smoke_validation_check_count": len(smoke["validation_checks"]),
        "bridge_rung_count": bridge["summary"]["rung_count"],
        "bridge_closed_before_hidden_state_count": bridge_disposition_counts.get(
            "bridge_rung_closed_before_hidden_state",
            0,
        ),
        "bridge_prompt_visible_positive_control_count": bridge_disposition_counts.get(
            "prompt_visible_positive_control",
            0,
        ),
        "hidden_state_allowed_count": smoke["summary"]["hidden_state_allowed_count"],
        "behavior_ready_count": smoke["summary"]["behavior_ready_count"],
        "top_failure_axes": dict(failure_axis_counts.most_common(12)),
        "top_exported_diagnostics": dict(exported_diagnostic_counts.most_common(12)),
        "substantive_failed_criteria_counts": dict(
            sorted(substantive_criteria_counts.items())
        ),
        "mc028_other_number_count": mc028_audit["operation_atomic_other_number_count"],
        "mc028_other_number_rate": mc028_audit["operation_atomic_other_number_rate"],
        "mc028_other_number_exclusive_buckets": mc028_audit["exclusive_bucket_counts"],
        "mc029_rules_only_operation_atomic_rate": mc029_audit[
            "rules_only_operation_atomic_rate"
        ],
        "mc029_rules_only_answer_absent_unknown_rate": mc029_audit[
            "rules_only_answer_absent_unknown_rate"
        ],
        "mc029_query_before_worked_example_other_count": mc029_audit[
            "query_before_worked_example_other_count"
        ],
        "mc029_min_other_number_template": mc029_audit["min_other_number_template"],
        "mc029_min_other_number_rate": mc029_audit["min_other_number_rate"],
        "mc030_baseline_operation_atomic_rate": mc030_audit[
            "baseline_operation_atomic_rate"
        ],
        "mc030_baseline_answer_absent_unknown_rate": mc030_audit[
            "baseline_answer_absent_unknown_rate"
        ],
        "mc030_min_other_number_template": mc030_audit["min_other_number_template"],
        "mc030_min_other_number_rate": mc030_audit["min_other_number_rate"],
        "mc030_min_other_number_template_operation_atomic_rate": mc030_audit[
            "min_other_number_template_operation_atomic_rate"
        ],
        "mc031_valid_checksum_local_rate": mc031_audit[
            "valid_checksum_local_rate"
        ],
        "mc031_invalid_checksum_local_rate": mc031_audit[
            "invalid_checksum_local_rate"
        ],
        "mc031_invalid_checksum_atomic_or_lure_rate": mc031_audit[
            "invalid_checksum_atomic_or_lure_rate"
        ],
        "mc031_answer_absent_unknown_rate": mc031_audit[
            "answer_absent_unknown_rate"
        ],
        "mc032_match_conflict_local_rate": mc032_audit[
            "match_conflict_local_rate"
        ],
        "mc032_mismatch_conflict_local_rate": mc032_audit[
            "mismatch_conflict_local_rate"
        ],
        "mc032_mismatch_conflict_atomic_or_lure_rate": mc032_audit[
            "mismatch_conflict_atomic_or_lure_rate"
        ],
        "mc032_mismatch_conflict_side_number_rate": mc032_audit[
            "mismatch_conflict_side_number_rate"
        ],
        "mc033_match_conflict_local_rate": mc033_audit[
            "match_conflict_local_rate"
        ],
        "mc033_match_conflict_atomic_rate": mc033_audit[
            "match_conflict_atomic_rate"
        ],
        "mc033_mismatch_conflict_atomic_rate": mc033_audit[
            "mismatch_conflict_atomic_rate"
        ],
        "mc033_mismatch_conflict_local_rate": mc033_audit[
            "mismatch_conflict_local_rate"
        ],
        "mc033_mismatch_conflict_lure_rate": mc033_audit[
            "mismatch_conflict_lure_rate"
        ],
    }

    validation_checks = build_validation_checks(
        smoke,
        bridge,
        route,
        mc028_audit,
        mc029_audit,
        mc030_audit,
        mc031_audit,
        mc032_audit,
        mc033_audit,
    )
    return {
        "schema_version": 1,
        "updated_at": "2026-07-01",
        "source": "code/control_surface_error_taxonomy.py",
        "purpose": (
            "Make post-atlas typed failures first-class evidence. This artifact "
            "does not promote smoke runs into atlas rows; it records why they "
            "died before hidden-state work."
        ),
        "refs": {
            "smoke_diagnostics": rel(SMOKE_DIAGNOSTICS_PATH),
            "bridge_ladder": rel(BRIDGE_LADDER_PATH),
            "route_disposition": rel(ROUTE_DISPOSITION_PATH),
            "mc028_behavior_artifact": mc028_card["artifact_path"],
            "mc029_behavior_artifact": mc029_card["artifact_path"],
            "mc030_behavior_artifact": mc030_card["artifact_path"],
            "mc031_behavior_artifact": mc031_card["artifact_path"],
            "mc032_behavior_artifact": mc032_card["artifact_path"],
            "mc033_behavior_artifact": mc033_card["artifact_path"],
        },
        "summary": summary,
        "card_failures": card_rows,
        "failure_axis_counts": dict(sorted(failure_axis_counts.items())),
        "exported_diagnostic_counts": dict(sorted(exported_diagnostic_counts.items())),
        "substantive_failed_criteria_counts": dict(
            sorted(substantive_criteria_counts.items())
        ),
        "mc028_other_number_audit": mc028_audit,
        "mc029_factorial_audit": mc029_audit,
        "mc030_guard_audit": mc030_audit,
        "mc031_checksum_audit": mc031_audit,
        "mc032_crosstable_audit": mc032_audit,
        "mc033_fact_claim_audit": mc033_audit,
        "claim_boundary": {
            "allowed": (
                "The post-atlas bridge program has a typed failure map. MC028's "
                "full-source bare-integer failure is a learned-branch other-number "
                "leak, not a direct atomic recall failure or null failure. MC029 "
                "shows that factorized prompt changes move the branch/null/error "
                "axes without producing a valid bridge substrate. MC030 shows that "
                "simple absence guards do not repair that tradeoff. MC031 shows "
                "that a statusless checksum reliability cue can keep controls and "
                "nulls clean while invalid-source rows still collapse to local "
                "answers. MC032 shows that replacing checksum validity with "
                "cross-table consistency does not repair the learned branch; "
                "the mismatch branch still collapses toward local numbers without "
                "side-number leakage. MC033 closes the post-MC032 repair path: "
                "a learned-fact claim cue keeps controls and nulls clean but fails "
                "both match and mismatch routing."
            ),
            "forbidden": (
                "This taxonomy does not establish a hidden signature, intervention, "
                "or mechanism card, and it does not make MC028, MC029, MC030, MC031, MC032, or MC033 behavior-ready."
            ),
        },
        "next_experiment_pressure": [
            "Treat simple absence-guard repair as closed for the current operation-leak route.",
            "Treat wrong learned numbers as a distinct failure family from local-copy collapse and UNKNOWN abstention.",
            "A future operation bridge must be materially different, not another local prompt guard around the same rules-only contract.",
            "A future statusless reliability bridge must prove invalid-source routing, not just preserve direct controls and answer-absent nulls.",
            "A future post-checksum bridge must beat MC032 by producing learned atomic/lure selections in mismatch rows without side-number leakage.",
            "Treat the post-MC032 fact-claim repair route as closed unless a materially different non-bridge substrate is specified.",
            "Keep answer-schema changes out of bridge promotion claims unless the schema is the experimental variable.",
        ],
        "validation_checks": validation_checks,
    }


def build_validation_checks(
    smoke: dict[str, Any],
    bridge: dict[str, Any],
    route: dict[str, Any],
    mc028_audit: dict[str, Any],
    mc029_audit: dict[str, Any],
    mc030_audit: dict[str, Any],
    mc031_audit: dict[str, Any],
    mc032_audit: dict[str, Any],
    mc033_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    bucket_total = sum(mc028_audit["exclusive_bucket_counts"].values())
    bridge_counts = route["summary"]["bridge_disposition_counts"]
    checks = [
        {
            "id": "all_smoke_cards_remain_pre_hidden_state",
            "actual": smoke["summary"]["hidden_state_allowed_count"],
            "predicate": "== 0",
            "passed": smoke["summary"]["hidden_state_allowed_count"] == 0,
            "why": "The error taxonomy must not silently promote smoke diagnostics.",
        },
        {
            "id": "smoke_taxonomy_covers_mc017_to_mc033",
            "actual": smoke["summary"]["card_count"],
            "predicate": "== 17",
            "passed": smoke["summary"]["card_count"] == 17,
            "why": "The taxonomy currently covers the MC017-MC033 bridge-smoke run set.",
        },
        {
            "id": "bridge_ladder_count_matches_taxonomy_scope",
            "actual": bridge["summary"]["rung_count"],
            "predicate": "== 24",
            "passed": bridge["summary"]["rung_count"] == 24,
            "why": "The current bridge ladder spans MC010-MC033.",
        },
        {
            "id": "bridge_dispositions_preserve_closed_before_hidden_state_majority",
            "actual": bridge_counts,
            "predicate": "23 closed before hidden-state and 1 prompt-visible positive control",
            "passed": bridge_counts.get("bridge_rung_closed_before_hidden_state") == 23
            and bridge_counts.get("prompt_visible_positive_control") == 1,
            "why": "MC012 is the only prompt-visible positive control; all other bridge rungs are closed before hidden-state work.",
        },
        {
            "id": "mc028_other_number_bucket_total_matches_other_count",
            "actual": {
                "bucket_total": bucket_total,
                "other_number_count": mc028_audit["operation_atomic_other_number_count"],
            },
            "predicate": "bucket_total == other_number_count == 28",
            "passed": bucket_total == mc028_audit["operation_atomic_other_number_count"] == 28,
            "why": "Every MC028 other-number row must be assigned one exclusive failure bucket.",
        },
        {
            "id": "mc028_atomic_branch_fails_while_controls_remain_clean",
            "actual": {
                "operation_atomic_atomic_rate": mc028_audit["operation_atomic_atomic_rate"],
                "direct_atomic_control_atomic_rate": mc028_audit["direct_atomic_control_atomic_rate"],
                "answer_absent_unknown_rate": mc028_audit["answer_absent_unknown_rate"],
                "rule_absent_unknown_rate": mc028_audit["rule_absent_unknown_rate"],
            },
            "predicate": "operation atomic < 0.85; direct atomic/null controls == 1.0",
            "passed": mc028_audit["operation_atomic_atomic_rate"] < 0.85
            and mc028_audit["direct_atomic_control_atomic_rate"] == 1.0
            and mc028_audit["answer_absent_unknown_rate"] == 1.0
            and mc028_audit["rule_absent_unknown_rate"] == 1.0,
            "why": "MC028 is a learned-branch selection failure, not a direct-control or null failure.",
        },
        {
            "id": "mc029_factorial_moves_axes_without_repairing_substrate",
            "actual": {
                "any_variant_behavior_gate_passed": mc029_audit[
                    "any_variant_behavior_gate_passed"
                ],
                "rules_only_operation_atomic_rate": mc029_audit[
                    "rules_only_operation_atomic_rate"
                ],
                "rules_only_answer_absent_unknown_rate": mc029_audit[
                    "rules_only_answer_absent_unknown_rate"
                ],
                "query_before_worked_example_other_count": mc029_audit[
                    "query_before_worked_example_other_count"
                ],
                "min_other_number_template": mc029_audit["min_other_number_template"],
                "min_other_number_rate": mc029_audit["min_other_number_rate"],
            },
            "predicate": (
                "no variant passes; rules-only atomic >= 0.85; "
                "rules-only answer-absent UNKNOWN < 0.90; query-before worked examples >= 50; "
                "query-row-last other-number < 0.10"
            ),
            "passed": (
                not mc029_audit["any_variant_behavior_gate_passed"]
                and mc029_audit["rules_only_operation_atomic_rate"] >= 0.85
                and mc029_audit["rules_only_answer_absent_unknown_rate"] < 0.90
                and mc029_audit["query_before_worked_example_other_count"] >= 50
                and mc029_audit["min_other_number_template"] == "query_row_last"
                and mc029_audit["min_other_number_rate"] < 0.10
            ),
            "why": "MC029 is a branch/null/example-leak tradeoff, not a repaired bridge substrate.",
        },
        {
            "id": "mc030_absence_guards_do_not_repair_rules_only_tradeoff",
            "actual": {
                "any_variant_behavior_gate_passed": mc030_audit[
                    "any_variant_behavior_gate_passed"
                ],
                "baseline_operation_atomic_rate": mc030_audit[
                    "baseline_operation_atomic_rate"
                ],
                "baseline_answer_absent_unknown_rate": mc030_audit[
                    "baseline_answer_absent_unknown_rate"
                ],
                "max_answer_absent_unknown_template": mc030_audit[
                    "max_answer_absent_unknown_template"
                ],
                "max_answer_absent_unknown_rate": mc030_audit[
                    "max_answer_absent_unknown_rate"
                ],
                "min_other_number_template": mc030_audit["min_other_number_template"],
                "min_other_number_rate": mc030_audit["min_other_number_rate"],
                "min_other_number_template_operation_atomic_rate": mc030_audit[
                    "min_other_number_template_operation_atomic_rate"
                ],
            },
            "predicate": (
                "no variant passes; baseline branch >= 0.85 but null < 0.90; "
                "best null remains below 0.90; query-last other-number < 0.10 "
                "with atomic branch < 0.50"
            ),
            "passed": (
                not mc030_audit["any_variant_behavior_gate_passed"]
                and mc030_audit["baseline_operation_atomic_rate"] >= 0.85
                and mc030_audit["baseline_answer_absent_unknown_rate"] < 0.90
                and mc030_audit["max_answer_absent_unknown_rate"] < 0.90
                and mc030_audit["min_other_number_template"]
                == "decision_order_guard_query_last"
                and mc030_audit["min_other_number_rate"] < 0.10
                and mc030_audit["min_other_number_template_operation_atomic_rate"] < 0.50
            ),
            "why": "MC030 closes simple absence-guard repair for the MC029 rules-only branch/null tradeoff.",
        },
        {
            "id": "mc031_statusless_checksum_collapses_invalid_branch",
            "actual": {
                "synthetic_lookup_local_rate": mc031_audit[
                    "synthetic_lookup_local_rate"
                ],
                "real_atomic_control_atomic_rate": mc031_audit[
                    "real_atomic_control_atomic_rate"
                ],
                "answer_absent_unknown_rate": mc031_audit[
                    "answer_absent_unknown_rate"
                ],
                "valid_checksum_local_rate": mc031_audit[
                    "valid_checksum_local_rate"
                ],
                "invalid_checksum_local_rate": mc031_audit[
                    "invalid_checksum_local_rate"
                ],
                "invalid_checksum_atomic_or_lure_rate": mc031_audit[
                    "invalid_checksum_atomic_or_lure_rate"
                ],
            },
            "predicate": (
                "controls/nulls == 1.0; valid checksum local >= 0.85; "
                "invalid checksum local == 1.0 and atomic/lure == 0.0"
            ),
            "passed": (
                mc031_audit["synthetic_lookup_local_rate"] == 1.0
                and mc031_audit["real_atomic_control_atomic_rate"] == 1.0
                and mc031_audit["answer_absent_unknown_rate"] == 1.0
                and mc031_audit["valid_checksum_local_rate"] >= 0.85
                and mc031_audit["invalid_checksum_local_rate"] == 1.0
                and mc031_audit["invalid_checksum_atomic_or_lure_rate"] == 0.0
            ),
            "why": "MC031 is a statusless reliability-cue failure, not a direct-control or null failure.",
        },
        {
            "id": "mc032_crosstable_mismatch_collapses_toward_local",
            "actual": {
                "synthetic_lookup_local_rate": mc032_audit[
                    "synthetic_lookup_local_rate"
                ],
                "real_atomic_control_atomic_rate": mc032_audit[
                    "real_atomic_control_atomic_rate"
                ],
                "answer_absent_unknown_rate": mc032_audit[
                    "answer_absent_unknown_rate"
                ],
                "mismatch_conflict_local_rate": mc032_audit[
                    "mismatch_conflict_local_rate"
                ],
                "mismatch_conflict_atomic_or_lure_rate": mc032_audit[
                    "mismatch_conflict_atomic_or_lure_rate"
                ],
                "mismatch_conflict_side_number_rate": mc032_audit[
                    "mismatch_conflict_side_number_rate"
                ],
            },
            "predicate": (
                "controls/nulls == 1.0; mismatch local >= 0.70; "
                "mismatch atomic/lure == 0.0 and side-number == 0.0"
            ),
            "passed": (
                mc032_audit["synthetic_lookup_local_rate"] == 1.0
                and mc032_audit["real_atomic_control_atomic_rate"] == 1.0
                and mc032_audit["answer_absent_unknown_rate"] == 1.0
                and mc032_audit["mismatch_conflict_local_rate"] >= 0.70
                and mc032_audit["mismatch_conflict_atomic_or_lure_rate"] == 0.0
                and mc032_audit["mismatch_conflict_side_number_rate"] == 0.0
            ),
            "why": "MC032 is a statusless cross-table failure, not a direct-control, null, or side-number-copying failure.",
        },
        {
            "id": "mc033_fact_claim_route_fails_both_branches",
            "actual": {
                "synthetic_lookup_local_rate": mc033_audit[
                    "synthetic_lookup_local_rate"
                ],
                "real_atomic_control_atomic_rate": mc033_audit[
                    "real_atomic_control_atomic_rate"
                ],
                "answer_absent_unknown_rate": mc033_audit[
                    "answer_absent_unknown_rate"
                ],
                "match_conflict_local_rate": mc033_audit[
                    "match_conflict_local_rate"
                ],
                "match_conflict_atomic_rate": mc033_audit[
                    "match_conflict_atomic_rate"
                ],
                "mismatch_conflict_atomic_rate": mc033_audit[
                    "mismatch_conflict_atomic_rate"
                ],
                "mismatch_conflict_lure_rate": mc033_audit[
                    "mismatch_conflict_lure_rate"
                ],
            },
            "predicate": (
                "controls/nulls == 1.0; match local < 0.85; "
                "mismatch atomic < 0.85 and lure >= 0.40"
            ),
            "passed": (
                mc033_audit["synthetic_lookup_local_rate"] == 1.0
                and mc033_audit["real_atomic_control_atomic_rate"] == 1.0
                and mc033_audit["answer_absent_unknown_rate"] == 1.0
                and mc033_audit["match_conflict_local_rate"] < 0.85
                and mc033_audit["mismatch_conflict_atomic_rate"] < 0.85
                and mc033_audit["mismatch_conflict_lure_rate"] >= 0.40
            ),
            "why": "MC033 closes the post-MC032 repair route: learned-fact claim checking does not stabilize either branch.",
        },
    ]
    return checks


def validate_error_taxonomy(payload: dict[str, Any]) -> None:
    failed = [check for check in payload["validation_checks"] if not check["passed"]]
    if failed:
        raise AssertionError(f"error taxonomy validation failed: {failed}")


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    mc028 = payload["mc028_other_number_audit"]
    mc029 = payload["mc029_factorial_audit"]
    mc030 = payload["mc030_guard_audit"]
    mc031 = payload["mc031_checksum_audit"]
    mc032 = payload["mc032_crosstable_audit"]
    mc033 = payload["mc033_fact_claim_audit"]
    lines = [
        "# Control-Surface Error Taxonomy",
        "",
        "Date: 2026-07-01",
        "",
        "Status: generated typed-failure taxonomy implemented and validated.",
        "",
        "This document interprets:",
        "",
        f"> `{payload['refs']['smoke_diagnostics']}`",
        "",
        f"> `{payload['refs']['bridge_ladder']}`",
        "",
        f"> `{payload['refs']['route_disposition']}`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_error_taxonomy.py`",
        "",
        "Command:",
        "",
        "```powershell",
        "python code\\control_surface_error_taxonomy.py --write",
        "python code\\control_surface_error_taxonomy.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Purpose",
        "",
        "The point is to stop treating killed bridge attempts as prose-only",
        "tombstones. Each failure is a measured datum about where behavior lives:",
        "answer interface, prompt-local source salience, route-rule following,",
        "learned-memory selection, null behavior, or output geometry.",
        "",
        "## Current Facts",
        "",
        f"- smoke cards covered: `{summary['smoke_card_count']}`;",
        f"- smoke validation checks: `{summary['smoke_validation_check_count']}`;",
        f"- bridge rungs covered: `{summary['bridge_rung_count']}`;",
        f"- bridge rungs closed before hidden-state work: `{summary['bridge_closed_before_hidden_state_count']}`;",
        f"- prompt-visible positive-control bridge rungs: `{summary['bridge_prompt_visible_positive_control_count']}`;",
        f"- hidden-state-allowed smoke cards: `{summary['hidden_state_allowed_count']}`;",
        f"- behavior-ready smoke cards: `{summary['behavior_ready_count']}`.",
        "",
        "## Top Failure Axes",
        "",
        "| Axis | Count |",
        "| --- | ---: |",
    ]
    for axis, count in summary["top_failure_axes"].items():
        lines.append(f"| `{axis}` | {count} |")

    lines.extend(
        [
            "",
            "## MC028 Other-Number Leak",
            "",
            "MC028 is the current sharpest example of why a failed behavior substrate",
            "can still be informative. Bare integers preserve direct controls and",
            "nulls at full-source scale, but the learned atomic operation branch",
            "does not merely copy the prompt-local row. It leaks wrong numbers from",
            "several different surfaces.",
            "",
            "| Measure | Value |",
            "| --- | ---: |",
            f"| operation-atomic rows | {mc028['operation_atomic_rows']} |",
            f"| operation-atomic atomic rate | {mc028['operation_atomic_atomic_rate']:.3f} |",
            f"| operation-atomic other-number count | {mc028['operation_atomic_other_number_count']} |",
            f"| operation-atomic other-number rate | {mc028['operation_atomic_other_number_rate']:.3f} |",
            f"| operation-local local rate | {mc028['operation_local_local_rate']:.3f} |",
            f"| direct atomic-control atomic rate | {mc028['direct_atomic_control_atomic_rate']:.3f} |",
            f"| answer-absent UNKNOWN rate | {mc028['answer_absent_unknown_rate']:.3f} |",
            f"| rule-absent UNKNOWN rate | {mc028['rule_absent_unknown_rate']:.3f} |",
            "",
            "Exclusive wrong-number buckets:",
            "",
            "| Bucket | Rows |",
            "| --- | ---: |",
        ]
    )
    for bucket, count in mc028["exclusive_bucket_counts"].items():
        lines.append(f"| `{bucket}` | {count} |")

    lines.extend(
        [
            "",
            "Overlap labels:",
            "",
            "| Label | Rows |",
            "| --- | ---: |",
        ]
    )
    for label, count in mc028["overlap_counts"].items():
        lines.append(f"| `{label}` | {count} |")

    lines.extend(
        [
            "",
            "## MC029 Factorized Leak Tradeoff",
            "",
            "MC029 tested whether the MC028 other-number leak could be repaired by",
            "separating numeric examples, label-only examples, rule-only prompts,",
            "query-before-example ordering, and query-row-last salience. The result",
            "is a sharper failure: individual factors move different error axes,",
            "but no variant becomes a valid bridge substrate.",
            "",
            "| Variant | Op-atomic atomic | Op-atomic other | Op-local local | Answer-absent UNKNOWN | Worked-example other rows | Gate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in mc029["variant_rows"]:
        gate = "pass" if row["variant_gate_passed"] else "fail"
        lines.append(
            "| "
            f"`{row['template']}` | "
            f"{row['operation_atomic_atomic_rate']:.3f} | "
            f"{row['operation_atomic_other_number_rate']:.3f} | "
            f"{row['operation_local_local_rate']:.3f} | "
            f"{row['answer_absent_unknown_rate']:.3f} | "
            f"{row['worked_example_other_count']} | "
            f"{gate} |"
        )

    lines.extend(
        [
            "",
            mc029["interpretation"],
            "",
            "## MC030 Absence-Guard Repair Failure",
            "",
            "MC030 tested the narrowest repair implied by MC029: keep the rules-only",
            "branch gain, but add explicit absence guards to restore answer-absent",
            "null reliability. The repair failed. Guards either made answer-absent",
            "rows worse or reduced other-number leakage by collapsing the learned",
            "atomic branch.",
            "",
            "| Variant | Op-atomic atomic | Op-atomic other | Op-local local | Rule-null UNKNOWN | Answer-absent UNKNOWN | Gate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in mc030["variant_rows"]:
        gate = "pass" if row["variant_gate_passed"] else "fail"
        lines.append(
            "| "
            f"`{row['template']}` | "
            f"{row['operation_atomic_atomic_rate']:.3f} | "
            f"{row['operation_atomic_other_number_rate']:.3f} | "
            f"{row['operation_local_local_rate']:.3f} | "
            f"{row['operation_rule_absent_unknown_rate']:.3f} | "
            f"{row['answer_absent_unknown_rate']:.3f} | "
            f"{gate} |"
        )

    lines.extend(
        [
            "",
            mc030["interpretation"],
            "",
            "## MC031 Statusless Checksum Collapse",
            "",
            "MC031 tested whether a statusless arithmetic-checksum cue could replace",
            "visible reliability labels and route invalid-source rows to learned",
            "atomic facts. The controls and nulls stayed clean in smoke, but the",
            "invalid-checksum branch still selected local numbers.",
            "",
            "| Measure | Value |",
            "| --- | ---: |",
            f"| selected template | `{mc031['selected_template']}` |",
            f"| synthetic lookup local rate | {mc031['synthetic_lookup_local_rate']:.3f} |",
            f"| familiar lookup local rate | {mc031['familiar_lookup_local_rate']:.3f} |",
            f"| real atomic-control atomic rate | {mc031['real_atomic_control_atomic_rate']:.3f} |",
            f"| answer-absent UNKNOWN rate | {mc031['answer_absent_unknown_rate']:.3f} |",
            f"| valid-checksum local rate | {mc031['valid_checksum_local_rate']:.3f} |",
            f"| invalid-checksum local rate | {mc031['invalid_checksum_local_rate']:.3f} |",
            f"| invalid-checksum atomic/lure rate | {mc031['invalid_checksum_atomic_or_lure_rate']:.3f} |",
            f"| checksum-absent local rate | {mc031['checksum_absent_local_rate']:.3f} |",
            "",
            mc031["interpretation"],
            "",
            "## MC032 Cross-Table Consistency Collapse",
            "",
            "MC032 tested whether MC031's result was checksum-specific. It removed",
            "arithmetic validity and used agreement between two neutral local tables",
            "instead. The controls and nulls stayed clean, side-number leakage stayed",
            "absent, and the mismatch branch still avoided learned atomic answers.",
            "",
            "| Measure | Value |",
            "| --- | ---: |",
            f"| selected template | `{mc032['selected_template']}` |",
            f"| synthetic lookup local rate | {mc032['synthetic_lookup_local_rate']:.3f} |",
            f"| familiar lookup local rate | {mc032['familiar_lookup_local_rate']:.3f} |",
            f"| real atomic-control atomic rate | {mc032['real_atomic_control_atomic_rate']:.3f} |",
            f"| answer-absent UNKNOWN rate | {mc032['answer_absent_unknown_rate']:.3f} |",
            f"| match-conflict local rate | {mc032['match_conflict_local_rate']:.3f} |",
            f"| mismatch-conflict local rate | {mc032['mismatch_conflict_local_rate']:.3f} |",
            f"| mismatch-conflict atomic/lure rate | {mc032['mismatch_conflict_atomic_or_lure_rate']:.3f} |",
            f"| mismatch-conflict side-number rate | {mc032['mismatch_conflict_side_number_rate']:.3f} |",
            f"| crosscheck-absent local rate | {mc032['crosscheck_absent_local_rate']:.3f} |",
            "",
            mc032["interpretation"],
            "",
            "## MC033 Fact-Claim Closeout",
            "",
            "MC033 tested the one repair pass allowed after MC032: replace checksum",
            "and cross-table cues with a row-local standard-number claim checked",
            "against learned atomic memory. Controls and nulls stayed clean, but",
            "the branch rule did not become stable.",
            "",
            "| Measure | Value |",
            "| --- | ---: |",
            f"| selected template | `{mc033['selected_template']}` |",
            f"| synthetic lookup local rate | {mc033['synthetic_lookup_local_rate']:.3f} |",
            f"| familiar lookup local rate | {mc033['familiar_lookup_local_rate']:.3f} |",
            f"| real atomic-control atomic rate | {mc033['real_atomic_control_atomic_rate']:.3f} |",
            f"| answer-absent UNKNOWN rate | {mc033['answer_absent_unknown_rate']:.3f} |",
            f"| match-conflict local rate | {mc033['match_conflict_local_rate']:.3f} |",
            f"| match-conflict atomic rate | {mc033['match_conflict_atomic_rate']:.3f} |",
            f"| mismatch-conflict atomic rate | {mc033['mismatch_conflict_atomic_rate']:.3f} |",
            f"| mismatch-conflict local rate | {mc033['mismatch_conflict_local_rate']:.3f} |",
            f"| mismatch-conflict claimed-number/lure rate | {mc033['mismatch_conflict_lure_rate']:.3f} |",
            f"| fact-claim-absent local rate | {mc033['fact_claim_absent_local_rate']:.3f} |",
            "",
            mc033["interpretation"],
            "",
            "",
            "## Interpretation",
            "",
            mc028["interpretation"],
            "",
            "The immediate consequence is that the next bridge experiment should",
            "not just try another answer schema. MC029 already manipulated worked",
            "examples, query anchoring, and prompt-row salience separately; that",
            "factorization moved the failure axes without repairing the substrate.",
            "MC030 then tested the local absence-guard repair and found that this",
            "also fails: the unguarded rules-only baseline remains the best",
            "branch/null compromise, while the guards worsen null rows or collapse",
            "the learned branch.",
            "MC031 adds a separate statusless reliability-cue failure: the model can",
            "obey direct controls and nulls while still refusing to use the checksum",
            "cue to leave the local branch.",
            "MC032 removes the checksum-specific objection and preserves the same",
            "boundary: cross-table mismatch still fails to reach learned atomic facts,",
            "and the error is not second-table side-number copying.",
            "MC033 then removes the cross-table cue and uses a row-local learned-fact",
            "claim; this still does not repair the bridge, because the match branch",
            "does not reliably stay local and the mismatch branch leaks local and",
            "claimed wrong numbers.",
            "The current new failure families are wrong learned-number selection under",
            "route pressure, branch/null tradeoff, statusless invalid-source local",
            "collapse, statusless cross-table local collapse, and fact-claim branch",
            "instability, distinct from generic parse failure and UNKNOWN abstention.",
            "",
            "## Claim Boundary",
            "",
            f"Allowed: {payload['claim_boundary']['allowed']}",
            "",
            f"Forbidden: {payload['claim_boundary']['forbidden']}",
            "",
        ]
    )
    return "\n".join(lines)


def write_error_taxonomy() -> dict[str, Any]:
    payload = build_control_surface_error_taxonomy()
    validate_error_taxonomy(payload)
    write_json(ERROR_TAXONOMY_PATH, payload)
    ERROR_TAXONOMY_REPORT_PATH.write_text(render_report(payload), encoding="utf-8", newline="\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write data and report artifacts")
    parser.add_argument("--json", action="store_true", help="print taxonomy JSON")
    args = parser.parse_args()

    payload = build_control_surface_error_taxonomy()
    validate_error_taxonomy(payload)
    if args.write:
        write_error_taxonomy()
        print(
            f"wrote {rel(ERROR_TAXONOMY_PATH)} and {rel(ERROR_TAXONOMY_REPORT_PATH)} "
            f"with {payload['summary']['smoke_card_count']} smoke cards"
        )
        return
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print("error taxonomy ok")
    print("summary:", json.dumps(payload["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
