"""Build the current reliability matrix for control-surface claims.

This layer turns the mechanism-card gates into an auditable row matrix. A
control surface is not reliable because it has a good probe or a good primary
effect; it is reliable only when behavior, signature, intervention, nulls,
locality, robustness, and transfer are all accounted for.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from control_surface_artifacts import ATLAS_PATH, ROOT, load_json
from control_surface_comparison import COMPARISON_PATH
from control_surface_route_disposition import ROUTE_DISPOSITION_PATH
from control_surface_transfer_matrix import TRANSFER_MATRIX_PATH


RELIABILITY_MATRIX_PATH = ROOT / "data" / "control_surface_reliability_matrix.json"
RELIABILITY_MATRIX_REPORT_PATH = ROOT / "research" / "32_CONTROL_SURFACE_RELIABILITY_MATRIX.md"

OUTPUT_DIAGNOSTICS = {
    "OUTPUT_MARGIN_CONFUND",
    "CANDIDATE_SCORE_CONFUND",
    "GLOBAL_OUTPUT_CONFOUNDED_LEADTIME",
    "GLOBAL_MARGIN_SEPARATION_BLOCKS_MATCHING",
    "MARGIN_OVERLAP_TABLE_FAILED",
    "STRICT_FINAL_MARGIN_OVERLAP_ABSENT",
    "APPROXIMATE_PAIR_MATCHING_FAILED_MARGIN_BASELINES",
    "SOURCE_PATH_FINAL_MARGIN_SHADOW",
    "GREEDY_FINAL_MARGIN_SIGN_BARRIER",
    "DELAYED_INTERFACE_CANDIDATE_SCORE_VISIBLE",
}

PROMPT_DIAGNOSTICS = {
    "PROMPT_FORMAT_CONFUND",
    "REQUESTED_MODE_CONFUND",
    "PROMPT_REWRITE_EQUIVALENCE",
    "PROMPT_AUTHORITY_DIAL",
    "PROMPT_CONTRACT_PARSEABILITY",
    "RELIABILITY_PROMPT_CHANNEL_VISIBLE",
    "DERIVED_CODE_TYPED_SLOT_PROMPT_VISIBLE",
}

SHUFFLE_OR_SIGNATURE_DIAGNOSTICS = {
    "SHUFFLED_SELECTION_OVERFIT",
    "SIGNATURE_NOT_CAUSAL",
    "CANDIDATE_DECOUPLED_SHUFFLE_OVERFIT",
}

NULL_DIAGNOSTICS = {
    "NULL_ROW_LOW_MARGIN_FLIP",
    "MODEL_SIZE_NULL_FRAGILITY",
    "SYMBOLIC_NULL_CONTROL_FAILED",
}

BEHAVIOR_BLOCK_DIAGNOSTICS = {
    "BEHAVIOR_SUBSTRATE_FAILED",
    "CONTRAST_ABSENT",
    "SYMBOLIC_CONFLICT_PARSEABILITY_FAILED",
    "SYMBOLIC_CONFLICT_CONTRAST_WEAK",
    "DERIVED_CODE_CONTROL_CONFLICT_TRADEOFF",
    "TWO_HOP_SYNTHETIC_LOOKUP_FAILED",
    "TWO_HOP_REAL_MEMORY_CONTROL_FAILED",
    "TWO_HOP_CONFLICT_CONTRAST_ABSENT",
    "NUMERIC_CONFLICT_CONTRAST_ABSENT",
    "STATUS_CHANNEL_ABLATION_COLLAPSED_CONTRAST",
    "CALIBRATION_INFERENCE_CONFLICT_COLLAPSED",
    "PARITY_GATE_NOT_FOLLOWED",
    "ALPHABET_GATE_LOCAL_COLLAPSE",
    "FEATURE_LABEL_GATE_DID_NOT_RESCUE",
}

ROUTE_TO_RELIABILITY_CLASS = {
    "bounded_mechanism_frozen": "bounded_reliability_reference",
    "closed_before_hidden_state": "not_reliable_behavior_or_bridge_blocked",
    "failed_intervention_or_mechanism_route": "not_reliable_failed_intervention_route",
    "monitor_only_closed": "not_reliable_monitor_only_no_lever",
    "monitor_only_conditional_revisit": "not_reliable_monitor_only_no_lever",
    "output_shadow_diagnostic_baseline": "not_reliable_output_shadow_diagnostic",
    "prompt_visible_positive_control": "not_reliable_prompt_visible_positive_control",
    "diagnostic_baseline_only": "not_reliable_diagnostic_baseline_only",
}


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


def by_id(items: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {item[key]: item for item in items}


def behavior_gate_status(row: dict[str, Any], route: dict[str, Any]) -> str:
    diagnostics = set(row["diagnostics"])
    gate = row["behavior_gate"].lower()
    if route["disposition"] == "prompt_visible_positive_control":
        return "passed_but_prompt_visible"
    if route["disposition"] == "closed_before_hidden_state":
        return "failed_or_closed_before_hidden_state"
    if diagnostics & BEHAVIOR_BLOCK_DIAGNOSTICS:
        return "failed_or_unstable_behavior_substrate"
    if gate.startswith("failed") or "behavior_gate_failed" in gate:
        return "failed"
    if gate.startswith("passed") or "passed" in gate or "partially_repaired" in gate:
        return "passed_or_partial"
    return "diagnostic_only"


def signature_gate_status(row: dict[str, Any], route: dict[str, Any]) -> str:
    diagnostics = set(row["diagnostics"])
    lead_state = row["lead_time"]["state"]
    if row["verdict"]["class"] == "bounded_mechanism_card":
        return "bounded_internal_signature"
    if route["disposition"] in {"closed_before_hidden_state", "prompt_visible_positive_control"}:
        return "not_allowed_or_not_reached"
    if lead_state in {"zero_lead_time", "lead_time_output_shadow"}:
        return "output_visible_or_zero_lead"
    if lead_state == "lead_time_monitor_only":
        if diagnostics & OUTPUT_DIAGNOSTICS:
            return "monitor_only_output_or_candidate_confounded"
        return "monitor_only_no_lever"
    if diagnostics & SHUFFLE_OR_SIGNATURE_DIAGNOSTICS:
        return "not_causal_or_shuffle_fragile"
    if diagnostics & OUTPUT_DIAGNOSTICS:
        return "output_or_candidate_confounded"
    return "not_cleanly_established"


def intervention_gate_status(row: dict[str, Any]) -> str:
    state = row["intervention"]["state"]
    if row["verdict"]["class"] == "promoted_mechanism_card":
        return "passed"
    if row["verdict"]["class"] == "bounded_mechanism_card":
        return "bounded_causal_dirty"
    if state == "failed":
        return "failed"
    if state == "behavior_control_only":
        return "behavior_control_only_not_mechanism"
    if state in {"not_allowed", "not_tested"}:
        return state
    return state


def null_locality_status(row: dict[str, Any]) -> str:
    diagnostics = set(row["diagnostics"])
    null_value = row["mixture_profile"]["null_locality"]
    if row["id"] == "mc005_associative_lookup":
        return "bounded_low_margin_and_model_size_fragile"
    if null_value == "failed" or diagnostics & NULL_DIAGNOSTICS:
        return "failed_or_fragile"
    if null_value == "bounded":
        return "bounded"
    if null_value == "behavior_only":
        return "behavior_only"
    return null_value


def local_path_status(row: dict[str, Any]) -> str:
    local_value = row["mixture_profile"]["local_internal_path"]
    if row["id"] == "mc005_associative_lookup":
        return "localized_but_bounded"
    if local_value == "high":
        return "localized"
    if local_value in {"low", "none"}:
        return f"{local_value}_locality"
    return local_value


def robustness_status(
    row: dict[str, Any],
    audit_entry: dict[str, Any] | None,
) -> str:
    diagnostics = set(row["diagnostics"])
    if row["id"] == "mc005_associative_lookup":
        return "bounded_by_null_side_rows_and_model_size"
    if audit_entry and (
        audit_entry.get("failed_criteria_count", 0) > 0
        or audit_entry.get("false_criteria_count", 0) > 0
        or audit_entry.get("null_criteria_count", 0) > 0
    ):
        return "failed_or_incomplete_control_criteria"
    if diagnostics & (OUTPUT_DIAGNOSTICS | PROMPT_DIAGNOSTICS | SHUFFLE_OR_SIGNATURE_DIAGNOSTICS):
        return "confounded_by_visible_or_shuffle_controls"
    return "not_comprehensively_documented"


def transfer_status(transfer_entry: dict[str, Any]) -> str:
    return transfer_entry["transfer_class"]


def reliability_class(
    row: dict[str, Any],
    route: dict[str, Any],
    transfer_entry: dict[str, Any],
) -> str:
    if (
        row["verdict"]["class"] == "promoted_mechanism_card"
        and transfer_entry["transfer_class"] == "transfer_ready_mechanism"
    ):
        return "full_reliability_mechanism"
    return ROUTE_TO_RELIABILITY_CLASS.get(
        route["disposition"],
        "not_reliable_diagnostic_baseline_only",
    )


def failed_or_missing_gates(
    row: dict[str, Any],
    gate_statuses: dict[str, str],
    transfer_entry: dict[str, Any],
) -> list[str]:
    missing: list[str] = []

    if gate_statuses["behavior"] not in {
        "passed_or_partial",
        "passed_but_prompt_visible",
    }:
        missing.append("behavior_substrate")
    if gate_statuses["behavior"] == "passed_but_prompt_visible":
        missing.append("prompt_channel_locality")

    if gate_statuses["signature"] != "bounded_internal_signature":
        missing.append("control-surviving_signature")
    if gate_statuses["signature"] in {
        "output_visible_or_zero_lead",
        "monitor_only_output_or_candidate_confounded",
        "output_or_candidate_confounded",
    }:
        missing.append("output_or_candidate_margin_separation")
    if gate_statuses["signature"] == "not_causal_or_shuffle_fragile":
        missing.append("shuffle_or_causal_signature_stability")

    if gate_statuses["intervention"] != "passed":
        missing.append("clean_predicted_intervention")
    if gate_statuses["intervention"] in {"not_allowed", "not_tested"}:
        missing.append("intervention_test")

    if gate_statuses["null_locality"] not in {"clean", "localized"}:
        missing.append("null_and_locality_cleanliness")
    if row["mixture_profile"]["local_internal_path"] not in {"high"}:
        missing.append("local_internal_path")
    if gate_statuses["robustness_side_effects"] != "clean":
        missing.append("robustness_and_side_effects")
    if transfer_entry["transfer_class"] != "transfer_ready_mechanism":
        missing.append("transfer_or_widening")

    return sorted(set(missing))


def build_reliability_entries(
    atlas: dict[str, Any],
    comparison: dict[str, Any],
    route_disposition: dict[str, Any],
    transfer_matrix: dict[str, Any],
) -> list[dict[str, Any]]:
    route_by_row = by_id(route_disposition["route_entries"], "row_id")
    transfer_by_row = by_id(transfer_matrix["transfer_entries"], "row_id")
    audit_by_row = by_id(comparison["claim_audit"]["rows"], "row_id")
    entries: list[dict[str, Any]] = []

    for row in atlas["rows"]:
        route = route_by_row[row["id"]]
        transfer_entry = transfer_by_row[row["id"]]
        audit_entry = audit_by_row.get(row["id"])
        gates = {
            "behavior": behavior_gate_status(row, route),
            "signature": signature_gate_status(row, route),
            "intervention": intervention_gate_status(row),
            "null_locality": null_locality_status(row),
            "local_path": local_path_status(row),
            "robustness_side_effects": robustness_status(row, audit_entry),
            "transfer": transfer_status(transfer_entry),
        }
        klass = reliability_class(row, route, transfer_entry)
        entries.append(
            {
                "row_id": row["id"],
                "family": row["family"],
                "reliability_class": klass,
                "route_disposition": route["disposition"],
                "verdict": row["verdict"]["class"],
                "lead_time_state": row["lead_time"]["state"],
                "intervention_state": row["intervention"]["state"],
                "gate_statuses": gates,
                "failed_or_missing_gates": failed_or_missing_gates(
                    row,
                    gates,
                    transfer_entry,
                ),
                "failed_criteria_count": audit_entry.get("failed_criteria_count", 0)
                if audit_entry
                else 0,
                "false_criteria_count": audit_entry.get("false_criteria_count", 0)
                if audit_entry
                else 0,
                "null_criteria_count": audit_entry.get("null_criteria_count", 0)
                if audit_entry
                else 0,
                "diagnostics": row["diagnostics"],
                "allowed_claims": row["allowed_claims"],
                "forbidden_claims": row["forbidden_claims"],
                "next_decision": row["next_decision"],
                "evidence": row["evidence"],
            }
        )
    return entries


def build_validation_checks(
    atlas: dict[str, Any],
    route_disposition: dict[str, Any],
    transfer_matrix: dict[str, Any],
    entries: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    row_ids = sorted(row["id"] for row in atlas["rows"])
    entry_ids = sorted(entry["row_id"] for entry in entries)
    class_counts = summary["reliability_class_counts"]
    gate_counts = summary["gate_status_counts"]
    route_counts = route_disposition["summary"]["atlas_disposition_counts"]

    expected_class_counts: dict[str, int] = {}
    for route_class, count in route_counts.items():
        reliability = ROUTE_TO_RELIABILITY_CLASS[route_class]
        expected_class_counts[reliability] = expected_class_counts.get(reliability, 0) + count
    expected_class_counts = dict(sorted(expected_class_counts.items()))

    checks = [
        {
            "id": "reliability_entries_cover_each_atlas_row_once",
            "actual": entry_ids,
            "predicate": "sorted reliability rows == sorted atlas rows",
            "passed": entry_ids == row_ids,
            "why": "Every atlas row needs an explicit reliability disposition.",
        },
        {
            "id": "reliability_classes_match_route_dispositions",
            "actual": {
                "matrix": class_counts,
                "expected_from_routes": expected_class_counts,
            },
            "predicate": "class counts are route-disposition-derived",
            "passed": class_counts == expected_class_counts,
            "why": "Reliability classes must be an auditable derivation from the route ledger.",
        },
        {
            "id": "no_full_reliability_mechanism",
            "actual": class_counts.get("full_reliability_mechanism", 0),
            "predicate": "== 0",
            "passed": class_counts.get("full_reliability_mechanism", 0) == 0,
            "why": "No current surface clears every mechanism-card reliability gate.",
        },
        {
            "id": "mc005_is_only_bounded_reliability_reference",
            "actual": [
                entry["row_id"]
                for entry in entries
                if entry["reliability_class"] == "bounded_reliability_reference"
            ],
            "predicate": "== ['mc005_associative_lookup']",
            "passed": [
                entry["row_id"]
                for entry in entries
                if entry["reliability_class"] == "bounded_reliability_reference"
            ]
            == ["mc005_associative_lookup"],
            "why": "MC005 is the one near-mechanism specimen, still bounded by null/model-size fragility.",
        },
        {
            "id": "monitor_only_rows_are_mc004_and_mc006",
            "actual": [
                entry["row_id"]
                for entry in entries
                if entry["reliability_class"] == "not_reliable_monitor_only_no_lever"
            ],
            "predicate": "== ['mc004_in_context_binding', 'mc006_parametric_fact_override']",
            "passed": [
                entry["row_id"]
                for entry in entries
                if entry["reliability_class"] == "not_reliable_monitor_only_no_lever"
            ]
            == ["mc004_in_context_binding", "mc006_parametric_fact_override"],
            "why": "MC004 and MC006 are the current predecision monitor rows without a reliable lever.",
        },
        {
            "id": "mc012_is_prompt_visible_positive_control",
            "actual": [
                entry["row_id"]
                for entry in entries
                if entry["reliability_class"]
                == "not_reliable_prompt_visible_positive_control"
            ],
            "predicate": "== ['mc012_reliability_labeled_numeric_arbitration']",
            "passed": [
                entry["row_id"]
                for entry in entries
                if entry["reliability_class"]
                == "not_reliable_prompt_visible_positive_control"
            ]
            == ["mc012_reliability_labeled_numeric_arbitration"],
            "why": "MC012 is useful as a prompt-visible positive control, not a hidden mechanism claim.",
        },
        {
            "id": "every_row_has_failed_or_missing_gates",
            "actual": {
                entry["row_id"]: entry["failed_or_missing_gates"]
                for entry in entries
                if not entry["failed_or_missing_gates"]
            },
            "predicate": "empty dict",
            "passed": all(entry["failed_or_missing_gates"] for entry in entries),
            "why": "No current row is allowed to look reliability-complete.",
        },
        {
            "id": "no_transfer_ready_mechanism_crosscheck",
            "actual": transfer_matrix["summary"]["transfer_ready_mechanism_count"],
            "predicate": "== 0",
            "passed": transfer_matrix["summary"]["transfer_ready_mechanism_count"] == 0,
            "why": "Reliability cannot be full without transfer-ready mechanisms.",
        },
        {
            "id": "clean_predicted_intervention_missing_is_majority",
            "actual": summary["missing_gate_counts"].get("clean_predicted_intervention", 0),
            "predicate": ">= 18",
            "passed": summary["missing_gate_counts"].get("clean_predicted_intervention", 0) >= 18,
            "why": "The current atlas is dominated by absent, failed, behavior-only, or bounded-dirty interventions.",
        },
        {
            "id": "signature_gate_has_output_confounded_rows",
            "actual": gate_counts["signature"],
            "predicate": "output-visible or monitor-output-confounded rows exist",
            "passed": gate_counts["signature"].get("output_visible_or_zero_lead", 0) > 0
            and gate_counts["signature"].get(
                "monitor_only_output_or_candidate_confounded",
                0,
            )
            > 0,
            "why": "The matrix must preserve the output-geometry shadow finding.",
        },
    ]
    return checks


def build_control_surface_reliability_matrix() -> dict[str, Any]:
    atlas = load_json(ATLAS_PATH)
    comparison = load_json(COMPARISON_PATH)
    route_disposition = load_json(ROUTE_DISPOSITION_PATH)
    transfer_matrix = load_json(TRANSFER_MATRIX_PATH)

    entries = build_reliability_entries(
        atlas,
        comparison,
        route_disposition,
        transfer_matrix,
    )
    row_count = len(entries)
    class_counts = dict(
        sorted(Counter(entry["reliability_class"] for entry in entries).items())
    )
    missing_gate_counts = dict(
        sorted(
            Counter(
                gate
                for entry in entries
                for gate in entry["failed_or_missing_gates"]
            ).items()
        )
    )
    gate_status_counts = {
        gate: dict(
            sorted(Counter(entry["gate_statuses"][gate] for entry in entries).items())
        )
        for gate in [
            "behavior",
            "signature",
            "intervention",
            "null_locality",
            "local_path",
            "robustness_side_effects",
            "transfer",
        ]
    }
    summary = {
        "row_count": row_count,
        "reliability_class_counts": class_counts,
        "reliability_class_ratios": {
            key: ratio(value, row_count) for key, value in class_counts.items()
        },
        "gate_status_counts": gate_status_counts,
        "missing_gate_counts": missing_gate_counts,
        "full_reliability_count": class_counts.get("full_reliability_mechanism", 0),
        "bounded_reliability_count": class_counts.get(
            "bounded_reliability_reference",
            0,
        ),
        "monitor_only_count": class_counts.get("not_reliable_monitor_only_no_lever", 0),
        "prompt_visible_positive_control_count": class_counts.get(
            "not_reliable_prompt_visible_positive_control",
            0,
        ),
        "behavior_or_bridge_blocked_count": class_counts.get(
            "not_reliable_behavior_or_bridge_blocked",
            0,
        ),
        "output_shadow_diagnostic_count": class_counts.get(
            "not_reliable_output_shadow_diagnostic",
            0,
        ),
        "transfer_ready_mechanism_count": transfer_matrix["summary"][
            "transfer_ready_mechanism_count"
        ],
    }
    checks = build_validation_checks(
        atlas,
        route_disposition,
        transfer_matrix,
        entries,
        summary,
    )
    return {
        "schema_version": 1,
        "updated_at": atlas.get("updated_at"),
        "purpose": (
            "Record row-level reliability status for each control-surface claim, "
            "making typed failures and missing gates the primary evidence object."
        ),
        "sources": {
            "atlas": rel(ATLAS_PATH),
            "comparison": rel(COMPARISON_PATH),
            "route_disposition": rel(ROUTE_DISPOSITION_PATH),
            "transfer_matrix": rel(TRANSFER_MATRIX_PATH),
        },
        "mechanism_card_gates": [
            "behavior",
            "signature",
            "intervention",
            "null_locality",
            "local_path",
            "robustness_side_effects",
            "transfer",
        ],
        "summary": summary,
        "reliability_entries": entries,
        "validation_checks": checks,
        "allowed_claim": (
            "The current atlas has one bounded reliability reference and zero full "
            "reliability mechanisms. Typed failures are not bookkeeping; they are "
            "the measured shape of where control-surface claims break."
        ),
        "forbidden_claim": (
            "No row may be described as reliable mechanism control unless all "
            "signature, intervention, null/locality, robustness, side-effect, "
            "and transfer gates are clean under the recorded controls."
        ),
    }


def validate_reliability_matrix(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("reliability matrix schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"reliability matrix source missing: {rel_path}")
    row_ids = [entry["row_id"] for entry in payload.get("reliability_entries", [])]
    if len(row_ids) != len(set(row_ids)):
        raise AssertionError("reliability matrix has duplicate atlas rows")
    if payload["summary"]["full_reliability_count"] != 0:
        raise AssertionError("reliability matrix must not report full mechanisms")
    if payload["summary"]["bounded_reliability_count"] != 1:
        raise AssertionError("reliability matrix should have exactly one bounded reference")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(f"reliability matrix checks failed: {failed_checks}")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Control-Surface Reliability Matrix",
        "",
        "Date: 2026-07-01",
        "",
        "Status: generated reliability matrix implemented and validated.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/control_surface_reliability_matrix.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_reliability_matrix.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_reliability_matrix.py --write",
        "python code\\control_surface_reliability_matrix.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Purpose",
        "",
        "This matrix makes reliability explicit. Each row is scored across the",
        "mechanism-card gates: behavior, signature, intervention, null/locality,",
        "local internal path, robustness/side effects, and transfer. The point is",
        "to make failed claims die cleanly and make bounded claims stay bounded.",
        "",
        "## Generated Facts",
        "",
        f"- atlas rows: {summary['row_count']};",
        f"- full reliability mechanisms: {summary['full_reliability_count']};",
        f"- bounded reliability references: {summary['bounded_reliability_count']};",
        f"- monitor-only rows: {summary['monitor_only_count']};",
        f"- behavior/bridge blocked rows: {summary['behavior_or_bridge_blocked_count']};",
        f"- output-shadow diagnostic rows: {summary['output_shadow_diagnostic_count']};",
        f"- transfer-ready mechanisms: {summary['transfer_ready_mechanism_count']}.",
        "",
        "## Reliability Classes",
        "",
        "| Reliability Class | Count | Ratio |",
        "| --- | ---: | ---: |",
    ]
    for klass, count in summary["reliability_class_counts"].items():
        lines.append(
            f"| `{klass}` | {count} | "
            f"{format_value(summary['reliability_class_ratios'][klass])} |"
        )

    lines.extend(
        [
            "",
            "## Missing Gates",
            "",
            "| Missing or Failed Gate | Count |",
            "| --- | ---: |",
        ]
    )
    for gate, count in summary["missing_gate_counts"].items():
        lines.append(f"| `{gate}` | {count} |")

    lines.extend(
        [
            "",
            "## Row Matrix",
            "",
            "| Row | Reliability Class | Behavior | Signature | Intervention | Null/Locality | Transfer | Missing Gates |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for entry in payload["reliability_entries"]:
        gates = entry["gate_statuses"]
        lines.append(
            f"| `{entry['row_id']}` | `{entry['reliability_class']}` | "
            f"`{gates['behavior']}` | `{gates['signature']}` | "
            f"`{gates['intervention']}` | `{gates['null_locality']}` | "
            f"`{gates['transfer']}` | "
            f"`{', '.join(entry['failed_or_missing_gates'])}` |"
        )

    lines.extend(
        [
            "",
            "## Validation Checks",
            "",
            "| Check | Passed | Actual |",
            "| --- | --- | --- |",
        ]
    )
    for check in payload["validation_checks"]:
        lines.append(
            f"| `{check['id']}` | `{str(check['passed']).lower()}` | "
            f"`{format_value(check['actual'])}` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The most important result is not that the current atlas lacks a clean",
            "promoted mechanism. The important result is that the failures are typed.",
            "MC005 is the bounded reference: behavior, source-value signal, localized",
            "attention/write mediation, and causal effect exist, but null locality and",
            "model-size robustness prevent full reliability. MC004 and MC006 show",
            "predecision monitors without reliable levers. MC001, MC001B, and MC003",
            "are output-shadow diagnostics. MC012 is a prompt-visible positive control.",
            "The remaining bridge rows mostly die before hidden-state work is justified.",
            "",
            "This is the claim-killing engine in machine-readable form. Future work",
            "should improve the distribution by changing rows between classes, not by",
            "softening the gates.",
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

    payload = build_control_surface_reliability_matrix()
    validate_reliability_matrix(payload)
    if args.write:
        write_json(RELIABILITY_MATRIX_PATH, payload)
        RELIABILITY_MATRIX_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RELIABILITY_MATRIX_REPORT_PATH.write_text(
            render_markdown(payload),
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "passed": True,
                "row_count": payload["summary"]["row_count"],
                "reliability_class_counts": payload["summary"][
                    "reliability_class_counts"
                ],
                "missing_gate_counts": payload["summary"]["missing_gate_counts"],
                "full_reliability_count": payload["summary"][
                    "full_reliability_count"
                ],
                "bounded_reliability_count": payload["summary"][
                    "bounded_reliability_count"
                ],
                "validation_check_count": len(payload["validation_checks"]),
                "output_path": rel(RELIABILITY_MATRIX_PATH),
                "report_path": rel(RELIABILITY_MATRIX_REPORT_PATH),
            },
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
