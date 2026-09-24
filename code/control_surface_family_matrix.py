"""Build the cross-family control-surface matrix.

This is the compact atlas table the project needs for cumulative work: one row
per behavior family, with prompt/output/source/internal/frontier/reliability/
transfer/gate cells joined from the generated layers.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from control_surface_artifacts import ATLAS_PATH, ROOT, load_json
from control_surface_decision_frontier import DECISION_FRONTIER_PATH
from control_surface_gate_geometry import GATE_GEOMETRY_PATH
from control_surface_mixture_law import MIXTURE_LAW_PATH
from control_surface_reliability_matrix import RELIABILITY_MATRIX_PATH
from control_surface_route_disposition import ROUTE_DISPOSITION_PATH
from control_surface_transfer_matrix import TRANSFER_MATRIX_PATH


FAMILY_MATRIX_PATH = ROOT / "data" / "control_surface_family_matrix.json"
FAMILY_MATRIX_REPORT_PATH = ROOT / "research" / "41_CONTROL_SURFACE_FAMILY_MATRIX.md"

BOOLEAN_AXES = [
    "prompt_contract_visible",
    "output_geometry_visible",
    "source_or_prompt_token_dependent",
    "internal_monitor_present",
    "internal_causal_surface",
    "null_boundary_or_locality_limited",
    "transfer_unproven_or_failed",
]

MATRIX_COLUMNS = [
    "row_id",
    "family",
    "behavior_domain",
    "models",
    "behavior_gate",
    "verdict",
    "primary_blocker",
    "terminal_stage",
    "route_disposition",
    "frontier_class",
    "lead_time_state",
    "intervention_state",
    "reliability_class",
    "transfer_class",
    "transfer_value",
    "prompt_contract_visible",
    "output_geometry_visible",
    "source_or_prompt_token_dependent",
    "internal_monitor_present",
    "internal_causal_surface",
    "null_boundary_or_locality_limited",
    "transfer_unproven_or_failed",
    "failed_or_missing_gates",
    "claim_bar_action",
    "next_decision",
]


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


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def by_id(items: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = item[key]
        if item_id in result:
            raise AssertionError(f"duplicate {key}: {item_id}")
        result[item_id] = item
    return result


def behavior_domain(row_id: str, family: str) -> str:
    text = f"{row_id} {family}".lower()
    if "truth" in text or "agreement" in text:
        return "truth_agreement"
    if "known" in text or "context support" in text:
        return "known_unknown_or_context_support"
    if "delayed_copy" in row_id:
        return "delayed_copy"
    if "in_context_binding" in row_id:
        return "in_context_binding"
    if "associative_lookup" in row_id:
        return "synthetic_lookup"
    if "parametric_fact" in row_id or "capital-fact" in text:
        return "parametric_fact_override"
    if "familiar_entity" in row_id:
        return "semi_synthetic_familiar_entity"
    if "symbolic" in row_id or "derived_code" in row_id:
        return "symbolic_or_derived_code_arbitration"
    if "two_hop" in row_id:
        return "two_hop_arbitration"
    if "numeric" in row_id or "atomic_number" in row_id:
        return "numeric_or_status_arbitration"
    return "other"


def build_matrix_rows(
    atlas: dict[str, Any],
    mixture: dict[str, Any],
    frontier: dict[str, Any],
    route: dict[str, Any],
    reliability: dict[str, Any],
    transfer: dict[str, Any],
    gate: dict[str, Any],
) -> list[dict[str, Any]]:
    mixture_by_id = by_id(mixture["row_profiles"], "id")
    frontier_by_id = by_id(frontier["frontier_rows"], "id")
    route_by_id = by_id(route["route_entries"], "row_id")
    reliability_by_id = by_id(reliability["reliability_entries"], "row_id")
    transfer_by_id = by_id(transfer["transfer_entries"], "row_id")
    gate_by_id = by_id(gate["gate_entries"], "row_id")

    rows: list[dict[str, Any]] = []
    for atlas_row in atlas["rows"]:
        row_id = atlas_row["id"]
        missing_layers = [
            name
            for name, mapping in [
                ("mixture", mixture_by_id),
                ("frontier", frontier_by_id),
                ("route", route_by_id),
                ("reliability", reliability_by_id),
                ("transfer", transfer_by_id),
                ("gate", gate_by_id),
            ]
            if row_id not in mapping
        ]
        if missing_layers:
            raise AssertionError(f"{row_id}: missing joined layers {missing_layers}")

        mixture_row = mixture_by_id[row_id]
        frontier_row = frontier_by_id[row_id]
        route_row = route_by_id[row_id]
        reliability_row = reliability_by_id[row_id]
        transfer_row = transfer_by_id[row_id]
        gate_row = gate_by_id[row_id]
        pressure = set(mixture_row["pressure_classes"])
        matrix_row = {
            "row_id": row_id,
            "family": atlas_row["family"],
            "behavior_domain": behavior_domain(row_id, atlas_row["family"]),
            "models": atlas_row["models"],
            "model_count": len(atlas_row["models"]),
            "behavior_contract": atlas_row["behavior_contract"],
            "behavior_gate": atlas_row["behavior_gate"],
            "verdict": atlas_row["verdict"]["class"],
            "primary_blocker": mixture_row["primary_blocker"],
            "terminal_stage": gate_row["terminal_stage"],
            "terminal_stage_description": gate_row["terminal_stage_description"],
            "route_disposition": route_row["disposition"],
            "frontier_class": frontier_row["frontier_class"],
            "lead_time_state": frontier_row["lead_time_state"],
            "lead_time_internal_signal": frontier_row["lead_time_internal_signal"],
            "intervention_state": atlas_row["intervention"]["state"],
            "causal_control": atlas_row["mixture_profile"]["causal_control"],
            "local_internal_path": atlas_row["mixture_profile"]["local_internal_path"],
            "null_locality": atlas_row["mixture_profile"]["null_locality"],
            "reliability_class": reliability_row["reliability_class"],
            "gate_statuses": reliability_row["gate_statuses"],
            "failed_or_missing_gates": reliability_row["failed_or_missing_gates"],
            "transfer_class": transfer_row["transfer_class"],
            "transfer_value": transfer_row["transfer_value"],
            "required_transfer_gate": transfer_row["required_gate"],
            "prompt_contract_visible": "prompt_contract_visible" in pressure,
            "output_geometry_visible": "output_geometry_visible" in pressure,
            "source_or_prompt_token_dependent": "source_or_prompt_token_dependent" in pressure,
            "internal_monitor_present": "internal_monitor_present" in pressure,
            "internal_causal_surface": "internal_causal_surface" in pressure,
            "null_boundary_or_locality_limited": "null_boundary_or_locality_limited" in pressure,
            "transfer_unproven_or_failed": "transfer_unproven_or_failed" in pressure,
            "pressure_classes": mixture_row["pressure_classes"],
            "diagnostics": atlas_row["diagnostics"],
            "allowed_claim_count": len(atlas_row["allowed_claims"]),
            "forbidden_claim_count": len(atlas_row["forbidden_claims"]),
            "claim_bar_action": gate_row["claim_bar_action"],
            "promotion_rule": route_row["promotion_rule"],
            "death_rule": route_row["death_rule"],
            "allowed_action": route_row["allowed_action"],
            "next_decision": atlas_row["next_decision"],
            "evidence_count": len(atlas_row["evidence"]),
        }
        rows.append(matrix_row)
    return rows


def count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(row[key] for row in rows).items()))


def count_true(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row[key] is True)


def rows_by_value(rows: list[dict[str, Any]], key: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in rows:
        result.setdefault(row[key], []).append(row["row_id"])
    return {key: sorted(value) for key, value in sorted(result.items())}


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_count = len(rows)
    axis_counts = {axis: count_true(rows, axis) for axis in BOOLEAN_AXES}
    return {
        "row_count": row_count,
        "matrix_column_count": len(MATRIX_COLUMNS),
        "complete_join_row_count": sum(1 for row in rows if not row.get("join_gaps")),
        "behavior_domain_counts": count_values(rows, "behavior_domain"),
        "verdict_counts": count_values(rows, "verdict"),
        "primary_blocker_counts": count_values(rows, "primary_blocker"),
        "terminal_stage_counts": count_values(rows, "terminal_stage"),
        "frontier_class_counts": count_values(rows, "frontier_class"),
        "reliability_class_counts": count_values(rows, "reliability_class"),
        "transfer_class_counts": count_values(rows, "transfer_class"),
        "transfer_value_counts": count_values(rows, "transfer_value"),
        "boolean_axis_counts": axis_counts,
        "boolean_axis_ratios": {
            axis: ratio(count, row_count) for axis, count in sorted(axis_counts.items())
        },
        "promotion_ready_row_count": sum(
            1
            for row in rows
            if row["verdict"] == "promoted_mechanism_card"
            or row["reliability_class"] == "full_reliability_mechanism"
            or row["transfer_class"] == "transfer_ready_mechanism"
        ),
        "bounded_reference_rows": [
            row["row_id"]
            for row in rows
            if row["verdict"] == "bounded_mechanism_card"
            or row["reliability_class"] == "bounded_reliability_reference"
        ],
        "monitor_only_rows": [
            row["row_id"]
            for row in rows
            if row["frontier_class"] == "predecision_monitor_no_lever"
        ],
        "matrix_columns": MATRIX_COLUMNS,
    }


def build_validation_checks(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    mixture: dict[str, Any],
    frontier: dict[str, Any],
    reliability: dict[str, Any],
    transfer: dict[str, Any],
    gate: dict[str, Any],
) -> list[dict[str, Any]]:
    row_ids = sorted(row["row_id"] for row in rows)
    boolean_counts = summary["boolean_axis_counts"]
    checks = [
        {
            "id": "matrix_has_one_row_per_atlas_family",
            "predicate": "row_count == 19 and row ids unique",
            "actual": {"row_count": summary["row_count"], "unique_row_count": len(set(row_ids))},
            "passed": summary["row_count"] == 19 and len(set(row_ids)) == 19,
            "why": "The family matrix must cover every atlas row exactly once.",
        },
        {
            "id": "matrix_columns_are_declared",
            "predicate": "matrix column count matches declared columns",
            "actual": {
                "matrix_column_count": summary["matrix_column_count"],
                "declared_column_count": len(MATRIX_COLUMNS),
            },
            "passed": summary["matrix_column_count"] == len(MATRIX_COLUMNS),
            "why": "Downstream work should depend on a stable cross-family column set.",
        },
        {
            "id": "primary_blockers_match_mixture_law",
            "predicate": "matrix primary blockers == mixture law primary blockers",
            "actual": {
                "matrix": summary["primary_blocker_counts"],
                "mixture_law": mixture["summary"]["primary_blocker_counts"],
            },
            "passed": summary["primary_blocker_counts"]
            == mixture["summary"]["primary_blocker_counts"],
            "why": "The matrix must be a derived table, not a new blocker taxonomy.",
        },
        {
            "id": "frontier_counts_match_decision_frontier",
            "predicate": "matrix frontier counts == decision frontier counts",
            "actual": {
                "matrix": summary["frontier_class_counts"],
                "decision_frontier": frontier["summary"]["frontier_class_counts"],
            },
            "passed": summary["frontier_class_counts"]
            == frontier["summary"]["frontier_class_counts"],
            "why": "Lead-time cells must match the canonical decision-frontier layer.",
        },
        {
            "id": "reliability_counts_match_reliability_matrix",
            "predicate": "matrix reliability counts == reliability matrix counts",
            "actual": {
                "matrix": summary["reliability_class_counts"],
                "reliability_matrix": reliability["summary"]["reliability_class_counts"],
            },
            "passed": summary["reliability_class_counts"]
            == reliability["summary"]["reliability_class_counts"],
            "why": "Reliability cells must match the canonical reliability matrix.",
        },
        {
            "id": "transfer_counts_match_transfer_matrix",
            "predicate": "matrix transfer counts == transfer matrix counts",
            "actual": {
                "matrix": summary["transfer_class_counts"],
                "transfer_matrix": transfer["summary"]["transfer_class_counts"],
            },
            "passed": summary["transfer_class_counts"]
            == transfer["summary"]["transfer_class_counts"],
            "why": "Transfer cells must match the canonical transfer matrix.",
        },
        {
            "id": "terminal_stage_counts_match_gate_geometry",
            "predicate": "matrix terminal stages == gate geometry terminal stages",
            "actual": {
                "matrix": summary["terminal_stage_counts"],
                "gate_geometry": gate["summary"]["terminal_stage_counts"],
            },
            "passed": summary["terminal_stage_counts"]
            == gate["summary"]["terminal_stage_counts"],
            "why": "Gate-stage cells must match the canonical gate-geometry layer.",
        },
        {
            "id": "boolean_axis_counts_match_mixture_pressure_counts",
            "predicate": "selected matrix boolean axes == mixture pressure counts",
            "actual": {
                "matrix": boolean_counts,
                "mixture_law": {
                    axis: mixture["summary"]["pressure_class_counts"].get(axis, 0)
                    for axis in BOOLEAN_AXES
                },
            },
            "passed": all(
                boolean_counts[axis]
                == mixture["summary"]["pressure_class_counts"].get(axis, 0)
                for axis in BOOLEAN_AXES
            ),
            "why": "Boolean cross-family cells are direct pressure-class memberships.",
        },
        {
            "id": "mc005_is_sole_bounded_internal_reference",
            "predicate": "bounded reference rows == ['mc005_associative_lookup']",
            "actual": summary["bounded_reference_rows"],
            "passed": summary["bounded_reference_rows"] == ["mc005_associative_lookup"],
            "why": "The matrix must preserve MC005 as the only bounded internal-causal reference.",
        },
        {
            "id": "mc006_and_mc004_are_monitor_only_rows",
            "predicate": "monitor-only rows == MC004 and MC006",
            "actual": summary["monitor_only_rows"],
            "passed": summary["monitor_only_rows"]
            == ["mc004_in_context_binding", "mc006_parametric_fact_override"],
            "why": "The matrix must preserve the current lead-time monitor-only frontier.",
        },
        {
            "id": "no_promotion_ready_rows",
            "predicate": "promotion_ready_row_count == 0",
            "actual": summary["promotion_ready_row_count"],
            "passed": summary["promotion_ready_row_count"] == 0,
            "why": "The cross-family table must not imply a promoted/full/transfer-ready mechanism.",
        },
        {
            "id": "each_row_has_claim_boundaries_and_evidence",
            "predicate": "allowed/forbidden/evidence counts all nonzero",
            "actual": [
                [
                    row["row_id"],
                    row["allowed_claim_count"],
                    row["forbidden_claim_count"],
                    row["evidence_count"],
                ]
                for row in rows
                if row["allowed_claim_count"] == 0
                or row["forbidden_claim_count"] == 0
                or row["evidence_count"] == 0
            ],
            "passed": all(
                row["allowed_claim_count"] > 0
                and row["forbidden_claim_count"] > 0
                and row["evidence_count"] > 0
                for row in rows
            ),
            "why": "A cross-family row without claim boundaries or evidence is not audit-ready.",
        },
    ]
    return checks


def build_control_surface_family_matrix() -> dict[str, Any]:
    atlas = load_json(ATLAS_PATH)
    mixture = load_json(MIXTURE_LAW_PATH)
    frontier = load_json(DECISION_FRONTIER_PATH)
    route = load_json(ROUTE_DISPOSITION_PATH)
    reliability = load_json(RELIABILITY_MATRIX_PATH)
    transfer = load_json(TRANSFER_MATRIX_PATH)
    gate = load_json(GATE_GEOMETRY_PATH)
    rows = build_matrix_rows(atlas, mixture, frontier, route, reliability, transfer, gate)
    summary = build_summary(rows)
    validation_checks = build_validation_checks(
        rows, summary, mixture, frontier, reliability, transfer, gate
    )
    return {
        "schema_version": 1,
        "updated_at": atlas.get("updated_at"),
        "purpose": (
            "Provide one compact cross-family matrix for the current small-LLM "
            "control-surface genome: every atlas row joined to its prompt/output/"
            "source/internal/frontier/reliability/transfer/gate cells."
        ),
        "sources": {
            "atlas": rel(ATLAS_PATH),
            "mixture_law": rel(MIXTURE_LAW_PATH),
            "decision_frontier": rel(DECISION_FRONTIER_PATH),
            "route_disposition": rel(ROUTE_DISPOSITION_PATH),
            "reliability_matrix": rel(RELIABILITY_MATRIX_PATH),
            "transfer_matrix": rel(TRANSFER_MATRIX_PATH),
            "gate_geometry": rel(GATE_GEOMETRY_PATH),
        },
        "classification_rule": {
            "rows": "One row per atlas behavior family.",
            "columns": "Columns are direct joins from existing generated layers; boolean axes are mixture-law pressure class memberships.",
            "behavior_domain": "A coarse human-readable grouping derived from row id and family name for scanning only; it is not a mechanism claim.",
        },
        "summary": summary,
        "rows_by_primary_blocker": rows_by_value(rows, "primary_blocker"),
        "rows_by_terminal_stage": rows_by_value(rows, "terminal_stage"),
        "rows_by_frontier_class": rows_by_value(rows, "frontier_class"),
        "rows_by_reliability_class": rows_by_value(rows, "reliability_class"),
        "rows_by_transfer_class": rows_by_value(rows, "transfer_class"),
        "matrix_rows": rows,
        "validation_checks": validation_checks,
        "allowed_claim": (
            "This matrix is the current cross-family atlas table: it shows how "
            "each behavior family distributes across visible prompt/output/source "
            "surfaces, internal monitors, bounded causal evidence, reliability "
            "gates, transfer status, and terminal claim boundary."
        ),
        "forbidden_claim": (
            "This matrix is not a promotion artifact and does not claim a general "
            "truth vector, general knowledge vector, full-reliability mechanism, "
            "or transfer-ready control surface."
        ),
    }


def validate_family_matrix(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("family matrix schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"family matrix source missing: {rel_path}")
    rows = payload.get("matrix_rows", [])
    row_ids = [row["row_id"] for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise AssertionError("family matrix has duplicate row ids")
    if payload["summary"]["row_count"] != 19:
        raise AssertionError("family matrix expected 19 rows")
    if payload["summary"]["matrix_columns"] != MATRIX_COLUMNS:
        raise AssertionError("family matrix columns changed unexpectedly")
    if payload["summary"]["promotion_ready_row_count"] != 0:
        raise AssertionError("family matrix must not report promotion-ready rows")
    if payload["summary"]["bounded_reference_rows"] != ["mc005_associative_lookup"]:
        raise AssertionError("family matrix expected MC005 as sole bounded reference")
    if payload["summary"]["boolean_axis_counts"]["prompt_contract_visible"] != 19:
        raise AssertionError("family matrix expected universal prompt-contract pressure")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(f"family matrix checks failed: {failed_checks}")


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Control-Surface Family Matrix",
        "",
        f"Source updated_at: {payload['updated_at']}",
        "",
        "Status: generated cross-family matrix implemented and validated.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/control_surface_family_matrix.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_family_matrix.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_family_matrix.py --write",
        "python code\\control_surface_family_matrix.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Purpose",
        "",
        "This is the compact cross-family table for the current atlas. It joins",
        "each behavior family to the mixture-law, decision-frontier, route,",
        "reliability, transfer, and gate-geometry cells that define its current",
        "claim boundary.",
        "",
        "## Summary",
        "",
        f"- rows: {summary['row_count']};",
        f"- columns: {summary['matrix_column_count']};",
        f"- promotion-ready rows: {summary['promotion_ready_row_count']};",
        f"- bounded reference rows: {format_value(summary['bounded_reference_rows'])};",
        f"- monitor-only rows: {format_value(summary['monitor_only_rows'])}.",
        "",
        "Boolean axis counts:",
        "",
    ]
    for axis, count in summary["boolean_axis_counts"].items():
        lines.append(
            f"- `{axis}`: {count}/{summary['row_count']} "
            f"({format_value(summary['boolean_axis_ratios'][axis])});"
        )

    lines.extend(
        [
            "",
            "## Matrix",
            "",
            "| Row | Domain | Verdict | Blocker | Stage | Frontier | Reliability | Transfer | Axes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["matrix_rows"]:
        axes = [
            axis
            for axis in BOOLEAN_AXES
            if row[axis]
        ]
        lines.append(
            f"| `{row['row_id']}` | `{row['behavior_domain']}` | `{row['verdict']}` | "
            f"`{row['primary_blocker']}` | `{row['terminal_stage']}` | "
            f"`{row['frontier_class']}` | `{row['reliability_class']}` | "
            f"`{row['transfer_class']}` | {'<br>'.join(f'`{axis}`' for axis in axes)} |"
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

    payload = build_control_surface_family_matrix()
    validate_family_matrix(payload)
    if args.write:
        write_json(FAMILY_MATRIX_PATH, payload)
        FAMILY_MATRIX_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        FAMILY_MATRIX_REPORT_PATH.write_text(
            render_markdown(payload),
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "passed": True,
                "row_count": payload["summary"]["row_count"],
                "matrix_column_count": payload["summary"]["matrix_column_count"],
                "promotion_ready_row_count": payload["summary"][
                    "promotion_ready_row_count"
                ],
                "bounded_reference_rows": payload["summary"]["bounded_reference_rows"],
                "monitor_only_rows": payload["summary"]["monitor_only_rows"],
                "output_path": rel(FAMILY_MATRIX_PATH),
                "report_path": rel(FAMILY_MATRIX_REPORT_PATH),
            },
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
