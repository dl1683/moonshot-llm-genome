"""Build the compact current genome snapshot.

The project now has several validated layers: atlas rows, mixture law, gate
geometry, route disposition, transfer, reliability, bridge closure, and the
next queue. This module fuses them into one small artifact that states the
current global shape without forcing readers or later tools to scrape the long
overview.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from control_surface_artifacts import ATLAS_PATH, ROOT, load_json


GENOME_SNAPSHOT_PATH = ROOT / "data" / "control_surface_genome_snapshot.json"
GENOME_SNAPSHOT_REPORT_PATH = ROOT / "research" / "35_CONTROL_SURFACE_GENOME_SNAPSHOT.md"

ARTIFACT_INDEX_PATH = ROOT / "data" / "control_surface_artifact_index.json"
COMPARISON_PATH = ROOT / "data" / "control_surface_comparison.json"
LAW_AUDIT_PATH = ROOT / "data" / "control_surface_law_audit.json"
NEXT_QUEUE_PATH = ROOT / "data" / "control_surface_next_experiment_queue.json"
SMOKE_DIAGNOSTICS_PATH = ROOT / "data" / "control_surface_smoke_diagnostics.json"
BRIDGE_LADDER_PATH = ROOT / "data" / "control_surface_bridge_ladder.json"
MIXTURE_LAW_PATH = ROOT / "data" / "control_surface_mixture_law.json"
DECISION_FRONTIER_PATH = ROOT / "data" / "control_surface_decision_frontier.json"
ROUTE_DISPOSITION_PATH = ROOT / "data" / "control_surface_route_disposition.json"
TRANSFER_MATRIX_PATH = ROOT / "data" / "control_surface_transfer_matrix.json"
RELIABILITY_MATRIX_PATH = ROOT / "data" / "control_surface_reliability_matrix.json"
ERROR_TAXONOMY_PATH = ROOT / "data" / "control_surface_error_taxonomy.json"
GATE_GEOMETRY_PATH = ROOT / "data" / "control_surface_gate_geometry.json"


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


def group_rows_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row["row_id"])
    return {name: sorted(values) for name, values in sorted(grouped.items())}


def compact_top_queue(queue: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": item["id"],
            "hypothesis_id": item["hypothesis_id"],
            "priority_class": item["priority_class"],
            "priority_score": item["priority_score"],
            "reason_codes": item["reason_codes"],
            "action_type": item["action_type"],
            "next_test": item["next_test"],
            "bridge_closure_constraint_count": len(
                item.get("bridge_closure_context", {}).get("active_constraints", [])
            ),
        }
        for item in queue["top_items"]
    ]


def global_claims(
    mixture_law: dict[str, Any],
    gate_geometry: dict[str, Any],
    reliability_matrix: dict[str, Any],
    next_queue: dict[str, Any],
) -> dict[str, list[str]]:
    mixture_summary = mixture_law["summary"]
    gate_summary = gate_geometry["summary"]
    reliability_summary = reliability_matrix["summary"]
    bridge_closure = next_queue["summary"]["bridge_closure"]
    return {
        "allowed": [
            (
                "The current small-model control-surface map is mostly prompt- "
                "and output-coupled: prompt-contract-visible pressure appears "
                f"in {mixture_summary['pressure_class_counts']['prompt_contract_visible']}/"
                f"{mixture_summary['row_count']} rows and output-geometry-visible "
                f"pressure in {mixture_summary['pressure_class_counts']['output_geometry_visible']}/"
                f"{mixture_summary['row_count']} rows."
            ),
            (
                "The mechanism-card bar has a measured geometry: "
                f"{gate_summary['ordered_gate_buckets']['pre_signature_blocked']} rows "
                "stop before signature work, "
                f"{gate_summary['ordered_gate_buckets']['signature_stage_blocked']} "
                "stop at signature-stage controls, "
                f"{gate_summary['ordered_gate_buckets']['intervention_stage_blocked']} "
                "stops at failed intervention, and "
                f"{gate_summary['ordered_gate_buckets']['reliability_stage_bounded']} "
                "is bounded at reliability."
            ),
            (
                "MC005 is the single bounded internal-causal reference specimen; "
                "it is not promoted because null, locality, side-effect, and "
                "transfer gates are still not clean."
            ),
            (
                "The bridge program is still behavior-substrate work: "
                f"{bridge_closure['bridge_rung_count']} bridge rungs, "
                f"{bridge_closure['hidden_state_allowed_count']} hidden-state-allowed "
                "rungs, and no clean unconfounded bridge."
            ),
        ],
        "forbidden": [
            "Do not claim a broad truth, honesty, factuality, or knowledge vector.",
            "Do not call output-visible, candidate-visible, prompt-visible, or monitor-only signals mechanisms.",
            "Do not reopen MC030 operation-leak guards, the MC031 statusless checksum route, the MC032 cross-table route, or the MC033 fact-claim route without a materially different branch/null/local/side-number/source-validity gate.",
            (
                "Do not report full reliability: "
                f"{reliability_summary['full_reliability_count']} rows currently "
                "clear the full reliability bar."
            ),
        ],
    }


def build_control_surface_genome_snapshot() -> dict[str, Any]:
    atlas = load_json(ATLAS_PATH)
    artifact_index = load_json(ARTIFACT_INDEX_PATH)
    comparison = load_json(COMPARISON_PATH)
    law_audit = load_json(LAW_AUDIT_PATH)
    next_queue = load_json(NEXT_QUEUE_PATH)
    smoke_diagnostics = load_json(SMOKE_DIAGNOSTICS_PATH)
    bridge_ladder = load_json(BRIDGE_LADDER_PATH)
    mixture_law = load_json(MIXTURE_LAW_PATH)
    decision_frontier = load_json(DECISION_FRONTIER_PATH)
    route_disposition = load_json(ROUTE_DISPOSITION_PATH)
    transfer_matrix = load_json(TRANSFER_MATRIX_PATH)
    reliability_matrix = load_json(RELIABILITY_MATRIX_PATH)
    error_taxonomy = load_json(ERROR_TAXONOMY_PATH)
    gate_geometry = load_json(GATE_GEOMETRY_PATH)

    row_count = len(atlas["rows"])
    gate_entries = gate_geometry["gate_entries"]
    route_entries = route_disposition["route_entries"]
    reliability_entries = reliability_matrix["reliability_entries"]
    rows_by_terminal_stage = group_rows_by_key(gate_entries, "terminal_stage")
    rows_by_route = group_rows_by_key(route_entries, "disposition")
    rows_by_reliability = group_rows_by_key(reliability_entries, "reliability_class")
    row_verdict_counts = dict(
        sorted(Counter(row["verdict"]["class"] for row in atlas["rows"]).items())
    )
    law_status_counts = dict(
        sorted(Counter(item["audit_level"] for item in law_audit["hypotheses"]).items())
    )

    snapshot = {
        "schema_version": 1,
        "updated_at": atlas.get("updated_at"),
        "purpose": (
            "Provide one compact, machine-readable current-state snapshot of the "
            "small-model control-surface genome map."
        ),
        "sources": {
            "atlas": rel(ATLAS_PATH),
            "artifact_index": rel(ARTIFACT_INDEX_PATH),
            "comparison": rel(COMPARISON_PATH),
            "law_audit": rel(LAW_AUDIT_PATH),
            "next_queue": rel(NEXT_QUEUE_PATH),
            "smoke_diagnostics": rel(SMOKE_DIAGNOSTICS_PATH),
            "bridge_ladder": rel(BRIDGE_LADDER_PATH),
            "mixture_law": rel(MIXTURE_LAW_PATH),
            "decision_frontier": rel(DECISION_FRONTIER_PATH),
            "route_disposition": rel(ROUTE_DISPOSITION_PATH),
            "transfer_matrix": rel(TRANSFER_MATRIX_PATH),
            "reliability_matrix": rel(RELIABILITY_MATRIX_PATH),
            "error_taxonomy": rel(ERROR_TAXONOMY_PATH),
            "gate_geometry": rel(GATE_GEOMETRY_PATH),
        },
        "summary": {
            "row_count": row_count,
            "artifact_count": len(artifact_index["artifacts"]),
            "verdict_counts": row_verdict_counts,
            "promoted_mechanism_count": row_verdict_counts.get(
                "promoted_mechanism_card",
                0,
            ),
            "bounded_mechanism_count": row_verdict_counts.get(
                "bounded_mechanism_card",
                0,
            ),
            "diagnostic_or_failed_count": row_verdict_counts.get(
                "diagnostic_note",
                0,
            )
            + row_verdict_counts.get("failed_mechanism_card", 0),
            "law_audit_level_counts": law_status_counts,
            "top_queue_ids": next_queue["summary"]["top_queue_ids"],
        },
        "mixture_shape": {
            "dominant_axis_ratios": mixture_law["summary"]["dominant_axis_ratios"],
            "pressure_class_counts": mixture_law["summary"]["pressure_class_counts"],
            "pressure_class_ratios": mixture_law["summary"]["pressure_class_ratios"],
            "primary_blocker_counts": mixture_law["summary"]["primary_blocker_counts"],
            "primary_blocker_rows": mixture_law["primary_blocker_rows"],
        },
        "gate_shape": {
            "terminal_stage_counts": gate_geometry["summary"]["terminal_stage_counts"],
            "terminal_stage_ratios": gate_geometry["summary"]["terminal_stage_ratios"],
            "ordered_gate_buckets": gate_geometry["summary"]["ordered_gate_buckets"],
            "ordered_gate_bucket_ratios": gate_geometry["summary"][
                "ordered_gate_bucket_ratios"
            ],
            "rows_by_terminal_stage": rows_by_terminal_stage,
        },
        "route_shape": {
            "atlas_disposition_counts": route_disposition["summary"][
                "atlas_disposition_counts"
            ],
            "rows_by_route_disposition": rows_by_route,
            "hidden_state_ready_route_count": route_disposition["summary"][
                "hidden_state_ready_route_count"
            ],
        },
        "bridge_shape": {
            "rung_count": bridge_ladder["summary"]["rung_count"],
            "smoke_rung_count": bridge_ladder["summary"]["smoke_rung_count"],
            "hidden_state_allowed_count": bridge_ladder["summary"][
                "hidden_state_allowed_count"
            ],
            "clean_unconfounded_bridge_count": bridge_ladder["summary"][
                "clean_unconfounded_bridge_count"
            ],
            "terminal_stage_counts": gate_geometry["summary"][
                "bridge_terminal_stage_counts"
            ],
            "recent_closed_rung_ids": next_queue["summary"]["bridge_closure"][
                "recent_closed_rung_ids"
            ],
            "active_constraints": next_queue["bridge_closure_context"]["constraints"],
        },
        "frontier_shape": {
            "frontier_class_counts": decision_frontier["summary"][
                "frontier_class_counts"
            ],
            "monitor_only_rows": decision_frontier["summary"]["monitor_only_rows"],
            "predecision_causal_candidate_count": decision_frontier["summary"][
                "predecision_causal_candidate_count"
            ],
        },
        "reliability_shape": {
            "reliability_class_counts": reliability_matrix["summary"][
                "reliability_class_counts"
            ],
            "rows_by_reliability_class": rows_by_reliability,
            "missing_gate_counts": reliability_matrix["summary"]["missing_gate_counts"],
            "full_reliability_count": reliability_matrix["summary"][
                "full_reliability_count"
            ],
        },
        "transfer_shape": {
            "transfer_class_counts": transfer_matrix["summary"][
                "transfer_class_counts"
            ],
            "transfer_ready_mechanism_count": transfer_matrix["summary"][
                "transfer_ready_mechanism_count"
            ],
        },
        "error_shape": {
            "smoke_card_count": smoke_diagnostics["summary"]["card_count"],
            "error_taxonomy_smoke_card_count": error_taxonomy["summary"][
                "smoke_card_count"
            ],
            "error_taxonomy_bridge_rung_count": error_taxonomy["summary"][
                "bridge_rung_count"
            ],
            "mc028_other_number_count": error_taxonomy["summary"][
                "mc028_other_number_count"
            ],
            "mc030_baseline_operation_atomic_rate": error_taxonomy["summary"][
                "mc030_baseline_operation_atomic_rate"
            ],
            "mc030_baseline_answer_absent_unknown_rate": error_taxonomy["summary"][
                "mc030_baseline_answer_absent_unknown_rate"
            ],
        },
        "next_pressure": {
            "priority_counts": next_queue["summary"]["priority_counts"],
            "immediate_or_high_count": next_queue["summary"][
                "immediate_or_high_count"
            ],
            "top_items": compact_top_queue(next_queue),
        },
        "global_claims": global_claims(
            mixture_law,
            gate_geometry,
            reliability_matrix,
            next_queue,
        ),
    }
    snapshot["validation_checks"] = build_validation_checks(snapshot)
    return snapshot


def build_validation_checks(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    row_count = snapshot["summary"]["row_count"]
    gate_counts = snapshot["gate_shape"]["terminal_stage_counts"]
    bridge_shape = snapshot["bridge_shape"]
    checks = [
        {
            "id": "terminal_stage_counts_partition_rows",
            "actual": {"stage_counts": gate_counts, "row_count": row_count},
            "predicate": "sum(stage_counts) == row_count",
            "passed": sum(gate_counts.values()) == row_count,
            "why": "The compact snapshot must preserve the gate-geometry row partition.",
        },
        {
            "id": "no_promoted_mechanisms",
            "actual": snapshot["summary"]["promoted_mechanism_count"],
            "predicate": "== 0",
            "passed": snapshot["summary"]["promoted_mechanism_count"] == 0,
            "why": "The current genome snapshot must not imply a promoted mechanism.",
        },
        {
            "id": "mc005_only_bounded_mechanism",
            "actual": snapshot["reliability_shape"]["rows_by_reliability_class"].get(
                "bounded_reliability_reference",
                [],
            ),
            "predicate": "== ['mc005_associative_lookup']",
            "passed": snapshot["reliability_shape"]["rows_by_reliability_class"].get(
                "bounded_reliability_reference",
                [],
            )
            == ["mc005_associative_lookup"],
            "why": "MC005 is the only bounded internal-causal reference row.",
        },
        {
            "id": "bridge_has_no_hidden_state_allowed_rungs",
            "actual": bridge_shape["hidden_state_allowed_count"],
            "predicate": "== 0",
            "passed": bridge_shape["hidden_state_allowed_count"] == 0,
            "why": "The bridge program remains behavior-substrate work.",
        },
        {
            "id": "bridge_has_no_clean_unconfounded_rungs",
            "actual": bridge_shape["clean_unconfounded_bridge_count"],
            "predicate": "== 0",
            "passed": bridge_shape["clean_unconfounded_bridge_count"] == 0,
            "why": "No bridge route currently licenses hidden-state mechanism work.",
        },
        {
            "id": "bridge_closure_recent_rungs_are_mc030_mc033",
            "actual": bridge_shape["recent_closed_rung_ids"],
            "predicate": "== ['MC030', 'MC031', 'MC032', 'MC033']",
            "passed": bridge_shape["recent_closed_rung_ids"]
            == ["MC030", "MC031", "MC032", "MC033"],
            "why": "The snapshot must preserve the latest bridge-closure context.",
        },
        {
            "id": "full_reliability_is_zero",
            "actual": snapshot["reliability_shape"]["full_reliability_count"],
            "predicate": "== 0",
            "passed": snapshot["reliability_shape"]["full_reliability_count"] == 0,
            "why": "No row clears the full reliability bar.",
        },
        {
            "id": "transfer_ready_is_zero",
            "actual": snapshot["transfer_shape"]["transfer_ready_mechanism_count"],
            "predicate": "== 0",
            "passed": snapshot["transfer_shape"]["transfer_ready_mechanism_count"] == 0,
            "why": "No row has transfer-ready mechanism status.",
        },
        {
            "id": "global_claims_are_nonempty",
            "actual": {
                "allowed": len(snapshot["global_claims"]["allowed"]),
                "forbidden": len(snapshot["global_claims"]["forbidden"]),
            },
            "predicate": "allowed > 0 and forbidden > 0",
            "passed": bool(snapshot["global_claims"]["allowed"])
            and bool(snapshot["global_claims"]["forbidden"]),
            "why": "The snapshot must be claim-bounded, not just a metric dump.",
        },
    ]
    return checks


def validate_genome_snapshot(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("genome snapshot schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"genome snapshot source missing: {rel_path}")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(f"genome snapshot checks failed: {failed_checks}")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    gate_shape = payload["gate_shape"]
    mixture_shape = payload["mixture_shape"]
    bridge_shape = payload["bridge_shape"]
    lines = [
        "# Control-Surface Genome Snapshot",
        "",
        "Date: 2026-07-01",
        "",
        "Status: generated compact genome snapshot implemented and validated.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/control_surface_genome_snapshot.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_genome_snapshot.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_genome_snapshot.py --write",
        "python code\\control_surface_genome_snapshot.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Purpose",
        "",
        "This is the compact current-state object for the small-model control",
        "surface genome. It fuses the atlas, mixture law, gate geometry, route",
        "disposition, bridge ladder, reliability matrix, transfer matrix, error",
        "taxonomy, and next queue.",
        "",
        "The point is not to add another interpretation layer. The point is to",
        "make the current global claim state executable: what is controllable,",
        "what only looks controllable, what is prompt-visible, what is",
        "output-visible, what is internal, what is causal, and where each claim",
        "breaks.",
        "",
        "## Current Shape",
        "",
        f"- atlas rows: {summary['row_count']};",
        f"- linked result artifacts: {summary['artifact_count']};",
        f"- verdict counts: `{format_value(summary['verdict_counts'])}`;",
        f"- bounded mechanisms: {summary['bounded_mechanism_count']};",
        f"- promoted mechanisms: {format_value(summary['promoted_mechanism_count'])};",
        f"- diagnostic or failed rows: {summary['diagnostic_or_failed_count']}.",
        "",
        "## Mixture Shape",
        "",
        "| Pressure Class | Count | Ratio |",
        "| --- | ---: | ---: |",
    ]
    for key, count in mixture_shape["pressure_class_counts"].items():
        lines.append(
            f"| `{key}` | {count} | "
            f"{format_value(mixture_shape['pressure_class_ratios'][key])} |"
        )

    lines.extend(
        [
            "",
            "## Gate Shape",
            "",
            "| Terminal Stage | Count | Ratio |",
            "| --- | ---: | ---: |",
        ]
    )
    for key, count in gate_shape["terminal_stage_counts"].items():
        lines.append(
            f"| `{key}` | {count} | "
            f"{format_value(gate_shape['terminal_stage_ratios'][key])} |"
        )

    lines.extend(
        [
            "",
            "## Bridge Shape",
            "",
            f"- bridge rungs: {bridge_shape['rung_count']};",
            f"- hidden-state-allowed bridge rungs: {bridge_shape['hidden_state_allowed_count']};",
            f"- clean unconfounded bridge rungs: {bridge_shape['clean_unconfounded_bridge_count']};",
            f"- recent closed rungs: `{format_value(bridge_shape['recent_closed_rung_ids'])}`.",
            "",
            "## Global Allowed Claims",
            "",
        ]
    )
    for claim in payload["global_claims"]["allowed"]:
        lines.append(f"- {claim}")

    lines.extend(["", "## Global Forbidden Claims", ""])
    for claim in payload["global_claims"]["forbidden"]:
        lines.append(f"- {claim}")

    lines.extend(
        [
            "",
            "## Top Next Pressures",
            "",
            "| Queue Item | Priority | Reason Codes | Next Test |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in payload["next_pressure"]["top_items"]:
        lines.append(
            f"| `{item['id']}` | `{item['priority_class']}` | "
            f"`{format_value(item['reason_codes'])}` | {item['next_test']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The current genome is not a list of discovered mechanisms. It is a",
            "measured distribution of control-surface failure and survival modes.",
            "Most of the map is prompt-contract-visible, output-visible,",
            "source/prompt-token-dependent, behavior-substrate-blocked, or",
            "monitor-only. The one internal causal specimen is bounded rather",
            "than promoted.",
            "",
            "This compact snapshot is the artifact future experiments should move.",
            "A new result matters if it changes one of these ratios, moves a row",
            "to a later gate stage, creates a real transfer-ready mechanism, or",
            "sharpens a failure bucket.",
            "",
            "## Validation",
            "",
            "The snapshot is validated as part of `python code\\validate_control_surface_atlas.py`.",
            "Validation fails if it loses the gate partition, reports promoted",
            "mechanisms, loses MC005 as the sole bounded reference, reopens bridge",
            "hidden-state work, drops MC030-MC033 closure context, or emits global",
            "claims without forbidden-claim boundaries.",
            "",
        ]
    )
    return "\n".join(lines)


def write_genome_snapshot(
    output_path: Path = GENOME_SNAPSHOT_PATH,
    report_path: Path = GENOME_SNAPSHOT_REPORT_PATH,
) -> dict[str, Any]:
    payload = build_control_surface_genome_snapshot()
    validate_genome_snapshot(payload)
    write_json(output_path, payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(payload), encoding="utf-8", newline="\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write genome snapshot artifacts")
    parser.add_argument("--json", action="store_true", help="print genome snapshot JSON")
    args = parser.parse_args()

    payload = build_control_surface_genome_snapshot()
    validate_genome_snapshot(payload)

    if args.write:
        write_genome_snapshot()
        print(
            f"wrote {GENOME_SNAPSHOT_PATH.relative_to(ROOT).as_posix()} and "
            f"{GENOME_SNAPSHOT_REPORT_PATH.relative_to(ROOT).as_posix()} "
            f"with {payload['summary']['row_count']} rows"
        )
        return
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"genome snapshot ok: {payload['summary']['row_count']} rows")
    print(
        "verdict_counts:",
        json.dumps(payload["summary"]["verdict_counts"], sort_keys=True),
    )
    print(
        "ordered_gate_buckets:",
        json.dumps(payload["gate_shape"]["ordered_gate_buckets"], sort_keys=True),
    )
    print(
        "bridge_shape:",
        json.dumps(
            {
                "rungs": payload["bridge_shape"]["rung_count"],
                "hidden_state_allowed": payload["bridge_shape"][
                    "hidden_state_allowed_count"
                ],
                "clean_unconfounded": payload["bridge_shape"][
                    "clean_unconfounded_bridge_count"
                ],
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
