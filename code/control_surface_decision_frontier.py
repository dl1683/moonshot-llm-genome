"""Build the current decision-frontier map.

The mixture law says where behavior lives. This layer asks when behavior
becomes visible enough to study internally: unreached because the behavior
substrate failed, zero/output-shadowed at the answer interface, monitor-only
before a lever exists, or bounded-causal but not primarily a timing result.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from control_surface_artifacts import ATLAS_PATH, ROOT, load_json
from control_surface_comparison import COMPARISON_PATH
from control_surface_mixture_law import MIXTURE_LAW_PATH


DECISION_FRONTIER_PATH = ROOT / "data" / "control_surface_decision_frontier.json"
DECISION_FRONTIER_REPORT_PATH = ROOT / "research" / "29_CONTROL_SURFACE_DECISION_FRONTIER.md"


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


def frontier_class(row: dict[str, Any]) -> str:
    state = row["lead_time"]["state"]
    if state == "not_reached":
        return "frontier_not_reached"
    if state in {"zero_lead_time", "lead_time_output_shadow"}:
        return "output_visible_at_or_before_frontier"
    if state == "lead_time_monitor_only":
        return "predecision_monitor_no_lever"
    if state in {"lead_time_causal_dirty", "lead_time_causal_clean"}:
        return "predecision_causal_candidate"
    if state == "not_primary_axis":
        return "causal_surface_not_timing_frontier"
    return "unknown_frontier_state"


def frontier_claim(row: dict[str, Any]) -> str:
    klass = frontier_class(row)
    if klass == "frontier_not_reached":
        return "Hidden-state timing work is not justified because the behavior or bridge gate failed, collapsed, or is prompt-visible."
    if klass == "output_visible_at_or_before_frontier":
        return "The observable decision is already captured by output/candidate geometry or zero-lead behavior under current controls."
    if klass == "predecision_monitor_no_lever":
        return "A predecision hidden monitor exists, but intervention, transfer, shuffle, or final-margin controls block a causal claim."
    if klass == "predecision_causal_candidate":
        return "A predecision causal candidate exists and should be stress-tested for locality, nulls, and transfer."
    if klass == "causal_surface_not_timing_frontier":
        return "A bounded causal surface exists, but the current result is an attention/write mediation surface rather than a lead-time frontier."
    return "The row uses an unknown lead-time state and needs manual review."


def next_action(row: dict[str, Any]) -> str:
    klass = frontier_class(row)
    if klass == "frontier_not_reached":
        return "Repair behavior substrate or close the route; do not probe hidden states."
    if klass == "output_visible_at_or_before_frontier":
        return "Change the answer interface or row geometry before treating hidden probes as upstream mechanisms."
    if klass == "predecision_monitor_no_lever":
        return "Either run a materially different causal stress test or record the row as monitor-only."
    if klass == "predecision_causal_candidate":
        return "Run intervention, locality, null, and transfer gates before promotion."
    if klass == "causal_surface_not_timing_frontier":
        return "Keep timing claims separate from the bounded causal surface; finish null/locality boundaries."
    return "Review lead-time schema."


def build_frontier_rows(atlas: dict[str, Any], mixture_law: dict[str, Any]) -> list[dict[str, Any]]:
    blocker_by_row = {
        profile["id"]: profile["primary_blocker"]
        for profile in mixture_law.get("row_profiles", [])
    }
    rows = []
    for row in atlas["rows"]:
        rows.append(
            {
                "id": row["id"],
                "family": row["family"],
                "frontier_class": frontier_class(row),
                "frontier_claim": frontier_claim(row),
                "next_action": next_action(row),
                "primary_blocker": blocker_by_row.get(row["id"]),
                "lead_time_state": row["lead_time"]["state"],
                "lead_time_summary": row["lead_time"]["summary"],
                "lead_time_internal_signal": row["mixture_profile"]["lead_time_internal_signal"],
                "output_geometry": row["mixture_profile"]["output_geometry"],
                "intervention_state": row["intervention"]["state"],
                "verdict": row["verdict"]["class"],
                "behavior_gate": row["behavior_gate"],
                "diagnostics": row["diagnostics"],
                "evidence": row["evidence"],
            }
        )
    return rows


def build_validation_checks(
    atlas: dict[str, Any],
    comparison: dict[str, Any],
    frontier_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    row_ids = sorted(row["id"] for row in atlas["rows"])
    frontier_ids = sorted(row["id"] for row in frontier_rows)
    class_counts = summary["frontier_class_counts"]
    lead_counts = comparison["report"]["lead_time_counts"]

    checks = [
        {
            "id": "frontier_rows_cover_each_atlas_row_once",
            "actual": frontier_ids,
            "predicate": "sorted frontier rows == sorted atlas rows",
            "passed": frontier_ids == row_ids,
            "why": "Every atlas row must have exactly one decision-frontier class.",
        },
        {
            "id": "frontier_counts_match_comparison_lead_time_counts",
            "actual": {
                "frontier_not_reached": class_counts.get("frontier_not_reached", 0),
                "not_reached": lead_counts.get("not_reached", 0),
                "output_visible": class_counts.get("output_visible_at_or_before_frontier", 0),
                "zero_plus_shadow": lead_counts.get("zero_lead_time", 0)
                + lead_counts.get("lead_time_output_shadow", 0),
                "monitor_only": class_counts.get("predecision_monitor_no_lever", 0),
                "lead_time_monitor_only": lead_counts.get("lead_time_monitor_only", 0),
            },
            "predicate": "derived frontier counts equal comparison lead-time counts",
            "passed": class_counts.get("frontier_not_reached", 0) == lead_counts.get("not_reached", 0)
            and class_counts.get("output_visible_at_or_before_frontier", 0)
            == lead_counts.get("zero_lead_time", 0) + lead_counts.get("lead_time_output_shadow", 0)
            and class_counts.get("predecision_monitor_no_lever", 0)
            == lead_counts.get("lead_time_monitor_only", 0),
            "why": "The frontier map must be a derived view of the canonical atlas/comparison state.",
        },
        {
            "id": "no_predecision_causal_candidate_yet",
            "actual": class_counts.get("predecision_causal_candidate", 0),
            "predicate": "== 0",
            "passed": class_counts.get("predecision_causal_candidate", 0) == 0,
            "why": "No current atlas row has a lead-time causal-clean or causal-dirty verdict.",
        },
        {
            "id": "monitor_only_rows_are_mc004_and_mc006",
            "actual": [
                row["id"]
                for row in frontier_rows
                if row["frontier_class"] == "predecision_monitor_no_lever"
            ],
            "predicate": "== ['mc004_in_context_binding', 'mc006_parametric_fact_override']",
            "passed": [
                row["id"]
                for row in frontier_rows
                if row["frontier_class"] == "predecision_monitor_no_lever"
            ]
            == ["mc004_in_context_binding", "mc006_parametric_fact_override"],
            "why": "MC004 and MC006 are the current predecision-monitor rows.",
        },
        {
            "id": "output_visible_frontier_exceeds_monitor_only_frontier",
            "actual": {
                "output_visible_at_or_before_frontier": class_counts.get("output_visible_at_or_before_frontier", 0),
                "predecision_monitor_no_lever": class_counts.get("predecision_monitor_no_lever", 0),
            },
            "predicate": "output-visible rows > monitor-only rows",
            "passed": class_counts.get("output_visible_at_or_before_frontier", 0)
            > class_counts.get("predecision_monitor_no_lever", 0),
            "why": "Current hidden timing evidence is more often an output shadow than an upstream lever.",
        },
        {
            "id": "mc005_is_not_counted_as_lead_time_promotion",
            "actual": [
                row["frontier_class"]
                for row in frontier_rows
                if row["id"] == "mc005_associative_lookup"
            ],
            "predicate": "== ['causal_surface_not_timing_frontier']",
            "passed": [
                row["frontier_class"]
                for row in frontier_rows
                if row["id"] == "mc005_associative_lookup"
            ]
            == ["causal_surface_not_timing_frontier"],
            "why": "MC005 is bounded causal mediation, not a lead-time causal mechanism.",
        },
    ]
    return checks


def build_control_surface_decision_frontier() -> dict[str, Any]:
    atlas = load_json(ATLAS_PATH)
    comparison = load_json(COMPARISON_PATH)
    mixture_law = load_json(MIXTURE_LAW_PATH)
    frontier_rows = build_frontier_rows(atlas, mixture_law)
    class_counts = dict(sorted(Counter(row["frontier_class"] for row in frontier_rows).items()))
    class_ratios = {
        klass: ratio(count, len(frontier_rows))
        for klass, count in class_counts.items()
    }
    lead_signal_counts = dict(
        sorted(Counter(row["lead_time_internal_signal"] for row in frontier_rows).items())
    )
    monitor_rows = [
        row["id"]
        for row in frontier_rows
        if row["frontier_class"] == "predecision_monitor_no_lever"
    ]
    output_visible_rows = [
        row["id"]
        for row in frontier_rows
        if row["frontier_class"] == "output_visible_at_or_before_frontier"
    ]
    summary = {
        "row_count": len(frontier_rows),
        "frontier_class_counts": class_counts,
        "frontier_class_ratios": class_ratios,
        "lead_time_state_counts": comparison["report"]["lead_time_counts"],
        "lead_time_internal_signal_counts": lead_signal_counts,
        "monitor_only_rows": monitor_rows,
        "output_visible_frontier_rows": output_visible_rows,
        "predecision_causal_candidate_count": class_counts.get("predecision_causal_candidate", 0),
        "causal_not_timing_rows": [
            row["id"]
            for row in frontier_rows
            if row["frontier_class"] == "causal_surface_not_timing_frontier"
        ],
    }
    checks = build_validation_checks(atlas, comparison, frontier_rows, summary)
    return {
        "schema_version": 1,
        "updated_at": atlas.get("updated_at"),
        "purpose": (
            "Make commitment timing a first-class control-surface axis: which "
            "rows never reach timing work, which are output-visible, which have "
            "predecision monitors without levers, and which causal results are "
            "not primarily timing-frontier claims."
        ),
        "sources": {
            "atlas": rel(ATLAS_PATH),
            "comparison": rel(COMPARISON_PATH),
            "mixture_law": rel(MIXTURE_LAW_PATH),
        },
        "classification_rule": {
            "frontier_not_reached": "lead_time.state == not_reached",
            "output_visible_at_or_before_frontier": "lead_time.state in {zero_lead_time, lead_time_output_shadow}",
            "predecision_monitor_no_lever": "lead_time.state == lead_time_monitor_only",
            "predecision_causal_candidate": "lead_time.state in {lead_time_causal_dirty, lead_time_causal_clean}",
            "causal_surface_not_timing_frontier": "lead_time.state == not_primary_axis",
        },
        "summary": summary,
        "frontier_rows": frontier_rows,
        "validation_checks": checks,
        "allowed_claim": (
            "In the current atlas, lead-time evidence is mostly absent, output-visible, "
            "or monitor-only. MC004 and MC006 are predecision monitor rows; no row is "
            "a promoted or bounded predecision causal lever."
        ),
        "forbidden_claim": (
            "This artifact does not show that any current lead-time signal is a causal "
            "control vector, nor that MC005's bounded causal lookup surface is a "
            "general commitment-timing mechanism."
        ),
    }


def validate_decision_frontier(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("decision frontier schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"decision frontier source missing: {rel_path}")
    row_ids = [row["id"] for row in payload.get("frontier_rows", [])]
    if len(row_ids) != len(set(row_ids)):
        raise AssertionError("decision frontier has duplicate rows")
    if payload["summary"]["predecision_causal_candidate_count"] != 0:
        raise AssertionError("decision frontier must not report lead-time causal candidates")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(f"decision frontier checks failed: {failed_checks}")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Control-Surface Decision Frontier",
        "",
        "Date: 2026-07-01",
        "",
        "Status: generated decision-frontier layer implemented and validated.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/control_surface_decision_frontier.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_decision_frontier.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_decision_frontier.py --write",
        "python code\\control_surface_decision_frontier.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Purpose",
        "",
        "The decision frontier makes lead-time a measured axis. It asks whether a",
        "behavior family has no valid timing substrate, is already output-visible,",
        "has a predecision monitor without a lever, or has a causal result that is",
        "not actually a timing-frontier claim.",
        "",
        "## Generated Facts",
        "",
        f"- atlas rows: {summary['row_count']};",
        f"- predecision causal candidates: {summary['predecision_causal_candidate_count']};",
        f"- monitor-only rows: {len(summary['monitor_only_rows'])};",
        f"- output-visible frontier rows: {len(summary['output_visible_frontier_rows'])};",
        f"- causal-not-timing rows: {len(summary['causal_not_timing_rows'])}.",
        "",
        "## Frontier Classes",
        "",
        "| Frontier Class | Count | Ratio |",
        "| --- | ---: | ---: |",
    ]
    for klass, count in summary["frontier_class_counts"].items():
        lines.append(
            f"| `{klass}` | {count} | {format_value(summary['frontier_class_ratios'][klass])} |"
        )

    lines.extend(
        [
            "",
            "## Row Map",
            "",
            "| Row | Frontier | Lead-Time State | Primary Blocker | Next Action |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["frontier_rows"]:
        lines.append(
            f"| `{row['id']}` | `{row['frontier_class']}` | `{row['lead_time_state']}` | "
            f"`{row['primary_blocker']}` | {row['next_action']} |"
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
            "The current frontier map says there is no lead-time causal lever in the",
            "atlas. MC004 and MC006 have predecision monitors, but they are not",
            "intervention-ready. MC001/MC001B/MC001G/MC003-style timing evidence is",
            "output-visible or zero-lead under current controls. MC005 remains the",
            "bounded causal result, but its claim is source-value attention/write",
            "mediation, not commitment timing.",
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

    payload = build_control_surface_decision_frontier()
    validate_decision_frontier(payload)
    if args.write:
        write_json(DECISION_FRONTIER_PATH, payload)
        DECISION_FRONTIER_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        DECISION_FRONTIER_REPORT_PATH.write_text(
            render_markdown(payload),
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "passed": True,
                "row_count": payload["summary"]["row_count"],
                "frontier_class_counts": payload["summary"]["frontier_class_counts"],
                "monitor_only_rows": payload["summary"]["monitor_only_rows"],
                "predecision_causal_candidate_count": payload["summary"]["predecision_causal_candidate_count"],
                "validation_check_count": len(payload["validation_checks"]),
                "output_path": rel(DECISION_FRONTIER_PATH),
                "report_path": rel(DECISION_FRONTIER_REPORT_PATH),
            },
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
