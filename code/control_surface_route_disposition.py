"""Build the current route-disposition ledger.

This is the claim-killing layer. It turns atlas rows and bridge rungs into
explicit dispositions: closed before hidden-state work, output-shadow baseline,
monitor-only, prompt-visible positive control, bounded frozen mechanism, or
failed intervention route.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from control_surface_artifacts import ATLAS_PATH, ROOT, load_json
from control_surface_bridge_ladder import BRIDGE_LADDER_PATH
from control_surface_decision_frontier import DECISION_FRONTIER_PATH
from control_surface_mixture_law import MIXTURE_LAW_PATH
from control_surface_next_queue import NEXT_QUEUE_PATH


ROUTE_DISPOSITION_PATH = ROOT / "data" / "control_surface_route_disposition.json"
ROUTE_DISPOSITION_REPORT_PATH = ROOT / "research" / "30_CONTROL_SURFACE_ROUTE_DISPOSITION.md"


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


def by_id(items: list[dict[str, Any]], key: str = "id") -> dict[str, dict[str, Any]]:
    return {item[key]: item for item in items}


def row_disposition(row: dict[str, Any], frontier: dict[str, Any], mixture: dict[str, Any]) -> str:
    diagnostics = set(row["diagnostics"])
    if row["verdict"]["class"] == "bounded_mechanism_card":
        return "bounded_mechanism_frozen"
    if "RELIABILITY_PROMPT_CHANNEL_VISIBLE" in diagnostics:
        return "prompt_visible_positive_control"
    if "DELAYED_CITY_ROUTE_CLOSED_MONITOR_ONLY" in diagnostics:
        return "monitor_only_closed"
    if frontier["frontier_class"] == "predecision_monitor_no_lever":
        return "monitor_only_conditional_revisit"
    if row["verdict"]["class"] == "failed_mechanism_card" or row["intervention"]["state"] == "failed":
        return "failed_intervention_or_mechanism_route"
    if row["lead_time"]["state"] == "not_reached" or row["intervention"]["state"] == "not_allowed":
        return "closed_before_hidden_state"
    if mixture["primary_blocker"] == "output_geometry_shadow":
        return "output_shadow_diagnostic_baseline"
    return "diagnostic_baseline_only"


DISPOSITION_POLICIES = {
    "bounded_mechanism_frozen": {
        "allowed_action": "Use as reference specimen; do not continue ordinary repair of the same route.",
        "promotion_rule": "Promote only with a materially different intervention or transfer/null-locality result that repairs the bounded null boundary without collateral damage.",
        "death_rule": "Ordinary same-route patching is dead; repeated null-locality failure keeps the result bounded.",
    },
    "prompt_visible_positive_control": {
        "allowed_action": "Use as positive control; do not run hidden-state mechanism work.",
        "promotion_rule": "Reopen only if the same contrast survives prompt-channel ablation, matching, or an equivalent prompt-locality control.",
        "death_rule": "If the contrast disappears when visible source-status text is removed, the mechanism route is dead under this contract.",
    },
    "monitor_only_closed": {
        "allowed_action": "Record as monitor-only; do not continue the current prompt family as a promotion route.",
        "promotion_rule": "Reopen only with a materially different behavior family or a preregistered known-confounded causal stress test.",
        "death_rule": "Final/candidate-margin, transfer-bank, or shuffle-null failure closes the current monitor route.",
    },
    "monitor_only_conditional_revisit": {
        "allowed_action": "Revisit only with larger source-disjoint controls or a materially different causal stress test.",
        "promotion_rule": "Promote only if a local intervention changes holdout behavior while preserving nulls, side rows, prompt rewrites, and transfer checks.",
        "death_rule": "Shuffle, subgroup, source-disjoint, or output/candidate baseline failure keeps the row monitor-only.",
    },
    "failed_intervention_or_mechanism_route": {
        "allowed_action": "Treat as a failed mechanism route unless a new prompt contract is preregistered.",
        "promotion_rule": "Reopen only if a new substrate and intervention beat output, prompt, shuffle, null, and side-effect controls.",
        "death_rule": "Current intervention family is dead when matched controls equal or exceed the effect.",
    },
    "closed_before_hidden_state": {
        "allowed_action": "Do not probe hidden states; rebuild the behavior substrate or close the branch.",
        "promotion_rule": "Start hidden-state work only after behavior, parseability, nulls, source-disjoint holdout, output/candidate baselines, and prompt-channel controls pass.",
        "death_rule": "Behavior-gate collapse, prompt-visible label channel, null failure, or rule-following failure kills the current route.",
    },
    "output_shadow_diagnostic_baseline": {
        "allowed_action": "Use as output-geometry diagnostic; change interface or row geometry before probing again.",
        "promotion_rule": "Reopen only if a hidden signature beats output/candidate margins on source-disjoint holdout and supports a local intervention.",
        "death_rule": "If output/candidate geometry matches the hidden result, the current mechanism claim is dead.",
    },
    "diagnostic_baseline_only": {
        "allowed_action": "Use as a diagnostic baseline only.",
        "promotion_rule": "Define a new preregistered behavior and intervention route before promotion.",
        "death_rule": "No mechanism claim survives without signature, intervention, and reliability gates.",
    },
}


def bridge_rung_disposition(rung: dict[str, Any]) -> str:
    if rung["local_vs_learned_mixture"] == "clean_prompt_visible":
        return "prompt_visible_positive_control"
    if rung["hidden_state_allowed"]:
        return "hidden_state_allowed_bridge_candidate"
    if rung["behavior_ready"]:
        return "behavior_ready_but_not_signature_ready"
    return "bridge_rung_closed_before_hidden_state"


def build_route_entries(
    atlas: dict[str, Any],
    frontier_payload: dict[str, Any],
    mixture_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    frontier_by_row = by_id(frontier_payload["frontier_rows"])
    mixture_by_row = by_id(mixture_payload["row_profiles"])
    entries: list[dict[str, Any]] = []
    for row in atlas["rows"]:
        frontier = frontier_by_row[row["id"]]
        mixture = mixture_by_row[row["id"]]
        disposition = row_disposition(row, frontier, mixture)
        policy = DISPOSITION_POLICIES[disposition]
        entries.append(
            {
                "row_id": row["id"],
                "family": row["family"],
                "disposition": disposition,
                "allowed_action": policy["allowed_action"],
                "promotion_rule": policy["promotion_rule"],
                "death_rule": policy["death_rule"],
                "next_decision": row["next_decision"],
                "primary_blocker": mixture["primary_blocker"],
                "frontier_class": frontier["frontier_class"],
                "behavior_gate": row["behavior_gate"],
                "lead_time_state": row["lead_time"]["state"],
                "intervention_state": row["intervention"]["state"],
                "verdict": row["verdict"]["class"],
                "diagnostics": row["diagnostics"],
                "evidence": row["evidence"],
            }
        )
    return entries


def build_bridge_entries(bridge_ladder: dict[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for rung in bridge_ladder["rungs"]:
        disposition = bridge_rung_disposition(rung)
        policy = DISPOSITION_POLICIES.get(
            disposition,
            DISPOSITION_POLICIES["closed_before_hidden_state"],
        )
        entries.append(
            {
                "card_id": rung["card_id"],
                "row_id": rung.get("row_id"),
                "source": rung["source"],
                "title": rung["title"],
                "disposition": disposition,
                "allowed_action": policy["allowed_action"],
                "promotion_rule": policy["promotion_rule"],
                "death_rule": policy["death_rule"],
                "behavior_ready": rung["behavior_ready"],
                "signature_ready": rung["signature_ready"],
                "hidden_state_allowed": rung["hidden_state_allowed"],
                "local_vs_learned_mixture": rung["local_vs_learned_mixture"],
                "dominant_failure": rung["dominant_failure"],
                "claim_boundary": rung["claim_boundary"],
                "evidence_path": rung["evidence_path"],
            }
        )
    return entries


def build_validation_checks(
    atlas: dict[str, Any],
    route_entries: list[dict[str, Any]],
    bridge_entries: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    row_ids = sorted(row["id"] for row in atlas["rows"])
    route_ids = sorted(entry["row_id"] for entry in route_entries)
    disposition_counts = summary["atlas_disposition_counts"]
    bridge_counts = summary["bridge_disposition_counts"]
    checks = [
        {
            "id": "route_entries_cover_each_atlas_row_once",
            "actual": route_ids,
            "predicate": "sorted route rows == sorted atlas rows",
            "passed": route_ids == row_ids,
            "why": "Every atlas row must have exactly one route disposition.",
        },
        {
            "id": "mc005_is_bounded_frozen",
            "actual": [
                entry["disposition"]
                for entry in route_entries
                if entry["row_id"] == "mc005_associative_lookup"
            ],
            "predicate": "== ['bounded_mechanism_frozen']",
            "passed": [
                entry["disposition"]
                for entry in route_entries
                if entry["row_id"] == "mc005_associative_lookup"
            ]
            == ["bounded_mechanism_frozen"],
            "why": "MC005 is the bounded reference specimen, not an ordinary active repair route.",
        },
        {
            "id": "mc006_is_monitor_only_closed",
            "actual": [
                entry["disposition"]
                for entry in route_entries
                if entry["row_id"] == "mc006_parametric_fact_override"
            ],
            "predicate": "== ['monitor_only_closed']",
            "passed": [
                entry["disposition"]
                for entry in route_entries
                if entry["row_id"] == "mc006_parametric_fact_override"
            ]
            == ["monitor_only_closed"],
            "why": "The current MC006 delayed-city route is closed as monitor-only.",
        },
        {
            "id": "mc012_is_prompt_visible_positive_control",
            "actual": [
                entry["disposition"]
                for entry in route_entries
                if entry["row_id"] == "mc012_reliability_labeled_numeric_arbitration"
            ],
            "predicate": "== ['prompt_visible_positive_control']",
            "passed": [
                entry["disposition"]
                for entry in route_entries
                if entry["row_id"] == "mc012_reliability_labeled_numeric_arbitration"
            ]
            == ["prompt_visible_positive_control"],
            "why": "MC012 is useful as a positive control but blocked by visible source-status text.",
        },
        {
            "id": "no_active_hidden_state_route",
            "actual": disposition_counts,
            "predicate": "no disposition is active_hidden_state_search",
            "passed": "active_hidden_state_search" not in disposition_counts,
            "why": "No current route is hidden-state-ready under the atlas gates.",
        },
        {
            "id": "bridge_rungs_have_no_hidden_state_allowed_candidate",
            "actual": {
                "hidden_state_allowed_bridge_candidate": bridge_counts.get(
                    "hidden_state_allowed_bridge_candidate",
                    0,
                ),
                "bridge_entries": len(bridge_entries),
            },
            "predicate": "hidden_state_allowed_bridge_candidate == 0",
            "passed": bridge_counts.get("hidden_state_allowed_bridge_candidate", 0) == 0,
            "why": "The bridge ladder must not reopen hidden-state work without a passing bridge substrate.",
        },
        {
            "id": "closed_or_bounded_routes_dominate",
            "actual": {
                "closed_before_hidden_state": disposition_counts.get(
                    "closed_before_hidden_state",
                    0,
                ),
                "bounded_mechanism_frozen": disposition_counts.get(
                    "bounded_mechanism_frozen",
                    0,
                ),
                "row_count": len(route_entries),
            },
            "predicate": "closed_before_hidden_state + bounded_mechanism_frozen >= 10",
            "passed": disposition_counts.get("closed_before_hidden_state", 0)
            + disposition_counts.get("bounded_mechanism_frozen", 0)
            >= 10,
            "why": "The current atlas is mostly a closed-route and bounded-result ledger.",
        },
    ]
    return checks


def build_control_surface_route_disposition() -> dict[str, Any]:
    atlas = load_json(ATLAS_PATH)
    frontier_payload = load_json(DECISION_FRONTIER_PATH)
    mixture_payload = load_json(MIXTURE_LAW_PATH)
    bridge_ladder = load_json(BRIDGE_LADDER_PATH)
    next_queue = load_json(NEXT_QUEUE_PATH)
    route_entries = build_route_entries(atlas, frontier_payload, mixture_payload)
    bridge_entries = build_bridge_entries(bridge_ladder)
    disposition_counts = dict(
        sorted(Counter(entry["disposition"] for entry in route_entries).items())
    )
    bridge_counts = dict(
        sorted(Counter(entry["disposition"] for entry in bridge_entries).items())
    )
    row_count = len(route_entries)
    summary = {
        "row_count": row_count,
        "atlas_disposition_counts": disposition_counts,
        "atlas_disposition_ratios": {
            key: ratio(value, row_count) for key, value in disposition_counts.items()
        },
        "bridge_rung_count": len(bridge_entries),
        "bridge_disposition_counts": bridge_counts,
        "bridge_hidden_state_allowed_count": sum(
            1 for entry in bridge_entries if entry["hidden_state_allowed"]
        ),
        "hidden_state_ready_route_count": disposition_counts.get(
            "active_hidden_state_search",
            0,
        ),
        "top_queue_ids": next_queue["summary"]["top_queue_ids"],
        "immediate_or_high_queue_count": next_queue["summary"][
            "immediate_or_high_count"
        ],
    }
    checks = build_validation_checks(atlas, route_entries, bridge_entries, summary)
    return {
        "schema_version": 1,
        "updated_at": atlas.get("updated_at"),
        "purpose": (
            "Classify each atlas row and bridge rung by route disposition so "
            "promotion, bounded status, closure, and conditional revisit rules "
            "are explicit and validated."
        ),
        "sources": {
            "atlas": rel(ATLAS_PATH),
            "decision_frontier": rel(DECISION_FRONTIER_PATH),
            "mixture_law": rel(MIXTURE_LAW_PATH),
            "bridge_ladder": rel(BRIDGE_LADDER_PATH),
            "next_queue": rel(NEXT_QUEUE_PATH),
        },
        "disposition_policies": DISPOSITION_POLICIES,
        "summary": summary,
        "route_entries": route_entries,
        "bridge_rung_entries": bridge_entries,
        "validation_checks": checks,
        "allowed_claim": (
            "The current route ledger has no active hidden-state-ready route: "
            "MC005 is bounded and frozen, MC006 is monitor-only closed, MC012 is "
            "a prompt-visible positive control, and most bridge routes are closed "
            "before hidden-state work."
        ),
        "forbidden_claim": (
            "This ledger does not promote a new mechanism. It prevents closed, "
            "prompt-visible, output-shadowed, or monitor-only routes from being "
            "treated as active mechanism candidates."
        ),
    }


def validate_route_disposition(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("route disposition schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"route disposition source missing: {rel_path}")
    row_ids = [entry["row_id"] for entry in payload.get("route_entries", [])]
    if len(row_ids) != len(set(row_ids)):
        raise AssertionError("route disposition has duplicate atlas rows")
    if payload["summary"]["hidden_state_ready_route_count"] != 0:
        raise AssertionError("route disposition must not report hidden-state-ready routes")
    if payload["summary"]["bridge_hidden_state_allowed_count"] != 0:
        raise AssertionError("bridge rungs must not allow hidden-state work")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(f"route disposition checks failed: {failed_checks}")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Control-Surface Route Disposition",
        "",
        "Date: 2026-07-01",
        "",
        "Status: generated route-disposition layer implemented and validated.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/control_surface_route_disposition.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_route_disposition.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_route_disposition.py --write",
        "python code\\control_surface_route_disposition.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Purpose",
        "",
        "The route-disposition ledger is the claim-killing layer. It says which",
        "routes are closed, bounded, monitor-only, prompt-visible positive",
        "controls, output-shadow diagnostics, or failed interventions. It also",
        "states what would be required to reopen or promote a route.",
        "",
        "## Generated Facts",
        "",
        f"- atlas rows: {summary['row_count']};",
        f"- bridge rungs: {summary['bridge_rung_count']};",
        f"- hidden-state-ready atlas routes: {summary['hidden_state_ready_route_count']};",
        f"- bridge rungs allowing hidden-state work: {summary['bridge_hidden_state_allowed_count']};",
        f"- immediate/high next-queue items: {summary['immediate_or_high_queue_count']}.",
        "",
        "## Atlas Dispositions",
        "",
        "| Disposition | Count | Ratio |",
        "| --- | ---: | ---: |",
    ]
    for disposition, count in summary["atlas_disposition_counts"].items():
        lines.append(
            f"| `{disposition}` | {count} | "
            f"{format_value(summary['atlas_disposition_ratios'][disposition])} |"
        )

    lines.extend(
        [
            "",
            "## Row Ledger",
            "",
            "| Row | Disposition | Primary Blocker | Frontier | Allowed Action |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for entry in payload["route_entries"]:
        lines.append(
            f"| `{entry['row_id']}` | `{entry['disposition']}` | "
            f"`{entry['primary_blocker']}` | `{entry['frontier_class']}` | "
            f"{entry['allowed_action']} |"
        )

    lines.extend(
        [
            "",
            "## Bridge Rungs",
            "",
            "| Card | Source | Disposition | Mixture | Boundary |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for entry in payload["bridge_rung_entries"]:
        lines.append(
            f"| `{entry['card_id']}` | `{entry['source']}` | `{entry['disposition']}` | "
            f"`{entry['local_vs_learned_mixture']}` | {entry['claim_boundary']} |"
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
            "This ledger makes negative results operational. A closed route is not a",
            "loose invitation to keep trying adjacent prompt tweaks; it is a branch",
            "that can only be reopened by satisfying the listed promotion rule.",
            "MC005 remains the bounded reference specimen, MC006 is closed",
            "monitor-only under the current prompt families, and MC012 remains a",
            "prompt-visible positive control rather than a mechanism candidate.",
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

    payload = build_control_surface_route_disposition()
    validate_route_disposition(payload)
    if args.write:
        write_json(ROUTE_DISPOSITION_PATH, payload)
        ROUTE_DISPOSITION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        ROUTE_DISPOSITION_REPORT_PATH.write_text(
            render_markdown(payload),
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "passed": True,
                "row_count": payload["summary"]["row_count"],
                "atlas_disposition_counts": payload["summary"]["atlas_disposition_counts"],
                "bridge_disposition_counts": payload["summary"]["bridge_disposition_counts"],
                "hidden_state_ready_route_count": payload["summary"]["hidden_state_ready_route_count"],
                "validation_check_count": len(payload["validation_checks"]),
                "output_path": rel(ROUTE_DISPOSITION_PATH),
                "report_path": rel(ROUTE_DISPOSITION_REPORT_PATH),
            },
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
