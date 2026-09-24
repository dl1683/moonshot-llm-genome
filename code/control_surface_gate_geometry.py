"""Build the gate-geometry map for control-surface claims.

The reliability matrix says which gates are missing. This layer asks where a
claim first stops in the ordered funnel: behavior substrate, prompt-channel
locality, signature, intervention, or reliability. The point is to make the
shape of the bar itself measurable instead of treating failed controls as
unstructured tombstones.
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
from control_surface_error_taxonomy import ERROR_TAXONOMY_PATH
from control_surface_reliability_matrix import RELIABILITY_MATRIX_PATH
from control_surface_route_disposition import ROUTE_DISPOSITION_PATH


GATE_GEOMETRY_PATH = ROOT / "data" / "control_surface_gate_geometry.json"
GATE_GEOMETRY_REPORT_PATH = ROOT / "research" / "34_CONTROL_SURFACE_GATE_GEOMETRY.md"


TERMINAL_STAGE_ORDER = [
    "pre_signature_behavior_substrate",
    "pre_signature_prompt_channel_locality",
    "signature_output_geometry_shadow",
    "signature_monitor_no_lever",
    "intervention_failed",
    "reliability_null_boundary",
    "promoted_mechanism",
]

TERMINAL_STAGE_DESCRIPTIONS = {
    "pre_signature_behavior_substrate": (
        "The behavior table, bridge substrate, nulls, parseability, or conflict "
        "mixture failed before hidden-state work is allowed."
    ),
    "pre_signature_prompt_channel_locality": (
        "The behavior contrast exists, but a visible prompt channel carries the "
        "rule, so hidden-state mechanism work is not licensed."
    ),
    "signature_output_geometry_shadow": (
        "The behavior table exists, but the hidden signal is output/candidate "
        "geometry or zero-lead shadow under current controls."
    ),
    "signature_monitor_no_lever": (
        "A predecision or monitor-like signal exists, but it has no clean "
        "causal lever or is still candidate/output confounded."
    ),
    "intervention_failed": (
        "A plausible signal or control route reached intervention work, but "
        "the intervention failed controls, locality, or causal criteria."
    ),
    "reliability_null_boundary": (
        "The internal causal surface is bounded by null, locality, side-effect, "
        "or transfer fragility."
    ),
    "promoted_mechanism": "The surface clears signature, intervention, and reliability gates.",
}

ROUTE_STAGE = {
    "closed_before_hidden_state": "pre_signature_behavior_substrate",
    "prompt_visible_positive_control": "pre_signature_prompt_channel_locality",
    "output_shadow_diagnostic_baseline": "signature_output_geometry_shadow",
    "monitor_only_closed": "signature_monitor_no_lever",
    "monitor_only_conditional_revisit": "signature_monitor_no_lever",
    "failed_intervention_or_mechanism_route": "intervention_failed",
    "bounded_mechanism_frozen": "reliability_null_boundary",
}

BRIDGE_STAGE = {
    "bridge_rung_closed_before_hidden_state": "pre_signature_behavior_substrate",
    "prompt_visible_positive_control": "pre_signature_prompt_channel_locality",
}

CLAIM_BAR_ACTION = {
    "pre_signature_behavior_substrate": (
        "Do not probe hidden states; rebuild the behavior table or record a "
        "diagnostic failure."
    ),
    "pre_signature_prompt_channel_locality": (
        "Use as positive control only; remove or match the visible prompt "
        "channel before claiming a hidden surface."
    ),
    "signature_output_geometry_shadow": (
        "Treat hidden AUCs as dashboard readings until output/candidate geometry "
        "is matched or beaten."
    ),
    "signature_monitor_no_lever": (
        "Preserve as monitor-only evidence until an intervention changes "
        "behavior without collateral damage."
    ),
    "intervention_failed": (
        "Kill or redesign the intervention family; do not promote behavior "
        "control as mechanism control."
    ),
    "reliability_null_boundary": (
        "Freeze as bounded reference unless a materially different route repairs "
        "nulls, locality, robustness, and transfer."
    ),
    "promoted_mechanism": "Eligible for promoted mechanism-card use.",
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


def ordered_counts(values: list[str]) -> dict[str, int]:
    counts = Counter(values)
    return {
        stage: counts[stage]
        for stage in TERMINAL_STAGE_ORDER
        if counts.get(stage, 0)
    }


def terminal_stage_for_route(disposition: str) -> str:
    if disposition not in ROUTE_STAGE:
        raise ValueError(f"unknown route disposition for gate geometry: {disposition}")
    return ROUTE_STAGE[disposition]


def terminal_stage_for_bridge(disposition: str) -> str:
    if disposition not in BRIDGE_STAGE:
        raise ValueError(f"unknown bridge disposition for gate geometry: {disposition}")
    return BRIDGE_STAGE[disposition]


def build_gate_entries(
    atlas: dict[str, Any],
    route_disposition: dict[str, Any],
    reliability_matrix: dict[str, Any],
    decision_frontier: dict[str, Any],
) -> list[dict[str, Any]]:
    route_by_row = by_id(route_disposition["route_entries"], "row_id")
    reliability_by_row = by_id(reliability_matrix["reliability_entries"], "row_id")
    frontier_by_row = by_id(decision_frontier["frontier_rows"], "id")

    entries: list[dict[str, Any]] = []
    for row in atlas["rows"]:
        row_id = row["id"]
        route = route_by_row[row_id]
        reliability = reliability_by_row[row_id]
        frontier = frontier_by_row[row_id]
        stage = terminal_stage_for_route(route["disposition"])
        entries.append(
            {
                "row_id": row_id,
                "family": row["family"],
                "terminal_stage": stage,
                "terminal_stage_description": TERMINAL_STAGE_DESCRIPTIONS[stage],
                "claim_bar_action": CLAIM_BAR_ACTION[stage],
                "route_disposition": route["disposition"],
                "reliability_class": reliability["reliability_class"],
                "frontier_class": frontier["frontier_class"],
                "primary_blocker": route["primary_blocker"],
                "verdict": row["verdict"]["class"],
                "lead_time_state": row["lead_time"]["state"],
                "intervention_state": row["intervention"]["state"],
                "gate_statuses": reliability["gate_statuses"],
                "failed_or_missing_gates": reliability["failed_or_missing_gates"],
                "diagnostics": row["diagnostics"],
                "allowed_claims": row["allowed_claims"],
                "forbidden_claims": row["forbidden_claims"],
                "next_decision": row["next_decision"],
                "evidence": row["evidence"],
            }
        )
    return entries


def build_bridge_gate_entries(route_disposition: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for rung in route_disposition["bridge_rung_entries"]:
        stage = terminal_stage_for_bridge(rung["disposition"])
        entries.append(
            {
                "card_id": rung["card_id"],
                "row_id": rung["row_id"],
                "title": rung["title"],
                "terminal_stage": stage,
                "terminal_stage_description": TERMINAL_STAGE_DESCRIPTIONS[stage],
                "claim_bar_action": CLAIM_BAR_ACTION[stage],
                "disposition": rung["disposition"],
                "behavior_ready": rung["behavior_ready"],
                "signature_ready": rung["signature_ready"],
                "hidden_state_allowed": rung["hidden_state_allowed"],
                "local_vs_learned_mixture": rung["local_vs_learned_mixture"],
                "dominant_failure": rung["dominant_failure"],
                "claim_boundary": rung["claim_boundary"],
                "allowed_action": rung["allowed_action"],
                "promotion_rule": rung["promotion_rule"],
                "death_rule": rung["death_rule"],
                "evidence_path": rung["evidence_path"],
            }
        )
    return entries


def build_claim_bar(summary_counts: dict[str, int], row_count: int) -> list[dict[str, Any]]:
    return [
        {
            "stage": stage,
            "count": summary_counts.get(stage, 0),
            "ratio": ratio(summary_counts.get(stage, 0), row_count),
            "description": TERMINAL_STAGE_DESCRIPTIONS[stage],
            "action": CLAIM_BAR_ACTION[stage],
        }
        for stage in TERMINAL_STAGE_ORDER
    ]


def build_validation_checks(
    atlas: dict[str, Any],
    gate_entries: list[dict[str, Any]],
    bridge_gate_entries: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    row_ids = sorted(row["id"] for row in atlas["rows"])
    gate_ids = sorted(entry["row_id"] for entry in gate_entries)
    stage_counts = summary["terminal_stage_counts"]
    bridge_counts = summary["bridge_terminal_stage_counts"]
    checks = [
        {
            "id": "gate_entries_cover_each_atlas_row_once",
            "actual": gate_ids,
            "predicate": "sorted gate rows == sorted atlas rows",
            "passed": gate_ids == row_ids,
            "why": "Every atlas row needs exactly one terminal claim-bar stage.",
        },
        {
            "id": "terminal_stage_counts_sum_to_row_count",
            "actual": {"stage_counts": stage_counts, "row_count": summary["row_count"]},
            "predicate": "sum(stage_counts) == row_count",
            "passed": sum(stage_counts.values()) == summary["row_count"],
            "why": "The row funnel must partition the atlas.",
        },
        {
            "id": "bridge_terminal_stage_counts_sum_to_rung_count",
            "actual": {
                "bridge_counts": bridge_counts,
                "bridge_rung_count": summary["bridge_rung_count"],
            },
            "predicate": "sum(bridge_counts) == bridge_rung_count",
            "passed": sum(bridge_counts.values()) == summary["bridge_rung_count"],
            "why": "The bridge funnel must partition the bridge ladder.",
        },
        {
            "id": "current_atlas_has_no_promoted_mechanism",
            "actual": stage_counts.get("promoted_mechanism", 0),
            "predicate": "== 0",
            "passed": stage_counts.get("promoted_mechanism", 0) == 0,
            "why": "No current row clears signature, intervention, and reliability gates.",
        },
        {
            "id": "pre_signature_blocks_are_majority",
            "actual": summary["ordered_gate_buckets"]["pre_signature_blocked"],
            "predicate": ">= 12",
            "passed": summary["ordered_gate_buckets"]["pre_signature_blocked"] >= 12,
            "why": "Most current claims die before hidden-state work is licensed.",
        },
        {
            "id": "signature_stage_blocks_are_five_rows",
            "actual": summary["ordered_gate_buckets"]["signature_stage_blocked"],
            "predicate": "== 5",
            "passed": summary["ordered_gate_buckets"]["signature_stage_blocked"] == 5,
            "why": "The current signature-stage residue is exactly three output shadows plus two monitor-only rows.",
        },
        {
            "id": "mc005_is_reliability_boundary",
            "actual": [
                entry["row_id"]
                for entry in gate_entries
                if entry["terminal_stage"] == "reliability_null_boundary"
            ],
            "predicate": "== ['mc005_associative_lookup']",
            "passed": [
                entry["row_id"]
                for entry in gate_entries
                if entry["terminal_stage"] == "reliability_null_boundary"
            ]
            == ["mc005_associative_lookup"],
            "why": "MC005 is the only current internal-causal surface, bounded by null/reliability issues.",
        },
        {
            "id": "mc012_is_prompt_channel_boundary",
            "actual": [
                entry["row_id"]
                for entry in gate_entries
                if entry["terminal_stage"] == "pre_signature_prompt_channel_locality"
            ],
            "predicate": "== ['mc012_reliability_labeled_numeric_arbitration']",
            "passed": [
                entry["row_id"]
                for entry in gate_entries
                if entry["terminal_stage"] == "pre_signature_prompt_channel_locality"
            ]
            == ["mc012_reliability_labeled_numeric_arbitration"],
            "why": "MC012 is the one behavior-ready bridge blocked by visible source-status text.",
        },
        {
            "id": "mc001g_is_intervention_failed_boundary",
            "actual": [
                entry["row_id"]
                for entry in gate_entries
                if entry["terminal_stage"] == "intervention_failed"
            ],
            "predicate": "== ['mc001g_gemma_truth_agreement']",
            "passed": [
                entry["row_id"]
                for entry in gate_entries
                if entry["terminal_stage"] == "intervention_failed"
            ]
            == ["mc001g_gemma_truth_agreement"],
            "why": "The Gemma truth/agreement route is the current failed-intervention specimen.",
        },
        {
            "id": "bridge_ladder_has_no_signature_or_intervention_stage",
            "actual": bridge_counts,
            "predicate": "only pre-signature stages are present",
            "passed": set(bridge_counts).issubset(
                {
                    "pre_signature_behavior_substrate",
                    "pre_signature_prompt_channel_locality",
                }
            ),
            "why": "The bridge program remains behavior-substrate work; hidden-state work is still forbidden.",
        },
    ]
    return checks


def build_control_surface_gate_geometry() -> dict[str, Any]:
    atlas = load_json(ATLAS_PATH)
    route_disposition = load_json(ROUTE_DISPOSITION_PATH)
    reliability_matrix = load_json(RELIABILITY_MATRIX_PATH)
    decision_frontier = load_json(DECISION_FRONTIER_PATH)
    bridge_ladder = load_json(BRIDGE_LADDER_PATH)
    error_taxonomy = load_json(ERROR_TAXONOMY_PATH)

    gate_entries = build_gate_entries(
        atlas,
        route_disposition,
        reliability_matrix,
        decision_frontier,
    )
    bridge_gate_entries = build_bridge_gate_entries(route_disposition)
    row_count = len(gate_entries)
    bridge_rung_count = len(bridge_gate_entries)
    stage_counts = ordered_counts([entry["terminal_stage"] for entry in gate_entries])
    bridge_stage_counts = ordered_counts(
        [entry["terminal_stage"] for entry in bridge_gate_entries]
    )
    ordered_buckets = {
        "pre_signature_blocked": stage_counts.get(
            "pre_signature_behavior_substrate",
            0,
        )
        + stage_counts.get("pre_signature_prompt_channel_locality", 0),
        "signature_stage_blocked": stage_counts.get(
            "signature_output_geometry_shadow",
            0,
        )
        + stage_counts.get("signature_monitor_no_lever", 0),
        "intervention_stage_blocked": stage_counts.get("intervention_failed", 0),
        "reliability_stage_bounded": stage_counts.get("reliability_null_boundary", 0),
        "promoted": stage_counts.get("promoted_mechanism", 0),
    }
    summary = {
        "row_count": row_count,
        "bridge_rung_count": bridge_rung_count,
        "terminal_stage_counts": stage_counts,
        "terminal_stage_ratios": {
            stage: ratio(count, row_count) for stage, count in stage_counts.items()
        },
        "bridge_terminal_stage_counts": bridge_stage_counts,
        "bridge_terminal_stage_ratios": {
            stage: ratio(count, bridge_rung_count)
            for stage, count in bridge_stage_counts.items()
        },
        "ordered_gate_buckets": ordered_buckets,
        "ordered_gate_bucket_ratios": {
            bucket: ratio(count, row_count) for bucket, count in ordered_buckets.items()
        },
        "claim_bar": build_claim_bar(stage_counts, row_count),
        "bridge_hidden_state_allowed_count": route_disposition["summary"][
            "bridge_hidden_state_allowed_count"
        ],
        "bridge_clean_unconfounded_count": bridge_ladder["summary"][
            "clean_unconfounded_bridge_count"
        ],
        "error_taxonomy_bridge_rung_count": error_taxonomy["summary"][
            "bridge_rung_count"
        ],
        "error_taxonomy_smoke_card_count": error_taxonomy["summary"][
            "smoke_card_count"
        ],
        "mc028_other_number_count": error_taxonomy["summary"].get(
            "mc028_other_number_count",
        ),
        "promoted_mechanism_count": stage_counts.get("promoted_mechanism", 0),
    }
    checks = build_validation_checks(atlas, gate_entries, bridge_gate_entries, summary)
    return {
        "schema_version": 1,
        "updated_at": atlas.get("updated_at"),
        "purpose": (
            "Map the ordered geometry of the mechanism-card bar. This artifact "
            "shows where atlas rows and bridge rungs first stop: behavior "
            "substrate, prompt-channel locality, output-shadow signature, "
            "monitor-only signature, failed intervention, reliability boundary, "
            "or promoted mechanism."
        ),
        "sources": {
            "atlas": rel(ATLAS_PATH),
            "route_disposition": rel(ROUTE_DISPOSITION_PATH),
            "reliability_matrix": rel(RELIABILITY_MATRIX_PATH),
            "decision_frontier": rel(DECISION_FRONTIER_PATH),
            "bridge_ladder": rel(BRIDGE_LADDER_PATH),
            "error_taxonomy": rel(ERROR_TAXONOMY_PATH),
        },
        "terminal_stage_order": TERMINAL_STAGE_ORDER,
        "terminal_stage_descriptions": TERMINAL_STAGE_DESCRIPTIONS,
        "claim_bar_actions": CLAIM_BAR_ACTION,
        "summary": summary,
        "gate_entries": gate_entries,
        "bridge_gate_entries": bridge_gate_entries,
        "validation_checks": checks,
        "allowed_claim": (
            "The current atlas has a measured claim-bar geometry: most rows die "
            "before signature work, the signature-stage residue is output-shadow "
            "or monitor-only, one route failed intervention, and MC005 is the "
            "only bounded reliability specimen."
        ),
        "forbidden_claim": (
            "This funnel does not prove that the current bar is the final bar or "
            "that killed routes are false mechanisms in every prompt contract. "
            "It says where current evidence stops under the current controls."
        ),
    }


def validate_gate_geometry(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("gate geometry schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"gate geometry source missing: {rel_path}")
    if payload.get("terminal_stage_order") != TERMINAL_STAGE_ORDER:
        raise AssertionError("gate geometry terminal stage order drifted")
    row_ids = [entry["row_id"] for entry in payload.get("gate_entries", [])]
    if len(row_ids) != len(set(row_ids)):
        raise AssertionError("gate geometry has duplicate atlas row entries")
    stage_counts = payload["summary"]["terminal_stage_counts"]
    if sum(stage_counts.values()) != payload["summary"]["row_count"]:
        raise AssertionError("gate geometry terminal stages do not partition rows")
    if payload["summary"]["promoted_mechanism_count"] != 0:
        raise AssertionError("gate geometry must not report promoted mechanisms")
    if payload["summary"]["bridge_hidden_state_allowed_count"] != 0:
        raise AssertionError("gate geometry bridge rungs must not allow hidden-state work")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(f"gate geometry checks failed: {failed_checks}")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Control-Surface Gate Geometry",
        "",
        "Date: 2026-07-01",
        "",
        "Status: generated gate-geometry layer implemented and validated.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/control_surface_gate_geometry.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_gate_geometry.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_gate_geometry.py --write",
        "python code\\control_surface_gate_geometry.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Purpose",
        "",
        "The gate-geometry layer makes the shape of the mechanism-card bar a",
        "first-class artifact. A failed control is not just a tombstone; it is",
        "evidence about where current behavior families stop under the current",
        "controls.",
        "",
        "The ordered funnel is:",
        "",
        "1. behavior substrate;",
        "2. prompt-channel locality;",
        "3. signature not explained by output/candidate geometry;",
        "4. monitor-to-lever transition;",
        "5. predicted intervention;",
        "6. reliability, nulls, locality, side effects, and transfer;",
        "7. promoted mechanism.",
        "",
        "## Generated Facts",
        "",
        f"- atlas rows: {summary['row_count']};",
        f"- bridge rungs: {summary['bridge_rung_count']};",
        f"- promoted mechanisms: {summary['promoted_mechanism_count']};",
        f"- bridge rungs allowing hidden-state work: {summary['bridge_hidden_state_allowed_count']};",
        f"- clean unconfounded bridge rungs: {summary['bridge_clean_unconfounded_count']};",
        f"- MC028 other-number rows in taxonomy: {summary['mc028_other_number_count']}.",
        "",
        "## Row Funnel",
        "",
        "| Terminal Stage | Count | Ratio | Meaning |",
        "| --- | ---: | ---: | --- |",
    ]
    for item in summary["claim_bar"]:
        lines.append(
            f"| `{item['stage']}` | {item['count']} | {format_value(item['ratio'])} | "
            f"{item['description']} |"
        )

    lines.extend(
        [
            "",
            "## Ordered Buckets",
            "",
            "| Bucket | Count | Ratio |",
            "| --- | ---: | ---: |",
        ]
    )
    for bucket, count in summary["ordered_gate_buckets"].items():
        lines.append(
            f"| `{bucket}` | {count} | "
            f"{format_value(summary['ordered_gate_bucket_ratios'][bucket])} |"
        )

    lines.extend(
        [
            "",
            "## Row Ledger",
            "",
            "| Row | Terminal Stage | Route | Frontier | Action |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for entry in payload["gate_entries"]:
        lines.append(
            f"| `{entry['row_id']}` | `{entry['terminal_stage']}` | "
            f"`{entry['route_disposition']}` | `{entry['frontier_class']}` | "
            f"{entry['claim_bar_action']} |"
        )

    lines.extend(
        [
            "",
            "## Bridge Funnel",
            "",
            "| Terminal Stage | Count | Ratio |",
            "| --- | ---: | ---: |",
        ]
    )
    for stage, count in summary["bridge_terminal_stage_counts"].items():
        lines.append(
            f"| `{stage}` | {count} | "
            f"{format_value(summary['bridge_terminal_stage_ratios'][stage])} |"
        )

    lines.extend(
        [
            "",
            "## Bridge Ledger",
            "",
            "| Card | Terminal Stage | Mixture | Dominant Failure |",
            "| --- | --- | --- | --- |",
        ]
    )
    for entry in payload["bridge_gate_entries"]:
        lines.append(
            f"| `{entry['card_id']}` | `{entry['terminal_stage']}` | "
            f"`{entry['local_vs_learned_mixture']}` | {entry['dominant_failure']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The current project does not merely have many failed mechanism-card",
            "routes. It has a measured distribution of where those routes fail.",
            "That distribution is now part of the genome map.",
            "",
            "The dominant fact is pre-signature closure. Twelve of 19 atlas rows",
            "stop before hidden-state mechanism work is licensed: 11 at behavior",
            "or bridge-substrate quality and MC012 at visible prompt-channel",
            "locality. Five rows stop at the signature stage: three are",
            "output-geometry shadows and two are monitor-only no-lever rows.",
            "One route has a failed intervention, and MC005 is the only bounded",
            "internal-causal reliability specimen.",
            "",
            "This is the concrete answer to asymptotic conservatism: the bar is",
            "not merely getting stricter in prose. Its failure geometry is now",
            "measured, versioned, and validated.",
            "",
            "## What This Proves",
            "",
            "It proves that the current atlas has an ordered claim-bar geometry",
            "derived from validated route, reliability, decision-frontier, bridge,",
            "and taxonomy artifacts.",
            "",
            "It proves that killed controls are structured data: most deaths are",
            "pre-signature, the signature-stage residue is output-shadow or",
            "monitor-only, intervention failure is currently one row, and the",
            "sole internal-causal result is bounded rather than promoted.",
            "",
            "## What It Does Not Prove",
            "",
            "It does not prove that the current bar is final, optimal, or fair for",
            "every behavior family. It documents the bar that current claims have",
            "actually faced.",
            "",
            "It does not prove that a route killed under one prompt contract is",
            "impossible under a materially different contract. It does require",
            "that such a route be reopened explicitly rather than smuggled back in",
            "as a near-duplicate repair.",
            "",
            "## Next Use",
            "",
            "After each future result, update this layer and ask:",
            "",
            "> Did the new result move a row to a later terminal stage, or did it",
            "> only add another example to an already measured death bucket?",
            "",
            "That question keeps the project aimed at the genome law rather than",
            "at isolated attractive results.",
            "",
        ]
    )
    return "\n".join(lines)


def write_gate_geometry(
    output_path: Path = GATE_GEOMETRY_PATH,
    report_path: Path = GATE_GEOMETRY_REPORT_PATH,
) -> dict[str, Any]:
    payload = build_control_surface_gate_geometry()
    validate_gate_geometry(payload)
    write_json(output_path, payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(payload), encoding="utf-8", newline="\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write gate-geometry artifacts")
    parser.add_argument("--json", action="store_true", help="print gate-geometry JSON")
    args = parser.parse_args()

    payload = build_control_surface_gate_geometry()
    validate_gate_geometry(payload)

    if args.write:
        write_gate_geometry()
        print(
            f"wrote {GATE_GEOMETRY_PATH.relative_to(ROOT).as_posix()} and "
            f"{GATE_GEOMETRY_REPORT_PATH.relative_to(ROOT).as_posix()} "
            f"with {payload['summary']['row_count']} rows"
        )
        return
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"gate geometry ok: {payload['summary']['row_count']} rows")
    print(
        "terminal_stage_counts:",
        json.dumps(payload["summary"]["terminal_stage_counts"], sort_keys=True),
    )
    print(
        "ordered_gate_buckets:",
        json.dumps(payload["summary"]["ordered_gate_buckets"], sort_keys=True),
    )
    print(
        "bridge_terminal_stage_counts:",
        json.dumps(
            payload["summary"]["bridge_terminal_stage_counts"],
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
