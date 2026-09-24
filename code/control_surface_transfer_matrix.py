"""Build the current transfer and widening matrix.

The genome map is not complete until it says which surfaces generalize. This
layer classifies each atlas row by transfer status and records what kind of
widening test, if any, is legitimate from the current evidence.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from control_surface_artifacts import ATLAS_PATH, ROOT, load_json
from control_surface_comparison import COMPARISON_PATH
from control_surface_next_queue import NEXT_QUEUE_PATH
from control_surface_route_disposition import ROUTE_DISPOSITION_PATH


TRANSFER_MATRIX_PATH = ROOT / "data" / "control_surface_transfer_matrix.json"
TRANSFER_MATRIX_REPORT_PATH = ROOT / "research" / "31_CONTROL_SURFACE_TRANSFER_MATRIX.md"


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


def transfer_class(row: dict[str, Any], route_entry: dict[str, Any]) -> str:
    transfer = row["mixture_profile"]["transfer"]
    disposition = route_entry["disposition"]
    diagnostics = set(row["diagnostics"])
    if row["verdict"]["class"] == "promoted_mechanism_card":
        return "transfer_ready_mechanism"
    if row["id"] == "mc005_associative_lookup":
        return "bounded_transfer_fragile_reference"
    if transfer == "failed" or diagnostics & {
        "LOCKED_COORDINATE_TRANSFER_FAILED",
        "EXPANDED_CANDIDATE_DECOUPLED_BANK_INSUFFICIENT",
        "TRANSFER_ROLE_REPAIR_BANK_INSUFFICIENT",
        "DELAYED_CITY_ROUTE_CLOSED_MONITOR_ONLY",
    }:
        return "transfer_failed_or_bank_insufficient"
    if disposition == "prompt_visible_positive_control":
        return "prompt_visible_no_transfer_claim"
    if transfer in {"low", "medium"} and row["verdict"]["class"] != "bounded_mechanism_card":
        return "diagnostic_cross_model_evidence_only"
    if disposition in {
        "closed_before_hidden_state",
        "failed_intervention_or_mechanism_route",
        "monitor_only_closed",
    }:
        return "closed_route_no_widening"
    if disposition == "monitor_only_conditional_revisit":
        return "conditional_widening_requires_new_controls"
    if disposition == "output_shadow_diagnostic_baseline":
        return "output_shadow_widening_baseline"
    return "transfer_untested_no_claim"


TRANSFER_POLICIES = {
    "transfer_ready_mechanism": {
        "widening_action": "Run preregistered cross-model null/locality/side-effect transfer.",
        "required_gate": "Already promoted locally; transfer must preserve mechanism gates.",
    },
    "bounded_transfer_fragile_reference": {
        "widening_action": "Use as reference specimen only; any widening test must match null panels and side rows before celebrating primary effect.",
        "required_gate": "Primary effect plus null locality, source-disjoint holdout, side effects, and model-size robustness.",
    },
    "transfer_failed_or_bank_insufficient": {
        "widening_action": "Do not continue this transfer route as promotion; reopen only with materially different behavior family or transfer-bank construction.",
        "required_gate": "At least two transfer-ready templates or cross-model rows with candidate/output controls and shuffle/null gates.",
    },
    "prompt_visible_no_transfer_claim": {
        "widening_action": "Do not widen as mechanism; first remove or match the prompt-visible answer channel.",
        "required_gate": "Prompt-channel locality before transfer.",
    },
    "diagnostic_cross_model_evidence_only": {
        "widening_action": "Use as cross-model confound evidence, not transfer success.",
        "required_gate": "A new substrate must beat output/prompt/shuffle controls before transfer matters.",
    },
    "closed_route_no_widening": {
        "widening_action": "Do not widen; repair behavior substrate first or keep the route closed.",
        "required_gate": "Behavior, parseability, nulls, source-disjoint holdout, and baseline gates.",
    },
    "conditional_widening_requires_new_controls": {
        "widening_action": "Widen only after larger source-disjoint controls or a materially different causal stress test.",
        "required_gate": "Lead-time monitor must survive subgroup, shuffle, output/candidate, and intervention checks.",
    },
    "output_shadow_widening_baseline": {
        "widening_action": "Use as output-geometry widening baseline; do not call transfer until hidden effects beat output/candidate controls.",
        "required_gate": "Hidden signature and intervention beat matched output/candidate baselines on holdout.",
    },
    "transfer_untested_no_claim": {
        "widening_action": "No transfer claim; define a local substrate before widening.",
        "required_gate": "Local behavior and mechanism gates first.",
    },
}


def build_transfer_entries(
    atlas: dict[str, Any],
    route_disposition: dict[str, Any],
) -> list[dict[str, Any]]:
    route_by_row = by_id(route_disposition["route_entries"], "row_id")
    entries = []
    for row in atlas["rows"]:
        route = route_by_row[row["id"]]
        klass = transfer_class(row, route)
        policy = TRANSFER_POLICIES[klass]
        entries.append(
            {
                "row_id": row["id"],
                "family": row["family"],
                "models": row["models"],
                "model_count": len(row["models"]),
                "transfer_value": row["mixture_profile"]["transfer"],
                "transfer_class": klass,
                "widening_action": policy["widening_action"],
                "required_gate": policy["required_gate"],
                "route_disposition": route["disposition"],
                "verdict": row["verdict"]["class"],
                "intervention_state": row["intervention"]["state"],
                "causal_control": row["mixture_profile"]["causal_control"],
                "null_locality": row["mixture_profile"]["null_locality"],
                "diagnostics": row["diagnostics"],
                "next_decision": row["next_decision"],
                "evidence": row["evidence"],
            }
        )
    return entries


def transfer_queue_items(next_queue: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": item["id"],
            "priority_class": item["priority_class"],
            "priority_score": item["priority_score"],
            "hypothesis_id": item["hypothesis_id"],
            "next_test": item["next_test"],
            "reason_codes": item["reason_codes"],
        }
        for item in next_queue["queue"]
        if "TRANSFER_GAP" in item["reason_codes"]
    ]


def build_validation_checks(
    atlas: dict[str, Any],
    comparison: dict[str, Any],
    entries: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    row_ids = sorted(row["id"] for row in atlas["rows"])
    entry_ids = sorted(entry["row_id"] for entry in entries)
    transfer_counts = comparison["mixture_axis_counts"]["transfer"]["counts"]
    class_counts = summary["transfer_class_counts"]
    value_counts = summary["transfer_value_counts"]
    checks = [
        {
            "id": "transfer_entries_cover_each_atlas_row_once",
            "actual": entry_ids,
            "predicate": "sorted transfer rows == sorted atlas rows",
            "passed": entry_ids == row_ids,
            "why": "Every atlas row needs a transfer disposition.",
        },
        {
            "id": "transfer_values_match_comparison_axis_counts",
            "actual": {
                "matrix": value_counts,
                "comparison": transfer_counts,
            },
            "predicate": "matrix transfer_value_counts == comparison transfer counts",
            "passed": value_counts == transfer_counts,
            "why": "The matrix must be a derived view of the canonical atlas transfer axis.",
        },
        {
            "id": "no_transfer_ready_mechanism",
            "actual": class_counts.get("transfer_ready_mechanism", 0),
            "predicate": "== 0",
            "passed": class_counts.get("transfer_ready_mechanism", 0) == 0,
            "why": "No current row is a promoted mechanism with transfer evidence.",
        },
        {
            "id": "mc005_is_bounded_transfer_fragile_reference",
            "actual": [
                entry["transfer_class"]
                for entry in entries
                if entry["row_id"] == "mc005_associative_lookup"
            ],
            "predicate": "== ['bounded_transfer_fragile_reference']",
            "passed": [
                entry["transfer_class"]
                for entry in entries
                if entry["row_id"] == "mc005_associative_lookup"
            ]
            == ["bounded_transfer_fragile_reference"],
            "why": "MC005 has the strongest surface but is not transfer-clean due null/model-size fragility.",
        },
        {
            "id": "mc006_transfer_route_failed",
            "actual": [
                entry["transfer_class"]
                for entry in entries
                if entry["row_id"] == "mc006_parametric_fact_override"
            ],
            "predicate": "== ['transfer_failed_or_bank_insufficient']",
            "passed": [
                entry["transfer_class"]
                for entry in entries
                if entry["row_id"] == "mc006_parametric_fact_override"
            ]
            == ["transfer_failed_or_bank_insufficient"],
            "why": "MC006 transfer failed through locked-coordinate, expanded-bank, and transfer-role repair routes.",
        },
        {
            "id": "untested_transfer_is_majority",
            "actual": value_counts.get("untested", 0),
            "predicate": "untested >= 10",
            "passed": value_counts.get("untested", 0) >= 10,
            "why": "Most current rows are not widened yet.",
        },
        {
            "id": "transfer_gap_queue_items_exist",
            "actual": summary["transfer_gap_queue_item_count"],
            "predicate": ">= 2",
            "passed": summary["transfer_gap_queue_item_count"] >= 2,
            "why": "The next queue should preserve transfer/null widening pressure.",
        },
    ]
    return checks


def build_control_surface_transfer_matrix() -> dict[str, Any]:
    atlas = load_json(ATLAS_PATH)
    comparison = load_json(COMPARISON_PATH)
    route_disposition = load_json(ROUTE_DISPOSITION_PATH)
    next_queue = load_json(NEXT_QUEUE_PATH)
    entries = build_transfer_entries(atlas, route_disposition)
    queue_items = transfer_queue_items(next_queue)
    class_counts = dict(sorted(Counter(entry["transfer_class"] for entry in entries).items()))
    value_counts = dict(sorted(Counter(entry["transfer_value"] for entry in entries).items()))
    row_count = len(entries)
    summary = {
        "row_count": row_count,
        "transfer_class_counts": class_counts,
        "transfer_class_ratios": {
            key: ratio(value, row_count) for key, value in class_counts.items()
        },
        "transfer_value_counts": value_counts,
        "transfer_value_ratios": {
            key: ratio(value, row_count) for key, value in value_counts.items()
        },
        "transfer_ready_mechanism_count": class_counts.get("transfer_ready_mechanism", 0),
        "transfer_gap_queue_item_count": len(queue_items),
        "transfer_gap_top_queue_ids": [item["id"] for item in queue_items[:5]],
        "multi_model_row_count": sum(1 for entry in entries if entry["model_count"] > 1),
    }
    checks = build_validation_checks(atlas, comparison, entries, summary)
    return {
        "schema_version": 1,
        "updated_at": atlas.get("updated_at"),
        "purpose": (
            "Classify transfer and widening status for each atlas row, separating "
            "bounded reference evidence, failed transfer routes, diagnostic cross-model "
            "evidence, closed routes, and rows with no transfer claim."
        ),
        "sources": {
            "atlas": rel(ATLAS_PATH),
            "comparison": rel(COMPARISON_PATH),
            "route_disposition": rel(ROUTE_DISPOSITION_PATH),
            "next_queue": rel(NEXT_QUEUE_PATH),
        },
        "transfer_policies": TRANSFER_POLICIES,
        "summary": summary,
        "transfer_entries": entries,
        "transfer_gap_queue_items": queue_items,
        "validation_checks": checks,
        "allowed_claim": (
            "The current atlas has no transfer-ready mechanism. MC005 is a bounded "
            "transfer-fragile reference specimen, MC006 transfer is failed or bank-"
            "insufficient, and most rows are untested or closed before widening."
        ),
        "forbidden_claim": (
            "Medium or cross-model evidence in this matrix is not a transfer success "
            "unless signature, intervention, null, locality, side-effect, and output/"
            "candidate controls survive on the widened target."
        ),
    }


def validate_transfer_matrix(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("transfer matrix schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"transfer matrix source missing: {rel_path}")
    row_ids = [entry["row_id"] for entry in payload.get("transfer_entries", [])]
    if len(row_ids) != len(set(row_ids)):
        raise AssertionError("transfer matrix has duplicate atlas rows")
    if payload["summary"]["transfer_ready_mechanism_count"] != 0:
        raise AssertionError("transfer matrix must not report transfer-ready mechanisms")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(f"transfer matrix checks failed: {failed_checks}")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Control-Surface Transfer Matrix",
        "",
        "Date: 2026-07-01",
        "",
        "Status: generated transfer/widening matrix implemented and validated.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/control_surface_transfer_matrix.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_transfer_matrix.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_transfer_matrix.py --write",
        "python code\\control_surface_transfer_matrix.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Purpose",
        "",
        "The transfer matrix records which control surfaces are local, fragile,",
        "failed under transfer, or not yet eligible for widening. It prevents",
        "cross-model anecdotes from becoming transfer claims.",
        "",
        "## Generated Facts",
        "",
        f"- atlas rows: {summary['row_count']};",
        f"- transfer-ready mechanisms: {summary['transfer_ready_mechanism_count']};",
        f"- multi-model rows: {summary['multi_model_row_count']};",
        f"- transfer-gap queue items: {summary['transfer_gap_queue_item_count']}.",
        "",
        "## Transfer Classes",
        "",
        "| Transfer Class | Count | Ratio |",
        "| --- | ---: | ---: |",
    ]
    for klass, count in summary["transfer_class_counts"].items():
        lines.append(
            f"| `{klass}` | {count} | {format_value(summary['transfer_class_ratios'][klass])} |"
        )

    lines.extend(
        [
            "",
            "## Row Matrix",
            "",
            "| Row | Transfer Value | Transfer Class | Route Disposition | Required Gate |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for entry in payload["transfer_entries"]:
        lines.append(
            f"| `{entry['row_id']}` | `{entry['transfer_value']}` | "
            f"`{entry['transfer_class']}` | `{entry['route_disposition']}` | "
            f"{entry['required_gate']} |"
        )

    lines.extend(
        [
            "",
            "## Transfer-Gap Queue Items",
            "",
            "| Queue Item | Priority | Hypothesis | Next Test |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in payload["transfer_gap_queue_items"]:
        lines.append(
            f"| `{item['id']}` | `{item['priority_class']}` | "
            f"`{item['hypothesis_id']}` | {item['next_test']} |"
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
            "The current transfer picture is mostly absence, fragility, or failure.",
            "MC005 is useful as the bounded reference specimen, but its transfer",
            "story is explicitly null/model-size fragile. MC006 is the negative",
            "transfer exemplar: locked-coordinate, expanded-bank, and transfer-role",
            "repair routes are not enough. Most bridge rows are closed before",
            "transfer is even a meaningful question.",
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

    payload = build_control_surface_transfer_matrix()
    validate_transfer_matrix(payload)
    if args.write:
        write_json(TRANSFER_MATRIX_PATH, payload)
        TRANSFER_MATRIX_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        TRANSFER_MATRIX_REPORT_PATH.write_text(
            render_markdown(payload),
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "passed": True,
                "row_count": payload["summary"]["row_count"],
                "transfer_class_counts": payload["summary"]["transfer_class_counts"],
                "transfer_value_counts": payload["summary"]["transfer_value_counts"],
                "transfer_ready_mechanism_count": payload["summary"]["transfer_ready_mechanism_count"],
                "validation_check_count": len(payload["validation_checks"]),
                "output_path": rel(TRANSFER_MATRIX_PATH),
                "report_path": rel(TRANSFER_MATRIX_REPORT_PATH),
            },
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
