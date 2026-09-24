"""Build the offensive-doctrine harness for future control-surface work.

The gap-closure plan says what to do next. This layer turns that plan into a
branch-intake contract: every future line must name the gap it targets, the
generated layer it expects to change, and the rule that kills it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from control_surface_artifacts import ROOT, load_json


OFFENSIVE_DOCTRINE_PATH = ROOT / "data" / "control_surface_offensive_doctrine.json"
OFFENSIVE_DOCTRINE_REPORT_PATH = (
    ROOT / "research" / "39_CONTROL_SURFACE_OFFENSIVE_DOCTRINE.md"
)

COVERAGE_GAPS_PATH = ROOT / "data" / "control_surface_coverage_gaps.json"
GAP_CLOSURE_PLAN_PATH = ROOT / "data" / "control_surface_gap_closure_plan.json"
GENOME_SNAPSHOT_PATH = ROOT / "data" / "control_surface_genome_snapshot.json"
NEXT_QUEUE_PATH = ROOT / "data" / "control_surface_next_experiment_queue.json"

REQUIRED_DECISION_FIELDS = {
    "promotion_rule",
    "bound_rule",
    "kill_rule",
    "containment_rule",
    "export_rule",
}

REQUIRED_BRANCH_FIELDS = [
    "target_gap_ids",
    "expected_generated_layer_change",
    "promotion_rule",
    "bound_rule",
    "kill_rule",
    "containment_rule",
    "export_rule",
    "first_artifact",
    "iteration_budget",
    "forbidden_moves",
    "minimum_evidence_packet",
    "allowed_outputs",
]

OUTCOME_CLASSES = [
    "promoted_mechanism_card",
    "bounded_mechanism_card",
    "failed_mechanism_card",
    "diagnostic_note",
]

TRACK_LAYER_TARGETS = {
    "bridge_substrate": [
        "data/control_surface_bridge_ladder.json",
        "data/control_surface_smoke_diagnostics.json",
        "data/control_surface_gate_geometry.json",
        "data/control_surface_coverage_gaps.json",
    ],
    "deepening_closeout": [
        "data/control_surface_atlas.json",
        "data/control_surface_reliability_matrix.json",
        "data/control_surface_route_disposition.json",
        "data/control_surface_gate_geometry.json",
    ],
    "knowledge_frontier_closeout": [
        "data/control_surface_decision_frontier.json",
        "data/control_surface_route_disposition.json",
        "data/control_surface_coverage_gaps.json",
        "data/control_surface_error_taxonomy.json",
    ],
    "law_replication": [
        "data/control_surface_axis_interactions.json",
        "data/control_surface_law_hypotheses.json",
        "data/control_surface_law_audit.json",
        "data/control_surface_coverage_gaps.json",
    ],
    "process_harness": [
        "data/control_surface_offensive_doctrine.json",
        "research/39_CONTROL_SURFACE_OFFENSIVE_DOCTRINE.md",
        "code/validate_control_surface_atlas.py",
    ],
    "widening_probe": [
        "data/control_surface_transfer_matrix.json",
        "data/control_surface_reliability_matrix.json",
        "data/control_surface_coverage_gaps.json",
        "data/control_surface_genome_snapshot.json",
    ],
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def compact_queue_ref(ref: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": ref["id"],
        "priority_class": ref["priority_class"],
        "priority_score": ref["priority_score"],
        "hypothesis_id": ref["hypothesis_id"],
        "action_type": ref["action_type"],
        "next_test": ref["next_test"],
    }


def branch_contract_from_order(order: dict[str, Any]) -> dict[str, Any]:
    target_gap_ids = order["primary_gap_ids"] + order["secondary_gap_ids"]
    return {
        "work_order_id": order["id"],
        "title": order["title"],
        "urgency": order["urgency"],
        "track_type": order["track_type"],
        "target_gap_ids": target_gap_ids,
        "source_rows": order["source_rows"],
        "expected_generated_layer_change": TRACK_LAYER_TARGETS[order["track_type"]],
        "first_artifact": order["first_artifact"],
        "iteration_budget": order["iteration_budget"],
        "minimum_evidence_packet": order["required_evidence"],
        "promotion_rule": order["decision_rules"]["promotion_rule"],
        "bound_rule": order["decision_rules"]["bound_rule"],
        "kill_rule": order["decision_rules"]["kill_rule"],
        "containment_rule": order["decision_rules"]["containment_rule"],
        "export_rule": order["decision_rules"]["export_rule"],
        "forbidden_moves": order["forbidden_moves"],
        "queue_refs": [compact_queue_ref(ref) for ref in order["queue_refs"]],
        "allowed_outputs": OUTCOME_CLASSES,
        "must_change_map_geometry": (
            "A result must change at least one listed generated layer, named "
            "coverage gap, atlas field, reliability class, transfer class, "
            "gate-geometry cell, or exported diagnostic."
        ),
    }


def build_validation_checks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    contracts = payload["branch_contracts"]
    current_snapshot = payload["source_snapshot"]
    missing_contract_fields = {
        contract["work_order_id"]: [
            field
            for field in REQUIRED_BRANCH_FIELDS
            if not contract.get(field)
        ]
        for contract in contracts
    }
    missing_contract_fields = {
        key: value for key, value in missing_contract_fields.items() if value
    }
    decision_rule_failures = [
        contract["work_order_id"]
        for contract in contracts
        if not all(contract.get(field) for field in REQUIRED_DECISION_FIELDS)
    ]
    missing_layer_targets = [
        contract["work_order_id"]
        for contract in contracts
        if not contract["expected_generated_layer_change"]
    ]
    no_kill_rule = [
        contract["work_order_id"]
        for contract in contracts
        if not contract.get("kill_rule")
    ]
    branch_ids = [contract["work_order_id"] for contract in contracts]
    repeated_ids = sorted(
        work_id for work_id, count in Counter(branch_ids).items() if count > 1
    )
    covered_critical_gap_ids = sorted(
        {
            gap_id
            for contract in contracts
            for gap_id in contract["target_gap_ids"]
            if gap_id in payload["critical_gap_ids"]
        }
    )
    missing_critical_gap_ids = sorted(
        set(payload["critical_gap_ids"]) - set(covered_critical_gap_ids)
    )
    return [
        {
            "id": "all_work_orders_have_branch_contracts",
            "predicate": "contract_count == source_work_order_count",
            "actual": {
                "contract_count": len(contracts),
                "source_work_order_count": payload["summary"]["source_work_order_count"],
            },
            "passed": len(contracts) == payload["summary"]["source_work_order_count"],
            "why": "The harness must cover every currently planned branch.",
        },
        {
            "id": "branch_contracts_have_required_fields",
            "predicate": "empty dict",
            "actual": missing_contract_fields,
            "passed": not missing_contract_fields,
            "why": "Every branch needs enough information to fail fast.",
        },
        {
            "id": "branch_contracts_have_all_decision_rules",
            "predicate": "empty list",
            "actual": decision_rule_failures,
            "passed": not decision_rule_failures,
            "why": "Promotion, bound, kill, containment, and export rules must be explicit.",
        },
        {
            "id": "branch_contracts_have_kill_rules",
            "predicate": "empty list",
            "actual": no_kill_rule,
            "passed": not no_kill_rule,
            "why": "The harness exists to prevent half-alive branches.",
        },
        {
            "id": "branch_contracts_name_generated_layer_changes",
            "predicate": "empty list",
            "actual": missing_layer_targets,
            "passed": not missing_layer_targets,
            "why": "Future work must state which generated map geometry it can move.",
        },
        {
            "id": "branch_contract_ids_are_unique",
            "predicate": "empty list",
            "actual": repeated_ids,
            "passed": not repeated_ids,
            "why": "Each active branch contract must be addressable by a stable id.",
        },
        {
            "id": "critical_gaps_have_active_contracts",
            "predicate": "empty missing_critical_gap_ids",
            "actual": {
                "critical_gap_ids": payload["critical_gap_ids"],
                "covered_critical_gap_ids": covered_critical_gap_ids,
                "missing_critical_gap_ids": missing_critical_gap_ids,
            },
            "passed": not missing_critical_gap_ids,
            "why": "The four largest missing claims must be directly guarded.",
        },
        {
            "id": "harness_preserves_no_promotion_claim",
            "predicate": "promoted == 0 and full_reliability == 0",
            "actual": {
                "promoted_mechanism_count": current_snapshot[
                    "promoted_mechanism_count"
                ],
                "full_reliability_count": current_snapshot["full_reliability_count"],
            },
            "passed": (
                current_snapshot["promoted_mechanism_count"] == 0
                and current_snapshot["full_reliability_count"] == 0
            ),
            "why": "The harness governs future evidence; it is not itself a mechanism win.",
        },
    ]


def build_control_surface_offensive_doctrine() -> dict[str, Any]:
    coverage_gaps = load_json(COVERAGE_GAPS_PATH)
    gap_plan = load_json(GAP_CLOSURE_PLAN_PATH)
    genome_snapshot = load_json(GENOME_SNAPSHOT_PATH)
    next_queue = load_json(NEXT_QUEUE_PATH)
    branch_contracts = [
        branch_contract_from_order(order) for order in gap_plan["work_orders"]
    ]
    critical_gap_ids = sorted(
        gap["id"]
        for gap in coverage_gaps["coverage_gaps"]
        if gap["severity"] == "critical"
    )
    track_counts = dict(
        sorted(Counter(contract["track_type"] for contract in branch_contracts).items())
    )
    urgency_counts = dict(
        sorted(Counter(contract["urgency"] for contract in branch_contracts).items())
    )
    layer_target_counts = dict(
        sorted(
            Counter(
                layer
                for contract in branch_contracts
                for layer in contract["expected_generated_layer_change"]
            ).items()
        )
    )
    payload = {
        "schema_version": 1,
        "updated_at": gap_plan.get("updated_at"),
        "purpose": (
            "Turn the gap-closure plan into an executable branch-intake "
            "harness so future work must target a named gap, declare its "
            "death rule, and identify the generated atlas layer it can change."
        ),
        "sources": {
            "coverage_gaps": rel(COVERAGE_GAPS_PATH),
            "gap_closure_plan": rel(GAP_CLOSURE_PLAN_PATH),
            "genome_snapshot": rel(GENOME_SNAPSHOT_PATH),
            "next_queue": rel(NEXT_QUEUE_PATH),
        },
        "source_snapshot": {
            "row_count": genome_snapshot["summary"]["row_count"],
            "promoted_mechanism_count": genome_snapshot["summary"][
                "promoted_mechanism_count"
            ],
            "bounded_mechanism_count": genome_snapshot["summary"][
                "bounded_mechanism_count"
            ],
            "full_reliability_count": genome_snapshot["reliability_shape"][
                "full_reliability_count"
            ],
            "transfer_ready_mechanism_count": genome_snapshot["transfer_shape"][
                "transfer_ready_mechanism_count"
            ],
            "hidden_state_allowed_bridge_count": genome_snapshot["bridge_shape"][
                "hidden_state_allowed_count"
            ],
            "top_queue_ids": next_queue["summary"]["top_queue_ids"],
        },
        "critical_gap_ids": critical_gap_ids,
        "required_branch_fields": REQUIRED_BRANCH_FIELDS,
        "allowed_output_classes": OUTCOME_CLASSES,
        "global_rules": [
            "Start no branch without a named coverage gap.",
            "Start no branch without promote, bound, kill, containment, and export rules.",
            "Treat a typed failure as a result only if it changes the atlas, taxonomy, gap map, closure plan, or future branch contract.",
            "Require generated map movement before claiming progress beyond a local story.",
            "Treat hidden-state work as forbidden until behavior, null, holdout, prompt-channel, and output/candidate controls license it.",
        ],
        "branch_contracts": branch_contracts,
        "summary": {
            "source_work_order_count": gap_plan["summary"]["work_order_count"],
            "branch_contract_count": len(branch_contracts),
            "critical_gap_count": len(critical_gap_ids),
            "track_type_counts": track_counts,
            "urgency_counts": urgency_counts,
            "generated_layer_target_counts": layer_target_counts,
            "top_queue_coverage_count": gap_plan["summary"][
                "top_queue_coverage_count"
            ],
        },
        "allowed_claim": (
            "Future branches now have a generated intake contract: they must "
            "name target gaps, declare promote/bound/kill/contain/export rules, "
            "identify expected generated-layer movement, and emit one of four "
            "allowed closeout classes."
        ),
        "forbidden_claim": (
            "This harness does not close the mechanism, reliability, transfer, "
            "or bridge gaps. It prevents new work from bypassing them."
        ),
    }
    payload["validation_checks"] = build_validation_checks(payload)
    return payload


def validate_offensive_doctrine(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("offensive doctrine schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"offensive doctrine source missing: {rel_path}")
    if payload.get("allowed_output_classes") != OUTCOME_CLASSES:
        raise AssertionError("offensive doctrine allowed output classes drifted")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(f"offensive doctrine checks failed: {failed_checks}")


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Control-Surface Offensive Doctrine",
        "",
        "Date: 2026-07-01",
        "",
        "Status: generated offensive-doctrine harness implemented and validated.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/control_surface_offensive_doctrine.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_offensive_doctrine.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_offensive_doctrine.py --write",
        "python code\\control_surface_offensive_doctrine.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Purpose",
        "",
        "This layer is the operational complement to the defensive audit stack.",
        "It makes future branch quality machine-checkable before a run begins:",
        "the branch must name the gap it targets, the generated layer it can",
        "change, the evidence packet it owes, and the exact rule that kills it.",
        "",
        "## Generated Facts",
        "",
        f"- branch contracts: {summary['branch_contract_count']};",
        f"- source work orders: {summary['source_work_order_count']};",
        f"- critical gaps covered by contracts: {summary['critical_gap_count']};",
        f"- urgency counts: `{format_value(summary['urgency_counts'])}`;",
        f"- track type counts: `{format_value(summary['track_type_counts'])}`;",
        f"- top queue coverage: {summary['top_queue_coverage_count']}.",
        "",
        "## Global Rules",
        "",
    ]
    for rule in payload["global_rules"]:
        lines.append(f"- {rule}")

    lines.extend(
        [
            "",
            "## Required Branch Fields",
            "",
        ]
    )
    for field in payload["required_branch_fields"]:
        lines.append(f"- `{field}`")

    lines.extend(
        [
            "",
            "## Branch Contracts",
            "",
            "| Work Order | Urgency | Track | Target Gaps | Expected Layer Change | Kill Rule |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for contract in payload["branch_contracts"]:
        lines.append(
            f"| `{contract['work_order_id']}` | `{contract['urgency']}` | "
            f"`{contract['track_type']}` | "
            f"`{format_value(contract['target_gap_ids'])}` | "
            f"`{format_value(contract['expected_generated_layer_change'])}` | "
            f"{contract['kill_rule']} |"
        )

    lines.extend(
        [
            "",
            "## Allowed Outputs",
            "",
        ]
    )
    for output_class in payload["allowed_output_classes"]:
        lines.append(f"- `{output_class}`")

    lines.extend(
        [
            "",
            "## What This Proves",
            "",
            "It proves that the project now has a generated intake harness for new",
            "branches. A future experiment is not just a promising idea; it is a",
            "contract against a named gap, expected generated-layer movement, and",
            "a predeclared death rule.",
            "",
            "## What It Does Not Prove",
            "",
            "It does not prove that a control surface has been found, transferred,",
            "or made reliable. It only prevents future evidence from entering the",
            "atlas as an attractive but unbounded story.",
            "",
        ]
    )
    return "\n".join(lines)


def write_offensive_doctrine(
    output_path: Path = OFFENSIVE_DOCTRINE_PATH,
    report_path: Path = OFFENSIVE_DOCTRINE_REPORT_PATH,
) -> dict[str, Any]:
    payload = build_control_surface_offensive_doctrine()
    validate_offensive_doctrine(payload)
    write_json(output_path, payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(payload), encoding="utf-8", newline="\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="write offensive-doctrine artifacts",
    )
    parser.add_argument("--json", action="store_true", help="print doctrine JSON")
    args = parser.parse_args()

    payload = build_control_surface_offensive_doctrine()
    validate_offensive_doctrine(payload)

    if args.write:
        write_offensive_doctrine()
        print(
            f"wrote {OFFENSIVE_DOCTRINE_PATH.relative_to(ROOT).as_posix()} and "
            f"{OFFENSIVE_DOCTRINE_REPORT_PATH.relative_to(ROOT).as_posix()} "
            f"with {payload['summary']['branch_contract_count']} branch contracts"
        )
        return
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(
        "offensive doctrine ok: "
        f"{payload['summary']['branch_contract_count']} branch contracts"
    )
    print(
        "urgency_counts:",
        json.dumps(payload["summary"]["urgency_counts"], sort_keys=True),
    )
    print(
        "track_type_counts:",
        json.dumps(payload["summary"]["track_type_counts"], sort_keys=True),
    )


if __name__ == "__main__":
    main()
