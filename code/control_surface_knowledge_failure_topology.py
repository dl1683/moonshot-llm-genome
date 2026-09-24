"""Build the knowledge-substrate failure topology and second-wave plan.

The KSQ outcome matrix says how the first six behavior-substrate candidates
failed. This layer turns those failures into a small decision topology: which
branches are narrow repair candidates, which are template-fragility audits, and
which require materially new substrates before any hidden-state work.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from control_surface_artifacts import ROOT, load_json
from control_surface_knowledge_first_run_outcomes import (
    KNOWLEDGE_FIRST_RUN_OUTCOMES_PATH,
)


KNOWLEDGE_FAILURE_TOPOLOGY_PATH = (
    ROOT / "data" / "control_surface_knowledge_failure_topology.json"
)
KNOWLEDGE_FAILURE_TOPOLOGY_REPORT_PATH = (
    ROOT / "research" / "48_CONTROL_SURFACE_KNOWLEDGE_FAILURE_TOPOLOGY.md"
)

TOPOLOGY_NODES = [
    {
        "node_id": "structural_substrate_construction_solved",
        "node_type": "positive_infrastructure_result",
        "candidate_ids": [
            "ksq001_familiar_entity_prior_counterbalance",
            "ksq002_familiar_entity_source_rewrite_equivalence",
            "ksq003_bridge_statusless_evidence_aggregation",
            "ksq004_bridge_answer_interface_minimal_pairs",
            "ksq005_uncertainty_grounded_answerability",
            "ksq006_uncertainty_context_support_counterfactuals",
        ],
        "diagnostic_classes": [],
        "summary": "All six KSQ candidates passed structural construction on 40 sources.",
        "second_wave_decision": "reuse_harness",
        "why": "The next bottleneck is behavioral reliability, not row construction.",
    },
    {
        "node_id": "full_behavior_parseability_near_misses",
        "node_type": "narrow_repair_candidate",
        "candidate_ids": [
            "ksq001_familiar_entity_prior_counterbalance",
            "ksq002_familiar_entity_source_rewrite_equivalence",
        ],
        "diagnostic_classes": [
            "FAMILIAR_PRIOR_MIXTURE_PARSEABILITY_TRADEOFF",
            "SOURCE_REWRITE_HOLDOUT_PARSEABILITY_FAILURE",
        ],
        "summary": (
            "Two familiar-entity candidates reached full behavior before failing "
            "parseability or source-disjoint rewrite reliability."
        ),
        "second_wave_decision": "allow_one_repair_iteration_each",
        "why": "Both failures occurred after direct/null controls survived, so a bounded repair is informative.",
    },
    {
        "node_id": "bridge_template_and_local_table_boundary",
        "node_type": "bridge_boundary",
        "candidate_ids": [
            "ksq003_bridge_statusless_evidence_aggregation",
            "ksq004_bridge_answer_interface_minimal_pairs",
        ],
        "diagnostic_classes": [
            "STATUSLESS_EVIDENCE_LOCAL_TABLE_DOMINANCE",
            "ANSWER_INTERFACE_TEMPLATE_FRAGILITY",
        ],
        "summary": (
            "The bridge level splits between smoke-stage local-table dominance "
            "and full-stage template fragility."
        ),
        "second_wave_decision": "adjudicate_template_fragility_before_new_bridge_probe",
        "why": "KSQ004 has a stronger partial bridge than KSQ003, but neither licenses hidden-state work.",
    },
    {
        "node_id": "real_uncertainty_answer_channel_failures",
        "node_type": "material_redesign_required",
        "candidate_ids": [
            "ksq005_uncertainty_grounded_answerability",
            "ksq006_uncertainty_context_support_counterfactuals",
        ],
        "diagnostic_classes": [
            "GROUNDED_ANSWERABILITY_UNKNOWN_UNSUPPORTED_FAILURE",
            "CONTEXT_SUPPORT_CLAIM_OR_MENTION_CONTROL_FAILURE",
        ],
        "summary": (
            "Both real-uncertainty candidates fail at smoke because abstention or "
            "relation-free controls are not stable."
        ),
        "second_wave_decision": "redesign_before_more_runs",
        "why": "Correction-looking behavior is easier than grounded abstention or relation-sensitive support.",
    },
]

SECOND_WAVE_WORK_ORDERS = [
    {
        "work_order_id": "repair_ksq002_source_rewrite_holdout",
        "priority": "immediate",
        "track": "narrow_repair",
        "source_node_id": "full_behavior_parseability_near_misses",
        "candidate_ids": ["ksq002_familiar_entity_source_rewrite_equivalence"],
        "target_question": (
            "Can the source-disjoint rewrite holdout cross the predeclared "
            "parseability gate without weakening deletion/query-only locality?"
        ),
        "required_evidence": [
            "Full 40-source rerun with source-disjoint rewrite holdout parseability >= 0.90.",
            "Neutral rewrite artificial-value rate remains within 0.15 of baseline.",
            "Source deletion and query-only controls remain UNKNOWN-dominant.",
            "Candidate/output margins are reported on the selected template.",
            "Status card states whether hidden-state work remains forbidden or is newly admitted.",
        ],
        "promotion_rule": (
            "Admit only to behavior-ready status if rewrite, deletion, query-only, "
            "holdout, prompt audit, and margin-reporting gates all pass."
        ),
        "kill_rule": (
            "Kill ordinary source-rewrite repair if a single predeclared repair "
            "rerun still misses source-disjoint holdout parseability or breaks "
            "deletion/query-only controls."
        ),
        "containment_rule": (
            "If repaired, the claim is source-rewrite behavior admission only; "
            "no mechanism claim exists until a signature beats prompt/output "
            "controls and an intervention is tested."
        ),
        "export_rule": (
            "Export SOURCE_REWRITE_HOLDOUT_PARSEABILITY_FAILURE if the boundary "
            "persists, or SOURCE_REWRITE_BEHAVIOR_ADMITTED if the full gate passes."
        ),
        "forbidden_moves": [
            "Do not drop the source-disjoint holdout to improve pass rate.",
            "Do not remove deletion or query-only controls.",
            "Do not start hidden-state probing from the existing failed full run.",
        ],
    },
    {
        "work_order_id": "adjudicate_ksq004_template_invariance",
        "priority": "high",
        "track": "template_boundary",
        "source_node_id": "bridge_template_and_local_table_boundary",
        "candidate_ids": ["ksq004_bridge_answer_interface_minimal_pairs"],
        "target_question": (
            "Is the bridge behavior genuinely template-local to question_form, "
            "or can it survive a second compact/neutral surface?"
        ),
        "required_evidence": [
            "Predeclare template families before scoring.",
            "Report conflict, side-answer leakage, null/holdout, and candidate margins per template.",
            "Show whether question_form full behavior passes when selected directly.",
            "Show whether any compact/neutral relation-key template reaches the same branch balance.",
            "Preserve matched local/atomic direct controls under the same answer interface.",
        ],
        "promotion_rule": (
            "Admit bridge behavior only if at least two predeclared templates "
            "pass conflict, side-leakage, null, holdout, and margin-reporting gates."
        ),
        "kill_rule": (
            "Kill same-family answer-interface repair if question_form is the "
            "only passing surface or compact/neutral templates keep collapsing "
            "expected-atomic rows to local answers."
        ),
        "containment_rule": (
            "A single passing template is a template-specific behavior note, not "
            "a learned-memory bridge substrate."
        ),
        "export_rule": (
            "Export ANSWER_INTERFACE_TEMPLATE_FRAGILITY if the split persists, "
            "or TEMPLATE_INVARIANT_BRIDGE_BEHAVIOR if two templates pass."
        ),
        "forbidden_moves": [
            "Do not treat first-token numeric margins as meaningful when sequence candidate scoring is the real output baseline.",
            "Do not call question_form alone a robust bridge.",
            "Do not start hidden-state probing while compact/neutral templates fail.",
        ],
    },
    {
        "work_order_id": "bound_ksq001_familiar_prior_parseability",
        "priority": "medium",
        "track": "bounded_repair_or_closeout",
        "source_node_id": "full_behavior_parseability_near_misses",
        "candidate_ids": ["ksq001_familiar_entity_prior_counterbalance"],
        "target_question": (
            "Can familiar-prior conflict parseability be repaired without "
            "removing the weak local-versus-prior mixture?"
        ),
        "required_evidence": [
            "Full 40-source conflict parseability >= 0.90.",
            "Both artificial and real-prior/lure outcomes remain present.",
            "Direct local lookup, direct real-prior recall, nulls, and holdout survive.",
            "Answer-shape and candidate/output baselines are reported.",
        ],
        "promotion_rule": (
            "Admit only if parseability repair preserves mixture and all direct/null/holdout gates."
        ),
        "kill_rule": (
            "Kill ordinary KSQ001 repair if parseability improves only by collapsing to local lookup, prior recall, or UNKNOWN."
        ),
        "containment_rule": (
            "The surviving claim is familiar-prior behavior pressure, not a knowledge substrate."
        ),
        "export_rule": (
            "Export FAMILIAR_PRIOR_MIXTURE_PARSEABILITY_TRADEOFF unless a clean full behavior substrate appears."
        ),
        "forbidden_moves": [
            "Do not use a parser repair that changes the behavior branch distribution without reporting it.",
            "Do not count direct controls as conflict success.",
            "Do not run hidden-state probes on the current failed full behavior table.",
        ],
    },
    {
        "work_order_id": "redesign_statusless_bridge_substrate",
        "priority": "medium",
        "track": "material_redesign",
        "source_node_id": "bridge_template_and_local_table_boundary",
        "candidate_ids": ["ksq003_bridge_statusless_evidence_aggregation"],
        "target_question": (
            "What bridge substrate can route learned/local branches without "
            "visible status cues and without local-table dominance?"
        ),
        "required_evidence": [
            "A materially new branch-selection family, not another symbol/parity ablation.",
            "Direct learned recall and local lookup remain clean.",
            "Mismatch/conflict rows produce the intended learned branch above gate.",
            "Single-feature ablations go UNKNOWN instead of local.",
            "Source-disjoint holdout and candidate/output baselines are reported.",
        ],
        "promotion_rule": (
            "Admit only if conflict and ablation behavior both pass while direct controls and nulls stay clean."
        ),
        "kill_rule": (
            "Kill same-family evidence aggregation if the next material design again sends mismatch or ablation rows to local answers."
        ),
        "containment_rule": (
            "Until then, KSQ003 remains a local-table-dominance diagnostic."
        ),
        "export_rule": (
            "Export STATUSLESS_EVIDENCE_LOCAL_TABLE_DOMINANCE as the default bridge failure unless a new substrate passes."
        ),
        "forbidden_moves": [
            "Do not add visible trusted/untrusted labels.",
            "Do not reuse checksum, row-code, or fact-claim validity cues from the killed bridge family.",
            "Do not start hidden-state work before conflict and ablation gates pass.",
        ],
    },
    {
        "work_order_id": "redesign_real_uncertainty_answerability",
        "priority": "medium",
        "track": "material_redesign",
        "source_node_id": "real_uncertainty_answer_channel_failures",
        "candidate_ids": [
            "ksq005_uncertainty_grounded_answerability",
            "ksq006_uncertainty_context_support_counterfactuals",
        ],
        "target_question": (
            "Can real uncertainty be made behavior-ready when answerability and "
            "relation support are tested against claim-only and mention-only controls?"
        ),
        "required_evidence": [
            "Unknown and unsupported rows abstain or say UNKNOWN with >= 0.80 stability.",
            "Supported rows still answer correctly.",
            "Contradicting rows reject false context without relying on answer-shape artifacts.",
            "Claim-only and mention-only controls do not reproduce supported answers.",
            "Parser, answer schema, and output/candidate baselines are reported.",
        ],
        "promotion_rule": (
            "Admit only if abstention, support, contradiction, and claim/mention controls all pass together."
        ),
        "kill_rule": (
            "Kill the current uncertainty route if the next redesign again fixes correction-looking rows but fails unknown/unsupported or claim/mention controls."
        ),
        "containment_rule": (
            "Correction of false familiar facts remains a behavior diagnostic until grounded abstention and relation support pass."
        ),
        "export_rule": (
            "Export GROUNDED_ANSWERABILITY_UNKNOWN_UNSUPPORTED_FAILURE and "
            "CONTEXT_SUPPORT_CLAIM_OR_MENTION_CONTROL_FAILURE unless both routes pass."
        ),
        "forbidden_moves": [
            "Do not call contradicted-context correction uncertainty control.",
            "Do not hide claim-only or city-mention controls.",
            "Do not start hidden-state probing from a smoke-stage abstention failure.",
        ],
    },
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def by_candidate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = row["candidate_id"]
        if candidate_id in result:
            raise AssertionError(f"duplicate KSQ outcome {candidate_id}")
        result[candidate_id] = row
    return result


def build_topology_nodes(outcomes: dict[str, Any]) -> list[dict[str, Any]]:
    rows_by_candidate = by_candidate(outcomes["outcome_rows"])
    nodes = []
    for node in TOPOLOGY_NODES:
        candidate_rows = [rows_by_candidate[candidate_id] for candidate_id in node["candidate_ids"]]
        nodes.append(
            {
                **node,
                "candidate_count": len(candidate_rows),
                "terminal_gate_counts": dict(
                    sorted(Counter(row["terminal_gate"] for row in candidate_rows).items())
                ),
                "level_counts": dict(
                    sorted(Counter(row["level_id"] for row in candidate_rows).items())
                ),
                "behavior_ready_count": sum(
                    1 for row in candidate_rows if row["behavior_ready"] is True
                ),
                "hidden_state_allowed_count": sum(
                    1 for row in candidate_rows if row["hidden_state_allowed"] is True
                ),
                "candidate_summaries": [
                    {
                        "candidate_id": row["candidate_id"],
                        "level_id": row["level_id"],
                        "terminal_gate": row["terminal_gate"],
                        "exported_diagnostic_class": row["exported_diagnostic_class"],
                        "failure_axes": row["failure_axes"],
                        "survived_controls": row["survived_controls"],
                    }
                    for row in candidate_rows
                ],
            }
        )
    return nodes


def build_summary(nodes: list[dict[str, Any]], outcomes: dict[str, Any]) -> dict[str, Any]:
    work_orders = SECOND_WAVE_WORK_ORDERS
    return {
        "topology_node_count": len(nodes),
        "work_order_count": len(work_orders),
        "outcome_count": outcomes["summary"]["outcome_count"],
        "structural_passed_count": outcomes["summary"]["structural_passed_count"],
        "behavior_ready_count": outcomes["summary"]["behavior_ready_count"],
        "hidden_state_allowed_count": outcomes["summary"]["hidden_state_allowed_count"],
        "node_type_counts": dict(sorted(Counter(node["node_type"] for node in nodes).items())),
        "second_wave_decision_counts": dict(
            sorted(Counter(node["second_wave_decision"] for node in nodes).items())
        ),
        "work_order_priority_counts": dict(
            sorted(Counter(order["priority"] for order in work_orders).items())
        ),
        "work_order_track_counts": dict(
            sorted(Counter(order["track"] for order in work_orders).items())
        ),
        "candidate_coverage_count": len(
            {
                candidate_id
                for order in work_orders
                for candidate_id in order["candidate_ids"]
            }
        ),
        "diagnostic_coverage_count": len(
            {
                diagnostic
                for node in nodes
                for diagnostic in node["diagnostic_classes"]
            }
        ),
        "near_miss_candidate_ids": [
            candidate_id
            for node in nodes
            if node["node_type"] == "narrow_repair_candidate"
            for candidate_id in node["candidate_ids"]
        ],
        "redesign_candidate_ids": [
            candidate_id
            for node in nodes
            if node["node_type"] == "material_redesign_required"
            for candidate_id in node["candidate_ids"]
        ],
    }


def build_validation_checks(
    nodes: list[dict[str, Any]], summary: dict[str, Any], outcomes: dict[str, Any]
) -> list[dict[str, Any]]:
    outcome_candidate_ids = sorted(row["candidate_id"] for row in outcomes["outcome_rows"])
    node_candidate_ids = sorted(
        {
            candidate_id
            for node in nodes
            for candidate_id in node["candidate_ids"]
        }
    )
    work_order_candidate_ids = sorted(
        {
            candidate_id
            for order in SECOND_WAVE_WORK_ORDERS
            for candidate_id in order["candidate_ids"]
        }
    )
    source_node_ids = {node["node_id"] for node in nodes}
    work_order_source_failures = [
        order["work_order_id"]
        for order in SECOND_WAVE_WORK_ORDERS
        if order["source_node_id"] not in source_node_ids
    ]
    missing_decision_rules = [
        order["work_order_id"]
        for order in SECOND_WAVE_WORK_ORDERS
        if not all(
            order.get(key)
            for key in [
                "promotion_rule",
                "kill_rule",
                "containment_rule",
                "export_rule",
                "forbidden_moves",
                "required_evidence",
            ]
        )
    ]
    hidden_state_mentions = [
        order["work_order_id"]
        for order in SECOND_WAVE_WORK_ORDERS
        if "hidden-state" not in " ".join(order["forbidden_moves"])
        and "hidden state" not in order["containment_rule"]
    ]
    return [
        {
            "id": "topology_covers_all_ksq_outcomes",
            "predicate": "node candidate ids == outcome candidate ids",
            "actual": {
                "node_candidate_ids": node_candidate_ids,
                "outcome_candidate_ids": outcome_candidate_ids,
            },
            "passed": node_candidate_ids == outcome_candidate_ids,
            "why": "Every executed KSQ row must appear in the failure topology.",
        },
        {
            "id": "second_wave_covers_all_ksq_outcomes",
            "predicate": "work order candidate ids == outcome candidate ids",
            "actual": {
                "work_order_candidate_ids": work_order_candidate_ids,
                "outcome_candidate_ids": outcome_candidate_ids,
            },
            "passed": work_order_candidate_ids == outcome_candidate_ids,
            "why": "Every executed KSQ row needs either a repair, closeout, or redesign decision.",
        },
        {
            "id": "no_hidden_state_license_created",
            "predicate": "behavior_ready_count == 0 and hidden_state_allowed_count == 0",
            "actual": {
                "behavior_ready_count": summary["behavior_ready_count"],
                "hidden_state_allowed_count": summary["hidden_state_allowed_count"],
            },
            "passed": summary["behavior_ready_count"] == 0
            and summary["hidden_state_allowed_count"] == 0,
            "why": "The topology must not turn failed behavior gates into probe licenses.",
        },
        {
            "id": "work_orders_bind_existing_nodes",
            "predicate": "empty list",
            "actual": work_order_source_failures,
            "passed": not work_order_source_failures,
            "why": "Second-wave work orders should be grounded in topology nodes.",
        },
        {
            "id": "work_orders_have_decision_rules",
            "predicate": "empty list",
            "actual": missing_decision_rules,
            "passed": not missing_decision_rules,
            "why": "Every second-wave branch needs promotion, kill, containment, export, and evidence rules.",
        },
        {
            "id": "work_orders_preserve_hidden_state_bar",
            "predicate": "empty list",
            "actual": hidden_state_mentions,
            "passed": not hidden_state_mentions,
            "why": "Every second-wave branch must explicitly preserve the no-hidden-state-work boundary.",
        },
        {
            "id": "near_miss_and_redesign_are_separated",
            "predicate": "at least one near miss and one redesign candidate",
            "actual": {
                "near_miss_candidate_ids": summary["near_miss_candidate_ids"],
                "redesign_candidate_ids": summary["redesign_candidate_ids"],
            },
            "passed": bool(summary["near_miss_candidate_ids"])
            and bool(summary["redesign_candidate_ids"]),
            "why": "The topology should not flatten all failures into one generic bucket.",
        },
    ]


def build_control_surface_knowledge_failure_topology() -> dict[str, Any]:
    outcomes = load_json(KNOWLEDGE_FIRST_RUN_OUTCOMES_PATH)
    nodes = build_topology_nodes(outcomes)
    summary = build_summary(nodes, outcomes)
    payload = {
        "schema_version": 1,
        "updated_at": outcomes.get("updated_at"),
        "purpose": (
            "Classify the six executed KSQ first-run failures into a topology "
            "of repairable near-misses, template boundaries, material redesign "
            "requirements, and second-wave work orders."
        ),
        "sources": {
            "knowledge_first_run_outcomes": rel(KNOWLEDGE_FIRST_RUN_OUTCOMES_PATH),
        },
        "classification_rule": {
            "near_miss": "A candidate reached full behavior and failed a narrow parseability or holdout boundary while direct/null controls survived.",
            "template_boundary": "A candidate shows materially different behavior across prompt templates or answer-interface forms.",
            "material_redesign": "A candidate fails at smoke because core controls, abstention, relation support, or branch routing are not stable.",
            "hidden_state_bar": "No topology node or work order licenses hidden-state work until a future behavior substrate passes admission gates.",
        },
        "summary": summary,
        "topology_nodes": nodes,
        "second_wave_work_orders": SECOND_WAVE_WORK_ORDERS,
        "validation_checks": build_validation_checks(nodes, summary, outcomes),
        "allowed_claim": (
            "The first KSQ wave now has an explicit failure topology: source-rewrite "
            "and familiar-prior parseability are bounded repair candidates; bridge "
            "behavior is split between template fragility and local-table dominance; "
            "real uncertainty needs material redesign before more probing."
        ),
        "forbidden_claim": (
            "This topology does not make any KSQ row behavior-ready, does not "
            "license hidden-state search, and does not claim a knowledge-control "
            "surface."
        ),
    }
    return payload


def validate_knowledge_failure_topology(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("knowledge failure topology schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"knowledge failure topology source missing: {rel_path}")
    summary = payload["summary"]
    if summary["topology_node_count"] != 4:
        raise AssertionError("knowledge failure topology expected four nodes")
    if summary["work_order_count"] != 5:
        raise AssertionError("knowledge failure topology expected five work orders")
    if summary["candidate_coverage_count"] != 6:
        raise AssertionError("knowledge failure topology must cover six candidates")
    if summary["behavior_ready_count"] != 0 or summary["hidden_state_allowed_count"] != 0:
        raise AssertionError("knowledge failure topology must not license hidden-state work")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(f"knowledge failure topology checks failed: {failed_checks}")


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Control-Surface Knowledge Failure Topology",
        "",
        f"Source updated_at: {payload['updated_at']}",
        "",
        "Status: generated KSQ failure topology implemented and validated.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/control_surface_knowledge_failure_topology.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_knowledge_failure_topology.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_knowledge_failure_topology.py --write",
        "python code\\control_surface_knowledge_failure_topology.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Summary",
        "",
        f"- topology nodes: `{summary['topology_node_count']}`",
        f"- second-wave work orders: `{summary['work_order_count']}`",
        f"- candidate coverage: `{summary['candidate_coverage_count']}/6`",
        f"- behavior-ready rows: `{summary['behavior_ready_count']}`",
        f"- hidden-state-allowed rows: `{summary['hidden_state_allowed_count']}`",
        f"- node types: `{format_value(summary['node_type_counts'])}`",
        f"- work-order priorities: `{format_value(summary['work_order_priority_counts'])}`",
        "",
        "## Topology Nodes",
        "",
        "| Node | Type | Candidates | Decision |",
        "| --- | --- | --- | --- |",
    ]
    for node in payload["topology_nodes"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{node['node_id']}`",
                    f"`{node['node_type']}`",
                    format_value(node["candidate_ids"]),
                    f"`{node['second_wave_decision']}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Second-Wave Work Orders", ""])
    for order in payload["second_wave_work_orders"]:
        lines.extend(
            [
                f"### `{order['work_order_id']}`",
                "",
                f"- priority: `{order['priority']}`",
                f"- track: `{order['track']}`",
                f"- candidates: `{format_value(order['candidate_ids'])}`",
                f"- target question: {order['target_question']}",
                f"- required evidence: `{format_value(order['required_evidence'])}`",
                f"- promotion rule: {order['promotion_rule']}",
                f"- kill rule: {order['kill_rule']}",
                f"- containment rule: {order['containment_rule']}",
                f"- export rule: {order['export_rule']}",
                f"- forbidden moves: `{format_value(order['forbidden_moves'])}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Validation Checks",
            "",
            "| Check | Passed |",
            "| --- | --- |",
        ]
    )
    for check in payload["validation_checks"]:
        lines.append(f"| `{check['id']}` | `{str(check['passed']).lower()}` |")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    payload = build_control_surface_knowledge_failure_topology()
    validate_knowledge_failure_topology(payload)
    if args.write:
        write_json(KNOWLEDGE_FAILURE_TOPOLOGY_PATH, payload)
        KNOWLEDGE_FAILURE_TOPOLOGY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        KNOWLEDGE_FAILURE_TOPOLOGY_REPORT_PATH.write_text(
            render_markdown(payload), encoding="utf-8", newline="\n"
        )
    print(
        json.dumps(
            {
                "topology_node_count": payload["summary"]["topology_node_count"],
                "work_order_count": payload["summary"]["work_order_count"],
                "candidate_coverage_count": payload["summary"][
                    "candidate_coverage_count"
                ],
                "node_type_counts": payload["summary"]["node_type_counts"],
                "work_order_priority_counts": payload["summary"][
                    "work_order_priority_counts"
                ],
                "behavior_ready_count": payload["summary"]["behavior_ready_count"],
                "hidden_state_allowed_count": payload["summary"][
                    "hidden_state_allowed_count"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
