"""Build the knowledge-ladder coverage map.

The family matrix says where every row currently sits. This layer asks the
knowledge-specific question: how far has the project climbed from synthetic
lookup toward real factual correction, abstention, and uncertainty?
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from control_surface_artifacts import ROOT, load_json
from control_surface_bridge_ladder import BRIDGE_LADDER_PATH
from control_surface_family_matrix import FAMILY_MATRIX_PATH
from mc005_reference_specimen_audit import MC005_REFERENCE_SPECIMEN_AUDIT_PATH
from mc006_predecision_frontier_audit import MC006_PREDECISION_FRONTIER_AUDIT_PATH
from post_mc033_bridge_closeout_audit import POST_MC033_BRIDGE_CLOSEOUT_PATH


KNOWLEDGE_LADDER_PATH = ROOT / "data" / "control_surface_knowledge_ladder.json"
KNOWLEDGE_LADDER_REPORT_PATH = ROOT / "research" / "42_CONTROL_SURFACE_KNOWLEDGE_LADDER.md"

LADDER_LEVELS = [
    {
        "level_id": "level_1_synthetic_lookup",
        "label": "Pure synthetic lookup",
        "description": "The answer is fully in the prompt and source-value paths can be localized.",
        "row_ids": ["mc005_associative_lookup"],
        "target_claim": "Can a prompt-local source-value lookup surface become a bounded mechanism reference?",
    },
    {
        "level_id": "level_2_semi_synthetic_familiar_entity",
        "label": "Semi-synthetic familiar entities",
        "description": "Familiar entity names are used, but values remain task-local or artificial.",
        "row_ids": ["mc007_semi_synthetic_familiar_entity_lookup"],
        "target_claim": "Do familiar semantic priors preserve or disrupt source-value lookup?",
    },
    {
        "level_id": "level_3_symbolic_or_learned_memory_bridge",
        "label": "Symbolic / learned-memory bridge",
        "description": "Task-local codes, numeric facts, status cues, or source-validity rules compete with learned atomic or factual memory.",
        "row_ids": [
            "mc008_symbolic_fact_code_arbitration",
            "mc009_derived_code_arbitration",
            "mc010_two_hop_fact_code_arbitration",
            "mc011_atomic_number_code_arbitration",
            "mc012_reliability_labeled_numeric_arbitration",
            "mc013_status_channel_ablation_numeric_arbitration",
            "mc014_inferred_reliability_numeric_arbitration",
            "mc015_parity_gated_numeric_arbitration",
            "mc016_alphabet_gated_numeric_arbitration",
        ],
        "target_claim": "Can a bridge substrate force reliable learned/local arbitration before hidden-state work starts?",
    },
    {
        "level_id": "level_4_parametric_fact_override",
        "label": "Strong parametric fact override",
        "description": "Real capital facts compete with task-local fictional overrides under generated-answer contracts.",
        "row_ids": ["mc006_parametric_fact_override"],
        "target_claim": "Does a knowledge-like predecision monitor become a reliable causal lever?",
    },
    {
        "level_id": "level_5_real_abstention_uncertainty",
        "label": "Real abstention / uncertainty",
        "description": "Known/unknown and context-support behavior attempts target factual uncertainty, abstention, or correction.",
        "row_ids": ["mc002_known_unknown", "mc002b_context_support"],
        "target_claim": "Can real factual correction or uncertainty behavior become a mechanism-ready substrate?",
    },
]

AUXILIARY_ROWS = [
    "mc001_qwen3_0p6b_truth_agreement",
    "mc001b_qwen3_1p7b_truth_agreement",
    "mc001g_gemma_truth_agreement",
    "mc003_delayed_copy",
    "mc004_in_context_binding",
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
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def by_row_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = row["row_id"]
        if row_id in result:
            raise AssertionError(f"duplicate matrix row {row_id}")
        result[row_id] = row
    return result


def count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(row[key] for row in rows).items()))


def level_status(rows: list[dict[str, Any]], level_id: str) -> tuple[str, int]:
    if not rows:
        return "missing", 0
    if any(row["verdict"] == "promoted_mechanism_card" for row in rows):
        return "promoted_mechanism", 5
    if any(row["reliability_class"] == "bounded_reliability_reference" for row in rows):
        return "bounded_reference", 4
    if any(row["frontier_class"] == "predecision_monitor_no_lever" for row in rows):
        return "monitor_only_no_lever", 3
    if any(row["terminal_stage"] == "pre_signature_prompt_channel_locality" for row in rows):
        return "prompt_visible_or_behavior_blocked", 2
    if all(row["terminal_stage"] == "pre_signature_behavior_substrate" for row in rows):
        return "behavior_substrate_blocked", 1
    if any(row["primary_blocker"] == "behavior_substrate_or_bridge_blocked" for row in rows):
        return "behavior_substrate_blocked", 1
    if level_id == "level_5_real_abstention_uncertainty":
        return "attempted_not_mechanism_ready", 1
    return "diagnostic_only", 1


def build_level_entries(
    family_matrix: dict[str, Any],
    bridge_ladder: dict[str, Any],
    mc005: dict[str, Any],
    mc006: dict[str, Any],
    post_mc033: dict[str, Any],
) -> list[dict[str, Any]]:
    rows_by_id = by_row_id(family_matrix["matrix_rows"])
    levels = []
    for level in LADDER_LEVELS:
        rows = [rows_by_id[row_id] for row_id in level["row_ids"]]
        status, stage_rank = level_status(rows, level["level_id"])
        hidden_state_blocked_count = sum(
            1
            for row in rows
            if row["terminal_stage"].startswith("pre_signature")
            or row["reliability_class"].startswith("not_reliable")
        )
        entry = {
            "level_id": level["level_id"],
            "label": level["label"],
            "description": level["description"],
            "target_claim": level["target_claim"],
            "row_ids": level["row_ids"],
            "row_count": len(rows),
            "status": status,
            "stage_rank": stage_rank,
            "verdict_counts": count_values(rows, "verdict"),
            "primary_blocker_counts": count_values(rows, "primary_blocker"),
            "terminal_stage_counts": count_values(rows, "terminal_stage"),
            "frontier_class_counts": count_values(rows, "frontier_class"),
            "reliability_class_counts": count_values(rows, "reliability_class"),
            "transfer_class_counts": count_values(rows, "transfer_class"),
            "prompt_contract_visible_count": sum(1 for row in rows if row["prompt_contract_visible"]),
            "output_geometry_visible_count": sum(1 for row in rows if row["output_geometry_visible"]),
            "source_or_prompt_token_dependent_count": sum(
                1 for row in rows if row["source_or_prompt_token_dependent"]
            ),
            "internal_monitor_present_count": sum(1 for row in rows if row["internal_monitor_present"]),
            "internal_causal_surface_count": sum(1 for row in rows if row["internal_causal_surface"]),
            "hidden_state_blocked_count": hidden_state_blocked_count,
            "promotion_ready_count": sum(
                1
                for row in rows
                if row["verdict"] == "promoted_mechanism_card"
                or row["reliability_class"] == "full_reliability_mechanism"
                or row["transfer_class"] == "transfer_ready_mechanism"
            ),
            "rows": [
                {
                    "row_id": row["row_id"],
                    "family": row["family"],
                    "primary_blocker": row["primary_blocker"],
                    "terminal_stage": row["terminal_stage"],
                    "frontier_class": row["frontier_class"],
                    "reliability_class": row["reliability_class"],
                    "transfer_class": row["transfer_class"],
                    "verdict": row["verdict"],
                    "next_decision": row["next_decision"],
                }
                for row in rows
            ],
        }
        if level["level_id"] == "level_1_synthetic_lookup":
            entry["anchor_audit"] = mc005["summary"]
            entry["allowed_interpretation"] = (
                "Synthetic lookup supplies the only bounded causal reference, "
                "but null reliability and transfer remain boundaries."
            )
        elif level["level_id"] == "level_4_parametric_fact_override":
            entry["anchor_audit"] = mc006["summary"]
            entry["allowed_interpretation"] = (
                "Capital-fact override supplies a matched behavior substrate "
                "and predecision monitor, but no reliable causal lever."
            )
        elif level["level_id"] == "level_3_symbolic_or_learned_memory_bridge":
            entry["bridge_context"] = {
                "bridge_rung_count": bridge_ladder["summary"]["rung_count"],
                "smoke_rung_count": bridge_ladder["summary"]["smoke_rung_count"],
                "hidden_state_allowed_count": bridge_ladder["summary"][
                    "hidden_state_allowed_count"
                ],
                "clean_unconfounded_bridge_count": bridge_ladder["summary"][
                    "clean_unconfounded_bridge_count"
                ],
                "post_mc033_same_family_route_status": post_mc033["summary"][
                    "same_family_route_status"
                ],
                "recent_closed_rung_ids": post_mc033["summary"][
                    "recent_closed_rung_ids"
                ],
            }
            entry["allowed_interpretation"] = (
                "The bridge ladder is a substrate-failure map, not a hidden "
                "knowledge mechanism."
            )
        else:
            entry["allowed_interpretation"] = (
                "This level is represented only as a diagnostic or blocked "
                "behavior substrate under current controls."
            )
        levels.append(entry)
    return levels


def build_auxiliary_rows(family_matrix: dict[str, Any]) -> list[dict[str, Any]]:
    rows_by_id = by_row_id(family_matrix["matrix_rows"])
    return [
        {
            "row_id": row_id,
            "family": rows_by_id[row_id]["family"],
            "role": "diagnostic_support_not_ladder_level",
            "behavior_domain": rows_by_id[row_id]["behavior_domain"],
            "frontier_class": rows_by_id[row_id]["frontier_class"],
            "primary_blocker": rows_by_id[row_id]["primary_blocker"],
            "terminal_stage": rows_by_id[row_id]["terminal_stage"],
        }
        for row_id in AUXILIARY_ROWS
    ]


def build_summary(
    levels: list[dict[str, Any]],
    auxiliary_rows: list[dict[str, Any]],
    family_matrix: dict[str, Any],
    bridge_ladder: dict[str, Any],
) -> dict[str, Any]:
    row_count = sum(level["row_count"] for level in levels)
    status_counts = dict(sorted(Counter(level["status"] for level in levels).items()))
    return {
        "level_count": len(levels),
        "ladder_row_count": row_count,
        "auxiliary_row_count": len(auxiliary_rows),
        "family_matrix_row_count": family_matrix["summary"]["row_count"],
        "status_counts": status_counts,
        "max_stage_rank": max(level["stage_rank"] for level in levels),
        "promoted_level_count": sum(1 for level in levels if level["status"] == "promoted_mechanism"),
        "bounded_reference_level_count": sum(1 for level in levels if level["status"] == "bounded_reference"),
        "monitor_only_level_count": sum(1 for level in levels if level["status"] == "monitor_only_no_lever"),
        "behavior_or_prompt_blocked_level_count": sum(
            1
            for level in levels
            if level["status"]
            in {"behavior_substrate_blocked", "prompt_visible_or_behavior_blocked"}
        ),
        "real_abstention_uncertainty_ready_count": sum(
            1
            for level in levels
            if level["level_id"] == "level_5_real_abstention_uncertainty"
            and level["stage_rank"] >= 3
        ),
        "bridge_rung_count": bridge_ladder["summary"]["rung_count"],
        "bridge_hidden_state_allowed_count": bridge_ladder["summary"][
            "hidden_state_allowed_count"
        ],
        "bridge_clean_unconfounded_count": bridge_ladder["summary"][
            "clean_unconfounded_bridge_count"
        ],
        "coverage_ratio": ratio(row_count, family_matrix["summary"]["row_count"]),
    }


def build_validation_checks(
    levels: list[dict[str, Any]],
    auxiliary_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    family_matrix: dict[str, Any],
    bridge_ladder: dict[str, Any],
    mc005: dict[str, Any],
    mc006: dict[str, Any],
    post_mc033: dict[str, Any],
) -> list[dict[str, Any]]:
    ladder_row_ids = sorted(row_id for level in levels for row_id in level["row_ids"])
    auxiliary_row_ids = sorted(row["row_id"] for row in auxiliary_rows)
    matrix_row_ids = sorted(row["row_id"] for row in family_matrix["matrix_rows"])
    levels_by_id = {level["level_id"]: level for level in levels}
    checks = [
        {
            "id": "five_ladder_levels_declared",
            "predicate": "level_count == 5",
            "actual": summary["level_count"],
            "passed": summary["level_count"] == 5,
            "why": "The knowledge ladder must separate synthetic, semi-synthetic, bridge, parametric-fact, and real uncertainty levels.",
        },
        {
            "id": "ladder_and_auxiliary_rows_partition_family_matrix",
            "predicate": "ladder rows plus auxiliary rows == all matrix rows",
            "actual": {
                "ladder_plus_auxiliary": sorted(ladder_row_ids + auxiliary_row_ids),
                "family_matrix_rows": matrix_row_ids,
            },
            "passed": sorted(ladder_row_ids + auxiliary_row_ids) == matrix_row_ids
            and len(set(ladder_row_ids) & set(auxiliary_row_ids)) == 0,
            "why": "The ladder should account for every atlas row without pretending diagnostics are knowledge levels.",
        },
        {
            "id": "synthetic_lookup_level_is_mc005_bounded_reference",
            "predicate": "level 1 row == MC005 and status bounded_reference",
            "actual": {
                "row_ids": levels_by_id["level_1_synthetic_lookup"]["row_ids"],
                "status": levels_by_id["level_1_synthetic_lookup"]["status"],
                "mc005_route_status": mc005["summary"]["route_status"],
            },
            "passed": levels_by_id["level_1_synthetic_lookup"]["row_ids"]
            == ["mc005_associative_lookup"]
            and levels_by_id["level_1_synthetic_lookup"]["status"]
            == "bounded_reference"
            and mc005["summary"]["route_status"] == "bounded_frozen_not_promoted",
            "why": "MC005 is the only current bounded causal specimen and must anchor the ladder.",
        },
        {
            "id": "semi_synthetic_level_is_blocked_mc007",
            "predicate": "level 2 row == MC007 and hidden state blocked",
            "actual": {
                "row_ids": levels_by_id[
                    "level_2_semi_synthetic_familiar_entity"
                ]["row_ids"],
                "status": levels_by_id[
                    "level_2_semi_synthetic_familiar_entity"
                ]["status"],
            },
            "passed": levels_by_id["level_2_semi_synthetic_familiar_entity"][
                "row_ids"
            ]
            == ["mc007_semi_synthetic_familiar_entity_lookup"]
            and levels_by_id["level_2_semi_synthetic_familiar_entity"]["status"]
            == "behavior_substrate_blocked",
            "why": "The familiar-entity bridge is not mechanism-ready in the current map.",
        },
        {
            "id": "bridge_level_preserves_post_mc033_closeout",
            "predicate": "bridge has 9 atlas rows, 24 rungs, no hidden-state allowance",
            "actual": {
                "level_row_count": levels_by_id[
                    "level_3_symbolic_or_learned_memory_bridge"
                ]["row_count"],
                "bridge_rung_count": bridge_ladder["summary"]["rung_count"],
                "hidden_state_allowed": bridge_ladder["summary"][
                    "hidden_state_allowed_count"
                ],
                "clean_unconfounded": bridge_ladder["summary"][
                    "clean_unconfounded_bridge_count"
                ],
                "post_mc033_status": post_mc033["summary"][
                    "same_family_route_status"
                ],
            },
            "passed": levels_by_id[
                "level_3_symbolic_or_learned_memory_bridge"
            ]["row_count"]
            == 9
            and bridge_ladder["summary"]["rung_count"] == 24
            and bridge_ladder["summary"]["hidden_state_allowed_count"] == 0
            and bridge_ladder["summary"]["clean_unconfounded_bridge_count"] == 0
            and post_mc033["summary"]["same_family_route_status"]
            == "killed_after_mc033",
            "why": "The bridge ladder is currently a closure map, not an intervention substrate.",
        },
        {
            "id": "parametric_fact_level_is_mc006_monitor_only",
            "predicate": "level 4 row == MC006 and status monitor_only_no_lever",
            "actual": {
                "row_ids": levels_by_id["level_4_parametric_fact_override"][
                    "row_ids"
                ],
                "status": levels_by_id["level_4_parametric_fact_override"][
                    "status"
                ],
                "mc006_route_status": mc006["summary"]["route_status"],
            },
            "passed": levels_by_id["level_4_parametric_fact_override"]["row_ids"]
            == ["mc006_parametric_fact_override"]
            and levels_by_id["level_4_parametric_fact_override"]["status"]
            == "monitor_only_no_lever"
            and mc006["summary"]["route_status"] == "monitor_only_closed",
            "why": "MC006 is the current knowledge-like timing boundary, not a causal control surface.",
        },
        {
            "id": "real_uncertainty_level_not_mechanism_ready",
            "predicate": "level 5 rows are MC002/MC002B and stage rank < monitor",
            "actual": {
                "row_ids": levels_by_id["level_5_real_abstention_uncertainty"][
                    "row_ids"
                ],
                "stage_rank": levels_by_id[
                    "level_5_real_abstention_uncertainty"
                ]["stage_rank"],
                "ready_count": summary["real_abstention_uncertainty_ready_count"],
            },
            "passed": levels_by_id["level_5_real_abstention_uncertainty"][
                "row_ids"
            ]
            == ["mc002_known_unknown", "mc002b_context_support"]
            and levels_by_id["level_5_real_abstention_uncertainty"]["stage_rank"] < 3
            and summary["real_abstention_uncertainty_ready_count"] == 0,
            "why": "Real factual correction/uncertainty remains behavior-blocked, not mechanism-ready.",
        },
        {
            "id": "no_ladder_level_is_promoted",
            "predicate": "promoted_level_count == 0",
            "actual": summary["promoted_level_count"],
            "passed": summary["promoted_level_count"] == 0,
            "why": "The ladder must not imply a promoted knowledge mechanism.",
        },
    ]
    return checks


def build_control_surface_knowledge_ladder() -> dict[str, Any]:
    family_matrix = load_json(FAMILY_MATRIX_PATH)
    bridge_ladder = load_json(BRIDGE_LADDER_PATH)
    mc005 = load_json(MC005_REFERENCE_SPECIMEN_AUDIT_PATH)
    mc006 = load_json(MC006_PREDECISION_FRONTIER_AUDIT_PATH)
    post_mc033 = load_json(POST_MC033_BRIDGE_CLOSEOUT_PATH)
    levels = build_level_entries(
        family_matrix, bridge_ladder, mc005, mc006, post_mc033
    )
    auxiliary_rows = build_auxiliary_rows(family_matrix)
    summary = build_summary(levels, auxiliary_rows, family_matrix, bridge_ladder)
    checks = build_validation_checks(
        levels,
        auxiliary_rows,
        summary,
        family_matrix,
        bridge_ladder,
        mc005,
        mc006,
        post_mc033,
    )
    return {
        "schema_version": 1,
        "updated_at": family_matrix.get("updated_at"),
        "purpose": (
            "Map the current knowledge ladder from synthetic lookup to real "
            "factual correction/uncertainty, using the family matrix and bridge "
            "closeouts as evidence."
        ),
        "sources": {
            "family_matrix": rel(FAMILY_MATRIX_PATH),
            "bridge_ladder": rel(BRIDGE_LADDER_PATH),
            "mc005_reference_specimen": rel(MC005_REFERENCE_SPECIMEN_AUDIT_PATH),
            "mc006_predecision_frontier": rel(MC006_PREDECISION_FRONTIER_AUDIT_PATH),
            "post_mc033_bridge_closeout": rel(POST_MC033_BRIDGE_CLOSEOUT_PATH),
        },
        "classification_rule": {
            "levels": "Five predeclared knowledge-ladder levels, from prompt-local lookup to real factual uncertainty.",
            "status": "Derived from each level's best row status: promoted, bounded reference, monitor-only, prompt-visible/behavior-blocked, or behavior-blocked.",
            "auxiliary_rows": "Truth/agreement, delayed-copy, and in-context-binding rows are supporting diagnostics, not knowledge-ladder levels.",
        },
        "summary": summary,
        "levels": levels,
        "auxiliary_rows": auxiliary_rows,
        "validation_checks": checks,
        "allowed_claim": (
            "The project currently has a partial knowledge ladder: synthetic "
            "lookup reaches a bounded causal reference, capital-fact override "
            "reaches monitor-only timing evidence, bridge rows mostly fail before "
            "hidden-state work, and real abstention/uncertainty is not "
            "mechanism-ready."
        ),
        "forbidden_claim": (
            "This ladder does not establish a broad knowledge mechanism, a truth "
            "or knowledge vector, a promoted mechanism card, a reliable MC006 "
            "steering surface, or a real-world factual correction/refusal "
            "mechanism."
        ),
    }


def validate_knowledge_ladder(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("knowledge ladder schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"knowledge ladder source missing: {rel_path}")
    if payload["summary"]["level_count"] != 5:
        raise AssertionError("knowledge ladder expected five levels")
    if payload["summary"]["promoted_level_count"] != 0:
        raise AssertionError("knowledge ladder must not report promoted levels")
    if payload["summary"]["bounded_reference_level_count"] != 1:
        raise AssertionError("knowledge ladder expected one bounded reference level")
    if payload["summary"]["monitor_only_level_count"] != 1:
        raise AssertionError("knowledge ladder expected one monitor-only level")
    if payload["summary"]["bridge_hidden_state_allowed_count"] != 0:
        raise AssertionError("knowledge ladder must not allow hidden-state bridge work")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(f"knowledge ladder checks failed: {failed_checks}")


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Control-Surface Knowledge Ladder",
        "",
        f"Source updated_at: {payload['updated_at']}",
        "",
        "Status: generated knowledge-ladder coverage map implemented and validated.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/control_surface_knowledge_ladder.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_knowledge_ladder.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_knowledge_ladder.py --write",
        "python code\\control_surface_knowledge_ladder.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Purpose",
        "",
        "This map separates the knowledge-specific ladder from auxiliary",
        "diagnostic rows. It shows how far the current atlas gets from pure",
        "synthetic lookup toward real factual correction, refusal, and",
        "uncertainty behavior.",
        "",
        "## Summary",
        "",
        f"- levels: {summary['level_count']};",
        f"- ladder rows: {summary['ladder_row_count']};",
        f"- auxiliary diagnostic rows: {summary['auxiliary_row_count']};",
        f"- bounded-reference levels: {summary['bounded_reference_level_count']};",
        f"- monitor-only levels: {summary['monitor_only_level_count']};",
        f"- promoted levels: {summary['promoted_level_count']};",
        f"- bridge hidden-state-allowed rungs: {summary['bridge_hidden_state_allowed_count']};",
        f"- bridge clean unconfounded rungs: {summary['bridge_clean_unconfounded_count']};",
        f"- real abstention/uncertainty mechanism-ready levels: {summary['real_abstention_uncertainty_ready_count']}.",
        "",
        "## Ladder",
        "",
        "| Level | Status | Rank | Rows | Main Boundary |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for level in payload["levels"]:
        boundaries = []
        for key, values in [
            ("blocker", level["primary_blocker_counts"]),
            ("stage", level["terminal_stage_counts"]),
            ("frontier", level["frontier_class_counts"]),
            ("reliability", level["reliability_class_counts"]),
        ]:
            boundaries.append(f"{key}: {format_value(values)}")
        lines.append(
            f"| `{level['level_id']}` | `{level['status']}` | "
            f"{level['stage_rank']} | "
            f"{'<br>'.join(f'`{row_id}`' for row_id in level['row_ids'])} | "
            f"{'<br>'.join(boundaries)} |"
        )

    lines.extend(
        [
            "",
            "## Auxiliary Diagnostics",
            "",
            "| Row | Role | Domain | Frontier | Stage |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["auxiliary_rows"]:
        lines.append(
            f"| `{row['row_id']}` | `{row['role']}` | "
            f"`{row['behavior_domain']}` | `{row['frontier_class']}` | "
            f"`{row['terminal_stage']}` |"
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

    payload = build_control_surface_knowledge_ladder()
    validate_knowledge_ladder(payload)
    if args.write:
        write_json(KNOWLEDGE_LADDER_PATH, payload)
        KNOWLEDGE_LADDER_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        KNOWLEDGE_LADDER_REPORT_PATH.write_text(
            render_markdown(payload),
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "passed": True,
                "level_count": payload["summary"]["level_count"],
                "ladder_row_count": payload["summary"]["ladder_row_count"],
                "auxiliary_row_count": payload["summary"]["auxiliary_row_count"],
                "bounded_reference_level_count": payload["summary"][
                    "bounded_reference_level_count"
                ],
                "monitor_only_level_count": payload["summary"][
                    "monitor_only_level_count"
                ],
                "promoted_level_count": payload["summary"]["promoted_level_count"],
                "real_abstention_uncertainty_ready_count": payload["summary"][
                    "real_abstention_uncertainty_ready_count"
                ],
                "output_path": rel(KNOWLEDGE_LADDER_PATH),
                "report_path": rel(KNOWLEDGE_LADDER_REPORT_PATH),
            },
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
