#!/usr/bin/env python
"""Build the knowledge eleventh-wave outcome layer.

KSQ015 follows the KSQ014 catalog-slash smoke survivor with the required
full-source behavior test. It records the failure as a bridge-positive
substrate failure: catalog slash controls remain clean, but the exact bridge
itself is not reliable enough across all 40 sources and holdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ELEVENTH_WAVE_OUTCOMES_PATH = (
    ROOT / "data" / "control_surface_knowledge_eleventh_wave_outcomes.json"
)
ELEVENTH_WAVE_REPORT_PATH = (
    ROOT / "research" / "58_CONTROL_SURFACE_KNOWLEDGE_ELEVENTH_WAVE_OUTCOMES.md"
)

TENTH_WAVE_OUTCOMES_PATH = (
    ROOT / "data" / "control_surface_knowledge_tenth_wave_outcomes.json"
)
KSQ015_STRUCTURAL_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ015_CATALOG_SLASH_FULL_SOURCE_PACKET"
    / "ksq015_catalog_slash_full_source_first_run.json"
)
KSQ015_FULL_BEHAVIOR_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ015_CATALOG_SLASH_FULL_SOURCE_PACKET"
    / "ksq015_catalog_slash_full_source_full_behavior.json"
)
KSQ015_STATUS_PATH = (
    ROOT / "research" / "cards" / "KSQ015_CATALOG_SLASH_FULL_SOURCE_PACKET_STATUS.md"
)
KSQ015_PREREG_PATH = (
    ROOT / "research" / "prereg" / "KSQ015_CATALOG_SLASH_FULL_SOURCE_PACKET.md"
)
KSQ015_RUNNER_PATH = ROOT / "code" / "ksq015_catalog_slash_full_source_packet.py"

PANEL_ORDER = [
    "exact_bridge",
    "raw_answer_for_alt",
    "catalog_slash_entity_alt",
    "catalog_slash_decoy_alt",
    "catalog_slash_reversed_entity_alt",
    "catalog_slash_entity_only_control",
    "catalog_slash_decoy_only_control",
    "catalog_slash_reversed_entity_only_control",
    "query_only_control",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")


def failed_criteria(summary: dict[str, Any]) -> list[str]:
    criteria = summary.get("criteria")
    if not isinstance(criteria, dict):
        return []
    return sorted(key for key, value in criteria.items() if value is False)


def panel_metrics(summary: dict[str, Any], panel: str, *, holdout: bool = False) -> dict[str, Any]:
    panel_group = "holdout_panels" if holdout else "panels"
    item = summary["selected_template_summary"][panel_group][panel]
    result = {
        "label_counts": item["label_counts"],
        "parseable_rate": item["parseable_rate"],
        "evidence_answer_rate": item.get("evidence_answer_rate"),
        "raw_answer_channel_reproduced_rate": item.get(
            "raw_answer_channel_reproduced_rate"
        ),
        "slash_alt_override_rate": item.get("slash_alt_override_rate"),
        "any_abstain_rate": item.get("any_abstain_rate"),
        "control_abstain_rate": item.get("control_abstain_rate"),
        "control_reproduced_value_rate": item.get("control_reproduced_value_rate"),
    }
    for key in [
        "mean_target_minus_abstain_logit",
        "mean_abstain_minus_target_logit",
        "mean_alternate_minus_abstain_logit",
    ]:
        if key in item:
            result[key] = item[key]
    return result


def failed_bridge_rows(full: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in full["records"]:
        if record["panel"] not in {
            "exact_bridge",
            "catalog_slash_entity_alt",
            "catalog_slash_decoy_alt",
            "catalog_slash_reversed_entity_alt",
        }:
            continue
        if record.get("selected_label") == "evidence_answer":
            continue
        rows.append(
            {
                "source_id": record["source_id"],
                "split": record["split"],
                "panel": record["panel"],
                "selected_label": record.get("selected_label"),
                "selected_answer": record.get("selected_answer"),
                "first_line": record.get("first_line"),
            }
        )
    return rows


def ksq015_outcome(structural: dict[str, Any], full: dict[str, Any]) -> dict[str, Any]:
    summary = full["summary"]
    decision = summary["catalog_slash_decision"]
    return {
        "outcome_id": "ksq015_catalog_slash_full_source_packet",
        "parent_outcome_id": "ksq014_slash_locality_packet",
        "candidate_id": full["candidate_id"],
        "card_id": full["card_id"],
        "parent_card_id": full["parent_card_id"],
        "wave": "knowledge_eleventh_wave",
        "track": "catalog_slash_full_source_packet",
        "status": "completed",
        "route_decision": decision["route_decision"],
        "diagnostic_class": summary["diagnostic_class"],
        "exported_diagnostic_class": decision["exported_diagnostic_class"],
        "behavior_ready": decision["behavior_ready"],
        "signature_screen_allowed": decision["signature_screen_allowed"],
        "hidden_state_claim_allowed": decision["hidden_state_claim_allowed"],
        "intervention_allowed": decision["intervention_allowed"],
        "mechanism_claim_allowed": decision["mechanism_claim_allowed"],
        "selected_template": summary["selection"]["selected_template"],
        "failed_criteria": failed_criteria(summary),
        "failed_panels": summary["failed_panels"],
        "structural": {
            "path": rel(KSQ015_STRUCTURAL_PATH),
            "record_count": structural["summary"]["structural"]["record_count"],
            "source_count": structural["summary"]["structural"]["source_count"],
            "panel_count": len(structural["summary"]["structural"]["panel_counts"]),
            "template_count": len(structural["summary"]["structural"]["template_counts"]),
            "source_disjoint_holdout": structural["summary"]["criteria"][
                "source_disjoint_holdout"
            ],
        },
        "full_behavior": {
            "path": rel(KSQ015_FULL_BEHAVIOR_PATH),
            "record_count": summary["structural"]["record_count"],
            "source_count": summary["structural"]["source_count"],
            "candidate_and_output_margins_reported": summary["criteria"][
                "candidate_and_output_margins_reported"
            ],
            "panels": {
                panel: panel_metrics(summary, panel) for panel in PANEL_ORDER
            },
            "holdout_panels": {
                panel: panel_metrics(summary, panel, holdout=True)
                for panel in PANEL_ORDER
            },
        },
        "failed_bridge_rows": failed_bridge_rows(full),
        "artifact_paths": {
            "runner": rel(KSQ015_RUNNER_PATH),
            "prereg": rel(KSQ015_PREREG_PATH),
            "status_card": rel(KSQ015_STATUS_PATH),
            "structural_result": rel(KSQ015_STRUCTURAL_PATH),
            "full_behavior_result": rel(KSQ015_FULL_BEHAVIOR_PATH),
        },
        "allowed_claim": (
            "KSQ015 shows that the catalog slash locality smoke does not promote "
            "to a full-source behavior substrate under the tag_rows contract. "
            "The no-bridge catalog slash controls are clean at 40/40 abstain, "
            "and raw answer_for remains active at 39/40, but the exact bridge "
            "positive control itself falls to 35/40 with 5/40 code-token "
            "unparsed outputs. Catalog slash therefore inherits bridge "
            "extraction fragility rather than becoming a reliable behavior "
            "substrate."
        ),
        "forbidden_claims": [
            "KSQ015 is behavior-ready.",
            "KSQ015 licenses hidden-state probing, intervention, or mechanism claims.",
            "Catalog slash notation is promoted as a reliable repair surface.",
            "The KSQ014 smoke result transfers cleanly to full-source coverage.",
        ],
    }


def build_control_surface_knowledge_eleventh_wave_outcomes() -> dict[str, Any]:
    required_paths = [
        TENTH_WAVE_OUTCOMES_PATH,
        KSQ015_STRUCTURAL_PATH,
        KSQ015_FULL_BEHAVIOR_PATH,
        KSQ015_STATUS_PATH,
        KSQ015_PREREG_PATH,
        KSQ015_RUNNER_PATH,
    ]
    missing = [rel(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing eleventh-wave inputs: {missing}")

    structural = load_json(KSQ015_STRUCTURAL_PATH)
    full = load_json(KSQ015_FULL_BEHAVIOR_PATH)
    outcome = ksq015_outcome(structural, full)
    summary = {
        "outcome_count": 1,
        "completed_outcome_count": 1,
        "structural_passed_count": 1,
        "behavior_ready_count": 0,
        "signature_screen_allowed_count": 0,
        "hidden_state_claim_allowed_count": 0,
        "intervention_allowed_count": 0,
        "mechanism_claim_allowed_count": 0,
        "diagnostic_class_counts": {
            outcome["diagnostic_class"]: 1,
        },
        "exported_diagnostic_class_counts": {
            outcome["exported_diagnostic_class"]: 1,
        },
        "selected_template_counts": {
            outcome["selected_template"]: 1,
        },
        "total_structural_rows": outcome["structural"]["record_count"],
        "total_full_behavior_rows": outcome["full_behavior"]["record_count"],
    }
    payload = {
        "schema_version": 1,
        "updated_at": "2026-07-02",
        "layer_id": "control_surface_knowledge_eleventh_wave_outcomes",
        "source_layers": {
            "knowledge_tenth_wave_outcomes": rel(TENTH_WAVE_OUTCOMES_PATH),
        },
        "summary": summary,
        "outcomes": [outcome],
        "diagnostic_chain": {
            "chain_id": "catalog_slash_smoke_to_full_source",
            "source_question": (
                "Does the KSQ014 catalog slash locality survivor become a "
                "full-source behavior substrate when bare slash is removed?"
            ),
            "steps": [
                {
                    "step": 1,
                    "outcome_id": outcome["outcome_id"],
                    "test": "40-source tag_rows bridge with catalog slash entity, decoy, reversed, matched slash-only controls, query-only control, raw answer_for pressure, and candidate margins",
                    "result": outcome["exported_diagnostic_class"],
                    "boundary": (
                        "all catalog slash-only controls abstain 40/40 and raw "
                        "answer_for reproduces 39/40, but exact bridge falls "
                        "to 35/40 and source-disjoint holdout fails by one "
                        "unparsed exact/reversed bridge row"
                    ),
                }
            ],
            "conclusion": (
                "The catalog slash smoke survivor does not widen to a behavior "
                "substrate. The important finding is typed: catalog slash is "
                "local in no-bridge controls, but the underlying counted bridge "
                "surface is not reliable enough at full-source coverage."
            ),
            "next_required_evidence": [
                "Do not start a hidden-state signature screen from KSQ015.",
                "Treat the five code-token rows as bridge-extraction fragility, not slash override.",
                "Search for a bridge contract whose exact positive control clears full-source and holdout before re-testing catalog slash.",
                "Carry CATALOG_SLASH_BRIDGE_POSITIVE_FAILED into the failure taxonomy.",
            ],
        },
        "global_decision": {
            "new_behavior_substrate_admitted": False,
            "new_signature_screen_allowed": False,
            "hidden_state_work_allowed": False,
            "intervention_work_allowed": False,
            "mechanism_claim_allowed": False,
            "route_status": "eleventh_wave_completed_as_full_source_bridge_positive_failure",
            "primary_boundary": "CATALOG_SLASH_BRIDGE_POSITIVE_FAILED",
            "secondary_boundary": "CATALOG_SLASH_CONTROLS_CLEAN_FULL_SOURCE",
        },
    }
    validate_knowledge_eleventh_wave_outcomes(payload)
    return payload


def validate_knowledge_eleventh_wave_outcomes(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("eleventh-wave outcomes schema_version must be 1")
    if payload.get("layer_id") != "control_surface_knowledge_eleventh_wave_outcomes":
        raise ValueError("eleventh-wave outcomes layer_id changed")
    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != 1:
        raise ValueError("eleventh-wave outcomes must contain exactly one outcome")
    outcome = outcomes[0]
    if outcome.get("outcome_id") != "ksq015_catalog_slash_full_source_packet":
        raise ValueError("KSQ015 outcome ID changed")
    if outcome.get("parent_outcome_id") != "ksq014_slash_locality_packet":
        raise ValueError("KSQ015 parent outcome changed")
    if outcome.get("exported_diagnostic_class") != "CATALOG_SLASH_BRIDGE_POSITIVE_FAILED":
        raise ValueError("KSQ015 exported diagnostic changed")
    if outcome.get("selected_template") != "tag_rows":
        raise ValueError("KSQ015 selected template changed")
    if outcome["behavior_ready"] or outcome["hidden_state_claim_allowed"]:
        raise ValueError("KSQ015 must not admit behavior or hidden-state work")
    if (
        outcome["structural"]["record_count"] != 360
        or outcome["full_behavior"]["record_count"] != 360
    ):
        raise ValueError("KSQ015 row counts changed")
    if outcome["failed_panels"] != ["exact_bridge", "catalog_slash_reversed_entity_alt"]:
        raise ValueError("KSQ015 failed panels changed")
    full = outcome["full_behavior"]
    expected_counts = {
        "exact_bridge": {"evidence_answer": 35, "unparsed": 5},
        "raw_answer_for_alt": {
            "raw_answer_channel_reproduced": 39,
            "unparsed": 1,
        },
        "catalog_slash_entity_alt": {
            "abstain": 1,
            "evidence_answer": 36,
            "unparsed": 3,
        },
        "catalog_slash_decoy_alt": {
            "abstain": 3,
            "evidence_answer": 35,
            "unparsed": 2,
        },
        "catalog_slash_reversed_entity_alt": {"evidence_answer": 35, "unparsed": 5},
        "catalog_slash_entity_only_control": {"control_abstain": 40},
        "catalog_slash_decoy_only_control": {"control_abstain": 40},
        "catalog_slash_reversed_entity_only_control": {"control_abstain": 40},
        "query_only_control": {"control_abstain": 40},
    }
    for panel, expected in expected_counts.items():
        if full["panels"][panel]["label_counts"] != expected:
            raise ValueError(f"KSQ015 {panel} counts changed")
    expected_holdout_counts = {
        "exact_bridge": {"evidence_answer": 7, "unparsed": 1},
        "raw_answer_for_alt": {"raw_answer_channel_reproduced": 8},
        "catalog_slash_entity_alt": {"evidence_answer": 8},
        "catalog_slash_decoy_alt": {"abstain": 1, "evidence_answer": 7},
        "catalog_slash_reversed_entity_alt": {"evidence_answer": 7, "unparsed": 1},
        "catalog_slash_entity_only_control": {"control_abstain": 8},
        "catalog_slash_decoy_only_control": {"control_abstain": 8},
        "catalog_slash_reversed_entity_only_control": {"control_abstain": 8},
        "query_only_control": {"control_abstain": 8},
    }
    for panel, expected in expected_holdout_counts.items():
        if full["holdout_panels"][panel]["label_counts"] != expected:
            raise ValueError(f"KSQ015 holdout {panel} counts changed")
    if len(outcome["failed_bridge_rows"]) != 19:
        raise ValueError("KSQ015 failed bridge row count changed")
    summary = payload.get("summary")
    expected_summary = {
        "outcome_count": 1,
        "completed_outcome_count": 1,
        "structural_passed_count": 1,
        "behavior_ready_count": 0,
        "signature_screen_allowed_count": 0,
        "hidden_state_claim_allowed_count": 0,
        "intervention_allowed_count": 0,
        "mechanism_claim_allowed_count": 0,
        "total_structural_rows": 360,
        "total_full_behavior_rows": 360,
    }
    if not isinstance(summary, dict):
        raise ValueError("eleventh-wave outcomes missing summary")
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f"eleventh-wave summary changed: {key}")
    decision = payload.get("global_decision")
    if not isinstance(decision, dict):
        raise ValueError("eleventh-wave outcomes missing global decision")
    if decision.get("hidden_state_work_allowed") is not False:
        raise ValueError("eleventh-wave outcomes must not allow hidden-state work")
    if decision.get("primary_boundary") != "CATALOG_SLASH_BRIDGE_POSITIVE_FAILED":
        raise ValueError("eleventh-wave primary boundary changed")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    chain = payload["diagnostic_chain"]
    outcome = payload["outcomes"][0]
    panels = outcome["full_behavior"]["panels"]
    holdout = outcome["full_behavior"]["holdout_panels"]
    lines = [
        "# Control-Surface Knowledge Eleventh-Wave Outcomes",
        "",
        "Status: generated KSQ015 catalog-slash full-source outcome layer.",
        "",
        "Generated data:",
        "",
        "> `data/control_surface_knowledge_eleventh_wave_outcomes.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_knowledge_eleventh_wave_outcomes.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\ksq015_catalog_slash_full_source_packet.py --write --write-prereg --write-status-card",
        "python code\\ksq015_catalog_slash_full_source_packet.py --score --score-candidates --full-run --write --write-status-card --local-files-only",
        "python code\\control_surface_knowledge_eleventh_wave_outcomes.py --write",
        "python code\\control_surface_knowledge_eleventh_wave_outcomes.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Summary",
        "",
        f"- outcomes: `{summary['outcome_count']}`",
        f"- structural rows checked: `{summary['total_structural_rows']}`",
        f"- full behavior rows checked: `{summary['total_full_behavior_rows']}`",
        f"- behavior-ready outcomes: `{summary['behavior_ready_count']}`",
        f"- signature-screen licenses: `{summary['signature_screen_allowed_count']}`",
        f"- hidden-state licenses: `{summary['hidden_state_claim_allowed_count']}`",
        "",
        "## Diagnostic Chain",
        "",
        chain["conclusion"],
        "",
        "| Step | Outcome | Result | Boundary |",
        "| ---: | --- | --- | --- |",
    ]
    for step in chain["steps"]:
        lines.append(
            f"| {step['step']} | `{step['outcome_id']}` | "
            f"`{step['result']}` | {step['boundary']} |"
        )
    lines.extend(
        [
            "",
            "## Outcome",
            "",
            "| Outcome | Selected Template | Exported Diagnostic | Behavior Ready | Hidden State | Key Boundary |",
            "| --- | --- | --- | --- | --- | --- |",
            f"| `{outcome['outcome_id']}` | `{outcome['selected_template']}` | "
            f"`{outcome['exported_diagnostic_class']}` | "
            f"`{str(outcome['behavior_ready']).lower()}` | "
            f"`{str(outcome['hidden_state_claim_allowed']).lower()}` | "
            "catalog controls clean, exact bridge not full-source reliable |",
            "",
            "## Selected Panel Counts",
            "",
        ]
    )
    panel_labels = {
        "exact_bridge": "exact bridge",
        "raw_answer_for_alt": "raw answer_for alternate",
        "catalog_slash_entity_alt": "catalog slash entity alternate",
        "catalog_slash_decoy_alt": "catalog slash decoy alternate",
        "catalog_slash_reversed_entity_alt": "catalog slash reversed alternate",
        "catalog_slash_entity_only_control": "catalog slash entity-only control",
        "catalog_slash_decoy_only_control": "catalog slash decoy-only control",
        "catalog_slash_reversed_entity_only_control": "catalog slash reversed-only control",
        "query_only_control": "query-only control",
    }
    for panel in PANEL_ORDER:
        lines.append(
            f"- {panel_labels[panel]}: `{json.dumps(panels[panel]['label_counts'], sort_keys=True)}`"
        )
    lines.extend(
        [
            "",
            "## Holdout Counts",
            "",
        ]
    )
    for panel in PANEL_ORDER:
        lines.append(
            f"- {panel_labels[panel]}: `{json.dumps(holdout[panel]['label_counts'], sort_keys=True)}`"
        )
    lines.extend(
        [
            "",
            "## Failed Bridge Rows",
            "",
            "| Source | Split | Panel | Label | First Line |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in outcome["failed_bridge_rows"]:
        first_line = str(row["first_line"]).replace("|", "\\|")
        lines.append(
            f"| `{row['source_id']}` | `{row['split']}` | `{row['panel']}` | "
            f"`{row['selected_label']}` | `{first_line}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "KSQ015 kills the tempting read of KSQ014 as a ready catalog-slash "
            "substrate. The locality controls are clean, but the full-source "
            "positive bridge is not: the model sometimes returns code tokens "
            "instead of values. This is not a slash override; it is bridge "
            "extraction fragility under the current tag_rows contract.",
            "",
            "## Forbidden Claims",
            "",
            "- KSQ015 is behavior-ready.",
            "- KSQ015 licenses hidden-state probing, intervention, or mechanism claims.",
            "- Catalog slash notation is a reliable repair surface.",
            "- The KSQ014 smoke result transfers cleanly to full-source coverage.",
            "",
            "## Next Required Evidence",
            "",
        ]
    )
    for item in chain["next_required_evidence"]:
        lines.append(f"- {item}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    payload = build_control_surface_knowledge_eleventh_wave_outcomes()
    if args.write:
        write_json(ELEVENTH_WAVE_OUTCOMES_PATH, payload)
        write_report(ELEVENTH_WAVE_REPORT_PATH, payload)
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
