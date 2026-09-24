#!/usr/bin/env python
"""Build the knowledge tenth-wave outcome layer.

KSQ014 follows KSQ013 by attacking the one constructive nonfunction signal:
catalog slash text. It adds matched slash-only no-bridge controls and separates
catalog slash from bare slash. This layer records the result as behavior-only
diagnostic evidence, not as hidden-state admission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TENTH_WAVE_OUTCOMES_PATH = (
    ROOT / "data" / "control_surface_knowledge_tenth_wave_outcomes.json"
)
TENTH_WAVE_REPORT_PATH = (
    ROOT / "research" / "57_CONTROL_SURFACE_KNOWLEDGE_TENTH_WAVE_OUTCOMES.md"
)

NINTH_WAVE_OUTCOMES_PATH = (
    ROOT / "data" / "control_surface_knowledge_ninth_wave_outcomes.json"
)
KSQ014_STRUCTURAL_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ014_SLASH_LOCALITY_PACKET"
    / "ksq014_slash_locality_packet_first_run.json"
)
KSQ014_SMOKE_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ014_SLASH_LOCALITY_PACKET"
    / "ksq014_slash_locality_packet_smoke_limit10.json"
)
KSQ014_STATUS_PATH = (
    ROOT / "research" / "cards" / "KSQ014_SLASH_LOCALITY_PACKET_STATUS.md"
)
KSQ014_PREREG_PATH = ROOT / "research" / "prereg" / "KSQ014_SLASH_LOCALITY_PACKET.md"
KSQ014_RUNNER_PATH = ROOT / "code" / "ksq014_slash_locality_packet.py"


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


def panel_metrics(summary: dict[str, Any], panel: str) -> dict[str, Any]:
    item = summary["selected_template_summary"]["panels"][panel]
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


def ksq014_outcome(structural: dict[str, Any], smoke: dict[str, Any]) -> dict[str, Any]:
    summary = smoke["summary"]
    decision = summary["slash_locality_decision"]
    return {
        "outcome_id": "ksq014_slash_locality_packet",
        "parent_outcome_id": "ksq013_nonfunction_representation_screen",
        "candidate_id": smoke["candidate_id"],
        "card_id": smoke["card_id"],
        "parent_card_id": smoke["parent_card_id"],
        "wave": "knowledge_tenth_wave",
        "track": "slash_locality_packet",
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
            "path": rel(KSQ014_STRUCTURAL_PATH),
            "record_count": structural["summary"]["structural"]["record_count"],
            "source_count": structural["summary"]["structural"]["source_count"],
            "panel_count": len(structural["summary"]["structural"]["panel_counts"]),
            "template_count": len(structural["summary"]["structural"]["template_counts"]),
            "source_disjoint_holdout": structural["summary"]["criteria"][
                "source_disjoint_holdout"
            ],
        },
        "smoke": {
            "path": rel(KSQ014_SMOKE_PATH),
            "record_count": summary["structural"]["record_count"],
            "source_count": summary["structural"]["source_count"],
            "candidate_and_output_margins_reported": summary["criteria"][
                "candidate_and_output_margins_reported"
            ],
            "exact_bridge": panel_metrics(summary, "exact_bridge"),
            "raw_answer_for_alt": panel_metrics(summary, "raw_answer_for_alt"),
            "catalog_slash_entity_alt": panel_metrics(
                summary, "catalog_slash_entity_alt"
            ),
            "catalog_slash_decoy_alt": panel_metrics(
                summary, "catalog_slash_decoy_alt"
            ),
            "bare_slash_entity_alt": panel_metrics(summary, "bare_slash_entity_alt"),
            "catalog_slash_reversed_entity_alt": panel_metrics(
                summary, "catalog_slash_reversed_entity_alt"
            ),
            "catalog_slash_entity_only_control": panel_metrics(
                summary, "catalog_slash_entity_only_control"
            ),
            "catalog_slash_decoy_only_control": panel_metrics(
                summary, "catalog_slash_decoy_only_control"
            ),
            "bare_slash_entity_only_control": panel_metrics(
                summary, "bare_slash_entity_only_control"
            ),
            "catalog_slash_reversed_entity_only_control": panel_metrics(
                summary, "catalog_slash_reversed_entity_only_control"
            ),
            "query_only_control": panel_metrics(summary, "query_only_control"),
        },
        "artifact_paths": {
            "runner": rel(KSQ014_RUNNER_PATH),
            "prereg": rel(KSQ014_PREREG_PATH),
            "status_card": rel(KSQ014_STATUS_PATH),
            "structural_result": rel(KSQ014_STRUCTURAL_PATH),
            "smoke_result": rel(KSQ014_SMOKE_PATH),
        },
        "allowed_claim": (
            "KSQ014 repaired the KSQ013 missing slash-locality control in smoke. "
            "Exact bridge and raw answer_for both pass at 9/10. Catalog slash "
            "entity, decoy, and reversed variants preserve the bridge at 9/10 "
            "or better by the panel gate, while every matched slash-only "
            "no-bridge control abstains 10/10. Bare slash is not clean: its "
            "bridge panel answers only 6/10 with 4/10 unparsed."
        ),
        "forbidden_claims": [
            "KSQ014 is behavior-ready.",
            "KSQ014 licenses hidden-state probing, intervention, or mechanism claims.",
            "Bare slash notation is a clean repair surface.",
            "Catalog slash notation is promoted without a full-source behavior run.",
        ],
    }


def build_control_surface_knowledge_tenth_wave_outcomes() -> dict[str, Any]:
    required_paths = [
        NINTH_WAVE_OUTCOMES_PATH,
        KSQ014_STRUCTURAL_PATH,
        KSQ014_SMOKE_PATH,
        KSQ014_STATUS_PATH,
        KSQ014_PREREG_PATH,
        KSQ014_RUNNER_PATH,
    ]
    missing = [rel(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing tenth-wave inputs: {missing}")

    structural = load_json(KSQ014_STRUCTURAL_PATH)
    smoke = load_json(KSQ014_SMOKE_PATH)
    outcome = ksq014_outcome(structural, smoke)
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
        "total_smoke_rows": outcome["smoke"]["record_count"],
    }
    payload = {
        "schema_version": 1,
        "updated_at": "2026-07-02",
        "layer_id": "control_surface_knowledge_tenth_wave_outcomes",
        "source_layers": {
            "knowledge_ninth_wave_outcomes": rel(NINTH_WAVE_OUTCOMES_PATH),
        },
        "summary": summary,
        "outcomes": [outcome],
        "diagnostic_chain": {
            "chain_id": "nonfunction_representation_to_slash_locality",
            "source_question": (
                "Does the KSQ013 catalog slash survivor have matched no-bridge "
                "locality, and is bare slash equally safe?"
            ),
            "steps": [
                {
                    "step": 1,
                    "outcome_id": outcome["outcome_id"],
                    "test": "fixed tag_rows bridge with catalog slash entity, decoy, reversed, bare slash, matched slash-only controls, query-only control, and raw answer_for pressure",
                    "result": outcome["exported_diagnostic_class"],
                    "boundary": (
                        "catalog slash bridge variants pass smoke locality; "
                        "all slash-only controls abstain 10/10; bare slash "
                        "bridge fails at 6/10 answer and 4/10 unparsed"
                    ),
                }
            ],
            "conclusion": (
                "The slash result is now split. Catalog-labeled slash text has "
                "a clean 10-source locality smoke under the tested prompt "
                "contract, but bare slash notation is too parse-fragile. This "
                "licenses only a narrow full-source catalog-slash behavior test, "
                "not hidden-state work."
            ),
            "next_required_evidence": [
                "Run a KSQ015 full-source catalog-slash-only packet without bare slash.",
                "Keep raw answer_for as the active positive-control pressure channel.",
                "Require full-source bridge preservation, slash-only abstention, source-disjoint holdout, and margins before any signature screen.",
                "Do not merge catalog slash and bare slash into one nonfunction category.",
            ],
        },
        "global_decision": {
            "new_behavior_substrate_admitted": False,
            "new_signature_screen_allowed": False,
            "hidden_state_work_allowed": False,
            "intervention_work_allowed": False,
            "mechanism_claim_allowed": False,
            "route_status": "tenth_wave_completed_as_catalog_slash_smoke_boundary",
            "primary_boundary": "SLASH_LOCALITY_BRIDGE_LOSS",
            "secondary_boundary": "CATALOG_SLASH_LOCALITY_SMOKE_SURVIVED",
        },
    }
    validate_knowledge_tenth_wave_outcomes(payload)
    return payload


def validate_knowledge_tenth_wave_outcomes(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("tenth-wave outcomes schema_version must be 1")
    if payload.get("layer_id") != "control_surface_knowledge_tenth_wave_outcomes":
        raise ValueError("tenth-wave outcomes layer_id changed")
    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != 1:
        raise ValueError("tenth-wave outcomes must contain exactly one outcome")
    outcome = outcomes[0]
    if outcome.get("outcome_id") != "ksq014_slash_locality_packet":
        raise ValueError("KSQ014 outcome ID changed")
    if outcome.get("parent_outcome_id") != "ksq013_nonfunction_representation_screen":
        raise ValueError("KSQ014 parent outcome changed")
    if outcome.get("exported_diagnostic_class") != "SLASH_LOCALITY_BRIDGE_LOSS":
        raise ValueError("KSQ014 exported diagnostic changed")
    if outcome.get("selected_template") != "tag_rows":
        raise ValueError("KSQ014 selected template changed")
    if outcome["behavior_ready"] or outcome["hidden_state_claim_allowed"]:
        raise ValueError("KSQ014 must not admit behavior or hidden-state work")
    if outcome["structural"]["record_count"] != 440 or outcome["smoke"]["record_count"] != 110:
        raise ValueError("KSQ014 row counts changed")
    if outcome["failed_panels"] != ["bare_slash_entity_alt"]:
        raise ValueError("KSQ014 failed panels changed")
    smoke = outcome["smoke"]
    expected_counts = {
        "exact_bridge": {"evidence_answer": 9, "unparsed": 1},
        "raw_answer_for_alt": {
            "raw_answer_channel_reproduced": 9,
            "unparsed": 1,
        },
        "catalog_slash_entity_alt": {"evidence_answer": 9, "unparsed": 1},
        "catalog_slash_decoy_alt": {"abstain": 1, "evidence_answer": 9},
        "bare_slash_entity_alt": {"evidence_answer": 6, "unparsed": 4},
        "catalog_slash_reversed_entity_alt": {"evidence_answer": 9, "unparsed": 1},
        "catalog_slash_entity_only_control": {"control_abstain": 10},
        "catalog_slash_decoy_only_control": {"control_abstain": 10},
        "bare_slash_entity_only_control": {"control_abstain": 10},
        "catalog_slash_reversed_entity_only_control": {"control_abstain": 10},
        "query_only_control": {"control_abstain": 10},
    }
    for panel, expected in expected_counts.items():
        if smoke[panel]["label_counts"] != expected:
            raise ValueError(f"KSQ014 {panel} counts changed")
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
        "total_structural_rows": 440,
        "total_smoke_rows": 110,
    }
    if not isinstance(summary, dict):
        raise ValueError("tenth-wave outcomes missing summary")
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f"tenth-wave summary changed: {key}")
    decision = payload.get("global_decision")
    if not isinstance(decision, dict):
        raise ValueError("tenth-wave outcomes missing global decision")
    if decision.get("hidden_state_work_allowed") is not False:
        raise ValueError("tenth-wave outcomes must not allow hidden-state work")
    if decision.get("primary_boundary") != "SLASH_LOCALITY_BRIDGE_LOSS":
        raise ValueError("tenth-wave primary boundary changed")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    chain = payload["diagnostic_chain"]
    outcome = payload["outcomes"][0]
    lines = [
        "# Control-Surface Knowledge Tenth-Wave Outcomes",
        "",
        "Status: generated KSQ014 slash locality outcome layer.",
        "",
        "Generated data:",
        "",
        "> `data/control_surface_knowledge_tenth_wave_outcomes.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_knowledge_tenth_wave_outcomes.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_knowledge_tenth_wave_outcomes.py --write",
        "python code\\control_surface_knowledge_tenth_wave_outcomes.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Summary",
        "",
        f"- outcomes: `{summary['outcome_count']}`",
        f"- structural rows checked: `{summary['total_structural_rows']}`",
        f"- smoke rows checked: `{summary['total_smoke_rows']}`",
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
            "catalog slash locality survives smoke; bare slash bridge fails |",
            "",
            "## Selected Panel Counts",
            "",
            "- exact bridge: `9/10` answer, `1/10` unparsed",
            "- raw answer_for alternate: `9/10` reproduced, `1/10` unparsed",
            "- catalog slash entity alternate: `9/10` bridge answer, `1/10` unparsed",
            "- catalog slash decoy alternate: `9/10` bridge answer, `1/10` abstain",
            "- bare slash entity alternate: `6/10` bridge answer, `4/10` unparsed",
            "- catalog slash reversed alternate: `9/10` bridge answer, `1/10` unparsed",
            "- catalog slash entity-only control: `10/10` abstain",
            "- catalog slash decoy-only control: `10/10` abstain",
            "- bare slash entity-only control: `10/10` abstain",
            "- catalog slash reversed-only control: `10/10` abstain",
            "- query-only control: `10/10` abstain",
            "",
            "## Interpretation",
            "",
            "KSQ014 makes the slash result useful by splitting it. Catalog-labeled "
            "slash text passes the missing no-bridge locality smoke; bare slash "
            "does not preserve bridge parseability. The next admissible step is "
            "a full-source catalog-slash-only packet, not hidden-state work.",
            "",
            "## Forbidden Claims",
            "",
            "- KSQ014 is behavior-ready.",
            "- KSQ014 licenses hidden-state probing, intervention, or mechanism claims.",
            "- Bare slash notation is a clean repair surface.",
            "- Catalog slash notation is promoted without a full-source behavior run.",
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
    payload = build_control_surface_knowledge_tenth_wave_outcomes()
    if args.write:
        write_json(TENTH_WAVE_OUTCOMES_PATH, payload)
        write_report(TENTH_WAVE_REPORT_PATH, payload)
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
