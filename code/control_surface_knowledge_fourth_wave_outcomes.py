#!/usr/bin/env python
"""Build the knowledge fourth-wave outcome layer.

KSQ007B localized the real-uncertainty answerability leak to answer-bearing
slot-binding syntax. KSQ008 tests the first direct repair: move counted evidence
to a neutral row grammar and make answer_for syntax explicitly forbidden.

This layer records the result as a diagnostic boundary. It is not a hidden-state
admission layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FOURTH_WAVE_OUTCOMES_PATH = (
    ROOT / "data" / "control_surface_knowledge_fourth_wave_outcomes.json"
)
FOURTH_WAVE_REPORT_PATH = (
    ROOT / "research" / "51_CONTROL_SURFACE_KNOWLEDGE_FOURTH_WAVE_OUTCOMES.md"
)

THIRD_WAVE_OUTCOMES_PATH = (
    ROOT / "data" / "control_surface_knowledge_third_wave_outcomes.json"
)
KSQ008_STRUCTURAL_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ008_NEUTRAL_EVIDENCE_CHANNEL_REPAIR"
    / "ksq008_neutral_evidence_channel_repair_first_run.json"
)
KSQ008_SMOKE_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ008_NEUTRAL_EVIDENCE_CHANNEL_REPAIR"
    / "ksq008_neutral_evidence_channel_repair_smoke_limit10.json"
)
KSQ008_STATUS_PATH = (
    ROOT / "research" / "cards" / "KSQ008_NEUTRAL_EVIDENCE_CHANNEL_REPAIR_STATUS.md"
)
KSQ008_PREREG_PATH = (
    ROOT / "research" / "prereg" / "KSQ008_NEUTRAL_EVIDENCE_CHANNEL_REPAIR.md"
)
KSQ008_RUNNER_PATH = ROOT / "code" / "ksq008_neutral_evidence_channel_repair.py"


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
        "any_abstain_rate": item.get("any_abstain_rate"),
        "control_abstain_rate": item.get("control_abstain_rate"),
        "control_reproduced_value_rate": item.get("control_reproduced_value_rate"),
        "forbidden_answer_for_override_rate": item.get(
            "forbidden_answer_for_override_rate"
        ),
    }
    for key in [
        "mean_target_minus_abstain_logit",
        "mean_abstain_minus_target_logit",
    ]:
        if key in item:
            result[key] = item[key]
    return result


def ksq008_outcome(structural: dict[str, Any], smoke: dict[str, Any]) -> dict[str, Any]:
    summary = smoke["summary"]
    decision = summary["neutral_repair_decision"]
    return {
        "outcome_id": "ksq008_neutral_evidence_channel_repair",
        "parent_outcome_id": "ksq007b_claim_channel_boundary",
        "candidate_id": smoke["candidate_id"],
        "card_id": smoke["card_id"],
        "parent_card_id": smoke["parent_card_id"],
        "wave": "knowledge_fourth_wave",
        "track": "neutral_evidence_repair",
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
            "path": rel(KSQ008_STRUCTURAL_PATH),
            "record_count": structural["summary"]["structural"]["record_count"],
            "source_count": structural["summary"]["structural"]["source_count"],
            "panel_count": len(structural["summary"]["structural"]["panel_counts"]),
            "template_count": len(structural["summary"]["structural"]["template_counts"]),
            "source_disjoint_holdout": structural["summary"]["criteria"][
                "source_disjoint_holdout"
            ],
        },
        "smoke": {
            "path": rel(KSQ008_SMOKE_PATH),
            "record_count": summary["structural"]["record_count"],
            "source_count": summary["structural"]["source_count"],
            "candidate_and_output_margins_reported": summary["criteria"][
                "candidate_and_output_margins_reported"
            ],
            "exact_neutral_evidence": panel_metrics(summary, "exact_neutral_evidence"),
            "absent_neutral_evidence": panel_metrics(summary, "absent_neutral_evidence"),
            "unrelated_entity_neutral": panel_metrics(summary, "unrelated_entity_neutral"),
            "conflicting_neutral_evidence": panel_metrics(
                summary, "conflicting_neutral_evidence"
            ),
            "forbidden_bare_answer_for": panel_metrics(
                summary, "forbidden_bare_answer_for"
            ),
            "forbidden_claim_answer_for": panel_metrics(
                summary, "forbidden_claim_answer_for"
            ),
            "forbidden_prose_claim": panel_metrics(summary, "forbidden_prose_claim"),
            "uncounted_neutral_evidence": panel_metrics(
                summary, "uncounted_neutral_evidence"
            ),
            "quoted_neutral_evidence": panel_metrics(summary, "quoted_neutral_evidence"),
            "counted_wrong_schema_answer_for": panel_metrics(
                summary, "counted_wrong_schema_answer_for"
            ),
            "neutral_evidence_vs_forbidden_bare_alt": panel_metrics(
                summary, "neutral_evidence_vs_forbidden_bare_alt"
            ),
            "query_only_control": panel_metrics(summary, "query_only_control"),
        },
        "artifact_paths": {
            "runner": rel(KSQ008_RUNNER_PATH),
            "prereg": rel(KSQ008_PREREG_PATH),
            "status_card": rel(KSQ008_STATUS_PATH),
            "structural_result": rel(KSQ008_STRUCTURAL_PATH),
            "smoke_result": rel(KSQ008_SMOKE_PATH),
        },
        "allowed_claim": (
            "The first neutral-channel repair suppresses many forbidden channels "
            "under the selected field_registry template, but it loses positive "
            "evidence answering and still lets counted answer_for wrong-schema "
            "rows reproduce values on half the smoke rows."
        ),
        "forbidden_claims": [
            "KSQ008 repairs KSQ007B into a behavior-ready substrate.",
            "KSQ008 licenses hidden-state probing, intervention, or mechanism claims.",
            "Neutral evidence grammar alone solves answerability.",
        ],
    }


def build_control_surface_knowledge_fourth_wave_outcomes() -> dict[str, Any]:
    required_paths = [
        THIRD_WAVE_OUTCOMES_PATH,
        KSQ008_STRUCTURAL_PATH,
        KSQ008_SMOKE_PATH,
        KSQ008_STATUS_PATH,
        KSQ008_PREREG_PATH,
        KSQ008_RUNNER_PATH,
    ]
    missing = [rel(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing fourth-wave inputs: {missing}")

    structural = load_json(KSQ008_STRUCTURAL_PATH)
    smoke = load_json(KSQ008_SMOKE_PATH)
    outcome = ksq008_outcome(structural, smoke)
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
        "layer_id": "control_surface_knowledge_fourth_wave_outcomes",
        "source_layers": {
            "knowledge_third_wave_outcomes": rel(THIRD_WAVE_OUTCOMES_PATH),
        },
        "summary": summary,
        "outcomes": [outcome],
        "diagnostic_chain": {
            "chain_id": "slot_binding_boundary_to_neutral_evidence_repair",
            "source_question": (
                "After answer_for syntax is isolated as the leak, does moving "
                "counted evidence into a neutral grammar repair the behavior "
                "substrate?"
            ),
            "steps": [
                {
                    "step": 1,
                    "outcome_id": outcome["outcome_id"],
                    "test": "neutral counted evidence with forbidden answer_for and uncounted neutral controls",
                    "result": outcome["exported_diagnostic_class"],
                    "boundary": (
                        "selected field_registry template answers exact neutral "
                        "evidence only 5/10 and counted wrong-schema answer_for "
                        "rows reproduce 5/10"
                    ),
                }
            ],
            "conclusion": (
                "The first neutral-channel repair does not produce a behavior-ready "
                "substrate. It suppresses bare, claim, prose, uncounted, quoted, "
                "unrelated, absent, and query-only channels under the selected "
                "template, but the allowed neutral grammar is too weak for positive "
                "answering and still admits counted old-schema answer_for rows."
            ),
            "next_required_evidence": [
                "Repair positive neutral-evidence parseability without re-admitting old-schema answer_for rows.",
                "Separate schema-specificity failure from conflict-abstention failure.",
                "Only after a full source-disjoint behavior pass: report margins and consider a signature screen.",
            ],
        },
        "global_decision": {
            "new_behavior_substrate_admitted": False,
            "new_signature_screen_allowed": False,
            "hidden_state_work_allowed": False,
            "intervention_work_allowed": False,
            "mechanism_claim_allowed": False,
            "route_status": "fourth_wave_completed_as_failed_repair",
            "primary_boundary": "NEUTRAL_EVIDENCE_POSITIVE_FAILED",
            "secondary_boundary": "COUNTED_WRONG_SCHEMA_ANSWER_FOR_REPRODUCED",
        },
    }
    validate_knowledge_fourth_wave_outcomes(payload)
    return payload


def validate_knowledge_fourth_wave_outcomes(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("fourth-wave outcomes schema_version must be 1")
    if payload.get("layer_id") != "control_surface_knowledge_fourth_wave_outcomes":
        raise ValueError("fourth-wave outcomes layer_id changed")
    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != 1:
        raise ValueError("fourth-wave outcomes must contain exactly one outcome")
    outcome = outcomes[0]
    if outcome.get("outcome_id") != "ksq008_neutral_evidence_channel_repair":
        raise ValueError("KSQ008 outcome ID changed")
    if outcome.get("parent_outcome_id") != "ksq007b_claim_channel_boundary":
        raise ValueError("KSQ008 parent outcome changed")
    if outcome.get("exported_diagnostic_class") != "NEUTRAL_EVIDENCE_POSITIVE_FAILED":
        raise ValueError("KSQ008 exported diagnostic changed")
    if outcome.get("selected_template") != "field_registry":
        raise ValueError("KSQ008 selected template changed")
    if outcome["behavior_ready"] or outcome["hidden_state_claim_allowed"]:
        raise ValueError("KSQ008 must not admit behavior or hidden-state work")
    if outcome["structural"]["record_count"] != 1440 or outcome["smoke"]["record_count"] != 360:
        raise ValueError("KSQ008 row counts changed")
    if outcome["failed_panels"] != [
        "exact_neutral_evidence",
        "conflicting_neutral_evidence",
        "counted_wrong_schema_answer_for",
        "neutral_evidence_vs_forbidden_bare_alt",
    ]:
        raise ValueError("KSQ008 failed panels changed")
    smoke = outcome["smoke"]
    if smoke["exact_neutral_evidence"]["label_counts"] != {
        "abstain": 5,
        "evidence_answer": 5,
    }:
        raise ValueError("KSQ008 exact neutral evidence counts changed")
    if smoke["counted_wrong_schema_answer_for"]["label_counts"] != {
        "control_abstain": 5,
        "counted_wrong_schema_reproduced": 5,
    }:
        raise ValueError("KSQ008 wrong-schema answer_for counts changed")
    if smoke["neutral_evidence_vs_forbidden_bare_alt"]["label_counts"] != {
        "abstain": 8,
        "evidence_answer": 2,
    }:
        raise ValueError("KSQ008 mixed neutral/forbidden counts changed")
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
        "total_structural_rows": 1440,
        "total_smoke_rows": 360,
    }
    if not isinstance(summary, dict):
        raise ValueError("fourth-wave outcomes missing summary")
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f"fourth-wave summary changed: {key}")
    decision = payload.get("global_decision")
    if not isinstance(decision, dict):
        raise ValueError("fourth-wave outcomes missing global decision")
    if decision.get("hidden_state_work_allowed") is not False:
        raise ValueError("fourth-wave outcomes must not allow hidden-state work")
    if decision.get("primary_boundary") != "NEUTRAL_EVIDENCE_POSITIVE_FAILED":
        raise ValueError("fourth-wave primary boundary changed")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    chain = payload["diagnostic_chain"]
    outcome = payload["outcomes"][0]
    lines = [
        "# Control-Surface Knowledge Fourth-Wave Outcomes",
        "",
        "Status: generated KSQ008 neutral-channel repair outcome layer.",
        "",
        "Generated data:",
        "",
        "> `data/control_surface_knowledge_fourth_wave_outcomes.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_knowledge_fourth_wave_outcomes.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_knowledge_fourth_wave_outcomes.py --write",
        "python code\\control_surface_knowledge_fourth_wave_outcomes.py",
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
            "exact neutral evidence answered `5/10`; counted wrong-schema "
            "`answer_for` reproduced `5/10` |",
            "",
            "## Selected Panel Counts",
            "",
            "- exact neutral evidence: `5/10` answer, `5/10` abstain",
            "- counted wrong-schema answer_for: `5/10` reproduced, `5/10` abstain",
            "- neutral evidence versus forbidden bare alternate: `2/10` answer, `8/10` abstain",
            "- forbidden bare answer_for, claim answer_for, prose claim, uncounted neutral, quoted neutral, absent, unrelated, and query-only controls all abstained `10/10` under the selected template",
            "",
            "## Forbidden Claims",
            "",
            "- KSQ008 is behavior-ready.",
            "- KSQ008 licenses hidden-state probing, intervention, or mechanism claims.",
            "- Neutral evidence grammar alone solves answerability.",
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
    payload = build_control_surface_knowledge_fourth_wave_outcomes()
    if args.write:
        write_json(FOURTH_WAVE_OUTCOMES_PATH, payload)
        write_report(FOURTH_WAVE_REPORT_PATH, payload)
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
