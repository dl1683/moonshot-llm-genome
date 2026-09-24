#!/usr/bin/env python
"""Build the knowledge fifth-wave outcome layer.

KSQ008 showed that neutral evidence grammar did not repair the answer_for
syntax leak. KSQ009 tests a sharper schema-specific version: only ALLOW rows are
counted evidence, and answer_for rows are explicitly declared not to be ALLOW
rows.

This layer records the result as a diagnostic boundary. It is not a hidden-state
admission layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FIFTH_WAVE_OUTCOMES_PATH = (
    ROOT / "data" / "control_surface_knowledge_fifth_wave_outcomes.json"
)
FIFTH_WAVE_REPORT_PATH = (
    ROOT / "research" / "52_CONTROL_SURFACE_KNOWLEDGE_FIFTH_WAVE_OUTCOMES.md"
)

FOURTH_WAVE_OUTCOMES_PATH = (
    ROOT / "data" / "control_surface_knowledge_fourth_wave_outcomes.json"
)
KSQ009_STRUCTURAL_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ009_SCHEMA_SPECIFIC_VALUE_LOOKUP"
    / "ksq009_schema_specific_value_lookup_first_run.json"
)
KSQ009_SMOKE_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ009_SCHEMA_SPECIFIC_VALUE_LOOKUP"
    / "ksq009_schema_specific_value_lookup_smoke_limit10.json"
)
KSQ009_STATUS_PATH = (
    ROOT / "research" / "cards" / "KSQ009_SCHEMA_SPECIFIC_VALUE_LOOKUP_STATUS.md"
)
KSQ009_PREREG_PATH = (
    ROOT / "research" / "prereg" / "KSQ009_SCHEMA_SPECIFIC_VALUE_LOOKUP.md"
)
KSQ009_RUNNER_PATH = ROOT / "code" / "ksq009_schema_specific_value_lookup.py"


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
        "answer_for_alt_override_rate": item.get("answer_for_alt_override_rate"),
    }
    for key in [
        "mean_target_minus_abstain_logit",
        "mean_abstain_minus_target_logit",
    ]:
        if key in item:
            result[key] = item[key]
    return result


def ksq009_outcome(structural: dict[str, Any], smoke: dict[str, Any]) -> dict[str, Any]:
    summary = smoke["summary"]
    decision = summary["schema_specific_decision"]
    return {
        "outcome_id": "ksq009_schema_specific_value_lookup",
        "parent_outcome_id": "ksq008_neutral_evidence_channel_repair",
        "candidate_id": smoke["candidate_id"],
        "card_id": smoke["card_id"],
        "parent_card_id": smoke["parent_card_id"],
        "wave": "knowledge_fifth_wave",
        "track": "schema_specific_value_lookup",
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
            "path": rel(KSQ009_STRUCTURAL_PATH),
            "record_count": structural["summary"]["structural"]["record_count"],
            "source_count": structural["summary"]["structural"]["source_count"],
            "panel_count": len(structural["summary"]["structural"]["panel_counts"]),
            "template_count": len(structural["summary"]["structural"]["template_counts"]),
            "source_disjoint_holdout": structural["summary"]["criteria"][
                "source_disjoint_holdout"
            ],
        },
        "smoke": {
            "path": rel(KSQ009_SMOKE_PATH),
            "record_count": summary["structural"]["record_count"],
            "source_count": summary["structural"]["source_count"],
            "candidate_and_output_margins_reported": summary["criteria"][
                "candidate_and_output_margins_reported"
            ],
            "exact_allowed_value": panel_metrics(summary, "exact_allowed_value"),
            "absent_allowed_value": panel_metrics(summary, "absent_allowed_value"),
            "unrelated_entity_allowed_value": panel_metrics(
                summary, "unrelated_entity_allowed_value"
            ),
            "conflicting_allowed_values": panel_metrics(
                summary, "conflicting_allowed_values"
            ),
            "counted_wrong_schema_answer_for": panel_metrics(
                summary, "counted_wrong_schema_answer_for"
            ),
            "uncounted_wrong_schema_answer_for": panel_metrics(
                summary, "uncounted_wrong_schema_answer_for"
            ),
            "uncounted_allowed_value": panel_metrics(summary, "uncounted_allowed_value"),
            "quoted_allowed_value": panel_metrics(summary, "quoted_allowed_value"),
            "allowed_value_vs_uncounted_answer_for_alt": panel_metrics(
                summary, "allowed_value_vs_uncounted_answer_for_alt"
            ),
            "query_only_control": panel_metrics(summary, "query_only_control"),
        },
        "artifact_paths": {
            "runner": rel(KSQ009_RUNNER_PATH),
            "prereg": rel(KSQ009_PREREG_PATH),
            "status_card": rel(KSQ009_STATUS_PATH),
            "structural_result": rel(KSQ009_STRUCTURAL_PATH),
            "smoke_result": rel(KSQ009_SMOKE_PATH),
        },
        "allowed_claim": (
            "Schema-specific ALLOW row labels do not rescue the behavior "
            "substrate. The selected kv_lines template answers exact ALLOW "
            "rows only 2/10, while wrong-schema answer_for rows still reproduce "
            "7/10 to 9/10 and can override an ALLOW row 9/10."
        ),
        "forbidden_claims": [
            "KSQ009 is behavior-ready.",
            "KSQ009 licenses hidden-state probing, intervention, or mechanism claims.",
            "Adding an explicit ALLOW schema is sufficient to suppress answer_for syntax.",
        ],
    }


def build_control_surface_knowledge_fifth_wave_outcomes() -> dict[str, Any]:
    required_paths = [
        FOURTH_WAVE_OUTCOMES_PATH,
        KSQ009_STRUCTURAL_PATH,
        KSQ009_SMOKE_PATH,
        KSQ009_STATUS_PATH,
        KSQ009_PREREG_PATH,
        KSQ009_RUNNER_PATH,
    ]
    missing = [rel(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing fifth-wave inputs: {missing}")

    structural = load_json(KSQ009_STRUCTURAL_PATH)
    smoke = load_json(KSQ009_SMOKE_PATH)
    outcome = ksq009_outcome(structural, smoke)
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
        "layer_id": "control_surface_knowledge_fifth_wave_outcomes",
        "source_layers": {
            "knowledge_fourth_wave_outcomes": rel(FOURTH_WAVE_OUTCOMES_PATH),
        },
        "summary": summary,
        "outcomes": [outcome],
        "diagnostic_chain": {
            "chain_id": "neutral_evidence_repair_to_schema_specific_value_lookup",
            "source_question": (
                "After neutral evidence fails, does a schema-specific ALLOW "
                "row grammar preserve positive lookup while suppressing "
                "answer_for syntax?"
            ),
            "steps": [
                {
                    "step": 1,
                    "outcome_id": outcome["outcome_id"],
                    "test": "ALLOW rows versus counted, uncounted, quoted, and competing answer_for rows",
                    "result": outcome["exported_diagnostic_class"],
                    "boundary": (
                        "selected kv_lines template answers exact ALLOW rows "
                        "2/10, counted answer_for rows reproduce 9/10, "
                        "uncounted answer_for rows reproduce 7/10, and "
                        "uncounted answer_for alternates override ALLOW rows 9/10"
                    ),
                }
            ],
            "conclusion": (
                "The fifth-wave repair fails more sharply than KSQ008. The "
                "schema-specific ALLOW contract is visible enough for absent, "
                "unrelated, and query-only abstention, but not strong enough "
                "to make positive ALLOW lookup reliable or suppress the old "
                "answer_for micro-language. The answer_for string remains an "
                "answer-bearing channel even when declared out of schema."
            ),
            "next_required_evidence": [
                "Stop treating row labels alone as a likely repair for answerability.",
                "Build an interface where target extraction and counted evidence are separated from answer-like syntax.",
                "Measure whether the answer_for channel is a lexical parser prior, an output-format prior, or a prompt-contract prior before another repair attempt.",
            ],
        },
        "global_decision": {
            "new_behavior_substrate_admitted": False,
            "new_signature_screen_allowed": False,
            "hidden_state_work_allowed": False,
            "intervention_work_allowed": False,
            "mechanism_claim_allowed": False,
            "route_status": "fifth_wave_completed_as_failed_repair",
            "primary_boundary": "SCHEMA_SPECIFIC_POSITIVE_FAILED",
            "secondary_boundary": "ANSWER_FOR_SCHEMA_SPECIFICITY_FAILED",
        },
    }
    validate_knowledge_fifth_wave_outcomes(payload)
    return payload


def validate_knowledge_fifth_wave_outcomes(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("fifth-wave outcomes schema_version must be 1")
    if payload.get("layer_id") != "control_surface_knowledge_fifth_wave_outcomes":
        raise ValueError("fifth-wave outcomes layer_id changed")
    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != 1:
        raise ValueError("fifth-wave outcomes must contain exactly one outcome")
    outcome = outcomes[0]
    if outcome.get("outcome_id") != "ksq009_schema_specific_value_lookup":
        raise ValueError("KSQ009 outcome ID changed")
    if outcome.get("parent_outcome_id") != "ksq008_neutral_evidence_channel_repair":
        raise ValueError("KSQ009 parent outcome changed")
    if outcome.get("exported_diagnostic_class") != "SCHEMA_SPECIFIC_POSITIVE_FAILED":
        raise ValueError("KSQ009 exported diagnostic changed")
    if outcome.get("selected_template") != "kv_lines":
        raise ValueError("KSQ009 selected template changed")
    if outcome["behavior_ready"] or outcome["hidden_state_claim_allowed"]:
        raise ValueError("KSQ009 must not admit behavior or hidden-state work")
    if outcome["structural"]["record_count"] != 1200 or outcome["smoke"]["record_count"] != 300:
        raise ValueError("KSQ009 row counts changed")
    if outcome["failed_panels"] != [
        "exact_allowed_value",
        "conflicting_allowed_values",
        "counted_wrong_schema_answer_for",
        "uncounted_wrong_schema_answer_for",
        "uncounted_allowed_value",
        "quoted_allowed_value",
        "allowed_value_vs_uncounted_answer_for_alt",
    ]:
        raise ValueError("KSQ009 failed panels changed")
    smoke = outcome["smoke"]
    if smoke["exact_allowed_value"]["label_counts"] != {
        "abstain": 8,
        "evidence_answer": 2,
    }:
        raise ValueError("KSQ009 exact allowed value counts changed")
    if smoke["counted_wrong_schema_answer_for"]["label_counts"] != {
        "control_abstain": 1,
        "counted_answer_for_reproduced": 9,
    }:
        raise ValueError("KSQ009 counted answer_for counts changed")
    if smoke["uncounted_wrong_schema_answer_for"]["label_counts"] != {
        "control_abstain": 3,
        "uncounted_answer_for_reproduced": 7,
    }:
        raise ValueError("KSQ009 uncounted answer_for counts changed")
    if smoke["allowed_value_vs_uncounted_answer_for_alt"]["label_counts"] != {
        "answer_for_alt_overrode_allowed_value": 9,
        "unparsed": 1,
    }:
        raise ValueError("KSQ009 mixed ALLOW/answer_for counts changed")
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
        "total_structural_rows": 1200,
        "total_smoke_rows": 300,
    }
    if not isinstance(summary, dict):
        raise ValueError("fifth-wave outcomes missing summary")
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f"fifth-wave summary changed: {key}")
    decision = payload.get("global_decision")
    if not isinstance(decision, dict):
        raise ValueError("fifth-wave outcomes missing global decision")
    if decision.get("hidden_state_work_allowed") is not False:
        raise ValueError("fifth-wave outcomes must not allow hidden-state work")
    if decision.get("primary_boundary") != "SCHEMA_SPECIFIC_POSITIVE_FAILED":
        raise ValueError("fifth-wave primary boundary changed")
    if decision.get("secondary_boundary") != "ANSWER_FOR_SCHEMA_SPECIFICITY_FAILED":
        raise ValueError("fifth-wave secondary boundary changed")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    chain = payload["diagnostic_chain"]
    outcome = payload["outcomes"][0]
    lines = [
        "# Control-Surface Knowledge Fifth-Wave Outcomes",
        "",
        "Status: generated KSQ009 schema-specific value-lookup outcome layer.",
        "",
        "Generated data:",
        "",
        "> `data/control_surface_knowledge_fifth_wave_outcomes.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_knowledge_fifth_wave_outcomes.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_knowledge_fifth_wave_outcomes.py --write",
        "python code\\control_surface_knowledge_fifth_wave_outcomes.py",
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
            "exact ALLOW value answered `2/10`; counted `answer_for` "
            "reproduced `9/10`; uncounted `answer_for` reproduced `7/10`; "
            "uncounted `answer_for` alternate overrode ALLOW `9/10` |",
            "",
            "## Selected Panel Counts",
            "",
            "- exact ALLOW value: `2/10` answer, `8/10` abstain",
            "- counted wrong-schema answer_for: `9/10` reproduced, `1/10` abstain",
            "- uncounted wrong-schema answer_for: `7/10` reproduced, `3/10` abstain",
            "- ALLOW value versus uncounted answer_for alternate: `9/10` answer_for override, `1/10` unparsed",
            "- absent ALLOW, unrelated ALLOW, and query-only controls abstained `10/10`",
            "- uncounted ALLOW reproduced `2/10`; quoted ALLOW reproduced `4/10`; conflicting ALLOW rows selected a value `5/10`",
            "",
            "## Forbidden Claims",
            "",
            "- KSQ009 is behavior-ready.",
            "- KSQ009 licenses hidden-state probing, intervention, or mechanism claims.",
            "- Schema-specific ALLOW labels solve answerability or answer_for leakage.",
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
    payload = build_control_surface_knowledge_fifth_wave_outcomes()
    if args.write:
        write_json(FIFTH_WAVE_OUTCOMES_PATH, payload)
        write_report(FIFTH_WAVE_REPORT_PATH, payload)
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
