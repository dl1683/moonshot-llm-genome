#!/usr/bin/env python
"""Build the knowledge sixth-wave outcome layer.

KSQ010 tests the repair implied by KSQ009: separate entity targeting from value
selection with an opaque two-stage codebook. This layer records the result as a
diagnostic boundary. It is not a hidden-state admission layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SIXTH_WAVE_OUTCOMES_PATH = (
    ROOT / "data" / "control_surface_knowledge_sixth_wave_outcomes.json"
)
SIXTH_WAVE_REPORT_PATH = (
    ROOT / "research" / "53_CONTROL_SURFACE_KNOWLEDGE_SIXTH_WAVE_OUTCOMES.md"
)

FIFTH_WAVE_OUTCOMES_PATH = (
    ROOT / "data" / "control_surface_knowledge_fifth_wave_outcomes.json"
)
KSQ010_STRUCTURAL_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ010_TWO_STAGE_CODEBOOK_VALUE_LOOKUP"
    / "ksq010_two_stage_codebook_value_lookup_first_run.json"
)
KSQ010_SMOKE_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ010_TWO_STAGE_CODEBOOK_VALUE_LOOKUP"
    / "ksq010_two_stage_codebook_value_lookup_smoke_limit10.json"
)
KSQ010_STATUS_PATH = (
    ROOT / "research" / "cards" / "KSQ010_TWO_STAGE_CODEBOOK_VALUE_LOOKUP_STATUS.md"
)
KSQ010_PREREG_PATH = (
    ROOT / "research" / "prereg" / "KSQ010_TWO_STAGE_CODEBOOK_VALUE_LOOKUP.md"
)
KSQ010_RUNNER_PATH = ROOT / "code" / "ksq010_two_stage_codebook_value_lookup.py"


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


def ksq010_outcome(structural: dict[str, Any], smoke: dict[str, Any]) -> dict[str, Any]:
    summary = smoke["summary"]
    decision = summary["codebook_decision"]
    return {
        "outcome_id": "ksq010_two_stage_codebook_value_lookup",
        "parent_outcome_id": "ksq009_schema_specific_value_lookup",
        "candidate_id": smoke["candidate_id"],
        "card_id": smoke["card_id"],
        "parent_card_id": smoke["parent_card_id"],
        "wave": "knowledge_sixth_wave",
        "track": "two_stage_codebook_value_lookup",
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
            "path": rel(KSQ010_STRUCTURAL_PATH),
            "record_count": structural["summary"]["structural"]["record_count"],
            "source_count": structural["summary"]["structural"]["source_count"],
            "panel_count": len(structural["summary"]["structural"]["panel_counts"]),
            "template_count": len(structural["summary"]["structural"]["template_counts"]),
            "source_disjoint_holdout": structural["summary"]["criteria"][
                "source_disjoint_holdout"
            ],
        },
        "smoke": {
            "path": rel(KSQ010_SMOKE_PATH),
            "record_count": summary["structural"]["record_count"],
            "source_count": summary["structural"]["source_count"],
            "candidate_and_output_margins_reported": summary["criteria"][
                "candidate_and_output_margins_reported"
            ],
            "exact_codebook_bridge": panel_metrics(summary, "exact_codebook_bridge"),
            "missing_entity_code": panel_metrics(summary, "missing_entity_code"),
            "missing_code_value": panel_metrics(summary, "missing_code_value"),
            "unrelated_entity_code": panel_metrics(summary, "unrelated_entity_code"),
            "conflicting_entity_codes": panel_metrics(summary, "conflicting_entity_codes"),
            "conflicting_code_values": panel_metrics(summary, "conflicting_code_values"),
            "uncounted_entity_code": panel_metrics(summary, "uncounted_entity_code"),
            "uncounted_code_value": panel_metrics(summary, "uncounted_code_value"),
            "quoted_bridge_rows": panel_metrics(summary, "quoted_bridge_rows"),
            "counted_answer_for_only": panel_metrics(summary, "counted_answer_for_only"),
            "bridge_vs_answer_for_alt": panel_metrics(
                summary, "bridge_vs_answer_for_alt"
            ),
            "query_only_control": panel_metrics(summary, "query_only_control"),
        },
        "artifact_paths": {
            "runner": rel(KSQ010_RUNNER_PATH),
            "prereg": rel(KSQ010_PREREG_PATH),
            "status_card": rel(KSQ010_STATUS_PATH),
            "structural_result": rel(KSQ010_STRUCTURAL_PATH),
            "smoke_result": rel(KSQ010_SMOKE_PATH),
        },
        "allowed_claim": (
            "The two-stage codebook interface improves positive lookup relative "
            "to KSQ009 under the selected tag_rows template, but it still fails "
            "the behavior gate: exact bridges answer 8/10 with 2/10 unparsed, "
            "conflicts select values 10/10, counted answer_for rows reproduce "
            "10/10, and answer_for alternates override bridges 9/10."
        ),
        "forbidden_claims": [
            "KSQ010 is behavior-ready.",
            "KSQ010 licenses hidden-state probing, intervention, or mechanism claims.",
            "Separating entity-code and code-value rows is sufficient to suppress answer_for syntax.",
        ],
    }


def build_control_surface_knowledge_sixth_wave_outcomes() -> dict[str, Any]:
    required_paths = [
        FIFTH_WAVE_OUTCOMES_PATH,
        KSQ010_STRUCTURAL_PATH,
        KSQ010_SMOKE_PATH,
        KSQ010_STATUS_PATH,
        KSQ010_PREREG_PATH,
        KSQ010_RUNNER_PATH,
    ]
    missing = [rel(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing sixth-wave inputs: {missing}")

    structural = load_json(KSQ010_STRUCTURAL_PATH)
    smoke = load_json(KSQ010_SMOKE_PATH)
    outcome = ksq010_outcome(structural, smoke)
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
        "layer_id": "control_surface_knowledge_sixth_wave_outcomes",
        "source_layers": {
            "knowledge_fifth_wave_outcomes": rel(FIFTH_WAVE_OUTCOMES_PATH),
        },
        "summary": summary,
        "outcomes": [outcome],
        "diagnostic_chain": {
            "chain_id": "schema_specific_value_lookup_to_two_stage_codebook",
            "source_question": (
                "After schema labels fail, does separating entity targeting "
                "from value selection through an opaque codebook preserve "
                "positive lookup while suppressing answer_for syntax?"
            ),
            "steps": [
                {
                    "step": 1,
                    "outcome_id": outcome["outcome_id"],
                    "test": "entity-code rows plus code-value rows versus missing, conflict, locality, quoted, and answer_for controls",
                    "result": outcome["exported_diagnostic_class"],
                    "boundary": (
                        "selected tag_rows template answers exact bridges 8/10 "
                        "but leaves 2/10 unparsed; both conflict panels select "
                        "a value 10/10; answer_for-only rows reproduce 10/10; "
                        "answer_for alternates override bridges 9/10"
                    ),
                }
            ],
            "conclusion": (
                "The two-stage codebook interface moves the positive branch "
                "closer to viability, but it does not repair the substrate. "
                "The model can sometimes follow the entity-code/code-value "
                "path, yet the same prompt contract fails conflict abstention, "
                "locality, quoted-row exclusion, and answer_for competition."
            ),
            "next_required_evidence": [
                "Stop treating positive lookup alone as a useful admission signal for this branch.",
                "Probe whether answer_for competition is lexical, answer-position, or value-salience driven with answer_for string ablations.",
                "Repair conflict abstention and row locality before any full behavior run or hidden-state screen.",
            ],
        },
        "global_decision": {
            "new_behavior_substrate_admitted": False,
            "new_signature_screen_allowed": False,
            "hidden_state_work_allowed": False,
            "intervention_work_allowed": False,
            "mechanism_claim_allowed": False,
            "route_status": "sixth_wave_completed_as_failed_repair",
            "primary_boundary": "CODEBOOK_POSITIVE_FAILED",
            "secondary_boundary": "ANSWER_FOR_COMPETES_WITH_CODEBOOK_BRIDGE",
        },
    }
    validate_knowledge_sixth_wave_outcomes(payload)
    return payload


def validate_knowledge_sixth_wave_outcomes(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("sixth-wave outcomes schema_version must be 1")
    if payload.get("layer_id") != "control_surface_knowledge_sixth_wave_outcomes":
        raise ValueError("sixth-wave outcomes layer_id changed")
    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != 1:
        raise ValueError("sixth-wave outcomes must contain exactly one outcome")
    outcome = outcomes[0]
    if outcome.get("outcome_id") != "ksq010_two_stage_codebook_value_lookup":
        raise ValueError("KSQ010 outcome ID changed")
    if outcome.get("parent_outcome_id") != "ksq009_schema_specific_value_lookup":
        raise ValueError("KSQ010 parent outcome changed")
    if outcome.get("exported_diagnostic_class") != "CODEBOOK_POSITIVE_FAILED":
        raise ValueError("KSQ010 exported diagnostic changed")
    if outcome.get("selected_template") != "tag_rows":
        raise ValueError("KSQ010 selected template changed")
    if outcome["behavior_ready"] or outcome["hidden_state_claim_allowed"]:
        raise ValueError("KSQ010 must not admit behavior or hidden-state work")
    if outcome["structural"]["record_count"] != 1440 or outcome["smoke"]["record_count"] != 360:
        raise ValueError("KSQ010 row counts changed")
    if outcome["failed_panels"] != [
        "exact_codebook_bridge",
        "missing_code_value",
        "conflicting_entity_codes",
        "conflicting_code_values",
        "uncounted_entity_code",
        "uncounted_code_value",
        "quoted_bridge_rows",
        "counted_answer_for_only",
        "bridge_vs_answer_for_alt",
    ]:
        raise ValueError("KSQ010 failed panels changed")
    smoke = outcome["smoke"]
    if smoke["exact_codebook_bridge"]["label_counts"] != {
        "evidence_answer": 8,
        "unparsed": 2,
    }:
        raise ValueError("KSQ010 exact codebook counts changed")
    if smoke["conflicting_entity_codes"]["label_counts"] != {
        "conflict_value_selected": 10,
    }:
        raise ValueError("KSQ010 conflicting entity-code counts changed")
    if smoke["conflicting_code_values"]["label_counts"] != {
        "conflict_value_selected": 10,
    }:
        raise ValueError("KSQ010 conflicting code-value counts changed")
    if smoke["counted_answer_for_only"]["label_counts"] != {
        "counted_answer_for_reproduced": 10,
    }:
        raise ValueError("KSQ010 counted answer_for counts changed")
    if smoke["bridge_vs_answer_for_alt"]["label_counts"] != {
        "answer_for_alt_overrode_bridge": 9,
        "unparsed": 1,
    }:
        raise ValueError("KSQ010 bridge versus answer_for counts changed")
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
        raise ValueError("sixth-wave outcomes missing summary")
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f"sixth-wave summary changed: {key}")
    decision = payload.get("global_decision")
    if not isinstance(decision, dict):
        raise ValueError("sixth-wave outcomes missing global decision")
    if decision.get("hidden_state_work_allowed") is not False:
        raise ValueError("sixth-wave outcomes must not allow hidden-state work")
    if decision.get("primary_boundary") != "CODEBOOK_POSITIVE_FAILED":
        raise ValueError("sixth-wave primary boundary changed")
    if decision.get("secondary_boundary") != "ANSWER_FOR_COMPETES_WITH_CODEBOOK_BRIDGE":
        raise ValueError("sixth-wave secondary boundary changed")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    chain = payload["diagnostic_chain"]
    outcome = payload["outcomes"][0]
    lines = [
        "# Control-Surface Knowledge Sixth-Wave Outcomes",
        "",
        "Status: generated KSQ010 two-stage codebook outcome layer.",
        "",
        "Generated data:",
        "",
        "> `data/control_surface_knowledge_sixth_wave_outcomes.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_knowledge_sixth_wave_outcomes.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_knowledge_sixth_wave_outcomes.py --write",
        "python code\\control_surface_knowledge_sixth_wave_outcomes.py",
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
            "exact bridges answered `8/10`; conflict panels selected values "
            "`10/10`; answer_for-only reproduced `10/10`; answer_for alternate "
            "overrode bridge `9/10` |",
            "",
            "## Selected Panel Counts",
            "",
            "- exact codebook bridge: `8/10` answer, `2/10` unparsed",
            "- missing entity-code: `9/10` abstain, `1/10` unparsed",
            "- missing code-value: `10/10` unparsed",
            "- unrelated entity-code and query-only controls abstained `10/10`",
            "- conflicting entity-code and code-value panels selected a value `10/10` each",
            "- counted answer_for-only reproduced `10/10`",
            "- bridge versus answer_for alternate: `9/10` answer_for override, `1/10` unparsed",
            "- uncounted code-value reproduced `4/10`; quoted bridge reproduced `3/10`",
            "",
            "## Forbidden Claims",
            "",
            "- KSQ010 is behavior-ready.",
            "- KSQ010 licenses hidden-state probing, intervention, or mechanism claims.",
            "- Two-stage codebook rows solve answerability or answer_for leakage.",
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
    payload = build_control_surface_knowledge_sixth_wave_outcomes()
    if args.write:
        write_json(SIXTH_WAVE_OUTCOMES_PATH, payload)
        write_report(SIXTH_WAVE_REPORT_PATH, payload)
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
