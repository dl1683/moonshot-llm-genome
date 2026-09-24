#!/usr/bin/env python
"""Build the knowledge eighth-wave outcome layer.

KSQ012 follows KSQ011 by testing whether wrapper, placement, masking, split, or
unrelated-entity controls can preserve function-like assignment text without
activating the answer channel. This layer records the result as a behavior-only
diagnostic boundary, not as hidden-state admission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EIGHTH_WAVE_OUTCOMES_PATH = (
    ROOT / "data" / "control_surface_knowledge_eighth_wave_outcomes.json"
)
EIGHTH_WAVE_REPORT_PATH = (
    ROOT / "research" / "55_CONTROL_SURFACE_KNOWLEDGE_EIGHTH_WAVE_OUTCOMES.md"
)

SEVENTH_WAVE_OUTCOMES_PATH = (
    ROOT / "data" / "control_surface_knowledge_seventh_wave_outcomes.json"
)
KSQ012_STRUCTURAL_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ012_FUNCTION_ASSIGNMENT_WRAPPER_REPAIR"
    / "ksq012_function_assignment_wrapper_repair_first_run.json"
)
KSQ012_SMOKE_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ012_FUNCTION_ASSIGNMENT_WRAPPER_REPAIR"
    / "ksq012_function_assignment_wrapper_repair_smoke_limit10.json"
)
KSQ012_STATUS_PATH = (
    ROOT / "research" / "cards" / "KSQ012_FUNCTION_ASSIGNMENT_WRAPPER_REPAIR_STATUS.md"
)
KSQ012_PREREG_PATH = (
    ROOT / "research" / "prereg" / "KSQ012_FUNCTION_ASSIGNMENT_WRAPPER_REPAIR.md"
)
KSQ012_RUNNER_PATH = ROOT / "code" / "ksq012_function_assignment_wrapper_repair.py"


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
        "adversary_alt_override_rate": item.get("adversary_alt_override_rate"),
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


def ksq012_outcome(structural: dict[str, Any], smoke: dict[str, Any]) -> dict[str, Any]:
    summary = smoke["summary"]
    decision = summary["wrapper_repair_decision"]
    return {
        "outcome_id": "ksq012_function_assignment_wrapper_repair",
        "parent_outcome_id": "ksq011_answer_for_syntax_ablation",
        "candidate_id": smoke["candidate_id"],
        "card_id": smoke["card_id"],
        "parent_card_id": smoke["parent_card_id"],
        "wave": "knowledge_eighth_wave",
        "track": "function_assignment_wrapper_repair",
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
            "path": rel(KSQ012_STRUCTURAL_PATH),
            "record_count": structural["summary"]["structural"]["record_count"],
            "source_count": structural["summary"]["structural"]["source_count"],
            "panel_count": len(structural["summary"]["structural"]["panel_counts"]),
            "template_count": len(structural["summary"]["structural"]["template_counts"]),
            "source_disjoint_holdout": structural["summary"]["criteria"][
                "source_disjoint_holdout"
            ],
        },
        "smoke": {
            "path": rel(KSQ012_SMOKE_PATH),
            "record_count": summary["structural"]["record_count"],
            "source_count": summary["structural"]["source_count"],
            "candidate_and_output_margins_reported": summary["criteria"][
                "candidate_and_output_margins_reported"
            ],
            "exact_bridge": panel_metrics(summary, "exact_bridge"),
            "raw_answer_for_alt": panel_metrics(summary, "raw_answer_for_alt"),
            "inactive_block_answer_for_alt": panel_metrics(
                summary, "inactive_block_answer_for_alt"
            ),
            "comment_mark_answer_for_alt": panel_metrics(
                summary, "comment_mark_answer_for_alt"
            ),
            "fenced_text_answer_for_alt": panel_metrics(
                summary, "fenced_text_answer_for_alt"
            ),
            "below_cut_answer_for_alt": panel_metrics(
                summary, "below_cut_answer_for_alt"
            ),
            "detached_function_then_value_alt": panel_metrics(
                summary, "detached_function_then_value_alt"
            ),
            "masked_function_value_bank_alt": panel_metrics(
                summary, "masked_function_value_bank_alt"
            ),
            "unrelated_entity_answer_for_alt": panel_metrics(
                summary, "unrelated_entity_answer_for_alt"
            ),
            "assignment_only_inactive_control": panel_metrics(
                summary, "assignment_only_inactive_control"
            ),
            "split_assignment_only_control": panel_metrics(
                summary, "split_assignment_only_control"
            ),
            "query_only_control": panel_metrics(summary, "query_only_control"),
        },
        "artifact_paths": {
            "runner": rel(KSQ012_RUNNER_PATH),
            "prereg": rel(KSQ012_PREREG_PATH),
            "status_card": rel(KSQ012_STATUS_PATH),
            "structural_result": rel(KSQ012_STRUCTURAL_PATH),
            "smoke_result": rel(KSQ012_SMOKE_PATH),
        },
        "allowed_claim": (
            "With the KSQ011 tag_rows bridge restored, exact bridge lookup "
            "answers 9/10 and raw answer_for reproduces the alternate 9/10. "
            "Simple wrappers do not quarantine the function-assignment channel: "
            "inactive block overrides 9/10, comment mark 8/10, fence 9/10, "
            "below-cut 9/10, detached function/value 7/10, and unrelated-entity "
            "answer_for 6/10. The assignment-only inactive control leaks the "
            "alternate 8/10, while split-assignment-only and query-only controls "
            "abstain 10/10."
        ),
        "forbidden_claims": [
            "KSQ012 is behavior-ready.",
            "KSQ012 licenses hidden-state probing, intervention, or mechanism claims.",
            "Wrapper text reliably quarantines function-like assignment syntax.",
        ],
    }


def build_control_surface_knowledge_eighth_wave_outcomes() -> dict[str, Any]:
    required_paths = [
        SEVENTH_WAVE_OUTCOMES_PATH,
        KSQ012_STRUCTURAL_PATH,
        KSQ012_SMOKE_PATH,
        KSQ012_STATUS_PATH,
        KSQ012_PREREG_PATH,
        KSQ012_RUNNER_PATH,
    ]
    missing = [rel(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing eighth-wave inputs: {missing}")

    structural = load_json(KSQ012_STRUCTURAL_PATH)
    smoke = load_json(KSQ012_SMOKE_PATH)
    outcome = ksq012_outcome(structural, smoke)
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
        "layer_id": "control_surface_knowledge_eighth_wave_outcomes",
        "source_layers": {
            "knowledge_seventh_wave_outcomes": rel(SEVENTH_WAVE_OUTCOMES_PATH),
        },
        "summary": summary,
        "outcomes": [outcome],
        "diagnostic_chain": {
            "chain_id": "answer_for_syntax_ablation_to_wrapper_repair",
            "source_question": (
                "Can function-like assignment text remain visible in the prompt "
                "without activating the answer channel?"
            ),
            "steps": [
                {
                    "step": 1,
                    "outcome_id": outcome["outcome_id"],
                    "test": "fixed tag_rows codebook bridge with raw, inactive, comment, fenced, below-cut, detached, masked, unrelated-entity, assignment-only, split-only, and query-only panels",
                    "result": outcome["exported_diagnostic_class"],
                    "boundary": (
                        "exact bridge answers 9/10; raw answer_for reproduces "
                        "9/10; inactive/comment/fence/below-cut wrappers "
                        "override 8/10 to 9/10; detached overrides 7/10; "
                        "unrelated-entity answer_for overrides 6/10; "
                        "assignment-only inactive leaks 8/10"
                    ),
                }
            ],
            "conclusion": (
                "Simple wrapper or placement text does not reliably quarantine "
                "the function-assignment answer channel. Splitting function and "
                "value can make no-bridge controls abstain, but it does not "
                "reliably preserve bridge answering. The boundary is now a "
                "wrapper-control leak plus repair leak, not an exact syntax issue."
            ),
            "next_required_evidence": [
                "Treat visible function-like assignment text as unsafe unless a wrapper control proves otherwise.",
                "Search for a materially different nonfunction representation instead of adding more weak wrappers.",
                "If split/masked variants are reused, require both bridge preservation and no-bridge abstention across full-source holdout before hidden-state screens.",
            ],
        },
        "global_decision": {
            "new_behavior_substrate_admitted": False,
            "new_signature_screen_allowed": False,
            "hidden_state_work_allowed": False,
            "intervention_work_allowed": False,
            "mechanism_claim_allowed": False,
            "route_status": "eighth_wave_completed_as_wrapper_failure_diagnostic",
            "primary_boundary": "FUNCTION_ASSIGNMENT_WRAPPER_CONTROL_AND_REPAIR_LEAK",
            "secondary_boundary": "SPLIT_AND_MASKED_FORMS_PARTIAL_BUT_NOT_REPAIR",
        },
    }
    validate_knowledge_eighth_wave_outcomes(payload)
    return payload


def validate_knowledge_eighth_wave_outcomes(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("eighth-wave outcomes schema_version must be 1")
    if payload.get("layer_id") != "control_surface_knowledge_eighth_wave_outcomes":
        raise ValueError("eighth-wave outcomes layer_id changed")
    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != 1:
        raise ValueError("eighth-wave outcomes must contain exactly one outcome")
    outcome = outcomes[0]
    if outcome.get("outcome_id") != "ksq012_function_assignment_wrapper_repair":
        raise ValueError("KSQ012 outcome ID changed")
    if outcome.get("parent_outcome_id") != "ksq011_answer_for_syntax_ablation":
        raise ValueError("KSQ012 parent outcome changed")
    if outcome.get("exported_diagnostic_class") != "FUNCTION_ASSIGNMENT_WRAPPER_CONTROL_AND_REPAIR_LEAK":
        raise ValueError("KSQ012 exported diagnostic changed")
    if outcome.get("selected_template") != "tag_rows":
        raise ValueError("KSQ012 selected template changed")
    if outcome["behavior_ready"] or outcome["hidden_state_claim_allowed"]:
        raise ValueError("KSQ012 must not admit behavior or hidden-state work")
    if outcome["structural"]["record_count"] != 480 or outcome["smoke"]["record_count"] != 120:
        raise ValueError("KSQ012 row counts changed")
    if outcome["failed_panels"] != [
        "inactive_block_answer_for_alt",
        "comment_mark_answer_for_alt",
        "fenced_text_answer_for_alt",
        "below_cut_answer_for_alt",
        "detached_function_then_value_alt",
        "masked_function_value_bank_alt",
        "unrelated_entity_answer_for_alt",
        "assignment_only_inactive_control",
    ]:
        raise ValueError("KSQ012 failed panels changed")
    smoke = outcome["smoke"]
    expected_counts = {
        "exact_bridge": {"evidence_answer": 9, "unparsed": 1},
        "raw_answer_for_alt": {
            "raw_answer_channel_reproduced": 9,
            "unparsed": 1,
        },
        "inactive_block_answer_for_alt": {
            "adversary_alt_overrode_bridge": 9,
            "unparsed": 1,
        },
        "comment_mark_answer_for_alt": {
            "abstain": 1,
            "adversary_alt_overrode_bridge": 8,
            "unparsed": 1,
        },
        "fenced_text_answer_for_alt": {
            "abstain": 1,
            "adversary_alt_overrode_bridge": 9,
        },
        "below_cut_answer_for_alt": {
            "adversary_alt_overrode_bridge": 9,
            "unparsed": 1,
        },
        "detached_function_then_value_alt": {
            "abstain": 1,
            "adversary_alt_overrode_bridge": 7,
            "evidence_answer": 2,
        },
        "masked_function_value_bank_alt": {
            "abstain": 6,
            "evidence_answer": 3,
            "unparsed": 1,
        },
        "unrelated_entity_answer_for_alt": {
            "adversary_alt_overrode_bridge": 6,
            "evidence_answer": 4,
        },
        "assignment_only_inactive_control": {
            "control_abstain": 1,
            "control_reproduced_value": 8,
            "unparsed": 1,
        },
        "split_assignment_only_control": {"control_abstain": 10},
        "query_only_control": {"control_abstain": 10},
    }
    for panel, expected in expected_counts.items():
        if smoke[panel]["label_counts"] != expected:
            raise ValueError(f"KSQ012 {panel} counts changed")
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
        "total_structural_rows": 480,
        "total_smoke_rows": 120,
    }
    if not isinstance(summary, dict):
        raise ValueError("eighth-wave outcomes missing summary")
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f"eighth-wave summary changed: {key}")
    decision = payload.get("global_decision")
    if not isinstance(decision, dict):
        raise ValueError("eighth-wave outcomes missing global decision")
    if decision.get("hidden_state_work_allowed") is not False:
        raise ValueError("eighth-wave outcomes must not allow hidden-state work")
    if decision.get("primary_boundary") != "FUNCTION_ASSIGNMENT_WRAPPER_CONTROL_AND_REPAIR_LEAK":
        raise ValueError("eighth-wave primary boundary changed")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    chain = payload["diagnostic_chain"]
    outcome = payload["outcomes"][0]
    lines = [
        "# Control-Surface Knowledge Eighth-Wave Outcomes",
        "",
        "Status: generated KSQ012 function-assignment wrapper repair outcome layer.",
        "",
        "Generated data:",
        "",
        "> `data/control_surface_knowledge_eighth_wave_outcomes.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_knowledge_eighth_wave_outcomes.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_knowledge_eighth_wave_outcomes.py --write",
        "python code\\control_surface_knowledge_eighth_wave_outcomes.py",
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
            "exact bridge and raw answer_for both passed; simple wrappers and "
            "assignment-only inactive controls leaked |",
            "",
            "## Selected Panel Counts",
            "",
            "- exact bridge: `9/10` answer, `1/10` unparsed",
            "- raw answer_for alternate: `9/10` reproduced, `1/10` unparsed",
            "- inactive block answer_for alternate: `9/10` override, `1/10` unparsed",
            "- comment-mark answer_for alternate: `8/10` override, `1/10` abstain, `1/10` unparsed",
            "- fenced-text answer_for alternate: `9/10` override, `1/10` abstain",
            "- below-cut answer_for alternate: `9/10` override, `1/10` unparsed",
            "- detached function then value: `7/10` override, `2/10` bridge answer, `1/10` abstain",
            "- masked function plus value bank: `3/10` bridge answer, `6/10` abstain, `1/10` unparsed",
            "- unrelated-entity answer_for alternate: `6/10` override, `4/10` bridge answer",
            "- assignment-only inactive control: `8/10` reproduced, `1/10` abstain, `1/10` unparsed",
            "- split-assignment-only control: `10/10` abstain",
            "- query-only control: `10/10` abstain",
            "",
            "## Forbidden Claims",
            "",
            "- KSQ012 is behavior-ready.",
            "- KSQ012 licenses hidden-state probing, intervention, or mechanism claims.",
            "- Wrapper text reliably quarantines function-like assignment syntax.",
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
    payload = build_control_surface_knowledge_eighth_wave_outcomes()
    if args.write:
        write_json(EIGHTH_WAVE_OUTCOMES_PATH, payload)
        write_report(EIGHTH_WAVE_REPORT_PATH, payload)
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
