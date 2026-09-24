#!/usr/bin/env python
"""Build the knowledge seventh-wave outcome layer.

KSQ011 decomposes the answer_for competition seen in KSQ010 by varying the
adversarial alternate syntax while holding the selected two-stage codebook
interface fixed. This layer records the result as a diagnostic boundary. It is
not a hidden-state admission layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SEVENTH_WAVE_OUTCOMES_PATH = (
    ROOT / "data" / "control_surface_knowledge_seventh_wave_outcomes.json"
)
SEVENTH_WAVE_REPORT_PATH = (
    ROOT / "research" / "54_CONTROL_SURFACE_KNOWLEDGE_SEVENTH_WAVE_OUTCOMES.md"
)

SIXTH_WAVE_OUTCOMES_PATH = (
    ROOT / "data" / "control_surface_knowledge_sixth_wave_outcomes.json"
)
KSQ011_STRUCTURAL_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ011_ANSWER_FOR_SYNTAX_ABLATION"
    / "ksq011_answer_for_syntax_ablation_first_run.json"
)
KSQ011_SMOKE_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ011_ANSWER_FOR_SYNTAX_ABLATION"
    / "ksq011_answer_for_syntax_ablation_smoke_limit10.json"
)
KSQ011_STATUS_PATH = (
    ROOT / "research" / "cards" / "KSQ011_ANSWER_FOR_SYNTAX_ABLATION_STATUS.md"
)
KSQ011_PREREG_PATH = (
    ROOT / "research" / "prereg" / "KSQ011_ANSWER_FOR_SYNTAX_ABLATION.md"
)
KSQ011_RUNNER_PATH = ROOT / "code" / "ksq011_answer_for_syntax_ablation.py"


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
        "adversary_alt_override_rate": item.get("adversary_alt_override_rate"),
    }
    for key in [
        "mean_target_minus_abstain_logit",
        "mean_abstain_minus_target_logit",
        "mean_alternate_minus_abstain_logit",
    ]:
        if key in item:
            result[key] = item[key]
    return result


def ksq011_outcome(structural: dict[str, Any], smoke: dict[str, Any]) -> dict[str, Any]:
    summary = smoke["summary"]
    decision = summary["syntax_ablation_decision"]
    return {
        "outcome_id": "ksq011_answer_for_syntax_ablation",
        "parent_outcome_id": "ksq010_two_stage_codebook_value_lookup",
        "candidate_id": smoke["candidate_id"],
        "card_id": smoke["card_id"],
        "parent_card_id": smoke["parent_card_id"],
        "wave": "knowledge_seventh_wave",
        "track": "answer_for_syntax_ablation",
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
            "path": rel(KSQ011_STRUCTURAL_PATH),
            "record_count": structural["summary"]["structural"]["record_count"],
            "source_count": structural["summary"]["structural"]["source_count"],
            "panel_count": len(structural["summary"]["structural"]["panel_counts"]),
            "template_count": len(structural["summary"]["structural"]["template_counts"]),
            "source_disjoint_holdout": structural["summary"]["criteria"][
                "source_disjoint_holdout"
            ],
        },
        "smoke": {
            "path": rel(KSQ011_SMOKE_PATH),
            "record_count": summary["structural"]["record_count"],
            "source_count": summary["structural"]["source_count"],
            "candidate_and_output_margins_reported": summary["criteria"][
                "candidate_and_output_margins_reported"
            ],
            "exact_bridge": panel_metrics(summary, "exact_bridge"),
            "other_exact_answer_for_alt": panel_metrics(
                summary, "other_exact_answer_for_alt"
            ),
            "other_spaced_answer_for_alt": panel_metrics(
                summary, "other_spaced_answer_for_alt"
            ),
            "other_colon_answer_for_alt": panel_metrics(
                summary, "other_colon_answer_for_alt"
            ),
            "other_answer_to_alt": panel_metrics(summary, "other_answer_to_alt"),
            "other_value_for_alt": panel_metrics(summary, "other_value_for_alt"),
            "other_entity_equals_alt": panel_metrics(summary, "other_entity_equals_alt"),
            "other_prose_value_alt": panel_metrics(summary, "other_prose_value_alt"),
            "other_quoted_answer_for_alt": panel_metrics(
                summary, "other_quoted_answer_for_alt"
            ),
            "other_bare_alt_mention": panel_metrics(summary, "other_bare_alt_mention"),
            "adversary_only_exact_answer_for": panel_metrics(
                summary, "adversary_only_exact_answer_for"
            ),
            "query_only_control": panel_metrics(summary, "query_only_control"),
        },
        "artifact_paths": {
            "runner": rel(KSQ011_RUNNER_PATH),
            "prereg": rel(KSQ011_PREREG_PATH),
            "status_card": rel(KSQ011_STATUS_PATH),
            "structural_result": rel(KSQ011_STRUCTURAL_PATH),
            "smoke_result": rel(KSQ011_SMOKE_PATH),
        },
        "allowed_claim": (
            "With the two-stage tag_rows bridge held fixed, exact bridge lookup "
            "passes the smoke gate at 9/10 while function-like alternate "
            "assignment forms dominate: exact answer_for overrides 9/10, spaced "
            "answer_for 10/10, answer_to 9/10, value_for 8/10, quoted answer_for "
            "9/10, and prose value notes 7/10. Entity assignment and bare value "
            "mention mostly do not override."
        ),
        "forbidden_claims": [
            "KSQ011 is behavior-ready.",
            "KSQ011 licenses hidden-state probing, intervention, or mechanism claims.",
            "The answer_for boundary is merely an exact string artifact.",
        ],
    }


def build_control_surface_knowledge_seventh_wave_outcomes() -> dict[str, Any]:
    required_paths = [
        SIXTH_WAVE_OUTCOMES_PATH,
        KSQ011_STRUCTURAL_PATH,
        KSQ011_SMOKE_PATH,
        KSQ011_STATUS_PATH,
        KSQ011_PREREG_PATH,
        KSQ011_RUNNER_PATH,
    ]
    missing = [rel(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing seventh-wave inputs: {missing}")

    structural = load_json(KSQ011_STRUCTURAL_PATH)
    smoke = load_json(KSQ011_SMOKE_PATH)
    outcome = ksq011_outcome(structural, smoke)
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
        "layer_id": "control_surface_knowledge_seventh_wave_outcomes",
        "source_layers": {
            "knowledge_sixth_wave_outcomes": rel(SIXTH_WAVE_OUTCOMES_PATH),
        },
        "summary": summary,
        "outcomes": [outcome],
        "diagnostic_chain": {
            "chain_id": "two_stage_codebook_to_answer_for_syntax_ablation",
            "source_question": (
                "After answer_for alternates beat codebook bridges, is the "
                "failure an exact lexical artifact or a broader assignment/value "
                "answer channel?"
            ),
            "steps": [
                {
                    "step": 1,
                    "outcome_id": outcome["outcome_id"],
                    "test": "fixed tag_rows codebook bridge with exact, spaced, colon, answer_to, value_for, entity assignment, prose, quoted, and bare value adversaries",
                    "result": outcome["exported_diagnostic_class"],
                    "boundary": (
                        "exact bridge answers 9/10; exact answer_for overrides "
                        "9/10; spaced answer_for overrides 10/10; answer_to "
                        "overrides 9/10; value_for overrides 8/10; prose value "
                        "overrides 7/10; quoted answer_for overrides 9/10"
                    ),
                }
            ],
            "conclusion": (
                "The answer_for boundary is not merely one exact string. It is "
                "a broader function-like assignment answer channel. Plain "
                "entity assignment and bare value mention mostly preserve the "
                "codebook bridge, which separates assignment syntax from generic "
                "value salience."
            ),
            "next_required_evidence": [
                "Repair by removing function-like assignment syntax from adversarial text, not by escaping one exact answer_for spelling.",
                "Test whether sectioning, masking, or delayed adversary placement can preserve function-like text without activating the answer channel.",
                "Do not run hidden-state screens until the same packet passes conflict, locality, and function-assignment controls.",
            ],
        },
        "global_decision": {
            "new_behavior_substrate_admitted": False,
            "new_signature_screen_allowed": False,
            "hidden_state_work_allowed": False,
            "intervention_work_allowed": False,
            "mechanism_claim_allowed": False,
            "route_status": "seventh_wave_completed_as_diagnostic_ablation",
            "primary_boundary": "FUNCTION_ASSIGNMENT_ANSWER_CHANNEL_DOMINANCE",
            "secondary_boundary": "EXACT_BRIDGE_POSITIVE_PASSED",
        },
    }
    validate_knowledge_seventh_wave_outcomes(payload)
    return payload


def validate_knowledge_seventh_wave_outcomes(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("seventh-wave outcomes schema_version must be 1")
    if payload.get("layer_id") != "control_surface_knowledge_seventh_wave_outcomes":
        raise ValueError("seventh-wave outcomes layer_id changed")
    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != 1:
        raise ValueError("seventh-wave outcomes must contain exactly one outcome")
    outcome = outcomes[0]
    if outcome.get("outcome_id") != "ksq011_answer_for_syntax_ablation":
        raise ValueError("KSQ011 outcome ID changed")
    if outcome.get("parent_outcome_id") != "ksq010_two_stage_codebook_value_lookup":
        raise ValueError("KSQ011 parent outcome changed")
    if outcome.get("exported_diagnostic_class") != "FUNCTION_ASSIGNMENT_ANSWER_CHANNEL_DOMINANCE":
        raise ValueError("KSQ011 exported diagnostic changed")
    if outcome.get("selected_template") != "tag_rows":
        raise ValueError("KSQ011 selected template changed")
    if outcome["behavior_ready"] or outcome["hidden_state_claim_allowed"]:
        raise ValueError("KSQ011 must not admit behavior or hidden-state work")
    if outcome["structural"]["record_count"] != 480 or outcome["smoke"]["record_count"] != 120:
        raise ValueError("KSQ011 row counts changed")
    if outcome["failed_panels"] != [
        "other_exact_answer_for_alt",
        "other_spaced_answer_for_alt",
        "other_colon_answer_for_alt",
        "other_answer_to_alt",
        "other_value_for_alt",
        "other_prose_value_alt",
        "other_quoted_answer_for_alt",
        "other_bare_alt_mention",
        "adversary_only_exact_answer_for",
    ]:
        raise ValueError("KSQ011 failed panels changed")
    smoke = outcome["smoke"]
    expected_counts = {
        "exact_bridge": {"evidence_answer": 9, "unparsed": 1},
        "other_exact_answer_for_alt": {
            "adversary_alt_overrode_bridge": 9,
            "unparsed": 1,
        },
        "other_spaced_answer_for_alt": {"adversary_alt_overrode_bridge": 10},
        "other_colon_answer_for_alt": {
            "adversary_alt_overrode_bridge": 9,
            "evidence_answer": 1,
        },
        "other_answer_to_alt": {
            "adversary_alt_overrode_bridge": 9,
            "unparsed": 1,
        },
        "other_value_for_alt": {
            "adversary_alt_overrode_bridge": 8,
            "evidence_answer": 1,
            "unparsed": 1,
        },
        "other_entity_equals_alt": {
            "abstain": 1,
            "evidence_answer": 8,
            "unparsed": 1,
        },
        "other_prose_value_alt": {
            "adversary_alt_overrode_bridge": 7,
            "evidence_answer": 2,
            "unparsed": 1,
        },
        "other_quoted_answer_for_alt": {
            "adversary_alt_overrode_bridge": 9,
            "unparsed": 1,
        },
        "other_bare_alt_mention": {"evidence_answer": 8, "unparsed": 2},
        "adversary_only_exact_answer_for": {
            "adversary_only_answer_for_reproduced": 9,
            "unparsed": 1,
        },
        "query_only_control": {"control_abstain": 10},
    }
    for panel, expected in expected_counts.items():
        if smoke[panel]["label_counts"] != expected:
            raise ValueError(f"KSQ011 {panel} counts changed")
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
        raise ValueError("seventh-wave outcomes missing summary")
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f"seventh-wave summary changed: {key}")
    decision = payload.get("global_decision")
    if not isinstance(decision, dict):
        raise ValueError("seventh-wave outcomes missing global decision")
    if decision.get("hidden_state_work_allowed") is not False:
        raise ValueError("seventh-wave outcomes must not allow hidden-state work")
    if decision.get("primary_boundary") != "FUNCTION_ASSIGNMENT_ANSWER_CHANNEL_DOMINANCE":
        raise ValueError("seventh-wave primary boundary changed")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    chain = payload["diagnostic_chain"]
    outcome = payload["outcomes"][0]
    lines = [
        "# Control-Surface Knowledge Seventh-Wave Outcomes",
        "",
        "Status: generated KSQ011 answer-for syntax ablation outcome layer.",
        "",
        "Generated data:",
        "",
        "> `data/control_surface_knowledge_seventh_wave_outcomes.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_knowledge_seventh_wave_outcomes.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_knowledge_seventh_wave_outcomes.py --write",
        "python code\\control_surface_knowledge_seventh_wave_outcomes.py",
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
            "exact bridge answered `9/10`; function-like adversaries overrode "
            "`7/10` to `10/10`; entity assignment and bare mention mostly did not |",
            "",
            "## Selected Panel Counts",
            "",
            "- exact bridge: `9/10` answer, `1/10` unparsed",
            "- exact answer_for alternate: `9/10` override, `1/10` unparsed",
            "- spaced answer_for alternate: `10/10` override",
            "- colon answer_for alternate: `9/10` override, `1/10` bridge answer",
            "- answer_to alternate: `9/10` override, `1/10` unparsed",
            "- value_for alternate: `8/10` override, `1/10` bridge answer, `1/10` unparsed",
            "- prose value alternate: `7/10` override, `2/10` bridge answer, `1/10` unparsed",
            "- quoted answer_for alternate: `9/10` override, `1/10` unparsed",
            "- entity assignment: `8/10` bridge answer, `1/10` abstain, `1/10` unparsed",
            "- bare alternate mention: `8/10` bridge answer, `2/10` unparsed",
            "- adversary-only exact answer_for reproduced `9/10`; query-only abstained `10/10`",
            "",
            "## Forbidden Claims",
            "",
            "- KSQ011 is behavior-ready.",
            "- KSQ011 licenses hidden-state probing, intervention, or mechanism claims.",
            "- The answer_for boundary is merely one exact string artifact.",
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
    payload = build_control_surface_knowledge_seventh_wave_outcomes()
    if args.write:
        write_json(SEVENTH_WAVE_OUTCOMES_PATH, payload)
        write_report(SEVENTH_WAVE_REPORT_PATH, payload)
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
