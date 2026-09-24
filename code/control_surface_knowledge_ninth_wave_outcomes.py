#!/usr/bin/env python
"""Build the knowledge ninth-wave outcome layer.

KSQ013 follows KSQ012 by removing function-call and assignment syntax from the
candidate repair surfaces. It tests whether materially nonfunction text can
coexist with the counted two-stage codebook bridge without activating a value
channel or damaging bridge lookup. This layer records the result as a
behavior-only diagnostic boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NINTH_WAVE_OUTCOMES_PATH = (
    ROOT / "data" / "control_surface_knowledge_ninth_wave_outcomes.json"
)
NINTH_WAVE_REPORT_PATH = (
    ROOT / "research" / "56_CONTROL_SURFACE_KNOWLEDGE_NINTH_WAVE_OUTCOMES.md"
)

EIGHTH_WAVE_OUTCOMES_PATH = (
    ROOT / "data" / "control_surface_knowledge_eighth_wave_outcomes.json"
)
KSQ013_STRUCTURAL_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ013_NONFUNCTION_REPRESENTATION_SCREEN"
    / "ksq013_nonfunction_representation_screen_first_run.json"
)
KSQ013_SMOKE_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ013_NONFUNCTION_REPRESENTATION_SCREEN"
    / "ksq013_nonfunction_representation_screen_smoke_limit10.json"
)
KSQ013_STATUS_PATH = (
    ROOT / "research" / "cards" / "KSQ013_NONFUNCTION_REPRESENTATION_SCREEN_STATUS.md"
)
KSQ013_PREREG_PATH = (
    ROOT / "research" / "prereg" / "KSQ013_NONFUNCTION_REPRESENTATION_SCREEN.md"
)
KSQ013_RUNNER_PATH = ROOT / "code" / "ksq013_nonfunction_representation_screen.py"


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
        "nonfunction_alt_override_rate": item.get("nonfunction_alt_override_rate"),
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


def ksq013_outcome(structural: dict[str, Any], smoke: dict[str, Any]) -> dict[str, Any]:
    summary = smoke["summary"]
    decision = summary["nonfunction_representation_decision"]
    return {
        "outcome_id": "ksq013_nonfunction_representation_screen",
        "parent_outcome_id": "ksq012_function_assignment_wrapper_repair",
        "candidate_id": smoke["candidate_id"],
        "card_id": smoke["card_id"],
        "parent_card_id": smoke["parent_card_id"],
        "wave": "knowledge_ninth_wave",
        "track": "nonfunction_representation_screen",
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
            "path": rel(KSQ013_STRUCTURAL_PATH),
            "record_count": structural["summary"]["structural"]["record_count"],
            "source_count": structural["summary"]["structural"]["source_count"],
            "panel_count": len(structural["summary"]["structural"]["panel_counts"]),
            "template_count": len(structural["summary"]["structural"]["template_counts"]),
            "source_disjoint_holdout": structural["summary"]["criteria"][
                "source_disjoint_holdout"
            ],
        },
        "smoke": {
            "path": rel(KSQ013_SMOKE_PATH),
            "record_count": summary["structural"]["record_count"],
            "source_count": summary["structural"]["source_count"],
            "candidate_and_output_margins_reported": summary["criteria"][
                "candidate_and_output_margins_reported"
            ],
            "exact_bridge": panel_metrics(summary, "exact_bridge"),
            "raw_answer_for_alt": panel_metrics(summary, "raw_answer_for_alt"),
            "value_bank_alt": panel_metrics(summary, "value_bank_alt"),
            "metadata_value_alt": panel_metrics(summary, "metadata_value_alt"),
            "decoy_entity_pair_alt": panel_metrics(summary, "decoy_entity_pair_alt"),
            "entity_value_separate_alt": panel_metrics(
                summary, "entity_value_separate_alt"
            ),
            "entity_value_slash_alt": panel_metrics(summary, "entity_value_slash_alt"),
            "bare_alt_mention": panel_metrics(summary, "bare_alt_mention"),
            "value_bank_only_control": panel_metrics(
                summary, "value_bank_only_control"
            ),
            "decoy_pair_only_control": panel_metrics(
                summary, "decoy_pair_only_control"
            ),
            "entity_value_separate_only_control": panel_metrics(
                summary, "entity_value_separate_only_control"
            ),
            "query_only_control": panel_metrics(summary, "query_only_control"),
        },
        "artifact_paths": {
            "runner": rel(KSQ013_RUNNER_PATH),
            "prereg": rel(KSQ013_PREREG_PATH),
            "status_card": rel(KSQ013_STATUS_PATH),
            "structural_result": rel(KSQ013_STRUCTURAL_PATH),
            "smoke_result": rel(KSQ013_SMOKE_PATH),
        },
        "allowed_claim": (
            "With the two-stage tag_rows bridge fixed, exact bridge lookup "
            "answers 9/10 and raw answer_for still reproduces the alternate "
            "9/10. Nonfunction forms are less catastrophic than wrappers but "
            "not clean: value bank and decoy-pair bridge panels answer 8/10 "
            "with 2/10 unparsed, metadata answers 7/10, separated entity/value "
            "answers 6/10, bare alternate mention answers 8/10, and separated "
            "entity/value no-bridge control leaks the alternate 4/10. The "
            "catalog slash note is the only nonfunction bridge panel to match "
            "exact bridge in smoke at 9/10, but KSQ013 lacks a slash-only "
            "locality control."
        ),
        "forbidden_claims": [
            "KSQ013 is behavior-ready.",
            "KSQ013 licenses hidden-state probing, intervention, or mechanism claims.",
            "Nonfunction representations are now proven safe repair surfaces.",
            "The catalog slash note is admitted without a matching no-bridge control.",
        ],
    }


def build_control_surface_knowledge_ninth_wave_outcomes() -> dict[str, Any]:
    required_paths = [
        EIGHTH_WAVE_OUTCOMES_PATH,
        KSQ013_STRUCTURAL_PATH,
        KSQ013_SMOKE_PATH,
        KSQ013_STATUS_PATH,
        KSQ013_PREREG_PATH,
        KSQ013_RUNNER_PATH,
    ]
    missing = [rel(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing ninth-wave inputs: {missing}")

    structural = load_json(KSQ013_STRUCTURAL_PATH)
    smoke = load_json(KSQ013_SMOKE_PATH)
    outcome = ksq013_outcome(structural, smoke)
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
        "layer_id": "control_surface_knowledge_ninth_wave_outcomes",
        "source_layers": {
            "knowledge_eighth_wave_outcomes": rel(EIGHTH_WAVE_OUTCOMES_PATH),
        },
        "summary": summary,
        "outcomes": [outcome],
        "diagnostic_chain": {
            "chain_id": "wrapper_failure_to_nonfunction_representation_screen",
            "source_question": (
                "Can alternate-value text be made visible without acting like "
                "a function-assignment answer channel or damaging bridge lookup?"
            ),
            "steps": [
                {
                    "step": 1,
                    "outcome_id": outcome["outcome_id"],
                    "test": "fixed tag_rows codebook bridge with value-bank, metadata, decoy-pair, separated entity/value, catalog-slash, bare-mention, and no-bridge controls",
                    "result": outcome["exported_diagnostic_class"],
                    "boundary": (
                        "exact bridge and raw answer_for both reproduce 9/10; "
                        "most nonfunction bridge panels answer the bridge only "
                        "6/10 to 8/10 because of parse/abstain losses; separated "
                        "entity/value no-bridge control leaks 4/10; catalog slash "
                        "bridge panel answers 9/10 but lacks a slash-only control"
                    ),
                }
            ],
            "conclusion": (
                "Removing function-call and assignment syntax reduces the raw "
                "assignment-channel failure, but it does not yet produce a clean "
                "behavior substrate. The useful new signal is narrower: slash-style "
                "catalog text may preserve bridge lookup in smoke, but it must be "
                "attacked by a matching no-bridge locality control before any "
                "behavior admission."
            ),
            "next_required_evidence": [
                "Build a KSQ014 slash-only locality packet with matched bridge and no-bridge slash panels.",
                "Separate parse loss from genuine abstention loss before treating nonfunction forms as bridge failures.",
                "Keep raw answer_for as a positive-control pressure channel, not as a repair candidate.",
                "Do not license hidden-state screens until a full-source behavior substrate passes bridge, control, holdout, and margin gates.",
            ],
        },
        "global_decision": {
            "new_behavior_substrate_admitted": False,
            "new_signature_screen_allowed": False,
            "hidden_state_work_allowed": False,
            "intervention_work_allowed": False,
            "mechanism_claim_allowed": False,
            "route_status": "ninth_wave_completed_as_nonfunction_diagnostic",
            "primary_boundary": "NONFUNCTION_REPRESENTATION_CONTROL_AND_REPAIR_LEAK",
            "secondary_boundary": "CATALOG_SLASH_PROMISING_BUT_UNCONTROLLED",
        },
    }
    validate_knowledge_ninth_wave_outcomes(payload)
    return payload


def validate_knowledge_ninth_wave_outcomes(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("ninth-wave outcomes schema_version must be 1")
    if payload.get("layer_id") != "control_surface_knowledge_ninth_wave_outcomes":
        raise ValueError("ninth-wave outcomes layer_id changed")
    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != 1:
        raise ValueError("ninth-wave outcomes must contain exactly one outcome")
    outcome = outcomes[0]
    if outcome.get("outcome_id") != "ksq013_nonfunction_representation_screen":
        raise ValueError("KSQ013 outcome ID changed")
    if outcome.get("parent_outcome_id") != "ksq012_function_assignment_wrapper_repair":
        raise ValueError("KSQ013 parent outcome changed")
    if (
        outcome.get("exported_diagnostic_class")
        != "NONFUNCTION_REPRESENTATION_CONTROL_AND_REPAIR_LEAK"
    ):
        raise ValueError("KSQ013 exported diagnostic changed")
    if outcome.get("selected_template") != "tag_rows":
        raise ValueError("KSQ013 selected template changed")
    if outcome["behavior_ready"] or outcome["hidden_state_claim_allowed"]:
        raise ValueError("KSQ013 must not admit behavior or hidden-state work")
    if outcome["structural"]["record_count"] != 480 or outcome["smoke"]["record_count"] != 120:
        raise ValueError("KSQ013 row counts changed")
    if outcome["failed_panels"] != [
        "value_bank_alt",
        "metadata_value_alt",
        "decoy_entity_pair_alt",
        "entity_value_separate_alt",
        "bare_alt_mention",
        "entity_value_separate_only_control",
    ]:
        raise ValueError("KSQ013 failed panels changed")
    smoke = outcome["smoke"]
    expected_counts = {
        "exact_bridge": {"evidence_answer": 9, "unparsed": 1},
        "raw_answer_for_alt": {
            "raw_answer_channel_reproduced": 9,
            "unparsed": 1,
        },
        "value_bank_alt": {"evidence_answer": 8, "unparsed": 2},
        "metadata_value_alt": {"abstain": 1, "evidence_answer": 7, "unparsed": 2},
        "decoy_entity_pair_alt": {"evidence_answer": 8, "unparsed": 2},
        "entity_value_separate_alt": {
            "abstain": 1,
            "evidence_answer": 6,
            "unparsed": 3,
        },
        "entity_value_slash_alt": {"evidence_answer": 9, "unparsed": 1},
        "bare_alt_mention": {"evidence_answer": 8, "unparsed": 2},
        "value_bank_only_control": {"control_abstain": 10},
        "decoy_pair_only_control": {"control_abstain": 10},
        "entity_value_separate_only_control": {
            "control_abstain": 6,
            "control_reproduced_value": 4,
        },
        "query_only_control": {"control_abstain": 10},
    }
    for panel, expected in expected_counts.items():
        if smoke[panel]["label_counts"] != expected:
            raise ValueError(f"KSQ013 {panel} counts changed")
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
        raise ValueError("ninth-wave outcomes missing summary")
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f"ninth-wave summary changed: {key}")
    decision = payload.get("global_decision")
    if not isinstance(decision, dict):
        raise ValueError("ninth-wave outcomes missing global decision")
    if decision.get("hidden_state_work_allowed") is not False:
        raise ValueError("ninth-wave outcomes must not allow hidden-state work")
    if decision.get("primary_boundary") != "NONFUNCTION_REPRESENTATION_CONTROL_AND_REPAIR_LEAK":
        raise ValueError("ninth-wave primary boundary changed")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    chain = payload["diagnostic_chain"]
    outcome = payload["outcomes"][0]
    lines = [
        "# Control-Surface Knowledge Ninth-Wave Outcomes",
        "",
        "Status: generated KSQ013 nonfunction representation outcome layer.",
        "",
        "Generated data:",
        "",
        "> `data/control_surface_knowledge_ninth_wave_outcomes.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_knowledge_ninth_wave_outcomes.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_knowledge_ninth_wave_outcomes.py --write",
        "python code\\control_surface_knowledge_ninth_wave_outcomes.py",
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
            "nonfunction forms reduce wrapper failure but still split bridge "
            "preservation from no-bridge locality |",
            "",
            "## Selected Panel Counts",
            "",
            "- exact bridge: `9/10` answer, `1/10` unparsed",
            "- raw answer_for alternate: `9/10` reproduced, `1/10` unparsed",
            "- value-bank alternate: `8/10` bridge answer, `2/10` unparsed",
            "- metadata-value alternate: `7/10` bridge answer, `1/10` abstain, `2/10` unparsed",
            "- decoy-entity-pair alternate: `8/10` bridge answer, `2/10` unparsed",
            "- separated entity/value alternate: `6/10` bridge answer, `1/10` abstain, `3/10` unparsed",
            "- catalog slash alternate: `9/10` bridge answer, `1/10` unparsed",
            "- bare alternate mention: `8/10` bridge answer, `2/10` unparsed",
            "- value-bank-only control: `10/10` abstain",
            "- decoy-pair-only control: `10/10` abstain",
            "- separated entity/value-only control: `6/10` abstain, `4/10` reproduced value",
            "- query-only control: `10/10` abstain",
            "",
            "## Interpretation",
            "",
            "KSQ013 matters because it turns the reviewer comment into a stronger "
            "diagnostic. The failure is no longer merely that `answer_for` syntax "
            "is active. The failure is that nonfunction value mentions trade off "
            "bridge preservation, parseability, and no-bridge locality. A catalog "
            "slash note is the one smoke-passing bridge form, but it is not a "
            "claim until a slash-only control is built.",
            "",
            "## Forbidden Claims",
            "",
            "- KSQ013 is behavior-ready.",
            "- KSQ013 licenses hidden-state probing, intervention, or mechanism claims.",
            "- Nonfunction representations are safe repair surfaces.",
            "- The catalog slash note is admitted without a slash-only locality control.",
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
    payload = build_control_surface_knowledge_ninth_wave_outcomes()
    if args.write:
        write_json(NINTH_WAVE_OUTCOMES_PATH, payload)
        write_report(NINTH_WAVE_REPORT_PATH, payload)
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
