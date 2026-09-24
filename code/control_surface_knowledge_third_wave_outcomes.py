#!/usr/bin/env python
"""Build the knowledge third-wave outcome layer.

The second-wave KSQ layer killed the real-uncertainty relation-evidence route
but left an important question open: was the failure mainly familiar-world
prior pressure, or was the answerability grammar itself unstable?

KSQ007 and KSQ007B answer that question as a two-step diagnostic chain:

1. Remove real capital/city priors with nonce entities and nonce values.
2. If claim-only controls still leak, split the claim channel by syntax,
   section boundary, prose, quotes, wrong predicates, mentions, and query-only
   baselines.

This layer makes that chain machine-readable. It is an outcome layer, not a
hidden-state admission layer.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
THIRD_WAVE_OUTCOMES_PATH = (
    ROOT / "data" / "control_surface_knowledge_third_wave_outcomes.json"
)
THIRD_WAVE_REPORT_PATH = (
    ROOT / "research" / "50_CONTROL_SURFACE_KNOWLEDGE_THIRD_WAVE_OUTCOMES.md"
)

SECOND_WAVE_OUTCOMES_PATH = (
    ROOT / "data" / "control_surface_knowledge_second_wave_outcomes.json"
)
KSQ007_STRUCTURAL_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ007_NONCE_EVIDENCE_ANSWERABILITY"
    / "ksq007_nonce_evidence_answerability_first_run.json"
)
KSQ007_SMOKE_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ007_NONCE_EVIDENCE_ANSWERABILITY"
    / "ksq007_nonce_evidence_answerability_smoke_limit10.json"
)
KSQ007_STATUS_PATH = (
    ROOT / "research" / "cards" / "KSQ007_NONCE_EVIDENCE_ANSWERABILITY_STATUS.md"
)
KSQ007_PREREG_PATH = (
    ROOT / "research" / "prereg" / "KSQ007_NONCE_EVIDENCE_ANSWERABILITY.md"
)
KSQ007_RUNNER_PATH = ROOT / "code" / "ksq007_nonce_evidence_answerability_calibrator.py"

KSQ007B_STRUCTURAL_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ007_CLAIM_CHANNEL_BOUNDARY"
    / "ksq007_claim_channel_boundary_first_run.json"
)
KSQ007B_SMOKE_PATH = (
    ROOT
    / "results"
    / "cards"
    / "KSQ007_CLAIM_CHANNEL_BOUNDARY"
    / "ksq007_claim_channel_boundary_smoke_limit10.json"
)
KSQ007B_STATUS_PATH = (
    ROOT / "research" / "cards" / "KSQ007_CLAIM_CHANNEL_BOUNDARY_STATUS.md"
)
KSQ007B_PREREG_PATH = (
    ROOT / "research" / "prereg" / "KSQ007_CLAIM_CHANNEL_BOUNDARY.md"
)
KSQ007B_RUNNER_PATH = ROOT / "code" / "ksq007_claim_channel_boundary_audit.py"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")


def summary_decision(summary: dict[str, Any]) -> dict[str, Any]:
    for key in ["calibrator_decision", "claim_boundary_decision"]:
        value = summary.get(key)
        if isinstance(value, dict):
            return value
    raise ValueError("summary has no recognized decision block")


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
        "control_reproduced_value_rate": item.get("control_reproduced_value_rate"),
        "control_abstain_rate": item.get("control_abstain_rate"),
    }
    for key in [
        "evidence_answer_rate",
        "abstain_rate",
        "conflict_value_selected_rate",
        "mean_target_minus_abstain_logit",
        "mean_abstain_minus_target_logit",
    ]:
        if key in item:
            result[key] = item[key]
    return result


def ksq007_outcome(structural: dict[str, Any], smoke: dict[str, Any]) -> dict[str, Any]:
    summary = smoke["summary"]
    decision = summary_decision(summary)
    panels = summary["selected_template_summary"]["panels"]
    return {
        "outcome_id": "ksq007_nonce_evidence_answerability",
        "parent_outcome_id": "ksq005006_relation_evidence_answerability",
        "candidate_id": smoke["candidate_id"],
        "card_id": smoke["card_id"],
        "wave": "knowledge_third_wave",
        "track": "nonce_evidence_answerability_calibration",
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
        "structural": {
            "path": rel(KSQ007_STRUCTURAL_PATH),
            "record_count": structural["summary"]["structural"]["record_count"],
            "source_count": structural["summary"]["structural"]["source_count"],
            "panel_count": len(structural["summary"]["structural"]["panel_counts"]),
            "template_count": len(structural["summary"]["structural"]["template_counts"]),
            "source_disjoint_holdout": structural["summary"]["criteria"][
                "source_disjoint_holdout"
            ],
        },
        "smoke": {
            "path": rel(KSQ007_SMOKE_PATH),
            "record_count": summary["structural"]["record_count"],
            "source_count": summary["structural"]["source_count"],
            "candidate_and_output_margins_reported": summary["criteria"][
                "candidate_and_output_margins_reported"
            ],
            "exact_evidence_rows": panel_metrics(summary, "exact_evidence_rows"),
            "absent_evidence_rows": panel_metrics(summary, "absent_evidence_rows"),
            "unrelated_entity_rows": panel_metrics(summary, "unrelated_entity_rows"),
            "conflicting_evidence_rows": panel_metrics(summary, "conflicting_evidence_rows"),
            "claim_only_control": panel_metrics(summary, "claim_only_control"),
            "mention_only_control": panel_metrics(summary, "mention_only_control"),
            "query_only_control": panel_metrics(summary, "query_only_control"),
        },
        "artifact_paths": {
            "runner": rel(KSQ007_RUNNER_PATH),
            "prereg": rel(KSQ007_PREREG_PATH),
            "status_card": rel(KSQ007_STATUS_PATH),
            "structural_result": rel(KSQ007_STRUCTURAL_PATH),
            "smoke_result": rel(KSQ007_SMOKE_PATH),
        },
        "allowed_claim": (
            "Removing real capital priors and real city values repairs several "
            "answerability branches under the selected evidence_rows template, "
            "but claim-only text still acts as an answer channel."
        ),
        "forbidden_claims": [
            "KSQ007 is behavior-ready.",
            "KSQ007 licenses hidden-state probing or intervention.",
            "Nonce evidence proves uncertainty, factuality, or knowledge control.",
        ],
    }


def ksq007b_outcome(structural: dict[str, Any], smoke: dict[str, Any]) -> dict[str, Any]:
    summary = smoke["summary"]
    decision = summary_decision(summary)
    return {
        "outcome_id": "ksq007b_claim_channel_boundary",
        "parent_outcome_id": "ksq007_nonce_evidence_answerability",
        "candidate_id": smoke["candidate_id"],
        "card_id": smoke["card_id"],
        "parent_card_id": smoke["parent_card_id"],
        "wave": "knowledge_third_wave",
        "track": "claim_channel_boundary_audit",
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
        "failed_control_panels": summary["failed_control_panels"],
        "structural": {
            "path": rel(KSQ007B_STRUCTURAL_PATH),
            "record_count": structural["summary"]["structural"]["record_count"],
            "source_count": structural["summary"]["structural"]["source_count"],
            "panel_count": len(structural["summary"]["structural"]["panel_counts"]),
            "template_count": len(structural["summary"]["structural"]["template_counts"]),
            "source_disjoint_holdout": structural["summary"]["criteria"][
                "source_disjoint_holdout"
            ],
        },
        "smoke": {
            "path": rel(KSQ007B_SMOKE_PATH),
            "record_count": summary["structural"]["record_count"],
            "source_count": summary["structural"]["source_count"],
            "candidate_and_output_margins_reported": summary["criteria"][
                "candidate_and_output_margins_reported"
            ],
            "exact_evidence_positive": panel_metrics(summary, "exact_evidence_positive"),
            "claim_same_syntax": panel_metrics(summary, "claim_same_syntax"),
            "claim_prose": panel_metrics(summary, "claim_prose"),
            "bare_same_syntax": panel_metrics(summary, "bare_same_syntax"),
            "not_evidence_prefix": panel_metrics(summary, "not_evidence_prefix"),
            "other_block_evidence_row": panel_metrics(summary, "other_block_evidence_row"),
            "quoted_evidence_syntax": panel_metrics(summary, "quoted_evidence_syntax"),
            "wrong_predicate_claim": panel_metrics(summary, "wrong_predicate_claim"),
            "mention_only": panel_metrics(summary, "mention_only"),
            "query_only": panel_metrics(summary, "query_only"),
        },
        "artifact_paths": {
            "runner": rel(KSQ007B_RUNNER_PATH),
            "prereg": rel(KSQ007B_PREREG_PATH),
            "status_card": rel(KSQ007B_STATUS_PATH),
            "structural_result": rel(KSQ007B_STRUCTURAL_PATH),
            "smoke_result": rel(KSQ007B_SMOKE_PATH),
        },
        "allowed_claim": (
            "Counted/uncounted sections suppress literal claim labels, quotes, "
            "wrong predicates, NOT_EVIDENCE rows, mentions, and query-only rows, "
            "but bare answer_for(entity)=value syntax and prose claims still leak."
        ),
        "forbidden_claims": [
            "KSQ007B repairs KSQ007 into a behavior-ready substrate.",
            "KSQ007B licenses hidden-state probing or intervention.",
            "Section labels alone solve claim-channel leakage.",
        ],
    }


def build_chain(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {outcome["outcome_id"]: outcome for outcome in outcomes}
    ksq007 = by_id["ksq007_nonce_evidence_answerability"]
    ksq007b = by_id["ksq007b_claim_channel_boundary"]
    return {
        "chain_id": "real_uncertainty_to_slot_binding_boundary",
        "source_question": (
            "When real-uncertainty relation evidence fails, is the failure "
            "mostly familiar-world prior pressure, or a more general answerability "
            "grammar leak?"
        ),
        "steps": [
            {
                "step": 1,
                "outcome_id": ksq007["outcome_id"],
                "test": "remove real capital and real-city priors with nonce entities and values",
                "result": ksq007["exported_diagnostic_class"],
                "boundary": "claim-only rows still reproduce the nonce value too often",
            },
            {
                "step": 2,
                "outcome_id": ksq007b["outcome_id"],
                "test": "split the claim channel by syntax, section, prose, quote, predicate, mention, and query baselines",
                "result": ksq007b["exported_diagnostic_class"],
                "boundary": "bare answer_for(entity)=value syntax remains strongly answer-bearing",
            },
        ],
        "conclusion": (
            "The current real-uncertainty bottleneck is not exhausted by "
            "familiar priors or value mentions. A slot-binding assertion can "
            "become an answer channel even when the prompt marks it as outside "
            "the counted evidence contract."
        ),
        "next_required_evidence": [
            "A behavior substrate that preserves exact evidence answering while suppressing bare slot-binding claims.",
            "A source-disjoint full run after any claim-channel repair.",
            "Candidate/output margins on the selected repaired template.",
            "Only after behavior repair: hidden signatures that beat prompt/output/candidate baselines.",
        ],
    }


def build_control_surface_knowledge_third_wave_outcomes() -> dict[str, Any]:
    required_paths = [
        SECOND_WAVE_OUTCOMES_PATH,
        KSQ007_STRUCTURAL_PATH,
        KSQ007_SMOKE_PATH,
        KSQ007_STATUS_PATH,
        KSQ007_PREREG_PATH,
        KSQ007_RUNNER_PATH,
        KSQ007B_STRUCTURAL_PATH,
        KSQ007B_SMOKE_PATH,
        KSQ007B_STATUS_PATH,
        KSQ007B_PREREG_PATH,
        KSQ007B_RUNNER_PATH,
    ]
    missing = [rel(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing third-wave inputs: {missing}")

    ksq007_structural = load_json(KSQ007_STRUCTURAL_PATH)
    ksq007_smoke = load_json(KSQ007_SMOKE_PATH)
    ksq007b_structural = load_json(KSQ007B_STRUCTURAL_PATH)
    ksq007b_smoke = load_json(KSQ007B_SMOKE_PATH)
    outcomes = [
        ksq007_outcome(ksq007_structural, ksq007_smoke),
        ksq007b_outcome(ksq007b_structural, ksq007b_smoke),
    ]
    summary = {
        "outcome_count": len(outcomes),
        "completed_outcome_count": sum(1 for outcome in outcomes if outcome["status"] == "completed"),
        "structural_passed_count": len(outcomes),
        "behavior_ready_count": sum(1 for outcome in outcomes if outcome["behavior_ready"]),
        "signature_screen_allowed_count": sum(
            1 for outcome in outcomes if outcome["signature_screen_allowed"]
        ),
        "hidden_state_claim_allowed_count": sum(
            1 for outcome in outcomes if outcome["hidden_state_claim_allowed"]
        ),
        "intervention_allowed_count": sum(1 for outcome in outcomes if outcome["intervention_allowed"]),
        "mechanism_claim_allowed_count": sum(
            1 for outcome in outcomes if outcome["mechanism_claim_allowed"]
        ),
        "diagnostic_class_counts": dict(
            sorted(Counter(outcome["diagnostic_class"] for outcome in outcomes).items())
        ),
        "exported_diagnostic_class_counts": dict(
            sorted(Counter(outcome["exported_diagnostic_class"] for outcome in outcomes).items())
        ),
        "selected_template_counts": dict(
            sorted(Counter(outcome["selected_template"] for outcome in outcomes).items())
        ),
        "total_structural_rows": sum(outcome["structural"]["record_count"] for outcome in outcomes),
        "total_smoke_rows": sum(outcome["smoke"]["record_count"] for outcome in outcomes),
    }
    payload = {
        "schema_version": 1,
        "updated_at": "2026-07-02",
        "layer_id": "control_surface_knowledge_third_wave_outcomes",
        "source_layers": {
            "knowledge_second_wave_outcomes": rel(SECOND_WAVE_OUTCOMES_PATH),
        },
        "summary": summary,
        "outcomes": outcomes,
        "diagnostic_chain": build_chain(outcomes),
        "global_decision": {
            "new_behavior_substrate_admitted": False,
            "new_signature_screen_allowed": False,
            "hidden_state_work_allowed": False,
            "intervention_work_allowed": False,
            "mechanism_claim_allowed": False,
            "route_status": "third_wave_completed_as_diagnostic_boundary",
            "primary_boundary": "ANSWER_FOR_SYNTAX_CLAIM_LEAK",
        },
    }
    validate_knowledge_third_wave_outcomes(payload)
    return payload


def validate_knowledge_third_wave_outcomes(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("third-wave outcomes schema_version must be 1")
    if payload.get("layer_id") != "control_surface_knowledge_third_wave_outcomes":
        raise ValueError("third-wave outcomes layer_id changed")
    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != 2:
        raise ValueError("third-wave outcomes must contain exactly two outcomes")
    by_id = {outcome.get("outcome_id"): outcome for outcome in outcomes}
    if set(by_id) != {
        "ksq007_nonce_evidence_answerability",
        "ksq007b_claim_channel_boundary",
    }:
        raise ValueError("third-wave outcome IDs changed")

    ksq007 = by_id["ksq007_nonce_evidence_answerability"]
    if ksq007["exported_diagnostic_class"] != "NONCE_EVIDENCE_ANSWERABILITY_BOUNDARY":
        raise ValueError("KSQ007 exported diagnostic changed")
    if ksq007["selected_template"] != "evidence_rows":
        raise ValueError("KSQ007 selected template changed")
    if ksq007["behavior_ready"] or ksq007["hidden_state_claim_allowed"]:
        raise ValueError("KSQ007 must not admit behavior or hidden-state work")
    if ksq007["structural"]["record_count"] != 840 or ksq007["smoke"]["record_count"] != 210:
        raise ValueError("KSQ007 row counts changed")
    if ksq007["smoke"]["claim_only_control"]["label_counts"] != {
        "claim_only_reproduced": 6,
        "control_abstain": 4,
    }:
        raise ValueError("KSQ007 claim-only control counts changed")

    ksq007b = by_id["ksq007b_claim_channel_boundary"]
    if ksq007b["parent_outcome_id"] != "ksq007_nonce_evidence_answerability":
        raise ValueError("KSQ007B parent outcome changed")
    if ksq007b["exported_diagnostic_class"] != "ANSWER_FOR_SYNTAX_CLAIM_LEAK":
        raise ValueError("KSQ007B exported diagnostic changed")
    if ksq007b["selected_template"] != "counted_uncounted_sections":
        raise ValueError("KSQ007B selected template changed")
    if ksq007b["behavior_ready"] or ksq007b["hidden_state_claim_allowed"]:
        raise ValueError("KSQ007B must not admit behavior or hidden-state work")
    if ksq007b["structural"]["record_count"] != 1200 or ksq007b["smoke"]["record_count"] != 300:
        raise ValueError("KSQ007B row counts changed")
    if ksq007b["failed_control_panels"] != [
        "claim_prose",
        "bare_same_syntax",
        "other_block_evidence_row",
    ]:
        raise ValueError("KSQ007B failed control panels changed")
    if ksq007b["smoke"]["bare_same_syntax"]["label_counts"] != {
        "bare_same_syntax_reproduced": 7,
        "control_abstain": 3,
    }:
        raise ValueError("KSQ007B bare-syntax counts changed")

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("third-wave outcomes missing summary")
    expected_summary = {
        "outcome_count": 2,
        "completed_outcome_count": 2,
        "structural_passed_count": 2,
        "behavior_ready_count": 0,
        "signature_screen_allowed_count": 0,
        "hidden_state_claim_allowed_count": 0,
        "intervention_allowed_count": 0,
        "mechanism_claim_allowed_count": 0,
        "total_structural_rows": 2040,
        "total_smoke_rows": 510,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f"third-wave summary changed: {key}")
    decision = payload.get("global_decision")
    if not isinstance(decision, dict):
        raise ValueError("third-wave outcomes missing global decision")
    if decision.get("hidden_state_work_allowed") is not False:
        raise ValueError("third-wave outcomes must not allow hidden-state work")
    if decision.get("primary_boundary") != "ANSWER_FOR_SYNTAX_CLAIM_LEAK":
        raise ValueError("third-wave primary boundary changed")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    chain = payload["diagnostic_chain"]
    lines = [
        "# Control-Surface Knowledge Third-Wave Outcomes",
        "",
        "Status: generated post-second-wave diagnostic outcome layer.",
        "",
        "Generated data:",
        "",
        "> `data/control_surface_knowledge_third_wave_outcomes.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_knowledge_third_wave_outcomes.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_knowledge_third_wave_outcomes.py --write",
        "python code\\control_surface_knowledge_third_wave_outcomes.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Summary",
        "",
        f"- outcomes: `{summary['outcome_count']}`",
        f"- completed outcomes: `{summary['completed_outcome_count']}`",
        f"- structural rows checked: `{summary['total_structural_rows']}`",
        f"- smoke rows checked: `{summary['total_smoke_rows']}`",
        f"- behavior-ready outcomes: `{summary['behavior_ready_count']}`",
        f"- signature-screen licenses: `{summary['signature_screen_allowed_count']}`",
        f"- hidden-state licenses: `{summary['hidden_state_claim_allowed_count']}`",
        f"- intervention licenses: `{summary['intervention_allowed_count']}`",
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
            "## Outcomes",
            "",
            "| Outcome | Selected Template | Exported Diagnostic | Behavior Ready | Hidden State | Key Boundary |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for outcome in payload["outcomes"]:
        if outcome["outcome_id"] == "ksq007_nonce_evidence_answerability":
            key_boundary = "claim-only reproduced `6/10` under nonce evidence"
        else:
            key_boundary = "bare `answer_for(entity)=value` reproduced `7/10`"
        lines.append(
            f"| `{outcome['outcome_id']}` | `{outcome['selected_template']}` | "
            f"`{outcome['exported_diagnostic_class']}` | "
            f"`{str(outcome['behavior_ready']).lower()}` | "
            f"`{str(outcome['hidden_state_claim_allowed']).lower()}` | "
            f"{key_boundary} |"
        )
    lines.extend(
        [
            "",
            "## Forbidden Claims",
            "",
            "- KSQ007 or KSQ007B is behavior-ready.",
            "- KSQ007 or KSQ007B licenses hidden-state probing, intervention, or mechanism claims.",
            "- Removing real-world priors is sufficient to solve answerability.",
            "- Section labels alone solve claim-channel leakage.",
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
    payload = build_control_surface_knowledge_third_wave_outcomes()
    if args.write:
        write_json(THIRD_WAVE_OUTCOMES_PATH, payload)
        write_report(THIRD_WAVE_REPORT_PATH, payload)
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
