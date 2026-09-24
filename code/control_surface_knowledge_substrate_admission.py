"""Build the knowledge-substrate admission protocol.

The knowledge gap plan identifies levels that cannot advance without a new
behavior substrate. This layer defines the front-door protocol for those
future substrates: the gates they must clear before any hidden-state signature
or intervention work is licensed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from control_surface_artifacts import ROOT, load_json
from control_surface_bridge_ladder import BRIDGE_LADDER_PATH
from control_surface_gate_geometry import GATE_GEOMETRY_PATH
from control_surface_knowledge_gap_plan import KNOWLEDGE_GAP_PLAN_PATH
from control_surface_smoke_diagnostics import SMOKE_DIAGNOSTICS_PATH


KNOWLEDGE_SUBSTRATE_ADMISSION_PATH = (
    ROOT / "data" / "control_surface_knowledge_substrate_admission.json"
)
KNOWLEDGE_SUBSTRATE_ADMISSION_REPORT_PATH = (
    ROOT / "research" / "44_CONTROL_SURFACE_KNOWLEDGE_SUBSTRATE_ADMISSION.md"
)

ADMISSION_GATE_ORDER = [
    "material_novelty",
    "behavior_contract",
    "parseability_and_label_balance",
    "direct_controls",
    "conflict_mixture",
    "null_rows",
    "source_disjoint_holdout",
    "prompt_channel_locality",
    "output_candidate_baselines",
    "side_effect_and_leakage",
    "split_freeze",
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def by_key(items: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        item_key = item[key]
        if item_key in result:
            raise AssertionError(f"duplicate {key}: {item_key}")
        result[item_key] = item
    return result


def gate(
    gate_id: str,
    status: str,
    requirement: str,
    evidence_required: list[str],
    kill_condition: str,
) -> dict[str, Any]:
    if gate_id not in ADMISSION_GATE_ORDER:
        raise AssertionError(f"unknown admission gate {gate_id}")
    return {
        "gate_id": gate_id,
        "order": ADMISSION_GATE_ORDER.index(gate_id) + 1,
        "status": status,
        "requirement": requirement,
        "evidence_required": evidence_required,
        "kill_condition": kill_condition,
    }


def common_behavior_gates(level_id: str) -> list[dict[str, Any]]:
    return [
        gate(
            "behavior_contract",
            "required_before_signature",
            "Predeclare the exact prompt contract, answer interface, labels, parser, and allowed claim before any hidden-state collection.",
            [
                "frozen prompt templates",
                "frozen parser or scorer",
                "explicit positive, negative, null, and holdout panels",
                "declared allowed and forbidden claims",
            ],
            "Kill or redesign if the behavior cannot be scored without post-hoc parser or label changes.",
        ),
        gate(
            "parseability_and_label_balance",
            "required_before_signature",
            "Show parseability, label balance, and split balance on discovery, calibration, and holdout rows.",
            [
                "parseability by panel",
                "label counts by split",
                "source-disjoint split assignment",
                "minimum per-cell row counts",
            ],
            "Kill or resize if any primary label/split cell is too small to support holdout claims.",
        ),
        gate(
            "direct_controls",
            "required_before_signature",
            "Pass direct controls that isolate ordinary task competence from the claimed arbitration behavior.",
            [
                "source-local direct control",
                "learned/parametric direct control where relevant",
                "answer-interface control",
                "parser failure audit",
            ],
            "Kill if direct controls fail, because the substrate would not distinguish task failure from mechanism failure.",
        ),
        gate(
            "conflict_mixture",
            "required_before_signature",
            "Produce a nontrivial conflict mixture in the primary behavior instead of all-local, all-learned, all-null, or all-format behavior.",
            [
                "primary conflict counts",
                "per-label behavior rates",
                "non-holdout and holdout conflict balance",
                "subgroup robustness by source family",
            ],
            "Kill if the behavior collapses to one branch or reaches contrast only through a visible prompt channel.",
        ),
        gate(
            "null_rows",
            "required_before_signature",
            "Pass answer-absent, irrelevant-source, lure, or unsupported-context null rows appropriate to the level.",
            [
                "answer-absent null rates",
                "irrelevant or unsupported context rows",
                "lure/side-answer rows",
                "margin-stratified null analysis",
            ],
            "Kill or bound if primary rows pass but null rows expose local, learned, side-answer, or abstention leakage.",
        ),
        gate(
            "source_disjoint_holdout",
            "required_before_signature",
            "Preserve the behavior on source-disjoint holdout sources, not just template-disjoint or row-disjoint variants.",
            [
                "source-disjoint holdout ids",
                "holdout behavior table",
                "matched split metrics",
                "source-family subgroup metrics",
            ],
            "Kill if holdout behavior exists only on reused sources or source families used for selection.",
        ),
        gate(
            "prompt_channel_locality",
            "required_before_signature",
            "Remove, match, or ablate visible prompt channels that can explain the label or requested mode.",
            [
                "prompt-format baseline",
                "requested-mode baseline",
                "prompt rewrite equivalence",
                "visible label/status/channel ablation",
            ],
            "Kill hidden-state admission if prompt text alone identifies the behavior branch.",
        ),
        gate(
            "output_candidate_baselines",
            "required_before_signature",
            "Report output margin, candidate-score margin, next-token margin, and simple prompt baselines beside any future signature.",
            [
                "output margin baseline",
                "candidate-score margin baseline",
                "next-token margin baseline",
                "prompt-length and prompt-format baselines",
            ],
            "Kill or reclassify as output-visible if these baselines explain the candidate signature.",
        ),
        gate(
            "side_effect_and_leakage",
            "required_before_signature",
            "Audit side answers, other-number leakage, fluency, answer shape, and unrelated rows before intervention work.",
            [
                "side-answer rate",
                "other-number or lure leakage",
                "answer-shape distribution",
                "fluency or parse degradation check",
            ],
            "Kill or bound if the behavior is carried by side effects rather than the intended branch.",
        ),
        gate(
            "split_freeze",
            "required_before_signature",
            "Freeze discovery, calibration, holdout, null, and side-effect panels before hidden-state discovery.",
            [
                "frozen split manifest",
                "frozen row ids and source ids",
                "predeclared admission thresholds",
                "admission report linked before signature work",
            ],
            "Kill the admission packet if thresholds move after inspecting hidden states.",
        ),
    ]


def build_level_admission_specs() -> dict[str, dict[str, Any]]:
    return {
        "level_2_semi_synthetic_familiar_entity": {
            "admission_class": "new_familiar_entity_substrate",
            "candidate_substrate": "Familiar entity names with artificial task-local values under a contract that separates source lookup from semantic prior pressure.",
            "material_novelty_requirement": "Must change the current MC007 route by changing the behavior family or prompt contract, not just another authority dial or parser repair.",
            "admission_decision": "No hidden-state work until a fresh familiar-entity table passes all admission gates.",
            "known_failure_modes": [
                "PROMPT_AUTHORITY_CONFUND",
                "PROMPT_CONTRACT_PARSEABILITY",
                "BEHAVIOR_TABLE_CONDITION_CONFOUND",
                "semantic prior swamps artificial values",
            ],
            "extra_gates": [
                gate(
                    "material_novelty",
                    "required_before_behavior_run",
                    "Show that the proposal is materially outside MC007 V1-V4 ordinary prompt/authority/parser repairs.",
                    [
                        "diff against MC007 prompt route",
                        "new source/value generation rule",
                        "semantic-prior interference panel",
                        "authority channel removal or matching plan",
                    ],
                    "Reject the proposal if it is another MC007-style authority or parse repair.",
                )
            ],
            "promotion_rule": "Admit to signature search only if source-local artificial values and semantic-prior controls both pass on source-disjoint holdout while prompt-authority and output/candidate baselines fail to explain the behavior.",
            "death_rule": "Kill the substrate if behavior still depends on authority wording, parser choices, or semantic-prior leakage after one preregistered repair.",
            "containment_rule": "If only source-local lookup works, classify the level as a source-value diagnostic, not a familiar-entity knowledge bridge.",
            "export_rule": "Export PROMPT_AUTHORITY_CONFUND, SEMANTIC_PRIOR_INTERFERENCE, or BEHAVIOR_TABLE_CONDITION_CONFOUND.",
        },
        "level_3_symbolic_or_learned_memory_bridge": {
            "admission_class": "new_bridge_substrate_class",
            "candidate_substrate": "A local-versus-learned arbitration task outside MC007-MC033 that creates stable branch behavior without visible status labels or answer-interface shortcuts.",
            "material_novelty_requirement": "Must be materially outside MC007-MC033: source labels, row codes, operation handles, worked examples, answer schemas, absence guards, checksum cues, cross-table consistency, and row-local fact claims.",
            "admission_decision": "Same-family bridge route remains killed; only a new substrate class can request admission.",
            "known_failure_modes": [
                "POST_MC032_BRIDGE_ROUTE_CLOSED",
                "LOCAL_SOURCE_SALIENCE",
                "FACT_CLAIM_MISMATCH_LOCAL_AND_CLAIM_LEAK",
                "STATUSLESS_SOURCE_VALIDITY_LOCAL_DOMINANCE",
                "prompt-visible positive control",
            ],
            "extra_gates": [
                gate(
                    "material_novelty",
                    "required_before_behavior_run",
                    "Demonstrate the proposal is outside the MC007-MC033 bridge route family and not just a new source-validity cue.",
                    [
                        "closed-route exclusion checklist",
                        "new answer interface or source contract",
                        "reason the old bridge failure modes should not dominate",
                        "explicit comparison to MC031-MC033",
                    ],
                    "Reject the proposal if it only renames source validity, reliability, checksum, cross-table, or fact-claim cues.",
                )
            ],
            "promotion_rule": "Admit to hidden-state search only if branch, null, local, side-number, parseability, prompt-channel, source-disjoint, and output/candidate controls pass together.",
            "death_rule": "Kill the proposed bridge if it repeats a closed MC007-MC033 failure class or passes only as a prompt-visible positive control.",
            "containment_rule": "If it fails, add it to the bridge failure taxonomy; do not call the failure a mechanism absence in learned facts generally.",
            "export_rule": "Export the typed bridge failure into the bridge ladder, smoke diagnostics, error taxonomy, and gap plan.",
        },
        "level_5_real_abstention_uncertainty": {
            "admission_class": "new_real_uncertainty_substrate",
            "candidate_substrate": "Generated-answer factual correction, refusal, abstention, or uncertainty behavior with grounded known/unknown/support labels and no refusal-template leakage.",
            "material_novelty_requirement": "Must change task construction beyond MC002 and MC002B pressure prompts; labels and abstention behavior must be grounded before prompting.",
            "admission_decision": "No existing real-uncertainty substrate is admissible; a new behavior family and work order are required.",
            "known_failure_modes": [
                "REQUESTED_MODE_CONFOUND",
                "OUTPUT_MARGIN_CONFUND",
                "BEHAVIOR_TABLE_CONDITION_CONFOUND",
                "refusal-template leakage",
                "known/unknown label leakage",
            ],
            "extra_gates": [
                gate(
                    "material_novelty",
                    "required_before_behavior_run",
                    "Show that the proposal is not another MC002/MC002B prompt-pressure repair.",
                    [
                        "new factual label source",
                        "known/unknown/support split plan",
                        "abstention/null row definitions",
                        "refusal-template leakage controls",
                    ],
                    "Reject the proposal if the labels or abstention behavior are defined by the prompt wording itself.",
                )
            ],
            "promotion_rule": "Admit to signature search only after generated factual correction/refusal/uncertainty behavior passes prompt, output, label-balance, null, and source-disjoint holdout gates.",
            "death_rule": "Kill the substrate if abstention, refusal, or correction labels are explained by prompt wording, answer schema, output margin, or label leakage.",
            "containment_rule": "If only a prompt-visible refusal behavior appears, classify it as requested-mode behavior control, not uncertainty control.",
            "export_rule": "Export BEHAVIOR_TABLE_CONDITION_CONFOUND, REQUESTED_MODE_CONFOUND, OUTPUT_MARGIN_CONFUND, or LABEL_GROUNDING_FAILURE.",
        },
    }


def build_admission_packets(
    knowledge_gap_plan: dict[str, Any],
    bridge_ladder: dict[str, Any],
    smoke_diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    plans_by_id = by_key(knowledge_gap_plan["level_plans"], "level_id")
    specs = build_level_admission_specs()
    packets: list[dict[str, Any]] = []
    for level_id, spec in specs.items():
        source_plan = plans_by_id[level_id]
        gates = sorted(
            spec["extra_gates"] + common_behavior_gates(level_id),
            key=lambda item: item["order"],
        )
        packets.append(
            {
                "level_id": level_id,
                "label": source_plan["label"],
                "row_ids": source_plan["row_ids"],
                "ladder_status": source_plan["ladder_status"],
                "decision_state": source_plan["decision_state"],
                "admission_class": spec["admission_class"],
                "candidate_substrate": spec["candidate_substrate"],
                "material_novelty_requirement": spec["material_novelty_requirement"],
                "admission_decision": spec["admission_decision"],
                "current_route_state": {
                    "requires_new_behavior_substrate": source_plan[
                        "requires_new_behavior_substrate"
                    ],
                    "new_work_order_required": source_plan[
                        "new_work_order_required"
                    ],
                    "new_hidden_state_search_allowed": source_plan[
                        "new_hidden_state_search_allowed"
                    ],
                    "same_family_route_killed": source_plan[
                        "same_family_route_killed"
                    ],
                    "current_primary_blockers": source_plan["current_evidence"][
                        "primary_blocker_counts"
                    ],
                    "current_terminal_stages": source_plan["current_evidence"][
                        "terminal_stage_counts"
                    ],
                },
                "historical_context": {
                    "bridge_rung_count": bridge_ladder["summary"]["rung_count"],
                    "bridge_hidden_state_allowed_count": bridge_ladder["summary"][
                        "hidden_state_allowed_count"
                    ],
                    "bridge_clean_unconfounded_count": bridge_ladder["summary"][
                        "clean_unconfounded_bridge_count"
                    ],
                    "smoke_card_count": smoke_diagnostics["summary"]["card_count"],
                    "smoke_hidden_state_allowed_count": smoke_diagnostics["summary"][
                        "hidden_state_allowed_count"
                    ],
                    "smoke_behavior_ready_count": smoke_diagnostics["summary"][
                        "behavior_ready_count"
                    ],
                },
                "admission_gates": gates,
                "known_failure_modes": spec["known_failure_modes"],
                "promotion_rule": spec["promotion_rule"],
                "death_rule": spec["death_rule"],
                "containment_rule": spec["containment_rule"],
                "export_rule": spec["export_rule"],
                "hidden_state_license": "forbidden_until_all_admission_gates_pass",
            }
        )
    return packets


def build_summary(
    packets: list[dict[str, Any]],
    knowledge_gap_plan: dict[str, Any],
    bridge_ladder: dict[str, Any],
    smoke_diagnostics: dict[str, Any],
    gate_geometry: dict[str, Any],
) -> dict[str, Any]:
    admission_classes = dict(
        sorted(Counter(packet["admission_class"] for packet in packets).items())
    )
    gate_status_counts = dict(
        sorted(
            Counter(
                gate_item["status"]
                for packet in packets
                for gate_item in packet["admission_gates"]
            ).items()
        )
    )
    return {
        "packet_count": len(packets),
        "gate_count": sum(len(packet["admission_gates"]) for packet in packets),
        "unique_gate_count": len(ADMISSION_GATE_ORDER),
        "admission_classes": admission_classes,
        "gate_status_counts": gate_status_counts,
        "levels_requiring_new_behavior_substrate_count": knowledge_gap_plan[
            "summary"
        ]["levels_requiring_new_behavior_substrate_count"],
        "new_work_order_required_level_count": knowledge_gap_plan["summary"][
            "new_work_order_required_level_count"
        ],
        "new_hidden_state_search_allowed_level_count": knowledge_gap_plan[
            "summary"
        ]["new_hidden_state_search_allowed_level_count"],
        "same_family_route_killed_level_count": knowledge_gap_plan["summary"][
            "same_family_route_killed_level_count"
        ],
        "bridge_rung_count": bridge_ladder["summary"]["rung_count"],
        "bridge_hidden_state_allowed_count": bridge_ladder["summary"][
            "hidden_state_allowed_count"
        ],
        "bridge_clean_unconfounded_count": bridge_ladder["summary"][
            "clean_unconfounded_bridge_count"
        ],
        "smoke_card_count": smoke_diagnostics["summary"]["card_count"],
        "smoke_hidden_state_allowed_count": smoke_diagnostics["summary"][
            "hidden_state_allowed_count"
        ],
        "smoke_behavior_ready_count": smoke_diagnostics["summary"][
            "behavior_ready_count"
        ],
        "pre_signature_blocked_rows": gate_geometry["summary"]["ordered_gate_buckets"][
            "pre_signature_blocked"
        ],
        "promoted_mechanism_count": gate_geometry["summary"][
            "promoted_mechanism_count"
        ],
    }


def build_validation_checks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    packets = payload["admission_packets"]
    packets_by_level = by_key(packets, "level_id")
    expected_level_ids = sorted(
        level["level_id"]
        for level in payload["source_snapshot"]["knowledge_gap_levels"]
        if level["requires_new_behavior_substrate"]
    )
    packet_level_ids = sorted(packet["level_id"] for packet in packets)
    gate_failures = [
        packet["level_id"]
        for packet in packets
        if sorted(gate_item["gate_id"] for gate_item in packet["admission_gates"])
        != sorted(ADMISSION_GATE_ORDER)
    ]
    hidden_state_licenses = [
        packet["level_id"]
        for packet in packets
        if packet["hidden_state_license"] != "forbidden_until_all_admission_gates_pass"
        or packet["current_route_state"]["new_hidden_state_search_allowed"]
    ]
    empty_rule_packets = [
        packet["level_id"]
        for packet in packets
        if not packet["promotion_rule"]
        or not packet["death_rule"]
        or not packet["containment_rule"]
        or not packet["export_rule"]
    ]
    return [
        {
            "id": "covers_every_new_substrate_level",
            "predicate": "packet levels == gap-plan levels requiring new behavior substrate",
            "actual": {
                "packet_level_ids": packet_level_ids,
                "expected_level_ids": expected_level_ids,
            },
            "passed": packet_level_ids == expected_level_ids,
            "why": "Admission protocol should cover exactly the levels that need fresh behavior substrates.",
        },
        {
            "id": "each_packet_has_all_admission_gates",
            "predicate": "empty list",
            "actual": gate_failures,
            "passed": not gate_failures,
            "why": "Every future knowledge substrate must pass the same front-door gate types.",
        },
        {
            "id": "hidden_state_license_remains_forbidden",
            "predicate": "empty list and summary hidden-state count == 0",
            "actual": {
                "packet_failures": hidden_state_licenses,
                "new_hidden_state_search_allowed_level_count": payload["summary"][
                    "new_hidden_state_search_allowed_level_count"
                ],
            },
            "passed": not hidden_state_licenses
            and payload["summary"]["new_hidden_state_search_allowed_level_count"] == 0,
            "why": "The protocol is an admission gate, not a license to probe hidden states.",
        },
        {
            "id": "bridge_route_stays_killed_until_new_class",
            "predicate": "bridge level killed and bridge hidden-state count == 0",
            "actual": {
                "bridge_same_family_route_killed": packets_by_level[
                    "level_3_symbolic_or_learned_memory_bridge"
                ]["current_route_state"]["same_family_route_killed"],
                "bridge_hidden_state_allowed_count": payload["summary"][
                    "bridge_hidden_state_allowed_count"
                ],
                "material_novelty": packets_by_level[
                    "level_3_symbolic_or_learned_memory_bridge"
                ]["material_novelty_requirement"],
            },
            "passed": packets_by_level["level_3_symbolic_or_learned_memory_bridge"][
                "current_route_state"
            ]["same_family_route_killed"]
            and payload["summary"]["bridge_hidden_state_allowed_count"] == 0
            and "MC007-MC033" in packets_by_level[
                "level_3_symbolic_or_learned_memory_bridge"
            ]["material_novelty_requirement"],
            "why": "The bridge family has 24 rungs and no clean hidden-state route.",
        },
        {
            "id": "real_uncertainty_requires_label_grounding",
            "predicate": "level 5 material novelty names labels and abstention",
            "actual": {
                "material_novelty": packets_by_level[
                    "level_5_real_abstention_uncertainty"
                ]["material_novelty_requirement"],
                "known_failure_modes": packets_by_level[
                    "level_5_real_abstention_uncertainty"
                ]["known_failure_modes"],
            },
            "passed": "labels" in packets_by_level[
                "level_5_real_abstention_uncertainty"
            ]["material_novelty_requirement"]
            and "abstention" in packets_by_level[
                "level_5_real_abstention_uncertainty"
            ]["candidate_substrate"],
            "why": "The real uncertainty level must be grounded before prompt or output controls.",
        },
        {
            "id": "packets_have_decision_rules",
            "predicate": "empty list",
            "actual": empty_rule_packets,
            "passed": not empty_rule_packets,
            "why": "Admission must say how a new substrate promotes, dies, narrows, or exports.",
        },
        {
            "id": "does_not_claim_promoted_mechanism",
            "predicate": "promoted_mechanism_count == 0",
            "actual": payload["summary"]["promoted_mechanism_count"],
            "passed": payload["summary"]["promoted_mechanism_count"] == 0
            and "not a mechanism result" in payload["forbidden_claim"],
            "why": "A front-door protocol is not evidence that any control surface exists.",
        },
    ]


def build_control_surface_knowledge_substrate_admission() -> dict[str, Any]:
    knowledge_gap_plan = load_json(KNOWLEDGE_GAP_PLAN_PATH)
    bridge_ladder = load_json(BRIDGE_LADDER_PATH)
    smoke_diagnostics = load_json(SMOKE_DIAGNOSTICS_PATH)
    gate_geometry = load_json(GATE_GEOMETRY_PATH)
    packets = build_admission_packets(
        knowledge_gap_plan, bridge_ladder, smoke_diagnostics
    )
    summary = build_summary(
        packets,
        knowledge_gap_plan,
        bridge_ladder,
        smoke_diagnostics,
        gate_geometry,
    )
    payload = {
        "schema_version": 1,
        "updated_at": knowledge_gap_plan.get("updated_at"),
        "purpose": (
            "Define the admission protocol for knowledge-ladder levels that "
            "need a new behavior substrate before hidden-state signatures or "
            "interventions are licensed."
        ),
        "sources": {
            "knowledge_gap_plan": rel(KNOWLEDGE_GAP_PLAN_PATH),
            "bridge_ladder": rel(BRIDGE_LADDER_PATH),
            "smoke_diagnostics": rel(SMOKE_DIAGNOSTICS_PATH),
            "gate_geometry": rel(GATE_GEOMETRY_PATH),
        },
        "source_snapshot": {
            "knowledge_gap_levels": [
                {
                    "level_id": level["level_id"],
                    "requires_new_behavior_substrate": level[
                        "requires_new_behavior_substrate"
                    ],
                    "new_work_order_required": level["new_work_order_required"],
                    "new_hidden_state_search_allowed": level[
                        "new_hidden_state_search_allowed"
                    ],
                    "same_family_route_killed": level["same_family_route_killed"],
                }
                for level in knowledge_gap_plan["level_plans"]
            ],
            "bridge_rung_count": bridge_ladder["summary"]["rung_count"],
            "smoke_card_count": smoke_diagnostics["summary"]["card_count"],
            "pre_signature_blocked_rows": gate_geometry["summary"][
                "ordered_gate_buckets"
            ]["pre_signature_blocked"],
        },
        "admission_gate_order": ADMISSION_GATE_ORDER,
        "summary": summary,
        "admission_packets": packets,
        "allowed_claim": (
            "The project now has a front-door protocol for future knowledge "
            "substrates: familiar-entity, bridge, and real-uncertainty levels "
            "must pass material novelty, behavior, controls, nulls, holdouts, "
            "prompt-channel, output/candidate, leakage, and split-freeze gates "
            "before hidden-state work."
        ),
        "forbidden_claim": (
            "This is not a mechanism result, not a hidden-state license, not a "
            "claim that MC007-MC033 can be repaired by same-family cue changes, "
            "and not evidence for real-world factual correction or uncertainty "
            "control."
        ),
    }
    payload["validation_checks"] = build_validation_checks(payload)
    return payload


def validate_knowledge_substrate_admission(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("knowledge substrate admission schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(
                f"knowledge substrate admission source missing: {rel_path}"
            )
    if payload["summary"]["packet_count"] != 3:
        raise AssertionError("knowledge substrate admission expected three packets")
    if payload["summary"]["unique_gate_count"] != len(ADMISSION_GATE_ORDER):
        raise AssertionError("knowledge substrate admission gate count mismatch")
    if payload["summary"]["new_hidden_state_search_allowed_level_count"] != 0:
        raise AssertionError("knowledge substrate admission must not license search")
    if payload["summary"]["bridge_hidden_state_allowed_count"] != 0:
        raise AssertionError("knowledge substrate admission bridge count must be zero")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(
            f"knowledge substrate admission checks failed: {failed_checks}"
        )


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Control-Surface Knowledge Substrate Admission",
        "",
        f"Source updated_at: {payload['updated_at']}",
        "",
        "Status: generated admission protocol implemented and validated.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/control_surface_knowledge_substrate_admission.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_knowledge_substrate_admission.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_knowledge_substrate_admission.py --write",
        "python code\\control_surface_knowledge_substrate_admission.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Purpose",
        "",
        "This protocol is the front door for future knowledge-family behavior",
        "substrates. It is deliberately prior to hidden-state work. A candidate",
        "substrate has to pass these gates before signature, steering, editing,",
        "or surgery is licensed.",
        "",
        "## Generated Facts",
        "",
        f"- admission packets: {summary['packet_count']};",
        f"- admission gates per packet: {summary['unique_gate_count']};",
        f"- total gate entries: {summary['gate_count']};",
        f"- levels requiring new behavior substrate: {summary['levels_requiring_new_behavior_substrate_count']};",
        f"- future-work-order levels: {summary['new_work_order_required_level_count']};",
        f"- new hidden-state search allowed levels: {summary['new_hidden_state_search_allowed_level_count']};",
        f"- same-family route-killed levels: {summary['same_family_route_killed_level_count']};",
        f"- bridge rungs: {summary['bridge_rung_count']};",
        f"- bridge hidden-state-allowed rungs: {summary['bridge_hidden_state_allowed_count']};",
        f"- bridge clean unconfounded rungs: {summary['bridge_clean_unconfounded_count']};",
        f"- smoke cards: {summary['smoke_card_count']};",
        f"- smoke hidden-state-allowed cards: {summary['smoke_hidden_state_allowed_count']};",
        f"- pre-signature-blocked atlas rows: {summary['pre_signature_blocked_rows']};",
        f"- promoted mechanisms: {summary['promoted_mechanism_count']}.",
        "",
        "## Admission Packets",
        "",
        "| Level | Admission Class | Current Status | Hidden-State License |",
        "| --- | --- | --- | --- |",
    ]
    for packet in payload["admission_packets"]:
        lines.append(
            f"| {packet['label']} | `{packet['admission_class']}` | "
            f"`{packet['ladder_status']}` | `{packet['hidden_state_license']}` |"
        )

    lines.extend(["", "## Gate Order", ""])
    for index, gate_id in enumerate(payload["admission_gate_order"], start=1):
        lines.append(f"{index}. `{gate_id}`")

    for packet in payload["admission_packets"]:
        lines.extend(
            [
                "",
                f"## {packet['label']}",
                "",
                f"- level id: `{packet['level_id']}`;",
                f"- row ids: `{format_value(packet['row_ids'])}`;",
                f"- admission decision: {packet['admission_decision']}",
                f"- candidate substrate: {packet['candidate_substrate']}",
                f"- material novelty: {packet['material_novelty_requirement']}",
                "",
                "Decision rules:",
                f"- `promotion_rule`: {packet['promotion_rule']}",
                f"- `death_rule`: {packet['death_rule']}",
                f"- `containment_rule`: {packet['containment_rule']}",
                f"- `export_rule`: {packet['export_rule']}",
                "",
                "Known failure modes:",
            ]
        )
        for mode in packet["known_failure_modes"]:
            lines.append(f"- `{mode}`")
        lines.extend(["", "Admission gates:"])
        for gate_item in packet["admission_gates"]:
            lines.append(
                f"- `{gate_item['gate_id']}`: {gate_item['requirement']}"
            )

    lines.extend(
        [
            "",
            "## What This Proves",
            "",
            "It proves that the project has a generated front-door rule for",
            "future knowledge substrate proposals. The rule is tied to the",
            "current gap plan, bridge ladder, smoke diagnostics, and gate geometry",
            "rather than to review prose.",
            "",
            "## What It Does Not Prove",
            "",
            "It does not prove that a new substrate exists. It does not permit",
            "hidden-state search. It does not upgrade any existing knowledge",
            "level to a mechanism claim.",
            "",
        ]
    )
    return "\n".join(lines)


def write_knowledge_substrate_admission(
    output_path: Path = KNOWLEDGE_SUBSTRATE_ADMISSION_PATH,
    report_path: Path = KNOWLEDGE_SUBSTRATE_ADMISSION_REPORT_PATH,
) -> dict[str, Any]:
    payload = build_control_surface_knowledge_substrate_admission()
    validate_knowledge_substrate_admission(payload)
    write_json(output_path, payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(payload), encoding="utf-8", newline="\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write admission artifacts")
    parser.add_argument("--json", action="store_true", help="print admission JSON")
    args = parser.parse_args()

    payload = build_control_surface_knowledge_substrate_admission()
    validate_knowledge_substrate_admission(payload)

    if args.write:
        write_knowledge_substrate_admission()
        print(
            f"wrote {KNOWLEDGE_SUBSTRATE_ADMISSION_PATH.relative_to(ROOT).as_posix()} and "
            f"{KNOWLEDGE_SUBSTRATE_ADMISSION_REPORT_PATH.relative_to(ROOT).as_posix()} "
            f"with {payload['summary']['packet_count']} admission packets"
        )
        return
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(
        "knowledge substrate admission ok: "
        f"{payload['summary']['packet_count']} packets, "
        f"{payload['summary']['gate_count']} gate entries"
    )
    print(
        "admission_classes:",
        json.dumps(payload["summary"]["admission_classes"], sort_keys=True),
    )
    print(
        "hidden_state_allowed:",
        payload["summary"]["new_hidden_state_search_allowed_level_count"],
    )


if __name__ == "__main__":
    main()
