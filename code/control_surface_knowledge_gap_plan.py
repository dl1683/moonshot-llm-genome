"""Build the knowledge-ladder missing-evidence plan.

The knowledge ladder says how far the project has climbed from synthetic
lookup toward real factual uncertainty. This layer turns that ladder into
decision-bound missing-evidence items: what is absent, what work is licensed,
what route is killed, and what claim is forbidden at each level.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from control_surface_artifacts import ROOT, load_json
from control_surface_family_matrix import FAMILY_MATRIX_PATH
from control_surface_gap_closure_plan import GAP_CLOSURE_PLAN_PATH
from control_surface_knowledge_ladder import KNOWLEDGE_LADDER_PATH
from control_surface_offensive_doctrine import OFFENSIVE_DOCTRINE_PATH


KNOWLEDGE_GAP_PLAN_PATH = ROOT / "data" / "control_surface_knowledge_gap_plan.json"
KNOWLEDGE_GAP_PLAN_REPORT_PATH = (
    ROOT / "research" / "43_CONTROL_SURFACE_KNOWLEDGE_GAP_PLAN.md"
)

REQUIRED_DECISION_RULES = {
    "promotion_rule",
    "death_rule",
    "containment_rule",
    "export_rule",
}


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


def compact_contract(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "work_order_id": contract["work_order_id"],
        "title": contract["title"],
        "urgency": contract["urgency"],
        "track_type": contract["track_type"],
        "target_gap_ids": contract["target_gap_ids"],
        "first_artifact": contract["first_artifact"],
        "iteration_budget": contract["iteration_budget"],
        "promotion_rule": contract["promotion_rule"],
        "kill_rule": contract["kill_rule"],
        "containment_rule": contract["containment_rule"],
        "export_rule": contract["export_rule"],
    }


def level_gap_specs() -> list[dict[str, Any]]:
    return [
        {
            "level_id": "level_1_synthetic_lookup",
            "decision_state": "bounded_reference_needs_reliability_and_width",
            "missing_evidence": [
                {
                    "id": "answer_absent_null_boundary",
                    "kind": "reliability",
                    "description": "Repair or explicitly bound the answer-absent low-margin null flips without weakening the high-margin lookup mediation claim.",
                },
                {
                    "id": "transfer_panel",
                    "kind": "widening",
                    "description": "Measure whether primary mediation, null locality, side effects, and prompt robustness transfer together beyond the current Qwen-dominant setting.",
                },
                {
                    "id": "intervention_family_verdict",
                    "kind": "route_disposition",
                    "description": "Decide whether write replacement is promoted, bounded, or killed after a fixed repair budget.",
                },
            ],
            "next_allowed_action": (
                "Use the MC005 reference closeout and transfer-width probe. "
                "Do not keep adding same-route repairs unless the preregistered "
                "repair budget changes the reliability boundary."
            ),
            "requires_new_behavior_substrate": False,
            "requires_predecision_or_reliability": True,
            "new_hidden_state_search_allowed": False,
            "new_work_order_required": False,
            "same_family_route_killed": False,
            "linked_work_order_ids": [
                "close_mc005_reference_specimen",
                "run_width_transfer_probe",
            ],
            "decision_rules": {
                "promotion_rule": "Promote only if high-margin lookup mediation stays strong and the answer-absent null boundary closes under locality, side-row, fluency, and holdout panels.",
                "death_rule": "Kill the write-replacement route if null flips or collateral side effects persist across the fixed repair variants.",
                "containment_rule": "If the primary effect survives but nulls remain fragile, keep MC005 as a bounded synthetic lookup reference, not a general knowledge mechanism.",
                "export_rule": "Export NULL_ROW_LOW_MARGIN_FLIP, TRANSFER_PRIMARY_BEFORE_RELIABILITY, or TRANSFER_PRIMARY_FAILED as reusable diagnostics.",
            },
            "forbidden_moves": [
                "Do not describe MC005 as full reliability while answer-absent null flips remain.",
                "Do not call a primary-effect replication transfer unless null locality and side-effect panels also transfer.",
                "Do not broaden the claim to knowledge control.",
            ],
            "insight": "The best current mechanism-like object is also the best calibration object for how reliability boundaries break.",
        },
        {
            "level_id": "level_2_semi_synthetic_familiar_entity",
            "decision_state": "material_new_substrate_required",
            "missing_evidence": [
                {
                    "id": "clean_familiar_entity_behavior_substrate",
                    "kind": "behavior",
                    "description": "A familiar-entity task where local artificial values are behaviorally stable before hidden-state work.",
                },
                {
                    "id": "prompt_authority_controls",
                    "kind": "control",
                    "description": "Prompt-authority, format, source-disjoint, null, and output/candidate baselines that do not explain the behavior.",
                },
                {
                    "id": "semantic_prior_interference_panel",
                    "kind": "robustness",
                    "description": "A panel that measures whether familiar entity priors help, conflict with, or swamp task-local lookup.",
                },
            ],
            "next_allowed_action": (
                "Treat MC007 as blocked under the current substrate. A future "
                "attempt needs a materially new familiar-entity contract before "
                "any signature search."
            ),
            "requires_new_behavior_substrate": True,
            "requires_predecision_or_reliability": False,
            "new_hidden_state_search_allowed": False,
            "new_work_order_required": True,
            "same_family_route_killed": True,
            "linked_work_order_ids": ["close_post_mc033_bridge_substrate_family"],
            "decision_rules": {
                "promotion_rule": "Promote this level only when a familiar-entity behavior substrate passes direct, null, prompt-channel, holdout, and output/candidate gates before probing.",
                "death_rule": "Kill same-family MC007-style repairs if the behavior remains parse-, authority-, or prompt-contract-dominated.",
                "containment_rule": "Until then, the level only says familiar entities are a bridge stressor, not a localizable control surface.",
                "export_rule": "Export PROMPT_AUTHORITY_CONFUND or BEHAVIOR_TABLE_CONDITION_CONFOUND if the next substrate fails for visible reasons.",
            },
            "forbidden_moves": [
                "Do not run hidden-state probes on another MC007-style behavior failure.",
                "Do not treat real entity names as knowledge evidence when the artificial value contract is unstable.",
                "Do not reuse the same authority-dial route as a new substrate.",
            ],
            "insight": "This level is where semantic familiarity first enters, and current evidence says the behavior contract fails before mechanism work.",
        },
        {
            "level_id": "level_3_symbolic_or_learned_memory_bridge",
            "decision_state": "same_family_bridge_route_killed",
            "missing_evidence": [
                {
                    "id": "clean_unconfounded_bridge_substrate",
                    "kind": "behavior",
                    "description": "A bridge substrate outside MC007-MC033 that passes branch, null, local, side-number, parseability, prompt-channel, source-disjoint, and output/candidate controls.",
                },
                {
                    "id": "hidden_state_permission_gate",
                    "kind": "permission",
                    "description": "A documented gate allowing hidden-state work only after the bridge behavior substrate is clean.",
                },
                {
                    "id": "learned_local_arbitration_stability",
                    "kind": "robustness",
                    "description": "Evidence that learned/local arbitration remains stable across conflict mixtures instead of collapsing to prompt-visible or output-visible cues.",
                },
            ],
            "next_allowed_action": (
                "Close the same-family bridge route after MC033. A future bridge "
                "must be a new substrate class, not another source-validity cue."
            ),
            "requires_new_behavior_substrate": True,
            "requires_predecision_or_reliability": False,
            "new_hidden_state_search_allowed": False,
            "new_work_order_required": True,
            "same_family_route_killed": True,
            "linked_work_order_ids": ["close_post_mc033_bridge_substrate_family"],
            "decision_rules": {
                "promotion_rule": "Promote a bridge level to hidden-state work only if a materially new substrate clears all behavior, null, locality, holdout, and output/candidate gates together.",
                "death_rule": "Keep the MC007-MC033 same-family route killed unless a new proposal changes the substrate class, not merely the cue wording.",
                "containment_rule": "The current bridge evidence supports a substrate-failure taxonomy, not a symbolic or learned-memory mechanism.",
                "export_rule": "Export POST_MC032_BRIDGE_ROUTE_CLOSED, FACT_CLAIM_MISMATCH_LOCAL_AND_CLAIM_LEAK, and STATUSLESS_SOURCE_VALIDITY_LOCAL_DOMINANCE.",
            },
            "forbidden_moves": [
                "Do not start probes on MC031-MC033-style rows.",
                "Do not reopen status labels, row codes, operation handles, worked examples, answer schemas, absence guards, checksum, consistency, or row-local fact-claim cues.",
                "Do not call a prompt-visible positive control a knowledge mechanism.",
            ],
            "insight": "The bridge ladder is not an embarrassing absence of mechanisms; it is a map of which bridge substrates die under controls.",
        },
        {
            "level_id": "level_4_parametric_fact_override",
            "decision_state": "monitor_only_predecision_frontier",
            "missing_evidence": [
                {
                    "id": "candidate_output_decoupled_signature",
                    "kind": "signature",
                    "description": "An earlier signal that survives same-stage output geometry, final candidate margin, shuffle, and source-disjoint holdout controls.",
                },
                {
                    "id": "causal_predecision_lever",
                    "kind": "intervention",
                    "description": "A steering or editing route that changes generated capital-fact override behavior before answer commitment without broad collateral effects.",
                },
                {
                    "id": "decision_timing_boundary",
                    "kind": "lead_time",
                    "description": "A measured boundary for whether MC006 has usable lead time or is already output-visible by the final decision state.",
                },
            ],
            "next_allowed_action": (
                "Use the MC006 predecision frontier closeout. Probe earlier "
                "positions only under same-stage and final-margin controls; do "
                "not steer final-position high-AUC signatures."
            ),
            "requires_new_behavior_substrate": False,
            "requires_predecision_or_reliability": True,
            "new_hidden_state_search_allowed": False,
            "new_work_order_required": False,
            "same_family_route_killed": False,
            "linked_work_order_ids": ["close_mc006_predecision_frontier"],
            "decision_rules": {
                "promotion_rule": "Promote to intervention only if an earlier source/path signal predicts on holdout while beating same-stage output geometry and final candidate margin.",
                "death_rule": "Kill final-token MC006 probing if another margin-matched pass is explained by candidate score or final output margin.",
                "containment_rule": "If the route fails, keep MC006 as a decision-timing and output-visibility result, not a truth or knowledge vector.",
                "export_rule": "Export FINAL_STATE_OUTPUT_VISIBLE or PREDECISION_MONITOR_NO_LEVER depending on where the controlled failure lands.",
            },
            "forbidden_moves": [
                "Do not steer V15/V16-style final-token perfect AUCs.",
                "Do not treat final margin as a nuisance if it is the faithful downstream decision state.",
                "Do not use lonely AUCs without same-stage and final-stage baselines.",
            ],
            "insight": "MC006 is valuable because it measures where knowledge-like behavior becomes output-visible, even without a usable lever.",
        },
        {
            "level_id": "level_5_real_abstention_uncertainty",
            "decision_state": "real_uncertainty_behavior_substrate_absent",
            "missing_evidence": [
                {
                    "id": "real_uncertainty_behavior_table",
                    "kind": "behavior",
                    "description": "A balanced generated-answer table for factual correction, refusal, or uncertainty behavior that passes prompt, label, and output-margin controls.",
                },
                {
                    "id": "abstention_null_and_side_effect_gates",
                    "kind": "reliability",
                    "description": "Null rows, side-effect checks, fluency, locality, and holdouts that distinguish abstention from formatting or refusal-template artifacts.",
                },
                {
                    "id": "known_unknown_label_grounding",
                    "kind": "label_quality",
                    "description": "A label source for known, unknown, corrected, unsupported, and abstain cases that does not leak through prompt text or answer interface.",
                },
            ],
            "next_allowed_action": (
                "Open a new behavior-family work order only after the behavior "
                "table is clean. Hidden-state work is not licensed for the "
                "current MC002/MC002B substrates."
            ),
            "requires_new_behavior_substrate": True,
            "requires_predecision_or_reliability": False,
            "new_hidden_state_search_allowed": False,
            "new_work_order_required": True,
            "same_family_route_killed": False,
            "linked_work_order_ids": [],
            "decision_rules": {
                "promotion_rule": "Promote this level to signature search only after real uncertainty behavior passes behavior, prompt, output, label-balance, null, and holdout gates.",
                "death_rule": "Kill any uncertainty route whose labels or refusal behavior are explained by prompt wording, answer schema, or output margin.",
                "containment_rule": "Until then, this level remains a future target and cannot support factual correction, refusal, or uncertainty mechanism claims.",
                "export_rule": "Export BEHAVIOR_TABLE_CONDITION_CONFOUND, REQUESTED_MODE_CONFOUND, or OUTPUT_MARGIN_CONFUND if the substrate fails.",
            },
            "forbidden_moves": [
                "Do not probe hidden states on MC002/MC002B as if they were mechanism-ready.",
                "Do not collapse factual correction, refusal, and uncertainty into one label before the behavior table passes.",
                "Do not publish a real-world abstention control claim from synthetic or prompt-visible behavior.",
            ],
            "insight": "The real safety-relevant level is still below the behavior gate; that is a result about the current map, not a reason to pretend the ladder is higher.",
        },
    ]


def build_level_plans(
    knowledge_ladder: dict[str, Any],
    offensive_doctrine: dict[str, Any],
) -> list[dict[str, Any]]:
    levels_by_id = by_key(knowledge_ladder["levels"], "level_id")
    contracts_by_id = by_key(offensive_doctrine["branch_contracts"], "work_order_id")
    plans: list[dict[str, Any]] = []
    for spec in level_gap_specs():
        level = levels_by_id[spec["level_id"]]
        linked_ids = spec["linked_work_order_ids"]
        plans.append(
            {
                **spec,
                "label": level["label"],
                "ladder_status": level["status"],
                "stage_rank": level["stage_rank"],
                "row_ids": level["row_ids"],
                "row_count": level["row_count"],
                "current_evidence": {
                    "verdict_counts": level["verdict_counts"],
                    "primary_blocker_counts": level["primary_blocker_counts"],
                    "terminal_stage_counts": level["terminal_stage_counts"],
                    "frontier_class_counts": level["frontier_class_counts"],
                    "reliability_class_counts": level["reliability_class_counts"],
                    "transfer_class_counts": level["transfer_class_counts"],
                    "internal_monitor_present_count": level[
                        "internal_monitor_present_count"
                    ],
                    "internal_causal_surface_count": level[
                        "internal_causal_surface_count"
                    ],
                    "promotion_ready_count": level["promotion_ready_count"],
                },
                "work_order_links": [
                    compact_contract(contracts_by_id[work_id])
                    for work_id in linked_ids
                ],
            }
        )
    return plans


def build_summary(
    level_plans: list[dict[str, Any]],
    knowledge_ladder: dict[str, Any],
    offensive_doctrine: dict[str, Any],
) -> dict[str, Any]:
    linked_ids = sorted(
        {
            work_id
            for level in level_plans
            for work_id in level["linked_work_order_ids"]
        }
    )
    return {
        "level_count": len(level_plans),
        "missing_evidence_item_count": sum(
            len(level["missing_evidence"]) for level in level_plans
        ),
        "linked_existing_work_order_count": len(linked_ids),
        "linked_existing_work_order_ids": linked_ids,
        "new_work_order_required_level_count": sum(
            1 for level in level_plans if level["new_work_order_required"]
        ),
        "levels_requiring_new_behavior_substrate_count": sum(
            1 for level in level_plans if level["requires_new_behavior_substrate"]
        ),
        "levels_requiring_predecision_or_reliability_count": sum(
            1 for level in level_plans if level["requires_predecision_or_reliability"]
        ),
        "new_hidden_state_search_allowed_level_count": sum(
            1 for level in level_plans if level["new_hidden_state_search_allowed"]
        ),
        "same_family_route_killed_level_count": sum(
            1 for level in level_plans if level["same_family_route_killed"]
        ),
        "decision_state_counts": dict(
            sorted(Counter(level["decision_state"] for level in level_plans).items())
        ),
        "ladder_status_counts": dict(
            sorted(Counter(level["ladder_status"] for level in level_plans).items())
        ),
        "promoted_level_count": knowledge_ladder["summary"]["promoted_level_count"],
        "bounded_reference_level_count": knowledge_ladder["summary"][
            "bounded_reference_level_count"
        ],
        "monitor_only_level_count": knowledge_ladder["summary"][
            "monitor_only_level_count"
        ],
        "behavior_or_prompt_blocked_level_count": knowledge_ladder["summary"][
            "behavior_or_prompt_blocked_level_count"
        ],
        "bridge_hidden_state_allowed_count": knowledge_ladder["summary"][
            "bridge_hidden_state_allowed_count"
        ],
        "bridge_clean_unconfounded_count": knowledge_ladder["summary"][
            "bridge_clean_unconfounded_count"
        ],
        "real_abstention_uncertainty_ready_count": knowledge_ladder["summary"][
            "real_abstention_uncertainty_ready_count"
        ],
        "offensive_doctrine_branch_contract_count": offensive_doctrine["summary"][
            "branch_contract_count"
        ],
    }


def build_validation_checks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    level_plans = payload["level_plans"]
    level_ids = [level["level_id"] for level in level_plans]
    expected_level_ids = payload["source_snapshot"]["knowledge_ladder_level_ids"]
    ladder_status_by_level = payload["source_snapshot"]["ladder_status_by_level"]
    offensive_work_order_ids = set(payload["source_snapshot"]["offensive_work_order_ids"])
    linked_work_order_ids = {
        work_id for level in level_plans for work_id in level["linked_work_order_ids"]
    }
    missing_work_order_links = sorted(linked_work_order_ids - offensive_work_order_ids)
    rule_failures = [
        level["level_id"]
        for level in level_plans
        if REQUIRED_DECISION_RULES - set(level["decision_rules"])
        or any(
            not level["decision_rules"].get(rule_id)
            for rule_id in REQUIRED_DECISION_RULES
        )
    ]
    levels_by_id = by_key(level_plans, "level_id")
    return [
        {
            "id": "all_ladder_levels_have_gap_plans",
            "predicate": "planned level ids == knowledge ladder level ids",
            "actual": {
                "planned": sorted(level_ids),
                "expected": expected_level_ids,
            },
            "passed": sorted(level_ids) == expected_level_ids
            and len(level_ids) == len(set(level_ids)),
            "why": "The gap plan must cover every knowledge-ladder level exactly once.",
        },
        {
            "id": "level_statuses_match_knowledge_ladder",
            "predicate": "plan ladder_status == source ladder status",
            "actual": {
                level["level_id"]: {
                    "plan": level["ladder_status"],
                    "source": ladder_status_by_level[level["level_id"]],
                }
                for level in level_plans
            },
            "passed": all(
                level["ladder_status"] == ladder_status_by_level[level["level_id"]]
                for level in level_plans
            ),
            "why": "The plan must not improve or degrade ladder statuses by prose.",
        },
        {
            "id": "no_new_hidden_state_search_is_licensed",
            "predicate": "new_hidden_state_search_allowed_level_count == 0",
            "actual": payload["summary"]["new_hidden_state_search_allowed_level_count"],
            "passed": payload["summary"]["new_hidden_state_search_allowed_level_count"]
            == 0,
            "why": "Current knowledge-ladder progress is below the bar for new hidden-state search.",
        },
        {
            "id": "all_existing_work_order_links_are_real",
            "predicate": "empty list",
            "actual": missing_work_order_links,
            "passed": not missing_work_order_links,
            "why": "Every linked work order must exist in the offensive doctrine harness.",
        },
        {
            "id": "every_level_has_decision_rules",
            "predicate": "empty list",
            "actual": rule_failures,
            "passed": not rule_failures,
            "why": "Each level needs promotion, death, containment, and export rules.",
        },
        {
            "id": "synthetic_lookup_is_reliability_and_transfer_gap",
            "predicate": "level 1 linked to MC005 closeout and transfer width, not new behavior substrate",
            "actual": {
                "linked": levels_by_id["level_1_synthetic_lookup"][
                    "linked_work_order_ids"
                ],
                "requires_new_behavior_substrate": levels_by_id[
                    "level_1_synthetic_lookup"
                ]["requires_new_behavior_substrate"],
                "requires_predecision_or_reliability": levels_by_id[
                    "level_1_synthetic_lookup"
                ]["requires_predecision_or_reliability"],
            },
            "passed": set(
                levels_by_id["level_1_synthetic_lookup"]["linked_work_order_ids"]
            )
            == {"close_mc005_reference_specimen", "run_width_transfer_probe"}
            and not levels_by_id["level_1_synthetic_lookup"][
                "requires_new_behavior_substrate"
            ]
            and levels_by_id["level_1_synthetic_lookup"][
                "requires_predecision_or_reliability"
            ],
            "why": "MC005 should be closed or widened, not replaced with a new substrate.",
        },
        {
            "id": "bridge_level_is_route_killed_and_hidden_state_forbidden",
            "predicate": "level 3 same-family route killed and bridge hidden-state count == 0",
            "actual": {
                "same_family_route_killed": levels_by_id[
                    "level_3_symbolic_or_learned_memory_bridge"
                ]["same_family_route_killed"],
                "new_hidden_state_search_allowed": levels_by_id[
                    "level_3_symbolic_or_learned_memory_bridge"
                ]["new_hidden_state_search_allowed"],
                "bridge_hidden_state_allowed_count": payload["summary"][
                    "bridge_hidden_state_allowed_count"
                ],
            },
            "passed": levels_by_id["level_3_symbolic_or_learned_memory_bridge"][
                "same_family_route_killed"
            ]
            and not levels_by_id["level_3_symbolic_or_learned_memory_bridge"][
                "new_hidden_state_search_allowed"
            ]
            and payload["summary"]["bridge_hidden_state_allowed_count"] == 0,
            "why": "The bridge level is a substrate-failure map until a new class passes controls.",
        },
        {
            "id": "mc006_level_is_monitor_only_closeout",
            "predicate": "level 4 linked to MC006 closeout and no final-token steering",
            "actual": {
                "status": levels_by_id["level_4_parametric_fact_override"][
                    "ladder_status"
                ],
                "linked": levels_by_id["level_4_parametric_fact_override"][
                    "linked_work_order_ids"
                ],
                "forbidden_moves": levels_by_id[
                    "level_4_parametric_fact_override"
                ]["forbidden_moves"],
            },
            "passed": levels_by_id["level_4_parametric_fact_override"][
                "ladder_status"
            ]
            == "monitor_only_no_lever"
            and levels_by_id["level_4_parametric_fact_override"][
                "linked_work_order_ids"
            ]
            == ["close_mc006_predecision_frontier"]
            and any(
                "final-token" in move
                for move in levels_by_id["level_4_parametric_fact_override"][
                    "forbidden_moves"
                ]
            ),
            "why": "MC006 must stay a timing/frontier route unless it beats final output geometry.",
        },
        {
            "id": "real_uncertainty_requires_new_behavior_family",
            "predicate": "level 5 new_work_order_required and ready_count == 0",
            "actual": {
                "new_work_order_required": levels_by_id[
                    "level_5_real_abstention_uncertainty"
                ]["new_work_order_required"],
                "real_ready_count": payload["summary"][
                    "real_abstention_uncertainty_ready_count"
                ],
            },
            "passed": levels_by_id["level_5_real_abstention_uncertainty"][
                "new_work_order_required"
            ]
            and payload["summary"]["real_abstention_uncertainty_ready_count"] == 0,
            "why": "Real uncertainty remains below the behavior gate and needs a fresh substrate.",
        },
        {
            "id": "broad_knowledge_claims_remain_forbidden",
            "predicate": "promoted_level_count == 0 and forbidden claim names broad claims",
            "actual": {
                "promoted_level_count": payload["summary"]["promoted_level_count"],
                "forbidden_claim": payload["forbidden_claim"],
            },
            "passed": payload["summary"]["promoted_level_count"] == 0
            and "truth or knowledge vector" in payload["forbidden_claim"]
            and "real-world factual" in payload["forbidden_claim"],
            "why": "The gap plan must not launder missing evidence into broad claims.",
        },
    ]


def build_control_surface_knowledge_gap_plan() -> dict[str, Any]:
    knowledge_ladder = load_json(KNOWLEDGE_LADDER_PATH)
    family_matrix = load_json(FAMILY_MATRIX_PATH)
    gap_closure_plan = load_json(GAP_CLOSURE_PLAN_PATH)
    offensive_doctrine = load_json(OFFENSIVE_DOCTRINE_PATH)
    level_plans = build_level_plans(knowledge_ladder, offensive_doctrine)
    summary = build_summary(level_plans, knowledge_ladder, offensive_doctrine)
    payload = {
        "schema_version": 1,
        "updated_at": knowledge_ladder.get("updated_at"),
        "purpose": (
            "Convert the five-level knowledge ladder into missing-evidence "
            "decisions: which level needs reliability, transfer, a new behavior "
            "substrate, a predecision lever, or a route death."
        ),
        "sources": {
            "knowledge_ladder": rel(KNOWLEDGE_LADDER_PATH),
            "family_matrix": rel(FAMILY_MATRIX_PATH),
            "gap_closure_plan": rel(GAP_CLOSURE_PLAN_PATH),
            "offensive_doctrine": rel(OFFENSIVE_DOCTRINE_PATH),
        },
        "source_snapshot": {
            "family_matrix_row_count": family_matrix["summary"]["row_count"],
            "knowledge_ladder_level_ids": sorted(
                level["level_id"] for level in knowledge_ladder["levels"]
            ),
            "ladder_status_by_level": {
                level["level_id"]: level["status"]
                for level in knowledge_ladder["levels"]
            },
            "gap_closure_work_order_count": gap_closure_plan["summary"][
                "work_order_count"
            ],
            "offensive_work_order_ids": sorted(
                contract["work_order_id"]
                for contract in offensive_doctrine["branch_contracts"]
            ),
        },
        "classification_rule": {
            "plan_level": "One missing-evidence plan per predeclared knowledge-ladder level.",
            "linked_work_orders": "Only existing offensive-doctrine work orders are linked; future substrates are marked new_work_order_required.",
            "hidden_state_permission": "A level may license new hidden-state work only after behavior, null, locality, holdout, and output/candidate gates pass.",
        },
        "summary": summary,
        "level_plans": level_plans,
        "allowed_claim": (
            "The project has a generated knowledge-gap plan: MC005 needs "
            "reliability/transfer closure, MC006 needs a margin-decoupled "
            "predecision lever or timing closeout, the bridge and familiar-"
            "entity levels need materially new substrates, and real uncertainty "
            "is still below the behavior gate."
        ),
        "forbidden_claim": (
            "This plan does not establish a truth or knowledge vector, a "
            "promoted knowledge mechanism, a reliable MC006 steering route, "
            "or a real-world factual correction/refusal/uncertainty mechanism."
        ),
    }
    payload["validation_checks"] = build_validation_checks(payload)
    return payload


def validate_knowledge_gap_plan(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("knowledge gap plan schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"knowledge gap plan source missing: {rel_path}")
    if payload["summary"]["level_count"] != 5:
        raise AssertionError("knowledge gap plan expected five levels")
    if payload["summary"]["promoted_level_count"] != 0:
        raise AssertionError("knowledge gap plan must not report promoted levels")
    if payload["summary"]["new_hidden_state_search_allowed_level_count"] != 0:
        raise AssertionError("knowledge gap plan must not allow new hidden-state search")
    if payload["summary"]["levels_requiring_new_behavior_substrate_count"] != 3:
        raise AssertionError("knowledge gap plan expected three new-substrate levels")
    if payload["summary"]["new_work_order_required_level_count"] != 3:
        raise AssertionError("knowledge gap plan expected three future-work-order levels")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(f"knowledge gap plan checks failed: {failed_checks}")


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Control-Surface Knowledge Gap Plan",
        "",
        f"Source updated_at: {payload['updated_at']}",
        "",
        "Status: generated missing-evidence plan implemented and validated.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/control_surface_knowledge_gap_plan.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_knowledge_gap_plan.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_knowledge_gap_plan.py --write",
        "python code\\control_surface_knowledge_gap_plan.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Purpose",
        "",
        "This layer converts the knowledge ladder into explicit missing-evidence",
        "decisions. It treats typed failures as evidence about where the behavior",
        "currently lives instead of treating them as failed prose claims.",
        "",
        "## Generated Facts",
        "",
        f"- levels: {summary['level_count']};",
        f"- missing-evidence items: {summary['missing_evidence_item_count']};",
        f"- linked existing work orders: {summary['linked_existing_work_order_count']};",
        f"- levels needing new behavior substrates: {summary['levels_requiring_new_behavior_substrate_count']};",
        f"- levels needing predecision/reliability closure: {summary['levels_requiring_predecision_or_reliability_count']};",
        f"- levels requiring future work orders: {summary['new_work_order_required_level_count']};",
        f"- new hidden-state search allowed levels: {summary['new_hidden_state_search_allowed_level_count']};",
        f"- same-family route-killed levels: {summary['same_family_route_killed_level_count']};",
        f"- promoted levels: {summary['promoted_level_count']};",
        f"- bounded-reference levels: {summary['bounded_reference_level_count']};",
        f"- monitor-only levels: {summary['monitor_only_level_count']};",
        f"- real uncertainty mechanism-ready levels: {summary['real_abstention_uncertainty_ready_count']}.",
        "",
        "## Level Plan",
        "",
        "| Level | Status | Decision State | Missing Items | Existing Work Orders | New Work Order? |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for level in payload["level_plans"]:
        linked = ", ".join(f"`{work_id}`" for work_id in level["linked_work_order_ids"])
        if not linked:
            linked = "none"
        lines.append(
            f"| {level['label']} | `{level['ladder_status']}` | "
            f"`{level['decision_state']}` | {len(level['missing_evidence'])} | "
            f"{linked} | `{level['new_work_order_required']}` |"
        )

    for level in payload["level_plans"]:
        lines.extend(
            [
                "",
                f"## {level['label']}",
                "",
                f"- level id: `{level['level_id']}`;",
                f"- row ids: `{format_value(level['row_ids'])}`;",
                f"- decision state: `{level['decision_state']}`;",
                f"- next allowed action: {level['next_allowed_action']}",
                f"- insight: {level['insight']}",
                "",
                "Missing evidence:",
            ]
        )
        for item in level["missing_evidence"]:
            lines.append(
                f"- `{item['id']}` ({item['kind']}): {item['description']}"
            )
        lines.extend(["", "Decision rules:"])
        for rule_id in [
            "promotion_rule",
            "death_rule",
            "containment_rule",
            "export_rule",
        ]:
            lines.append(f"- `{rule_id}`: {level['decision_rules'][rule_id]}")
        lines.extend(["", "Forbidden moves:"])
        for move in level["forbidden_moves"]:
            lines.append(f"- {move}")
        if level["work_order_links"]:
            lines.extend(["", "Linked work orders:"])
            for link in level["work_order_links"]:
                lines.append(
                    f"- `{link['work_order_id']}` ({link['track_type']}, "
                    f"{link['urgency']}): {link['title']}"
                )

    lines.extend(
        [
            "",
            "## What This Proves",
            "",
            "It proves that the knowledge ambition has been converted into a",
            "level-by-level evidence ledger. The current map has one bounded",
            "synthetic reference, one monitor-only parametric-fact frontier,",
            "three levels that need new behavior substrates or fresh work orders,",
            "and zero promoted knowledge mechanisms.",
            "",
            "## What It Does Not Prove",
            "",
            "It does not prove a broad knowledge mechanism, a truth vector, a",
            "reliable steering route for MC006, or real-world factual correction",
            "or abstention control.",
            "",
        ]
    )
    return "\n".join(lines)


def write_knowledge_gap_plan(
    output_path: Path = KNOWLEDGE_GAP_PLAN_PATH,
    report_path: Path = KNOWLEDGE_GAP_PLAN_REPORT_PATH,
) -> dict[str, Any]:
    payload = build_control_surface_knowledge_gap_plan()
    validate_knowledge_gap_plan(payload)
    write_json(output_path, payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(payload), encoding="utf-8", newline="\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write knowledge-gap artifacts")
    parser.add_argument("--json", action="store_true", help="print knowledge-gap JSON")
    args = parser.parse_args()

    payload = build_control_surface_knowledge_gap_plan()
    validate_knowledge_gap_plan(payload)

    if args.write:
        write_knowledge_gap_plan()
        print(
            f"wrote {KNOWLEDGE_GAP_PLAN_PATH.relative_to(ROOT).as_posix()} and "
            f"{KNOWLEDGE_GAP_PLAN_REPORT_PATH.relative_to(ROOT).as_posix()} "
            f"with {payload['summary']['missing_evidence_item_count']} missing-evidence items"
        )
        return
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(
        "knowledge gap plan ok: "
        f"{payload['summary']['level_count']} levels, "
        f"{payload['summary']['missing_evidence_item_count']} missing-evidence items"
    )
    print(
        "decision_state_counts:",
        json.dumps(payload["summary"]["decision_state_counts"], sort_keys=True),
    )
    print(
        "linked_work_orders:",
        json.dumps(payload["summary"]["linked_existing_work_order_ids"]),
    )


if __name__ == "__main__":
    main()
