"""Build the MC006 predecision-frontier audit.

MC006 is the atlas' knowledge-like timing case: the behavior substrate exists
and predecision monitors exist, but output/candidate geometry, steering,
shuffle, and transfer gates block a causal control claim. This module keeps
that boundary validator-backed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from control_surface_artifacts import ATLAS_PATH, ROOT, load_json
from control_surface_decision_frontier import DECISION_FRONTIER_PATH
from control_surface_gate_geometry import GATE_GEOMETRY_PATH
from control_surface_offensive_doctrine import OFFENSIVE_DOCTRINE_PATH
from control_surface_reliability_matrix import RELIABILITY_MATRIX_PATH
from control_surface_route_disposition import ROUTE_DISPOSITION_PATH
from control_surface_transfer_matrix import TRANSFER_MATRIX_PATH


MC006_PREDECISION_FRONTIER_AUDIT_PATH = (
    ROOT / "data" / "mc006_predecision_frontier_audit.json"
)
MC006_PREDECISION_FRONTIER_AUDIT_REPORT_PATH = (
    ROOT / "research" / "cards" / "MC006_PREDECISION_FRONTIER_AUDIT.md"
)
MC006_CLOSEOUT_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "MC006"
    / "mc006_predecision_frontier_closeout_audit_20260701T200738.json"
)

WORK_ORDER_ID = "close_mc006_predecision_frontier"
ROW_ID = "mc006_parametric_fact_override"


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")


def format_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def get_work_order(doctrine: dict[str, Any]) -> dict[str, Any]:
    matches = [
        contract
        for contract in doctrine["branch_contracts"]
        if contract["work_order_id"] == WORK_ORDER_ID
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one offensive-doctrine contract {WORK_ORDER_ID}, "
            f"got {len(matches)}"
        )
    return matches[0]


def get_by_row_id(entries: list[dict[str, Any]], row_id: str, label: str) -> dict[str, Any]:
    matches = [entry for entry in entries if entry.get("row_id") == row_id]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one {label} entry {row_id}, got {len(matches)}")
    return matches[0]


def get_by_id(entries: list[dict[str, Any]], row_id: str, label: str) -> dict[str, Any]:
    matches = [entry for entry in entries if entry.get("id") == row_id]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one {label} entry {row_id}, got {len(matches)}")
    return matches[0]


def get_atlas_row(atlas: dict[str, Any]) -> dict[str, Any]:
    matches = [row for row in atlas["rows"] if row["id"] == ROW_ID]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one atlas row {ROW_ID}, got {len(matches)}")
    return matches[0]


def build_source_snapshot(
    atlas_row: dict[str, Any],
    frontier_entry: dict[str, Any],
    route_entry: dict[str, Any],
    reliability_entry: dict[str, Any],
    transfer_entry: dict[str, Any],
    gate_entry: dict[str, Any],
    closeout: dict[str, Any],
) -> dict[str, Any]:
    return {
        "atlas": {
            "verdict": atlas_row["verdict"]["class"],
            "lead_time_state": atlas_row["lead_time"]["state"],
            "intervention_state": atlas_row["intervention"]["state"],
            "diagnostics": atlas_row["diagnostics"],
            "allowed_claims": atlas_row["allowed_claims"],
            "forbidden_claims": atlas_row["forbidden_claims"],
            "next_decision": atlas_row["next_decision"],
        },
        "decision_frontier": {
            "frontier_class": frontier_entry["frontier_class"],
            "lead_time_state": frontier_entry["lead_time_state"],
            "lead_time_internal_signal": frontier_entry["lead_time_internal_signal"],
            "output_geometry": frontier_entry["output_geometry"],
            "primary_blocker": frontier_entry["primary_blocker"],
            "next_action": frontier_entry["next_action"],
        },
        "route_disposition": {
            "disposition": route_entry["disposition"],
            "death_rule": route_entry["death_rule"],
            "allowed_action": route_entry["allowed_action"],
            "promotion_rule": route_entry["promotion_rule"],
        },
        "reliability": {
            "reliability_class": reliability_entry["reliability_class"],
            "gate_statuses": reliability_entry["gate_statuses"],
            "failed_or_missing_gates": reliability_entry["failed_or_missing_gates"],
        },
        "transfer": {
            "transfer_class": transfer_entry["transfer_class"],
            "transfer_value": transfer_entry["transfer_value"],
            "required_gate": transfer_entry["required_gate"],
            "widening_action": transfer_entry["widening_action"],
        },
        "gate_geometry": {
            "terminal_stage": gate_entry["terminal_stage"],
            "primary_blocker": gate_entry["primary_blocker"],
            "claim_bar_action": gate_entry["claim_bar_action"],
        },
        "closeout": {
            "verdict": closeout["decision"]["verdict"],
            "route_status": closeout["decision"]["route_status"],
            "hidden_state_work_allowed": closeout["decision"][
                "hidden_state_work_allowed"
            ],
            "intervention_allowed": closeout["decision"]["intervention_allowed"],
            "ordinary_route_repairs_allowed": closeout["decision"][
                "ordinary_route_repairs_allowed"
            ],
            "known_confounded_causal_stress_allowed": closeout["decision"][
                "known_confounded_causal_stress_allowed"
            ],
            "materially_new_behavior_family_required": closeout["decision"][
                "materially_new_behavior_family_required"
            ],
            "criteria": closeout["criteria"],
            "behavior_substrate": closeout["evidence"]["behavior_substrate"],
            "predecision_monitor": closeout["evidence"]["predecision_monitor"],
            "promotion_blockers": closeout["evidence"]["promotion_blockers"],
            "delayed_interface_boundary": closeout["evidence"][
                "delayed_interface_boundary"
            ],
        },
    }


def build_validation_checks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    snapshot = payload["source_snapshot"]
    criteria = snapshot["closeout"]["criteria"]
    behavior = snapshot["closeout"]["behavior_substrate"]
    monitor = snapshot["closeout"]["predecision_monitor"]
    blockers = snapshot["closeout"]["promotion_blockers"]
    delayed = snapshot["closeout"]["delayed_interface_boundary"]
    claim_flags = payload["claim_boundary"]["claim_flags"]
    generated_layer_statuses = {
        "frontier_class": snapshot["decision_frontier"]["frontier_class"],
        "route_disposition": snapshot["route_disposition"]["disposition"],
        "reliability_class": snapshot["reliability"]["reliability_class"],
        "transfer_class": snapshot["transfer"]["transfer_class"],
        "terminal_stage": snapshot["gate_geometry"]["terminal_stage"],
    }
    checks = [
        {
            "id": "work_order_contract_matches_mc006_predecision_frontier",
            "predicate": f"== {WORK_ORDER_ID}",
            "actual": payload["work_order"]["id"],
            "passed": payload["work_order"]["id"] == WORK_ORDER_ID,
            "why": "The audit must come from the offensive-doctrine MC006 predecision-frontier contract.",
        },
        {
            "id": "atlas_row_is_monitor_only_diagnostic",
            "predicate": "diagnostic_note with lead_time_monitor_only and failed intervention",
            "actual": {
                "verdict": snapshot["atlas"]["verdict"],
                "lead_time_state": snapshot["atlas"]["lead_time_state"],
                "intervention_state": snapshot["atlas"]["intervention_state"],
            },
            "passed": snapshot["atlas"]["verdict"] == "diagnostic_note"
            and snapshot["atlas"]["lead_time_state"] == "lead_time_monitor_only"
            and snapshot["atlas"]["intervention_state"] == "failed",
            "why": "MC006 should be recorded as monitor-only, not as a mechanism card.",
        },
        {
            "id": "closeout_decision_closes_hidden_state_and_ordinary_repair",
            "predicate": "monitor_only_closed, no hidden-state work, no intervention, no ordinary repair",
            "actual": {
                "route_status": snapshot["closeout"]["route_status"],
                "hidden_state_work_allowed": snapshot["closeout"][
                    "hidden_state_work_allowed"
                ],
                "intervention_allowed": snapshot["closeout"]["intervention_allowed"],
                "ordinary_route_repairs_allowed": snapshot["closeout"][
                    "ordinary_route_repairs_allowed"
                ],
            },
            "passed": snapshot["closeout"]["route_status"] == "monitor_only_closed"
            and snapshot["closeout"]["hidden_state_work_allowed"] is False
            and snapshot["closeout"]["intervention_allowed"] is False
            and snapshot["closeout"]["ordinary_route_repairs_allowed"] is False,
            "why": "The delayed-city/predecision route must be closed before more hidden-state searches.",
        },
        {
            "id": "behavior_substrate_passed_before_signature_claim",
            "predicate": "V14 substrate passed with 30 binary rows and holdout labels on both sides",
            "actual": behavior,
            "passed": criteria["behavior_substrate_passed"] is True
            and behavior["selected_binary_rows"] == 30
            and behavior["holdout_label_counts"]["true_answer"] >= 2
            and behavior["holdout_label_counts"]["override_answer"] >= 2,
            "why": "The route is knowledge-like because it has a matched generated behavior substrate.",
        },
        {
            "id": "predecision_monitor_exists_but_does_not_beat_global_geometry",
            "predicate": "holdout AUC 1.0, beats same-position output, fails candidate/final margins",
            "actual": monitor,
            "passed": criteria["predecision_monitor_supported"] is True
            and monitor["selected_position"] == "after_mapping_line"
            and monitor["selected_layer"] == 4
            and monitor["selected_holdout_auc"] == 1.0
            and monitor["beats_same_position_output"] is True
            and monitor["beats_candidate_score"] is False
            and monitor["beats_final_next_token_output"] is False,
            "why": "MC006 has a real monitor, but not an upstream control surface.",
        },
        {
            "id": "final_or_candidate_geometry_blocks_promotion",
            "predicate": "geometry blocker true and source/path shadow recorded",
            "actual": {
                "final_or_candidate_geometry_blocks_promotion": criteria[
                    "final_or_candidate_geometry_blocks_promotion"
                ],
                "v18_global_margin_separation": blockers[
                    "v18_global_margin_separation"
                ],
                "v21_pair_matching_failed": blockers["v21_pair_matching_failed"],
                "v22_source_path_shadow": blockers["v22_source_path_shadow"],
            },
            "passed": criteria["final_or_candidate_geometry_blocks_promotion"] is True
            and blockers["v18_global_margin_separation"]
            == "global_margin_separation_blocks_matching"
            and blockers["v21_pair_matching_failed"]
            == "approximate_pair_matching_failed_margin_baselines"
            and blockers["v22_source_path_shadow"] == "source_path_final_margin_shadow",
            "why": "The hidden monitor remains downstream-visible through candidate/output geometry.",
        },
        {
            "id": "causal_shuffle_and_transfer_routes_are_closed",
            "predicate": "intervention failed, shuffle overfit, transfer/bank insufficient",
            "actual": {
                "causal_and_transfer_routes_closed": criteria[
                    "causal_and_transfer_routes_closed"
                ],
                "candidate_decoupled_hidden_selection_failed_shuffle_null": criteria[
                    "candidate_decoupled_hidden_selection_failed_shuffle_null"
                ],
                "v17_intervention_failed": blockers["v17_intervention_failed"],
                "v25_shuffle_overfit": blockers["v25_shuffle_overfit"],
                "v26_transfer_failed": blockers["v26_transfer_failed"],
                "v28_transfer_role_insufficient": blockers[
                    "v28_transfer_role_insufficient"
                ],
            },
            "passed": criteria["causal_and_transfer_routes_closed"] is True
            and criteria["candidate_decoupled_hidden_selection_failed_shuffle_null"]
            is True
            and blockers["v17_intervention_failed"] == "intervention_failed"
            and blockers["v25_shuffle_overfit"]
            == "candidate_decoupled_hidden_shuffle_overfit"
            and blockers["v26_transfer_failed"] == "locked_coordinate_transfer_failed"
            and blockers["v28_transfer_role_insufficient"]
            == "transfer_role_repair_bank_insufficient",
            "why": "Output-geometry controls are not the only reason the route is closed.",
        },
        {
            "id": "delayed_interface_does_not_repair_route",
            "predicate": "first-token barrier broken but candidate-decoupled/transfer bank still not ready",
            "actual": delayed,
            "passed": delayed["v24_selected_first_token_city_rate_at_most_0p1"] is True
            and delayed["v24_selected_final_city_margin_overlap_exists"] is True
            and delayed["v25_candidate_decoupled_template_found"] is True
            and delayed["v25_hidden_beats_shuffle_null"] is False
            and delayed["v28_any_transfer_candidate_decoupled_template"] is True
            and delayed["v28_transfer_repair_bank_ready"] is False,
            "why": "The delayed-city wrapper reduces one output barrier but does not produce a reliable control surface.",
        },
        {
            "id": "generated_layers_agree_on_monitor_only_status",
            "predicate": "frontier, route, reliability, transfer, and gate status align",
            "actual": generated_layer_statuses,
            "passed": generated_layer_statuses
            == {
                "frontier_class": "predecision_monitor_no_lever",
                "route_disposition": "monitor_only_closed",
                "reliability_class": "not_reliable_monitor_only_no_lever",
                "transfer_class": "transfer_failed_or_bank_insufficient",
                "terminal_stage": "signature_monitor_no_lever",
            },
            "why": "All generated map layers must agree that MC006 is monitor-only.",
        },
        {
            "id": "claim_boundary_forbids_knowledge_vector_and_steering",
            "predicate": "all forbidden claim flags false",
            "actual": claim_flags,
            "passed": not any(claim_flags.values()),
            "why": "The knowledge-like route must not be overclaimed as a knowledge vector.",
        },
        {
            "id": "future_work_requires_new_family_or_labeled_stress_test",
            "predicate": "ordinary repairs false; new family and confounded stress true",
            "actual": payload["future_work_admission_rule"],
            "passed": payload["future_work_admission_rule"][
                "ordinary_route_repairs_allowed"
            ]
            is False
            and payload["future_work_admission_rule"][
                "known_confounded_causal_stress_allowed"
            ]
            is True
            and payload["future_work_admission_rule"][
                "materially_new_behavior_family_required"
            ]
            is True,
            "why": "Future MC006 work must not restart ordinary prompt repair inside the closed route.",
        },
    ]
    return checks


def build_mc006_predecision_frontier_audit() -> dict[str, Any]:
    doctrine = load_json(OFFENSIVE_DOCTRINE_PATH)
    atlas = load_json(ATLAS_PATH)
    decision_frontier = load_json(DECISION_FRONTIER_PATH)
    route = load_json(ROUTE_DISPOSITION_PATH)
    reliability = load_json(RELIABILITY_MATRIX_PATH)
    transfer = load_json(TRANSFER_MATRIX_PATH)
    gate = load_json(GATE_GEOMETRY_PATH)
    closeout = load_json(MC006_CLOSEOUT_RESULT_PATH)
    work_order = get_work_order(doctrine)
    atlas_row = get_atlas_row(atlas)
    frontier_entry = get_by_id(
        decision_frontier["frontier_rows"], ROW_ID, "decision frontier"
    )
    route_entry = get_by_row_id(route["route_entries"], ROW_ID, "route disposition")
    reliability_entry = get_by_row_id(
        reliability["reliability_entries"], ROW_ID, "reliability"
    )
    transfer_entry = get_by_row_id(transfer["transfer_entries"], ROW_ID, "transfer")
    gate_entry = get_by_row_id(gate["gate_entries"], ROW_ID, "gate geometry")
    source_snapshot = build_source_snapshot(
        atlas_row,
        frontier_entry,
        route_entry,
        reliability_entry,
        transfer_entry,
        gate_entry,
        closeout,
    )
    payload = {
        "schema_version": 1,
        "purpose": (
            "Freeze MC006 as the knowledge-like predecision monitor route: "
            "a matched behavior substrate and predecision monitors exist, but "
            "final/candidate geometry, failed steering, shuffled-label nulls, "
            "and transfer-bank insufficiency block mechanism promotion."
        ),
        "sources": {
            "atlas": rel(ATLAS_PATH),
            "offensive_doctrine": rel(OFFENSIVE_DOCTRINE_PATH),
            "decision_frontier": rel(DECISION_FRONTIER_PATH),
            "route_disposition": rel(ROUTE_DISPOSITION_PATH),
            "reliability_matrix": rel(RELIABILITY_MATRIX_PATH),
            "transfer_matrix": rel(TRANSFER_MATRIX_PATH),
            "gate_geometry": rel(GATE_GEOMETRY_PATH),
            "closeout_result": rel(MC006_CLOSEOUT_RESULT_PATH),
        },
        "work_order": {
            "id": work_order["work_order_id"],
            "urgency": work_order["urgency"],
            "track_type": work_order["track_type"],
            "target_gap_ids": work_order["target_gap_ids"],
            "first_artifact": work_order["first_artifact"],
            "iteration_budget": work_order["iteration_budget"],
            "promotion_rule": work_order["promotion_rule"],
            "bound_rule": work_order["bound_rule"],
            "kill_rule": work_order["kill_rule"],
            "containment_rule": work_order["containment_rule"],
            "export_rule": work_order["export_rule"],
        },
        "row_id": ROW_ID,
        "source_snapshot": source_snapshot,
        "future_work_admission_rule": {
            "ordinary_route_repairs_allowed": False,
            "known_confounded_causal_stress_allowed": source_snapshot["closeout"][
                "known_confounded_causal_stress_allowed"
            ],
            "materially_new_behavior_family_required": source_snapshot["closeout"][
                "materially_new_behavior_family_required"
            ],
            "allowed_future_work": [
                "known-confounded causal stress test explicitly labeled as such",
                "materially new behavior family or prompt contract that reopens behavior-substrate work from first principles",
            ],
            "forbidden_future_work": work_order["forbidden_moves"],
        },
        "claim_boundary": {
            "allowed_claim": (
                "MC006 has a matched generated capital-fact override substrate "
                "and predecision monitor signals, but the V14-V28 delayed-city "
                "route is monitor-only rather than causal."
            ),
            "forbidden_claim": (
                "MC006 does not supply a knowledge vector, promoted or bounded "
                "mechanism card, licensed steering direction, or ordinary "
                "same-route repair path."
            ),
            "claim_flags": {
                "knowledge_vector_claimed": False,
                "promoted_mechanism_claimed": False,
                "bounded_mechanism_claimed": False,
                "steering_authorized": False,
                "ordinary_route_repair_allowed": False,
            },
        },
        "summary": {
            "row_id": ROW_ID,
            "verdict": source_snapshot["atlas"]["verdict"],
            "route_status": source_snapshot["closeout"]["route_status"],
            "frontier_class": source_snapshot["decision_frontier"]["frontier_class"],
            "terminal_stage": source_snapshot["gate_geometry"]["terminal_stage"],
            "reliability_class": source_snapshot["reliability"]["reliability_class"],
            "transfer_class": source_snapshot["transfer"]["transfer_class"],
            "behavior_substrate_passed": source_snapshot["closeout"]["criteria"][
                "behavior_substrate_passed"
            ],
            "predecision_monitor_supported": source_snapshot["closeout"][
                "criteria"
            ]["predecision_monitor_supported"],
            "final_or_candidate_geometry_blocks_promotion": source_snapshot[
                "closeout"
            ]["criteria"]["final_or_candidate_geometry_blocks_promotion"],
            "promotion_gate_passed": source_snapshot["closeout"]["criteria"][
                "promotion_gate_passed"
            ],
            "monitor_only_closeout_gate_passed": source_snapshot["closeout"][
                "criteria"
            ]["monitor_only_closeout_gate_passed"],
        },
    }
    payload["validation_checks"] = build_validation_checks(payload)
    return payload


def validate_mc006_predecision_frontier_audit(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("MC006 predecision-frontier audit schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"MC006 predecision-frontier source missing: {rel_path}")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(f"MC006 predecision-frontier checks failed: {failed_checks}")


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    work_order = payload["work_order"]
    closeout = payload["source_snapshot"]["closeout"]
    behavior = closeout["behavior_substrate"]
    monitor = closeout["predecision_monitor"]
    blockers = closeout["promotion_blockers"]
    delayed = closeout["delayed_interface_boundary"]
    future = payload["future_work_admission_rule"]
    lines = [
        "# MC006 Predecision Frontier Audit",
        "",
        "Date: 2026-07-01",
        "",
        "Status: generated monitor-only knowledge-frontier audit; no hidden-state license.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/mc006_predecision_frontier_audit.json`",
        "",
        "Builder:",
        "",
        "> `code/mc006_predecision_frontier_audit.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\mc006_predecision_frontier_audit.py --write",
        "python code\\mc006_predecision_frontier_audit.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Purpose",
        "",
        payload["purpose"],
        "",
        "## Generated Facts",
        "",
        f"- row id: `{summary['row_id']}`;",
        f"- verdict: `{summary['verdict']}`;",
        f"- route status: `{summary['route_status']}`;",
        f"- frontier class: `{summary['frontier_class']}`;",
        f"- terminal stage: `{summary['terminal_stage']}`;",
        f"- reliability class: `{summary['reliability_class']}`;",
        f"- transfer class: `{summary['transfer_class']}`;",
        f"- behavior substrate passed: {summary['behavior_substrate_passed']};",
        f"- predecision monitor supported: {summary['predecision_monitor_supported']};",
        f"- final/candidate geometry blocks promotion: {summary['final_or_candidate_geometry_blocks_promotion']};",
        f"- promotion gate passed: {summary['promotion_gate_passed']};",
        f"- monitor-only closeout gate passed: {summary['monitor_only_closeout_gate_passed']}.",
        "",
        "## Decisive Metrics",
        "",
        f"- V14 selected template: `{behavior['selected_template']}`;",
        f"- V14 selected binary rows: `{behavior['selected_binary_rows']}`;",
        f"- V14 holdout label counts: `{format_value(behavior['holdout_label_counts'])}`;",
        f"- V16 selected position/layer: `{monitor['selected_position']}` / `{monitor['selected_layer']}`;",
        f"- V16 selected holdout AUC: `{monitor['selected_holdout_auc']}`;",
        f"- V16 beats same-position output: `{monitor['beats_same_position_output']}`;",
        f"- V16 beats candidate score: `{monitor['beats_candidate_score']}`;",
        f"- V16 beats final next-token output: `{monitor['beats_final_next_token_output']}`;",
        f"- V17 blocker: `{blockers['v17_intervention_failed']}`;",
        f"- V22 blocker: `{blockers['v22_source_path_shadow']}`;",
        f"- V25 hidden beats shuffle null: `{delayed['v25_hidden_beats_shuffle_null']}`;",
        f"- V28 transfer repair bank ready: `{delayed['v28_transfer_repair_bank_ready']}`.",
        "",
        "## Decision Rules",
        "",
        f"- work order: `{work_order['id']}`;",
        f"- iteration budget: {work_order['iteration_budget']}",
        f"- promotion rule: {work_order['promotion_rule']}",
        f"- bound rule: {work_order['bound_rule']}",
        f"- kill rule: {work_order['kill_rule']}",
        f"- containment rule: {work_order['containment_rule']}",
        f"- export rule: {work_order['export_rule']}",
        "",
        "## Future Work Admission",
        "",
        f"- ordinary route repairs allowed: {future['ordinary_route_repairs_allowed']};",
        f"- known-confounded causal stress allowed: {future['known_confounded_causal_stress_allowed']};",
        f"- materially new behavior family required: {future['materially_new_behavior_family_required']};",
        f"- allowed future work: `{format_value(future['allowed_future_work'])}`;",
        f"- forbidden future work: `{format_value(future['forbidden_future_work'])}`.",
        "",
        "## Claim Boundary",
        "",
        payload["claim_boundary"]["allowed_claim"],
        "",
        payload["claim_boundary"]["forbidden_claim"],
        "",
        "## What This Proves",
        "",
        "It proves that the knowledge-like MC006 route is a timing diagnostic:",
        "the model can expose predecision monitors before the final answer",
        "interface, but those monitors are not reliable causal levers under the",
        "current prompt family.",
        "",
        "## What It Does Not Prove",
        "",
        "It does not prove a knowledge vector, a promoted or bounded mechanism",
        "card, an actionable steering direction, or transfer-ready factual",
        "control.",
        "",
    ]
    return "\n".join(lines)


def write_mc006_predecision_frontier_audit(
    output_path: Path = MC006_PREDECISION_FRONTIER_AUDIT_PATH,
    report_path: Path = MC006_PREDECISION_FRONTIER_AUDIT_REPORT_PATH,
) -> dict[str, Any]:
    payload = build_mc006_predecision_frontier_audit()
    validate_mc006_predecision_frontier_audit(payload)
    write_json(output_path, payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(payload), encoding="utf-8", newline="\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="write MC006 predecision-frontier artifacts",
    )
    parser.add_argument("--json", action="store_true", help="print audit JSON")
    args = parser.parse_args()

    payload = build_mc006_predecision_frontier_audit()
    validate_mc006_predecision_frontier_audit(payload)

    if args.write:
        write_mc006_predecision_frontier_audit()
        print(
            f"wrote {MC006_PREDECISION_FRONTIER_AUDIT_PATH.relative_to(ROOT).as_posix()} "
            f"and {MC006_PREDECISION_FRONTIER_AUDIT_REPORT_PATH.relative_to(ROOT).as_posix()} "
            f"for {payload['summary']['row_id']}"
        )
        return
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(
        "MC006 predecision frontier audit ok: "
        f"{payload['summary']['route_status']}, "
        f"{payload['summary']['frontier_class']}, "
        f"promotion={payload['summary']['promotion_gate_passed']}"
    )


if __name__ == "__main__":
    main()
