"""Build the MC005 reference-specimen audit.

MC005 is the atlas' bounded positive control: it has a real internal causal
surface, but reliability and transfer gates prevent full promotion. This module
keeps that status machine-checkable across the generated stack.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from control_surface_artifacts import ATLAS_PATH, ROOT, load_json
from control_surface_gate_geometry import GATE_GEOMETRY_PATH
from control_surface_offensive_doctrine import OFFENSIVE_DOCTRINE_PATH
from control_surface_reliability_matrix import RELIABILITY_MATRIX_PATH
from control_surface_route_disposition import ROUTE_DISPOSITION_PATH
from control_surface_transfer_matrix import TRANSFER_MATRIX_PATH


MC005_REFERENCE_SPECIMEN_AUDIT_PATH = (
    ROOT / "data" / "mc005_reference_specimen_audit.json"
)
MC005_REFERENCE_SPECIMEN_AUDIT_REPORT_PATH = (
    ROOT / "research" / "cards" / "MC005_REFERENCE_SPECIMEN_AUDIT.md"
)
MC005_CLOSEOUT_RESULT_PATH = (
    ROOT
    / "results"
    / "cards"
    / "MC005"
    / "mc005_write_replacement_closeout_audit_20260701T200017.json"
)

WORK_ORDER_ID = "close_mc005_reference_specimen"
ROW_ID = "mc005_associative_lookup"


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


def get_atlas_row(atlas: dict[str, Any]) -> dict[str, Any]:
    matches = [row for row in atlas["rows"] if row["id"] == ROW_ID]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one atlas row {ROW_ID}, got {len(matches)}")
    return matches[0]


def build_source_snapshot(
    atlas_row: dict[str, Any],
    route_entry: dict[str, Any],
    reliability_entry: dict[str, Any],
    transfer_entry: dict[str, Any],
    gate_entry: dict[str, Any],
    closeout: dict[str, Any],
) -> dict[str, Any]:
    v31 = closeout["evidence"]["write_replacement_family"]["v31_margin_boundary"]
    v29 = closeout["evidence"]["write_replacement_family"]["v29_attention_write"]
    return {
        "atlas": {
            "verdict": atlas_row["verdict"]["class"],
            "intervention_state": atlas_row["intervention"]["state"],
            "diagnostics": atlas_row["diagnostics"],
            "allowed_claims": atlas_row["allowed_claims"],
            "forbidden_claims": atlas_row["forbidden_claims"],
            "next_decision": atlas_row["next_decision"],
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
            "bounded_mechanism_preserved": closeout["decision"][
                "bounded_mechanism_preserved"
            ],
            "full_promotion_allowed": closeout["decision"]["full_promotion_allowed"],
            "same_write_route_closed": closeout["decision"]["same_write_route_closed"],
            "hidden_state_or_new_intervention_allowed": closeout["decision"][
                "hidden_state_or_new_intervention_allowed"
            ],
            "criteria": closeout["criteria"],
            "v29_recovery": {
                "target_write_delta_recovery_vs_direct": v29[
                    "target_write_delta_recovery_vs_direct"
                ],
                "target_write_win_loss_recovery_vs_direct": v29[
                    "target_write_win_loss_recovery_vs_direct"
                ],
                "write_source_controls_pass": v29["write_source_controls_pass"],
                "write_nulls_clean": v29["write_nulls_clean"],
            },
            "v31_boundary": {
                "lookup_target_write_mean_delta": v31[
                    "lookup_target_write_mean_delta"
                ],
                "lookup_target_win_loss": v31["lookup_target_win_loss"],
                "lookup_loss_margin_bands": v31["lookup_loss_margin_bands"],
                "combined_null_flip_count": v31["combined_null_flip_count"],
                "combined_null_flip_margin_bands": v31[
                    "combined_null_flip_margin_bands"
                ],
                "v30_imported_flips_abs_margin_le_0p5": v31[
                    "v30_imported_flips_abs_margin_le_0p5"
                ],
            },
        },
    }


def build_validation_checks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    snapshot = payload["source_snapshot"]
    criteria = snapshot["closeout"]["criteria"]
    v29 = snapshot["closeout"]["v29_recovery"]
    v31 = snapshot["closeout"]["v31_boundary"]
    claim_flags = payload["claim_boundary"]["claim_flags"]
    generated_layer_statuses = {
        "route_disposition": snapshot["route_disposition"]["disposition"],
        "reliability_class": snapshot["reliability"]["reliability_class"],
        "transfer_class": snapshot["transfer"]["transfer_class"],
        "terminal_stage": snapshot["gate_geometry"]["terminal_stage"],
    }
    lookup_high_margin = (
        v31["lookup_loss_margin_bands"]["gt_2"] == v31["lookup_target_win_loss"]
        and v31["lookup_target_win_loss"] == 11
    )
    null_boundary_bands_match = (
        v31["combined_null_flip_count"] == 5
        and v31["combined_null_flip_margin_bands"]["le_0p25"] == 4
        and v31["combined_null_flip_margin_bands"]["0p5_1"] == 1
        and v31["combined_null_flip_margin_bands"]["gt_2"] == 0
    )
    checks = [
        {
            "id": "work_order_contract_matches_mc005_reference_specimen",
            "predicate": f"== {WORK_ORDER_ID}",
            "actual": payload["work_order"]["id"],
            "passed": payload["work_order"]["id"] == WORK_ORDER_ID,
            "why": "The audit must come from the offensive-doctrine MC005 closeout contract.",
        },
        {
            "id": "atlas_row_is_bounded_mechanism_card",
            "predicate": "verdict == bounded_mechanism_card",
            "actual": snapshot["atlas"]["verdict"],
            "passed": snapshot["atlas"]["verdict"] == "bounded_mechanism_card",
            "why": "MC005 is the atlas reference specimen, not a promoted mechanism.",
        },
        {
            "id": "closeout_decision_freezes_bounded_not_promoted",
            "predicate": "bounded true, promotion false, same route closed",
            "actual": {
                "bounded": snapshot["closeout"]["bounded_mechanism_preserved"],
                "promotion": snapshot["closeout"]["full_promotion_allowed"],
                "same_write_route_closed": snapshot["closeout"][
                    "same_write_route_closed"
                ],
                "route_status": snapshot["closeout"]["route_status"],
            },
            "passed": snapshot["closeout"]["bounded_mechanism_preserved"] is True
            and snapshot["closeout"]["full_promotion_allowed"] is False
            and snapshot["closeout"]["same_write_route_closed"] is True
            and snapshot["closeout"]["route_status"] == "bounded_frozen_not_promoted",
            "why": "The closeout must force a verdict rather than leave MC005 in repair mode.",
        },
        {
            "id": "lookup_write_effect_exact_and_source_controls_pass",
            "predicate": "exact recovery == 1.0 and controls pass",
            "actual": {
                "lookup_write_effect_exact": criteria["lookup_write_effect_exact"],
                "write_source_controls_pass": criteria["write_source_controls_pass"],
                "target_write_delta_recovery_vs_direct": v29[
                    "target_write_delta_recovery_vs_direct"
                ],
                "target_write_win_loss_recovery_vs_direct": v29[
                    "target_write_win_loss_recovery_vs_direct"
                ],
            },
            "passed": criteria["lookup_write_effect_exact"] is True
            and criteria["write_source_controls_pass"] is True
            and v29["target_write_delta_recovery_vs_direct"] == 1.0
            and v29["target_write_win_loss_recovery_vs_direct"] == 1.0,
            "why": "The bounded positive control depends on exact lookup mediation, not just a probe.",
        },
        {
            "id": "strict_null_locality_blocks_promotion",
            "predicate": "strict nulls false and null boundary reproduced true",
            "actual": {
                "strict_answer_absent_nulls_clean": criteria[
                    "strict_answer_absent_nulls_clean"
                ],
                "null_boundary_reproduced": criteria["null_boundary_reproduced"],
                "write_nulls_clean": v29["write_nulls_clean"],
            },
            "passed": criteria["strict_answer_absent_nulls_clean"] is False
            and criteria["null_boundary_reproduced"] is True
            and v29["write_nulls_clean"] is False,
            "why": "MC005 is bounded specifically because the write route is not strictly null-local.",
        },
        {
            "id": "lookup_losses_are_high_margin",
            "predicate": "11/11 lookup losses in >2 margin band",
            "actual": {
                "lookup_target_win_loss": v31["lookup_target_win_loss"],
                "lookup_loss_margin_bands": v31["lookup_loss_margin_bands"],
                "lookup_target_write_mean_delta": v31["lookup_target_write_mean_delta"],
            },
            "passed": lookup_high_margin and v31["lookup_target_write_mean_delta"] < -6.0,
            "why": "The primary lookup effect is separated from the low-margin null flips.",
        },
        {
            "id": "null_flips_are_low_to_moderate_margin_not_fixed_by_0p5_cutoff",
            "predicate": "5 flips, 4 <=0.25, 1 in 0.5-1, cutoff criterion false",
            "actual": {
                "combined_null_flip_count": v31["combined_null_flip_count"],
                "combined_null_flip_margin_bands": v31[
                    "combined_null_flip_margin_bands"
                ],
                "v30_imported_flips_abs_margin_le_0p5": v31[
                    "v30_imported_flips_abs_margin_le_0p5"
                ],
            },
            "passed": null_boundary_bands_match
            and v31["v30_imported_flips_abs_margin_le_0p5"] is False,
            "why": "The null boundary is reproducible and broader than the strict 0.5 margin explanation.",
        },
        {
            "id": "alternative_intervention_family_does_not_promote",
            "predicate": "alternative family tested and not promotable",
            "actual": {
                "alternative_local_intervention_family_tested": criteria[
                    "alternative_local_intervention_family_tested"
                ],
                "alternative_local_intervention_family_not_promotable": criteria[
                    "alternative_local_intervention_family_not_promotable"
                ],
            },
            "passed": criteria["alternative_local_intervention_family_tested"] is True
            and criteria["alternative_local_intervention_family_not_promotable"] is True,
            "why": "The audit must compare the write route with another local intervention family.",
        },
        {
            "id": "generated_layers_agree_on_reference_status",
            "predicate": "bounded route, bounded reliability, fragile transfer, reliability boundary",
            "actual": generated_layer_statuses,
            "passed": generated_layer_statuses
            == {
                "route_disposition": "bounded_mechanism_frozen",
                "reliability_class": "bounded_reliability_reference",
                "transfer_class": "bounded_transfer_fragile_reference",
                "terminal_stage": "reliability_null_boundary",
            },
            "why": "All generated map layers must treat MC005 as the same bounded reference specimen.",
        },
        {
            "id": "claim_boundary_forbids_promotion_transfer_and_general_knowledge",
            "predicate": "all forbidden claim flags false",
            "actual": claim_flags,
            "passed": not any(claim_flags.values()),
            "why": "The reference specimen must not become a broad knowledge-control claim.",
        },
        {
            "id": "future_work_admission_is_not_same_route_repair",
            "predicate": "same_route_repair_allowed == false",
            "actual": payload["future_work_admission_rule"]["same_route_repair_allowed"],
            "passed": payload["future_work_admission_rule"][
                "same_route_repair_allowed"
            ]
            is False,
            "why": "The write-replacement route is frozen unless future work changes intervention family or width-transfer question.",
        },
    ]
    return checks


def build_mc005_reference_specimen_audit() -> dict[str, Any]:
    doctrine = load_json(OFFENSIVE_DOCTRINE_PATH)
    atlas = load_json(ATLAS_PATH)
    route = load_json(ROUTE_DISPOSITION_PATH)
    reliability = load_json(RELIABILITY_MATRIX_PATH)
    transfer = load_json(TRANSFER_MATRIX_PATH)
    gate = load_json(GATE_GEOMETRY_PATH)
    closeout = load_json(MC005_CLOSEOUT_RESULT_PATH)
    work_order = get_work_order(doctrine)
    atlas_row = get_atlas_row(atlas)
    route_entry = get_by_row_id(route["route_entries"], ROW_ID, "route disposition")
    reliability_entry = get_by_row_id(
        reliability["reliability_entries"], ROW_ID, "reliability"
    )
    transfer_entry = get_by_row_id(transfer["transfer_entries"], ROW_ID, "transfer")
    gate_entry = get_by_row_id(gate["gate_entries"], ROW_ID, "gate geometry")
    source_snapshot = build_source_snapshot(
        atlas_row,
        route_entry,
        reliability_entry,
        transfer_entry,
        gate_entry,
        closeout,
    )
    payload = {
        "schema_version": 1,
        "purpose": (
            "Freeze MC005 as the calibrated bounded internal-causal reference "
            "specimen: exact high-margin lookup mediation exists, but null "
            "locality and transfer keep it below full mechanism-card promotion."
        ),
        "sources": {
            "atlas": rel(ATLAS_PATH),
            "offensive_doctrine": rel(OFFENSIVE_DOCTRINE_PATH),
            "route_disposition": rel(ROUTE_DISPOSITION_PATH),
            "reliability_matrix": rel(RELIABILITY_MATRIX_PATH),
            "transfer_matrix": rel(TRANSFER_MATRIX_PATH),
            "gate_geometry": rel(GATE_GEOMETRY_PATH),
            "closeout_result": rel(MC005_CLOSEOUT_RESULT_PATH),
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
            "same_route_repair_allowed": False,
            "allowed_future_work": [
                "materially new intervention family with preregistered null-locality rationale",
                "matched transfer panel with null reliability as a first-class gate",
                "comparative baseline for a new bridge or mechanism-card family",
            ],
            "forbidden_future_work": work_order["forbidden_moves"],
        },
        "claim_boundary": {
            "allowed_claim": (
                "MC005 is a bounded internal-causal reference specimen for "
                "Qwen3-1.7B associative lookup under the tested Response: "
                "contract: layers 24-26 final-query attention writes exactly "
                "mediate high-margin lookup rows."
            ),
            "forbidden_claim": (
                "MC005 is not a promoted mechanism card, not fully reliable, "
                "not transfer-ready, and not evidence for general factual or "
                "knowledge control."
            ),
            "claim_flags": {
                "promoted_mechanism_claimed": False,
                "full_reliability_claimed": False,
                "transfer_ready_claimed": False,
                "general_knowledge_control_claimed": False,
                "same_route_repair_loop_allowed": False,
            },
        },
        "summary": {
            "row_id": ROW_ID,
            "verdict": source_snapshot["atlas"]["verdict"],
            "route_status": source_snapshot["closeout"]["route_status"],
            "terminal_stage": source_snapshot["gate_geometry"]["terminal_stage"],
            "reliability_class": source_snapshot["reliability"]["reliability_class"],
            "transfer_class": source_snapshot["transfer"]["transfer_class"],
            "lookup_write_effect_exact": source_snapshot["closeout"]["criteria"][
                "lookup_write_effect_exact"
            ],
            "strict_answer_absent_nulls_clean": source_snapshot["closeout"][
                "criteria"
            ]["strict_answer_absent_nulls_clean"],
            "combined_null_flip_count": source_snapshot["closeout"]["v31_boundary"][
                "combined_null_flip_count"
            ],
            "same_route_repair_allowed": False,
        },
    }
    payload["validation_checks"] = build_validation_checks(payload)
    return payload


def validate_mc005_reference_specimen_audit(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("MC005 reference-specimen audit schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"MC005 reference-specimen source missing: {rel_path}")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(f"MC005 reference-specimen checks failed: {failed_checks}")


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    work_order = payload["work_order"]
    closeout = payload["source_snapshot"]["closeout"]
    v31 = closeout["v31_boundary"]
    future = payload["future_work_admission_rule"]
    lines = [
        "# MC005 Reference Specimen Audit",
        "",
        "Date: 2026-07-01",
        "",
        "Status: generated bounded-reference audit; no full promotion.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/mc005_reference_specimen_audit.json`",
        "",
        "Builder:",
        "",
        "> `code/mc005_reference_specimen_audit.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\mc005_reference_specimen_audit.py --write",
        "python code\\mc005_reference_specimen_audit.py",
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
        f"- terminal stage: `{summary['terminal_stage']}`;",
        f"- reliability class: `{summary['reliability_class']}`;",
        f"- transfer class: `{summary['transfer_class']}`;",
        f"- lookup write effect exact: {summary['lookup_write_effect_exact']};",
        f"- strict answer-absent nulls clean: {summary['strict_answer_absent_nulls_clean']};",
        f"- combined null flip count: {summary['combined_null_flip_count']};",
        f"- same-route repair allowed: {summary['same_route_repair_allowed']}.",
        "",
        "## Decisive Metrics",
        "",
        f"- V29 delta recovery versus direct source masking: `{closeout['v29_recovery']['target_write_delta_recovery_vs_direct']}`;",
        f"- V29 target-win-loss recovery versus direct source masking: `{closeout['v29_recovery']['target_write_win_loss_recovery_vs_direct']}`;",
        f"- V31 lookup target write mean delta: `{v31['lookup_target_write_mean_delta']}`;",
        f"- V31 lookup target-win loss: `{v31['lookup_target_win_loss']}`;",
        f"- V31 lookup loss margin bands: `{format_value(v31['lookup_loss_margin_bands'])}`;",
        f"- V31 null flip margin bands: `{format_value(v31['combined_null_flip_margin_bands'])}`;",
        f"- V30 imported flips inside strict 0.5 margin cutoff: `{v31['v30_imported_flips_abs_margin_le_0p5']}`.",
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
        f"- same-route repair allowed: {future['same_route_repair_allowed']};",
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
        "It proves that the atlas has one calibrated bounded internal-causal",
        "reference specimen: a local intervention exactly mediates the primary",
        "lookup behavior while reliability gates still stop full promotion.",
        "",
        "## What It Does Not Prove",
        "",
        "It does not prove full mechanism-card reliability, transfer to other",
        "models, or general control of factual knowledge.",
        "",
    ]
    return "\n".join(lines)


def write_mc005_reference_specimen_audit(
    output_path: Path = MC005_REFERENCE_SPECIMEN_AUDIT_PATH,
    report_path: Path = MC005_REFERENCE_SPECIMEN_AUDIT_REPORT_PATH,
) -> dict[str, Any]:
    payload = build_mc005_reference_specimen_audit()
    validate_mc005_reference_specimen_audit(payload)
    write_json(output_path, payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(payload), encoding="utf-8", newline="\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="write MC005 reference-specimen artifacts",
    )
    parser.add_argument("--json", action="store_true", help="print audit JSON")
    args = parser.parse_args()

    payload = build_mc005_reference_specimen_audit()
    validate_mc005_reference_specimen_audit(payload)

    if args.write:
        write_mc005_reference_specimen_audit()
        print(
            f"wrote {MC005_REFERENCE_SPECIMEN_AUDIT_PATH.relative_to(ROOT).as_posix()} "
            f"and {MC005_REFERENCE_SPECIMEN_AUDIT_REPORT_PATH.relative_to(ROOT).as_posix()} "
            f"for {payload['summary']['row_id']}"
        )
        return
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(
        "MC005 reference specimen audit ok: "
        f"{payload['summary']['verdict']}, "
        f"{payload['summary']['route_status']}, "
        f"null flips={payload['summary']['combined_null_flip_count']}"
    )


if __name__ == "__main__":
    main()
