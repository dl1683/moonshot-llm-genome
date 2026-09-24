"""Build the post-MC033 bridge-substrate closeout audit.

The offensive doctrine says the post-MC033 bridge family should be closed
unless a future experiment changes substrate class. This module turns that
research decision into a generated, validator-backed artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from control_surface_artifacts import ROOT, load_json
from control_surface_bridge_ladder import BRIDGE_LADDER_PATH
from control_surface_error_taxonomy import ERROR_TAXONOMY_PATH
from control_surface_genome_snapshot import GENOME_SNAPSHOT_PATH
from control_surface_offensive_doctrine import OFFENSIVE_DOCTRINE_PATH
from control_surface_smoke_diagnostics import SMOKE_DIAGNOSTICS_PATH


POST_MC033_BRIDGE_CLOSEOUT_PATH = (
    ROOT / "data" / "post_mc033_bridge_closeout_audit.json"
)
POST_MC033_BRIDGE_CLOSEOUT_REPORT_PATH = (
    ROOT / "research" / "cards" / "POST_MC033_BRIDGE_SUBSTRATE_CLOSEOUT_STATUS.md"
)

WORK_ORDER_ID = "close_post_mc033_bridge_substrate_family"
SAME_FAMILY_SEQUENCE = ["MC031", "MC032", "MC033"]
RECENT_CLOSED_RUNG_IDS = ["MC030", "MC031", "MC032", "MC033"]


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


def by_card_id(items: list[dict[str, Any]], card_id: str) -> dict[str, Any]:
    matches = [item for item in items if item.get("card_id") == card_id]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one {card_id} entry, got {len(matches)}")
    return matches[0]


def selected_audit_metrics(card_id: str, error_taxonomy: dict[str, Any]) -> dict[str, Any]:
    if card_id == "MC031":
        audit = error_taxonomy["mc031_checksum_audit"]
        return {
            "synthetic_lookup_local_rate": audit["synthetic_lookup_local_rate"],
            "real_atomic_control_atomic_rate": audit["real_atomic_control_atomic_rate"],
            "answer_absent_unknown_rate": audit["answer_absent_unknown_rate"],
            "valid_checksum_local_rate": audit["valid_checksum_local_rate"],
            "invalid_checksum_local_rate": audit["invalid_checksum_local_rate"],
            "invalid_checksum_atomic_or_lure_rate": audit[
                "invalid_checksum_atomic_or_lure_rate"
            ],
        }
    if card_id == "MC032":
        audit = error_taxonomy["mc032_crosstable_audit"]
        return {
            "synthetic_lookup_local_rate": audit["synthetic_lookup_local_rate"],
            "real_atomic_control_atomic_rate": audit["real_atomic_control_atomic_rate"],
            "answer_absent_unknown_rate": audit["answer_absent_unknown_rate"],
            "match_conflict_local_rate": audit["match_conflict_local_rate"],
            "mismatch_conflict_local_rate": audit["mismatch_conflict_local_rate"],
            "mismatch_conflict_atomic_or_lure_rate": audit[
                "mismatch_conflict_atomic_or_lure_rate"
            ],
            "mismatch_conflict_side_number_rate": audit[
                "mismatch_conflict_side_number_rate"
            ],
        }
    if card_id == "MC033":
        audit = error_taxonomy["mc033_fact_claim_audit"]
        return {
            "synthetic_lookup_local_rate": audit["synthetic_lookup_local_rate"],
            "real_atomic_control_atomic_rate": audit["real_atomic_control_atomic_rate"],
            "answer_absent_unknown_rate": audit["answer_absent_unknown_rate"],
            "match_conflict_local_rate": audit["match_conflict_local_rate"],
            "match_conflict_atomic_rate": audit["match_conflict_atomic_rate"],
            "mismatch_conflict_atomic_rate": audit["mismatch_conflict_atomic_rate"],
            "mismatch_conflict_local_rate": audit["mismatch_conflict_local_rate"],
            "mismatch_conflict_lure_rate": audit["mismatch_conflict_lure_rate"],
        }
    raise AssertionError(f"unexpected same-family card {card_id}")


def audit_interpretation(card_id: str, error_taxonomy: dict[str, Any]) -> str:
    key_by_id = {
        "MC031": "mc031_checksum_audit",
        "MC032": "mc032_crosstable_audit",
        "MC033": "mc033_fact_claim_audit",
    }
    return error_taxonomy[key_by_id[card_id]]["interpretation"]


def build_closure_sequence(
    bridge_ladder: dict[str, Any],
    smoke_diagnostics: dict[str, Any],
    error_taxonomy: dict[str, Any],
) -> list[dict[str, Any]]:
    sequence = []
    for card_id in SAME_FAMILY_SEQUENCE:
        rung = by_card_id(bridge_ladder["rungs"], card_id)
        smoke = by_card_id(smoke_diagnostics["cards"], card_id)
        sequence.append(
            {
                "card_id": card_id,
                "title": rung["title"],
                "repair_attempt": rung["repair_attempt"],
                "dominant_failure": rung["dominant_failure"],
                "behavior_outcome": rung["behavior_outcome"],
                "local_vs_learned_mixture": rung["local_vs_learned_mixture"],
                "hidden_state_allowed": rung["hidden_state_allowed"],
                "behavior_ready": rung["behavior_ready"],
                "signature_ready": rung["signature_ready"],
                "diagnostic_class": smoke["diagnostic_class"],
                "exported_diagnostics": smoke["exported_diagnostics"],
                "failure_axes": smoke["failure_axes"],
                "metrics": selected_audit_metrics(card_id, error_taxonomy),
                "interpretation": audit_interpretation(card_id, error_taxonomy),
                "evidence_path": rung["evidence_path"],
                "status_card_path": rung["status_card_path"],
            }
        )
    return sequence


def build_validation_checks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    bridge = payload["source_snapshot"]["bridge_ladder_summary"]
    smoke = payload["source_snapshot"]["smoke_diagnostics_summary"]
    genome_bridge = payload["source_snapshot"]["genome_bridge_shape"]
    sequence = payload["closure_sequence"]
    by_id = {entry["card_id"]: entry for entry in sequence}
    mc031 = by_id["MC031"]["metrics"]
    mc032 = by_id["MC032"]["metrics"]
    mc033 = by_id["MC033"]["metrics"]
    direct_controls_clean = all(
        entry["metrics"]["synthetic_lookup_local_rate"] == 1.0
        and entry["metrics"]["real_atomic_control_atomic_rate"] == 1.0
        and entry["metrics"]["answer_absent_unknown_rate"] == 1.0
        for entry in sequence
    )
    hidden_state_flags = [entry["hidden_state_allowed"] for entry in sequence]
    readiness_flags = [
        [
            entry["card_id"],
            entry["behavior_ready"],
            entry["signature_ready"],
            entry["hidden_state_allowed"],
        ]
        for entry in sequence
    ]
    checks = [
        {
            "id": "work_order_contract_matches_post_mc033_closeout",
            "predicate": f"== {WORK_ORDER_ID}",
            "actual": payload["work_order"]["id"],
            "passed": payload["work_order"]["id"] == WORK_ORDER_ID,
            "why": "The closeout must come from the offensive-doctrine bridge-substrate work order.",
        },
        {
            "id": "same_family_sequence_recorded",
            "predicate": f"== {SAME_FAMILY_SEQUENCE}",
            "actual": [entry["card_id"] for entry in sequence],
            "passed": [entry["card_id"] for entry in sequence] == SAME_FAMILY_SEQUENCE,
            "why": "The closeout must record MC031-MC033 as the same-family source-validity sequence.",
        },
        {
            "id": "recent_closed_bridge_rungs_preserved",
            "predicate": f"== {RECENT_CLOSED_RUNG_IDS}",
            "actual": genome_bridge["recent_closed_rung_ids"],
            "passed": genome_bridge["recent_closed_rung_ids"] == RECENT_CLOSED_RUNG_IDS,
            "why": "The generated snapshot must still treat MC030-MC033 as the recent bridge closure set.",
        },
        {
            "id": "bridge_ladder_scope_preserved",
            "predicate": "24 rungs, 17 smoke rungs",
            "actual": {
                "rung_count": bridge["rung_count"],
                "smoke_rung_count": bridge["smoke_rung_count"],
            },
            "passed": bridge["rung_count"] == 24 and bridge["smoke_rung_count"] == 17,
            "why": "The closeout is scoped to the current MC010-MC033 bridge ladder.",
        },
        {
            "id": "no_bridge_hidden_state_or_clean_unconfounded_candidate",
            "predicate": "hidden_state_allowed == 0 and clean_unconfounded == 0",
            "actual": {
                "hidden_state_allowed_count": bridge["hidden_state_allowed_count"],
                "clean_unconfounded_bridge_count": bridge[
                    "clean_unconfounded_bridge_count"
                ],
            },
            "passed": bridge["hidden_state_allowed_count"] == 0
            and bridge["clean_unconfounded_bridge_count"] == 0,
            "why": "A bridge-family closeout cannot license hidden-state work by accident.",
        },
        {
            "id": "smoke_layer_stays_pre_hidden_state",
            "predicate": "hidden_state_allowed_count == 0",
            "actual": smoke["hidden_state_allowed_count"],
            "passed": smoke["hidden_state_allowed_count"] == 0,
            "why": "Smoke diagnostics remain behavior-substrate evidence, not mechanism evidence.",
        },
        {
            "id": "mc031_statusless_checksum_invalid_branch_collapses",
            "predicate": "invalid local == 1.0 and invalid atomic/lure == 0.0",
            "actual": {
                "invalid_checksum_local_rate": mc031["invalid_checksum_local_rate"],
                "invalid_checksum_atomic_or_lure_rate": mc031[
                    "invalid_checksum_atomic_or_lure_rate"
                ],
            },
            "passed": mc031["invalid_checksum_local_rate"] == 1.0
            and mc031["invalid_checksum_atomic_or_lure_rate"] == 0.0,
            "why": "MC031 fails at source-validity arbitration despite clean direct controls and nulls.",
        },
        {
            "id": "mc032_crosstable_mismatch_not_learned_branch",
            "predicate": "mismatch atomic/lure == 0.0, local >= 0.7, side == 0.0",
            "actual": {
                "mismatch_conflict_atomic_or_lure_rate": mc032[
                    "mismatch_conflict_atomic_or_lure_rate"
                ],
                "mismatch_conflict_local_rate": mc032["mismatch_conflict_local_rate"],
                "mismatch_conflict_side_number_rate": mc032[
                    "mismatch_conflict_side_number_rate"
                ],
            },
            "passed": mc032["mismatch_conflict_atomic_or_lure_rate"] == 0.0
            and mc032["mismatch_conflict_local_rate"] >= 0.7
            and mc032["mismatch_conflict_side_number_rate"] == 0.0,
            "why": "MC032 removes the checksum-specific explanation but still fails learned routing.",
        },
        {
            "id": "mc033_fact_claim_route_fails_both_branches",
            "predicate": "match local < 0.85, mismatch atomic < 0.85, lure >= 0.4",
            "actual": {
                "match_conflict_local_rate": mc033["match_conflict_local_rate"],
                "mismatch_conflict_atomic_rate": mc033[
                    "mismatch_conflict_atomic_rate"
                ],
                "mismatch_conflict_lure_rate": mc033["mismatch_conflict_lure_rate"],
            },
            "passed": mc033["match_conflict_local_rate"] < 0.85
            and mc033["mismatch_conflict_atomic_rate"] < 0.85
            and mc033["mismatch_conflict_lure_rate"] >= 0.4,
            "why": "MC033 does not repair routing by making the cue a learned-fact comparison.",
        },
        {
            "id": "direct_controls_and_nulls_survive_same_family_smokes",
            "predicate": "all selected direct controls and answer-absent nulls == 1.0",
            "actual": {
                entry["card_id"]: {
                    "synthetic_lookup_local_rate": entry["metrics"][
                        "synthetic_lookup_local_rate"
                    ],
                    "real_atomic_control_atomic_rate": entry["metrics"][
                        "real_atomic_control_atomic_rate"
                    ],
                    "answer_absent_unknown_rate": entry["metrics"][
                        "answer_absent_unknown_rate"
                    ],
                }
                for entry in sequence
            },
            "passed": direct_controls_clean,
            "why": "The family boundary is not just bad controls or broken null rows.",
        },
        {
            "id": "same_family_rows_forbid_hidden_state_work",
            "predicate": "all hidden_state_allowed == false",
            "actual": hidden_state_flags,
            "passed": not any(hidden_state_flags),
            "why": "MC031-MC033 remain pre-signature bridge diagnostics.",
        },
        {
            "id": "same_family_rows_not_behavior_or_signature_ready",
            "predicate": "all behavior_ready/signature_ready/hidden_state_allowed == false",
            "actual": readiness_flags,
            "passed": all(
                not entry["behavior_ready"]
                and not entry["signature_ready"]
                and not entry["hidden_state_allowed"]
                for entry in sequence
            ),
            "why": "The closeout must not promote failed behavior substrates.",
        },
        {
            "id": "claim_boundary_forbids_mechanism_claim",
            "predicate": "all mechanism claim flags false",
            "actual": payload["claim_boundary"]["claim_flags"],
            "passed": not any(payload["claim_boundary"]["claim_flags"].values()),
            "why": "The artifact is a route closeout, not a mechanism card.",
        },
        {
            "id": "future_bridge_requires_material_substrate_change",
            "predicate": "material_substrate_change_required == true",
            "actual": payload["future_bridge_admission_rule"][
                "material_substrate_change_required"
            ],
            "passed": payload["future_bridge_admission_rule"][
                "material_substrate_change_required"
            ]
            is True,
            "why": "Same-family source-validity wording repairs are killed after MC033.",
        },
    ]
    return checks


def build_post_mc033_bridge_closeout_audit() -> dict[str, Any]:
    doctrine = load_json(OFFENSIVE_DOCTRINE_PATH)
    genome_snapshot = load_json(GENOME_SNAPSHOT_PATH)
    bridge_ladder = load_json(BRIDGE_LADDER_PATH)
    smoke_diagnostics = load_json(SMOKE_DIAGNOSTICS_PATH)
    error_taxonomy = load_json(ERROR_TAXONOMY_PATH)
    work_order = get_work_order(doctrine)
    closure_sequence = build_closure_sequence(
        bridge_ladder, smoke_diagnostics, error_taxonomy
    )
    payload = {
        "schema_version": 1,
        "purpose": (
            "Close the same-family post-MC033 bridge-substrate route as a "
            "diagnostic family unless a future experiment changes substrate "
            "class and clears the full bridge admission packet."
        ),
        "sources": {
            "offensive_doctrine": rel(OFFENSIVE_DOCTRINE_PATH),
            "genome_snapshot": rel(GENOME_SNAPSHOT_PATH),
            "bridge_ladder": rel(BRIDGE_LADDER_PATH),
            "smoke_diagnostics": rel(SMOKE_DIAGNOSTICS_PATH),
            "error_taxonomy": rel(ERROR_TAXONOMY_PATH),
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
        "source_snapshot": {
            "bridge_ladder_summary": {
                key: bridge_ladder["summary"][key]
                for key in [
                    "rung_count",
                    "smoke_rung_count",
                    "behavior_ready_count",
                    "signature_ready_count",
                    "hidden_state_allowed_count",
                    "clean_unconfounded_bridge_count",
                ]
            },
            "smoke_diagnostics_summary": {
                key: smoke_diagnostics["summary"][key]
                for key in [
                    "card_count",
                    "behavior_ready_count",
                    "signature_ready_count",
                    "hidden_state_allowed_count",
                ]
            },
            "genome_bridge_shape": genome_snapshot["bridge_shape"],
            "error_taxonomy_summary": {
                key: error_taxonomy["summary"][key]
                for key in [
                    "smoke_card_count",
                    "bridge_rung_count",
                    "hidden_state_allowed_count",
                    "bridge_closed_before_hidden_state_count",
                ]
            },
        },
        "recent_closed_rung_ids": genome_snapshot["bridge_shape"][
            "recent_closed_rung_ids"
        ],
        "same_family_sequence_ids": SAME_FAMILY_SEQUENCE,
        "closure_sequence": closure_sequence,
        "future_bridge_admission_rule": {
            "material_substrate_change_required": True,
            "same_family_route_status": "killed_after_mc033",
            "forbidden_same_family_moves": work_order["forbidden_moves"],
            "minimum_evidence_packet": work_order["minimum_evidence_packet"],
            "hidden_state_license_rule": work_order["promotion_rule"],
            "allowed_output_classes": work_order["allowed_outputs"],
        },
        "claim_boundary": {
            "allowed_claim": (
                "MC031-MC033 form a same-family statusless source-validity "
                "closure sequence: direct controls and answer-absent nulls "
                "can remain clean while learned/local bridge routing still "
                "fails under checksum, cross-table, and fact-claim cues."
            ),
            "forbidden_claim": (
                "This closeout does not claim a hidden signature, causal "
                "intervention, mechanism card, or general knowledge-control "
                "surface."
            ),
            "claim_flags": {
                "behavior_substrate_promoted": False,
                "hidden_signature_claimed": False,
                "intervention_claimed": False,
                "mechanism_claimed": False,
                "general_knowledge_control_claimed": False,
            },
        },
        "summary": {
            "same_family_sequence_count": len(closure_sequence),
            "same_family_sequence_ids": SAME_FAMILY_SEQUENCE,
            "recent_closed_rung_ids": genome_snapshot["bridge_shape"][
                "recent_closed_rung_ids"
            ],
            "bridge_rung_count": bridge_ladder["summary"]["rung_count"],
            "smoke_rung_count": bridge_ladder["summary"]["smoke_rung_count"],
            "hidden_state_allowed_count": bridge_ladder["summary"][
                "hidden_state_allowed_count"
            ],
            "clean_unconfounded_bridge_count": bridge_ladder["summary"][
                "clean_unconfounded_bridge_count"
            ],
            "same_family_route_status": "killed_after_mc033",
        },
    }
    payload["validation_checks"] = build_validation_checks(payload)
    return payload


def validate_post_mc033_bridge_closeout_audit(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("post-MC033 bridge closeout schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"post-MC033 bridge closeout source missing: {rel_path}")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(f"post-MC033 bridge closeout checks failed: {failed_checks}")


def render_markdown(payload: dict[str, Any]) -> str:
    work_order = payload["work_order"]
    bridge = payload["source_snapshot"]["bridge_ladder_summary"]
    rule = payload["future_bridge_admission_rule"]
    lines = [
        "# Post-MC033 Bridge Substrate Closeout Status",
        "",
        "Date: 2026-07-01",
        "",
        "Status: generated route closeout; no hidden-state license.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/post_mc033_bridge_closeout_audit.json`",
        "",
        "Builder:",
        "",
        "> `code/post_mc033_bridge_closeout_audit.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\post_mc033_bridge_closeout_audit.py --write",
        "python code\\post_mc033_bridge_closeout_audit.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Purpose",
        "",
        payload["purpose"],
        "",
        "## Generated Facts",
        "",
        f"- bridge rungs: {bridge['rung_count']};",
        f"- smoke rungs: {bridge['smoke_rung_count']};",
        f"- hidden-state-allowed bridge rungs: {bridge['hidden_state_allowed_count']};",
        f"- clean unconfounded bridge candidates: {bridge['clean_unconfounded_bridge_count']};",
        f"- recent closed rung ids: `{format_value(payload['recent_closed_rung_ids'])}`;",
        f"- same-family closure sequence: `{format_value(payload['same_family_sequence_ids'])}`.",
        "",
        "## Closure Sequence",
        "",
        "| Card | Repair Attempt | Decisive Failure | Key Metrics |",
        "| --- | --- | --- | --- |",
    ]
    for entry in payload["closure_sequence"]:
        metrics = entry["metrics"]
        key_metrics = ", ".join(
            f"{key}={value}" for key, value in sorted(metrics.items())
        )
        lines.append(
            f"| `{entry['card_id']}` | {entry['repair_attempt']} | "
            f"{entry['dominant_failure']} | `{key_metrics}` |"
        )

    lines.extend(
        [
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
            "## Required Future Bridge Class",
            "",
            "A future bridge is admitted only if it is materially outside the",
            "same-family source-validity repairs already killed here.",
            "",
            f"- route status: `{rule['same_family_route_status']}`;",
            f"- material substrate change required: {rule['material_substrate_change_required']};",
            f"- forbidden same-family moves: `{format_value(rule['forbidden_same_family_moves'])}`;",
            f"- minimum evidence packet: `{format_value(rule['minimum_evidence_packet'])}`;",
            f"- hidden-state license rule: {rule['hidden_state_license_rule']}",
            "",
            "## Claim Boundary",
            "",
            payload["claim_boundary"]["allowed_claim"],
            "",
            payload["claim_boundary"]["forbidden_claim"],
            "",
            "## What This Proves",
            "",
            "It proves that the post-MC033 bridge route has a machine-checked",
            "death condition. The same family can preserve direct controls and",
            "nulls while still failing the learned/local routing behavior the",
            "bridge needs.",
            "",
            "## What It Does Not Prove",
            "",
            "It does not prove a hidden signature, a causal intervention, a",
            "mechanism card, or a general knowledge-control surface. It is a",
            "bounded diagnostic closeout and an admission rule for future work.",
            "",
        ]
    )
    return "\n".join(lines)


def write_post_mc033_bridge_closeout_audit(
    output_path: Path = POST_MC033_BRIDGE_CLOSEOUT_PATH,
    report_path: Path = POST_MC033_BRIDGE_CLOSEOUT_REPORT_PATH,
) -> dict[str, Any]:
    payload = build_post_mc033_bridge_closeout_audit()
    validate_post_mc033_bridge_closeout_audit(payload)
    write_json(output_path, payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(payload), encoding="utf-8", newline="\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="write post-MC033 bridge closeout artifacts",
    )
    parser.add_argument("--json", action="store_true", help="print closeout JSON")
    args = parser.parse_args()

    payload = build_post_mc033_bridge_closeout_audit()
    validate_post_mc033_bridge_closeout_audit(payload)

    if args.write:
        write_post_mc033_bridge_closeout_audit()
        print(
            f"wrote {POST_MC033_BRIDGE_CLOSEOUT_PATH.relative_to(ROOT).as_posix()} "
            f"and {POST_MC033_BRIDGE_CLOSEOUT_REPORT_PATH.relative_to(ROOT).as_posix()} "
            f"with {payload['summary']['same_family_sequence_count']} same-family cards"
        )
        return
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(
        "post-MC033 bridge closeout ok: "
        f"{payload['summary']['same_family_sequence_count']} same-family cards, "
        f"{payload['summary']['hidden_state_allowed_count']} hidden-state-allowed rungs"
    )
    print(
        "recent_closed_rung_ids:",
        json.dumps(payload["summary"]["recent_closed_rung_ids"], sort_keys=True),
    )


if __name__ == "__main__":
    main()
