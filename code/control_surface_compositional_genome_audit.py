"""Build the compositional genome audit.

This layer promotes the current reviewer insight into a validator-backed
artifact: the "genome" is not a single mechanism card. It is the measured
distribution of where behavior lives across prompt contract, output geometry,
source/token dependence, internal monitors, causal intervention, reliability,
and transfer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from control_surface_artifacts import ATLAS_PATH, ROOT, load_json
from control_surface_decision_frontier import DECISION_FRONTIER_PATH
from control_surface_mixture_law import MIXTURE_LAW_PATH
from control_surface_reliability_matrix import RELIABILITY_MATRIX_PATH
from control_surface_transfer_matrix import TRANSFER_MATRIX_PATH
from mc005_reference_specimen_audit import MC005_REFERENCE_SPECIMEN_AUDIT_PATH
from mc006_predecision_frontier_audit import MC006_PREDECISION_FRONTIER_AUDIT_PATH
from post_mc033_bridge_closeout_audit import POST_MC033_BRIDGE_CLOSEOUT_PATH


COMPOSITIONAL_GENOME_AUDIT_PATH = (
    ROOT / "data" / "control_surface_compositional_genome_audit.json"
)
COMPOSITIONAL_GENOME_AUDIT_REPORT_PATH = (
    ROOT / "research" / "40_CONTROL_SURFACE_COMPOSITIONAL_GENOME_AUDIT.md"
)

MC005_ROW_ID = "mc005_associative_lookup"
MC006_ROW_ID = "mc006_parametric_fact_override"
MONITOR_ONLY_ROWS = ["mc004_in_context_binding", MC006_ROW_ID]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")


def ratio(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(count / total, 6)


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def axis_entry(
    axis_id: str,
    count: int,
    denominator: int,
    denominator_label: str,
    interpretation: str,
) -> dict[str, Any]:
    return {
        "axis_id": axis_id,
        "count": count,
        "denominator": denominator,
        "denominator_label": denominator_label,
        "ratio": ratio(count, denominator),
        "interpretation": interpretation,
    }


def get_atlas_row(atlas: dict[str, Any], row_id: str) -> dict[str, Any]:
    matches = [row for row in atlas["rows"] if row["id"] == row_id]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one atlas row {row_id}, got {len(matches)}")
    return matches[0]


def build_distribution_axes(
    mixture: dict[str, Any],
    frontier: dict[str, Any],
    reliability: dict[str, Any],
    transfer: dict[str, Any],
    post_mc033: dict[str, Any],
) -> list[dict[str, Any]]:
    row_count = mixture["summary"]["row_count"]
    bridge_count = post_mc033["summary"]["bridge_rung_count"]
    pressure = mixture["summary"]["pressure_class_counts"]
    frontier_counts = frontier["summary"]["frontier_class_counts"]
    transfer_values = transfer["summary"]["transfer_value_counts"]
    return [
        axis_entry(
            "prompt_contract_visible",
            pressure.get("prompt_contract_visible", 0),
            row_count,
            "atlas_rows",
            "Prompt wording, authority, format, or contract remains visible in every current atlas row.",
        ),
        axis_entry(
            "output_geometry_visible",
            pressure.get("output_geometry_visible", 0),
            row_count,
            "atlas_rows",
            "Most rows expose the behavior through answer/candidate/output geometry before a clean internal mechanism claim can be made.",
        ),
        axis_entry(
            "source_or_prompt_token_dependent",
            pressure.get("source_or_prompt_token_dependent", 0),
            row_count,
            "atlas_rows",
            "Source tokens, local prompt values, or prompt-side routing pressure explain a majority of current behavior surfaces.",
        ),
        axis_entry(
            "behavior_or_bridge_substrate_blocked",
            pressure.get("behavior_or_bridge_substrate_blocked", 0),
            row_count,
            "atlas_rows",
            "The largest primary failure mode is still getting a behavior substrate that justifies hidden-state work.",
        ),
        axis_entry(
            "internal_monitor_present",
            pressure.get("internal_monitor_present", 0),
            row_count,
            "atlas_rows",
            "Internal monitors exist, but most are output shadows, non-causal signatures, or monitor-only rows.",
        ),
        axis_entry(
            "predecision_monitor_no_lever",
            frontier_counts.get("predecision_monitor_no_lever", 0),
            row_count,
            "atlas_rows",
            "Only MC004 and MC006 currently show predecision monitor evidence, and neither supplies a reliable lever.",
        ),
        axis_entry(
            "internal_causal_surface",
            pressure.get("internal_causal_surface", 0),
            row_count,
            "atlas_rows",
            "Only MC005 is currently counted as a bounded internal-causal surface.",
        ),
        axis_entry(
            "bounded_reliability_reference",
            reliability["summary"]["bounded_reliability_count"],
            row_count,
            "atlas_rows",
            "The reliability matrix has one bounded reference and no full-reliability mechanism.",
        ),
        axis_entry(
            "full_reliability_mechanism",
            reliability["summary"]["full_reliability_count"],
            row_count,
            "atlas_rows",
            "No row currently passes all behavior, signature, intervention, null/locality, robustness, and transfer gates.",
        ),
        axis_entry(
            "transfer_ready_mechanism",
            transfer["summary"]["transfer_ready_mechanism_count"],
            row_count,
            "atlas_rows",
            "No mechanism is currently transfer-ready.",
        ),
        axis_entry(
            "transfer_untested_or_missing",
            transfer_values.get("untested", 0),
            row_count,
            "atlas_rows",
            "Transfer remains mostly untested or unavailable, so generality claims are still barred.",
        ),
        axis_entry(
            "clean_unconfounded_bridge_substrate",
            post_mc033["summary"]["clean_unconfounded_bridge_count"],
            bridge_count,
            "bridge_rungs",
            "The bridge ladder has not produced a clean unconfounded knowledge-like substrate for hidden-state work.",
        ),
    ]


def build_anchor_findings(
    atlas: dict[str, Any],
    mixture: dict[str, Any],
    frontier: dict[str, Any],
    mc005: dict[str, Any],
    mc006: dict[str, Any],
    post_mc033: dict[str, Any],
) -> list[dict[str, Any]]:
    mc005_row = get_atlas_row(atlas, MC005_ROW_ID)
    mc006_row = get_atlas_row(atlas, MC006_ROW_ID)
    monitor_rows = frontier["summary"]["monitor_only_rows"]
    return [
        {
            "anchor_id": "mc005_reference_specimen",
            "row_id": MC005_ROW_ID,
            "role": "bounded_internal_causal_reference",
            "current_status": mc005["summary"],
            "mixture_memberships": [
                "internal_causal_surface",
                "source_or_prompt_token_dependent",
                "null_boundary_or_locality_limited",
            ],
            "insight": (
                "MC005 shows that a narrow internal causal surface can be real "
                "while still failing full reliability through null-locality and "
                "model-size transfer boundaries."
            ),
            "allowed_claims": mc005_row["allowed_claims"],
            "forbidden_claims": mc005_row["forbidden_claims"],
        },
        {
            "anchor_id": "mc006_predecision_monitor",
            "row_id": MC006_ROW_ID,
            "role": "knowledge_like_monitor_without_lever",
            "current_status": mc006["summary"],
            "mixture_memberships": [
                "output_geometry_visible",
                "internal_monitor_present",
                "predecision_monitor_no_lever",
                "transfer_unproven_or_failed",
            ],
            "insight": (
                "MC006 shows why lead-time must be measured separately from "
                "causality: a matched generated behavior substrate and early "
                "monitor can coexist with failed steering, output/candidate "
                "visibility, shuffled-label failure, and transfer-bank closure."
            ),
            "allowed_claims": mc006_row["allowed_claims"],
            "forbidden_claims": mc006_row["forbidden_claims"],
        },
        {
            "anchor_id": "post_mc033_bridge_closeout",
            "row_id": "bridge_ladder_mc031_mc033",
            "role": "bridge_substrate_death_condition",
            "current_status": post_mc033["summary"],
            "mixture_memberships": [
                "behavior_or_bridge_substrate_blocked",
                "source_or_prompt_token_dependent",
                "clean_unconfounded_bridge_substrate_absent",
            ],
            "insight": (
                "The bridge sequence shows that clean direct controls and nulls "
                "can coexist with failed learned/local arbitration, so direct "
                "control success cannot license hidden-state work."
            ),
            "allowed_claims": [post_mc033["claim_boundary"]["allowed_claim"]],
            "forbidden_claims": [post_mc033["claim_boundary"]["forbidden_claim"]],
        },
        {
            "anchor_id": "monitor_only_frontier",
            "row_id": "mc004_and_mc006",
            "role": "lead_time_boundary",
            "current_status": {
                "monitor_only_rows": monitor_rows,
                "predecision_causal_candidate_count": frontier["summary"][
                    "predecision_causal_candidate_count"
                ],
            },
            "mixture_memberships": ["predecision_monitor_no_lever"],
            "insight": (
                "The current lead-time frontier has monitors, not levers: MC004 "
                "and MC006 are informative timing rows, but no atlas row is a "
                "predecision causal candidate."
            ),
            "allowed_claims": [
                "Lead-time is a first-class measured axis in the current atlas.",
                "Monitor-only rows can be useful evidence without being mechanism cards.",
            ],
            "forbidden_claims": [
                "Any current predecision monitor is a promoted causal control surface.",
            ],
        },
    ]


def build_insight_claims(
    mixture: dict[str, Any],
    frontier: dict[str, Any],
    reliability: dict[str, Any],
    transfer: dict[str, Any],
) -> list[dict[str, Any]]:
    pressure = mixture["summary"]["pressure_class_counts"]
    row_count = mixture["summary"]["row_count"]
    return [
        {
            "claim_id": "mixture_is_current_genome_object",
            "status": "supported_current_map",
            "evidence": {
                "row_count": row_count,
                "prompt_contract_visible": pressure.get("prompt_contract_visible", 0),
                "output_geometry_visible": pressure.get("output_geometry_visible", 0),
                "source_or_prompt_token_dependent": pressure.get(
                    "source_or_prompt_token_dependent", 0
                ),
                "internal_causal_surface": pressure.get("internal_causal_surface", 0),
            },
            "interpretation": (
                "The central present result is the distribution of behavior "
                "across visible, source-dependent, monitor-only, and bounded "
                "causal surfaces."
            ),
        },
        {
            "claim_id": "typed_failures_are_primary_data",
            "status": "supported_current_map",
            "evidence": {
                "primary_blocker_counts": mixture["summary"]["primary_blocker_counts"],
                "frontier_class_counts": frontier["summary"]["frontier_class_counts"],
                "reliability_class_counts": reliability["summary"][
                    "reliability_class_counts"
                ],
            },
            "interpretation": (
                "Output shadows, prompt-visible positives, bridge blocks, null "
                "boundaries, and transfer failures are not just dead ends; they "
                "are reusable diagnostic classes."
            ),
        },
        {
            "claim_id": "lead_time_is_boundary_not_promotion",
            "status": "supported_current_map",
            "evidence": {
                "monitor_only_rows": frontier["summary"]["monitor_only_rows"],
                "predecision_causal_candidate_count": frontier["summary"][
                    "predecision_causal_candidate_count"
                ],
            },
            "interpretation": (
                "A signal can appear before the local output readout and still "
                "fail as a causal control surface once global output, shuffle, "
                "intervention, and transfer controls are applied."
            ),
        },
        {
            "claim_id": "reliability_and_transfer_are_current_bottlenecks",
            "status": "supported_current_map",
            "evidence": {
                "full_reliability_count": reliability["summary"][
                    "full_reliability_count"
                ],
                "bounded_reliability_count": reliability["summary"][
                    "bounded_reliability_count"
                ],
                "transfer_ready_mechanism_count": transfer["summary"][
                    "transfer_ready_mechanism_count"
                ],
                "transfer_value_counts": transfer["summary"]["transfer_value_counts"],
            },
            "interpretation": (
                "The project can discuss control-surface structure, but broad "
                "mechanism and generality claims remain barred."
            ),
        },
    ]


def build_claim_boundary(
    mixture: dict[str, Any],
    reliability: dict[str, Any],
    transfer: dict[str, Any],
) -> dict[str, Any]:
    return {
        "claim_flags": {
            "mixture_distribution_measured": True,
            "typed_failure_taxonomy_supported": True,
            "lead_time_axis_first_class": True,
            "bounded_internal_reference_exists": mixture["summary"][
                "bounded_internal_causal_count"
            ]
            == 1,
            "promoted_mechanism_exists": mixture["summary"][
                "promoted_mechanism_count"
            ]
            > 0,
            "full_reliability_mechanism_exists": reliability["summary"][
                "full_reliability_count"
            ]
            > 0,
            "transfer_ready_mechanism_exists": transfer["summary"][
                "transfer_ready_mechanism_count"
            ]
            > 0,
            "general_truth_vector_found": False,
            "general_knowledge_control_surface_found": False,
        },
        "allowed_claim": (
            "The project has a validator-backed compositional map of the "
            "current small-model control-surface mixture: prompt-contract, "
            "output-geometry, source/token, monitor-only, bounded-causal, "
            "null-locality, and transfer axes explain why rows pass, fail, or "
            "remain bounded."
        ),
        "forbidden_claim": (
            "This audit does not promote any mechanism card, does not find a "
            "general truth vector, does not establish a general knowledge "
            "control surface, and does not license hidden-state work on bridge "
            "routes closed before behavior substrate readiness."
        ),
    }


def build_validation_checks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    summary = payload["summary"]
    axes = {entry["axis_id"]: entry for entry in payload["distribution_axes"]}
    claim_flags = payload["claim_boundary"]["claim_flags"]
    source = payload["source_snapshot"]
    primary = source["mixture"]["primary_blocker_counts"]
    checks = [
        {
            "id": "source_row_counts_align",
            "predicate": "all atlas-layer row counts == 19",
            "actual": {
                "atlas": summary["row_count"],
                "mixture": source["mixture"]["row_count"],
                "frontier": source["frontier"]["row_count"],
                "reliability": source["reliability"]["row_count"],
                "transfer": source["transfer"]["row_count"],
            },
            "passed": summary["row_count"] == 19
            and source["mixture"]["row_count"] == 19
            and source["frontier"]["row_count"] == 19
            and source["reliability"]["row_count"] == 19
            and source["transfer"]["row_count"] == 19,
            "why": "The audit must summarize the same 19-row atlas as every generated layer.",
        },
        {
            "id": "prompt_contract_pressure_is_universal",
            "predicate": "prompt_contract_visible == row_count",
            "actual": axes["prompt_contract_visible"],
            "passed": axes["prompt_contract_visible"]["count"] == summary["row_count"],
            "why": "The current map says every behavior family still exposes prompt-contract pressure.",
        },
        {
            "id": "visible_surfaces_dominate_internal_causal_surface",
            "predicate": "output and source counts both exceed internal causal count",
            "actual": {
                "output_geometry_visible": axes["output_geometry_visible"]["count"],
                "source_or_prompt_token_dependent": axes[
                    "source_or_prompt_token_dependent"
                ]["count"],
                "internal_causal_surface": axes["internal_causal_surface"]["count"],
            },
            "passed": axes["output_geometry_visible"]["count"]
            > axes["internal_causal_surface"]["count"]
            and axes["source_or_prompt_token_dependent"]["count"]
            > axes["internal_causal_surface"]["count"],
            "why": "The current genome object is dominated by visible and source-dependent surfaces.",
        },
        {
            "id": "no_promoted_full_or_transfer_ready_mechanism",
            "predicate": "promoted == 0, full reliability == 0, transfer ready == 0",
            "actual": {
                "promoted": source["mixture"]["promoted_mechanism_count"],
                "full_reliability": axes["full_reliability_mechanism"]["count"],
                "transfer_ready": axes["transfer_ready_mechanism"]["count"],
            },
            "passed": source["mixture"]["promoted_mechanism_count"] == 0
            and axes["full_reliability_mechanism"]["count"] == 0
            and axes["transfer_ready_mechanism"]["count"] == 0,
            "why": "The compositional audit must not quietly upgrade the state of the project.",
        },
        {
            "id": "mc005_is_only_bounded_internal_reference",
            "predicate": "MC005 is sole bounded internal causal and bounded reliability reference",
            "actual": {
                "internal_causal_rows": source["mixture_rows"][
                    "internal_causal_surface"
                ],
                "mc005_summary": source["mc005"],
                "bounded_reliability_count": axes["bounded_reliability_reference"][
                    "count"
                ],
            },
            "passed": source["mixture_rows"]["internal_causal_surface"] == [MC005_ROW_ID]
            and source["mc005"]["route_status"] == "bounded_frozen_not_promoted"
            and source["mc005"]["verdict"] == "bounded_mechanism_card"
            and axes["bounded_reliability_reference"]["count"] == 1,
            "why": "MC005 anchors the positive side of the map without becoming a promoted card.",
        },
        {
            "id": "mc006_is_monitor_only_closed",
            "predicate": "MC006 monitor supported, promotion false, geometry blocks promotion",
            "actual": source["mc006"],
            "passed": source["mc006"]["predecision_monitor_supported"] is True
            and source["mc006"]["promotion_gate_passed"] is False
            and source["mc006"]["route_status"] == "monitor_only_closed"
            and source["mc006"]["final_or_candidate_geometry_blocks_promotion"] is True,
            "why": "MC006 is evidence for the lead-time boundary, not a causal knowledge lever.",
        },
        {
            "id": "monitor_only_rows_are_not_causal_candidates",
            "predicate": "monitor rows == MC004/MC006 and predecision causal candidates == 0",
            "actual": {
                "monitor_only_rows": source["frontier"]["monitor_only_rows"],
                "predecision_causal_candidate_count": source["frontier"][
                    "predecision_causal_candidate_count"
                ],
            },
            "passed": source["frontier"]["monitor_only_rows"] == MONITOR_ONLY_ROWS
            and source["frontier"]["predecision_causal_candidate_count"] == 0,
            "why": "Lead-time is measured, but no current row has become a predecision causal candidate.",
        },
        {
            "id": "bridge_closeout_keeps_hidden_state_disallowed",
            "predicate": "post-MC033 hidden_state_allowed == 0 and clean_unconfounded == 0",
            "actual": source["post_mc033"],
            "passed": source["post_mc033"]["hidden_state_allowed_count"] == 0
            and source["post_mc033"]["clean_unconfounded_bridge_count"] == 0,
            "why": "The bridge closeout cannot be used to justify hidden-state work.",
        },
        {
            "id": "primary_blockers_partition_atlas",
            "predicate": "primary blocker counts sum to row_count",
            "actual": primary,
            "passed": sum(primary.values()) == summary["row_count"],
            "why": "Typed failures and bounded successes must form a row-level partition.",
        },
        {
            "id": "claim_boundary_forbids_general_vector_language",
            "predicate": "all broad mechanism flags are false",
            "actual": claim_flags,
            "passed": claim_flags["promoted_mechanism_exists"] is False
            and claim_flags["full_reliability_mechanism_exists"] is False
            and claim_flags["transfer_ready_mechanism_exists"] is False
            and claim_flags["general_truth_vector_found"] is False
            and claim_flags["general_knowledge_control_surface_found"] is False,
            "why": "The audit captures a map of pressure points, not a general truth or knowledge vector.",
        },
    ]
    return checks


def build_control_surface_compositional_genome_audit() -> dict[str, Any]:
    atlas = load_json(ATLAS_PATH)
    mixture = load_json(MIXTURE_LAW_PATH)
    frontier = load_json(DECISION_FRONTIER_PATH)
    reliability = load_json(RELIABILITY_MATRIX_PATH)
    transfer = load_json(TRANSFER_MATRIX_PATH)
    mc005 = load_json(MC005_REFERENCE_SPECIMEN_AUDIT_PATH)
    mc006 = load_json(MC006_PREDECISION_FRONTIER_AUDIT_PATH)
    post_mc033 = load_json(POST_MC033_BRIDGE_CLOSEOUT_PATH)

    row_count = len(atlas["rows"])
    distribution_axes = build_distribution_axes(
        mixture, frontier, reliability, transfer, post_mc033
    )
    claim_boundary = build_claim_boundary(mixture, reliability, transfer)
    axis_by_id = {axis["axis_id"]: axis for axis in distribution_axes}
    summary = {
        "row_count": row_count,
        "prompt_contract_visible_count": axis_by_id[
            "prompt_contract_visible"
        ]["count"],
        "output_geometry_visible_count": axis_by_id[
            "output_geometry_visible"
        ]["count"],
        "source_or_prompt_token_dependent_count": axis_by_id[
            "source_or_prompt_token_dependent"
        ]["count"],
        "behavior_or_bridge_substrate_blocked_count": axis_by_id[
            "behavior_or_bridge_substrate_blocked"
        ]["count"],
        "internal_monitor_present_count": axis_by_id[
            "internal_monitor_present"
        ]["count"],
        "predecision_monitor_no_lever_count": axis_by_id[
            "predecision_monitor_no_lever"
        ]["count"],
        "internal_causal_surface_count": axis_by_id[
            "internal_causal_surface"
        ]["count"],
        "bounded_reliability_reference_count": axis_by_id[
            "bounded_reliability_reference"
        ]["count"],
        "full_reliability_mechanism_count": axis_by_id[
            "full_reliability_mechanism"
        ]["count"],
        "transfer_ready_mechanism_count": axis_by_id[
            "transfer_ready_mechanism"
        ]["count"],
        "clean_unconfounded_bridge_substrate_count": axis_by_id[
            "clean_unconfounded_bridge_substrate"
        ]["count"],
        "promoted_mechanism_count": mixture["summary"]["promoted_mechanism_count"],
        "dominant_current_object": "measured_compositional_mixture",
    }
    source_snapshot = {
        "mixture": mixture["summary"],
        "mixture_rows": mixture["pressure_class_rows"],
        "frontier": frontier["summary"],
        "reliability": reliability["summary"],
        "transfer": transfer["summary"],
        "mc005": mc005["summary"],
        "mc006": mc006["summary"],
        "post_mc033": post_mc033["summary"],
    }
    payload = {
        "schema_version": 1,
        "updated_at": atlas.get("updated_at"),
        "purpose": (
            "Make the current genome-level insight machine-readable: the main "
            "object is the distribution of behavior across control-surface axes, "
            "not the next isolated mechanism-card candidate."
        ),
        "sources": {
            "atlas": rel(ATLAS_PATH),
            "mixture_law": rel(MIXTURE_LAW_PATH),
            "decision_frontier": rel(DECISION_FRONTIER_PATH),
            "reliability_matrix": rel(RELIABILITY_MATRIX_PATH),
            "transfer_matrix": rel(TRANSFER_MATRIX_PATH),
            "mc005_reference_specimen": rel(MC005_REFERENCE_SPECIMEN_AUDIT_PATH),
            "mc006_predecision_frontier": rel(MC006_PREDECISION_FRONTIER_AUDIT_PATH),
            "post_mc033_bridge_closeout": rel(POST_MC033_BRIDGE_CLOSEOUT_PATH),
        },
        "classification_rule": {
            "distribution_axes": (
                "Non-exclusive row or bridge-rung ratios copied from generated "
                "layers and interpreted as compositional genome axes."
            ),
            "anchor_findings": (
                "Selected rows and closeouts that make the mixture law concrete: "
                "one bounded causal reference, one knowledge-like monitor-only "
                "route, one bridge-substrate death condition, and the current "
                "lead-time frontier."
            ),
        },
        "summary": summary,
        "distribution_axes": distribution_axes,
        "anchor_findings": build_anchor_findings(
            atlas, mixture, frontier, mc005, mc006, post_mc033
        ),
        "insight_claims": build_insight_claims(
            mixture, frontier, reliability, transfer
        ),
        "source_snapshot": source_snapshot,
        "claim_boundary": claim_boundary,
        "allowed_claim": claim_boundary["allowed_claim"],
        "forbidden_claim": claim_boundary["forbidden_claim"],
    }
    payload["validation_checks"] = build_validation_checks(payload)
    return payload


def validate_compositional_genome_audit(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("compositional genome audit schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"compositional genome source missing: {rel_path}")
    axes = {entry["axis_id"]: entry for entry in payload.get("distribution_axes", [])}
    required_axes = {
        "prompt_contract_visible",
        "output_geometry_visible",
        "source_or_prompt_token_dependent",
        "behavior_or_bridge_substrate_blocked",
        "internal_monitor_present",
        "predecision_monitor_no_lever",
        "internal_causal_surface",
        "bounded_reliability_reference",
        "full_reliability_mechanism",
        "transfer_ready_mechanism",
        "transfer_untested_or_missing",
        "clean_unconfounded_bridge_substrate",
    }
    if set(axes) != required_axes:
        raise AssertionError(f"unexpected compositional axes: {sorted(axes)}")
    summary = payload["summary"]
    if summary["dominant_current_object"] != "measured_compositional_mixture":
        raise AssertionError("compositional audit must keep mixture as dominant object")
    if summary["promoted_mechanism_count"] != 0:
        raise AssertionError("compositional audit must not promote a mechanism")
    if summary["internal_causal_surface_count"] != 1:
        raise AssertionError("expected exactly one internal causal surface")
    if summary["full_reliability_mechanism_count"] != 0:
        raise AssertionError("expected zero full-reliability mechanisms")
    if summary["transfer_ready_mechanism_count"] != 0:
        raise AssertionError("expected zero transfer-ready mechanisms")
    if axes["prompt_contract_visible"]["ratio"] != 1.0:
        raise AssertionError("prompt-contract pressure should remain universal")
    flags = payload["claim_boundary"]["claim_flags"]
    if flags["general_truth_vector_found"] or flags["general_knowledge_control_surface_found"]:
        raise AssertionError("broad vector/control flags must remain false")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(f"compositional genome checks failed: {failed_checks}")


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Control-Surface Compositional Genome Audit",
        "",
        f"Source updated_at: {payload['updated_at']}",
        "",
        "Status: generated compositional-genome audit implemented and validated.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/control_surface_compositional_genome_audit.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_compositional_genome_audit.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_compositional_genome_audit.py --write",
        "python code\\control_surface_compositional_genome_audit.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Purpose",
        "",
        "This audit makes the project-level insight explicit: the current genome",
        "object is not a single truth vector, knowledge vector, or mechanism card.",
        "It is the measured distribution of where behavior lives across prompt,",
        "output, source-token, internal-monitor, causal, reliability, and transfer",
        "axes.",
        "",
        "## Generated Facts",
        "",
        f"- atlas rows: {summary['row_count']};",
        f"- promoted mechanism cards: {summary['promoted_mechanism_count']};",
        f"- prompt-contract-visible rows: {summary['prompt_contract_visible_count']};",
        f"- output-geometry-visible rows: {summary['output_geometry_visible_count']};",
        f"- source-or-prompt-token-dependent rows: {summary['source_or_prompt_token_dependent_count']};",
        f"- behavior-or-bridge-substrate-blocked rows: {summary['behavior_or_bridge_substrate_blocked_count']};",
        f"- internal-monitor-present rows: {summary['internal_monitor_present_count']};",
        f"- predecision-monitor-with-no-lever rows: {summary['predecision_monitor_no_lever_count']};",
        f"- internal-causal-surface rows: {summary['internal_causal_surface_count']};",
        f"- full-reliability mechanisms: {summary['full_reliability_mechanism_count']};",
        f"- transfer-ready mechanisms: {summary['transfer_ready_mechanism_count']};",
        f"- clean unconfounded bridge substrates: {summary['clean_unconfounded_bridge_substrate_count']}.",
        "",
        "## Distribution Axes",
        "",
        "| Axis | Count | Denominator | Ratio | Interpretation |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for axis in payload["distribution_axes"]:
        lines.append(
            f"| `{axis['axis_id']}` | {axis['count']} | "
            f"{axis['denominator']} `{axis['denominator_label']}` | "
            f"{format_value(axis['ratio'])} | {axis['interpretation']} |"
        )

    lines.extend(
        [
            "",
            "## Anchor Findings",
            "",
            "| Anchor | Role | Status | Insight |",
            "| --- | --- | --- | --- |",
        ]
    )
    for anchor in payload["anchor_findings"]:
        status = anchor["current_status"]
        if isinstance(status, dict):
            compact_status = {
                key: status[key]
                for key in sorted(status)
                if key
                in {
                    "row_id",
                    "verdict",
                    "route_status",
                    "frontier_class",
                    "terminal_stage",
                    "reliability_class",
                    "transfer_class",
                    "hidden_state_allowed_count",
                    "clean_unconfounded_bridge_count",
                    "predecision_causal_candidate_count",
                    "monitor_only_rows",
                }
            }
        else:
            compact_status = status
        lines.append(
            f"| `{anchor['anchor_id']}` | `{anchor['role']}` | "
            f"`{format_value(compact_status)}` | {anchor['insight']} |"
        )

    lines.extend(
        [
            "",
            "## Insight Claims",
            "",
            "| Claim | Status | Interpretation |",
            "| --- | --- | --- |",
        ]
    )
    for claim in payload["insight_claims"]:
        lines.append(
            f"| `{claim['claim_id']}` | `{claim['status']}` | "
            f"{claim['interpretation']} |"
        )

    lines.extend(
        [
            "",
            "## Validation Checks",
            "",
            "| Check | Passed | Actual |",
            "| --- | --- | --- |",
        ]
    )
    for check in payload["validation_checks"]:
        lines.append(
            f"| `{check['id']}` | `{str(check['passed']).lower()}` | "
            f"`{format_value(check['actual'])}` |"
        )

    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            payload["allowed_claim"],
            "",
            payload["forbidden_claim"],
            "",
            "## Interpretation",
            "",
            "The audit converts negative results into measured structure. MC005 is the",
            "calibration object for bounded internal causality; MC006 is the",
            "knowledge-like timing boundary; the post-MC033 bridge closeout is the",
            "substrate death condition; and the distribution axes are the current",
            "genome-level object. The next useful experiments should change these",
            "ratios or explain why they are stable.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write JSON and markdown artifacts")
    args = parser.parse_args()

    payload = build_control_surface_compositional_genome_audit()
    validate_compositional_genome_audit(payload)
    if args.write:
        write_json(COMPOSITIONAL_GENOME_AUDIT_PATH, payload)
        COMPOSITIONAL_GENOME_AUDIT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        COMPOSITIONAL_GENOME_AUDIT_REPORT_PATH.write_text(
            render_markdown(payload),
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "passed": True,
                "row_count": payload["summary"]["row_count"],
                "distribution_axis_count": len(payload["distribution_axes"]),
                "promoted_mechanism_count": payload["summary"][
                    "promoted_mechanism_count"
                ],
                "internal_causal_surface_count": payload["summary"][
                    "internal_causal_surface_count"
                ],
                "output_geometry_visible_count": payload["summary"][
                    "output_geometry_visible_count"
                ],
                "source_or_prompt_token_dependent_count": payload["summary"][
                    "source_or_prompt_token_dependent_count"
                ],
                "validation_check_count": len(payload["validation_checks"]),
                "output_path": rel(COMPOSITIONAL_GENOME_AUDIT_PATH),
                "report_path": rel(COMPOSITIONAL_GENOME_AUDIT_REPORT_PATH),
            },
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
