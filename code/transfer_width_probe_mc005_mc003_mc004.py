"""Build the transfer-width probe packet for MC005/MC003/MC004.

This is a preregistration generator, not a model runner. It turns the
offensive-doctrine work order into an executable contract for the first real
widening probe: a bounded MC005 reference transfer plus MC003/MC004 diagnostic
baselines, with null locality treated as co-equal to primary effect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from control_surface_artifacts import ATLAS_PATH, ROOT, load_json
from control_surface_offensive_doctrine import OFFENSIVE_DOCTRINE_PATH
from control_surface_transfer_matrix import TRANSFER_MATRIX_PATH


TRANSFER_WIDTH_PROBE_PATH = (
    ROOT / "data" / "transfer_width_probe_mc005_mc003_mc004.json"
)
TRANSFER_WIDTH_PROBE_REPORT_PATH = (
    ROOT / "research" / "prereg" / "TRANSFER_WIDTH_PROBE_MC005_MC003_MC004.md"
)

WORK_ORDER_ID = "run_width_transfer_probe"
TARGET_ROW_IDS = [
    "mc005_associative_lookup",
    "mc003_delayed_copy",
    "mc004_in_context_binding",
]
PRIMARY_NON_QWEN_TARGETS = [
    "google/gemma-2-2b-it",
    "google/gemma-2-2b",
]
REQUIRED_PANEL_TYPES = {
    "primary_effect",
    "null_locality",
    "side_effects",
    "prompt_robustness",
    "output_shadow_baseline",
    "monitor_only_baseline",
}


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


def by_id(items: list[dict[str, Any]], key: str = "id") -> dict[str, dict[str, Any]]:
    return {item[key]: item for item in items}


def get_contract(doctrine: dict[str, Any]) -> dict[str, Any]:
    for contract in doctrine["branch_contracts"]:
        if contract["work_order_id"] == WORK_ORDER_ID:
            return contract
    raise AssertionError(f"missing offensive-doctrine contract {WORK_ORDER_ID}")


def compact_row(row: dict[str, Any], transfer_entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": row["id"],
        "family": row["family"],
        "models": row["models"],
        "verdict": row["verdict"]["class"],
        "lead_time_state": row["lead_time"]["state"],
        "intervention_state": row["intervention"]["state"],
        "null_locality": row["mixture_profile"]["null_locality"],
        "transfer_value": transfer_entry["transfer_value"],
        "transfer_class": transfer_entry["transfer_class"],
        "route_disposition": transfer_entry["route_disposition"],
        "diagnostics": row["diagnostics"],
        "evidence": row["evidence"],
    }


def build_row_roles(
    atlas_rows: dict[str, dict[str, Any]],
    transfer_entries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    mc005 = compact_row(
        atlas_rows["mc005_associative_lookup"],
        transfer_entries["mc005_associative_lookup"],
    )
    mc003 = compact_row(
        atlas_rows["mc003_delayed_copy"],
        transfer_entries["mc003_delayed_copy"],
    )
    mc004 = compact_row(
        atlas_rows["mc004_in_context_binding"],
        transfer_entries["mc004_in_context_binding"],
    )
    return [
        {
            **mc005,
            "probe_role": "bounded_reference_surface",
            "transfer_question": (
                "Does source-value lookup mediation transfer with null locality, "
                "side rows, source-disjoint holdouts, and prompt robustness, or "
                "does the primary effect appear before reliability?"
            ),
            "expected_failure_to_measure": "TRANSFER_PRIMARY_BEFORE_RELIABILITY",
        },
        {
            **mc003,
            "probe_role": "output_shadow_diagnostic_baseline",
            "transfer_question": (
                "Does a delayed-copy lead-time-looking signal remain an output "
                "shadow under the same transfer harness?"
            ),
            "expected_failure_to_measure": "OUTPUT_SHADOW_REPLICATES_UNDER_WIDTH",
        },
        {
            **mc004,
            "probe_role": "predecision_monitor_diagnostic_baseline",
            "transfer_question": (
                "Does an in-context-binding predecision monitor transfer as a "
                "monitor-only surface without becoming a lever?"
            ),
            "expected_failure_to_measure": "PREDECISION_MONITOR_NO_LEVER_REPLICATES",
        },
    ]


def panel(
    panel_id: str,
    row_id: str,
    panel_type: str,
    purpose: str,
    required_measurements: list[str],
    success_rule: str,
    bound_rule: str,
    kill_rule: str,
    failure_exports: list[str],
    generated_layer_if_moves: list[str],
) -> dict[str, Any]:
    return {
        "id": panel_id,
        "row_id": row_id,
        "panel_type": panel_type,
        "purpose": purpose,
        "required_measurements": required_measurements,
        "success_rule": success_rule,
        "bound_rule": bound_rule,
        "kill_rule": kill_rule,
        "failure_exports": failure_exports,
        "generated_layer_if_moves": generated_layer_if_moves,
    }


def build_panels() -> list[dict[str, Any]]:
    return [
        panel(
            "mc005_gemma_primary_lookup_effect",
            "mc005_associative_lookup",
            "primary_effect",
            "Replicate the MC005 answer-present lookup behavior and intervention direction on a non-Qwen target.",
            [
                "answer-present lookup accuracy on source-disjoint holdout",
                "source-value source-mask or homologous local intervention effect",
                "target versus distractor source-row specificity",
                "late-band coordinate mapping stated by fractional depth, not copied layer numbers",
            ],
            (
                "Pass only if the non-Qwen target preserves high answer-present "
                "lookup accuracy and the local intervention moves behavior in "
                "the MC005-predicted direction without relying on Qwen layer numbers."
            ),
            (
                "Bound if the answer-present primary effect reproduces but the "
                "intervention is broader, weaker, or not locally homologous."
            ),
            (
                "Kill transfer for the MC005 write surface if the primary lookup "
                "effect itself fails on two non-Qwen or prompt-family targets."
            ),
            ["TRANSFER_PRIMARY_FAILED"],
            [
                "data/control_surface_transfer_matrix.json",
                "data/control_surface_gate_geometry.json",
            ],
        ),
        panel(
            "mc005_gemma_answer_absent_null_locality",
            "mc005_associative_lookup",
            "null_locality",
            "Test whether answer-absent rows remain stable under the same transfer intervention.",
            [
                "answer-absent null rows stratified by target-model margin",
                "off-target null flips under intervention",
                "low-margin and high-margin null strata reported separately",
                "comparison against no-intervention, source-deletion, and neutral-rewrite controls",
            ],
            (
                "Pass only if answer-absent null flips stay below the predeclared "
                "trivial threshold across strata, holdouts, and intervention variants."
            ),
            (
                "Bound if answer-present lookup transfers but answer-absent null "
                "flips recur, especially in low-to-moderate-margin rows."
            ),
            (
                "Kill a transfer-success claim immediately if null flips are "
                "averaged away or reported without margin strata."
            ),
            [
                "TRANSFER_PRIMARY_BEFORE_RELIABILITY",
                "NULL_ROW_LOW_MARGIN_FLIP_REPLICATES_UNDER_WIDTH",
            ],
            [
                "data/control_surface_reliability_matrix.json",
                "data/control_surface_transfer_matrix.json",
            ],
        ),
        panel(
            "mc005_gemma_side_rows_and_prompt_robustness",
            "mc005_associative_lookup",
            "side_effects",
            "Measure whether prompt robustness and side rows transfer with the primary effect.",
            [
                "layout, lexicon, pair-count, and long-context prompt variants",
                "query-only path comparison",
                "neutral rewrite comparison",
                "side-row value and distractor-row effects",
                "fluency or parseability side effects",
            ],
            (
                "Pass only if primary effect, side rows, prompt variants, and "
                "fluency remain within predefined tolerances together."
            ),
            (
                "Bound if the primary effect survives but robustness or side-row "
                "locality fails."
            ),
            (
                "Kill transfer-ready status if side rows or prompt variants fail "
                "while the primary effect appears clean."
            ),
            [
                "TRANSFER_PRIMARY_BEFORE_RELIABILITY",
                "TRANSFER_SIDE_EFFECT_BOUNDARY",
            ],
            [
                "data/control_surface_reliability_matrix.json",
                "data/control_surface_coverage_gaps.json",
            ],
        ),
        panel(
            "mc005_gemma_prompt_robustness",
            "mc005_associative_lookup",
            "prompt_robustness",
            "Keep MC005 transfer from becoming a single-format replication.",
            [
                "same grammar as source Qwen panel",
                "Response-marker variant",
                "neutral marker variant",
                "source-disjoint prompt-family holdout",
            ],
            (
                "Pass only if the effect survives the preregistered prompt-family "
                "holdouts without changing the claim boundary."
            ),
            (
                "Bound if transfer exists only under the original Response: "
                "contract."
            ),
            (
                "Kill broad transfer language if one prompt contract carries the "
                "entire result."
            ),
            ["PROMPT_CONTRACT_TRANSFER_FRAGILITY"],
            [
                "data/control_surface_transfer_matrix.json",
                "data/control_surface_coverage_gaps.json",
            ],
        ),
        panel(
            "mc003_delayed_copy_output_shadow_baseline",
            "mc003_delayed_copy",
            "output_shadow_baseline",
            "Use MC003 as a negative widening control for lead-time-looking output shadows.",
            [
                "behavior table balance",
                "same-stage output margin baseline",
                "final output or candidate margin baseline",
                "shuffled-label selection null",
                "source-disjoint holdout",
            ],
            (
                "This panel is not expected to promote. It passes as a diagnostic "
                "baseline if output/candidate or shuffle controls still explain "
                "the signal."
            ),
            (
                "If a stronger signal appears, it remains bounded until it beats "
                "same-stage and final-stage baselines and earns an intervention plan."
            ),
            (
                "Kill any transfer interpretation if the MC003 signal is reported "
                "without output/candidate baselines."
            ),
            ["OUTPUT_SHADOW_REPLICATES_UNDER_WIDTH"],
            [
                "data/control_surface_decision_frontier.json",
                "data/control_surface_axis_interactions.json",
            ],
        ),
        panel(
            "mc004_binding_monitor_only_baseline",
            "mc004_in_context_binding",
            "monitor_only_baseline",
            "Use MC004 as the predecision-monitor baseline for transfer-width interpretation.",
            [
                "pre-update or pre-answer coordinate selected before final answer",
                "subgroup robustness",
                "source-disjoint holdout",
                "shuffled-label selection null",
                "same-stage and final-stage output/candidate controls",
            ],
            (
                "This panel only opens a future intervention route if the monitor "
                "beats subgroup, shuffle, same-stage, and final-stage controls."
            ),
            (
                "Bound as monitor-only if the lead-time signal exists but remains "
                "fragile under subgroup or final-stage controls."
            ),
            (
                "Kill any lever claim if the signal is predictive but fails "
                "shuffle, subgroup, or output/candidate controls."
            ),
            ["PREDECISION_MONITOR_NO_LEVER_REPLICATES"],
            [
                "data/control_surface_decision_frontier.json",
                "data/control_surface_route_disposition.json",
            ],
        ),
    ]


def build_validation_checks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    panel_types = {panel["panel_type"] for panel in payload["probe_panels"]}
    missing_panel_types = sorted(REQUIRED_PANEL_TYPES - panel_types)
    target_rows = [role["row_id"] for role in payload["row_roles"]]
    non_qwen_targets = [
        model
        for model in payload["target_models"]["primary_non_qwen_targets"]
        if "qwen" not in model.lower()
    ]
    checks = [
        {
            "id": "work_order_contract_matches_transfer_probe",
            "predicate": f"== {WORK_ORDER_ID}",
            "actual": payload["work_order"]["id"],
            "passed": payload["work_order"]["id"] == WORK_ORDER_ID,
            "why": "The packet must be generated from the offensive-doctrine widening contract.",
        },
        {
            "id": "target_rows_are_reference_and_diagnostics",
            "predicate": f"== {TARGET_ROW_IDS}",
            "actual": target_rows,
            "passed": target_rows == TARGET_ROW_IDS,
            "why": "The probe must include one bounded reference row and two diagnostic rows.",
        },
        {
            "id": "bounded_reference_is_mc005",
            "predicate": "mc005 verdict bounded_mechanism_card",
            "actual": payload["row_roles"][0],
            "passed": (
                payload["row_roles"][0]["row_id"] == "mc005_associative_lookup"
                and payload["row_roles"][0]["verdict"] == "bounded_mechanism_card"
            ),
            "why": "The transfer probe needs a bounded mechanism-like source specimen.",
        },
        {
            "id": "non_qwen_target_is_predeclared",
            "predicate": "at least one non-Qwen target",
            "actual": payload["target_models"]["primary_non_qwen_targets"],
            "passed": bool(non_qwen_targets),
            "why": "Qwen-only widening cannot support small-model generalization.",
        },
        {
            "id": "required_panel_types_present",
            "predicate": "empty missing_panel_types",
            "actual": missing_panel_types,
            "passed": not missing_panel_types,
            "why": "Transfer must measure primary effect, nulls, side rows, prompt robustness, and diagnostic baselines.",
        },
        {
            "id": "all_panels_have_decision_fields",
            "predicate": "empty list",
            "actual": [
                panel["id"]
                for panel in payload["probe_panels"]
                if not (
                    panel["success_rule"]
                    and panel["bound_rule"]
                    and panel["kill_rule"]
                    and panel["failure_exports"]
                    and panel["generated_layer_if_moves"]
                )
            ],
            "passed": all(
                panel["success_rule"]
                and panel["bound_rule"]
                and panel["kill_rule"]
                and panel["failure_exports"]
                and panel["generated_layer_if_moves"]
                for panel in payload["probe_panels"]
            ),
            "why": "Every panel must be promotable, bounded, killed, or exported.",
        },
        {
            "id": "no_transfer_success_claimed_by_prereg",
            "predicate": "transfer_success_claimed == false",
            "actual": payload["claim_boundary"]["transfer_success_claimed"],
            "passed": payload["claim_boundary"]["transfer_success_claimed"] is False,
            "why": "This is a preregistered probe packet, not a result.",
        },
    ]
    return checks


def build_transfer_width_probe() -> dict[str, Any]:
    atlas = load_json(ATLAS_PATH)
    transfer_matrix = load_json(TRANSFER_MATRIX_PATH)
    doctrine = load_json(OFFENSIVE_DOCTRINE_PATH)
    contract = get_contract(doctrine)
    rows = by_id(atlas["rows"])
    transfer_entries = by_id(transfer_matrix["transfer_entries"], "row_id")
    row_roles = build_row_roles(rows, transfer_entries)
    payload = {
        "schema_version": 1,
        "updated_at": atlas.get("updated_at"),
        "purpose": (
            "Pre-register the first width-transfer probe: MC005 as the bounded "
            "reference specimen, MC003 as an output-shadow diagnostic baseline, "
            "and MC004 as a predecision monitor-only diagnostic baseline."
        ),
        "sources": {
            "atlas": rel(ATLAS_PATH),
            "transfer_matrix": rel(TRANSFER_MATRIX_PATH),
            "offensive_doctrine": rel(OFFENSIVE_DOCTRINE_PATH),
        },
        "work_order": {
            "id": contract["work_order_id"],
            "urgency": contract["urgency"],
            "track_type": contract["track_type"],
            "target_gap_ids": contract["target_gap_ids"],
            "first_artifact": contract["first_artifact"],
            "iteration_budget": contract["iteration_budget"],
            "promotion_rule": contract["promotion_rule"],
            "bound_rule": contract["bound_rule"],
            "kill_rule": contract["kill_rule"],
            "containment_rule": contract["containment_rule"],
            "export_rule": contract["export_rule"],
        },
        "target_models": {
            "source_models_from_rows": {
                row_id: rows[row_id]["models"] for row_id in TARGET_ROW_IDS
            },
            "primary_non_qwen_targets": PRIMARY_NON_QWEN_TARGETS,
            "why_non_qwen": (
                "The transfer matrix already records Qwen-dominant evidence; "
                "the first width probe must test a materially comparable "
                "non-Qwen model family before any small-model generalization claim."
            ),
        },
        "row_roles": row_roles,
        "probe_panels": build_panels(),
        "claim_boundary": {
            "transfer_success_claimed": False,
            "allowed_claim": (
                "This packet makes transfer measurable by requiring primary "
                "effect, null locality, side rows, prompt robustness, and "
                "diagnostic baselines before any transfer field may improve."
            ),
            "forbidden_claim": (
                "This packet does not prove MC005 transfers, does not license "
                "MC003 or MC004 interventions, and does not reduce the current "
                "count of zero transfer-ready mechanisms."
            ),
        },
    }
    payload["validation_checks"] = build_validation_checks(payload)
    return payload


def validate_transfer_width_probe(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("transfer width probe schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"transfer width probe source missing: {rel_path}")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(f"transfer width probe checks failed: {failed_checks}")


def render_markdown(payload: dict[str, Any]) -> str:
    work_order = payload["work_order"]
    lines = [
        "# Transfer Width Probe: MC005 / MC003 / MC004",
        "",
        "Date: 2026-07-01",
        "",
        "Status: preregistered transfer-width packet; no transfer result claimed.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/transfer_width_probe_mc005_mc003_mc004.json`",
        "",
        "Builder:",
        "",
        "> `code/transfer_width_probe_mc005_mc003_mc004.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\transfer_width_probe_mc005_mc003_mc004.py --write",
        "python code\\transfer_width_probe_mc005_mc003_mc004.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Target Gap",
        "",
        f"- work order: `{work_order['id']}`;",
        f"- urgency: `{work_order['urgency']}`;",
        f"- track: `{work_order['track_type']}`;",
        f"- target gaps: `{format_value(work_order['target_gap_ids'])}`;",
        f"- iteration budget: {work_order['iteration_budget']}.",
        "",
        "## Purpose",
        "",
        "This packet makes transfer failure or success measurable instead of",
        "assumed. It deliberately pairs one bounded reference specimen with two",
        "diagnostic rows:",
        "",
    ]
    for role in payload["row_roles"]:
        lines.append(
            f"- `{role['row_id']}`: `{role['probe_role']}`; "
            f"transfer class `{role['transfer_class']}`; "
            f"question: {role['transfer_question']}"
        )

    lines.extend(
        [
            "",
            "## Target Models",
            "",
            f"- primary non-Qwen targets: `{format_value(payload['target_models']['primary_non_qwen_targets'])}`;",
            f"- rationale: {payload['target_models']['why_non_qwen']}",
            "",
            "## Probe Panels",
            "",
            "| Panel | Row | Type | Required Measurements | Success Rule | Failure Exports |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for probe_panel in payload["probe_panels"]:
        lines.append(
            f"| `{probe_panel['id']}` | `{probe_panel['row_id']}` | "
            f"`{probe_panel['panel_type']}` | "
            f"`{format_value(probe_panel['required_measurements'])}` | "
            f"{probe_panel['success_rule']} | "
            f"`{format_value(probe_panel['failure_exports'])}` |"
        )

    lines.extend(
        [
            "",
            "## Decision Rules",
            "",
            f"- promotion rule: {work_order['promotion_rule']}",
            f"- bound rule: {work_order['bound_rule']}",
            f"- kill rule: {work_order['kill_rule']}",
            f"- containment rule: {work_order['containment_rule']}",
            f"- export rule: {work_order['export_rule']}",
            "",
            "## Claim Boundary",
            "",
            payload["claim_boundary"]["allowed_claim"],
            "",
            payload["claim_boundary"]["forbidden_claim"],
            "",
            "## What This Proves",
            "",
            "It proves that the first transfer-width test is now a branch",
            "contract, not a vague widening wish. A future result cannot count",
            "as transfer unless the primary effect, null locality, side rows,",
            "prompt robustness, and diagnostic baselines are reported together.",
            "",
            "## What It Does Not Prove",
            "",
            "It does not prove any new model result. It does not move MC005 beyond",
            "bounded transfer-fragile status, and it does not make MC003 or MC004",
            "intervention-ready.",
            "",
        ]
    )
    return "\n".join(lines)


def write_transfer_width_probe(
    output_path: Path = TRANSFER_WIDTH_PROBE_PATH,
    report_path: Path = TRANSFER_WIDTH_PROBE_REPORT_PATH,
) -> dict[str, Any]:
    payload = build_transfer_width_probe()
    validate_transfer_width_probe(payload)
    write_json(output_path, payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(payload), encoding="utf-8", newline="\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="write transfer-width preregistration artifacts",
    )
    parser.add_argument("--json", action="store_true", help="print probe JSON")
    args = parser.parse_args()

    payload = build_transfer_width_probe()
    validate_transfer_width_probe(payload)

    if args.write:
        write_transfer_width_probe()
        print(
            f"wrote {TRANSFER_WIDTH_PROBE_PATH.relative_to(ROOT).as_posix()} and "
            f"{TRANSFER_WIDTH_PROBE_REPORT_PATH.relative_to(ROOT).as_posix()} "
            f"with {len(payload['probe_panels'])} panels"
        )
        return
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(
        "transfer width probe ok: "
        f"{len(payload['row_roles'])} rows, {len(payload['probe_panels'])} panels"
    )
    print(
        "target_gaps:",
        json.dumps(payload["work_order"]["target_gap_ids"], sort_keys=True),
    )
    print(
        "non_qwen_targets:",
        json.dumps(
            payload["target_models"]["primary_non_qwen_targets"],
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
