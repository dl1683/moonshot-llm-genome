#!/usr/bin/env python
"""Executable closeout audit for the MC006 predecision frontier route."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "cards" / "MC006"

DEFAULT_PATHS = {
    "v14": RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v14_parser_normalized_20260701T000538.json",
    "v15": RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v15_parser_normalized_signature_20260701T000911.json",
    "v16": RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v16_pre_output_position_signature_20260701T002113.json",
    "v17": RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v17_pre_output_steering_stress_20260701T003620.json",
    "v18": RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v18_margin_matched_leadtime_20260701T022343.json",
    "v19": RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v19_overlapping_margin_table_20260701T023921.json",
    "v20": RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v20_strict_overlap_selection_audit_20260701T025043.json",
    "v21": RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v21_pair_matched_leadtime_20260701T030507.json",
    "v22": RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v22_source_path_leadtime_curve_20260701T032420.json",
    "v23": RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v23_final_margin_sign_barrier_20260701T033829.json",
    "v24": RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v24_delayed_city_interface_20260701T040731.json",
    "v25": RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v25_candidate_decoupled_template_20260701T041843.json",
    "v26": RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v26_locked_coordinate_transfer_20260701T043334.json",
    "v27": RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v27_expanded_candidate_decoupled_bank_20260701T045409.json",
    "v28": RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v28_transfer_role_repair_bank_20260701T051821.json",
}

EXPECTED_RUN_TYPES = {
    "v14": "parametric_fact_override_v14_parser_normalized",
    "v15": "parametric_fact_override_v15_parser_normalized_signature",
    "v16": "parametric_fact_override_v16_pre_output_position_signature",
    "v17": "parametric_fact_override_v17_pre_output_steering_stress",
    "v18": "parametric_fact_override_v18_margin_matched_leadtime",
    "v19": "parametric_fact_override_v19_overlapping_margin_table",
    "v20": "parametric_fact_override_v20_strict_overlap_selection_audit",
    "v21": "parametric_fact_override_v21_pair_matched_leadtime",
    "v22": "parametric_fact_override_v22_source_path_leadtime_curve",
    "v23": "parametric_fact_override_v23_final_margin_sign_barrier",
    "v24": "parametric_fact_override_v24_delayed_city_interface",
    "v25": "parametric_fact_override_v25_candidate_decoupled_template",
    "v26": "parametric_fact_override_v26_locked_coordinate_transfer",
    "v27": "parametric_fact_override_v27_expanded_candidate_decoupled_bank",
    "v28": "parametric_fact_override_v28_transfer_role_repair_bank",
}

EXPECTED_DIAGNOSTICS = {
    "v14": "parser_normalized_generated_substrate_passed",
    "v15": "candidate_score_confounded",
    "v16": "leadtime_signal_supported_but_output_global_confounded",
    "v17": "intervention_failed",
    "v18": "global_margin_separation_blocks_matching",
    "v19": "non_holdout_candidate_margin_overlap_failed",
    "v20": "strict_final_margin_overlap_absent",
    "v21": "approximate_pair_matching_failed_margin_baselines",
    "v22": "source_path_final_margin_shadow",
    "v23": "greedy_final_margin_sign_barrier",
    "v24": "delayed_city_interface_decouples_first_token_margin",
    "v25": "candidate_decoupled_hidden_shuffle_overfit",
    "v26": "locked_coordinate_transfer_failed",
    "v27": "expanded_candidate_decoupled_bank_insufficient",
    "v28": "transfer_role_repair_bank_insufficient",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def require_equal(errors: list[str], label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        errors.append(f"{label}: expected {expected!r}, got {actual!r}")


def require_bool(errors: list[str], label: str, actual: Any, expected: bool) -> None:
    if bool(actual) is not expected:
        errors.append(f"{label}: expected {expected!r}, got {actual!r}")


def summary(artifacts: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    return artifacts[key]["summary"]


def criteria(artifacts: dict[str, dict[str, Any]], key: str, field: str = "criteria") -> dict[str, Any]:
    return summary(artifacts, key).get(field, {})


def validate_identity(artifacts: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for key, artifact in artifacts.items():
        require_equal(errors, f"{key} run_type", artifact.get("run_type"), EXPECTED_RUN_TYPES[key])
        require_equal(errors, f"{key} diagnostic_class", summary(artifacts, key).get("diagnostic_class"), EXPECTED_DIAGNOSTICS[key])
    return errors


def validate_route_sentinels(artifacts: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    c14 = criteria(artifacts, "v14")
    require_bool(errors, "V14 selected_binary_rows_at_least_30", c14.get("selected_binary_rows_at_least_30"), True)
    require_bool(errors, "V14 selected_holdout_true_at_least_2", c14.get("selected_holdout_true_at_least_2"), True)
    require_bool(errors, "V14 selected_holdout_override_at_least_2", c14.get("selected_holdout_override_at_least_2"), True)

    c15 = criteria(artifacts, "v15")
    require_bool(errors, "V15 selected_hidden_holdout_auc_at_least_0p85", c15.get("selected_hidden_holdout_auc_at_least_0p85"), True)
    require_bool(errors, "V15 selected_hidden_beats_candidate_score_by_0p02", c15.get("selected_hidden_beats_candidate_score_by_0p02"), False)
    require_bool(errors, "V15 selected_hidden_beats_next_token_output_by_0p02", c15.get("selected_hidden_beats_next_token_output_by_0p02"), False)

    lead16 = criteria(artifacts, "v16", "leadtime_criteria")
    mech16 = criteria(artifacts, "v16", "mechanism_criteria")
    require_bool(errors, "V16 selected_pre_output_hidden_holdout_auc_at_least_0p85", lead16.get("selected_pre_output_hidden_holdout_auc_at_least_0p85"), True)
    require_bool(errors, "V16 selected_hidden_beats_same_position_output_by_0p02", lead16.get("selected_hidden_beats_same_position_output_by_0p02"), True)
    require_bool(errors, "V16 selected_hidden_beats_candidate_score_by_0p02", mech16.get("selected_hidden_beats_candidate_score_by_0p02"), False)
    require_bool(errors, "V16 selected_hidden_beats_final_next_token_output_by_0p02", mech16.get("selected_hidden_beats_final_next_token_output_by_0p02"), False)

    mech17 = criteria(artifacts, "v17", "mechanism_criteria")
    causal17 = criteria(artifacts, "v17", "causal_criteria")
    require_bool(errors, "V17 baseline_reproduces_at_least_6_of_8_holdout_labels", mech17.get("baseline_reproduces_at_least_6_of_8_holdout_labels"), True)
    require_bool(errors, "V17 plus_selected_increases_margin_by_0p25", causal17.get("plus_selected_increases_margin_by_0p25"), False)
    require_bool(errors, "V17 selected_abs_effect_beats_controls_by_0p25", causal17.get("selected_abs_effect_beats_controls_by_0p25"), False)
    require_bool(errors, "V17 not_global_output_confounded", mech17.get("not_global_output_confounded"), False)

    c18 = criteria(artifacts, "v18")
    require_bool(errors, "V18 selected_pre_output_reproduced", c18.get("selected_pre_output_reproduced"), True)
    require_bool(errors, "V18 candidate_margin_holdout_overlap_exists", c18.get("candidate_margin_holdout_overlap_exists"), False)
    require_bool(errors, "V18 final_output_margin_holdout_overlap_exists", c18.get("final_output_margin_holdout_overlap_exists"), False)
    require_bool(errors, "V18 residualized_hidden_beats_0p85_after_both_global_margins", c18.get("residualized_hidden_beats_0p85_after_both_global_margins"), False)

    c19 = criteria(artifacts, "v19")
    require_bool(errors, "V19 selected_binary_rows_at_least_30", c19.get("selected_binary_rows_at_least_30"), True)
    require_bool(errors, "V19 selected_non_holdout_candidate_margin_overlap", c19.get("selected_non_holdout_candidate_margin_overlap"), False)
    require_bool(errors, "V19 selected_holdout_final_margin_overlap", c19.get("selected_holdout_final_margin_overlap"), False)

    c20 = criteria(artifacts, "v20")
    require_bool(errors, "V20 pooled_non_holdout_candidate_overlap", c20.get("pooled_non_holdout_candidate_overlap"), True)
    require_bool(errors, "V20 pooled_non_holdout_final_overlap", c20.get("pooled_non_holdout_final_overlap"), False)
    require_bool(errors, "V20 any_single_template_full_gate_passed", c20.get("any_single_template_full_gate_passed"), False)

    c21 = criteria(artifacts, "v21")
    require_bool(errors, "V21 pair_matched_diagnostic_ready", c21.get("pair_matched_diagnostic_ready"), True)
    require_bool(
        errors,
        "V21 candidate_and_final_margin_holdout_pair_accuracy_at_most_0p65",
        c21.get("candidate_and_final_margin_holdout_pair_accuracy_at_most_0p65"),
        False,
    )
    require_bool(errors, "V21 selected_hidden_beats_candidate_and_final_margins_by_0p05", c21.get("selected_hidden_beats_candidate_and_final_margins_by_0p05"), False)

    c22 = criteria(artifacts, "v22")
    require_bool(errors, "V22 curve_any_hidden_holdout_supported", c22.get("curve_any_hidden_holdout_supported"), True)
    require_bool(errors, "V22 curve_any_hidden_beats_same_position_output_by_0p05", c22.get("curve_any_hidden_beats_same_position_output_by_0p05"), True)
    require_bool(errors, "V22 curve_any_hidden_beats_candidate_and_final_margins_by_0p05", c22.get("curve_any_hidden_beats_candidate_and_final_margins_by_0p05"), False)

    c23 = criteria(artifacts, "v23")
    require_bool(errors, "V23 v18_final_sign_barrier", c23.get("v18_final_sign_barrier"), True)
    require_bool(errors, "V23 v19_selected_final_sign_barrier", c23.get("v19_selected_final_sign_barrier"), True)
    require_bool(errors, "V23 v19_candidate_raw_overlap_exists", c23.get("v19_candidate_raw_overlap_exists"), True)

    c24 = criteria(artifacts, "v24")
    require_bool(errors, "V24 selected_binary_rows_at_least_30", c24.get("selected_binary_rows_at_least_30"), True)
    require_bool(errors, "V24 selected_first_token_city_rate_at_most_0p1", c24.get("selected_first_token_city_rate_at_most_0p1"), True)
    require_bool(errors, "V24 selected_final_city_margin_overlap_exists", c24.get("selected_final_city_margin_overlap_exists"), True)

    c25 = criteria(artifacts, "v25")
    require_bool(errors, "V25 candidate_decoupled_template_found", c25.get("candidate_decoupled_template_found"), True)
    require_bool(errors, "V25 hidden_beats_candidate_controls", c25.get("hidden_beats_candidate_controls"), True)
    require_bool(errors, "V25 hidden_beats_shuffle_null", c25.get("hidden_beats_shuffle_null"), False)
    require_bool(errors, "V25 signature_gate", c25.get("signature_gate"), False)

    c26 = criteria(artifacts, "v26")
    require_bool(errors, "V26 locked_coordinate_available", c26.get("locked_coordinate_available"), True)
    require_bool(errors, "V26 source_holdout_reproduced", c26.get("source_holdout_reproduced"), True)
    require_bool(errors, "V26 all_transfer_hidden_auc_at_least_0p75", c26.get("all_transfer_hidden_auc_at_least_0p75"), False)
    require_bool(errors, "V26 locked_transfer_gate", c26.get("locked_transfer_gate"), False)

    c27 = criteria(artifacts, "v27")
    require_bool(errors, "V27 candidate_decoupled_templates_at_least_4", c27.get("candidate_decoupled_templates_at_least_4"), True)
    require_bool(errors, "V27 source_ready_templates_at_least_2", c27.get("source_ready_templates_at_least_2"), True)
    require_bool(errors, "V27 transfer_ready_templates_at_least_2", c27.get("transfer_ready_templates_at_least_2"), False)
    require_bool(errors, "V27 expanded_bank_ready", c27.get("expanded_bank_ready"), False)

    c28 = criteria(artifacts, "v28")
    require_bool(errors, "V28 any_transfer_candidate_decoupled_template", c28.get("any_transfer_candidate_decoupled_template"), True)
    require_bool(errors, "V28 transfer_ready_templates_at_least_2", c28.get("transfer_ready_templates_at_least_2"), False)
    require_bool(errors, "V28 combined_ready_templates_at_least_5", c28.get("combined_ready_templates_at_least_5"), False)
    require_bool(errors, "V28 transfer_repair_bank_ready", c28.get("transfer_repair_bank_ready"), False)
    return errors


def build_closeout(paths: dict[str, Path]) -> dict[str, Any]:
    artifacts = {key: read_json(path) for key, path in paths.items()}
    validation_errors = validate_identity(artifacts)
    validation_errors.extend(validate_route_sentinels(artifacts))

    c14 = criteria(artifacts, "v14")
    lead16 = criteria(artifacts, "v16", "leadtime_criteria")
    mech16 = criteria(artifacts, "v16", "mechanism_criteria")
    c18 = criteria(artifacts, "v18")
    c22 = criteria(artifacts, "v22")
    c25 = criteria(artifacts, "v25")
    c28 = criteria(artifacts, "v28")

    behavior_substrate_passed = bool(
        c14.get("selected_binary_rows_at_least_30")
        and c14.get("selected_holdout_true_at_least_2")
        and c14.get("selected_holdout_override_at_least_2")
    )
    predecision_monitor_supported = bool(
        lead16.get("selected_pre_output_hidden_holdout_auc_at_least_0p85")
        and lead16.get("selected_hidden_beats_same_position_output_by_0p02")
        and c22.get("curve_any_hidden_holdout_supported")
    )
    final_or_candidate_geometry_blocks = bool(
        not mech16.get("selected_hidden_beats_candidate_score_by_0p02")
        and not mech16.get("selected_hidden_beats_final_next_token_output_by_0p02")
        and not c18.get("candidate_margin_holdout_overlap_exists")
        and not c18.get("final_output_margin_holdout_overlap_exists")
        and not c22.get("curve_any_hidden_beats_candidate_and_final_margins_by_0p05")
    )
    causal_and_transfer_routes_closed = bool(
        summary(artifacts, "v17").get("diagnostic_class") == "intervention_failed"
        and not criteria(artifacts, "v26").get("locked_transfer_gate")
        and not criteria(artifacts, "v27").get("expanded_bank_ready")
        and not criteria(artifacts, "v28").get("transfer_repair_bank_ready")
    )
    candidate_decoupled_search_failed = bool(
        c25.get("candidate_decoupled_template_found")
        and not c25.get("hidden_beats_shuffle_null")
        and not c25.get("signature_gate")
    )

    promotion_gate = bool(
        not validation_errors
        and behavior_substrate_passed
        and predecision_monitor_supported
        and not final_or_candidate_geometry_blocks
        and not candidate_decoupled_search_failed
        and not causal_and_transfer_routes_closed
    )
    monitor_only_closeout_gate = bool(
        not validation_errors
        and behavior_substrate_passed
        and predecision_monitor_supported
        and final_or_candidate_geometry_blocks
        and candidate_decoupled_search_failed
        and causal_and_transfer_routes_closed
    )

    if promotion_gate:
        verdict = "hidden_state_allowed_predecision_route"
        route_status = "promotion_route_open"
    elif monitor_only_closeout_gate:
        verdict = "diagnostic_note"
        route_status = "monitor_only_closed"
    else:
        verdict = "failed_or_invalid_closeout"
        route_status = "not_closed"

    source_sha256 = {key: sha256_file(path) for key, path in paths.items()}
    return {
        "schema_version": 1,
        "run_type": "mc006_predecision_frontier_closeout_audit",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "card_id": "MC006",
        "model_id": summary(artifacts, "v16").get("model_id"),
        "source_paths": {key: rel(path) for key, path in paths.items()},
        "source_sha256": source_sha256,
        "validation_errors": validation_errors,
        "criteria": {
            "artifact_sentinels_valid": not validation_errors,
            "behavior_substrate_passed": behavior_substrate_passed,
            "predecision_monitor_supported": predecision_monitor_supported,
            "final_or_candidate_geometry_blocks_promotion": final_or_candidate_geometry_blocks,
            "candidate_decoupled_hidden_selection_failed_shuffle_null": candidate_decoupled_search_failed,
            "causal_and_transfer_routes_closed": causal_and_transfer_routes_closed,
            "promotion_gate_passed": promotion_gate,
            "monitor_only_closeout_gate_passed": monitor_only_closeout_gate,
        },
        "decision": {
            "verdict": verdict,
            "route_status": route_status,
            "hidden_state_work_allowed": False,
            "intervention_allowed": False,
            "ordinary_route_repairs_allowed": False,
            "known_confounded_causal_stress_allowed": True,
            "materially_new_behavior_family_required": True,
        },
        "evidence": {
            "behavior_substrate": {
                "v14_diagnostic": summary(artifacts, "v14")["diagnostic_class"],
                "selected_template": summary(artifacts, "v14")["selection"]["selected_template"],
                "selected_binary_rows": summary(artifacts, "v14")["selected_template_summary"]["binary_count"],
                "holdout_label_counts": summary(artifacts, "v14")["selected_template_summary"][
                    "selected_label_counts_by_split"
                ]["holdout"],
            },
            "predecision_monitor": {
                "v16_diagnostic": summary(artifacts, "v16")["diagnostic_class"],
                "selected_position": summary(artifacts, "v16")["selected_pre_output_candidate"]["position"],
                "selected_layer": summary(artifacts, "v16")["selected_pre_output_candidate"]["layer"],
                "selected_holdout_auc": summary(artifacts, "v16")["selected_pre_output_candidate"]["holdout_auc"],
                "beats_same_position_output": lead16["selected_hidden_beats_same_position_output_by_0p02"],
                "beats_candidate_score": mech16["selected_hidden_beats_candidate_score_by_0p02"],
                "beats_final_next_token_output": mech16["selected_hidden_beats_final_next_token_output_by_0p02"],
            },
            "promotion_blockers": {
                "v17_intervention_failed": summary(artifacts, "v17")["diagnostic_class"],
                "v18_global_margin_separation": summary(artifacts, "v18")["diagnostic_class"],
                "v21_pair_matching_failed": summary(artifacts, "v21")["diagnostic_class"],
                "v22_source_path_shadow": summary(artifacts, "v22")["diagnostic_class"],
                "v25_shuffle_overfit": summary(artifacts, "v25")["diagnostic_class"],
                "v26_transfer_failed": summary(artifacts, "v26")["diagnostic_class"],
                "v27_bank_insufficient": summary(artifacts, "v27")["diagnostic_class"],
                "v28_transfer_role_insufficient": summary(artifacts, "v28")["diagnostic_class"],
            },
            "delayed_interface_boundary": {
                "v23_diagnostic": summary(artifacts, "v23")["diagnostic_class"],
                "v24_diagnostic": summary(artifacts, "v24")["diagnostic_class"],
                "v24_selected_template": summary(artifacts, "v24")["selection"]["selected_template"],
                "v24_selected_first_token_city_rate_at_most_0p1": criteria(artifacts, "v24")[
                    "selected_first_token_city_rate_at_most_0p1"
                ],
                "v24_selected_final_city_margin_overlap_exists": criteria(artifacts, "v24")[
                    "selected_final_city_margin_overlap_exists"
                ],
                "v25_candidate_decoupled_template_found": c25["candidate_decoupled_template_found"],
                "v25_hidden_beats_shuffle_null": c25["hidden_beats_shuffle_null"],
                "v28_any_transfer_candidate_decoupled_template": c28["any_transfer_candidate_decoupled_template"],
                "v28_transfer_repair_bank_ready": c28["transfer_repair_bank_ready"],
            },
        },
        "allowed_claim": (
            "MC006 has a matched generated behavior substrate and predecision monitor signals, but the V14-V28 "
            "route is closed as monitor-only because output/candidate geometry, failed steering, shuffle nulls, "
            "and transfer-bank insufficiency block mechanism promotion."
        ),
        "forbidden_claim": (
            "This audit does not establish a knowledge vector, does not license hidden-state steering from V16/V25, "
            "and does not permit ordinary delayed-city route repairs as promotion attempts."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for key, path in DEFAULT_PATHS.items():
        parser.add_argument(f"--{key}-artifact", type=Path, default=path)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--artifact-prefix", default="mc006_predecision_frontier_closeout_audit")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    paths = {key: getattr(args, f"{key}_artifact") for key in DEFAULT_PATHS}
    missing = [f"{key}: {path}" for key, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing source artifacts: " + "; ".join(missing))

    result = build_closeout(paths)
    if result["validation_errors"]:
        print(json.dumps(result["validation_errors"], indent=2, ensure_ascii=True))
        return 2

    if args.no_write:
        print(json.dumps(result["decision"], indent=2, ensure_ascii=True))
        return 0

    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_{stamp}.json"
    write_json(output_path, result)
    print(f"wrote {output_path}")
    print(json.dumps(result["decision"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
