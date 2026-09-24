#!/usr/bin/env python
"""Executable closeout audit for the MC005 write-replacement route."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "cards" / "MC005"

DEFAULT_V27_PATH = RESULT_DIR / "mc005_qwen3_1p7b_response_marker_v27_signature_intervention_20260630T212004.json"
DEFAULT_V28_PATH = RESULT_DIR / "mc005_qwen3_1p7b_response_marker_v28_donor_replacement_20260630T213704.json"
DEFAULT_V29_PATH = RESULT_DIR / "mc005_qwen3_1p7b_response_marker_v29_attention_write_replacement_20260630T214645.json"
DEFAULT_V30_PATH = RESULT_DIR / "mc005_qwen3_1p7b_response_marker_v30_write_null_sweep_20260630T215937.json"
DEFAULT_V31_PATH = RESULT_DIR / "mc005_qwen3_1p7b_response_marker_v31_margin_boundary_20260630T220916.json"


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


def validate_v27(v27: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    summary = v27.get("summary", {})
    criteria = summary.get("criteria", {})
    require_equal(errors, "V27 run_type", v27.get("run_type"), "associative_lookup_response_marker_v27_signature_intervention")
    require_equal(errors, "V27 diagnostic_class", summary.get("diagnostic_class"), "plus_target_no_row_effect")
    require_bool(errors, "V27 passed", summary.get("passed"), False)
    require_bool(errors, "V27 source_signature_valid", criteria.get("source_signature_valid"), True)
    require_bool(errors, "V27 base_holdout_parent_valid", criteria.get("base_holdout_parent_valid"), True)
    require_bool(
        errors,
        "V27 plus_target_all_three_fraction_gain_at_least_0p10",
        criteria.get("plus_target_all_three_fraction_gain_at_least_0p10"),
        False,
    )
    require_bool(errors, "V27 residual_nulls_clean", criteria.get("residual_nulls_clean"), True)
    return errors


def validate_v28(v28: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    summary = v28.get("summary", {})
    criteria = summary.get("criteria", {})
    require_equal(errors, "V28 run_type", v28.get("run_type"), "associative_lookup_response_marker_v28_donor_replacement")
    require_equal(errors, "V28 diagnostic_class", summary.get("diagnostic_class"), "source_control_failed")
    require_bool(errors, "V28 passed", summary.get("passed"), False)
    require_bool(errors, "V28 source_artifacts_valid", criteria.get("source_artifacts_valid"), True)
    require_bool(errors, "V28 source_signature_valid", criteria.get("source_signature_valid"), True)
    require_bool(
        errors,
        "V28 positive_target_all_three_fraction_gain_at_least_0p10",
        criteria.get("positive_target_all_three_fraction_gain_at_least_0p10"),
        True,
    )
    require_bool(errors, "V28 positive_parent_source_controls_pass", criteria.get("positive_parent_source_controls_pass"), False)
    require_bool(errors, "V28 replacement_nulls_clean", criteria.get("replacement_nulls_clean"), False)
    return errors


def validate_v29(v29: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    summary = v29.get("summary", {})
    criteria = summary.get("criteria", {})
    recovery = summary.get("recovery", {})
    require_equal(errors, "V29 run_type", v29.get("run_type"), "associative_lookup_response_marker_v29_attention_write_replacement")
    require_equal(errors, "V29 diagnostic_class", summary.get("diagnostic_class"), "null_failed")
    require_bool(errors, "V29 passed", summary.get("passed"), False)
    require_bool(errors, "V29 direct_source_controls_pass", criteria.get("direct_source_controls_pass"), True)
    require_bool(errors, "V29 write_source_controls_pass", criteria.get("write_source_controls_pass"), True)
    require_bool(errors, "V29 write_nulls_clean", criteria.get("write_nulls_clean"), False)
    require_equal(errors, "V29 target_write_delta_recovery_vs_direct", recovery.get("target_write_delta_recovery_vs_direct"), 1.0)
    require_equal(
        errors,
        "V29 target_write_win_loss_recovery_vs_direct",
        recovery.get("target_write_win_loss_recovery_vs_direct"),
        1.0,
    )
    return errors


def validate_v30(v30: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    summary = v30.get("summary", {})
    criteria = summary.get("criteria", {})
    require_equal(errors, "V30 run_type", v30.get("run_type"), "associative_lookup_response_marker_v30_write_null_sweep")
    require_equal(errors, "V30 diagnostic_class", summary.get("diagnostic_class"), "fresh_write_null_failed")
    require_bool(errors, "V30 passed", summary.get("passed"), False)
    require_bool(errors, "V30 source_artifact_valid", criteria.get("source_artifact_valid"), True)
    require_bool(errors, "V30 v29_artifact_valid", criteria.get("v29_artifact_valid"), True)
    require_bool(errors, "V30 v29_seed251_failure_reproduced", criteria.get("v29_seed251_failure_reproduced"), True)
    require_bool(errors, "V30 fresh_no_large_mean_delta", criteria.get("fresh_no_large_mean_delta"), True)
    require_bool(errors, "V30 fresh_no_target_win_change", criteria.get("fresh_no_target_win_change"), False)
    return errors


def validate_v31(v31: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    summary = v31.get("summary", {})
    criteria = summary.get("criteria", {})
    require_equal(errors, "V31 run_type", v31.get("run_type"), "associative_lookup_response_marker_v31_margin_boundary")
    require_equal(errors, "V31 diagnostic_class", summary.get("diagnostic_class"), "null_boundary_broad")
    require_bool(errors, "V31 passed", summary.get("passed"), False)
    require_bool(errors, "V31 source_artifacts_valid", criteria.get("source_artifacts_valid"), True)
    require_bool(errors, "V31 v31_lookup_target_effect_reproduced", criteria.get("v31_lookup_target_effect_reproduced"), True)
    require_bool(errors, "V31 lookup_loss_rows_baseline_margin_ge_2", criteria.get("lookup_loss_rows_baseline_margin_ge_2"), True)
    require_bool(errors, "V31 v31_fresh_null_mean_deltas_within_0p25", criteria.get("v31_fresh_null_mean_deltas_within_0p25"), True)
    require_bool(errors, "V31 v30_imported_flips_abs_margin_le_0p5", criteria.get("v30_imported_flips_abs_margin_le_0p5"), False)
    return errors


def build_closeout(
    paths: dict[str, Path],
) -> dict[str, Any]:
    artifacts = {name: read_json(path) for name, path in paths.items()}
    validation_errors = []
    validation_errors.extend(validate_v27(artifacts["v27"]))
    validation_errors.extend(validate_v28(artifacts["v28"]))
    validation_errors.extend(validate_v29(artifacts["v29"]))
    validation_errors.extend(validate_v30(artifacts["v30"]))
    validation_errors.extend(validate_v31(artifacts["v31"]))

    summaries = {name: artifact["summary"] for name, artifact in artifacts.items()}
    v27_criteria = summaries["v27"]["criteria"]
    v28_criteria = summaries["v28"]["criteria"]
    v29_criteria = summaries["v29"]["criteria"]
    v29_recovery = summaries["v29"]["recovery"]
    v30_criteria = summaries["v30"]["criteria"]
    v31_criteria = summaries["v31"]["criteria"]
    v31_lookup = summaries["v31"]["lookup_target_write"]["target_write_summary"]

    artifact_hashes = {name: sha256_file(path) for name, path in paths.items()}
    lookup_write_effect_exact = (
        float(v29_recovery["target_write_delta_recovery_vs_direct"]) == 1.0
        and float(v29_recovery["target_write_win_loss_recovery_vs_direct"]) == 1.0
    )
    write_source_controls_pass = bool(v29_criteria["direct_source_controls_pass"] and v29_criteria["write_source_controls_pass"])
    strict_nulls_clean = bool(
        v29_criteria["write_nulls_clean"]
        and v30_criteria["fresh_no_target_win_change"]
        and int(summaries["v31"]["combined_null_flip_count"]) == 0
    )
    null_boundary_reproduced = bool(
        not v29_criteria["write_nulls_clean"]
        and not v30_criteria["fresh_no_target_win_change"]
        and int(summaries["v31"]["combined_null_flip_count"]) > 0
    )
    high_margin_lookup_separated = bool(
        v31_criteria["lookup_loss_rows_baseline_margin_ge_2"]
        and int(summaries["v31"]["lookup_loss_margin_bands"]["gt_2"]) == int(v31_lookup["target_win_loss"])
    )
    alternate_family_tested = True
    alternate_family_not_promotable = bool(
        not summaries["v27"]["passed"]
        and not summaries["v28"]["passed"]
        and v27_criteria["residual_nulls_clean"]
        and not v28_criteria["positive_parent_source_controls_pass"]
        and not v28_criteria["replacement_nulls_clean"]
    )

    promotion_gate = bool(
        not validation_errors
        and lookup_write_effect_exact
        and write_source_controls_pass
        and strict_nulls_clean
        and high_margin_lookup_separated
    )
    bound_gate = bool(
        not validation_errors
        and lookup_write_effect_exact
        and write_source_controls_pass
        and null_boundary_reproduced
        and high_margin_lookup_separated
        and alternate_family_tested
    )

    if promotion_gate:
        verdict = "promoted_mechanism_card"
        route_status = "promoted"
    elif bound_gate:
        verdict = "bounded_mechanism_card"
        route_status = "bounded_frozen_not_promoted"
    else:
        verdict = "failed_or_invalid_closeout"
        route_status = "not_closed"

    criteria = {
        "artifact_sentinels_valid": not validation_errors,
        "lookup_write_effect_exact": lookup_write_effect_exact,
        "write_source_controls_pass": write_source_controls_pass,
        "strict_answer_absent_nulls_clean": strict_nulls_clean,
        "null_boundary_reproduced": null_boundary_reproduced,
        "lookup_losses_high_margin": high_margin_lookup_separated,
        "alternative_local_intervention_family_tested": alternate_family_tested,
        "alternative_local_intervention_family_not_promotable": alternate_family_not_promotable,
        "promotion_gate_passed": promotion_gate,
        "bounded_gate_passed": bound_gate,
    }
    decision_rules = {
        "promotion_rule_result": (
            "failed: lookup mediation is exact and source controls pass, but strict answer-absent null locality fails"
        ),
        "bound_rule_result": (
            "passed: high-margin lookup mediation remains exact while rare low-to-moderate-margin null flips persist"
        ),
        "kill_rule_result": (
            "kill exact write-replacement promotion attempts unless a materially new intervention family is preregistered"
        ),
        "containment_rule_result": (
            "allowed claim is source-visible associative lookup mediation in Qwen3-1.7B under the tested Response contract"
        ),
        "export_rule_result": (
            "export NULL_ROW_LOW_MARGIN_FLIP, MODEL_SIZE_NULL_FRAGILITY, and ATTENTION_WRITE_MEDIATION_BOUNDED"
        ),
    }
    return {
        "schema_version": 1,
        "run_type": "mc005_write_replacement_closeout_audit",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "card_id": "MC005",
        "model_id": summaries["v31"]["model_id"],
        "source_paths": {name: rel(path) for name, path in paths.items()},
        "source_sha256": artifact_hashes,
        "validation_errors": validation_errors,
        "criteria": criteria,
        "decision": {
            "verdict": verdict,
            "route_status": route_status,
            "hidden_state_or_new_intervention_allowed": False,
            "full_promotion_allowed": promotion_gate,
            "bounded_mechanism_preserved": bound_gate,
            "same_write_route_closed": bool(bound_gate and not promotion_gate),
        },
        "decision_rules": decision_rules,
        "evidence": {
            "alternative_intervention_family": {
                "v27_additive_residual": {
                    "diagnostic_class": summaries["v27"]["diagnostic_class"],
                    "passed": summaries["v27"]["passed"],
                    "residual_nulls_clean": v27_criteria["residual_nulls_clean"],
                    "plus_target_all_three_fraction_gain_at_least_0p10": v27_criteria[
                        "plus_target_all_three_fraction_gain_at_least_0p10"
                    ],
                },
                "v28_donor_replacement": {
                    "diagnostic_class": summaries["v28"]["diagnostic_class"],
                    "passed": summaries["v28"]["passed"],
                    "positive_target_all_three_fraction_gain_at_least_0p10": v28_criteria[
                        "positive_target_all_three_fraction_gain_at_least_0p10"
                    ],
                    "positive_parent_source_controls_pass": v28_criteria["positive_parent_source_controls_pass"],
                    "replacement_nulls_clean": v28_criteria["replacement_nulls_clean"],
                },
            },
            "write_replacement_family": {
                "v29_attention_write": {
                    "diagnostic_class": summaries["v29"]["diagnostic_class"],
                    "passed": summaries["v29"]["passed"],
                    "target_write_delta_recovery_vs_direct": v29_recovery["target_write_delta_recovery_vs_direct"],
                    "target_write_win_loss_recovery_vs_direct": v29_recovery[
                        "target_write_win_loss_recovery_vs_direct"
                    ],
                    "write_source_controls_pass": v29_criteria["write_source_controls_pass"],
                    "write_nulls_clean": v29_criteria["write_nulls_clean"],
                },
                "v30_null_sweep": {
                    "diagnostic_class": summaries["v30"]["diagnostic_class"],
                    "fresh_no_large_mean_delta": v30_criteria["fresh_no_large_mean_delta"],
                    "fresh_no_target_win_change": v30_criteria["fresh_no_target_win_change"],
                    "panels": {
                        panel_name: {
                            "seed_count": panel["seed_count"],
                            "row_count": panel["row_count"],
                            "clean_seed_count": panel["clean_seed_count"],
                            "changed_row_count": panel["changed_row_count"],
                        }
                        for panel_name, panel in summaries["v30"]["panels"].items()
                    },
                },
                "v31_margin_boundary": {
                    "diagnostic_class": summaries["v31"]["diagnostic_class"],
                    "lookup_target_write_mean_delta": v31_lookup["mean_delta"],
                    "lookup_target_win_loss": v31_lookup["target_win_loss"],
                    "combined_null_flip_count": summaries["v31"]["combined_null_flip_count"],
                    "combined_null_flip_margin_bands": summaries["v31"]["combined_null_flip_margin_bands"],
                    "lookup_loss_margin_bands": summaries["v31"]["lookup_loss_margin_bands"],
                    "v30_imported_flips_abs_margin_le_0p5": v31_criteria["v30_imported_flips_abs_margin_le_0p5"],
                    "v31_fresh_null_mean_deltas_within_0p25": v31_criteria[
                        "v31_fresh_null_mean_deltas_within_0p25"
                    ],
                },
            },
        },
        "allowed_claim": (
            "MC005 remains a bounded mechanism card: layers 24-26 final-query attention writes exactly mediate "
            "the tested high-margin associative lookup effect, but strict answer-absent null locality blocks promotion."
        ),
        "forbidden_claim": (
            "This audit does not promote MC005, does not license another same-route write-replacement repair loop, "
            "and does not generalize the surface to factual knowledge or other model sizes."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v27-artifact", type=Path, default=DEFAULT_V27_PATH)
    parser.add_argument("--v28-artifact", type=Path, default=DEFAULT_V28_PATH)
    parser.add_argument("--v29-artifact", type=Path, default=DEFAULT_V29_PATH)
    parser.add_argument("--v30-artifact", type=Path, default=DEFAULT_V30_PATH)
    parser.add_argument("--v31-artifact", type=Path, default=DEFAULT_V31_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--artifact-prefix", default="mc005_write_replacement_closeout_audit")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    paths = {
        "v27": args.v27_artifact,
        "v28": args.v28_artifact,
        "v29": args.v29_artifact,
        "v30": args.v30_artifact,
        "v31": args.v31_artifact,
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
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
