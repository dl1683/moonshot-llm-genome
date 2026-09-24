"""Build the current control-surface mixture law.

The comparison artifact records raw atlas counts. This layer turns those counts
into the project's main genome-level object: which rows are prompt-visible,
output-visible, source-dependent, internally causal, monitor-only, null-limited,
or blocked before hidden-state work is justified.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from control_surface_artifacts import ATLAS_PATH, ROOT, load_json
from control_surface_bridge_ladder import BRIDGE_LADDER_PATH
from control_surface_comparison import COMPARISON_PATH
from control_surface_smoke_diagnostics import SMOKE_DIAGNOSTICS_PATH


MIXTURE_LAW_PATH = ROOT / "data" / "control_surface_mixture_law.json"
MIXTURE_LAW_REPORT_PATH = ROOT / "research" / "28_CONTROL_SURFACE_MIXTURE_LAW.md"

STRONG_VALUES = {"high", "dominant", "behavior_only", "bounded", "failed"}
PROMPT_DIAGNOSTICS = {
    "PROMPT_FORMAT_CONFUND",
    "REQUESTED_MODE_CONFUND",
    "PROMPT_REWRITE_EQUIVALENCE",
    "PROMPT_AUTHORITY_DIAL",
    "PROMPT_CONTRACT_PARSEABILITY",
    "RELIABILITY_PROMPT_CHANNEL_VISIBLE",
    "DERIVED_CODE_TYPED_SLOT_PROMPT_VISIBLE",
}
OUTPUT_DIAGNOSTICS = {
    "OUTPUT_MARGIN_CONFUND",
    "CANDIDATE_SCORE_CONFUND",
    "GLOBAL_OUTPUT_CONFOUNDED_LEADTIME",
    "GLOBAL_MARGIN_SEPARATION_BLOCKS_MATCHING",
    "MARGIN_OVERLAP_TABLE_FAILED",
    "STRICT_FINAL_MARGIN_OVERLAP_ABSENT",
    "APPROXIMATE_PAIR_MATCHING_FAILED_MARGIN_BASELINES",
    "SOURCE_PATH_FINAL_MARGIN_SHADOW",
    "GREEDY_FINAL_MARGIN_SIGN_BARRIER",
    "DELAYED_INTERFACE_CANDIDATE_SCORE_VISIBLE",
}
SOURCE_DIAGNOSTICS = {
    "SOURCE_DELETION_NOT_CIRCUIT",
    "SOURCE_DECLARATION_CITY_MISMATCH",
    "SOURCE_PATH_FINAL_MARGIN_SHADOW",
    "MC005_BOUNDED_ATTENTION_WRITE_MEDIATION",
}
SIGNATURE_DIAGNOSTICS = {
    "SHUFFLED_SELECTION_OVERFIT",
    "SIGNATURE_NOT_CAUSAL",
    "CANDIDATE_DECOUPLED_SHUFFLE_OVERFIT",
    "LOCKED_COORDINATE_TRANSFER_FAILED",
}
TRANSFER_DIAGNOSTICS = {
    "LOCKED_COORDINATE_TRANSFER_FAILED",
    "EXPANDED_CANDIDATE_DECOUPLED_BANK_INSUFFICIENT",
    "TRANSFER_ROLE_REPAIR_BANK_INSUFFICIENT",
}
BRIDGE_BLOCK_DIAGNOSTICS = {
    "BEHAVIOR_SUBSTRATE_FAILED",
    "CONTRAST_ABSENT",
    "SYMBOLIC_NULL_CONTROL_FAILED",
    "SYMBOLIC_CONFLICT_PARSEABILITY_FAILED",
    "SYMBOLIC_CONFLICT_CONTRAST_WEAK",
    "DERIVED_CODE_CONTROL_CONFLICT_TRADEOFF",
    "TWO_HOP_SYNTHETIC_LOOKUP_FAILED",
    "TWO_HOP_REAL_MEMORY_CONTROL_FAILED",
    "TWO_HOP_CONFLICT_CONTRAST_ABSENT",
    "NUMERIC_CONFLICT_CONTRAST_ABSENT",
    "STATUS_CHANNEL_ABLATION_COLLAPSED_CONTRAST",
    "CALIBRATION_INFERENCE_CONFLICT_COLLAPSED",
    "PARITY_GATE_NOT_FOLLOWED",
    "ALPHABET_GATE_LOCAL_COLLAPSE",
    "FEATURE_LABEL_GATE_DID_NOT_RESCUE",
}
NULL_DIAGNOSTICS = {
    "NULL_ROW_LOW_MARGIN_FLIP",
    "MODEL_SIZE_NULL_FRAGILITY",
    "SYMBOLIC_NULL_CONTROL_FAILED",
}


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


def is_strong(value: str) -> bool:
    return value in STRONG_VALUES


def has_any(row: dict[str, Any], diagnostics: set[str]) -> bool:
    return bool(set(row.get("diagnostics", [])) & diagnostics)


def row_pressure_classes(row: dict[str, Any]) -> list[str]:
    mixture = row["mixture_profile"]
    diagnostics = set(row["diagnostics"])
    classes: list[str] = []

    if is_strong(mixture["prompt_authority"]) or is_strong(mixture["prompt_format"]) or diagnostics & PROMPT_DIAGNOSTICS:
        classes.append("prompt_contract_visible")
    if is_strong(mixture["output_geometry"]) or diagnostics & OUTPUT_DIAGNOSTICS:
        classes.append("output_geometry_visible")
    if is_strong(mixture["source_token_dependence"]) or diagnostics & SOURCE_DIAGNOSTICS:
        classes.append("source_or_prompt_token_dependent")
    if (
        row["lead_time"]["state"] in {"lead_time_monitor_only", "lead_time_output_shadow"}
        or mixture["lead_time_internal_signal"] in {"medium", "high"}
    ):
        classes.append("internal_monitor_present")
    if row["verdict"]["class"] in {"promoted_mechanism_card", "bounded_mechanism_card"} or mixture["causal_control"] == "bounded":
        classes.append("internal_causal_surface")
    if row["intervention"]["state"] == "failed" or diagnostics & SIGNATURE_DIAGNOSTICS:
        classes.append("signature_or_intervention_failed")
    if (
        row["behavior_gate"].startswith("failed")
        or "behavior_gate_failed" in row["behavior_gate"]
        or diagnostics & BRIDGE_BLOCK_DIAGNOSTICS
    ):
        classes.append("behavior_or_bridge_substrate_blocked")
    if mixture["null_locality"] in {"bounded", "failed"} or diagnostics & NULL_DIAGNOSTICS:
        classes.append("null_boundary_or_locality_limited")
    if mixture["transfer"] in {"failed", "low"} or diagnostics & TRANSFER_DIAGNOSTICS:
        classes.append("transfer_unproven_or_failed")

    return classes


def primary_blocker(row: dict[str, Any]) -> str:
    diagnostics = set(row["diagnostics"])
    if row["verdict"]["class"] == "promoted_mechanism_card":
        return "promoted_mechanism"
    if row["verdict"]["class"] == "bounded_mechanism_card":
        return "bounded_internal_causal_with_null_boundary"
    if "RELIABILITY_PROMPT_CHANNEL_VISIBLE" in diagnostics:
        return "prompt_visible_positive_control"
    if (
        row["behavior_gate"].startswith("failed")
        or "behavior_gate_failed" in row["behavior_gate"]
        or diagnostics & BRIDGE_BLOCK_DIAGNOSTICS
    ):
        return "behavior_substrate_or_bridge_blocked"
    if diagnostics & OUTPUT_DIAGNOSTICS:
        return "output_geometry_shadow"
    if row["intervention"]["state"] == "failed" or diagnostics & SIGNATURE_DIAGNOSTICS:
        return "signature_or_intervention_not_causal"
    if diagnostics & PROMPT_DIAGNOSTICS:
        return "prompt_contract_visible"
    return "diagnostic_open"


def class_rows(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in rows:
        for pressure_class in row_pressure_classes(row):
            result.setdefault(pressure_class, []).append(row["id"])
    return {key: sorted(value) for key, value in sorted(result.items())}


def primary_blocker_rows(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in rows:
        result.setdefault(primary_blocker(row), []).append(row["id"])
    return {key: sorted(value) for key, value in sorted(result.items())}


def axis_strength_summary(comparison: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for axis, payload in comparison["mixture_axis_counts"].items():
        summary[axis] = {
            "counts": payload["counts"],
            "strong_or_failed_count": payload["strong_or_failed_count"],
            "strong_or_failed_ratio": payload["strong_or_failed_ratio"],
        }
    return summary


def build_validation_checks(
    atlas: dict[str, Any],
    comparison: dict[str, Any],
    bridge_ladder: dict[str, Any],
    class_map: dict[str, list[str]],
    blocker_map: dict[str, list[str]],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    row_ids = sorted(row["id"] for row in atlas["rows"])
    blocker_row_ids = sorted(row_id for rows in blocker_map.values() for row_id in rows)
    comparison_shape = comparison["current_genome_shape"]

    checks = [
        {
            "id": "primary_blockers_cover_each_atlas_row_once",
            "actual": blocker_row_ids,
            "predicate": "sorted blocker rows == sorted atlas rows",
            "passed": blocker_row_ids == row_ids,
            "why": "The mixture law must assign exactly one primary blocker to every atlas row.",
        },
        {
            "id": "prompt_contract_is_universal_current_pressure",
            "actual": summary["pressure_class_counts"].get("prompt_contract_visible", 0),
            "predicate": "== row_count",
            "passed": summary["pressure_class_counts"].get("prompt_contract_visible", 0) == summary["row_count"],
            "why": "Every current atlas row has high/dominant prompt authority or prompt-channel evidence.",
        },
        {
            "id": "zero_promoted_mechanisms_preserved",
            "actual": comparison_shape["promoted_mechanism_ratio"],
            "predicate": "== 0.0",
            "passed": comparison_shape["promoted_mechanism_ratio"] == 0.0,
            "why": "The current law must not silently promote a mechanism card.",
        },
        {
            "id": "exactly_one_bounded_internal_causal_surface",
            "actual": blocker_map.get("bounded_internal_causal_with_null_boundary", []),
            "predicate": "== ['mc005_associative_lookup']",
            "passed": blocker_map.get("bounded_internal_causal_with_null_boundary", []) == ["mc005_associative_lookup"],
            "why": "MC005 is the only bounded internal-causal surface in the current atlas.",
        },
        {
            "id": "output_visibility_exceeds_internal_causal_surfaces",
            "actual": {
                "output_geometry_visible": summary["pressure_class_counts"].get("output_geometry_visible", 0),
                "internal_causal_surface": summary["pressure_class_counts"].get("internal_causal_surface", 0),
            },
            "predicate": "output visible rows > internal causal rows",
            "passed": summary["pressure_class_counts"].get("output_geometry_visible", 0) > summary["pressure_class_counts"].get("internal_causal_surface", 0),
            "why": "The present genome shape is dominated by visible behavior surfaces, not clean causal mechanisms.",
        },
        {
            "id": "bridge_ladder_has_no_unconfounded_hidden_state_candidate",
            "actual": {
                "clean_unconfounded": bridge_ladder["summary"]["clean_unconfounded_bridge_count"],
                "hidden_state_allowed": bridge_ladder["summary"]["hidden_state_allowed_count"],
            },
            "predicate": "clean_unconfounded == 0 and hidden_state_allowed == 0",
            "passed": bridge_ladder["summary"]["clean_unconfounded_bridge_count"] == 0
            and bridge_ladder["summary"]["hidden_state_allowed_count"] == 0,
            "why": "The post-MC010 bridge program has not produced a hidden-state-ready knowledge bridge.",
        },
    ]
    return checks


def build_control_surface_mixture_law() -> dict[str, Any]:
    atlas = load_json(ATLAS_PATH)
    comparison = load_json(COMPARISON_PATH)
    smoke = load_json(SMOKE_DIAGNOSTICS_PATH)
    bridge_ladder = load_json(BRIDGE_LADDER_PATH)
    rows = atlas["rows"]
    class_map = class_rows(rows)
    blocker_map = primary_blocker_rows(rows)
    pressure_counts = {key: len(value) for key, value in class_map.items()}
    blocker_counts = {key: len(value) for key, value in blocker_map.items()}
    row_count = len(rows)
    pressure_ratios = {
        key: ratio(value, row_count)
        for key, value in sorted(pressure_counts.items())
    }
    blocker_ratios = {
        key: ratio(value, row_count)
        for key, value in sorted(blocker_counts.items())
    }
    axis_summary = axis_strength_summary(comparison)

    row_profiles = [
        {
            "id": row["id"],
            "family": row["family"],
            "primary_blocker": primary_blocker(row),
            "pressure_classes": row_pressure_classes(row),
            "behavior_gate": row["behavior_gate"],
            "verdict": row["verdict"]["class"],
            "lead_time": row["lead_time"]["state"],
            "intervention": row["intervention"]["state"],
            "mixture_profile": row["mixture_profile"],
            "diagnostics": row["diagnostics"],
        }
        for row in rows
    ]

    summary = {
        "row_count": row_count,
        "pressure_class_counts": pressure_counts,
        "pressure_class_ratios": pressure_ratios,
        "primary_blocker_counts": blocker_counts,
        "primary_blocker_ratios": blocker_ratios,
        "promoted_mechanism_count": sum(1 for row in rows if row["verdict"]["class"] == "promoted_mechanism_card"),
        "bounded_internal_causal_count": len(blocker_map.get("bounded_internal_causal_with_null_boundary", [])),
        "bridge_ladder_rung_count": bridge_ladder["summary"]["rung_count"],
        "bridge_clean_unconfounded_count": bridge_ladder["summary"]["clean_unconfounded_bridge_count"],
        "smoke_hidden_state_allowed_count": smoke["summary"]["hidden_state_allowed_count"],
        "dominant_axis_ratios": {
            "prompt_authority": axis_summary["prompt_authority"]["strong_or_failed_ratio"],
            "output_geometry": axis_summary["output_geometry"]["strong_or_failed_ratio"],
            "source_token_dependence": axis_summary["source_token_dependence"]["strong_or_failed_ratio"],
            "local_internal_path": axis_summary["local_internal_path"]["strong_or_failed_ratio"],
            "causal_control": axis_summary["causal_control"]["strong_or_failed_ratio"],
            "transfer": axis_summary["transfer"]["strong_or_failed_ratio"],
        },
    }
    checks = build_validation_checks(
        atlas,
        comparison,
        bridge_ladder,
        class_map,
        blocker_map,
        summary,
    )
    return {
        "schema_version": 1,
        "updated_at": atlas.get("updated_at"),
        "purpose": (
            "Quantify the current control-surface mixture: where behavior lives "
            "across prompt contracts, output geometry, source tokens, internal "
            "monitors, bounded causal surfaces, null boundaries, and bridge failures."
        ),
        "sources": {
            "atlas": rel(ATLAS_PATH),
            "comparison": rel(COMPARISON_PATH),
            "smoke_diagnostics": rel(SMOKE_DIAGNOSTICS_PATH),
            "bridge_ladder": rel(BRIDGE_LADDER_PATH),
        },
        "classification_rule": {
            "pressure_classes": "Non-exclusive row memberships derived from mixture_profile, diagnostics, lead_time, intervention, and verdict fields.",
            "primary_blocker": "One row-level outcome chosen by ordered priority: promoted, bounded internal causal, prompt-visible positive control, behavior/bridge blocked, output shadow, signature/intervention failure, prompt contract, diagnostic open.",
        },
        "summary": summary,
        "axis_strength_summary": axis_summary,
        "pressure_class_rows": class_map,
        "primary_blocker_rows": blocker_map,
        "row_profiles": row_profiles,
        "validation_checks": checks,
        "allowed_claim": (
            "The current small-LLM genome map is a measured mixture distribution: "
            "prompt-contract and output/source-visible pressures dominate, MC005 is "
            "the only bounded internal-causal row, and the MC010-MC022 bridge has "
            "not produced a hidden-state-ready unconfounded knowledge substrate."
        ),
        "forbidden_claim": (
            "This artifact does not prove a general truth vector, a broad knowledge "
            "mechanism, or a deployable factual-control intervention."
        ),
    }


def validate_mixture_law(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("mixture law schema_version must be 1")
    for rel_path in payload.get("sources", {}).values():
        if not (ROOT / rel_path).exists():
            raise AssertionError(f"mixture law source missing: {rel_path}")
    row_ids = sorted(profile["id"] for profile in payload.get("row_profiles", []))
    blocker_ids = sorted(
        row_id
        for rows in payload.get("primary_blocker_rows", {}).values()
        for row_id in rows
    )
    if row_ids != blocker_ids:
        raise AssertionError("primary_blocker_rows must cover every row exactly once")
    summary = payload["summary"]
    if summary["promoted_mechanism_count"] != 0:
        raise AssertionError("mixture law must not report promoted mechanism cards")
    if summary["bounded_internal_causal_count"] != 1:
        raise AssertionError("mixture law expected exactly one bounded internal-causal row")
    failed_checks = [
        check for check in payload.get("validation_checks", []) if not check.get("passed")
    ]
    if failed_checks:
        raise AssertionError(f"mixture law checks failed: {failed_checks}")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Control-Surface Mixture Law",
        "",
        "Date: 2026-07-01",
        "",
        "Status: generated mixture-law layer implemented and validated.",
        "",
        "Machine-readable artifact:",
        "",
        "> `data/control_surface_mixture_law.json`",
        "",
        "Builder:",
        "",
        "> `code/control_surface_mixture_law.py`",
        "",
        "Commands:",
        "",
        "```powershell",
        "python code\\control_surface_mixture_law.py --write",
        "python code\\control_surface_mixture_law.py",
        "python code\\validate_control_surface_atlas.py",
        "```",
        "",
        "## Purpose",
        "",
        "The mixture law turns the atlas from a list of mechanism-card attempts",
        "into a measured distribution of where behavior currently lives. Rows can",
        "belong to multiple pressure classes, but each row also gets one primary",
        "blocker so the project can track whether future experiments change the",
        "shape of the map.",
        "",
        "## Generated Facts",
        "",
        f"- atlas rows: {summary['row_count']};",
        f"- promoted mechanism cards: {summary['promoted_mechanism_count']};",
        f"- bounded internal-causal rows: {summary['bounded_internal_causal_count']};",
        f"- bridge ladder rungs: {summary['bridge_ladder_rung_count']};",
        f"- clean unconfounded bridge rungs: {summary['bridge_clean_unconfounded_count']};",
        f"- smoke hidden-state-allowed cards: {summary['smoke_hidden_state_allowed_count']}.",
        "",
        "Strong-or-failed axis ratios:",
        "",
    ]
    for axis, value in summary["dominant_axis_ratios"].items():
        lines.append(f"- `{axis}`: {format_value(value)};")

    lines.extend(
        [
            "",
            "## Primary Blockers",
            "",
            "| Primary Blocker | Count | Ratio | Rows |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for blocker, rows in payload["primary_blocker_rows"].items():
        lines.append(
            f"| `{blocker}` | {len(rows)} | {format_value(summary['primary_blocker_ratios'][blocker])} | "
            f"{'<br>'.join(f'`{row_id}`' for row_id in rows)} |"
        )

    lines.extend(
        [
            "",
            "## Pressure Classes",
            "",
            "| Pressure Class | Count | Ratio |",
            "| --- | ---: | ---: |",
        ]
    )
    for pressure_class, rows in payload["pressure_class_rows"].items():
        lines.append(
            f"| `{pressure_class}` | {len(rows)} | "
            f"{format_value(summary['pressure_class_ratios'][pressure_class])} |"
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
            "## Interpretation",
            "",
            "The central result is not that the project has found a broad knowledge",
            "mechanism. It has not. The central result is that the current behavior",
            "families distribute heavily over prompt contracts, output geometry, and",
            "source-token/path surfaces, while bounded internal causality is currently",
            "represented by one narrow MC005 row.",
            "",
            "That makes typed failure a primary datum. A result killed by output",
            "geometry, prompt visibility, bridge-substrate collapse, null locality,",
            "or transfer fragility is not just a dead mechanism claim; it is one more",
            "measurement of the mixture.",
            "",
            "## Claim Boundary",
            "",
            payload["allowed_claim"],
            "",
            payload["forbidden_claim"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write JSON and markdown artifacts")
    args = parser.parse_args()

    payload = build_control_surface_mixture_law()
    validate_mixture_law(payload)
    if args.write:
        write_json(MIXTURE_LAW_PATH, payload)
        MIXTURE_LAW_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        MIXTURE_LAW_REPORT_PATH.write_text(
            render_markdown(payload),
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "passed": True,
                "row_count": payload["summary"]["row_count"],
                "primary_blocker_counts": payload["summary"]["primary_blocker_counts"],
                "pressure_class_counts": payload["summary"]["pressure_class_counts"],
                "validation_check_count": len(payload["validation_checks"]),
                "output_path": rel(MIXTURE_LAW_PATH),
                "report_path": rel(MIXTURE_LAW_REPORT_PATH),
            },
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
