#!/usr/bin/env python
"""MC006 V20 offline strict-overlap selection audit for the V19 prompt bank."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from mc006_parametric_fact_override_v15_parser_normalized_signature import sha256_file


CARD_ID = "MC006"
RUN_TYPE = "parametric_fact_override_v20_strict_overlap_selection_audit"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/MC006")
DEFAULT_SOURCE_ARTIFACT = (
    RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v19_overlapping_margin_table_20260701T023921.json"
)
SOURCE_RUN_TYPE = "parametric_fact_override_v19_overlapping_margin_table"
SOURCE_DIAGNOSTIC_CLASS = "non_holdout_candidate_margin_overlap_failed"
PRIMARY_LABELS = ("true_answer", "override_answer")
MARGIN_FIELDS = ("candidate_score_margin", "final_next_token_margin")
MATCH_THRESHOLDS_Z = (0.25, 0.5, 1.0)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_source(source: dict[str, Any]) -> dict[str, Any]:
    summary = source.get("summary", {})
    records = source.get("records", [])
    criteria = {
        "expected_run_type": source.get("run_type") == SOURCE_RUN_TYPE,
        "expected_diagnostic_class": summary.get("diagnostic_class") == SOURCE_DIAGNOSTIC_CLASS,
        "source_not_signature_ready": summary.get("signature_ready") is False,
        "has_400_records": isinstance(records, list) and len(records) == 400,
        "has_template_summary": isinstance(summary.get("by_template"), dict),
    }
    return {"criteria": criteria, "passed": all(criteria.values())}


def is_binary(row: dict[str, Any]) -> bool:
    return bool(row.get("is_binary")) and row.get("selected_label") in PRIMARY_LABELS


def split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split == "all":
        return rows
    if split == "non_holdout":
        return [row for row in rows if row["split"] != "holdout"]
    return [row for row in rows if row["split"] == split]


def train_stats(rows: list[dict[str, Any]], field: str) -> tuple[float, float]:
    train = [
        float(row[field])
        for row in rows
        if row.get(field) is not None and row.get("split") != "holdout"
    ]
    if not train:
        return 0.0, 1.0
    mean = float(np.mean(train))
    std = float(np.std(train))
    if std < 1e-9:
        std = 1.0
    return mean, std


def z_value(row: dict[str, Any], field: str, mean: float, std: float) -> float:
    return (float(row[field]) - mean) / std


def class_range(
    rows: list[dict[str, Any]],
    field: str,
    stats_by_field: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    mean, std = stats_by_field[field]
    payload: dict[str, Any] = {}
    for label in PRIMARY_LABELS:
        raw_values = np.asarray(
            [float(row[field]) for row in rows if row["selected_label"] == label and row.get(field) is not None],
            dtype=np.float64,
        )
        z_values = np.asarray(
            [z_value(row, field, mean, std) for row in rows if row["selected_label"] == label and row.get(field) is not None],
            dtype=np.float64,
        )
        payload[label] = {
            "count": int(raw_values.size),
            "raw_min": float(raw_values.min()) if raw_values.size else None,
            "raw_max": float(raw_values.max()) if raw_values.size else None,
            "raw_mean": float(raw_values.mean()) if raw_values.size else None,
            "z_min": float(z_values.min()) if z_values.size else None,
            "z_max": float(z_values.max()) if z_values.size else None,
            "z_mean": float(z_values.mean()) if z_values.size else None,
        }
    true_range = payload["true_answer"]
    override_range = payload["override_answer"]
    if true_range["count"] and override_range["count"]:
        raw_overlap_low = max(float(true_range["raw_min"]), float(override_range["raw_min"]))
        raw_overlap_high = min(float(true_range["raw_max"]), float(override_range["raw_max"]))
        z_overlap_low = max(float(true_range["z_min"]), float(override_range["z_min"]))
        z_overlap_high = min(float(true_range["z_max"]), float(override_range["z_max"]))
        payload["raw_overlap_exists"] = raw_overlap_high >= raw_overlap_low
        payload["raw_overlap_width"] = (
            float(raw_overlap_high - raw_overlap_low) if raw_overlap_high >= raw_overlap_low else 0.0
        )
        payload["raw_separation_gap"] = (
            float(raw_overlap_low - raw_overlap_high) if raw_overlap_high < raw_overlap_low else 0.0
        )
        payload["z_overlap_exists"] = z_overlap_high >= z_overlap_low
        payload["z_overlap_width"] = (
            float(z_overlap_high - z_overlap_low) if z_overlap_high >= z_overlap_low else 0.0
        )
        payload["z_separation_gap"] = (
            float(z_overlap_low - z_overlap_high) if z_overlap_high < z_overlap_low else 0.0
        )
        payload["true_min_minus_override_max_z"] = float(true_range["z_min"] - override_range["z_max"])
    else:
        payload["raw_overlap_exists"] = False
        payload["raw_overlap_width"] = 0.0
        payload["raw_separation_gap"] = None
        payload["z_overlap_exists"] = False
        payload["z_overlap_width"] = 0.0
        payload["z_separation_gap"] = None
        payload["true_min_minus_override_max_z"] = None
    return payload


def joint_pair_counts(
    rows: list[dict[str, Any]],
    stats_by_field: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    true_rows = [row for row in rows if row["selected_label"] == "true_answer"]
    override_rows = [row for row in rows if row["selected_label"] == "override_answer"]
    payload: dict[str, Any] = {
        "true_rows": len(true_rows),
        "override_rows": len(override_rows),
        "possible_pairs": len(true_rows) * len(override_rows),
    }
    for threshold in MATCH_THRESHOLDS_Z:
        pairs = []
        for true_row in true_rows:
            for override_row in override_rows:
                deltas = []
                for field in MARGIN_FIELDS:
                    if true_row.get(field) is None or override_row.get(field) is None:
                        deltas = []
                        break
                    mean, std = stats_by_field[field]
                    deltas.append(abs(z_value(true_row, field, mean, std) - z_value(override_row, field, mean, std)))
                if deltas and max(deltas) <= threshold:
                    pairs.append(
                        {
                            "true_id": true_row["id"],
                            "override_id": override_row["id"],
                            "max_abs_z_delta": float(max(deltas)),
                            "candidate_abs_z_delta": float(deltas[0]),
                            "final_abs_z_delta": float(deltas[1]),
                        }
                    )
        key = f"z_le_{str(threshold).replace('.', 'p')}"
        payload[key] = {
            "pairs": len(pairs),
            "mean_max_abs_z_delta": float(np.mean([row["max_abs_z_delta"] for row in pairs])) if pairs else None,
            "example_pairs": pairs[:10],
        }
    return payload


def scope_audit(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    binary_rows = [row for row in rows if is_binary(row)]
    stats_by_field = {field: train_stats(binary_rows, field) for field in MARGIN_FIELDS}
    by_split: dict[str, Any] = {}
    for split in ("all", "non_holdout", "discovery", "calibration", "holdout"):
        scoped = split_rows(binary_rows, split)
        by_split[split] = {
            "row_count": len(scoped),
            "label_counts": dict(sorted(Counter(row["selected_label"] for row in scoped).items())),
            "margins": {
                field: class_range(scoped, field, stats_by_field)
                for field in MARGIN_FIELDS
            },
            "joint_margin_pairs": joint_pair_counts(scoped, stats_by_field),
        }
    full_gate = {
        "binary_rows_at_least_30": len(binary_rows) >= 30,
        "non_holdout_true_at_least_6": by_split["non_holdout"]["label_counts"].get("true_answer", 0) >= 6,
        "non_holdout_override_at_least_6": by_split["non_holdout"]["label_counts"].get("override_answer", 0) >= 6,
        "holdout_true_at_least_2": by_split["holdout"]["label_counts"].get("true_answer", 0) >= 2,
        "holdout_override_at_least_2": by_split["holdout"]["label_counts"].get("override_answer", 0) >= 2,
        "non_holdout_candidate_overlap": by_split["non_holdout"]["margins"]["candidate_score_margin"]["z_overlap_exists"],
        "non_holdout_final_overlap": by_split["non_holdout"]["margins"]["final_next_token_margin"]["z_overlap_exists"],
        "holdout_candidate_overlap": by_split["holdout"]["margins"]["candidate_score_margin"]["z_overlap_exists"],
        "holdout_final_overlap": by_split["holdout"]["margins"]["final_next_token_margin"]["z_overlap_exists"],
    }
    return {
        "name": name,
        "binary_row_count": len(binary_rows),
        "full_gate": full_gate,
        "full_gate_passed": all(full_gate.values()),
        "by_split": by_split,
    }


def summarize(source: dict[str, Any], source_validation: dict[str, Any]) -> dict[str, Any]:
    records = source["records"]
    templates = list(source["summary"]["by_template"].keys())
    binary_rows = [row for row in records if is_binary(row)]
    pooled = scope_audit("pooled_all_templates", records)
    selected_template = source["summary"]["selection"]["selected_template"]
    selected = scope_audit(
        f"selected_template:{selected_template}",
        [row for row in records if row["template"] == selected_template],
    )
    template_audits = {
        template: scope_audit(
            template,
            [row for row in records if row["template"] == template],
        )
        for template in templates
    }
    any_template_full_gate = any(item["full_gate_passed"] for item in template_audits.values())
    any_template_holdout_final_overlap = any(
        item["by_split"]["holdout"]["margins"]["final_next_token_margin"]["z_overlap_exists"]
        for item in template_audits.values()
    )
    any_template_non_holdout_final_overlap = any(
        item["by_split"]["non_holdout"]["margins"]["final_next_token_margin"]["z_overlap_exists"]
        for item in template_audits.values()
    )
    pooled_holdout_joint_pairs_0p5 = int(
        pooled["by_split"]["holdout"]["joint_margin_pairs"]["z_le_0p5"]["pairs"]
    )
    selected_holdout_joint_pairs_0p5 = int(
        selected["by_split"]["holdout"]["joint_margin_pairs"]["z_le_0p5"]["pairs"]
    )
    criteria = {
        "source_artifact_valid": source_validation["passed"],
        "pooled_non_holdout_candidate_overlap": pooled["by_split"]["non_holdout"]["margins"][
            "candidate_score_margin"
        ]["z_overlap_exists"],
        "pooled_non_holdout_final_overlap": pooled["by_split"]["non_holdout"]["margins"][
            "final_next_token_margin"
        ]["z_overlap_exists"],
        "pooled_holdout_candidate_overlap": pooled["by_split"]["holdout"]["margins"][
            "candidate_score_margin"
        ]["z_overlap_exists"],
        "pooled_holdout_final_overlap": pooled["by_split"]["holdout"]["margins"][
            "final_next_token_margin"
        ]["z_overlap_exists"],
        "any_single_template_full_gate_passed": any_template_full_gate,
        "any_template_non_holdout_final_overlap": any_template_non_holdout_final_overlap,
        "any_template_holdout_final_overlap": any_template_holdout_final_overlap,
        "pooled_holdout_joint_pairs_at_0p5z": pooled_holdout_joint_pairs_0p5 > 0,
        "selected_holdout_joint_pairs_at_0p5z": selected_holdout_joint_pairs_0p5 > 0,
    }
    if not source_validation["passed"]:
        diagnostic_class = "source_artifact_invalid"
    elif any_template_full_gate:
        diagnostic_class = "strict_overlap_template_candidate_found"
    elif not criteria["pooled_non_holdout_final_overlap"] and not criteria["pooled_holdout_final_overlap"]:
        diagnostic_class = "strict_final_margin_overlap_absent"
    elif not criteria["pooled_holdout_final_overlap"]:
        diagnostic_class = "holdout_final_margin_overlap_absent"
    elif criteria["pooled_holdout_joint_pairs_at_0p5z"]:
        diagnostic_class = "pair_matched_only_not_strict_overlap"
    else:
        diagnostic_class = "strict_overlap_selection_failed"
    return {
        "source_validation": source_validation,
        "source_v19_diagnostic_class": source["summary"]["diagnostic_class"],
        "selected_v19_template": selected_template,
        "binary_row_count": len(binary_rows),
        "pooled_audit": pooled,
        "selected_template_audit": selected,
        "template_audits": template_audits,
        "criteria": criteria,
        "passed": diagnostic_class == "strict_overlap_template_candidate_found",
        "signature_ready": diagnostic_class == "strict_overlap_template_candidate_found",
        "pair_matched_diagnostic_ready": criteria["pooled_holdout_joint_pairs_at_0p5z"],
        "diagnostic_class": diagnostic_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--source-artifact", type=Path, default=DEFAULT_SOURCE_ARTIFACT)
    parser.add_argument(
        "--artifact-prefix",
        default="mc006_qwen3_1p7b_parametric_fact_override_v20_strict_overlap_selection_audit",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    args = parser.parse_args()

    started = time.time()
    source_hash = sha256_file(args.source_artifact)
    source = read_json(args.source_artifact)
    source_validation = validate_source(source)
    summary = summarize(source, source_validation)
    result = {
        "card_id": args.card_id,
        "run_type": RUN_TYPE,
        "model_id": args.model_id,
        "source_artifact": str(args.source_artifact),
        "source_artifact_sha256": source_hash,
        "elapsed_s": time.time() - started,
        "summary": summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)
    compact = {
        "diagnostic_class": summary["diagnostic_class"],
        "passed": summary["passed"],
        "signature_ready": summary["signature_ready"],
        "pair_matched_diagnostic_ready": summary["pair_matched_diagnostic_ready"],
        "criteria": summary["criteria"],
        "pooled_holdout_final_margin": summary["pooled_audit"]["by_split"]["holdout"]["margins"][
            "final_next_token_margin"
        ],
        "pooled_holdout_joint_pairs_0p5": summary["pooled_audit"]["by_split"]["holdout"][
            "joint_margin_pairs"
        ]["z_le_0p5"]["pairs"],
        "selected_holdout_joint_pairs_0p5": summary["selected_template_audit"]["by_split"]["holdout"][
            "joint_margin_pairs"
        ]["z_le_0p5"]["pairs"],
        "output_path": str(output_path),
    }
    print(json.dumps(compact, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
