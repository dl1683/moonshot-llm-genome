#!/usr/bin/env python
"""MC005 V11 focused diagnostic for the Qwen3-0.6B answer-absent weak null."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc001_qwen3_controlled_v10_head_localization import set_eager_attention
from mc005_associative_lookup_reliability_v4 import (
    SELECTED_BAND,
    SELECTED_LAYERS,
    score_rows_atlas,
)
from mc005_associative_lookup_response_marker_v9 import (
    MARKER_TEXT,
    Scenario,
    arm_is_clean,
    arm_is_weak,
    build_rows,
    evaluate_scenario,
)
from mc005_associative_lookup_source_edge import CARD_ID, RESULT_DIR


MODEL_ID = "Qwen/Qwen3-0.6B"
DIAGNOSTIC_SEEDS = [17, 23, 31, 37, 41]
ROW_COUNT = 128
LOW_MARGIN_THRESHOLD = 1.0
ARM_KEYS = (
    "source_value",
    "non_source_control_value",
    "earlier_neutral_colon",
    "final_label",
    "final_colon",
)


def classify_null_scaled(summary: dict[str, Any]) -> str:
    rows = int(summary["rows"])
    baseline_floor = int(0.75 * rows)
    if int(summary["baseline_clean_rows"]) < baseline_floor:
        return "invalid_baseline"
    arms = list(summary["arms"].values())
    if all(arm_is_clean(arm) for arm in arms):
        return "clean_null"
    if all(arm_is_weak(arm) for arm in arms):
        return "weak_null"
    return "side_effect"


def evaluate_scaled(
    scenario: Scenario,
    seed: int,
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    arms: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    summary = evaluate_scenario(scenario, seed, rows, baseline, arms)
    summary["label"] = classify_null_scaled(summary)
    summary["baseline_floor"] = int(0.75 * int(summary["rows"]))
    return summary


def flip_label(before: bool, after: bool) -> str:
    if before and not after:
        return "target_to_distractor"
    if (not before) and after:
        return "distractor_to_target"
    return "none"


def row_diagnostics(
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    arm: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    diagnostics = []
    for row in rows:
        row_id = str(row["id"])
        base = baseline[row_id]
        after = arm[row_id]
        baseline_margin = float(base["target_minus_distractor_margin"])
        arm_margin = float(after["target_minus_distractor_margin"])
        before_win = bool(base["target_wins"])
        after_win = bool(after["target_wins"])
        diagnostics.append(
            {
                "row_id": row_id,
                "source_id": row["source_id"],
                "target_value": row["target_value"],
                "distractor_value": row["distractor_value"],
                "control_value": row.get("control_value"),
                "source_focus_value": row.get("source_focus_value"),
                "baseline_margin": baseline_margin,
                "arm_margin": arm_margin,
                "delta": arm_margin - baseline_margin,
                "baseline_abs_margin": abs(baseline_margin),
                "baseline_target_wins": before_win,
                "arm_target_wins": after_win,
                "baseline_greedy_label": base["greedy_label"],
                "arm_greedy_label": after["greedy_label"],
                "flip": flip_label(before_win, after_win),
            }
        )
    return diagnostics


def summarize_row_diagnostics(per_row: list[dict[str, Any]]) -> dict[str, Any]:
    flip_rows = [row for row in per_row if row["flip"] != "none"]
    return {
        "row_count": len(per_row),
        "flip_count": len(flip_rows),
        "flip_type_counts": dict(Counter(row["flip"] for row in per_row)),
        "low_margin_flip_count": sum(
            float(row["baseline_abs_margin"]) <= LOW_MARGIN_THRESHOLD for row in flip_rows
        ),
        "max_abs_baseline_margin_flip": max(
            (float(row["baseline_abs_margin"]) for row in flip_rows),
            default=None,
        ),
        "flip_rows": flip_rows,
    }


def summarize_suite(
    full_summaries: list[dict[str, Any]],
    first32_summaries: list[dict[str, Any]],
    first32_diagnostics: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    full_label_counts = dict(Counter(str(summary["label"]) for summary in full_summaries))
    first32_label_counts = dict(Counter(str(summary["label"]) for summary in first32_summaries))
    full_all_clean = all(summary["label"] == "clean_null" for summary in full_summaries)
    seed23_summary = next(summary for summary in first32_summaries if int(summary["scenario"]["seed"]) == 23)
    seed23_control = seed23_summary["arms"]["non_source_control_value"]
    seed23_v10_pattern = (
        seed23_summary["label"] == "weak_null"
        and int(seed23_control["target_win_loss"]) <= -2
        and abs(float(seed23_control["mean_delta"])) <= 0.50
    )
    seed23_control_rows = first32_diagnostics[23]["non_source_control_value"]
    seed23_flips = seed23_control_rows["flip_rows"]
    seed23_all_flips_low_margin = bool(seed23_flips) and all(
        float(row["baseline_abs_margin"]) <= LOW_MARGIN_THRESHOLD for row in seed23_flips
    )

    if not seed23_v10_pattern:
        diagnostic_class = "v10_not_reproduced"
    elif full_all_clean and seed23_all_flips_low_margin:
        diagnostic_class = "sample_fragile_low_margin"
    elif full_all_clean:
        diagnostic_class = "sample_fragile"
    else:
        diagnostic_class = "persistent_boundary"

    return {
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "marker": MARKER_TEXT,
        "seeds": [summary["scenario"]["seed"] for summary in full_summaries],
        "row_count_per_seed": int(full_summaries[0]["rows"]) if full_summaries else 0,
        "first32_label_counts": first32_label_counts,
        "full_label_counts": full_label_counts,
        "full_all_clean": full_all_clean,
        "seed23_first32_v10_pattern_reproduced": seed23_v10_pattern,
        "seed23_first32_non_source_control_flip_count": len(seed23_flips),
        "seed23_first32_non_source_control_all_flips_low_margin": seed23_all_flips_low_margin,
        "low_margin_threshold": LOW_MARGIN_THRESHOLD,
        "diagnostic_class": diagnostic_class,
    }


def parse_seeds(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_0p6b_response_marker_v11_weak_null")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--row-count", type=int, default=ROW_COUNT)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DIAGNOSTIC_SEEDS))
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    scenario = Scenario(
        "answer_absent_response_null",
        "null",
        5,
        ARM_KEYS,
        row_count=args.row_count,
    )

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    set_eager_attention(model)
    model.eval()

    started = time.time()
    full_summaries = []
    first32_summaries = []
    all_rows = []
    row_diagnostic_summaries: dict[int, dict[str, dict[str, Any]]] = {}
    first32_row_diagnostic_summaries: dict[int, dict[str, dict[str, Any]]] = {}
    row_diagnostic_details: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for seed in seeds:
        rows = build_rows(tokenizer, scenario, seed, "mc005_v11")
        all_rows.extend(rows)
        baseline = score_rows_atlas(rows, tokenizer, model, args.batch_size)
        arms = {
            arm_key: score_rows_atlas(rows, tokenizer, model, args.batch_size, arm_key)
            for arm_key in ARM_KEYS
        }
        full_summary = evaluate_scaled(scenario, seed, rows, baseline, arms)
        first32_rows = rows[:32]
        first32_summary = evaluate_scenario(scenario, seed, first32_rows, baseline, arms)
        full_summaries.append(full_summary)
        first32_summaries.append(first32_summary)

        row_diagnostic_summaries[seed] = {}
        first32_row_diagnostic_summaries[seed] = {}
        row_diagnostic_details[seed] = {}
        for arm_key, arm_features in arms.items():
            per_row = row_diagnostics(rows, baseline, arm_features)
            row_diagnostic_details[seed][arm_key] = per_row
            row_diagnostic_summaries[seed][arm_key] = summarize_row_diagnostics(per_row)
            first32_row_diagnostic_summaries[seed][arm_key] = summarize_row_diagnostics(per_row[:32])

        full_bits = " ".join(
            f"{key}={full_summary['arms'][key]['mean_delta']:.4f}/"
            f"{full_summary['arms'][key]['target_win_loss']}"
            for key in ARM_KEYS
        )
        first32_bits = " ".join(
            f"{key}={first32_summary['arms'][key]['mean_delta']:.4f}/"
            f"{first32_summary['arms'][key]['target_win_loss']}"
            for key in ARM_KEYS
        )
        print(
            f"[v11 seed={seed} full128] label={full_summary['label']} "
            f"clean={full_summary['baseline_clean_rows']}/{full_summary['rows']} {full_bits}"
        )
        print(
            f"[v11 seed={seed} first32] label={first32_summary['label']} "
            f"clean={first32_summary['baseline_clean_rows']}/{first32_summary['rows']} {first32_bits}"
        )

    suite_summary = summarize_suite(full_summaries, first32_summaries, first32_row_diagnostic_summaries)
    elapsed = time.time() - started
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_response_marker_v11_weak_null_diagnostic",
        "model_id": args.model_id,
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "elapsed_s": elapsed,
        "summary": suite_summary,
        "full_summaries": full_summaries,
        "first32_summaries": first32_summaries,
        "row_diagnostic_summaries": row_diagnostic_summaries,
        "first32_row_diagnostic_summaries": first32_row_diagnostic_summaries,
        "row_diagnostic_details": row_diagnostic_details,
        "rows": all_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)
    print(json.dumps({**suite_summary, "output_path": str(output_path)}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
