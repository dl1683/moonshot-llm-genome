#!/usr/bin/env python
"""MC005 V14 focused Qwen3-1.7B pair16 answer-absent null diagnostic."""

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
from mc005_associative_lookup_response_marker_v11_weak_null_diagnostic import (
    LOW_MARGIN_THRESHOLD,
    classify_null_scaled,
    row_diagnostics,
    summarize_row_diagnostics,
)
from mc005_associative_lookup_response_marker_v9 import (
    MARKER_TEXT,
    Scenario,
    arm_is_clean,
    arm_is_weak,
    build_rows,
    evaluate_scenario,
)
from mc005_associative_lookup_source_edge import CARD_ID, MODEL_ID, RESULT_DIR


DIAGNOSTIC_SEEDS = [17, 23, 31, 37, 41]
PAIR_COUNTS = [12, 14, 16]
ROW_COUNT = 128
ARM_KEYS = (
    "source_value",
    "non_source_control_value",
    "earlier_neutral_colon",
    "final_label",
    "final_colon",
)


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def arm_label(arm: dict[str, Any]) -> str:
    if arm_is_clean(arm):
        return "clean"
    if arm_is_weak(arm):
        return "weak"
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
    for arm in summary["arms"].values():
        arm["arm_label"] = arm_label(arm)
    return summary


def quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = pos - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def summarize_delta_distribution(per_row: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = sorted(float(row["delta"]) for row in per_row)
    abs_deltas = sorted(abs(value) for value in deltas)
    baseline_abs_margins = sorted(float(row["baseline_abs_margin"]) for row in per_row)
    if not deltas:
        return {
            "row_count": 0,
            "mean_delta": None,
            "min_delta": None,
            "max_delta": None,
        }
    return {
        "row_count": len(deltas),
        "mean_delta": sum(deltas) / len(deltas),
        "min_delta": min(deltas),
        "q05_delta": quantile(deltas, 0.05),
        "q25_delta": quantile(deltas, 0.25),
        "q50_delta": quantile(deltas, 0.50),
        "q75_delta": quantile(deltas, 0.75),
        "q95_delta": quantile(deltas, 0.95),
        "max_delta": max(deltas),
        "max_abs_delta": max(abs_deltas),
        "abs_delta_gt_0p50_count": sum(value > 0.50 for value in abs_deltas),
        "abs_delta_gt_1p00_count": sum(value > 1.00 for value in abs_deltas),
        "baseline_abs_margin_q05": quantile(baseline_abs_margins, 0.05),
        "baseline_abs_margin_q50": quantile(baseline_abs_margins, 0.50),
        "baseline_abs_margin_q95": quantile(baseline_abs_margins, 0.95),
        "low_margin_row_count": sum(value <= LOW_MARGIN_THRESHOLD for value in baseline_abs_margins),
    }


def nonclean_arms(summary: dict[str, Any]) -> list[str]:
    return [
        key
        for key, arm in summary["arms"].items()
        if str(arm.get("arm_label")) != "clean"
    ]


def classify_diagnostic(summaries: list[dict[str, Any]]) -> str:
    if any(summary["label"] == "invalid_baseline" for summary in summaries):
        return "invalid_baseline"
    if all(summary["label"] == "clean_null" for summary in summaries):
        return "all_clean_sample_fragile"

    nonclean = [summary for summary in summaries if summary["label"] != "clean_null"]
    nonclean_arm_keys = {
        arm_key
        for summary in nonclean
        for arm_key in nonclean_arms(summary)
    }
    nonclean_pair_counts = {int(summary["scenario"]["pair_count"]) for summary in nonclean}
    if nonclean_arm_keys == {"final_label"}:
        if 12 in nonclean_pair_counts:
            return "broader_answer_absent_boundary"
        if 14 in nonclean_pair_counts:
            return "length_gradient_final_label_boundary"
        if nonclean_pair_counts == {16}:
            return "pair16_specific_final_label_boundary"
    return "broader_answer_absent_boundary"


def summarize_suite(
    summaries: list[dict[str, Any]],
    row_diagnostic_summaries: dict[int, dict[int, dict[str, dict[str, Any]]]],
    final_label_distributions: dict[int, dict[int, dict[str, Any]]],
    seeds: list[int],
    pair_counts: list[int],
    row_count: int,
    model_id: str,
) -> dict[str, Any]:
    label_counts = dict(Counter(str(summary["label"]) for summary in summaries))
    pair_label_counts: dict[str, dict[str, int]] = {}
    pair_clean_seed_counts: dict[str, int] = {str(pair_count): 0 for pair_count in pair_counts}
    arm_nonclean_counts: dict[str, dict[str, int]] = {}
    final_label_mean_delta_by_pair_seed: dict[str, dict[str, float]] = {}

    for summary in summaries:
        pair_count = str(summary["scenario"]["pair_count"])
        label = str(summary["label"])
        pair_label_counts.setdefault(pair_count, {})
        pair_label_counts[pair_count][label] = pair_label_counts[pair_count].get(label, 0) + 1
        if label == "clean_null":
            pair_clean_seed_counts[pair_count] += 1
        arm_nonclean_counts.setdefault(pair_count, {})
        for arm_key in nonclean_arms(summary):
            arm_nonclean_counts[pair_count][arm_key] = arm_nonclean_counts[pair_count].get(arm_key, 0) + 1
        seed = str(summary["scenario"]["seed"])
        final_label_mean_delta_by_pair_seed.setdefault(pair_count, {})
        final_label_mean_delta_by_pair_seed[pair_count][seed] = float(
            summary["arms"]["final_label"]["mean_delta"]
        )

    passed = all(summary["label"] == "clean_null" for summary in summaries)
    return {
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "marker": MARKER_TEXT,
        "model_id": model_id,
        "seeds": seeds,
        "pair_counts": pair_counts,
        "row_count_per_pair_seed": row_count,
        "scenario_count": len(summaries),
        "label_counts": label_counts,
        "pair_label_counts": pair_label_counts,
        "pair_clean_seed_counts": pair_clean_seed_counts,
        "arm_nonclean_counts": arm_nonclean_counts,
        "final_label_mean_delta_by_pair_seed": final_label_mean_delta_by_pair_seed,
        "final_label_distributions": final_label_distributions,
        "row_diagnostic_summaries": row_diagnostic_summaries,
        "low_margin_threshold": LOW_MARGIN_THRESHOLD,
        "all_answer_absent_nulls_clean": passed,
        "passed": passed,
        "diagnostic_class": classify_diagnostic(summaries),
    }


def make_scenario(pair_count: int, row_count: int) -> Scenario:
    return Scenario(
        f"answer_absent_pair{pair_count}_response_null",
        "null",
        pair_count,
        ARM_KEYS,
        row_count=row_count,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_response_marker_v14_pair16_null")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--row-count", type=int, default=ROW_COUNT)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DIAGNOSTIC_SEEDS))
    parser.add_argument("--pair-counts", default=",".join(str(pair_count) for pair_count in PAIR_COUNTS))
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    seeds = parse_ints(args.seeds)
    pair_counts = parse_ints(args.pair_counts)

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
    summaries = []
    all_rows = []
    row_diagnostic_summaries: dict[int, dict[int, dict[str, dict[str, Any]]]] = {}
    row_diagnostic_details: dict[int, dict[int, dict[str, list[dict[str, Any]]]]] = {}
    final_label_distributions: dict[int, dict[int, dict[str, Any]]] = {}

    for pair_count in pair_counts:
        row_diagnostic_summaries[pair_count] = {}
        row_diagnostic_details[pair_count] = {}
        final_label_distributions[pair_count] = {}
        scenario = make_scenario(pair_count, args.row_count)
        for seed in seeds:
            rows = build_rows(tokenizer, scenario, seed, "mc005_v14")
            for row in rows:
                row["split"] = "response_marker_v14_pair16_null_diagnostic"
            all_rows.extend(rows)
            baseline = score_rows_atlas(rows, tokenizer, model, args.batch_size)
            arms = {
                arm_key: score_rows_atlas(rows, tokenizer, model, args.batch_size, arm_key)
                for arm_key in ARM_KEYS
            }
            summary = evaluate_scaled(scenario, seed, rows, baseline, arms)
            summaries.append(summary)

            row_diagnostic_summaries[pair_count][seed] = {}
            row_diagnostic_details[pair_count][seed] = {}
            for arm_key, arm_features in arms.items():
                per_row = row_diagnostics(rows, baseline, arm_features)
                row_diagnostic_details[pair_count][seed][arm_key] = per_row
                row_diagnostic_summaries[pair_count][seed][arm_key] = summarize_row_diagnostics(per_row)
                if arm_key == "final_label":
                    final_label_distributions[pair_count][seed] = summarize_delta_distribution(per_row)

            arm_bits = " ".join(
                f"{key}={summary['arms'][key]['mean_delta']:.4f}/"
                f"{summary['arms'][key]['target_win_loss']}/"
                f"{summary['arms'][key]['arm_label']}"
                for key in ARM_KEYS
            )
            print(
                f"[v14 pair={pair_count} seed={seed} full{len(rows)}] "
                f"label={summary['label']} clean={summary['baseline_clean_rows']}/"
                f"{summary['rows']} {arm_bits}"
            )

    suite_summary = summarize_suite(
        summaries,
        row_diagnostic_summaries,
        final_label_distributions,
        seeds,
        pair_counts,
        args.row_count,
        args.model_id,
    )
    elapsed = time.time() - started
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_response_marker_v14_pair16_null_diagnostic",
        "model_id": args.model_id,
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "elapsed_s": elapsed,
        "summary": suite_summary,
        "scenario_summaries": summaries,
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
