#!/usr/bin/env python
"""MC005 V22 row-level interaction diagnostic for layers 24-26."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc001_qwen3_controlled_v10_head_localization import set_eager_attention
from mc005_associative_lookup_reliability_v4 import SELECTED_BAND, SELECTED_LAYERS
from mc005_associative_lookup_response_marker_v15_fine_localization import (
    Candidate,
    baseline_summary,
    candidate_payload,
    score_rows_path,
    summarize_path_arm,
)
from mc005_associative_lookup_response_marker_v9 import (
    MARKER_TEXT,
    Scenario,
    arm_is_clean,
    arm_is_weak,
    build_rows,
)
from mc005_associative_lookup_source_edge import CARD_ID, MODEL_ID, RESULT_DIR


LOOKUP_SEEDS = [191, 193]
NULL_SEEDS = [197, 199]
PAIR_COUNT = 16
LOOKUP_ROW_COUNT = 128
NULL_ROW_COUNT = 96
LOOKUP_ARM_KEYS = ("target_value", "distractor_value", "random_value")
NULL_ARM_KEYS = (
    "source_value",
    "non_source_control_value",
    "earlier_neutral_colon",
    "final_label",
    "final_colon",
)

PARENT = Candidate("slice_l24_26_all", (24, 25, 26), None, False)
PAIR_PATHS = [
    Candidate("slice_l24_25_all", (24, 25), None, False),
    Candidate("slice_l24_26_all_pair", (24, 26), None, False),
    Candidate("slice_l25_26_all", (25, 26), None, False),
]
SINGLE_PATHS = [
    Candidate("single_l24_all", (24,), None, False),
    Candidate("single_l25_all", (25,), None, False),
    Candidate("single_l26_all", (26,), None, False),
]
TARGET_PATHS = [PARENT] + PAIR_PATHS + SINGLE_PATHS

PAIR_NAMES = [path.name for path in PAIR_PATHS]
SINGLE_NAMES = [path.name for path in SINGLE_PATHS]

PAIR_PLUS_SINGLE_FAMILIES = {
    "pair_l24_25_plus_single_l26": ("slice_l24_25_all", "single_l26_all"),
    "pair_l24_26_plus_single_l25": ("slice_l24_26_all_pair", "single_l25_all"),
    "pair_l25_26_plus_single_l24": ("slice_l25_26_all", "single_l24_all"),
}


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def build_lookup_rows(tokenizer: Any, seeds: list[int], row_count: int) -> list[dict[str, Any]]:
    scenario = Scenario(
        "lookup_pair16_response",
        "lookup",
        PAIR_COUNT,
        LOOKUP_ARM_KEYS,
        row_count=row_count,
    )
    rows = []
    for seed in seeds:
        seed_rows = build_rows(tokenizer, scenario, seed, "mc005_v22")
        for row in seed_rows:
            row["split"] = "lookup_row_interaction_v22"
        rows.extend(seed_rows)
    return rows


def build_null_rows(tokenizer: Any, seed: int, row_count: int) -> list[dict[str, Any]]:
    scenario = Scenario(
        "answer_absent_pair16_response_null",
        "null",
        PAIR_COUNT,
        NULL_ARM_KEYS,
        row_count=row_count,
    )
    rows = build_rows(tokenizer, scenario, seed, "mc005_v22")
    for row in rows:
        row["split"] = "answer_absent_null_holdout_v22"
    return rows


def margin(features: dict[str, Any]) -> float:
    return float(features["target_minus_distractor_margin"])


def row_delta(row_id: str, baseline: dict[str, dict[str, Any]], arm: dict[str, dict[str, Any]]) -> float:
    return margin(arm[row_id]) - margin(baseline[row_id])


def negative_effect_share(component_delta: float, parent_delta: float) -> float | None:
    if parent_delta >= -1e-12:
        return None
    return max(0.0, -component_delta) / max(1e-12, -parent_delta)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * q
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    frac = rank - lower
    return float(ordered[lower] * (1.0 - frac) + ordered[upper] * frac)


def score_target_paths(
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    raw_by_path = {}
    summary_by_path = {}
    for path in TARGET_PATHS:
        raw = score_rows_path(rows, tokenizer, model, batch_size, path, "target_value")
        raw_by_path[path.name] = raw
        summary_by_path[path.name] = {
            "candidate": candidate_payload(path, model),
            "target_value": summarize_path_arm(rows, baseline, raw),
        }
        target = summary_by_path[path.name]["target_value"]
        print(
            f"[v22 target {path.name}] "
            f"mean_delta={target['mean_delta']:.4f} win_loss={target['target_win_loss']}"
        )
    return raw_by_path, summary_by_path


def score_parent_source_controls(
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
) -> dict[str, Any]:
    arms = {}
    for arm_key in LOOKUP_ARM_KEYS:
        arm = score_rows_path(rows, tokenizer, model, batch_size, PARENT, arm_key)
        arms[arm_key] = summarize_path_arm(rows, baseline, arm)
    return {"candidate": candidate_payload(PARENT, model), "arms": arms}


def compute_row_diagnostics(
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    raw_by_path: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    diagnostics = []
    parent_raw = raw_by_path[PARENT.name]
    for row in rows:
        row_id = row["id"]
        parent_delta = row_delta(row_id, baseline, parent_raw)
        pair_deltas = {
            name: row_delta(row_id, baseline, raw_by_path[name])
            for name in PAIR_NAMES
        }
        single_deltas = {
            name: row_delta(row_id, baseline, raw_by_path[name])
            for name in SINGLE_NAMES
        }
        pair_negative_shares = {
            name: negative_effect_share(delta, parent_delta)
            for name, delta in pair_deltas.items()
        }
        singles_sum = sum(single_deltas.values())
        singles_residual = parent_delta - singles_sum
        pair_plus_single_sums = {
            family: pair_deltas[pair_name] + single_deltas[single_name]
            for family, (pair_name, single_name) in PAIR_PLUS_SINGLE_FAMILIES.items()
        }
        pair_plus_single_residuals = {
            family: parent_delta - component_sum
            for family, component_sum in pair_plus_single_sums.items()
        }
        baseline_target_wins = bool(baseline[row_id]["target_wins"])
        parent_target_wins = bool(parent_raw[row_id]["target_wins"])
        parent_effect_row = baseline_target_wins and parent_delta <= -1.0
        all_pairs_below_0p60 = (
            parent_effect_row
            and all(
                share is not None and share < 0.60
                for share in pair_negative_shares.values()
            )
        )
        all_pair_plus_single_superadditive = all(
            residual <= -0.5
            for residual in pair_plus_single_residuals.values()
        )
        all_three_margin_row = (
            parent_effect_row
            and all_pairs_below_0p60
            and singles_residual <= -1.0
            and all_pair_plus_single_superadditive
        )
        parent_flip_pair_resistant = (
            baseline_target_wins
            and not parent_target_wins
            and all(bool(raw_by_path[name][row_id]["target_wins"]) for name in PAIR_NAMES)
        )
        diagnostics.append(
            {
                "row_id": row_id,
                "seed": row["seed"],
                "baseline_margin": margin(baseline[row_id]),
                "baseline_target_wins": baseline_target_wins,
                "parent_margin": margin(parent_raw[row_id]),
                "parent_target_wins": parent_target_wins,
                "parent_delta": parent_delta,
                "pair_deltas": pair_deltas,
                "single_deltas": single_deltas,
                "pair_negative_effect_shares": pair_negative_shares,
                "singles_sum": singles_sum,
                "singles_residual": singles_residual,
                "pair_plus_single_sums": pair_plus_single_sums,
                "pair_plus_single_residuals": pair_plus_single_residuals,
                "parent_effect_row": parent_effect_row,
                "all_pairs_below_0p60": all_pairs_below_0p60,
                "all_pair_plus_single_superadditive": all_pair_plus_single_superadditive,
                "all_three_margin_row": all_three_margin_row,
                "parent_flip_pair_resistant": parent_flip_pair_resistant,
            }
        )
    return diagnostics


def summarize_row_diagnostics(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    parent_effect_rows = [row for row in diagnostics if row["parent_effect_row"]]
    all_three_rows = [row for row in diagnostics if row["all_three_margin_row"]]
    parent_flip_rows = [
        row for row in diagnostics
        if row["baseline_target_wins"] and not row["parent_target_wins"]
    ]
    pair_resistant_flip_rows = [row for row in diagnostics if row["parent_flip_pair_resistant"]]
    parent_effect_count = len(parent_effect_rows)
    all_three_count = len(all_three_rows)
    parent_effect_denominator = max(1, parent_effect_count)
    parent_effect_singles_residuals = [
        float(row["singles_residual"])
        for row in parent_effect_rows
    ]
    parent_effect_best_pair_shares = [
        max(
            float(share)
            for share in row["pair_negative_effect_shares"].values()
            if share is not None
        )
        for row in parent_effect_rows
    ]
    return {
        "row_count": len(diagnostics),
        "baseline_target_win_count": sum(1 for row in diagnostics if row["baseline_target_wins"]),
        "parent_effect_row_count": parent_effect_count,
        "all_three_margin_row_count": all_three_count,
        "all_three_margin_row_fraction_of_parent_effect_rows": all_three_count / parent_effect_denominator,
        "parent_flip_row_count": len(parent_flip_rows),
        "parent_flip_pair_resistant_row_count": len(pair_resistant_flip_rows),
        "parent_effect_singles_residual_mean": mean(parent_effect_singles_residuals),
        "parent_effect_singles_residual_median": median(parent_effect_singles_residuals),
        "parent_effect_singles_residual_p25": percentile(parent_effect_singles_residuals, 0.25),
        "parent_effect_singles_residual_p75": percentile(parent_effect_singles_residuals, 0.75),
        "parent_effect_best_pair_share_mean": mean(parent_effect_best_pair_shares),
        "parent_effect_best_pair_share_median": median(parent_effect_best_pair_shares),
        "parent_effect_best_pair_share_p90": percentile(parent_effect_best_pair_shares, 0.90),
        "all_three_margin_row_ids": [row["row_id"] for row in all_three_rows],
        "parent_flip_pair_resistant_row_ids": [row["row_id"] for row in pair_resistant_flip_rows],
    }


def classify_null(
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    arms: dict[str, Any],
) -> str:
    baseline_floor = int(0.75 * len(rows))
    baseline_clean_rows = sum(1 for row in rows if baseline[row["id"]]["target_wins"])
    if baseline_clean_rows < baseline_floor:
        return "invalid_baseline"
    arm_values = list(arms.values())
    if all(arm_is_clean(arm) for arm in arm_values):
        return "clean_null"
    if all(arm_is_weak(arm) for arm in arm_values):
        return "weak_null"
    return "side_effect"


def classify_diagnostic(criteria: dict[str, bool]) -> str:
    if all(criteria.values()):
        return "row_level_interaction_supported"
    if (
        not criteria["lookup_baseline_valid"]
        or not criteria["parent_effect"]
        or not criteria["parent_source_controls_pass"]
    ):
        return "parent_not_replicated"
    if not criteria["parent_null_holdouts_clean"]:
        return "null_failed"
    if (
        not criteria["all_three_margin_row_count_at_least_24"]
        or not criteria["all_three_margin_row_fraction_at_least_0p40"]
        or not criteria["median_singles_residual_at_most_minus_1"]
    ):
        return "mean_only_interaction"
    if not criteria["parent_flip_pair_resistant_rows_at_least_8"]:
        return "flip_support_failed"
    return "mixed_failure"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_response_marker_v22_row_interaction")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lookup-row-count", type=int, default=LOOKUP_ROW_COUNT)
    parser.add_argument("--null-row-count", type=int, default=NULL_ROW_COUNT)
    parser.add_argument("--lookup-seeds", default=",".join(str(seed) for seed in LOOKUP_SEEDS))
    parser.add_argument("--null-seeds", default=",".join(str(seed) for seed in NULL_SEEDS))
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    lookup_seeds = parse_ints(args.lookup_seeds)
    null_seeds = parse_ints(args.null_seeds)

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
    lookup_rows = build_lookup_rows(tokenizer, lookup_seeds, args.lookup_row_count)
    lookup_baseline = score_rows_path(lookup_rows, tokenizer, model, args.batch_size)
    raw_by_path, path_results = score_target_paths(
        lookup_rows,
        lookup_baseline,
        tokenizer,
        model,
        args.batch_size,
    )
    parent_controls = score_parent_source_controls(
        lookup_rows,
        lookup_baseline,
        tokenizer,
        model,
        args.batch_size,
    )
    row_diagnostics = compute_row_diagnostics(lookup_rows, lookup_baseline, raw_by_path)
    row_summary = summarize_row_diagnostics(row_diagnostics)
    print(
        "[v22 rows] "
        f"parent_effect={row_summary['parent_effect_row_count']} "
        f"all_three={row_summary['all_three_margin_row_count']} "
        f"flip_pair_resistant={row_summary['parent_flip_pair_resistant_row_count']}"
    )

    null_results = {}
    null_rows_by_seed = {}
    for seed in null_seeds:
        rows = build_null_rows(tokenizer, seed, args.null_row_count)
        null_rows_by_seed[seed] = rows
        baseline = score_rows_path(rows, tokenizer, model, args.batch_size)
        arms = {}
        for arm_key in NULL_ARM_KEYS:
            arm = score_rows_path(rows, tokenizer, model, args.batch_size, PARENT, arm_key)
            arms[arm_key] = summarize_path_arm(rows, baseline, arm)
        label = classify_null(rows, baseline, arms)
        null_results[seed] = {
            "seed": seed,
            "label": label,
            "baseline": baseline_summary(rows, baseline),
            "baseline_floor": int(0.75 * len(rows)),
            "candidate": candidate_payload(PARENT, model),
            "arms": arms,
        }
        print(
            f"[v22 null seed={seed}] label={label} "
            + " ".join(
                f"{key}={arms[key]['mean_delta']:.4f}/{arms[key]['target_win_loss']}/{arms[key]['arm_label']}"
                for key in NULL_ARM_KEYS
            )
        )

    lookup_summary = baseline_summary(lookup_rows, lookup_baseline)
    parent_target = path_results[PARENT.name]["target_value"]
    parent_distractor = parent_controls["arms"]["distractor_value"]
    parent_random = parent_controls["arms"]["random_value"]
    criteria = {
        "lookup_baseline_valid": int(lookup_summary["target_wins"]) >= int(0.75 * len(lookup_rows)),
        "parent_effect": (
            float(parent_target["mean_delta"]) <= -1.0
            and int(parent_target["target_win_loss"]) >= 3
        ),
        "parent_source_controls_pass": (
            float(parent_target["mean_delta"]) <= float(parent_distractor["mean_delta"]) - 0.50
            and float(parent_target["mean_delta"]) <= float(parent_random["mean_delta"]) - 0.50
        ),
        "all_three_margin_row_count_at_least_24": int(row_summary["all_three_margin_row_count"]) >= 24,
        "all_three_margin_row_fraction_at_least_0p40": (
            float(row_summary["all_three_margin_row_fraction_of_parent_effect_rows"]) >= 0.40
        ),
        "parent_flip_pair_resistant_rows_at_least_8": (
            int(row_summary["parent_flip_pair_resistant_row_count"]) >= 8
        ),
        "median_singles_residual_at_most_minus_1": (
            row_summary["parent_effect_singles_residual_median"] is not None
            and float(row_summary["parent_effect_singles_residual_median"]) <= -1.0
        ),
        "parent_null_holdouts_clean": all(result["label"] == "clean_null" for result in null_results.values()),
    }
    elapsed = time.time() - started
    summary = {
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "marker": MARKER_TEXT,
        "model_id": args.model_id,
        "pair_count": PAIR_COUNT,
        "lookup_seeds": lookup_seeds,
        "null_seeds": null_seeds,
        "lookup_row_count_per_seed": args.lookup_row_count,
        "null_row_count_per_seed": args.null_row_count,
        "lookup_row_count": len(lookup_rows),
        "lookup_baseline": lookup_summary,
        "parent_candidate": candidate_payload(PARENT, model),
        "pair_candidates": [candidate_payload(path, model) for path in PAIR_PATHS],
        "single_candidates": [candidate_payload(path, model) for path in SINGLE_PATHS],
        "row_summary": row_summary,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "diagnostic_class": classify_diagnostic(criteria),
    }
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_response_marker_v22_row_interaction",
        "model_id": args.model_id,
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "elapsed_s": elapsed,
        "summary": summary,
        "path_results": path_results,
        "parent_source_controls": parent_controls,
        "row_diagnostics": row_diagnostics,
        "null_results": null_results,
        "rows": lookup_rows + [
            row for seed in null_seeds for row in null_rows_by_seed[seed]
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)
    print(json.dumps({**summary, "output_path": str(output_path)}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
