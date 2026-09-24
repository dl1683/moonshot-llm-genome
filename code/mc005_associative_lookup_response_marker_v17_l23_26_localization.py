#!/usr/bin/env python
"""MC005 V17 held-out localization gate for slice_l23_26_all."""

from __future__ import annotations

import argparse
import json
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


LOOKUP_SEEDS = [43, 47]
NULL_SEEDS = [53, 59]
PAIR_COUNT = 16
LOOKUP_ROW_COUNT = 96
NULL_ROW_COUNT = 96
LOOKUP_ARM_KEYS = ("target_value", "distractor_value", "random_value")
NULL_ARM_KEYS = (
    "source_value",
    "non_source_control_value",
    "earlier_neutral_colon",
    "final_label",
    "final_colon",
)

FULL = Candidate("full_l20_26_all", tuple(SELECTED_LAYERS), None, False)
PRIMARY = Candidate("slice_l23_26_all", (23, 24, 25, 26), None, True)
SMALLER_SLICES = [
    Candidate("slice_l20_22_all", (20, 21, 22), None, False),
    Candidate("slice_l23_24_all", (23, 24), None, False),
    Candidate("slice_l25_26_all", (25, 26), None, False),
]


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
        seed_rows = build_rows(tokenizer, scenario, seed, "mc005_v17")
        for row in seed_rows:
            row["split"] = "lookup_holdout_v17"
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
    rows = build_rows(tokenizer, scenario, seed, "mc005_v17")
    for row in rows:
        row["split"] = "answer_absent_null_holdout_v17"
    return rows


def classify_null(rows: list[dict[str, Any]], baseline: dict[str, dict[str, Any]], arms: dict[str, Any]) -> str:
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


def score_lookup_path(
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    path: Candidate,
    arm_keys: tuple[str, ...],
) -> dict[str, Any]:
    arms = {}
    for arm_key in arm_keys:
        arm = score_rows_path(rows, tokenizer, model, batch_size, path, arm_key)
        arms[arm_key] = summarize_path_arm(rows, baseline, arm)
    return {"candidate": candidate_payload(path, model), "arms": arms}


def effect_share(candidate_target: dict[str, Any], full_target: dict[str, Any]) -> float | None:
    full_magnitude = abs(float(full_target["mean_delta"]))
    if full_magnitude <= 1e-12:
        return None
    return abs(float(candidate_target["mean_delta"])) / full_magnitude


def classify_diagnostic(criteria: dict[str, bool]) -> str:
    if all(criteria.values()):
        return "l23_26_localization_supported"
    if not criteria["candidate_target_effect"]:
        return "no_l23_26_effect"
    if not criteria["candidate_source_controls_pass"]:
        return "source_control_failed"
    if not criteria["candidate_null_holdouts_clean"]:
        return "null_failed"
    if not criteria["candidate_beats_smaller_slices"]:
        return "smaller_slice_not_separated"
    if not criteria["candidate_effect_share_at_least_0p60"]:
        return "full_band_still_required"
    return "mixed_failure"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_response_marker_v17_l23_26_localization")
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

    lookup_results = {
        FULL.name: score_lookup_path(
            lookup_rows,
            lookup_baseline,
            tokenizer,
            model,
            args.batch_size,
            FULL,
            LOOKUP_ARM_KEYS,
        ),
        PRIMARY.name: score_lookup_path(
            lookup_rows,
            lookup_baseline,
            tokenizer,
            model,
            args.batch_size,
            PRIMARY,
            LOOKUP_ARM_KEYS,
        ),
    }
    for path in SMALLER_SLICES:
        lookup_results[path.name] = score_lookup_path(
            lookup_rows,
            lookup_baseline,
            tokenizer,
            model,
            args.batch_size,
            path,
            ("target_value",),
        )
    for name, result in lookup_results.items():
        target = result["arms"]["target_value"]
        print(
            f"[v17 lookup {name}] target_mean_delta={target['mean_delta']:.4f} "
            f"win_loss={target['target_win_loss']}"
        )

    null_results = {}
    for seed in null_seeds:
        rows = build_null_rows(tokenizer, seed, args.null_row_count)
        baseline = score_rows_path(rows, tokenizer, model, args.batch_size)
        arms = {}
        for arm_key in NULL_ARM_KEYS:
            arm = score_rows_path(rows, tokenizer, model, args.batch_size, PRIMARY, arm_key)
            arms[arm_key] = summarize_path_arm(rows, baseline, arm)
        label = classify_null(rows, baseline, arms)
        null_results[seed] = {
            "seed": seed,
            "label": label,
            "baseline": baseline_summary(rows, baseline),
            "baseline_floor": int(0.75 * len(rows)),
            "candidate": candidate_payload(PRIMARY, model),
            "arms": arms,
        }
        print(
            f"[v17 null seed={seed}] label={label} "
            + " ".join(
                f"{key}={arms[key]['mean_delta']:.4f}/{arms[key]['target_win_loss']}/{arms[key]['arm_label']}"
                for key in NULL_ARM_KEYS
            )
        )

    full_target = lookup_results[FULL.name]["arms"]["target_value"]
    candidate_target = lookup_results[PRIMARY.name]["arms"]["target_value"]
    candidate_distractor = lookup_results[PRIMARY.name]["arms"]["distractor_value"]
    candidate_random = lookup_results[PRIMARY.name]["arms"]["random_value"]
    share = effect_share(candidate_target, full_target)
    smaller_target_deltas = {
        path.name: float(lookup_results[path.name]["arms"]["target_value"]["mean_delta"])
        for path in SMALLER_SLICES
    }
    criteria = {
        "full_benchmark_effect": float(full_target["mean_delta"]) <= -1.0 and int(full_target["target_win_loss"]) >= 3,
        "candidate_target_effect": float(candidate_target["mean_delta"]) <= -1.0
        and int(candidate_target["target_win_loss"]) >= 3,
        "candidate_source_controls_pass": (
            float(candidate_target["mean_delta"]) <= float(candidate_distractor["mean_delta"]) - 0.50
            and float(candidate_target["mean_delta"]) <= float(candidate_random["mean_delta"]) - 0.50
        ),
        "candidate_effect_share_at_least_0p60": share is not None and share >= 0.60,
        "candidate_beats_smaller_slices": all(
            float(candidate_target["mean_delta"]) <= delta - 0.50
            for delta in smaller_target_deltas.values()
        ),
        "candidate_null_holdouts_clean": all(result["label"] == "clean_null" for result in null_results.values()),
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
        "lookup_baseline": baseline_summary(lookup_rows, lookup_baseline),
        "full_candidate": candidate_payload(FULL, model),
        "primary_candidate": candidate_payload(PRIMARY, model),
        "smaller_slice_controls": [candidate_payload(path, model) for path in SMALLER_SLICES],
        "candidate_effect_share_of_full": share,
        "smaller_slice_target_deltas": smaller_target_deltas,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "diagnostic_class": classify_diagnostic(criteria),
    }
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_response_marker_v17_l23_26_localization",
        "model_id": args.model_id,
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "elapsed_s": elapsed,
        "summary": summary,
        "lookup_results": lookup_results,
        "null_results": null_results,
        "rows": lookup_rows + [
            row
            for seed in null_seeds
            for row in build_null_rows(tokenizer, seed, args.null_row_count)
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
