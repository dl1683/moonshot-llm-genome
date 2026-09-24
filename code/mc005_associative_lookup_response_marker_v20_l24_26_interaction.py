#!/usr/bin/env python
"""MC005 V20 interaction diagnostic for the Qwen3-1.7B layers-24-26 surface."""

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


LOOKUP_SEEDS = [113, 127]
NULL_SEEDS = [131, 137]
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

GLOBAL_FULL = Candidate("full_l20_26_all", tuple(SELECTED_LAYERS), None, False)
GRANDPARENT = Candidate("slice_l23_26_all", (23, 24, 25, 26), None, False)
PARENT = Candidate("slice_l24_26_all", (24, 25, 26), None, False)

PATHS = [
    GLOBAL_FULL,
    GRANDPARENT,
    PARENT,
    Candidate("slice_l24_25_all", (24, 25), None, False),
    Candidate("slice_l24_26_all_pair", (24, 26), None, False),
    Candidate("slice_l25_26_all", (25, 26), None, False),
    Candidate("single_l24_all", (24,), None, False),
    Candidate("single_l25_all", (25,), None, False),
    Candidate("single_l26_all", (26,), None, False),
]

CONTROL_PATHS = {
    "slice_l24_26_all",
    "slice_l24_25_all",
    "slice_l24_26_all_pair",
    "slice_l25_26_all",
}

PAIR_PATHS = [
    "slice_l24_25_all",
    "slice_l24_26_all_pair",
    "slice_l25_26_all",
]
SINGLE_PATHS = ["single_l24_all", "single_l25_all", "single_l26_all"]

ADDITIVITY_FAMILIES = {
    "singles_l24_plus_l25_plus_l26": SINGLE_PATHS,
    "pair_l24_25_plus_single_l26": ["slice_l24_25_all", "single_l26_all"],
    "pair_l24_26_plus_single_l25": ["slice_l24_26_all_pair", "single_l25_all"],
    "pair_l25_26_plus_single_l24": ["slice_l25_26_all", "single_l24_all"],
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
        seed_rows = build_rows(tokenizer, scenario, seed, "mc005_v20")
        for row in seed_rows:
            row["split"] = "lookup_interaction_v20"
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
    rows = build_rows(tokenizer, scenario, seed, "mc005_v20")
    for row in rows:
        row["split"] = "answer_absent_null_holdout_v20"
    return rows


def score_path(
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    path: Candidate,
) -> dict[str, Any]:
    arms = {}
    arm_keys = LOOKUP_ARM_KEYS if path.name in CONTROL_PATHS else ("target_value",)
    for arm_key in arm_keys:
        arm = score_rows_path(rows, tokenizer, model, batch_size, path, arm_key)
        arms[arm_key] = summarize_path_arm(rows, baseline, arm)
    return {"candidate": candidate_payload(path, model), "arms": arms}


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


def effect_share(child_target: dict[str, Any], parent_target: dict[str, Any]) -> float | None:
    parent_mag = abs(float(parent_target["mean_delta"]))
    if parent_mag <= 1e-12:
        return None
    return abs(float(child_target["mean_delta"])) / parent_mag


def analyze_additivity(parent_delta: float, component_deltas: list[float]) -> dict[str, Any]:
    component_sum = sum(component_deltas)
    residual = parent_delta - component_sum
    parent_mag = abs(parent_delta)
    residual_fraction = residual / parent_mag if parent_mag > 1e-12 else None
    if residual_fraction is None:
        label = "invalid_parent_effect"
    elif abs(residual_fraction) <= 0.20:
        label = "additive"
    elif residual_fraction < -0.20:
        label = "superadditive_parent"
    else:
        label = "subadditive_or_overlap"
    return {
        "parent_mean_delta": parent_delta,
        "component_mean_deltas": component_deltas,
        "component_sum": component_sum,
        "residual": residual,
        "residual_fraction": residual_fraction,
        "label": label,
    }


def classify_diagnostic(criteria: dict[str, bool]) -> str:
    if all(criteria.values()):
        return "l24_26_three_layer_interaction_supported"
    if not criteria["parent_effect"] or not criteria["parent_source_controls_pass"]:
        return "parent_not_replicated"
    if not criteria["all_pairs_below_0p60_parent_share"]:
        return "pair_candidate_reopened"
    if not criteria["singles_superadditive_parent_residual"]:
        return "not_superadditive"
    if not criteria["parent_null_holdouts_clean"]:
        return "null_failed"
    return "mixed_failure"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_response_marker_v20_l24_26_interaction")
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
    path_results = {}
    for path in PATHS:
        path_results[path.name] = score_path(
            lookup_rows,
            lookup_baseline,
            tokenizer,
            model,
            args.batch_size,
            path,
        )
        target = path_results[path.name]["arms"]["target_value"]
        print(
            f"[v20 path {path.name}] target_mean_delta={target['mean_delta']:.4f} "
            f"win_loss={target['target_win_loss']}"
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
            f"[v20 null seed={seed}] label={label} "
            + " ".join(
                f"{key}={arms[key]['mean_delta']:.4f}/{arms[key]['target_win_loss']}/{arms[key]['arm_label']}"
                for key in NULL_ARM_KEYS
            )
        )

    parent_target = path_results[PARENT.name]["arms"]["target_value"]
    parent_delta = float(parent_target["mean_delta"])
    pair_effect_shares = {
        name: effect_share(path_results[name]["arms"]["target_value"], parent_target)
        for name in PAIR_PATHS
    }
    single_effect_shares = {
        name: effect_share(path_results[name]["arms"]["target_value"], parent_target)
        for name in SINGLE_PATHS
    }
    additivity_results = {
        name: analyze_additivity(
            parent_delta,
            [
                float(path_results[component]["arms"]["target_value"]["mean_delta"])
                for component in components
            ],
        )
        for name, components in ADDITIVITY_FAMILIES.items()
    }
    parent_distractor = path_results[PARENT.name]["arms"]["distractor_value"]
    parent_random = path_results[PARENT.name]["arms"]["random_value"]
    criteria = {
        "parent_effect": parent_delta <= -1.0 and int(parent_target["target_win_loss"]) >= 3,
        "parent_source_controls_pass": (
            parent_delta <= float(parent_distractor["mean_delta"]) - 0.50
            and parent_delta <= float(parent_random["mean_delta"]) - 0.50
        ),
        "all_pairs_below_0p60_parent_share": all(
            share is not None and share < 0.60
            for share in pair_effect_shares.values()
        ),
        "singles_superadditive_parent_residual": (
            additivity_results["singles_l24_plus_l25_plus_l26"]["label"] == "superadditive_parent"
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
        "lookup_baseline": baseline_summary(lookup_rows, lookup_baseline),
        "parent_candidate": candidate_payload(PARENT, model),
        "grandparent_candidate": candidate_payload(GRANDPARENT, model),
        "global_reference_candidate": candidate_payload(GLOBAL_FULL, model),
        "paths": [candidate_payload(path, model) for path in PATHS],
        "control_paths": sorted(CONTROL_PATHS),
        "pair_effect_shares_of_parent": pair_effect_shares,
        "single_effect_shares_of_parent": single_effect_shares,
        "additivity_results": additivity_results,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "diagnostic_class": classify_diagnostic(criteria),
    }
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_response_marker_v20_l24_26_interaction",
        "model_id": args.model_id,
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "elapsed_s": elapsed,
        "summary": summary,
        "path_results": path_results,
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
