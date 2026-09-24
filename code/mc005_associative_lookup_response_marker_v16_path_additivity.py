#!/usr/bin/env python
"""MC005 V16 path additivity diagnostic for the Qwen3-1.7B Response surface."""

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
from mc005_associative_lookup_response_marker_v9 import MARKER_TEXT, Scenario, build_rows
from mc005_associative_lookup_source_edge import CARD_ID, MODEL_ID, RESULT_DIR


SEEDS = [17, 23, 31]
PAIR_COUNT = 16
ROW_COUNT = 128
ARM_KEYS = ("target_value", "distractor_value", "random_value")

PATHS = [
    Candidate("full_l20_26_all", tuple(SELECTED_LAYERS), None, False),
    Candidate("full_l20_26_lower_heads", tuple(SELECTED_LAYERS), tuple(range(0, 8)), False),
    Candidate("full_l20_26_upper_heads", tuple(SELECTED_LAYERS), tuple(range(8, 16)), False),
    Candidate("full_l20_26_even_heads", tuple(SELECTED_LAYERS), tuple(range(0, 16, 2)), False),
    Candidate("full_l20_26_odd_heads", tuple(SELECTED_LAYERS), tuple(range(1, 16, 2)), False),
    Candidate("slice_l20_22_all", (20, 21, 22), None, False),
    Candidate("slice_l23_24_all", (23, 24), None, False),
    Candidate("slice_l25_26_all", (25, 26), None, False),
    Candidate("slice_l20_24_all", (20, 21, 22, 23, 24), None, False),
    Candidate("slice_l23_26_all", (23, 24, 25, 26), None, False),
    Candidate("slice_l20_22_l25_26_all", (20, 21, 22, 25, 26), None, False),
]

CONTROL_PATHS = {
    "full_l20_26_all",
    "full_l20_26_lower_heads",
    "full_l20_26_upper_heads",
}

PARTITION_FAMILIES = {
    "heads_lower_plus_upper": ["full_l20_26_lower_heads", "full_l20_26_upper_heads"],
    "heads_even_plus_odd": ["full_l20_26_even_heads", "full_l20_26_odd_heads"],
    "layers_20_22_plus_23_24_plus_25_26": [
        "slice_l20_22_all",
        "slice_l23_24_all",
        "slice_l25_26_all",
    ],
    "layers_20_24_plus_25_26": ["slice_l20_24_all", "slice_l25_26_all"],
    "layers_20_22_plus_23_26": ["slice_l20_22_all", "slice_l23_26_all"],
    "layers_20_22_plus_25_26_omit_middle": ["slice_l20_22_all", "slice_l25_26_all"],
}


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def build_lookup_rows(tokenizer: Any, seeds: list[int], row_count: int) -> list[dict[str, Any]]:
    scenario = Scenario(
        "lookup_pair16_response",
        "lookup",
        PAIR_COUNT,
        ARM_KEYS,
        row_count=row_count,
    )
    rows = []
    for seed in seeds:
        seed_rows = build_rows(tokenizer, scenario, seed, "mc005_v16")
        for row in seed_rows:
            row["split"] = "path_additivity_v16"
        rows.extend(seed_rows)
    return rows


def row_deltas(
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    arm: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        row_id = str(row["id"])
        baseline_margin = float(baseline[row_id]["target_minus_distractor_margin"])
        arm_margin = float(arm[row_id]["target_minus_distractor_margin"])
        out.append(
            {
                "row_id": row_id,
                "seed": int(row["seed"]),
                "source_id": row["source_id"],
                "target_value": row["target_value"],
                "distractor_value": row["distractor_value"],
                "baseline_margin": baseline_margin,
                "arm_margin": arm_margin,
                "delta": arm_margin - baseline_margin,
                "baseline_target_wins": bool(baseline[row_id]["target_wins"]),
                "arm_target_wins": bool(arm[row_id]["target_wins"]),
            }
        )
    return out


def analyze_family(full_delta: float, component_deltas: list[float]) -> dict[str, Any]:
    component_sum = sum(component_deltas)
    residual = full_delta - component_sum
    full_magnitude = abs(full_delta)
    residual_fraction = residual / full_magnitude if full_magnitude > 1e-12 else None
    if residual_fraction is None:
        label = "invalid_full_effect"
    elif abs(residual_fraction) <= 0.20:
        label = "additive"
    elif residual_fraction < -0.20:
        label = "superadditive_full"
    else:
        label = "subadditive_or_overlap"
    return {
        "full_mean_delta": full_delta,
        "component_mean_deltas": component_deltas,
        "component_sum": component_sum,
        "residual": residual,
        "residual_fraction": residual_fraction,
        "label": label,
    }


def classify_suite(family_results: dict[str, dict[str, Any]]) -> str:
    head_labels = {
        str(family_results[name]["label"])
        for name in ("heads_lower_plus_upper", "heads_even_plus_odd")
    }
    layer_labels = {
        str(family_results[name]["label"])
        for name in (
            "layers_20_22_plus_23_24_plus_25_26",
            "layers_20_24_plus_25_26",
            "layers_20_22_plus_23_26",
        )
    }
    has_head_super = "superadditive_full" in head_labels
    has_layer_super = "superadditive_full" in layer_labels
    has_subadditive = any(
        str(result["label"]) == "subadditive_or_overlap"
        for result in family_results.values()
    )
    if head_labels == {"additive"} and layer_labels == {"additive"}:
        return "distributed_additive"
    if has_head_super and has_layer_super:
        return "mixed_superadditive"
    if has_head_super:
        return "head_superadditive"
    if has_layer_super:
        return "layer_superadditive"
    if has_subadditive:
        return "subadditive_or_overlap"
    return "mixed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_response_marker_v16_path_additivity")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--row-count", type=int, default=ROW_COUNT)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    seeds = parse_ints(args.seeds)

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
    rows = build_lookup_rows(tokenizer, seeds, args.row_count)
    baseline = score_rows_path(rows, tokenizer, model, args.batch_size)
    path_results: dict[str, dict[str, Any]] = {}
    row_delta_details: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for path in PATHS:
        path_results[path.name] = {"candidate": candidate_payload(path, model), "arms": {}}
        row_delta_details[path.name] = {}
        arm_names = list(ARM_KEYS) if path.name in CONTROL_PATHS else ["target_value"]
        for arm_key in arm_names:
            arm = score_rows_path(rows, tokenizer, model, args.batch_size, path, arm_key)
            path_results[path.name]["arms"][arm_key] = summarize_path_arm(rows, baseline, arm)
            row_delta_details[path.name][arm_key] = row_deltas(rows, baseline, arm)
        target = path_results[path.name]["arms"]["target_value"]
        print(
            f"[v16 path {path.name}] target_mean_delta={target['mean_delta']:.4f} "
            f"win_loss={target['target_win_loss']}"
        )

    full_delta = float(path_results["full_l20_26_all"]["arms"]["target_value"]["mean_delta"])
    family_results = {
        name: analyze_family(
            full_delta,
            [
                float(path_results[component]["arms"]["target_value"]["mean_delta"])
                for component in components
            ],
        )
        for name, components in PARTITION_FAMILIES.items()
    }
    suite_class = classify_suite(family_results)
    elapsed = time.time() - started
    summary = {
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "marker": MARKER_TEXT,
        "model_id": args.model_id,
        "pair_count": PAIR_COUNT,
        "seeds": seeds,
        "row_count_per_seed": args.row_count,
        "row_count": len(rows),
        "paths": [candidate_payload(path, model) for path in PATHS],
        "control_paths": sorted(CONTROL_PATHS),
        "baseline": baseline_summary(rows, baseline),
        "full_target_mean_delta": full_delta,
        "full_target_win_loss": int(path_results["full_l20_26_all"]["arms"]["target_value"]["target_win_loss"]),
        "family_results": family_results,
        "diagnostic_class": suite_class,
    }
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_response_marker_v16_path_additivity",
        "model_id": args.model_id,
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "elapsed_s": elapsed,
        "summary": summary,
        "path_results": path_results,
        "row_delta_details": row_delta_details,
        "rows": rows,
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
