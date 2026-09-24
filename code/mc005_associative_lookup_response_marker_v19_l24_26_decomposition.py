#!/usr/bin/env python
"""MC005 V19 decomposition gate for the Qwen3-1.7B layers-24-26 surface."""

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


DISCOVERY_SEEDS = [89, 97]
LOOKUP_HOLDOUT_SEEDS = [101, 103]
NULL_HOLDOUT_SEEDS = [107, 109]
PAIR_COUNT = 16
DISCOVERY_ROW_COUNT = 64
LOOKUP_HOLDOUT_ROW_COUNT = 96
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

LOWER_HEADS = tuple(range(0, 8))
UPPER_HEADS = tuple(range(8, 16))

SELECTABLE_CANDIDATES = [
    Candidate("single_l24_all", (24,), None, True),
    Candidate("single_l25_all", (25,), None, True),
    Candidate("single_l26_all", (26,), None, True),
    Candidate("slice_l24_25_all", (24, 25), None, True),
    Candidate("slice_l25_26_all", (25, 26), None, True),
    Candidate("l24_26_lower_heads", (24, 25, 26), LOWER_HEADS, True),
    Candidate("l24_26_upper_heads", (24, 25, 26), UPPER_HEADS, True),
    Candidate("l24_26_even_heads", (24, 25, 26), tuple(range(0, 16, 2)), True),
    Candidate("l24_26_odd_heads", (24, 25, 26), tuple(range(1, 16, 2)), True),
    Candidate("l24_26_heads_0_3", (24, 25, 26), tuple(range(0, 4)), True),
    Candidate("l24_26_heads_4_7", (24, 25, 26), tuple(range(4, 8)), True),
    Candidate("l24_26_heads_8_11", (24, 25, 26), tuple(range(8, 12)), True),
    Candidate("l24_26_heads_12_15", (24, 25, 26), tuple(range(12, 16)), True),
    Candidate("single_l24_lower_heads", (24,), LOWER_HEADS, True),
    Candidate("single_l24_upper_heads", (24,), UPPER_HEADS, True),
    Candidate("single_l25_lower_heads", (25,), LOWER_HEADS, True),
    Candidate("single_l25_upper_heads", (25,), UPPER_HEADS, True),
    Candidate("single_l26_lower_heads", (26,), LOWER_HEADS, True),
    Candidate("single_l26_upper_heads", (26,), UPPER_HEADS, True),
    Candidate("slice_l24_25_lower_heads", (24, 25), LOWER_HEADS, True),
    Candidate("slice_l24_25_upper_heads", (24, 25), UPPER_HEADS, True),
    Candidate("slice_l25_26_lower_heads", (25, 26), LOWER_HEADS, True),
    Candidate("slice_l25_26_upper_heads", (25, 26), UPPER_HEADS, True),
]


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def build_lookup_rows(
    tokenizer: Any,
    seeds: list[int],
    row_count: int,
    split: str,
) -> list[dict[str, Any]]:
    scenario = Scenario(
        "lookup_pair16_response",
        "lookup",
        PAIR_COUNT,
        LOOKUP_ARM_KEYS,
        row_count=row_count,
    )
    rows = []
    for seed in seeds:
        seed_rows = build_rows(tokenizer, scenario, seed, "mc005_v19")
        for row in seed_rows:
            row["split"] = split
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
    rows = build_rows(tokenizer, scenario, seed, "mc005_v19")
    for row in rows:
        row["split"] = "answer_absent_null_holdout_v19"
    return rows


def score_target_path(
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    path: Candidate,
) -> dict[str, Any]:
    arm = score_rows_path(rows, tokenizer, model, batch_size, path, "target_value")
    return {
        "candidate": candidate_payload(path, model),
        "target_value": summarize_path_arm(rows, baseline, arm),
    }


def score_controlled_path(
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    path: Candidate,
) -> dict[str, Any]:
    arms = {}
    for arm_key in LOOKUP_ARM_KEYS:
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


def select_candidate(discovery_results: dict[str, dict[str, Any]]) -> str:
    selectable = [
        (name, result["target_value"])
        for name, result in discovery_results.items()
        if result["candidate"]["selectable"]
    ]
    return min(
        selectable,
        key=lambda item: (
            float(item[1]["mean_delta"]),
            -int(item[1]["target_win_loss"]),
            item[0],
        ),
    )[0]


def effect_share(selected_target: dict[str, Any], parent_target: dict[str, Any]) -> float | None:
    parent_mag = abs(float(parent_target["mean_delta"]))
    if parent_mag <= 1e-12:
        return None
    return abs(float(selected_target["mean_delta"])) / parent_mag


def holdout_rank_stable(
    selected_name: str,
    holdout_targets: dict[str, dict[str, Any]],
    tolerance: float,
) -> bool:
    selected_delta = float(holdout_targets[selected_name]["target_value"]["mean_delta"])
    best_delta = min(
        float(result["target_value"]["mean_delta"])
        for result in holdout_targets.values()
        if result["candidate"]["selectable"]
    )
    return selected_delta <= best_delta + tolerance


def classify_diagnostic(criteria: dict[str, bool]) -> str:
    if all(criteria.values()):
        return "compact_l24_26_decomposition_supported"
    if not criteria["parent_holdout_effect"] or not criteria["parent_source_controls_pass"]:
        return "parent_not_replicated"
    if not criteria["selected_holdout_effect"]:
        return "no_smaller_effect"
    if not criteria["selected_source_controls_pass"]:
        return "selected_source_control_failed"
    if not criteria["selected_null_holdouts_clean"]:
        return "null_failed"
    if not criteria["selected_effect_share_at_least_0p60_of_parent"]:
        return "l24_26_block_still_required"
    if not criteria["selected_holdout_rank_stable"]:
        return "discovery_unstable"
    return "mixed_failure"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_response_marker_v19_l24_26_decomposition")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--discovery-row-count", type=int, default=DISCOVERY_ROW_COUNT)
    parser.add_argument("--lookup-holdout-row-count", type=int, default=LOOKUP_HOLDOUT_ROW_COUNT)
    parser.add_argument("--null-row-count", type=int, default=NULL_ROW_COUNT)
    parser.add_argument("--discovery-seeds", default=",".join(str(seed) for seed in DISCOVERY_SEEDS))
    parser.add_argument("--lookup-holdout-seeds", default=",".join(str(seed) for seed in LOOKUP_HOLDOUT_SEEDS))
    parser.add_argument("--null-holdout-seeds", default=",".join(str(seed) for seed in NULL_HOLDOUT_SEEDS))
    parser.add_argument("--rank-tolerance", type=float, default=0.25)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    discovery_seeds = parse_ints(args.discovery_seeds)
    lookup_holdout_seeds = parse_ints(args.lookup_holdout_seeds)
    null_holdout_seeds = parse_ints(args.null_holdout_seeds)
    candidate_by_name = {candidate.name: candidate for candidate in SELECTABLE_CANDIDATES}
    context_paths = [GLOBAL_FULL, GRANDPARENT, PARENT] + SELECTABLE_CANDIDATES

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
    discovery_rows = build_lookup_rows(
        tokenizer,
        discovery_seeds,
        args.discovery_row_count,
        "lookup_discovery_v19",
    )
    holdout_rows = build_lookup_rows(
        tokenizer,
        lookup_holdout_seeds,
        args.lookup_holdout_row_count,
        "lookup_holdout_v19",
    )

    discovery_baseline = score_rows_path(discovery_rows, tokenizer, model, args.batch_size)
    holdout_baseline = score_rows_path(holdout_rows, tokenizer, model, args.batch_size)

    discovery_results = {}
    for path in context_paths:
        discovery_results[path.name] = score_target_path(
            discovery_rows,
            discovery_baseline,
            tokenizer,
            model,
            args.batch_size,
            path,
        )
        target = discovery_results[path.name]["target_value"]
        print(
            f"[v19 discovery {path.name}] "
            f"mean_delta={target['mean_delta']:.4f} win_loss={target['target_win_loss']}"
        )

    selected_name = select_candidate(discovery_results)
    selected = candidate_by_name[selected_name]
    print(f"[v19 selected] {selected_name}")

    holdout_targets = {}
    for path in context_paths:
        holdout_targets[path.name] = score_target_path(
            holdout_rows,
            holdout_baseline,
            tokenizer,
            model,
            args.batch_size,
            path,
        )
        target = holdout_targets[path.name]["target_value"]
        print(
            f"[v19 holdout-target {path.name}] "
            f"mean_delta={target['mean_delta']:.4f} win_loss={target['target_win_loss']}"
        )

    parent_holdout = score_controlled_path(
        holdout_rows,
        holdout_baseline,
        tokenizer,
        model,
        args.batch_size,
        PARENT,
    )
    selected_holdout = score_controlled_path(
        holdout_rows,
        holdout_baseline,
        tokenizer,
        model,
        args.batch_size,
        selected,
    )
    for label, result in (("parent", parent_holdout), ("selected", selected_holdout)):
        target = result["arms"]["target_value"]
        distractor = result["arms"]["distractor_value"]
        random = result["arms"]["random_value"]
        print(
            f"[v19 holdout {label}] target={target['mean_delta']:.4f}/{target['target_win_loss']} "
            f"distractor={distractor['mean_delta']:.4f}/{distractor['target_win_loss']} "
            f"random={random['mean_delta']:.4f}/{random['target_win_loss']}"
        )

    null_results = {}
    null_rows_by_seed = {}
    for seed in null_holdout_seeds:
        rows = build_null_rows(tokenizer, seed, args.null_row_count)
        null_rows_by_seed[seed] = rows
        baseline = score_rows_path(rows, tokenizer, model, args.batch_size)
        arms = {}
        for arm_key in NULL_ARM_KEYS:
            arm = score_rows_path(rows, tokenizer, model, args.batch_size, selected, arm_key)
            arms[arm_key] = summarize_path_arm(rows, baseline, arm)
        label = classify_null(rows, baseline, arms)
        null_results[seed] = {
            "seed": seed,
            "label": label,
            "baseline": baseline_summary(rows, baseline),
            "baseline_floor": int(0.75 * len(rows)),
            "candidate": candidate_payload(selected, model),
            "arms": arms,
        }
        print(
            f"[v19 null seed={seed}] label={label} "
            + " ".join(
                f"{key}={arms[key]['mean_delta']:.4f}/{arms[key]['target_win_loss']}/{arms[key]['arm_label']}"
                for key in NULL_ARM_KEYS
            )
        )

    parent_target = parent_holdout["arms"]["target_value"]
    parent_distractor = parent_holdout["arms"]["distractor_value"]
    parent_random = parent_holdout["arms"]["random_value"]
    selected_target = selected_holdout["arms"]["target_value"]
    selected_distractor = selected_holdout["arms"]["distractor_value"]
    selected_random = selected_holdout["arms"]["random_value"]
    share = effect_share(selected_target, parent_target)
    rank_stable = holdout_rank_stable(selected_name, holdout_targets, args.rank_tolerance)
    holdout_sorted = sorted(
        (
            {
                "name": name,
                "mean_delta": float(result["target_value"]["mean_delta"]),
                "target_win_loss": int(result["target_value"]["target_win_loss"]),
            }
            for name, result in holdout_targets.items()
            if result["candidate"]["selectable"]
        ),
        key=lambda item: (item["mean_delta"], -item["target_win_loss"], item["name"]),
    )
    criteria = {
        "parent_holdout_effect": float(parent_target["mean_delta"]) <= -1.0
        and int(parent_target["target_win_loss"]) >= 3,
        "parent_source_controls_pass": (
            float(parent_target["mean_delta"]) <= float(parent_distractor["mean_delta"]) - 0.50
            and float(parent_target["mean_delta"]) <= float(parent_random["mean_delta"]) - 0.50
        ),
        "selected_holdout_effect": float(selected_target["mean_delta"]) <= -1.0
        and int(selected_target["target_win_loss"]) >= 3,
        "selected_source_controls_pass": (
            float(selected_target["mean_delta"]) <= float(selected_distractor["mean_delta"]) - 0.50
            and float(selected_target["mean_delta"]) <= float(selected_random["mean_delta"]) - 0.50
        ),
        "selected_effect_share_at_least_0p60_of_parent": share is not None and share >= 0.60,
        "selected_holdout_rank_stable": rank_stable,
        "selected_null_holdouts_clean": all(result["label"] == "clean_null" for result in null_results.values()),
    }

    elapsed = time.time() - started
    summary = {
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "marker": MARKER_TEXT,
        "model_id": args.model_id,
        "pair_count": PAIR_COUNT,
        "discovery_seeds": discovery_seeds,
        "lookup_holdout_seeds": lookup_holdout_seeds,
        "null_holdout_seeds": null_holdout_seeds,
        "discovery_row_count_per_seed": args.discovery_row_count,
        "lookup_holdout_row_count_per_seed": args.lookup_holdout_row_count,
        "null_row_count_per_seed": args.null_row_count,
        "discovery_row_count": len(discovery_rows),
        "lookup_holdout_row_count": len(holdout_rows),
        "parent_candidate": candidate_payload(PARENT, model),
        "grandparent_candidate": candidate_payload(GRANDPARENT, model),
        "global_reference_candidate": candidate_payload(GLOBAL_FULL, model),
        "selectable_candidates": [candidate_payload(candidate, model) for candidate in SELECTABLE_CANDIDATES],
        "selected_candidate": candidate_payload(selected, model),
        "discovery_baseline": baseline_summary(discovery_rows, discovery_baseline),
        "lookup_holdout_baseline": baseline_summary(holdout_rows, holdout_baseline),
        "selected_effect_share_of_parent": share,
        "holdout_rank_tolerance": args.rank_tolerance,
        "holdout_selectable_ranking": holdout_sorted,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "diagnostic_class": classify_diagnostic(criteria),
    }
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_response_marker_v19_l24_26_decomposition",
        "model_id": args.model_id,
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "elapsed_s": elapsed,
        "summary": summary,
        "discovery_results": discovery_results,
        "holdout_targets": holdout_targets,
        "parent_holdout": parent_holdout,
        "selected_holdout": selected_holdout,
        "null_results": null_results,
        "rows": discovery_rows + holdout_rows + [
            row for seed in null_holdout_seeds for row in null_rows_by_seed[seed]
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
