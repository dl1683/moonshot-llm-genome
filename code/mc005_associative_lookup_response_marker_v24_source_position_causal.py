#!/usr/bin/env python
"""MC005 V24 causal source-position diagnostic for layers 24-26."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import dataclass
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
    choose_distractor_index,
    choose_random_index,
    lookup_positions,
    render_lookup_prompt,
    sample_disjoint_row_words,
)
from mc005_associative_lookup_source_edge import (
    CARD_ID,
    KEY_WORDS,
    MODEL_ID,
    RESULT_DIR,
    VALUE_WORDS,
    first_token_id,
    valid_words,
)


LOOKUP_SEEDS = [211, 223]
NULL_SEEDS = [227, 229]
PAIR_COUNT = 16
BASE_FAMILIES_PER_SEED = 32
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


@dataclass(frozen=True)
class PositionVariant:
    name: str
    target_slot: int
    position_group: str


POSITION_VARIANTS = [
    PositionVariant("early_slot0", 0, "early"),
    PositionVariant("early_slot1", 1, "early"),
    PositionVariant("mid_slot10", 10, "mid_late"),
    PositionVariant("late_slot13", 13, "mid_late"),
]


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def place_target_pair(
    pairs: list[tuple[str, str]],
    target_original_index: int,
    target_slot: int,
) -> list[tuple[str, str]]:
    target_pair = pairs[target_original_index]
    rest = [pair for index, pair in enumerate(pairs) if index != target_original_index]
    ordered = rest[:]
    ordered.insert(target_slot, target_pair)
    return ordered


def pair_index(pairs: list[tuple[str, str]], pair: tuple[str, str]) -> int:
    for index, candidate in enumerate(pairs):
        if candidate == pair:
            return index
    raise ValueError(f"pair not found: {pair!r}")


def build_position_rows_for_seed(
    tokenizer: Any,
    seed: int,
    base_family_count: int,
) -> list[dict[str, Any]]:
    keys = valid_words(tokenizer, KEY_WORDS)
    values = valid_words(tokenizer, VALUE_WORDS)
    if len(keys) < PAIR_COUNT or len(values) < PAIR_COUNT:
        raise ValueError(f"not enough single-token words: keys={len(keys)} values={len(values)}")
    rng = random.Random(seed + PAIR_COUNT + 2400)
    rows = []
    for family_index in range(base_family_count):
        local = random.Random(rng.randint(0, 10_000_000) + family_index)
        row_keys, row_values = sample_disjoint_row_words(local, keys, values, PAIR_COUNT, PAIR_COUNT)
        base_pairs = list(zip(row_keys, row_values, strict=True))
        target_original_index = family_index % PAIR_COUNT
        distractor_original_index = choose_distractor_index(target_original_index, PAIR_COUNT, family_index)
        random_original_index = choose_random_index(
            {target_original_index, distractor_original_index},
            PAIR_COUNT,
            family_index,
        )
        target_pair = base_pairs[target_original_index]
        distractor_pair = base_pairs[distractor_original_index]
        random_pair = base_pairs[random_original_index]
        family_id = f"seed{seed}_family{family_index:03d}"
        for variant in POSITION_VARIANTS:
            pairs = place_target_pair(base_pairs, target_original_index, variant.target_slot)
            query_index = pair_index(pairs, target_pair)
            distractor_index = pair_index(pairs, distractor_pair)
            random_index = pair_index(pairs, random_pair)
            query_key, target_value = target_pair
            _, distractor_value = distractor_pair
            prompt, spans = render_lookup_prompt(pairs, query_key)
            positions, token_length = lookup_positions(
                tokenizer,
                prompt,
                spans,
                query_index,
                distractor_index,
                random_index,
            )
            rows.append(
                {
                    "id": f"mc005_v24_{family_id}_{variant.name}",
                    "source_id": f"{family_id}_{variant.name}",
                    "base_family_id": family_id,
                    "scenario": "lookup_pair16_response_source_position_causal",
                    "seed": seed,
                    "split": "lookup_source_position_causal_v24",
                    "mode": "lookup",
                    "marker": MARKER_TEXT,
                    "variant": variant.name,
                    "position_group": variant.position_group,
                    "target_slot": variant.target_slot,
                    "pairs": [{"key": key, "value": value} for key, value in pairs],
                    "base_pairs": [{"key": key, "value": value} for key, value in base_pairs],
                    "pair_count": PAIR_COUNT,
                    "query_pair_index": query_index,
                    "distractor_pair_index": distractor_index,
                    "random_value_index": random_index,
                    "target_original_index": target_original_index,
                    "distractor_original_index": distractor_original_index,
                    "random_original_index": random_original_index,
                    "query_key": query_key,
                    "target_value": target_value,
                    "distractor_value": distractor_value,
                    "random_value": random_pair[1],
                    "target_token_id": first_token_id(tokenizer, target_value),
                    "distractor_token_id": first_token_id(tokenizer, distractor_value),
                    "positions": positions,
                    "arm_keys": list(LOOKUP_ARM_KEYS),
                    "rendered_prompt": prompt,
                    "token_length": token_length,
                }
            )
    return rows


def build_position_rows(tokenizer: Any, seeds: list[int], base_family_count: int) -> list[dict[str, Any]]:
    return [
        row
        for seed in seeds
        for row in build_position_rows_for_seed(tokenizer, seed, base_family_count)
    ]


def build_null_rows(tokenizer: Any, seed: int, row_count: int) -> list[dict[str, Any]]:
    scenario = Scenario(
        "answer_absent_pair16_response_null",
        "null",
        PAIR_COUNT,
        NULL_ARM_KEYS,
        row_count=row_count,
    )
    rows = build_rows(tokenizer, scenario, seed, "mc005_v24")
    for row in rows:
        row["split"] = "answer_absent_null_holdout_v24"
    return rows


def margin(features: dict[str, Any]) -> float:
    return float(features["target_minus_distractor_margin"])


def row_delta(row_id: str, baseline: dict[str, dict[str, Any]], arm: dict[str, dict[str, Any]]) -> float:
    return margin(arm[row_id]) - margin(baseline[row_id])


def negative_effect_share(component_delta: float, parent_delta: float) -> float | None:
    if parent_delta >= -1e-12:
        return None
    return max(0.0, -component_delta) / max(1e-12, -parent_delta)


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
            f"[v24 target {path.name}] "
            f"mean_delta={target['mean_delta']:.4f} win_loss={target['target_win_loss']}"
        )
    return raw_by_path, summary_by_path


def score_parent_source_controls(
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    raw_arms = {}
    summary_arms = {}
    for arm_key in LOOKUP_ARM_KEYS:
        raw = score_rows_path(rows, tokenizer, model, batch_size, PARENT, arm_key)
        raw_arms[arm_key] = raw
        summary_arms[arm_key] = summarize_path_arm(rows, baseline, raw)
    return raw_arms, {"candidate": candidate_payload(PARENT, model), "arms": summary_arms}


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
        best_pair_share = None
        if any(share is not None for share in pair_negative_shares.values()):
            best_pair_share = max(float(share) for share in pair_negative_shares.values() if share is not None)
        diagnostics.append(
            {
                "row_id": row_id,
                "base_family_id": row["base_family_id"],
                "variant": row["variant"],
                "position_group": row["position_group"],
                "target_slot": int(row["target_slot"]),
                "target_source_position": int(row["positions"]["target_value"]),
                "baseline_margin": margin(baseline[row_id]),
                "baseline_target_wins": baseline_target_wins,
                "parent_margin": margin(parent_raw[row_id]),
                "parent_target_wins": parent_target_wins,
                "parent_delta": parent_delta,
                "pair_deltas": pair_deltas,
                "single_deltas": single_deltas,
                "pair_negative_effect_shares": pair_negative_shares,
                "best_pair_share": best_pair_share,
                "singles_sum": singles_sum,
                "singles_residual": singles_residual,
                "pair_plus_single_sums": pair_plus_single_sums,
                "pair_plus_single_residuals": pair_plus_single_residuals,
                "parent_effect_row": parent_effect_row,
                "all_pairs_below_0p60": all_pairs_below_0p60,
                "all_pair_plus_single_superadditive": all_pair_plus_single_superadditive,
                "all_three_margin_row": all_three_margin_row,
            }
        )
    return diagnostics


def group_row_summary(rows: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    row_ids = {row["id"] for row in rows}
    group_diags = [diag for diag in diagnostics if diag["row_id"] in row_ids]
    parent_effect = [diag for diag in group_diags if diag["parent_effect_row"]]
    all_three = [diag for diag in parent_effect if diag["all_three_margin_row"]]
    return {
        "row_count": len(group_diags),
        "baseline_target_win_count": sum(1 for diag in group_diags if diag["baseline_target_wins"]),
        "parent_effect_row_count": len(parent_effect),
        "all_three_margin_row_count": len(all_three),
        "all_three_fraction_of_parent_effect_rows": (
            len(all_three) / len(parent_effect) if parent_effect else None
        ),
        "median_best_pair_share": median([
            float(diag["best_pair_share"])
            for diag in parent_effect
            if diag["best_pair_share"] is not None
        ]),
        "median_singles_residual": median([float(diag["singles_residual"]) for diag in parent_effect]),
        "median_parent_delta": median([float(diag["parent_delta"]) for diag in parent_effect]),
        "median_target_source_position": median([float(diag["target_source_position"]) for diag in group_diags]),
    }


def summarize_groups(
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    raw_controls: dict[str, dict[str, dict[str, Any]]],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {
        "early": [row for row in rows if row["position_group"] == "early"],
        "mid_late": [row for row in rows if row["position_group"] == "mid_late"],
    }
    for variant in POSITION_VARIANTS:
        groups[variant.name] = [row for row in rows if row["variant"] == variant.name]

    summaries = {}
    for name, group_rows in groups.items():
        control_arms = {
            arm_key: summarize_path_arm(group_rows, baseline, raw_controls[arm_key])
            for arm_key in LOOKUP_ARM_KEYS
        }
        summaries[name] = {
            "baseline": baseline_summary(group_rows, baseline),
            "parent_source_controls": control_arms,
            "row_summary": group_row_summary(group_rows, diagnostics),
        }
    return summaries


def paired_summary(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for diag in diagnostics:
        by_family.setdefault(diag["base_family_id"], []).append(diag)
    family_records = []
    for family_id, rows in sorted(by_family.items()):
        early = [row for row in rows if row["position_group"] == "early"]
        mid_late = [row for row in rows if row["position_group"] == "mid_late"]
        early_any = any(row["all_three_margin_row"] for row in early)
        mid_late_any = any(row["all_three_margin_row"] for row in mid_late)
        early_parent = any(row["parent_effect_row"] for row in early)
        mid_late_parent = any(row["parent_effect_row"] for row in mid_late)
        early_shares = [
            float(row["best_pair_share"]) for row in early
            if row["parent_effect_row"] and row["best_pair_share"] is not None
        ]
        mid_late_shares = [
            float(row["best_pair_share"]) for row in mid_late
            if row["parent_effect_row"] and row["best_pair_share"] is not None
        ]
        family_records.append(
            {
                "base_family_id": family_id,
                "early_any_all_three": early_any,
                "mid_late_any_all_three": mid_late_any,
                "paired_all_three_gain": int(mid_late_any) - int(early_any),
                "early_parent_effect_any": early_parent,
                "mid_late_parent_effect_any": mid_late_parent,
                "early_median_best_pair_share": median(early_shares),
                "mid_late_median_best_pair_share": median(mid_late_shares),
                "best_pair_share_change_mid_late_minus_early": (
                    median(mid_late_shares) - median(early_shares)
                    if median(mid_late_shares) is not None and median(early_shares) is not None
                    else None
                ),
            }
        )
    mid_late_only = sum(
        (not row["early_any_all_three"]) and row["mid_late_any_all_three"]
        for row in family_records
    )
    early_only = sum(
        row["early_any_all_three"] and (not row["mid_late_any_all_three"])
        for row in family_records
    )
    both = sum(row["early_any_all_three"] and row["mid_late_any_all_three"] for row in family_records)
    neither = sum((not row["early_any_all_three"]) and (not row["mid_late_any_all_three"]) for row in family_records)
    share_changes = [
        float(row["best_pair_share_change_mid_late_minus_early"])
        for row in family_records
        if row["best_pair_share_change_mid_late_minus_early"] is not None
    ]
    return {
        "family_count": len(family_records),
        "mid_late_only_all_three_count": mid_late_only,
        "early_only_all_three_count": early_only,
        "both_all_three_count": both,
        "neither_all_three_count": neither,
        "net_mid_late_minus_early_all_three_count": mid_late_only - early_only,
        "median_best_pair_share_change_mid_late_minus_early": median(share_changes),
        "families": family_records,
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
        return "source_position_causal_supported"
    if (
        not criteria["all_position_group_baselines_valid"]
        or not criteria["all_position_group_parent_effects_pass"]
        or not criteria["all_position_group_source_controls_pass"]
    ):
        return "position_parent_failed"
    if not criteria["parent_null_holdouts_clean"]:
        return "null_failed"
    if not criteria["mid_late_all_three_fraction_beats_early_by_0p20"]:
        return "position_fraction_not_causal"
    if not criteria["paired_mid_late_net_gain_at_least_12"]:
        return "paired_position_not_causal"
    if not criteria["mid_late_best_pair_share_lower_by_0p05"]:
        return "best_pair_share_not_shifted"
    return "mixed_failure"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_response_marker_v24_source_position_causal")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-families-per-seed", type=int, default=BASE_FAMILIES_PER_SEED)
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
    lookup_rows = build_position_rows(tokenizer, lookup_seeds, args.base_families_per_seed)
    lookup_baseline = score_rows_path(lookup_rows, tokenizer, model, args.batch_size)
    raw_by_path, path_results = score_target_paths(
        lookup_rows,
        lookup_baseline,
        tokenizer,
        model,
        args.batch_size,
    )
    raw_controls, parent_controls = score_parent_source_controls(
        lookup_rows,
        lookup_baseline,
        tokenizer,
        model,
        args.batch_size,
    )
    row_diagnostics = compute_row_diagnostics(lookup_rows, lookup_baseline, raw_by_path)
    group_summaries = summarize_groups(lookup_rows, lookup_baseline, raw_controls, row_diagnostics)
    paired = paired_summary(row_diagnostics)
    print(
        "[v24 groups] "
        f"early_all3={group_summaries['early']['row_summary']['all_three_margin_row_count']} "
        f"mid_late_all3={group_summaries['mid_late']['row_summary']['all_three_margin_row_count']} "
        f"paired_net={paired['net_mid_late_minus_early_all_three_count']}"
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
            f"[v24 null seed={seed}] label={label} "
            + " ".join(
                f"{key}={arms[key]['mean_delta']:.4f}/{arms[key]['target_win_loss']}/{arms[key]['arm_label']}"
                for key in NULL_ARM_KEYS
            )
        )

    early = group_summaries["early"]["row_summary"]
    mid_late = group_summaries["mid_late"]["row_summary"]
    early_fraction = float(early["all_three_fraction_of_parent_effect_rows"] or 0.0)
    mid_late_fraction = float(mid_late["all_three_fraction_of_parent_effect_rows"] or 0.0)
    early_best_pair = early["median_best_pair_share"]
    mid_late_best_pair = mid_late["median_best_pair_share"]

    criteria = {
        "all_position_group_baselines_valid": all(
            int(group_summaries[group]["baseline"]["target_wins"])
            >= int(0.75 * group_summaries[group]["baseline"]["rows"])
            for group in ("early", "mid_late")
        ),
        "all_position_group_parent_effects_pass": all(
            float(group_summaries[group]["parent_source_controls"]["target_value"]["mean_delta"]) <= -1.0
            and int(group_summaries[group]["parent_source_controls"]["target_value"]["target_win_loss"]) >= 3
            for group in ("early", "mid_late")
        ),
        "all_position_group_source_controls_pass": all(
            float(group_summaries[group]["parent_source_controls"]["target_value"]["mean_delta"])
            <= float(group_summaries[group]["parent_source_controls"]["distractor_value"]["mean_delta"]) - 0.50
            and float(group_summaries[group]["parent_source_controls"]["target_value"]["mean_delta"])
            <= float(group_summaries[group]["parent_source_controls"]["random_value"]["mean_delta"]) - 0.50
            for group in ("early", "mid_late")
        ),
        "mid_late_all_three_fraction_beats_early_by_0p20": mid_late_fraction - early_fraction >= 0.20,
        "paired_mid_late_net_gain_at_least_12": int(paired["net_mid_late_minus_early_all_three_count"]) >= 12,
        "mid_late_best_pair_share_lower_by_0p05": (
            mid_late_best_pair is not None
            and early_best_pair is not None
            and float(mid_late_best_pair) <= float(early_best_pair) - 0.05
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
        "base_families_per_seed": args.base_families_per_seed,
        "lookup_row_count": len(lookup_rows),
        "lookup_baseline": baseline_summary(lookup_rows, lookup_baseline),
        "parent_candidate": candidate_payload(PARENT, model),
        "pair_candidates": [candidate_payload(path, model) for path in PAIR_PATHS],
        "single_candidates": [candidate_payload(path, model) for path in SINGLE_PATHS],
        "position_variants": [variant.__dict__ for variant in POSITION_VARIANTS],
        "group_summaries": group_summaries,
        "paired_summary": {key: value for key, value in paired.items() if key != "families"},
        "criteria": criteria,
        "passed": all(criteria.values()),
        "diagnostic_class": classify_diagnostic(criteria),
    }
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_response_marker_v24_source_position_causal",
        "model_id": args.model_id,
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "elapsed_s": elapsed,
        "summary": summary,
        "path_results": path_results,
        "parent_source_controls": parent_controls,
        "row_diagnostics": row_diagnostics,
        "paired_families": paired["families"],
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
