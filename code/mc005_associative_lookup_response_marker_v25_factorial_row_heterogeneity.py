#!/usr/bin/env python
"""MC005 V25 factorial row-heterogeneity diagnostic for layers 24-26."""

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


LOOKUP_SEEDS = [233, 239]
NULL_SEEDS = [251, 257]
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
class FactorialVariant:
    name: str
    target_slot: int
    target_position_group: str
    distractor_relation: str
    distractor_slot: int


FACTORIAL_VARIANTS = [
    FactorialVariant("early_near", 0, "early", "near", 1),
    FactorialVariant("early_far", 0, "early", "far", 13),
    FactorialVariant("late_near", 13, "mid_late", "near", 14),
    FactorialVariant("late_far", 13, "mid_late", "far", 0),
]
CELL_NAMES = ["early_near", "early_far", "mid_late_near", "mid_late_far"]


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def place_target_and_distractor_pairs(
    pairs: list[tuple[str, str]],
    target_original_index: int,
    distractor_original_index: int,
    target_slot: int,
    distractor_slot: int,
) -> list[tuple[str, str]]:
    if target_slot == distractor_slot:
        raise ValueError("target_slot and distractor_slot must differ")
    target_pair = pairs[target_original_index]
    distractor_pair = pairs[distractor_original_index]
    fixed = {target_slot: target_pair, distractor_slot: distractor_pair}
    rest = [
        pair
        for index, pair in enumerate(pairs)
        if index not in {target_original_index, distractor_original_index}
    ]
    rest_iter = iter(rest)
    ordered = []
    for slot in range(len(pairs)):
        if slot in fixed:
            ordered.append(fixed[slot])
        else:
            ordered.append(next(rest_iter))
    return ordered


def pair_index(pairs: list[tuple[str, str]], pair: tuple[str, str]) -> int:
    for index, candidate in enumerate(pairs):
        if candidate == pair:
            return index
    raise ValueError(f"pair not found: {pair!r}")


def build_factorial_rows_for_seed(
    tokenizer: Any,
    seed: int,
    base_family_count: int,
) -> list[dict[str, Any]]:
    keys = valid_words(tokenizer, KEY_WORDS)
    values = valid_words(tokenizer, VALUE_WORDS)
    if len(keys) < PAIR_COUNT or len(values) < PAIR_COUNT:
        raise ValueError(f"not enough single-token words: keys={len(keys)} values={len(values)}")
    rng = random.Random(seed + PAIR_COUNT + 2500)
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
        for variant in FACTORIAL_VARIANTS:
            pairs = place_target_and_distractor_pairs(
                base_pairs,
                target_original_index,
                distractor_original_index,
                variant.target_slot,
                variant.distractor_slot,
            )
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
                    "id": f"mc005_v25_{family_id}_{variant.name}",
                    "source_id": f"{family_id}_{variant.name}",
                    "base_family_id": family_id,
                    "scenario": "lookup_pair16_response_factorial_row_heterogeneity",
                    "seed": seed,
                    "split": "lookup_factorial_row_heterogeneity_v25",
                    "mode": "lookup",
                    "marker": MARKER_TEXT,
                    "variant": variant.name,
                    "position_group": variant.target_position_group,
                    "target_position_group": variant.target_position_group,
                    "distractor_relation": variant.distractor_relation,
                    "target_slot": variant.target_slot,
                    "distractor_slot": variant.distractor_slot,
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
                    "target_distractor_token_distance": (
                        int(positions["distractor_value"]) - int(positions["target_value"])
                    ),
                    "target_before_distractor": (
                        int(positions["target_value"]) < int(positions["distractor_value"])
                    ),
                    "arm_keys": list(LOOKUP_ARM_KEYS),
                    "rendered_prompt": prompt,
                    "token_length": token_length,
                }
            )
    return rows


def build_factorial_rows(tokenizer: Any, seeds: list[int], base_family_count: int) -> list[dict[str, Any]]:
    return [
        row
        for seed in seeds
        for row in build_factorial_rows_for_seed(tokenizer, seed, base_family_count)
    ]


def build_null_rows(tokenizer: Any, seed: int, row_count: int) -> list[dict[str, Any]]:
    scenario = Scenario(
        "answer_absent_pair16_response_null",
        "null",
        PAIR_COUNT,
        NULL_ARM_KEYS,
        row_count=row_count,
    )
    rows = build_rows(tokenizer, scenario, seed, "mc005_v25")
    for row in rows:
        row["split"] = "answer_absent_null_holdout_v25"
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
            f"[v25 target {path.name}] "
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
                "seed": int(row["seed"]),
                "variant": row["variant"],
                "position_group": row["position_group"],
                "target_position_group": row["target_position_group"],
                "distractor_relation": row["distractor_relation"],
                "target_slot": int(row["target_slot"]),
                "distractor_slot": int(row["distractor_slot"]),
                "target_source_position": int(row["positions"]["target_value"]),
                "distractor_source_position": int(row["positions"]["distractor_value"]),
                "target_distractor_token_distance": int(row["target_distractor_token_distance"]),
                "target_before_distractor": bool(row["target_before_distractor"]),
                "target_value": row["target_value"],
                "distractor_value": row["distractor_value"],
                "target_token_id": int(row["target_token_id"]),
                "distractor_token_id": int(row["distractor_token_id"]),
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


def add_baseline_margin_bands(diagnostics: list[dict[str, Any]]) -> None:
    parent_rows = [row for row in diagnostics if row["parent_effect_row"]]
    ordered = sorted(parent_rows, key=lambda row: (float(row["baseline_margin"]), row["row_id"]))
    split = len(ordered) // 2
    low_ids = {row["row_id"] for row in ordered[:split]}
    high_ids = {row["row_id"] for row in ordered[split:]}
    for row in diagnostics:
        if row["row_id"] in low_ids:
            row["baseline_margin_band"] = "low_margin"
        elif row["row_id"] in high_ids:
            row["baseline_margin_band"] = "high_margin"
        else:
            row["baseline_margin_band"] = "non_parent_effect"


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
        "median_distractor_source_position": median([
            float(diag["distractor_source_position"]) for diag in group_diags
        ]),
        "median_target_distractor_token_distance": median([
            float(diag["target_distractor_token_distance"]) for diag in group_diags
        ]),
        "all_three_family_count": len({diag["base_family_id"] for diag in all_three}),
        "all_three_seed_count": len({int(diag["seed"]) for diag in all_three}),
    }


def rows_matching_diagnostics(
    rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    predicate: Any,
) -> list[dict[str, Any]]:
    row_ids = {diag["row_id"] for diag in diagnostics if predicate(diag)}
    return [row for row in rows if row["id"] in row_ids]


def summarize_groups(
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    raw_controls: dict[str, dict[str, dict[str, Any]]],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"overall": rows}
    for target_group in ("early", "mid_late"):
        groups[target_group] = [
            row for row in rows if row["target_position_group"] == target_group
        ]
    for relation in ("near", "far"):
        groups[f"distractor_{relation}"] = [
            row for row in rows if row["distractor_relation"] == relation
        ]
    groups["early_near"] = [
        row
        for row in rows
        if row["target_position_group"] == "early" and row["distractor_relation"] == "near"
    ]
    groups["early_far"] = [
        row
        for row in rows
        if row["target_position_group"] == "early" and row["distractor_relation"] == "far"
    ]
    groups["mid_late_near"] = [
        row
        for row in rows
        if row["target_position_group"] == "mid_late" and row["distractor_relation"] == "near"
    ]
    groups["mid_late_far"] = [
        row
        for row in rows
        if row["target_position_group"] == "mid_late" and row["distractor_relation"] == "far"
    ]
    for variant in FACTORIAL_VARIANTS:
        groups[variant.name] = [row for row in rows if row["variant"] == variant.name]
    for margin_band in ("low_margin", "high_margin"):
        groups[margin_band] = rows_matching_diagnostics(
            rows,
            diagnostics,
            lambda diag, margin_band=margin_band: diag["baseline_margin_band"] == margin_band,
        )
        for target_group in ("early", "mid_late"):
            groups[f"{target_group}_{margin_band}"] = rows_matching_diagnostics(
                rows,
                diagnostics,
                lambda diag, margin_band=margin_band, target_group=target_group: (
                    diag["baseline_margin_band"] == margin_band
                    and diag["target_position_group"] == target_group
                ),
            )
        for cell in CELL_NAMES:
            target_group, relation = cell.rsplit("_", 1)
            groups[f"{cell}_{margin_band}"] = rows_matching_diagnostics(
                rows,
                diagnostics,
                lambda diag, margin_band=margin_band, target_group=target_group, relation=relation: (
                    diag["baseline_margin_band"] == margin_band
                    and diag["target_position_group"] == target_group
                    and diag["distractor_relation"] == relation
                ),
            )

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


def family_summary(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for diag in diagnostics:
        by_family.setdefault(diag["base_family_id"], []).append(diag)
    family_records = []
    for family_id, rows in sorted(by_family.items()):
        early = [row for row in rows if row["target_position_group"] == "early"]
        mid_late = [row for row in rows if row["target_position_group"] == "mid_late"]
        near = [row for row in rows if row["distractor_relation"] == "near"]
        far = [row for row in rows if row["distractor_relation"] == "far"]
        early_any = any(row["all_three_margin_row"] for row in early)
        mid_late_any = any(row["all_three_margin_row"] for row in mid_late)
        near_any = any(row["all_three_margin_row"] for row in near)
        far_any = any(row["all_three_margin_row"] for row in far)
        early_parent = any(row["parent_effect_row"] for row in early)
        mid_late_parent = any(row["parent_effect_row"] for row in mid_late)
        all_three_variants = [
            row["variant"] for row in rows if row["all_three_margin_row"]
        ]
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
                "near_any_all_three": near_any,
                "far_any_all_three": far_any,
                "all_three_variants": sorted(all_three_variants),
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
        "near_any_all_three_count": sum(row["near_any_all_three"] for row in family_records),
        "far_any_all_three_count": sum(row["far_any_all_three"] for row in family_records),
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


def fraction(summary: dict[str, Any]) -> float:
    value = summary["row_summary"]["all_three_fraction_of_parent_effect_rows"]
    return float(value) if value is not None else 0.0


def factorial_contrasts(
    group_summaries: dict[str, Any],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    cell_records = []
    for cell in CELL_NAMES:
        summary = group_summaries[cell]["row_summary"]
        cell_records.append(
            {
                "cell": cell,
                "parent_effect_row_count": int(summary["parent_effect_row_count"]),
                "all_three_margin_row_count": int(summary["all_three_margin_row_count"]),
                "all_three_fraction": fraction(group_summaries[cell]),
                "all_three_family_count": int(summary["all_three_family_count"]),
                "all_three_seed_count": int(summary["all_three_seed_count"]),
            }
        )
    best_cell = max(cell_records, key=lambda row: (row["all_three_fraction"], row["cell"]))
    worst_cell = min(cell_records, key=lambda row: (row["all_three_fraction"], row["cell"]))
    near_position_effect = fraction(group_summaries["mid_late_near"]) - fraction(group_summaries["early_near"])
    far_position_effect = fraction(group_summaries["mid_late_far"]) - fraction(group_summaries["early_far"])
    margin_contrasts = {}
    for target_group in ("early", "mid_late"):
        high = fraction(group_summaries[f"{target_group}_high_margin"])
        low = fraction(group_summaries[f"{target_group}_low_margin"])
        margin_contrasts[target_group] = {
            "high_margin_fraction": high,
            "low_margin_fraction": low,
            "high_minus_low": high - low,
            "high_parent_effect_rows": int(
                group_summaries[f"{target_group}_high_margin"]["row_summary"]["parent_effect_row_count"]
            ),
            "low_parent_effect_rows": int(
                group_summaries[f"{target_group}_low_margin"]["row_summary"]["parent_effect_row_count"]
            ),
        }
    best_cell_all_three = [
        row
        for row in diagnostics
        if row["all_three_margin_row"]
        and (
            f"{row['target_position_group']}_{row['distractor_relation']}"
            == best_cell["cell"]
        )
    ]
    return {
        "cell_records": cell_records,
        "best_cell": best_cell,
        "worst_cell": worst_cell,
        "cell_fraction_range": best_cell["all_three_fraction"] - worst_cell["all_three_fraction"],
        "near_position_effect_mid_late_minus_early": near_position_effect,
        "far_position_effect_mid_late_minus_early": far_position_effect,
        "distractor_modulation_abs_difference": abs(near_position_effect - far_position_effect),
        "margin_contrasts": margin_contrasts,
        "max_margin_high_minus_low": max(
            item["high_minus_low"] for item in margin_contrasts.values()
        ),
        "best_cell_all_three_family_count": len({
            row["base_family_id"] for row in best_cell_all_three
        }),
        "best_cell_all_three_seed_count": len({int(row["seed"]) for row in best_cell_all_three}),
    }


def classify_diagnostic(criteria: dict[str, bool]) -> str:
    if all(criteria.values()):
        return "factorial_row_heterogeneity_supported"
    if (
        not criteria["overall_lookup_baseline_valid"]
        or not criteria["all_factorial_cell_baselines_valid"]
        or not criteria["all_factorial_cell_parent_effects_pass"]
        or not criteria["all_factorial_cell_source_controls_pass"]
        or not criteria["parent_null_holdouts_clean"]
    ):
        return "factorial_parent_failed"
    if (
        not criteria["all_cells_parent_effect_rows_at_least_24"]
        or not criteria["cell_all_three_fraction_range_at_least_0p25"]
    ):
        return "factorial_cell_contrast_failed"
    if not criteria["distractor_modulates_position_effect_at_least_0p15"]:
        return "distractor_modulation_failed"
    if not criteria["margin_high_beats_low_within_position_by_0p20"]:
        return "margin_factor_failed"
    if not criteria["best_cell_all_three_spans_at_least_8_families_and_both_seeds"]:
        return "family_artifact_failed"
    return "mixed_failure"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_response_marker_v25_factorial_row_heterogeneity")
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
    lookup_rows = build_factorial_rows(tokenizer, lookup_seeds, args.base_families_per_seed)
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
    add_baseline_margin_bands(row_diagnostics)
    group_summaries = summarize_groups(lookup_rows, lookup_baseline, raw_controls, row_diagnostics)
    families = family_summary(row_diagnostics)
    contrasts = factorial_contrasts(group_summaries, row_diagnostics)
    print(
        "[v25 cells] "
        + " ".join(
            f"{cell}={group_summaries[cell]['row_summary']['all_three_margin_row_count']}/"
            f"{group_summaries[cell]['row_summary']['parent_effect_row_count']}"
            for cell in CELL_NAMES
        )
        + f" range={contrasts['cell_fraction_range']:.4f}"
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
            f"[v25 null seed={seed}] label={label} "
            + " ".join(
                f"{key}={arms[key]['mean_delta']:.4f}/{arms[key]['target_win_loss']}/{arms[key]['arm_label']}"
                for key in NULL_ARM_KEYS
            )
        )

    lookup_summary = baseline_summary(lookup_rows, lookup_baseline)

    criteria = {
        "overall_lookup_baseline_valid": int(lookup_summary["target_wins"]) >= int(0.75 * len(lookup_rows)),
        "all_factorial_cell_baselines_valid": all(
            int(group_summaries[cell]["baseline"]["target_wins"])
            >= int(0.75 * group_summaries[cell]["baseline"]["rows"])
            for cell in CELL_NAMES
        ),
        "all_factorial_cell_parent_effects_pass": all(
            float(group_summaries[cell]["parent_source_controls"]["target_value"]["mean_delta"]) <= -1.0
            and int(group_summaries[cell]["parent_source_controls"]["target_value"]["target_win_loss"]) >= 3
            for cell in CELL_NAMES
        ),
        "all_factorial_cell_source_controls_pass": all(
            float(group_summaries[cell]["parent_source_controls"]["target_value"]["mean_delta"])
            <= float(group_summaries[cell]["parent_source_controls"]["distractor_value"]["mean_delta"]) - 0.50
            and float(group_summaries[cell]["parent_source_controls"]["target_value"]["mean_delta"])
            <= float(group_summaries[cell]["parent_source_controls"]["random_value"]["mean_delta"]) - 0.50
            for cell in CELL_NAMES
        ),
        "parent_null_holdouts_clean": all(result["label"] == "clean_null" for result in null_results.values()),
        "all_cells_parent_effect_rows_at_least_24": all(
            int(group_summaries[cell]["row_summary"]["parent_effect_row_count"]) >= 24
            for cell in CELL_NAMES
        ),
        "cell_all_three_fraction_range_at_least_0p25": (
            float(contrasts["cell_fraction_range"]) >= 0.25
        ),
        "distractor_modulates_position_effect_at_least_0p15": (
            float(contrasts["distractor_modulation_abs_difference"]) >= 0.15
        ),
        "margin_high_beats_low_within_position_by_0p20": (
            float(contrasts["max_margin_high_minus_low"]) >= 0.20
        ),
        "best_cell_all_three_spans_at_least_8_families_and_both_seeds": (
            int(contrasts["best_cell_all_three_family_count"]) >= 8
            and int(contrasts["best_cell_all_three_seed_count"]) >= 2
        ),
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
        "lookup_baseline": lookup_summary,
        "parent_candidate": candidate_payload(PARENT, model),
        "pair_candidates": [candidate_payload(path, model) for path in PAIR_PATHS],
        "single_candidates": [candidate_payload(path, model) for path in SINGLE_PATHS],
        "factorial_variants": [variant.__dict__ for variant in FACTORIAL_VARIANTS],
        "group_summaries": group_summaries,
        "factorial_contrasts": contrasts,
        "family_summary": {key: value for key, value in families.items() if key != "families"},
        "criteria": criteria,
        "passed": all(criteria.values()),
        "diagnostic_class": classify_diagnostic(criteria),
    }
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_response_marker_v25_factorial_row_heterogeneity",
        "model_id": args.model_id,
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "elapsed_s": elapsed,
        "summary": summary,
        "path_results": path_results,
        "parent_source_controls": parent_controls,
        "row_diagnostics": row_diagnostics,
        "factorial_families": families["families"],
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
