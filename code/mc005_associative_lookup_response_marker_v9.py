#!/usr/bin/env python
"""MC005 V9 compact Response-marker reliability atlas."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc001_qwen3_controlled_v10_head_localization import set_eager_attention
from mc005_associative_lookup_late_band_v3 import token_position_for_char
from mc005_associative_lookup_reliability_v4 import (
    SELECTED_BAND,
    SELECTED_LAYERS,
    greedy_counts,
    rotated_sample_pool,
    score_rows_atlas,
)
from mc005_associative_lookup_source_edge import (
    CARD_ID,
    KEY_WORDS,
    MODEL_ID,
    RESULT_DIR,
    VALUE_WORDS,
    first_token_id,
    summarize_arm,
    valid_words,
)


SWEEP_SEEDS = [17, 23, 31]
MARKER_TEXT = "Response"


@dataclass(frozen=True)
class Scenario:
    name: str
    mode: str
    pair_count: int
    arm_keys: tuple[str, ...]
    row_count: int = 32
    lexicon_offset: int = 0


SCENARIOS = [
    Scenario("lookup_pair5_response", "lookup", 5, ("target_value", "distractor_value", "random_value")),
    Scenario("lookup_pair8_response", "lookup", 8, ("target_value", "distractor_value", "random_value")),
    Scenario("offtarget_pair5_response", "null", 5, ("irrelevant_value", "irrelevant_key", "random_value")),
    Scenario(
        "answer_absent_response_null",
        "null",
        5,
        ("source_value", "non_source_control_value", "earlier_neutral_colon", "final_label", "final_colon"),
    ),
]


def add_line(lines: list[str], cursor: int, line: str) -> int:
    lines.append(line)
    return cursor + len(line) + 1


def choose_distractor_index(query_index: int, pair_count: int, row_index: int) -> int:
    candidates = [index for index in range(pair_count) if index != query_index]
    return candidates[row_index % len(candidates)]


def choose_random_index(excluded: set[int], pair_count: int, row_index: int) -> int:
    candidates = [index for index in range(pair_count) if index not in excluded]
    if not candidates:
        candidates = [index for index in range(pair_count)]
    return candidates[row_index % len(candidates)]


def sample_disjoint_row_words(
    local: random.Random,
    keys: list[str],
    values: list[str],
    pair_count: int,
    value_count: int,
) -> tuple[list[str], list[str]]:
    row_keys = local.sample(keys, pair_count)
    key_set = set(row_keys)
    value_candidates = [value for value in values if value not in key_set]
    if len(value_candidates) < value_count:
        raise ValueError(
            f"not enough key-disjoint values: values={len(value_candidates)} value_count={value_count}"
        )
    return row_keys, local.sample(value_candidates, value_count)


def render_lookup_prompt(
    pairs: list[tuple[str, str]],
    query_key: str,
) -> tuple[str, dict[str, dict[str, int]]]:
    lines = ["Reference pairs:"]
    cursor = len(lines[0]) + 1
    spans: dict[str, dict[str, int]] = {}
    for pair_index, (key, value) in enumerate(pairs):
        line = f"- {key}: {value}"
        spans[f"pair_{pair_index}"] = {
            "key": cursor + line.index(key),
            "colon": cursor + line.index(":"),
            "value": cursor + line.index(value),
        }
        cursor = add_line(lines, cursor, line)
    cursor = add_line(lines, cursor, f"Query key: {query_key}")
    final_line = f"{MARKER_TEXT}:"
    spans["final_marker"] = {
        "label": cursor + final_line.index(MARKER_TEXT),
        "colon": cursor + final_line.index(":"),
    }
    cursor = add_line(lines, cursor, final_line)
    _ = cursor
    return "\n".join(lines), spans


def render_answer_absent_prompt(
    pairs: list[tuple[str, str]],
    target_answer: str,
    distractor_answer: str,
    control_value: str,
    source_pair_index: int,
) -> tuple[str, dict[str, int]]:
    lines = ["Reference pairs:"]
    cursor = len(lines[0]) + 1
    positions: dict[str, int] = {}
    for pair_index, (key, value) in enumerate(pairs):
        line = f"- {key}: {value}"
        if pair_index == source_pair_index:
            positions["source_value_char"] = cursor + line.index(value)
        cursor = add_line(lines, cursor, line)
    line = f"Control word: {control_value}"
    positions["non_source_control_value_char"] = cursor + line.index(control_value)
    cursor = add_line(lines, cursor, line)
    line = "Neutral marker: ready"
    positions["earlier_neutral_colon_char"] = cursor + line.index(":")
    cursor = add_line(lines, cursor, line)
    cursor = add_line(lines, cursor, f"Required word: {target_answer}")
    cursor = add_line(lines, cursor, f"Distractor word: {distractor_answer}")
    cursor = add_line(lines, cursor, "Write the required word after the final marker.")
    final_line = f"{MARKER_TEXT}:"
    positions["final_label_char"] = cursor + final_line.index(MARKER_TEXT)
    positions["final_colon_char"] = cursor + final_line.index(":")
    cursor = add_line(lines, cursor, final_line)
    _ = cursor
    return "\n".join(lines), positions


def lookup_positions(
    tokenizer: Any,
    prompt: str,
    spans: dict[str, dict[str, int]],
    query_index: int,
    distractor_index: int,
    random_index: int,
    source_focus_index: int | None = None,
) -> tuple[dict[str, int], int]:
    offset_enc = tokenizer(prompt, return_offsets_mapping=True)
    offsets = [[int(left), int(right)] for left, right in offset_enc["offset_mapping"]]
    positions = {
        "target_value": token_position_for_char(offsets, spans[f"pair_{query_index}"]["value"]),
        "distractor_value": token_position_for_char(offsets, spans[f"pair_{distractor_index}"]["value"]),
        "random_value": token_position_for_char(offsets, spans[f"pair_{random_index}"]["value"]),
        "final_label": token_position_for_char(offsets, spans["final_marker"]["label"]),
        "final_colon": token_position_for_char(offsets, spans["final_marker"]["colon"]),
    }
    if source_focus_index is not None:
        positions["irrelevant_value"] = token_position_for_char(offsets, spans[f"pair_{source_focus_index}"]["value"])
        positions["irrelevant_key"] = token_position_for_char(offsets, spans[f"pair_{source_focus_index}"]["key"])
    return positions, len(offset_enc["input_ids"])


def answer_absent_positions(
    tokenizer: Any,
    prompt: str,
    char_positions: dict[str, int],
) -> tuple[dict[str, int], int]:
    offset_enc = tokenizer(prompt, return_offsets_mapping=True)
    offsets = [[int(left), int(right)] for left, right in offset_enc["offset_mapping"]]
    positions = {
        "source_value": token_position_for_char(offsets, char_positions["source_value_char"]),
        "non_source_control_value": token_position_for_char(
            offsets,
            char_positions["non_source_control_value_char"],
        ),
        "earlier_neutral_colon": token_position_for_char(offsets, char_positions["earlier_neutral_colon_char"]),
        "final_label": token_position_for_char(offsets, char_positions["final_label_char"]),
        "final_colon": token_position_for_char(offsets, char_positions["final_colon_char"]),
    }
    return positions, len(offset_enc["input_ids"])


def build_lookup_rows(
    tokenizer: Any,
    scenario: Scenario,
    seed: int,
    row_id_prefix: str = "mc005_v9",
) -> list[dict[str, Any]]:
    keys = rotated_sample_pool(valid_words(tokenizer, KEY_WORDS), scenario.lexicon_offset)
    values = rotated_sample_pool(valid_words(tokenizer, VALUE_WORDS), scenario.lexicon_offset)
    if len(keys) < scenario.pair_count or len(values) < scenario.pair_count:
        raise ValueError(
            f"scenario {scenario.name} lacks single-token words: "
            f"keys={len(keys)} values={len(values)} pair_count={scenario.pair_count}"
        )
    rng = random.Random(seed + scenario.lexicon_offset + scenario.pair_count)
    rows = []
    for row_index in range(scenario.row_count):
        local = random.Random(rng.randint(0, 10_000_000) + row_index)
        row_keys, row_values = sample_disjoint_row_words(
            local,
            keys,
            values,
            scenario.pair_count,
            scenario.pair_count,
        )
        pairs = list(zip(row_keys, row_values, strict=True))
        query_index = row_index % scenario.pair_count
        distractor_index = choose_distractor_index(query_index, scenario.pair_count, row_index)
        random_index = choose_random_index({query_index, distractor_index}, scenario.pair_count, row_index)
        query_key, target_value = pairs[query_index]
        _, distractor_value = pairs[distractor_index]
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
                "id": f"{row_id_prefix}_seed{seed}_{scenario.name}_{row_index:03d}",
                "source_id": f"seed{seed}_{scenario.name}_{row_index:03d}",
                "scenario": scenario.name,
                "seed": seed,
                "split": "response_marker_v9",
                "mode": scenario.mode,
                "marker": MARKER_TEXT,
                "pairs": [{"key": key, "value": value} for key, value in pairs],
                "pair_count": scenario.pair_count,
                "query_pair_index": query_index,
                "distractor_pair_index": distractor_index,
                "random_value_index": random_index,
                "query_key": query_key,
                "target_value": target_value,
                "distractor_value": distractor_value,
                "target_token_id": first_token_id(tokenizer, target_value),
                "distractor_token_id": first_token_id(tokenizer, distractor_value),
                "positions": positions,
                "arm_keys": list(scenario.arm_keys),
                "rendered_prompt": prompt,
                "token_length": token_length,
            }
        )
    return rows


def build_offtarget_rows(
    tokenizer: Any,
    scenario: Scenario,
    seed: int,
    row_id_prefix: str = "mc005_v9",
) -> list[dict[str, Any]]:
    keys = rotated_sample_pool(valid_words(tokenizer, KEY_WORDS), scenario.lexicon_offset)
    values = rotated_sample_pool(valid_words(tokenizer, VALUE_WORDS), scenario.lexicon_offset)
    if len(keys) < scenario.pair_count or len(values) < scenario.pair_count:
        raise ValueError(
            f"scenario {scenario.name} lacks single-token words: "
            f"keys={len(keys)} values={len(values)} pair_count={scenario.pair_count}"
        )
    rng = random.Random(seed + scenario.lexicon_offset + scenario.pair_count + 100)
    rows = []
    for row_index in range(scenario.row_count):
        local = random.Random(rng.randint(0, 10_000_000) + row_index)
        row_keys, row_values = sample_disjoint_row_words(
            local,
            keys,
            values,
            scenario.pair_count,
            scenario.pair_count,
        )
        pairs = list(zip(row_keys, row_values, strict=True))
        query_index = row_index % scenario.pair_count
        source_focus_index = (query_index + 1) % scenario.pair_count
        distractor_index = choose_random_index({query_index, source_focus_index}, scenario.pair_count, row_index)
        random_index = choose_random_index({query_index, source_focus_index, distractor_index}, scenario.pair_count, row_index)
        query_key, target_value = pairs[query_index]
        _, distractor_value = pairs[distractor_index]
        prompt, spans = render_lookup_prompt(pairs, query_key)
        positions, token_length = lookup_positions(
            tokenizer,
            prompt,
            spans,
            query_index,
            distractor_index,
            random_index,
            source_focus_index=source_focus_index,
        )
        rows.append(
            {
                "id": f"{row_id_prefix}_seed{seed}_{scenario.name}_{row_index:03d}",
                "source_id": f"seed{seed}_{scenario.name}_{row_index:03d}",
                "scenario": scenario.name,
                "seed": seed,
                "split": "response_marker_v9",
                "mode": scenario.mode,
                "marker": MARKER_TEXT,
                "pairs": [{"key": key, "value": value} for key, value in pairs],
                "pair_count": scenario.pair_count,
                "query_pair_index": query_index,
                "source_focus_pair_index": source_focus_index,
                "distractor_pair_index": distractor_index,
                "random_value_index": random_index,
                "query_key": query_key,
                "target_value": target_value,
                "distractor_value": distractor_value,
                "source_focus_value": pairs[source_focus_index][1],
                "target_token_id": first_token_id(tokenizer, target_value),
                "distractor_token_id": first_token_id(tokenizer, distractor_value),
                "positions": positions,
                "arm_keys": list(scenario.arm_keys),
                "rendered_prompt": prompt,
                "token_length": token_length,
            }
        )
    return rows


def build_answer_absent_rows(
    tokenizer: Any,
    scenario: Scenario,
    seed: int,
    row_id_prefix: str = "mc005_v9",
) -> list[dict[str, Any]]:
    keys = rotated_sample_pool(valid_words(tokenizer, KEY_WORDS), scenario.lexicon_offset)
    values = rotated_sample_pool(valid_words(tokenizer, VALUE_WORDS), scenario.lexicon_offset)
    needed_values = scenario.pair_count + 3
    if len(keys) < scenario.pair_count or len(values) < needed_values:
        raise ValueError(
            f"scenario {scenario.name} lacks single-token words: "
            f"keys={len(keys)} values={len(values)} needed_values={needed_values}"
        )
    rng = random.Random(seed + scenario.lexicon_offset + scenario.pair_count + 200)
    rows = []
    for row_index in range(scenario.row_count):
        local = random.Random(rng.randint(0, 10_000_000) + row_index)
        row_keys, row_values = sample_disjoint_row_words(
            local,
            keys,
            values,
            scenario.pair_count,
            needed_values,
        )
        source_values = row_values[: scenario.pair_count]
        target_answer = row_values[scenario.pair_count]
        distractor_answer = row_values[scenario.pair_count + 1]
        control_value = row_values[scenario.pair_count + 2]
        pairs = list(zip(row_keys, source_values, strict=True))
        source_pair_index = row_index % scenario.pair_count
        prompt, char_positions = render_answer_absent_prompt(
            pairs,
            target_answer,
            distractor_answer,
            control_value,
            source_pair_index,
        )
        positions, token_length = answer_absent_positions(tokenizer, prompt, char_positions)
        rows.append(
            {
                "id": f"{row_id_prefix}_seed{seed}_{scenario.name}_{row_index:03d}",
                "source_id": f"seed{seed}_{scenario.name}_{row_index:03d}",
                "scenario": scenario.name,
                "seed": seed,
                "split": "response_marker_v9",
                "mode": scenario.mode,
                "marker": MARKER_TEXT,
                "pairs": [{"key": key, "value": value} for key, value in pairs],
                "pair_count": scenario.pair_count,
                "source_focus_pair_index": source_pair_index,
                "target_value": target_answer,
                "distractor_value": distractor_answer,
                "source_focus_value": pairs[source_pair_index][1],
                "control_value": control_value,
                "target_token_id": first_token_id(tokenizer, target_answer),
                "distractor_token_id": first_token_id(tokenizer, distractor_answer),
                "positions": positions,
                "arm_keys": list(scenario.arm_keys),
                "rendered_prompt": prompt,
                "token_length": token_length,
            }
        )
    return rows


def build_rows(
    tokenizer: Any,
    scenario: Scenario,
    seed: int,
    row_id_prefix: str = "mc005_v9",
) -> list[dict[str, Any]]:
    if scenario.name.startswith("lookup_"):
        return build_lookup_rows(tokenizer, scenario, seed, row_id_prefix)
    if scenario.name.startswith("offtarget_"):
        return build_offtarget_rows(tokenizer, scenario, seed, row_id_prefix)
    return build_answer_absent_rows(tokenizer, scenario, seed, row_id_prefix)


def classify_lookup(summary: dict[str, Any]) -> str:
    if int(summary["baseline_clean_rows"]) < 16:
        return "breaks"
    target = summary["arms"]["target_value"]
    non_target_arms = [arm for key, arm in summary["arms"].items() if key != "target_value"]
    works = (
        float(target["mean_delta"]) <= -1.0
        and int(target["target_win_loss"]) >= 3
        and all(float(target["mean_delta"]) <= float(arm["mean_delta"]) - 0.50 for arm in non_target_arms)
    )
    if works:
        return "works"
    if float(target["mean_delta"]) < 0:
        return "weak"
    return "breaks"


def arm_is_clean(arm: dict[str, Any]) -> bool:
    return abs(float(arm["mean_delta"])) <= 0.50 and abs(int(arm["target_win_loss"])) <= 1


def arm_is_weak(arm: dict[str, Any]) -> bool:
    return abs(float(arm["mean_delta"])) <= 1.00 and abs(int(arm["target_win_loss"])) <= 2


def classify_null(summary: dict[str, Any]) -> str:
    if int(summary["baseline_clean_rows"]) < 24:
        return "invalid_baseline"
    arms = list(summary["arms"].values())
    if all(arm_is_clean(arm) for arm in arms):
        return "clean_null"
    if all(arm_is_weak(arm) for arm in arms):
        return "weak_null"
    return "side_effect"


def evaluate_scenario(
    scenario: Scenario,
    seed: int,
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    arms: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm_key, arm_features in arms.items():
        arm_summary = summarize_arm(rows, baseline, arm_features)
        arm_summary["greedy_counts"] = greedy_counts(arm_features, rows)
        arm_summary["abs_target_win_change"] = abs(int(arm_summary["target_win_loss"]))
        arm_summaries[arm_key] = arm_summary
    summary = {
        "scenario": {
            "name": scenario.name,
            "mode": scenario.mode,
            "seed": seed,
            "marker": MARKER_TEXT,
            "pair_count": scenario.pair_count,
            "row_count": len(rows),
            "arm_keys": list(scenario.arm_keys),
        },
        "rows": len(rows),
        "baseline_clean_rows": sum(1 for row in rows if baseline[row["id"]]["target_wins"]),
        "baseline_mean_margin": sum(
            float(baseline[row["id"]]["target_minus_distractor_margin"]) for row in rows
        )
        / max(1, len(rows)),
        "baseline_greedy_counts": greedy_counts(baseline, rows),
        "arms": arm_summaries,
    }
    summary["max_abs_mean_delta"] = max(abs(float(arm["mean_delta"])) for arm in arm_summaries.values())
    summary["max_abs_target_win_loss"] = max(abs(int(arm["target_win_loss"])) for arm in arm_summaries.values())
    summary["label"] = classify_lookup(summary) if scenario.mode == "lookup" else classify_null(summary)
    return summary


def summarize_suite(scenario_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts: dict[str, int] = {}
    scenario_label_counts: dict[str, dict[str, int]] = {}
    for summary in scenario_summaries:
        label = str(summary["label"])
        label_counts[label] = label_counts.get(label, 0) + 1
        name = str(summary["scenario"]["name"])
        scenario_label_counts.setdefault(name, {})
        scenario_label_counts[name][label] = scenario_label_counts[name].get(label, 0) + 1

    lookup = [summary for summary in scenario_summaries if summary["scenario"]["mode"] == "lookup"]
    nulls = [summary for summary in scenario_summaries if summary["scenario"]["mode"] == "null"]
    all_lookup_work = all(summary["label"] == "works" for summary in lookup)
    all_nulls_clean = all(summary["label"] == "clean_null" for summary in nulls)
    return {
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "marker": MARKER_TEXT,
        "seeds": SWEEP_SEEDS,
        "scenario_count": len(scenario_summaries),
        "lookup_count": len(lookup),
        "null_count": len(nulls),
        "label_counts": label_counts,
        "scenario_label_counts": scenario_label_counts,
        "lookup_work_count": sum(1 for summary in lookup if summary["label"] == "works"),
        "clean_null_count": sum(1 for summary in nulls if summary["label"] == "clean_null"),
        "all_lookup_scenarios_work": all_lookup_work,
        "all_null_scenarios_clean": all_nulls_clean,
        "passed": all_lookup_work and all_nulls_clean,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_associative_lookup_response_marker_v9")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--run-type", default="associative_lookup_response_marker_v9")
    parser.add_argument("--progress-label", default="response-atlas")
    parser.add_argument("--row-id-prefix", default="mc005_v9")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

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
    scenario_summaries = []
    all_rows = []
    for seed in SWEEP_SEEDS:
        for scenario in SCENARIOS:
            rows = build_rows(tokenizer, scenario, seed, args.row_id_prefix)
            all_rows.extend(rows)
            baseline = score_rows_atlas(rows, tokenizer, model, args.batch_size)
            arms = {
                arm_key: score_rows_atlas(rows, tokenizer, model, args.batch_size, arm_key)
                for arm_key in scenario.arm_keys
            }
            summary = evaluate_scenario(scenario, seed, rows, baseline, arms)
            scenario_summaries.append(summary)
            arm_bits = " ".join(
                f"{key}={summary['arms'][key]['mean_delta']:.4f}/"
                f"{summary['arms'][key]['target_win_loss']}"
                for key in scenario.arm_keys
            )
            print(
                f"[{args.progress_label} seed={seed} scenario={scenario.name}] "
                f"label={summary['label']} clean={summary['baseline_clean_rows']}/{summary['rows']} {arm_bits}"
            )

    suite_summary = summarize_suite(scenario_summaries)
    elapsed = time.time() - started
    result = {
        "card_id": args.card_id,
        "run_type": args.run_type,
        "model_id": args.model_id,
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "elapsed_s": elapsed,
        "summary": suite_summary,
        "scenario_summaries": scenario_summaries,
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
