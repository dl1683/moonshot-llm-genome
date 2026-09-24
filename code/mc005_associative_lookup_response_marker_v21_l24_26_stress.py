#!/usr/bin/env python
"""MC005 V21 stress test for the Qwen3-1.7B layers-24-26 interaction block."""

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
from mc005_associative_lookup_reliability_v4 import SELECTED_BAND, SELECTED_LAYERS, rotated_sample_pool
from mc005_associative_lookup_response_marker_v15_fine_localization import (
    Candidate,
    baseline_summary,
    candidate_payload,
    score_rows_path,
    summarize_path_arm,
)
from mc005_associative_lookup_response_marker_v9 import (
    MARKER_TEXT,
    Scenario as ResponseScenario,
    arm_is_clean,
    arm_is_weak,
    build_rows,
    choose_distractor_index,
    choose_random_index,
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


LOOKUP_ARM_KEYS = ("target_value", "distractor_value", "random_value")
NULL_ARM_KEYS = (
    "source_value",
    "non_source_control_value",
    "earlier_neutral_colon",
    "final_label",
    "final_colon",
)
NULL_SEEDS = [179, 181]
NULL_ROW_COUNT = 96
PARENT = Candidate("slice_l24_26_all", (24, 25, 26), None, False)


@dataclass(frozen=True)
class StressScenario:
    name: str
    pair_count: int
    layout: str
    lexicon_offset: int
    seed: int
    row_count: int = 64


STRESS_SCENARIOS = [
    StressScenario("pair16_dash_base", 16, "dash_colon", 0, 149),
    StressScenario("pair16_arrow_layout", 16, "arrow", 0, 151),
    StressScenario("pair16_sentence_layout", 16, "sentence", 0, 157),
    StressScenario("pair16_dash_lexicon_shift", 16, "dash_colon", 24, 163),
    StressScenario("pair20_dash_pairstress", 20, "dash_colon", 0, 167),
    StressScenario("pair20_arrow_lexicon_stress", 20, "arrow", 24, 173),
]


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def add_line(lines: list[str], cursor: int, line: str) -> int:
    lines.append(line)
    return cursor + len(line) + 1


def render_stress_prompt(
    pairs: list[tuple[str, str]],
    query_key: str,
    layout: str,
) -> tuple[str, dict[str, dict[str, int]]]:
    if layout == "dash_colon":
        lines = ["Reference pairs:"]
        cursor = len(lines[0]) + 1
        spans: dict[str, dict[str, int]] = {}
        for pair_index, (key, value) in enumerate(pairs):
            line = f"- {key}: {value}"
            spans[f"pair_{pair_index}"] = {
                "key": cursor + line.index(key),
                "value": cursor + line.index(value),
            }
            cursor = add_line(lines, cursor, line)
    elif layout == "arrow":
        lines = ["Lookup table:"]
        cursor = len(lines[0]) + 1
        spans = {}
        for pair_index, (key, value) in enumerate(pairs):
            line = f"{key} -> {value}"
            spans[f"pair_{pair_index}"] = {
                "key": cursor + line.index(key),
                "value": cursor + line.index(value),
            }
            cursor = add_line(lines, cursor, line)
    elif layout == "sentence":
        lines = ["Reference sentences:"]
        cursor = len(lines[0]) + 1
        spans = {}
        for pair_index, (key, value) in enumerate(pairs):
            line = f"The value for {key} is {value}."
            spans[f"pair_{pair_index}"] = {
                "key": cursor + line.index(key),
                "value": cursor + line.index(value),
            }
            cursor = add_line(lines, cursor, line)
    else:
        raise ValueError(f"unknown layout {layout!r}")
    cursor = add_line(lines, cursor, f"Query key: {query_key}")
    final_line = f"{MARKER_TEXT}:"
    spans["final_marker"] = {
        "label": cursor + final_line.index(MARKER_TEXT),
        "colon": cursor + final_line.index(":"),
    }
    cursor = add_line(lines, cursor, final_line)
    _ = cursor
    return "\n".join(lines), spans


def lookup_positions(
    tokenizer: Any,
    prompt: str,
    spans: dict[str, dict[str, int]],
    query_index: int,
    distractor_index: int,
    random_index: int,
) -> tuple[dict[str, int], int]:
    offset_enc = tokenizer(prompt, return_offsets_mapping=True)
    offsets = [[int(left), int(right)] for left, right in offset_enc["offset_mapping"]]
    return {
        "target_value": token_position_for_char(offsets, spans[f"pair_{query_index}"]["value"]),
        "distractor_value": token_position_for_char(offsets, spans[f"pair_{distractor_index}"]["value"]),
        "random_value": token_position_for_char(offsets, spans[f"pair_{random_index}"]["value"]),
        "final_label": token_position_for_char(offsets, spans["final_marker"]["label"]),
        "final_colon": token_position_for_char(offsets, spans["final_marker"]["colon"]),
    }, len(offset_enc["input_ids"])


def build_stress_rows(tokenizer: Any, scenario: StressScenario) -> list[dict[str, Any]]:
    keys = rotated_sample_pool(valid_words(tokenizer, KEY_WORDS), scenario.lexicon_offset)
    values = rotated_sample_pool(valid_words(tokenizer, VALUE_WORDS), scenario.lexicon_offset)
    if len(keys) < scenario.pair_count or len(values) < scenario.pair_count:
        raise ValueError(
            f"scenario {scenario.name} lacks single-token words: "
            f"keys={len(keys)} values={len(values)} pair_count={scenario.pair_count}"
        )
    rng = random.Random(scenario.seed + scenario.lexicon_offset + scenario.pair_count)
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
        prompt, spans = render_stress_prompt(pairs, query_key, scenario.layout)
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
                "id": f"mc005_v21_seed{scenario.seed}_{scenario.name}_{row_index:03d}",
                "source_id": f"seed{scenario.seed}_{scenario.name}_{row_index:03d}",
                "scenario": scenario.name,
                "seed": scenario.seed,
                "split": "lookup_stress_v21",
                "mode": "lookup",
                "marker": MARKER_TEXT,
                "layout": scenario.layout,
                "lexicon_offset": scenario.lexicon_offset,
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
                "arm_keys": list(LOOKUP_ARM_KEYS),
                "rendered_prompt": prompt,
                "token_length": token_length,
            }
        )
    return rows


def build_null_rows(tokenizer: Any, seed: int, row_count: int) -> list[dict[str, Any]]:
    scenario = ResponseScenario(
        "answer_absent_pair16_response_null",
        "null",
        16,
        NULL_ARM_KEYS,
        row_count=row_count,
    )
    rows = build_rows(tokenizer, scenario, seed, "mc005_v21")
    for row in rows:
        row["split"] = "answer_absent_null_holdout_v21"
    return rows


def classify_lookup(rows: list[dict[str, Any]], arms: dict[str, Any]) -> str:
    baseline_floor = int(0.75 * len(rows))
    if int(arms["baseline"]["target_wins"]) < baseline_floor:
        return "invalid_baseline"
    target = arms["target_value"]
    distractor = arms["distractor_value"]
    random_arm = arms["random_value"]
    if (
        float(target["mean_delta"]) <= -1.0
        and int(target["target_win_loss"]) >= 3
        and float(target["mean_delta"]) <= float(distractor["mean_delta"]) - 0.50
        and float(target["mean_delta"]) <= float(random_arm["mean_delta"]) - 0.50
    ):
        return "works"
    if float(target["mean_delta"]) < 0:
        return "weak"
    return "breaks"


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


def score_lookup_scenario(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
) -> dict[str, Any]:
    baseline = score_rows_path(rows, tokenizer, model, batch_size)
    arms: dict[str, Any] = {"baseline": baseline_summary(rows, baseline)}
    for arm_key in LOOKUP_ARM_KEYS:
        arm = score_rows_path(rows, tokenizer, model, batch_size, PARENT, arm_key)
        arms[arm_key] = summarize_path_arm(rows, baseline, arm)
    label = classify_lookup(rows, arms)
    return {
        "scenario": {
            "name": rows[0]["scenario"],
            "layout": rows[0]["layout"],
            "pair_count": rows[0]["pair_count"],
            "lexicon_offset": rows[0]["lexicon_offset"],
            "seed": rows[0]["seed"],
            "row_count": len(rows),
        },
        "candidate": candidate_payload(PARENT, model),
        "label": label,
        "arms": arms,
    }


def classify_diagnostic(criteria: dict[str, bool], lookup_results: dict[str, Any], null_results: dict[int, Any]) -> str:
    if all(criteria.values()):
        return "l24_26_stress_supported"
    invalid = [name for name, result in lookup_results.items() if result["label"] == "invalid_baseline"]
    weak_or_breaks = [name for name, result in lookup_results.items() if result["label"] in {"weak", "breaks"}]
    null_bad = [seed for seed, result in null_results.items() if result["label"] != "clean_null"]
    if invalid:
        return "stress_baseline_failed"
    if weak_or_breaks:
        return "stress_effect_failed"
    if not criteria["all_lookup_source_controls_pass"]:
        return "stress_source_control_failed"
    if null_bad:
        return "stress_null_failed"
    return "mixed_failure"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_response_marker_v21_l24_26_stress")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--null-row-count", type=int, default=NULL_ROW_COUNT)
    parser.add_argument("--null-seeds", default=",".join(str(seed) for seed in NULL_SEEDS))
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

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
    lookup_rows_by_scenario = {
        scenario.name: build_stress_rows(tokenizer, scenario)
        for scenario in STRESS_SCENARIOS
    }
    lookup_results = {}
    for name, rows in lookup_rows_by_scenario.items():
        result = score_lookup_scenario(rows, tokenizer, model, args.batch_size)
        lookup_results[name] = result
        target = result["arms"]["target_value"]
        print(
            f"[v21 lookup {name}] label={result['label']} "
            f"target={target['mean_delta']:.4f}/{target['target_win_loss']}"
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
            f"[v21 null seed={seed}] label={label} "
            + " ".join(
                f"{key}={arms[key]['mean_delta']:.4f}/{arms[key]['target_win_loss']}/{arms[key]['arm_label']}"
                for key in NULL_ARM_KEYS
            )
        )

    criteria = {
        "all_lookup_baselines_valid": all(result["label"] != "invalid_baseline" for result in lookup_results.values()),
        "all_lookup_target_effects_pass": all(result["label"] == "works" for result in lookup_results.values()),
        "all_lookup_source_controls_pass": all(
            float(result["arms"]["target_value"]["mean_delta"])
            <= float(result["arms"]["distractor_value"]["mean_delta"]) - 0.50
            and float(result["arms"]["target_value"]["mean_delta"])
            <= float(result["arms"]["random_value"]["mean_delta"]) - 0.50
            for result in lookup_results.values()
        ),
        "null_holdouts_clean": all(result["label"] == "clean_null" for result in null_results.values()),
    }
    elapsed = time.time() - started
    summary = {
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "marker": MARKER_TEXT,
        "model_id": args.model_id,
        "parent_candidate": candidate_payload(PARENT, model),
        "stress_scenarios": [scenario.__dict__ for scenario in STRESS_SCENARIOS],
        "null_seeds": null_seeds,
        "null_row_count_per_seed": args.null_row_count,
        "lookup_scenario_count": len(lookup_results),
        "lookup_labels": {name: result["label"] for name, result in lookup_results.items()},
        "lookup_work_count": sum(1 for result in lookup_results.values() if result["label"] == "works"),
        "lookup_count": len(lookup_results),
        "criteria": criteria,
        "passed": all(criteria.values()),
        "diagnostic_class": classify_diagnostic(criteria, lookup_results, null_results),
    }
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_response_marker_v21_l24_26_stress",
        "model_id": args.model_id,
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "elapsed_s": elapsed,
        "summary": summary,
        "lookup_results": lookup_results,
        "null_results": null_results,
        "rows": [
            row
            for rows in lookup_rows_by_scenario.values()
            for row in rows
        ]
        + [row for seed in null_seeds for row in null_rows_by_seed[seed]],
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
