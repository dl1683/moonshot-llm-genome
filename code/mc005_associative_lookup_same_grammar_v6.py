#!/usr/bin/env python
"""MC005 V6 same-grammar side-effect decomposition."""

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


@dataclass(frozen=True)
class Scenario:
    name: str
    surface: str
    row_count: int = 32
    pair_count: int = 5
    lexicon_offset: int = 0
    arm_keys: tuple[str, ...] = ("irrelevant_value", "irrelevant_key", "irrelevant_colon")


SCENARIOS = [
    Scenario(
        "query_other_pair_original",
        "query_other_pair",
        arm_keys=("irrelevant_value", "irrelevant_key", "irrelevant_colon"),
    ),
    Scenario(
        "query_other_pair_control_word",
        "query_other_pair_control_word",
        arm_keys=("irrelevant_value", "irrelevant_key", "irrelevant_colon", "non_source_control_value"),
    ),
    Scenario(
        "answer_absent_reference_control",
        "answer_absent_reference_control",
        arm_keys=(
            "irrelevant_value",
            "irrelevant_key",
            "irrelevant_colon",
            "non_source_control_value",
            "final_query_label",
        ),
    ),
]


def render_reference_pairs(
    pairs: list[tuple[str, str]],
    query_key: str,
    control_value: str | None = None,
) -> tuple[str, dict[str, dict[str, int]]]:
    lines = ["Reference pairs:"]
    spans: dict[str, dict[str, int]] = {}
    cursor = len(lines[0]) + 1
    for pair_index, (key, value) in enumerate(pairs):
        line = f"- {key}: {value}"
        spans[f"pair_{pair_index}"] = {
            "key": cursor + line.index(key),
            "colon": cursor + line.index(":"),
            "value": cursor + line.index(value),
        }
        lines.append(line)
        cursor += len(line) + 1
    if control_value is not None:
        line = f"Control word: {control_value}"
        spans["control"] = {"value": cursor + line.index(control_value)}
        lines.append(line)
        cursor += len(line) + 1
    query_line = f"- {query_key}:"
    spans["final_query"] = {"label": cursor + query_line.index(query_key)}
    lines.append(query_line)
    return "\n".join(lines), spans


def render_answer_absent_reference(
    pairs: list[tuple[str, str]],
    target_answer: str,
    distractor_answer: str,
    control_value: str,
    query_label: str,
) -> tuple[str, dict[str, dict[str, int]]]:
    lines = ["Reference pairs:"]
    spans: dict[str, dict[str, int]] = {}
    cursor = len(lines[0]) + 1
    for pair_index, (key, value) in enumerate(pairs):
        line = f"- {key}: {value}"
        spans[f"pair_{pair_index}"] = {
            "key": cursor + line.index(key),
            "colon": cursor + line.index(":"),
            "value": cursor + line.index(value),
        }
        lines.append(line)
        cursor += len(line) + 1

    for name, line in [
        ("control", f"Control word: {control_value}"),
        ("target", f"Required word: {target_answer}"),
        ("distractor", f"Distractor word: {distractor_answer}"),
        ("rule", "After the final label, write the required word."),
    ]:
        if name == "control":
            spans[name] = {"value": cursor + line.index(control_value)}
        elif name == "target":
            spans[name] = {"value": cursor + line.index(target_answer)}
        elif name == "distractor":
            spans[name] = {"value": cursor + line.index(distractor_answer)}
        lines.append(line)
        cursor += len(line) + 1

    query_line = f"- {query_label}:"
    spans["final_query"] = {"label": cursor + query_line.index(query_label)}
    lines.append(query_line)
    return "\n".join(lines), spans


def token_positions_from_spans(
    tokenizer: Any,
    prompt: str,
    source_pair_index: int,
    spans: dict[str, dict[str, int]],
) -> tuple[dict[str, int], int]:
    offset_enc = tokenizer(prompt, return_offsets_mapping=True)
    offsets = [[int(left), int(right)] for left, right in offset_enc["offset_mapping"]]
    pair_spans = spans[f"pair_{source_pair_index}"]
    positions = {
        "irrelevant_value": token_position_for_char(offsets, pair_spans["value"]),
        "irrelevant_key": token_position_for_char(offsets, pair_spans["key"]),
        "irrelevant_colon": token_position_for_char(offsets, pair_spans["colon"]),
        "final_query_label": token_position_for_char(offsets, spans["final_query"]["label"]),
    }
    if "control" in spans:
        positions["non_source_control_value"] = token_position_for_char(offsets, spans["control"]["value"])
    return positions, len(offset_enc["input_ids"])


def choose_distractor_index(query_index: int, source_index: int, pair_count: int, row_index: int) -> int:
    candidates = [index for index in range(pair_count) if index not in {query_index, source_index}]
    return candidates[row_index % len(candidates)]


def build_query_other_pair_rows(tokenizer: Any, scenario: Scenario, seed: int) -> list[dict[str, Any]]:
    keys = rotated_sample_pool(valid_words(tokenizer, KEY_WORDS), scenario.lexicon_offset)
    values = rotated_sample_pool(valid_words(tokenizer, VALUE_WORDS), scenario.lexicon_offset)
    needed_values = scenario.pair_count + (1 if scenario.surface == "query_other_pair_control_word" else 0)
    if len(keys) < scenario.pair_count or len(values) < needed_values:
        raise ValueError(
            f"scenario {scenario.name} lacks single-token words: "
            f"keys={len(keys)} values={len(values)} needed_values={needed_values}"
        )

    rng = random.Random(seed + scenario.lexicon_offset + scenario.pair_count)
    rows = []
    for row_index in range(scenario.row_count):
        local = random.Random(rng.randint(0, 10_000_000) + row_index)
        row_keys = local.sample(keys, scenario.pair_count)
        row_values = local.sample(values, needed_values)
        pairs = list(zip(row_keys, row_values[: scenario.pair_count], strict=True))
        control_value = row_values[-1] if scenario.surface == "query_other_pair_control_word" else None

        query_pair_index = row_index % scenario.pair_count
        source_pair_index = (query_pair_index + 1) % scenario.pair_count
        distractor_pair_index = choose_distractor_index(
            query_pair_index,
            source_pair_index,
            scenario.pair_count,
            row_index,
        )
        query_key, target_value = pairs[query_pair_index]
        _, distractor_value = pairs[distractor_pair_index]
        prompt, spans = render_reference_pairs(pairs, query_key, control_value)
        positions, token_length = token_positions_from_spans(tokenizer, prompt, source_pair_index, spans)
        row = {
            "id": f"mc005_v6_{scenario.name}_{row_index:03d}",
            "source_id": f"{scenario.name}_{row_index:03d}",
            "scenario": scenario.name,
            "surface": scenario.surface,
            "split": "same_grammar_v6",
            "mode": "same_grammar_decomposition",
            "pairs": [{"key": key, "value": value} for key, value in pairs],
            "pair_count": scenario.pair_count,
            "query_pair_index": query_pair_index,
            "source_focus_pair_index": source_pair_index,
            "distractor_pair_index": distractor_pair_index,
            "query_key": query_key,
            "target_value": target_value,
            "distractor_value": distractor_value,
            "source_focus_value": pairs[source_pair_index][1],
            "control_value": control_value,
            "target_token_id": first_token_id(tokenizer, target_value),
            "distractor_token_id": first_token_id(tokenizer, distractor_value),
            "positions": positions,
            "arm_keys": list(scenario.arm_keys),
            "rendered_prompt": prompt,
            "token_length": token_length,
        }
        rows.append(row)
    return rows


def build_answer_absent_rows(tokenizer: Any, scenario: Scenario, seed: int) -> list[dict[str, Any]]:
    keys = rotated_sample_pool(valid_words(tokenizer, KEY_WORDS), scenario.lexicon_offset)
    values = rotated_sample_pool(valid_words(tokenizer, VALUE_WORDS), scenario.lexicon_offset)
    needed_values = scenario.pair_count + 3
    if len(keys) < scenario.pair_count + 1 or len(values) < needed_values:
        raise ValueError(
            f"scenario {scenario.name} lacks single-token words: "
            f"keys={len(keys)} values={len(values)} needed_values={needed_values}"
        )

    rng = random.Random(seed + scenario.lexicon_offset + scenario.pair_count)
    rows = []
    for row_index in range(scenario.row_count):
        local = random.Random(rng.randint(0, 10_000_000) + row_index)
        sampled_keys = local.sample(keys, scenario.pair_count + 1)
        row_keys = sampled_keys[: scenario.pair_count]
        query_label = sampled_keys[-1]
        row_values = local.sample(values, needed_values)
        source_values = row_values[: scenario.pair_count]
        target_answer = row_values[scenario.pair_count]
        distractor_answer = row_values[scenario.pair_count + 1]
        control_value = row_values[scenario.pair_count + 2]
        pairs = list(zip(row_keys, source_values, strict=True))
        source_pair_index = row_index % scenario.pair_count
        prompt, spans = render_answer_absent_reference(
            pairs,
            target_answer,
            distractor_answer,
            control_value,
            query_label,
        )
        positions, token_length = token_positions_from_spans(tokenizer, prompt, source_pair_index, spans)
        rows.append(
            {
                "id": f"mc005_v6_{scenario.name}_{row_index:03d}",
                "source_id": f"{scenario.name}_{row_index:03d}",
                "scenario": scenario.name,
                "surface": scenario.surface,
                "split": "same_grammar_v6",
                "mode": "same_grammar_decomposition",
                "pairs": [{"key": key, "value": value} for key, value in pairs],
                "pair_count": scenario.pair_count,
                "query_pair_index": None,
                "source_focus_pair_index": source_pair_index,
                "distractor_pair_index": None,
                "query_key": query_label,
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


def build_rows(tokenizer: Any, scenario: Scenario, seed: int) -> list[dict[str, Any]]:
    if scenario.surface == "answer_absent_reference_control":
        return build_answer_absent_rows(tokenizer, scenario, seed)
    return build_query_other_pair_rows(tokenizer, scenario, seed)


def classify_arm(arm: dict[str, Any]) -> str:
    abs_delta = abs(float(arm["mean_delta"]))
    abs_win_change = abs(int(arm["target_win_loss"]))
    if abs_delta <= 0.50 and abs_win_change <= 1:
        return "clean"
    if abs_delta <= 1.00 and abs_win_change <= 2:
        return "weak"
    return "side_effect"


def classify_scenario(summary: dict[str, Any]) -> str:
    if int(summary["baseline_clean_rows"]) < 24:
        return "invalid_baseline"
    arm_labels = {key: arm["arm_label"] for key, arm in summary["arms"].items()}
    if all(label == "clean" for label in arm_labels.values()):
        return "clean_null"
    non_value_labels = [label for key, label in arm_labels.items() if key != "irrelevant_value"]
    if arm_labels.get("irrelevant_value") != "clean" and all(label == "clean" for label in non_value_labels):
        return "value_specific_boundary"
    return "structural_boundary"


def evaluate_scenario(
    scenario: Scenario,
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    arms: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm_key, arm_features in arms.items():
        arm_summary = summarize_arm(rows, baseline, arm_features)
        arm_summary["greedy_counts"] = greedy_counts(arm_features, rows)
        arm_summary["arm_label"] = classify_arm(arm_summary)
        arm_summaries[arm_key] = arm_summary

    summary = {
        "scenario": scenario.__dict__,
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
    summary["label"] = classify_scenario(summary)
    return summary


def summarize_suite(scenario_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts: dict[str, int] = {}
    arm_label_counts: dict[str, int] = {}
    for summary in scenario_summaries:
        label = str(summary["label"])
        label_counts[label] = label_counts.get(label, 0) + 1
        for arm in summary["arms"].values():
            arm_label = str(arm["arm_label"])
            arm_label_counts[arm_label] = arm_label_counts.get(arm_label, 0) + 1
    clean_all = all(summary["label"] == "clean_null" for summary in scenario_summaries)
    answer_absent = [
        summary
        for summary in scenario_summaries
        if summary["scenario"]["name"] == "answer_absent_reference_control"
    ]
    answer_absent_clean = all(summary["label"] == "clean_null" for summary in answer_absent)
    return {
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "scenario_count": len(scenario_summaries),
        "label_counts": label_counts,
        "arm_label_counts": arm_label_counts,
        "clean_null_count": sum(1 for summary in scenario_summaries if summary["label"] == "clean_null"),
        "value_specific_boundary_count": sum(
            1 for summary in scenario_summaries if summary["label"] == "value_specific_boundary"
        ),
        "structural_boundary_count": sum(
            1 for summary in scenario_summaries if summary["label"] == "structural_boundary"
        ),
        "answer_absent_clean": answer_absent_clean,
        "same_grammar_null_repaired": clean_all,
        "passed": clean_all,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_associative_lookup_same_grammar_v6")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
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
    for scenario_index, scenario in enumerate(SCENARIOS):
        rows = build_rows(tokenizer, scenario, args.seed + scenario_index * 1000)
        all_rows.extend(rows)
        baseline = score_rows_atlas(rows, tokenizer, model, args.batch_size)
        arms = {
            arm_key: score_rows_atlas(rows, tokenizer, model, args.batch_size, arm_key)
            for arm_key in scenario.arm_keys
        }
        summary = evaluate_scenario(scenario, rows, baseline, arms)
        scenario_summaries.append(summary)
        arm_bits = " ".join(
            f"{key}={summary['arms'][key]['mean_delta']:.4f}/{summary['arms'][key]['arm_label']}"
            for key in scenario.arm_keys
        )
        print(
            f"[same-grammar {scenario.name}] label={summary['label']} "
            f"clean={summary['baseline_clean_rows']}/{summary['rows']} {arm_bits}"
        )

    suite_summary = summarize_suite(scenario_summaries)
    elapsed = time.time() - started
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_same_grammar_v6",
        "model_id": args.model_id,
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "seed": args.seed,
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
