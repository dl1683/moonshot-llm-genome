#!/usr/bin/env python
"""MC005 V8 final-marker punctuation robustness test."""

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
SOURCE_CONTROL_ARMS = {"source_value", "non_source_control_value", "earlier_neutral_colon"}
FINAL_LABEL_ARMS = {"final_label"}
FINAL_PUNCTUATION_ARMS = {"final_colon"}


@dataclass(frozen=True)
class Marker:
    name: str
    text: str


MARKERS = [
    Marker("answer", "Answer"),
    Marker("response", "Response"),
    Marker("output", "Output"),
    Marker("result", "Result"),
]


ARM_KEYS = (
    "source_value",
    "non_source_control_value",
    "earlier_neutral_colon",
    "final_label",
    "final_colon",
)


def add_line(lines: list[str], cursor: int, line: str) -> int:
    lines.append(line)
    return cursor + len(line) + 1


def render_prompt(
    pairs: list[tuple[str, str]],
    target_answer: str,
    distractor_answer: str,
    control_value: str,
    marker: Marker,
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
    final_line = f"{marker.text}:"
    positions["final_label_char"] = cursor + final_line.index(marker.text)
    positions["final_colon_char"] = cursor + final_line.index(":")
    cursor = add_line(lines, cursor, final_line)
    _ = cursor
    return "\n".join(lines), positions


def token_positions(
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


def build_rows_for_marker(
    tokenizer: Any,
    marker: Marker,
    seed: int,
    row_count: int,
    pair_count: int,
    lexicon_offset: int,
) -> list[dict[str, Any]]:
    keys = rotated_sample_pool(valid_words(tokenizer, KEY_WORDS), lexicon_offset)
    values = rotated_sample_pool(valid_words(tokenizer, VALUE_WORDS), lexicon_offset)
    needed_values = pair_count + 3
    if len(keys) < pair_count or len(values) < needed_values:
        raise ValueError(
            f"marker {marker.name} lacks single-token words: "
            f"keys={len(keys)} values={len(values)} needed_values={needed_values}"
        )

    rng = random.Random(seed + lexicon_offset + pair_count)
    rows = []
    for row_index in range(row_count):
        local = random.Random(rng.randint(0, 10_000_000) + row_index)
        row_keys = local.sample(keys, pair_count)
        row_values = local.sample(values, needed_values)
        source_values = row_values[:pair_count]
        target_answer = row_values[pair_count]
        distractor_answer = row_values[pair_count + 1]
        control_value = row_values[pair_count + 2]
        pairs = list(zip(row_keys, source_values, strict=True))
        source_pair_index = row_index % pair_count
        prompt, char_positions = render_prompt(
            pairs,
            target_answer,
            distractor_answer,
            control_value,
            marker,
            source_pair_index,
        )
        positions, token_length = token_positions(tokenizer, prompt, char_positions)
        rows.append(
            {
                "id": f"mc005_v8_seed{seed}_{marker.name}_{row_index:03d}",
                "source_id": f"seed{seed}_{marker.name}_{row_index:03d}",
                "scenario": marker.name,
                "marker": marker.name,
                "marker_text": marker.text,
                "seed": seed,
                "split": "final_marker_v8",
                "mode": "final_marker_punctuation",
                "pairs": [{"key": key, "value": value} for key, value in pairs],
                "pair_count": pair_count,
                "source_focus_pair_index": source_pair_index,
                "target_value": target_answer,
                "distractor_value": distractor_answer,
                "source_focus_value": pairs[source_pair_index][1],
                "control_value": control_value,
                "target_token_id": first_token_id(tokenizer, target_answer),
                "distractor_token_id": first_token_id(tokenizer, distractor_answer),
                "positions": positions,
                "arm_keys": list(ARM_KEYS),
                "rendered_prompt": prompt,
                "token_length": token_length,
            }
        )
    return rows


def classify_arm(arm: dict[str, Any]) -> str:
    abs_delta = abs(float(arm["mean_delta"]))
    abs_win_change = abs(int(arm["target_win_loss"]))
    if abs_delta <= 0.50 and abs_win_change <= 1:
        return "clean"
    if abs_delta <= 1.00 and abs_win_change <= 2:
        return "weak"
    return "side_effect"


def boundary_classes(arm_summaries: dict[str, dict[str, Any]]) -> set[str]:
    classes: set[str] = set()
    for arm_key, arm in arm_summaries.items():
        if arm["arm_label"] == "clean":
            continue
        if arm_key in SOURCE_CONTROL_ARMS:
            classes.add("structural_boundary")
        elif arm_key in FINAL_LABEL_ARMS:
            classes.add("final_label_boundary")
        elif arm_key in FINAL_PUNCTUATION_ARMS:
            classes.add("final_punctuation_boundary")
        else:
            classes.add("unknown_boundary")
    return classes


def classify_scenario(summary: dict[str, Any]) -> str:
    if int(summary["baseline_clean_rows"]) < 24:
        return "invalid_baseline"
    classes = boundary_classes(summary["arms"])
    if not classes:
        return "clean_null"
    if len(classes) == 1:
        return next(iter(classes))
    return "mixed_boundary"


def evaluate_scenario(
    marker: Marker,
    seed: int,
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    arms: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm_key, arm_features in arms.items():
        arm_summary = summarize_arm(rows, baseline, arm_features)
        arm_summary["greedy_counts"] = greedy_counts(arm_features, rows)
        arm_summary["arm_label"] = classify_arm(arm_summary)
        arm_summary["abs_target_win_change"] = abs(int(arm_summary["target_win_loss"]))
        arm_summaries[arm_key] = arm_summary

    summary = {
        "scenario": {
            "name": marker.name,
            "marker": marker.name,
            "marker_text": marker.text,
            "seed": seed,
            "row_count": len(rows),
            "arm_keys": list(ARM_KEYS),
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
    summary["boundary_classes"] = sorted(boundary_classes(arm_summaries))
    summary["label"] = classify_scenario(summary)
    return summary


def summarize_suite(scenario_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts: dict[str, int] = {}
    arm_label_counts: dict[str, int] = {}
    marker_counts: dict[str, dict[str, int]] = {}
    for summary in scenario_summaries:
        label = str(summary["label"])
        label_counts[label] = label_counts.get(label, 0) + 1
        marker = str(summary["scenario"]["marker"])
        marker_counts.setdefault(marker, {})
        marker_counts[marker][label] = marker_counts[marker].get(label, 0) + 1
        for arm in summary["arms"].values():
            arm_label = str(arm["arm_label"])
            arm_label_counts[arm_label] = arm_label_counts.get(arm_label, 0) + 1

    structural_nonclean = 0
    final_label_nonclean = 0
    final_punctuation_nonclean = 0
    marker_clean_counts: dict[str, int] = {marker.name: 0 for marker in MARKERS}
    for summary in scenario_summaries:
        if summary["label"] == "clean_null":
            marker_clean_counts[str(summary["scenario"]["marker"])] += 1
        for arm_key, arm in summary["arms"].items():
            if arm["arm_label"] == "clean":
                continue
            if arm_key in SOURCE_CONTROL_ARMS:
                structural_nonclean += 1
            elif arm_key in FINAL_LABEL_ARMS:
                final_label_nonclean += 1
            elif arm_key in FINAL_PUNCTUATION_ARMS:
                final_punctuation_nonclean += 1

    passed = all(summary["label"] == "clean_null" for summary in scenario_summaries)
    return {
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "seeds": SWEEP_SEEDS,
        "marker_count": len(MARKERS),
        "scenario_count": len(scenario_summaries),
        "label_counts": label_counts,
        "marker_label_counts": marker_counts,
        "arm_label_counts": arm_label_counts,
        "clean_null_count": sum(1 for summary in scenario_summaries if summary["label"] == "clean_null"),
        "structural_nonclean_arm_count": structural_nonclean,
        "final_label_nonclean_arm_count": final_label_nonclean,
        "final_punctuation_nonclean_arm_count": final_punctuation_nonclean,
        "all_structural_controls_clean": structural_nonclean == 0,
        "all_final_labels_clean": final_label_nonclean == 0,
        "all_final_punctuation_clean": final_punctuation_nonclean == 0,
        "marker_clean_counts": marker_clean_counts,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_associative_lookup_final_marker_v8")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--row-count", type=int, default=32)
    parser.add_argument("--pair-count", type=int, default=5)
    parser.add_argument("--lexicon-offset", type=int, default=0)
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
        for marker in MARKERS:
            rows = build_rows_for_marker(
                tokenizer,
                marker,
                seed,
                args.row_count,
                args.pair_count,
                args.lexicon_offset,
            )
            all_rows.extend(rows)
            baseline = score_rows_atlas(rows, tokenizer, model, args.batch_size)
            arms = {
                arm_key: score_rows_atlas(rows, tokenizer, model, args.batch_size, arm_key)
                for arm_key in ARM_KEYS
            }
            summary = evaluate_scenario(marker, seed, rows, baseline, arms)
            scenario_summaries.append(summary)
            arm_bits = " ".join(
                f"{key}={summary['arms'][key]['mean_delta']:.4f}/"
                f"{summary['arms'][key]['target_win_loss']}/"
                f"{summary['arms'][key]['arm_label']}"
                for key in ARM_KEYS
            )
            print(
                f"[final-marker seed={seed} marker={marker.name}] "
                f"label={summary['label']} clean={summary['baseline_clean_rows']}/{summary['rows']} {arm_bits}"
            )

    suite_summary = summarize_suite(scenario_summaries)
    elapsed = time.time() - started
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_final_marker_v8",
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
