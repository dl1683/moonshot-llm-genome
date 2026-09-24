#!/usr/bin/env python
"""MC005 V12 Qwen3-0.6B marker-specific answer-absent null diagnostic."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc001_qwen3_controlled_v10_head_localization import set_eager_attention
from mc005_associative_lookup_reliability_v4 import (
    SELECTED_BAND,
    SELECTED_LAYERS,
    greedy_counts,
    rotated_sample_pool,
    score_rows_atlas,
)
from mc005_associative_lookup_response_marker_v11_weak_null_diagnostic import (
    LOW_MARGIN_THRESHOLD,
    classify_null_scaled,
    row_diagnostics,
    summarize_row_diagnostics,
)
from mc005_associative_lookup_response_marker_v9 import (
    Scenario,
    answer_absent_positions,
    arm_is_clean,
    arm_is_weak,
    evaluate_scenario,
    sample_disjoint_row_words,
)
from mc005_associative_lookup_source_edge import (
    CARD_ID,
    KEY_WORDS,
    RESULT_DIR,
    VALUE_WORDS,
    first_token_id,
    summarize_arm,
    valid_words,
)


MODEL_ID = "Qwen/Qwen3-0.6B"
DIAGNOSTIC_SEEDS = [17, 23, 31, 37, 41]
ROW_COUNT = 128
PAIR_COUNT = 5
ARM_KEYS = (
    "source_value",
    "non_source_control_value",
    "earlier_neutral_colon",
    "final_label",
    "final_colon",
)


@dataclass(frozen=True)
class Marker:
    name: str
    text: str


MARKERS = [
    Marker("response", "Response"),
    Marker("output", "Output"),
    Marker("answer", "Answer"),
    Marker("result", "Result"),
]


def add_line(lines: list[str], cursor: int, line: str) -> int:
    lines.append(line)
    return cursor + len(line) + 1


def render_answer_absent_prompt_for_marker(
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


def build_rows_for_marker(
    tokenizer: Any,
    marker: Marker,
    seed: int,
    row_count: int,
    pair_count: int,
    lexicon_offset: int = 0,
) -> list[dict[str, Any]]:
    keys = rotated_sample_pool(valid_words(tokenizer, KEY_WORDS), lexicon_offset)
    values = rotated_sample_pool(valid_words(tokenizer, VALUE_WORDS), lexicon_offset)
    needed_values = pair_count + 3
    if len(keys) < pair_count or len(values) < needed_values:
        raise ValueError(
            f"marker {marker.name} lacks single-token words: "
            f"keys={len(keys)} values={len(values)} needed_values={needed_values}"
        )
    rng = random.Random(seed + lexicon_offset + pair_count + 200)
    rows = []
    for row_index in range(row_count):
        local = random.Random(rng.randint(0, 10_000_000) + row_index)
        row_keys, row_values = sample_disjoint_row_words(local, keys, values, pair_count, needed_values)
        source_values = row_values[:pair_count]
        target_answer = row_values[pair_count]
        distractor_answer = row_values[pair_count + 1]
        control_value = row_values[pair_count + 2]
        pairs = list(zip(row_keys, source_values, strict=True))
        source_pair_index = row_index % pair_count
        prompt, char_positions = render_answer_absent_prompt_for_marker(
            pairs,
            target_answer,
            distractor_answer,
            control_value,
            marker,
            source_pair_index,
        )
        positions, token_length = answer_absent_positions(tokenizer, prompt, char_positions)
        rows.append(
            {
                "id": f"mc005_v12_seed{seed}_{marker.name}_{row_index:03d}",
                "source_id": f"seed{seed}_{marker.name}_{row_index:03d}",
                "scenario": f"answer_absent_{marker.name}_null",
                "seed": seed,
                "split": "marker_specificity_v12",
                "mode": "answer_absent_null",
                "marker": marker.name,
                "marker_text": marker.text,
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


def evaluate_marker(
    marker: Marker,
    seed: int,
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    arms: dict[str, dict[str, dict[str, Any]]],
    scaled: bool,
) -> dict[str, Any]:
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm_key, arm_features in arms.items():
        arm_summary = summarize_arm(rows, baseline, arm_features)
        arm_summary["greedy_counts"] = greedy_counts(arm_features, rows)
        arm_summary["abs_target_win_change"] = abs(int(arm_summary["target_win_loss"]))
        if arm_is_clean(arm_summary):
            arm_summary["arm_label"] = "clean"
        elif arm_is_weak(arm_summary):
            arm_summary["arm_label"] = "weak"
        else:
            arm_summary["arm_label"] = "side_effect"
        arm_summaries[arm_key] = arm_summary
    summary = {
        "scenario": {
            "name": f"answer_absent_{marker.name}_null",
            "mode": "null",
            "seed": seed,
            "marker": marker.name,
            "marker_text": marker.text,
            "pair_count": PAIR_COUNT,
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
    if scaled:
        summary["label"] = classify_null_scaled(summary)
        summary["baseline_floor"] = int(0.75 * int(summary["rows"]))
    else:
        scenario = Scenario(
            f"answer_absent_{marker.name}_null",
            "null",
            PAIR_COUNT,
            ARM_KEYS,
            row_count=len(rows),
        )
        summary["label"] = evaluate_scenario(scenario, seed, rows, baseline, arms)["label"]
        summary["baseline_floor"] = 24
    return summary


def summarize_suite(
    full_summaries: list[dict[str, Any]],
    first32_summaries: list[dict[str, Any]],
    first32_diag_summaries: dict[str, dict[int, dict[str, dict[str, Any]]]],
    markers: list[Marker],
    seeds: list[int],
    model_id: str,
) -> dict[str, Any]:
    full_label_counts = dict(Counter(str(summary["label"]) for summary in full_summaries))
    first32_label_counts = dict(Counter(str(summary["label"]) for summary in first32_summaries))
    marker_full_counts: dict[str, dict[str, int]] = {}
    marker_first32_counts: dict[str, dict[str, int]] = {}
    marker_clean_seed_counts: dict[str, int] = {marker.name: 0 for marker in markers}
    for summary in full_summaries:
        marker = str(summary["scenario"]["marker"])
        label = str(summary["label"])
        marker_full_counts.setdefault(marker, {})
        marker_full_counts[marker][label] = marker_full_counts[marker].get(label, 0) + 1
        if label == "clean_null":
            marker_clean_seed_counts[marker] += 1
    for summary in first32_summaries:
        marker = str(summary["scenario"]["marker"])
        label = str(summary["label"])
        marker_first32_counts.setdefault(marker, {})
        marker_first32_counts[marker][label] = marker_first32_counts[marker].get(label, 0) + 1

    all_seed_clean_markers = [
        marker.name
        for marker in markers
        if marker.name in marker_clean_seed_counts and marker_clean_seed_counts[marker.name] == len(seeds)
    ]
    response_clean = "response" in all_seed_clean_markers
    output_clean = "output" in all_seed_clean_markers
    if output_clean and not response_clean:
        diagnostic_class = "output_repairs_boundary"
    elif output_clean and response_clean:
        diagnostic_class = "response_and_output_clean"
    elif any(marker != "response" for marker in all_seed_clean_markers):
        diagnostic_class = "alternate_marker_repairs_boundary"
    else:
        diagnostic_class = "no_clean_marker"

    response_seed23 = next(
        summary
        for summary in first32_summaries
        if summary["scenario"]["marker"] == "response" and int(summary["scenario"]["seed"]) == 23
    )
    response_seed23_control = response_seed23["arms"]["non_source_control_value"]
    response_seed23_v11_pattern = (
        response_seed23["label"] == "weak_null"
        and int(response_seed23_control["target_win_loss"]) <= -2
        and abs(float(response_seed23_control["mean_delta"])) <= 0.50
    )
    response_seed23_flips = first32_diag_summaries["response"][23]["non_source_control_value"]["flip_rows"]
    return {
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "model_id": model_id,
        "seeds": seeds,
        "markers": [marker.name for marker in markers],
        "row_count_per_marker_seed": int(full_summaries[0]["rows"]) if full_summaries else 0,
        "full_label_counts": full_label_counts,
        "first32_label_counts": first32_label_counts,
        "marker_full_label_counts": marker_full_counts,
        "marker_first32_label_counts": marker_first32_counts,
        "marker_clean_seed_counts": marker_clean_seed_counts,
        "all_seed_clean_markers": all_seed_clean_markers,
        "response_seed23_first32_v11_pattern_reproduced": response_seed23_v11_pattern,
        "response_seed23_first32_non_source_control_flip_count": len(response_seed23_flips),
        "response_seed23_first32_non_source_control_all_flips_low_margin": bool(response_seed23_flips)
        and all(float(row["baseline_abs_margin"]) <= LOW_MARGIN_THRESHOLD for row in response_seed23_flips),
        "diagnostic_class": diagnostic_class,
    }


def parse_seeds(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def select_markers(raw: str) -> list[Marker]:
    wanted = {part.strip().lower() for part in raw.split(",") if part.strip()}
    markers = [marker for marker in MARKERS if marker.name in wanted]
    missing = wanted.difference({marker.name for marker in markers})
    if missing:
        raise ValueError(f"unknown markers: {sorted(missing)}")
    return markers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_0p6b_response_marker_v12_marker_specificity")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--row-count", type=int, default=ROW_COUNT)
    parser.add_argument("--pair-count", type=int, default=PAIR_COUNT)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DIAGNOSTIC_SEEDS))
    parser.add_argument("--markers", default=",".join(marker.name for marker in MARKERS))
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    markers = select_markers(args.markers)

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
    full_summaries = []
    first32_summaries = []
    all_rows = []
    row_diagnostic_summaries: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    first32_row_diagnostic_summaries: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    row_diagnostic_details: dict[str, dict[int, dict[str, list[dict[str, Any]]]]] = {}
    for marker in markers:
        row_diagnostic_summaries[marker.name] = {}
        first32_row_diagnostic_summaries[marker.name] = {}
        row_diagnostic_details[marker.name] = {}
        for seed in seeds:
            rows = build_rows_for_marker(tokenizer, marker, seed, args.row_count, args.pair_count)
            all_rows.extend(rows)
            baseline = score_rows_atlas(rows, tokenizer, model, args.batch_size)
            arms = {
                arm_key: score_rows_atlas(rows, tokenizer, model, args.batch_size, arm_key)
                for arm_key in ARM_KEYS
            }
            full_summary = evaluate_marker(marker, seed, rows, baseline, arms, scaled=True)
            first32_rows = rows[:32]
            first32_summary = evaluate_marker(marker, seed, first32_rows, baseline, arms, scaled=False)
            full_summaries.append(full_summary)
            first32_summaries.append(first32_summary)

            row_diagnostic_summaries[marker.name][seed] = {}
            first32_row_diagnostic_summaries[marker.name][seed] = {}
            row_diagnostic_details[marker.name][seed] = {}
            for arm_key, arm_features in arms.items():
                per_row = row_diagnostics(rows, baseline, arm_features)
                row_diagnostic_details[marker.name][seed][arm_key] = per_row
                row_diagnostic_summaries[marker.name][seed][arm_key] = summarize_row_diagnostics(per_row)
                first32_row_diagnostic_summaries[marker.name][seed][arm_key] = summarize_row_diagnostics(
                    per_row[:32]
                )

            full_bits = " ".join(
                f"{key}={full_summary['arms'][key]['mean_delta']:.4f}/"
                f"{full_summary['arms'][key]['target_win_loss']}/"
                f"{full_summary['arms'][key]['arm_label']}"
                for key in ARM_KEYS
            )
            first32_bits = " ".join(
                f"{key}={first32_summary['arms'][key]['mean_delta']:.4f}/"
                f"{first32_summary['arms'][key]['target_win_loss']}/"
                f"{first32_summary['arms'][key]['arm_label']}"
                for key in ARM_KEYS
            )
            print(
                f"[v12 marker={marker.name} seed={seed} full{len(rows)}] "
                f"label={full_summary['label']} clean={full_summary['baseline_clean_rows']}/"
                f"{full_summary['rows']} {full_bits}"
            )
            print(
                f"[v12 marker={marker.name} seed={seed} first32] "
                f"label={first32_summary['label']} clean={first32_summary['baseline_clean_rows']}/"
                f"{first32_summary['rows']} {first32_bits}"
            )

    suite_summary = summarize_suite(
        full_summaries,
        first32_summaries,
        first32_row_diagnostic_summaries,
        markers,
        seeds,
        args.model_id,
    )
    elapsed = time.time() - started
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_response_marker_v12_0p6b_marker_specificity",
        "model_id": args.model_id,
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "elapsed_s": elapsed,
        "summary": suite_summary,
        "full_summaries": full_summaries,
        "first32_summaries": first32_summaries,
        "row_diagnostic_summaries": row_diagnostic_summaries,
        "first32_row_diagnostic_summaries": first32_row_diagnostic_summaries,
        "row_diagnostic_details": row_diagnostic_details,
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
