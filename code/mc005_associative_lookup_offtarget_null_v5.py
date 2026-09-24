#!/usr/bin/env python
"""MC005 V5 off-target null suite for the late-band source-value intervention."""

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
    Scenario as V4Scenario,
    build_scenario_rows,
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
class NullScenario:
    name: str
    surface: str
    role: str
    row_count: int = 32
    pair_count: int = 5
    lexicon_offset: int = 0


SCENARIOS = [
    NullScenario("same_grammar_query_other_pair", "same_grammar", "diagnostic"),
    NullScenario("reference_explicit_answer", "reference_explicit", "primary"),
    NullScenario("sentence_reference_explicit_answer", "sentence_reference_explicit", "primary"),
    NullScenario("no_reference_note_explicit_answer", "no_reference_note", "primary"),
    NullScenario("nonlookup_marker_answer", "nonlookup_marker", "primary"),
]


def render_reference_explicit_answer(
    pairs: list[tuple[str, str]],
    answer: str,
) -> tuple[str, list[dict[str, Any]]]:
    lines = ["Reference pairs:"]
    spans: list[dict[str, Any]] = []
    cursor = len(lines[0]) + 1
    for pair_index, (key, value) in enumerate(pairs):
        line = f"- {key}: {value}"
        spans.append({"source_index": pair_index, "value": value, "char_start": cursor + line.index(value)})
        lines.append(line)
        cursor += len(line) + 1
    lines.append("Instruction: ignore the reference pairs.")
    lines.append(f"The required answer word is {answer}.")
    lines.append("Answer:")
    return "\n".join(lines), spans


def render_sentence_reference_explicit_answer(
    pairs: list[tuple[str, str]],
    answer: str,
) -> tuple[str, list[dict[str, Any]]]:
    lines = ["Reference notes:"]
    spans: list[dict[str, Any]] = []
    cursor = len(lines[0]) + 1
    for pair_index, (key, value) in enumerate(pairs):
        line = f"The note for {key} says {value}."
        spans.append({"source_index": pair_index, "value": value, "char_start": cursor + line.index(value)})
        lines.append(line)
        cursor += len(line) + 1
    lines.append("Ignore the notes.")
    lines.append(f"The required answer word is {answer}.")
    lines.append("Answer:")
    return "\n".join(lines), spans


def render_no_reference_note_explicit_answer(
    source_values: list[str],
    answer: str,
) -> tuple[str, list[dict[str, Any]]]:
    line = (
        f"Background note: {source_values[0]}, {source_values[1]}, and {source_values[2]} "
        "are archived sample words."
    )
    lines = [
        line,
        f"The required answer word is {answer}.",
        "Answer:",
    ]
    spans = []
    for source_index, value in enumerate(source_values):
        spans.append({"source_index": source_index, "value": value, "char_start": line.index(value)})
    return "\n".join(lines), spans


def render_nonlookup_marker_answer(
    source_values: list[str],
    answer: str,
) -> tuple[str, list[dict[str, Any]]]:
    line = f"Unrelated word list: {source_values[0]} / {source_values[1]} / {source_values[2]}."
    lines = [
        line,
        f"Marker answer: {answer}",
        "Answer:",
    ]
    spans = []
    for source_index, value in enumerate(source_values):
        spans.append({"source_index": source_index, "value": value, "char_start": line.index(value)})
    return "\n".join(lines), spans


def encoded_positions(
    tokenizer: Any,
    prompt: str,
    spans: list[dict[str, Any]],
) -> tuple[dict[str, int], int]:
    offset_enc = tokenizer(prompt, return_offsets_mapping=True)
    offsets = [[int(left), int(right)] for left, right in offset_enc["offset_mapping"]]
    source_positions: dict[int, int] = {}
    for span in spans:
        source_positions[int(span["source_index"])] = token_position_for_char(offsets, int(span["char_start"]))
    return {
        "target_value": source_positions[0],
        "distractor_value": source_positions[1],
        "random_value": source_positions[2],
        "final_query_key": len(offset_enc["input_ids"]) - 1,
    }, len(offset_enc["input_ids"])


def build_same_grammar_rows(
    tokenizer: Any,
    scenario: NullScenario,
    seed: int,
) -> list[dict[str, Any]]:
    v4_scenario = V4Scenario(
        scenario.name,
        scenario.pair_count,
        "dash_colon",
        scenario.lexicon_offset,
        "off_target",
        scenario.row_count,
    )
    rows = build_scenario_rows(tokenizer, v4_scenario, seed)
    for row in rows:
        row["id"] = str(row["id"]).replace("mc005_v4_", "mc005_v5_")
        row["split"] = "offtarget_null_v5"
        row["v5_role"] = scenario.role
        row["surface"] = scenario.surface
    return rows


def build_repaired_null_rows(
    tokenizer: Any,
    scenario: NullScenario,
    seed: int,
) -> list[dict[str, Any]]:
    keys = rotated_sample_pool(valid_words(tokenizer, KEY_WORDS), scenario.lexicon_offset)
    values = rotated_sample_pool(valid_words(tokenizer, VALUE_WORDS), scenario.lexicon_offset)
    needed_values = max(7, scenario.pair_count + 2)
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
        source_values = row_values[: scenario.pair_count]
        target_answer = row_values[scenario.pair_count]
        distractor_answer = row_values[scenario.pair_count + 1]
        pairs = list(zip(row_keys, source_values, strict=True))

        if scenario.surface == "reference_explicit":
            prompt, spans = render_reference_explicit_answer(pairs, target_answer)
        elif scenario.surface == "sentence_reference_explicit":
            prompt, spans = render_sentence_reference_explicit_answer(pairs, target_answer)
        elif scenario.surface == "no_reference_note":
            prompt, spans = render_no_reference_note_explicit_answer(source_values[:3], target_answer)
        elif scenario.surface == "nonlookup_marker":
            prompt, spans = render_nonlookup_marker_answer(source_values[:3], target_answer)
        else:
            raise ValueError(f"unknown null surface {scenario.surface!r}")

        positions, token_length = encoded_positions(tokenizer, prompt, spans)
        rows.append(
            {
                "id": f"mc005_v5_{scenario.name}_{row_index:03d}",
                "source_id": f"{scenario.name}_{row_index:03d}",
                "scenario": scenario.name,
                "mode": "off_target_null",
                "split": "offtarget_null_v5",
                "surface": scenario.surface,
                "v5_role": scenario.role,
                "pairs": [{"key": key, "value": value} for key, value in pairs],
                "pair_count": scenario.pair_count,
                "query_key": "",
                "target_value": target_answer,
                "distractor_value": distractor_answer,
                "source_focus_value": source_values[0],
                "source_distractor_value": source_values[1],
                "random_value": source_values[2],
                "target_token_id": first_token_id(tokenizer, target_answer),
                "distractor_token_id": first_token_id(tokenizer, distractor_answer),
                "positions": positions,
                "rendered_prompt": prompt,
                "token_length": token_length,
            }
        )
    return rows


def build_null_rows(tokenizer: Any, scenario: NullScenario, seed: int) -> list[dict[str, Any]]:
    if scenario.surface == "same_grammar":
        return build_same_grammar_rows(tokenizer, scenario, seed)
    return build_repaired_null_rows(tokenizer, scenario, seed)


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
    scenario: NullScenario,
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    primary_arm: dict[str, dict[str, Any]],
    secondary_arm: dict[str, dict[str, Any]],
    random_arm: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        "scenario": scenario.__dict__,
        "rows": len(rows),
        "baseline_clean_rows": sum(1 for row in rows if baseline[row["id"]]["target_wins"]),
        "baseline_mean_margin": sum(
            float(baseline[row["id"]]["target_minus_distractor_margin"]) for row in rows
        )
        / max(1, len(rows)),
        "baseline_greedy_counts": greedy_counts(baseline, rows),
        "arms": {
            "primary": {
                **summarize_arm(rows, baseline, primary_arm),
                "greedy_counts": greedy_counts(primary_arm, rows),
            },
            "secondary": {
                **summarize_arm(rows, baseline, secondary_arm),
                "greedy_counts": greedy_counts(secondary_arm, rows),
            },
            "random": {
                **summarize_arm(rows, baseline, random_arm),
                "greedy_counts": greedy_counts(random_arm, rows),
            },
        },
    }
    summary["max_abs_mean_delta"] = max(abs(float(arm["mean_delta"])) for arm in summary["arms"].values())
    summary["max_abs_target_win_loss"] = max(abs(int(arm["target_win_loss"])) for arm in summary["arms"].values())
    summary["label"] = classify_null(summary)
    return summary


def summarize_suite(scenario_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts: dict[str, int] = {}
    for summary in scenario_summaries:
        label = str(summary["label"])
        label_counts[label] = label_counts.get(label, 0) + 1
    primary = [summary for summary in scenario_summaries if summary["scenario"]["role"] == "primary"]
    diagnostic = [summary for summary in scenario_summaries if summary["scenario"]["role"] == "diagnostic"]
    strict_passed = all(summary["label"] == "clean_null" for summary in scenario_summaries)
    primary_passed = all(summary["label"] == "clean_null" for summary in primary)
    return {
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "scenario_count": len(scenario_summaries),
        "primary_scenario_count": len(primary),
        "diagnostic_scenario_count": len(diagnostic),
        "label_counts": label_counts,
        "clean_null_count": sum(1 for summary in scenario_summaries if summary["label"] == "clean_null"),
        "primary_clean_null_count": sum(1 for summary in primary if summary["label"] == "clean_null"),
        "diagnostic_clean_null_count": sum(1 for summary in diagnostic if summary["label"] == "clean_null"),
        "primary_repaired_null_passed": primary_passed,
        "strict_null_suite_passed": strict_passed,
        "passed": strict_passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_associative_lookup_offtarget_null_v5")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
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
        rows = build_null_rows(tokenizer, scenario, args.seed + scenario_index * 1000)
        all_rows.extend(rows)
        baseline = score_rows_atlas(rows, tokenizer, model, args.batch_size)
        primary_arm = score_rows_atlas(rows, tokenizer, model, args.batch_size, "target_value")
        secondary_arm = score_rows_atlas(rows, tokenizer, model, args.batch_size, "distractor_value")
        random_arm = score_rows_atlas(rows, tokenizer, model, args.batch_size, "random_value")
        summary = evaluate_scenario(scenario, rows, baseline, primary_arm, secondary_arm, random_arm)
        scenario_summaries.append(summary)
        print(
            f"[null {scenario.name}] label={summary['label']} "
            f"role={scenario.role} clean={summary['baseline_clean_rows']}/{summary['rows']} "
            f"primary_delta={summary['arms']['primary']['mean_delta']:.4f} "
            f"secondary_delta={summary['arms']['secondary']['mean_delta']:.4f} "
            f"random_delta={summary['arms']['random']['mean_delta']:.4f} "
            f"max_abs_delta={summary['max_abs_mean_delta']:.4f}"
        )

    suite_summary = summarize_suite(scenario_summaries)
    elapsed = time.time() - started
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_offtarget_null_v5",
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
