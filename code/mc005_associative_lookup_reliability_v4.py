#!/usr/bin/env python
"""MC005 V4 reliability atlas for the late-band source-value control surface."""

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

from mc001_qwen3_controlled_v10_head_localization import (
    install_source_mask_hook,
    set_eager_attention,
    tokenizer_padding_side,
)
from mc005_associative_lookup_late_band_v3 import (
    BANDS,
    final_query_position,
    render_prompt,
    token_position_for_char,
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


SELECTED_BAND = "late_20_26"
SELECTED_LAYERS = BANDS[SELECTED_BAND]


@dataclass(frozen=True)
class Scenario:
    name: str
    pair_count: int
    layout: str
    lexicon_offset: int
    mode: str
    row_count: int = 32


SCENARIOS = [
    Scenario("pair3_dash_base", 3, "dash_colon", 0, "lookup"),
    Scenario("pair5_dash_base", 5, "dash_colon", 0, "lookup"),
    Scenario("pair8_dash_base", 8, "dash_colon", 0, "lookup"),
    Scenario("pair5_arrow_base", 5, "arrow", 0, "lookup"),
    Scenario("pair8_arrow_base", 8, "arrow", 0, "lookup"),
    Scenario("pair5_sentence_base", 5, "sentence", 0, "lookup"),
    Scenario("pair5_arrow_holdout_lexicon", 5, "arrow", 24, "lookup"),
    Scenario("pair8_dash_holdout_lexicon", 8, "dash_colon", 24, "lookup"),
    Scenario("offtarget_pair5_dash", 5, "dash_colon", 0, "off_target"),
]


def rotated_sample_pool(items: list[str], offset: int) -> list[str]:
    if not items:
        return []
    offset = offset % len(items)
    return items[offset:] + items[:offset]


def render_prompt_v4(pairs: list[tuple[str, str]], query_key: str, layout: str) -> tuple[str, list[dict[str, Any]]]:
    if layout in {"dash_colon", "arrow"}:
        return render_prompt(pairs, query_key, layout)
    if layout == "sentence":
        lines = ["Reference sentences:"]
        value_spans: list[dict[str, Any]] = []
        cursor = len(lines[0]) + 1
        for pair_index, (key, value) in enumerate(pairs):
            line = f"The value for {key} is {value}."
            value_start = cursor + line.index(value)
            value_spans.append({"pair_index": pair_index, "value": value, "char_start": value_start})
            lines.append(line)
            cursor += len(line) + 1
        lines.append(f"The value for {query_key} is")
        return "\n".join(lines), value_spans
    raise ValueError(f"unknown layout {layout!r}")


def final_query_position_v4(prompt: str, offsets: list[list[int]], query_key: str, layout: str) -> int:
    if layout in {"dash_colon", "arrow"}:
        return final_query_position(prompt, offsets, query_key, layout)
    if layout == "sentence":
        needle = f"The value for {query_key} is"
        char_index = prompt.rindex(needle) + needle.index(query_key)
        return token_position_for_char(offsets, char_index)
    raise ValueError(f"unknown layout {layout!r}")


def build_scenario_rows(
    tokenizer: Any,
    scenario: Scenario,
    seed: int,
) -> list[dict[str, Any]]:
    base_keys = rotated_sample_pool(valid_words(tokenizer, KEY_WORDS), scenario.lexicon_offset)
    base_values = rotated_sample_pool(valid_words(tokenizer, VALUE_WORDS), scenario.lexicon_offset)
    if len(base_keys) < scenario.pair_count or len(base_values) < scenario.pair_count:
        raise ValueError(
            f"scenario {scenario.name} lacks single-token words: "
            f"keys={len(base_keys)} values={len(base_values)} pair_count={scenario.pair_count}"
        )

    rng = random.Random(seed + scenario.lexicon_offset + scenario.pair_count)
    rows = []
    for row_index in range(scenario.row_count):
        local = random.Random(rng.randint(0, 10_000_000) + row_index)
        row_keys = local.sample(base_keys, scenario.pair_count)
        row_values = local.sample(base_values, scenario.pair_count)
        pairs = list(zip(row_keys, row_values, strict=True))
        query_pair_index = row_index % scenario.pair_count
        source_focus_pair_index = query_pair_index
        if scenario.mode == "off_target":
            source_focus_pair_index = (query_pair_index + 1) % scenario.pair_count
        distractor_pair_index = (query_pair_index + 1 + (row_index % (scenario.pair_count - 1))) % scenario.pair_count
        if distractor_pair_index == query_pair_index:
            distractor_pair_index = (query_pair_index + 1) % scenario.pair_count
        source_distractor_index = (source_focus_pair_index + 1) % scenario.pair_count
        if source_distractor_index == query_pair_index and scenario.mode == "off_target":
            source_distractor_index = (source_distractor_index + 1) % scenario.pair_count
        unavailable = {query_pair_index, source_focus_pair_index, source_distractor_index}
        random_candidates = [index for index in range(scenario.pair_count) if index not in unavailable]
        if not random_candidates:
            random_candidates = [index for index in range(scenario.pair_count) if index != source_focus_pair_index]
        random_value_index = random_candidates[row_index % len(random_candidates)]

        query_key, output_target_value = pairs[query_pair_index]
        _, output_distractor_value = pairs[distractor_pair_index]
        prompt, value_spans = render_prompt_v4(pairs, query_key, scenario.layout)
        offset_enc = tokenizer(prompt, return_offsets_mapping=True)
        offsets = [[int(left), int(right)] for left, right in offset_enc["offset_mapping"]]
        input_ids = [int(value) for value in offset_enc["input_ids"]]
        value_positions: dict[str, int] = {}
        for span in value_spans:
            pair_index = int(span["pair_index"])
            pos = token_position_for_char(offsets, int(span["char_start"]))
            value_positions[str(pair_index)] = int(pos)
        rows.append(
            {
                "id": f"mc005_v4_{scenario.name}_{row_index:03d}",
                "source_id": f"{scenario.name}_{row_index:03d}",
                "scenario": scenario.name,
                "mode": scenario.mode,
                "split": "atlas",
                "layout": scenario.layout,
                "lexicon_offset": scenario.lexicon_offset,
                "pairs": [{"key": key, "value": value} for key, value in pairs],
                "pair_count": scenario.pair_count,
                "query_pair_index": query_pair_index,
                "source_focus_pair_index": source_focus_pair_index,
                "distractor_pair_index": distractor_pair_index,
                "source_distractor_index": source_distractor_index,
                "random_value_index": random_value_index,
                "query_key": query_key,
                "target_value": output_target_value,
                "distractor_value": output_distractor_value,
                "source_focus_value": pairs[source_focus_pair_index][1],
                "source_distractor_value": pairs[source_distractor_index][1],
                "random_value": pairs[random_value_index][1],
                "target_token_id": first_token_id(tokenizer, output_target_value),
                "distractor_token_id": first_token_id(tokenizer, output_distractor_value),
                "positions": {
                    "target_value": value_positions[str(source_focus_pair_index)],
                    "distractor_value": value_positions[str(source_distractor_index)],
                    "random_value": value_positions[str(random_value_index)],
                    "final_query_key": final_query_position_v4(prompt, offsets, query_key, scenario.layout),
                },
                "rendered_prompt": prompt,
                "token_length": len(input_ids),
            }
        )
    return rows


def shifted_positions_for_batch(rows: list[dict[str, Any]], source_key: str, seq_lens: list[int], max_len: int) -> list[list[int]]:
    shifted: list[list[int]] = []
    for row, seq_len in zip(rows, seq_lens, strict=True):
        shifted.append([int(row["positions"][source_key]) + (max_len - seq_len)])
    return shifted


def next_token_features(logits: torch.Tensor, row: dict[str, Any]) -> dict[str, Any]:
    target_logit = float(logits[int(row["target_token_id"])])
    distractor_logit = float(logits[int(row["distractor_token_id"])])
    margin = target_logit - distractor_logit
    greedy_token_id = int(torch.argmax(logits).item())
    if greedy_token_id == int(row["target_token_id"]):
        greedy_label = "target"
    elif greedy_token_id == int(row["distractor_token_id"]):
        greedy_label = "distractor"
    else:
        greedy_label = "other"
    return {
        "target_logit": target_logit,
        "distractor_logit": distractor_logit,
        "target_minus_distractor_margin": margin,
        "target_wins": margin > 0.0,
        "greedy_token_id": greedy_token_id,
        "greedy_token": "",
        "greedy_label": greedy_label,
    }


def score_rows_atlas(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    source_key: str | None = None,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    heads = list(range(model.config.num_attention_heads))
    with tokenizer_padding_side(tokenizer, "left"):
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            inputs = tokenizer([row["rendered_prompt"] for row in batch], return_tensors="pt", padding=True).to(
                model.device
            )
            handles = []
            if source_key is not None:
                seq_lens = [int(value) for value in inputs["attention_mask"].sum(dim=1).detach().cpu().tolist()]
                max_len = int(inputs["input_ids"].shape[-1])
                shifted = shifted_positions_for_batch(batch, source_key, seq_lens, max_len)
                handles = [
                    install_source_mask_hook(model, int(layer), heads, shifted)
                    for layer in SELECTED_LAYERS
                ]
            try:
                with torch.inference_mode():
                    out = model(**inputs, use_cache=False, logits_to_keep=1)
            finally:
                for handle in handles:
                    handle.remove()
            for index, row in enumerate(batch):
                feature = next_token_features(out.logits[index, -1, :].detach().float(), row)
                feature["greedy_token"] = tokenizer.decode([feature["greedy_token_id"]])
                features[row["id"]] = feature
    return features


def greedy_counts(features: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"target": 0, "distractor": 0, "other": 0}
    for row in rows:
        counts[str(features[row["id"]]["greedy_label"])] += 1
    return counts


def classify_lookup(summary: dict[str, Any]) -> str:
    target = summary["arms"]["target"]
    distractor = summary["arms"]["distractor"]
    random_arm = summary["arms"]["random"]
    if summary["baseline_clean_rows"] < 16:
        return "breaks"
    works = (
        target["mean_delta"] <= -1.0
        and target["target_win_loss"] >= 3
        and target["mean_delta"] <= distractor["mean_delta"] - 0.50
        and target["mean_delta"] <= random_arm["mean_delta"] - 0.50
    )
    if works:
        return "works"
    if target["mean_delta"] < 0:
        return "weak"
    return "breaks"


def classify_off_target(summary: dict[str, Any]) -> str:
    target = summary["arms"]["target"]
    if abs(float(target["mean_delta"])) <= 0.50 and int(target["target_win_loss"]) <= 1:
        return "clean_null"
    return "side_effect"


def evaluate_scenario(
    scenario: Scenario,
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    target_arm: dict[str, dict[str, Any]],
    distractor_arm: dict[str, dict[str, Any]],
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
            "target": {**summarize_arm(rows, baseline, target_arm), "greedy_counts": greedy_counts(target_arm, rows)},
            "distractor": {
                **summarize_arm(rows, baseline, distractor_arm),
                "greedy_counts": greedy_counts(distractor_arm, rows),
            },
            "random": {**summarize_arm(rows, baseline, random_arm), "greedy_counts": greedy_counts(random_arm, rows)},
        },
    }
    summary["label"] = classify_off_target(summary) if scenario.mode == "off_target" else classify_lookup(summary)
    return summary


def summarize_atlas(scenario_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts: dict[str, int] = {}
    for summary in scenario_summaries:
        label = str(summary["label"])
        label_counts[label] = label_counts.get(label, 0) + 1
    lookup_scenarios = [summary for summary in scenario_summaries if summary["scenario"]["mode"] == "lookup"]
    off_target_scenarios = [summary for summary in scenario_summaries if summary["scenario"]["mode"] == "off_target"]
    return {
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "scenario_count": len(scenario_summaries),
        "label_counts": label_counts,
        "lookup_work_count": sum(1 for summary in lookup_scenarios if summary["label"] == "works"),
        "lookup_count": len(lookup_scenarios),
        "off_target_clean_null_count": sum(1 for summary in off_target_scenarios if summary["label"] == "clean_null"),
        "off_target_count": len(off_target_scenarios),
        "all_lookup_scenarios_work": all(summary["label"] == "works" for summary in lookup_scenarios),
        "all_off_target_scenarios_clean_null": all(
            summary["label"] == "clean_null" for summary in off_target_scenarios
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_associative_lookup_reliability_v4")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11)
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
        rows = build_scenario_rows(tokenizer, scenario, args.seed + scenario_index * 1000)
        all_rows.extend(rows)
        baseline = score_rows_atlas(rows, tokenizer, model, args.batch_size)
        target_arm = score_rows_atlas(rows, tokenizer, model, args.batch_size, "target_value")
        distractor_arm = score_rows_atlas(rows, tokenizer, model, args.batch_size, "distractor_value")
        random_arm = score_rows_atlas(rows, tokenizer, model, args.batch_size, "random_value")
        summary = evaluate_scenario(scenario, rows, baseline, target_arm, distractor_arm, random_arm)
        scenario_summaries.append(summary)
        print(
            f"[atlas {scenario.name}] label={summary['label']} "
            f"clean={summary['baseline_clean_rows']}/{summary['rows']} "
            f"target_delta={summary['arms']['target']['mean_delta']:.4f} "
            f"target_loss={summary['arms']['target']['target_win_loss']}"
        )

    atlas_summary = summarize_atlas(scenario_summaries)
    atlas_summary["passed"] = (
        atlas_summary["all_lookup_scenarios_work"]
        and atlas_summary["all_off_target_scenarios_clean_null"]
    )
    elapsed = time.time() - started
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_reliability_v4",
        "model_id": args.model_id,
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "seed": args.seed,
        "elapsed_s": elapsed,
        "summary": atlas_summary,
        "scenario_summaries": scenario_summaries,
        "rows": all_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)
    print(json.dumps({**atlas_summary, "output_path": str(output_path)}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
