#!/usr/bin/env python
"""Fast answer-option logit steering sweep for MC-001 Qwen3-0.6B."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mc001_qwen3_smoke import classify, summarize
from mc001_qwen3_steer import (
    MODEL_ID,
    RESULT_DIR,
    build_direction,
    latest_smoke_result,
    layer_module_for_hidden_index,
    load_model,
    load_records,
    split_records,
)


def parse_float_list(raw: str) -> list[float]:
    return [float(value) for value in raw.split(",") if value.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value.strip()]


def letter_token_ids(tokenizer: Any, letter: str) -> list[int]:
    ids: set[int] = set()
    for text in (letter, f" {letter}", f"\n{letter}", f"{letter}.", f" {letter}."):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if encoded:
            ids.add(int(encoded[0]))
    return sorted(ids)


def build_option_token_map(tokenizer: Any) -> dict[str, list[int]]:
    return {letter: letter_token_ids(tokenizer, letter) for letter in ["A", "B", "C", "D"]}


def score_options(logits: torch.Tensor, option_token_ids: dict[str, list[int]]) -> dict[str, float]:
    scores = {}
    for letter, ids in option_token_ids.items():
        token_logits = logits[torch.tensor(ids, device=logits.device)]
        scores[letter] = float(token_logits.max().item())
    return scores


def score_record(
    record: dict[str, Any],
    tokenizer: Any,
    model: Any,
    hidden_index: int,
    direction: torch.Tensor,
    alpha: float,
    median_hidden_norm: float,
    option_token_ids: dict[str, list[int]],
) -> dict[str, Any]:
    delta = direction.to(model.device, dtype=model.dtype) * (alpha * median_hidden_norm)

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0].clone()
            hidden[:, -1, :] = hidden[:, -1, :] + delta
            return (hidden,) + output[1:]
        hidden = output.clone()
        hidden[:, -1, :] = hidden[:, -1, :] + delta
        return hidden

    handle = layer_module_for_hidden_index(model, hidden_index).register_forward_hook(hook)
    try:
        inputs = tokenizer(record["rendered_prompt"], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model(**inputs, use_cache=False)
    finally:
        handle.remove()

    logits = out.logits[0, -1, :].detach().float()
    option_scores = score_options(logits, option_token_ids)
    parsed = max(option_scores, key=option_scores.get)
    label = classify(parsed, record["correct_answer"], record["wrong_answer"])
    correct_score = option_scores[record["correct_answer"]]
    wrong_score = option_scores[record["wrong_answer"]]
    return {
        **record,
        "hidden_index": hidden_index,
        "hook_layer_index": hidden_index - 1,
        "alpha": alpha,
        "option_scores": option_scores,
        "correct_minus_wrong_logit": correct_score - wrong_score,
        "parsed_answer": parsed,
        "label": label,
    }


def summarize_with_margin(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize(rows)
    margins = [float(row["correct_minus_wrong_logit"]) for row in rows]
    summary["mean_correct_minus_wrong_logit"] = float(np.mean(margins)) if margins else None
    summary["median_correct_minus_wrong_logit"] = float(np.median(margins)) if margins else None
    return summary


def summarize_grid(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"hidden_{row['hidden_index']}__alpha_{row['alpha']}"
        grouped[key].append(row)
    return {
        key: summarize_with_margin(group_rows)
        for key, group_rows in sorted(
            grouped.items(),
            key=lambda kv: (
                int(kv[0].split("__", 1)[0].replace("hidden_", "")),
                float(kv[0].split("alpha_", 1)[1]),
            ),
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=None)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--hidden-indices", default="5,7,14")
    parser.add_argument("--alphas", default="-3,-1,0,1,3")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    args = parser.parse_args()

    source = args.result or latest_smoke_result()
    records = load_records(source)
    train, eval_rows = split_records(records)
    tokenizer, model = load_model(args.model_id)
    option_token_ids = build_option_token_map(tokenizer)
    hidden_indices = parse_int_list(args.hidden_indices)
    alphas = parse_float_list(args.alphas)

    started = time.time()
    rows: list[dict[str, Any]] = []
    directions = {}
    for hidden_index in hidden_indices:
        direction_info = build_direction(train, tokenizer, model, hidden_index)
        direction = torch.tensor(direction_info["direction"])
        directions[str(hidden_index)] = {k: v for k, v in direction_info.items() if k != "direction"}
        for alpha in alphas:
            for index, record in enumerate(eval_rows, start=1):
                row = score_record(
                    record,
                    tokenizer,
                    model,
                    hidden_index,
                    direction,
                    alpha,
                    direction_info["median_hidden_norm"],
                    option_token_ids,
                )
                rows.append(row)
                print(
                    f"[h{hidden_index:02d} a{alpha:+.1f} {index:03d}/{len(eval_rows):03d}] "
                    f"{record['id']} -> {row['parsed_answer']} margin={row['correct_minus_wrong_logit']:+.3f} {row['label']}"
                )

    output = {
        "card_id": "MC-001",
        "run_type": "qwen3_0p6b_logit_direction_sweep",
        "model_id": args.model_id,
        "source_result": str(source),
        "hidden_indices": hidden_indices,
        "alphas": alphas,
        "train_n": len(train),
        "eval_n_per_grid_cell": len(eval_rows),
        "option_token_ids": option_token_ids,
        "directions": directions,
        "elapsed_s": time.time() - started,
        "summary_by_grid_cell": summarize_grid(rows),
        "records": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"mc001_qwen3_0p6b_logit_steer_factual_ladder_{stamp}.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(output["summary_by_grid_cell"], indent=2, ensure_ascii=True))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
