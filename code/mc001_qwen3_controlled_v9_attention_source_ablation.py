#!/usr/bin/env python
"""Attention-source ablation audit for MC-001 Qwen3-0.6B.

V8 closed dense answer-prefix steering. V9 tests a path-local question: does
blocking attention to the user hint source tokens reduce wrong-hint agreement
more cleanly than matched question or answer-instruction source masking?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from mc001_qwen3_controlled import (
    CARD_ID,
    MANIFEST_DIR,
    MODEL_ID,
    RESULT_DIR,
    classify,
    forward_features,
    generate_one,
    load_model,
    option_token_ids,
    parse_answer,
    summarize_many,
)
from mc001_qwen3_controlled_v2 import (
    add_margin_metadata,
    iter_records,
    margin_bin_counts,
    write_manifest,
)
from mc001_qwen3_controlled_v3 import (
    answer_distribution,
    rows_digest,
    score_cell,
    summarize_margin_bins,
)
from mc001_qwen3_controlled_v6_prefill_controls import validation_filters
from mc001_qwen3_smoke import format_for_model


ValidationFilter = Callable[[dict[str, Any]], bool]


def line_span(rendered: str, marker: str) -> tuple[int, int] | None:
    start = rendered.find(marker)
    if start < 0:
        return None
    end = rendered.find("\n", start)
    if end < 0:
        end = len(rendered)
    return start, end


def token_positions_for_span(tokenizer: Any, rendered: str, start: int, end: int) -> list[int]:
    encoded = tokenizer(rendered, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    positions: list[int] = []
    for index, offset in enumerate(offsets):
        left, right = [int(value) for value in offset]
        if left == right == 0:
            continue
        if right > start and left < end:
            positions.append(index)
    return positions


def token_positions_for_spans(tokenizer: Any, rendered: str, spans: list[tuple[int, int]]) -> list[int]:
    out: set[int] = set()
    for start, end in spans:
        out.update(token_positions_for_span(tokenizer, rendered, start, end))
    return sorted(out)


def find_hint_answer_spans(rendered: str, hint_span: tuple[int, int] | None, record: dict[str, Any]) -> list[tuple[int, int]]:
    if hint_span is None:
        return []
    if record["condition"] == "correct_hint":
        target = record["correct_answer"]
    elif record["condition"].startswith("wrong_") or record["condition"] == "anti_wrong":
        target = record["wrong_answer"]
    else:
        return []
    start, end = hint_span
    segment = rendered[start:end]
    spans: list[tuple[int, int]] = []
    for index, char in enumerate(segment):
        if char != target:
            continue
        before = segment[index - 1] if index > 0 else " "
        after = segment[index + 1] if index + 1 < len(segment) else " "
        if before.isalnum() or after.isalnum():
            continue
        spans.append((start + index, start + index + 1))
    return spans


def matched_positions(candidates: list[int], count: int) -> list[int]:
    if count <= 0 or not candidates:
        return []
    if len(candidates) <= count:
        return list(candidates)
    if count == 1:
        return [candidates[len(candidates) // 2]]
    selected = []
    for i in range(count):
        index = round(i * (len(candidates) - 1) / (count - 1))
        selected.append(candidates[index])
    return sorted(set(selected))


def random_matched_positions(row_id: str, candidates: list[int], excluded: set[int], count: int) -> list[int]:
    pool = [pos for pos in candidates if pos not in excluded]
    if count <= 0 or not pool:
        return []
    seed = int(hashlib.sha256(row_id.encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(seed)
    if len(pool) <= count:
        return sorted(pool)
    return sorted(rng.sample(pool, count))


def build_path_positions(tokenizer: Any, rendered: str, record: dict[str, Any]) -> dict[str, list[int]]:
    hint_span = line_span(rendered, "User hint:")
    question_span = line_span(rendered, "Question:")
    answer_span = line_span(rendered, "Answer with only one letter:")

    hint_line = token_positions_for_span(tokenizer, rendered, *hint_span) if hint_span else []
    hint_answer = token_positions_for_spans(tokenizer, rendered, find_hint_answer_spans(rendered, hint_span, record))
    question = token_positions_for_span(tokenizer, rendered, *question_span) if question_span else []
    answer_instruction = token_positions_for_span(tokenizer, rendered, *answer_span) if answer_span else []

    all_content = sorted(set(question + answer_instruction + hint_line))
    excluded = set(hint_line)
    return {
        "hint_line": hint_line,
        "hint_answer": hint_answer,
        "question_matched_hint_line": matched_positions(question, len(hint_line)),
        "answer_instruction_matched_hint_line": matched_positions(answer_instruction, len(hint_line)),
        "random_matched_hint_line": random_matched_positions(record["id"], all_content, excluded, len(hint_line)),
    }


def render_row(tokenizer: Any, record: dict[str, Any]) -> dict[str, Any]:
    rendered = format_for_model(tokenizer, record["prompt"])
    path_positions = build_path_positions(tokenizer, rendered, record)
    return {
        **record,
        "variant": "controlled_v9_attention_source_ablation",
        "rendered_prompt": rendered,
        "path_position_counts": {key: len(value) for key, value in path_positions.items()},
        "path_positions": path_positions,
    }


def generate_with_attention_mask(
    record: dict[str, Any],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    mask_positions: list[int],
) -> dict[str, Any]:
    inputs = tokenizer(record["rendered_prompt"], return_tensors="pt").to(model.device)
    if mask_positions:
        attention_mask = inputs["attention_mask"].clone()
        valid = [pos for pos in mask_positions if 0 <= pos < attention_mask.shape[1]]
        if valid:
            attention_mask[:, valid] = 0
            inputs["attention_mask"] = attention_mask
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
    parsed = parse_answer(completion, record)
    label = classify(parsed, record["correct_answer"], record["wrong_answer"])
    return {"completion": completion, "parsed_answer": parsed, "label": label}


def arm_specs() -> list[dict[str, Any]]:
    return [
        {"name": "baseline", "kind": "baseline"},
        {"name": "mask_hint_line", "kind": "mask", "position_group": "hint_line"},
        {"name": "mask_hint_answer", "kind": "mask", "position_group": "hint_answer"},
        {"name": "mask_question_matched_hint_line", "kind": "mask", "position_group": "question_matched_hint_line"},
        {"name": "mask_answer_instruction_matched_hint_line", "kind": "mask", "position_group": "answer_instruction_matched_hint_line"},
        {"name": "mask_random_matched_hint_line", "kind": "mask", "position_group": "random_matched_hint_line"},
    ]


def summarize_filtered(arm_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for filter_name, predicate in validation_filters().items():
        output[filter_name] = {
            arm: summarize_many([row for row in rows if predicate(row)])
            for arm, rows in sorted(arm_rows.items())
        }
    return output


def split_condition_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(f"{row['split']}::{row['condition']}" for row in rows).items()))


def position_count_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({key for row in rows for key in row["path_position_counts"]})
    output: dict[str, Any] = {}
    for group in groups:
        counts = [int(row["path_position_counts"].get(group, 0)) for row in rows]
        output[group] = {
            "min": min(counts) if counts else None,
            "max": max(counts) if counts else None,
            "mean": float(np.mean(counts)) if counts else None,
            "nonzero_n": sum(count > 0 for count in counts),
        }
    return output


def run_arms(
    eval_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    arm_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    specs = arm_specs()
    for spec in specs:
        arm_name = spec["name"]
        for index, row in enumerate(eval_rows, start=1):
            mask_positions: list[int] = []
            if spec["kind"] == "baseline":
                base = baseline_by_id[row["id"]]
                result = {key: base[key] for key in ["completion", "parsed_answer", "label"]}
            else:
                mask_positions = row["path_positions"][spec["position_group"]]
                result = generate_with_attention_mask(row, tokenizer, model, max_new_tokens, mask_positions)
            arm_row = {
                **row,
                **result,
                "arm": arm_name,
                "mask_position_group": spec.get("position_group"),
                "mask_position_count": len(mask_positions),
            }
            arm_rows[arm_name].append(arm_row)
            print(f"[v9 gen {arm_name} {index:03d}/{len(eval_rows):03d}] {row['id']} mask={len(mask_positions)} -> {arm_row['parsed_answer']!r} {arm_row['label']}")

    return {
        "arm_settings": specs,
        "summary_by_arm": {key: summarize_many(value) for key, value in sorted(arm_rows.items())},
        "summary_by_arm_margin_bin": {key: summarize_margin_bins(value) for key, value in sorted(arm_rows.items())},
        "summary_filtered": summarize_filtered(arm_rows),
        "score_by_arm": {key: score_cell(value) for key, value in sorted(arm_rows.items())},
        "answer_distribution_by_arm": {key: answer_distribution(value) for key, value in sorted(arm_rows.items())},
        "records": [row for rows in arm_rows.values() for row in rows],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DIR / "mc001_qwen3_0p6b_controlled_v9_attention_source_ablation_manifest.jsonl")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(9)
    random.seed(9)
    np.random.seed(9)

    records = iter_records()
    digest = write_manifest(args.manifest, records)
    if args.manifest_only:
        print(f"wrote {len(records)} records to {args.manifest}")
        print(f"sha256={digest}")
        return 0

    tokenizer, model = load_model(args.model_id)
    token_ids = option_token_ids(tokenizer)
    hidden_indices = [13, 14]
    started = time.time()

    rendered_rows = [render_row(tokenizer, record) for record in records]
    baseline_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rendered_rows, start=1):
        result = generate_one(row, tokenizer, model, args.max_new_tokens)
        baseline_rows.append({**row, **result})
        print(f"[v9 baseline {index:03d}/{len(rendered_rows):03d}] {row['id']} -> {result['parsed_answer']!r} {result['label']}")

    features: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(baseline_rows, start=1):
        features[row["id"]] = forward_features(row, tokenizer, model, token_ids, hidden_indices)
        print(f"[v9 features {index:03d}/{len(baseline_rows):03d}] {row['id']}")

    baseline_rows = add_margin_metadata(baseline_rows, features)
    discovery = [row for row in baseline_rows if row["split"] == "discovery"]
    calibration = [row for row in baseline_rows if row["split"] == "calibration"]
    holdout = [row for row in baseline_rows if row["split"] == "holdout"]
    paraphrase = [row for row in baseline_rows if row["split"] == "paraphrase_holdout"]
    eval_rows = [
        row
        for row in baseline_rows
        if row["split"] in {"calibration", "holdout", "paraphrase_holdout"}
    ]

    generation = run_arms(eval_rows, baseline_rows, tokenizer, model, args.max_new_tokens)

    output = {
        "card_id": CARD_ID,
        "run_type": "qwen3_0p6b_controlled_v9_attention_source_ablation",
        "model_id": args.model_id,
        "manifest": str(args.manifest),
        "manifest_sha256": digest,
        "records_digest": rows_digest(records),
        "records_n": len(records),
        "hidden_indices_for_baseline_features": hidden_indices,
        "max_new_tokens": args.max_new_tokens,
        "elapsed_s": time.time() - started,
        "baseline_summary": summarize_many(baseline_rows),
        "position_count_summary": position_count_summary(baseline_rows),
        "candidate_counts": {
            "discovery": len(discovery),
            "calibration": len(calibration),
            "holdout": len(holdout),
            "paraphrase_holdout": len(paraphrase),
            "validation_rows": len(eval_rows),
        },
        "candidate_margin_bin_counts": {
            "discovery": margin_bin_counts(discovery),
            "calibration": margin_bin_counts(calibration),
            "holdout": margin_bin_counts(holdout),
            "paraphrase_holdout": margin_bin_counts(paraphrase),
            "validation_wrong_hint": margin_bin_counts([
                row
                for row in eval_rows
                if row["condition"].startswith("wrong_") and row["label"] in {"truth_following", "user_agreement_error"}
            ]),
        },
        "selection_rule": {
            "splits": ["calibration", "holdout", "paraphrase_holdout"],
            "conditions": sorted({row["condition"] for row in eval_rows}),
            "n": len(eval_rows),
            "split_condition_counts": split_condition_counts(eval_rows),
            "purpose": "attention-source path-local ablation audit",
        },
        "generation_validation": generation,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"mc001_qwen3_0p6b_controlled_v9_attention_source_ablation_{timestamp}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output_path": str(output_path),
        "baseline": output["baseline_summary"]["overall"],
        "candidate_counts": output["candidate_counts"],
        "position_count_summary": output["position_count_summary"],
        "summary_filtered": output["generation_validation"]["summary_filtered"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
