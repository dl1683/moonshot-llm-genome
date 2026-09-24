#!/usr/bin/env python
"""Input-mask semantics audit for MC-001 Qwen3-0.6B.

V9 produced a strong coarse source-token effect by zeroing source-token
positions in the input attention mask. V10/V11 preserved the prompt and only
blocked generation-query attention to those source positions, recovering a much
weaker effect. V12 compares those semantics directly against prompt rewrites.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mc001_qwen3_controlled import (
    CARD_ID,
    MANIFEST_DIR,
    MODEL_ID,
    RESULT_DIR,
    classify,
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
from mc001_qwen3_controlled_v9_attention_source_ablation import generate_with_attention_mask
from mc001_qwen3_controlled_v10_head_localization import (
    batched_next_token_features,
    position_count_summary,
    render_row,
    set_eager_attention,
    split_condition_counts,
    summarize_filtered,
)
from mc001_qwen3_controlled_v11_layer_band_source import (
    Band,
    SourceBandSpec,
    generate_with_layer_band_mask,
)
from mc001_qwen3_smoke import format_for_model


def target_hint_answer(record: dict[str, Any]) -> str | None:
    if record["condition"] == "correct_hint":
        return str(record["correct_answer"])
    if record["condition"].startswith("wrong_") or record["condition"] == "anti_wrong":
        return str(record["wrong_answer"])
    return None


def find_user_hint_line(prompt: str) -> tuple[int, int] | None:
    start = prompt.find("User hint:")
    if start < 0:
        return None
    end = prompt.find("\n", start)
    if end < 0:
        end = len(prompt)
    return start, end


def standalone_target_spans(line: str, target: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(target)}(?![A-Za-z0-9])")
    for match in pattern.finditer(line):
        spans.append((match.start(), match.end()))
    return spans


def rewrite_hint_answer_in_line(line: str, target: str, replacement: str) -> tuple[str, int]:
    spans = standalone_target_spans(line, target)
    if not spans:
        return line, 0
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        parts.append(line[cursor:start])
        parts.append(replacement)
        cursor = end
    parts.append(line[cursor:])
    return "".join(parts), len(spans)


def rewrite_prompt(record: dict[str, Any], rewrite_kind: str) -> tuple[str, dict[str, Any]]:
    prompt = str(record["prompt"])
    hint_span = find_user_hint_line(prompt)
    if hint_span is None:
        return prompt, {"rewrite_applied": False, "rewrite_count": 0}

    start, end = hint_span
    before = prompt[:start]
    hint_line = prompt[start:end]
    after = prompt[end:]
    target = target_hint_answer(record)

    if rewrite_kind == "delete_hint_line":
        if before.endswith("\n") and after.startswith("\n"):
            return before + after[1:], {"rewrite_applied": True, "rewrite_count": 1}
        return before + after, {"rewrite_applied": True, "rewrite_count": 1}

    if rewrite_kind == "neutralize_hint_line":
        neutral = "User hint: The suggested option is hidden. Solve independently."
        return before + neutral + after, {"rewrite_applied": True, "rewrite_count": 1}

    if target is None:
        return prompt, {"rewrite_applied": False, "rewrite_count": 0}

    if rewrite_kind == "delete_hint_answer":
        new_line, count = rewrite_hint_answer_in_line(hint_line, target, "")
        return before + new_line + after, {"rewrite_applied": count > 0, "rewrite_count": count}

    if rewrite_kind == "placeholder_hint_answer_X":
        new_line, count = rewrite_hint_answer_in_line(hint_line, target, "X")
        return before + new_line + after, {"rewrite_applied": count > 0, "rewrite_count": count}

    raise ValueError(f"unknown rewrite_kind: {rewrite_kind}")


def generate_with_rewrite(
    row: dict[str, Any],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    rewrite_kind: str,
) -> dict[str, Any]:
    rewritten_prompt, metadata = rewrite_prompt(row, rewrite_kind)
    rendered = format_for_model(tokenizer, rewritten_prompt)
    rewritten_row = {**row, "prompt": rewritten_prompt, "rendered_prompt": rendered}
    result = generate_one(rewritten_row, tokenizer, model, max_new_tokens)
    return {**result, **metadata, "rewritten_prompt": rewritten_prompt}


def all_layer_spec(source_group: str, num_layers: int) -> SourceBandSpec:
    return SourceBandSpec(source_group, Band("all_L00_L27", tuple(range(num_layers))))


def arm_specs() -> list[dict[str, Any]]:
    return [
        {"name": "baseline", "kind": "baseline"},
        {"name": "input_mask_hint_line", "kind": "input_mask", "position_group": "hint_line"},
        {"name": "input_mask_hint_answer", "kind": "input_mask", "position_group": "hint_answer"},
        {"name": "input_mask_hint_non_answer", "kind": "input_mask", "position_group": "hint_non_answer"},
        {"name": "input_mask_answer_instruction_matched_hint_line", "kind": "input_mask", "position_group": "answer_instruction_matched_hint_line"},
        {"name": "input_mask_random_matched_hint_line", "kind": "input_mask", "position_group": "random_matched_hint_line"},
        {"name": "input_mask_answer_instruction_matched_hint_answer", "kind": "input_mask", "position_group": "answer_instruction_matched_hint_answer"},
        {"name": "input_mask_random_matched_hint_answer", "kind": "input_mask", "position_group": "random_matched_hint_answer"},
        {"name": "rewrite_delete_hint_line", "kind": "rewrite", "rewrite_kind": "delete_hint_line"},
        {"name": "rewrite_neutralize_hint_line", "kind": "rewrite", "rewrite_kind": "neutralize_hint_line"},
        {"name": "rewrite_delete_hint_answer", "kind": "rewrite", "rewrite_kind": "delete_hint_answer"},
        {"name": "rewrite_placeholder_hint_answer_X", "kind": "rewrite", "rewrite_kind": "placeholder_hint_answer_X"},
        {"name": "query_mask_hint_line_all", "kind": "query_mask_all_layers", "source_group": "hint_line"},
        {"name": "query_mask_hint_answer_all", "kind": "query_mask_all_layers", "source_group": "hint_answer"},
        {"name": "query_mask_hint_non_answer_all", "kind": "query_mask_all_layers", "source_group": "hint_non_answer"},
    ]


def run_generation_validation(
    eval_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    heads = list(range(model.config.num_attention_heads))
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    arm_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for spec in specs:
        arm_name = spec["name"]
        for index, row in enumerate(eval_rows, start=1):
            extra: dict[str, Any] = {}
            if spec["kind"] == "baseline":
                base = baseline_by_id[row["id"]]
                result = {key: base[key] for key in ["completion", "parsed_answer", "label"]}
            elif spec["kind"] == "input_mask":
                positions = row["path_positions"].get(spec["position_group"], [])
                result = generate_with_attention_mask(row, tokenizer, model, max_new_tokens, positions)
                extra = {
                    "position_group": spec["position_group"],
                    "mask_position_count": len(positions),
                }
            elif spec["kind"] == "rewrite":
                result = generate_with_rewrite(row, tokenizer, model, max_new_tokens, spec["rewrite_kind"])
                extra = {"rewrite_kind": spec["rewrite_kind"]}
            elif spec["kind"] == "query_mask_all_layers":
                source_group = spec["source_group"]
                source_spec = all_layer_spec(source_group, model.config.num_hidden_layers)
                result = generate_with_layer_band_mask(row, tokenizer, model, max_new_tokens, source_spec, heads)
                extra = {
                    "source_group": source_group,
                    "layer_band": source_spec.band.name,
                    "mask_layers": list(source_spec.band.layers),
                    "mask_heads": heads,
                    "source_position_count": len(row["path_positions"].get(source_group, [])),
                }
            else:
                raise ValueError(f"unknown arm kind: {spec['kind']}")

            arm_row = {
                **row,
                **result,
                **extra,
                "arm": arm_name,
                "arm_kind": spec["kind"],
            }
            arm_rows[arm_name].append(arm_row)
            print(f"[v12 gen {arm_name} {index:03d}/{len(eval_rows):03d}] {row['id']} -> {arm_row['parsed_answer']!r} {arm_row['label']}")

    return {
        "arm_settings": specs,
        "summary_by_arm": {key: summarize_many(value) for key, value in sorted(arm_rows.items())},
        "summary_by_arm_margin_bin": {key: summarize_margin_bins(value) for key, value in sorted(arm_rows.items())},
        "summary_filtered": summarize_filtered(arm_rows),
        "score_by_arm": {key: score_cell(value) for key, value in sorted(arm_rows.items())},
        "answer_distribution_by_arm": {key: answer_distribution(value) for key, value in sorted(arm_rows.items())},
        "records": [row for rows in arm_rows.values() for row in rows],
    }


def compact_arm_summary(generation: dict[str, Any], section: str = "wrong_hint_agreement_favored") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    filtered = generation["summary_filtered"][section]
    for arm, value in filtered.items():
        overall = value["overall"] if isinstance(value, dict) and "overall" in value else value
        out.append({
            "arm": arm,
            "n": int(overall["n"]),
            "parseable_n": int(overall["parseable_n"]),
            "truth_following_n": int(overall["truth_following_n"]),
            "user_agreement_error_n": int(overall["user_agreement_error_n"]),
            "other_error_n": int(overall["other_error_n"]),
            "truth_following_rate_parseable": overall["truth_following_rate_parseable"],
            "user_agreement_error_rate_parseable": overall["user_agreement_error_rate_parseable"],
            "other_error_rate_parseable": overall["other_error_rate_parseable"],
        })
    return sorted(
        out,
        key=lambda row: (
            row["truth_following_n"],
            -row["user_agreement_error_n"],
            -row["other_error_n"],
            row["arm"],
        ),
        reverse=True,
    )


def rewrite_count_summary(generation: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in generation["arm_settings"]:
        if arm["kind"] != "rewrite":
            continue
        rows = [row for row in generation["records"] if row["arm"] == arm["name"]]
        counts = [int(row.get("rewrite_count", 0)) for row in rows]
        out[arm["name"]] = {
            "min": min(counts) if counts else None,
            "max": max(counts) if counts else None,
            "mean": float(np.mean(counts)) if counts else None,
            "applied_n": sum(bool(row.get("rewrite_applied")) for row in rows),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DIR / "mc001_qwen3_0p6b_controlled_v12_input_mask_semantics_manifest.jsonl")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-arms", type=int, default=0)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(12)
    random.seed(12)
    np.random.seed(12)

    records = iter_records()
    digest = write_manifest(args.manifest, records)
    if args.manifest_only:
        print(f"wrote {len(records)} records to {args.manifest}")
        print(f"sha256={digest}")
        return 0

    tokenizer, model = load_model(args.model_id)
    set_eager_attention(model)
    token_ids = option_token_ids(tokenizer)
    started = time.time()

    rendered_rows = [render_row(tokenizer, record) for record in records]
    baseline_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rendered_rows, start=1):
        result = generate_one(row, tokenizer, model, args.max_new_tokens)
        baseline_rows.append({**row, **result})
        print(f"[v12 baseline {index:03d}/{len(rendered_rows):03d}] {row['id']} -> {result['parsed_answer']!r} {result['label']}")

    baseline_features = batched_next_token_features(
        baseline_rows,
        tokenizer,
        model,
        token_ids,
        args.batch_size,
    )
    baseline_rows = add_margin_metadata(baseline_rows, baseline_features)

    validation_rows = [
        row
        for row in baseline_rows
        if row["split"] in {"calibration", "holdout", "paraphrase_holdout"}
    ]
    hard_rows = [
        row
        for row in validation_rows
        if row["condition"].startswith("wrong_") and row["baseline_margin_bin"] == "agreement_favored"
    ]
    side_rows = [
        row
        for row in validation_rows
        if row["condition"] in {"no_hint", "correct_hint", "anti_wrong"}
    ]
    eval_rows_by_id = {row["id"]: row for row in hard_rows + side_rows}
    eval_rows = list(eval_rows_by_id.values())
    print(f"[v12 selection] validation={len(validation_rows)} hard={len(hard_rows)} side={len(side_rows)} eval={len(eval_rows)}")

    specs = arm_specs()
    if args.max_arms:
        specs = [specs[0]] + specs[1 : args.max_arms + 1]
    print("[v12 arms]")
    for spec in specs:
        print(f"  {spec['name']} kind={spec['kind']}")

    generation = run_generation_validation(
        eval_rows,
        baseline_rows,
        specs,
        tokenizer,
        model,
        args.max_new_tokens,
    )

    discovery = [row for row in baseline_rows if row["split"] == "discovery"]
    calibration = [row for row in baseline_rows if row["split"] == "calibration"]
    holdout = [row for row in baseline_rows if row["split"] == "holdout"]
    paraphrase = [row for row in baseline_rows if row["split"] == "paraphrase_holdout"]

    output = {
        "card_id": CARD_ID,
        "run_type": "qwen3_0p6b_controlled_v12_input_mask_semantics",
        "model_id": args.model_id,
        "attention_implementation": "eager",
        "manifest": str(args.manifest),
        "manifest_sha256": digest,
        "records_digest": rows_digest(records),
        "records_n": len(records),
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "elapsed_s": time.time() - started,
        "baseline_summary": summarize_many(baseline_rows),
        "position_count_summary": position_count_summary(baseline_rows),
        "candidate_counts": {
            "discovery": len(discovery),
            "calibration": len(calibration),
            "holdout": len(holdout),
            "paraphrase_holdout": len(paraphrase),
            "validation_rows": len(validation_rows),
            "hard_agreement_favored_wrong_hint_rows": len(hard_rows),
            "side_effect_rows": len(side_rows),
            "generation_eval_rows": len(eval_rows),
        },
        "candidate_margin_bin_counts": {
            "discovery": margin_bin_counts(discovery),
            "calibration": margin_bin_counts(calibration),
            "holdout": margin_bin_counts(holdout),
            "paraphrase_holdout": margin_bin_counts(paraphrase),
            "validation_wrong_hint": margin_bin_counts([
                row
                for row in validation_rows
                if row["condition"].startswith("wrong_") and row["label"] in {"truth_following", "user_agreement_error"}
            ]),
        },
        "selection_rule": {
            "hard_rows": "validation split, wrong-hint condition, baseline_margin_bin == agreement_favored",
            "side_rows": "validation split, condition in no_hint/correct_hint/anti_wrong",
            "split_condition_counts": split_condition_counts(eval_rows),
        },
        "semantics_tested": {
            "input_mask": "V9-style source positions set to 0 in the input attention_mask before generation",
            "rewrite": "literal prompt rewrites before chat-template rendering",
            "query_mask_all_layers": "V11-style prompt-preserving generation-query source mask across all layers",
        },
        "generation_validation": generation,
        "rewrite_count_summary": rewrite_count_summary(generation),
        "hard_bin_top": compact_arm_summary(generation, "wrong_hint_agreement_favored"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"mc001_qwen3_0p6b_controlled_v12_input_mask_semantics_{timestamp}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output_path": str(output_path),
        "baseline": output["baseline_summary"]["overall"],
        "candidate_counts": output["candidate_counts"],
        "hard_bin_top": output["hard_bin_top"][:15],
        "rewrite_count_summary": output["rewrite_count_summary"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
