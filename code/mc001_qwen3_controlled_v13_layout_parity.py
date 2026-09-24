#!/usr/bin/env python
"""Length/tokenization parity sanity check for MC-001 Qwen3-0.6B.

V12 showed that literal prompt rewrites nearly match V9-style input masking.
V13 tests the remaining caveat: whether the small residual gap is plausibly
caused by prompt length or tokenization layout rather than source semantics.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mc001_qwen3_controlled import (
    CARD_ID,
    MANIFEST_DIR,
    MODEL_ID,
    RESULT_DIR,
    generate_one,
    load_model,
    option_token_ids,
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
from mc001_qwen3_controlled_v12_input_mask_semantics import (
    compact_arm_summary,
    find_user_hint_line,
    rewrite_prompt as v12_rewrite_prompt,
)
from mc001_qwen3_smoke import format_for_model


def rendered_token_count(tokenizer: Any, prompt: str) -> int:
    rendered = format_for_model(tokenizer, prompt)
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def char_matched_neutral_line(hint_line: str) -> str:
    """Return a neutral replacement with exactly the original character length."""
    base = "User hint: hidden."
    filler = " x"
    target_len = len(hint_line)
    if target_len <= len(base):
        return base[:target_len]
    repeats = ((target_len - len(base)) // len(filler)) + 2
    return (base + filler * repeats)[:target_len]


def token_matched_neutral_line(
    tokenizer: Any,
    before: str,
    hint_line: str,
    after: str,
) -> tuple[str, dict[str, Any]]:
    """Return a neutral replacement with rendered prompt token count closest to original."""
    original_prompt = before + hint_line + after
    target_tokens = rendered_token_count(tokenizer, original_prompt)
    base = "User hint: hidden."

    best_line = base
    best_tokens = rendered_token_count(tokenizer, before + best_line + after)
    best_key = (abs(best_tokens - target_tokens), abs(len(best_line) - len(hint_line)), best_tokens)

    # Repeated " x" is usually token-count smooth for Qwen tokenizers. The loop
    # intentionally over-searches; tokenizer calls are cheap relative to generation.
    for repeats in range(0, 256):
        candidate = base + (" x" * repeats)
        tokens = rendered_token_count(tokenizer, before + candidate + after)
        key = (abs(tokens - target_tokens), abs(len(candidate) - len(hint_line)), tokens)
        if key < best_key:
            best_line = candidate
            best_tokens = tokens
            best_key = key
        if tokens == target_tokens and len(candidate) >= len(hint_line):
            break

    return best_line, {
        "original_rendered_token_count": target_tokens,
        "rewritten_rendered_token_count": best_tokens,
        "rendered_token_delta": best_tokens - target_tokens,
        "original_hint_line_chars": len(hint_line),
        "rewritten_hint_line_chars": len(best_line),
        "hint_line_char_delta": len(best_line) - len(hint_line),
    }


def rewrite_prompt_v13(
    record: dict[str, Any],
    tokenizer: Any,
    rewrite_kind: str,
) -> tuple[str, dict[str, Any]]:
    prompt = str(record["prompt"])
    hint_span = find_user_hint_line(prompt)
    original_tokens = rendered_token_count(tokenizer, prompt)

    if hint_span is None:
        return prompt, {
            "rewrite_applied": False,
            "rewrite_count": 0,
            "original_rendered_token_count": original_tokens,
            "rewritten_rendered_token_count": original_tokens,
            "rendered_token_delta": 0,
            "original_hint_line_chars": 0,
            "rewritten_hint_line_chars": 0,
            "hint_line_char_delta": 0,
        }

    start, end = hint_span
    before = prompt[:start]
    hint_line = prompt[start:end]
    after = prompt[end:]

    if rewrite_kind == "char_matched_neutral_hint_line":
        replacement = char_matched_neutral_line(hint_line)
        rewritten = before + replacement + after
        rewritten_tokens = rendered_token_count(tokenizer, rewritten)
        return rewritten, {
            "rewrite_applied": True,
            "rewrite_count": 1,
            "original_rendered_token_count": original_tokens,
            "rewritten_rendered_token_count": rewritten_tokens,
            "rendered_token_delta": rewritten_tokens - original_tokens,
            "original_hint_line_chars": len(hint_line),
            "rewritten_hint_line_chars": len(replacement),
            "hint_line_char_delta": len(replacement) - len(hint_line),
        }

    if rewrite_kind == "token_matched_neutral_hint_line":
        replacement, metadata = token_matched_neutral_line(tokenizer, before, hint_line, after)
        return before + replacement + after, {
            "rewrite_applied": True,
            "rewrite_count": 1,
            **metadata,
        }

    rewritten, metadata = v12_rewrite_prompt(record, rewrite_kind)
    rewritten_tokens = rendered_token_count(tokenizer, rewritten)
    return rewritten, {
        **metadata,
        "original_rendered_token_count": original_tokens,
        "rewritten_rendered_token_count": rewritten_tokens,
        "rendered_token_delta": rewritten_tokens - original_tokens,
        "original_hint_line_chars": len(hint_line),
        "rewritten_hint_line_chars": len(rewritten[start: rewritten.find("\n", start) if rewritten.find("\n", start) >= 0 else len(rewritten)]),
        "hint_line_char_delta": len(rewritten) - len(prompt),
    }


def generate_with_rewrite_v13(
    row: dict[str, Any],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    rewrite_kind: str,
) -> dict[str, Any]:
    rewritten_prompt, metadata = rewrite_prompt_v13(row, tokenizer, rewrite_kind)
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
        {"name": "rewrite_neutralize_hint_line", "kind": "rewrite", "rewrite_kind": "neutralize_hint_line"},
        {"name": "rewrite_char_matched_neutral_hint_line", "kind": "rewrite", "rewrite_kind": "char_matched_neutral_hint_line"},
        {"name": "rewrite_token_matched_neutral_hint_line", "kind": "rewrite", "rewrite_kind": "token_matched_neutral_hint_line"},
        {"name": "rewrite_delete_hint_line", "kind": "rewrite", "rewrite_kind": "delete_hint_line"},
        {"name": "query_mask_hint_line_all", "kind": "query_mask_all_layers", "source_group": "hint_line"},
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
                result = generate_with_rewrite_v13(row, tokenizer, model, max_new_tokens, spec["rewrite_kind"])
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
            print(f"[v13 gen {arm_name} {index:03d}/{len(eval_rows):03d}] {row['id']} -> {arm_row['parsed_answer']!r} {arm_row['label']}")

    return {
        "arm_settings": specs,
        "summary_by_arm": {key: summarize_many(value) for key, value in sorted(arm_rows.items())},
        "summary_by_arm_margin_bin": {key: summarize_margin_bins(value) for key, value in sorted(arm_rows.items())},
        "summary_filtered": summarize_filtered(arm_rows),
        "score_by_arm": {key: score_cell(value) for key, value in sorted(arm_rows.items())},
        "answer_distribution_by_arm": {key: answer_distribution(value) for key, value in sorted(arm_rows.items())},
        "records": [row for rows in arm_rows.values() for row in rows],
    }


def rewrite_parity_summary(generation: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in generation["arm_settings"]:
        if arm["kind"] != "rewrite":
            continue
        rows = [row for row in generation["records"] if row["arm"] == arm["name"]]
        token_deltas = [int(row.get("rendered_token_delta", 0)) for row in rows]
        char_deltas = [int(row.get("hint_line_char_delta", 0)) for row in rows]
        counts = [int(row.get("rewrite_count", 0)) for row in rows]
        out[arm["name"]] = {
            "applied_n": sum(bool(row.get("rewrite_applied")) for row in rows),
            "rewrite_count_min": min(counts) if counts else None,
            "rewrite_count_max": max(counts) if counts else None,
            "rewrite_count_mean": float(np.mean(counts)) if counts else None,
            "rendered_token_delta_min": min(token_deltas) if token_deltas else None,
            "rendered_token_delta_max": max(token_deltas) if token_deltas else None,
            "rendered_token_delta_mean": float(np.mean(token_deltas)) if token_deltas else None,
            "hint_line_char_delta_min": min(char_deltas) if char_deltas else None,
            "hint_line_char_delta_max": max(char_deltas) if char_deltas else None,
            "hint_line_char_delta_mean": float(np.mean(char_deltas)) if char_deltas else None,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DIR / "mc001_qwen3_0p6b_controlled_v13_layout_parity_manifest.jsonl")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-arms", type=int, default=0)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(13)
    random.seed(13)
    np.random.seed(13)

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
        print(f"[v13 baseline {index:03d}/{len(rendered_rows):03d}] {row['id']} -> {result['parsed_answer']!r} {result['label']}")

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
    print(f"[v13 selection] validation={len(validation_rows)} hard={len(hard_rows)} side={len(side_rows)} eval={len(eval_rows)}")

    specs = arm_specs()
    if args.max_arms:
        specs = [specs[0]] + specs[1 : args.max_arms + 1]
    print("[v13 arms]")
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
        "run_type": "qwen3_0p6b_controlled_v13_layout_parity",
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
        "parity_tested": {
            "char_matched": "neutral User hint line with exactly the same character length as the original hint line",
            "token_matched": "neutral User hint line selected to minimize rendered prompt token-count delta",
            "references": "V12 neutral/delete rewrites, V9-style input mask, and V11-style all-layer query mask for hint_line",
        },
        "generation_validation": generation,
        "rewrite_parity_summary": rewrite_parity_summary(generation),
        "hard_bin_top": compact_arm_summary(generation, "wrong_hint_agreement_favored"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"mc001_qwen3_0p6b_controlled_v13_layout_parity_{timestamp}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output_path": str(output_path),
        "baseline": output["baseline_summary"]["overall"],
        "candidate_counts": output["candidate_counts"],
        "hard_bin_top": output["hard_bin_top"],
        "rewrite_parity_summary": output["rewrite_parity_summary"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
