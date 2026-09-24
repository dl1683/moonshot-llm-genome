#!/usr/bin/env python
"""Additive prompt-prefill controls for MC-001 Qwen3-0.6B.

V5 showed that the raw h14 direction still changes hard agreement-favored
wrong-hint rows when applied only during the prompt/prefix prefill pass. V6
tests the missing controls on the full validation surface: matched random,
nearby-layer, wrong-token, residualized, and no/correct side-effect rows.
"""

from __future__ import annotations

import argparse
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
    build_direction,
    build_random_direction,
    candidate_rows,
    forward_features,
    generate_one,
    load_model,
    option_token_ids,
    summarize_many,
)
from mc001_qwen3_controlled_v2 import (
    add_margin_metadata,
    guard_prompt,
    iter_records,
    margin_bin_counts,
    write_manifest,
)
from mc001_qwen3_controlled_v3 import (
    answer_distribution,
    build_residualized_direction,
    rows_digest,
    score_cell,
    summarize_margin_bins,
)
from mc001_qwen3_controlled_v5_prefill_audit import generate_with_direction
from mc001_qwen3_smoke import format_for_model


ValidationFilter = Callable[[dict[str, Any]], bool]


def direction_without_vector(direction: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in direction.items() if key != "direction"}


def nearby_direction(direction: dict[str, Any], hidden_index: int, kind: str) -> dict[str, Any]:
    out = {**direction, "hidden_index": hidden_index}
    out["direction_kind"] = kind
    return out


def arm_specs(raw_h14: dict[str, Any], residual_h14: dict[str, Any]) -> list[dict[str, Any]]:
    raw_random = build_random_direction(raw_h14, seed=61003)
    raw_random["direction_kind"] = "raw_h14_matched_random"
    residual_random = build_random_direction(residual_h14, seed=61004)
    residual_random["direction_kind"] = "residual_h14_matched_random"
    return [
        {"name": "baseline", "kind": "baseline"},
        {"name": "prompt_guard", "kind": "prompt_guard"},
        {"name": "raw_h14_prefill_a0.50", "kind": "direction", "direction": raw_h14, "alpha": 0.50, "mode": "prefill_only", "token_position": "last"},
        {"name": "raw_h14_prefill_a1.00", "kind": "direction", "direction": raw_h14, "alpha": 1.00, "mode": "prefill_only", "token_position": "last"},
        {"name": "raw_random_prefill_a0.50", "kind": "direction", "direction": raw_random, "alpha": 0.50, "mode": "prefill_only", "token_position": "last"},
        {"name": "raw_random_prefill_a1.00", "kind": "direction", "direction": raw_random, "alpha": 1.00, "mode": "prefill_only", "token_position": "last"},
        {
            "name": "raw_h14_on_h13_prefill_a0.50",
            "kind": "direction",
            "direction": nearby_direction(raw_h14, 13, "raw_h14_vector_on_h13_nearby"),
            "alpha": 0.50,
            "mode": "prefill_only",
            "token_position": "last",
        },
        {
            "name": "raw_h14_on_h13_prefill_a1.00",
            "kind": "direction",
            "direction": nearby_direction(raw_h14, 13, "raw_h14_vector_on_h13_nearby"),
            "alpha": 1.00,
            "mode": "prefill_only",
            "token_position": "last",
        },
        {"name": "raw_h14_wrong_token_prefill_a0.50", "kind": "direction", "direction": raw_h14, "alpha": 0.50, "mode": "prefill_only", "token_position": "first"},
        {"name": "residual_h14_prefill_a0.50", "kind": "direction", "direction": residual_h14, "alpha": 0.50, "mode": "prefill_only", "token_position": "last"},
        {"name": "residual_random_prefill_a0.50", "kind": "direction", "direction": residual_random, "alpha": 0.50, "mode": "prefill_only", "token_position": "last"},
        {
            "name": "residual_h14_on_h13_prefill_a0.50",
            "kind": "direction",
            "direction": nearby_direction(residual_h14, 13, "residual_h14_vector_on_h13_nearby"),
            "alpha": 0.50,
            "mode": "prefill_only",
            "token_position": "last",
        },
    ]


def validation_filters() -> dict[str, ValidationFilter]:
    return {
        "all_validation": lambda row: True,
        "wrong_hint_all": lambda row: row["condition"].startswith("wrong_"),
        "wrong_hint_agreement_favored": lambda row: row["condition"].startswith("wrong_") and row["baseline_margin_bin"] == "agreement_favored",
        "wrong_hint_ambiguous_or_agreement": lambda row: row["condition"].startswith("wrong_") and row["baseline_margin_bin"] in {"agreement_favored", "ambiguous"},
        "no_and_correct_hint": lambda row: row["condition"] in {"no_hint", "correct_hint"},
        "anti_wrong": lambda row: row["condition"] == "anti_wrong",
    }


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


def run_arms(
    eval_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    arm_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in specs:
        arm_name = spec["name"]
        for index, row in enumerate(eval_rows, start=1):
            if spec["kind"] == "baseline":
                base = baseline_by_id[row["id"]]
                result = {key: base[key] for key in ["completion", "parsed_answer", "label"]}
            elif spec["kind"] == "prompt_guard":
                guarded = {**row, "rendered_prompt": format_for_model(tokenizer, guard_prompt(row["prompt"]))}
                result = generate_one(guarded, tokenizer, model, max_new_tokens)
            else:
                result = generate_with_direction(
                    row,
                    spec["direction"],
                    tokenizer,
                    model,
                    max_new_tokens,
                    alpha=float(spec["alpha"]),
                    mode=spec["mode"],
                    token_position=spec.get("token_position", "last"),
                )
            arm_row = {
                **row,
                **result,
                "arm": arm_name,
                "mode": spec.get("mode"),
                "direction_alpha": spec.get("alpha"),
                "direction_hidden_index": spec.get("direction", {}).get("hidden_index") if spec.get("direction") else None,
                "direction_kind": spec.get("direction", {}).get("direction_kind") if spec.get("direction") else None,
                "token_position": spec.get("token_position"),
            }
            arm_rows[arm_name].append(arm_row)
            print(f"[v6 gen {arm_name} {index:03d}/{len(eval_rows):03d}] {row['id']} -> {arm_row['parsed_answer']!r} {arm_row['label']}")

    return {
        "arm_settings": [
            {
                key: (direction_without_vector(value) if key == "direction" else value)
                for key, value in spec.items()
            }
            for spec in specs
        ],
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
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DIR / "mc001_qwen3_0p6b_controlled_v6_prefill_controls_manifest.jsonl")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(6)
    random.seed(6)
    np.random.seed(6)

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

    baseline_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        rendered = format_for_model(tokenizer, record["prompt"])
        row = {**record, "variant": "controlled_v6_prefill_controls", "rendered_prompt": rendered}
        result = generate_one(row, tokenizer, model, args.max_new_tokens)
        baseline_rows.append({**row, **result})
        print(f"[v6 baseline {index:03d}/{len(records):03d}] {record['id']} -> {result['parsed_answer']!r} {result['label']}")

    features: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(baseline_rows, start=1):
        features[row["id"]] = forward_features(row, tokenizer, model, token_ids, hidden_indices)
        print(f"[v6 features {index:03d}/{len(baseline_rows):03d}] {row['id']}")

    baseline_rows = add_margin_metadata(baseline_rows, features)
    discovery = candidate_rows(baseline_rows, split="discovery")
    calibration = candidate_rows(baseline_rows, split="calibration")
    holdout = candidate_rows(baseline_rows, split="holdout")
    paraphrase = candidate_rows(baseline_rows, split="paraphrase_holdout")

    raw_h14 = build_direction(discovery, features, 14)
    raw_h14["direction_kind"] = "raw_truth_minus_agreement_last_prompt_token"
    residual_h14 = build_residualized_direction(discovery, features, 14)
    residual_h14["direction_kind"] = "residualized_truth_minus_agreement_last_prompt_token"
    specs = arm_specs(raw_h14, residual_h14)

    eval_rows = [
        row
        for row in baseline_rows
        if row["split"] in {"calibration", "holdout", "paraphrase_holdout"}
    ]
    generation = run_arms(eval_rows, baseline_rows, specs, tokenizer, model, args.max_new_tokens)

    output = {
        "card_id": CARD_ID,
        "run_type": "qwen3_0p6b_controlled_v6_prefill_controls",
        "model_id": args.model_id,
        "manifest": str(args.manifest),
        "manifest_sha256": digest,
        "records_digest": rows_digest(records),
        "records_n": len(records),
        "hidden_indices": hidden_indices,
        "max_new_tokens": args.max_new_tokens,
        "elapsed_s": time.time() - started,
        "baseline_summary": summarize_many(baseline_rows),
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
            "purpose": "full validation and side-effect rows for additive prompt-prefill controls",
        },
        "directions": {
            "raw_h14": direction_without_vector(raw_h14),
            "residual_h14": direction_without_vector(residual_h14),
        },
        "generation_validation": generation,
        "baseline_records": [
            {key: value for key, value in row.items() if key != "rendered_prompt"}
            for row in baseline_rows
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"mc001_qwen3_0p6b_controlled_v6_prefill_controls_{stamp}.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({
        "baseline": output["baseline_summary"]["overall"],
        "candidate_counts": output["candidate_counts"],
        "candidate_margin_bin_counts": output["candidate_margin_bin_counts"],
        "directions": output["directions"],
        "summary_filtered": {
            key: {
                arm: value["overall"]
                for arm, value in arms.items()
            }
            for key, arms in output["generation_validation"]["summary_filtered"].items()
        },
        "answer_distribution_by_arm": output["generation_validation"]["answer_distribution_by_arm"],
        "output_path": str(output_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
