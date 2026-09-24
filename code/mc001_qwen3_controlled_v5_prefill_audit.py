#!/usr/bin/env python
"""Prefill-only intervention audit for MC-001 Qwen3-0.6B.

Earlier generation hooks modified the last token on every `generate()` forward
pass. During decoding, that means the hook also touches each newly generated
token. V5 separates all-step steering from prompt-prefill-only steering and
patching on the hard agreement-favored wrong-hint rows.
"""

from __future__ import annotations

import argparse
import json
import random
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
    build_direction,
    candidate_rows,
    classify,
    generate_one,
    load_model,
    option_token_ids,
    parse_answer,
    summarize_many,
)
from mc001_qwen3_controlled_v2 import (
    add_margin_metadata,
    guard_prompt,
    iter_records,
    margin_bin_counts,
    summarize_margin_bins,
    write_manifest,
)
from mc001_qwen3_controlled_v3 import answer_distribution, rows_digest
from mc001_qwen3_controlled_v4_patch import (
    build_row_indexes,
    feature_positions,
    random_other_donor,
    same_question_donor,
    selected_eval_rows,
    summarize_donors,
)
from mc001_qwen3_smoke import format_for_model
from mc001_qwen3_steer import layer_module_for_hidden_index


def prompt_pass_only(hidden: torch.Tensor) -> bool:
    return hidden.shape[1] > 1


def position_index(token_position: str) -> int:
    return 0 if token_position == "first" else -1


def generate_with_direction(
    record: dict[str, Any],
    direction: dict[str, Any],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    alpha: float,
    mode: str,
    token_position: str = "last",
) -> dict[str, Any]:
    hidden_index = int(direction["hidden_index"])
    unit = torch.tensor(direction["direction"], device=model.device, dtype=model.dtype)
    median_hidden_norm = float(direction["median_hidden_norm"])
    delta = unit * (float(alpha) * median_hidden_norm)

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
            if mode == "prefill_only" and not prompt_pass_only(hidden):
                return output
            patched = hidden.clone()
            patched[:, position_index(token_position), :] = patched[:, position_index(token_position), :] + delta
            return (patched,) + output[1:]
        if mode == "prefill_only" and not prompt_pass_only(output):
            return output
        patched = output.clone()
        patched[:, position_index(token_position), :] = patched[:, position_index(token_position), :] + delta
        return patched

    handle = layer_module_for_hidden_index(model, hidden_index).register_forward_hook(hook)
    try:
        return generate_raw(record, tokenizer, model, max_new_tokens)
    finally:
        handle.remove()


def generate_with_patch(
    record: dict[str, Any],
    donor_feature: dict[str, Any],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    hidden_index: int,
    mix: float,
    mode: str,
    source_position: str = "last",
    target_position: str = "last",
) -> dict[str, Any]:
    donor_vec = torch.tensor(
        donor_feature["hidden"][str(hidden_index)][source_position],
        device=model.device,
        dtype=model.dtype,
    )

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
            if mode == "prefill_only" and not prompt_pass_only(hidden):
                return output
            patched = hidden.clone()
            pos = position_index(target_position)
            patched[:, pos, :] = patched[:, pos, :] * (1.0 - mix) + donor_vec * mix
            return (patched,) + output[1:]
        if mode == "prefill_only" and not prompt_pass_only(output):
            return output
        patched = output.clone()
        pos = position_index(target_position)
        patched[:, pos, :] = patched[:, pos, :] * (1.0 - mix) + donor_vec * mix
        return patched

    handle = layer_module_for_hidden_index(model, hidden_index).register_forward_hook(hook)
    try:
        return generate_raw(record, tokenizer, model, max_new_tokens)
    finally:
        handle.remove()


def generate_raw(record: dict[str, Any], tokenizer: Any, model: Any, max_new_tokens: int) -> dict[str, Any]:
    inputs = tokenizer(record["rendered_prompt"], return_tensors="pt").to(model.device)
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


def direction_features(features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        row_id: {
            "hidden": {
                hidden_index: values["last"]
                for hidden_index, values in feature["hidden"].items()
            }
        }
        for row_id, feature in features.items()
    }


def arm_specs() -> list[dict[str, Any]]:
    return [
        {"name": "baseline", "kind": "baseline"},
        {"name": "prompt_guard", "kind": "prompt_guard"},
        {"name": "raw_h14_all_steps_a0.50", "kind": "direction", "alpha": 0.50, "mode": "all_steps"},
        {"name": "raw_h14_prefill_a0.50", "kind": "direction", "alpha": 0.50, "mode": "prefill_only"},
        {"name": "raw_h14_prefill_a1.00", "kind": "direction", "alpha": 1.00, "mode": "prefill_only"},
        {"name": "same_no_hint_h14_prefill_m0.50", "kind": "same_question", "donor_condition": "no_hint", "hidden_index": 14, "mix": 0.50},
        {"name": "same_no_hint_h14_prefill_m1.00", "kind": "same_question", "donor_condition": "no_hint", "hidden_index": 14, "mix": 1.00},
        {"name": "same_correct_hint_h14_prefill_m1.00", "kind": "same_question", "donor_condition": "correct_hint", "hidden_index": 14, "mix": 1.00},
        {"name": "nearby_same_no_hint_h13_prefill_m1.00", "kind": "same_question", "donor_condition": "no_hint", "hidden_index": 13, "mix": 1.00},
        {
            "name": "wrong_token_same_no_hint_h14_prefill_m1.00",
            "kind": "same_question",
            "donor_condition": "no_hint",
            "hidden_index": 14,
            "mix": 1.00,
            "target_position": "first",
        },
        {"name": "same_correct_letter_other_no_hint_h14_prefill_m1.00", "kind": "same_correct_random", "hidden_index": 14, "mix": 1.00},
        {"name": "random_other_no_hint_h14_prefill_m1.00", "kind": "random_other", "hidden_index": 14, "mix": 1.00},
    ]


def donor_for_spec(row: dict[str, Any], spec: dict[str, Any], indexes: dict[str, Any]) -> dict[str, Any] | None:
    if spec["kind"] == "same_question":
        return same_question_donor(row, spec["donor_condition"], indexes)
    if spec["kind"] == "same_correct_random":
        return random_other_donor(row, indexes, same_correct=True)
    if spec["kind"] == "random_other":
        return random_other_donor(row, indexes, same_correct=False)
    return None


def run_arms(
    eval_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    raw_direction: dict[str, Any],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    indexes = build_row_indexes(baseline_rows)
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    arm_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    donor_rows: list[dict[str, Any]] = []
    for spec in arm_specs():
        arm_name = spec["name"]
        for index, row in enumerate(eval_rows, start=1):
            donor = donor_for_spec(row, spec, indexes)
            donor_meta: dict[str, Any] = {}
            if donor is not None:
                donor_meta = {
                    "donor_id": donor["id"],
                    "donor_condition": donor["condition"],
                    "donor_label": donor["label"],
                    "donor_correct_answer": donor["correct_answer"],
                    "donor_parsed_answer": donor["parsed_answer"],
                }
                donor_rows.append({**row, "arm": arm_name, **donor_meta})

            if spec["kind"] == "baseline":
                base = baseline_by_id[row["id"]]
                result = {key: base[key] for key in ["completion", "parsed_answer", "label"]}
            elif spec["kind"] == "prompt_guard":
                guarded = {**row, "rendered_prompt": format_for_model(tokenizer, guard_prompt(row["prompt"]))}
                result = generate_one(guarded, tokenizer, model, max_new_tokens)
            elif spec["kind"] == "direction":
                result = generate_with_direction(
                    row,
                    raw_direction,
                    tokenizer,
                    model,
                    max_new_tokens,
                    alpha=float(spec["alpha"]),
                    mode=spec["mode"],
                )
            else:
                assert donor is not None
                result = generate_with_patch(
                    row,
                    features[donor["id"]],
                    tokenizer,
                    model,
                    max_new_tokens,
                    hidden_index=int(spec["hidden_index"]),
                    mix=float(spec["mix"]),
                    mode="prefill_only",
                    source_position=spec.get("source_position", "last"),
                    target_position=spec.get("target_position", "last"),
                )

            arm_row = {
                **row,
                **result,
                "arm": arm_name,
                **donor_meta,
                "mode": spec.get("mode", "prefill_only" if spec["kind"] not in {"baseline", "prompt_guard"} else None),
                "patch_hidden_index": spec.get("hidden_index"),
                "patch_mix": spec.get("mix"),
                "direction_alpha": spec.get("alpha"),
                "patch_target_position": spec.get("target_position", "last"),
            }
            arm_rows[arm_name].append(arm_row)
            print(f"[v5 gen {arm_name} {index:03d}/{len(eval_rows):03d}] {row['id']} -> {arm_row['parsed_answer']!r} {arm_row['label']}")
    return {
        "arm_settings": arm_specs(),
        "summary_by_arm": {key: summarize_many(value) for key, value in sorted(arm_rows.items())},
        "summary_by_arm_margin_bin": {key: summarize_margin_bins(value) for key, value in sorted(arm_rows.items())},
        "answer_distribution_by_arm": {key: answer_distribution(value) for key, value in sorted(arm_rows.items())},
        "donor_summary_by_arm": summarize_donors(donor_rows),
        "records": [row for rows in arm_rows.values() for row in rows],
    }


def split_condition_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(f"{row['split']}::{row['condition']}" for row in rows).items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DIR / "mc001_qwen3_0p6b_controlled_v5_prefill_manifest.jsonl")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--hidden-indices", default="13,14")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(5)
    random.seed(5)
    np.random.seed(5)

    records = iter_records()
    digest = write_manifest(args.manifest, records)
    if args.manifest_only:
        print(f"wrote {len(records)} records to {args.manifest}")
        print(f"sha256={digest}")
        return 0

    hidden_indices = [int(value) for value in args.hidden_indices.split(",") if value.strip()]
    if 14 not in hidden_indices:
        hidden_indices.append(14)
    tokenizer, model = load_model(args.model_id)
    token_ids = option_token_ids(tokenizer)
    started = time.time()

    baseline_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        rendered = format_for_model(tokenizer, record["prompt"])
        row = {**record, "variant": "controlled_v5_prefill_audit", "rendered_prompt": rendered}
        result = generate_one(row, tokenizer, model, args.max_new_tokens)
        baseline_rows.append({**row, **result})
        print(f"[v5 baseline {index:03d}/{len(records):03d}] {record['id']} -> {result['parsed_answer']!r} {result['label']}")

    features: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(baseline_rows, start=1):
        features[row["id"]] = feature_positions(row, tokenizer, model, token_ids, hidden_indices)
        print(f"[v5 features {index:03d}/{len(baseline_rows):03d}] {row['id']}")

    baseline_rows = add_margin_metadata(baseline_rows, features)
    discovery = candidate_rows(baseline_rows, split="discovery")
    raw_direction = build_direction(discovery, direction_features(features), 14)
    raw_direction["direction_kind"] = "raw_truth_minus_agreement_last_prompt_token"
    eval_rows = selected_eval_rows(baseline_rows)
    generation = run_arms(
        eval_rows,
        baseline_rows,
        features,
        raw_direction,
        tokenizer,
        model,
        args.max_new_tokens,
    )

    output = {
        "card_id": CARD_ID,
        "run_type": "qwen3_0p6b_controlled_v5_prefill_audit",
        "model_id": args.model_id,
        "manifest": str(args.manifest),
        "manifest_sha256": digest,
        "records_digest": rows_digest(records),
        "records_n": len(records),
        "hidden_indices": hidden_indices,
        "max_new_tokens": args.max_new_tokens,
        "elapsed_s": time.time() - started,
        "selection_rule": {
            "splits": ["calibration", "holdout", "paraphrase_holdout"],
            "conditions": sorted({row["condition"] for row in eval_rows}),
            "baseline_margin_bin": "agreement_favored",
            "n": len(eval_rows),
            "split_condition_counts": split_condition_counts(eval_rows),
        },
        "baseline_summary": summarize_many(baseline_rows),
        "candidate_counts": {
            "discovery": len(discovery),
            "selected_eval": len(eval_rows),
        },
        "candidate_margin_bin_counts": {
            "selected_eval": margin_bin_counts([
                row
                for row in eval_rows
                if row["label"] in {"truth_following", "user_agreement_error"}
            ])
        },
        "direction": {
            key: value
            for key, value in raw_direction.items()
            if key != "direction"
        },
        "generation_validation": generation,
        "baseline_records": [
            {key: value for key, value in row.items() if key != "rendered_prompt"}
            for row in baseline_rows
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"mc001_qwen3_0p6b_controlled_v5_prefill_{stamp}.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({
        "selection_rule": output["selection_rule"],
        "baseline": output["baseline_summary"]["overall"],
        "direction": output["direction"],
        "generation_summary_by_arm": output["generation_validation"]["summary_by_arm"],
        "answer_distribution_by_arm": output["generation_validation"]["answer_distribution_by_arm"],
        "output_path": str(output_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
