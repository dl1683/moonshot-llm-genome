#!/usr/bin/env python
"""Agreement-bin activation patching pass for MC-001 Qwen3-0.6B.

V1-V3 showed that global dense directions expose a control surface but fail
mechanism promotion. V4 changes method: for hard agreement-favored wrong-hint
rows, patch same-question no-hint/correct-hint hidden states into the
wrong-hint recipient and compare against answer-token and locality controls.
"""

from __future__ import annotations

import argparse
import hashlib
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
    classify,
    generate_one,
    load_model,
    option_scores,
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
from mc001_qwen3_smoke import format_for_model
from mc001_qwen3_steer import layer_module_for_hidden_index


WRONG_HINT_CONDITIONS = {
    "wrong_marked_false",
    "wrong_untrusted",
    "wrong_unsure",
    "wrong_direct",
    "wrong_high",
}


def deterministic_index(key: str, size: int) -> int:
    if size <= 0:
        raise ValueError("cannot choose from an empty candidate list")
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % size


def feature_positions(
    record: dict[str, Any],
    tokenizer: Any,
    model: Any,
    token_ids: dict[str, list[int]],
    hidden_indices: list[int],
) -> dict[str, Any]:
    inputs = tokenizer(record["rendered_prompt"], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    scores = option_scores(out.logits[0, -1, :].detach().float(), token_ids)
    pred = max(scores, key=scores.get)
    return {
        "option_scores": scores,
        "next_token_answer": pred,
        "next_token_label": classify(pred, record["correct_answer"], record["wrong_answer"]),
        "correct_minus_wrong_logit": scores[record["correct_answer"]] - scores[record["wrong_answer"]],
        "hidden": {
            str(index): {
                "first": out.hidden_states[index][0, 0, :].detach().float().cpu().numpy(),
                "last": out.hidden_states[index][0, -1, :].detach().float().cpu().numpy(),
            }
            for index in hidden_indices
        },
    }


def generate_with_patch(
    record: dict[str, Any],
    donor_feature: dict[str, Any],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    hidden_index: int,
    mix: float,
    source_position: str = "last",
    target_position: str = "last",
) -> dict[str, Any]:
    donor_vec = torch.tensor(
        donor_feature["hidden"][str(hidden_index)][source_position],
        device=model.device,
        dtype=model.dtype,
    )

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        pos = 0 if target_position == "first" else -1
        if isinstance(output, tuple):
            hidden = output[0].clone()
            hidden[:, pos, :] = hidden[:, pos, :] * (1.0 - mix) + donor_vec * mix
            return (hidden,) + output[1:]
        hidden = output.clone()
        hidden[:, pos, :] = hidden[:, pos, :] * (1.0 - mix) + donor_vec * mix
        return hidden

    handle = layer_module_for_hidden_index(model, hidden_index).register_forward_hook(hook)
    try:
        inputs = tokenizer(record["rendered_prompt"], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
    parsed = parse_answer(completion, record)
    label = classify(parsed, record["correct_answer"], record["wrong_answer"])
    return {"completion": completion, "parsed_answer": parsed, "label": label}


def question_key(row: dict[str, Any], condition: str) -> tuple[str, str, str, str]:
    return (row["split"], row["form"], row["question"], condition)


def build_row_indexes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_question_condition = {
        question_key(row, row["condition"]): row
        for row in rows
    }
    no_hint_pool = [
        row
        for row in rows
        if row["condition"] == "no_hint"
    ]
    by_split_form_no_hint: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_split_form_correct_no_hint: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in no_hint_pool:
        by_split_form_no_hint[(row["split"], row["form"])].append(row)
        by_split_form_correct_no_hint[(row["split"], row["form"], row["correct_answer"])].append(row)
    for values in by_split_form_no_hint.values():
        values.sort(key=lambda item: item["id"])
    for values in by_split_form_correct_no_hint.values():
        values.sort(key=lambda item: item["id"])
    return {
        "by_question_condition": by_question_condition,
        "by_split_form_no_hint": by_split_form_no_hint,
        "by_split_form_correct_no_hint": by_split_form_correct_no_hint,
    }


def same_question_donor(row: dict[str, Any], condition: str, indexes: dict[str, Any]) -> dict[str, Any]:
    donor = indexes["by_question_condition"].get(question_key(row, condition))
    if donor is None:
        raise KeyError(f"missing {condition} donor for {row['id']}")
    return donor


def random_other_donor(row: dict[str, Any], indexes: dict[str, Any], same_correct: bool) -> dict[str, Any]:
    if same_correct:
        candidates = indexes["by_split_form_correct_no_hint"][(row["split"], row["form"], row["correct_answer"])]
    else:
        candidates = indexes["by_split_form_no_hint"][(row["split"], row["form"])]
    candidates = [candidate for candidate in candidates if candidate["question"] != row["question"]]
    if not candidates:
        raise KeyError(f"missing random donor for {row['id']}")
    tag = "same_correct" if same_correct else "random"
    return candidates[deterministic_index(f"{tag}:{row['id']}", len(candidates))]


def selected_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["split"] in {"calibration", "holdout", "paraphrase_holdout"}
        and row["condition"] in WRONG_HINT_CONDITIONS
        and row["baseline_margin_bin"] == "agreement_favored"
    ]


def arm_specs() -> list[dict[str, Any]]:
    return [
        {"name": "baseline", "kind": "baseline"},
        {"name": "prompt_guard", "kind": "prompt_guard"},
        {"name": "same_no_hint_h14_m0.50", "kind": "same_question", "donor_condition": "no_hint", "hidden_index": 14, "mix": 0.50},
        {"name": "same_no_hint_h14_m1.00", "kind": "same_question", "donor_condition": "no_hint", "hidden_index": 14, "mix": 1.00},
        {"name": "same_correct_hint_h14_m0.50", "kind": "same_question", "donor_condition": "correct_hint", "hidden_index": 14, "mix": 0.50},
        {"name": "same_correct_hint_h14_m1.00", "kind": "same_question", "donor_condition": "correct_hint", "hidden_index": 14, "mix": 1.00},
        {"name": "same_no_hint_h7_m1.00", "kind": "same_question", "donor_condition": "no_hint", "hidden_index": 7, "mix": 1.00},
        {"name": "nearby_same_no_hint_h13_m1.00", "kind": "same_question", "donor_condition": "no_hint", "hidden_index": 13, "mix": 1.00},
        {
            "name": "wrong_token_same_no_hint_h14_m1.00",
            "kind": "same_question",
            "donor_condition": "no_hint",
            "hidden_index": 14,
            "mix": 1.00,
            "target_position": "first",
        },
        {"name": "same_correct_letter_other_no_hint_h14_m1.00", "kind": "same_correct_random", "hidden_index": 14, "mix": 1.00},
        {"name": "random_other_no_hint_h14_m1.00", "kind": "random_other", "hidden_index": 14, "mix": 1.00},
    ]


def donor_for_spec(row: dict[str, Any], spec: dict[str, Any], indexes: dict[str, Any]) -> dict[str, Any] | None:
    if spec["kind"] == "same_question":
        return same_question_donor(row, spec["donor_condition"], indexes)
    if spec["kind"] == "same_correct_random":
        return random_other_donor(row, indexes, same_correct=True)
    if spec["kind"] == "random_other":
        return random_other_donor(row, indexes, same_correct=False)
    return None


def run_patch_arms(
    eval_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    specs = arm_specs()
    indexes = build_row_indexes(baseline_rows)
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    arm_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    donor_rows: list[dict[str, Any]] = []
    for spec in specs:
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
                    source_position=spec.get("source_position", "last"),
                    target_position=spec.get("target_position", "last"),
                )

            arm_row = {
                **row,
                **result,
                "arm": arm_name,
                **donor_meta,
                "patch_hidden_index": spec.get("hidden_index"),
                "patch_mix": spec.get("mix"),
                "patch_source_position": spec.get("source_position", "last"),
                "patch_target_position": spec.get("target_position", "last"),
            }
            arm_rows[arm_name].append(arm_row)
            print(f"[v4 gen {arm_name} {index:03d}/{len(eval_rows):03d}] {row['id']} -> {arm_row['parsed_answer']!r} {arm_row['label']}")
    return {
        "arm_settings": specs,
        "summary_by_arm": {key: summarize_many(value) for key, value in sorted(arm_rows.items())},
        "summary_by_arm_margin_bin": {key: summarize_margin_bins(value) for key, value in sorted(arm_rows.items())},
        "answer_distribution_by_arm": {key: answer_distribution(value) for key, value in sorted(arm_rows.items())},
        "donor_summary_by_arm": summarize_donors(donor_rows),
        "records": [row for rows in arm_rows.values() for row in rows],
    }


def summarize_donors(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    output: dict[str, Any] = {}
    for arm, arm_rows in sorted(by_arm.items()):
        output[arm] = {
            "n": len(arm_rows),
            "donor_label_counts": dict(sorted(Counter(row["donor_label"] for row in arm_rows).items())),
            "donor_condition_counts": dict(sorted(Counter(row["donor_condition"] for row in arm_rows).items())),
            "donor_parsed_answer_counts": dict(sorted(Counter(str(row["donor_parsed_answer"]) for row in arm_rows).items())),
        }
    return output


def split_condition_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(f"{row['split']}::{row['condition']}" for row in rows).items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DIR / "mc001_qwen3_0p6b_controlled_v4_patch_manifest.jsonl")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--hidden-indices", default="7,13,14")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(4)
    random.seed(4)
    np.random.seed(4)

    records = iter_records()
    digest = write_manifest(args.manifest, records)
    if args.manifest_only:
        print(f"wrote {len(records)} records to {args.manifest}")
        print(f"sha256={digest}")
        return 0

    hidden_indices = [int(value) for value in args.hidden_indices.split(",") if value.strip()]
    tokenizer, model = load_model(args.model_id)
    token_ids = option_token_ids(tokenizer)
    started = time.time()

    baseline_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        rendered = format_for_model(tokenizer, record["prompt"])
        row = {**record, "variant": "controlled_v4_patch", "rendered_prompt": rendered}
        result = generate_one(row, tokenizer, model, args.max_new_tokens)
        baseline_rows.append({**row, **result})
        print(f"[v4 baseline {index:03d}/{len(records):03d}] {record['id']} -> {result['parsed_answer']!r} {result['label']}")

    features: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(baseline_rows, start=1):
        features[row["id"]] = feature_positions(row, tokenizer, model, token_ids, hidden_indices)
        print(f"[v4 features {index:03d}/{len(baseline_rows):03d}] {row['id']}")

    baseline_rows = add_margin_metadata(baseline_rows, features)
    eval_rows = selected_eval_rows(baseline_rows)
    generation = run_patch_arms(
        eval_rows,
        baseline_rows,
        features,
        tokenizer,
        model,
        args.max_new_tokens,
    )

    output = {
        "card_id": CARD_ID,
        "run_type": "qwen3_0p6b_controlled_v4_agreement_bin_patch",
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
            "conditions": sorted(WRONG_HINT_CONDITIONS),
            "baseline_margin_bin": "agreement_favored",
            "n": len(eval_rows),
            "split_condition_counts": split_condition_counts(eval_rows),
        },
        "baseline_summary": summarize_many(baseline_rows),
        "candidate_margin_bin_counts": {
            "calibration": margin_bin_counts([row for row in baseline_rows if row["split"] == "calibration" and row["condition"] in WRONG_HINT_CONDITIONS and row["label"] in {"truth_following", "user_agreement_error"}]),
            "holdout": margin_bin_counts([row for row in baseline_rows if row["split"] == "holdout" and row["condition"] in WRONG_HINT_CONDITIONS and row["label"] in {"truth_following", "user_agreement_error"}]),
            "paraphrase_holdout": margin_bin_counts([row for row in baseline_rows if row["split"] == "paraphrase_holdout" and row["condition"] in WRONG_HINT_CONDITIONS and row["label"] in {"truth_following", "user_agreement_error"}]),
        },
        "generation_validation": generation,
        "baseline_records": [
            {key: value for key, value in row.items() if key != "rendered_prompt"}
            for row in baseline_rows
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"mc001_qwen3_0p6b_controlled_v4_patch_{stamp}.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({
        "selection_rule": output["selection_rule"],
        "baseline": output["baseline_summary"]["overall"],
        "generation_summary_by_arm": output["generation_validation"]["summary_by_arm"],
        "answer_distribution_by_arm": output["generation_validation"]["answer_distribution_by_arm"],
        "output_path": str(output_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
