#!/usr/bin/env python
"""Cross-layer logit-transport audit for MC-001 Qwen3-0.6B.

V6 closed dense prompt-prefill steering as an h14-local mechanism path because
the h14 vector was equally or more active when applied at nearby h13. V7 does
not run another generation sweep. It measures the single-forward option-logit
effect of that same vector across layers and controls.
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
    classify,
    forward_features,
    generate_one,
    load_model,
    option_scores,
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
    build_residualized_direction,
    rows_digest,
    summarize_margin_bins,
)
from mc001_qwen3_controlled_v6_prefill_controls import (
    direction_without_vector,
    validation_filters,
)
from mc001_qwen3_smoke import format_for_model
from mc001_qwen3_steer import layer_module_for_hidden_index


ValidationFilter = Callable[[dict[str, Any]], bool]


def position_index(token_position: str) -> int:
    return 0 if token_position == "first" else -1


def tensor_delta(direction: dict[str, Any], model: Any, alpha: float) -> torch.Tensor:
    unit = torch.tensor(direction["direction"], device=model.device, dtype=model.dtype)
    return unit * (float(alpha) * float(direction["median_hidden_norm"]))


def forward_option_audit(
    record: dict[str, Any],
    tokenizer: Any,
    model: Any,
    token_ids: dict[str, list[int]],
    direction: dict[str, Any] | None = None,
    apply_hidden_index: int | None = None,
    alpha: float = 0.0,
    token_position: str = "last",
) -> dict[str, Any]:
    handle = None
    if direction is not None:
        if apply_hidden_index is None:
            apply_hidden_index = int(direction["hidden_index"])
        delta = tensor_delta(direction, model, alpha)

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            if isinstance(output, tuple):
                hidden = output[0].clone()
                hidden[:, position_index(token_position), :] = hidden[:, position_index(token_position), :] + delta
                return (hidden,) + output[1:]
            hidden = output.clone()
            hidden[:, position_index(token_position), :] = hidden[:, position_index(token_position), :] + delta
            return hidden

        handle = layer_module_for_hidden_index(model, int(apply_hidden_index)).register_forward_hook(hook)

    try:
        inputs = tokenizer(record["rendered_prompt"], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model(**inputs, use_cache=False)
    finally:
        if handle is not None:
            handle.remove()

    scores = option_scores(out.logits[0, -1, :].detach().float(), token_ids)
    pred = max(scores, key=scores.get)
    label = classify(pred, record["correct_answer"], record["wrong_answer"])
    margin = scores[record["correct_answer"]] - scores[record["wrong_answer"]]
    return {
        "option_scores": scores,
        "next_token_answer": pred,
        "label": label,
        "correct_minus_wrong_logit": margin,
    }


def arm_specs(raw_h14: dict[str, Any], residual_h14: dict[str, Any]) -> list[dict[str, Any]]:
    raw_random = build_random_direction(raw_h14, seed=71003)
    raw_random["direction_kind"] = "raw_h14_matched_random"
    residual_random = build_random_direction(residual_h14, seed=71004)
    residual_random["direction_kind"] = "residual_h14_matched_random"
    return [
        {"name": "baseline_logits", "kind": "baseline"},
        {"name": "raw_h14_at_h7_a0.50", "kind": "direction", "direction": raw_h14, "apply_hidden_index": 7, "alpha": 0.50, "token_position": "last"},
        {"name": "raw_h14_at_h13_a0.50", "kind": "direction", "direction": raw_h14, "apply_hidden_index": 13, "alpha": 0.50, "token_position": "last"},
        {"name": "raw_h14_at_h14_a0.50", "kind": "direction", "direction": raw_h14, "apply_hidden_index": 14, "alpha": 0.50, "token_position": "last"},
        {"name": "raw_h14_at_h20_a0.50", "kind": "direction", "direction": raw_h14, "apply_hidden_index": 20, "alpha": 0.50, "token_position": "last"},
        {"name": "raw_h14_at_h13_a1.00", "kind": "direction", "direction": raw_h14, "apply_hidden_index": 13, "alpha": 1.00, "token_position": "last"},
        {"name": "raw_h14_at_h14_a1.00", "kind": "direction", "direction": raw_h14, "apply_hidden_index": 14, "alpha": 1.00, "token_position": "last"},
        {"name": "raw_h14_h14_wrong_token_a0.50", "kind": "direction", "direction": raw_h14, "apply_hidden_index": 14, "alpha": 0.50, "token_position": "first"},
        {"name": "raw_random_at_h13_a0.50", "kind": "direction", "direction": raw_random, "apply_hidden_index": 13, "alpha": 0.50, "token_position": "last"},
        {"name": "raw_random_at_h14_a0.50", "kind": "direction", "direction": raw_random, "apply_hidden_index": 14, "alpha": 0.50, "token_position": "last"},
        {"name": "residual_h14_at_h13_a0.50", "kind": "direction", "direction": residual_h14, "apply_hidden_index": 13, "alpha": 0.50, "token_position": "last"},
        {"name": "residual_h14_at_h14_a0.50", "kind": "direction", "direction": residual_h14, "apply_hidden_index": 14, "alpha": 0.50, "token_position": "last"},
        {"name": "residual_random_at_h13_a0.50", "kind": "direction", "direction": residual_random, "apply_hidden_index": 13, "alpha": 0.50, "token_position": "last"},
        {"name": "residual_random_at_h14_a0.50", "kind": "direction", "direction": residual_random, "apply_hidden_index": 14, "alpha": 0.50, "token_position": "last"},
    ]


def split_condition_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(f"{row['split']}::{row['condition']}" for row in rows).items()))


def margin_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "positive_margin_n": 0,
            "positive_margin_rate": None,
            "mean_margin": None,
            "median_margin": None,
            "mean_delta_margin": None,
            "median_delta_margin": None,
        }
    margins = np.array([float(row["correct_minus_wrong_logit"]) for row in rows], dtype=np.float64)
    deltas = np.array([float(row.get("delta_correct_minus_wrong_logit", 0.0)) for row in rows], dtype=np.float64)
    positive = int(np.sum(margins > 0.0))
    return {
        "n": len(rows),
        "positive_margin_n": positive,
        "positive_margin_rate": positive / len(rows),
        "mean_margin": float(np.mean(margins)),
        "median_margin": float(np.median(margins)),
        "mean_delta_margin": float(np.mean(deltas)),
        "median_delta_margin": float(np.median(deltas)),
    }


def summarize_filtered(arm_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for filter_name, predicate in validation_filters().items():
        output[filter_name] = {
            arm: {
                "label_summary": summarize_many([row for row in rows if predicate(row)]),
                "margin_summary": margin_summary([row for row in rows if predicate(row)]),
            }
            for arm, rows in sorted(arm_rows.items())
        }
    return output


def paired_delta_similarity(records: list[dict[str, Any]], arm_a: str, arm_b: str, predicate: ValidationFilter) -> dict[str, Any]:
    rows_a = {row["id"]: row for row in records if row["arm"] == arm_a and predicate(row)}
    rows_b = {row["id"]: row for row in records if row["arm"] == arm_b and predicate(row)}
    ids = sorted(set(rows_a) & set(rows_b))
    if not ids:
        return {"n": 0}
    a = np.array([float(rows_a[row_id]["delta_correct_minus_wrong_logit"]) for row_id in ids], dtype=np.float64)
    b = np.array([float(rows_b[row_id]["delta_correct_minus_wrong_logit"]) for row_id in ids], dtype=np.float64)
    if len(ids) > 1 and float(np.std(a)) > 0.0 and float(np.std(b)) > 0.0:
        corr = float(np.corrcoef(a, b)[0, 1])
    else:
        corr = None
    return {
        "n": len(ids),
        "arm_a": arm_a,
        "arm_b": arm_b,
        "mean_delta_a": float(np.mean(a)),
        "mean_delta_b": float(np.mean(b)),
        "median_delta_a": float(np.median(a)),
        "median_delta_b": float(np.median(b)),
        "mean_abs_delta_diff": float(np.mean(np.abs(a - b))),
        "median_abs_delta_diff": float(np.median(np.abs(a - b))),
        "delta_pearson": corr,
    }


def transport_similarity(records: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [
        ("raw_h14_at_h13_a0.50", "raw_h14_at_h14_a0.50"),
        ("raw_h14_at_h13_a1.00", "raw_h14_at_h14_a1.00"),
        ("residual_h14_at_h13_a0.50", "residual_h14_at_h14_a0.50"),
        ("raw_random_at_h13_a0.50", "raw_random_at_h14_a0.50"),
        ("residual_random_at_h13_a0.50", "residual_random_at_h14_a0.50"),
    ]
    return {
        filter_name: {
            f"{left}__vs__{right}": paired_delta_similarity(records, left, right, predicate)
            for left, right in pairs
        }
        for filter_name, predicate in validation_filters().items()
    }


def run_transport(
    eval_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    token_ids: dict[str, list[int]],
) -> dict[str, Any]:
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    arm_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in specs:
        arm_name = spec["name"]
        for index, row in enumerate(eval_rows, start=1):
            if spec["kind"] == "baseline":
                result = {
                    "option_scores": baseline_by_id[row["id"]]["option_scores"],
                    "next_token_answer": baseline_by_id[row["id"]]["baseline_next_token_answer"],
                    "label": baseline_by_id[row["id"]]["baseline_next_token_label"],
                    "correct_minus_wrong_logit": baseline_by_id[row["id"]]["baseline_correct_minus_wrong_logit"],
                }
            else:
                result = forward_option_audit(
                    row,
                    tokenizer,
                    model,
                    token_ids,
                    direction=spec["direction"],
                    apply_hidden_index=int(spec["apply_hidden_index"]),
                    alpha=float(spec["alpha"]),
                    token_position=spec.get("token_position", "last"),
                )
            delta = float(result["correct_minus_wrong_logit"]) - float(row["baseline_correct_minus_wrong_logit"])
            arm_row = {
                **row,
                **result,
                "arm": arm_name,
                "apply_hidden_index": spec.get("apply_hidden_index"),
                "direction_source_hidden_index": spec.get("direction", {}).get("hidden_index") if spec.get("direction") else None,
                "direction_kind": spec.get("direction", {}).get("direction_kind") if spec.get("direction") else None,
                "direction_alpha": spec.get("alpha"),
                "token_position": spec.get("token_position"),
                "delta_correct_minus_wrong_logit": delta,
            }
            arm_rows[arm_name].append(arm_row)
            print(
                f"[v7 logit {arm_name} {index:03d}/{len(eval_rows):03d}] "
                f"{row['id']} -> {arm_row['next_token_answer']} {arm_row['label']} "
                f"delta={delta:+.3f}"
            )

    records = [row for rows in arm_rows.values() for row in rows]
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
        "margin_summary_by_arm": {key: margin_summary(value) for key, value in sorted(arm_rows.items())},
        "summary_filtered": summarize_filtered(arm_rows),
        "transport_similarity": transport_similarity(records),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DIR / "mc001_qwen3_0p6b_controlled_v7_transport_manifest.jsonl")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)

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
        row = {**record, "variant": "controlled_v7_transport", "rendered_prompt": rendered}
        result = generate_one(row, tokenizer, model, args.max_new_tokens)
        baseline_rows.append({**row, **result})
        print(f"[v7 baseline {index:03d}/{len(records):03d}] {record['id']} -> {result['parsed_answer']!r} {result['label']}")

    features: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(baseline_rows, start=1):
        features[row["id"]] = forward_features(row, tokenizer, model, token_ids, hidden_indices)
        print(f"[v7 features {index:03d}/{len(baseline_rows):03d}] {row['id']}")

    baseline_rows = add_margin_metadata(baseline_rows, features)
    for row in baseline_rows:
        row["option_scores"] = features[row["id"]]["option_scores"]

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
    transport = run_transport(eval_rows, baseline_rows, specs, tokenizer, model, token_ids)

    output = {
        "card_id": CARD_ID,
        "run_type": "qwen3_0p6b_controlled_v7_transport",
        "model_id": args.model_id,
        "manifest": str(args.manifest),
        "manifest_sha256": digest,
        "records_digest": rows_digest(records),
        "records_n": len(records),
        "hidden_indices_for_direction_training": hidden_indices,
        "max_new_tokens_for_baseline_labels": args.max_new_tokens,
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
            "purpose": "single-forward cross-layer option-logit transport audit",
        },
        "directions": {
            "raw_h14": direction_without_vector(raw_h14),
            "residual_h14": direction_without_vector(residual_h14),
        },
        "transport_validation": transport,
        "baseline_records": [
            {key: value for key, value in row.items() if key != "rendered_prompt"}
            for row in baseline_rows
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"mc001_qwen3_0p6b_controlled_v7_transport_{stamp}.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({
        "baseline": output["baseline_summary"]["overall"],
        "candidate_counts": output["candidate_counts"],
        "candidate_margin_bin_counts": output["candidate_margin_bin_counts"],
        "directions": output["directions"],
        "summary_filtered": {
            key: {
                arm: value["margin_summary"]
                for arm, value in arms.items()
            }
            for key, arms in output["transport_validation"]["summary_filtered"].items()
        },
        "transport_similarity": output["transport_validation"]["transport_similarity"],
        "output_path": str(output_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
