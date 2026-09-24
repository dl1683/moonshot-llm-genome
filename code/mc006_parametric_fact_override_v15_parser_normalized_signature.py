#!/usr/bin/env python
"""MC006 V15 hidden-signature diagnostic for the V14 normalized table."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc003_delayed_copy_signature import auc_score, percentile


CARD_ID = "MC006"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/MC006")
DEFAULT_SOURCE_ARTIFACT = (
    RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v14_parser_normalized_20260701T000538.json"
)
RUN_TYPE = "parametric_fact_override_v15_parser_normalized_signature"
SOURCE_RUN_TYPE = "parametric_fact_override_v14_parser_normalized"
SOURCE_DIAGNOSTIC_CLASS = "parser_normalized_generated_substrate_passed"
PRIMARY_LABELS = ("true_answer", "override_answer")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_source(result: dict[str, Any]) -> None:
    if result.get("run_type") != SOURCE_RUN_TYPE:
        raise ValueError(f"unexpected run_type: {result.get('run_type')!r}")
    summary = result.get("summary", {})
    if summary.get("diagnostic_class") != SOURCE_DIAGNOSTIC_CLASS:
        raise ValueError(f"unexpected diagnostic_class: {summary.get('diagnostic_class')!r}")
    if not summary.get("passed"):
        raise ValueError("V14 source artifact did not pass")
    if not summary.get("parser_delta", {}).get("passed"):
        raise ValueError("V14 parser-delta controls did not pass")
    if "selected_template_rows" not in summary:
        raise ValueError("V14 artifact missing selected_template_rows")


def first_answer_token_id(tokenizer: Any, answer: str) -> int:
    ids = tokenizer(f" {answer}", add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"tokenless answer: {answer!r}")
    return int(ids[0])


def token_logprob(model: Any, tokenizer: Any, rendered_prompt: str, answer: str) -> dict[str, Any]:
    full_text = f"{rendered_prompt} {answer}"
    prompt_ids = tokenizer(rendered_prompt, return_tensors="pt")["input_ids"][0]
    full_inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    full_ids = full_inputs["input_ids"][0]
    start = int(prompt_ids.shape[0])
    if start >= int(full_ids.shape[0]):
        return {"token_count": 0, "sum_logprob": -math.inf, "mean_logprob": -math.inf}
    with torch.inference_mode():
        logits = model(**full_inputs).logits[0]
    token_ids = full_ids[start:]
    logprobs = torch.log_softmax(logits[start - 1 : -1], dim=-1)
    selected = logprobs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
    sum_logprob = float(selected.sum().detach().cpu())
    token_count = int(token_ids.shape[0])
    return {
        "token_count": token_count,
        "sum_logprob": sum_logprob,
        "mean_logprob": sum_logprob / token_count,
    }


def build_records(source_result: dict[str, Any], tokenizer: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary_rows = []
    side_rows = []
    selected_template = source_result["summary"]["selection"]["selected_template"]
    for row in source_result["summary"]["selected_template_rows"]:
        if row["selected_label"] not in PRIMARY_LABELS:
            side_rows.append(
                {
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "split": row["split"],
                    "selected_label": row["selected_label"],
                    "selected_answer": row["selected_answer"],
                    "first_line": row["first_line"],
                }
            )
            continue
        prompt_ids = tokenizer(row["prompt"], return_tensors="pt")["input_ids"][0]
        label = 1 if row["selected_label"] == "true_answer" else 0
        primary_rows.append(
            {
                "id": row["id"].replace("_v13_", "_v15_"),
                "source_row_id": row["id"],
                "source_id": row["source_id"],
                "split": row["split"],
                "template": selected_template,
                "condition": row["condition"],
                "country": row["country"],
                "true_capital": row["true_capital"],
                "override_capital": row["override_capital"],
                "lure_capital": row["lure_capital"],
                "selected_label": row["selected_label"],
                "selected_answer": row["selected_answer"],
                "generated_text": row["generated_text"],
                "generated_token_ids": row["generated_token_ids"],
                "generated_first_token_id": int(row["generated_token_ids"][0]) if row["generated_token_ids"] else -1,
                "v13_selected_label": row.get("v13_selected_label"),
                "label_changed_by_normalization": bool(row.get("label_changed_by_normalization")),
                "binary_label": label,
                "label_name": row["selected_label"],
                "prompt": row["prompt"],
                "prompt_token_count": int(prompt_ids.shape[0]),
                "final_prompt_token_id": int(prompt_ids[-1]),
            }
        )
    primary_rows.sort(key=lambda item: (item["split"], item["source_id"]))
    side_rows.sort(key=lambda item: (item["split"], item["source_id"]))
    return primary_rows, side_rows


def structural_check(records: list[dict[str, Any]], side_rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(row["label_name"] for row in records)
    split_label_counts: dict[str, dict[str, int]] = {}
    for split in ("discovery", "calibration", "holdout"):
        split_rows = [row for row in records if row["split"] == split]
        split_label_counts[split] = dict(sorted(Counter(row["label_name"] for row in split_rows).items()))
    duplicate_ids = [row_id for row_id, count in Counter(row["id"] for row in records).items() if count > 1]
    source_ids = {row["source_id"] for row in records}
    holdout_labels = split_label_counts.get("holdout", {})
    side_label_counts = Counter(row["selected_label"] for row in side_rows)
    criteria = {
        "exactly_30_binary_rows": len(records) == 30,
        "exactly_10_side_rows": len(side_rows) == 10,
        "expected_label_counts": label_counts.get("true_answer", 0) == 21
        and label_counts.get("override_answer", 0) == 9,
        "expected_holdout_counts": holdout_labels.get("true_answer", 0) == 6
        and holdout_labels.get("override_answer", 0) == 2,
        "holdout_has_at_least_2_each": holdout_labels.get("true_answer", 0) >= 2
        and holdout_labels.get("override_answer", 0) >= 2,
        "source_ids_unique": len(source_ids) == len(records),
        "no_duplicate_record_ids": not duplicate_ids,
    }
    return {
        "criteria": criteria,
        "passed": all(criteria.values()),
        "row_count": len(records),
        "side_row_count": len(side_rows),
        "source_count": len(source_ids),
        "label_counts": dict(sorted(label_counts.items())),
        "split_label_counts": split_label_counts,
        "side_label_counts": dict(sorted(side_label_counts.items())),
        "duplicate_ids": duplicate_ids,
    }


def labels_for(records: list[dict[str, Any]]) -> np.ndarray:
    return np.array([int(row["binary_label"]) for row in records], dtype=np.int64)


def split_mask(records: list[dict[str, Any]], split: str) -> np.ndarray:
    return np.array([row["split"] == split for row in records], dtype=bool)


def orient_auc(
    train_scores: np.ndarray,
    y_train: np.ndarray,
    holdout_scores: np.ndarray,
    y_holdout: np.ndarray,
) -> dict[str, Any]:
    train_auc = auc_score([float(score) for score in train_scores], [int(label) for label in y_train])
    if train_auc is None:
        return {"discovery_auc": None, "holdout_auc": None, "orientation": None}
    orientation = 1.0
    if train_auc < 0.5:
        train_scores = -train_scores
        holdout_scores = -holdout_scores
        train_auc = auc_score([float(score) for score in train_scores], [int(label) for label in y_train])
        orientation = -1.0
    holdout_auc = auc_score([float(score) for score in holdout_scores], [int(label) for label in y_holdout])
    return {
        "discovery_auc": float(train_auc) if train_auc is not None else None,
        "holdout_auc": float(holdout_auc) if holdout_auc is not None else None,
        "orientation": orientation,
        "train_scores": train_scores,
        "holdout_scores": holdout_scores,
    }


def fit_direction(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_holdout: np.ndarray,
    y_holdout: np.ndarray,
) -> dict[str, Any]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    z_train = (x_train - mean) / std
    z_holdout = (x_holdout - mean) / std
    pos = z_train[y_train == 1]
    neg = z_train[y_train == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("direction fit requires both labels")
    direction = pos.mean(axis=0) - neg.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        train_scores = np.zeros(len(z_train), dtype=np.float32)
        holdout_scores = np.zeros(len(z_holdout), dtype=np.float32)
    else:
        direction = direction / norm
        train_scores = z_train @ direction
        holdout_scores = z_holdout @ direction
    oriented = orient_auc(train_scores, y_train, holdout_scores, y_holdout)
    return {
        "discovery_auc": oriented["discovery_auc"],
        "holdout_auc": oriented["holdout_auc"],
        "orientation": oriented["orientation"],
        "direction_norm": norm,
        "train_scores": oriented["train_scores"],
        "holdout_scores": oriented["holdout_scores"],
    }


def metric_payload(fit: dict[str, Any]) -> dict[str, Any]:
    return {
        "discovery_auc": fit["discovery_auc"],
        "holdout_auc": fit["holdout_auc"],
        "orientation": fit["orientation"],
        "direction_norm": fit.get("direction_norm"),
    }


def first_token_margin(row: dict[str, Any], tokenizer: Any, logits: torch.Tensor) -> float:
    true_token = first_answer_token_id(tokenizer, row["true_capital"])
    override_token = first_answer_token_id(tokenizer, row["override_capital"])
    return float(logits[true_token] - logits[override_token])


def collect_features_and_baselines(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, list[float]], list[dict[str, Any]]]:
    hidden_chunks: list[list[torch.Tensor]] | None = None
    next_token_margins: list[float] = []
    scored_records: list[dict[str, Any]] = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            inputs = tokenizer(
                [row["prompt"] for row in batch],
                return_tensors="pt",
                padding=True,
            ).to(model.device)
            with torch.inference_mode():
                out = model(**inputs, output_hidden_states=True, use_cache=False)
            logits = out.logits[:, -1, :].detach().float().cpu()
            hidden_states = out.hidden_states[1:]
            if hidden_chunks is None:
                hidden_chunks = [[] for _ in hidden_states]
            for layer_index, hidden in enumerate(hidden_states):
                hidden_chunks[layer_index].append(hidden[:, -1, :].detach().float().cpu())
            for offset, row in enumerate(batch):
                next_token_margins.append(first_token_margin(row, tokenizer, logits[offset]))
            print(f"[v15 hidden] rows {start + len(batch)}/{len(records)}")
    finally:
        tokenizer.padding_side = old_padding_side
    for index, row in enumerate(records, start=1):
        true_score = token_logprob(model, tokenizer, row["prompt"], row["true_capital"])
        override_score = token_logprob(model, tokenizer, row["prompt"], row["override_capital"])
        scored = {
            **row,
            "true_candidate_score": true_score,
            "override_candidate_score": override_score,
            "true_minus_override_mean_logprob": float(
                true_score["mean_logprob"] - override_score["mean_logprob"]
            ),
        }
        scored_records.append(scored)
        print(
            f"[v15 candidate] {index:03d}/{len(records):03d} {row['id']} "
            f"true-override={scored['true_minus_override_mean_logprob']:+.4f}"
        )
    if hidden_chunks is None:
        raise ValueError("no hidden features collected")
    hidden_features = {
        f"layer_{layer_index}": torch.cat(chunks, dim=0).numpy().astype(np.float32)
        for layer_index, chunks in enumerate(hidden_chunks)
    }
    baselines = {
        "candidate_score_margin": [
            float(row["true_minus_override_mean_logprob"]) for row in scored_records
        ],
        "next_token_output_margin": next_token_margins,
        "prompt_length": [float(row["prompt_token_count"]) for row in scored_records],
        "final_token_id": [float(row["final_prompt_token_id"]) for row in scored_records],
        "true_candidate_token_count": [
            float(row["true_candidate_score"]["token_count"]) for row in scored_records
        ],
        "override_candidate_token_count": [
            float(row["override_candidate_score"]["token_count"]) for row in scored_records
        ],
        "generated_first_token_id": [float(row["generated_first_token_id"]) for row in scored_records],
        "label_changed_by_normalization": [
            1.0 if row["label_changed_by_normalization"] else 0.0 for row in scored_records
        ],
    }
    return hidden_features, baselines, scored_records


def score_internal_candidates(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    discovery: np.ndarray,
    holdout: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y_train = labels[discovery]
    y_holdout = labels[holdout]
    candidates_payload = []
    selected: dict[str, Any] | None = None
    for name, matrix in sorted(features.items(), key=lambda item: int(item[0].split("_")[1])):
        layer = int(name.split("_")[1])
        fit = fit_direction(matrix[discovery], y_train, matrix[holdout], y_holdout)
        record = {"name": name, "layer": layer, **metric_payload(fit)}
        candidates_payload.append(record)
        if selected is None or (
            -1.0 if record["discovery_auc"] is None else float(record["discovery_auc"]),
            -1.0 if record["holdout_auc"] is None else float(record["holdout_auc"]),
            -int(record["layer"]),
        ) > (
            -1.0 if selected["discovery_auc"] is None else float(selected["discovery_auc"]),
            -1.0 if selected["holdout_auc"] is None else float(selected["holdout_auc"]),
            -int(selected["layer"]),
        ):
            selected = record
    assert selected is not None
    return candidates_payload, selected


def baseline_results(
    baselines: dict[str, list[float]],
    labels: np.ndarray,
    discovery: np.ndarray,
    holdout: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    y_train = labels[discovery]
    y_holdout = labels[holdout]
    results = {}
    for name, values in baselines.items():
        matrix = np.asarray(values, dtype=np.float32).reshape(-1, 1)
        fit = fit_direction(matrix[discovery], y_train, matrix[holdout], y_holdout)
        results[name] = {**metric_payload(fit), "score": name}
    best = max(
        results.items(),
        key=lambda item: -1.0 if item[1]["holdout_auc"] is None else float(item[1]["holdout_auc"]),
    )
    return results, {"name": best[0], **best[1]}


def shuffled_selection_null(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    discovery: np.ndarray,
    holdout: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    y_train = labels[discovery]
    y_holdout = labels[holdout]
    aucs = []
    selected_counts: Counter[str] = Counter()
    feature_items = sorted(features.items(), key=lambda item: int(item[0].split("_")[1]))
    for _ in range(iterations):
        shuffled = y_train.copy()
        rng.shuffle(shuffled)
        selected: dict[str, Any] | None = None
        for name, matrix in feature_items:
            layer = int(name.split("_")[1])
            fit = fit_direction(matrix[discovery], shuffled, matrix[holdout], y_holdout)
            record = {"name": name, "layer": layer, **metric_payload(fit)}
            if selected is None or (
                -1.0 if record["discovery_auc"] is None else float(record["discovery_auc"]),
                -int(record["layer"]),
            ) > (
                -1.0 if selected["discovery_auc"] is None else float(selected["discovery_auc"]),
                -int(selected["layer"]),
            ):
                selected = record
        assert selected is not None
        selected_counts[str(selected["name"])] += 1
        if selected["holdout_auc"] is not None:
            aucs.append(float(selected["holdout_auc"]))
    return {
        "iterations": iterations,
        "holdout_auc_p95": percentile(aucs, 0.95),
        "holdout_auc_max": max(aucs) if aucs else None,
        "holdout_auc_mean": float(sum(aucs) / len(aucs)) if aucs else None,
        "selected_counts_top10": selected_counts.most_common(10),
    }


def classify(criteria: dict[str, bool]) -> str:
    if not criteria["source_artifact_and_structural_passed"]:
        return "source_artifact_invalid"
    if not criteria["holdout_class_balance_at_least_2_each"]:
        return "holdout_balance_failed"
    if not criteria["selected_hidden_holdout_auc_at_least_0p85"]:
        return "internal_signal_failed"
    if not criteria["selected_hidden_beats_candidate_score_by_0p02"]:
        return "candidate_score_confounded"
    if not criteria["selected_hidden_beats_next_token_output_by_0p02"]:
        return "output_margin_confounded"
    if not criteria["selected_hidden_beats_prompt_length_by_0p02"]:
        return "prompt_length_confounded"
    if not criteria["selected_hidden_beats_final_token_id_by_0p02"]:
        return "final_token_confounded"
    if not criteria["selected_hidden_beats_shuffle_p95_by_0p05"]:
        return "shuffle_null_confounded"
    if all(criteria.values()):
        return "hidden_signature_supported"
    return "mixed_signature_failure"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--source-artifact", type=Path, default=DEFAULT_SOURCE_ARTIFACT)
    parser.add_argument(
        "--artifact-prefix",
        default="mc006_qwen3_1p7b_parametric_fact_override_v15_parser_normalized_signature",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shuffle-iterations", type=int, default=100)
    parser.add_argument("--shuffle-seed", type=int, default=26015)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.time()
    source_hash = sha256_file(args.source_artifact)
    source_result = read_json(args.source_artifact)
    validate_source(source_result)

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    records, side_rows = build_records(source_result, tokenizer)
    structural = structural_check(records, side_rows)
    if not structural["passed"]:
        raise ValueError(f"structural check failed before model scoring: {structural}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    hidden_features, baselines, scored_records = collect_features_and_baselines(
        records,
        tokenizer,
        model,
        args.batch_size,
    )
    labels = labels_for(scored_records)
    discovery = np.array([row["split"] != "holdout" for row in scored_records], dtype=bool)
    holdout = split_mask(scored_records, "holdout")
    internal_candidates, selected_internal = score_internal_candidates(
        hidden_features,
        labels,
        discovery,
        holdout,
    )
    baseline_payload, best_baseline = baseline_results(baselines, labels, discovery, holdout)
    shuffle_null = shuffled_selection_null(
        hidden_features,
        labels,
        discovery,
        holdout,
        args.shuffle_iterations,
        args.shuffle_seed,
    )

    holdout_counts = structural["split_label_counts"]["holdout"]
    selected_auc = float(selected_internal["holdout_auc"] or 0.0)
    candidate_auc = float(baseline_payload["candidate_score_margin"]["holdout_auc"] or 0.0)
    next_token_auc = float(baseline_payload["next_token_output_margin"]["holdout_auc"] or 0.0)
    prompt_length_auc = float(baseline_payload["prompt_length"]["holdout_auc"] or 0.0)
    final_token_auc = float(baseline_payload["final_token_id"]["holdout_auc"] or 0.0)
    shuffle_p95 = float(shuffle_null["holdout_auc_p95"])
    criteria = {
        "source_artifact_and_structural_passed": structural["passed"],
        "holdout_class_balance_at_least_2_each": holdout_counts.get("true_answer", 0) >= 2
        and holdout_counts.get("override_answer", 0) >= 2,
        "selected_hidden_holdout_auc_at_least_0p85": selected_auc >= 0.85,
        "selected_hidden_beats_candidate_score_by_0p02": selected_auc >= candidate_auc + 0.02,
        "selected_hidden_beats_next_token_output_by_0p02": selected_auc >= next_token_auc + 0.02,
        "selected_hidden_beats_prompt_length_by_0p02": selected_auc >= prompt_length_auc + 0.02,
        "selected_hidden_beats_final_token_id_by_0p02": selected_auc >= final_token_auc + 0.02,
        "selected_hidden_beats_shuffle_p95_by_0p05": selected_auc >= shuffle_p95 + 0.05,
    }
    diagnostic_class = classify(criteria)
    summary = {
        "model_id": args.model_id,
        "source_artifact": str(args.source_artifact),
        "source_artifact_sha256": source_hash,
        "primary_row_contract": "V14 selected-template true/override rows",
        "excluded_side_rows": side_rows,
        "label_definition": {"1": "true_answer", "0": "override_answer"},
        "structural": structural,
        "candidate_position": "final_prompt_token",
        "candidate_layer_count": len(hidden_features),
        "selected_internal_candidate": selected_internal,
        "best_baseline": best_baseline,
        "baseline_results": baseline_payload,
        "shuffle_selection_null": shuffle_null,
        "criteria": criteria,
        "passed": diagnostic_class == "hidden_signature_supported",
        "diagnostic_class": diagnostic_class,
    }
    result = {
        "card_id": args.card_id,
        "run_type": RUN_TYPE,
        "model_id": args.model_id,
        "elapsed_s": time.time() - started,
        "summary": summary,
        "internal_candidate_results": internal_candidates,
        "records": [{key: value for key, value in row.items() if key != "prompt"} for row in scored_records],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)
    print(json.dumps({**summary, "output_path": str(output_path)}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
