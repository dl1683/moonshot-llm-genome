#!/usr/bin/env python
"""MC006 V12 matched-surface signature diagnostic for real-after-fiction rows."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v2_repair_20260630T222815.json"
)
RUN_TYPE = "parametric_fact_override_v12_real_after_fiction_signature"
SOURCE_RUN_TYPE = "parametric_fact_override_v2_repair"
PRIMARY_CONDITION = "real_after_fiction"
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


def first_answer_token_id(tokenizer: Any, answer: str) -> int:
    ids = tokenizer(f" {answer}", add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"tokenless answer: {answer!r}")
    return int(ids[0])


def validate_source(result: dict[str, Any]) -> None:
    if result.get("run_type") != SOURCE_RUN_TYPE:
        raise ValueError(f"unexpected run_type: {result.get('run_type')!r}")
    summary = result.get("summary", {})
    structural = summary.get("structural", {})
    if not structural.get("passed"):
        raise ValueError("V2 source artifact structural checks did not pass")
    if "records" not in result:
        raise ValueError("V2 artifact missing records")


def candidate_score(row: dict[str, Any], label: str) -> dict[str, Any]:
    return next(candidate for candidate in row["candidate_scores"] if candidate["label"] == label)


def build_records(source_result: dict[str, Any], tokenizer: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary_rows = []
    side_rows = []
    for row in source_result["records"]:
        if row["condition"] != PRIMARY_CONDITION:
            continue
        if row["selected_label"] not in PRIMARY_LABELS:
            side_rows.append(
                {
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "split": row["split"],
                    "selected_label": row["selected_label"],
                    "selected_answer": row["selected_answer"],
                }
            )
            continue
        prompt_ids = tokenizer(row["prompt"], return_tensors="pt")["input_ids"][0]
        true_score = candidate_score(row, "true_answer")
        override_score = candidate_score(row, "override_answer")
        label = 1 if row["selected_label"] == "true_answer" else 0
        primary_rows.append(
            {
                "id": row["id"].replace("_v2_", "_v12_"),
                "source_row_id": row["id"],
                "source_id": row["source_id"],
                "split": row["split"],
                "condition": row["condition"],
                "country": row["country"],
                "true_capital": row["true_capital"],
                "override_capital": row["override_capital"],
                "lure_capital": row["lure_capital"],
                "selected_label": row["selected_label"],
                "selected_answer": row["selected_answer"],
                "binary_label": label,
                "label_name": row["selected_label"],
                "prompt": row["prompt"],
                "prompt_token_count": int(prompt_ids.shape[0]),
                "final_prompt_token_id": int(prompt_ids[-1]),
                "true_minus_override_mean_logprob": float(row["true_minus_override_mean_logprob"]),
                "true_candidate_mean_logprob": float(true_score["mean_logprob"]),
                "override_candidate_mean_logprob": float(override_score["mean_logprob"]),
                "true_candidate_token_count": int(true_score["token_count"]),
                "override_candidate_token_count": int(override_score["token_count"]),
                "override_token_count": int(override_score["token_count"]),
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
    side_label_counts = Counter(row["selected_label"] for row in side_rows)
    holdout_labels = split_label_counts.get("holdout", {})
    criteria = {
        "exactly_30_binary_rows": len(records) == 30,
        "exactly_10_side_lure_rows": len(side_rows) == 10 and side_label_counts.get("lure_answer", 0) == 10,
        "expected_label_counts": label_counts.get("true_answer", 0) == 9
        and label_counts.get("override_answer", 0) == 21,
        "expected_discovery_counts": split_label_counts.get("discovery", {}).get("true_answer", 0) == 6
        and split_label_counts.get("discovery", {}).get("override_answer", 0) == 12,
        "expected_calibration_counts": split_label_counts.get("calibration", {}).get("true_answer", 0) == 2
        and split_label_counts.get("calibration", {}).get("override_answer", 0) == 4,
        "holdout_has_both_labels": {"true_answer", "override_answer"}.issubset(holdout_labels),
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


def collect_features(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    hidden_chunks: list[list[torch.Tensor]] | None = None
    next_token_margins: list[float] = []
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
            print(f"[v12 hidden] rows {start + len(batch)}/{len(records)}")
    finally:
        tokenizer.padding_side = old_padding_side
    if hidden_chunks is None:
        raise ValueError("no hidden features collected")
    hidden_features = {
        f"layer_{layer_index}": torch.cat(chunks, dim=0).numpy().astype(np.float32)
        for layer_index, chunks in enumerate(hidden_chunks)
    }
    baselines = {
        "candidate_score_margin": [float(row["true_minus_override_mean_logprob"]) for row in records],
        "next_token_output_margin": next_token_margins,
        "prompt_length": [float(row["prompt_token_count"]) for row in records],
        "final_token_id": [float(row["final_prompt_token_id"]) for row in records],
        "override_token_count": [float(row["override_token_count"]) for row in records],
    }
    return hidden_features, baselines


def score_internal_candidates(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    discovery: np.ndarray,
    holdout: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y_train = labels[discovery]
    y_holdout = labels[holdout]
    candidates = []
    selected: dict[str, Any] | None = None
    for name, matrix in sorted(features.items(), key=lambda item: int(item[0].split("_")[1])):
        layer = int(name.split("_")[1])
        fit = fit_direction(matrix[discovery], y_train, matrix[holdout], y_holdout)
        record = {"name": name, "layer": layer, **metric_payload(fit)}
        candidates.append(record)
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
    return candidates, selected


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
    if not criteria["structural_passed"]:
        return "structural_invalid"
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
        default="mc006_qwen3_1p7b_parametric_fact_override_v12_real_after_fiction_signature",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shuffle-iterations", type=int, default=100)
    parser.add_argument("--shuffle-seed", type=int, default=26012)
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

    hidden_features, baselines = collect_features(records, tokenizer, model, args.batch_size)
    labels = labels_for(records)
    discovery = np.array([row["split"] != "holdout" for row in records], dtype=bool)
    holdout = split_mask(records, "holdout")
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
    shuffle_p95 = float(shuffle_null["holdout_auc_p95"])
    criteria = {
        "structural_passed": structural["passed"],
        "holdout_class_balance_at_least_2_each": holdout_counts.get("true_answer", 0) >= 2
        and holdout_counts.get("override_answer", 0) >= 2,
        "selected_hidden_holdout_auc_at_least_0p85": selected_auc >= 0.85,
        "selected_hidden_beats_candidate_score_by_0p02": selected_auc >= candidate_auc + 0.02,
        "selected_hidden_beats_next_token_output_by_0p02": selected_auc >= next_token_auc + 0.02,
        "selected_hidden_beats_prompt_length_by_0p02": selected_auc >= prompt_length_auc + 0.02,
        "selected_hidden_beats_shuffle_p95_by_0p05": selected_auc >= shuffle_p95 + 0.05,
    }
    diagnostic_class = classify(criteria)
    summary = {
        "model_id": args.model_id,
        "source_artifact": str(args.source_artifact),
        "source_artifact_sha256": source_hash,
        "primary_row_contract": "V2 real_after_fiction rows with true/override selected labels",
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
        "records": [{key: value for key, value in row.items() if key != "prompt"} for row in records],
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
