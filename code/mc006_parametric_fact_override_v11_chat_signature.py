#!/usr/bin/env python
"""MC006 V11 hidden-state signature audit for the V10 chat substrate."""

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
    RESULT_DIR
    / "mc006_qwen3_1p7b_parametric_fact_override_v10_chat_generated_20260630T232318.json"
)
RUN_TYPE = "parametric_fact_override_v11_chat_signature"
SOURCE_RUN_TYPE = "parametric_fact_override_v10_chat_generated"
SOURCE_DIAGNOSTIC_CLASS = "chat_generated_v10_substrate_passed"
CONDITIONS = ("mode_real", "mode_fictional")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_generation_token_id(tokenizer: Any, answer: str) -> int:
    ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"tokenless answer: {answer!r}")
    return int(ids[0])


def validate_source(result: dict[str, Any]) -> None:
    if result.get("run_type") != SOURCE_RUN_TYPE:
        raise ValueError(f"unexpected run_type: {result.get('run_type')!r}")
    summary = result.get("summary", {})
    if summary.get("diagnostic_class") != SOURCE_DIAGNOSTIC_CLASS:
        raise ValueError(f"unexpected diagnostic_class: {summary.get('diagnostic_class')!r}")
    if not summary.get("passed"):
        raise ValueError("V10 source artifact did not pass")
    if "records" not in result or "source_contrasts" not in summary:
        raise ValueError("V10 artifact missing records or source_contrasts")


def clean_source_ids(source_result: dict[str, Any]) -> set[str]:
    return {
        row["source_id"]
        for row in source_result["summary"]["source_contrasts"]
        if row.get("clean_contrast")
    }


def build_records(source_result: dict[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    clean_sources = clean_source_ids(source_result)
    rows = []
    for row in source_result["records"]:
        if row["source_id"] not in clean_sources:
            continue
        if row["condition"] not in CONDITIONS:
            raise ValueError(f"unexpected condition: {row['condition']!r}")
        if not row.get("parseable"):
            raise ValueError(f"primary row is not parseable: {row['id']}")
        if row["selected_label"] != row["expected_label"]:
            raise ValueError(f"primary row is not behavior-clean: {row['id']}")
        prompt_ids = tokenizer(row["chat_prompt"], return_tensors="pt")["input_ids"][0]
        label = 1 if row["condition"] == "mode_real" else 0
        rows.append(
            {
                "id": row["id"].replace("_v10_", "_v11_"),
                "source_row_id": row["id"],
                "source_id": row["source_id"],
                "split": row["split"],
                "condition": row["condition"],
                "country": row["country"],
                "true_capital": row["true_capital"],
                "override_capital": row["override_capital"],
                "lure_capital": row["lure_capital"],
                "requested_mode": row["requested_mode"],
                "selected_label": row["selected_label"],
                "selected_answer": row["selected_answer"],
                "generated_text": row["generated_text"],
                "binary_label": label,
                "label_name": "mode_real" if label == 1 else "mode_fictional",
                "chat_prompt": row["chat_prompt"],
                "rendered_token_count": int(prompt_ids.shape[0]),
                "final_prompt_token_id": int(prompt_ids[-1]),
                "generated_first_token_id": int(row["generated_token_ids"][0]),
            }
        )
    rows.sort(key=lambda item: (item["split"], item["source_id"], item["condition"]))
    return rows


def structural_check(records: list[dict[str, Any]]) -> dict[str, Any]:
    source_ids = {row["source_id"] for row in records}
    holdout_sources = {row["source_id"] for row in records if row["split"] == "holdout"}
    condition_counts = Counter(row["condition"] for row in records)
    label_counts = Counter(row["label_name"] for row in records)
    split_label_counts: dict[str, dict[str, int]] = {}
    for split in ("discovery", "calibration", "holdout"):
        split_rows = [row for row in records if row["split"] == split]
        split_label_counts[split] = dict(sorted(Counter(row["label_name"] for row in split_rows).items()))
    source_condition_counts = Counter((row["source_id"], row["condition"]) for row in records)
    duplicate_ids = [row_id for row_id, count in Counter(row["id"] for row in records).items() if count > 1]
    split_by_source: dict[str, set[str]] = {}
    for row in records:
        split_by_source.setdefault(row["source_id"], set()).add(row["split"])
    source_split_valid = all(len(splits) == 1 for splits in split_by_source.values())
    criteria = {
        "exactly_20_clean_sources": len(source_ids) == 20,
        "exactly_5_holdout_sources": len(holdout_sources) == 5,
        "exactly_40_rows": len(records) == 40,
        "two_conditions_per_source": all(count == 1 for count in source_condition_counts.values())
        and len(source_condition_counts) == 20 * len(CONDITIONS),
        "twenty_rows_per_condition": all(condition_counts.get(condition, 0) == 20 for condition in CONDITIONS),
        "balanced_binary_labels": label_counts.get("mode_real", 0) == 20
        and label_counts.get("mode_fictional", 0) == 20,
        "source_split_valid": source_split_valid,
        "no_duplicate_record_ids": not duplicate_ids,
        "discovery_has_both_labels": {"mode_real", "mode_fictional"}.issubset(
            split_label_counts.get("discovery", {})
        ),
        "holdout_has_both_labels": {"mode_real", "mode_fictional"}.issubset(
            split_label_counts.get("holdout", {})
        ),
    }
    return {
        "criteria": criteria,
        "passed": all(criteria.values()),
        "source_count": len(source_ids),
        "holdout_source_count": len(holdout_sources),
        "row_count": len(records),
        "condition_counts": dict(sorted(condition_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "split_label_counts": split_label_counts,
        "duplicate_ids": duplicate_ids,
    }


def split_mask(records: list[dict[str, Any]], split: str) -> np.ndarray:
    return np.array([row["split"] == split for row in records], dtype=bool)


def labels_for(records: list[dict[str, Any]]) -> np.ndarray:
    return np.array([int(row["binary_label"]) for row in records], dtype=np.int64)


def orient_auc(
    train_scores: np.ndarray,
    y_train: np.ndarray,
    holdout_scores: np.ndarray,
    y_holdout: np.ndarray,
) -> dict[str, Any]:
    train_auc = auc_score([float(score) for score in train_scores], [int(label) for label in y_train])
    if train_auc is None:
        return {
            "discovery_auc": None,
            "holdout_auc": None,
            "orientation": None,
            "train_scores": train_scores,
            "holdout_scores": holdout_scores,
        }
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
    true_token = first_generation_token_id(tokenizer, row["true_capital"])
    override_token = first_generation_token_id(tokenizer, row["override_capital"])
    return float(logits[true_token] - logits[override_token])


def collect_features(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    hidden_chunks: list[list[torch.Tensor]] | None = None
    output_margins: list[float] = []
    requested_mode_scores: list[float] = []
    prompt_lengths: list[float] = []
    final_token_ids: list[float] = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            inputs = tokenizer(
                [row["chat_prompt"] for row in batch],
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
                output_margins.append(first_token_margin(row, tokenizer, logits[offset]))
                requested_mode_scores.append(1.0 if row["requested_mode"] == "REAL_WORLD_CAPITAL" else 0.0)
                prompt_lengths.append(float(row["rendered_token_count"]))
                final_token_ids.append(float(row["final_prompt_token_id"]))
            print(f"[v11 hidden] rows {start + len(batch)}/{len(records)}")
    finally:
        tokenizer.padding_side = old_padding_side
    if hidden_chunks is None:
        raise ValueError("no hidden features collected")
    hidden_features = {
        f"layer_{layer_index}": torch.cat(chunks, dim=0).numpy().astype(np.float32)
        for layer_index, chunks in enumerate(hidden_chunks)
    }
    baselines = {
        "output_margin": output_margins,
        "requested_mode": requested_mode_scores,
        "prompt_length": prompt_lengths,
        "final_token_id": final_token_ids,
    }
    return hidden_features, baselines


def score_internal_candidates(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    discovery: np.ndarray,
    holdout: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    y_train = labels[discovery]
    y_holdout = labels[holdout]
    candidates = []
    selected: dict[str, Any] | None = None
    selected_fit: dict[str, Any] | None = None
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
            selected_fit = fit
    assert selected is not None and selected_fit is not None
    return candidates, selected, selected_fit


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


def holdout_source_ordering(
    records: list[dict[str, Any]],
    holdout_scores: np.ndarray,
    holdout: np.ndarray,
) -> dict[str, Any]:
    holdout_records = [row for row, keep in zip(records, holdout, strict=True) if keep]
    result = {}
    for source_id in sorted({row["source_id"] for row in holdout_records}):
        mode_real_scores = [
            float(holdout_scores[index])
            for index, row in enumerate(holdout_records)
            if row["source_id"] == source_id and row["condition"] == "mode_real"
        ]
        mode_fictional_scores = [
            float(holdout_scores[index])
            for index, row in enumerate(holdout_records)
            if row["source_id"] == source_id and row["condition"] == "mode_fictional"
        ]
        real_score = mode_real_scores[0] if mode_real_scores else float("nan")
        fictional_score = mode_fictional_scores[0] if mode_fictional_scores else float("nan")
        result[source_id] = {
            "mode_real_rows": len(mode_real_scores),
            "mode_fictional_rows": len(mode_fictional_scores),
            "mode_real_score": real_score,
            "mode_fictional_score": fictional_score,
            "passes": real_score > fictional_score,
        }
    return result


def classify(criteria: dict[str, bool]) -> str:
    if not criteria["structural_passed"]:
        return "structural_invalid"
    if not criteria["selected_hidden_holdout_auc_at_least_0p85"]:
        return "internal_signal_failed"
    if not criteria["selected_hidden_beats_requested_mode_by_0p02"]:
        return "requested_mode_confounded"
    if not criteria["selected_hidden_beats_output_margin_by_0p02"]:
        return "output_margin_confounded"
    if not criteria["selected_hidden_beats_prompt_length_by_0p02"]:
        return "prompt_length_confounded"
    if not criteria["selected_hidden_beats_shuffle_p95_by_0p05"]:
        return "shuffle_null_confounded"
    if not criteria["holdout_source_mode_real_gt_fictional_all"]:
        return "source_pair_instability"
    if all(criteria.values()):
        return "hidden_signature_supported"
    return "mixed_signature_failure"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--source-artifact", type=Path, default=DEFAULT_SOURCE_ARTIFACT)
    parser.add_argument("--artifact-prefix", default="mc006_qwen3_1p7b_parametric_fact_override_v11_chat_signature")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shuffle-iterations", type=int, default=100)
    parser.add_argument("--shuffle-seed", type=int, default=26011)
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

    records = build_records(source_result, tokenizer)
    structural = structural_check(records)
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
    internal_candidates, selected_internal, selected_fit = score_internal_candidates(
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
    holdout_source_pairs = holdout_source_ordering(
        records,
        np.asarray(selected_fit["holdout_scores"], dtype=np.float32),
        holdout,
    )

    selected_auc = float(selected_internal["holdout_auc"] or 0.0)
    output_auc = float(baseline_payload["output_margin"]["holdout_auc"] or 0.0)
    requested_mode_auc = float(baseline_payload["requested_mode"]["holdout_auc"] or 0.0)
    prompt_length_auc = float(baseline_payload["prompt_length"]["holdout_auc"] or 0.0)
    shuffle_p95 = float(shuffle_null["holdout_auc_p95"])
    criteria = {
        "structural_passed": structural["passed"],
        "selected_hidden_holdout_auc_at_least_0p85": selected_auc >= 0.85,
        "selected_hidden_beats_output_margin_by_0p02": selected_auc >= output_auc + 0.02,
        "selected_hidden_beats_requested_mode_by_0p02": selected_auc >= requested_mode_auc + 0.02,
        "selected_hidden_beats_prompt_length_by_0p02": selected_auc >= prompt_length_auc + 0.02,
        "selected_hidden_beats_shuffle_p95_by_0p05": selected_auc >= shuffle_p95 + 0.05,
        "holdout_source_mode_real_gt_fictional_all": all(
            row["passes"] for row in holdout_source_pairs.values()
        ),
    }
    diagnostic_class = classify(criteria)
    elapsed = time.time() - started
    summary = {
        "model_id": args.model_id,
        "source_artifact": str(args.source_artifact),
        "source_artifact_sha256": source_hash,
        "primary_row_contract": "V10 clean source-level contrasts only",
        "label_definition": {
            "1": "mode_real",
            "0": "mode_fictional",
        },
        "structural": structural,
        "candidate_position": "final_rendered_prompt_token",
        "candidate_layer_count": len(hidden_features),
        "selected_internal_candidate": selected_internal,
        "best_baseline": best_baseline,
        "baseline_results": baseline_payload,
        "shuffle_selection_null": shuffle_null,
        "holdout_source_ordering": holdout_source_pairs,
        "criteria": criteria,
        "passed": diagnostic_class == "hidden_signature_supported",
        "diagnostic_class": diagnostic_class,
    }
    result = {
        "card_id": args.card_id,
        "run_type": RUN_TYPE,
        "model_id": args.model_id,
        "elapsed_s": elapsed,
        "summary": summary,
        "internal_candidate_results": internal_candidates,
        "records": [
            {key: value for key, value in row.items() if key != "chat_prompt"}
            for row in records
        ],
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
