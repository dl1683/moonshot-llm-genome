#!/usr/bin/env python
"""MC005 V26 internal row-signature diagnostic over the V25 artifact."""

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

from mc001_qwen3_controlled_v10_head_localization import set_eager_attention, tokenizer_padding_side
from mc003_delayed_copy_signature import auc_score, percentile
from mc005_associative_lookup_reliability_v4 import SELECTED_BAND, SELECTED_LAYERS
from mc005_associative_lookup_source_edge import CARD_ID, MODEL_ID, RESULT_DIR


DEFAULT_V25_PATH = (
    RESULT_DIR
    / "mc005_qwen3_1p7b_response_marker_v25_factorial_row_heterogeneity_20260630T205454.json"
)
DISCOVERY_SEED = 233
HOLDOUT_SEED = 239
CANDIDATE_LAYERS = [20, 21, 22, 23, 24, 25, 26]
CANDIDATE_POSITIONS = ["target_value", "distractor_value", "random_value", "final_label", "final_colon"]
SCALAR_BASELINES = [
    "baseline_margin",
    "target_source_position",
    "distractor_source_position",
    "target_distractor_token_distance",
    "abs_target_distractor_token_distance",
    "target_slot",
    "distractor_slot",
]
LAYOUT_LINEAR_FIELDS = [
    "baseline_margin",
    "target_source_position",
    "distractor_source_position",
    "target_distractor_token_distance",
    "abs_target_distractor_token_distance",
    "target_slot",
    "distractor_slot",
    "target_before_distractor",
]


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_v25(v25: dict[str, Any]) -> None:
    if v25.get("run_type") != "associative_lookup_response_marker_v25_factorial_row_heterogeneity":
        raise ValueError(f"unexpected source run_type: {v25.get('run_type')!r}")
    summary = v25.get("summary", {})
    if summary.get("diagnostic_class") != "factorial_cell_contrast_failed":
        raise ValueError(f"unexpected V25 diagnostic class: {summary.get('diagnostic_class')!r}")
    if "row_diagnostics" not in v25 or "rows" not in v25:
        raise ValueError("V25 artifact missing rows or row_diagnostics")


def build_records(v25: dict[str, Any]) -> list[dict[str, Any]]:
    rows_by_id = {
        row["id"]: row
        for row in v25["rows"]
        if row.get("mode") == "lookup"
    }
    records = []
    for diag in v25["row_diagnostics"]:
        if not diag.get("parent_effect_row"):
            continue
        row = rows_by_id[diag["row_id"]]
        record = {
            "row_id": diag["row_id"],
            "base_family_id": diag["base_family_id"],
            "seed": int(diag["seed"]),
            "split": "discovery" if int(diag["seed"]) == DISCOVERY_SEED else "holdout",
            "label": int(bool(diag["all_three_margin_row"])),
            "target_position_group": diag["target_position_group"],
            "distractor_relation": diag["distractor_relation"],
            "baseline_margin": float(diag["baseline_margin"]),
            "target_source_position": float(diag["target_source_position"]),
            "distractor_source_position": float(diag["distractor_source_position"]),
            "target_distractor_token_distance": float(diag["target_distractor_token_distance"]),
            "abs_target_distractor_token_distance": abs(float(diag["target_distractor_token_distance"])),
            "target_slot": float(diag["target_slot"]),
            "distractor_slot": float(diag["distractor_slot"]),
            "target_before_distractor": 1.0 if bool(diag["target_before_distractor"]) else 0.0,
            "rendered_prompt": row["rendered_prompt"],
            "positions": row["positions"],
        }
        records.append(record)
    records.sort(key=lambda item: (int(item["seed"]), item["row_id"]))
    seeds = {int(record["seed"]) for record in records}
    if seeds != {DISCOVERY_SEED, HOLDOUT_SEED}:
        raise ValueError(f"unexpected V25 lookup seeds: {sorted(seeds)}")
    return records


def labels_for(records: list[dict[str, Any]]) -> np.ndarray:
    return np.array([int(record["label"]) for record in records], dtype=np.int64)


def split_mask(records: list[dict[str, Any]], split: str) -> np.ndarray:
    return np.array([record["split"] == split for record in records], dtype=bool)


def label_balance(records: list[dict[str, Any]], labels: np.ndarray) -> dict[str, Any]:
    result = {}
    for split in ("discovery", "holdout"):
        mask = split_mask(records, split)
        split_labels = labels[mask]
        positive = int(split_labels.sum())
        negative = int(len(split_labels) - positive)
        result[split] = {
            "rows": int(len(split_labels)),
            "positive_all_three": positive,
            "negative_non_all_three": negative,
            "valid": positive >= 20 and negative >= 40,
        }
    return result


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


def metric_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "discovery_auc": metrics["discovery_auc"],
        "holdout_auc": metrics["holdout_auc"],
        "orientation": metrics["orientation"],
        "direction_norm": metrics.get("direction_norm"),
    }


def collect_hidden_features(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    layers: list[int],
    positions: list[str],
) -> dict[str, np.ndarray]:
    chunks: dict[str, list[torch.Tensor]] = {
        f"l{layer}_{position}": []
        for layer in layers
        for position in positions
    }
    with tokenizer_padding_side(tokenizer, "left"):
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            inputs = tokenizer(
                [record["rendered_prompt"] for record in batch],
                return_tensors="pt",
                padding=True,
            ).to(model.device)
            seq_lens = [int(value) for value in inputs["attention_mask"].sum(dim=1).detach().cpu().tolist()]
            max_len = int(inputs["input_ids"].shape[-1])
            shifted = {
                position: [
                    int(record["positions"][position]) + (max_len - seq_len)
                    for record, seq_len in zip(batch, seq_lens, strict=True)
                ]
                for position in positions
            }
            with torch.inference_mode():
                out = model(**inputs, output_hidden_states=True, use_cache=False)
            for layer in layers:
                hidden = out.hidden_states[layer + 1].detach().float().cpu()
                for position in positions:
                    vectors = torch.stack([
                        hidden[index, shifted[position][index], :]
                        for index in range(len(batch))
                    ])
                    chunks[f"l{layer}_{position}"].append(vectors)
            print(f"[v26 hidden] rows {start + len(batch)}/{len(records)}")
    return {
        key: torch.cat(value, dim=0).numpy().astype(np.float32)
        for key, value in chunks.items()
    }


def feature_matrix(records: list[dict[str, Any]], fields: list[str]) -> np.ndarray:
    return np.array(
        [[float(record[field]) for field in fields] for record in records],
        dtype=np.float32,
    )


def score_internal_candidates(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    discovery: np.ndarray,
    holdout: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    y_train = labels[discovery]
    y_holdout = labels[holdout]
    candidates = []
    selected_fit: dict[str, Any] | None = None
    for name, matrix in features.items():
        layer_part, position = name.split("_", 1)
        layer = int(layer_part[1:])
        fit = fit_direction(matrix[discovery], y_train, matrix[holdout], y_holdout)
        record = {
            "name": name,
            "layer": layer,
            "position": position,
            **metric_payload(fit),
        }
        candidates.append(record)
        if selected_fit is None or (
            -1.0 if record["discovery_auc"] is None else float(record["discovery_auc"]),
            -1.0 if record["holdout_auc"] is None else float(record["holdout_auc"]),
            -int(record["layer"]),
            record["position"],
        ) > (
            -1.0 if selected_fit["record"]["discovery_auc"] is None else float(selected_fit["record"]["discovery_auc"]),
            -1.0 if selected_fit["record"]["holdout_auc"] is None else float(selected_fit["record"]["holdout_auc"]),
            -int(selected_fit["record"]["layer"]),
            selected_fit["record"]["position"],
        ):
            selected_fit = {"record": record, "fit": fit}
    assert selected_fit is not None
    candidates.sort(
        key=lambda row: (
            -1.0 if row["discovery_auc"] is None else -float(row["discovery_auc"]),
            -1.0 if row["holdout_auc"] is None else -float(row["holdout_auc"]),
            int(row["layer"]),
            row["position"],
        )
    )
    return candidates, selected_fit["record"], selected_fit["fit"]


def score_baselines(
    records: list[dict[str, Any]],
    labels: np.ndarray,
    discovery: np.ndarray,
    holdout: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    y_train = labels[discovery]
    y_holdout = labels[holdout]
    results = {}
    for field in SCALAR_BASELINES:
        matrix = feature_matrix(records, [field])
        fit = fit_direction(matrix[discovery], y_train, matrix[holdout], y_holdout)
        results[field] = {**metric_payload(fit), "fields": [field]}
    layout = feature_matrix(records, LAYOUT_LINEAR_FIELDS)
    layout_fit = fit_direction(layout[discovery], y_train, layout[holdout], y_holdout)
    results["layout_margin_linear"] = {
        **metric_payload(layout_fit),
        "fields": LAYOUT_LINEAR_FIELDS,
    }
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
    feature_items = sorted(features.items())
    for _ in range(iterations):
        shuffled = y_train.copy()
        rng.shuffle(shuffled)
        selected: dict[str, Any] | None = None
        for name, matrix in feature_items:
            layer_part, position = name.split("_", 1)
            layer = int(layer_part[1:])
            fit = fit_direction(matrix[discovery], shuffled, matrix[holdout], y_holdout)
            record = {
                "name": name,
                "layer": layer,
                "position": position,
                **metric_payload(fit),
            }
            if selected is None or (
                -1.0 if record["discovery_auc"] is None else float(record["discovery_auc"]),
                -int(record["layer"]),
                record["position"],
            ) > (
                -1.0 if selected["discovery_auc"] is None else float(selected["discovery_auc"]),
                -int(selected["layer"]),
                selected["position"],
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


def subgroup_aucs(
    records: list[dict[str, Any]],
    labels: np.ndarray,
    holdout: np.ndarray,
    selected_scores: np.ndarray,
) -> dict[str, Any]:
    holdout_records = [record for record, keep in zip(records, holdout, strict=True) if keep]
    holdout_labels = labels[holdout]
    groups = {}
    for field in ("target_position_group", "distractor_relation"):
        values = sorted({str(record[field]) for record in holdout_records})
        for value in values:
            indices = [index for index, record in enumerate(holdout_records) if str(record[field]) == value]
            group_labels = holdout_labels[indices]
            positive = int(group_labels.sum())
            negative = int(len(group_labels) - positive)
            key = f"{field}_{value}"
            eligible = positive >= 8 and negative >= 8
            groups[key] = {
                "field": field,
                "value": value,
                "rows": len(indices),
                "positive_all_three": positive,
                "negative_non_all_three": negative,
                "eligible": eligible,
                "auc": auc_score(
                    [float(selected_scores[index]) for index in indices],
                    [int(label) for label in group_labels],
                )
                if eligible
                else None,
            }
    return groups


def classify_diagnostic(criteria: dict[str, bool]) -> str:
    if all(criteria.values()):
        return "internal_row_signature_supported"
    if not criteria["label_balance_valid"]:
        return "label_balance_failed"
    if not criteria["selected_internal_holdout_auc_at_least_0p70"]:
        return "internal_holdout_signal_failed"
    if not criteria["selected_internal_beats_best_baseline_by_0p05"]:
        return "baseline_confounded"
    if not criteria["selected_internal_beats_shuffle_p95_by_0p03"]:
        return "shuffle_null_confounded"
    if not criteria["eligible_subgroups_auc_at_least_0p60"]:
        return "subgroup_unstable"
    return "mixed_failure"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--source-artifact", type=Path, default=DEFAULT_V25_PATH)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_response_marker_v26_internal_row_signature")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--candidate-layers", default=",".join(str(layer) for layer in CANDIDATE_LAYERS))
    parser.add_argument("--shuffle-iterations", type=int, default=100)
    parser.add_argument("--shuffle-seed", type=int, default=26001)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    candidate_layers = parse_ints(args.candidate_layers)
    source_hash = sha256_file(args.source_artifact)
    v25 = read_json(args.source_artifact)
    validate_v25(v25)
    records = build_records(v25)
    labels = labels_for(records)
    discovery = split_mask(records, "discovery")
    holdout = split_mask(records, "holdout")
    balance = label_balance(records, labels)

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    set_eager_attention(model)
    model.eval()

    started = time.time()
    hidden_features = collect_hidden_features(
        records,
        tokenizer,
        model,
        args.batch_size,
        candidate_layers,
        CANDIDATE_POSITIONS,
    )
    internal_candidates, selected_internal, selected_fit = score_internal_candidates(
        hidden_features,
        labels,
        discovery,
        holdout,
    )
    baselines, best_baseline = score_baselines(records, labels, discovery, holdout)
    shuffle_null = shuffled_selection_null(
        hidden_features,
        labels,
        discovery,
        holdout,
        args.shuffle_iterations,
        args.shuffle_seed,
    )
    groups = subgroup_aucs(records, labels, holdout, np.asarray(selected_fit["holdout_scores"]))

    selected_holdout_auc = float(selected_internal["holdout_auc"] or 0.0)
    best_baseline_auc = float(best_baseline["holdout_auc"] or 0.0)
    shuffle_p95 = float(shuffle_null["holdout_auc_p95"])
    criteria = {
        "label_balance_valid": all(item["valid"] for item in balance.values()),
        "selected_internal_holdout_auc_at_least_0p70": selected_holdout_auc >= 0.70,
        "selected_internal_beats_best_baseline_by_0p05": selected_holdout_auc >= best_baseline_auc + 0.05,
        "selected_internal_beats_shuffle_p95_by_0p03": selected_holdout_auc >= shuffle_p95 + 0.03,
        "eligible_subgroups_auc_at_least_0p60": all(
            (not group["eligible"]) or (group["auc"] is not None and float(group["auc"]) >= 0.60)
            for group in groups.values()
        ),
    }
    elapsed = time.time() - started

    summary = {
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "model_id": args.model_id,
        "source_artifact": str(args.source_artifact),
        "source_artifact_sha256": source_hash,
        "discovery_seed": DISCOVERY_SEED,
        "holdout_seed": HOLDOUT_SEED,
        "candidate_layers": candidate_layers,
        "candidate_positions": CANDIDATE_POSITIONS,
        "row_count": len(records),
        "label_balance": balance,
        "selected_internal_candidate": selected_internal,
        "best_baseline": best_baseline,
        "shuffle_selection_null": shuffle_null,
        "holdout_subgroups": groups,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "diagnostic_class": classify_diagnostic(criteria),
    }
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_response_marker_v26_internal_row_signature",
        "model_id": args.model_id,
        "elapsed_s": elapsed,
        "summary": summary,
        "internal_candidate_results": internal_candidates,
        "baseline_results": baselines,
        "records": [
            {key: value for key, value in record.items() if key not in {"rendered_prompt"}}
            for record in records
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
