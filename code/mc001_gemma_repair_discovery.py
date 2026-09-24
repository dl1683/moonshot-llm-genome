#!/usr/bin/env python
"""MC001G dense signature discovery on the repaired Gemma behavior substrate."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc001_logit_smoke import letter_token_ids, render_for_score


PRIMARY_CONDITIONS = ("wrong_disclaimed", "wrong_unsure")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def item_id(row: dict[str, Any]) -> str:
    return row["id"].split("__", 1)[0]


def select_clean_items(rows: list[dict[str, Any]]) -> dict[str, str]:
    by_item_condition: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        by_item_condition[(item_id(row), row["condition"])] = row

    clean: dict[str, str] = {}
    all_items = sorted({item_id(row) for row in rows})
    for item in all_items:
        no_hint = by_item_condition.get((item, "no_hint"))
        correct_hint = by_item_condition.get((item, "correct_hint"))
        if (
            no_hint
            and correct_hint
            and no_hint["label"] == "truth_following"
            and correct_hint["label"] == "truth_following"
        ):
            clean[item] = no_hint["correct_answer"]
    return clean


def split_clean_items(clean_items: dict[str, str]) -> dict[str, str]:
    split: dict[str, str] = {}
    by_letter: dict[str, list[str]] = defaultdict(list)
    for item, letter in clean_items.items():
        by_letter[letter].append(item)
    for letter, items in by_letter.items():
        for index, item in enumerate(sorted(items)):
            split[item] = "holdout" if index % 3 == 2 else "discovery"
    return split


def label_value(label: str) -> int:
    if label == "truth_following":
        return 1
    if label == "user_agreement_error":
        return 0
    raise ValueError(f"unsupported label for binary signature discovery: {label}")


def one_hot(rows: list[dict[str, Any]], fields: list[str]) -> tuple[np.ndarray, list[str]]:
    names: list[str] = []
    columns: list[np.ndarray] = []
    for field in fields:
        values = sorted({str(row[field]) for row in rows})
        for value in values:
            names.append(f"{field}={value}")
            columns.append(np.array([1.0 if str(row[field]) == value else 0.0 for row in rows]))
    if not columns:
        return np.zeros((len(rows), 0), dtype=np.float32), names
    return np.stack(columns, axis=1).astype(np.float32), names


def orient_auc(train_scores: np.ndarray, y_train: np.ndarray, eval_scores: np.ndarray, y_eval: np.ndarray) -> dict[str, Any]:
    train_auc = float(roc_auc_score(y_train, train_scores))
    orientation = 1.0
    if train_auc < 0.5:
        train_scores = -train_scores
        eval_scores = -eval_scores
        train_auc = float(roc_auc_score(y_train, train_scores))
        orientation = -1.0
    return {
        "train_auc": train_auc,
        "holdout_auc": float(roc_auc_score(y_eval, eval_scores)),
        "orientation": orientation,
    }


def logistic_auc(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray) -> dict[str, Any]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=0),
    )
    model.fit(x_train, y_train)
    train_scores = model.decision_function(x_train)
    eval_scores = model.decision_function(x_eval)
    return orient_auc(train_scores, y_train, eval_scores, y_eval)


def direction_auc(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray) -> dict[str, Any]:
    truth_mean = x_train[y_train == 1].mean(axis=0)
    agree_mean = x_train[y_train == 0].mean(axis=0)
    direction = truth_mean - agree_mean
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("zero direction norm")
    direction = direction / norm
    return {
        **orient_auc(x_train @ direction, y_train, x_eval @ direction, y_eval),
        "direction_norm": norm,
    }


def score_baselines(rows: list[dict[str, Any]], y: np.ndarray, split_mask: np.ndarray) -> dict[str, Any]:
    train = split_mask
    holdout = ~split_mask
    result: dict[str, Any] = {}

    margin = np.array([[row["correct_minus_wrong_logprob"]] for row in rows], dtype=np.float32)
    result["margin"] = orient_auc(margin[train, 0], y[train], margin[holdout, 0], y[holdout])

    for name, fields in {
        "condition": ["condition"],
        "correct_letter": ["correct_answer"],
        "wrong_letter": ["wrong_answer"],
        "condition_correct_wrong": ["condition", "correct_answer", "wrong_answer"],
        "baseline_answer": ["parsed_answer"],
    }.items():
        features, feature_names = one_hot(rows, fields)
        result[name] = {
            **logistic_auc(features[train], y[train], features[holdout], y[holdout]),
            "features": feature_names,
        }
    return result


def collect_rows(
    rows: list[dict[str, Any]],
    clean_items: dict[str, str],
    split_items: dict[str, str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        item = item_id(row)
        if item not in clean_items:
            continue
        if row["condition"] not in PRIMARY_CONDITIONS:
            continue
        if row["label"] not in {"truth_following", "user_agreement_error"}:
            continue
        selected.append({**row, "item_id": item, "split": split_items[item]})
    return selected


def run_model_rows(
    rows: list[dict[str, Any]],
    model_id: str,
    render_mode: str,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    option_ids = letter_token_ids(tokenizer)
    scored_rows: list[dict[str, Any]] = []
    hidden_by_layer: list[list[np.ndarray]] | None = None

    for index, row in enumerate(rows, start=1):
        rendered = render_for_score(tokenizer, row["prompt"], render_mode)
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits[0, -1].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        scores = {
            letter: max(float(log_probs[token_id].item()) for token_id in token_ids)
            for letter, token_ids in option_ids.items()
        }
        parsed = max(scores, key=scores.get)
        if hidden_by_layer is None:
            hidden_by_layer = [[] for _ in outputs.hidden_states[1:]]
        for layer_index, hidden in enumerate(outputs.hidden_states[1:]):
            hidden_by_layer[layer_index].append(hidden[0, -1].float().cpu().numpy())
        scored_rows.append(
            {
                **row,
                "index": index,
                "rendered_prompt": rendered,
                "parsed_answer_recomputed": parsed,
                "option_logprobs_recomputed": scores,
                "correct_minus_wrong_logprob_recomputed": scores[row["correct_answer"]] - scores[row["wrong_answer"]],
            }
        )
        print(f"[{index:03d}/{len(rows):03d}] {row['id']} split={row['split']} label={row['label']}")

    assert hidden_by_layer is not None
    return scored_rows, [np.stack(layer_rows, axis=0).astype(np.float32) for layer_rows in hidden_by_layer]


def counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/gemma-2-2b")
    parser.add_argument("--repair-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cards/MC001G"))
    parser.add_argument("--artifact-prefix", default="mc001g_gemma2_2b_repair")
    parser.add_argument("--render-mode", choices=["raw", "chat"], default="raw")
    args = parser.parse_args()

    torch.manual_seed(0)
    repair = load_json(args.repair_result)
    repair_rows = repair["records"]
    clean_items = select_clean_items(repair_rows)
    split_items = split_clean_items(clean_items)
    selected_rows = collect_rows(repair_rows, clean_items, split_items)
    y = np.array([label_value(row["label"]) for row in selected_rows], dtype=np.int64)
    split_mask = np.array([row["split"] == "discovery" for row in selected_rows], dtype=bool)

    if len(set(y[split_mask])) != 2 or len(set(y[~split_mask])) != 2:
        raise RuntimeError("discovery and holdout splits must both contain truth and agreement labels")

    started = time.time()
    scored_rows, hidden_layers = run_model_rows(selected_rows, args.model_id, args.render_mode)
    elapsed = time.time() - started

    # Use recomputed margins for baselines in the same forward pass as the hidden states.
    for row in scored_rows:
        row["correct_minus_wrong_logprob"] = row["correct_minus_wrong_logprob_recomputed"]
        row["parsed_answer"] = row["parsed_answer_recomputed"]

    baselines = score_baselines(scored_rows, y, split_mask)
    layer_metrics: list[dict[str, Any]] = []
    for layer_index, hidden in enumerate(hidden_layers):
        layer_metrics.append(
            {
                "layer": layer_index,
                "direction": direction_auc(hidden[split_mask], y[split_mask], hidden[~split_mask], y[~split_mask]),
                "logistic": logistic_auc(hidden[split_mask], y[split_mask], hidden[~split_mask], y[~split_mask]),
            }
        )

    best_direction = max(layer_metrics, key=lambda row: row["direction"]["holdout_auc"])
    best_logistic = max(layer_metrics, key=lambda row: row["logistic"]["holdout_auc"])
    result = {
        "card_id": "MC001G",
        "run_type": f"{args.artifact_prefix}_dense_signature_discovery",
        "model_id": args.model_id,
        "repair_result": str(args.repair_result),
        "render_mode": args.render_mode,
        "primary_conditions": list(PRIMARY_CONDITIONS),
        "clean_item_count": len(clean_items),
        "clean_items_by_correct": dict(sorted(Counter(clean_items.values()).items())),
        "selected_row_count": len(selected_rows),
        "selected_label_counts": counts(selected_rows, "label"),
        "split_counts": counts(selected_rows, "split"),
        "split_label_counts": {
            split: dict(sorted(Counter(row["label"] for row in selected_rows if row["split"] == split).items()))
            for split in ["discovery", "holdout"]
        },
        "elapsed_s": elapsed,
        "baselines": baselines,
        "layer_metrics": layer_metrics,
        "best_direction_layer": best_direction,
        "best_logistic_layer": best_logistic,
        "records": scored_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_dense_signature_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)

    print(json.dumps({
        "clean_item_count": result["clean_item_count"],
        "selected_label_counts": result["selected_label_counts"],
        "split_label_counts": result["split_label_counts"],
        "baselines": result["baselines"],
        "best_direction_layer": result["best_direction_layer"],
        "best_logistic_layer": result["best_logistic_layer"],
        "output_path": str(output_path),
    }, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
