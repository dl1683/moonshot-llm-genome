#!/usr/bin/env python
"""MC001G residualized dense signature discovery for repaired Gemma rows."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mc001_gemma_repair_discovery import (
    collect_rows,
    counts,
    direction_auc,
    label_value,
    load_json,
    logistic_auc,
    one_hot,
    run_model_rows,
    score_baselines,
    select_clean_items,
    split_clean_items,
)


NUISANCE_FIELDS = ["condition", "correct_answer", "wrong_answer", "parsed_answer"]


def nuisance_design(rows: list[dict[str, Any]], train_mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    margin = np.array([row["correct_minus_wrong_logprob"] for row in rows], dtype=np.float32)
    train_margin = margin[train_mask]
    margin_std = float(train_margin.std())
    if margin_std == 0.0:
        margin_std = 1.0
    margin_z = ((margin - float(train_margin.mean())) / margin_std).reshape(-1, 1)
    categorical, names = one_hot(rows, NUISANCE_FIELDS)
    intercept = np.ones((len(rows), 1), dtype=np.float32)
    design = np.concatenate([intercept, margin_z.astype(np.float32), categorical], axis=1)
    return design, ["intercept", "margin_z", *names]


def residualize(
    values: np.ndarray,
    design: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(design[train_mask], values[train_mask], rcond=None)
    return (values - design @ beta).astype(np.float32)


def residualized_baseline(rows: list[dict[str, Any]], y: np.ndarray, split_mask: np.ndarray) -> dict[str, Any]:
    design, names = nuisance_design(rows, split_mask)
    return {
        **logistic_auc(design[split_mask], y[split_mask], design[~split_mask], y[~split_mask]),
        "features": names,
    }


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

    for row in scored_rows:
        row["correct_minus_wrong_logprob"] = row["correct_minus_wrong_logprob_recomputed"]
        row["parsed_answer"] = row["parsed_answer_recomputed"]

    baselines = score_baselines(scored_rows, y, split_mask)
    nuisance, nuisance_features = nuisance_design(scored_rows, split_mask)
    nuisance_baseline = residualized_baseline(scored_rows, y, split_mask)

    layer_metrics: list[dict[str, Any]] = []
    for layer_index, hidden in enumerate(hidden_layers):
        residual_hidden = residualize(hidden, nuisance, split_mask)
        layer_metrics.append(
            {
                "layer": layer_index,
                "raw_direction": direction_auc(hidden[split_mask], y[split_mask], hidden[~split_mask], y[~split_mask]),
                "raw_logistic": logistic_auc(hidden[split_mask], y[split_mask], hidden[~split_mask], y[~split_mask]),
                "residual_direction": direction_auc(
                    residual_hidden[split_mask],
                    y[split_mask],
                    residual_hidden[~split_mask],
                    y[~split_mask],
                ),
                "residual_logistic": logistic_auc(
                    residual_hidden[split_mask],
                    y[split_mask],
                    residual_hidden[~split_mask],
                    y[~split_mask],
                ),
                "residual_train_norm_mean": float(np.linalg.norm(residual_hidden[split_mask], axis=1).mean()),
                "residual_holdout_norm_mean": float(np.linalg.norm(residual_hidden[~split_mask], axis=1).mean()),
            }
        )

    best_residual_direction = max(layer_metrics, key=lambda row: row["residual_direction"]["holdout_auc"])
    best_residual_logistic = max(layer_metrics, key=lambda row: row["residual_logistic"]["holdout_auc"])
    result = {
        "card_id": "MC001G",
        "run_type": f"{args.artifact_prefix}_residualized_dense_signature_discovery",
        "model_id": args.model_id,
        "repair_result": str(args.repair_result),
        "render_mode": args.render_mode,
        "nuisance_features": nuisance_features,
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
        "nuisance_baseline": nuisance_baseline,
        "layer_metrics": layer_metrics,
        "best_residual_direction_layer": best_residual_direction,
        "best_residual_logistic_layer": best_residual_logistic,
        "records": scored_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_residualized_dense_signature_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)

    print(json.dumps({
        "clean_item_count": result["clean_item_count"],
        "selected_label_counts": result["selected_label_counts"],
        "split_label_counts": result["split_label_counts"],
        "margin_baseline": result["baselines"]["margin"],
        "nuisance_baseline": result["nuisance_baseline"],
        "best_residual_direction_layer": result["best_residual_direction_layer"],
        "best_residual_logistic_layer": result["best_residual_logistic_layer"],
        "output_path": str(output_path),
    }, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
