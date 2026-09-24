#!/usr/bin/env python
"""MC001G pre-hint margin matched dense discovery for repaired Gemma rows."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
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
    orient_auc,
    run_model_rows,
    select_clean_items,
    split_clean_items,
)


MATCH_CONDITIONS = ("wrong_disclaimed", "wrong_unsure")


def item_id(row: dict[str, Any]) -> str:
    return row["id"].split("__", 1)[0]


def attach_item_margins(rows: list[dict[str, Any]], repair_rows: list[dict[str, Any]], bin_width: float) -> list[dict[str, Any]]:
    by_item_condition = {(item_id(row), row["condition"]): row for row in repair_rows}
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = row["item_id"]
        no_hint = by_item_condition[(item, "no_hint")]
        correct_hint = by_item_condition[(item, "correct_hint")]
        no_margin = float(no_hint["correct_minus_wrong_logprob"])
        correct_hint_margin = float(correct_hint["correct_minus_wrong_logprob"])
        enriched.append(
            {
                **row,
                "no_hint_margin": no_margin,
                "correct_hint_margin": correct_hint_margin,
                "no_hint_margin_bin": int(np.floor(no_margin / bin_width)),
            }
        )
    return enriched


def matched_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[(row["split"], row["no_hint_margin_bin"])][row["label"]].append(row)

    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        truth = sorted(grouped[key]["truth_following"], key=lambda row: row["id"])
        agree = sorted(grouped[key]["user_agreement_error"], key=lambda row: row["id"])
        take = min(len(truth), len(agree))
        selected.extend(truth[:take])
        selected.extend(agree[:take])
    return sorted(selected, key=lambda row: (row["split"], row["no_hint_margin_bin"], row["label"], row["id"]))


def scalar_auc(rows: list[dict[str, Any]], y: np.ndarray, train_mask: np.ndarray, field: str) -> dict[str, Any]:
    values = np.array([float(row[field]) for row in rows], dtype=np.float32)
    return orient_auc(values[train_mask], y[train_mask], values[~train_mask], y[~train_mask])


def baseline_auc(rows: list[dict[str, Any]], y: np.ndarray, train_mask: np.ndarray) -> dict[str, Any]:
    result = {
        "same_row_margin": scalar_auc(rows, y, train_mask, "correct_minus_wrong_logprob"),
        "no_hint_margin": scalar_auc(rows, y, train_mask, "no_hint_margin"),
        "correct_hint_margin": scalar_auc(rows, y, train_mask, "correct_hint_margin"),
    }
    for name, fields in {
        "no_hint_margin_bin": ["no_hint_margin_bin"],
        "condition": ["condition"],
        "correct_letter": ["correct_answer"],
        "wrong_letter": ["wrong_answer"],
        "condition_correct_wrong": ["condition", "correct_answer", "wrong_answer"],
        "no_hint_bin_condition_letters": ["no_hint_margin_bin", "condition", "correct_answer", "wrong_answer"],
    }.items():
        features, feature_names = one_hot(rows, fields)
        result[name] = {
            **logistic_auc(features[train_mask], y[train_mask], features[~train_mask], y[~train_mask]),
            "features": feature_names,
        }
    return result


def summarize_match(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in ["discovery", "holdout"]:
        split_rows = [row for row in rows if row["split"] == split]
        summary[split] = {
            "n": len(split_rows),
            "label_counts": dict(sorted(Counter(row["label"] for row in split_rows).items())),
            "bins": {
                str(bin_id): dict(sorted(Counter(row["label"] for row in bin_rows).items()))
                for bin_id, bin_rows in sorted(
                    {
                        bin_id: [row for row in split_rows if row["no_hint_margin_bin"] == bin_id]
                        for bin_id in sorted({row["no_hint_margin_bin"] for row in split_rows})
                    }.items()
                )
            },
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/gemma-2-2b")
    parser.add_argument("--repair-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cards/MC001G"))
    parser.add_argument("--artifact-prefix", default="mc001g_gemma2_2b_repair")
    parser.add_argument("--render-mode", choices=["raw", "chat"], default="raw")
    parser.add_argument("--bin-width", type=float, default=0.5)
    args = parser.parse_args()

    torch.manual_seed(0)
    repair = load_json(args.repair_result)
    repair_rows = repair["records"]
    clean_items = select_clean_items(repair_rows)
    split_items = split_clean_items(clean_items)
    primary_rows = collect_rows(repair_rows, clean_items, split_items)
    primary_rows = [row for row in primary_rows if row["condition"] in MATCH_CONDITIONS]
    enriched = attach_item_margins(primary_rows, repair_rows, args.bin_width)
    selected_rows = matched_rows(enriched)

    y = np.array([label_value(row["label"]) for row in selected_rows], dtype=np.int64)
    split_mask = np.array([row["split"] == "discovery" for row in selected_rows], dtype=bool)
    if len(set(y[split_mask])) != 2 or len(set(y[~split_mask])) != 2:
        raise RuntimeError("matched discovery and holdout splits must both contain truth and agreement labels")

    started = time.time()
    scored_rows, hidden_layers = run_model_rows(selected_rows, args.model_id, args.render_mode)
    elapsed = time.time() - started
    for row in scored_rows:
        row["correct_minus_wrong_logprob"] = row["correct_minus_wrong_logprob_recomputed"]
        row["parsed_answer"] = row["parsed_answer_recomputed"]

    baselines = baseline_auc(scored_rows, y, split_mask)
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
        "run_type": f"{args.artifact_prefix}_pre_hint_margin_matched_discovery",
        "model_id": args.model_id,
        "repair_result": str(args.repair_result),
        "render_mode": args.render_mode,
        "bin_width": args.bin_width,
        "match_conditions": list(MATCH_CONDITIONS),
        "clean_item_count": len(clean_items),
        "primary_row_count": len(primary_rows),
        "selected_row_count": len(selected_rows),
        "selected_label_counts": counts(selected_rows, "label"),
        "split_counts": counts(selected_rows, "split"),
        "match_summary": summarize_match(selected_rows),
        "elapsed_s": elapsed,
        "baselines": baselines,
        "layer_metrics": layer_metrics,
        "best_direction_layer": best_direction,
        "best_logistic_layer": best_logistic,
        "records": scored_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_pre_hint_margin_matched_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)

    print(json.dumps({
        "selected_label_counts": result["selected_label_counts"],
        "match_summary": result["match_summary"],
        "baselines": result["baselines"],
        "best_direction_layer": result["best_direction_layer"],
        "best_logistic_layer": result["best_logistic_layer"],
        "output_path": str(output_path),
    }, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
