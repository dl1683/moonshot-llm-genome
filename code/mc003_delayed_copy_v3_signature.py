#!/usr/bin/env python
"""MC003 V3 condition-balanced early-position signature gate."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from mc003_delayed_copy_early_signature import (
    POSITIONS,
    collect_features,
    evaluate_position_layers,
    position_by_name,
    target_first_subgroup_results,
)
from mc003_delayed_copy_signature import (
    CARD_ID,
    MODEL_ID,
    RESULT_DIR,
    auc_score,
    direction_scores,
    fit_direction,
    indices_for,
    labels_for,
    percentile,
    scores_for,
)


SELECTED_CONDITION = "wrong_hint_balanced"


def load_condition_rows(path: Path, condition: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        result = json.load(f)
    rows = [
        row
        for row in result["records"]
        if row["condition"] == condition and row["label"] in {"target_correct", "distractor_followed"}
    ]
    if not rows:
        raise ValueError(f"no rows found for condition {condition!r}")
    labels = {row["label"] for row in rows}
    if labels != {"target_correct", "distractor_followed"}:
        raise ValueError(f"condition {condition!r} does not contain both target and distractor labels: {labels}")
    splits = {row["split"] for row in rows}
    if splits != {"discovery", "holdout"}:
        raise ValueError(f"condition {condition!r} does not contain discovery and holdout rows: {splits}")
    return rows


def evaluate(rows: list[dict[str, Any]], position_vectors: dict[str, list[list[torch.Tensor]]]) -> dict[str, Any]:
    discovery = indices_for(rows, split="discovery")
    holdout = indices_for(rows, split="holdout")
    per_position = {
        position.name: evaluate_position_layers(rows, position.name, position_vectors[position.name])
        for position in POSITIONS
    }
    positions = position_by_name()
    selected = max(
        (result["selected_layer_result"] for result in per_position.values()),
        key=lambda row: (
            -1.0 if row["discovery_auc"] is None else row["discovery_auc"],
            -positions[row["position"]].order,
            -row["layer"],
        ),
    )
    selected_position = str(selected["position"])
    selected_layer = int(selected["layer"])
    selected_scores = per_position[selected_position]["selected_layer_scores"]
    target_first_subgroups = target_first_subgroup_results(rows, selected_scores, holdout)

    selected_vectors = torch.stack(position_vectors[selected_position][selected_layer])
    labels = [int(row["binary_label"]) for row in rows]
    discovery_labels = labels_for(rows, discovery)
    selected_holdout_labels = labels_for(rows, holdout)
    rng = random.Random(0)
    shuffle_aucs = []
    for _ in range(100):
        shuffled = discovery_labels[:]
        rng.shuffle(shuffled)
        shuffled_labels = labels[:]
        for index, row_index in enumerate(discovery):
            shuffled_labels[row_index] = shuffled[index]
        direction = fit_direction(selected_vectors, shuffled_labels, discovery)
        scores = direction_scores(selected_vectors, direction)
        auc = auc_score(scores_for(scores, holdout), selected_holdout_labels)
        if auc is not None:
            shuffle_aucs.append(auc)

    selected_output_auc = per_position[selected_position]["output_margin_control"]["holdout_auc"]
    output_auc = selected_output_auc if selected_output_auc is not None else float("nan")
    selected_holdout_auc = selected["holdout_auc"]
    shuffle_p95 = percentile(shuffle_aucs, 0.95)
    primary_conditions = sorted({row["condition"] for row in rows if row["primary_row"]})
    criteria = {
        "selected_hidden_holdout_auc_at_least_0_85": selected_holdout_auc is not None and selected_holdout_auc >= 0.85,
        "selected_hidden_above_shuffle_p95_by_0_05": selected_holdout_auc is not None
        and selected_holdout_auc >= shuffle_p95 + 0.05,
        "target_order_subgroup_aucs_at_least_0_75": all(
            subgroup["auc"] is not None and subgroup["auc"] >= 0.75 for subgroup in target_first_subgroups.values()
        ),
        "hidden_auc_beats_same_position_output_margin_by_0_02": selected_holdout_auc is not None
        and selected_holdout_auc >= output_auc + 0.02,
        "single_condition_primary_rows": primary_conditions == [SELECTED_CONDITION],
    }

    return {
        "selected_condition": SELECTED_CONDITION,
        "primary_conditions": primary_conditions,
        "primary_rows": len(indices_for(rows, primary_only=True)),
        "discovery_primary_rows": len(discovery),
        "holdout_primary_rows": len(holdout),
        "positions": [
            {
                "name": position.name,
                "order": position.order,
                "suffix": position.suffix,
                "description": position.description,
            }
            for position in POSITIONS
        ],
        "position_results": {
            name: {key: value for key, value in result.items() if key != "selected_layer_scores"}
            for name, result in per_position.items()
        },
        "selected_position": selected_position,
        "selected_layer": selected_layer,
        "selected_result": selected,
        "selected_position_output_margin_control": per_position[selected_position]["output_margin_control"],
        "target_first_holdout_subgroups": target_first_subgroups,
        "shuffle_null": {
            "selected_position": selected_position,
            "selected_layer": selected_layer,
            "iterations": len(shuffle_aucs),
            "auc_p95": shuffle_p95,
            "auc_max": max(shuffle_aucs) if shuffle_aucs else None,
        },
        "criteria": criteria,
        "diagnostic_signature_candidate": all(
            value
            for key, value in criteria.items()
            if key != "hidden_auc_beats_same_position_output_margin_by_0_02"
        ),
        "passed": all(criteria.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--behavior-result", type=Path, required=True)
    parser.add_argument("--condition", default=SELECTED_CONDITION)
    parser.add_argument("--artifact-prefix", default="mc003_gemma2_2b_it_delayed_copy_v3_condition_balanced_signature")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--render-mode", choices=["raw", "chat"], default="chat")
    parser.add_argument("--limit", type=int, default=0, help="0 means all selected-condition rows")
    args = parser.parse_args()

    if args.condition != SELECTED_CONDITION:
        raise ValueError(f"condition is frozen to {SELECTED_CONDITION!r}; got {args.condition!r}")

    torch.manual_seed(0)
    rows = load_condition_rows(args.behavior_result, args.condition)
    if args.limit:
        rows = rows[: args.limit]
    started = time.time()
    feature_rows, position_vectors = collect_features(rows, args.model_id, args.render_mode)
    summary = evaluate(feature_rows, position_vectors)
    elapsed = time.time() - started
    result = {
        "card_id": args.card_id,
        "run_type": f"{args.artifact_prefix}_hidden_signature",
        "model_id": args.model_id,
        "behavior_result": str(args.behavior_result),
        "selected_condition": args.condition,
        "render_mode": args.render_mode,
        "candidate_positions": [position.__dict__ for position in POSITIONS],
        "limit": args.limit or None,
        "elapsed_s": elapsed,
        "summary": summary,
        "records": feature_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_{args.render_mode}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)

    print(json.dumps({**summary, "output_path": str(output_path)}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
