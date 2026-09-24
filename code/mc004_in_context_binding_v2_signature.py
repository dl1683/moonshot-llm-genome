#!/usr/bin/env python
"""MC004 V2 condition-balanced early-position signature gate."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc002_known_unknown_smoke import render_for_generation
from mc003_delayed_copy_signature import (
    CARD_ID as DEFAULT_CARD_ID,
    MODEL_ID,
    auc_score,
    direction_scores,
    first_answer_token_id,
    fit_direction,
    indices_for,
    labels_for,
    percentile,
    scores_for,
)


CARD_ID = "MC004"
RESULT_DIR = Path("results/cards/MC004")
SELECTED_CONDITION = "update_prefer_latest"


@dataclass(frozen=True)
class PositionSpec:
    name: str
    suffix: str
    order: int
    description: str


POSITIONS = [
    PositionSpec(
        name="prompt_end",
        suffix="",
        order=0,
        description="final token of rendered prompt before the model answers",
    ),
    PositionSpec(
        name="after_answer_prefix",
        suffix="Answer:",
        order=1,
        description="after teacher-forced Answer prefix",
    ),
    PositionSpec(
        name="after_final_prefix",
        suffix="FINAL:",
        order=2,
        description="after teacher-forced FINAL prefix",
    ),
]


def load_condition_rows(path: Path, condition: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        result = json.load(f)
    rows = []
    for row in result["records"]:
        if row["condition"] != condition or row["label"] not in {"target_correct", "distractor_followed"}:
            continue
        mapped = dict(row)
        mapped["target_word"] = row["target_code"]
        mapped["distractor_word"] = row["distractor_code"]
        rows.append(mapped)
    if not rows:
        raise ValueError(f"no usable rows found for condition {condition!r}")
    labels = {row["label"] for row in rows}
    if labels != {"target_correct", "distractor_followed"}:
        raise ValueError(f"condition {condition!r} does not contain both target/update labels: {labels}")
    return rows


def collect_features(
    rows: list[dict[str, Any]],
    model_id: str,
    render_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, list[list[torch.Tensor]]]]:
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

    feature_rows: list[dict[str, Any]] = []
    position_vectors: dict[str, list[list[torch.Tensor]]] = {}

    for index, row in enumerate(rows, start=1):
        rendered = render_for_generation(tokenizer, row["prompt"], render_mode)
        target_token = first_answer_token_id(tokenizer, row["target_word"])
        distractor_token = first_answer_token_id(tokenizer, row["distractor_word"])
        output_margins: dict[str, float] = {}

        for position in POSITIONS:
            prefix = f"{rendered}{position.suffix}"
            inputs = tokenizer(prefix, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits[0, -1].detach().float().cpu()
            output_margins[position.name] = float(logits[target_token] - logits[distractor_token])
            hidden_states = outputs.hidden_states[1:]
            if position.name not in position_vectors:
                position_vectors[position.name] = [[] for _ in hidden_states]
            for layer_index, hidden in enumerate(hidden_states):
                position_vectors[position.name][layer_index].append(hidden[0, -1].detach().float().cpu())

        feature_rows.append(
            {
                "id": row["id"],
                "source_id": row["source_id"],
                "split": row["split"],
                "condition": row["condition"],
                "target_first": row["target_first"],
                "label": row["label"],
                "binary_label": 1 if row["label"] == "target_correct" else 0,
                "target_word": row["target_word"],
                "distractor_word": row["distractor_word"],
                "target_first_token_id": target_token,
                "distractor_first_token_id": distractor_token,
                "position_output_margins": output_margins,
                "primary_row": row["condition"] == SELECTED_CONDITION,
            }
        )
        margins = ", ".join(f"{name}={value:.4f}" for name, value in output_margins.items())
        print(f"[{index:03d}/{len(rows):03d}] {row['id']} label={row['label']} {margins}")

    return feature_rows, position_vectors


def position_by_name() -> dict[str, PositionSpec]:
    return {position.name: position for position in POSITIONS}


def target_first_subgroup_results(rows: list[dict[str, Any]], scores: list[float], holdout: list[int]) -> dict[str, Any]:
    subgroups: dict[str, Any] = {}
    for value in (True, False):
        subgroup = [index for index in holdout if rows[index]["target_first"] is value]
        labels = labels_for(rows, subgroup)
        subgroups[str(value).lower()] = {
            "n": len(subgroup),
            "positive_count": sum(labels),
            "negative_count": len(labels) - sum(labels),
            "auc": auc_score(scores_for(scores, subgroup), labels),
        }
    return subgroups


def evaluate_position_layers(
    rows: list[dict[str, Any]],
    position_name: str,
    layer_vectors: list[list[torch.Tensor]],
) -> dict[str, Any]:
    labels = [int(row["binary_label"]) for row in rows]
    discovery = indices_for(rows, split="discovery")
    holdout = indices_for(rows, split="holdout")
    output_scores = [float(row["position_output_margins"][position_name]) for row in rows]
    output_discovery_auc = auc_score(scores_for(output_scores, discovery), labels_for(rows, discovery))
    output_holdout_auc = auc_score(scores_for(output_scores, holdout), labels_for(rows, holdout))

    layer_results = []
    all_layer_scores: list[list[float]] = []
    for layer_index, vectors_by_row in enumerate(layer_vectors):
        vectors = torch.stack(vectors_by_row)
        direction = fit_direction(vectors, labels, discovery)
        scores = direction_scores(vectors, direction)
        all_layer_scores.append(scores)
        layer_results.append(
            {
                "position": position_name,
                "layer": layer_index,
                "discovery_auc": auc_score(scores_for(scores, discovery), labels_for(rows, discovery)),
                "holdout_auc": auc_score(scores_for(scores, holdout), labels_for(rows, holdout)),
            }
        )

    selected = max(
        layer_results,
        key=lambda row: (-1.0 if row["discovery_auc"] is None else row["discovery_auc"], -row["layer"]),
    )
    return {
        "position": position_name,
        "output_margin_control": {
            "score": "target_first_token_minus_update_first_token_logit",
            "discovery_auc": output_discovery_auc,
            "holdout_auc": output_holdout_auc,
        },
        "layer_results": layer_results,
        "selected_layer": int(selected["layer"]),
        "selected_layer_result": selected,
        "selected_layer_scores": all_layer_scores[int(selected["layer"])],
    }


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
    parser.add_argument("--artifact-prefix", default="mc004_gemma2_2b_it_in_context_binding_v2_signature")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--render-mode", choices=["raw", "chat"], default="chat")
    parser.add_argument("--limit", type=int, default=0, help="0 means all selected-condition rows")
    args = parser.parse_args()

    _ = DEFAULT_CARD_ID
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
