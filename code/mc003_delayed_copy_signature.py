#!/usr/bin/env python
"""MC003 delayed-copy hidden-signature discovery gate.

This uses the passed MC003 V2 behavior substrate. It probes hidden states at a
teacher-forced answer prefix ("WAIT\nFINAL:") and compares residual directions
against shuffled-label nulls and a first-token output-margin baseline.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc002_known_unknown_smoke import render_for_generation


MODEL_ID = "google/gemma-2-2b-it"
CARD_ID = "MC003"
RESULT_DIR = Path("results/cards/MC003")
ANSWER_PREFIX = "WAIT\nFINAL:"
PRIMARY_EXCLUDED_CONDITIONS = {"wrong_hint_forced"}


def load_behavior_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        result = json.load(f)
    rows = result["records"]
    for row in rows:
        if row["label"] not in {"target_correct", "distractor_followed"}:
            raise ValueError(f"unsupported label in behavior rows: {row['label']}")
    return rows


def first_answer_token_id(tokenizer: Any, answer: str) -> int:
    ids = tokenizer(f" {answer}", add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"tokenless answer: {answer!r}")
    return int(ids[0])


def auc_score(scores: list[float], labels: list[int]) -> float | None:
    pos = [score for score, label in zip(scores, labels) if label == 1]
    neg = [score for score, label in zip(scores, labels) if label == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for pos_score in pos:
        for neg_score in neg:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * q))))
    return sorted_values[index]


def collect_features(
    rows: list[dict[str, Any]],
    model_id: str,
    render_mode: str,
) -> tuple[list[dict[str, Any]], list[list[torch.Tensor]]]:
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
    layer_vectors: list[list[torch.Tensor]] | None = None

    for index, row in enumerate(rows, start=1):
        rendered = render_for_generation(tokenizer, row["prompt"], render_mode)
        prefix = f"{rendered}{ANSWER_PREFIX}"
        inputs = tokenizer(prefix, return_tensors="pt").to(model.device)
        target_token = first_answer_token_id(tokenizer, row["target_word"])
        distractor_token = first_answer_token_id(tokenizer, row["distractor_word"])
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits[0, -1].detach().float().cpu()
        output_margin = float(logits[target_token] - logits[distractor_token])
        hidden_states = outputs.hidden_states[1:]
        if layer_vectors is None:
            layer_vectors = [[] for _ in hidden_states]
        for layer_index, hidden in enumerate(hidden_states):
            layer_vectors[layer_index].append(hidden[0, -1].detach().float().cpu())
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
                "first_token_output_margin": output_margin,
                "primary_row": row["condition"] not in PRIMARY_EXCLUDED_CONDITIONS,
            }
        )
        print(
            f"[{index:03d}/{len(rows):03d}] {row['id']} label={row['label']} "
            f"margin={output_margin:.4f}"
        )

    if layer_vectors is None:
        raise ValueError("no rows collected")
    return feature_rows, layer_vectors


def indices_for(rows: list[dict[str, Any]], *, split: str | None = None, primary_only: bool = True) -> list[int]:
    return [
        index
        for index, row in enumerate(rows)
        if (split is None or row["split"] == split) and (not primary_only or row["primary_row"])
    ]


def labels_for(rows: list[dict[str, Any]], indices: list[int]) -> list[int]:
    return [int(rows[index]["binary_label"]) for index in indices]


def scores_for(scores: list[float], indices: list[int]) -> list[float]:
    return [float(scores[index]) for index in indices]


def direction_scores(vectors: torch.Tensor, direction: torch.Tensor) -> list[float]:
    norm = float(direction.norm())
    if norm == 0.0:
        return [0.0 for _ in range(vectors.shape[0])]
    unit = direction / direction.norm()
    return [float(score) for score in vectors.matmul(unit)]


def fit_direction(vectors: torch.Tensor, labels: list[int], indices: list[int]) -> torch.Tensor:
    pos_indices = [index for index in indices if labels[index] == 1]
    neg_indices = [index for index in indices if labels[index] == 0]
    if not pos_indices or not neg_indices:
        return torch.zeros(vectors.shape[1], dtype=vectors.dtype)
    return vectors[pos_indices].mean(dim=0) - vectors[neg_indices].mean(dim=0)


def evaluate_layers(rows: list[dict[str, Any]], layer_vectors: list[list[torch.Tensor]]) -> dict[str, Any]:
    labels = [int(row["binary_label"]) for row in rows]
    discovery = indices_for(rows, split="discovery")
    holdout = indices_for(rows, split="holdout")
    output_scores = [float(row["first_token_output_margin"]) for row in rows]
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
                "layer": layer_index,
                "discovery_auc": auc_score(scores_for(scores, discovery), labels_for(rows, discovery)),
                "holdout_auc": auc_score(scores_for(scores, holdout), labels_for(rows, holdout)),
            }
        )

    selected = max(
        layer_results,
        key=lambda row: (
            -1.0 if row["discovery_auc"] is None else row["discovery_auc"],
            -1.0 if row["holdout_auc"] is None else row["holdout_auc"],
            -row["layer"],
        ),
    )
    selected_layer = int(selected["layer"])
    selected_scores = all_layer_scores[selected_layer]
    selected_holdout_scores = scores_for(selected_scores, holdout)
    selected_holdout_labels = labels_for(rows, holdout)
    target_first_subgroups: dict[str, Any] = {}
    for value in (True, False):
        subgroup = [index for index in holdout if rows[index]["target_first"] is value]
        target_first_subgroups[str(value).lower()] = {
            "n": len(subgroup),
            "positive_count": sum(labels_for(rows, subgroup)),
            "negative_count": len(subgroup) - sum(labels_for(rows, subgroup)),
            "auc": auc_score(scores_for(selected_scores, subgroup), labels_for(rows, subgroup)),
        }

    rng = random.Random(0)
    selected_vectors = torch.stack(layer_vectors[selected_layer])
    shuffle_aucs = []
    discovery_labels = labels_for(rows, discovery)
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
    shuffle_p95 = percentile(shuffle_aucs, 0.95)
    selected_holdout_auc = selected["holdout_auc"]
    output_holdout = output_holdout_auc if output_holdout_auc is not None else float("nan")
    criteria = {
        "selected_layer_holdout_auc_at_least_0_85": selected_holdout_auc is not None and selected_holdout_auc >= 0.85,
        "selected_layer_above_shuffle_p95_by_0_05": selected_holdout_auc is not None
        and selected_holdout_auc >= shuffle_p95 + 0.05,
        "target_order_subgroup_aucs_at_least_0_75": all(
            subgroup["auc"] is not None and subgroup["auc"] >= 0.75 for subgroup in target_first_subgroups.values()
        ),
        "hidden_auc_beats_output_margin_auc_by_0_02": selected_holdout_auc is not None
        and selected_holdout_auc >= output_holdout + 0.02,
    }

    return {
        "primary_rows": len(indices_for(rows, primary_only=True)),
        "discovery_primary_rows": len(discovery),
        "holdout_primary_rows": len(holdout),
        "output_margin_control": {
            "score": "first_token_target_minus_distractor_logit",
            "discovery_auc": output_discovery_auc,
            "holdout_auc": output_holdout_auc,
        },
        "layer_results": layer_results,
        "selected_layer": selected_layer,
        "selected_layer_result": selected,
        "target_first_holdout_subgroups": target_first_subgroups,
        "shuffle_null": {
            "selected_layer": selected_layer,
            "iterations": len(shuffle_aucs),
            "auc_p95": shuffle_p95,
            "auc_max": max(shuffle_aucs) if shuffle_aucs else None,
        },
        "criteria": criteria,
        "diagnostic_signature_candidate": all(
            value for key, value in criteria.items() if key != "hidden_auc_beats_output_margin_auc_by_0_02"
        ),
        "passed": all(criteria.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--behavior-result", type=Path, required=True)
    parser.add_argument("--artifact-prefix", default="mc003_gemma2_2b_it_delayed_copy_v2_signature")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--render-mode", choices=["raw", "chat"], default="chat")
    parser.add_argument("--limit", type=int, default=0, help="0 means all behavior rows")
    args = parser.parse_args()

    torch.manual_seed(0)
    rows = load_behavior_rows(args.behavior_result)
    if args.limit:
        rows = rows[: args.limit]
    started = time.time()
    feature_rows, layer_vectors = collect_features(rows, args.model_id, args.render_mode)
    summary = evaluate_layers(feature_rows, layer_vectors)
    elapsed = time.time() - started
    result = {
        "card_id": args.card_id,
        "run_type": f"{args.artifact_prefix}_hidden_signature",
        "model_id": args.model_id,
        "behavior_result": str(args.behavior_result),
        "render_mode": args.render_mode,
        "answer_prefix": ANSWER_PREFIX,
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
