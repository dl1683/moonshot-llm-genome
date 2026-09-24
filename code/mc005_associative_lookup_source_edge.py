#!/usr/bin/env python
"""MC005 associative lookup source-edge audit."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc001_qwen3_controlled_v10_head_localization import (
    install_source_mask_hook,
    set_eager_attention,
    tokenizer_padding_side,
)
from mc003_delayed_copy_signature import auc_score, percentile


CARD_ID = "MC005"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/MC005")

KEY_WORDS = [
    "river",
    "garden",
    "market",
    "planet",
    "window",
    "forest",
    "castle",
    "bridge",
    "engine",
    "circle",
    "flame",
    "island",
    "letter",
    "mirror",
    "rocket",
    "school",
    "temple",
    "valley",
    "button",
    "camera",
    "desert",
    "harbor",
    "jacket",
    "ladder",
    "needle",
    "pocket",
    "saddle",
    "ticket",
    "winter",
    "yellow",
    "anchor",
    "bottle",
    "candle",
    "dragon",
    "fabric",
    "glacier",
    "jungle",
    "magnet",
    "nectar",
    "orbit",
    "puzzle",
    "quartz",
    "sailor",
    "tunnel",
    "walnut",
    "yogurt",
    "basket",
    "hammer",
    "lantern",
    "marble",
    "pepper",
    "ribbon",
    "silver",
    "velvet",
    "violet",
    "copper",
    "orange",
    "cotton",
    "helmet",
    "pencil",
]

VALUE_WORDS = [
    "velvet",
    "copper",
    "silver",
    "orange",
    "marble",
    "cotton",
    "pepper",
    "thunder",
    "violet",
    "harbor",
    "ladder",
    "pencil",
    "winter",
    "sugar",
    "helmet",
    "ribbon",
    "anchor",
    "bottle",
    "candle",
    "dragon",
    "fabric",
    "glacier",
    "jungle",
    "magnet",
    "nectar",
    "orbit",
    "puzzle",
    "quartz",
    "sailor",
    "tunnel",
    "walnut",
    "yogurt",
    "basket",
    "camera",
    "desert",
    "engine",
    "forest",
    "garden",
    "island",
    "jacket",
    "market",
    "mirror",
    "planet",
    "rocket",
    "school",
    "temple",
    "ticket",
    "valley",
    "window",
    "yellow",
    "bridge",
    "button",
    "circle",
    "flame",
    "hammer",
    "lantern",
    "needle",
    "pocket",
    "river",
    "saddle",
]


def first_token_id(tokenizer: Any, word: str) -> int:
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if not ids:
        raise ValueError(f"word {word!r} produced no token ids")
    return int(ids[0])


def is_single_scored_token(tokenizer: Any, word: str) -> bool:
    return len(tokenizer.encode(" " + word, add_special_tokens=False)) == 1


def valid_words(tokenizer: Any, words: list[str]) -> list[str]:
    return [word for word in words if is_single_scored_token(tokenizer, word)]


def split_for_index(index: int) -> str:
    return "holdout" if index % 3 == 0 else "discovery"


def render_prompt(pairs: list[tuple[str, str]], query_key: str) -> str:
    lines = ["Reference pairs:"]
    lines.extend(f"- {key}: {value}" for key, value in pairs)
    lines.append(f"- {query_key}:")
    return "\n".join(lines)


def token_position_for_word(prompt: str, offsets: list[list[int]], word: str, occurrence: int = 0) -> int:
    start = 0
    char_index = -1
    for _ in range(occurrence + 1):
        char_index = prompt.index(word, start)
        start = char_index + 1
    candidates = [index for index, (left, right) in enumerate(offsets) if left <= char_index < right]
    if not candidates:
        raise ValueError(f"could not find token position for {word!r} occurrence {occurrence}")
    return int(candidates[0])


def build_rows(
    tokenizer: Any,
    row_count: int,
    pair_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    keys = valid_words(tokenizer, KEY_WORDS)
    values = valid_words(tokenizer, VALUE_WORDS)
    if len(keys) < pair_count or len(values) < pair_count:
        raise ValueError(f"not enough single-token words: keys={len(keys)} values={len(values)}")

    rows: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for row_index in range(row_count):
        local = random.Random(rng.randint(0, 10_000_000) + row_index)
        row_keys = local.sample(keys, pair_count)
        row_values = local.sample(values, pair_count)
        pairs = list(zip(row_keys, row_values, strict=True))
        query_pair_index = row_index % pair_count
        distractor_pair_index = (query_pair_index + 1 + (row_index % (pair_count - 1))) % pair_count
        if distractor_pair_index == query_pair_index:
            distractor_pair_index = (query_pair_index + 1) % pair_count
        query_key, target_value = pairs[query_pair_index]
        _, distractor_value = pairs[distractor_pair_index]
        other_value_indices = [
            index for index in range(pair_count) if index not in {query_pair_index, distractor_pair_index}
        ]
        random_value_index = other_value_indices[row_index % len(other_value_indices)]
        prompt = render_prompt(pairs, query_key)
        offset_enc = tokenizer(prompt, return_offsets_mapping=True)
        offsets = [[int(left), int(right)] for left, right in offset_enc["offset_mapping"]]
        input_ids = [int(value) for value in offset_enc["input_ids"]]
        value_positions = {}
        for pair_index, (_, value) in enumerate(pairs):
            pos = token_position_for_word(prompt, offsets, value)
            value_positions[str(pair_index)] = pos
            expected_token_id = first_token_id(tokenizer, value)
            if input_ids[pos] != expected_token_id:
                decoded = tokenizer.decode([input_ids[pos]])
                raise ValueError(
                    f"position token mismatch for {value!r}: position={pos} "
                    f"decoded={decoded!r} expected_id={expected_token_id}"
                )
        final_key_position = token_position_for_word(prompt, offsets, query_key, occurrence=1)
        rows.append(
            {
                "id": f"mc005_row_{row_index:03d}",
                "source_id": f"lookup_{row_index:03d}",
                "split": split_for_index(row_index),
                "pairs": [{"key": key, "value": value} for key, value in pairs],
                "pair_count": pair_count,
                "query_pair_index": query_pair_index,
                "distractor_pair_index": distractor_pair_index,
                "random_value_index": random_value_index,
                "query_key": query_key,
                "target_value": target_value,
                "distractor_value": distractor_value,
                "random_value": pairs[random_value_index][1],
                "target_token_id": first_token_id(tokenizer, target_value),
                "distractor_token_id": first_token_id(tokenizer, distractor_value),
                "positions": {
                    "target_value": int(value_positions[str(query_pair_index)]),
                    "distractor_value": int(value_positions[str(distractor_pair_index)]),
                    "random_value": int(value_positions[str(random_value_index)]),
                    "final_query_key": int(final_key_position),
                },
                "target_before_distractor": query_pair_index < distractor_pair_index,
                "query_pair_low_half": query_pair_index < (pair_count / 2),
                "rendered_prompt": prompt,
                "token_length": len(input_ids),
            }
        )
    return rows


def shifted_positions_for_batch(rows: list[dict[str, Any]], source_key: str, seq_lens: list[int], max_len: int) -> list[list[int]]:
    shifted: list[list[int]] = []
    for row, seq_len in zip(rows, seq_lens, strict=True):
        pad = max_len - seq_len
        shifted.append([int(row["positions"][source_key]) + pad])
    return shifted


def install_intervention_hooks(
    model: Any,
    rows: list[dict[str, Any]],
    seq_lens: list[int],
    max_len: int,
    source_key: str,
    layer_index: int | None,
    head_indices: list[int],
    all_layers: bool,
) -> list[Any]:
    layers = range(model.config.num_hidden_layers) if all_layers else [int(layer_index)]
    shifted = shifted_positions_for_batch(rows, source_key, seq_lens, max_len)
    return [
        install_source_mask_hook(model, int(layer), head_indices, shifted)
        for layer in layers
    ]


def score_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    source_key: str | None = None,
    layer_index: int | None = None,
    head_indices: list[int] | None = None,
    all_layers: bool = False,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    heads = head_indices or list(range(model.config.num_attention_heads))
    with tokenizer_padding_side(tokenizer, "left"):
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            inputs = tokenizer([row["rendered_prompt"] for row in batch], return_tensors="pt", padding=True).to(
                model.device
            )
            handles: list[Any] = []
            if source_key is not None:
                if layer_index is None and not all_layers:
                    raise ValueError("layer_index is required unless all_layers is true")
                seq_lens = [int(value) for value in inputs["attention_mask"].sum(dim=1).detach().cpu().tolist()]
                max_len = int(inputs["input_ids"].shape[-1])
                handles = install_intervention_hooks(
                    model,
                    batch,
                    seq_lens,
                    max_len,
                    source_key,
                    layer_index,
                    heads,
                    all_layers,
                )
            try:
                with torch.inference_mode():
                    out = model(**inputs, use_cache=False, logits_to_keep=1)
            finally:
                for handle in handles:
                    handle.remove()
            for index, row in enumerate(batch):
                logits = out.logits[index, -1, :].detach().float()
                target_logit = float(logits[int(row["target_token_id"])])
                distractor_logit = float(logits[int(row["distractor_token_id"])])
                margin = target_logit - distractor_logit
                features[row["id"]] = {
                    "target_logit": target_logit,
                    "distractor_logit": distractor_logit,
                    "target_minus_distractor_margin": margin,
                    "target_wins": margin > 0.0,
                }
    return features


def collect_attention_scores(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
) -> tuple[list[int], list[str], list[dict[str, Any]], list[list[list[float]]]]:
    n_layers = int(model.config.num_hidden_layers)
    n_heads = int(model.config.num_attention_heads)
    labels: list[int] = []
    splits: list[str] = []
    candidate_records: list[dict[str, Any]] = []
    scores = [[[0.0 for _ in range(2 * len(rows))] for _ in range(n_heads)] for _ in range(n_layers)]

    for row_index, row in enumerate(rows, start=1):
        inputs = tokenizer(row["rendered_prompt"], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model(**inputs, output_attentions=True, use_cache=False, logits_to_keep=1)
        target_pos = int(row["positions"]["target_value"])
        distractor_pos = int(row["positions"]["distractor_value"])
        candidate_base = 2 * (row_index - 1)
        labels.extend([1, 0])
        splits.extend([row["split"], row["split"]])
        candidate_records.extend(
            [
                {
                    "row_id": row["id"],
                    "split": row["split"],
                    "candidate": "target_value",
                    "label": 1,
                    "target_before_distractor": row["target_before_distractor"],
                    "query_pair_low_half": row["query_pair_low_half"],
                },
                {
                    "row_id": row["id"],
                    "split": row["split"],
                    "candidate": "distractor_value",
                    "label": 0,
                    "target_before_distractor": row["target_before_distractor"],
                    "query_pair_low_half": row["query_pair_low_half"],
                },
            ]
        )
        for layer_index, attention in enumerate(out.attentions):
            final_attention = attention[0, :, -1, :].detach().float().cpu()
            for head_index in range(n_heads):
                scores[layer_index][head_index][candidate_base] = float(final_attention[head_index, target_pos])
                scores[layer_index][head_index][candidate_base + 1] = float(final_attention[head_index, distractor_pos])
        print(f"[attention {row_index:03d}/{len(rows):03d}] {row['id']} split={row['split']}")

    return labels, splits, candidate_records, scores


def indices_where(values: list[str], wanted: str) -> list[int]:
    return [index for index, value in enumerate(values) if value == wanted]


def select_attention_head(
    labels: list[int],
    splits: list[str],
    scores: list[list[list[float]]],
) -> dict[str, Any]:
    discovery = indices_where(splits, "discovery")
    holdout = indices_where(splits, "holdout")
    layer_results = []
    for layer_index, layer_scores in enumerate(scores):
        for head_index, head_scores in enumerate(layer_scores):
            discovery_auc = auc_score([head_scores[i] for i in discovery], [labels[i] for i in discovery])
            holdout_auc = auc_score([head_scores[i] for i in holdout], [labels[i] for i in holdout])
            layer_results.append(
                {
                    "layer": layer_index,
                    "head": head_index,
                    "discovery_auc": discovery_auc,
                    "holdout_auc": holdout_auc,
                    "mean_discovery_score": float(sum(head_scores[i] for i in discovery) / max(1, len(discovery))),
                }
            )
    selected = max(
        layer_results,
        key=lambda row: (
            -1.0 if row["discovery_auc"] is None else row["discovery_auc"],
            row["mean_discovery_score"],
            -row["layer"],
            -row["head"],
        ),
    )
    return {"selected": selected, "layer_head_results": layer_results}


def subgroup_aucs(
    candidate_records: list[dict[str, Any]],
    labels: list[int],
    splits: list[str],
    selected_scores: list[float],
) -> dict[str, Any]:
    holdout = [index for index, split in enumerate(splits) if split == "holdout"]
    groups: dict[str, Any] = {}
    for field in ("target_before_distractor", "query_pair_low_half"):
        for value in (True, False):
            indices = [
                index
                for index in holdout
                if bool(candidate_records[index][field]) is value
            ]
            group_labels = [labels[index] for index in indices]
            key = f"{field}_{str(value).lower()}"
            groups[key] = {
                "n": len(indices),
                "positive_count": sum(group_labels),
                "negative_count": len(group_labels) - sum(group_labels),
                "auc": auc_score([selected_scores[index] for index in indices], group_labels),
            }
    return groups


def shuffled_selection_null(
    labels: list[int],
    splits: list[str],
    scores: list[list[list[float]]],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    discovery = indices_where(splits, "discovery")
    holdout = indices_where(splits, "holdout")
    true_holdout_labels = [labels[i] for i in holdout]
    aucs = []
    selected_counts: Counter[str] = Counter()
    for _ in range(iterations):
        shuffled_labels = labels[:]
        discovery_labels = [labels[i] for i in discovery]
        rng.shuffle(discovery_labels)
        for offset, index in enumerate(discovery):
            shuffled_labels[index] = discovery_labels[offset]
        best: dict[str, Any] | None = None
        for layer_index, layer_scores in enumerate(scores):
            for head_index, head_scores in enumerate(layer_scores):
                discovery_auc = auc_score(
                    [head_scores[i] for i in discovery],
                    [shuffled_labels[i] for i in discovery],
                )
                candidate = {
                    "layer": layer_index,
                    "head": head_index,
                    "discovery_auc": discovery_auc,
                }
                if best is None or (
                    -1.0 if candidate["discovery_auc"] is None else candidate["discovery_auc"],
                    -candidate["layer"],
                    -candidate["head"],
                ) > (
                    -1.0 if best["discovery_auc"] is None else best["discovery_auc"],
                    -best["layer"],
                    -best["head"],
                ):
                    best = candidate
        assert best is not None
        selected_counts[f"l{best['layer']}_h{best['head']}"] += 1
        selected_scores = scores[int(best["layer"])][int(best["head"])]
        holdout_auc = auc_score([selected_scores[i] for i in holdout], true_holdout_labels)
        if holdout_auc is not None:
            aucs.append(float(holdout_auc))
    return {
        "iterations": iterations,
        "auc_p95": percentile(aucs, 0.95),
        "auc_max": max(aucs) if aucs else None,
        "selected_counts_top5": selected_counts.most_common(5),
    }


def summarize_arm(
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    arm: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    deltas = [
        float(arm[row["id"]]["target_minus_distractor_margin"])
        - float(baseline[row["id"]]["target_minus_distractor_margin"])
        for row in rows
    ]
    before_wins = [bool(baseline[row["id"]]["target_wins"]) for row in rows]
    after_wins = [bool(arm[row["id"]]["target_wins"]) for row in rows]
    before_margins = [float(baseline[row["id"]]["target_minus_distractor_margin"]) for row in rows]
    after_margins = [float(arm[row["id"]]["target_minus_distractor_margin"]) for row in rows]
    return {
        "n": len(rows),
        "baseline_target_wins": sum(before_wins),
        "arm_target_wins": sum(after_wins),
        "target_win_loss": sum(before_wins) - sum(after_wins),
        "flipped_target_to_distractor": sum(before and not after for before, after in zip(before_wins, after_wins, strict=True)),
        "flipped_distractor_to_target": sum((not before) and after for before, after in zip(before_wins, after_wins, strict=True)),
        "baseline_mean_margin": sum(before_margins) / max(1, len(before_margins)),
        "arm_mean_margin": sum(after_margins) / max(1, len(after_margins)),
        "mean_delta": sum(deltas) / max(1, len(deltas)),
        "min_delta": min(deltas) if deltas else None,
        "max_delta": max(deltas) if deltas else None,
    }


def select_causal_layer(
    discovery_rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    baseline: dict[str, dict[str, Any]],
    batch_size: int,
) -> dict[str, Any]:
    layer_results = []
    all_heads = list(range(model.config.num_attention_heads))
    for layer_index in range(model.config.num_hidden_layers):
        arm = score_rows(
            discovery_rows,
            tokenizer,
            model,
            batch_size,
            source_key="target_value",
            layer_index=layer_index,
            head_indices=all_heads,
        )
        summary = summarize_arm(discovery_rows, baseline, arm)
        layer_results.append({"layer": layer_index, **summary})
        print(
            f"[causal-layer {layer_index:02d}] "
            f"mean_delta={summary['mean_delta']:.4f} win_loss={summary['target_win_loss']}"
        )
    selected = min(layer_results, key=lambda row: (row["mean_delta"], -row["target_win_loss"], row["layer"]))
    return {"selected": selected, "layer_results": layer_results}


def evaluate(
    rows: list[dict[str, Any]],
    clean_rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    labels: list[int],
    splits: list[str],
    candidate_records: list[dict[str, Any]],
    attention_scores: list[list[list[float]]],
    attention_selection: dict[str, Any],
    attention_null: dict[str, Any],
    causal_selection: dict[str, Any],
    intervention_arms: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    discovery_rows = [row for row in clean_rows if row["split"] == "discovery"]
    holdout_rows = [row for row in clean_rows if row["split"] == "holdout"]
    selected_attention = attention_selection["selected"]
    selected_scores = attention_scores[int(selected_attention["layer"])][int(selected_attention["head"])]
    groups = subgroup_aucs(candidate_records, labels, splits, selected_scores)
    selected_causal = causal_selection["selected"]
    arm_summaries = {
        name: summarize_arm(holdout_rows, baseline, arm)
        for name, arm in intervention_arms.items()
    }

    behavior_target_wins = sum(1 for row in rows if baseline[row["id"]]["target_wins"])
    clean_target_wins = sum(1 for row in clean_rows if baseline[row["id"]]["target_wins"])
    target_layer = arm_summaries["selected_layer_target_value"]
    distractor_layer = arm_summaries["selected_layer_distractor_value"]
    random_layer = arm_summaries["selected_layer_random_value"]
    wrong_layer = arm_summaries["wrong_layer_target_value"]
    full_target = arm_summaries["full_path_target_value"]
    full_distractor = arm_summaries["full_path_distractor_value"]
    selected_head = arm_summaries["selected_head_target_value"]
    criteria = {
        "behavior_clean_rows_at_least_36": len(clean_rows) >= 36,
        "behavior_holdout_clean_rows_at_least_12": len(holdout_rows) >= 12,
        "behavior_clean_rate_at_least_0_75": (len(clean_rows) / max(1, len(rows))) >= 0.75,
        "signature_holdout_auc_at_least_0_85": selected_attention["holdout_auc"] is not None
        and selected_attention["holdout_auc"] >= 0.85,
        "signature_above_shuffle_p95_by_0_05": selected_attention["holdout_auc"] is not None
        and selected_attention["holdout_auc"] >= attention_null["auc_p95"] + 0.05,
        "signature_subgroup_aucs_at_least_0_75": all(
            group["auc"] is not None and group["auc"] >= 0.75 for group in groups.values()
        ),
        "selected_layer_target_mask_reduces_margin_by_0_25": target_layer["mean_delta"] <= -0.25,
        "selected_layer_target_mask_beats_distractor_by_0_25": target_layer["mean_delta"]
        <= distractor_layer["mean_delta"] - 0.25,
        "selected_layer_target_mask_beats_random_by_0_25": target_layer["mean_delta"]
        <= random_layer["mean_delta"] - 0.25,
        "selected_layer_target_mask_beats_wrong_layer_by_0_25": target_layer["mean_delta"]
        <= wrong_layer["mean_delta"] - 0.25,
        "selected_layer_target_mask_causes_target_win_loss": target_layer["target_win_loss"] >= 1,
        "full_path_target_mask_beats_distractor_by_0_50": full_target["mean_delta"]
        <= full_distractor["mean_delta"] - 0.50,
    }
    return {
        "model_behavior": {
            "rows": len(rows),
            "target_wins": behavior_target_wins,
            "target_win_rate": behavior_target_wins / max(1, len(rows)),
            "clean_rows": len(clean_rows),
            "clean_target_wins": clean_target_wins,
            "discovery_clean_rows": len(discovery_rows),
            "holdout_clean_rows": len(holdout_rows),
            "pair_count": rows[0]["pair_count"] if rows else None,
        },
        "selected_attention_head": selected_attention,
        "selected_attention_head_subgroups": groups,
        "attention_shuffle_selection_null": attention_null,
        "selected_causal_layer": selected_causal,
        "intervention_holdout_arms": arm_summaries,
        "criteria": criteria,
        "behavior_gate_passed": all(
            criteria[key]
            for key in (
                "behavior_clean_rows_at_least_36",
                "behavior_holdout_clean_rows_at_least_12",
                "behavior_clean_rate_at_least_0_75",
            )
        ),
        "signature_gate_passed": all(
            criteria[key]
            for key in (
                "signature_holdout_auc_at_least_0_85",
                "signature_above_shuffle_p95_by_0_05",
                "signature_subgroup_aucs_at_least_0_75",
            )
        ),
        "selected_layer_intervention_gate_passed": all(
            criteria[key]
            for key in (
                "selected_layer_target_mask_reduces_margin_by_0_25",
                "selected_layer_target_mask_beats_distractor_by_0_25",
                "selected_layer_target_mask_beats_random_by_0_25",
                "selected_layer_target_mask_beats_wrong_layer_by_0_25",
                "selected_layer_target_mask_causes_target_win_loss",
            )
        ),
        "full_path_intervention_diagnostic_passed": criteria["full_path_target_mask_beats_distractor_by_0_50"],
        "selected_head_intervention_mean_delta": selected_head["mean_delta"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_associative_lookup_source_edge")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--row-count", type=int, default=48)
    parser.add_argument("--pair-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shuffle-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

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
    rows = build_rows(tokenizer, args.row_count, args.pair_count, args.seed)
    baseline = score_rows(rows, tokenizer, model, args.batch_size)
    for row in rows:
        row["baseline"] = baseline[row["id"]]
        row["clean"] = bool(baseline[row["id"]]["target_wins"])
    clean_rows = [row for row in rows if row["clean"]]
    if len(clean_rows) < 2:
        raise ValueError("not enough clean rows for MC005")
    print(
        f"[behavior] clean={len(clean_rows)}/{len(rows)} "
        f"discovery={sum(row['split'] == 'discovery' for row in clean_rows)} "
        f"holdout={sum(row['split'] == 'holdout' for row in clean_rows)}"
    )

    labels, splits, candidate_records, attention_scores = collect_attention_scores(clean_rows, tokenizer, model)
    attention_selection = select_attention_head(labels, splits, attention_scores)
    selected_attention = attention_selection["selected"]
    print(
        "[attention-select] "
        f"layer={selected_attention['layer']} head={selected_attention['head']} "
        f"discovery_auc={selected_attention['discovery_auc']} "
        f"holdout_auc={selected_attention['holdout_auc']}"
    )
    attention_null = shuffled_selection_null(
        labels,
        splits,
        attention_scores,
        args.shuffle_iterations,
        seed=args.seed + 100,
    )
    discovery_rows = [row for row in clean_rows if row["split"] == "discovery"]
    holdout_rows = [row for row in clean_rows if row["split"] == "holdout"]
    causal_selection = select_causal_layer(discovery_rows, tokenizer, model, baseline, args.batch_size)
    selected_causal_layer = int(causal_selection["selected"]["layer"])
    wrong_layer = selected_causal_layer - 4 if selected_causal_layer >= 4 else selected_causal_layer + 4
    all_heads = list(range(model.config.num_attention_heads))
    selected_head_layer = int(selected_attention["layer"])
    selected_head_index = int(selected_attention["head"])
    intervention_arms = {
        "selected_layer_target_value": score_rows(
            holdout_rows,
            tokenizer,
            model,
            args.batch_size,
            source_key="target_value",
            layer_index=selected_causal_layer,
            head_indices=all_heads,
        ),
        "selected_layer_distractor_value": score_rows(
            holdout_rows,
            tokenizer,
            model,
            args.batch_size,
            source_key="distractor_value",
            layer_index=selected_causal_layer,
            head_indices=all_heads,
        ),
        "selected_layer_random_value": score_rows(
            holdout_rows,
            tokenizer,
            model,
            args.batch_size,
            source_key="random_value",
            layer_index=selected_causal_layer,
            head_indices=all_heads,
        ),
        "wrong_layer_target_value": score_rows(
            holdout_rows,
            tokenizer,
            model,
            args.batch_size,
            source_key="target_value",
            layer_index=wrong_layer,
            head_indices=all_heads,
        ),
        "selected_head_target_value": score_rows(
            holdout_rows,
            tokenizer,
            model,
            args.batch_size,
            source_key="target_value",
            layer_index=selected_head_layer,
            head_indices=[selected_head_index],
        ),
        "full_path_target_value": score_rows(
            holdout_rows,
            tokenizer,
            model,
            args.batch_size,
            source_key="target_value",
            head_indices=all_heads,
            all_layers=True,
        ),
        "full_path_distractor_value": score_rows(
            holdout_rows,
            tokenizer,
            model,
            args.batch_size,
            source_key="distractor_value",
            head_indices=all_heads,
            all_layers=True,
        ),
    }
    summary = evaluate(
        rows,
        clean_rows,
        baseline,
        labels,
        splits,
        candidate_records,
        attention_scores,
        attention_selection,
        attention_null,
        causal_selection,
        intervention_arms,
    )
    summary["passed"] = (
        summary["behavior_gate_passed"]
        and summary["signature_gate_passed"]
        and summary["selected_layer_intervention_gate_passed"]
    )
    elapsed = time.time() - started
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_source_edge",
        "model_id": args.model_id,
        "row_count": args.row_count,
        "pair_count": args.pair_count,
        "seed": args.seed,
        "shuffle_iterations": args.shuffle_iterations,
        "elapsed_s": elapsed,
        "summary": summary,
        "rows": rows,
        "candidate_records": candidate_records,
        "attention_selection": {
            "selected": attention_selection["selected"],
            "layer_head_results": attention_selection["layer_head_results"],
        },
        "causal_layer_selection": causal_selection,
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
