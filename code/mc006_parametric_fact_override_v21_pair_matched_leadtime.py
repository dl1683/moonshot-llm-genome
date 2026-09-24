#!/usr/bin/env python
"""MC006 V21 approximate pair-matched lead-time diagnostic for the V19 table.

V20 found no strict final-margin overlap, but did find pooled 0.5z joint
candidate/final-margin pairs. This runner asks the narrower diagnostic question:
inside those approximate matched pairs, does a pre-output hidden direction lead
candidate/final/output-visible baselines, or is the apparent signature still a
typed output-margin failure?
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc003_delayed_copy_signature import auc_score
from mc006_parametric_fact_override_v15_parser_normalized_signature import (
    first_token_margin,
    sha256_file,
)


CARD_ID = "MC006"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/MC006")
DEFAULT_V19_ARTIFACT = (
    RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v19_overlapping_margin_table_20260701T023921.json"
)
DEFAULT_V20_ARTIFACT = (
    RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v20_strict_overlap_selection_audit_20260701T025043.json"
)
RUN_TYPE = "parametric_fact_override_v21_pair_matched_leadtime"
V19_RUN_TYPE = "parametric_fact_override_v19_overlapping_margin_table"
V19_DIAGNOSTIC_CLASS = "non_holdout_candidate_margin_overlap_failed"
V20_RUN_TYPE = "parametric_fact_override_v20_strict_overlap_selection_audit"
V20_DIAGNOSTIC_CLASS = "strict_final_margin_overlap_absent"
PRIMARY_LABELS = ("true_answer", "override_answer")
MARGIN_FIELDS = ("candidate_score_margin", "final_next_token_margin")

SELECTABLE_POSITIONS = (
    "after_mapping_line",
    "after_instruction_line",
    "after_question_line",
    "after_return_line",
)
REFERENCE_POSITIONS = ("final_prompt_token",)
ALL_POSITIONS = SELECTABLE_POSITIONS + REFERENCE_POSITIONS
POSITION_ORDER = {name: index for index, name in enumerate(ALL_POSITIONS)}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_sources(v19: dict[str, Any], v20: dict[str, Any]) -> dict[str, Any]:
    v19_summary = v19.get("summary", {})
    v20_summary = v20.get("summary", {})
    criteria = {
        "v19_expected_run_type": v19.get("run_type") == V19_RUN_TYPE,
        "v19_expected_diagnostic_class": v19_summary.get("diagnostic_class") == V19_DIAGNOSTIC_CLASS,
        "v19_not_signature_ready": v19_summary.get("signature_ready") is False,
        "v19_has_400_records": isinstance(v19.get("records"), list) and len(v19.get("records", [])) == 400,
        "v20_expected_run_type": v20.get("run_type") == V20_RUN_TYPE,
        "v20_expected_diagnostic_class": v20_summary.get("diagnostic_class") == V20_DIAGNOSTIC_CLASS,
        "v20_pair_matched_ready": v20_summary.get("pair_matched_diagnostic_ready") is True,
        "v20_not_signature_ready": v20_summary.get("signature_ready") is False,
    }
    return {"criteria": criteria, "passed": all(criteria.values())}


def is_binary(row: dict[str, Any]) -> bool:
    return bool(row.get("is_binary")) and row.get("selected_label") in PRIMARY_LABELS


def build_records(v19: dict[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    records = []
    for row in v19["records"]:
        if not is_binary(row):
            continue
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        if not prompt_ids:
            raise ValueError(f"tokenless prompt for {row['id']}")
        label = 1 if row["selected_label"] == "true_answer" else 0
        records.append(
            {
                "id": row["id"].replace("_v19_", "_v21_"),
                "source_row_id": row["id"],
                "source_id": row["source_id"],
                "split": row["split"],
                "template": row["template"],
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
                "prompt_token_count": len(prompt_ids),
                "final_prompt_token_id": int(prompt_ids[-1]),
                "generated_text": row.get("generated_text"),
                "generated_token_ids": row.get("generated_token_ids", []),
                "candidate_score_margin": float(row["candidate_score_margin"]),
                "final_next_token_margin": float(row["final_next_token_margin"]),
            }
        )
    records.sort(key=lambda item: (item["split"], item["template"], item["source_id"]))
    return records


def structural_check(records: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(row["label_name"] for row in records)
    split_label_counts: dict[str, dict[str, int]] = {}
    for split in ("discovery", "calibration", "holdout"):
        split_rows = [row for row in records if row["split"] == split]
        split_label_counts[split] = dict(sorted(Counter(row["label_name"] for row in split_rows).items()))
    duplicate_ids = [row_id for row_id, count in Counter(row["id"] for row in records).items() if count > 1]
    source_ids = {row["source_id"] for row in records}
    source_row_ids = {row["source_row_id"] for row in records}
    holdout_labels = split_label_counts.get("holdout", {})
    criteria = {
        "binary_rows_at_least_200": len(records) >= 200,
        "non_holdout_true_at_least_20": label_counts.get("true_answer", 0)
        - holdout_labels.get("true_answer", 0)
        >= 20,
        "non_holdout_override_at_least_20": label_counts.get("override_answer", 0)
        - holdout_labels.get("override_answer", 0)
        >= 20,
        "holdout_true_at_least_10": holdout_labels.get("true_answer", 0) >= 10,
        "holdout_override_at_least_10": holdout_labels.get("override_answer", 0) >= 10,
        "source_row_ids_unique": len(source_row_ids) == len(records),
        "no_duplicate_record_ids": not duplicate_ids,
    }
    return {
        "criteria": criteria,
        "passed": all(criteria.values()),
        "row_count": len(records),
        "label_counts": dict(sorted(label_counts.items())),
        "split_label_counts": split_label_counts,
        "source_count": len(source_ids),
        "source_row_count": len(source_row_ids),
        "duplicate_ids": duplicate_ids,
    }


def line_boundary_offsets(prompt: str) -> dict[str, int]:
    lines = prompt.splitlines(keepends=True)
    if not lines:
        raise ValueError("empty prompt")
    entries = []
    cursor = 0
    for line in lines:
        line_without_break = line.rstrip("\r\n")
        entries.append(
            {
                "start": cursor,
                "end": cursor + len(line_without_break),
                "text": line_without_break,
                "stripped": line_without_break.strip(),
            }
        )
        cursor += len(line)
    if cursor != len(prompt):
        raise ValueError("prompt line accounting mismatch")
    nonempty = [entry for entry in entries if entry["stripped"]]
    if not nonempty:
        raise ValueError(f"prompt has no nonempty lines: {prompt!r}")
    question_entries = [entry for entry in nonempty if "?" in entry["text"]]
    answer_indices = [
        index
        for index, entry in enumerate(nonempty)
        if entry["stripped"].lower().startswith("answer")
    ]
    if answer_indices:
        answer_index = answer_indices[0]
        before_answer = nonempty[:answer_index]
    else:
        before_answer = nonempty

    mapping_entry = nonempty[0]
    instruction_entry = nonempty[1] if len(nonempty) > 1 else mapping_entry
    question_entry = question_entries[-1] if question_entries else instruction_entry
    return_entry = before_answer[-1] if before_answer else question_entry
    offsets: dict[str, int] = {
        "after_mapping_line": int(mapping_entry["end"]),
        "after_instruction_line": int(instruction_entry["end"]),
        "after_question_line": int(question_entry["end"]),
        "after_return_line": int(return_entry["end"]),
    }
    offsets["final_prompt_token"] = len(prompt)
    return offsets


def token_positions_for_prompt(tokenizer: Any, prompt: str) -> dict[str, dict[str, Any]]:
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = [int(token_id) for token_id in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    if not input_ids:
        raise ValueError("tokenless prompt")
    if len(input_ids) != len(offsets):
        raise ValueError("token/offset length mismatch")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("V21 requires a fast tokenizer for auditable offset mapping")

    positions: dict[str, dict[str, Any]] = {}
    for name, char_offset in line_boundary_offsets(prompt).items():
        candidates = [
            index
            for index, (start, end) in enumerate(offsets)
            if end <= char_offset and end > start
        ]
        if not candidates:
            candidates = [
                index
                for index, (start, end) in enumerate(offsets)
                if start < char_offset <= end and end > start
            ]
        if not candidates:
            raise ValueError(f"could not map {name} at char offset {char_offset}")
        token_index = candidates[-1]
        start, end = offsets[token_index]
        positions[name] = {
            "token_index": token_index,
            "token_id": input_ids[token_index],
            "prefix_token_count": token_index + 1,
            "boundary_char_offset": char_offset,
            "token_char_span": [start, end],
            "token_text": prompt[start:end],
        }
    return positions


def train_stats(records: list[dict[str, Any]], field: str) -> tuple[float, float]:
    values = np.asarray(
        [float(row[field]) for row in records if row["split"] != "holdout"],
        dtype=np.float64,
    )
    if values.size == 0:
        return 0.0, 1.0
    std = float(values.std())
    if std < 1e-9:
        std = 1.0
    return float(values.mean()), std


def z_value(row: dict[str, Any], field: str, stats_by_field: dict[str, tuple[float, float]]) -> float:
    mean, std = stats_by_field[field]
    return (float(row[field]) - mean) / std


def split_mask(records: list[dict[str, Any]], split: str) -> np.ndarray:
    if split == "all":
        return np.ones(len(records), dtype=bool)
    if split == "non_holdout":
        return np.array([row["split"] != "holdout" for row in records], dtype=bool)
    return np.array([row["split"] == split for row in records], dtype=bool)


def label_array(records: list[dict[str, Any]]) -> np.ndarray:
    return np.array([int(row["binary_label"]) for row in records], dtype=np.int64)


def matched_pairs(
    records: list[dict[str, Any]],
    stats_by_field: dict[str, tuple[float, float]],
    split: str,
    threshold_z: float,
) -> list[dict[str, Any]]:
    indexed = [(index, row) for index, row in enumerate(records) if split_mask(records, split)[index]]
    true_rows = [(index, row) for index, row in indexed if row["selected_label"] == "true_answer"]
    override_rows = [(index, row) for index, row in indexed if row["selected_label"] == "override_answer"]
    pairs = []
    for true_index, true_row in true_rows:
        for override_index, override_row in override_rows:
            deltas = {
                field: abs(
                    z_value(true_row, field, stats_by_field)
                    - z_value(override_row, field, stats_by_field)
                )
                for field in MARGIN_FIELDS
            }
            max_delta = max(deltas.values())
            if max_delta <= threshold_z:
                pairs.append(
                    {
                        "true_index": true_index,
                        "override_index": override_index,
                        "true_id": true_row["id"],
                        "override_id": override_row["id"],
                        "max_abs_z_delta": float(max_delta),
                        "candidate_abs_z_delta": float(deltas["candidate_score_margin"]),
                        "final_abs_z_delta": float(deltas["final_next_token_margin"]),
                    }
                )
    return pairs


def pair_participant_indices(pairs: list[dict[str, Any]]) -> np.ndarray:
    indices = sorted(
        {
            int(pair["true_index"])
            for pair in pairs
        }
        | {
            int(pair["override_index"])
            for pair in pairs
        }
    )
    return np.asarray(indices, dtype=np.int64)


def auc_payload(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float | None:
    if int(mask.sum()) == 0:
        return None
    return auc_score(
        [float(score) for score in scores[mask]],
        [int(label) for label in labels[mask]],
    )


def pair_accuracy(scores: np.ndarray, pairs: list[dict[str, Any]]) -> float | None:
    if not pairs:
        return None
    wins = 0.0
    for pair in pairs:
        true_score = float(scores[int(pair["true_index"])])
        override_score = float(scores[int(pair["override_index"])])
        if true_score > override_score:
            wins += 1.0
        elif true_score == override_score:
            wins += 0.5
    return float(wins / len(pairs))


def fit_direction_scores(
    matrix: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    orientation_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    if train_indices.size == 0:
        raise ValueError("direction fit needs train indices")
    x_train = matrix[train_indices]
    y_train = labels[train_indices]
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    z_all = (matrix - mean) / std
    pos = z_all[train_indices[y_train == 1]]
    neg = z_all[train_indices[y_train == 0]]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("direction fit requires both labels")
    direction = pos.mean(axis=0) - neg.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        scores = np.zeros(matrix.shape[0], dtype=np.float64)
    else:
        direction = direction / norm
        scores = (z_all @ direction).astype(np.float64)
    orientation = 1.0
    train_pair_accuracy = pair_accuracy(scores, orientation_pairs)
    if train_pair_accuracy is not None and train_pair_accuracy < 0.5:
        scores = -scores
        orientation = -1.0
        train_pair_accuracy = pair_accuracy(scores, orientation_pairs)
    return {
        "scores": scores,
        "orientation": orientation,
        "direction_norm": norm,
        "train_pair_accuracy": train_pair_accuracy,
    }


def orient_scalar_scores(
    values: np.ndarray,
    orientation_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    scores = values.astype(np.float64).copy()
    orientation = 1.0
    train_pair_accuracy = pair_accuracy(scores, orientation_pairs)
    if train_pair_accuracy is not None and train_pair_accuracy < 0.5:
        scores = -scores
        orientation = -1.0
        train_pair_accuracy = pair_accuracy(scores, orientation_pairs)
    return {
        "scores": scores,
        "orientation": orientation,
        "train_pair_accuracy": train_pair_accuracy,
    }


def score_payload(
    name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
    pairs_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "name": name,
        "auc_by_split": {
            split_name: auc_payload(scores, labels, mask)
            for split_name, mask in masks.items()
        },
        "pair_accuracy_by_split": {
            split_name: pair_accuracy(scores, pairs)
            for split_name, pairs in pairs_by_split.items()
        },
    }


def collect_features_and_baselines(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, list[float]]],
    list[dict[str, Any]],
]:
    hidden_accumulator: dict[str, list[list[torch.Tensor]]] | None = None
    position_baselines = {
        position: {
            "next_token_output_margin": [],
            "prefix_token_count": [],
            "position_token_id": [],
        }
        for position in ALL_POSITIONS
    }
    scored_records: list[dict[str, Any]] = []

    for index, row in enumerate(records, start=1):
        positions = token_positions_for_prompt(tokenizer, row["prompt"])
        inputs = tokenizer(
            row["prompt"],
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        logits = out.logits[0].detach().float().cpu()
        hidden_states = [hidden[0].detach().float().cpu() for hidden in out.hidden_states[1:]]
        if hidden_accumulator is None:
            hidden_accumulator = {
                position: [[] for _ in hidden_states]
                for position in ALL_POSITIONS
            }
        for position, meta in positions.items():
            token_index = int(meta["token_index"])
            for layer_index, hidden in enumerate(hidden_states):
                hidden_accumulator[position][layer_index].append(hidden[token_index])
            position_baselines[position]["next_token_output_margin"].append(
                first_token_margin(row, tokenizer, logits[token_index])
            )
            position_baselines[position]["prefix_token_count"].append(
                float(meta["prefix_token_count"])
            )
            position_baselines[position]["position_token_id"].append(float(meta["token_id"]))
        scored_records.append({**row, "position_metadata": positions})
        print(
            f"[v21 row] {index:03d}/{len(records):03d} {row['source_row_id']} "
            f"label={row['selected_label']} template={row['template']}"
        )

    if hidden_accumulator is None:
        raise ValueError("no hidden features collected")
    features_by_position = {
        position: {
            f"layer_{layer_index}": torch.stack(layer_values, dim=0).numpy().astype(np.float32)
            for layer_index, layer_values in enumerate(layer_lists)
        }
        for position, layer_lists in hidden_accumulator.items()
    }
    return features_by_position, position_baselines, scored_records


def score_internal_candidates(
    features_by_position: dict[str, dict[str, np.ndarray]],
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
    train_indices: np.ndarray,
    train_pairs: list[dict[str, Any]],
    holdout_pairs: list[dict[str, Any]],
    positions: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    pairs_by_split = {"non_holdout": train_pairs, "holdout": holdout_pairs}
    candidates = []
    selected: dict[str, Any] | None = None
    selected_scores: np.ndarray | None = None
    for position in positions:
        for layer_name, matrix in sorted(
            features_by_position[position].items(),
            key=lambda item: int(item[0].split("_")[1]),
        ):
            layer = int(layer_name.split("_")[1])
            fit = fit_direction_scores(matrix, labels, train_indices, train_pairs)
            scores = fit["scores"]
            payload = {
                "name": f"{position}/{layer_name}",
                "position": position,
                "position_order": POSITION_ORDER[position],
                "layer": layer,
                "orientation": fit["orientation"],
                "direction_norm": fit["direction_norm"],
                **score_payload(
                    f"{position}/{layer_name}",
                    scores,
                    labels,
                    masks,
                    pairs_by_split,
                ),
            }
            candidates.append(payload)
            selection_key = (
                -1.0
                if payload["pair_accuracy_by_split"]["non_holdout"] is None
                else float(payload["pair_accuracy_by_split"]["non_holdout"]),
                -1.0
                if payload["auc_by_split"]["non_holdout"] is None
                else float(payload["auc_by_split"]["non_holdout"]),
                -int(payload["position_order"]),
                -int(payload["layer"]),
            )
            if selected is None:
                selected = payload
                selected_scores = scores
                continue
            selected_key = (
                -1.0
                if selected["pair_accuracy_by_split"]["non_holdout"] is None
                else float(selected["pair_accuracy_by_split"]["non_holdout"]),
                -1.0
                if selected["auc_by_split"]["non_holdout"] is None
                else float(selected["auc_by_split"]["non_holdout"]),
                -int(selected["position_order"]),
                -int(selected["layer"]),
            )
            if selection_key > selected_key:
                selected = payload
                selected_scores = scores
    if selected is None or selected_scores is None:
        raise ValueError("no internal candidate selected")
    return candidates, selected, selected_scores


def summarize_pairs(
    pairs: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    participant_indices = pair_participant_indices(pairs)
    participant_labels = Counter(records[index]["selected_label"] for index in participant_indices)
    participant_templates = Counter(records[index]["template"] for index in participant_indices)
    return {
        "pairs": len(pairs),
        "participants": int(participant_indices.size),
        "participant_label_counts": dict(sorted(participant_labels.items())),
        "participant_template_counts": dict(sorted(participant_templates.items())),
        "mean_max_abs_z_delta": float(np.mean([pair["max_abs_z_delta"] for pair in pairs])) if pairs else None,
        "example_pairs": pairs[:10],
    }


def classify(criteria: dict[str, bool]) -> str:
    if not criteria["source_artifacts_valid"] or not criteria["structural_passed"]:
        return "source_artifact_invalid"
    if not criteria["strict_final_margin_overlap_absent"]:
        return "unexpected_source_boundary"
    if not criteria["non_holdout_pair_count_at_least_20"] or not criteria["holdout_pair_count_at_least_20"]:
        return "pair_matched_leadtime_underpowered"
    if not criteria["candidate_and_final_margin_holdout_pair_accuracy_at_most_0p65"]:
        return "approximate_pair_matching_failed_margin_baselines"
    if not criteria["selected_hidden_holdout_pair_accuracy_at_least_0p65"]:
        return "pair_matched_hidden_not_supported"
    if not criteria["selected_hidden_beats_same_position_output_by_0p05"]:
        return "pair_matched_hidden_same_position_output_confounded"
    if not criteria["selected_hidden_beats_candidate_and_final_margins_by_0p05"]:
        return "pair_matched_hidden_output_margin_confounded"
    if not criteria["selected_hidden_beats_prompt_shape_by_0p05"]:
        return "pair_matched_hidden_prompt_shape_confounded"
    return "pair_matched_leadtime_supported_bounded"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--v19-artifact", type=Path, default=DEFAULT_V19_ARTIFACT)
    parser.add_argument("--v20-artifact", type=Path, default=DEFAULT_V20_ARTIFACT)
    parser.add_argument(
        "--artifact-prefix",
        default="mc006_qwen3_1p7b_parametric_fact_override_v21_pair_matched_leadtime",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--pair-threshold-z", type=float, default=0.5)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.time()
    v19_hash = sha256_file(args.v19_artifact)
    v20_hash = sha256_file(args.v20_artifact)
    v19 = read_json(args.v19_artifact)
    v20 = read_json(args.v20_artifact)
    source_validation = validate_sources(v19, v20)

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("V21 requires a fast tokenizer for offset mapping")

    records = build_records(v19, tokenizer)
    structural = structural_check(records)
    stats_by_field = {field: train_stats(records, field) for field in MARGIN_FIELDS}
    pairs_by_split = {
        split: matched_pairs(records, stats_by_field, split, args.pair_threshold_z)
        for split in ("non_holdout", "holdout")
    }
    train_indices = pair_participant_indices(pairs_by_split["non_holdout"])
    if train_indices.size == 0:
        train_indices = np.where(split_mask(records, "non_holdout"))[0]

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    features_by_position, position_baselines, scored_records = collect_features_and_baselines(
        records,
        tokenizer,
        model,
    )
    labels = label_array(scored_records)
    masks = {
        "all": split_mask(scored_records, "all"),
        "non_holdout": split_mask(scored_records, "non_holdout"),
        "discovery": split_mask(scored_records, "discovery"),
        "calibration": split_mask(scored_records, "calibration"),
        "holdout": split_mask(scored_records, "holdout"),
    }

    pre_output_candidates, selected_pre_output, selected_scores = score_internal_candidates(
        features_by_position,
        labels,
        masks,
        train_indices,
        pairs_by_split["non_holdout"],
        pairs_by_split["holdout"],
        SELECTABLE_POSITIONS,
    )
    final_reference_candidates, selected_final_reference, final_reference_scores = score_internal_candidates(
        features_by_position,
        labels,
        masks,
        train_indices,
        pairs_by_split["non_holdout"],
        pairs_by_split["holdout"],
        REFERENCE_POSITIONS,
    )

    selected_position = selected_pre_output["position"]
    scalar_sources = {
        "candidate_score_margin": np.asarray(
            [row["candidate_score_margin"] for row in scored_records],
            dtype=np.float64,
        ),
        "v19_final_next_token_margin": np.asarray(
            [row["final_next_token_margin"] for row in scored_records],
            dtype=np.float64,
        ),
        "same_position_next_token_output_margin": np.asarray(
            position_baselines[selected_position]["next_token_output_margin"],
            dtype=np.float64,
        ),
        "final_prompt_next_token_output_margin": np.asarray(
            position_baselines["final_prompt_token"]["next_token_output_margin"],
            dtype=np.float64,
        ),
        "prompt_token_count": np.asarray(
            [row["prompt_token_count"] for row in scored_records],
            dtype=np.float64,
        ),
        "same_position_prefix_token_count": np.asarray(
            position_baselines[selected_position]["prefix_token_count"],
            dtype=np.float64,
        ),
        "same_position_token_id": np.asarray(
            position_baselines[selected_position]["position_token_id"],
            dtype=np.float64,
        ),
    }
    scalar_baselines = {}
    scalar_scores = {}
    for name, values in scalar_sources.items():
        oriented = orient_scalar_scores(values, pairs_by_split["non_holdout"])
        scalar_scores[name] = oriented["scores"]
        scalar_baselines[name] = {
            "orientation": oriented["orientation"],
            **score_payload(
                name,
                oriented["scores"],
                labels,
                masks,
                pairs_by_split,
            ),
        }

    selected_holdout_pair_accuracy = float(
        selected_pre_output["pair_accuracy_by_split"]["holdout"] or 0.0
    )
    candidate_holdout_pair_accuracy = float(
        scalar_baselines["candidate_score_margin"]["pair_accuracy_by_split"]["holdout"] or 0.0
    )
    v19_final_holdout_pair_accuracy = float(
        scalar_baselines["v19_final_next_token_margin"]["pair_accuracy_by_split"]["holdout"] or 0.0
    )
    same_position_output_holdout_pair_accuracy = float(
        scalar_baselines["same_position_next_token_output_margin"]["pair_accuracy_by_split"]["holdout"] or 0.0
    )
    prompt_token_count_holdout_pair_accuracy = float(
        scalar_baselines["prompt_token_count"]["pair_accuracy_by_split"]["holdout"] or 0.0
    )
    same_position_prefix_count_holdout_pair_accuracy = float(
        scalar_baselines["same_position_prefix_token_count"]["pair_accuracy_by_split"]["holdout"] or 0.0
    )

    criteria = {
        "source_artifacts_valid": source_validation["passed"],
        "structural_passed": structural["passed"],
        "strict_final_margin_overlap_absent": v20["summary"]["diagnostic_class"] == V20_DIAGNOSTIC_CLASS,
        "pair_matched_diagnostic_ready": v20["summary"].get("pair_matched_diagnostic_ready") is True,
        "non_holdout_pair_count_at_least_20": len(pairs_by_split["non_holdout"]) >= 20,
        "holdout_pair_count_at_least_20": len(pairs_by_split["holdout"]) >= 20,
        "candidate_and_final_margin_holdout_pair_accuracy_at_most_0p65": (
            candidate_holdout_pair_accuracy <= 0.65
            and v19_final_holdout_pair_accuracy <= 0.65
        ),
        "selected_hidden_holdout_pair_accuracy_at_least_0p65": selected_holdout_pair_accuracy >= 0.65,
        "selected_hidden_beats_same_position_output_by_0p05": (
            selected_holdout_pair_accuracy >= same_position_output_holdout_pair_accuracy + 0.05
        ),
        "selected_hidden_beats_candidate_and_final_margins_by_0p05": (
            selected_holdout_pair_accuracy >= candidate_holdout_pair_accuracy + 0.05
            and selected_holdout_pair_accuracy >= v19_final_holdout_pair_accuracy + 0.05
        ),
        "selected_hidden_beats_prompt_shape_by_0p05": (
            selected_holdout_pair_accuracy >= prompt_token_count_holdout_pair_accuracy + 0.05
            and selected_holdout_pair_accuracy >= same_position_prefix_count_holdout_pair_accuracy + 0.05
        ),
    }
    diagnostic_class = classify(criteria)
    diagnostic_supported = diagnostic_class == "pair_matched_leadtime_supported_bounded"

    pair_summaries = {
        split: summarize_pairs(pairs, scored_records)
        for split, pairs in pairs_by_split.items()
    }
    row_scores = []
    for index, row in enumerate(scored_records):
        row_scores.append(
            {
                "id": row["id"],
                "source_row_id": row["source_row_id"],
                "source_id": row["source_id"],
                "split": row["split"],
                "template": row["template"],
                "label_name": row["label_name"],
                "binary_label": int(labels[index]),
                "selected_hidden_score": float(selected_scores[index]),
                "selected_final_reference_hidden_score": float(final_reference_scores[index]),
                "candidate_score_margin": float(row["candidate_score_margin"]),
                "v19_final_next_token_margin": float(row["final_next_token_margin"]),
                "same_position_next_token_output_margin": float(
                    scalar_sources["same_position_next_token_output_margin"][index]
                ),
                "final_prompt_next_token_output_margin": float(
                    scalar_sources["final_prompt_next_token_output_margin"][index]
                ),
                "prompt_token_count": int(row["prompt_token_count"]),
            }
        )

    summary = {
        "model_id": args.model_id,
        "source_artifacts": {
            "v19": str(args.v19_artifact),
            "v20": str(args.v20_artifact),
        },
        "source_artifact_sha256": {
            "v19": v19_hash,
            "v20": v20_hash,
        },
        "primary_row_contract": (
            "V19 pooled binary true/override rows; V20 strict final-margin overlap is absent, "
            "so this is an approximate pair-matched lead-time diagnostic only."
        ),
        "pair_threshold_z": args.pair_threshold_z,
        "source_validation": source_validation,
        "structural": structural,
        "stats_by_margin_field": {
            field: {"mean": mean, "std": std}
            for field, (mean, std) in stats_by_field.items()
        },
        "pair_summaries": pair_summaries,
        "train_pair_participant_count": int(train_indices.size),
        "selected_pre_output_candidate": selected_pre_output,
        "selected_final_reference_candidate": selected_final_reference,
        "scalar_baseline_results": scalar_baselines,
        "criteria": criteria,
        "diagnostic_supported": diagnostic_supported,
        "signature_ready": False,
        "intervention_ready": False,
        "passed": False,
        "diagnostic_class": diagnostic_class,
        "claim_boundary": (
            "Even if supported, V21 cannot promote MC006: it lacks strict final-margin overlap "
            "and performs no intervention. It can only classify whether approximate pair-matched "
            "pre-output lead-time is promising, underpowered, or still explained by baselines."
        ),
    }
    result = {
        "card_id": args.card_id,
        "run_type": RUN_TYPE,
        "model_id": args.model_id,
        "elapsed_s": time.time() - started,
        "summary": summary,
        "pre_output_candidate_results": pre_output_candidates,
        "final_reference_candidate_results": final_reference_candidates,
        "row_scores": row_scores,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)

    compact = {
        "diagnostic_class": diagnostic_class,
        "diagnostic_supported": diagnostic_supported,
        "signature_ready": summary["signature_ready"],
        "intervention_ready": summary["intervention_ready"],
        "pair_summaries": pair_summaries,
        "selected_pre_output_candidate": selected_pre_output,
        "selected_final_reference_candidate": selected_final_reference,
        "selected_holdout_pair_accuracy": selected_holdout_pair_accuracy,
        "candidate_holdout_pair_accuracy": candidate_holdout_pair_accuracy,
        "v19_final_holdout_pair_accuracy": v19_final_holdout_pair_accuracy,
        "same_position_output_holdout_pair_accuracy": same_position_output_holdout_pair_accuracy,
        "criteria": criteria,
        "output_path": str(output_path),
    }
    print(json.dumps(compact, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
