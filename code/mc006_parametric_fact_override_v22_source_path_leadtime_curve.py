#!/usr/bin/env python
"""MC006 V22 source/path lead-time curve diagnostic.

V21 closed approximate pair matching as a promotion route. This runner asks a
different, bounded question: where do source/path hidden monitors appear across
the prompt, and do any of them survive same-position output, final-output, and
candidate-score controls?
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

from mc006_parametric_fact_override_v15_parser_normalized_signature import (
    first_token_margin,
    sha256_file,
)
from mc006_parametric_fact_override_v21_pair_matched_leadtime import (
    CARD_ID,
    DEFAULT_V19_ARTIFACT,
    DEFAULT_V20_ARTIFACT,
    MODEL_ID,
    RESULT_DIR,
    RUN_TYPE as V21_RUN_TYPE,
    V19_DIAGNOSTIC_CLASS,
    V19_RUN_TYPE,
    V20_DIAGNOSTIC_CLASS,
    V20_RUN_TYPE,
    auc_payload,
    build_records,
    fit_direction_scores,
    label_array,
    matched_pairs,
    orient_scalar_scores,
    pair_accuracy,
    pair_participant_indices,
    read_json,
    score_payload,
    split_mask,
    structural_check,
    summarize_pairs,
    train_stats,
)


RUN_TYPE = "parametric_fact_override_v22_source_path_leadtime_curve"
DEFAULT_V21_ARTIFACT = (
    RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v21_pair_matched_leadtime_20260701T030507.json"
)
V21_DIAGNOSTIC_CLASS = "approximate_pair_matching_failed_margin_baselines"
MARGIN_FIELDS = ("candidate_score_margin", "final_next_token_margin")

SOURCE_PATH_POSITIONS = (
    "mapping_country_token",
    "mapping_value_token",
    "question_country_token",
)
LINE_POSITIONS = (
    "after_mapping_line",
    "after_instruction_line",
    "after_question_line",
    "after_return_line",
)
REFERENCE_POSITIONS = ("final_prompt_token",)
ALL_POSITIONS = SOURCE_PATH_POSITIONS + LINE_POSITIONS + REFERENCE_POSITIONS
SELECTABLE_POSITIONS = SOURCE_PATH_POSITIONS + LINE_POSITIONS
POSITION_ORDER = {name: index for index, name in enumerate(ALL_POSITIONS)}


def validate_sources(v19: dict[str, Any], v20: dict[str, Any], v21: dict[str, Any]) -> dict[str, Any]:
    v19_summary = v19.get("summary", {})
    v20_summary = v20.get("summary", {})
    v21_summary = v21.get("summary", {})
    criteria = {
        "v19_expected_run_type": v19.get("run_type") == V19_RUN_TYPE,
        "v19_expected_diagnostic_class": v19_summary.get("diagnostic_class") == V19_DIAGNOSTIC_CLASS,
        "v19_not_signature_ready": v19_summary.get("signature_ready") is False,
        "v20_expected_run_type": v20.get("run_type") == V20_RUN_TYPE,
        "v20_expected_diagnostic_class": v20_summary.get("diagnostic_class") == V20_DIAGNOSTIC_CLASS,
        "v20_pair_matched_ready": v20_summary.get("pair_matched_diagnostic_ready") is True,
        "v21_expected_run_type": v21.get("run_type") == V21_RUN_TYPE,
        "v21_expected_diagnostic_class": v21_summary.get("diagnostic_class") == V21_DIAGNOSTIC_CLASS,
        "v21_not_signature_ready": v21_summary.get("signature_ready") is False,
    }
    return {"criteria": criteria, "passed": all(criteria.values())}


def prompt_line_entries(prompt: str) -> list[dict[str, Any]]:
    lines = prompt.splitlines(keepends=True)
    if not lines:
        raise ValueError("empty prompt")
    entries = []
    cursor = 0
    for line in lines:
        text = line.rstrip("\r\n")
        entries.append(
            {
                "start": cursor,
                "end": cursor + len(text),
                "text": text,
                "stripped": text.strip(),
            }
        )
        cursor += len(line)
    if cursor != len(prompt):
        raise ValueError("prompt line accounting mismatch")
    nonempty = [entry for entry in entries if entry["stripped"]]
    if not nonempty:
        raise ValueError(f"prompt has no nonempty lines: {prompt!r}")
    return nonempty


def find_all_spans(prompt: str, term: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    while True:
        index = prompt.find(term, start)
        if index < 0:
            break
        spans.append((index, index + len(term)))
        start = index + max(1, len(term))
    return spans


def choose_span(prompt: str, term: str, line: dict[str, Any] | None, occurrence: str) -> tuple[int, int]:
    spans = find_all_spans(prompt, term)
    if line is not None:
        line_spans = [
            span
            for span in spans
            if span[0] >= int(line["start"]) and span[1] <= int(line["end"])
        ]
        if line_spans:
            spans = line_spans
    if not spans:
        raise ValueError(f"could not find {term!r} in prompt")
    if occurrence == "last":
        return spans[-1]
    return spans[0]


def line_boundary_offsets(prompt: str) -> dict[str, int]:
    nonempty = prompt_line_entries(prompt)
    question_entries = [entry for entry in nonempty if "?" in entry["text"]]
    answer_indices = [
        index
        for index, entry in enumerate(nonempty)
        if entry["stripped"].lower().startswith("answer")
    ]
    before_answer = nonempty[: answer_indices[0]] if answer_indices else nonempty
    mapping_entry = nonempty[0]
    instruction_entry = nonempty[1] if len(nonempty) > 1 else mapping_entry
    question_entry = question_entries[-1] if question_entries else before_answer[-1]
    return_entry = before_answer[-1] if before_answer else question_entry
    return {
        "after_mapping_line": int(mapping_entry["end"]),
        "after_instruction_line": int(instruction_entry["end"]),
        "after_question_line": int(question_entry["end"]),
        "after_return_line": int(return_entry["end"]),
        "final_prompt_token": len(prompt),
    }


def token_index_for_char_end(
    offsets: list[tuple[int, int]],
    char_end: int,
) -> int:
    candidates = [
        index
        for index, (start, end) in enumerate(offsets)
        if end <= char_end and end > start
    ]
    if not candidates:
        candidates = [
            index
            for index, (start, end) in enumerate(offsets)
            if start < char_end <= end and end > start
        ]
    if not candidates:
        raise ValueError(f"could not map char end {char_end} to token")
    return candidates[-1]


def token_positions_for_record(tokenizer: Any, row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    prompt = row["prompt"]
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
        raise ValueError("V22 requires a fast tokenizer for auditable offset mapping")

    lines = prompt_line_entries(prompt)
    mapping_line = lines[0]
    country_spans = find_all_spans(prompt, row["country"])
    if not country_spans:
        raise ValueError(f"could not find country {row['country']!r}")
    mapping_country_span = choose_span(prompt, row["country"], mapping_line, "first")
    mapping_value_span = choose_span(prompt, row["override_capital"], mapping_line, "first")
    question_country_span = country_spans[-1]

    char_targets = {
        "mapping_country_token": mapping_country_span[1],
        "mapping_value_token": mapping_value_span[1],
        "question_country_token": question_country_span[1],
        **line_boundary_offsets(prompt),
    }
    positions: dict[str, dict[str, Any]] = {}
    for name, char_end in char_targets.items():
        token_index = token_index_for_char_end(offsets, int(char_end))
        start, end = offsets[token_index]
        positions[name] = {
            "token_index": token_index,
            "token_id": input_ids[token_index],
            "prefix_token_count": token_index + 1,
            "boundary_char_offset": int(char_end),
            "token_char_span": [start, end],
            "token_text": prompt[start:end],
            "position_order": POSITION_ORDER[name],
        }
    return positions


def collect_features_and_baselines(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, list[float]]],
    list[dict[str, Any]],
    dict[str, Any],
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
    mapping_failures = []

    for index, row in enumerate(records, start=1):
        try:
            positions = token_positions_for_record(tokenizer, row)
        except Exception as exc:
            mapping_failures.append({"id": row["id"], "source_row_id": row["source_row_id"], "error": str(exc)})
            continue
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
            f"[v22 row] {index:03d}/{len(records):03d} {row['source_row_id']} "
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
    mapping_summary = {
        "attempted_rows": len(records),
        "mapped_rows": len(scored_records),
        "mapping_failures": mapping_failures,
        "position_names": list(ALL_POSITIONS),
    }
    return features_by_position, position_baselines, scored_records, mapping_summary


def candidate_record(
    position: str,
    layer_name: str,
    fit: dict[str, Any],
    scores: np.ndarray,
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
    pairs_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    layer = int(layer_name.split("_")[1])
    return {
        "name": f"{position}/{layer_name}",
        "position": position,
        "position_group": "source_path" if position in SOURCE_PATH_POSITIONS else "line_or_reference",
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


def candidate_key(record: dict[str, Any], primary_metric: str) -> tuple[float, float, int, int]:
    if primary_metric == "pair_accuracy":
        primary = record["pair_accuracy_by_split"]["non_holdout"]
        secondary = record["auc_by_split"]["non_holdout"]
    else:
        primary = record["auc_by_split"]["non_holdout"]
        secondary = record["pair_accuracy_by_split"]["non_holdout"]
    return (
        -1.0 if primary is None else float(primary),
        -1.0 if secondary is None else float(secondary),
        -int(record["position_order"]),
        -int(record["layer"]),
    )


def score_internal_candidates(
    features_by_position: dict[str, dict[str, np.ndarray]],
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
    train_indices: np.ndarray,
    pairs_by_split: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    candidates = []
    selected_by_pair: dict[str, Any] | None = None
    selected_by_auc: dict[str, Any] | None = None
    for position in ALL_POSITIONS:
        for layer_name, matrix in sorted(
            features_by_position[position].items(),
            key=lambda item: int(item[0].split("_")[1]),
        ):
            fit = fit_direction_scores(matrix, labels, train_indices, pairs_by_split["non_holdout"])
            record = candidate_record(
                position,
                layer_name,
                fit,
                fit["scores"],
                labels,
                masks,
                pairs_by_split,
            )
            candidates.append(record)
            if position in SELECTABLE_POSITIONS:
                if selected_by_pair is None or candidate_key(record, "pair_accuracy") > candidate_key(
                    selected_by_pair, "pair_accuracy"
                ):
                    selected_by_pair = record
                if selected_by_auc is None or candidate_key(record, "auc") > candidate_key(
                    selected_by_auc, "auc"
                ):
                    selected_by_auc = record
    if selected_by_pair is None or selected_by_auc is None:
        raise ValueError("no selectable source/path candidates")
    return candidates, selected_by_pair, selected_by_auc


def per_position_curve(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    curve: dict[str, Any] = {}
    for position in ALL_POSITIONS:
        position_candidates = [record for record in candidates if record["position"] == position]
        by_pair = max(position_candidates, key=lambda record: candidate_key(record, "pair_accuracy"))
        by_auc = max(position_candidates, key=lambda record: candidate_key(record, "auc"))
        curve[position] = {
            "best_by_non_holdout_pair_accuracy": by_pair,
            "best_by_non_holdout_auc": by_auc,
        }
    return curve


def scalar_baseline_payload(
    name: str,
    values: np.ndarray,
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
    pairs_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    oriented = orient_scalar_scores(values, pairs_by_split["non_holdout"])
    return {
        "orientation": oriented["orientation"],
        **score_payload(
            name,
            oriented["scores"],
            labels,
            masks,
            pairs_by_split,
        ),
    }


def build_position_baselines(
    position_baselines: dict[str, dict[str, list[float]]],
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
    pairs_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for position, baselines in position_baselines.items():
        payload[position] = {
            name: scalar_baseline_payload(
                f"{position}.{name}",
                np.asarray(values, dtype=np.float64),
                labels,
                masks,
                pairs_by_split,
            )
            for name, values in baselines.items()
        }
    return payload


def classify(criteria: dict[str, bool]) -> str:
    if not criteria["source_artifacts_valid"] or not criteria["structural_passed"]:
        return "source_artifact_invalid"
    if not criteria["position_mapping_complete"]:
        return "source_path_position_mapping_failed"
    if not criteria["holdout_pair_count_at_least_20"]:
        return "source_path_pair_audit_underpowered"
    if not criteria["curve_any_hidden_holdout_supported"]:
        return "source_path_leadtime_not_supported"
    if not criteria["curve_any_hidden_beats_same_position_output_by_0p05"]:
        return "source_path_same_position_output_confounded"
    if not criteria["curve_any_hidden_beats_candidate_and_final_margins_by_0p05"]:
        return "source_path_final_margin_shadow"
    return "source_path_leadtime_monitor_only"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--v19-artifact", type=Path, default=DEFAULT_V19_ARTIFACT)
    parser.add_argument("--v20-artifact", type=Path, default=DEFAULT_V20_ARTIFACT)
    parser.add_argument("--v21-artifact", type=Path, default=DEFAULT_V21_ARTIFACT)
    parser.add_argument(
        "--artifact-prefix",
        default="mc006_qwen3_1p7b_parametric_fact_override_v22_source_path_leadtime_curve",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--pair-threshold-z", type=float, default=0.5)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.time()
    v19_hash = sha256_file(args.v19_artifact)
    v20_hash = sha256_file(args.v20_artifact)
    v21_hash = sha256_file(args.v21_artifact)
    v19 = read_json(args.v19_artifact)
    v20 = read_json(args.v20_artifact)
    v21 = read_json(args.v21_artifact)
    source_validation = validate_sources(v19, v20, v21)

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("V22 requires a fast tokenizer for offset mapping")

    records = build_records(v19, tokenizer)
    for row in records:
        row["id"] = row["id"].replace("_v21_", "_v22_")
    structural = structural_check(records)
    stats_by_field = {field: train_stats(records, field) for field in MARGIN_FIELDS}
    pairs_by_split = {
        split: matched_pairs(records, stats_by_field, split, args.pair_threshold_z)
        for split in ("non_holdout", "holdout")
    }

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    features_by_position, position_baselines, scored_records, mapping_summary = (
        collect_features_and_baselines(records, tokenizer, model)
    )
    labels = label_array(scored_records)
    masks = {
        "all": split_mask(scored_records, "all"),
        "non_holdout": split_mask(scored_records, "non_holdout"),
        "discovery": split_mask(scored_records, "discovery"),
        "calibration": split_mask(scored_records, "calibration"),
        "holdout": split_mask(scored_records, "holdout"),
    }
    train_indices = np.where(masks["non_holdout"])[0]
    if train_indices.size == 0:
        raise ValueError("no non-holdout rows")

    candidates, selected_by_pair, selected_by_auc = score_internal_candidates(
        features_by_position,
        labels,
        masks,
        train_indices,
        pairs_by_split,
    )
    curve = per_position_curve(candidates)
    position_baseline_results = build_position_baselines(
        position_baselines,
        labels,
        masks,
        pairs_by_split,
    )
    global_baselines = {
        "candidate_score_margin": scalar_baseline_payload(
            "candidate_score_margin",
            np.asarray([row["candidate_score_margin"] for row in scored_records], dtype=np.float64),
            labels,
            masks,
            pairs_by_split,
        ),
        "v19_final_next_token_margin": scalar_baseline_payload(
            "v19_final_next_token_margin",
            np.asarray([row["final_next_token_margin"] for row in scored_records], dtype=np.float64),
            labels,
            masks,
            pairs_by_split,
        ),
        "final_prompt_next_token_output_margin": position_baseline_results["final_prompt_token"][
            "next_token_output_margin"
        ],
        "prompt_token_count": scalar_baseline_payload(
            "prompt_token_count",
            np.asarray([row["prompt_token_count"] for row in scored_records], dtype=np.float64),
            labels,
            masks,
            pairs_by_split,
        ),
    }

    selected_position = selected_by_pair["position"]
    selected_pair_holdout = float(selected_by_pair["pair_accuracy_by_split"]["holdout"] or 0.0)
    selected_pair_auc = float(selected_by_pair["auc_by_split"]["holdout"] or 0.0)
    selected_auc_position = selected_by_auc["position"]
    selected_auc_holdout_pair = float(selected_by_auc["pair_accuracy_by_split"]["holdout"] or 0.0)
    selected_auc_holdout_auc = float(selected_by_auc["auc_by_split"]["holdout"] or 0.0)
    same_position_output_pair = float(
        position_baseline_results[selected_position]["next_token_output_margin"][
            "pair_accuracy_by_split"
        ]["holdout"]
        or 0.0
    )
    selected_auc_same_position_output_pair = float(
        position_baseline_results[selected_auc_position]["next_token_output_margin"][
            "pair_accuracy_by_split"
        ]["holdout"]
        or 0.0
    )
    candidate_pair = float(
        global_baselines["candidate_score_margin"]["pair_accuracy_by_split"]["holdout"] or 0.0
    )
    v19_final_pair = float(
        global_baselines["v19_final_next_token_margin"]["pair_accuracy_by_split"]["holdout"] or 0.0
    )
    final_prompt_pair = float(
        global_baselines["final_prompt_next_token_output_margin"]["pair_accuracy_by_split"]["holdout"] or 0.0
    )
    curve_supported = []
    curve_same_position_wins = []
    curve_global_margin_wins = []
    for position, item in curve.items():
        if position not in SELECTABLE_POSITIONS:
            continue
        same_output_pair = float(
            position_baseline_results[position]["next_token_output_margin"][
                "pair_accuracy_by_split"
            ]["holdout"]
            or 0.0
        )
        for key in ("best_by_non_holdout_pair_accuracy", "best_by_non_holdout_auc"):
            record = item[key]
            holdout_pair = float(record["pair_accuracy_by_split"]["holdout"] or 0.0)
            holdout_auc = float(record["auc_by_split"]["holdout"] or 0.0)
            supported = holdout_pair >= 0.65 and holdout_auc >= 0.65
            curve_supported.append(supported)
            curve_same_position_wins.append(supported and holdout_pair >= same_output_pair + 0.05)
            curve_global_margin_wins.append(
                supported
                and holdout_pair >= candidate_pair + 0.05
                and holdout_pair >= v19_final_pair + 0.05
                and holdout_pair >= final_prompt_pair + 0.05
            )

    criteria = {
        "source_artifacts_valid": source_validation["passed"],
        "structural_passed": structural["passed"],
        "position_mapping_complete": mapping_summary["mapped_rows"] == mapping_summary["attempted_rows"],
        "holdout_pair_count_at_least_20": len(pairs_by_split["holdout"]) >= 20,
        "selected_pair_hidden_holdout_pair_accuracy_at_least_0p65": selected_pair_holdout >= 0.65,
        "selected_pair_hidden_holdout_auc_at_least_0p65": selected_pair_auc >= 0.65,
        "selected_auc_hidden_holdout_pair_accuracy_at_least_0p65": selected_auc_holdout_pair >= 0.65,
        "selected_auc_hidden_holdout_auc_at_least_0p65": selected_auc_holdout_auc >= 0.65,
        "any_selected_hidden_holdout_supported": (
            (selected_pair_holdout >= 0.65 and selected_pair_auc >= 0.65)
            or (selected_auc_holdout_pair >= 0.65 and selected_auc_holdout_auc >= 0.65)
        ),
        "selected_pair_hidden_beats_same_position_output_by_0p05": (
            selected_pair_holdout >= same_position_output_pair + 0.05
        ),
        "selected_auc_hidden_beats_same_position_output_by_0p05": (
            selected_auc_holdout_pair >= selected_auc_same_position_output_pair + 0.05
        ),
        "any_selected_hidden_beats_same_position_output_by_0p05": (
            selected_pair_holdout >= same_position_output_pair + 0.05
            or selected_auc_holdout_pair >= selected_auc_same_position_output_pair + 0.05
        ),
        "selected_pair_hidden_beats_candidate_and_final_margins_by_0p05": (
            selected_pair_holdout >= candidate_pair + 0.05
            and selected_pair_holdout >= v19_final_pair + 0.05
            and selected_pair_holdout >= final_prompt_pair + 0.05
        ),
        "selected_auc_hidden_beats_candidate_and_final_margins_by_0p05": (
            selected_auc_holdout_pair >= candidate_pair + 0.05
            and selected_auc_holdout_pair >= v19_final_pair + 0.05
            and selected_auc_holdout_pair >= final_prompt_pair + 0.05
        ),
        "any_selected_hidden_beats_candidate_and_final_margins_by_0p05": (
            (
                selected_pair_holdout >= candidate_pair + 0.05
                and selected_pair_holdout >= v19_final_pair + 0.05
                and selected_pair_holdout >= final_prompt_pair + 0.05
            )
            or (
                selected_auc_holdout_pair >= candidate_pair + 0.05
                and selected_auc_holdout_pair >= v19_final_pair + 0.05
                and selected_auc_holdout_pair >= final_prompt_pair + 0.05
            )
        ),
        "curve_any_hidden_holdout_supported": any(curve_supported),
        "curve_any_hidden_beats_same_position_output_by_0p05": any(curve_same_position_wins),
        "curve_any_hidden_beats_candidate_and_final_margins_by_0p05": any(curve_global_margin_wins),
    }
    diagnostic_class = classify(criteria)

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
                "candidate_score_margin": float(row["candidate_score_margin"]),
                "v19_final_next_token_margin": float(row["final_next_token_margin"]),
                "position_metadata": row["position_metadata"],
            }
        )

    summary = {
        "model_id": args.model_id,
        "source_artifacts": {
            "v19": str(args.v19_artifact),
            "v20": str(args.v20_artifact),
            "v21": str(args.v21_artifact),
        },
        "source_artifact_sha256": {
            "v19": v19_hash,
            "v20": v20_hash,
            "v21": v21_hash,
        },
        "primary_row_contract": (
            "V19 pooled binary true/override rows; V22 maps source/path token "
            "positions after V21 killed approximate pair matching as promotion."
        ),
        "pair_threshold_z": args.pair_threshold_z,
        "source_validation": source_validation,
        "structural": structural,
        "mapping_summary": mapping_summary,
        "pair_summaries": {
            split: summarize_pairs(pairs, scored_records)
            for split, pairs in pairs_by_split.items()
        },
        "selected_by_non_holdout_pair_accuracy": selected_by_pair,
        "selected_by_non_holdout_auc": selected_by_auc,
        "per_position_curve": curve,
        "position_baseline_results": position_baseline_results,
        "global_baseline_results": global_baselines,
        "criteria": criteria,
        "diagnostic_supported": diagnostic_class
        in {"source_path_final_margin_shadow", "source_path_leadtime_monitor_only"},
        "signature_ready": False,
        "intervention_ready": False,
        "passed": False,
        "diagnostic_class": diagnostic_class,
        "claim_boundary": (
            "V22 is a source/path lead-time map, not a mechanism claim. It cannot "
            "promote MC006 while final candidate/output margins remain dominant "
            "and no intervention is tested."
        ),
    }
    result = {
        "card_id": args.card_id,
        "run_type": RUN_TYPE,
        "model_id": args.model_id,
        "elapsed_s": time.time() - started,
        "summary": summary,
        "candidate_results": candidates,
        "row_scores": row_scores,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)

    compact = {
        "diagnostic_class": diagnostic_class,
        "diagnostic_supported": summary["diagnostic_supported"],
        "signature_ready": summary["signature_ready"],
        "intervention_ready": summary["intervention_ready"],
        "mapping_summary": mapping_summary,
        "pair_summaries": {
            split: {
                key: value
                for key, value in payload.items()
                if key != "example_pairs"
            }
            for split, payload in summary["pair_summaries"].items()
        },
        "selected_by_non_holdout_pair_accuracy": selected_by_pair,
        "selected_by_non_holdout_auc": selected_by_auc,
        "selected_same_position_output_holdout_pair_accuracy": same_position_output_pair,
        "selected_auc_same_position_output_holdout_pair_accuracy": selected_auc_same_position_output_pair,
        "candidate_holdout_pair_accuracy": candidate_pair,
        "v19_final_holdout_pair_accuracy": v19_final_pair,
        "final_prompt_holdout_pair_accuracy": final_prompt_pair,
        "criteria": criteria,
        "output_path": str(output_path),
    }
    print(json.dumps(compact, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
