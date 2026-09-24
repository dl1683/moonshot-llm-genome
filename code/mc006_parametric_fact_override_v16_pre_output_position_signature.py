#!/usr/bin/env python
"""MC006 V16 pre-output hidden-signature diagnostic for the V14 table."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc003_delayed_copy_signature import percentile
from mc006_parametric_fact_override_v15_parser_normalized_signature import (
    first_token_margin,
    fit_direction,
    labels_for,
    metric_payload,
    read_json,
    sha256_file,
    split_mask,
    structural_check,
    token_logprob,
    validate_source,
)


CARD_ID = "MC006"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/MC006")
DEFAULT_SOURCE_ARTIFACT = (
    RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v14_parser_normalized_20260701T000538.json"
)
RUN_TYPE = "parametric_fact_override_v16_pre_output_position_signature"
PRIMARY_LABELS = ("true_answer", "override_answer")

SELECTABLE_POSITIONS = (
    "after_mapping_line",
    "after_instruction_line",
    "after_question_line",
    "after_return_line",
)
REFERENCE_POSITIONS = ("final_prompt_token",)
ALL_POSITIONS = SELECTABLE_POSITIONS + REFERENCE_POSITIONS
POSITION_ORDER = {name: index for index, name in enumerate(ALL_POSITIONS)}


def build_records(source_result: dict[str, Any], tokenizer: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary_rows = []
    side_rows = []
    selected_template = source_result["summary"]["selection"]["selected_template"]
    for row in source_result["summary"]["selected_template_rows"]:
        if row["selected_label"] not in PRIMARY_LABELS:
            side_rows.append(
                {
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "split": row["split"],
                    "selected_label": row["selected_label"],
                    "selected_answer": row["selected_answer"],
                    "first_line": row["first_line"],
                }
            )
            continue
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        label = 1 if row["selected_label"] == "true_answer" else 0
        primary_rows.append(
            {
                "id": row["id"].replace("_v13_", "_v16_"),
                "source_row_id": row["id"],
                "source_id": row["source_id"],
                "split": row["split"],
                "template": selected_template,
                "condition": row["condition"],
                "country": row["country"],
                "true_capital": row["true_capital"],
                "override_capital": row["override_capital"],
                "lure_capital": row["lure_capital"],
                "selected_label": row["selected_label"],
                "selected_answer": row["selected_answer"],
                "generated_text": row["generated_text"],
                "generated_token_ids": row["generated_token_ids"],
                "generated_first_token_id": int(row["generated_token_ids"][0]) if row["generated_token_ids"] else -1,
                "v13_selected_label": row.get("v13_selected_label"),
                "label_changed_by_normalization": bool(row.get("label_changed_by_normalization")),
                "binary_label": label,
                "label_name": row["selected_label"],
                "prompt": row["prompt"],
                "prompt_token_count": len(prompt_ids),
                "final_prompt_token_id": int(prompt_ids[-1]),
            }
        )
    primary_rows.sort(key=lambda item: (item["split"], item["source_id"]))
    side_rows.sort(key=lambda item: (item["split"], item["source_id"]))
    return primary_rows, side_rows


def line_boundary_offsets(prompt: str) -> dict[str, int]:
    lines = prompt.splitlines(keepends=True)
    if len(lines) != 5:
        raise ValueError(f"expected 5 prompt lines, found {len(lines)}: {prompt!r}")
    names = list(ALL_POSITIONS)
    offsets: dict[str, int] = {}
    cursor = 0
    for name, line in zip(names, lines):
        line_without_break = line.rstrip("\r\n")
        offsets[name] = cursor + len(line_without_break)
        cursor += len(line)
    if cursor != len(prompt):
        raise ValueError("prompt line accounting mismatch")
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
        raise ValueError("V16 requires a fast tokenizer for auditable offset mapping")

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


def orient_scalar_baseline(
    values: list[float],
    labels: np.ndarray,
    discovery: np.ndarray,
    holdout: np.ndarray,
    score_name: str,
) -> dict[str, Any]:
    matrix = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    fit = fit_direction(matrix[discovery], labels[discovery], matrix[holdout], labels[holdout])
    return {**metric_payload(fit), "score": score_name}


def score_internal_candidates(
    features_by_position: dict[str, dict[str, np.ndarray]],
    labels: np.ndarray,
    discovery: np.ndarray,
    holdout: np.ndarray,
    positions: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates_payload = []
    selected: dict[str, Any] | None = None
    for position in positions:
        layer_items = sorted(
            features_by_position[position].items(),
            key=lambda item: int(item[0].split("_")[1]),
        )
        for name, matrix in layer_items:
            layer = int(name.split("_")[1])
            fit = fit_direction(
                matrix[discovery],
                labels[discovery],
                matrix[holdout],
                labels[holdout],
            )
            record = {
                "name": f"{position}/{name}",
                "position": position,
                "position_order": POSITION_ORDER[position],
                "layer": layer,
                **metric_payload(fit),
            }
            candidates_payload.append(record)
            if selected is None or (
                -1.0 if record["discovery_auc"] is None else float(record["discovery_auc"]),
                -1.0 if record["holdout_auc"] is None else float(record["holdout_auc"]),
                -int(record["position_order"]),
                -int(record["layer"]),
            ) > (
                -1.0 if selected["discovery_auc"] is None else float(selected["discovery_auc"]),
                -1.0 if selected["holdout_auc"] is None else float(selected["holdout_auc"]),
                -int(selected["position_order"]),
                -int(selected["layer"]),
            ):
                selected = record
    assert selected is not None
    return candidates_payload, selected


def collect_features_and_baselines(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, list[float]]],
    dict[str, list[float]],
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
    global_baselines = {
        "candidate_score_margin": [],
        "prompt_length": [],
        "final_token_id": [],
        "true_candidate_token_count": [],
        "override_candidate_token_count": [],
        "generated_first_token_id": [],
        "label_changed_by_normalization": [],
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

        true_score = token_logprob(model, tokenizer, row["prompt"], row["true_capital"])
        override_score = token_logprob(model, tokenizer, row["prompt"], row["override_capital"])
        candidate_margin = float(true_score["mean_logprob"] - override_score["mean_logprob"])
        global_baselines["candidate_score_margin"].append(candidate_margin)
        global_baselines["prompt_length"].append(float(row["prompt_token_count"]))
        global_baselines["final_token_id"].append(float(row["final_prompt_token_id"]))
        global_baselines["true_candidate_token_count"].append(float(true_score["token_count"]))
        global_baselines["override_candidate_token_count"].append(float(override_score["token_count"]))
        global_baselines["generated_first_token_id"].append(float(row["generated_first_token_id"]))
        global_baselines["label_changed_by_normalization"].append(
            1.0 if row["label_changed_by_normalization"] else 0.0
        )

        scored_records.append(
            {
                **row,
                "position_metadata": positions,
                "true_candidate_score": true_score,
                "override_candidate_score": override_score,
                "true_minus_override_mean_logprob": candidate_margin,
            }
        )
        print(
            f"[v16 row] {index:03d}/{len(records):03d} {row['id']} "
            f"candidate_true_minus_override={candidate_margin:+.4f}"
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
    return features_by_position, position_baselines, global_baselines, scored_records


def shuffled_selection_null(
    features_by_position: dict[str, dict[str, np.ndarray]],
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
    feature_items = []
    for position in SELECTABLE_POSITIONS:
        for name, matrix in sorted(
            features_by_position[position].items(),
            key=lambda item: int(item[0].split("_")[1]),
        ):
            layer = int(name.split("_")[1])
            feature_items.append((position, POSITION_ORDER[position], layer, name, matrix))
    for _ in range(iterations):
        shuffled = y_train.copy()
        rng.shuffle(shuffled)
        selected: dict[str, Any] | None = None
        for position, position_order, layer, name, matrix in feature_items:
            fit = fit_direction(matrix[discovery], shuffled, matrix[holdout], y_holdout)
            record = {
                "name": f"{position}/{name}",
                "position": position,
                "position_order": position_order,
                "layer": layer,
                **metric_payload(fit),
            }
            if selected is None or (
                -1.0 if record["discovery_auc"] is None else float(record["discovery_auc"]),
                -int(record["position_order"]),
                -int(record["layer"]),
            ) > (
                -1.0 if selected["discovery_auc"] is None else float(selected["discovery_auc"]),
                -int(selected["position_order"]),
                -int(selected["layer"]),
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


def classify(leadtime_criteria: dict[str, bool], mechanism_criteria: dict[str, bool]) -> str:
    if not leadtime_criteria["source_artifact_and_structural_passed"]:
        return "source_artifact_invalid"
    if not leadtime_criteria["holdout_class_balance_at_least_2_each"]:
        return "holdout_balance_failed"
    if not leadtime_criteria["selected_pre_output_hidden_holdout_auc_at_least_0p85"]:
        return "internal_signal_failed"
    if not leadtime_criteria["selected_hidden_beats_same_position_output_by_0p02"]:
        return "same_position_output_confounded"
    if not leadtime_criteria["selected_hidden_beats_prefix_token_count_by_0p02"]:
        return "prefix_length_confounded"
    if not leadtime_criteria["selected_hidden_beats_position_token_id_by_0p02"]:
        return "position_token_confounded"
    if not leadtime_criteria["selected_hidden_beats_shuffle_p95_by_0p05"]:
        return "shuffle_null_confounded"
    if not all(mechanism_criteria.values()):
        return "leadtime_signal_supported_but_output_global_confounded"
    return "pre_output_hidden_signature_supported"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--source-artifact", type=Path, default=DEFAULT_SOURCE_ARTIFACT)
    parser.add_argument(
        "--artifact-prefix",
        default="mc006_qwen3_1p7b_parametric_fact_override_v16_pre_output_position_signature",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--shuffle-iterations", type=int, default=100)
    parser.add_argument("--shuffle-seed", type=int, default=26016)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.time()
    source_hash = sha256_file(args.source_artifact)
    source_result = read_json(args.source_artifact)
    validate_source(source_result)

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("V16 requires a fast tokenizer for offset mapping")

    records, side_rows = build_records(source_result, tokenizer)
    structural = structural_check(records, side_rows)
    if not structural["passed"]:
        raise ValueError(f"structural check failed before model scoring: {structural}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    features_by_position, position_baselines, global_baselines, scored_records = (
        collect_features_and_baselines(records, tokenizer, model)
    )
    labels = labels_for(scored_records)
    discovery = np.array([row["split"] != "holdout" for row in scored_records], dtype=bool)
    holdout = split_mask(scored_records, "holdout")

    pre_output_candidates, selected_pre_output = score_internal_candidates(
        features_by_position,
        labels,
        discovery,
        holdout,
        SELECTABLE_POSITIONS,
    )
    final_reference_candidates, selected_final_reference = score_internal_candidates(
        features_by_position,
        labels,
        discovery,
        holdout,
        REFERENCE_POSITIONS,
    )

    position_baseline_payload: dict[str, dict[str, Any]] = {}
    for position, baselines in position_baselines.items():
        position_baseline_payload[position] = {
            name: orient_scalar_baseline(values, labels, discovery, holdout, f"{position}.{name}")
            for name, values in baselines.items()
        }
    global_baseline_payload = {
        name: orient_scalar_baseline(values, labels, discovery, holdout, name)
        for name, values in global_baselines.items()
    }
    global_baseline_payload["final_next_token_output_margin"] = position_baseline_payload[
        "final_prompt_token"
    ]["next_token_output_margin"]

    shuffle_null = shuffled_selection_null(
        features_by_position,
        labels,
        discovery,
        holdout,
        args.shuffle_iterations,
        args.shuffle_seed,
    )

    holdout_counts = structural["split_label_counts"]["holdout"]
    selected_position = selected_pre_output["position"]
    selected_auc = float(selected_pre_output["holdout_auc"] or 0.0)
    same_position_output_auc = float(
        position_baseline_payload[selected_position]["next_token_output_margin"]["holdout_auc"] or 0.0
    )
    prefix_count_auc = float(
        position_baseline_payload[selected_position]["prefix_token_count"]["holdout_auc"] or 0.0
    )
    position_token_auc = float(
        position_baseline_payload[selected_position]["position_token_id"]["holdout_auc"] or 0.0
    )
    candidate_auc = float(global_baseline_payload["candidate_score_margin"]["holdout_auc"] or 0.0)
    final_output_auc = float(
        global_baseline_payload["final_next_token_output_margin"]["holdout_auc"] or 0.0
    )
    shuffle_p95 = float(shuffle_null["holdout_auc_p95"])

    leadtime_criteria = {
        "source_artifact_and_structural_passed": structural["passed"],
        "holdout_class_balance_at_least_2_each": holdout_counts.get("true_answer", 0) >= 2
        and holdout_counts.get("override_answer", 0) >= 2,
        "selected_pre_output_hidden_holdout_auc_at_least_0p85": selected_auc >= 0.85,
        "selected_hidden_beats_same_position_output_by_0p02": selected_auc >= same_position_output_auc + 0.02,
        "selected_hidden_beats_prefix_token_count_by_0p02": selected_auc >= prefix_count_auc + 0.02,
        "selected_hidden_beats_position_token_id_by_0p02": selected_auc >= position_token_auc + 0.02,
        "selected_hidden_beats_shuffle_p95_by_0p05": selected_auc >= shuffle_p95 + 0.05,
    }
    mechanism_criteria = {
        **leadtime_criteria,
        "selected_hidden_beats_candidate_score_by_0p02": selected_auc >= candidate_auc + 0.02,
        "selected_hidden_beats_final_next_token_output_by_0p02": selected_auc >= final_output_auc + 0.02,
    }
    diagnostic_class = classify(leadtime_criteria, mechanism_criteria)

    summary = {
        "model_id": args.model_id,
        "source_artifact": str(args.source_artifact),
        "source_artifact_sha256": source_hash,
        "primary_row_contract": "V14 selected-template true/override rows",
        "excluded_side_rows": side_rows,
        "label_definition": {"1": "true_answer", "0": "override_answer"},
        "structural": structural,
        "selectable_positions": list(SELECTABLE_POSITIONS),
        "reference_positions": list(REFERENCE_POSITIONS),
        "selected_pre_output_candidate": selected_pre_output,
        "selected_final_reference_candidate": selected_final_reference,
        "position_baseline_results": position_baseline_payload,
        "global_baseline_results": global_baseline_payload,
        "shuffle_selection_null": shuffle_null,
        "leadtime_criteria": leadtime_criteria,
        "mechanism_criteria": mechanism_criteria,
        "leadtime_supported": all(leadtime_criteria.values()),
        "passed": diagnostic_class == "pre_output_hidden_signature_supported",
        "diagnostic_class": diagnostic_class,
    }
    result = {
        "card_id": args.card_id,
        "run_type": RUN_TYPE,
        "model_id": args.model_id,
        "elapsed_s": time.time() - started,
        "summary": summary,
        "pre_output_candidate_results": pre_output_candidates,
        "final_reference_candidate_results": final_reference_candidates,
        "records": [{key: value for key, value in row.items() if key != "prompt"} for row in scored_records],
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
