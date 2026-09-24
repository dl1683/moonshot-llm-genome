#!/usr/bin/env python
"""MC006 V18 margin-matched lead-time audit for the V14/V16 table."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc003_delayed_copy_signature import auc_score
from mc006_parametric_fact_override_v15_parser_normalized_signature import (
    fit_direction,
    labels_for,
    orient_auc,
    split_mask,
)
from mc006_parametric_fact_override_v16_pre_output_position_signature import (
    CARD_ID,
    DEFAULT_SOURCE_ARTIFACT,
    MODEL_ID,
    RESULT_DIR,
    REFERENCE_POSITIONS,
    SELECTABLE_POSITIONS,
    build_records,
    collect_features_and_baselines,
    score_internal_candidates,
    sha256_file,
    structural_check,
    validate_source,
)


RUN_TYPE = "parametric_fact_override_v18_margin_matched_leadtime"
MATCH_THRESHOLDS_Z = (0.25, 0.5, 1.0, 2.0)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_scores(n: int, train_mask: np.ndarray, holdout_mask: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    scores = np.full(n, np.nan, dtype=np.float64)
    scores[train_mask] = np.asarray(fit["train_scores"], dtype=np.float64)
    scores[holdout_mask] = np.asarray(fit["holdout_scores"], dtype=np.float64)
    return scores


def auc_payload(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float | None:
    if int(mask.sum()) == 0:
        return None
    return auc_score(
        [float(score) for score in scores[mask]],
        [int(label) for label in labels[mask]],
    )


def scalar_fit_scores(
    values: list[float],
    labels: np.ndarray,
    train_mask: np.ndarray,
    holdout_mask: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    matrix = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    fit = fit_direction(matrix[train_mask], labels[train_mask], matrix[holdout_mask], labels[holdout_mask])
    return fit, merge_scores(len(values), train_mask, holdout_mask, fit)


def feature_fit_scores(
    matrix: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    holdout_mask: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    fit = fit_direction(matrix[train_mask], labels[train_mask], matrix[holdout_mask], labels[holdout_mask])
    return fit, merge_scores(matrix.shape[0], train_mask, holdout_mask, fit)


def row_split_masks(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(records), dtype=bool),
        "non_holdout": np.array([row["split"] != "holdout" for row in records], dtype=bool),
        "holdout": split_mask(records, "holdout"),
        "discovery": split_mask(records, "discovery"),
        "calibration": split_mask(records, "calibration"),
    }


def class_range(values: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label_value, label_name in ((1, "true_answer"), (0, "override_answer")):
        selected = values[mask & (labels == label_value)]
        if selected.size:
            result[label_name] = {
                "count": int(selected.size),
                "min": float(np.min(selected)),
                "max": float(np.max(selected)),
                "mean": float(np.mean(selected)),
            }
        else:
            result[label_name] = {"count": 0, "min": None, "max": None, "mean": None}
    true_range = result["true_answer"]
    override_range = result["override_answer"]
    if true_range["count"] and override_range["count"]:
        overlap_low = max(float(true_range["min"]), float(override_range["min"]))
        overlap_high = min(float(true_range["max"]), float(override_range["max"]))
        result["overlap_exists"] = overlap_high >= overlap_low
        result["overlap_width"] = float(overlap_high - overlap_low) if overlap_high >= overlap_low else 0.0
        result["separation_gap"] = float(overlap_low - overlap_high) if overlap_high < overlap_low else 0.0
    else:
        result["overlap_exists"] = False
        result["overlap_width"] = 0.0
        result["separation_gap"] = None
    return result


def standardized(values: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    train_values = values[train_mask]
    mean = float(np.mean(train_values))
    std = float(np.std(train_values))
    if std < 1e-9:
        std = 1.0
    return (values - mean) / std


def matched_pair_summary(
    confound_z: np.ndarray,
    hidden_scores: np.ndarray,
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split_name, mask in masks.items():
        true_indices = np.where(mask & (labels == 1))[0]
        override_indices = np.where(mask & (labels == 0))[0]
        split_payload: dict[str, Any] = {
            "true_rows": int(true_indices.size),
            "override_rows": int(override_indices.size),
            "possible_pairs": int(true_indices.size * override_indices.size),
        }
        for threshold in MATCH_THRESHOLDS_Z:
            pairs = []
            correct = 0
            for true_index in true_indices:
                for override_index in override_indices:
                    delta = abs(float(confound_z[true_index] - confound_z[override_index]))
                    if delta <= threshold:
                        pairs.append(delta)
                        if float(hidden_scores[true_index]) > float(hidden_scores[override_index]):
                            correct += 1
            key = f"z_le_{str(threshold).replace('.', 'p')}"
            split_payload[key] = {
                "pairs": len(pairs),
                "hidden_pair_accuracy": float(correct / len(pairs)) if pairs else None,
                "mean_abs_z_delta": float(np.mean(pairs)) if pairs else None,
            }
        result[split_name] = split_payload
    return result


def residualize(
    target_scores: np.ndarray,
    confounds: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    holdout_mask: np.ndarray,
) -> dict[str, Any]:
    x_train = confounds[train_mask]
    x_holdout = confounds[holdout_mask]
    design_train = np.concatenate([np.ones((x_train.shape[0], 1)), x_train], axis=1)
    design_holdout = np.concatenate([np.ones((x_holdout.shape[0], 1)), x_holdout], axis=1)
    beta, *_ = np.linalg.lstsq(design_train, target_scores[train_mask], rcond=None)
    train_residual = target_scores[train_mask] - design_train @ beta
    holdout_residual = target_scores[holdout_mask] - design_holdout @ beta
    oriented = orient_auc(
        train_residual.astype(np.float32),
        labels[train_mask],
        holdout_residual.astype(np.float32),
        labels[holdout_mask],
    )
    all_residual = np.full_like(target_scores, np.nan, dtype=np.float64)
    all_residual[train_mask] = np.asarray(oriented["train_scores"], dtype=np.float64)
    all_residual[holdout_mask] = np.asarray(oriented["holdout_scores"], dtype=np.float64)
    return {
        "discovery_auc": oriented["discovery_auc"],
        "holdout_auc": oriented["holdout_auc"],
        "orientation": oriented["orientation"],
        "coefficients": [float(value) for value in beta],
        "scores": all_residual,
    }


def score_payload(name: str, scores: np.ndarray, labels: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "name": name,
        "auc_by_split": {
            split_name: auc_payload(scores, labels, mask)
            for split_name, mask in masks.items()
        },
    }


def margin_payload(
    name: str,
    values: np.ndarray,
    hidden_scores: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    confound_z = standardized(values, train_mask)
    return {
        "name": name,
        "range_by_split": {
            split_name: class_range(values, labels, mask)
            for split_name, mask in masks.items()
        },
        "z_range_by_split": {
            split_name: class_range(confound_z, labels, mask)
            for split_name, mask in masks.items()
        },
        "matched_pairs_by_split": matched_pair_summary(confound_z, hidden_scores, labels, masks),
    }


def classify(criteria: dict[str, bool]) -> str:
    if not criteria["source_artifact_and_structural_passed"]:
        return "source_artifact_invalid"
    if not criteria["selected_pre_output_reproduced"]:
        return "leadtime_reproduction_failed"
    if not criteria["candidate_margin_holdout_overlap_exists"] and not criteria["final_output_margin_holdout_overlap_exists"]:
        return "global_margin_separation_blocks_matching"
    if not criteria["residualized_hidden_beats_0p85_after_both_global_margins"]:
        return "margin_residualized_leadtime_erased"
    if not criteria["matched_holdout_pairs_exist_at_0p5z"]:
        return "margin_matching_underpowered"
    return "leadtime_survives_margin_controls"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--source-artifact", type=Path, default=DEFAULT_SOURCE_ARTIFACT)
    parser.add_argument(
        "--artifact-prefix",
        default="mc006_qwen3_1p7b_parametric_fact_override_v18_margin_matched_leadtime",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
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
        raise ValueError("V18 requires a fast tokenizer for offset mapping")

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
    train_mask = np.array([row["split"] != "holdout" for row in scored_records], dtype=bool)
    holdout_mask = split_mask(scored_records, "holdout")
    masks = row_split_masks(scored_records)

    pre_output_candidates, selected_pre_output = score_internal_candidates(
        features_by_position,
        labels,
        train_mask,
        holdout_mask,
        SELECTABLE_POSITIONS,
    )
    final_reference_candidates, selected_final_reference = score_internal_candidates(
        features_by_position,
        labels,
        train_mask,
        holdout_mask,
        REFERENCE_POSITIONS,
    )

    selected_matrix = features_by_position[selected_pre_output["position"]][
        f"layer_{selected_pre_output['layer']}"
    ]
    selected_fit, selected_scores = feature_fit_scores(selected_matrix, labels, train_mask, holdout_mask)

    final_matrix = features_by_position[selected_final_reference["position"]][
        f"layer_{selected_final_reference['layer']}"
    ]
    final_hidden_fit, final_hidden_scores = feature_fit_scores(final_matrix, labels, train_mask, holdout_mask)

    scalar_scores: dict[str, np.ndarray] = {}
    scalar_payload: dict[str, dict[str, Any]] = {}
    scalar_sources = {
        "same_position_next_token_output_margin": position_baselines[selected_pre_output["position"]][
            "next_token_output_margin"
        ],
        "final_next_token_output_margin": position_baselines["final_prompt_token"]["next_token_output_margin"],
        "candidate_score_margin": global_baselines["candidate_score_margin"],
        "prompt_length": global_baselines["prompt_length"],
        "generated_first_token_id": global_baselines["generated_first_token_id"],
    }
    for name, values in scalar_sources.items():
        fit, scores = scalar_fit_scores(values, labels, train_mask, holdout_mask)
        scalar_scores[name] = scores
        scalar_payload[name] = {
            "discovery_auc": fit["discovery_auc"],
            "holdout_auc": fit["holdout_auc"],
            "orientation": fit["orientation"],
            "direction_norm": fit.get("direction_norm"),
        }

    candidate_values = np.asarray(global_baselines["candidate_score_margin"], dtype=np.float64)
    final_output_values = np.asarray(
        position_baselines["final_prompt_token"]["next_token_output_margin"],
        dtype=np.float64,
    )
    same_position_values = np.asarray(
        position_baselines[selected_pre_output["position"]]["next_token_output_margin"],
        dtype=np.float64,
    )

    residuals = {
        "hidden_residual_after_candidate_margin": residualize(
            selected_scores,
            standardized(candidate_values, train_mask).reshape(-1, 1),
            labels,
            train_mask,
            holdout_mask,
        ),
        "hidden_residual_after_final_output_margin": residualize(
            selected_scores,
            standardized(final_output_values, train_mask).reshape(-1, 1),
            labels,
            train_mask,
            holdout_mask,
        ),
        "hidden_residual_after_candidate_and_final_output": residualize(
            selected_scores,
            np.stack(
                [
                    standardized(candidate_values, train_mask),
                    standardized(final_output_values, train_mask),
                ],
                axis=1,
            ),
            labels,
            train_mask,
            holdout_mask,
        ),
    }

    margin_audits = {
        "candidate_score_margin": margin_payload(
            "candidate_score_margin",
            candidate_values,
            selected_scores,
            labels,
            train_mask,
            masks,
        ),
        "final_next_token_output_margin": margin_payload(
            "final_next_token_output_margin",
            final_output_values,
            selected_scores,
            labels,
            train_mask,
            masks,
        ),
        "same_position_next_token_output_margin": margin_payload(
            "same_position_next_token_output_margin",
            same_position_values,
            selected_scores,
            labels,
            train_mask,
            masks,
        ),
    }

    row_scores = []
    residual_score_payload = {
        name: payload["scores"]
        for name, payload in residuals.items()
    }
    for index, row in enumerate(scored_records):
        row_scores.append(
            {
                "id": row["id"],
                "source_id": row["source_id"],
                "split": row["split"],
                "label_name": row["label_name"],
                "binary_label": int(labels[index]),
                "selected_hidden_score": float(selected_scores[index]),
                "selected_final_hidden_score": float(final_hidden_scores[index]),
                "candidate_score_margin_raw": float(candidate_values[index]),
                "final_next_token_output_margin_raw": float(final_output_values[index]),
                "same_position_next_token_output_margin_raw": float(same_position_values[index]),
                "candidate_score_margin_oriented_score": float(scalar_scores["candidate_score_margin"][index]),
                "final_next_token_output_margin_oriented_score": float(
                    scalar_scores["final_next_token_output_margin"][index]
                ),
                "same_position_next_token_output_margin_oriented_score": float(
                    scalar_scores["same_position_next_token_output_margin"][index]
                ),
                "hidden_residual_after_candidate_margin": float(
                    residual_score_payload["hidden_residual_after_candidate_margin"][index]
                ),
                "hidden_residual_after_final_output_margin": float(
                    residual_score_payload["hidden_residual_after_final_output_margin"][index]
                ),
                "hidden_residual_after_candidate_and_final_output": float(
                    residual_score_payload["hidden_residual_after_candidate_and_final_output"][index]
                ),
            }
        )

    residual_summary = {
        name: {
            "discovery_auc": payload["discovery_auc"],
            "holdout_auc": payload["holdout_auc"],
            "orientation": payload["orientation"],
            "coefficients": payload["coefficients"],
        }
        for name, payload in residuals.items()
    }

    candidate_holdout_overlap = bool(
        margin_audits["candidate_score_margin"]["z_range_by_split"]["holdout"]["overlap_exists"]
    )
    final_holdout_overlap = bool(
        margin_audits["final_next_token_output_margin"]["z_range_by_split"]["holdout"]["overlap_exists"]
    )
    candidate_pairs_0p5 = int(
        margin_audits["candidate_score_margin"]["matched_pairs_by_split"]["holdout"]["z_le_0p5"]["pairs"]
    )
    final_pairs_0p5 = int(
        margin_audits["final_next_token_output_margin"]["matched_pairs_by_split"]["holdout"]["z_le_0p5"]["pairs"]
    )
    residual_both_auc = residual_summary[
        "hidden_residual_after_candidate_and_final_output"
    ]["holdout_auc"]

    criteria = {
        "source_artifact_and_structural_passed": structural["passed"],
        "selected_pre_output_reproduced": selected_pre_output["position"] == "after_mapping_line"
        and int(selected_pre_output["layer"]) == 4
        and float(selected_pre_output["holdout_auc"] or 0.0) >= 1.0,
        "candidate_margin_holdout_overlap_exists": candidate_holdout_overlap,
        "final_output_margin_holdout_overlap_exists": final_holdout_overlap,
        "matched_holdout_pairs_exist_at_0p5z": candidate_pairs_0p5 > 0 and final_pairs_0p5 > 0,
        "residualized_hidden_beats_0p85_after_candidate_margin": float(
            residual_summary["hidden_residual_after_candidate_margin"]["holdout_auc"] or 0.0
        )
        >= 0.85,
        "residualized_hidden_beats_0p85_after_final_output_margin": float(
            residual_summary["hidden_residual_after_final_output_margin"]["holdout_auc"] or 0.0
        )
        >= 0.85,
        "residualized_hidden_beats_0p85_after_both_global_margins": float(residual_both_auc or 0.0) >= 0.85,
    }
    diagnostic_class = classify(criteria)

    summary = {
        "model_id": args.model_id,
        "source_artifact": str(args.source_artifact),
        "source_artifact_sha256": source_hash,
        "primary_row_contract": "V14 selected-template true/override rows; V18 re-scores V16 lead-time surface with row-level margin controls.",
        "structural": structural,
        "selected_pre_output_candidate": selected_pre_output,
        "selected_final_reference_candidate": selected_final_reference,
        "selected_pre_output_refit": {
            "discovery_auc": selected_fit["discovery_auc"],
            "holdout_auc": selected_fit["holdout_auc"],
            "orientation": selected_fit["orientation"],
            "direction_norm": selected_fit.get("direction_norm"),
        },
        "selected_final_reference_refit": {
            "discovery_auc": final_hidden_fit["discovery_auc"],
            "holdout_auc": final_hidden_fit["holdout_auc"],
            "orientation": final_hidden_fit["orientation"],
            "direction_norm": final_hidden_fit.get("direction_norm"),
        },
        "scalar_baseline_results": scalar_payload,
        "residualized_hidden_results": residual_summary,
        "margin_audits": margin_audits,
        "criteria": criteria,
        "passed": diagnostic_class == "leadtime_survives_margin_controls",
        "signature_ready": diagnostic_class == "leadtime_survives_margin_controls",
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
        "row_scores": row_scores,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)

    print(
        json.dumps(
            {
                "diagnostic_class": diagnostic_class,
                "passed": summary["passed"],
                "signature_ready": summary["signature_ready"],
                "criteria": criteria,
                "selected_pre_output_candidate": selected_pre_output,
                "scalar_baseline_results": scalar_payload,
                "residualized_hidden_results": residual_summary,
                "candidate_holdout_margin_audit": margin_audits["candidate_score_margin"][
                    "z_range_by_split"
                ]["holdout"],
                "final_output_holdout_margin_audit": margin_audits["final_next_token_output_margin"][
                    "z_range_by_split"
                ]["holdout"],
                "output_path": str(output_path),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
