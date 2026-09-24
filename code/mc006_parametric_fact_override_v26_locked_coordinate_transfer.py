#!/usr/bin/env python
"""MC006 V26 locked-coordinate transfer audit.

V25 found a candidate-score-decoupled delayed-city template, but the selected
hidden monitor failed split-preserving shuffled-label selected-search nulls.
V26 removes the free layer/position search: it locks the V25 coordinate and asks
whether a direction fit on the selected candidate-decoupled template transfers
to another candidate-decoupled delayed-city template.
"""

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
from mc006_parametric_fact_override_v15_parser_normalized_signature import sha256_file
from mc006_parametric_fact_override_v24_delayed_city_interface import (
    MODEL_ID,
    RESULT_DIR,
    collect_features_and_position_margins,
)
from mc006_parametric_fact_override_v25_candidate_decoupled_template import (
    DEFAULT_V24_ARTIFACT,
    PRIMARY_LABELS,
    auc_score,
    label_int,
    read_json,
    template_candidate,
    validate_v24,
)


CARD_ID = "MC006"
RUN_TYPE = "parametric_fact_override_v26_locked_coordinate_transfer"
V25_RUN_TYPE = "parametric_fact_override_v25_candidate_decoupled_template"
V25_DIAGNOSTIC_CLASS = "candidate_decoupled_hidden_shuffle_overfit"
DEFAULT_V25_ARTIFACT = (
    RESULT_DIR
    / "mc006_qwen3_1p7b_parametric_fact_override_v25_candidate_decoupled_template_20260701T041843.json"
)
CONTROL_FIELDS = (
    "final_next_token_city_margin",
    "city_candidate_score_margin",
    "json_candidate_score_margin",
)


def validate_v25(v25: dict[str, Any]) -> dict[str, Any]:
    summary = v25.get("summary", {})
    hidden = v25.get("hidden_screen", {})
    selected = hidden.get("selected_hidden") or {}
    criteria = {
        "expected_run_type": v25.get("run_type") == V25_RUN_TYPE,
        "expected_diagnostic_class": summary.get("diagnostic_class") == V25_DIAGNOSTIC_CLASS,
        "v25_not_signature_ready": summary.get("signature_ready") is False,
        "v25_intervention_not_ready": summary.get("intervention_ready") is False,
        "v25_candidate_decoupled": summary.get("criteria", {}).get("candidate_decoupled_template_found") is True,
        "v25_hidden_available": bool(selected.get("position") and selected.get("layer") is not None),
    }
    return {"criteria": criteria, "passed": all(criteria.values())}


def label_counts(labels: np.ndarray) -> dict[str, int]:
    names = ["override_answer" if int(label) == 0 else "true_answer" for label in labels]
    return dict(sorted(Counter(names).items()))


def split_masks(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "discovery": np.asarray([row["split"] == "discovery" for row in rows], dtype=bool),
        "calibration": np.asarray([row["split"] == "calibration" for row in rows], dtype=bool),
        "holdout": np.asarray([row["split"] == "holdout" for row in rows], dtype=bool),
        "non_holdout": np.asarray([row["split"] != "holdout" for row in rows], dtype=bool),
        "all": np.ones(len(rows), dtype=bool),
    }


def fit_locked_direction(x_train: np.ndarray, y_train: np.ndarray) -> dict[str, Any]:
    if len(set(int(label) for label in y_train.tolist())) < 2:
        raise ValueError("locked direction requires both labels in train set")
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    z_train = (x_train - mean) / std
    pos = z_train[y_train == 1]
    neg = z_train[y_train == 0]
    direction = pos.mean(axis=0) - neg.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        direction = np.zeros(x_train.shape[1], dtype=np.float32)
        train_scores = np.zeros(len(z_train), dtype=np.float32)
    else:
        direction = direction / norm
        train_scores = z_train @ direction
    train_auc = auc_score([float(score) for score in train_scores], [int(label) for label in y_train])
    orientation = 1.0
    if train_auc is not None and train_auc < 0.5:
        orientation = -1.0
        train_scores = -train_scores
        train_auc = auc_score([float(score) for score in train_scores], [int(label) for label in y_train])
    return {
        "mean": mean,
        "std": std,
        "direction": direction,
        "orientation": orientation,
        "direction_norm": norm,
        "train_auc": float(train_auc) if train_auc is not None else None,
    }


def score_with_fit(fit: dict[str, Any], x_eval: np.ndarray, y_eval: np.ndarray) -> dict[str, Any]:
    if len(x_eval) == 0:
        return {"row_count": 0, "label_counts": {}, "auc": None, "score_range": None}
    z_eval = (x_eval - fit["mean"]) / fit["std"]
    if float(fit["direction_norm"]) < 1e-12:
        scores = np.zeros(len(z_eval), dtype=np.float32)
    else:
        scores = z_eval @ fit["direction"]
    scores = scores * float(fit["orientation"])
    auc = auc_score([float(score) for score in scores], [int(label) for label in y_eval])
    return {
        "row_count": int(len(x_eval)),
        "label_counts": label_counts(y_eval),
        "auc": float(auc) if auc is not None else None,
        "score_range": {
            "min": float(np.min(scores)) if len(scores) else None,
            "max": float(np.max(scores)) if len(scores) else None,
            "mean": float(np.mean(scores)) if len(scores) else None,
        },
    }


def fit_and_score(
    source_train_values: np.ndarray,
    source_train_labels: np.ndarray,
    eval_values: np.ndarray,
    eval_labels: np.ndarray,
) -> dict[str, Any]:
    fit = fit_locked_direction(source_train_values, source_train_labels)
    scored = score_with_fit(fit, eval_values, eval_labels)
    return {
        "train_auc": fit["train_auc"],
        "direction_norm": fit["direction_norm"],
        "orientation": fit["orientation"],
        **scored,
    }


def collect_locked_template(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    locked_position: str,
    locked_layer: int,
) -> dict[str, Any]:
    features, position_margins, scored_rows, mapping = collect_features_and_position_margins(rows, tokenizer, model)
    layer_key = f"layer_{locked_layer}"
    matrix = features[locked_position][layer_key]
    labels = np.asarray([label_int(row) for row in scored_rows], dtype=np.int64)
    scalar_fields = {
        field: np.asarray([float(row[field]) for row in scored_rows], dtype=np.float32).reshape(-1, 1)
        for field in CONTROL_FIELDS
    }
    locked_position_margin_name = f"{locked_position}_next_token_city_margin"
    scalar_fields[locked_position_margin_name] = np.asarray(
        position_margins[locked_position]["next_token_city_margin"],
        dtype=np.float32,
    ).reshape(-1, 1)
    return {
        "rows": scored_rows,
        "mapping": mapping,
        "features": matrix,
        "labels": labels,
        "masks": split_masks(scored_rows),
        "scalar_fields": scalar_fields,
    }


def shuffled_train_label_null(
    source_train_values: np.ndarray,
    source_train_labels: np.ndarray,
    eval_values: np.ndarray,
    eval_labels: np.ndarray,
    runs: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    records = []
    for index in range(runs):
        shuffled = [int(label) for label in source_train_labels.tolist()]
        rng.shuffle(shuffled)
        shuffled_labels = np.asarray(shuffled, dtype=np.int64)
        result = fit_and_score(source_train_values, shuffled_labels, eval_values, eval_labels)
        records.append(
            {
                "run": index,
                "train_auc": result["train_auc"],
                "eval_auc": result["auc"],
                "direction_norm": result["direction_norm"],
            }
        )
    eval_aucs = [float(record["eval_auc"] or 0.0) for record in records]
    return {
        "runs": runs,
        "seed": seed,
        "eval_auc_p95": percentile(eval_aucs, 0.95),
        "max_eval_auc": max(eval_aucs) if eval_aucs else None,
        "records": records,
    }


def score_transfer_template(
    template: str,
    source: dict[str, Any],
    target: dict[str, Any],
    fit: dict[str, Any],
    source_train_mask: np.ndarray,
    shuffle_runs: int,
    shuffle_seed: int,
) -> dict[str, Any]:
    target_masks = target["masks"]
    hidden_all = score_with_fit(fit, target["features"][target_masks["all"]], target["labels"][target_masks["all"]])
    hidden_holdout = score_with_fit(
        fit,
        target["features"][target_masks["holdout"]],
        target["labels"][target_masks["holdout"]],
    )
    controls = {}
    for field, source_values in source["scalar_fields"].items():
        controls[field] = {
            "all": fit_and_score(
                source_values[source_train_mask],
                source["labels"][source_train_mask],
                target["scalar_fields"][field][target_masks["all"]],
                target["labels"][target_masks["all"]],
            ),
            "holdout": fit_and_score(
                source_values[source_train_mask],
                source["labels"][source_train_mask],
                target["scalar_fields"][field][target_masks["holdout"]],
                target["labels"][target_masks["holdout"]],
            ),
        }
    best_control_holdout_auc = max(float(payload["holdout"]["auc"] or 0.0) for payload in controls.values())
    train_null = shuffled_train_label_null(
        source["features"][source_train_mask],
        source["labels"][source_train_mask],
        target["features"][target_masks["holdout"]],
        target["labels"][target_masks["holdout"]],
        shuffle_runs,
        shuffle_seed,
    )
    hidden_holdout_auc = float(hidden_holdout["auc"] or 0.0)
    return {
        "template": template,
        "row_count": len(target["rows"]),
        "split_counts": dict(sorted(Counter(row["split"] for row in target["rows"]).items())),
        "label_counts": dict(sorted(Counter(row["selected_label"] for row in target["rows"]).items())),
        "hidden_all": hidden_all,
        "hidden_holdout": hidden_holdout,
        "controls": controls,
        "best_control_holdout_auc": best_control_holdout_auc,
        "train_label_shuffle_null": train_null,
        "gate": {
            "holdout_auc_at_least_0p75": hidden_holdout_auc >= 0.75,
            "beats_best_control_holdout": hidden_holdout_auc > best_control_holdout_auc,
            "beats_train_shuffle_null_p95": hidden_holdout_auc > float(train_null["eval_auc_p95"] or 0.0),
        },
    }


def diagnostic_class(criteria: dict[str, bool]) -> str:
    if not criteria["v24_source_valid"] or not criteria["v25_source_valid"]:
        return "source_artifact_invalid"
    if not criteria["source_template_candidate_decoupled_ready"]:
        return "locked_coordinate_source_template_not_ready"
    if not criteria["transfer_template_found"]:
        return "locked_coordinate_transfer_template_absent"
    if not criteria["source_holdout_reproduced"]:
        return "locked_coordinate_source_reproduction_failed"
    if not criteria["all_transfer_hidden_auc_at_least_0p75"]:
        return "locked_coordinate_transfer_failed"
    if not criteria["all_transfer_hidden_beats_controls"]:
        return "locked_coordinate_transfer_control_confounded"
    if not criteria["all_transfer_hidden_beats_shuffle_null"]:
        return "locked_coordinate_transfer_shuffle_overfit"
    if criteria["locked_transfer_gate"]:
        return "locked_coordinate_transfer_signature_candidate"
    return "locked_coordinate_transfer_failed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--v24-artifact", type=Path, default=DEFAULT_V24_ARTIFACT)
    parser.add_argument("--v25-artifact", type=Path, default=DEFAULT_V25_ARTIFACT)
    parser.add_argument(
        "--artifact-prefix",
        default="mc006_qwen3_1p7b_parametric_fact_override_v26_locked_coordinate_transfer",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-runs", type=int, default=256)
    parser.add_argument("--shuffle-seed", type=int, default=26006)
    args = parser.parse_args()

    started = time.time()
    v24 = read_json(args.v24_artifact)
    v25 = read_json(args.v25_artifact)
    v24_validation = validate_v24(v24)
    v25_validation = validate_v25(v25)

    v25_hidden = v25["hidden_screen"]["selected_hidden"]
    locked_position = str(v25_hidden["position"])
    locked_layer = int(v25_hidden["layer"])
    source_template = str(v25["summary"]["selection"]["selected_template"])

    row_bank = list(v24["rows"])
    template_payloads = {
        template: template_candidate(row_bank, template)
        for template in v25["by_template"].keys()
    }
    source_payload = template_payloads[source_template]
    transfer_templates = [
        template
        for template, payload in template_payloads.items()
        if template != source_template and payload["candidate_decoupled_ready"]
    ]

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
    model.eval()

    source_features = collect_locked_template(
        source_payload["binary_rows"],
        tokenizer,
        model,
        locked_position,
        locked_layer,
    )
    source_train_mask = source_features["masks"]["non_holdout"]
    source_holdout_mask = source_features["masks"]["holdout"]
    locked_fit = fit_locked_direction(
        source_features["features"][source_train_mask],
        source_features["labels"][source_train_mask],
    )
    source_train = score_with_fit(
        locked_fit,
        source_features["features"][source_train_mask],
        source_features["labels"][source_train_mask],
    )
    source_holdout = score_with_fit(
        locked_fit,
        source_features["features"][source_holdout_mask],
        source_features["labels"][source_holdout_mask],
    )

    transfer_reports = {}
    for index, template in enumerate(transfer_templates):
        target_features = collect_locked_template(
            template_payloads[template]["binary_rows"],
            tokenizer,
            model,
            locked_position,
            locked_layer,
        )
        transfer_reports[template] = score_transfer_template(
            template,
            source_features,
            target_features,
            locked_fit,
            source_train_mask,
            args.shuffle_runs,
            args.shuffle_seed + index,
        )

    transfer_gates = [report["gate"] for report in transfer_reports.values()]
    criteria = {
        "v24_source_valid": v24_validation["passed"],
        "v25_source_valid": v25_validation["passed"],
        "locked_coordinate_available": bool(locked_position and locked_layer >= 0),
        "source_template_candidate_decoupled_ready": bool(source_payload["candidate_decoupled_ready"]),
        "transfer_template_found": bool(transfer_reports),
        "source_holdout_reproduced": float(source_holdout["auc"] or 0.0) >= 0.75,
        "all_transfer_hidden_auc_at_least_0p75": bool(transfer_gates)
        and all(gate["holdout_auc_at_least_0p75"] for gate in transfer_gates),
        "all_transfer_hidden_beats_controls": bool(transfer_gates)
        and all(gate["beats_best_control_holdout"] for gate in transfer_gates),
        "all_transfer_hidden_beats_shuffle_null": bool(transfer_gates)
        and all(gate["beats_train_shuffle_null_p95"] for gate in transfer_gates),
    }
    criteria["locked_transfer_gate"] = all(criteria.values())
    diag = diagnostic_class(criteria)

    summary = {
        "diagnostic_class": diag,
        "passed": criteria["locked_transfer_gate"],
        "diagnostic_supported": True,
        "signature_ready": criteria["locked_transfer_gate"],
        "intervention_ready": criteria["locked_transfer_gate"],
        "v24_validation": v24_validation,
        "v25_validation": v25_validation,
        "locked_coordinate": {
            "source": "v25_selected_hidden",
            "name": v25_hidden["name"],
            "position": locked_position,
            "layer": locked_layer,
        },
        "source_template": source_template,
        "transfer_templates": transfer_templates,
        "source_template_summary": {
            key: value for key, value in source_payload.items() if key != "binary_rows"
        },
        "transfer_template_summaries": {
            template: {key: value for key, value in template_payloads[template].items() if key != "binary_rows"}
            for template in transfer_templates
        },
        "criteria": criteria,
    }

    output_path = args.output_dir / f"{args.artifact_prefix}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    result = {
        "card_id": args.card_id,
        "run_type": RUN_TYPE,
        "model_id": args.model_id,
        "source_artifacts": [str(args.v24_artifact), str(args.v25_artifact)],
        "source_artifact_hashes": [
            {"path": str(args.v24_artifact), "sha256": sha256_file(args.v24_artifact)},
            {"path": str(args.v25_artifact), "sha256": sha256_file(args.v25_artifact)},
        ],
        "summary": summary,
        "source_fit": {
            "train_split": "non_holdout",
            "train_auc": locked_fit["train_auc"],
            "direction_norm": locked_fit["direction_norm"],
            "orientation": locked_fit["orientation"],
            "source_train": source_train,
            "source_holdout": source_holdout,
            "mapping": source_features["mapping"],
        },
        "transfer_reports": transfer_reports,
        "started_at": started,
        "finished_at": time.time(),
        "duration_seconds": time.time() - started,
    }
    result["output_path"] = str(output_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(summary["criteria"], indent=2))
    print(
        "RESULT "
        f"path={output_path} diagnostic={diag} "
        f"source_template={source_template} "
        f"transfer_templates={','.join(transfer_templates) or 'none'} "
        f"signature_ready={summary['signature_ready']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

