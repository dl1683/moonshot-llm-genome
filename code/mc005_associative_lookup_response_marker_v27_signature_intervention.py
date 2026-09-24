#!/usr/bin/env python
"""MC005 V27 causal intervention on the V26 l20 target-value row signature."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc001_qwen3_controlled_v10_head_localization import (
    install_source_mask_hook,
    set_eager_attention,
    tokenizer_padding_side,
)
from mc003_delayed_copy_signature import auc_score
from mc005_associative_lookup_response_marker_v15_fine_localization import (
    baseline_summary,
    candidate_payload,
    next_token_features,
    summarize_path_arm,
)
from mc005_associative_lookup_response_marker_v25_factorial_row_heterogeneity import (
    NULL_ARM_KEYS,
    PARENT,
    TARGET_PATHS,
    compute_row_diagnostics,
    group_row_summary,
)
from mc005_associative_lookup_response_marker_v26_internal_row_signature import (
    DISCOVERY_SEED,
    HOLDOUT_SEED,
    build_records,
    labels_for,
    read_json,
    split_mask,
    validate_v25,
)
from mc005_associative_lookup_source_edge import CARD_ID, MODEL_ID, RESULT_DIR


DEFAULT_V25_PATH = (
    RESULT_DIR
    / "mc005_qwen3_1p7b_response_marker_v25_factorial_row_heterogeneity_20260630T205454.json"
)
DEFAULT_V26_PATH = (
    RESULT_DIR
    / "mc005_qwen3_1p7b_response_marker_v26_internal_row_signature_20260630T210810.json"
)
SIGNATURE_LAYER = 20
SIGNATURE_HIDDEN_INDEX = SIGNATURE_LAYER + 1
SIGNATURE_POSITION = "target_value"
SCORE_SHIFT = 4.0
LOOKUP_INTERVENTIONS = [
    ("none", None, 0.0, "target_value"),
    ("plus_target", "signature", 1.0, "target_value"),
    ("minus_target", "signature", -1.0, "target_value"),
    ("random_target", "random", 1.0, "target_value"),
    ("plus_final_colon", "signature", 1.0, "final_colon"),
    ("plus_distractor", "signature", 1.0, "distractor_value"),
]
NULL_INTERVENTIONS = [
    ("plus_source_value", "signature", 1.0, "source_value"),
    ("plus_final_colon", "signature", 1.0, "final_colon"),
    ("random_source_value", "random", 1.0, "source_value"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_v26(v26: dict[str, Any], v25_sha256: str) -> None:
    if v26.get("run_type") != "associative_lookup_response_marker_v26_internal_row_signature":
        raise ValueError(f"unexpected V26 run_type: {v26.get('run_type')!r}")
    summary = v26.get("summary", {})
    if summary.get("diagnostic_class") != "internal_row_signature_supported":
        raise ValueError(f"unexpected V26 diagnostic class: {summary.get('diagnostic_class')!r}")
    selected = summary.get("selected_internal_candidate", {})
    if selected.get("name") != "l20_target_value":
        raise ValueError(f"unexpected V26 selected candidate: {selected.get('name')!r}")
    if summary.get("source_artifact_sha256") != v25_sha256:
        raise ValueError("V26 source-artifact hash does not match V25 artifact")


def rows_by_id(v25: dict[str, Any], mode: str) -> dict[str, dict[str, Any]]:
    return {
        row["id"]: row
        for row in v25["rows"]
        if row.get("mode") == mode
    }


def collect_vectors(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    layer: int,
    position: str,
) -> np.ndarray:
    chunks = []
    with tokenizer_padding_side(tokenizer, "left"):
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            inputs = tokenizer(
                [record["rendered_prompt"] for record in batch],
                return_tensors="pt",
                padding=True,
            ).to(model.device)
            seq_lens = [int(value) for value in inputs["attention_mask"].sum(dim=1).detach().cpu().tolist()]
            max_len = int(inputs["input_ids"].shape[-1])
            shifted = [
                int(record["positions"][position]) + (max_len - seq_len)
                for record, seq_len in zip(batch, seq_lens, strict=True)
            ]
            with torch.inference_mode():
                out = model(**inputs, output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states[layer + 1].detach().float().cpu()
            chunks.append(torch.stack([hidden[index, shifted[index], :] for index in range(len(batch))]))
    return torch.cat(chunks, dim=0).numpy().astype(np.float32)


def build_signature_delta(
    vectors: np.ndarray,
    labels: np.ndarray,
    discovery: np.ndarray,
    holdout: np.ndarray,
    score_shift: float,
    random_seed: int,
) -> dict[str, Any]:
    x_train = vectors[discovery]
    y_train = labels[discovery]
    x_holdout = vectors[holdout]
    y_holdout = labels[holdout]
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    z_train = (x_train - mean) / std
    z_holdout = (x_holdout - mean) / std
    pos = z_train[y_train == 1]
    neg = z_train[y_train == 0]
    raw_direction = pos.mean(axis=0) - neg.mean(axis=0)
    direction_norm = float(np.linalg.norm(raw_direction))
    if direction_norm < 1e-12:
        raise ValueError("zero V27 signature direction")
    unit_z = (raw_direction / direction_norm).astype(np.float32)
    train_scores = z_train @ unit_z
    holdout_scores = z_holdout @ unit_z
    discovery_auc = auc_score([float(value) for value in train_scores], [int(value) for value in y_train])
    holdout_auc = auc_score([float(value) for value in holdout_scores], [int(value) for value in y_holdout])
    gradient = (unit_z.reshape(1, -1) / std).reshape(-1).astype(np.float32)
    denom = float(np.dot(gradient, gradient))
    if denom < 1e-12:
        raise ValueError("zero raw gradient for V27 signature direction")
    delta = (float(score_shift) * gradient / denom).astype(np.float32)
    rng = np.random.default_rng(random_seed)
    random_delta = rng.normal(size=delta.shape).astype(np.float32)
    random_delta = random_delta / max(float(np.linalg.norm(random_delta)), 1e-12)
    random_delta = random_delta * float(np.linalg.norm(delta))
    expected_shift = float(np.dot(gradient, delta))
    return {
        "mean": mean.reshape(-1).astype(np.float32),
        "std": std.reshape(-1).astype(np.float32),
        "unit_z": unit_z,
        "gradient": gradient,
        "delta": delta,
        "random_delta": random_delta.astype(np.float32),
        "direction_norm": direction_norm,
        "delta_norm": float(np.linalg.norm(delta)),
        "random_delta_norm": float(np.linalg.norm(random_delta)),
        "expected_signature_score_shift": expected_shift,
        "discovery_auc": discovery_auc,
        "holdout_auc": holdout_auc,
    }


def install_residual_delta_hook(
    model: Any,
    layer: int,
    positions_by_batch: list[int],
    delta: torch.Tensor,
) -> Any:
    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0].clone()
            for batch_index, position in enumerate(positions_by_batch):
                hidden[batch_index, int(position), :] = hidden[batch_index, int(position), :] + delta
            return (hidden,) + output[1:]
        hidden = output.clone()
        for batch_index, position in enumerate(positions_by_batch):
            hidden[batch_index, int(position), :] = hidden[batch_index, int(position), :] + delta
        return hidden

    return model.model.layers[layer].register_forward_hook(hook)


def shifted_positions(rows: list[dict[str, Any]], key: str, seq_lens: list[int], max_len: int) -> list[int]:
    return [
        int(row["positions"][key]) + (max_len - seq_len)
        for row, seq_len in zip(rows, seq_lens, strict=True)
    ]


def score_rows_intervention(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    residual_position: str | None,
    residual_delta: np.ndarray | None,
    candidate: Any | None = None,
    source_key: str | None = None,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    heads = []
    if candidate is not None:
        heads = list(candidate.heads) if candidate.heads is not None else list(range(model.config.num_attention_heads))
    with tokenizer_padding_side(tokenizer, "left"):
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            inputs = tokenizer(
                [row["rendered_prompt"] for row in batch],
                return_tensors="pt",
                padding=True,
            ).to(model.device)
            seq_lens = [int(value) for value in inputs["attention_mask"].sum(dim=1).detach().cpu().tolist()]
            max_len = int(inputs["input_ids"].shape[-1])
            handles = []
            if residual_position is not None and residual_delta is not None:
                delta = torch.tensor(residual_delta, device=model.device, dtype=model.dtype)
                handles.append(
                    install_residual_delta_hook(
                        model,
                        SIGNATURE_LAYER,
                        shifted_positions(batch, residual_position, seq_lens, max_len),
                        delta,
                    )
                )
            if candidate is not None and source_key is not None:
                source_positions = [
                    [position] for position in shifted_positions(batch, source_key, seq_lens, max_len)
                ]
                handles.extend(
                    install_source_mask_hook(model, int(layer), heads, source_positions)
                    for layer in candidate.layers
                )
            try:
                with torch.inference_mode():
                    out = model(**inputs, use_cache=False, logits_to_keep=1)
            finally:
                for handle in handles:
                    handle.remove()
            for index, row in enumerate(batch):
                features[row["id"]] = next_token_features(out.logits[index, -1, :].detach().float(), row, tokenizer)
    return features


def delta_for(kind: str | None, sign: float, direction: dict[str, Any]) -> np.ndarray | None:
    if kind is None:
        return None
    if kind == "signature":
        return (float(sign) * direction["delta"]).astype(np.float32)
    if kind == "random":
        return (float(sign) * direction["random_delta"]).astype(np.float32)
    raise ValueError(f"unknown intervention kind: {kind}")


def score_lookup_condition(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    direction: dict[str, Any],
    intervention: tuple[str, str | None, float, str],
) -> dict[str, Any]:
    name, kind, sign, position = intervention
    delta = delta_for(kind, sign, direction)
    residual_position = position if delta is not None else None
    baseline = score_rows_intervention(rows, tokenizer, model, batch_size, residual_position, delta)
    raw_by_path = {}
    path_results = {}
    for path in TARGET_PATHS:
        raw = score_rows_intervention(
            rows,
            tokenizer,
            model,
            batch_size,
            residual_position,
            delta,
            path,
            "target_value",
        )
        raw_by_path[path.name] = raw
        path_results[path.name] = {
            "candidate": candidate_payload(path, model),
            "target_value": summarize_path_arm(rows, baseline, raw),
        }
        target = path_results[path.name]["target_value"]
        print(f"[v27 {name} {path.name}] delta={target['mean_delta']:.4f} loss={target['target_win_loss']}")
    diagnostics = compute_row_diagnostics(rows, baseline, raw_by_path)
    row_summary = group_row_summary(rows, diagnostics)
    return {
        "name": name,
        "kind": kind,
        "sign": sign,
        "position": position,
        "delta_norm": float(np.linalg.norm(delta)) if delta is not None else 0.0,
        "baseline": baseline_summary(rows, baseline),
        "path_results": path_results,
        "row_summary": row_summary,
        "row_diagnostics": diagnostics,
        "baseline_raw": baseline,
        "raw_by_path": raw_by_path,
    }


def score_parent_source_controls(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    direction: dict[str, Any],
) -> dict[str, Any]:
    delta = direction["delta"]
    baseline = score_rows_intervention(rows, tokenizer, model, batch_size, "target_value", delta)
    arms = {}
    for source_key in ("target_value", "distractor_value", "random_value"):
        raw = score_rows_intervention(
            rows,
            tokenizer,
            model,
            batch_size,
            "target_value",
            delta,
            PARENT,
            source_key,
        )
        arms[source_key] = summarize_path_arm(rows, baseline, raw)
    return {
        "candidate": candidate_payload(PARENT, model),
        "baseline": baseline_summary(rows, baseline),
        "arms": arms,
    }


def fraction(summary: dict[str, Any]) -> float:
    value = summary["all_three_fraction_of_parent_effect_rows"]
    return float(value) if value is not None else 0.0


def compare_conditions(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    none = conditions["none"]["row_summary"]
    result = {}
    for name, condition in conditions.items():
        row_summary = condition["row_summary"]
        result[name] = {
            "all_three_count_delta_vs_none": (
                int(row_summary["all_three_margin_row_count"]) - int(none["all_three_margin_row_count"])
            ),
            "all_three_fraction_delta_vs_none": fraction(row_summary) - fraction(none),
            "median_best_pair_share_delta_vs_none": (
                float(row_summary["median_best_pair_share"]) - float(none["median_best_pair_share"])
                if row_summary["median_best_pair_share"] is not None
                and none["median_best_pair_share"] is not None
                else None
            ),
            "median_singles_residual_delta_vs_none": (
                float(row_summary["median_singles_residual"]) - float(none["median_singles_residual"])
                if row_summary["median_singles_residual"] is not None
                and none["median_singles_residual"] is not None
                else None
            ),
        }
    return result


def summarize_null_arm(
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    arm: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summary = summarize_path_arm(rows, baseline, arm)
    summary["residual_null_clean"] = (
        int(summary["target_win_loss"]) == 0 and abs(float(summary["mean_delta"])) <= 0.25
    )
    return summary


def score_nulls(
    null_rows_by_seed: dict[int, list[dict[str, Any]]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    direction: dict[str, Any],
) -> dict[str, Any]:
    results = {}
    for seed, rows in sorted(null_rows_by_seed.items()):
        baseline = score_rows_intervention(rows, tokenizer, model, batch_size, None, None)
        arms = {}
        for name, kind, sign, position in NULL_INTERVENTIONS:
            delta = delta_for(kind, sign, direction)
            raw = score_rows_intervention(rows, tokenizer, model, batch_size, position, delta)
            arms[name] = summarize_null_arm(rows, baseline, raw)
        base = baseline_summary(rows, baseline)
        floor = int(0.75 * len(rows))
        label = "clean_null" if int(base["target_wins"]) >= floor and all(
            arm["residual_null_clean"] for arm in arms.values()
        ) else "side_effect"
        results[seed] = {
            "seed": seed,
            "label": label,
            "baseline": base,
            "baseline_floor": floor,
            "arms": arms,
        }
        print(
            f"[v27 null {seed}] {label} "
            + " ".join(f"{key}={arm['mean_delta']:.4f}/{arm['target_win_loss']}" for key, arm in arms.items())
        )
    return results


def classify(criteria: dict[str, bool]) -> str:
    if all(criteria.values()):
        return "signature_intervention_supported"
    if not criteria["source_signature_valid"]:
        return "source_signature_invalid"
    if not criteria["base_holdout_parent_valid"]:
        return "base_parent_invalid"
    if not criteria["plus_target_all_three_fraction_gain_at_least_0p10"]:
        return "plus_target_no_row_effect"
    if not criteria["plus_target_best_pair_share_drop_at_least_0p05"]:
        return "plus_target_no_pair_share_effect"
    if not criteria["minus_target_opposes_plus"]:
        return "minus_not_opposed"
    if not criteria["controls_smaller_than_plus"]:
        return "control_matched_effect"
    if not criteria["plus_parent_source_controls_pass"]:
        return "source_control_failed"
    if not criteria["residual_nulls_clean"]:
        return "null_failed"
    return "mixed_failure"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--v25-artifact", type=Path, default=DEFAULT_V25_PATH)
    parser.add_argument("--v26-artifact", type=Path, default=DEFAULT_V26_PATH)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_response_marker_v27_signature_intervention")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--score-shift", type=float, default=SCORE_SHIFT)
    parser.add_argument("--random-seed", type=int, default=27001)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    v25_hash = sha256_file(args.v25_artifact)
    v26_hash = sha256_file(args.v26_artifact)
    v25 = read_json(args.v25_artifact)
    v26 = read_json(args.v26_artifact)
    validate_v25(v25)
    validate_v26(v26, v25_hash)
    records = build_records(v25)
    labels = labels_for(records)
    discovery = np.array([record["seed"] == DISCOVERY_SEED for record in records], dtype=bool)
    holdout = np.array([record["seed"] == HOLDOUT_SEED for record in records], dtype=bool)
    lookup_by_id = rows_by_id(v25, "lookup")
    holdout_rows = [lookup_by_id[record["row_id"]] for record in records if record["seed"] == HOLDOUT_SEED]
    null_rows = [row for row in v25["rows"] if row.get("mode") == "null"]
    null_rows_by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in null_rows:
        null_rows_by_seed.setdefault(int(row["seed"]), []).append(row)

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
    vectors = collect_vectors(records, tokenizer, model, args.batch_size, SIGNATURE_LAYER, SIGNATURE_POSITION)
    direction = build_signature_delta(vectors, labels, discovery, holdout, args.score_shift, args.random_seed)
    print(
        "[v27 direction] "
        f"discovery_auc={direction['discovery_auc']:.4f} "
        f"holdout_auc={direction['holdout_auc']:.4f} "
        f"delta_norm={direction['delta_norm']:.4f} "
        f"expected_shift={direction['expected_signature_score_shift']:.4f}"
    )

    condition_results: dict[str, Any] = {}
    for intervention in LOOKUP_INTERVENTIONS:
        condition = score_lookup_condition(holdout_rows, tokenizer, model, args.batch_size, direction, intervention)
        condition_results[condition["name"]] = {
            key: value
            for key, value in condition.items()
            if key not in {"baseline_raw", "raw_by_path", "row_diagnostics"}
        }
        condition_results[condition["name"]]["row_diagnostics"] = condition["row_diagnostics"]
        print(
            f"[v27 condition {condition['name']}] "
            f"all3={condition['row_summary']['all_three_margin_row_count']}/"
            f"{condition['row_summary']['parent_effect_row_count']} "
            f"best_pair={condition['row_summary']['median_best_pair_share']}"
        )

    comparisons = compare_conditions(condition_results)
    parent_source_controls = score_parent_source_controls(
        holdout_rows,
        tokenizer,
        model,
        args.batch_size,
        direction,
    )
    null_results = score_nulls(null_rows_by_seed, tokenizer, model, args.batch_size, direction)

    none_summary = condition_results["none"]["row_summary"]
    plus_summary = condition_results["plus_target"]["row_summary"]
    minus_summary = condition_results["minus_target"]["row_summary"]
    plus_gain = float(comparisons["plus_target"]["all_three_fraction_delta_vs_none"])
    plus_pair_delta = float(comparisons["plus_target"]["median_best_pair_share_delta_vs_none"])
    minus_gain = float(comparisons["minus_target"]["all_three_fraction_delta_vs_none"])
    minus_pair = minus_summary["median_best_pair_share"]
    plus_pair = plus_summary["median_best_pair_share"]
    control_names = ["random_target", "plus_final_colon", "plus_distractor"]
    controls_smaller = all(
        abs(float(comparisons[name]["all_three_fraction_delta_vs_none"])) <= 0.03
        or float(comparisons[name]["all_three_fraction_delta_vs_none"]) < plus_gain / 2.0
        for name in control_names
    )
    parent_arms = parent_source_controls["arms"]
    criteria = {
        "source_signature_valid": (
            v26.get("summary", {}).get("selected_internal_candidate", {}).get("name") == "l20_target_value"
            and abs(float(direction["expected_signature_score_shift"]) - float(args.score_shift)) <= 1e-4
        ),
        "base_holdout_parent_valid": (
            int(condition_results["none"]["baseline"]["target_wins"]) >= int(0.75 * len(holdout_rows))
            and int(none_summary["parent_effect_row_count"]) >= 80
        ),
        "plus_target_all_three_fraction_gain_at_least_0p10": plus_gain >= 0.10,
        "plus_target_best_pair_share_drop_at_least_0p05": plus_pair_delta <= -0.05,
        "minus_target_opposes_plus": (
            minus_gain <= 0.03
            or (
                minus_pair is not None
                and plus_pair is not None
                and float(minus_pair) >= float(plus_pair) + 0.03
            )
        ),
        "controls_smaller_than_plus": controls_smaller,
        "plus_parent_source_controls_pass": (
            float(parent_arms["target_value"]["mean_delta"]) <= float(parent_arms["distractor_value"]["mean_delta"]) - 0.50
            and float(parent_arms["target_value"]["mean_delta"]) <= float(parent_arms["random_value"]["mean_delta"]) - 0.50
        ),
        "residual_nulls_clean": all(result["label"] == "clean_null" for result in null_results.values()),
    }

    summary = {
        "model_id": args.model_id,
        "v25_artifact": str(args.v25_artifact),
        "v25_artifact_sha256": v25_hash,
        "v26_artifact": str(args.v26_artifact),
        "v26_artifact_sha256": v26_hash,
        "signature_layer": SIGNATURE_LAYER,
        "signature_hidden_index": SIGNATURE_HIDDEN_INDEX,
        "signature_position": SIGNATURE_POSITION,
        "score_shift": float(args.score_shift),
        "discovery_seed": DISCOVERY_SEED,
        "holdout_seed": HOLDOUT_SEED,
        "holdout_row_count": len(holdout_rows),
        "direction": {
            "discovery_auc": direction["discovery_auc"],
            "holdout_auc": direction["holdout_auc"],
            "direction_norm": direction["direction_norm"],
            "delta_norm": direction["delta_norm"],
            "random_delta_norm": direction["random_delta_norm"],
            "expected_signature_score_shift": direction["expected_signature_score_shift"],
        },
        "condition_summaries": {
            name: {key: value for key, value in result.items() if key != "row_diagnostics"}
            for name, result in condition_results.items()
        },
        "comparisons": comparisons,
        "parent_source_controls_plus_target": parent_source_controls,
        "null_results": null_results,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "diagnostic_class": classify(criteria),
    }
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_response_marker_v27_signature_intervention",
        "model_id": args.model_id,
        "elapsed_s": time.time() - started,
        "summary": summary,
        "row_diagnostics_by_condition": {
            name: result["row_diagnostics"] for name, result in condition_results.items()
        },
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
