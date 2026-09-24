#!/usr/bin/env python
"""MC005 V28 donor activation replacement for the V26 row signature."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from mc005_associative_lookup_response_marker_v15_fine_localization import (
    baseline_summary,
    candidate_payload,
    next_token_features,
    summarize_path_arm,
)
from mc005_associative_lookup_response_marker_v25_factorial_row_heterogeneity import (
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
    validate_v25,
)
from mc005_associative_lookup_response_marker_v27_signature_intervention import (
    DEFAULT_V25_PATH,
    DEFAULT_V26_PATH,
    SIGNATURE_LAYER,
    SIGNATURE_POSITION,
    build_signature_delta,
    collect_vectors,
    compare_conditions,
    fraction,
    rows_by_id,
    sha256_file,
    shifted_positions,
    validate_v26,
)
from mc005_associative_lookup_source_edge import CARD_ID, MODEL_ID, RESULT_DIR


DEFAULT_V27_PATH = (
    RESULT_DIR
    / "mc005_qwen3_1p7b_response_marker_v27_signature_intervention_20260630T212004.json"
)
PATCH_POSITIONS = ("target_value", "distractor_value", "final_colon")
LOOKUP_PATCHES = [
    ("none", None, None, None),
    ("positive_target", "positive", "target_value", "target_value"),
    ("negative_target", "negative", "target_value", "target_value"),
    ("random_target", "random", "target_value", "target_value"),
    ("positive_final_colon", "positive", "final_colon", "final_colon"),
    ("positive_distractor", "positive", "distractor_value", "distractor_value"),
]
NULL_PATCHES = [
    ("positive_source_value", "positive", "source_value", "target_value"),
    ("positive_final_colon", "positive", "final_colon", "final_colon"),
    ("random_source_value", "random", "source_value", "target_value"),
]
NUMERIC_MATCH_FIELDS = (
    "baseline_margin",
    "target_slot",
    "distractor_slot",
    "target_distractor_token_distance",
)


def stable_int(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def validate_v27(v27: dict[str, Any], v25_hash: str, v26_hash: str) -> None:
    if v27.get("run_type") != "associative_lookup_response_marker_v27_signature_intervention":
        raise ValueError(f"unexpected V27 run_type: {v27.get('run_type')!r}")
    summary = v27.get("summary", {})
    if summary.get("diagnostic_class") != "plus_target_no_row_effect":
        raise ValueError(f"unexpected V27 diagnostic: {summary.get('diagnostic_class')!r}")
    if bool(summary.get("passed")):
        raise ValueError("V27 unexpectedly passed")
    if summary.get("v25_artifact_sha256") != v25_hash:
        raise ValueError("V27 V25 hash does not match current V25 artifact")
    if summary.get("v26_artifact_sha256") != v26_hash:
        raise ValueError("V27 V26 hash does not match current V26 artifact")


def donor_pools(labels: np.ndarray, discovery: np.ndarray) -> dict[str, list[int]]:
    discovery_indices = [index for index, keep in enumerate(discovery) if keep]
    positive = [index for index in discovery_indices if int(labels[index]) == 1]
    negative = [index for index in discovery_indices if int(labels[index]) == 0]
    if not positive or not negative:
        raise ValueError("donor replacement requires positive and negative discovery donors")
    return {
        "positive": positive,
        "negative": negative,
        "random": discovery_indices,
    }


def donor_scales(records: list[dict[str, Any]], discovery: np.ndarray) -> dict[str, float]:
    scales = {}
    discovery_records = [record for record, keep in zip(records, discovery, strict=True) if keep]
    for field in NUMERIC_MATCH_FIELDS:
        values = np.array([float(record[field]) for record in discovery_records], dtype=np.float32)
        scale = float(values.std())
        scales[field] = scale if scale > 1e-6 else 1.0
    return scales


def donor_distance(target: dict[str, Any], donor: dict[str, Any], scales: dict[str, float]) -> float:
    score = 0.0
    if target["target_position_group"] != donor["target_position_group"]:
        score += 10.0
    if target["distractor_relation"] != donor["distractor_relation"]:
        score += 5.0
    for field in NUMERIC_MATCH_FIELDS:
        score += abs(float(target[field]) - float(donor[field])) / scales[field]
    return score


def select_lookup_donor(
    target: dict[str, Any],
    records: list[dict[str, Any]],
    pool: list[int],
    mode: str,
    condition_name: str,
) -> tuple[int, float | None]:
    if mode == "random":
        donor_index = pool[stable_int(condition_name, target["row_id"], "random_donor") % len(pool)]
        return donor_index, None
    scales = select_lookup_donor.scales  # type: ignore[attr-defined]
    best_index = min(
        pool,
        key=lambda index: (
            donor_distance(target, records[index], scales),
            records[index]["row_id"],
        ),
    )
    return best_index, donor_distance(target, records[best_index], scales)


def donor_summary(matches: list[dict[str, Any]]) -> dict[str, Any]:
    labels: dict[str, int] = {}
    groups: dict[str, int] = {}
    unique = set()
    distances = []
    for match in matches:
        labels[str(match["donor_label"])] = labels.get(str(match["donor_label"]), 0) + 1
        group = str(match.get("donor_target_position_group"))
        groups[group] = groups.get(group, 0) + 1
        unique.add(str(match["donor_row_id"]))
        if match.get("match_distance") is not None:
            distances.append(float(match["match_distance"]))
    return {
        "rows": len(matches),
        "unique_donors": len(unique),
        "donor_label_counts": labels,
        "donor_target_position_group_counts": groups,
        "mean_match_distance": float(sum(distances) / len(distances)) if distances else None,
        "max_match_distance": max(distances) if distances else None,
    }


def build_lookup_patches(
    rows: list[dict[str, Any]],
    records_by_row_id: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
    labels: np.ndarray,
    pools: dict[str, list[int]],
    position_vectors: dict[str, np.ndarray],
    condition_name: str,
    donor_mode: str,
    donor_position: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    patches = {}
    matches = []
    pool = pools[donor_mode]
    for row in rows:
        target_record = records_by_row_id[row["id"]]
        donor_index, distance = select_lookup_donor(target_record, records, pool, donor_mode, condition_name)
        donor = records[donor_index]
        patches[row["id"]] = position_vectors[donor_position][donor_index].astype(np.float32)
        matches.append(
            {
                "row_id": row["id"],
                "donor_row_id": donor["row_id"],
                "donor_label": int(labels[donor_index]),
                "donor_seed": int(donor["seed"]),
                "donor_position": donor_position,
                "donor_target_position_group": donor["target_position_group"],
                "donor_distractor_relation": donor["distractor_relation"],
                "match_distance": distance,
            }
        )
    return patches, matches


def build_null_patches(
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    labels: np.ndarray,
    pools: dict[str, list[int]],
    position_vectors: dict[str, np.ndarray],
    condition_name: str,
    donor_mode: str,
    donor_position: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    patches = {}
    matches = []
    pool = pools[donor_mode]
    for row in rows:
        donor_index = pool[stable_int(condition_name, row["id"], donor_mode, donor_position) % len(pool)]
        donor = records[donor_index]
        patches[row["id"]] = position_vectors[donor_position][donor_index].astype(np.float32)
        matches.append(
            {
                "row_id": row["id"],
                "donor_row_id": donor["row_id"],
                "donor_label": int(labels[donor_index]),
                "donor_seed": int(donor["seed"]),
                "donor_position": donor_position,
                "donor_target_position_group": donor["target_position_group"],
                "donor_distractor_relation": donor["distractor_relation"],
                "match_distance": None,
            }
        )
    return patches, matches


def install_residual_replacement_hook(
    model: Any,
    layer: int,
    positions_by_batch: list[int],
    patch_vectors: torch.Tensor,
) -> Any:
    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0].clone()
            for batch_index, position in enumerate(positions_by_batch):
                hidden[batch_index, int(position), :] = patch_vectors[batch_index]
            return (hidden,) + output[1:]
        hidden = output.clone()
        for batch_index, position in enumerate(positions_by_batch):
            hidden[batch_index, int(position), :] = patch_vectors[batch_index]
        return hidden

    return model.model.layers[layer].register_forward_hook(hook)


def score_rows_patch(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    patch_position: str | None,
    patch_vectors_by_row_id: dict[str, np.ndarray] | None,
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
            if patch_position is not None and patch_vectors_by_row_id is not None:
                patch_vectors = torch.tensor(
                    np.stack([patch_vectors_by_row_id[row["id"]] for row in batch]),
                    device=model.device,
                    dtype=model.dtype,
                )
                handles.append(
                    install_residual_replacement_hook(
                        model,
                        SIGNATURE_LAYER,
                        shifted_positions(batch, patch_position, seq_lens, max_len),
                        patch_vectors,
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


def score_lookup_condition(
    rows: list[dict[str, Any]],
    records_by_row_id: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
    labels: np.ndarray,
    pools: dict[str, list[int]],
    position_vectors: dict[str, np.ndarray],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    patch: tuple[str, str | None, str | None, str | None],
) -> dict[str, Any]:
    name, donor_mode, patch_position, donor_position = patch
    patch_vectors = None
    matches: list[dict[str, Any]] = []
    if donor_mode is not None and patch_position is not None and donor_position is not None:
        patch_vectors, matches = build_lookup_patches(
            rows,
            records_by_row_id,
            records,
            labels,
            pools,
            position_vectors,
            name,
            donor_mode,
            donor_position,
        )
    baseline = score_rows_patch(rows, tokenizer, model, batch_size, patch_position, patch_vectors)
    raw_by_path = {}
    path_results = {}
    for path in TARGET_PATHS:
        raw = score_rows_patch(
            rows,
            tokenizer,
            model,
            batch_size,
            patch_position,
            patch_vectors,
            path,
            "target_value",
        )
        raw_by_path[path.name] = raw
        path_results[path.name] = {
            "candidate": candidate_payload(path, model),
            "target_value": summarize_path_arm(rows, baseline, raw),
        }
        target = path_results[path.name]["target_value"]
        print(f"[v28 {name} {path.name}] delta={target['mean_delta']:.4f} loss={target['target_win_loss']}")
    diagnostics = compute_row_diagnostics(rows, baseline, raw_by_path)
    row_summary = group_row_summary(rows, diagnostics)
    return {
        "name": name,
        "donor_mode": donor_mode,
        "patch_position": patch_position,
        "donor_position": donor_position,
        "baseline": baseline_summary(rows, baseline),
        "path_results": path_results,
        "row_summary": row_summary,
        "donor_summary": donor_summary(matches),
        "donor_matches": matches,
        "row_diagnostics": diagnostics,
        "baseline_raw": baseline,
        "raw_by_path": raw_by_path,
        "patch_vectors": patch_vectors,
    }


def score_parent_source_controls(
    rows: list[dict[str, Any]],
    records_by_row_id: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
    labels: np.ndarray,
    pools: dict[str, list[int]],
    position_vectors: dict[str, np.ndarray],
    tokenizer: Any,
    model: Any,
    batch_size: int,
) -> dict[str, Any]:
    patch_vectors, matches = build_lookup_patches(
        rows,
        records_by_row_id,
        records,
        labels,
        pools,
        position_vectors,
        "positive_target",
        "positive",
        "target_value",
    )
    baseline = score_rows_patch(rows, tokenizer, model, batch_size, "target_value", patch_vectors)
    arms = {}
    for source_key in ("target_value", "distractor_value", "random_value"):
        raw = score_rows_patch(
            rows,
            tokenizer,
            model,
            batch_size,
            "target_value",
            patch_vectors,
            PARENT,
            source_key,
        )
        arms[source_key] = summarize_path_arm(rows, baseline, raw)
    return {
        "candidate": candidate_payload(PARENT, model),
        "baseline": baseline_summary(rows, baseline),
        "donor_summary": donor_summary(matches),
        "arms": arms,
    }


def summarize_null_arm(
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    arm: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summary = summarize_path_arm(rows, baseline, arm)
    summary["replacement_null_clean"] = (
        int(summary["target_win_loss"]) == 0 and abs(float(summary["mean_delta"])) <= 0.25
    )
    return summary


def score_nulls(
    null_rows_by_seed: dict[int, list[dict[str, Any]]],
    records: list[dict[str, Any]],
    labels: np.ndarray,
    pools: dict[str, list[int]],
    position_vectors: dict[str, np.ndarray],
    tokenizer: Any,
    model: Any,
    batch_size: int,
) -> dict[str, Any]:
    results = {}
    for seed, rows in sorted(null_rows_by_seed.items()):
        baseline = score_rows_patch(rows, tokenizer, model, batch_size, None, None)
        arms = {}
        donor_summaries = {}
        for name, donor_mode, patch_position, donor_position in NULL_PATCHES:
            assert donor_mode is not None and patch_position is not None and donor_position is not None
            patch_vectors, matches = build_null_patches(
                rows,
                records,
                labels,
                pools,
                position_vectors,
                name,
                donor_mode,
                donor_position,
            )
            raw = score_rows_patch(rows, tokenizer, model, batch_size, patch_position, patch_vectors)
            arms[name] = summarize_null_arm(rows, baseline, raw)
            donor_summaries[name] = donor_summary(matches)
        base = baseline_summary(rows, baseline)
        floor = int(0.75 * len(rows))
        label = "clean_null" if int(base["target_wins"]) >= floor and all(
            arm["replacement_null_clean"] for arm in arms.values()
        ) else "side_effect"
        results[seed] = {
            "seed": seed,
            "label": label,
            "baseline": base,
            "baseline_floor": floor,
            "arms": arms,
            "donor_summaries": donor_summaries,
        }
        print(
            f"[v28 null {seed}] {label} "
            + " ".join(f"{key}={arm['mean_delta']:.4f}/{arm['target_win_loss']}" for key, arm in arms.items())
        )
    return results


def classify(criteria: dict[str, bool]) -> str:
    if all(criteria.values()):
        return "donor_replacement_supported"
    if not criteria["source_artifacts_valid"]:
        return "source_artifact_invalid"
    if not criteria["source_signature_valid"]:
        return "source_signature_invalid"
    if not criteria["base_holdout_parent_valid"]:
        return "base_parent_invalid"
    if not criteria["positive_target_all_three_fraction_gain_at_least_0p10"]:
        return "positive_target_no_row_effect"
    if not criteria["positive_target_best_pair_share_drop_at_least_0p05"]:
        return "positive_target_no_pair_share_effect"
    if not criteria["negative_target_opposes_positive"]:
        return "negative_not_opposed"
    if not criteria["controls_smaller_than_positive"]:
        return "control_matched_effect"
    if not criteria["positive_parent_source_controls_pass"]:
        return "source_control_failed"
    if not criteria["replacement_nulls_clean"]:
        return "null_failed"
    return "mixed_failure"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--v25-artifact", type=Path, default=DEFAULT_V25_PATH)
    parser.add_argument("--v26-artifact", type=Path, default=DEFAULT_V26_PATH)
    parser.add_argument("--v27-artifact", type=Path, default=DEFAULT_V27_PATH)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_response_marker_v28_donor_replacement")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=28001)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    v25_hash = sha256_file(args.v25_artifact)
    v26_hash = sha256_file(args.v26_artifact)
    v27_hash = sha256_file(args.v27_artifact)
    v25 = read_json(args.v25_artifact)
    v26 = read_json(args.v26_artifact)
    v27 = read_json(args.v27_artifact)
    validate_v25(v25)
    validate_v26(v26, v25_hash)
    validate_v27(v27, v25_hash, v26_hash)

    records = build_records(v25)
    labels = labels_for(records)
    discovery = np.array([record["seed"] == DISCOVERY_SEED for record in records], dtype=bool)
    holdout = np.array([record["seed"] == HOLDOUT_SEED for record in records], dtype=bool)
    lookup_by_id = rows_by_id(v25, "lookup")
    holdout_rows = [lookup_by_id[record["row_id"]] for record in records if record["seed"] == HOLDOUT_SEED]
    records_by_row_id = {record["row_id"]: record for record in records}
    null_rows = [row for row in v25["rows"] if row.get("mode") == "null"]
    null_rows_by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in null_rows:
        null_rows_by_seed.setdefault(int(row["seed"]), []).append(row)

    select_lookup_donor.scales = donor_scales(records, discovery)  # type: ignore[attr-defined]
    pools = donor_pools(labels, discovery)

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
    target_vectors = collect_vectors(records, tokenizer, model, args.batch_size, SIGNATURE_LAYER, SIGNATURE_POSITION)
    direction = build_signature_delta(
        target_vectors,
        labels,
        discovery,
        holdout,
        4.0,
        args.random_seed,
    )
    position_vectors = {"target_value": target_vectors}
    for position in ("distractor_value", "final_colon"):
        position_vectors[position] = collect_vectors(records, tokenizer, model, args.batch_size, SIGNATURE_LAYER, position)
    print(
        "[v28 direction] "
        f"discovery_auc={direction['discovery_auc']:.4f} "
        f"holdout_auc={direction['holdout_auc']:.4f}"
    )

    condition_results: dict[str, Any] = {}
    for patch in LOOKUP_PATCHES:
        condition = score_lookup_condition(
            holdout_rows,
            records_by_row_id,
            records,
            labels,
            pools,
            position_vectors,
            tokenizer,
            model,
            args.batch_size,
            patch,
        )
        condition_results[condition["name"]] = {
            key: value
            for key, value in condition.items()
            if key not in {"baseline_raw", "raw_by_path", "row_diagnostics", "patch_vectors"}
        }
        condition_results[condition["name"]]["row_diagnostics"] = condition["row_diagnostics"]
        print(
            f"[v28 condition {condition['name']}] "
            f"all3={condition['row_summary']['all_three_margin_row_count']}/"
            f"{condition['row_summary']['parent_effect_row_count']} "
            f"best_pair={condition['row_summary']['median_best_pair_share']}"
        )

    comparisons = compare_conditions(condition_results)
    parent_source_controls = score_parent_source_controls(
        holdout_rows,
        records_by_row_id,
        records,
        labels,
        pools,
        position_vectors,
        tokenizer,
        model,
        args.batch_size,
    )
    null_results = score_nulls(
        null_rows_by_seed,
        records,
        labels,
        pools,
        position_vectors,
        tokenizer,
        model,
        args.batch_size,
    )

    none_summary = condition_results["none"]["row_summary"]
    positive_summary = condition_results["positive_target"]["row_summary"]
    negative_summary = condition_results["negative_target"]["row_summary"]
    positive_gain = float(comparisons["positive_target"]["all_three_fraction_delta_vs_none"])
    positive_pair_delta = float(comparisons["positive_target"]["median_best_pair_share_delta_vs_none"])
    negative_gain = float(comparisons["negative_target"]["all_three_fraction_delta_vs_none"])
    negative_pair = negative_summary["median_best_pair_share"]
    positive_pair = positive_summary["median_best_pair_share"]
    control_names = ["random_target", "positive_final_colon", "positive_distractor"]
    controls_smaller = all(
        abs(float(comparisons[name]["all_three_fraction_delta_vs_none"])) <= 0.03
        or float(comparisons[name]["all_three_fraction_delta_vs_none"]) < positive_gain / 2.0
        for name in control_names
    )
    parent_arms = parent_source_controls["arms"]
    criteria = {
        "source_artifacts_valid": True,
        "source_signature_valid": (
            v26.get("summary", {}).get("selected_internal_candidate", {}).get("name") == "l20_target_value"
            and float(direction["holdout_auc"]) >= 0.70
        ),
        "base_holdout_parent_valid": (
            int(condition_results["none"]["baseline"]["target_wins"]) >= int(0.75 * len(holdout_rows))
            and int(none_summary["parent_effect_row_count"]) >= 80
        ),
        "positive_target_all_three_fraction_gain_at_least_0p10": positive_gain >= 0.10,
        "positive_target_best_pair_share_drop_at_least_0p05": positive_pair_delta <= -0.05,
        "negative_target_opposes_positive": (
            negative_gain <= 0.03
            or (
                negative_pair is not None
                and positive_pair is not None
                and float(negative_pair) >= float(positive_pair) + 0.03
            )
        ),
        "controls_smaller_than_positive": controls_smaller,
        "positive_parent_source_controls_pass": (
            float(parent_arms["target_value"]["mean_delta"]) <= float(parent_arms["distractor_value"]["mean_delta"]) - 0.50
            and float(parent_arms["target_value"]["mean_delta"]) <= float(parent_arms["random_value"]["mean_delta"]) - 0.50
        ),
        "replacement_nulls_clean": all(result["label"] == "clean_null" for result in null_results.values()),
    }

    summary = {
        "model_id": args.model_id,
        "v25_artifact": str(args.v25_artifact),
        "v25_artifact_sha256": v25_hash,
        "v26_artifact": str(args.v26_artifact),
        "v26_artifact_sha256": v26_hash,
        "v27_artifact": str(args.v27_artifact),
        "v27_artifact_sha256": v27_hash,
        "signature_layer": SIGNATURE_LAYER,
        "signature_position": SIGNATURE_POSITION,
        "discovery_seed": DISCOVERY_SEED,
        "holdout_seed": HOLDOUT_SEED,
        "holdout_row_count": len(holdout_rows),
        "donor_pool_counts": {key: len(value) for key, value in pools.items()},
        "donor_match_fields": list(NUMERIC_MATCH_FIELDS),
        "direction": {
            "discovery_auc": direction["discovery_auc"],
            "holdout_auc": direction["holdout_auc"],
            "direction_norm": direction["direction_norm"],
        },
        "condition_summaries": {
            name: {key: value for key, value in result.items() if key != "row_diagnostics"}
            for name, result in condition_results.items()
        },
        "comparisons": comparisons,
        "parent_source_controls_positive_target": parent_source_controls,
        "null_results": null_results,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "diagnostic_class": classify(criteria),
    }
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_response_marker_v28_donor_replacement",
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
