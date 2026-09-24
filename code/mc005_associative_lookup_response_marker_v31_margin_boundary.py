#!/usr/bin/env python
"""MC005 V31 margin-boundary audit for attention-write replacement."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc001_qwen3_controlled_v10_head_localization import set_eager_attention
from mc005_associative_lookup_response_marker_v15_fine_localization import (
    baseline_summary,
    score_rows_path,
    summarize_path_arm,
)
from mc005_associative_lookup_response_marker_v25_factorial_row_heterogeneity import build_null_rows
from mc005_associative_lookup_response_marker_v26_internal_row_signature import read_json, validate_v25
from mc005_associative_lookup_response_marker_v27_signature_intervention import DEFAULT_V25_PATH, sha256_file
from mc005_associative_lookup_response_marker_v29_attention_write_replacement import (
    NULL_SOURCE_KEYS,
    WRITE_LAYERS,
    collect_masked_attention_writes,
    parent_effect_holdout_rows,
    score_rows_write_replacement,
)
from mc005_associative_lookup_response_marker_v30_write_null_sweep import (
    DEFAULT_V29_PATH,
    changed_row_audit,
    validate_v29,
)
from mc005_associative_lookup_source_edge import CARD_ID, MODEL_ID, RESULT_DIR


DEFAULT_V30_PATH = RESULT_DIR / "mc005_qwen3_1p7b_response_marker_v30_write_null_sweep_20260630T215937.json"
FRESH_SEEDS = (311, 313, 317, 331, 337, 347, 349, 353)
FRESH_ROW_COUNT = 128


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def validate_v30(v30: dict[str, Any], v25_hash: str, v29_hash: str) -> None:
    if v30.get("run_type") != "associative_lookup_response_marker_v30_write_null_sweep":
        raise ValueError(f"unexpected V30 run_type: {v30.get('run_type')!r}")
    summary = v30.get("summary", {})
    if summary.get("diagnostic_class") != "fresh_write_null_failed":
        raise ValueError(f"unexpected V30 diagnostic class: {summary.get('diagnostic_class')!r}")
    if summary.get("source_artifact_sha256") != v25_hash:
        raise ValueError("V30 V25 source hash does not match current V25 source artifact")
    if summary.get("v29_artifact_sha256") != v29_hash:
        raise ValueError("V30 V29 source hash does not match current V29 source artifact")
    criteria = summary.get("criteria", {})
    required = {
        "source_artifact_valid": True,
        "v29_artifact_valid": True,
        "v29_seed251_failure_reproduced": True,
        "fresh_no_large_mean_delta": True,
        "fresh_no_target_win_change": False,
    }
    for key, expected in required.items():
        if bool(criteria.get(key)) is not expected:
            raise ValueError(f"unexpected V30 criterion {key}: {criteria.get(key)!r}")


def margin(features: dict[str, Any]) -> float:
    return float(features["target_minus_distractor_margin"])


def margin_band(value: float) -> str:
    abs_value = abs(float(value))
    if abs_value <= 0.25:
        return "le_0p25"
    if abs_value <= 0.5:
        return "0p25_0p5"
    if abs_value <= 1.0:
        return "0p5_1"
    if abs_value <= 2.0:
        return "1_2"
    return "gt_2"


def empty_band_counts() -> dict[str, int]:
    return {
        "le_0p25": 0,
        "0p25_0p5": 0,
        "0p5_1": 0,
        "1_2": 0,
        "gt_2": 0,
    }


def band_counts_for_margins(values: list[float]) -> dict[str, int]:
    counts = empty_band_counts()
    for value in values:
        counts[margin_band(value)] += 1
    return counts


def add_margin_band(row_change: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row_change)
    enriched["abs_baseline_margin"] = abs(float(enriched["baseline_margin"]))
    enriched["baseline_margin_band"] = margin_band(float(enriched["baseline_margin"]))
    return enriched


def changed_with_margin_bands(
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    arm: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [add_margin_band(row) for row in changed_row_audit(rows, baseline, arm)]


def imported_v30_flips(v30: dict[str, Any]) -> list[dict[str, Any]]:
    flips = []
    panels = v30["summary"]["panels"]
    for panel_name, panel in panels.items():
        for seed, seed_result in panel["seeds"].items():
            for arm, rows in seed_result["changed_rows"].items():
                for row in rows:
                    enriched = add_margin_band(row)
                    enriched["panel"] = panel_name
                    enriched["seed"] = int(seed)
                    enriched["arm"] = arm
                    flips.append(enriched)
    flips.sort(key=lambda row: (row["panel"], row["seed"], row["arm"], row["row_id"]))
    return flips


def score_lookup_target_write(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
) -> dict[str, Any]:
    baseline = score_rows_path(rows, tokenizer, model, batch_size)
    writes = collect_masked_attention_writes(rows, tokenizer, model, batch_size, "target_value", WRITE_LAYERS)
    raw = score_rows_write_replacement(rows, tokenizer, model, batch_size, writes, WRITE_LAYERS)
    summary = summarize_path_arm(rows, baseline, raw)
    changes = changed_with_margin_bands(rows, baseline, raw)
    target_win_losses = [row for row in changes if int(row["target_win_loss_contribution"]) == 1]
    target_win_gains = [row for row in changes if int(row["target_win_loss_contribution"]) == -1]
    return {
        "row_count": len(rows),
        "baseline": baseline_summary(rows, baseline),
        "target_write_summary": summary,
        "baseline_margin_bands": band_counts_for_margins([margin(baseline[row["id"]]) for row in rows]),
        "changed_rows": changes,
        "target_win_loss_rows": target_win_losses,
        "target_win_gain_rows": target_win_gains,
        "target_win_loss_margin_bands": band_counts_for_margins(
            [float(row["baseline_margin"]) for row in target_win_losses]
        ),
    }


def score_null_seed(
    seed: int,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
) -> dict[str, Any]:
    baseline = score_rows_path(rows, tokenizer, model, batch_size)
    base = baseline_summary(rows, baseline)
    arms = {}
    changed_rows = {}
    for source_key in NULL_SOURCE_KEYS:
        writes = collect_masked_attention_writes(rows, tokenizer, model, batch_size, source_key, WRITE_LAYERS)
        raw = score_rows_write_replacement(rows, tokenizer, model, batch_size, writes, WRITE_LAYERS)
        summary = summarize_path_arm(rows, baseline, raw)
        summary["strict_mean_delta_within_0p25"] = abs(float(summary["mean_delta"])) <= 0.25
        arms[source_key] = summary
        changed_rows[source_key] = changed_with_margin_bands(rows, baseline, raw)
    label = "clean_null" if all(int(arm["target_win_loss"]) == 0 for arm in arms.values()) else "side_effect"
    print(
        f"[v31 null seed={seed}] {label} "
        + " ".join(f"{key}={arm['mean_delta']:.4f}/{arm['target_win_loss']}" for key, arm in arms.items())
    )
    return {
        "seed": seed,
        "label": label,
        "baseline": base,
        "baseline_margin_bands": band_counts_for_margins([margin(baseline[row["id"]]) for row in rows]),
        "arms": arms,
        "changed_rows": changed_rows,
        "changed_row_count": sum(len(rows_for_arm) for rows_for_arm in changed_rows.values()),
    }


def score_fresh_null_panel(
    tokenizer: Any,
    model: Any,
    batch_size: int,
    seeds: list[int],
    row_count: int,
) -> dict[str, Any]:
    seed_results = {}
    for seed in seeds:
        rows = build_null_rows(tokenizer, seed, row_count)
        seed_results[str(seed)] = score_null_seed(seed, rows, tokenizer, model, batch_size)
    return {
        "seed_count": len(seed_results),
        "row_count": sum(int(result["baseline"]["rows"]) for result in seed_results.values()),
        "clean_seed_count": sum(1 for result in seed_results.values() if result["label"] == "clean_null"),
        "changed_row_count": sum(int(result["changed_row_count"]) for result in seed_results.values()),
        "seeds": seed_results,
    }


def flatten_fresh_flips(panel: dict[str, Any]) -> list[dict[str, Any]]:
    flips = []
    for seed, seed_result in panel["seeds"].items():
        for arm, rows in seed_result["changed_rows"].items():
            for row in rows:
                enriched = dict(row)
                enriched["panel"] = "fresh_null_margin_128"
                enriched["seed"] = int(seed)
                enriched["arm"] = arm
                flips.append(enriched)
    flips.sort(key=lambda row: (row["seed"], row["arm"], row["row_id"]))
    return flips


def all_fresh_arms(panel: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        arm
        for seed_result in panel["seeds"].values()
        for arm in seed_result["arms"].values()
    ]


def classify(criteria: dict[str, bool]) -> str:
    if not criteria["source_artifacts_valid"]:
        return "source_artifact_invalid"
    if not criteria["v31_lookup_target_effect_reproduced"]:
        return "lookup_effect_not_reproduced"
    if not criteria["v30_imported_flips_abs_margin_le_0p5"]:
        return "null_boundary_broad"
    if not criteria["v31_fresh_null_flips_abs_margin_le_0p5"]:
        return "null_boundary_broad"
    if not criteria["v31_fresh_null_mean_deltas_within_0p25"]:
        return "null_boundary_broad"
    if not criteria["lookup_loss_rows_baseline_margin_ge_2"]:
        return "lookup_effect_near_margin_confounded"
    if all(criteria.values()):
        return "margin_boundary_supported"
    return "mixed_margin_boundary"


def write_result(output_dir: Path, artifact_prefix: str, result: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = output_dir / f"{artifact_prefix}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--source-artifact", type=Path, default=DEFAULT_V25_PATH)
    parser.add_argument("--v29-artifact", type=Path, default=DEFAULT_V29_PATH)
    parser.add_argument("--v30-artifact", type=Path, default=DEFAULT_V30_PATH)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_response_marker_v31_margin_boundary")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fresh-seeds", default=",".join(str(seed) for seed in FRESH_SEEDS))
    parser.add_argument("--fresh-row-count", type=int, default=FRESH_ROW_COUNT)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.time()
    v25 = read_json(args.source_artifact)
    v29 = read_json(args.v29_artifact)
    v30 = read_json(args.v30_artifact)
    v25_hash = sha256_file(args.source_artifact)
    v29_hash = sha256_file(args.v29_artifact)
    v30_hash = sha256_file(args.v30_artifact)
    validation_errors = []
    try:
        validate_v25(v25)
        validate_v29(v29, v25_hash)
        validate_v30(v30, v25_hash, v29_hash)
    except Exception as exc:  # noqa: BLE001
        validation_errors.append(str(exc))

    if validation_errors:
        criteria = {
            "source_artifacts_valid": False,
            "v30_imported_flips_abs_margin_le_0p5": False,
            "v31_fresh_null_flips_abs_margin_le_0p5": False,
            "v31_fresh_null_mean_deltas_within_0p25": False,
            "v31_lookup_target_effect_reproduced": False,
            "lookup_loss_rows_baseline_margin_ge_2": False,
        }
        summary = {
            "model_id": args.model_id,
            "source_artifact": str(args.source_artifact),
            "source_artifact_sha256": v25_hash,
            "v29_artifact": str(args.v29_artifact),
            "v29_artifact_sha256": v29_hash,
            "v30_artifact": str(args.v30_artifact),
            "v30_artifact_sha256": v30_hash,
            "validation_errors": validation_errors,
            "criteria": criteria,
            "passed": False,
            "diagnostic_class": classify(criteria),
        }
        result = {
            "card_id": args.card_id,
            "run_type": "associative_lookup_response_marker_v31_margin_boundary",
            "model_id": args.model_id,
            "elapsed_s": time.time() - started,
            "summary": summary,
        }
        output_path = write_result(args.output_dir, args.artifact_prefix, result)
        print(json.dumps({**summary, "output_path": str(output_path)}, indent=2, ensure_ascii=True))
        return 1

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

    lookup_rows = parent_effect_holdout_rows(v25)
    lookup = score_lookup_target_write(lookup_rows, tokenizer, model, args.batch_size)
    fresh_seeds = parse_ints(args.fresh_seeds)
    fresh_null = score_fresh_null_panel(tokenizer, model, args.batch_size, fresh_seeds, args.fresh_row_count)
    v30_flips = imported_v30_flips(v30)
    v31_fresh_flips = flatten_fresh_flips(fresh_null)
    lookup_summary = lookup["target_write_summary"]
    lookup_loss_rows = lookup["target_win_loss_rows"]
    criteria = {
        "source_artifacts_valid": True,
        "v30_imported_flips_abs_margin_le_0p5": all(
            float(row["abs_baseline_margin"]) <= 0.5 for row in v30_flips
        ),
        "v31_fresh_null_flips_abs_margin_le_0p5": all(
            float(row["abs_baseline_margin"]) <= 0.5 for row in v31_fresh_flips
        ),
        "v31_fresh_null_mean_deltas_within_0p25": all(
            abs(float(arm["mean_delta"])) <= 0.25 for arm in all_fresh_arms(fresh_null)
        ),
        "v31_lookup_target_effect_reproduced": (
            float(lookup_summary["mean_delta"]) <= -5.0
            and int(lookup_summary["target_win_loss"]) >= 8
        ),
        "lookup_loss_rows_baseline_margin_ge_2": (
            bool(lookup_loss_rows)
            and all(float(row["baseline_margin"]) >= 2.0 for row in lookup_loss_rows)
        ),
    }
    diagnostic_class = classify(criteria)
    summary = {
        "model_id": args.model_id,
        "source_artifact": str(args.source_artifact),
        "source_artifact_sha256": v25_hash,
        "v29_artifact": str(args.v29_artifact),
        "v29_artifact_sha256": v29_hash,
        "v30_artifact": str(args.v30_artifact),
        "v30_artifact_sha256": v30_hash,
        "write_layers": list(WRITE_LAYERS),
        "fresh_seeds": fresh_seeds,
        "fresh_row_count": args.fresh_row_count,
        "lookup_target_write": lookup,
        "fresh_null_margin_128": fresh_null,
        "v30_imported_null_flips": v30_flips,
        "v31_fresh_null_flips": v31_fresh_flips,
        "combined_null_flip_count": len(v30_flips) + len(v31_fresh_flips),
        "combined_null_flip_margin_bands": band_counts_for_margins(
            [float(row["baseline_margin"]) for row in [*v30_flips, *v31_fresh_flips]]
        ),
        "lookup_loss_margin_bands": lookup["target_win_loss_margin_bands"],
        "criteria": criteria,
        "passed": diagnostic_class == "margin_boundary_supported",
        "diagnostic_class": diagnostic_class,
    }
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_response_marker_v31_margin_boundary",
        "model_id": args.model_id,
        "elapsed_s": time.time() - started,
        "summary": summary,
    }
    output_path = write_result(args.output_dir, args.artifact_prefix, result)
    print(json.dumps({**summary, "output_path": str(output_path)}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
