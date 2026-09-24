#!/usr/bin/env python
"""MC005 V30 answer-absent attention-write null sweep."""

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
    null_arm_clean,
    score_rows_write_replacement,
)
from mc005_associative_lookup_source_edge import CARD_ID, MODEL_ID, RESULT_DIR


DEFAULT_V29_PATH = (
    RESULT_DIR / "mc005_qwen3_1p7b_response_marker_v29_attention_write_replacement_20260630T214645.json"
)
FRESH_SEEDS = (263, 269, 271, 277, 281, 283, 293, 307)
FRESH_ROW_COUNT = 128


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def validate_v29(v29: dict[str, Any], v25_hash: str) -> None:
    if v29.get("run_type") != "associative_lookup_response_marker_v29_attention_write_replacement":
        raise ValueError(f"unexpected V29 run_type: {v29.get('run_type')!r}")
    summary = v29.get("summary", {})
    if summary.get("diagnostic_class") != "null_failed":
        raise ValueError(f"unexpected V29 diagnostic class: {summary.get('diagnostic_class')!r}")
    if summary.get("source_artifact_sha256") != v25_hash:
        raise ValueError("V29 source hash does not match V25 source artifact")
    recovery = summary.get("recovery", {})
    if float(recovery.get("target_write_delta_recovery_vs_direct", 0.0)) != 1.0:
        raise ValueError("V29 target write delta recovery is not exact")
    if float(recovery.get("target_write_win_loss_recovery_vs_direct", 0.0)) != 1.0:
        raise ValueError("V29 target write win-loss recovery is not exact")
    failure = summary["null_results"]["251"]["arms"]["non_source_control_value"]
    if int(failure.get("target_win_loss")) != -1:
        raise ValueError("V29 seed-251 non-source-control null failure sentinel missing")
    if bool(failure.get("write_null_clean")):
        raise ValueError("V29 seed-251 non-source-control null failure sentinel marked clean")


def artifact_null_rows_by_seed(v25: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    rows_by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in v25["rows"]:
        if row.get("mode") != "null":
            continue
        rows_by_seed.setdefault(int(row["seed"]), []).append(row)
    for rows in rows_by_seed.values():
        rows.sort(key=lambda row: row["id"])
    return rows_by_seed


def clean_arm(baseline_clean: bool, summary: dict[str, Any]) -> bool:
    return bool(baseline_clean and null_arm_clean(summary))


def target_win_loss_contribution(before: bool, after: bool) -> int:
    return int(before) - int(after)


def changed_row_audit(
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    arm: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    changed = []
    for row in rows:
        row_id = row["id"]
        before = baseline[row_id]
        after = arm[row_id]
        before_wins = bool(before["target_wins"])
        after_wins = bool(after["target_wins"])
        if before_wins == after_wins:
            continue
        before_margin = float(before["target_minus_distractor_margin"])
        after_margin = float(after["target_minus_distractor_margin"])
        changed.append(
            {
                "row_id": row_id,
                "seed": int(row["seed"]),
                "baseline_target_wins": before_wins,
                "arm_target_wins": after_wins,
                "target_win_loss_contribution": target_win_loss_contribution(before_wins, after_wins),
                "baseline_margin": before_margin,
                "arm_margin": after_margin,
                "margin_delta": after_margin - before_margin,
                "baseline_greedy_label": before["greedy_label"],
                "arm_greedy_label": after["greedy_label"],
                "baseline_greedy_token": before["greedy_token"],
                "arm_greedy_token": after["greedy_token"],
                "target_value": row.get("target_value"),
                "distractor_value": row.get("distractor_value"),
                "source_focus_value": row.get("source_focus_value"),
                "control_value": row.get("control_value"),
            }
        )
    return changed


def score_null_seed(
    panel: str,
    seed: int,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
) -> dict[str, Any]:
    baseline = score_rows_path(rows, tokenizer, model, batch_size)
    base = baseline_summary(rows, baseline)
    baseline_floor = int(0.75 * len(rows))
    baseline_clean = int(base["target_wins"]) >= baseline_floor
    arms = {}
    changed_rows = {}
    for source_key in NULL_SOURCE_KEYS:
        writes = collect_masked_attention_writes(rows, tokenizer, model, batch_size, source_key, WRITE_LAYERS)
        raw = score_rows_write_replacement(rows, tokenizer, model, batch_size, writes, WRITE_LAYERS)
        summary = summarize_path_arm(rows, baseline, raw)
        summary["baseline_clean"] = baseline_clean
        summary["strict_target_win_no_change"] = int(summary["target_win_loss"]) == 0
        summary["strict_mean_delta_within_0p25"] = abs(float(summary["mean_delta"])) <= 0.25
        summary["write_null_clean"] = clean_arm(baseline_clean, summary)
        arms[source_key] = summary
        changed_rows[source_key] = changed_row_audit(rows, baseline, raw)
    if not baseline_clean:
        label = "invalid_baseline"
    elif all(arm["write_null_clean"] for arm in arms.values()):
        label = "clean_null"
    else:
        label = "side_effect"
    print(
        f"[v30 {panel} seed={seed}] {label} "
        + " ".join(f"{key}={arm['mean_delta']:.4f}/{arm['target_win_loss']}" for key, arm in arms.items())
    )
    return {
        "seed": seed,
        "label": label,
        "baseline": base,
        "baseline_floor": baseline_floor,
        "baseline_clean": baseline_clean,
        "arms": arms,
        "changed_rows": changed_rows,
        "changed_row_count": sum(len(rows_for_arm) for rows_for_arm in changed_rows.values()),
    }


def score_panel(
    panel: str,
    rows_by_seed: dict[int, list[dict[str, Any]]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
) -> dict[str, Any]:
    seed_results = {}
    for seed, rows in sorted(rows_by_seed.items()):
        seed_results[str(seed)] = score_null_seed(panel, seed, rows, tokenizer, model, batch_size)
    return {
        "seed_count": len(seed_results),
        "row_count": sum(int(result["baseline"]["rows"]) for result in seed_results.values()),
        "clean_seed_count": sum(1 for result in seed_results.values() if result["label"] == "clean_null"),
        "changed_row_count": sum(int(result["changed_row_count"]) for result in seed_results.values()),
        "seeds": seed_results,
    }


def fresh_rows_by_seed(tokenizer: Any, seeds: list[int], row_count: int) -> dict[int, list[dict[str, Any]]]:
    return {seed: build_null_rows(tokenizer, seed, row_count) for seed in seeds}


def all_fresh_arms(fresh_panel: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        arm
        for seed_result in fresh_panel["seeds"].values()
        for arm in seed_result["arms"].values()
    ]


def classify(criteria: dict[str, bool]) -> str:
    if not criteria["source_artifact_valid"] or not criteria["v29_artifact_valid"]:
        return "source_artifact_invalid"
    if not criteria["v29_seed251_failure_reproduced"]:
        return "v29_replay_failed"
    if not criteria["fresh_seeds_clean"]:
        return "fresh_write_null_failed"
    if not criteria["fresh_no_large_mean_delta"]:
        return "fresh_write_null_failed"
    if not criteria["fresh_no_target_win_change"]:
        return "fresh_write_null_failed"
    if all(criteria.values()):
        return "write_null_sweep_clean"
    return "mixed_null_boundary"


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
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_response_marker_v30_write_null_sweep")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fresh-seeds", default=",".join(str(seed) for seed in FRESH_SEEDS))
    parser.add_argument("--fresh-row-count", type=int, default=FRESH_ROW_COUNT)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.time()
    v25 = read_json(args.source_artifact)
    v29 = read_json(args.v29_artifact)
    source_hash = sha256_file(args.source_artifact)
    v29_hash = sha256_file(args.v29_artifact)
    validation_errors = []
    try:
        validate_v25(v25)
    except Exception as exc:  # noqa: BLE001
        validation_errors.append(f"V25: {exc}")
    try:
        validate_v29(v29, source_hash)
    except Exception as exc:  # noqa: BLE001
        validation_errors.append(f"V29: {exc}")

    if validation_errors:
        criteria = {
            "source_artifact_valid": False,
            "v29_artifact_valid": False,
            "v29_seed251_failure_reproduced": False,
            "fresh_seeds_clean": False,
            "fresh_no_large_mean_delta": False,
            "fresh_no_target_win_change": False,
        }
        summary = {
            "model_id": args.model_id,
            "source_artifact": str(args.source_artifact),
            "source_artifact_sha256": source_hash,
            "v29_artifact": str(args.v29_artifact),
            "v29_artifact_sha256": v29_hash,
            "validation_errors": validation_errors,
            "criteria": criteria,
            "passed": False,
            "diagnostic_class": classify(criteria),
        }
        result = {
            "card_id": args.card_id,
            "run_type": "associative_lookup_response_marker_v30_write_null_sweep",
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

    replay_rows = artifact_null_rows_by_seed(v25)
    fresh_seeds = parse_ints(args.fresh_seeds)
    fresh_rows = fresh_rows_by_seed(tokenizer, fresh_seeds, args.fresh_row_count)
    panels = {
        "v25_replay_96": score_panel("v25_replay_96", replay_rows, tokenizer, model, args.batch_size),
        "fresh_128": score_panel("fresh_128", fresh_rows, tokenizer, model, args.batch_size),
    }
    replay_failure = panels["v25_replay_96"]["seeds"]["251"]["arms"]["non_source_control_value"]
    fresh_arms = all_fresh_arms(panels["fresh_128"])
    criteria = {
        "source_artifact_valid": True,
        "v29_artifact_valid": True,
        "v29_seed251_failure_reproduced": (
            int(replay_failure["target_win_loss"]) == -1
            and not bool(replay_failure["write_null_clean"])
        ),
        "fresh_seeds_clean": all(result["label"] == "clean_null" for result in panels["fresh_128"]["seeds"].values()),
        "fresh_no_large_mean_delta": all(abs(float(arm["mean_delta"])) <= 0.25 for arm in fresh_arms),
        "fresh_no_target_win_change": all(int(arm["target_win_loss"]) == 0 for arm in fresh_arms),
    }
    diagnostic_class = classify(criteria)
    summary = {
        "model_id": args.model_id,
        "source_artifact": str(args.source_artifact),
        "source_artifact_sha256": source_hash,
        "v29_artifact": str(args.v29_artifact),
        "v29_artifact_sha256": v29_hash,
        "write_layers": list(WRITE_LAYERS),
        "null_source_keys": list(NULL_SOURCE_KEYS),
        "fresh_seeds": fresh_seeds,
        "fresh_row_count": args.fresh_row_count,
        "panels": panels,
        "criteria": criteria,
        "passed": diagnostic_class == "write_null_sweep_clean",
        "diagnostic_class": diagnostic_class,
    }
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_response_marker_v30_write_null_sweep",
        "model_id": args.model_id,
        "elapsed_s": time.time() - started,
        "summary": summary,
    }
    output_path = write_result(args.output_dir, args.artifact_prefix, result)
    print(json.dumps({**summary, "output_path": str(output_path)}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
