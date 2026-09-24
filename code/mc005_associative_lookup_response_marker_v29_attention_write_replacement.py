#!/usr/bin/env python
"""MC005 V29 attention-write replacement diagnostic for layers 24-26."""

from __future__ import annotations

import argparse
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
    next_token_features,
    score_rows_path,
    summarize_path_arm,
)
from mc005_associative_lookup_response_marker_v25_factorial_row_heterogeneity import (
    PARENT,
)
from mc005_associative_lookup_response_marker_v26_internal_row_signature import (
    HOLDOUT_SEED,
    read_json,
    validate_v25,
)
from mc005_associative_lookup_response_marker_v27_signature_intervention import (
    DEFAULT_V25_PATH,
    sha256_file,
    shifted_positions,
)
from mc005_associative_lookup_source_edge import CARD_ID, MODEL_ID, RESULT_DIR


WRITE_LAYERS = (24, 25, 26)
FINAL_QUERY_POSITION = "final_colon"
LOOKUP_SOURCE_KEYS = ("target_value", "distractor_value", "random_value")
NULL_SOURCE_KEYS = ("source_value", "non_source_control_value", "final_colon")
WRITE_LAYER_SETS = {
    "write_l24_26": (24, 25, 26),
    "write_l24": (24,),
    "write_l25": (25,),
    "write_l26": (26,),
}


def rows_by_mode(v25: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    return [row for row in v25["rows"] if row.get("mode") == mode]


def parent_effect_holdout_rows(v25: dict[str, Any]) -> list[dict[str, Any]]:
    lookup = {row["id"]: row for row in rows_by_mode(v25, "lookup")}
    rows = []
    for diag in v25["row_diagnostics"]:
        if int(diag["seed"]) == HOLDOUT_SEED and bool(diag["parent_effect_row"]):
            rows.append(lookup[diag["row_id"]])
    rows.sort(key=lambda row: row["id"])
    return rows


def null_rows_by_seed(v25: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for row in rows_by_mode(v25, "null"):
        result.setdefault(int(row["seed"]), []).append(row)
    for rows in result.values():
        rows.sort(key=lambda row: row["id"])
    return result


def install_attn_capture_hook(
    model: Any,
    layer: int,
    row_ids: list[str],
    positions_by_batch: list[int],
    sink: dict[int, dict[str, np.ndarray]],
) -> Any:
    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        attn_output = output[0] if isinstance(output, tuple) else output
        for batch_index, row_id in enumerate(row_ids):
            sink.setdefault(layer, {})[row_id] = (
                attn_output[batch_index, int(positions_by_batch[batch_index]), :]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        return output

    return model.model.layers[layer].self_attn.register_forward_hook(hook)


def collect_masked_attention_writes(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    source_key: str,
    layers: tuple[int, ...] = WRITE_LAYERS,
) -> dict[int, dict[str, np.ndarray]]:
    writes: dict[int, dict[str, np.ndarray]] = {}
    heads = list(range(model.config.num_attention_heads))
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
            final_positions = shifted_positions(batch, FINAL_QUERY_POSITION, seq_lens, max_len)
            source_positions = [[position] for position in shifted_positions(batch, source_key, seq_lens, max_len)]
            handles = []
            for layer in layers:
                handles.append(install_source_mask_hook(model, int(layer), heads, source_positions))
                handles.append(
                    install_attn_capture_hook(
                        model,
                        int(layer),
                        [row["id"] for row in batch],
                        final_positions,
                        writes,
                    )
                )
            try:
                with torch.inference_mode():
                    model(**inputs, use_cache=False, logits_to_keep=1)
            finally:
                for handle in handles:
                    handle.remove()
    return writes


def install_attn_replace_hook(
    model: Any,
    layer: int,
    row_ids: list[str],
    positions_by_batch: list[int],
    replacements: dict[int, dict[str, np.ndarray]],
) -> Any:
    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, tuple):
            attn_output = output[0].clone()
            for batch_index, row_id in enumerate(row_ids):
                vector = torch.tensor(
                    replacements[layer][row_id],
                    device=attn_output.device,
                    dtype=attn_output.dtype,
                )
                attn_output[batch_index, int(positions_by_batch[batch_index]), :] = vector
            return (attn_output,) + output[1:]
        attn_output = output.clone()
        for batch_index, row_id in enumerate(row_ids):
            vector = torch.tensor(
                replacements[layer][row_id],
                device=attn_output.device,
                dtype=attn_output.dtype,
            )
            attn_output[batch_index, int(positions_by_batch[batch_index]), :] = vector
        return attn_output

    return model.model.layers[layer].self_attn.register_forward_hook(hook)


def score_rows_write_replacement(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    replacements: dict[int, dict[str, np.ndarray]] | None = None,
    replace_layers: tuple[int, ...] = WRITE_LAYERS,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
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
            if replacements is not None:
                final_positions = shifted_positions(batch, FINAL_QUERY_POSITION, seq_lens, max_len)
                row_ids = [row["id"] for row in batch]
                handles = [
                    install_attn_replace_hook(model, int(layer), row_ids, final_positions, replacements)
                    for layer in replace_layers
                ]
            try:
                with torch.inference_mode():
                    out = model(**inputs, use_cache=False, logits_to_keep=1)
            finally:
                for handle in handles:
                    handle.remove()
            for index, row in enumerate(batch):
                features[row["id"]] = next_token_features(out.logits[index, -1, :].detach().float(), row, tokenizer)
    return features


def score_lookup(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
) -> dict[str, Any]:
    baseline = score_rows_path(rows, tokenizer, model, batch_size)
    direct = {}
    replacement = {}
    for source_key in LOOKUP_SOURCE_KEYS:
        direct_raw = score_rows_path(rows, tokenizer, model, batch_size, PARENT, source_key)
        direct[source_key] = summarize_path_arm(rows, baseline, direct_raw)
        writes = collect_masked_attention_writes(rows, tokenizer, model, batch_size, source_key)
        replacement[source_key] = {}
        for name, layers in WRITE_LAYER_SETS.items():
            raw = score_rows_write_replacement(rows, tokenizer, model, batch_size, writes, layers)
            replacement[source_key][name] = summarize_path_arm(rows, baseline, raw)
            arm = replacement[source_key][name]
            print(
                f"[v29 lookup {source_key} {name}] "
                f"delta={arm['mean_delta']:.4f} loss={arm['target_win_loss']}"
            )
    return {
        "baseline": baseline_summary(rows, baseline),
        "direct_parent_source_masks": direct,
        "write_replacements": replacement,
    }


def null_arm_clean(summary: dict[str, Any]) -> bool:
    return int(summary["target_win_loss"]) == 0 and abs(float(summary["mean_delta"])) <= 0.25


def score_nulls(
    rows_by_seed: dict[int, list[dict[str, Any]]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
) -> dict[str, Any]:
    results = {}
    for seed, rows in sorted(rows_by_seed.items()):
        baseline = score_rows_path(rows, tokenizer, model, batch_size)
        arms = {}
        for source_key in NULL_SOURCE_KEYS:
            writes = collect_masked_attention_writes(rows, tokenizer, model, batch_size, source_key)
            raw = score_rows_write_replacement(rows, tokenizer, model, batch_size, writes, WRITE_LAYERS)
            summary = summarize_path_arm(rows, baseline, raw)
            summary["write_null_clean"] = null_arm_clean(summary)
            arms[source_key] = summary
        base = baseline_summary(rows, baseline)
        floor = int(0.75 * len(rows))
        label = "clean_null" if int(base["target_wins"]) >= floor and all(
            arm["write_null_clean"] for arm in arms.values()
        ) else "side_effect"
        results[str(seed)] = {
            "seed": seed,
            "label": label,
            "baseline": base,
            "baseline_floor": floor,
            "arms": arms,
        }
        print(
            f"[v29 null {seed}] {label} "
            + " ".join(f"{key}={arm['mean_delta']:.4f}/{arm['target_win_loss']}" for key, arm in arms.items())
        )
    return results


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= 1e-12:
        return None
    return abs(float(numerator)) / abs(float(denominator))


def classify(criteria: dict[str, bool]) -> str:
    if all(criteria.values()):
        return "attention_write_replacement_supported"
    if not criteria["source_artifact_valid"]:
        return "source_artifact_invalid"
    if not criteria["direct_parent_reference_valid"]:
        return "direct_parent_invalid"
    if not criteria["direct_source_controls_pass"]:
        return "direct_source_control_failed"
    if not criteria["target_write_delta_recovery_at_least_0p70"]:
        return "write_delta_recovery_failed"
    if not criteria["target_write_win_loss_recovery_at_least_0p50"]:
        return "write_win_loss_recovery_failed"
    if not criteria["write_source_controls_pass"]:
        return "write_source_control_failed"
    if not criteria["write_nulls_clean"]:
        return "null_failed"
    return "mixed_failure"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--source-artifact", type=Path, default=DEFAULT_V25_PATH)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_response_marker_v29_attention_write_replacement")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    source_hash = sha256_file(args.source_artifact)
    v25 = read_json(args.source_artifact)
    validate_v25(v25)
    lookup_rows = parent_effect_holdout_rows(v25)
    nulls = null_rows_by_seed(v25)

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
    lookup = score_lookup(lookup_rows, tokenizer, model, args.batch_size)
    null_results = score_nulls(nulls, tokenizer, model, args.batch_size)

    direct_target = lookup["direct_parent_source_masks"]["target_value"]
    direct_distractor = lookup["direct_parent_source_masks"]["distractor_value"]
    direct_random = lookup["direct_parent_source_masks"]["random_value"]
    write_target = lookup["write_replacements"]["target_value"]["write_l24_26"]
    write_distractor = lookup["write_replacements"]["distractor_value"]["write_l24_26"]
    write_random = lookup["write_replacements"]["random_value"]["write_l24_26"]
    delta_recovery = safe_ratio(float(write_target["mean_delta"]), float(direct_target["mean_delta"]))
    win_loss_recovery = safe_ratio(float(write_target["target_win_loss"]), float(direct_target["target_win_loss"]))
    criteria = {
        "source_artifact_valid": True,
        "direct_parent_reference_valid": (
            int(lookup["baseline"]["target_wins"]) >= int(0.75 * len(lookup_rows))
            and float(direct_target["mean_delta"]) <= -5.0
            and int(direct_target["target_win_loss"]) >= 8
        ),
        "direct_source_controls_pass": (
            float(direct_target["mean_delta"]) <= float(direct_distractor["mean_delta"]) - 0.50
            and float(direct_target["mean_delta"]) <= float(direct_random["mean_delta"]) - 0.50
        ),
        "target_write_delta_recovery_at_least_0p70": delta_recovery is not None and delta_recovery >= 0.70,
        "target_write_win_loss_recovery_at_least_0p50": win_loss_recovery is not None and win_loss_recovery >= 0.50,
        "write_source_controls_pass": (
            float(write_target["mean_delta"]) <= float(write_distractor["mean_delta"]) - 0.50
            and float(write_target["mean_delta"]) <= float(write_random["mean_delta"]) - 0.50
        ),
        "write_nulls_clean": all(result["label"] == "clean_null" for result in null_results.values()),
    }
    summary = {
        "model_id": args.model_id,
        "source_artifact": str(args.source_artifact),
        "source_artifact_sha256": source_hash,
        "write_layers": list(WRITE_LAYERS),
        "final_query_position": FINAL_QUERY_POSITION,
        "holdout_seed": HOLDOUT_SEED,
        "holdout_row_count": len(lookup_rows),
        "lookup": lookup,
        "null_results": null_results,
        "recovery": {
            "target_write_delta_recovery_vs_direct": delta_recovery,
            "target_write_win_loss_recovery_vs_direct": win_loss_recovery,
        },
        "criteria": criteria,
        "passed": all(criteria.values()),
        "diagnostic_class": classify(criteria),
    }
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_response_marker_v29_attention_write_replacement",
        "model_id": args.model_id,
        "elapsed_s": time.time() - started,
        "summary": summary,
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
