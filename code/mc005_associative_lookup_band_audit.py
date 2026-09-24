#!/usr/bin/env python
"""MC005 V2 layer-band source-edge audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc001_qwen3_controlled_v10_head_localization import (
    install_source_mask_hook,
    set_eager_attention,
    tokenizer_padding_side,
)
from mc005_associative_lookup_source_edge import (
    CARD_ID,
    MODEL_ID,
    RESULT_DIR,
    build_rows,
    score_rows,
    summarize_arm,
)


BANDS = {
    "early_0_6": list(range(0, 7)),
    "mid_7_13": list(range(7, 14)),
    "signature_14_18": list(range(14, 19)),
    "late_19_23": list(range(19, 24)),
    "late_24_27": list(range(24, 28)),
    "late_20_26": list(range(20, 27)),
    "all_layers": list(range(0, 28)),
}

EARLIER_BANDS = {"early_0_6", "mid_7_13", "signature_14_18"}


def shifted_positions_for_batch(rows: list[dict[str, Any]], source_key: str, seq_lens: list[int], max_len: int) -> list[list[int]]:
    shifted = []
    for row, seq_len in zip(rows, seq_lens, strict=True):
        shifted.append([int(row["positions"][source_key]) + (max_len - seq_len)])
    return shifted


def score_rows_band(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    band_layers: list[int],
    source_key: str,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    head_indices = list(range(model.config.num_attention_heads))
    with tokenizer_padding_side(tokenizer, "left"):
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            inputs = tokenizer([row["rendered_prompt"] for row in batch], return_tensors="pt", padding=True).to(
                model.device
            )
            seq_lens = [int(value) for value in inputs["attention_mask"].sum(dim=1).detach().cpu().tolist()]
            max_len = int(inputs["input_ids"].shape[-1])
            shifted = shifted_positions_for_batch(batch, source_key, seq_lens, max_len)
            handles = [
                install_source_mask_hook(model, int(layer), head_indices, shifted)
                for layer in band_layers
            ]
            try:
                with torch.inference_mode():
                    out = model(**inputs, use_cache=False, logits_to_keep=1)
            finally:
                for handle in handles:
                    handle.remove()
            for index, row in enumerate(batch):
                logits = out.logits[index, -1, :].detach().float()
                target_logit = float(logits[int(row["target_token_id"])])
                distractor_logit = float(logits[int(row["distractor_token_id"])])
                margin = target_logit - distractor_logit
                features[row["id"]] = {
                    "target_logit": target_logit,
                    "distractor_logit": distractor_logit,
                    "target_minus_distractor_margin": margin,
                    "target_wins": margin > 0.0,
                }
    return features


def audit_band_source(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    baseline: dict[str, dict[str, Any]],
    batch_size: int,
    source_key: str,
) -> dict[str, Any]:
    results = {}
    for name, layers in BANDS.items():
        arm = score_rows_band(rows, tokenizer, model, batch_size, layers, source_key)
        results[name] = {
            "layers": layers,
            **summarize_arm(rows, baseline, arm),
        }
        print(
            f"[band {source_key} {name}] "
            f"mean_delta={results[name]['mean_delta']:.4f} "
            f"win_loss={results[name]['target_win_loss']}"
        )
    return results


def evaluate(
    rows: list[dict[str, Any]],
    clean_rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    discovery_target_results: dict[str, Any],
    holdout_target_results: dict[str, Any],
    holdout_distractor_results: dict[str, Any],
    holdout_random_results: dict[str, Any],
) -> dict[str, Any]:
    discovery_rows = [row for row in clean_rows if row["split"] == "discovery"]
    holdout_rows = [row for row in clean_rows if row["split"] == "holdout"]
    selected_name = min(
        discovery_target_results,
        key=lambda name: (
            discovery_target_results[name]["mean_delta"],
            -discovery_target_results[name]["target_win_loss"],
        ),
    )
    selected_target = holdout_target_results[selected_name]
    selected_distractor = holdout_distractor_results[selected_name]
    selected_random = holdout_random_results[selected_name]
    earlier_controls = {
        name: result
        for name, result in holdout_target_results.items()
        if name in EARLIER_BANDS and name != selected_name
    }
    criteria = {
        "fixed_clean_rows_match_mc005": len(clean_rows) == 41
        and len(discovery_rows) == 28
        and len(holdout_rows) == 13,
        "selected_band_is_not_all_layers": selected_name != "all_layers",
        "selected_band_target_reduces_margin_by_1_0": selected_target["mean_delta"] <= -1.0,
        "selected_band_target_flips_at_least_3": selected_target["target_win_loss"] >= 3,
        "selected_band_target_beats_distractor_by_0_50": selected_target["mean_delta"]
        <= selected_distractor["mean_delta"] - 0.50,
        "selected_band_target_beats_random_by_0_50": selected_target["mean_delta"]
        <= selected_random["mean_delta"] - 0.50,
        "selected_band_target_beats_all_earlier_bands_by_0_50": all(
            selected_target["mean_delta"] <= control["mean_delta"] - 0.50
            for control in earlier_controls.values()
        ),
    }
    behavior_target_wins = sum(1 for row in rows if baseline[row["id"]]["target_wins"])
    return {
        "model_behavior": {
            "rows": len(rows),
            "target_wins": behavior_target_wins,
            "clean_rows": len(clean_rows),
            "discovery_clean_rows": len(discovery_rows),
            "holdout_clean_rows": len(holdout_rows),
        },
        "candidate_bands": {name: layers for name, layers in BANDS.items()},
        "selected_band": selected_name,
        "selected_band_layers": BANDS[selected_name],
        "discovery_target_results": discovery_target_results,
        "holdout_target_results": holdout_target_results,
        "holdout_distractor_results": holdout_distractor_results,
        "holdout_random_results": holdout_random_results,
        "selected_holdout": {
            "target": selected_target,
            "distractor": selected_distractor,
            "random": selected_random,
            "earlier_target_controls": earlier_controls,
        },
        "criteria": criteria,
        "passed": all(criteria.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_associative_lookup_band_localization")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--row-count", type=int, default=48)
    parser.add_argument("--pair-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

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
    rows = build_rows(tokenizer, args.row_count, args.pair_count, args.seed)
    baseline = score_rows(rows, tokenizer, model, args.batch_size)
    for row in rows:
        row["baseline"] = baseline[row["id"]]
        row["clean"] = bool(baseline[row["id"]]["target_wins"])
    clean_rows = [row for row in rows if row["clean"]]
    discovery_rows = [row for row in clean_rows if row["split"] == "discovery"]
    holdout_rows = [row for row in clean_rows if row["split"] == "holdout"]
    print(
        f"[behavior] clean={len(clean_rows)}/{len(rows)} "
        f"discovery={len(discovery_rows)} holdout={len(holdout_rows)}"
    )

    discovery_target_results = audit_band_source(
        discovery_rows,
        tokenizer,
        model,
        baseline,
        args.batch_size,
        "target_value",
    )
    holdout_target_results = audit_band_source(
        holdout_rows,
        tokenizer,
        model,
        baseline,
        args.batch_size,
        "target_value",
    )
    holdout_distractor_results = audit_band_source(
        holdout_rows,
        tokenizer,
        model,
        baseline,
        args.batch_size,
        "distractor_value",
    )
    holdout_random_results = audit_band_source(
        holdout_rows,
        tokenizer,
        model,
        baseline,
        args.batch_size,
        "random_value",
    )

    summary = evaluate(
        rows,
        clean_rows,
        baseline,
        discovery_target_results,
        holdout_target_results,
        holdout_distractor_results,
        holdout_random_results,
    )
    elapsed = time.time() - started
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_band_localization",
        "model_id": args.model_id,
        "row_count": args.row_count,
        "pair_count": args.pair_count,
        "seed": args.seed,
        "elapsed_s": elapsed,
        "summary": summary,
        "rows": rows,
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
