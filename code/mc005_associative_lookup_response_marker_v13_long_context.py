#!/usr/bin/env python
"""MC005 V13 longer-context Response-marker atlas on Qwen3-1.7B."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc001_qwen3_controlled_v10_head_localization import set_eager_attention
from mc005_associative_lookup_reliability_v4 import (
    SELECTED_BAND,
    SELECTED_LAYERS,
    score_rows_atlas,
)
from mc005_associative_lookup_response_marker_v9 import (
    MARKER_TEXT,
    SWEEP_SEEDS,
    Scenario,
    build_rows,
    evaluate_scenario,
    summarize_suite,
)
from mc005_associative_lookup_source_edge import CARD_ID, MODEL_ID, RESULT_DIR


SCENARIOS = [
    Scenario("lookup_pair12_response", "lookup", 12, ("target_value", "distractor_value", "random_value")),
    Scenario("lookup_pair16_response", "lookup", 16, ("target_value", "distractor_value", "random_value")),
    Scenario("offtarget_pair12_response", "null", 12, ("irrelevant_value", "irrelevant_key", "random_value")),
    Scenario("offtarget_pair16_response", "null", 16, ("irrelevant_value", "irrelevant_key", "random_value")),
    Scenario(
        "answer_absent_pair12_response_null",
        "null",
        12,
        ("source_value", "non_source_control_value", "earlier_neutral_colon", "final_label", "final_colon"),
    ),
    Scenario(
        "answer_absent_pair16_response_null",
        "null",
        16,
        ("source_value", "non_source_control_value", "earlier_neutral_colon", "final_label", "final_colon"),
    ),
]


def build_v13_rows(tokenizer: Any, scenario: Scenario, seed: int) -> list[dict[str, Any]]:
    rows = build_rows(tokenizer, scenario, seed, "mc005_v13")
    for row in rows:
        row["split"] = "response_marker_v13_long_context"
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_response_marker_v13_long_context")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
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
    scenario_summaries = []
    all_rows = []
    for seed in SWEEP_SEEDS:
        for scenario in SCENARIOS:
            rows = build_v13_rows(tokenizer, scenario, seed)
            all_rows.extend(rows)
            baseline = score_rows_atlas(rows, tokenizer, model, args.batch_size)
            arms = {
                arm_key: score_rows_atlas(rows, tokenizer, model, args.batch_size, arm_key)
                for arm_key in scenario.arm_keys
            }
            summary = evaluate_scenario(scenario, seed, rows, baseline, arms)
            scenario_summaries.append(summary)
            arm_bits = " ".join(
                f"{key}={summary['arms'][key]['mean_delta']:.4f}/"
                f"{summary['arms'][key]['target_win_loss']}"
                for key in scenario.arm_keys
            )
            print(
                f"[v13 seed={seed} scenario={scenario.name}] "
                f"label={summary['label']} clean={summary['baseline_clean_rows']}/"
                f"{summary['rows']} {arm_bits}"
            )

    suite_summary = summarize_suite(scenario_summaries)
    suite_summary["row_count_per_scenario_seed"] = 32
    suite_summary["pair_counts"] = [12, 16]
    suite_summary["run_scope"] = "long_context_pair12_pair16"
    elapsed = time.time() - started
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_response_marker_v13_long_context",
        "model_id": args.model_id,
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "elapsed_s": elapsed,
        "summary": suite_summary,
        "scenario_summaries": scenario_summaries,
        "rows": all_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)
    print(json.dumps({**suite_summary, "output_path": str(output_path)}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
