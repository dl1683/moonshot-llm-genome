#!/usr/bin/env python
"""MC006 V4 source-selected hybrid repair for fictional-code pressure."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mc006_parametric_fact_override_v3_source_selected as v3


CARD_ID = "MC006"
RUN_TYPE = "parametric_fact_override_v4_source_selected_hybrid"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/MC006")
V3_ARTIFACT = (
    "results/cards/MC006/"
    "mc006_qwen3_1p7b_parametric_fact_override_v3_source_selected_20260630T223516.json"
)


def render_prompt(source: dict[str, Any], condition: str) -> str:
    if condition != "fictional_code_lookup":
        return v3.render_prompt(source, condition)
    country = source["country"]
    override_capital = source["override_capital"]
    lines = [
        "You are in a fictional codebook task, not real-world geography.",
        f"Rule: for the input country \"{country}\", the output city code is \"{override_capital}\".",
        f"Using only the fictional codebook rule, what city code should be output for {country}?",
        "Only output the city name.",
        "Output:",
    ]
    return "\n".join(lines)


def source_records() -> list[dict[str, Any]]:
    records = []
    for source_id, split, country, true_capital, override_capital, lure_capital in v3.SOURCES:
        source = {
            "source_id": source_id,
            "split": split,
            "country": country,
            "true_capital": true_capital,
            "override_capital": override_capital,
            "lure_capital": lure_capital,
        }
        for condition in v3.CONDITIONS:
            records.append(
                {
                    "id": f"mc006_v4_{source_id}_{condition}",
                    "card_id": CARD_ID,
                    "run_type": RUN_TYPE,
                    "calibration_artifact": v3.CALIBRATION_ARTIFACT,
                    "v3_artifact": V3_ARTIFACT,
                    "source_id": source_id,
                    "split": split,
                    "condition": condition,
                    "country": country,
                    "true_capital": true_capital,
                    "override_capital": override_capital,
                    "lure_capital": lure_capital,
                    "expected_label": v3.EXPECTED_LABEL_BY_CONDITION[condition],
                    "prompt": render_prompt(source, condition),
                }
            )
    return records


def classify(structural: dict[str, Any], criteria: dict[str, bool]) -> str:
    if not structural["passed"]:
        return "structural_invalid"
    if not criteria["direct_real_true_at_least_20"] or not criteria["true_fact_true_at_least_22"]:
        return "parametric_fact_failed"
    if not criteria["false_claim_true_at_least_20"] or not criteria["false_claim_override_at_most_3"]:
        return "false_claim_check_failed"
    if not criteria["fictional_override_at_least_20"]:
        return "fictional_code_pressure_failed"
    if not criteria["clean_contrast_sources_at_least_18"] or not criteria["holdout_clean_contrasts_at_least_5"]:
        return "holdout_contrast_failed"
    if all(criteria.values()):
        return "source_selected_v4_substrate_passed"
    return "mixed_behavior_failure"


def summarize(records: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    structural = v3.structural_check(records, outputs)
    by_condition = v3.condition_summary(outputs)
    contrasts = v3.source_contrasts(outputs)
    clean_contrasts = [row for row in contrasts if row["clean_contrast"]]
    holdout_clean = [row for row in clean_contrasts if row["split"] == "holdout"]
    criteria = {
        "direct_real_true_at_least_20": by_condition["direct_real_paraphrase"]["true_answer"] >= 20,
        "true_fact_true_at_least_22": by_condition["true_fact_paraphrase"]["true_answer"] >= 22,
        "false_claim_true_at_least_20": by_condition["false_claim_check"]["true_answer"] >= 20,
        "false_claim_override_at_most_3": by_condition["false_claim_check"]["override_answer"] <= 3,
        "fictional_override_at_least_20": by_condition["fictional_code_lookup"]["override_answer"] >= 20,
        "clean_contrast_sources_at_least_18": len(clean_contrasts) >= 18,
        "holdout_clean_contrasts_at_least_5": len(holdout_clean) >= 5,
    }
    diagnostic_class = classify(structural, criteria)
    return {
        "structural": structural,
        "by_condition": by_condition,
        "source_contrasts": contrasts,
        "clean_contrast_count": len(clean_contrasts),
        "holdout_clean_contrast_count": len(holdout_clean),
        "criteria": criteria,
        "passed": diagnostic_class == "source_selected_v4_substrate_passed",
        "diagnostic_class": diagnostic_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument(
        "--artifact-prefix",
        default="mc006_qwen3_1p7b_parametric_fact_override_v4_source_selected_hybrid",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.time()
    records = source_records()
    structural = v3.structural_check(records)
    if not structural["passed"]:
        raise ValueError(f"structural check failed before scoring: {structural}")

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

    outputs = v3.score_records(records, tokenizer, model)
    summary = summarize(records, outputs)
    result = {
        "card_id": args.card_id,
        "run_type": RUN_TYPE,
        "model_id": args.model_id,
        "calibration_artifact": v3.CALIBRATION_ARTIFACT,
        "v3_artifact": V3_ARTIFACT,
        "elapsed_s": time.time() - started,
        "sources": [
            {
                "source_id": source_id,
                "split": split,
                "country": country,
                "true_capital": true_capital,
                "override_capital": override_capital,
                "lure_capital": lure_capital,
            }
            for source_id, split, country, true_capital, override_capital, lure_capital in v3.SOURCES
        ],
        "records": outputs,
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
