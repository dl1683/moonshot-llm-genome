#!/usr/bin/env python
"""MC006 V3 source-selected parametric fact versus fictional codebook smoke."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CARD_ID = "MC006"
RUN_TYPE = "parametric_fact_override_v3_source_selected"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/MC006")
CALIBRATION_ARTIFACT = (
    "results/cards/MC006/"
    "mc006_qwen3_1p7b_parametric_fact_override_v2_repair_20260630T222815.json"
)


SOURCES = [
    ("brazil", "discovery", "Brazil", "Brasilia", "Recife", "Rio"),
    ("chile", "holdout", "Chile", "Santiago", "Iquique", "Valparaiso"),
    ("china", "calibration", "China", "Beijing", "Guangzhou", "Shanghai"),
    ("croatia", "discovery", "Croatia", "Zagreb", "Zadar", "Split"),
    ("cuba", "discovery", "Cuba", "Havana", "Matanzas", "Santiago"),
    ("france", "holdout", "France", "Paris", "Lille", "Lyon"),
    ("germany", "calibration", "Germany", "Berlin", "Dresden", "Hamburg"),
    ("ghana", "discovery", "Ghana", "Accra", "Tamale", "Kumasi"),
    ("hungary", "discovery", "Hungary", "Budapest", "Szeged", "Debrecen"),
    ("iceland", "discovery", "Iceland", "Reykjavik", "Selfoss", "Akureyri"),
    ("italy", "discovery", "Italy", "Rome", "Turin", "Milan"),
    ("japan", "holdout", "Japan", "Tokyo", "Nagoya", "Osaka"),
    ("nigeria", "discovery", "Nigeria", "Abuja", "Enugu", "Lagos"),
    ("norway", "discovery", "Norway", "Oslo", "Tromso", "Bergen"),
    ("poland", "discovery", "Poland", "Warsaw", "Lodz", "Krakow"),
    ("romania", "holdout", "Romania", "Bucharest", "Timisoara", "Cluj"),
    ("serbia", "discovery", "Serbia", "Belgrade", "Subotica", "Novi Sad"),
    ("slovakia", "holdout", "Slovakia", "Bratislava", "Zilina", "Kosice"),
    ("slovenia", "discovery", "Slovenia", "Ljubljana", "Koper", "Maribor"),
    ("spain", "discovery", "Spain", "Madrid", "Valencia", "Barcelona"),
    ("thailand", "discovery", "Thailand", "Bangkok", "Pattaya", "Chiang Mai"),
    ("tunisia", "holdout", "Tunisia", "Tunis", "Sousse", "Sfax"),
    ("vietnam", "discovery", "Vietnam", "Hanoi", "Da Nang", "Saigon"),
]


CONDITIONS = (
    "direct_real_paraphrase",
    "true_fact_paraphrase",
    "false_claim_check",
    "fictional_code_lookup",
)

EXPECTED_LABEL_BY_CONDITION = {
    "direct_real_paraphrase": "true_answer",
    "true_fact_paraphrase": "true_answer",
    "false_claim_check": "true_answer",
    "fictional_code_lookup": "override_answer",
}


def render_prompt(source: dict[str, Any], condition: str) -> str:
    country = source["country"]
    true_capital = source["true_capital"]
    override_capital = source["override_capital"]
    lines: list[str] = []
    if condition == "direct_real_paraphrase":
        lines.append("Geography lookup.")
        lines.append(f"Country: {country}")
        lines.append("Task: write the official national capital of the country above.")
        lines.append("Return exactly one city name.")
        lines.append("City:")
    elif condition == "true_fact_paraphrase":
        lines.append("Verified geography table:")
        lines.append(f"{country} => {true_capital}")
        lines.append(f"Using the verified table, what is the official national capital of {country}?")
        lines.append("Return exactly one city name.")
        lines.append("City:")
    elif condition == "false_claim_check":
        lines.append("A draft geography answer may be wrong.")
        lines.append(f"Draft answer: the official national capital of {country} is {override_capital}.")
        lines.append("Check the draft and return the correct official national capital.")
        lines.append("Return exactly one city name.")
        lines.append("City:")
    elif condition == "fictional_code_lookup":
        lines.append("Fictional lookup table. These are task labels, not real geography:")
        lines.append(f"{country} => {override_capital}")
        lines.append(f"For this fictional lookup table, what city label is assigned to {country}?")
        lines.append("Return exactly one city name.")
        lines.append("Label:")
    else:
        raise ValueError(f"unknown condition: {condition}")
    return "\n".join(lines)


def source_records() -> list[dict[str, Any]]:
    source_dicts = [
        {
            "source_id": source_id,
            "split": split,
            "country": country,
            "true_capital": true_capital,
            "override_capital": override_capital,
            "lure_capital": lure_capital,
        }
        for source_id, split, country, true_capital, override_capital, lure_capital in SOURCES
    ]
    records = []
    for source in source_dicts:
        for condition in CONDITIONS:
            records.append(
                {
                    "id": f"mc006_v3_{source['source_id']}_{condition}",
                    "card_id": CARD_ID,
                    "run_type": RUN_TYPE,
                    "calibration_artifact": CALIBRATION_ARTIFACT,
                    "source_id": source["source_id"],
                    "split": source["split"],
                    "condition": condition,
                    "country": source["country"],
                    "true_capital": source["true_capital"],
                    "override_capital": source["override_capital"],
                    "lure_capital": source["lure_capital"],
                    "expected_label": EXPECTED_LABEL_BY_CONDITION[condition],
                    "prompt": render_prompt(source, condition),
                }
            )
    return records


def candidates(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"answer": record["true_capital"], "label": "true_answer", "candidate_type": "true"},
        {"answer": record["override_capital"], "label": "override_answer", "candidate_type": "override"},
        {"answer": record["lure_capital"], "label": "lure_answer", "candidate_type": "lure"},
        {"answer": "UNKNOWN", "label": "unknown", "candidate_type": "unknown"},
    ]


def token_logprob(model: Any, tokenizer: Any, rendered_prompt: str, answer: str) -> dict[str, Any]:
    full_text = f"{rendered_prompt} {answer}"
    prompt_ids = tokenizer(rendered_prompt, return_tensors="pt")["input_ids"][0]
    full_inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    full_ids = full_inputs["input_ids"][0]
    start = int(prompt_ids.shape[0])
    if start >= int(full_ids.shape[0]):
        return {
            "answer": answer,
            "token_count": 0,
            "sum_logprob": -math.inf,
            "mean_logprob": -math.inf,
        }
    with torch.inference_mode():
        logits = model(**full_inputs).logits[0]
    token_ids = full_ids[start:]
    logprobs = torch.log_softmax(logits[start - 1 : -1], dim=-1)
    selected = logprobs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
    sum_logprob = float(selected.sum().detach().cpu())
    token_count = int(token_ids.shape[0])
    return {
        "answer": answer,
        "token_count": token_count,
        "sum_logprob": sum_logprob,
        "mean_logprob": sum_logprob / token_count,
    }


def score_records(records: list[dict[str, Any]], tokenizer: Any, model: Any) -> list[dict[str, Any]]:
    outputs = []
    for index, record in enumerate(records, start=1):
        scored = []
        for candidate in candidates(record):
            scored.append({**candidate, **token_logprob(model, tokenizer, record["prompt"], candidate["answer"])})
        winner = max(scored, key=lambda row: (row["mean_logprob"], row["sum_logprob"], row["candidate_type"]))
        true_score = next(row for row in scored if row["label"] == "true_answer")
        override_score = next(row for row in scored if row["label"] == "override_answer")
        output = {
            **record,
            "index": index,
            "selected_label": winner["label"],
            "selected_answer": winner["answer"],
            "selected_candidate_type": winner["candidate_type"],
            "expected_correct": winner["label"] == record["expected_label"],
            "true_minus_override_mean_logprob": float(true_score["mean_logprob"] - override_score["mean_logprob"]),
            "candidate_scores": sorted(scored, key=lambda row: row["mean_logprob"], reverse=True),
        }
        outputs.append(output)
        print(
            f"[{index:03d}/{len(records):03d}] {record['id']} -> {winner['label']} "
            f"{winner['answer']!r} expected={record['expected_label']} "
            f"true-override={output['true_minus_override_mean_logprob']:+.4f}"
        )
    return outputs


def counter(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def condition_summary(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for condition in CONDITIONS:
        rows = [row for row in outputs if row["condition"] == condition]
        result[condition] = {
            "rows": len(rows),
            "selected_label_counts": counter(rows, "selected_label"),
            "expected_correct": sum(1 for row in rows if row["expected_correct"]),
            "true_answer": sum(1 for row in rows if row["selected_label"] == "true_answer"),
            "override_answer": sum(1 for row in rows if row["selected_label"] == "override_answer"),
            "lure_answer": sum(1 for row in rows if row["selected_label"] == "lure_answer"),
            "unknown": sum(1 for row in rows if row["selected_label"] == "unknown"),
            "mean_true_minus_override": (
                sum(float(row["true_minus_override_mean_logprob"]) for row in rows) / max(1, len(rows))
            ),
        }
    return result


def source_contrasts(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in outputs:
        by_source[row["source_id"]][row["condition"]] = row
    contrasts = []
    for source_id, rows_by_condition in sorted(by_source.items()):
        clean = (
            rows_by_condition["direct_real_paraphrase"]["selected_label"] == "true_answer"
            and rows_by_condition["false_claim_check"]["selected_label"] == "true_answer"
            and rows_by_condition["fictional_code_lookup"]["selected_label"] == "override_answer"
        )
        contrasts.append(
            {
                "source_id": source_id,
                "split": rows_by_condition["direct_real_paraphrase"]["split"],
                "country": rows_by_condition["direct_real_paraphrase"]["country"],
                "true_capital": rows_by_condition["direct_real_paraphrase"]["true_capital"],
                "override_capital": rows_by_condition["direct_real_paraphrase"]["override_capital"],
                "clean_contrast": clean,
                "true_fact_clean": rows_by_condition["true_fact_paraphrase"]["selected_label"] == "true_answer",
                "labels": {
                    condition: rows_by_condition[condition]["selected_label"]
                    for condition in CONDITIONS
                },
                "selected_answers": {
                    condition: rows_by_condition[condition]["selected_answer"]
                    for condition in CONDITIONS
                },
            }
        )
    return contrasts


def structural_check(records: list[dict[str, Any]], outputs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    source_ids = {row["source_id"] for row in records}
    condition_counts = Counter(row["condition"] for row in records)
    source_condition_counts = Counter((row["source_id"], row["condition"]) for row in records)
    duplicate_ids = [row_id for row_id, count in Counter(row["id"] for row in records).items() if count > 1]
    duplicate_candidates = []
    for row in records:
        answers = [candidate["answer"] for candidate in candidates(row)]
        if len(set(answers)) != len(answers):
            duplicate_candidates.append(row["id"])
    split_by_source: dict[str, set[str]] = defaultdict(set)
    for row in records:
        split_by_source[row["source_id"]].add(row["split"])
    source_split_valid = all(len(splits) == 1 for splits in split_by_source.values())
    holdout_sources = {row["source_id"] for row in records if row["split"] == "holdout"}
    invalid_token_rows = []
    if outputs is not None:
        for row in outputs:
            for candidate in row["candidate_scores"]:
                if candidate["token_count"] <= 0:
                    invalid_token_rows.append({"id": row["id"], "answer": candidate["answer"]})
    criteria = {
        "exactly_23_sources": len(source_ids) == 23,
        "exactly_6_holdout_sources": len(holdout_sources) == 6,
        "exactly_92_rows": len(records) == 92,
        "four_conditions_per_source": all(count == 1 for count in source_condition_counts.values())
        and len(source_condition_counts) == 23 * len(CONDITIONS),
        "twenty_three_rows_per_condition": all(
            condition_counts.get(condition, 0) == 23 for condition in CONDITIONS
        ),
        "source_split_valid": source_split_valid,
        "no_duplicate_record_ids": not duplicate_ids,
        "no_duplicate_candidate_answers": not duplicate_candidates,
        "candidate_token_counts_valid": outputs is None or not invalid_token_rows,
    }
    return {
        "criteria": criteria,
        "passed": all(criteria.values()),
        "source_count": len(source_ids),
        "row_count": len(records),
        "holdout_source_count": len(holdout_sources),
        "condition_counts": dict(sorted(condition_counts.items())),
        "duplicate_ids": duplicate_ids,
        "duplicate_candidate_rows": duplicate_candidates,
        "invalid_token_rows": invalid_token_rows,
    }


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
        return "source_selected_v3_substrate_passed"
    return "mixed_behavior_failure"


def summarize(records: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    structural = structural_check(records, outputs)
    by_condition = condition_summary(outputs)
    contrasts = source_contrasts(outputs)
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
        "passed": diagnostic_class == "source_selected_v3_substrate_passed",
        "diagnostic_class": diagnostic_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc006_qwen3_1p7b_parametric_fact_override_v3_source_selected")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.time()
    records = source_records()
    structural = structural_check(records)
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

    outputs = score_records(records, tokenizer, model)
    summary = summarize(records, outputs)
    result = {
        "card_id": args.card_id,
        "run_type": RUN_TYPE,
        "model_id": args.model_id,
        "calibration_artifact": CALIBRATION_ARTIFACT,
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
            for source_id, split, country, true_capital, override_capital, lure_capital in SOURCES
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
