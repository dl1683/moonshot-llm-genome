#!/usr/bin/env python
"""MC006 V2 parametric fact versus fictional override behavior repair."""

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
RUN_TYPE = "parametric_fact_override_v2_repair"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/MC006")


SOURCES = [
    ("france", "France", "Paris", "Lille", "Lyon"),
    ("germany", "Germany", "Berlin", "Dresden", "Hamburg"),
    ("italy", "Italy", "Rome", "Turin", "Milan"),
    ("spain", "Spain", "Madrid", "Valencia", "Barcelona"),
    ("portugal", "Portugal", "Lisbon", "Coimbra", "Porto"),
    ("japan", "Japan", "Tokyo", "Nagoya", "Osaka"),
    ("china", "China", "Beijing", "Guangzhou", "Shanghai"),
    ("canada", "Canada", "Ottawa", "Winnipeg", "Toronto"),
    ("australia", "Australia", "Canberra", "Hobart", "Sydney"),
    ("brazil", "Brazil", "Brasilia", "Recife", "Rio"),
    ("egypt", "Egypt", "Cairo", "Luxor", "Alexandria"),
    ("kenya", "Kenya", "Nairobi", "Eldoret", "Mombasa"),
    ("ghana", "Ghana", "Accra", "Tamale", "Kumasi"),
    ("nigeria", "Nigeria", "Abuja", "Enugu", "Lagos"),
    ("morocco", "Morocco", "Rabat", "Agadir", "Casablanca"),
    ("tunisia", "Tunisia", "Tunis", "Sousse", "Sfax"),
    ("greece", "Greece", "Athens", "Heraklion", "Thessaloniki"),
    ("turkey", "Turkey", "Ankara", "Bursa", "Istanbul"),
    ("norway", "Norway", "Oslo", "Tromso", "Bergen"),
    ("sweden", "Sweden", "Stockholm", "Uppsala", "Gothenburg"),
    ("finland", "Finland", "Helsinki", "Turku", "Tampere"),
    ("denmark", "Denmark", "Copenhagen", "Aalborg", "Aarhus"),
    ("poland", "Poland", "Warsaw", "Lodz", "Krakow"),
    ("austria", "Austria", "Vienna", "Innsbruck", "Salzburg"),
    ("hungary", "Hungary", "Budapest", "Szeged", "Debrecen"),
    ("romania", "Romania", "Bucharest", "Timisoara", "Cluj"),
    ("bulgaria", "Bulgaria", "Sofia", "Plovdiv", "Varna"),
    ("serbia", "Serbia", "Belgrade", "Subotica", "Novi Sad"),
    ("croatia", "Croatia", "Zagreb", "Zadar", "Split"),
    ("slovenia", "Slovenia", "Ljubljana", "Koper", "Maribor"),
    ("slovakia", "Slovakia", "Bratislava", "Zilina", "Kosice"),
    ("czechia", "Czechia", "Prague", "Plzen", "Brno"),
    ("ireland", "Ireland", "Dublin", "Limerick", "Cork"),
    ("iceland", "Iceland", "Reykjavik", "Selfoss", "Akureyri"),
    ("peru", "Peru", "Lima", "Trujillo", "Cusco"),
    ("chile", "Chile", "Santiago", "Iquique", "Valparaiso"),
    ("colombia", "Colombia", "Bogota", "Cartagena", "Medellin"),
    ("cuba", "Cuba", "Havana", "Matanzas", "Santiago"),
    ("thailand", "Thailand", "Bangkok", "Pattaya", "Chiang Mai"),
    ("vietnam", "Vietnam", "Hanoi", "Da Nang", "Saigon"),
]


CONDITIONS = (
    "direct_real",
    "true_fact",
    "false_claim_audit",
    "fictional_override",
    "real_after_fiction",
)

EXPECTED_LABEL_BY_CONDITION = {
    "direct_real": "true_answer",
    "true_fact": "true_answer",
    "false_claim_audit": "true_answer",
    "fictional_override": "override_answer",
    "real_after_fiction": "true_answer",
}


def split_for_index(index: int) -> str:
    if index % 5 == 0:
        return "holdout"
    if index % 5 == 1:
        return "calibration"
    return "discovery"


def render_prompt(source: dict[str, Any], condition: str) -> str:
    country = source["country"]
    true_capital = source["true_capital"]
    override_capital = source["override_capital"]
    lines: list[str] = []
    if condition == "direct_real":
        lines.append("Answer a real-world geography question.")
        lines.append("The answer is the current national capital, not the largest city or a tourist city.")
        lines.append(f"Question: What is the current national capital of {country}?")
        lines.append("Only output the city name.")
        lines.append("Answer:")
    elif condition == "true_fact":
        lines.append("Use this verified geography fact.")
        lines.append(f"Verified fact: The current national capital of {country} is {true_capital}.")
        lines.append(f"Question: What is the current national capital of {country}?")
        lines.append("Only output the city name.")
        lines.append("Answer:")
    elif condition == "false_claim_audit":
        lines.append("Check this claim against real-world geography.")
        lines.append(f"Claim to check: The current national capital of {country} is {override_capital}.")
        lines.append("If the claim is false, output the correct current national capital.")
        lines.append("If the claim is true, output the claimed city.")
        lines.append("Only output the city name.")
        lines.append("Correct city:")
    elif condition == "fictional_override":
        lines.append("You are in a fictional codebook task, not real-world geography.")
        lines.append(f"Rule: for the input country \"{country}\", the output city code is \"{override_capital}\".")
        lines.append(f"Using only the fictional codebook rule, what city code should be output for {country}?")
        lines.append("Only output the city name.")
        lines.append("Output:")
    elif condition == "real_after_fiction":
        lines.append(f"A fictional codebook from a separate task says: {country} -> {override_capital}.")
        lines.append("That codebook is not real-world geography. Ignore it for this question.")
        lines.append(f"Question: What is the current real-world national capital of {country}?")
        lines.append("Only output the city name.")
        lines.append("Answer:")
    else:
        raise ValueError(f"unknown condition: {condition}")
    return "\n".join(lines)


def source_records() -> list[dict[str, Any]]:
    source_dicts = [
        {
            "source_id": source_id,
            "country": country,
            "true_capital": true_capital,
            "override_capital": override_capital,
            "lure_capital": lure_capital,
            "split": split_for_index(index),
        }
        for index, (source_id, country, true_capital, override_capital, lure_capital) in enumerate(SOURCES)
    ]
    records = []
    for source in source_dicts:
        for condition in CONDITIONS:
            record = {
                "id": f"mc006_v2_{source['source_id']}_{condition}",
                "card_id": CARD_ID,
                "run_type": RUN_TYPE,
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
            records.append(record)
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
            rows_by_condition["direct_real"]["selected_label"] == "true_answer"
            and rows_by_condition["false_claim_audit"]["selected_label"] == "true_answer"
            and rows_by_condition["fictional_override"]["selected_label"] == "override_answer"
            and rows_by_condition["real_after_fiction"]["selected_label"] == "true_answer"
        )
        all_true_context_clean = rows_by_condition["true_fact"]["selected_label"] == "true_answer"
        contrasts.append(
            {
                "source_id": source_id,
                "split": rows_by_condition["direct_real"]["split"],
                "country": rows_by_condition["direct_real"]["country"],
                "true_capital": rows_by_condition["direct_real"]["true_capital"],
                "override_capital": rows_by_condition["direct_real"]["override_capital"],
                "clean_contrast": clean,
                "true_fact_clean": all_true_context_clean,
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
    invalid_token_rows = []
    if outputs is not None:
        for row in outputs:
            for candidate in row["candidate_scores"]:
                if candidate["token_count"] <= 0:
                    invalid_token_rows.append({"id": row["id"], "answer": candidate["answer"]})
    criteria = {
        "exactly_40_sources": len(source_ids) == 40,
        "exactly_200_rows": len(records) == 200,
        "five_conditions_per_source": all(count == 1 for count in source_condition_counts.values())
        and len(source_condition_counts) == 40 * len(CONDITIONS),
        "forty_rows_per_condition": all(condition_counts.get(condition, 0) == 40 for condition in CONDITIONS),
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
        "condition_counts": dict(sorted(condition_counts.items())),
        "duplicate_ids": duplicate_ids,
        "duplicate_candidate_rows": duplicate_candidates,
        "invalid_token_rows": invalid_token_rows,
    }


def classify(structural: dict[str, Any], criteria: dict[str, bool]) -> str:
    if not structural["passed"]:
        return "structural_invalid"
    if not criteria["direct_real_true_at_least_32"] or not criteria["true_fact_true_at_least_36"]:
        return "parametric_fact_failed"
    if (
        not criteria["false_claim_audit_true_at_least_30"]
        or not criteria["false_claim_audit_override_at_most_8"]
    ):
        return "false_claim_locality_failed"
    if not criteria["fictional_override_override_at_least_24"]:
        return "override_pressure_failed"
    if (
        not criteria["real_after_fiction_true_at_least_30"]
        or not criteria["real_after_fiction_override_at_most_8"]
    ):
        return "real_after_fiction_failed"
    if not criteria["clean_contrast_sources_at_least_16"] or not criteria["holdout_clean_contrasts_at_least_6"]:
        return "holdout_contrast_failed"
    if all(criteria.values()):
        return "parametric_override_v2_substrate_passed"
    return "mixed_behavior_failure"


def summarize(records: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    structural = structural_check(records, outputs)
    by_condition = condition_summary(outputs)
    contrasts = source_contrasts(outputs)
    clean_contrasts = [row for row in contrasts if row["clean_contrast"]]
    holdout_clean = [row for row in clean_contrasts if row["split"] == "holdout"]
    criteria = {
        "direct_real_true_at_least_32": by_condition["direct_real"]["true_answer"] >= 32,
        "true_fact_true_at_least_36": by_condition["true_fact"]["true_answer"] >= 36,
        "false_claim_audit_true_at_least_30": by_condition["false_claim_audit"]["true_answer"] >= 30,
        "false_claim_audit_override_at_most_8": by_condition["false_claim_audit"]["override_answer"] <= 8,
        "fictional_override_override_at_least_24": by_condition["fictional_override"]["override_answer"] >= 24,
        "real_after_fiction_true_at_least_30": by_condition["real_after_fiction"]["true_answer"] >= 30,
        "real_after_fiction_override_at_most_8": by_condition["real_after_fiction"]["override_answer"] <= 8,
        "clean_contrast_sources_at_least_16": len(clean_contrasts) >= 16,
        "holdout_clean_contrasts_at_least_6": len(holdout_clean) >= 6,
    }
    diagnostic_class = classify(structural, criteria)
    return {
        "structural": structural,
        "by_condition": by_condition,
        "source_contrasts": contrasts,
        "clean_contrast_count": len(clean_contrasts),
        "holdout_clean_contrast_count": len(holdout_clean),
        "criteria": criteria,
        "passed": diagnostic_class == "parametric_override_v2_substrate_passed",
        "diagnostic_class": diagnostic_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc006_qwen3_1p7b_parametric_fact_override_v2_repair")
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
        "elapsed_s": time.time() - started,
        "sources": [
            {
                "source_id": source_id,
                "country": country,
                "true_capital": true_capital,
                "override_capital": override_capital,
                "lure_capital": lure_capital,
                "split": split_for_index(index),
            }
            for index, (source_id, country, true_capital, override_capital, lure_capital) in enumerate(SOURCES)
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
