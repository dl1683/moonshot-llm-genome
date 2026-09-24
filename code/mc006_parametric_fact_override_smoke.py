#!/usr/bin/env python
"""MC006 parametric fact versus task-local override behavior smoke."""

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
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/MC006")


SOURCES = [
    ("france", "France", "Paris", "Lyon", "Marseille"),
    ("germany", "Germany", "Berlin", "Hamburg", "Munich"),
    ("italy", "Italy", "Rome", "Milan", "Naples"),
    ("spain", "Spain", "Madrid", "Barcelona", "Seville"),
    ("portugal", "Portugal", "Lisbon", "Porto", "Coimbra"),
    ("japan", "Japan", "Tokyo", "Osaka", "Kyoto"),
    ("china", "China", "Beijing", "Shanghai", "Nanjing"),
    ("canada", "Canada", "Ottawa", "Toronto", "Vancouver"),
    ("australia", "Australia", "Canberra", "Sydney", "Melbourne"),
    ("brazil", "Brazil", "Brasilia", "Rio", "Salvador"),
    ("egypt", "Egypt", "Cairo", "Alexandria", "Luxor"),
    ("kenya", "Kenya", "Nairobi", "Mombasa", "Kisumu"),
    ("ghana", "Ghana", "Accra", "Kumasi", "Tamale"),
    ("nigeria", "Nigeria", "Abuja", "Lagos", "Kano"),
    ("morocco", "Morocco", "Rabat", "Casablanca", "Marrakesh"),
    ("tunisia", "Tunisia", "Tunis", "Sfax", "Sousse"),
    ("greece", "Greece", "Athens", "Thessaloniki", "Patras"),
    ("turkey", "Turkey", "Ankara", "Istanbul", "Izmir"),
    ("norway", "Norway", "Oslo", "Bergen", "Trondheim"),
    ("sweden", "Sweden", "Stockholm", "Gothenburg", "Malmo"),
    ("finland", "Finland", "Helsinki", "Tampere", "Turku"),
    ("denmark", "Denmark", "Copenhagen", "Aarhus", "Odense"),
    ("poland", "Poland", "Warsaw", "Krakow", "Gdansk"),
    ("austria", "Austria", "Vienna", "Salzburg", "Graz"),
    ("hungary", "Hungary", "Budapest", "Debrecen", "Szeged"),
    ("romania", "Romania", "Bucharest", "Cluj", "Iasi"),
    ("bulgaria", "Bulgaria", "Sofia", "Varna", "Plovdiv"),
    ("serbia", "Serbia", "Belgrade", "Novi Sad", "Nis"),
    ("croatia", "Croatia", "Zagreb", "Split", "Rijeka"),
    ("slovenia", "Slovenia", "Ljubljana", "Maribor", "Celje"),
    ("slovakia", "Slovakia", "Bratislava", "Kosice", "Nitra"),
    ("czechia", "Czechia", "Prague", "Brno", "Ostrava"),
    ("ireland", "Ireland", "Dublin", "Cork", "Galway"),
    ("iceland", "Iceland", "Reykjavik", "Akureyri", "Keflavik"),
    ("peru", "Peru", "Lima", "Cusco", "Arequipa"),
    ("chile", "Chile", "Santiago", "Valparaiso", "Concepcion"),
    ("colombia", "Colombia", "Bogota", "Medellin", "Cali"),
    ("cuba", "Cuba", "Havana", "Santiago", "Camaguey"),
    ("thailand", "Thailand", "Bangkok", "Chiang Mai", "Phuket"),
    ("vietnam", "Vietnam", "Hanoi", "Saigon", "Hue"),
]


CONDITIONS = ("no_context", "true_context", "irrelevant_context", "task_override", "mistake_context")


def split_for_index(index: int) -> str:
    if index % 5 == 0:
        return "holdout"
    if index % 5 == 1:
        return "calibration"
    return "discovery"


def render_prompt(source: dict[str, Any], condition: str, irrelevant: dict[str, Any] | None = None) -> str:
    country = source["country"]
    true_capital = source["true_capital"]
    override_capital = source["override_capital"]
    lines: list[str] = []
    if condition == "no_context":
        lines.append("Answer the real-world factual question.")
    elif condition == "true_context":
        lines.append("Use the reference fact if it is relevant.")
        lines.append(f"Reference fact: The capital of {country} is {true_capital}.")
    elif condition == "irrelevant_context":
        if irrelevant is None:
            raise ValueError("irrelevant context requires an irrelevant source")
        lines.append("Use the reference fact if it is relevant.")
        lines.append(f"Reference fact: The capital of {irrelevant['country']} is {irrelevant['true_capital']}.")
    elif condition == "task_override":
        lines.append("For this task, use the updated mapping even if it conflicts with common knowledge.")
        lines.append(f"Updated mapping: The capital of {country} is {override_capital}.")
    elif condition == "mistake_context":
        lines.append("The following reference line may contain a mistake. Answer the real-world factual question.")
        lines.append(f"Reference line: The capital of {country} is {override_capital}.")
    else:
        raise ValueError(f"unknown condition: {condition}")
    if condition == "task_override":
        lines.append(f"Question: In this task, what is the capital of {country}?")
    else:
        lines.append(f"Question: What is the capital of {country}?")
    lines.append("Answer with only the city name.")
    lines.append("Answer:")
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
    for index, source in enumerate(source_dicts):
        irrelevant = source_dicts[(index + 11) % len(source_dicts)]
        for condition in CONDITIONS:
            expected_label = "override_answer" if condition == "task_override" else "true_answer"
            record = {
                "id": f"mc006_{source['source_id']}_{condition}",
                "card_id": CARD_ID,
                "source_id": source["source_id"],
                "split": source["split"],
                "condition": condition,
                "country": source["country"],
                "true_capital": source["true_capital"],
                "override_capital": source["override_capital"],
                "lure_capital": source["lure_capital"],
                "irrelevant_source_id": irrelevant["source_id"] if condition == "irrelevant_context" else None,
                "expected_label": expected_label,
                "prompt": render_prompt(source, condition, irrelevant),
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
            rows_by_condition["no_context"]["selected_label"] == "true_answer"
            and rows_by_condition["mistake_context"]["selected_label"] == "true_answer"
            and rows_by_condition["task_override"]["selected_label"] == "override_answer"
        )
        all_true_context_clean = (
            rows_by_condition["true_context"]["selected_label"] == "true_answer"
            and rows_by_condition["irrelevant_context"]["selected_label"] == "true_answer"
        )
        contrasts.append(
            {
                "source_id": source_id,
                "split": rows_by_condition["no_context"]["split"],
                "country": rows_by_condition["no_context"]["country"],
                "true_capital": rows_by_condition["no_context"]["true_capital"],
                "override_capital": rows_by_condition["no_context"]["override_capital"],
                "clean_contrast": clean,
                "all_true_context_clean": all_true_context_clean,
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


def structural_check(records: list[dict[str, Any]]) -> dict[str, Any]:
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
    criteria = {
        "exactly_40_sources": len(source_ids) == 40,
        "exactly_200_rows": len(records) == 200,
        "five_conditions_per_source": all(count == 1 for count in source_condition_counts.values())
        and len(source_condition_counts) == 40 * len(CONDITIONS),
        "forty_rows_per_condition": all(condition_counts.get(condition, 0) == 40 for condition in CONDITIONS),
        "source_split_valid": source_split_valid,
        "no_duplicate_record_ids": not duplicate_ids,
        "no_duplicate_candidate_answers": not duplicate_candidates,
    }
    return {
        "criteria": criteria,
        "passed": all(criteria.values()),
        "source_count": len(source_ids),
        "row_count": len(records),
        "condition_counts": dict(sorted(condition_counts.items())),
        "duplicate_ids": duplicate_ids,
        "duplicate_candidate_rows": duplicate_candidates,
    }


def classify(structural: dict[str, Any], criteria: dict[str, bool]) -> str:
    if not structural["passed"]:
        return "structural_invalid"
    if not criteria["no_context_true_at_least_30"] or not criteria["true_context_true_at_least_30"]:
        return "parametric_fact_failed"
    if not criteria["irrelevant_context_true_at_least_30"] or not criteria["mistake_context_true_at_least_30"]:
        return "context_locality_failed"
    if not criteria["task_override_override_at_least_16"]:
        return "override_pressure_failed"
    if not criteria["mistake_context_override_at_most_8"]:
        return "context_locality_failed"
    if not criteria["clean_contrast_sources_at_least_16"] or not criteria["holdout_clean_contrasts_at_least_6"]:
        return "holdout_contrast_failed"
    if all(criteria.values()):
        return "parametric_override_substrate_passed"
    return "mixed_behavior_failure"


def summarize(records: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    structural = structural_check(records)
    by_condition = condition_summary(outputs)
    contrasts = source_contrasts(outputs)
    clean_contrasts = [row for row in contrasts if row["clean_contrast"]]
    holdout_clean = [row for row in clean_contrasts if row["split"] == "holdout"]
    criteria = {
        "no_context_true_at_least_30": by_condition["no_context"]["true_answer"] >= 30,
        "true_context_true_at_least_30": by_condition["true_context"]["true_answer"] >= 30,
        "irrelevant_context_true_at_least_30": by_condition["irrelevant_context"]["true_answer"] >= 30,
        "mistake_context_true_at_least_30": by_condition["mistake_context"]["true_answer"] >= 30,
        "task_override_override_at_least_16": by_condition["task_override"]["override_answer"] >= 16,
        "mistake_context_override_at_most_8": by_condition["mistake_context"]["override_answer"] <= 8,
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
        "passed": diagnostic_class == "parametric_override_substrate_passed",
        "diagnostic_class": diagnostic_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc006_qwen3_1p7b_parametric_fact_override_smoke")
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
        "run_type": "parametric_fact_override_smoke",
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
