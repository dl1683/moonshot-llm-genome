#!/usr/bin/env python
"""MC006 V6 condition-balanced behavior repair."""

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
RUN_TYPE = "parametric_fact_override_v6_condition_balanced"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/MC006")
V4_ARTIFACT = (
    "results/cards/MC006/"
    "mc006_qwen3_1p7b_parametric_fact_override_v4_source_selected_hybrid_20260630T223734.json"
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

CONDITIONS = ("balanced_real", "balanced_fictional")


def source_mapping(index: int) -> dict[str, str]:
    if index % 2 == 0:
        return {"A": "real", "B": "fictional"}
    return {"A": "fictional", "B": "real"}


def render_prompt(source: dict[str, Any], query_tag: str, mapping: dict[str, str]) -> str:
    def tag_line(tag: str) -> str:
        source_type = mapping[tag]
        if source_type == "real":
            return f"Source {tag}: REAL_WORLD_CAPITAL = {source['true_capital']}"
        if source_type == "fictional":
            return f"Source {tag}: FICTIONAL_CITY_CODE = {source['override_capital']}"
        raise ValueError(f"unknown source type: {source_type!r}")

    lines = [
        "Use this two-source answer card.",
        f"Country: {source['country']}",
        tag_line("A"),
        tag_line("B"),
        f"Query source: {query_tag}",
        f"Question: Which city should be returned for {source['country']}?",
        "Answer with only the city name.",
        "Answer:",
    ]
    return "\n".join(lines)


def source_records() -> list[dict[str, Any]]:
    records = []
    for index, (source_id, split, country, true_capital, override_capital, lure_capital) in enumerate(SOURCES):
        source = {
            "source_id": source_id,
            "split": split,
            "country": country,
            "true_capital": true_capital,
            "override_capital": override_capital,
            "lure_capital": lure_capital,
        }
        mapping = source_mapping(index)
        query_for_label = {value: tag for tag, value in mapping.items()}
        for condition in CONDITIONS:
            expected_source_type = "real" if condition == "balanced_real" else "fictional"
            expected_label = "true_answer" if expected_source_type == "real" else "override_answer"
            query_tag = query_for_label[expected_source_type]
            records.append(
                {
                    "id": f"mc006_v6_{source_id}_{condition}",
                    "card_id": CARD_ID,
                    "run_type": RUN_TYPE,
                    "source_artifact": V4_ARTIFACT,
                    "source_id": source_id,
                    "split": split,
                    "condition": condition,
                    "country": country,
                    "true_capital": true_capital,
                    "override_capital": override_capital,
                    "lure_capital": lure_capital,
                    "source_a_type": mapping["A"],
                    "source_b_type": mapping["B"],
                    "query_tag": query_tag,
                    "expected_source_type": expected_source_type,
                    "expected_label": expected_label,
                    "prompt": render_prompt(source, query_tag, mapping),
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
            f"[{index:03d}/{len(records):03d}] {record['id']} query={record['query_tag']} "
            f"-> {winner['label']} {winner['answer']!r} expected={record['expected_label']} "
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
            "query_tag_counts": counter(rows, "query_tag"),
        }
    return result


def source_contrasts(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in outputs:
        by_source[row["source_id"]][row["condition"]] = row
    contrasts = []
    for source_id, rows_by_condition in sorted(by_source.items()):
        clean = (
            rows_by_condition["balanced_real"]["selected_label"] == "true_answer"
            and rows_by_condition["balanced_fictional"]["selected_label"] == "override_answer"
        )
        base = rows_by_condition["balanced_real"]
        contrasts.append(
            {
                "source_id": source_id,
                "split": base["split"],
                "country": base["country"],
                "true_capital": base["true_capital"],
                "override_capital": base["override_capital"],
                "source_a_type": base["source_a_type"],
                "source_b_type": base["source_b_type"],
                "clean_contrast": clean,
                "labels": {
                    condition: rows_by_condition[condition]["selected_label"]
                    for condition in CONDITIONS
                },
                "query_tags": {
                    condition: rows_by_condition[condition]["query_tag"]
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
    holdout_sources = {row["source_id"] for row in records if row["split"] == "holdout"}
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
    tag_label_counts = Counter((row["query_tag"], row["expected_label"]) for row in records)
    tag_label_balance_valid = all(tag_label_counts.get((tag, label), 0) > 0 for tag in ("A", "B") for label in ("true_answer", "override_answer"))
    invalid_token_rows = []
    if outputs is not None:
        for row in outputs:
            for candidate in row["candidate_scores"]:
                if candidate["token_count"] <= 0:
                    invalid_token_rows.append({"id": row["id"], "answer": candidate["answer"]})
    criteria = {
        "exactly_22_sources": len(source_ids) == 22,
        "exactly_5_holdout_sources": len(holdout_sources) == 5,
        "exactly_44_rows": len(records) == 44,
        "two_conditions_per_source": all(count == 1 for count in source_condition_counts.values())
        and len(source_condition_counts) == 22 * len(CONDITIONS),
        "twenty_two_rows_per_condition": all(condition_counts.get(condition, 0) == 22 for condition in CONDITIONS),
        "source_split_valid": source_split_valid,
        "tag_label_balance_valid": tag_label_balance_valid,
        "no_duplicate_record_ids": not duplicate_ids,
        "no_duplicate_candidate_answers": not duplicate_candidates,
        "candidate_token_counts_valid": outputs is None or not invalid_token_rows,
    }
    return {
        "criteria": criteria,
        "passed": all(criteria.values()),
        "source_count": len(source_ids),
        "holdout_source_count": len(holdout_sources),
        "row_count": len(records),
        "condition_counts": dict(sorted(condition_counts.items())),
        "tag_label_counts": {f"{tag}_{label}": count for (tag, label), count in sorted(tag_label_counts.items())},
        "duplicate_ids": duplicate_ids,
        "duplicate_candidate_rows": duplicate_candidates,
        "invalid_token_rows": invalid_token_rows,
    }


def classify(structural: dict[str, Any], criteria: dict[str, bool]) -> str:
    if not structural["passed"]:
        return "structural_invalid"
    if not criteria["balanced_real_true_at_least_20"]:
        return "real_query_failed"
    if not criteria["balanced_fictional_override_at_least_20"]:
        return "fictional_query_failed"
    if not criteria["clean_contrast_sources_at_least_18"] or not criteria["holdout_clean_contrasts_at_least_4"]:
        return "source_contrast_failed"
    if not criteria["tag_label_balance_valid"]:
        return "tag_label_balance_failed"
    if all(criteria.values()):
        return "condition_balanced_v6_substrate_passed"
    return "mixed_behavior_failure"


def summarize(records: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    structural = structural_check(records, outputs)
    by_condition = condition_summary(outputs)
    contrasts = source_contrasts(outputs)
    clean_contrasts = [row for row in contrasts if row["clean_contrast"]]
    holdout_clean = [row for row in clean_contrasts if row["split"] == "holdout"]
    criteria = {
        "balanced_real_true_at_least_20": by_condition["balanced_real"]["true_answer"] >= 20,
        "balanced_fictional_override_at_least_20": by_condition["balanced_fictional"]["override_answer"] >= 20,
        "clean_contrast_sources_at_least_18": len(clean_contrasts) >= 18,
        "holdout_clean_contrasts_at_least_4": len(holdout_clean) >= 4,
        "tag_label_balance_valid": structural["criteria"]["tag_label_balance_valid"],
    }
    diagnostic_class = classify(structural, criteria)
    return {
        "structural": structural,
        "by_condition": by_condition,
        "source_contrasts": contrasts,
        "clean_contrast_count": len(clean_contrasts),
        "holdout_clean_contrast_count": len(holdout_clean),
        "criteria": criteria,
        "passed": diagnostic_class == "condition_balanced_v6_substrate_passed",
        "diagnostic_class": diagnostic_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc006_qwen3_1p7b_parametric_fact_override_v6_condition_balanced")
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
        "source_artifact": V4_ARTIFACT,
        "elapsed_s": time.time() - started,
        "sources": [
            {
                "source_id": source_id,
                "split": split,
                "country": country,
                "true_capital": true_capital,
                "override_capital": override_capital,
                "lure_capital": lure_capital,
                "mapping": source_mapping(index),
            }
            for index, (source_id, split, country, true_capital, override_capital, lure_capital) in enumerate(SOURCES)
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
