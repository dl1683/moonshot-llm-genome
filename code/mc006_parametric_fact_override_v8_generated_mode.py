#!/usr/bin/env python
"""MC006 V8 generated-answer same-prompt-family behavior repair."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CARD_ID = "MC006"
RUN_TYPE = "parametric_fact_override_v8_generated_mode"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/MC006")
SOURCE_ARTIFACTS = [
    (
        "results/cards/MC006/"
        "mc006_qwen3_1p7b_parametric_fact_override_v4_source_selected_hybrid_20260630T223734.json"
    ),
    (
        "results/cards/MC006/"
        "mc006_qwen3_1p7b_parametric_fact_override_v7_mode_gated_20260630T230257.json"
    ),
]


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

CONDITIONS = ("mode_real", "mode_fictional")
MODES = {
    "mode_real": ("REAL_WORLD_CAPITAL", "true_answer"),
    "mode_fictional": ("FICTIONAL_CITY_CODE", "override_answer"),
}


def render_prompt(source: dict[str, Any], requested_mode: str) -> str:
    lines = [
        "Use the requested answer mode for this country.",
        f"Country: {source['country']}",
        f"Fictional city code for this task: {source['override_capital']}",
        "Mode definitions:",
        "- REAL_WORLD_CAPITAL: answer the actual capital from real-world geography.",
        "- FICTIONAL_CITY_CODE: answer the fictional city code listed above.",
        f"Requested answer mode: {requested_mode}",
        f"Question: Which city should be returned for {source['country']}?",
        "Answer with only the city name.",
        "Answer:",
    ]
    return "\n".join(lines)


def source_records() -> list[dict[str, Any]]:
    records = []
    for source_id, split, country, true_capital, override_capital, lure_capital in SOURCES:
        source = {
            "source_id": source_id,
            "split": split,
            "country": country,
            "true_capital": true_capital,
            "override_capital": override_capital,
            "lure_capital": lure_capital,
        }
        for condition in CONDITIONS:
            requested_mode, expected_label = MODES[condition]
            records.append(
                {
                    "id": f"mc006_v8_{source_id}_{condition}",
                    "card_id": CARD_ID,
                    "run_type": RUN_TYPE,
                    "source_artifacts": SOURCE_ARTIFACTS,
                    "source_id": source_id,
                    "split": split,
                    "condition": condition,
                    "country": country,
                    "true_capital": true_capital,
                    "override_capital": override_capital,
                    "lure_capital": lure_capital,
                    "requested_mode": requested_mode,
                    "expected_label": expected_label,
                    "prompt": render_prompt(source, requested_mode),
                }
            )
    return records


def candidates(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"answer": record["true_capital"], "label": "true_answer"},
        {"answer": record["override_capital"], "label": "override_answer"},
        {"answer": record["lure_capital"], "label": "lure_answer"},
        {"answer": "UNKNOWN", "label": "unknown"},
    ]


def true_answer_prompt_leak(record: dict[str, Any]) -> bool:
    pattern = rf"\b{re.escape(record['true_capital'])}\b"
    return re.search(pattern, record["prompt"]) is not None


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def parse_generated_answer(record: dict[str, Any], generated_text: str) -> dict[str, Any]:
    stripped = generated_text.strip().lstrip("`'\" ")
    first_line = stripped.splitlines()[0] if stripped else ""
    normalized = normalize_text(first_line)
    matches = []
    for candidate in sorted(candidates(record), key=lambda row: len(row["answer"]), reverse=True):
        answer_norm = normalize_text(candidate["answer"])
        pattern = rf"^{re.escape(answer_norm)}(?=$|[\s\.,;:!\?\)\]\}}])"
        if re.search(pattern, normalized):
            matches.append(candidate)
    if len(matches) == 1:
        return {
            "parse_label": matches[0]["label"],
            "parse_answer": matches[0]["answer"],
            "parseable": True,
            "parse_rule": "strict_first_line_prefix",
            "first_line": first_line,
        }
    return {
        "parse_label": "unparsed",
        "parse_answer": None,
        "parseable": False,
        "parse_rule": "no_unique_strict_prefix",
        "first_line": first_line,
    }


def generate_answer(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]
    new_ids = output_ids[input_len:]
    generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return {
        "generated_text": generated_text,
        "generated_token_ids": [int(token_id) for token_id in new_ids.detach().cpu().tolist()],
        "generated_token_count": int(new_ids.shape[0]),
    }


def score_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    outputs = []
    for index, record in enumerate(records, start=1):
        generated = generate_answer(model, tokenizer, record["prompt"], max_new_tokens)
        parsed = parse_generated_answer(record, generated["generated_text"])
        output = {
            **record,
            "index": index,
            **generated,
            **parsed,
            "selected_label": parsed["parse_label"],
            "selected_answer": parsed["parse_answer"],
            "expected_correct": parsed["parse_label"] == record["expected_label"],
        }
        outputs.append(output)
        print(
            f"[{index:03d}/{len(records):03d}] {record['id']} mode={record['requested_mode']} "
            f"-> {output['selected_label']} {str(output['selected_answer'])!r} "
            f"expected={record['expected_label']} generated={generated['generated_text']!r}"
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
            "parseable": sum(1 for row in rows if row["parseable"]),
            "true_answer": sum(1 for row in rows if row["selected_label"] == "true_answer"),
            "override_answer": sum(1 for row in rows if row["selected_label"] == "override_answer"),
            "lure_answer": sum(1 for row in rows if row["selected_label"] == "lure_answer"),
            "unknown": sum(1 for row in rows if row["selected_label"] == "unknown"),
            "unparsed": sum(1 for row in rows if row["selected_label"] == "unparsed"),
            "requested_mode_counts": counter(rows, "requested_mode"),
        }
    return result


def source_contrasts(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in outputs:
        by_source[row["source_id"]][row["condition"]] = row
    contrasts = []
    for source_id, rows_by_condition in sorted(by_source.items()):
        clean = (
            rows_by_condition["mode_real"]["selected_label"] == "true_answer"
            and rows_by_condition["mode_fictional"]["selected_label"] == "override_answer"
        )
        base = rows_by_condition["mode_real"]
        contrasts.append(
            {
                "source_id": source_id,
                "split": base["split"],
                "country": base["country"],
                "true_capital": base["true_capital"],
                "override_capital": base["override_capital"],
                "clean_contrast": clean,
                "labels": {condition: rows_by_condition[condition]["selected_label"] for condition in CONDITIONS},
                "selected_answers": {
                    condition: rows_by_condition[condition]["selected_answer"]
                    for condition in CONDITIONS
                },
                "generated_first_lines": {
                    condition: rows_by_condition[condition]["first_line"]
                    for condition in CONDITIONS
                },
            }
        )
    return contrasts


def structural_check(records: list[dict[str, Any]]) -> dict[str, Any]:
    source_ids = {row["source_id"] for row in records}
    holdout_sources = {row["source_id"] for row in records if row["split"] == "holdout"}
    condition_counts = Counter(row["condition"] for row in records)
    mode_counts = Counter(row["requested_mode"] for row in records)
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
    prompt_leaks = [row["id"] for row in records if true_answer_prompt_leak(row)]
    criteria = {
        "exactly_22_sources": len(source_ids) == 22,
        "exactly_5_holdout_sources": len(holdout_sources) == 5,
        "exactly_44_rows": len(records) == 44,
        "two_conditions_per_source": all(count == 1 for count in source_condition_counts.values())
        and len(source_condition_counts) == 22 * len(CONDITIONS),
        "twenty_two_rows_per_condition": all(condition_counts.get(condition, 0) == 22 for condition in CONDITIONS),
        "twenty_two_rows_per_mode": all(count == 22 for count in mode_counts.values()) and len(mode_counts) == 2,
        "source_split_valid": source_split_valid,
        "no_duplicate_record_ids": not duplicate_ids,
        "no_duplicate_candidate_answers": not duplicate_candidates,
        "true_answer_not_prompt_listed": not prompt_leaks,
    }
    return {
        "criteria": criteria,
        "passed": all(criteria.values()),
        "source_count": len(source_ids),
        "holdout_source_count": len(holdout_sources),
        "row_count": len(records),
        "condition_counts": dict(sorted(condition_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "duplicate_ids": duplicate_ids,
        "duplicate_candidate_rows": duplicate_candidates,
        "prompt_leak_rows": prompt_leaks,
    }


def classify(structural: dict[str, Any], criteria: dict[str, bool]) -> str:
    if not structural["passed"]:
        if not structural["criteria"]["true_answer_not_prompt_listed"]:
            return "true_answer_prompt_leak"
        return "structural_invalid"
    if not criteria["parseable_rows_at_least_40"]:
        return "parseability_failed"
    if not criteria["mode_real_true_at_least_20"]:
        return "real_mode_failed"
    if not criteria["mode_fictional_override_at_least_20"]:
        return "fictional_mode_failed"
    if not criteria["clean_contrast_sources_at_least_18"] or not criteria["holdout_clean_contrasts_at_least_4"]:
        return "source_contrast_failed"
    if all(criteria.values()):
        return "generated_mode_v8_substrate_passed"
    return "mixed_behavior_failure"


def summarize(records: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    structural = structural_check(records)
    by_condition = condition_summary(outputs)
    contrasts = source_contrasts(outputs)
    clean_contrasts = [row for row in contrasts if row["clean_contrast"]]
    holdout_clean = [row for row in clean_contrasts if row["split"] == "holdout"]
    parseable_count = sum(1 for row in outputs if row["parseable"])
    criteria = {
        "parseable_rows_at_least_40": parseable_count >= 40,
        "mode_real_true_at_least_20": by_condition["mode_real"]["true_answer"] >= 20,
        "mode_fictional_override_at_least_20": by_condition["mode_fictional"]["override_answer"] >= 20,
        "clean_contrast_sources_at_least_18": len(clean_contrasts) >= 18,
        "holdout_clean_contrasts_at_least_4": len(holdout_clean) >= 4,
        "true_answer_not_prompt_listed": structural["criteria"]["true_answer_not_prompt_listed"],
    }
    diagnostic_class = classify(structural, criteria)
    return {
        "structural": structural,
        "by_condition": by_condition,
        "source_contrasts": contrasts,
        "parseable_count": parseable_count,
        "clean_contrast_count": len(clean_contrasts),
        "holdout_clean_contrast_count": len(holdout_clean),
        "criteria": criteria,
        "passed": diagnostic_class == "generated_mode_v8_substrate_passed",
        "diagnostic_class": diagnostic_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc006_qwen3_1p7b_parametric_fact_override_v8_generated_mode")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.time()
    records = source_records()
    structural = structural_check(records)
    if not structural["passed"]:
        raise ValueError(f"structural check failed before generation: {structural}")

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

    outputs = score_records(records, tokenizer, model, args.max_new_tokens)
    summary = summarize(records, outputs)
    result = {
        "card_id": args.card_id,
        "run_type": RUN_TYPE,
        "model_id": args.model_id,
        "source_artifacts": SOURCE_ARTIFACTS,
        "max_new_tokens": args.max_new_tokens,
        "decoding": {"do_sample": False},
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
