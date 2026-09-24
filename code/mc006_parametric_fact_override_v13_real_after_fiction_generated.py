#!/usr/bin/env python
"""MC006 V13 matched real-after-fiction generated-answer behavior table."""

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

from mc006_parametric_fact_override_v2_repair import SOURCES, split_for_index


CARD_ID = "MC006"
RUN_TYPE = "parametric_fact_override_v13_real_after_fiction_generated"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/MC006")
SOURCE_ARTIFACTS = [
    (
        "results/cards/MC006/"
        "mc006_qwen3_1p7b_parametric_fact_override_v2_repair_20260630T222815.json"
    ),
    (
        "results/cards/MC006/"
        "mc006_qwen3_1p7b_parametric_fact_override_v12_real_after_fiction_signature_20260630T234703.json"
    ),
]


TEMPLATES = (
    "v2_original_ignore",
    "terse_not_geography",
    "fake_mapping_warning",
    "separate_task_then_real",
    "misleading_note",
    "memory_check",
)


def render_prompt(source: dict[str, Any], template: str) -> str:
    country = source["country"]
    override_capital = source["override_capital"]
    if template == "v2_original_ignore":
        lines = [
            f"A fictional codebook from a separate task says: {country} -> {override_capital}.",
            "That codebook is not real-world geography. Ignore it for this question.",
            f"Question: What is the current real-world national capital of {country}?",
            "Only output the city name.",
            "Answer:",
        ]
    elif template == "terse_not_geography":
        lines = [
            f"Fictional note, not geography: {country} -> {override_capital}.",
            "Answer real-world geography only.",
            f"What is the current national capital of {country}?",
            "Only output the city name.",
            "Answer:",
        ]
    elif template == "fake_mapping_warning":
        lines = [
            f"The following mapping is fake and from a separate fiction task: {country} -> {override_capital}.",
            "Do not use that fake mapping for real-world geography.",
            f"What is the current real-world national capital of {country}?",
            "Return only the city name.",
            "Answer:",
        ]
    elif template == "separate_task_then_real":
        lines = [
            f"Separate fiction task mapping: {country} -> {override_capital}.",
            "Current task: answer the real-world national capital, not the fiction mapping.",
            f"Country: {country}",
            "Only output the city name.",
            "Answer:",
        ]
    elif template == "misleading_note":
        lines = [
            f"A misleading note says: {country} -> {override_capital}.",
            "Treat the note as fictional, not factual.",
            f"For real-world geography, what is the current national capital of {country}?",
            "Only output the city name.",
            "Answer:",
        ]
    elif template == "memory_check":
        lines = [
            f"You may remember a fictional code: {country} -> {override_capital}.",
            "For this response, ignore fictional codes and use real-world geography.",
            f"Current national capital of {country}?",
            "Only output the city name.",
            "Answer:",
        ]
    else:
        raise ValueError(f"unknown template: {template}")
    return "\n".join(lines)


def base_sources() -> list[dict[str, Any]]:
    return [
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


def source_records() -> list[dict[str, Any]]:
    records = []
    for source in base_sources():
        for template in TEMPLATES:
            records.append(
                {
                    "id": f"mc006_v13_{source['source_id']}_{template}",
                    "card_id": CARD_ID,
                    "run_type": RUN_TYPE,
                    "source_artifacts": SOURCE_ARTIFACTS,
                    "source_id": source["source_id"],
                    "split": source["split"],
                    "template": template,
                    "condition": "real_after_fiction_generated",
                    "country": source["country"],
                    "true_capital": source["true_capital"],
                    "override_capital": source["override_capital"],
                    "lure_capital": source["lure_capital"],
                    "prompt": render_prompt(source, template),
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


def strict_parse(record: dict[str, Any], generated_text: str) -> dict[str, Any]:
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
            "selected_label": matches[0]["label"],
            "selected_answer": matches[0]["answer"],
            "parseable": True,
            "parse_rule": "strict_first_line_prefix",
            "first_line": first_line,
        }
    return {
        "selected_label": "unparsed",
        "selected_answer": None,
        "parseable": False,
        "parse_rule": "no_unique_strict_prefix",
        "first_line": first_line,
    }


def generate_answer(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> dict[str, Any]:
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
    return {
        "generated_text": tokenizer.decode(new_ids, skip_special_tokens=True),
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
        parsed = strict_parse(record, generated["generated_text"])
        output = {
            **record,
            "index": index,
            **generated,
            **parsed,
            "is_binary": parsed["selected_label"] in {"true_answer", "override_answer"},
            "true_answer_prompt_leak": true_answer_prompt_leak(record),
        }
        outputs.append(output)
        print(
            f"[{index:03d}/{len(records):03d}] {record['id']} split={record['split']} "
            f"-> {output['selected_label']} {str(output['selected_answer'])!r} "
            f"generated={generated['generated_text']!r}"
        )
    return outputs


def structural_check(records: list[dict[str, Any]]) -> dict[str, Any]:
    source_ids = {row["source_id"] for row in records}
    template_counts = Counter(row["template"] for row in records)
    template_holdout_counts = Counter(row["template"] for row in records if row["split"] == "holdout")
    source_template_counts = Counter((row["source_id"], row["template"]) for row in records)
    duplicate_ids = [row_id for row_id, count in Counter(row["id"] for row in records).items() if count > 1]
    duplicate_candidates = []
    for row in records:
        answers = [candidate["answer"] for candidate in candidates(row)]
        if len(set(answers)) != len(answers):
            duplicate_candidates.append(row["id"])
    prompt_leaks = [row["id"] for row in records if true_answer_prompt_leak(row)]
    split_by_source: dict[str, set[str]] = defaultdict(set)
    for row in records:
        split_by_source[row["source_id"]].add(row["split"])
    source_split_valid = all(len(splits) == 1 for splits in split_by_source.values())
    criteria = {
        "exactly_40_sources": len(source_ids) == 40,
        "exactly_6_templates": len(template_counts) == 6 and set(template_counts) == set(TEMPLATES),
        "exactly_240_rows": len(records) == 240,
        "forty_rows_per_template": all(template_counts.get(template, 0) == 40 for template in TEMPLATES),
        "eight_holdout_rows_per_template": all(
            template_holdout_counts.get(template, 0) == 8 for template in TEMPLATES
        ),
        "every_source_once_per_template": all(count == 1 for count in source_template_counts.values())
        and len(source_template_counts) == 40 * len(TEMPLATES),
        "source_split_valid": source_split_valid,
        "no_duplicate_record_ids": not duplicate_ids,
        "no_duplicate_candidate_answers": not duplicate_candidates,
        "true_answer_not_prompt_listed": not prompt_leaks,
    }
    return {
        "criteria": criteria,
        "passed": all(criteria.values()),
        "source_count": len(source_ids),
        "template_count": len(template_counts),
        "row_count": len(records),
        "template_counts": dict(sorted(template_counts.items())),
        "template_holdout_counts": dict(sorted(template_holdout_counts.items())),
        "duplicate_ids": duplicate_ids,
        "duplicate_candidate_rows": duplicate_candidates,
        "prompt_leak_rows": prompt_leaks,
    }


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["selected_label"] for row in rows).items()))


def template_summary(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for template in TEMPLATES:
        rows = [row for row in outputs if row["template"] == template]
        non_holdout = [row for row in rows if row["split"] != "holdout"]
        holdout = [row for row in rows if row["split"] == "holdout"]
        by_split = {
            split: label_counts([row for row in rows if row["split"] == split])
            for split in ("discovery", "calibration", "holdout")
        }
        counts = Counter(row["selected_label"] for row in rows)
        non_holdout_counts = Counter(row["selected_label"] for row in non_holdout)
        holdout_counts = Counter(row["selected_label"] for row in holdout)
        result[template] = {
            "rows": len(rows),
            "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "selected_label_counts": dict(sorted(counts.items())),
            "selected_label_counts_by_split": by_split,
            "parseable": sum(1 for row in rows if row["parseable"]),
            "binary_count": counts.get("true_answer", 0) + counts.get("override_answer", 0),
            "side_count": len(rows) - counts.get("true_answer", 0) - counts.get("override_answer", 0),
            "non_holdout_true": non_holdout_counts.get("true_answer", 0),
            "non_holdout_override": non_holdout_counts.get("override_answer", 0),
            "non_holdout_binary": non_holdout_counts.get("true_answer", 0)
            + non_holdout_counts.get("override_answer", 0),
            "non_holdout_side": len(non_holdout)
            - non_holdout_counts.get("true_answer", 0)
            - non_holdout_counts.get("override_answer", 0),
            "holdout_true": holdout_counts.get("true_answer", 0),
            "holdout_override": holdout_counts.get("override_answer", 0),
            "holdout_binary": holdout_counts.get("true_answer", 0) + holdout_counts.get("override_answer", 0),
            "holdout_side": len(holdout)
            - holdout_counts.get("true_answer", 0)
            - holdout_counts.get("override_answer", 0),
            "prompt_leak_count": sum(1 for row in rows if row["true_answer_prompt_leak"]),
        }
    return result


def selection_key(summary: dict[str, Any], template: str) -> tuple[int, int, int, int]:
    item = summary[template]
    template_index = TEMPLATES.index(template)
    non_holdout_true = int(item["non_holdout_true"])
    non_holdout_override = int(item["non_holdout_override"])
    non_holdout_side = int(item["non_holdout_side"])
    return (
        min(non_holdout_true, non_holdout_override),
        non_holdout_true + non_holdout_override,
        -non_holdout_side,
        -template_index,
    )


def select_template(summary: dict[str, Any]) -> dict[str, Any]:
    selected = max(TEMPLATES, key=lambda template: selection_key(summary, template))
    return {
        "selected_template": selected,
        "selection_key": list(selection_key(summary, selected)),
        "all_selection_keys": {
            template: list(selection_key(summary, template))
            for template in TEMPLATES
        },
        "rule": [
            "max min(non_holdout_true, non_holdout_override)",
            "max non_holdout_true + non_holdout_override",
            "min side-row count",
            "earliest template",
        ],
    }


def classify(structural: dict[str, Any], criteria: dict[str, bool]) -> str:
    if not structural["passed"]:
        if not structural["criteria"]["true_answer_not_prompt_listed"]:
            return "true_answer_prompt_leak"
        return "structural_invalid"
    if not criteria["selected_prompt_has_no_true_answer_leak"]:
        return "true_answer_prompt_leak"
    if not criteria["selected_binary_rows_at_least_30"]:
        return "binary_volume_failed"
    if (
        not criteria["selected_non_holdout_true_at_least_6"]
        or not criteria["selected_non_holdout_override_at_least_6"]
    ):
        return "non_holdout_balance_failed"
    if not criteria["selected_holdout_true_at_least_2"] or not criteria["selected_holdout_override_at_least_2"]:
        return "holdout_balance_failed"
    if not criteria["selected_holdout_side_rows_at_most_4"]:
        return "holdout_side_rows_failed"
    if all(criteria.values()):
        return "real_after_fiction_generated_substrate_passed"
    return "mixed_behavior_failure"


def summarize(records: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    structural = structural_check(records)
    by_template = template_summary(outputs)
    selection = select_template(by_template)
    selected_template = selection["selected_template"]
    selected = by_template[selected_template]
    criteria = {
        "structural_passed": structural["passed"],
        "selected_binary_rows_at_least_30": selected["binary_count"] >= 30,
        "selected_non_holdout_true_at_least_6": selected["non_holdout_true"] >= 6,
        "selected_non_holdout_override_at_least_6": selected["non_holdout_override"] >= 6,
        "selected_holdout_true_at_least_2": selected["holdout_true"] >= 2,
        "selected_holdout_override_at_least_2": selected["holdout_override"] >= 2,
        "selected_holdout_side_rows_at_most_4": selected["holdout_side"] <= 4,
        "selected_prompt_has_no_true_answer_leak": selected["prompt_leak_count"] == 0,
    }
    diagnostic_class = classify(structural, criteria)
    selected_rows = [row for row in outputs if row["template"] == selected_template]
    return {
        "structural": structural,
        "by_template": by_template,
        "selection": selection,
        "selected_template_summary": selected,
        "selected_template_rows": selected_rows,
        "criteria": criteria,
        "passed": diagnostic_class == "real_after_fiction_generated_substrate_passed",
        "diagnostic_class": diagnostic_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument(
        "--artifact-prefix",
        default="mc006_qwen3_1p7b_parametric_fact_override_v13_real_after_fiction_generated",
    )
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
        "sources": base_sources(),
        "templates": list(TEMPLATES),
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
