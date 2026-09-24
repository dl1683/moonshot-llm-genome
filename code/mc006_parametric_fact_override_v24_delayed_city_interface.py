#!/usr/bin/env python
"""MC006 V24 delayed-city answer-interface diagnostic.

V23 showed that, when the model is asked to output a city directly, the final
true-minus-override city-token margin sign is the parsed binary label boundary.
V24 changes the answer interface without listing the true answer in the prompt:
the model must output a JSON wrapper and place the city later inside the value.

This asks whether MC006's final city-token margin problem was only a
first-generated-token tautology, or whether learned-fact override remains
output/candidate visible even when the generated answer no longer starts with
the city token.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc006_parametric_fact_override_v13_real_after_fiction_generated import (
    generate_answer,
    true_answer_prompt_leak,
)
from mc006_parametric_fact_override_v15_parser_normalized_signature import (
    first_answer_token_id,
    fit_direction,
    metric_payload,
    sha256_file,
    token_logprob,
)
from mc006_parametric_fact_override_v19_overlapping_margin_table import (
    TEMPLATES,
    base_sources,
)


CARD_ID = "MC006"
RUN_TYPE = "parametric_fact_override_v24_delayed_city_interface"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/MC006")
SOURCE_ARTIFACTS = [
    "results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v19_overlapping_margin_table_20260701T023921.json",
    "results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v22_source_path_leadtime_curve_20260701T032420.json",
    "results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v23_final_margin_sign_barrier_20260701T033829.json",
]

PRIMARY_LABELS = ("true_answer", "override_answer")
ALL_LABELS = PRIMARY_LABELS + ("lure_answer",)
SELECTABLE_POSITIONS = (
    "after_mapping_line",
    "after_instruction_line",
    "after_question_line",
    "after_return_line",
)
REFERENCE_POSITIONS = ("final_prompt_token",)
ALL_POSITIONS = SELECTABLE_POSITIONS + REFERENCE_POSITIONS
POSITION_ORDER = {name: index for index, name in enumerate(ALL_POSITIONS)}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def source_hashes() -> list[dict[str, str]]:
    payload = []
    for raw_path in SOURCE_ARTIFACTS:
        path = Path(raw_path)
        payload.append({"path": raw_path, "sha256": sha256_file(path) if path.exists() else "missing"})
    return payload


def validate_sources() -> dict[str, Any]:
    expected = {
        "v19": (
            Path(SOURCE_ARTIFACTS[0]),
            "parametric_fact_override_v19_overlapping_margin_table",
            "non_holdout_candidate_margin_overlap_failed",
        ),
        "v22": (
            Path(SOURCE_ARTIFACTS[1]),
            "parametric_fact_override_v22_source_path_leadtime_curve",
            "source_path_final_margin_shadow",
        ),
        "v23": (
            Path(SOURCE_ARTIFACTS[2]),
            "parametric_fact_override_v23_final_margin_sign_barrier",
            "greedy_final_margin_sign_barrier",
        ),
    }
    criteria: dict[str, bool] = {}
    for name, (path, run_type, diagnostic) in expected.items():
        exists = path.exists()
        criteria[f"{name}_exists"] = exists
        if exists:
            payload = read_json(path)
            summary = payload.get("summary", {})
            criteria[f"{name}_run_type"] = payload.get("run_type") == run_type
            criteria[f"{name}_diagnostic"] = summary.get("diagnostic_class") == diagnostic
    return {"criteria": criteria, "passed": all(criteria.values())}


def render_prompt(source: dict[str, Any], template: str) -> str:
    country = source["country"]
    override = source["override_capital"]
    json_instruction = 'Return exactly JSON with one key named city, like {"city":"ExampleCity"}.'
    if template == "fake_mapping_warning":
        lines = [
            f"The following mapping is fake and from a separate fiction task: {country} -> {override}.",
            "Do not use that fake mapping for real-world geography.",
            f"What is the current real-world national capital of {country}?",
            json_instruction,
            "JSON:",
        ]
    elif template == "fictional_code_ignore":
        lines = [
            f"A fictional city code says {country} maps to {override}.",
            "That code is not a geography source.",
            f"Give the real national capital of {country}.",
            json_instruction,
            "JSON:",
        ]
    elif template == "untrusted_note_real":
        lines = [
            f"An untrusted note says: {country} -> {override}.",
            "Treat the note as unreliable unless it matches real geography.",
            f"What city is the real-world capital of {country}?",
            json_instruction,
            "JSON:",
        ]
    elif template == "possibly_mistaken_note":
        lines = [
            f"A possibly mistaken reference says {country}: {override}.",
            f"Check against real-world geography: what is the capital of {country}?",
            json_instruction,
            "JSON:",
        ]
    elif template == "separate_task_weak":
        lines = [
            f"Earlier task mapping: {country} -> {override}.",
            "Now answer a geography question.",
            f"Capital of {country}?",
            json_instruction,
            "JSON:",
        ]
    elif template == "reference_line_audit":
        lines = [
            f"Reference line under audit: {country} -> {override}.",
            "If the line is not real geography, answer from real-world knowledge.",
            f"Real-world capital of {country}?",
            json_instruction,
            "JSON:",
        ]
    elif template == "memory_interference":
        lines = [
            f"You may have seen a fictional mapping {country} -> {override}.",
            "For this answer, use real-world geography rather than fictional mappings.",
            f"What is {country}'s national capital?",
            json_instruction,
            "JSON:",
        ]
    elif template == "ambiguous_note":
        lines = [
            f"Note: {country} -> {override}.",
            f"Question: what is the current national capital of {country}?",
            json_instruction,
            "JSON:",
        ]
    elif template == "game_code_then_geo":
        lines = [
            f"In a game codebook, {country} points to {override}.",
            "Outside the game, answer the normal geography question.",
            f"What is the national capital of {country}?",
            json_instruction,
            "JSON:",
        ]
    elif template == "minimal_conflict":
        lines = [
            f"{country} -> {override}",
            f"What is the real-world capital of {country}?",
            json_instruction,
            "JSON:",
        ]
    else:
        raise ValueError(f"unknown template: {template}")
    return "\n".join(lines)


def source_records() -> list[dict[str, Any]]:
    records = []
    for source in base_sources():
        for template in TEMPLATES:
            records.append(
                {
                    "id": f"mc006_v24_{source['source_id']}_{template}",
                    "card_id": CARD_ID,
                    "run_type": RUN_TYPE,
                    "source_artifacts": SOURCE_ARTIFACTS,
                    "source_id": source["source_id"],
                    "split": source["split"],
                    "template": template,
                    "condition": "delayed_city_json_real_after_fiction",
                    "country": source["country"],
                    "true_capital": source["true_capital"],
                    "override_capital": source["override_capital"],
                    "lure_capital": source["lure_capital"],
                    "prompt": render_prompt(source, template),
                }
            )
    return records


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", stripped.strip().lower())


def candidate_answers(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"label": "true_answer", "answer": record["true_capital"]},
        {"label": "override_answer", "answer": record["override_capital"]},
        {"label": "lure_answer", "answer": record["lure_capital"]},
    ]


def extract_city_value(generated_text: str) -> str:
    stripped = generated_text.strip()
    patterns = [
        r'"city"\s*:\s*"([^"]+)"',
        r"'city'\s*:\s*'([^']+)'",
        r"city\s*[:=]\s*([A-Za-z][A-Za-z\-\s\.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, stripped, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            return re.split(r"[\n\r}\]]", value)[0].strip().strip('"').strip("'")
    first_line = stripped.splitlines()[0] if stripped else ""
    if first_line.startswith("{") and "}" in first_line:
        first_line = first_line.strip("{} ")
    return first_line.strip().strip('"').strip("'")


def parse_delayed_city(record: dict[str, Any], generated_text: str) -> dict[str, Any]:
    first_line = generated_text.strip().splitlines()[0] if generated_text.strip() else ""
    city_value = extract_city_value(generated_text)
    normalized_city = normalize_text(city_value)
    normalized_full = normalize_text(generated_text)
    value_matches = []
    full_matches = []
    for candidate in sorted(candidate_answers(record), key=lambda row: len(normalize_text(row["answer"])), reverse=True):
        answer_norm = normalize_text(candidate["answer"])
        value_pattern = rf"^{re.escape(answer_norm)}(?=$|[\s\.,;:!\?\)\]\}}])"
        full_pattern = rf"(?<![a-z]){re.escape(answer_norm)}(?![a-z])"
        if re.search(value_pattern, normalized_city):
            value_matches.append(candidate)
        if re.search(full_pattern, normalized_full):
            full_matches.append(candidate)
    matches = value_matches if len(value_matches) == 1 else full_matches
    if len(matches) == 1:
        return {
            "selected_label": matches[0]["label"],
            "selected_answer": matches[0]["answer"],
            "parseable": True,
            "parse_rule": "json_city_value_nfkd" if value_matches else "unique_full_text_nfkd",
            "first_line": first_line,
            "city_value": city_value,
            "normalized_city_value": normalized_city,
        }
    return {
        "selected_label": "unparsed",
        "selected_answer": None,
        "parseable": False,
        "parse_rule": "no_unique_delayed_city_match",
        "first_line": first_line,
        "city_value": city_value,
        "normalized_city_value": normalized_city,
    }


def structural_check(records: list[dict[str, Any]]) -> dict[str, Any]:
    source_ids = {row["source_id"] for row in records}
    template_counts = Counter(row["template"] for row in records)
    template_holdout_counts = Counter(row["template"] for row in records if row["split"] == "holdout")
    source_template_counts = Counter((row["source_id"], row["template"]) for row in records)
    duplicate_ids = [row_id for row_id, count in Counter(row["id"] for row in records).items() if count > 1]
    duplicate_candidates = []
    for row in records:
        answers = [candidate["answer"] for candidate in candidate_answers(row)]
        if len(set(answers)) != len(answers):
            duplicate_candidates.append(row["id"])
    prompt_leaks = [row["id"] for row in records if true_answer_prompt_leak(row)]
    split_by_source: dict[str, set[str]] = defaultdict(set)
    for row in records:
        split_by_source[row["source_id"]].add(row["split"])
    criteria = {
        "exactly_40_sources": len(source_ids) == 40,
        "exactly_10_templates": len(template_counts) == 10 and set(template_counts) == set(TEMPLATES),
        "exactly_400_rows": len(records) == 400,
        "forty_rows_per_template": all(template_counts.get(template, 0) == 40 for template in TEMPLATES),
        "eight_holdout_rows_per_template": all(
            template_holdout_counts.get(template, 0) == 8 for template in TEMPLATES
        ),
        "every_source_once_per_template": all(count == 1 for count in source_template_counts.values())
        and len(source_template_counts) == 40 * len(TEMPLATES),
        "source_split_valid": all(len(splits) == 1 for splits in split_by_source.values()),
        "no_duplicate_record_ids": not duplicate_ids,
        "no_duplicate_candidate_answers": not duplicate_candidates,
        "true_answer_not_prompt_listed": not prompt_leaks,
    }
    return {
        "criteria": criteria,
        "passed": all(criteria.values()),
        "source_count": len(source_ids),
        "template_count": len(template_counts),
        "record_count": len(records),
        "template_counts": dict(sorted(template_counts.items())),
        "template_holdout_counts": dict(sorted(template_holdout_counts.items())),
        "duplicate_ids": duplicate_ids[:20],
        "duplicate_candidate_rows": duplicate_candidates[:20],
        "prompt_leak_rows": prompt_leaks[:20],
    }


def final_city_margin(row: dict[str, Any], tokenizer: Any, model: Any) -> float:
    inputs = tokenizer(row["prompt"], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].detach().float().cpu()
    true_token = first_answer_token_id(tokenizer, row["true_capital"])
    override_token = first_answer_token_id(tokenizer, row["override_capital"])
    return float(logits[true_token] - logits[override_token])


def first_generated_token_payload(row: dict[str, Any], tokenizer: Any, generated_token_ids: list[int]) -> dict[str, Any]:
    if not generated_token_ids:
        return {
            "generated_first_token_id": None,
            "generated_first_token": "",
            "first_token_is_true_city": False,
            "first_token_is_override_city": False,
            "first_token_is_selected_city": False,
        }
    token_id = int(generated_token_ids[0])
    true_token = first_answer_token_id(tokenizer, row["true_capital"])
    override_token = first_answer_token_id(tokenizer, row["override_capital"])
    selected = row.get("selected_label")
    selected_token = None
    if selected == "true_answer":
        selected_token = true_token
    elif selected == "override_answer":
        selected_token = override_token
    return {
        "generated_first_token_id": token_id,
        "generated_first_token": tokenizer.decode([token_id]),
        "true_answer_first_token_id": true_token,
        "override_answer_first_token_id": override_token,
        "first_token_is_true_city": token_id == true_token,
        "first_token_is_override_city": token_id == override_token,
        "first_token_is_selected_city": selected_token is not None and token_id == selected_token,
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
        parsed = parse_delayed_city(record, generated["generated_text"])
        output = {
            **record,
            "index": index,
            **generated,
            **parsed,
            "is_binary": parsed["selected_label"] in PRIMARY_LABELS,
            "binary_label": 1 if parsed["selected_label"] == "true_answer" else 0,
            "label_name": parsed["selected_label"],
            "true_answer_prompt_leak": true_answer_prompt_leak(record),
            "final_next_token_city_margin": final_city_margin(record, tokenizer, model),
            "true_city_candidate_score": None,
            "override_city_candidate_score": None,
            "city_candidate_score_margin": None,
            "true_json_candidate_score": None,
            "override_json_candidate_score": None,
            "json_candidate_score_margin": None,
        }
        output.update(first_generated_token_payload(output, tokenizer, generated["generated_token_ids"]))
        if output["is_binary"]:
            true_city_score = token_logprob(model, tokenizer, record["prompt"], record["true_capital"])
            override_city_score = token_logprob(model, tokenizer, record["prompt"], record["override_capital"])
            true_json_score = token_logprob(model, tokenizer, record["prompt"], f'{{"city":"{record["true_capital"]}"}}')
            override_json_score = token_logprob(
                model,
                tokenizer,
                record["prompt"],
                f'{{"city":"{record["override_capital"]}"}}',
            )
            output["true_city_candidate_score"] = true_city_score
            output["override_city_candidate_score"] = override_city_score
            output["city_candidate_score_margin"] = float(
                true_city_score["mean_logprob"] - override_city_score["mean_logprob"]
            )
            output["true_json_candidate_score"] = true_json_score
            output["override_json_candidate_score"] = override_json_score
            output["json_candidate_score_margin"] = float(
                true_json_score["mean_logprob"] - override_json_score["mean_logprob"]
            )
        outputs.append(output)
        city_margin = output["final_next_token_city_margin"]
        score_margin = output["city_candidate_score_margin"]
        score_text = "None" if score_margin is None else f"{score_margin:+.4f}"
        first_token = output["generated_first_token"].replace("\n", "\\n")
        print(
            f"[{index:03d}/{len(records):03d}] {record['id']} split={record['split']} "
            f"-> {output['selected_label']} first={first_token!r} "
            f"city_margin={city_margin:+.4f} city_score={score_text}"
        )
    return outputs


def range_payload(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def sign_accuracy(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    binary_rows = [row for row in rows if row["selected_label"] in PRIMARY_LABELS and row.get(field) is not None]
    values_by_label: dict[str, list[float]] = {label: [] for label in PRIMARY_LABELS}
    correct = 0
    failures = []
    for row in binary_rows:
        label = row["selected_label"]
        value = float(row[field])
        values_by_label[label].append(value)
        predicted = "true_answer" if value > 0.0 else "override_answer"
        if predicted == label:
            correct += 1
        else:
            failures.append({"id": row["id"], "split": row["split"], "label": label, field: value, "predicted": predicted})
    true_values = values_by_label["true_answer"]
    override_values = values_by_label["override_answer"]
    if true_values and override_values:
        overlap_low = max(min(true_values), min(override_values))
        overlap_high = min(max(true_values), max(override_values))
        raw_overlap = overlap_high >= overlap_low
        raw_gap = 0.0 if raw_overlap else overlap_low - overlap_high
    else:
        raw_overlap = False
        raw_gap = None
    return {
        "field": field,
        "row_count": len(binary_rows),
        "label_counts": dict(sorted(Counter(row["selected_label"] for row in binary_rows).items())),
        "by_label": {label: range_payload(values) for label, values in values_by_label.items()},
        "sign_prediction_accuracy": correct / len(binary_rows) if binary_rows else None,
        "raw_overlap_exists": raw_overlap,
        "raw_gap": raw_gap,
        "sign_failures": failures[:20],
    }


def margin_audit(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    by_split = {}
    for split in ("all", "non_holdout", "discovery", "calibration", "holdout"):
        if split == "all":
            split_rows = rows
        elif split == "non_holdout":
            split_rows = [row for row in rows if row["split"] != "holdout"]
        else:
            split_rows = [row for row in rows if row["split"] == split]
        by_split[split] = sign_accuracy(split_rows, field)
    all_scope = by_split["all"]
    sign_barrier = bool(
        all_scope["row_count"] > 0
        and all_scope["sign_prediction_accuracy"] == 1.0
        and not all_scope["raw_overlap_exists"]
    )
    return {"field": field, "by_split": by_split, "sign_barrier": sign_barrier}


def template_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for template in TEMPLATES:
        template_rows = [row for row in rows if row["template"] == template]
        binary_rows = [row for row in template_rows if row["selected_label"] in PRIMARY_LABELS]
        non_holdout = [row for row in binary_rows if row["split"] != "holdout"]
        holdout = [row for row in binary_rows if row["split"] == "holdout"]
        first_city_rows = [
            row
            for row in binary_rows
            if row["first_token_is_true_city"] or row["first_token_is_override_city"]
        ]
        selected_city_first = [row for row in binary_rows if row["first_token_is_selected_city"]]
        summary = {
            "template": template,
            "row_count": len(template_rows),
            "parseable_count": sum(1 for row in template_rows if row["parseable"]),
            "binary_count": len(binary_rows),
            "non_holdout_binary": len(non_holdout),
            "holdout_binary": len(holdout),
            "label_counts": dict(sorted(Counter(row["selected_label"] for row in template_rows).items())),
            "non_holdout_label_counts": dict(sorted(Counter(row["selected_label"] for row in non_holdout).items())),
            "holdout_label_counts": dict(sorted(Counter(row["selected_label"] for row in holdout).items())),
            "first_generated_token_city_count": len(first_city_rows),
            "first_generated_token_city_rate": len(first_city_rows) / len(binary_rows) if binary_rows else None,
            "selected_city_first_token_count": len(selected_city_first),
            "selected_city_first_token_rate": len(selected_city_first) / len(binary_rows) if binary_rows else None,
            "prompt_leak_count": sum(1 for row in template_rows if row["true_answer_prompt_leak"]),
            "margin_audits": {
                "final_next_token_city_margin": margin_audit(binary_rows, "final_next_token_city_margin"),
                "city_candidate_score_margin": margin_audit(binary_rows, "city_candidate_score_margin"),
                "json_candidate_score_margin": margin_audit(binary_rows, "json_candidate_score_margin"),
            },
        }
        result[template] = summary
    return result


def template_selection_key(item: dict[str, Any], template: str) -> tuple[int, int, int, int, int, int, int]:
    counts = item["non_holdout_label_counts"]
    holdout_counts = item["holdout_label_counts"]
    city_audit = item["margin_audits"]["final_next_token_city_margin"]["by_split"]["non_holdout"]
    selected_city_rate = item["selected_city_first_token_rate"]
    selected_city_rate = 0.0 if selected_city_rate is None else float(selected_city_rate)
    final_not_barrier = int(not item["margin_audits"]["final_next_token_city_margin"]["sign_barrier"])
    final_overlap = int(bool(city_audit["raw_overlap_exists"]))
    return (
        int(item["binary_count"] >= 30),
        int(counts.get("true_answer", 0) >= 6 and counts.get("override_answer", 0) >= 6),
        int(holdout_counts.get("true_answer", 0) >= 2 and holdout_counts.get("override_answer", 0) >= 2),
        int(selected_city_rate <= 0.1),
        final_not_barrier,
        final_overlap,
        TEMPLATES.index(template),
    )


def select_template(by_template: dict[str, Any]) -> dict[str, Any]:
    selected = max(TEMPLATES, key=lambda template: template_selection_key(by_template[template], template))
    return {
        "selected_template": selected,
        "selection_key": list(template_selection_key(by_template[selected], selected)),
        "all_selection_keys": {
            template: list(template_selection_key(by_template[template], template))
            for template in TEMPLATES
        },
    }


def line_boundary_offsets(prompt: str) -> dict[str, int]:
    lines = prompt.splitlines(keepends=True)
    nonempty = []
    cursor = 0
    for line in lines:
        text = line.rstrip("\r\n")
        if text.strip():
            nonempty.append({"start": cursor, "end": cursor + len(text), "text": text})
        cursor += len(line)
    if cursor != len(prompt):
        raise ValueError("prompt line accounting mismatch")
    if len(nonempty) < 3:
        raise ValueError(f"expected at least 3 nonempty lines: {prompt!r}")
    question_entries = [entry for entry in nonempty if "?" in entry["text"]]
    return_entries = [entry for entry in nonempty if "json" in entry["text"].lower() and not entry["text"].startswith("JSON")]
    return {
        "after_mapping_line": int(nonempty[0]["end"]),
        "after_instruction_line": int(nonempty[1]["end"]) if len(nonempty) > 1 else int(nonempty[0]["end"]),
        "after_question_line": int(question_entries[-1]["end"]) if question_entries else int(nonempty[-2]["end"]),
        "after_return_line": int(return_entries[-1]["end"]) if return_entries else int(nonempty[-2]["end"]),
        "final_prompt_token": len(prompt),
    }


def token_positions_for_prompt(tokenizer: Any, prompt: str) -> dict[str, dict[str, Any]]:
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = [int(token_id) for token_id in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    if not input_ids:
        raise ValueError("tokenless prompt")
    if len(input_ids) != len(offsets):
        raise ValueError("token/offset length mismatch")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("V24 requires a fast tokenizer for auditable offset mapping")
    positions: dict[str, dict[str, Any]] = {}
    for name, char_offset in line_boundary_offsets(prompt).items():
        candidates = [
            index
            for index, (start, end) in enumerate(offsets)
            if end <= char_offset and end > start
        ]
        if not candidates:
            candidates = [
                index
                for index, (start, end) in enumerate(offsets)
                if start < char_offset <= end and end > start
            ]
        if not candidates:
            raise ValueError(f"could not map {name} at char offset {char_offset}")
        token_index = candidates[-1]
        start, end = offsets[token_index]
        positions[name] = {
            "token_index": token_index,
            "token_id": input_ids[token_index],
            "prefix_token_count": token_index + 1,
            "boundary_char_offset": char_offset,
            "token_char_span": [start, end],
            "token_text": prompt[start:end],
            "position_order": POSITION_ORDER[name],
        }
    return positions


def collect_features_and_position_margins(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, list[float]]], list[dict[str, Any]], dict[str, Any]]:
    hidden_accumulator: dict[str, list[list[torch.Tensor]]] | None = None
    position_margins = {
        position: {
            "next_token_city_margin": [],
            "prefix_token_count": [],
            "position_token_id": [],
        }
        for position in ALL_POSITIONS
    }
    scored_records = []
    mapping_failures = []
    for row in records:
        try:
            positions = token_positions_for_prompt(tokenizer, row["prompt"])
        except Exception as exc:  # noqa: BLE001
            mapping_failures.append({"id": row["id"], "error": repr(exc)})
            continue
        encoded = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            output = model(**encoded, output_hidden_states=True)
        hidden_states = output.hidden_states
        logits = output.logits[0].detach().float().cpu()
        if hidden_accumulator is None:
            hidden_accumulator = {position: [[] for _ in range(len(hidden_states))] for position in ALL_POSITIONS}
        for position in ALL_POSITIONS:
            token_index = int(positions[position]["token_index"])
            true_token = first_answer_token_id(tokenizer, row["true_capital"])
            override_token = first_answer_token_id(tokenizer, row["override_capital"])
            position_margins[position]["next_token_city_margin"].append(
                float(logits[token_index, true_token] - logits[token_index, override_token])
            )
            position_margins[position]["prefix_token_count"].append(float(positions[position]["prefix_token_count"]))
            position_margins[position]["position_token_id"].append(float(positions[position]["token_id"]))
            for layer_index, layer_state in enumerate(hidden_states):
                hidden_accumulator[position][layer_index].append(layer_state[0, token_index].detach().float().cpu())
        scored_records.append({**row, "position_metadata": positions})
    if hidden_accumulator is None:
        raise ValueError("no hidden features collected")
    features_by_position = {
        position: {
            f"layer_{layer_index}": torch.stack(chunks).numpy()
            for layer_index, chunks in enumerate(layer_chunks)
        }
        for position, layer_chunks in hidden_accumulator.items()
    }
    mapping = {
        "attempted": len(records),
        "scored": len(scored_records),
        "failures": mapping_failures[:20],
        "complete": len(scored_records) == len(records) and not mapping_failures,
    }
    return features_by_position, position_margins, scored_records, mapping


def split_masks(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "discovery": np.asarray([row["split"] == "discovery" for row in records], dtype=bool),
        "calibration": np.asarray([row["split"] == "calibration" for row in records], dtype=bool),
        "holdout": np.asarray([row["split"] == "holdout" for row in records], dtype=bool),
        "non_holdout": np.asarray([row["split"] != "holdout" for row in records], dtype=bool),
    }


def scalar_baseline(values: list[float], labels: np.ndarray, masks: dict[str, np.ndarray], name: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    fit = fit_direction(array[masks["discovery"]], labels[masks["discovery"]], array[masks["holdout"]], labels[masks["holdout"]])
    return {**metric_payload(fit), "score": name}


def score_hidden_candidates(
    features_by_position: dict[str, dict[str, np.ndarray]],
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    candidates = []
    selected = None
    for position in SELECTABLE_POSITIONS:
        layer_items = sorted(
            features_by_position[position].items(),
            key=lambda item: int(item[0].split("_")[1]),
        )
        for name, matrix in layer_items:
            layer = int(name.split("_")[1])
            fit = fit_direction(
                matrix[masks["discovery"]],
                labels[masks["discovery"]],
                matrix[masks["holdout"]],
                labels[masks["holdout"]],
            )
            record = {
                "name": f"{position}/{name}",
                "position": position,
                "position_order": POSITION_ORDER[position],
                "layer": layer,
                **metric_payload(fit),
            }
            candidates.append(record)
            key = (
                -1.0 if record["discovery_auc"] is None else float(record["discovery_auc"]),
                -1.0 if record["holdout_auc"] is None else float(record["holdout_auc"]),
                -int(record["position_order"]),
                -int(record["layer"]),
            )
            if selected is None:
                selected = record
            else:
                selected_key = (
                    -1.0 if selected["discovery_auc"] is None else float(selected["discovery_auc"]),
                    -1.0 if selected["holdout_auc"] is None else float(selected["holdout_auc"]),
                    -int(selected["position_order"]),
                    -int(selected["layer"]),
                )
                if key > selected_key:
                    selected = record
    return candidates, selected


def summarize(records: list[dict[str, Any]], outputs: list[dict[str, Any]], source_validation: dict[str, Any]) -> dict[str, Any]:
    structural = structural_check(records)
    by_template = template_summary(outputs)
    selection = select_template(by_template)
    selected_template = selection["selected_template"]
    selected_summary = by_template[selected_template]
    selected_rows = [row for row in outputs if row["template"] == selected_template]
    selected_binary = [row for row in selected_rows if row["selected_label"] in PRIMARY_LABELS]
    counts = selected_summary["non_holdout_label_counts"]
    holdout_counts = selected_summary["holdout_label_counts"]
    final_audit = selected_summary["margin_audits"]["final_next_token_city_margin"]
    criteria = {
        "source_artifacts_valid": source_validation["passed"],
        "structural_passed": structural["passed"],
        "selected_binary_rows_at_least_30": selected_summary["binary_count"] >= 30,
        "selected_non_holdout_true_at_least_6": counts.get("true_answer", 0) >= 6,
        "selected_non_holdout_override_at_least_6": counts.get("override_answer", 0) >= 6,
        "selected_holdout_true_at_least_2": holdout_counts.get("true_answer", 0) >= 2,
        "selected_holdout_override_at_least_2": holdout_counts.get("override_answer", 0) >= 2,
        "selected_first_token_city_rate_at_most_0p1": (selected_summary["selected_city_first_token_rate"] or 0.0) <= 0.1,
        "selected_final_city_margin_not_sign_barrier": not final_audit["sign_barrier"],
        "selected_final_city_margin_overlap_exists": bool(
            final_audit["by_split"]["non_holdout"]["raw_overlap_exists"]
            or final_audit["by_split"]["holdout"]["raw_overlap_exists"]
        ),
    }
    behavior_ready = all(
        criteria[key]
        for key in (
            "source_artifacts_valid",
            "structural_passed",
            "selected_binary_rows_at_least_30",
            "selected_non_holdout_true_at_least_6",
            "selected_non_holdout_override_at_least_6",
            "selected_holdout_true_at_least_2",
            "selected_holdout_override_at_least_2",
            "selected_first_token_city_rate_at_most_0p1",
        )
    )
    if not criteria["source_artifacts_valid"] or not criteria["structural_passed"]:
        diagnostic_class = "source_artifact_invalid"
    elif not behavior_ready:
        diagnostic_class = "delayed_city_behavior_substrate_failed"
    elif criteria["selected_final_city_margin_not_sign_barrier"]:
        diagnostic_class = "delayed_city_interface_decouples_first_token_margin"
    else:
        diagnostic_class = "delayed_city_interface_still_city_margin_shadowed"
    return {
        "source_validation": source_validation,
        "structural": structural,
        "by_template": by_template,
        "selection": selection,
        "selected_template_summary": selected_summary,
        "selected_template_rows": selected_rows,
        "selected_binary_rows": selected_binary,
        "criteria": criteria,
        "behavior_ready": behavior_ready,
        "diagnostic_class": diagnostic_class,
        "signature_ready": False,
        "intervention_ready": False,
    }


def maybe_hidden_screen(
    summary: dict[str, Any],
    tokenizer: Any,
    model: Any,
) -> dict[str, Any]:
    records = summary["selected_binary_rows"]
    if not summary["behavior_ready"]:
        return {"ran": False, "reason": "behavior_not_ready"}
    if len(records) < 30:
        return {"ran": False, "reason": "too_few_binary_rows"}
    labels = np.asarray([1 if row["selected_label"] == "true_answer" else 0 for row in records], dtype=np.int64)
    masks = split_masks(records)
    if len(set(labels[masks["discovery"]].tolist())) < 2:
        return {"ran": False, "reason": "discovery_single_class"}
    if len(set(labels[masks["holdout"]].tolist())) < 2:
        return {"ran": False, "reason": "holdout_single_class"}
    features, position_margins, scored_records, mapping = collect_features_and_position_margins(records, tokenizer, model)
    hidden_candidates, selected_hidden = score_hidden_candidates(features, labels, masks)
    global_baselines = {
        "final_next_token_city_margin": scalar_baseline(
            [row["final_next_token_city_margin"] for row in scored_records],
            labels,
            masks,
            "final_next_token_city_margin",
        ),
        "city_candidate_score_margin": scalar_baseline(
            [row["city_candidate_score_margin"] for row in scored_records],
            labels,
            masks,
            "city_candidate_score_margin",
        ),
        "json_candidate_score_margin": scalar_baseline(
            [row["json_candidate_score_margin"] for row in scored_records],
            labels,
            masks,
            "json_candidate_score_margin",
        ),
    }
    position_baselines = {
        position: scalar_baseline(
            position_margins[position]["next_token_city_margin"],
            labels,
            masks,
            f"{position}_next_token_city_margin",
        )
        for position in ALL_POSITIONS
    }
    best_control_holdout = max(
        [
            float(item["holdout_auc"] or 0.0)
            for item in list(global_baselines.values()) + list(position_baselines.values())
        ]
    )
    selected_holdout = float((selected_hidden or {}).get("holdout_auc") or 0.0)
    selected_discovery = float((selected_hidden or {}).get("discovery_auc") or 0.0)
    signature_gate = bool(
        selected_hidden
        and selected_discovery >= 0.85
        and selected_holdout >= 0.75
        and selected_holdout > best_control_holdout
    )
    return {
        "ran": True,
        "mapping": mapping,
        "row_count": len(records),
        "label_counts": dict(sorted(Counter(row["selected_label"] for row in records).items())),
        "split_counts": dict(sorted(Counter(row["split"] for row in records).items())),
        "selected_hidden": selected_hidden,
        "top_hidden_candidates": sorted(
            hidden_candidates,
            key=lambda row: (
                -1.0 if row["discovery_auc"] is None else float(row["discovery_auc"]),
                -1.0 if row["holdout_auc"] is None else float(row["holdout_auc"]),
            ),
            reverse=True,
        )[:20],
        "global_baselines": global_baselines,
        "position_baselines": position_baselines,
        "best_control_holdout_auc": best_control_holdout,
        "signature_gate": signature_gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument(
        "--artifact-prefix",
        default="mc006_qwen3_1p7b_parametric_fact_override_v24_delayed_city_interface",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-hidden", action="store_true")
    args = parser.parse_args()

    started = time.time()
    records = source_records()
    source_validation = validate_sources()
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
    summary = summarize(records, outputs, source_validation)
    hidden_screen = (
        {"ran": False, "reason": "skip_hidden_requested"}
        if args.skip_hidden
        else maybe_hidden_screen(summary, tokenizer, model)
    )
    if hidden_screen.get("ran"):
        summary["signature_ready"] = bool(hidden_screen.get("signature_gate"))
    summary["intervention_ready"] = bool(summary["signature_ready"])
    if summary["signature_ready"]:
        summary["diagnostic_class"] = "delayed_city_hidden_signature_candidate"

    output_path = args.output_dir / f"{args.artifact_prefix}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    result = {
        "card_id": args.card_id,
        "run_type": RUN_TYPE,
        "model_id": args.model_id,
        "source_artifacts": SOURCE_ARTIFACTS,
        "source_artifact_hashes": source_hashes(),
        "max_new_tokens": args.max_new_tokens,
        "summary": {
            key: value
            for key, value in summary.items()
            if key not in {"selected_template_rows", "selected_binary_rows"}
        },
        "hidden_screen": hidden_screen,
        "rows": outputs,
        "started_at": started,
        "finished_at": time.time(),
        "duration_seconds": time.time() - started,
    }
    result["summary"]["selected_template_rows"] = summary["selected_template_rows"]
    result["summary"]["selected_binary_rows"] = summary["selected_binary_rows"]
    result["summary"]["signature_ready"] = bool(summary["signature_ready"])
    result["summary"]["intervention_ready"] = bool(summary["intervention_ready"])
    result["output_path"] = str(output_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["summary"]["criteria"], indent=2))
    print(
        "RESULT "
        f"path={output_path} diagnostic={result['summary']['diagnostic_class']} "
        f"behavior_ready={result['summary']['behavior_ready']} "
        f"signature_ready={result['summary']['signature_ready']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
