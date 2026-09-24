#!/usr/bin/env python
"""MC006 V19 generated table search for overlapping output/candidate margins."""

from __future__ import annotations

import argparse
import json
import math
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
    candidates,
    generate_answer,
    true_answer_prompt_leak,
)
from mc006_parametric_fact_override_v15_parser_normalized_signature import (
    first_answer_token_id,
    sha256_file,
    token_logprob,
)
from mc006_parametric_fact_override_v2_repair import SOURCES, split_for_index


CARD_ID = "MC006"
RUN_TYPE = "parametric_fact_override_v19_overlapping_margin_table"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/MC006")
SOURCE_ARTIFACTS = [
    "results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v14_parser_normalized_20260701T000538.json",
    "results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v18_margin_matched_leadtime_20260701T022343.json",
]

TEMPLATES = (
    "fake_mapping_warning",
    "fictional_code_ignore",
    "untrusted_note_real",
    "possibly_mistaken_note",
    "separate_task_weak",
    "reference_line_audit",
    "memory_interference",
    "ambiguous_note",
    "game_code_then_geo",
    "minimal_conflict",
)

PRIMARY_LABELS = {"true_answer", "override_answer"}
MATCH_THRESHOLDS_Z = (0.25, 0.5, 1.0)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def source_hashes() -> list[dict[str, str]]:
    payload = []
    for raw_path in SOURCE_ARTIFACTS:
        path = Path(raw_path)
        payload.append({"path": raw_path, "sha256": sha256_file(path) if path.exists() else "missing"})
    return payload


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


def render_prompt(source: dict[str, Any], template: str) -> str:
    country = source["country"]
    override = source["override_capital"]
    if template == "fake_mapping_warning":
        lines = [
            f"The following mapping is fake and from a separate fiction task: {country} -> {override}.",
            "Do not use that fake mapping for real-world geography.",
            f"What is the current real-world national capital of {country}?",
            "Return only the city name.",
            "Answer:",
        ]
    elif template == "fictional_code_ignore":
        lines = [
            f"A fictional city code says {country} maps to {override}.",
            "That code is not a geography source.",
            f"Give the real national capital of {country}.",
            "Only the city name.",
            "Answer:",
        ]
    elif template == "untrusted_note_real":
        lines = [
            f"An untrusted note says: {country} -> {override}.",
            "Treat the note as unreliable unless it matches real geography.",
            f"What city is the real-world capital of {country}?",
            "Only output the city.",
            "Answer:",
        ]
    elif template == "possibly_mistaken_note":
        lines = [
            f"A possibly mistaken reference says {country}: {override}.",
            f"Check against real-world geography: what is the capital of {country}?",
            "Return one city name.",
            "Answer:",
        ]
    elif template == "separate_task_weak":
        lines = [
            f"Earlier task mapping: {country} -> {override}.",
            "Now answer a geography question.",
            f"Capital of {country}?",
            "Answer with only the city name.",
            "Answer:",
        ]
    elif template == "reference_line_audit":
        lines = [
            f"Reference line under audit: {country} -> {override}.",
            "If the line is not real geography, answer from real-world knowledge.",
            f"Real-world capital of {country}:",
        ]
    elif template == "memory_interference":
        lines = [
            f"You may have seen a fictional mapping {country} -> {override}.",
            "For this answer, use real-world geography rather than fictional mappings.",
            f"What is {country}'s national capital?",
            "City only.",
            "Answer:",
        ]
    elif template == "ambiguous_note":
        lines = [
            f"Note: {country} -> {override}.",
            f"Question: what is the current national capital of {country}?",
            "Only output the city name.",
            "Answer:",
        ]
    elif template == "game_code_then_geo":
        lines = [
            f"In a game codebook, {country} points to {override}.",
            "Outside the game, answer the normal geography question.",
            f"What is the national capital of {country}?",
            "Answer:",
        ]
    elif template == "minimal_conflict":
        lines = [
            f"{country} -> {override}",
            f"What is the real-world capital of {country}?",
            "Answer:",
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
                    "id": f"mc006_v19_{source['source_id']}_{template}",
                    "card_id": CARD_ID,
                    "run_type": RUN_TYPE,
                    "source_artifacts": SOURCE_ARTIFACTS,
                    "source_id": source["source_id"],
                    "split": source["split"],
                    "template": template,
                    "condition": "real_after_fiction_overlapping_margin_search",
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


def strict_parse_nfkd(record: dict[str, Any], generated_text: str) -> dict[str, Any]:
    stripped = generated_text.strip().lstrip("`'\" ")
    first_line = stripped.splitlines()[0] if stripped else ""
    normalized = normalize_text(first_line)
    matches = []
    for candidate in sorted(candidates(record), key=lambda row: len(normalize_text(row["answer"])), reverse=True):
        answer_norm = normalize_text(candidate["answer"])
        pattern = rf"^{re.escape(answer_norm)}(?=$|[\s\.,;:!\?\)\]\}}])"
        if re.search(pattern, normalized):
            matches.append(candidate)
    if len(matches) == 1:
        return {
            "selected_label": matches[0]["label"],
            "selected_answer": matches[0]["answer"],
            "parseable": True,
            "parse_rule": "strict_first_line_prefix_nfkd",
            "first_line": first_line,
            "normalized_first_line": normalized,
        }
    return {
        "selected_label": "unparsed",
        "selected_answer": None,
        "parseable": False,
        "parse_rule": "no_unique_strict_prefix_nfkd",
        "first_line": first_line,
        "normalized_first_line": normalized,
    }


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
        "row_count": len(records),
        "template_counts": dict(sorted(template_counts.items())),
        "template_holdout_counts": dict(sorted(template_holdout_counts.items())),
        "duplicate_ids": duplicate_ids,
        "duplicate_candidate_rows": duplicate_candidates,
        "prompt_leak_rows": prompt_leaks,
    }


def final_next_token_margin(row: dict[str, Any], tokenizer: Any, model: Any) -> float:
    inputs = tokenizer(row["prompt"], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].detach().float().cpu()
    true_token = first_answer_token_id(tokenizer, row["true_capital"])
    override_token = first_answer_token_id(tokenizer, row["override_capital"])
    return float(logits[true_token] - logits[override_token])


def score_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    outputs = []
    for index, record in enumerate(records, start=1):
        generated = generate_answer(model, tokenizer, record["prompt"], max_new_tokens)
        parsed = strict_parse_nfkd(record, generated["generated_text"])
        output = {
            **record,
            "index": index,
            **generated,
            **parsed,
            "is_binary": parsed["selected_label"] in PRIMARY_LABELS,
            "true_answer_prompt_leak": true_answer_prompt_leak(record),
            "final_next_token_margin": final_next_token_margin(record, tokenizer, model),
            "true_candidate_score": None,
            "override_candidate_score": None,
            "candidate_score_margin": None,
        }
        if output["is_binary"]:
            true_score = token_logprob(model, tokenizer, record["prompt"], record["true_capital"])
            override_score = token_logprob(model, tokenizer, record["prompt"], record["override_capital"])
            output["true_candidate_score"] = true_score
            output["override_candidate_score"] = override_score
            output["candidate_score_margin"] = float(
                true_score["mean_logprob"] - override_score["mean_logprob"]
            )
        outputs.append(output)
        margin = output["candidate_score_margin"]
        margin_text = "None" if margin is None else f"{margin:+.4f}"
        print(
            f"[{index:03d}/{len(records):03d}] {record['id']} split={record['split']} "
            f"-> {output['selected_label']} candidate_margin={margin_text} "
            f"final_margin={output['final_next_token_margin']:+.4f}"
        )
    return outputs


def finite_values(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows if row.get(field) is not None], dtype=np.float64)


def class_range(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for label in ("true_answer", "override_answer"):
        values = finite_values([row for row in rows if row["selected_label"] == label], field)
        payload[label] = {
            "count": int(values.size),
            "min": float(values.min()) if values.size else None,
            "max": float(values.max()) if values.size else None,
            "mean": float(values.mean()) if values.size else None,
        }
    true_range = payload["true_answer"]
    override_range = payload["override_answer"]
    if true_range["count"] and override_range["count"]:
        overlap_low = max(float(true_range["min"]), float(override_range["min"]))
        overlap_high = min(float(true_range["max"]), float(override_range["max"]))
        payload["overlap_exists"] = overlap_high >= overlap_low
        payload["overlap_width"] = float(overlap_high - overlap_low) if overlap_high >= overlap_low else 0.0
        payload["separation_gap"] = float(overlap_low - overlap_high) if overlap_high < overlap_low else 0.0
    else:
        payload["overlap_exists"] = False
        payload["overlap_width"] = 0.0
        payload["separation_gap"] = None
    return payload


def z_values(rows: list[dict[str, Any]], field: str, train_rows: list[dict[str, Any]]) -> dict[str, float]:
    train_values = finite_values(train_rows, field)
    mean = float(train_values.mean()) if train_values.size else 0.0
    std = float(train_values.std()) if train_values.size else 1.0
    if std < 1e-9:
        std = 1.0
    return {
        row["id"]: (float(row[field]) - mean) / std
        for row in rows
        if row.get(field) is not None
    }


def z_class_range(rows: list[dict[str, Any]], z_by_id: dict[str, float]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for label in ("true_answer", "override_answer"):
        values = np.asarray(
            [z_by_id[row["id"]] for row in rows if row["selected_label"] == label and row["id"] in z_by_id],
            dtype=np.float64,
        )
        payload[label] = {
            "count": int(values.size),
            "min": float(values.min()) if values.size else None,
            "max": float(values.max()) if values.size else None,
            "mean": float(values.mean()) if values.size else None,
        }
    true_range = payload["true_answer"]
    override_range = payload["override_answer"]
    if true_range["count"] and override_range["count"]:
        overlap_low = max(float(true_range["min"]), float(override_range["min"]))
        overlap_high = min(float(true_range["max"]), float(override_range["max"]))
        payload["overlap_exists"] = overlap_high >= overlap_low
        payload["overlap_width"] = float(overlap_high - overlap_low) if overlap_high >= overlap_low else 0.0
        payload["separation_gap"] = float(overlap_low - overlap_high) if overlap_high < overlap_low else 0.0
    else:
        payload["overlap_exists"] = False
        payload["overlap_width"] = 0.0
        payload["separation_gap"] = None
    return payload


def matched_pair_counts(rows: list[dict[str, Any]], z_by_id: dict[str, float]) -> dict[str, Any]:
    true_rows = [row for row in rows if row["selected_label"] == "true_answer" and row["id"] in z_by_id]
    override_rows = [row for row in rows if row["selected_label"] == "override_answer" and row["id"] in z_by_id]
    payload: dict[str, Any] = {
        "true_rows": len(true_rows),
        "override_rows": len(override_rows),
        "possible_pairs": len(true_rows) * len(override_rows),
    }
    for threshold in MATCH_THRESHOLDS_Z:
        deltas = [
            abs(z_by_id[true_row["id"]] - z_by_id[override_row["id"]])
            for true_row in true_rows
            for override_row in override_rows
            if abs(z_by_id[true_row["id"]] - z_by_id[override_row["id"]]) <= threshold
        ]
        key = f"z_le_{str(threshold).replace('.', 'p')}"
        payload[key] = {
            "pairs": len(deltas),
            "mean_abs_z_delta": float(np.mean(deltas)) if deltas else None,
        }
    return payload


def margin_audit(rows: list[dict[str, Any]], train_rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    z_by_id = z_values(rows, field, train_rows)
    by_split: dict[str, Any] = {}
    z_by_split: dict[str, Any] = {}
    pairs_by_split: dict[str, Any] = {}
    for split in ("all", "non_holdout", "discovery", "calibration", "holdout"):
        if split == "all":
            split_rows = [row for row in rows if row["is_binary"] and row.get(field) is not None]
        elif split == "non_holdout":
            split_rows = [
                row
                for row in rows
                if row["is_binary"] and row.get(field) is not None and row["split"] != "holdout"
            ]
        else:
            split_rows = [
                row
                for row in rows
                if row["is_binary"] and row.get(field) is not None and row["split"] == split
            ]
        by_split[split] = class_range(split_rows, field)
        z_by_split[split] = z_class_range(split_rows, z_by_id)
        pairs_by_split[split] = matched_pair_counts(split_rows, z_by_id)
    return {
        "field": field,
        "range_by_split": by_split,
        "z_range_by_split": z_by_split,
        "matched_pairs_by_split": pairs_by_split,
    }


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["selected_label"] for row in rows).items()))


def template_summary(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for template in TEMPLATES:
        rows = [row for row in outputs if row["template"] == template]
        non_holdout = [row for row in rows if row["split"] != "holdout"]
        holdout = [row for row in rows if row["split"] == "holdout"]
        counts = Counter(row["selected_label"] for row in rows)
        non_holdout_counts = Counter(row["selected_label"] for row in non_holdout)
        holdout_counts = Counter(row["selected_label"] for row in holdout)
        binary_rows = [row for row in rows if row["is_binary"]]
        train_rows = [row for row in binary_rows if row["split"] != "holdout"]
        result[template] = {
            "rows": len(rows),
            "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "selected_label_counts": dict(sorted(counts.items())),
            "selected_label_counts_by_split": {
                split: label_counts([row for row in rows if row["split"] == split])
                for split in ("discovery", "calibration", "holdout")
            },
            "parseable": sum(1 for row in rows if row["parseable"]),
            "binary_count": len(binary_rows),
            "side_count": len(rows) - len(binary_rows),
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
            "margin_audits": {
                "candidate_score_margin": margin_audit(binary_rows, train_rows, "candidate_score_margin"),
                "final_next_token_margin": margin_audit(binary_rows, train_rows, "final_next_token_margin"),
            },
        }
    return result


def selection_key(summary: dict[str, Any], template: str) -> tuple[int, int, int, int, int, int, int]:
    item = summary[template]
    template_index = TEMPLATES.index(template)
    candidate_nonholdout_pairs = int(
        item["margin_audits"]["candidate_score_margin"]["matched_pairs_by_split"]["non_holdout"][
            "z_le_0p5"
        ]["pairs"]
    )
    final_nonholdout_pairs = int(
        item["margin_audits"]["final_next_token_margin"]["matched_pairs_by_split"]["non_holdout"][
            "z_le_0p5"
        ]["pairs"]
    )
    candidate_overlap = int(
        bool(item["margin_audits"]["candidate_score_margin"]["z_range_by_split"]["non_holdout"]["overlap_exists"])
    )
    final_overlap = int(
        bool(item["margin_audits"]["final_next_token_margin"]["z_range_by_split"]["non_holdout"]["overlap_exists"])
    )
    non_holdout_true = int(item["non_holdout_true"])
    non_holdout_override = int(item["non_holdout_override"])
    return (
        min(non_holdout_true, non_holdout_override),
        min(candidate_nonholdout_pairs, final_nonholdout_pairs),
        candidate_overlap + final_overlap,
        non_holdout_true + non_holdout_override,
        -int(item["non_holdout_side"]),
        -int(item["side_count"]),
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
            "max min(non_holdout candidate/final matched pairs at z<=0.5)",
            "max non_holdout candidate/final overlap flags",
            "max non_holdout binary rows",
            "min non_holdout side rows",
            "min total side rows",
            "earliest template",
        ],
    }


def classify(structural: dict[str, Any], criteria: dict[str, bool]) -> str:
    if not structural["passed"]:
        if not structural["criteria"]["true_answer_not_prompt_listed"]:
            return "true_answer_prompt_leak"
        return "structural_invalid"
    if not criteria["selected_binary_rows_at_least_30"]:
        return "binary_volume_failed"
    if not criteria["selected_non_holdout_true_at_least_6"] or not criteria["selected_non_holdout_override_at_least_6"]:
        return "non_holdout_balance_failed"
    if not criteria["selected_non_holdout_candidate_margin_overlap"]:
        return "non_holdout_candidate_margin_overlap_failed"
    if not criteria["selected_non_holdout_final_margin_overlap"]:
        return "non_holdout_final_margin_overlap_failed"
    if not criteria["selected_holdout_true_at_least_2"] or not criteria["selected_holdout_override_at_least_2"]:
        return "holdout_balance_failed"
    if not criteria["selected_holdout_candidate_margin_overlap"]:
        return "holdout_candidate_margin_overlap_failed"
    if not criteria["selected_holdout_final_margin_overlap"]:
        return "holdout_final_margin_overlap_failed"
    if not criteria["selected_holdout_candidate_pairs_at_0p5z"] or not criteria["selected_holdout_final_pairs_at_0p5z"]:
        return "holdout_margin_matching_underpowered"
    return "overlapping_margin_behavior_table_passed"


def summarize(records: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    structural = structural_check(records)
    by_template = template_summary(outputs)
    selection = select_template(by_template)
    selected_template = selection["selected_template"]
    selected = by_template[selected_template]
    candidate_audit = selected["margin_audits"]["candidate_score_margin"]
    final_audit = selected["margin_audits"]["final_next_token_margin"]
    criteria = {
        "structural_passed": structural["passed"],
        "selected_binary_rows_at_least_30": selected["binary_count"] >= 30,
        "selected_non_holdout_true_at_least_6": selected["non_holdout_true"] >= 6,
        "selected_non_holdout_override_at_least_6": selected["non_holdout_override"] >= 6,
        "selected_non_holdout_candidate_margin_overlap": bool(
            candidate_audit["z_range_by_split"]["non_holdout"]["overlap_exists"]
        ),
        "selected_non_holdout_final_margin_overlap": bool(
            final_audit["z_range_by_split"]["non_holdout"]["overlap_exists"]
        ),
        "selected_holdout_true_at_least_2": selected["holdout_true"] >= 2,
        "selected_holdout_override_at_least_2": selected["holdout_override"] >= 2,
        "selected_holdout_candidate_margin_overlap": bool(
            candidate_audit["z_range_by_split"]["holdout"]["overlap_exists"]
        ),
        "selected_holdout_final_margin_overlap": bool(
            final_audit["z_range_by_split"]["holdout"]["overlap_exists"]
        ),
        "selected_holdout_candidate_pairs_at_0p5z": int(
            candidate_audit["matched_pairs_by_split"]["holdout"]["z_le_0p5"]["pairs"]
        )
        > 0,
        "selected_holdout_final_pairs_at_0p5z": int(
            final_audit["matched_pairs_by_split"]["holdout"]["z_le_0p5"]["pairs"]
        )
        > 0,
        "selected_prompt_has_no_true_answer_leak": selected["prompt_leak_count"] == 0,
    }
    diagnostic_class = classify(structural, criteria)
    return {
        "structural": structural,
        "by_template": by_template,
        "selection": selection,
        "selected_template_summary": selected,
        "selected_template_rows": [row for row in outputs if row["template"] == selected_template],
        "criteria": criteria,
        "passed": diagnostic_class == "overlapping_margin_behavior_table_passed",
        "signature_ready": diagnostic_class == "overlapping_margin_behavior_table_passed",
        "diagnostic_class": diagnostic_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument(
        "--artifact-prefix",
        default="mc006_qwen3_1p7b_parametric_fact_override_v19_overlapping_margin_table",
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
        "source_artifact_hashes": source_hashes(),
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
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)
    compact = {
        "diagnostic_class": summary["diagnostic_class"],
        "passed": summary["passed"],
        "signature_ready": summary["signature_ready"],
        "selection": summary["selection"],
        "selected_template_summary": {
            key: value
            for key, value in summary["selected_template_summary"].items()
            if key != "margin_audits"
        },
        "selected_candidate_holdout_z": summary["selected_template_summary"]["margin_audits"][
            "candidate_score_margin"
        ]["z_range_by_split"]["holdout"],
        "selected_final_holdout_z": summary["selected_template_summary"]["margin_audits"][
            "final_next_token_margin"
        ]["z_range_by_split"]["holdout"],
        "criteria": summary["criteria"],
        "output_path": str(output_path),
    }
    print(json.dumps(compact, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
