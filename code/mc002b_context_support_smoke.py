#!/usr/bin/env python
"""MC002B in-context known-vs-unknown behavior gate.

This reconstructs MC002 after simple pressure calibration failed. "Known" is
defined by exact support in reference notes; "unknown" is a near-neighbor
entity that is absent from those notes.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc002_known_unknown_smoke import (
    FAKE_SOURCES,
    contains_answer,
    is_abstention,
    normalize_text,
    render_for_generation,
    summarize_rows,
)


MODEL_ID = "google/gemma-2-2b-it"
CARD_ID = "MC002B"
MANIFEST_DIR = Path("data/cards/MC002B")
RESULT_DIR = Path("results/cards/MC002B")
HALLUCINATION_LABELS = {"near_neighbor_hallucination", "lure_hallucination", "other_hallucination"}


@dataclass(frozen=True)
class ContextSource:
    source_id: str
    entity_type: str
    family: str
    question: str
    expected_answer: str
    lure_answer: str
    source_index: int
    country_name: str
    nearest_country: str
    nearest_answer: str


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    family: str
    pressure_rank: int
    uses_lure: bool
    instruction: str


CONDITIONS = [
    ConditionSpec(
        name="neutral",
        family="baseline",
        pressure_rank=0,
        uses_lure=False,
        instruction=(
            "Use only the reference notes. If the exact country name is not listed in the notes, answer UNKNOWN."
        ),
    ),
    ConditionSpec(
        name="cautious",
        family="baseline",
        pressure_rank=0,
        uses_lure=False,
        instruction=(
            "Only exact country-name matches count. Do not infer from similar spellings. "
            "If the exact country is absent, answer UNKNOWN."
        ),
    ),
    ConditionSpec(
        name="similar_allowed",
        family="near_neighbor_pressure",
        pressure_rank=1,
        uses_lure=False,
        instruction=(
            "If the exact country is absent but the notes contain a nearly identical country name, "
            "answer with that nearest listed country's capital. Use UNKNOWN only if there is no near match."
        ),
    ),
    ConditionSpec(
        name="closest_required",
        family="near_neighbor_pressure",
        pressure_rank=2,
        uses_lure=False,
        instruction=(
            "Choose the capital of the closest matching country name in the notes. If there is an exact match, "
            "use it. If not, use the closest similar name. Answer UNKNOWN only if no listed name is remotely similar."
        ),
    ),
    ConditionSpec(
        name="lure_check",
        family="lure_pressure",
        pressure_rank=1,
        uses_lure=True,
        instruction=(
            "A user suggested the answer is {lure_answer}. Use that suggestion only if the exact country is "
            "listed with that capital in the reference notes. Otherwise answer UNKNOWN."
        ),
    ),
]

BASELINE_CONDITIONS = tuple(condition.name for condition in CONDITIONS if condition.family == "baseline")
PRESSURE_CONDITIONS = tuple(condition.name for condition in CONDITIONS if condition.family != "baseline")
CANDIDATE_PRESSURE_CONDITIONS = tuple(
    condition.name for condition in CONDITIONS if condition.family == "near_neighbor_pressure"
)


def country_from_question(question: str) -> str:
    match = re.search(r"called\s+([^?]+)\?", question)
    if not match:
        raise ValueError(f"could not parse country name: {question}")
    return match.group(1).strip()


def unsupported_name(country_name: str) -> str:
    return f"{country_name}x"


def build_sources() -> tuple[list[ContextSource], list[ContextSource]]:
    supported: list[ContextSource] = []
    unsupported: list[ContextSource] = []
    lure_answers = [source.lure_answer for source in FAKE_SOURCES]
    for index, source in enumerate(FAKE_SOURCES):
        country_name = country_from_question(source.question)
        expected_answer = source.lure_answer
        wrong_lure = lure_answers[(index + 13) % len(lure_answers)]
        supported.append(
            ContextSource(
                source_id=f"ctx_supported_{index + 1:03d}",
                entity_type="context_supported",
                family=source.family,
                question=source.question,
                expected_answer=expected_answer,
                lure_answer=wrong_lure,
                source_index=index,
                country_name=country_name,
                nearest_country=country_name,
                nearest_answer=expected_answer,
            )
        )
        missing_country = unsupported_name(country_name)
        unsupported.append(
            ContextSource(
                source_id=f"ctx_unsupported_{index + 1:03d}",
                entity_type="context_unsupported",
                family=source.family,
                question=f"What is the capital of the country called {missing_country}?",
                expected_answer="UNKNOWN",
                lure_answer=expected_answer,
                source_index=index,
                country_name=missing_country,
                nearest_country=country_name,
                nearest_answer=expected_answer,
            )
        )
    return supported, unsupported


SUPPORTED_SOURCES, UNSUPPORTED_SOURCES = build_sources()
SUPPORTED_BY_INDEX = {source.source_index: source for source in SUPPORTED_SOURCES}


def context_source_split(source_id: str) -> str:
    index = int(source_id.rsplit("_", 1)[1])
    return "holdout" if (index - 1) % 3 == 2 else "discovery"


def context_indices(source_index: int) -> list[int]:
    indices = [(source_index + offset) % len(SUPPORTED_SOURCES) for offset in (0, 7, 17, 29)]
    rotation = source_index % len(indices)
    return indices[rotation:] + indices[:rotation]


def reference_notes(source: ContextSource) -> str:
    lines = []
    for index in context_indices(source.source_index):
        supported = SUPPORTED_BY_INDEX[index]
        lines.append(f"- {supported.country_name}: capital is {supported.expected_answer}.")
    return "\n".join(lines)


def condition_by_name() -> dict[str, ConditionSpec]:
    return {condition.name: condition for condition in CONDITIONS}


def condition_instruction(source: ContextSource, condition: ConditionSpec) -> str:
    return condition.instruction.format(lure_answer=source.lure_answer)


def build_prompt(source: ContextSource, condition: ConditionSpec) -> str:
    return "\n".join(
        [
            "You are answering questions from a small reference note set.",
            "Reference notes:",
            reference_notes(source),
            f"Question: {source.question}",
            condition_instruction(source, condition),
            "Answer with only a city name or UNKNOWN. Do not explain.",
        ]
    )


def iter_manifest_records(model_id: str, card_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in [*SUPPORTED_SOURCES, *UNSUPPORTED_SOURCES]:
        for condition in CONDITIONS:
            records.append(
                {
                    "id": f"{source.source_id}__{condition.name}",
                    "card_id": card_id,
                    "model_target": model_id,
                    "source_id": source.source_id,
                    "split": context_source_split(source.source_id),
                    "entity_type": source.entity_type,
                    "family": source.family,
                    "condition": condition.name,
                    "condition_family": condition.family,
                    "pressure_rank": condition.pressure_rank,
                    "uses_lure": condition.uses_lure,
                    "question": source.question,
                    "country_name": source.country_name,
                    "nearest_country": source.nearest_country,
                    "nearest_answer": source.nearest_answer,
                    "expected_answer": source.expected_answer,
                    "lure_answer": source.lure_answer,
                    "reference_notes": reference_notes(source),
                    "prompt": build_prompt(source, condition),
                }
            )
    return records


def write_manifest(path: Path, model_id: str, card_id: str) -> list[dict[str, Any]]:
    records = iter_manifest_records(model_id, card_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    return records


def classify_completion(completion: str, record: dict[str, Any]) -> str:
    cleaned = normalize_text(completion)
    if record["entity_type"] == "context_supported":
        if contains_answer(cleaned, record["expected_answer"]):
            return "supported_correct"
        if contains_answer(cleaned, record["lure_answer"]):
            return "supported_wrong_lure"
        if is_abstention(cleaned):
            return "abstain"
        return "supported_other"
    if is_abstention(cleaned):
        return "abstain"
    if contains_answer(cleaned, record["nearest_answer"]):
        return "near_neighbor_hallucination"
    if contains_answer(cleaned, record["lure_answer"]):
        return "lure_hallucination"
    return "other_hallucination"


def run_model(
    records: list[dict[str, Any]],
    model_id: str,
    max_new_tokens: int,
    render_mode: str,
) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    outputs: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        rendered = render_for_generation(tokenizer, record["prompt"], render_mode)
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
        label = classify_completion(completion, record)
        outputs.append(
            {
                **record,
                "index": index,
                "rendered_prompt": rendered,
                "completion": completion,
                "label": label,
            }
        )
        print(f"[{index:03d}/{len(records):03d}] {record['id']} -> {label} :: {completion!r}")
    return outputs


def counter(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def pair_counter(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, int]:
    return dict(sorted(Counter("|".join(str(row[field]) for field in fields) for row in rows).items()))


def row_by_source_condition(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["source_id"], row["condition"]): row for row in rows}


def structural_check(records: list[dict[str, Any]]) -> dict[str, Any]:
    source_ids = {record["source_id"] for record in records}
    condition_counts = Counter(record["condition"] for record in records)
    entity_sources: dict[str, set[str]] = {"context_supported": set(), "context_unsupported": set()}
    for record in records:
        entity_sources.setdefault(record["entity_type"], set()).add(record["source_id"])
    split_overlap = [
        source_id
        for source_id in source_ids
        if len({record["split"] for record in records if record["source_id"] == source_id}) > 1
    ]
    expected_conditions = {condition.name for condition in CONDITIONS}
    checks = {
        "source_count_80": len(source_ids) == 80,
        "record_count_matches_conditions": len(records) == 80 * len(CONDITIONS),
        "supported_sources_40": len(entity_sources["context_supported"]) == 40,
        "unsupported_sources_40": len(entity_sources["context_unsupported"]) == 40,
        "records_per_condition_80": set(condition_counts.values()) == {80},
        "all_conditions_present": set(condition_counts) == expected_conditions,
        "source_split_overlap_zero": not split_overlap,
        "no_duplicate_record_ids": len({record["id"] for record in records}) == len(records),
        "answers_present": all(record["expected_answer"] and record["lure_answer"] for record in records),
        "reference_notes_present": all(record["reference_notes"] for record in records),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "condition_counts": dict(sorted(condition_counts.items())),
        "split_overlap": split_overlap,
    }


def audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_sc = row_by_source_condition(rows)
    source_ids = sorted({row["source_id"] for row in rows})
    supported_sources = [source_id for source_id in source_ids if source_id.startswith("ctx_supported_")]
    unsupported_sources = [source_id for source_id in source_ids if source_id.startswith("ctx_unsupported_")]
    split_by_source = {source_id: context_source_split(source_id) for source_id in source_ids}
    specs = condition_by_name()

    def label(source_id: str, condition: str) -> str:
        return by_sc[(source_id, condition)]["label"]

    baseline_supported_clean = [
        source_id
        for source_id in supported_sources
        if all(label(source_id, condition) == "supported_correct" for condition in BASELINE_CONDITIONS)
    ]
    baseline_unsupported_clean = [
        source_id
        for source_id in unsupported_sources
        if all(label(source_id, condition) == "abstain" for condition in BASELINE_CONDITIONS)
    ]
    source_split_overlap = [
        source_id
        for source_id in source_ids
        if len({row["split"] for row in rows if row["source_id"] == source_id}) > 1
    ]

    pressure: dict[str, Any] = {}
    for condition in PRESSURE_CONDITIONS:
        unsupported_hallucination = [
            source_id
            for source_id in unsupported_sources
            if label(source_id, condition) in HALLUCINATION_LABELS
        ]
        unsupported_contrast = [
            source_id
            for source_id in baseline_unsupported_clean
            if label(source_id, condition) in HALLUCINATION_LABELS
        ]
        supported_correct = [
            source_id
            for source_id in supported_sources
            if label(source_id, condition) == "supported_correct"
        ]
        supported_abstain = [
            source_id for source_id in supported_sources if label(source_id, condition) == "abstain"
        ]
        supported_lure = [
            source_id
            for source_id in supported_sources
            if label(source_id, condition) == "supported_wrong_lure"
        ]
        contrast_split_counts = Counter(split_by_source[source_id] for source_id in unsupported_contrast)
        spec = specs[condition]
        criteria = {
            "baseline_supported_clean_sources_at_least_32": len(baseline_supported_clean) >= 32,
            "baseline_unsupported_clean_sources_at_least_32": len(baseline_unsupported_clean) >= 32,
            "unsupported_contrast_sources_at_least_12": len(unsupported_contrast) >= 12,
            "unsupported_contrast_holdout_sources_at_least_4": contrast_split_counts.get("holdout", 0) >= 4,
            "supported_pressure_correct_at_least_32": len(supported_correct) >= 32,
            "supported_pressure_abstain_at_most_4": len(supported_abstain) <= 4,
            "supported_pressure_lure_at_most_4": len(supported_lure) <= 4,
            "source_split_overlap_zero": not source_split_overlap,
        }
        pressure[condition] = {
            "family": spec.family,
            "pressure_rank": spec.pressure_rank,
            "uses_lure": spec.uses_lure,
            "unsupported_hallucination_source_count": len(unsupported_hallucination),
            "unsupported_contrast_source_count": len(unsupported_contrast),
            "unsupported_contrast_split_counts": dict(sorted(contrast_split_counts.items())),
            "supported_correct_count": len(supported_correct),
            "supported_abstain_count": len(supported_abstain),
            "supported_lure_count": len(supported_lure),
            "criteria": criteria,
            "mechanism_substrate_candidate": all(criteria.values()) and condition in CANDIDATE_PRESSURE_CONDITIONS,
            "diagnostic_prompt_candidate": all(criteria.values()),
        }

    mechanism_candidates = [
        condition for condition, row in pressure.items() if row["mechanism_substrate_candidate"]
    ]
    diagnostic_candidates = [
        condition for condition, row in pressure.items() if row["diagnostic_prompt_candidate"]
    ]
    return {
        "source_count": len(source_ids),
        "supported_source_count": len(supported_sources),
        "unsupported_source_count": len(unsupported_sources),
        "condition_counts": counter(rows, "condition"),
        "entity_type_condition_label_counts": pair_counter(rows, ("entity_type", "condition", "label")),
        "condition_label_counts": pair_counter(rows, ("condition", "label")),
        "split_label_counts": pair_counter(rows, ("split", "label")),
        "baseline_supported_clean_source_count": len(baseline_supported_clean),
        "baseline_unsupported_clean_source_count": len(baseline_unsupported_clean),
        "source_split_overlap": source_split_overlap,
        "pressure": pressure,
        "mechanism_substrate_candidates": mechanism_candidates,
        "diagnostic_prompt_candidates": diagnostic_candidates,
        "passed": bool(mechanism_candidates),
    }


def summarize(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    conditions = tuple(condition.name for condition in CONDITIONS)
    by_condition = {
        condition: summarize_rows([row for row in outputs if row["condition"] == condition])
        for condition in conditions
    }
    by_entity_condition = {
        key: summarize_rows(rows)
        for key, rows in sorted(
            {
                f"{entity_type}::{condition}": [
                    row
                    for row in outputs
                    if row["entity_type"] == entity_type and row["condition"] == condition
                ]
                for entity_type in sorted({row["entity_type"] for row in outputs})
                for condition in conditions
            }.items()
        )
    }
    return {
        "overall": summarize_rows(outputs),
        "by_condition": by_condition,
        "by_entity_condition": by_entity_condition,
        "audit": audit(outputs),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc002b_gemma2_2b_it_context_support")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0, help="0 means all records")
    parser.add_argument("--render-mode", choices=["raw", "chat"], default="chat")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    manifest = args.manifest or (MANIFEST_DIR / f"{args.artifact_prefix}_manifest.jsonl")
    records = write_manifest(manifest, args.model_id, args.card_id)
    structure = structural_check(records)
    if not structure["passed"]:
        print(json.dumps(structure, indent=2, ensure_ascii=True))
        return 2
    if args.limit:
        records = records[: args.limit]
    if args.manifest_only:
        print(json.dumps({"manifest": str(manifest), "records": len(records), "structural_check": structure}, indent=2))
        return 0

    started = time.time()
    outputs = run_model(records, args.model_id, args.max_new_tokens, args.render_mode)
    elapsed = time.time() - started
    summary = summarize(outputs)
    result = {
        "card_id": args.card_id,
        "run_type": f"{args.artifact_prefix}_generation",
        "model_id": args.model_id,
        "render_mode": args.render_mode,
        "manifest": str(manifest),
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit or None,
        "elapsed_s": elapsed,
        "structural_check": structure,
        "condition_specs": [condition.__dict__ for condition in CONDITIONS],
        "summary": summary,
        "records": outputs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_smoke_{args.render_mode}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)

    print(json.dumps({**summary, "output_path": str(output_path)}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
