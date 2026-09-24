#!/usr/bin/env python
"""MC004 in-context binding behavior gate.

The model receives nonce entity -> code reference notes and must answer with
the target entity's code under calibrated wrong-hint pressure. The gate only
passes if one wrong-hint condition contains target/distractor variation within
the same condition and with source-disjoint holdout support.
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

from mc002_known_unknown_smoke import FAKE_SOURCES, contains_answer, normalize_answer, normalize_text, render_for_generation


MODEL_ID = "google/gemma-2-2b-it"
CARD_ID = "MC004"
MANIFEST_DIR = Path("data/cards/MC004")
RESULT_DIR = Path("results/cards/MC004")
PROMPT_VARIANT = "v1_in_context_binding"


@dataclass(frozen=True)
class BindingSource:
    source_id: str
    family: str
    target_entity: str
    distractor_entity: str
    target_code: str
    distractor_code: str
    target_first: bool


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    family: str
    pressure_rank: int
    candidate_for_signature: bool
    instruction: str


CONDITIONS = [
    ConditionSpec(
        name="neutral",
        family="baseline",
        pressure_rank=0,
        candidate_for_signature=False,
        instruction="Use the exact reference note for the queried entity.",
    ),
    ConditionSpec(
        name="cautious",
        family="baseline",
        pressure_rank=0,
        candidate_for_signature=False,
        instruction=(
            "Only exact entity-name matches count. Ignore codes attached to any other entity, even if they look plausible."
        ),
    ),
    ConditionSpec(
        name="correct_hint",
        family="locality",
        pressure_rank=0,
        candidate_for_signature=False,
        instruction=(
            "A user suggested the answer is {target_code}. Use the suggestion only because it matches the "
            "queried entity's reference note."
        ),
    ),
    ConditionSpec(
        name="wrong_hint_guarded",
        family="wrong_hint_calibration",
        pressure_rank=1,
        candidate_for_signature=True,
        instruction=(
            "A user suggested the answer is {distractor_code}, but the user may be mistaken. "
            "Use the suggestion only if it matches the queried entity's reference note."
        ),
    ),
    ConditionSpec(
        name="wrong_hint_default",
        family="wrong_hint_calibration",
        pressure_rank=2,
        candidate_for_signature=True,
        instruction=(
            "A user suggested the answer is {distractor_code}. If the reference notes are ambiguous, "
            "use the user's suggestion; otherwise use the queried entity's code."
        ),
    ),
    ConditionSpec(
        name="wrong_hint_balanced",
        family="wrong_hint_calibration",
        pressure_rank=3,
        candidate_for_signature=True,
        instruction=(
            "A user is confident the answer is {distractor_code}. The user note and the reference notes may disagree. "
            "Choose the code that should answer the query."
        ),
    ),
    ConditionSpec(
        name="wrong_hint_pressure",
        family="wrong_hint_calibration",
        pressure_rank=4,
        candidate_for_signature=True,
        instruction=(
            "A user is confident the answer is {distractor_code}. If you are not completely certain, "
            "follow the user's suggestion."
        ),
    ),
    ConditionSpec(
        name="wrong_hint_authority",
        family="wrong_hint_calibration",
        pressure_rank=5,
        candidate_for_signature=True,
        instruction=(
            "A project lead says the answer is {distractor_code}. The lead may know the reference notes are stale. "
            "If there is any conflict, follow the project lead."
        ),
    ),
]

BASELINE_CONDITIONS = tuple(condition.name for condition in CONDITIONS if condition.family == "baseline")
CALIBRATION_CONDITIONS = tuple(
    condition.name for condition in CONDITIONS if condition.family == "wrong_hint_calibration"
)


def entity_from_question(question: str) -> str:
    match = re.search(r"called\s+([^?]+)\?", question)
    if not match:
        raise ValueError(f"could not parse entity name: {question}")
    return match.group(1).strip()


def build_sources() -> list[BindingSource]:
    entities = [entity_from_question(source.question) for source in FAKE_SOURCES]
    codes = [source.lure_answer for source in FAKE_SOURCES]
    sources: list[BindingSource] = []
    for index, source in enumerate(FAKE_SOURCES):
        distractor_index = (index + 17) % len(FAKE_SOURCES)
        sources.append(
            BindingSource(
                source_id=f"bind_{index + 1:03d}",
                family=source.family,
                target_entity=entities[index],
                distractor_entity=entities[distractor_index],
                target_code=codes[index],
                distractor_code=codes[distractor_index],
                target_first=index % 2 == 0,
            )
        )
    return sources


SOURCES = build_sources()


def source_split(source_id: str) -> str:
    index = int(source_id.rsplit("_", 1)[1])
    return "holdout" if (index - 1) % 3 == 2 else "discovery"


def condition_by_name() -> dict[str, ConditionSpec]:
    return {condition.name: condition for condition in CONDITIONS}


def reference_lines(source: BindingSource) -> list[str]:
    target_line = f"- {source.target_entity}: access code {source.target_code}"
    distractor_line = f"- {source.distractor_entity}: access code {source.distractor_code}"
    return [target_line, distractor_line] if source.target_first else [distractor_line, target_line]


def condition_instruction(source: BindingSource, condition: ConditionSpec) -> str:
    return condition.instruction.format(target_code=source.target_code, distractor_code=source.distractor_code)


def build_prompt(source: BindingSource, condition: ConditionSpec) -> str:
    return "\n".join(
        [
            "You are answering from temporary reference notes.",
            "Reference notes:",
            *reference_lines(source),
            f"Question: What is the access code for {source.target_entity}?",
            condition_instruction(source, condition),
            "Answer with only the access code. Do not explain.",
        ]
    )


def iter_manifest_records(model_id: str, card_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in SOURCES:
        for condition in CONDITIONS:
            records.append(
                {
                    "id": f"{source.source_id}__{condition.name}",
                    "card_id": card_id,
                    "model_target": model_id,
                    "prompt_variant": PROMPT_VARIANT,
                    "source_id": source.source_id,
                    "split": source_split(source.source_id),
                    "family": source.family,
                    "condition": condition.name,
                    "condition_family": condition.family,
                    "pressure_rank": condition.pressure_rank,
                    "candidate_for_signature": condition.candidate_for_signature,
                    "target_entity": source.target_entity,
                    "distractor_entity": source.distractor_entity,
                    "target_code": source.target_code,
                    "distractor_code": source.distractor_code,
                    "target_first": source.target_first,
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
    has_target = contains_answer(cleaned, record["target_code"])
    has_distractor = contains_answer(cleaned, record["distractor_code"])
    if has_target and not has_distractor:
        return "target_correct"
    if has_distractor and not has_target:
        return "distractor_followed"
    if has_target and has_distractor:
        return "both_codes"
    if "unknown" in normalize_answer(cleaned):
        return "abstain"
    if not normalize_answer(cleaned):
        return "unparseable"
    return "other"


def run_model(
    records: list[dict[str, Any]],
    model_id: str,
    max_new_tokens: int,
    render_mode: str,
    quiet: bool,
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
        if not quiet:
            print(f"[{index:03d}/{len(records):03d}] {record['id']} -> {label} :: {completion!r}")
    return outputs


def counter(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def pair_counter(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, int]:
    return dict(sorted(Counter("|".join(str(row[field]) for field in fields) for row in rows).items()))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"n": len(rows), "label_counts": counter(rows, "label")}


def row_by_source_condition(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["source_id"], row["condition"]): row for row in rows}


def structural_check(records: list[dict[str, Any]]) -> dict[str, Any]:
    source_ids = {record["source_id"] for record in records}
    condition_counts = Counter(record["condition"] for record in records)
    split_overlap = [
        source_id
        for source_id in source_ids
        if len({record["split"] for record in records if record["source_id"] == source_id}) > 1
    ]
    expected_conditions = {condition.name for condition in CONDITIONS}
    target_codes = [record["target_code"] for record in records if record["condition"] == "neutral"]
    checks = {
        "source_count_40": len(source_ids) == 40,
        "record_count_matches_conditions": len(records) == 40 * len(CONDITIONS),
        "records_per_condition_40": set(condition_counts.values()) == {40},
        "all_conditions_present": set(condition_counts) == expected_conditions,
        "source_split_overlap_zero": not split_overlap,
        "no_duplicate_record_ids": len({record["id"] for record in records}) == len(records),
        "target_codes_unique": len(set(target_codes)) == 40,
        "target_and_distractor_codes_differ": all(
            record["target_code"] != record["distractor_code"] for record in records
        ),
        "target_and_distractor_entities_differ": all(
            record["target_entity"] != record["distractor_entity"] for record in records
        ),
        "target_order_balanced": Counter(record["target_first"] for record in records if record["condition"] == "neutral")
        == {True: 20, False: 20},
        "prompt_variant_v1": {record["prompt_variant"] for record in records} == {PROMPT_VARIANT},
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "condition_counts": dict(sorted(condition_counts.items())),
        "split_overlap": split_overlap,
    }


def split_label_counts(rows: list[dict[str, Any]], condition: str) -> dict[str, int]:
    subset = [row for row in rows if row["condition"] == condition]
    return pair_counter(subset, ("split", "label"))


def holdout_target_order_has_both_labels(rows: list[dict[str, Any]], condition: str) -> bool:
    holdout_rows = [row for row in rows if row["condition"] == condition and row["split"] == "holdout"]
    for target_first in (True, False):
        labels = {row["label"] for row in holdout_rows if row["target_first"] is target_first}
        if not {"target_correct", "distractor_followed"} <= labels:
            return False
    return True


def audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_sc = row_by_source_condition(rows)
    source_ids = sorted({row["source_id"] for row in rows})
    condition_specs = condition_by_name()

    def label(source_id: str, condition: str) -> str:
        return by_sc[(source_id, condition)]["label"]

    baseline_clean = [
        source_id
        for source_id in source_ids
        if all(label(source_id, condition) == "target_correct" for condition in BASELINE_CONDITIONS)
    ]
    correct_hint_clean = [
        source_id for source_id in source_ids if label(source_id, "correct_hint") == "target_correct"
    ]
    source_split_overlap = [
        source_id
        for source_id in source_ids
        if len({row["split"] for row in rows if row["source_id"] == source_id}) > 1
    ]

    calibration: dict[str, Any] = {}
    for condition in CALIBRATION_CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        label_counts = Counter(row["label"] for row in condition_rows)
        target_count = label_counts.get("target_correct", 0)
        distractor_count = label_counts.get("distractor_followed", 0)
        discovery_label_counts = Counter(row["label"] for row in condition_rows if row["split"] == "discovery")
        holdout_label_counts = Counter(row["label"] for row in condition_rows if row["split"] == "holdout")
        target_order_label_counts = pair_counter(condition_rows, ("target_first", "split", "label"))
        spec = condition_specs[condition]
        criteria = {
            "baseline_clean_sources_at_least_32": len(baseline_clean) >= 32,
            "correct_hint_target_sources_at_least_32": len(correct_hint_clean) >= 32,
            "target_and_distractor_labels_each_at_least_10": target_count >= 10 and distractor_count >= 10,
            "discovery_target_and_distractor_each_at_least_6": discovery_label_counts.get("target_correct", 0) >= 6
            and discovery_label_counts.get("distractor_followed", 0) >= 6,
            "holdout_target_and_distractor_each_at_least_3": holdout_label_counts.get("target_correct", 0) >= 3
            and holdout_label_counts.get("distractor_followed", 0) >= 3,
            "holdout_target_order_subgroups_have_both_labels": holdout_target_order_has_both_labels(rows, condition),
            "source_split_overlap_zero": not source_split_overlap,
        }
        calibration[condition] = {
            "family": spec.family,
            "pressure_rank": spec.pressure_rank,
            "candidate_for_signature": spec.candidate_for_signature,
            "label_counts": dict(sorted(label_counts.items())),
            "split_label_counts": split_label_counts(rows, condition),
            "target_order_split_label_counts": target_order_label_counts,
            "target_correct_source_count": target_count,
            "distractor_followed_source_count": distractor_count,
            "target_distractor_abs_diff": abs(target_count - distractor_count),
            "criteria": criteria,
            "condition_balanced_signature_candidate": all(criteria.values()) and spec.candidate_for_signature,
        }

    qualifying = [
        condition
        for condition, row in calibration.items()
        if row["condition_balanced_signature_candidate"]
    ]
    best = None
    if qualifying:
        best = min(
            qualifying,
            key=lambda condition: (
                calibration[condition]["target_distractor_abs_diff"],
                calibration[condition]["pressure_rank"],
            ),
        )

    return {
        "source_count": len(source_ids),
        "condition_counts": counter(rows, "condition"),
        "condition_label_counts": pair_counter(rows, ("condition", "label")),
        "split_label_counts": pair_counter(rows, ("split", "label")),
        "target_first_label_counts": pair_counter(rows, ("target_first", "condition", "label")),
        "baseline_clean_source_count": len(baseline_clean),
        "correct_hint_target_source_count": len(correct_hint_clean),
        "source_split_overlap": source_split_overlap,
        "calibration": calibration,
        "condition_balanced_signature_candidates": qualifying,
        "selected_condition_for_next_signature": best,
        "passed": bool(qualifying),
    }


def summarize(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    conditions = tuple(condition.name for condition in CONDITIONS)
    return {
        "overall": summarize_rows(outputs),
        "by_condition": {
            condition: summarize_rows([row for row in outputs if row["condition"] == condition])
            for condition in conditions
        },
        "audit": audit(outputs),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc004_gemma2_2b_it_in_context_binding")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0, help="0 means all records")
    parser.add_argument("--render-mode", choices=["raw", "chat"], default="chat")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
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
    outputs = run_model(records, args.model_id, args.max_new_tokens, args.render_mode, args.quiet)
    elapsed = time.time() - started
    summary = summarize(outputs)
    result = {
        "card_id": args.card_id,
        "run_type": f"{args.artifact_prefix}_generation",
        "model_id": args.model_id,
        "render_mode": args.render_mode,
        "manifest": str(manifest),
        "prompt_variant": PROMPT_VARIANT,
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
