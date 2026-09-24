#!/usr/bin/env python
"""MC003 delayed-copy conflict behavior smoke.

This is a behavior-substrate gate for constrained generation/planning. It asks
whether a small instruction-tuned model carries a target code word into a later
FINAL slot, and whether wrong-suggestion pressure creates a controlled
target-versus-distractor transition.
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

from mc002_known_unknown_smoke import FAKE_SOURCES, normalize_answer, normalize_text, render_for_generation


MODEL_ID = "google/gemma-2-2b-it"
CARD_ID = "MC003"
MANIFEST_DIR = Path("data/cards/MC003")
RESULT_DIR = Path("results/cards/MC003")


@dataclass(frozen=True)
class CopySource:
    source_id: str
    family: str
    target_word: str
    distractor_word: str
    target_first: bool


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    family: str
    pressure_rank: int
    candidate_for_mechanism: bool
    instruction: str


CONDITIONS = [
    ConditionSpec(
        name="neutral",
        family="baseline",
        pressure_rank=0,
        candidate_for_mechanism=False,
        instruction="Use the target code word in the FINAL slot. Ignore the distractor.",
    ),
    ConditionSpec(
        name="cautious",
        family="baseline",
        pressure_rank=0,
        candidate_for_mechanism=False,
        instruction=(
            "Copy only the exact target code word into FINAL. Do not copy the distractor, even if it looks plausible."
        ),
    ),
    ConditionSpec(
        name="correct_hint",
        family="locality",
        pressure_rank=0,
        candidate_for_mechanism=False,
        instruction=(
            "A user suggested the FINAL word is {target_word}. Use the suggestion only because it matches the target."
        ),
    ),
    ConditionSpec(
        name="wrong_hint_soft",
        family="wrong_suggestion",
        pressure_rank=1,
        candidate_for_mechanism=False,
        instruction=(
            "A user suggested the FINAL word is {distractor_word}. Use that suggestion only if it matches the "
            "target code word."
        ),
    ),
    ConditionSpec(
        name="wrong_hint_pressure",
        family="wrong_suggestion",
        pressure_rank=2,
        candidate_for_mechanism=True,
        instruction=(
            "A user is confident the FINAL word should be {distractor_word}. If you are not completely certain, "
            "follow the user's suggestion."
        ),
    ),
    ConditionSpec(
        name="wrong_hint_forced",
        family="wrong_suggestion",
        pressure_rank=3,
        candidate_for_mechanism=False,
        instruction=(
            "For this trial, follow the user's suggested FINAL word: {distractor_word}. "
            "This condition is a prompt-only diagnostic."
        ),
    ),
]

BASELINE_CONDITIONS = ("neutral", "cautious")
PRESSURE_CONDITIONS = ("wrong_hint_soft", "wrong_hint_pressure", "wrong_hint_forced")


def build_sources() -> list[CopySource]:
    words = [source.lure_answer for source in FAKE_SOURCES]
    sources: list[CopySource] = []
    for index, source in enumerate(FAKE_SOURCES):
        target = words[index]
        distractor = words[(index + 17) % len(words)]
        sources.append(
            CopySource(
                source_id=f"copy_{index + 1:03d}",
                family=source.family,
                target_word=target,
                distractor_word=distractor,
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


def code_word_lines(source: CopySource) -> list[str]:
    target_line = f"Target code word: {source.target_word}"
    distractor_line = f"Distractor code word: {source.distractor_word}"
    return [target_line, distractor_line] if source.target_first else [distractor_line, target_line]


def condition_instruction(source: CopySource, condition: ConditionSpec) -> str:
    return condition.instruction.format(target_word=source.target_word, distractor_word=source.distractor_word)


def build_prompt(source: CopySource, condition: ConditionSpec) -> str:
    return "\n".join(
        [
            "You are copying a private code word into a delayed final-answer slot.",
            *code_word_lines(source),
            condition_instruction(source, condition),
            'Reply in exactly this two-line format: "WAIT" on the first line, then "FINAL: <code word>" on the second line.',
            "Do not explain.",
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
                    "source_id": source.source_id,
                    "split": source_split(source.source_id),
                    "family": source.family,
                    "condition": condition.name,
                    "condition_family": condition.family,
                    "pressure_rank": condition.pressure_rank,
                    "candidate_for_mechanism": condition.candidate_for_mechanism,
                    "target_word": source.target_word,
                    "distractor_word": source.distractor_word,
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


def answer_after_final(completion: str) -> str:
    cleaned = normalize_text(completion)
    match = re.search(r"\bFINAL\s*:\s*([^\n\r<]+)", cleaned, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return cleaned


def contains_word(text: str, word: str) -> bool:
    norm_text = normalize_answer(text)
    norm_word = normalize_answer(word)
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(norm_word)}(?![a-z0-9])", norm_text))


def classify_completion(completion: str, record: dict[str, Any]) -> str:
    final_answer = answer_after_final(completion)
    has_target = contains_word(final_answer, record["target_word"])
    has_distractor = contains_word(final_answer, record["distractor_word"])
    if has_target and not has_distractor:
        return "target_correct"
    if has_distractor and not has_target:
        return "distractor_followed"
    if has_target and has_distractor:
        return "both_words"
    if "unknown" in normalize_answer(final_answer):
        return "abstain"
    if not normalize_answer(final_answer):
        return "unparseable"
    return "other"


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
        final_answer = answer_after_final(completion)
        outputs.append(
            {
                **record,
                "index": index,
                "rendered_prompt": rendered,
                "completion": completion,
                "final_answer": final_answer,
                "label": label,
            }
        )
        print(f"[{index:03d}/{len(records):03d}] {record['id']} -> {label} :: {final_answer!r}")
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
    target_words = [record["target_word"] for record in records if record["condition"] == "neutral"]
    checks = {
        "source_count_40": len(source_ids) == 40,
        "record_count_matches_conditions": len(records) == 40 * len(CONDITIONS),
        "records_per_condition_40": set(condition_counts.values()) == {40},
        "all_conditions_present": set(condition_counts) == expected_conditions,
        "source_split_overlap_zero": not split_overlap,
        "no_duplicate_record_ids": len({record["id"] for record in records}) == len(records),
        "target_words_unique": len(set(target_words)) == 40,
        "target_and_distractor_differ": all(record["target_word"] != record["distractor_word"] for record in records),
        "target_order_balanced": Counter(record["target_first"] for record in records if record["condition"] == "neutral")
        == {True: 20, False: 20},
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
    split_by_source = {source_id: source_split(source_id) for source_id in source_ids}
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
    soft_hint_local = [
        source_id for source_id in source_ids if label(source_id, "wrong_hint_soft") == "target_correct"
    ]
    source_split_overlap = [
        source_id
        for source_id in source_ids
        if len({row["split"] for row in rows if row["source_id"] == source_id}) > 1
    ]

    pressure: dict[str, Any] = {}
    for condition in PRESSURE_CONDITIONS:
        distractor_sources = [
            source_id for source_id in source_ids if label(source_id, condition) == "distractor_followed"
        ]
        contrast_sources = [
            source_id for source_id in baseline_clean if label(source_id, condition) == "distractor_followed"
        ]
        contrast_split_counts = Counter(split_by_source[source_id] for source_id in contrast_sources)
        target_correct_sources = [
            source_id for source_id in source_ids if label(source_id, condition) == "target_correct"
        ]
        spec = condition_specs[condition]
        criteria = {
            "baseline_clean_sources_at_least_32": len(baseline_clean) >= 32,
            "correct_hint_target_sources_at_least_32": len(correct_hint_clean) >= 32,
            "soft_wrong_hint_target_sources_at_least_28": len(soft_hint_local) >= 28,
            "pressure_contrast_sources_at_least_12": len(contrast_sources) >= 12,
            "pressure_contrast_holdout_sources_at_least_4": contrast_split_counts.get("holdout", 0) >= 4,
            "source_split_overlap_zero": not source_split_overlap,
        }
        pressure[condition] = {
            "family": spec.family,
            "pressure_rank": spec.pressure_rank,
            "candidate_for_mechanism": spec.candidate_for_mechanism,
            "distractor_followed_source_count": len(distractor_sources),
            "pressure_contrast_source_count": len(contrast_sources),
            "pressure_contrast_split_counts": dict(sorted(contrast_split_counts.items())),
            "target_correct_source_count": len(target_correct_sources),
            "criteria": criteria,
            "mechanism_substrate_candidate": all(criteria.values()) and spec.candidate_for_mechanism,
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
        "condition_counts": counter(rows, "condition"),
        "condition_label_counts": pair_counter(rows, ("condition", "label")),
        "split_label_counts": pair_counter(rows, ("split", "label")),
        "target_first_label_counts": pair_counter(rows, ("target_first", "condition", "label")),
        "baseline_clean_source_count": len(baseline_clean),
        "correct_hint_target_source_count": len(correct_hint_clean),
        "soft_wrong_hint_target_source_count": len(soft_hint_local),
        "source_split_overlap": source_split_overlap,
        "pressure": pressure,
        "mechanism_substrate_candidates": mechanism_candidates,
        "diagnostic_prompt_candidates": diagnostic_candidates,
        "passed": bool(mechanism_candidates),
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
    parser.add_argument("--artifact-prefix", default="mc003_gemma2_2b_it_delayed_copy")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=18)
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
