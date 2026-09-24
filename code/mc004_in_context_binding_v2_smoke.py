#!/usr/bin/env python
"""MC004 V2 in-context binding update-conflict behavior gate."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from mc004_in_context_binding_smoke import (
    CARD_ID,
    MANIFEST_DIR,
    MODEL_ID,
    RESULT_DIR,
    SOURCES,
    ConditionSpec,
    classify_completion,
    counter,
    pair_counter,
    reference_lines,
    run_model,
    source_split,
    summarize_rows,
)


PROMPT_VARIANT = "v2_update_conflict"


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
        name="update_unverified",
        family="update_conflict_calibration",
        pressure_rank=1,
        candidate_for_signature=True,
        instruction=(
            "An unverified later update says {target_entity}'s access code might be {distractor_code}. "
            "Use the update only if it clearly overrides the reference notes."
        ),
    ),
    ConditionSpec(
        name="update_ambiguous",
        family="update_conflict_calibration",
        pressure_rank=2,
        candidate_for_signature=True,
        instruction=(
            "A later update says {target_entity}'s access code is {distractor_code}, but the original notes say "
            "{target_code}. Decide which source should answer the query."
        ),
    ),
    ConditionSpec(
        name="update_prefer_latest",
        family="update_conflict_calibration",
        pressure_rank=3,
        candidate_for_signature=True,
        instruction=(
            "A later update says {target_entity}'s access code is {distractor_code}. Later updates usually override "
            "older reference notes. Answer with the code that should currently apply."
        ),
    ),
    ConditionSpec(
        name="update_authoritative",
        family="update_conflict_calibration",
        pressure_rank=4,
        candidate_for_signature=True,
        instruction=(
            "An authoritative later update replaces the old note for {target_entity}: the current access code is "
            "{distractor_code}. Use the current code."
        ),
    ),
    ConditionSpec(
        name="update_forced",
        family="update_conflict_calibration",
        pressure_rank=5,
        candidate_for_signature=True,
        instruction=(
            "For this trial, ignore the original note for {target_entity}. The answer is {distractor_code}."
        ),
    ),
]

BASELINE_CONDITIONS = tuple(condition.name for condition in CONDITIONS if condition.family == "baseline")
CALIBRATION_CONDITIONS = tuple(
    condition.name for condition in CONDITIONS if condition.family == "update_conflict_calibration"
)


def condition_by_name() -> dict[str, ConditionSpec]:
    return {condition.name: condition for condition in CONDITIONS}


def condition_instruction(source: Any, condition: ConditionSpec) -> str:
    return condition.instruction.format(
        target_entity=source.target_entity,
        target_code=source.target_code,
        distractor_code=source.distractor_code,
    )


def build_prompt(source: Any, condition: ConditionSpec) -> str:
    return "\n".join(
        [
            "You are answering from temporary reference notes.",
            "Original reference notes:",
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
        "target_order_balanced": Counter(record["target_first"] for record in records if record["condition"] == "neutral")
        == {True: 20, False: 20},
        "prompt_variant_v2": {record["prompt_variant"] for record in records} == {PROMPT_VARIANT},
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "condition_counts": dict(sorted(condition_counts.items())),
        "split_overlap": split_overlap,
    }


def row_by_source_condition(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["source_id"], row["condition"]): row for row in rows}


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
    parser.add_argument("--artifact-prefix", default="mc004_gemma2_2b_it_in_context_binding_v2")
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
