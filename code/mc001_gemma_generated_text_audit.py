#!/usr/bin/env python
"""MC001G behavior audit for generated answer-text repair banks."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mc001_gemma_repair_discovery import collect_rows, item_id, load_json, select_clean_items


MATCH_CONDITIONS = ("wrong_disclaimed", "wrong_unsure")


def split_clean_items_by_source(clean_items: dict[str, str]) -> dict[str, str]:
    split: dict[str, str] = {}
    for index, item in enumerate(sorted(clean_items)):
        split[item] = "holdout" if index % 3 == 2 else "discovery"
    return split


def normalize_answer(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().strip(" .,:;")).lower()


def answer_shape(value: str) -> str:
    normalized = normalize_answer(value)
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", normalized):
        return "number"
    words = normalized.split()
    if len(words) == 1:
        return "single_word"
    return "multi_word"


def answer_len_bin(value: str) -> str:
    normalized = normalize_answer(value)
    if len(normalized) <= 2:
        return "char_1_2"
    if len(normalized) <= 6:
        return "char_3_6"
    if len(normalized) <= 12:
        return "char_7_12"
    return "char_13_plus"


def question_family(question: str) -> str:
    lowered = question.strip().lower()
    if lowered.startswith("in python"):
        return "code"
    if lowered.startswith("what is"):
        return "what_is"
    if lowered.startswith("which"):
        return "which"
    if lowered.startswith("how many"):
        return "how_many"
    if lowered.startswith("who"):
        return "who"
    if lowered.startswith("at standard"):
        return "at_standard"
    return lowered.split(maxsplit=1)[0] if lowered else "unknown"


def counter(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def pair_counter(rows: list[dict[str, Any]], fields: tuple[str, str]) -> dict[str, int]:
    counts = Counter("|".join(str(row[field]) for field in fields) for row in rows)
    return dict(sorted(counts.items()))


def matched_rows_by(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["split"],) + tuple(row[field] for field in fields)
        grouped[key][row["label"]].append(row)

    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        truth = sorted(grouped[key]["truth_following"], key=lambda row: row["id"])
        agree = sorted(grouped[key]["user_agreement_error"], key=lambda row: row["id"])
        take = min(len(truth), len(agree))
        selected.extend(truth[:take])
        selected.extend(agree[:take])
    return sorted(selected, key=lambda row: (row["split"],) + tuple(row[field] for field in fields) + (row["label"], row["id"]))


def holdout_coverage(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, bool]:
    discovery_keys = {
        tuple(row[field] for field in fields)
        for row in rows
        if row["split"] == "discovery"
    }
    holdout_keys = {
        tuple(row[field] for field in fields)
        for row in rows
        if row["split"] == "holdout"
    }
    return {"|".join(map(str, key)): key in discovery_keys for key in sorted(holdout_keys)}


def split_summary(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in ("discovery", "holdout"):
        split_rows = [row for row in rows if row["split"] == split]
        summary[split] = {
            "n": len(split_rows),
            "label_counts": counter(split_rows, "label"),
            "key_label_counts": {
                "|".join(map(str, key)): counter(key_rows, "label")
                for key, key_rows in sorted(
                    {
                        key: [
                            row
                            for row in split_rows
                            if tuple(row[field] for field in fields) == key
                        ]
                        for key in sorted({tuple(row[field] for field in fields) for row in split_rows})
                    }.items()
                )
            },
        }
    return summary


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        enriched.append(
            {
                **row,
                "correct_shape": answer_shape(row["correct_answer"]),
                "wrong_shape": answer_shape(row["wrong_answer"]),
                "correct_len_bin": answer_len_bin(row["correct_answer"]),
                "wrong_len_bin": answer_len_bin(row["wrong_answer"]),
                "question_family": question_family(row["question"]),
            }
        )
    return enriched


def condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in sorted({row["condition"] for row in rows}):
        condition_rows = [row for row in rows if row["condition"] == condition]
        parseable = [row for row in condition_rows if row["label"] != "unparseable"]
        result[condition] = {
            "n": len(condition_rows),
            "parseable_n": len(parseable),
            "label_counts": counter(condition_rows, "label"),
        }
    return result


def source_overlap(split_items: dict[str, str]) -> list[str]:
    by_source: dict[str, set[str]] = defaultdict(set)
    for item, split in split_items.items():
        by_source[item].add(split)
    return sorted(source for source, splits in by_source.items() if len(splits) > 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cards/MC001G"))
    parser.add_argument("--artifact-prefix", default="mc001g_gemma2_2b_generated_text")
    args = parser.parse_args()

    generation = load_json(args.generation_result)
    rows = generation["records"]
    clean_items = select_clean_items(rows)
    split_items = split_clean_items_by_source(clean_items)

    primary_rows = collect_rows(rows, clean_items, split_items)
    primary_rows = [row for row in primary_rows if row["condition"] in MATCH_CONDITIONS]
    enriched = enrich_rows(primary_rows)

    shape_fields = ("condition", "correct_shape", "wrong_shape")
    strict_fields = ("condition", "correct_shape", "wrong_shape", "correct_len_bin", "wrong_len_bin")
    shape_matched = matched_rows_by(enriched, shape_fields)
    strict_matched = matched_rows_by(enriched, strict_fields)

    result = {
        "card_id": "MC001G",
        "run_type": f"{args.artifact_prefix}_generation_audit",
        "generation_result": str(args.generation_result),
        "variant": generation.get("variant"),
        "model_id": generation.get("model_id"),
        "render_mode": generation.get("render_mode"),
        "max_new_tokens": generation.get("max_new_tokens"),
        "match_conditions": list(MATCH_CONDITIONS),
        "record_count": len(rows),
        "item_count": len({item_id(row) for row in rows}),
        "condition_summary": condition_summary(rows),
        "clean_item_count": len(clean_items),
        "clean_item_counts_by_answer_shape": dict(sorted(Counter(answer_shape(answer) for answer in clean_items.values()).items())),
        "source_overlap": source_overlap(split_items),
        "primary_row_count": len(enriched),
        "primary_label_counts": counter(enriched, "label"),
        "primary_condition_label_counts": pair_counter(enriched, ("condition", "label")),
        "primary_shape_label_counts": pair_counter(enriched, ("correct_shape", "label")),
        "shape_matched": {
            "fields": list(shape_fields),
            "n": len(shape_matched),
            "label_counts": counter(shape_matched, "label"),
            "split_counts": counter(shape_matched, "split"),
            "condition_label_counts": pair_counter(shape_matched, ("condition", "label")),
            "shape_label_counts": pair_counter(shape_matched, ("correct_shape", "label")),
            "summary": split_summary(shape_matched, shape_fields),
            "holdout_key_coverage": holdout_coverage(shape_matched, shape_fields),
        },
        "strict_matched": {
            "fields": list(strict_fields),
            "n": len(strict_matched),
            "label_counts": counter(strict_matched, "label"),
            "split_counts": counter(strict_matched, "split"),
            "condition_label_counts": pair_counter(strict_matched, ("condition", "label")),
            "shape_label_counts": pair_counter(strict_matched, ("correct_shape", "label")),
            "summary": split_summary(strict_matched, strict_fields),
            "holdout_key_coverage": holdout_coverage(strict_matched, strict_fields),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_generation_audit_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)

    print(json.dumps({**result, "records": None, "output_path": str(output_path)}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
