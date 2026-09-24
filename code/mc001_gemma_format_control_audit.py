#!/usr/bin/env python
"""MC001G behavior audit for option-position counterbalanced repair banks."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mc001_gemma_prehint_margin_discovery import attach_item_margins
from mc001_gemma_repair_discovery import collect_rows, load_json, select_clean_items, split_clean_items


MATCH_CONDITIONS = ("wrong_disclaimed", "wrong_unsure")


def source_group(item_id: str) -> str:
    match = re.match(r"^(gemma_repair_p\d{3})_[a-d]$", item_id)
    if match:
        return match.group(1)
    return item_id


def split_clean_items_by_source(clean_items: dict[str, str]) -> dict[str, str]:
    by_source: dict[str, list[str]] = defaultdict(list)
    for item in clean_items:
        by_source[source_group(item)].append(item)

    split: dict[str, str] = {}
    for index, source in enumerate(sorted(by_source)):
        split_name = "holdout" if index % 3 == 2 else "discovery"
        for item in by_source[source]:
            split[item] = split_name
    return split


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


def counter(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def pair_counter(rows: list[dict[str, Any]], fields: tuple[str, str]) -> dict[str, int]:
    counts = Counter("|".join(str(row[field]) for field in fields) for row in rows)
    return dict(sorted(counts.items()))


def bin_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in ("discovery", "holdout"):
        split_rows = [row for row in rows if row["split"] == split]
        summary[split] = {
            "n": len(split_rows),
            "label_counts": counter(split_rows, "label"),
            "bins": {
                str(bin_id): counter([row for row in split_rows if row["no_hint_margin_bin"] == bin_id], "label")
                for bin_id in sorted({row["no_hint_margin_bin"] for row in split_rows})
            },
        }
    return summary


def holdout_bin_coverage(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, bool]:
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


def source_overlap(split_items: dict[str, str]) -> list[str]:
    by_source: dict[str, set[str]] = defaultdict(set)
    for item, split in split_items.items():
        by_source[source_group(item)].add(split)
    return sorted(source for source, splits in by_source.items() if len(splits) > 1)


def clean_source_summary(clean_items: dict[str, str]) -> dict[str, Any]:
    by_source: dict[str, set[str]] = defaultdict(set)
    for item, correct in clean_items.items():
        by_source[source_group(item)].add(correct)
    return {
        "source_count": len(by_source),
        "format_complete_source_count": sum(letters == {"A", "B", "C", "D"} for letters in by_source.values()),
        "clean_positions_per_source": dict(sorted(Counter(len(letters) for letters in by_source.values()).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cards/MC001G"))
    parser.add_argument("--artifact-prefix", default="mc001g_gemma2_2b_format_control")
    parser.add_argument("--bin-width", type=float, default=0.5)
    parser.add_argument("--split-mode", choices=["source", "letter"], default="source")
    args = parser.parse_args()

    repair = load_json(args.repair_result)
    repair_rows = repair["records"]
    clean_items = select_clean_items(repair_rows)
    if args.split_mode == "source":
        split_items = split_clean_items_by_source(clean_items)
    else:
        split_items = split_clean_items(clean_items)

    primary_rows = collect_rows(repair_rows, clean_items, split_items)
    primary_rows = [row for row in primary_rows if row["condition"] in MATCH_CONDITIONS]
    enriched = attach_item_margins(primary_rows, repair_rows, args.bin_width)
    for row in enriched:
        row["source_group"] = source_group(row["item_id"])

    bin_matched = matched_rows_by(enriched, ("no_hint_margin_bin",))
    strict_matched = matched_rows_by(enriched, ("no_hint_margin_bin", "correct_answer"))

    result = {
        "card_id": "MC001G",
        "run_type": f"{args.artifact_prefix}_{args.split_mode}_audit",
        "repair_result": str(args.repair_result),
        "variant": repair.get("variant"),
        "model_id": repair.get("model_id"),
        "render_mode": repair.get("render_mode"),
        "split_mode": args.split_mode,
        "bin_width": args.bin_width,
        "match_conditions": list(MATCH_CONDITIONS),
        "clean_item_count": len(clean_items),
        "clean_item_counts_by_correct": dict(sorted(Counter(clean_items.values()).items())),
        "clean_source_summary": clean_source_summary(clean_items),
        "source_overlap": source_overlap(split_items),
        "primary_row_count": len(enriched),
        "primary_label_counts": counter(enriched, "label"),
        "primary_correct_label_counts": pair_counter(enriched, ("correct_answer", "label")),
        "bin_matched": {
            "n": len(bin_matched),
            "label_counts": counter(bin_matched, "label"),
            "split_counts": counter(bin_matched, "split"),
            "correct_label_counts": pair_counter(bin_matched, ("correct_answer", "label")),
            "condition_label_counts": pair_counter(bin_matched, ("condition", "label")),
            "summary": bin_summary(bin_matched),
            "holdout_bin_coverage": holdout_bin_coverage(bin_matched, ("no_hint_margin_bin",)),
        },
        "strict_matched": {
            "n": len(strict_matched),
            "label_counts": counter(strict_matched, "label"),
            "split_counts": counter(strict_matched, "split"),
            "correct_label_counts": pair_counter(strict_matched, ("correct_answer", "label")),
            "condition_label_counts": pair_counter(strict_matched, ("condition", "label")),
            "summary": bin_summary(strict_matched),
            "holdout_bin_letter_coverage": holdout_bin_coverage(strict_matched, ("no_hint_margin_bin", "correct_answer")),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_{args.split_mode}_audit_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)

    print(json.dumps({**result, "records": None, "output_path": str(output_path)}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
