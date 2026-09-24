#!/usr/bin/env python
"""MC006 V9 offline lenient-parser audit for V8 generated outputs."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CARD_ID = "MC006"
RUN_TYPE = "parametric_fact_override_v9_lenient_parse_audit"
SOURCE_RUN_TYPE = "parametric_fact_override_v8_generated_mode"
SOURCE_ARTIFACT = (
    "results/cards/MC006/"
    "mc006_qwen3_1p7b_parametric_fact_override_v8_generated_mode_20260630T230955.json"
)
RESULT_DIR = Path("results/cards/MC006")
CONDITIONS = ("mode_real", "mode_fictional")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def candidates(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"answer": record["true_capital"], "label": "true_answer"},
        {"answer": record["override_capital"], "label": "override_answer"},
        {"answer": record["lure_capital"], "label": "lure_answer"},
        {"answer": "UNKNOWN", "label": "unknown"},
    ]


def candidate_matches(record: dict[str, Any]) -> list[dict[str, Any]]:
    text = normalize_text(record.get("generated_text", ""))
    matches = []
    for candidate in candidates(record):
        answer = normalize_text(candidate["answer"])
        pattern = rf"(?<![a-z0-9]){re.escape(answer)}(?![a-z0-9])"
        found = list(re.finditer(pattern, text))
        if found:
            matches.append(
                {
                    "label": candidate["label"],
                    "answer": candidate["answer"],
                    "first_start": int(found[0].start()),
                    "match_count": len(found),
                }
            )
    return sorted(matches, key=lambda row: (row["first_start"], row["label"]))


def lenient_parse(record: dict[str, Any]) -> dict[str, Any]:
    matches = candidate_matches(record)
    if not matches:
        return {
            "lenient_label": "unparsed",
            "lenient_answer": None,
            "lenient_parseable": False,
            "lenient_parse_rule": "no_candidate_mention",
            "candidate_matches": [],
        }
    first_start = matches[0]["first_start"]
    earliest = [match for match in matches if match["first_start"] == first_start]
    if len(earliest) != 1:
        return {
            "lenient_label": "unparsed",
            "lenient_answer": None,
            "lenient_parseable": False,
            "lenient_parse_rule": "ambiguous_earliest_candidate",
            "candidate_matches": matches,
        }
    winner = earliest[0]
    return {
        "lenient_label": winner["label"],
        "lenient_answer": winner["answer"],
        "lenient_parseable": True,
        "lenient_parse_rule": "unique_earliest_candidate_mention",
        "candidate_matches": matches,
    }


def load_source(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def structural_check(source: dict[str, Any]) -> dict[str, Any]:
    records = source.get("records", [])
    source_ids = {row["source_id"] for row in records}
    holdout_sources = {row["source_id"] for row in records if row.get("split") == "holdout"}
    condition_counts = Counter(row.get("condition") for row in records)
    source_condition_counts = Counter((row.get("source_id"), row.get("condition")) for row in records)
    duplicate_candidates = []
    for row in records:
        answers = [candidate["answer"] for candidate in candidates(row)]
        if len(set(answers)) != len(answers):
            duplicate_candidates.append(row["id"])
    v8_structural = source.get("summary", {}).get("structural", {}).get("criteria", {})
    criteria = {
        "source_run_type_valid": source.get("run_type") == SOURCE_RUN_TYPE,
        "exactly_44_records": len(records) == 44,
        "exactly_22_sources": len(source_ids) == 22,
        "exactly_5_holdout_sources": len(holdout_sources) == 5,
        "two_conditions_per_source": all(count == 1 for count in source_condition_counts.values())
        and len(source_condition_counts) == 22 * len(CONDITIONS),
        "twenty_two_rows_per_condition": all(condition_counts.get(condition, 0) == 22 for condition in CONDITIONS),
        "no_duplicate_candidate_answers": not duplicate_candidates,
        "v8_true_answer_not_prompt_listed": v8_structural.get("true_answer_not_prompt_listed") is True,
    }
    return {
        "criteria": criteria,
        "passed": all(criteria.values()),
        "source_count": len(source_ids),
        "holdout_source_count": len(holdout_sources),
        "record_count": len(records),
        "condition_counts": dict(sorted((str(k), v) for k, v in condition_counts.items())),
        "duplicate_candidate_rows": duplicate_candidates,
    }


def condition_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for condition in CONDITIONS:
        rows = [row for row in records if row["condition"] == condition]
        result[condition] = {
            "rows": len(rows),
            "lenient_label_counts": dict(sorted(Counter(row["lenient_label"] for row in rows).items())),
            "strict_label_counts": dict(sorted(Counter(row["selected_label"] for row in rows).items())),
            "lenient_expected_correct": sum(1 for row in rows if row["lenient_label"] == row["expected_label"]),
            "strict_expected_correct": sum(1 for row in rows if row["selected_label"] == row["expected_label"]),
            "lenient_parseable": sum(1 for row in rows if row["lenient_parseable"]),
            "strict_parseable": sum(1 for row in rows if row["parseable"]),
            "true_answer": sum(1 for row in rows if row["lenient_label"] == "true_answer"),
            "override_answer": sum(1 for row in rows if row["lenient_label"] == "override_answer"),
            "lure_answer": sum(1 for row in rows if row["lenient_label"] == "lure_answer"),
            "unknown": sum(1 for row in rows if row["lenient_label"] == "unknown"),
            "unparsed": sum(1 for row in rows if row["lenient_label"] == "unparsed"),
        }
    return result


def source_contrasts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        by_source[row["source_id"]][row["condition"]] = row
    contrasts = []
    for source_id, rows_by_condition in sorted(by_source.items()):
        clean = (
            rows_by_condition["mode_real"]["lenient_label"] == "true_answer"
            and rows_by_condition["mode_fictional"]["lenient_label"] == "override_answer"
        )
        strict_clean = (
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
                "lenient_clean_contrast": clean,
                "strict_clean_contrast": strict_clean,
                "lenient_labels": {
                    condition: rows_by_condition[condition]["lenient_label"]
                    for condition in CONDITIONS
                },
                "strict_labels": {
                    condition: rows_by_condition[condition]["selected_label"]
                    for condition in CONDITIONS
                },
                "generated_first_lines": {
                    condition: rows_by_condition[condition]["first_line"]
                    for condition in CONDITIONS
                },
            }
        )
    return contrasts


def classify(structural: dict[str, Any], criteria: dict[str, bool]) -> str:
    if not structural["passed"]:
        return "source_artifact_invalid"
    if not criteria["lenient_parseable_rows_at_least_40"]:
        return "parseability_failed"
    if not criteria["mode_real_true_at_least_20"]:
        return "real_mode_failed"
    if not criteria["mode_fictional_override_at_least_20"]:
        return "fictional_mode_failed"
    if not criteria["clean_contrast_sources_at_least_18"] or not criteria["holdout_clean_contrasts_at_least_4"]:
        return "source_contrast_failed"
    if all(criteria.values()):
        return "lenient_parse_v9_rescue_passed"
    return "mixed_behavior_failure"


def summarize(source: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    structural = structural_check(source)
    by_condition = condition_summary(records)
    contrasts = source_contrasts(records)
    clean_contrasts = [row for row in contrasts if row["lenient_clean_contrast"]]
    holdout_clean = [row for row in clean_contrasts if row["split"] == "holdout"]
    parseable_count = sum(1 for row in records if row["lenient_parseable"])
    strict_parseable_count = sum(1 for row in records if row["parseable"])
    changed_rows = [
        {
            "id": row["id"],
            "source_id": row["source_id"],
            "condition": row["condition"],
            "strict_label": row["selected_label"],
            "lenient_label": row["lenient_label"],
            "first_line": row["first_line"],
        }
        for row in records
        if row["selected_label"] != row["lenient_label"]
    ]
    criteria = {
        "lenient_parseable_rows_at_least_40": parseable_count >= 40,
        "mode_real_true_at_least_20": by_condition["mode_real"]["true_answer"] >= 20,
        "mode_fictional_override_at_least_20": by_condition["mode_fictional"]["override_answer"] >= 20,
        "clean_contrast_sources_at_least_18": len(clean_contrasts) >= 18,
        "holdout_clean_contrasts_at_least_4": len(holdout_clean) >= 4,
    }
    diagnostic_class = classify(structural, criteria)
    return {
        "structural": structural,
        "by_condition": by_condition,
        "source_contrasts": contrasts,
        "strict_parseable_count": strict_parseable_count,
        "lenient_parseable_count": parseable_count,
        "strict_clean_contrast_count": source.get("summary", {}).get("clean_contrast_count"),
        "lenient_clean_contrast_count": len(clean_contrasts),
        "strict_holdout_clean_contrast_count": source.get("summary", {}).get("holdout_clean_contrast_count"),
        "lenient_holdout_clean_contrast_count": len(holdout_clean),
        "changed_row_count": len(changed_rows),
        "changed_rows": changed_rows,
        "criteria": criteria,
        "passed": diagnostic_class == "lenient_parse_v9_rescue_passed",
        "diagnostic_class": diagnostic_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-artifact", type=Path, default=Path(SOURCE_ARTIFACT))
    parser.add_argument("--artifact-prefix", default="mc006_qwen3_1p7b_parametric_fact_override_v9_lenient_parse_audit")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    args = parser.parse_args()

    started = time.time()
    source = load_source(args.source_artifact)
    records = []
    for row in source["records"]:
        parsed = lenient_parse(row)
        records.append(
            {
                **row,
                **parsed,
                "lenient_expected_correct": parsed["lenient_label"] == row["expected_label"],
            }
        )
    summary = summarize(source, records)
    result = {
        "card_id": CARD_ID,
        "run_type": RUN_TYPE,
        "source_artifact": str(args.source_artifact),
        "source_run_type": source.get("run_type"),
        "elapsed_s": time.time() - started,
        "records": records,
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
