#!/usr/bin/env python
"""MC006 V14 parser-normalized audit of the V13 generated table."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from mc006_parametric_fact_override_v13_real_after_fiction_generated import (
    RUN_TYPE as V13_RUN_TYPE,
)
from mc006_parametric_fact_override_v13_real_after_fiction_generated import (
    TEMPLATES,
    candidates,
    structural_check,
)


CARD_ID = "MC006"
RUN_TYPE = "parametric_fact_override_v14_parser_normalized"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/MC006")
DEFAULT_SOURCE_ARTIFACT = (
    RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v13_real_after_fiction_generated_20260630T235712.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", stripped.strip().lower())


def strict_parse_nfkd(record: dict[str, Any]) -> dict[str, Any]:
    generated_text = str(record.get("generated_text", ""))
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


def validate_source(source: dict[str, Any]) -> dict[str, Any]:
    criteria = {
        "expected_run_type": source.get("run_type") == V13_RUN_TYPE,
        "has_records": isinstance(source.get("records"), list),
        "source_structural_passed": bool(source.get("summary", {}).get("structural", {}).get("passed")),
    }
    if criteria["has_records"]:
        records = source["records"]
        structural = structural_check(records)
        criteria.update(
            {
                "exactly_40_sources": structural["criteria"]["exactly_40_sources"],
                "exactly_6_templates": structural["criteria"]["exactly_6_templates"],
                "exactly_240_rows": structural["criteria"]["exactly_240_rows"],
                "forty_rows_per_template": structural["criteria"]["forty_rows_per_template"],
                "eight_holdout_rows_per_template": structural["criteria"]["eight_holdout_rows_per_template"],
                "every_source_once_per_template": structural["criteria"]["every_source_once_per_template"],
                "no_duplicate_record_ids": structural["criteria"]["no_duplicate_record_ids"],
                "no_duplicate_candidate_answers": structural["criteria"]["no_duplicate_candidate_answers"],
                "true_answer_not_prompt_listed": structural["criteria"]["true_answer_not_prompt_listed"],
            }
        )
    return {"criteria": criteria, "passed": all(criteria.values())}


def reparse_records(source_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs = []
    for index, row in enumerate(source_records, start=1):
        parsed = strict_parse_nfkd(row)
        old_label = row.get("selected_label")
        output = {
            **row,
            "index": index,
            "v13_selected_label": old_label,
            "v13_selected_answer": row.get("selected_answer"),
            "v13_parseable": row.get("parseable"),
            "v13_parse_rule": row.get("parse_rule"),
            "v13_first_line": row.get("first_line"),
            **parsed,
            "is_binary": parsed["selected_label"] in {"true_answer", "override_answer"},
            "label_changed_by_normalization": old_label != parsed["selected_label"],
        }
        outputs.append(output)
        changed = " changed" if output["label_changed_by_normalization"] else ""
        print(
            f"[{index:03d}/{len(source_records):03d}] {row['id']} {old_label} -> "
            f"{output['selected_label']}{changed} first={output['first_line']!r}"
        )
    return outputs


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
        result[template] = {
            "rows": len(rows),
            "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "selected_label_counts": dict(sorted(counts.items())),
            "selected_label_counts_by_split": {
                split: label_counts([row for row in rows if row["split"] == split])
                for split in ("discovery", "calibration", "holdout")
            },
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
            "normalization_changed_rows": sum(1 for row in rows if row["label_changed_by_normalization"]),
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


def parser_delta_summary(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    changed = [row for row in outputs if row["label_changed_by_normalization"]]
    changed_from_candidate = [
        row["id"]
        for row in changed
        if row["v13_selected_label"] in {"true_answer", "override_answer", "lure_answer", "unknown"}
    ]
    changed_not_strict_nfkd = [
        row["id"]
        for row in changed
        if row["parse_rule"] != "strict_first_line_prefix_nfkd"
    ]
    criteria = {
        "changed_rows_from_unparsed_only": not changed_from_candidate,
        "no_candidate_to_candidate_changes": not changed_from_candidate,
        "changed_rows_at_most_5": len(changed) <= 5,
        "changed_rows_match_strict_nfkd": not changed_not_strict_nfkd,
    }
    return {
        "criteria": criteria,
        "passed": all(criteria.values()),
        "changed_row_count": len(changed),
        "changed_rows": [
            {
                "id": row["id"],
                "template": row["template"],
                "split": row["split"],
                "v13_selected_label": row["v13_selected_label"],
                "selected_label": row["selected_label"],
                "first_line": row["first_line"],
                "normalized_first_line": row["normalized_first_line"],
                "selected_answer": row["selected_answer"],
            }
            for row in changed
        ],
        "changed_from_candidate_rows": changed_from_candidate,
        "changed_not_strict_nfkd_rows": changed_not_strict_nfkd,
    }


def classify(source_validation: dict[str, Any], structural: dict[str, Any], delta: dict[str, Any], criteria: dict[str, bool]) -> str:
    if not source_validation["passed"] or not structural["passed"]:
        return "source_artifact_invalid"
    if not structural["criteria"]["true_answer_not_prompt_listed"]:
        return "true_answer_prompt_leak"
    if not delta["passed"]:
        return "normalization_delta_too_broad"
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
        return "parser_normalized_generated_substrate_passed"
    return "mixed_behavior_failure"


def summarize(
    source_validation: dict[str, Any],
    records: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    structural = structural_check(records)
    by_template = template_summary(outputs)
    selection = select_template(by_template)
    selected_template = selection["selected_template"]
    selected = by_template[selected_template]
    delta = parser_delta_summary(outputs)
    criteria = {
        "source_artifact_valid": source_validation["passed"],
        "structural_passed": structural["passed"],
        "normalization_delta_passed": delta["passed"],
        "selected_binary_rows_at_least_30": selected["binary_count"] >= 30,
        "selected_non_holdout_true_at_least_6": selected["non_holdout_true"] >= 6,
        "selected_non_holdout_override_at_least_6": selected["non_holdout_override"] >= 6,
        "selected_holdout_true_at_least_2": selected["holdout_true"] >= 2,
        "selected_holdout_override_at_least_2": selected["holdout_override"] >= 2,
        "selected_holdout_side_rows_at_most_4": selected["holdout_side"] <= 4,
        "selected_prompt_has_no_true_answer_leak": selected["prompt_leak_count"] == 0,
    }
    diagnostic_class = classify(source_validation, structural, delta, criteria)
    return {
        "source_validation": source_validation,
        "structural": structural,
        "parser_delta": delta,
        "by_template": by_template,
        "selection": selection,
        "selected_template_summary": selected,
        "selected_template_rows": [row for row in outputs if row["template"] == selected_template],
        "criteria": criteria,
        "passed": diagnostic_class == "parser_normalized_generated_substrate_passed",
        "diagnostic_class": diagnostic_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--source-artifact", type=Path, default=DEFAULT_SOURCE_ARTIFACT)
    parser.add_argument("--artifact-prefix", default="mc006_qwen3_1p7b_parametric_fact_override_v14_parser_normalized")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    args = parser.parse_args()

    started = time.time()
    source_hash = sha256_file(args.source_artifact)
    source = read_json(args.source_artifact)
    source_validation = validate_source(source)
    if not source_validation["passed"]:
        raise ValueError(f"source validation failed: {source_validation}")

    records = source["records"]
    outputs = reparse_records(records)
    summary = summarize(source_validation, records, outputs)
    result = {
        "card_id": args.card_id,
        "run_type": RUN_TYPE,
        "model_id": args.model_id,
        "source_artifact": str(args.source_artifact),
        "source_artifact_sha256": source_hash,
        "parser": {
            "normalization": "NFKD strip combining marks, lowercase, whitespace collapse",
            "rule": "strict_first_line_prefix_nfkd",
            "substring_search": False,
            "alias_list": False,
        },
        "elapsed_s": time.time() - started,
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
