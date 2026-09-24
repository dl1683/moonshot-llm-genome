#!/usr/bin/env python
"""MC007 V3 parseability repair for familiar-entity authority pressure."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc007_semi_synthetic_familiar_entity_lookup import (
    CARD_ID,
    MODEL_ID,
    RESULT_DIR,
    base_sources,
    candidates,
    generate_answer,
    next_token_candidate_logits,
    normalize_text,
    prompt_occurrences,
    rate,
    render_table,
    strict_parse,
    table_sources,
)


RUN_TYPE = "semi_synthetic_familiar_entity_lookup_v3_parse_repair"

PANELS = (
    "synthetic_control_lookup",
    "target_note_authority_50",
    "target_note_authority_30",
    "target_note_authority_0",
    "real_world_control",
    "answer_absent_null",
)

PRIMARY_PANELS = (
    "target_note_authority_50",
    "target_note_authority_30",
    "target_note_authority_0",
)

AUTHORITY_LEVEL_BY_PANEL = {
    "target_note_authority_50": 50,
    "target_note_authority_30": 30,
    "target_note_authority_0": 0,
}

TEMPLATES = (
    "target_only_note",
    "compact_note",
)


def authority_lines(level: int) -> list[str]:
    if level == 50:
        return [
            "Task-note authority: 50/100.",
            "The task note and ordinary geography memory are both relevant.",
        ]
    if level == 30:
        return [
            "Task-note authority: 30/100.",
            "Prefer ordinary geography memory when you know it.",
        ]
    if level == 0:
        return [
            "Task-note authority: 0/100.",
            "Ignore the task note for ordinary geography questions.",
        ]
    raise ValueError(f"unknown authority level: {level}")


def render_target_note(source: dict[str, Any], template: str) -> list[str]:
    country = source["country"]
    artificial = source["artificial_value"]
    if template == "target_only_note":
        return [
            "Task note:",
            "Country -> task city",
            f"{country} -> {artificial}",
        ]
    if template == "compact_note":
        return [
            f"Task note: in a separate lookup task, {country} maps to task city {artificial}.",
        ]
    raise ValueError(f"unknown template: {template}")


def render_prompt(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    panel: str,
    template: str,
) -> str:
    if panel == "synthetic_control_lookup":
        rows = table_sources(sources, source, include_query=True)
        table = render_table(rows, "synthetic_key")
        return "\n".join(
            [
                "Synthetic lookup task. These code keys have no real-world city memory.",
                *table,
                f"Query: {source['synthetic_key']}",
                "Reply with the city name only.",
                "The next token must be the city name, not a sentence.",
                "Answer:",
            ]
        )

    if panel == "real_world_control":
        return "\n".join(
            [
                "Ordinary real-world geography question.",
                f"Question: What is the current real-world national capital of {source['country']}?",
                "Reply with the city name only.",
                "The next token must be the city name, not a sentence.",
                "Do not write 'The', 'Answer', markdown, or an explanation.",
                "Answer:",
            ]
        )

    if panel == "answer_absent_null":
        rows = table_sources(sources, source, include_query=False)
        table = render_table(rows, "country")
        return "\n".join(
            [
                "If the queried key is absent from the table, answer UNKNOWN.",
                "This is a task-city lookup, not real-world geography.",
                "Use the table only for listed keys.",
                "",
                *table,
                "",
                f"Question: According to the table, what is the task city for {source['country']}?",
                "If the country is not listed, output UNKNOWN.",
                "Answer:",
            ]
        )

    if panel not in AUTHORITY_LEVEL_BY_PANEL:
        raise ValueError(f"unknown panel: {panel}")
    level = AUTHORITY_LEVEL_BY_PANEL[panel]
    lines = [
        *render_target_note(source, template),
        "",
        *authority_lines(level),
        f"Question: What is the current real-world national capital of {source['country']}?",
    ]
    if level > 0:
        lines.append("If the task note should override memory, output the task city.")
    lines.extend(
        [
            "Reply with the city name only.",
            "The next token must be the city name, not a sentence.",
            "Do not write 'The', 'Answer', markdown, or an explanation.",
            "Answer:",
        ]
    )
    return "\n".join(lines)


def source_records(limit_sources: int | None = None, templates: tuple[str, ...] = TEMPLATES) -> list[dict[str, Any]]:
    sources = base_sources(limit_sources)
    records = []
    for source in sources:
        for panel in PANELS:
            for template in templates:
                if panel == "answer_absent_null":
                    expected_label = "unknown"
                elif panel == "real_world_control":
                    expected_label = "real_prior"
                else:
                    expected_label = "artificial_value"
                authority_level = AUTHORITY_LEVEL_BY_PANEL.get(panel)
                records.append(
                    {
                        "id": f"mc007_v3_{source['source_id']}_{panel}_{template}",
                        "card_id": CARD_ID,
                        "run_type": RUN_TYPE,
                        "source_id": source["source_id"],
                        "split": source["split"],
                        "panel": panel,
                        "template": template,
                        "authority_level": authority_level,
                        "country": source["country"],
                        "synthetic_key": source["synthetic_key"],
                        "true_capital": source["true_capital"],
                        "override_capital": source["override_capital"],
                        "lure_capital": source["lure_capital"],
                        "artificial_value": source["artificial_value"],
                        "expected_label": expected_label,
                        "prompt": render_prompt(sources, source, panel, template),
                    }
                )
    return records


def prompt_audit(record: dict[str, Any]) -> dict[str, Any]:
    artificial_occurrences = prompt_occurrences(record["prompt"], record["artificial_value"])
    true_occurrences = prompt_occurrences(record["prompt"], record["true_capital"])
    if record["panel"] in {"answer_absent_null", "real_world_control"}:
        artificial_expected = 0
    else:
        artificial_expected = 1
    return {
        "artificial_value_occurrences": artificial_occurrences,
        "true_capital_occurrences": true_occurrences,
        "artificial_value_prompt_count_expected": artificial_occurrences == artificial_expected,
        "true_capital_not_prompt_listed": true_occurrences == 0,
    }


def score_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    score_candidates: bool,
    verbose: bool,
) -> list[dict[str, Any]]:
    outputs = []
    for index, record in enumerate(records, start=1):
        generated = generate_answer(model, tokenizer, record["prompt"], max_new_tokens)
        parsed = strict_parse(record, generated["generated_text"])
        audit = prompt_audit(record)
        output = {
            **record,
            "index": index,
            **generated,
            **parsed,
            **audit,
            "is_primary_panel": record["panel"] in PRIMARY_PANELS,
            "is_primary_binary": record["panel"] in PRIMARY_PANELS
            and parsed["selected_label"] in {"artificial_value", "real_prior", "lure_value"},
        }
        if score_candidates:
            output.update(next_token_candidate_logits(model, tokenizer, record["prompt"], record))
        outputs.append(output)
        if verbose:
            print(
                f"[{index:03d}/{len(records):03d}] {record['id']} split={record['split']} "
                f"panel={record['panel']} template={record['template']} authority={record['authority_level']} -> "
                f"{output['selected_label']} {str(output['selected_answer'])!r} generated={generated['generated_text']!r}"
            )
    return outputs


def structural_check(records: list[dict[str, Any]], templates: tuple[str, ...]) -> dict[str, Any]:
    source_ids = {row["source_id"] for row in records}
    duplicate_ids = [row_id for row_id, count in Counter(row["id"] for row in records).items() if count > 1]
    source_splits: dict[str, set[str]] = {}
    collision_rows = []
    prompt_audit_failures = []
    for row in records:
        source_splits.setdefault(row["source_id"], set()).add(row["split"])
        normalized_candidates = [
            normalize_text(row["artificial_value"]),
            normalize_text(row["true_capital"]),
            normalize_text(row["override_capital"]),
            normalize_text(row["lure_capital"]),
            normalize_text("UNKNOWN"),
        ]
        if len(set(normalized_candidates)) != len(normalized_candidates):
            collision_rows.append(row["id"])
        audit = prompt_audit(row)
        if not audit["artificial_value_prompt_count_expected"] or not audit["true_capital_not_prompt_listed"]:
            prompt_audit_failures.append(row["id"])
    expected_rows = len(source_ids) * len(PANELS) * len(templates)
    template_counts = Counter(row["template"] for row in records)
    panel_counts = Counter(row["panel"] for row in records)
    criteria = {
        "source_count_at_most_40": 1 <= len(source_ids) <= 40,
        "template_set_valid": set(template_counts) == set(templates),
        "panel_set_valid": set(panel_counts) == set(PANELS),
        "expected_row_count": len(records) == expected_rows,
        "no_duplicate_record_ids": not duplicate_ids,
        "source_split_disjoint": all(len(splits) == 1 for splits in source_splits.values()),
        "no_candidate_collisions": not collision_rows,
        "prompt_audit_passed": not prompt_audit_failures,
    }
    return {
        "criteria": criteria,
        "passed": all(criteria.values()),
        "source_count": len(source_ids),
        "template_count": len(template_counts),
        "panel_count": len(panel_counts),
        "row_count": len(records),
        "expected_row_count": expected_rows,
        "template_counts": dict(sorted(template_counts.items())),
        "panel_counts": dict(sorted(panel_counts.items())),
        "duplicate_ids": duplicate_ids,
        "candidate_collision_rows": collision_rows,
        "prompt_audit_failure_rows": prompt_audit_failures,
    }


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["selected_label"] for row in rows).items()))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parseable = sum(1 for row in rows if row.get("parseable"))
    artificial = sum(1 for row in rows if row.get("selected_label") == "artificial_value")
    real_prior = sum(1 for row in rows if row.get("selected_label") == "real_prior")
    lure = sum(1 for row in rows if row.get("selected_label") == "lure_value")
    unknown = sum(1 for row in rows if row.get("selected_label") == "unknown")
    primary_binary = sum(1 for row in rows if row.get("is_primary_binary"))
    result = {
        "rows": len(rows),
        "label_counts": label_counts(rows),
        "parseable": parseable,
        "parseable_rate": rate(parseable, len(rows)),
        "artificial_value": artificial,
        "artificial_value_rate": rate(artificial, len(rows)),
        "real_prior": real_prior,
        "real_prior_rate": rate(real_prior, len(rows)),
        "lure_value": lure,
        "lure_value_rate": rate(lure, len(rows)),
        "prior_or_lure": real_prior + lure,
        "prior_or_lure_rate": rate(real_prior + lure, len(rows)),
        "unknown": unknown,
        "unknown_rate": rate(unknown, len(rows)),
        "primary_binary": primary_binary,
    }
    if rows and "artificial_minus_real_prior_logit" in rows[0]:
        result["mean_artificial_minus_real_prior_logit"] = sum(
            float(row["artificial_minus_real_prior_logit"]) for row in rows
        ) / len(rows)
        result["mean_artificial_minus_lure_logit"] = sum(
            float(row["artificial_minus_lure_logit"]) for row in rows
        ) / len(rows)
        result["mean_unknown_minus_artificial_logit"] = sum(
            float(row["unknown_minus_artificial_logit"]) for row in rows
        ) / len(rows)
    return result


def template_summary(outputs: list[dict[str, Any]], templates: tuple[str, ...]) -> dict[str, Any]:
    result = {}
    for template in templates:
        rows = [row for row in outputs if row["template"] == template]
        panels = {panel: summarize_rows([row for row in rows if row["panel"] == panel]) for panel in PANELS}
        primary = [row for row in rows if row["panel"] in PRIMARY_PANELS]
        primary_non_holdout = [row for row in primary if row["split"] != "holdout"]
        primary_holdout = [row for row in primary if row["split"] == "holdout"]
        result[template] = {
            "rows": len(rows),
            "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "selected_label_counts": label_counts(rows),
            "panels": panels,
            "primary": summarize_rows(primary),
            "primary_non_holdout": summarize_rows(primary_non_holdout),
            "primary_holdout": summarize_rows(primary_holdout),
        }
    return result


def panel_split_summary(outputs: list[dict[str, Any]], template: str, panel: str) -> dict[str, Any]:
    rows = [row for row in outputs if row["template"] == template and row["panel"] == panel]
    non_holdout = [row for row in rows if row["split"] != "holdout"]
    holdout = [row for row in rows if row["split"] == "holdout"]
    return {
        "template": template,
        "panel": panel,
        "authority_level": AUTHORITY_LEVEL_BY_PANEL.get(panel),
        "all": summarize_rows(rows),
        "non_holdout": summarize_rows(non_holdout),
        "holdout": summarize_rows(holdout),
    }


def contrast_key(item: dict[str, Any]) -> tuple[int, int, int, float, int]:
    non_holdout = item["non_holdout"]
    holdout = item["holdout"]
    balance = min(int(non_holdout["artificial_value"]), int(non_holdout["prior_or_lure"]))
    holdout_balance = min(int(holdout["artificial_value"]), int(holdout["prior_or_lure"]))
    binary = int(non_holdout["primary_binary"]) + int(holdout["primary_binary"])
    parseable = float(item["all"]["parseable_rate"])
    authority_level = item.get("authority_level")
    centrality = -abs(int(authority_level if authority_level is not None else 30) - 30)
    return (balance, holdout_balance, binary, parseable, centrality)


def select_best_within_panel(outputs: list[dict[str, Any]], templates: tuple[str, ...]) -> dict[str, Any]:
    candidates_by_panel = [
        panel_split_summary(outputs, template, panel)
        for template in templates
        for panel in PRIMARY_PANELS
    ]
    selected = max(candidates_by_panel, key=contrast_key)
    return {
        "selected_template": selected["template"],
        "selected_panel": selected["panel"],
        "selected_authority_level": selected["authority_level"],
        "selection_key": list(contrast_key(selected)),
        "selected_summary": selected,
        "all_selection_keys": {
            f"{item['template']}::{item['panel']}": list(contrast_key(item))
            for item in candidates_by_panel
        },
        "rule": [
            "max min(non-holdout artificial, non-holdout real-prior-or-lure)",
            "max min(holdout artificial, holdout real-prior-or-lure)",
            "max binary rows",
            "max parseability",
            "prefer authority levels closest to 30",
        ],
    }


def select_cross_panel_template(summary: dict[str, Any], templates: tuple[str, ...]) -> dict[str, Any]:
    def key(template: str) -> tuple[int, int, int, float]:
        item = summary[template]
        non_holdout = item["primary_non_holdout"]
        holdout = item["primary_holdout"]
        balance = min(int(non_holdout["artificial_value"]), int(non_holdout["prior_or_lure"]))
        holdout_balance = min(int(holdout["artificial_value"]), int(holdout["prior_or_lure"]))
        binary = int(item["primary"]["primary_binary"])
        parseable = float(item["primary"]["parseable_rate"])
        return (balance, holdout_balance, binary, parseable)

    selected = max(templates, key=key)
    return {
        "selected_template": selected,
        "selection_key": list(key(selected)),
        "all_selection_keys": {template: list(key(template)) for template in templates},
        "rule": [
            "max cross-panel non-holdout class balance",
            "max cross-panel holdout class balance",
            "max primary binary rows",
            "max primary parseability",
        ],
    }


def classify(criteria: dict[str, bool], selected: dict[str, Any]) -> str:
    primary = selected["cross_panel_summary"]["primary"]
    if criteria["smoke_mode"]:
        return "smoke_only"
    if not criteria["structural_passed"]:
        return "structural_invalid"
    if not criteria["panel_a_artificial_adherence_at_least_90p"]:
        return "synthetic_control_failed"
    if not criteria["primary_binary_rows_at_least_40"]:
        return "behavior_substrate_failed"
    if not criteria["selected_prompt_audit_passed"]:
        return "prompt_leak_failed"
    if criteria["within_panel_behavior_gate_passed"]:
        return "behavior_contrast_substrate_passed"
    if criteria["cross_panel_behavior_gate_passed"]:
        return "prompt_authority_dial_contrast_only"
    if not criteria["primary_parseability_at_least_90p"]:
        return "parseability_repair_failed"
    if int(primary["prior_or_lure"]) == 0:
        return "prior_pressure_insufficient"
    if int(primary["artificial_value"]) == 0:
        return "semantic_prior_dominant"
    return "behavior_substrate_failed"


def summarize(
    records: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    templates: tuple[str, ...],
    full_run: bool,
) -> dict[str, Any]:
    structural = structural_check(records, templates)
    by_template = template_summary(outputs, templates)
    cross_selection = select_cross_panel_template(by_template, templates)
    within_selection = select_best_within_panel(outputs, templates)
    selected_template = cross_selection["selected_template"]
    selected = by_template[selected_template]
    primary = selected["primary"]
    non_holdout = selected["primary_non_holdout"]
    holdout = selected["primary_holdout"]
    within = within_selection["selected_summary"]
    selected_rows = [row for row in outputs if row["template"] == selected_template]

    cross_panel_contrast_present = int(primary["prior_or_lure"]) > 0 and int(primary["artificial_value"]) > 0
    primary_parseable = float(primary["parseable_rate"]) >= 0.90
    cross_panel_behavior_gate_passed = (
        cross_panel_contrast_present
        and primary_parseable
        and int(non_holdout["artificial_value"]) >= 10
        and int(non_holdout["prior_or_lure"]) >= 10
        and int(holdout["artificial_value"]) >= 4
        and int(holdout["prior_or_lure"]) >= 4
    )
    within_panel_contrast_present = (
        int(within["all"]["prior_or_lure"]) > 0 and int(within["all"]["artificial_value"]) > 0
    )
    within_panel_behavior_gate_passed = (
        within_panel_contrast_present
        and float(within["all"]["parseable_rate"]) >= 0.90
        and int(within["non_holdout"]["artificial_value"]) >= 8
        and int(within["non_holdout"]["prior_or_lure"]) >= 8
        and int(within["holdout"]["artificial_value"]) >= 2
        and int(within["holdout"]["prior_or_lure"]) >= 2
    )
    criteria = {
        "smoke_mode": not full_run,
        "structural_passed": structural["passed"],
        "full_source_count_is_40": structural["source_count"] == 40,
        "primary_binary_rows_at_least_40": int(primary["primary_binary"]) >= 40,
        "primary_parseability_at_least_90p": primary_parseable,
        "selected_prompt_audit_passed": all(
            row["artificial_value_prompt_count_expected"] and row["true_capital_not_prompt_listed"]
            for row in selected_rows
        ),
        "panel_a_artificial_adherence_at_least_90p": float(
            selected["panels"]["synthetic_control_lookup"]["artificial_value_rate"]
        )
        >= 0.90,
        "cross_panel_contrast_present": cross_panel_contrast_present,
        "cross_panel_behavior_gate_passed": cross_panel_behavior_gate_passed,
        "within_panel_contrast_present": within_panel_contrast_present,
        "within_panel_behavior_gate_passed": within_panel_behavior_gate_passed,
        "holdout_source_disjoint": structural["criteria"]["source_split_disjoint"],
    }
    selected_payload = {
        "cross_panel_selection": cross_selection,
        "within_panel_selection": within_selection,
        "cross_panel_summary": selected,
    }
    diagnostic_class = classify(criteria, selected_payload)
    return {
        "structural": structural,
        "by_template": by_template,
        "selection": selected_payload,
        "selected_template_summary": selected,
        "selected_template_rows": selected_rows,
        "criteria": criteria,
        "passed": diagnostic_class
        in {
            "behavior_contrast_substrate_passed",
            "prompt_authority_dial_contrast_only",
        },
        "signature_ready": diagnostic_class == "behavior_contrast_substrate_passed",
        "diagnostic_class": diagnostic_class,
    }


def parse_templates(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return TEMPLATES
    templates = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = sorted(set(templates) - set(TEMPLATES))
    if unknown:
        raise ValueError(f"unknown templates: {unknown}")
    if not templates:
        raise ValueError("at least one template is required")
    return templates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument(
        "--artifact-prefix",
        default="mc007_qwen3_1p7b_semi_synthetic_familiar_entity_lookup_v3_parse_repair",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--templates", default=None, help="Comma-separated subset of template names.")
    parser.add_argument("--score-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    started = time.time()
    templates = parse_templates(args.templates)
    records = source_records(args.limit_sources, templates)
    structural = structural_check(records, templates)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_type": RUN_TYPE,
                    "dry_run": True,
                    "model_id": args.model_id,
                    "templates": list(templates),
                    "source_count": structural["source_count"],
                    "record_count": len(records),
                    "structural": structural,
                    "example_records": records[: min(8, len(records))],
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return 0

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

    outputs = score_records(
        records,
        tokenizer,
        model,
        args.max_new_tokens,
        args.score_candidates,
        verbose=not args.quiet,
    )
    full_run = args.limit_sources is None and set(templates) == set(TEMPLATES)
    summary = summarize(records, outputs, templates, full_run)
    result = {
        "card_id": args.card_id,
        "run_type": RUN_TYPE,
        "model_id": args.model_id,
        "max_new_tokens": args.max_new_tokens,
        "decoding": {"do_sample": False},
        "score_candidates": args.score_candidates,
        "limit_sources": args.limit_sources,
        "templates": list(templates),
        "elapsed_s": time.time() - started,
        "sources": base_sources(args.limit_sources),
        "panels": list(PANELS),
        "records": outputs,
        "summary": summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)
    if args.quiet:
        print(
            json.dumps(
                {
                    "diagnostic_class": summary["diagnostic_class"],
                    "passed": summary["passed"],
                    "signature_ready": summary["signature_ready"],
                    "criteria": summary["criteria"],
                    "selection": {
                        "cross_panel_selection": summary["selection"]["cross_panel_selection"],
                        "within_panel_selection": {
                            key: value
                            for key, value in summary["selection"]["within_panel_selection"].items()
                            if key != "all_selection_keys"
                        },
                    },
                    "output_path": str(output_path),
                },
                indent=2,
                ensure_ascii=True,
            )
        )
    else:
        print(json.dumps({**summary, "output_path": str(output_path)}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
