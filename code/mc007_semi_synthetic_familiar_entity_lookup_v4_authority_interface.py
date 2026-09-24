#!/usr/bin/env python
"""MC007 V4 authority-interface behavior redesign."""

from __future__ import annotations

import argparse
import json
import re
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
    table_sources,
)


RUN_TYPE = "semi_synthetic_familiar_entity_lookup_v4_authority_interface"

PANELS = (
    "synthetic_control_lookup",
    "task_query_with_note",
    "real_query_with_note",
    "ambiguous_authority_30",
    "ambiguous_authority_0",
    "real_world_control",
    "answer_absent_null",
)

PRIMARY_EXPLICIT_PANELS = (
    "task_query_with_note",
    "real_query_with_note",
)

PRIMARY_CONFLICT_PANELS = (
    "ambiguous_authority_30",
    "ambiguous_authority_0",
)

PRIMARY_PANELS = PRIMARY_EXPLICIT_PANELS + PRIMARY_CONFLICT_PANELS

TEMPLATES = (
    "city_then_source",
    "source_then_city",
)

SOURCE_TO_LABEL = {
    "TASK": "artificial_value",
    "MEMORY": "real_prior",
    "UNKNOWN": "unknown",
}


def task_note_lines(source: dict[str, Any]) -> list[str]:
    return [
        "Temporary task note:",
        f"{source['country']} -> {source['artificial_value']}",
        "The task note is not a real-world geography statement.",
    ]


def context_lines(sources: list[dict[str, Any]], source: dict[str, Any], panel: str) -> list[str]:
    if panel == "synthetic_control_lookup":
        rows = table_sources(sources, source, include_query=True)
        return [
            "Synthetic lookup task. Code keys have no real-world city memory.",
            *render_table(rows, "synthetic_key"),
            f"Question: For this task, what city is associated with {source['synthetic_key']}?",
            "Use source TASK.",
        ]
    if panel == "task_query_with_note":
        return [
            *task_note_lines(source),
            "Question: For the temporary task, what task city is associated with this country?",
            "Use source TASK.",
        ]
    if panel == "real_query_with_note":
        return [
            *task_note_lines(source),
            f"Question: For ordinary real-world geography, what is the current national capital of {source['country']}?",
            "Use source MEMORY if you know the real-world capital.",
        ]
    if panel == "ambiguous_authority_30":
        return [
            *task_note_lines(source),
            "Task-note authority: 30/100.",
            "Prefer ordinary real-world memory when you know it.",
            f"Question: What is the current real-world national capital of {source['country']}?",
        ]
    if panel == "ambiguous_authority_0":
        return [
            *task_note_lines(source),
            "Task-note authority: 0/100.",
            "For real-world geography questions, ignore the temporary task note.",
            f"Question: What is the current real-world national capital of {source['country']}?",
        ]
    if panel == "real_world_control":
        return [
            "Ordinary real-world geography question.",
            f"Question: What is the current real-world national capital of {source['country']}?",
            "Use source MEMORY.",
        ]
    if panel == "answer_absent_null":
        rows = table_sources(sources, source, include_query=False)
        return [
            "If the queried key is absent from the table, answer UNKNOWN.",
            "This is a task-city lookup, not real-world geography.",
            *render_table(rows, "country"),
            f"Question: According to the table, what is the task city for {source['country']}?",
            "Use source UNKNOWN when the country is absent.",
        ]
    raise ValueError(f"unknown panel: {panel}")


def render_prompt(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    panel: str,
    template: str,
) -> str:
    lines = context_lines(sources, source, panel)
    if template == "city_then_source":
        lines.extend(
            [
                "Output exactly two lines.",
                "CITY: one city name, or UNKNOWN",
                "SOURCE: TASK, MEMORY, or UNKNOWN",
                "CITY:",
            ]
        )
    elif template == "source_then_city":
        lines.extend(
            [
                "Output exactly two lines.",
                "SOURCE: TASK, MEMORY, or UNKNOWN",
                "CITY: one city name, or UNKNOWN",
                "SOURCE:",
            ]
        )
    else:
        raise ValueError(f"unknown template: {template}")
    return "\n".join(lines)


def source_records(limit_sources: int | None = None, templates: tuple[str, ...] = TEMPLATES) -> list[dict[str, Any]]:
    sources = base_sources(limit_sources)
    records = []
    for source in sources:
        city_score_prompt = render_prompt(sources, source, "real_query_with_note", "city_then_source")
        for panel in PANELS:
            for template in templates:
                if panel == "answer_absent_null":
                    expected_label = "unknown"
                    expected_source = "UNKNOWN"
                elif panel in {"real_world_control", "real_query_with_note", "ambiguous_authority_30", "ambiguous_authority_0"}:
                    expected_label = "real_prior"
                    expected_source = "MEMORY"
                else:
                    expected_label = "artificial_value"
                    expected_source = "TASK"
                records.append(
                    {
                        "id": f"mc007_v4_{source['source_id']}_{panel}_{template}",
                        "card_id": CARD_ID,
                        "run_type": RUN_TYPE,
                        "source_id": source["source_id"],
                        "split": source["split"],
                        "panel": panel,
                        "template": template,
                        "country": source["country"],
                        "synthetic_key": source["synthetic_key"],
                        "true_capital": source["true_capital"],
                        "override_capital": source["override_capital"],
                        "lure_capital": source["lure_capital"],
                        "artificial_value": source["artificial_value"],
                        "expected_label": expected_label,
                        "expected_source": expected_source,
                        "prompt": render_prompt(sources, source, panel, template),
                        "city_score_prompt": city_score_prompt,
                    }
                )
    return records


def prompt_audit(record: dict[str, Any]) -> dict[str, Any]:
    artificial_occurrences = prompt_occurrences(record["prompt"], record["artificial_value"])
    true_occurrences = prompt_occurrences(record["prompt"], record["true_capital"])
    score_true_occurrences = prompt_occurrences(record["city_score_prompt"], record["true_capital"])
    if record["panel"] in {"answer_absent_null", "real_world_control"}:
        artificial_expected = 0
    else:
        artificial_expected = 1
    return {
        "artificial_value_occurrences": artificial_occurrences,
        "true_capital_occurrences": true_occurrences,
        "city_score_prompt_true_capital_occurrences": score_true_occurrences,
        "artificial_value_prompt_count_expected": artificial_occurrences == artificial_expected,
        "true_capital_not_prompt_listed": true_occurrences == 0 and score_true_occurrences == 0,
    }


def parse_source_choice(generated_text: str, template: str) -> dict[str, Any]:
    stripped = generated_text.strip().lstrip("`'\" ")
    first_line = stripped.splitlines()[0] if stripped else ""
    source_text = ""
    match = re.search(r"(?im)^\s*SOURCE\s*:\s*(TASK|MEMORY|UNKNOWN)\b", generated_text)
    if match:
        source_text = match.group(1).upper()
    elif template == "source_then_city":
        match = re.search(r"(?i)^\s*(TASK|MEMORY|UNKNOWN)\b", first_line)
        if match:
            source_text = match.group(1).upper()
    if source_text:
        return {
            "source_parseable": True,
            "source_choice": source_text,
            "source_selected_label": SOURCE_TO_LABEL[source_text],
            "source_parse_rule": "source_field_or_first_line",
        }
    return {
        "source_parseable": False,
        "source_choice": None,
        "source_selected_label": "unparsed",
        "source_parse_rule": "no_source_choice",
    }


def city_text_for_parse(generated_text: str, template: str) -> str:
    stripped = generated_text.strip().lstrip("`'\" ")
    if template == "city_then_source":
        first_line = stripped.splitlines()[0] if stripped else ""
        return re.sub(r"(?i)^\s*CITY\s*:\s*", "", first_line).strip()
    match = re.search(r"(?im)^\s*CITY\s*:\s*(.+?)\s*$", generated_text)
    if match:
        return match.group(1).strip()
    return ""


def parse_city_choice(record: dict[str, Any], generated_text: str) -> dict[str, Any]:
    city_text = city_text_for_parse(generated_text, record["template"])
    normalized = normalize_text(city_text)
    matches = []
    sorted_candidates = sorted(
        candidates(record),
        key=lambda row: len(normalize_text(row["answer"])),
        reverse=True,
    )
    for candidate in sorted_candidates:
        answer_norm = normalize_text(candidate["answer"])
        pattern = rf"^{re.escape(answer_norm)}(?=$|[\s\.,;:!\?\)\]\}}])"
        if re.search(pattern, normalized):
            matches.append(candidate)
    if len(matches) == 1:
        return {
            "selected_label": matches[0]["label"],
            "selected_answer": matches[0]["answer"],
            "parseable": True,
            "parse_rule": "city_field_strict_prefix_nfkd",
            "city_field": city_text,
            "normalized_city_field": normalized,
        }
    return {
        "selected_label": "unparsed",
        "selected_answer": None,
        "parseable": False,
        "parse_rule": "no_unique_city_field_prefix_nfkd",
        "city_field": city_text,
        "normalized_city_field": normalized,
    }


def parse_record(record: dict[str, Any], generated_text: str) -> dict[str, Any]:
    city = parse_city_choice(record, generated_text)
    source = parse_source_choice(generated_text, record["template"])
    source_label = source["source_selected_label"]
    city_label = city["selected_label"]
    consistent = (
        (source_label == "artificial_value" and city_label == "artificial_value")
        or (source_label == "real_prior" and city_label in {"real_prior", "lure_value"})
        or (source_label == "unknown" and city_label == "unknown")
    )
    return {
        **city,
        **source,
        "source_city_consistent": bool(source["source_parseable"] and city["parseable"] and consistent),
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
        parsed = parse_record(record, generated["generated_text"])
        audit = prompt_audit(record)
        output = {
            **record,
            "index": index,
            **generated,
            **parsed,
            **audit,
            "is_primary_panel": record["panel"] in PRIMARY_PANELS,
            "is_primary_explicit_panel": record["panel"] in PRIMARY_EXPLICIT_PANELS,
            "is_primary_conflict_panel": record["panel"] in PRIMARY_CONFLICT_PANELS,
            "is_primary_binary": record["panel"] in PRIMARY_PANELS
            and parsed["selected_label"] in {"artificial_value", "real_prior", "lure_value"},
            "is_primary_explicit_binary": record["panel"] in PRIMARY_EXPLICIT_PANELS
            and parsed["selected_label"] in {"artificial_value", "real_prior", "lure_value"},
            "is_primary_conflict_binary": record["panel"] in PRIMARY_CONFLICT_PANELS
            and parsed["selected_label"] in {"artificial_value", "real_prior", "lure_value"},
        }
        if score_candidates:
            output.update(next_token_candidate_logits(model, tokenizer, record["city_score_prompt"], record))
        outputs.append(output)
        if verbose:
            print(
                f"[{index:03d}/{len(records):03d}] {record['id']} split={record['split']} "
                f"panel={record['panel']} template={record['template']} -> "
                f"city={output['selected_label']} {str(output['selected_answer'])!r} "
                f"source={output['source_choice']} generated={generated['generated_text']!r}"
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


def label_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    def stable_value(row: dict[str, Any]) -> str:
        value = row.get(key)
        return "None" if value is None else str(value)

    return dict(sorted(Counter(stable_value(row) for row in rows).items()))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parseable = sum(1 for row in rows if row.get("parseable"))
    source_parseable = sum(1 for row in rows if row.get("source_parseable"))
    consistent = sum(1 for row in rows if row.get("source_city_consistent"))
    artificial = sum(1 for row in rows if row.get("selected_label") == "artificial_value")
    real_prior = sum(1 for row in rows if row.get("selected_label") == "real_prior")
    lure = sum(1 for row in rows if row.get("selected_label") == "lure_value")
    unknown = sum(1 for row in rows if row.get("selected_label") == "unknown")
    primary_binary = sum(1 for row in rows if row.get("is_primary_binary"))
    primary_explicit_binary = sum(1 for row in rows if row.get("is_primary_explicit_binary"))
    primary_conflict_binary = sum(1 for row in rows if row.get("is_primary_conflict_binary"))
    result = {
        "rows": len(rows),
        "label_counts": label_counts(rows, "selected_label"),
        "source_label_counts": label_counts(rows, "source_selected_label"),
        "source_choice_counts": label_counts(rows, "source_choice"),
        "parseable": parseable,
        "parseable_rate": rate(parseable, len(rows)),
        "source_parseable": source_parseable,
        "source_parseable_rate": rate(source_parseable, len(rows)),
        "source_city_consistent": consistent,
        "source_city_consistent_rate": rate(consistent, len(rows)),
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
        "primary_explicit_binary": primary_explicit_binary,
        "primary_conflict_binary": primary_conflict_binary,
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
        explicit = [row for row in rows if row["panel"] in PRIMARY_EXPLICIT_PANELS]
        conflict = [row for row in rows if row["panel"] in PRIMARY_CONFLICT_PANELS]
        primary = explicit + conflict
        result[template] = {
            "rows": len(rows),
            "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "selected_label_counts": label_counts(rows, "selected_label"),
            "source_label_counts": label_counts(rows, "source_selected_label"),
            "panels": panels,
            "primary": summarize_rows(primary),
            "primary_explicit": summarize_rows(explicit),
            "primary_explicit_non_holdout": summarize_rows([row for row in explicit if row["split"] != "holdout"]),
            "primary_explicit_holdout": summarize_rows([row for row in explicit if row["split"] == "holdout"]),
            "primary_conflict": summarize_rows(conflict),
            "primary_conflict_non_holdout": summarize_rows([row for row in conflict if row["split"] != "holdout"]),
            "primary_conflict_holdout": summarize_rows([row for row in conflict if row["split"] == "holdout"]),
        }
    return result


def explicit_template_key(item: dict[str, Any]) -> tuple[float, int, int, float, float]:
    task = item["panels"]["task_query_with_note"]
    real = item["panels"]["real_query_with_note"]
    synthetic = item["panels"]["synthetic_control_lookup"]
    real_control = item["panels"]["real_world_control"]
    null = item["panels"]["answer_absent_null"]
    explicit = item["primary_explicit"]
    non_holdout = item["primary_explicit_non_holdout"]
    holdout = item["primary_explicit_holdout"]
    control_floor = min(
        float(synthetic["artificial_value_rate"]),
        float(real_control["prior_or_lure_rate"]),
        float(null["unknown_rate"]),
    )
    balance = min(int(non_holdout["artificial_value"]), int(non_holdout["prior_or_lure"]))
    holdout_balance = min(int(holdout["artificial_value"]), int(holdout["prior_or_lure"]))
    return (
        control_floor,
        balance,
        holdout_balance,
        float(explicit["parseable_rate"]),
        min(float(task["artificial_value_rate"]), float(real["prior_or_lure_rate"])),
    )


def conflict_template_key(item: dict[str, Any]) -> tuple[float, int, int, float, int]:
    synthetic = item["panels"]["synthetic_control_lookup"]
    real_control = item["panels"]["real_world_control"]
    null = item["panels"]["answer_absent_null"]
    conflict = item["primary_conflict"]
    non_holdout = item["primary_conflict_non_holdout"]
    holdout = item["primary_conflict_holdout"]
    control_floor = min(
        float(synthetic["artificial_value_rate"]),
        float(real_control["prior_or_lure_rate"]),
        float(null["unknown_rate"]),
    )
    balance = min(int(non_holdout["artificial_value"]), int(non_holdout["prior_or_lure"]))
    holdout_balance = min(int(holdout["artificial_value"]), int(holdout["prior_or_lure"]))
    return (
        control_floor,
        balance,
        holdout_balance,
        float(conflict["parseable_rate"]),
        int(conflict["primary_conflict_binary"]),
    )


def select_templates(by_template: dict[str, Any], templates: tuple[str, ...]) -> dict[str, Any]:
    explicit_template = max(templates, key=lambda template: explicit_template_key(by_template[template]))
    conflict_template = max(templates, key=lambda template: conflict_template_key(by_template[template]))
    return {
        "explicit_selection": {
            "selected_template": explicit_template,
            "selection_key": list(explicit_template_key(by_template[explicit_template])),
            "all_selection_keys": {
                template: list(explicit_template_key(by_template[template]))
                for template in templates
            },
            "rule": [
                "max control-floor across synthetic, real-world, and null panels",
                "max non-holdout explicit artificial/prior balance",
                "max holdout explicit artificial/prior balance",
                "max explicit parseability",
                "max min(task artificial rate, real prior-or-lure rate)",
            ],
        },
        "conflict_selection": {
            "selected_template": conflict_template,
            "selection_key": list(conflict_template_key(by_template[conflict_template])),
            "all_selection_keys": {
                template: list(conflict_template_key(by_template[template]))
                for template in templates
            },
            "rule": [
                "max control-floor across synthetic, real-world, and null panels",
                "max non-holdout conflict artificial/prior balance",
                "max holdout conflict artificial/prior balance",
                "max conflict parseability",
                "max conflict binary rows",
            ],
        },
    }


def classify(criteria: dict[str, bool], selected: dict[str, Any]) -> str:
    if criteria["smoke_mode"]:
        return "smoke_only"
    if not criteria["structural_passed"]:
        return "structural_invalid"
    if not criteria["controls_clean"]:
        return "control_panel_failed"
    if criteria["within_conflict_behavior_gate_passed"]:
        return "conflict_behavior_substrate_passed"
    if criteria["explicit_authority_behavior_gate_passed"] and criteria["conflict_contrast_present"]:
        return "explicit_authority_passed_conflict_weak"
    if criteria["explicit_authority_behavior_gate_passed"]:
        return "explicit_authority_requested_mode_only"
    explicit = selected["explicit_summary"]["primary_explicit"]
    if float(explicit["parseable_rate"]) < 0.90:
        return "explicit_authority_parseability_failed"
    if int(explicit["artificial_value"]) == 0 or int(explicit["prior_or_lure"]) == 0:
        return "explicit_authority_contrast_failed"
    return "behavior_substrate_failed"


def summarize(
    records: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    templates: tuple[str, ...],
    full_run: bool,
) -> dict[str, Any]:
    structural = structural_check(records, templates)
    by_template = template_summary(outputs, templates)
    selection = select_templates(by_template, templates)
    explicit_summary = by_template[selection["explicit_selection"]["selected_template"]]
    conflict_summary = by_template[selection["conflict_selection"]["selected_template"]]

    explicit_task = explicit_summary["panels"]["task_query_with_note"]
    explicit_real = explicit_summary["panels"]["real_query_with_note"]
    explicit_all = explicit_summary["primary_explicit"]
    explicit_non_holdout = explicit_summary["primary_explicit_non_holdout"]
    explicit_holdout = explicit_summary["primary_explicit_holdout"]
    conflict_all = conflict_summary["primary_conflict"]
    conflict_non_holdout = conflict_summary["primary_conflict_non_holdout"]
    conflict_holdout = conflict_summary["primary_conflict_holdout"]

    explicit_authority_behavior_gate_passed = (
        float(explicit_all["parseable_rate"]) >= 0.90
        and float(explicit_task["artificial_value_rate"]) >= 0.80
        and float(explicit_real["prior_or_lure_rate"]) >= 0.80
        and int(explicit_non_holdout["artificial_value"]) >= 20
        and int(explicit_non_holdout["prior_or_lure"]) >= 20
        and int(explicit_holdout["artificial_value"]) >= 5
        and int(explicit_holdout["prior_or_lure"]) >= 5
    )
    conflict_contrast_present = int(conflict_all["artificial_value"]) > 0 and int(conflict_all["prior_or_lure"]) > 0
    within_conflict_behavior_gate_passed = (
        conflict_contrast_present
        and float(conflict_all["parseable_rate"]) >= 0.90
        and int(conflict_non_holdout["artificial_value"]) >= 8
        and int(conflict_non_holdout["prior_or_lure"]) >= 8
        and int(conflict_holdout["artificial_value"]) >= 2
        and int(conflict_holdout["prior_or_lure"]) >= 2
    )
    controls_clean = (
        float(explicit_summary["panels"]["synthetic_control_lookup"]["artificial_value_rate"]) >= 0.90
        and float(explicit_summary["panels"]["real_world_control"]["prior_or_lure_rate"]) >= 0.85
        and float(explicit_summary["panels"]["answer_absent_null"]["unknown_rate"]) >= 0.95
    )
    criteria = {
        "smoke_mode": not full_run,
        "structural_passed": structural["passed"],
        "full_source_count_is_40": structural["source_count"] == 40,
        "controls_clean": controls_clean,
        "synthetic_control_artificial_at_least_90p": float(
            explicit_summary["panels"]["synthetic_control_lookup"]["artificial_value_rate"]
        )
        >= 0.90,
        "real_world_control_prior_or_lure_at_least_85p": float(
            explicit_summary["panels"]["real_world_control"]["prior_or_lure_rate"]
        )
        >= 0.85,
        "answer_absent_null_unknown_at_least_95p": float(
            explicit_summary["panels"]["answer_absent_null"]["unknown_rate"]
        )
        >= 0.95,
        "explicit_authority_behavior_gate_passed": explicit_authority_behavior_gate_passed,
        "explicit_authority_parseability_at_least_90p": float(explicit_all["parseable_rate"]) >= 0.90,
        "explicit_task_artificial_at_least_80p": float(explicit_task["artificial_value_rate"]) >= 0.80,
        "explicit_real_prior_or_lure_at_least_80p": float(explicit_real["prior_or_lure_rate"]) >= 0.80,
        "conflict_contrast_present": conflict_contrast_present,
        "within_conflict_behavior_gate_passed": within_conflict_behavior_gate_passed,
        "holdout_source_disjoint": structural["criteria"]["source_split_disjoint"],
    }
    selected = {
        **selection,
        "explicit_summary": explicit_summary,
        "conflict_summary": conflict_summary,
    }
    diagnostic_class = classify(criteria, selected)
    return {
        "structural": structural,
        "by_template": by_template,
        "selection": selected,
        "criteria": criteria,
        "passed": diagnostic_class
        in {
            "conflict_behavior_substrate_passed",
            "explicit_authority_passed_conflict_weak",
            "explicit_authority_requested_mode_only",
        },
        "signature_ready": diagnostic_class == "conflict_behavior_substrate_passed",
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


def quiet_summary(summary: dict[str, Any], output_path: Path) -> dict[str, Any]:
    return {
        "diagnostic_class": summary["diagnostic_class"],
        "passed": summary["passed"],
        "signature_ready": summary["signature_ready"],
        "criteria": summary["criteria"],
        "selection": {
            "explicit_selection": summary["selection"]["explicit_selection"],
            "conflict_selection": summary["selection"]["conflict_selection"],
        },
        "explicit_primary": summary["selection"]["explicit_summary"]["primary_explicit"],
        "conflict_primary": summary["selection"]["conflict_summary"]["primary_conflict"],
        "output_path": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument(
        "--artifact-prefix",
        default="mc007_qwen3_1p7b_semi_synthetic_familiar_entity_lookup_v4_authority_interface",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=20)
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
        "primary_explicit_panels": list(PRIMARY_EXPLICIT_PANELS),
        "primary_conflict_panels": list(PRIMARY_CONFLICT_PANELS),
        "records": outputs,
        "summary": summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)
    payload = quiet_summary(summary, output_path) if args.quiet else {**summary, "output_path": str(output_path)}
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
