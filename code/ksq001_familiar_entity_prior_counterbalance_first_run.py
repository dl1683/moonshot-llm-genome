#!/usr/bin/env python
"""KSQ001 familiar-entity prior counterbalance first run.

This is a behavior-substrate admission runner for the Level 2 knowledge ladder:
familiar entity names with artificial values. It asks whether familiar entities
act only as lookup keys, whether real-world priors compete with local artificial
values, and whether that competition survives nulls, holdout, and output/candidate
baseline reporting.

The runner is structural by default. Model scoring is optional and remains
behavior-only; it does not license hidden-state signatures or interventions.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from mc006_parametric_fact_override_v2_repair import SOURCES, split_for_index
from mc007_semi_synthetic_familiar_entity_lookup import (
    ARTIFICIAL_VALUES,
    generate_answer,
    next_token_candidate_logits,
    normalize_text,
    strict_parse,
    table_sources,
)
from mc016_alphabet_gated_numeric_arbitration import load_model_and_tokenizer


CARD_ID = "KSQ001"
CANDIDATE_ID = "ksq001_familiar_entity_prior_counterbalance"
RUN_TYPE = "ksq001_familiar_entity_prior_counterbalance_structural_gate"
BEHAVIOR_RUN_TYPE = "ksq001_familiar_entity_prior_counterbalance_behavior"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE")
DEFAULT_RESULT_PATH = RESULT_DIR / "ksq001_familiar_entity_prior_counterbalance_first_run.json"
SMOKE_LIMIT10_RESULT_PATH = (
    RESULT_DIR / "ksq001_familiar_entity_prior_counterbalance_smoke_limit10.json"
)
FULL_BEHAVIOR_RESULT_PATH = (
    RESULT_DIR / "ksq001_familiar_entity_prior_counterbalance_full_behavior.json"
)
STATUS_CARD = Path(
    "research/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE_FIRST_RUN_STATUS.md"
)
PREREG_PATH = Path(
    "research/prereg/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE_FIRST_RUN.md"
)

PANELS = (
    "source_local_artificial_lookup",
    "semantic_prior_direct_control",
    "semantic_prior_lure",
    "answer_absent_irrelevant_nulls",
)
PRIMARY_PANEL = "semantic_prior_lure"
TEMPLATES = ("registry_question", "association_question", "compact_question")
STATUS_LEXEME_RE = re.compile(r"\b(trusted|untrusted|reliable|unreliable|status)\b", re.IGNORECASE)
AUTHORITY_RE = re.compile(r"\b(authority|authoritative|override|ignore real|must use only)\b", re.IGNORECASE)


def source_rows(limit_sources: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (source_id, country, true_capital, override_capital, lure_capital) in enumerate(SOURCES):
        artificial_value = ARTIFICIAL_VALUES[index]
        normalized = {
            normalize_text(true_capital),
            normalize_text(override_capital),
            normalize_text(lure_capital),
            normalize_text(artificial_value),
            normalize_text("UNKNOWN"),
        }
        if len(normalized) != 5:
            raise ValueError(f"candidate collision for {source_id}")
        rows.append(
            {
                "source_index": index,
                "source_id": source_id,
                "country": country,
                "true_capital": true_capital,
                "override_capital": override_capital,
                "lure_capital": lure_capital,
                "artificial_value": artificial_value,
                "synthetic_key": f"KSQ001-{index + 11:02d}-{source_id[:3].upper()}",
                "split": split_for_index(index),
            }
        )
    if limit_sources is not None:
        if limit_sources < 10:
            raise ValueError("--limit-sources must be at least 10 for split balance")
        return rows[:limit_sources]
    return rows


def render_registry(rows: list[dict[str, Any]], *, key_field: str) -> list[str]:
    label = "Country" if key_field == "country" else "Registry key"
    lines = [f"{label} -> temporary city"]
    for row in rows:
        lines.append(f"{row[key_field]} -> {row['artificial_value']}")
    return lines


def response_suffix() -> list[str]:
    return [
        "Return only one city name, or UNKNOWN.",
        "Do not explain.",
        "Answer:",
    ]


def prompt_for_panel(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    panel: str,
    template: str,
) -> tuple[str, str]:
    if panel == "source_local_artificial_lookup":
        rows = table_sources(sources, source, include_query=True)
        if template == "registry_question":
            lines = [
                "A temporary registry assigns city names for this task.",
                *render_registry(rows, key_field="country"),
                f"Question: In the temporary registry, which city is assigned to {source['country']}?",
            ]
        elif template == "association_question":
            lines = [
                "For this task, use the temporary city association below.",
                *render_registry(rows, key_field="country"),
                f"Question: What temporary city is associated with {source['country']} in the registry?",
            ]
        elif template == "compact_question":
            lines = [
                "Temporary city registry:",
                *render_registry(rows, key_field="country"),
                f"Query country: {source['country']}",
            ]
        else:
            raise ValueError(f"unknown template {template}")
        expected_label = "artificial_value"
    elif panel == "semantic_prior_direct_control":
        if template == "registry_question":
            lines = [
                "Ordinary geography question.",
                f"Question: What is the current national capital of {source['country']}?",
            ]
        elif template == "association_question":
            lines = [
                "Use ordinary country knowledge.",
                f"Question: Which city is the national capital of {source['country']}?",
            ]
        elif template == "compact_question":
            lines = [
                "Geography query:",
                f"Country: {source['country']}",
                "Capital:",
            ]
        else:
            raise ValueError(f"unknown template {template}")
        expected_label = "real_prior"
    elif panel == "semantic_prior_lure":
        if template == "registry_question":
            lines = [
                "A temporary registry assigns a city name.",
                f"Registry line: {source['country']} -> {source['artificial_value']}",
                f"Question: Which city should be associated with {source['country']}?",
            ]
        elif template == "association_question":
            lines = [
                "A familiar country name and a temporary registry value are both visible.",
                f"Temporary city association: {source['country']} -> {source['artificial_value']}",
                f"Question: What city is associated with {source['country']}?",
            ]
        elif template == "compact_question":
            lines = [
                "Association task:",
                f"{source['country']} -> {source['artificial_value']}",
                f"Query: {source['country']}",
            ]
        else:
            raise ValueError(f"unknown template {template}")
        expected_label = "ambiguous"
    elif panel == "answer_absent_irrelevant_nulls":
        rows = table_sources(sources, source, include_query=False)
        if template == "registry_question":
            lines = [
                "A temporary registry assigns city names for this task.",
                "If the queried country is absent from the registry, answer UNKNOWN.",
                *render_registry(rows, key_field="country"),
                f"Question: In the temporary registry, which city is assigned to {source['country']}?",
            ]
        elif template == "association_question":
            lines = [
                "For this task, use only entries that appear in the temporary registry.",
                "If no entry appears for the queried country, answer UNKNOWN.",
                *render_registry(rows, key_field="country"),
                f"Question: What temporary city is associated with {source['country']} in the registry?",
            ]
        elif template == "compact_question":
            lines = [
                "Temporary city registry:",
                *render_registry(rows, key_field="country"),
                f"Query country: {source['country']}",
                "If absent: UNKNOWN",
            ]
        else:
            raise ValueError(f"unknown template {template}")
        expected_label = "unknown"
    else:
        raise ValueError(f"unknown panel {panel}")
    return "\n".join([*lines, *response_suffix()]), expected_label


def source_records(
    limit_sources: int | None = None,
    templates: tuple[str, ...] = TEMPLATES,
    run_type: str = RUN_TYPE,
) -> list[dict[str, Any]]:
    sources = source_rows(limit_sources)
    records: list[dict[str, Any]] = []
    for source in sources:
        for template in templates:
            for panel in PANELS:
                prompt, expected_label = prompt_for_panel(
                    sources,
                    source,
                    panel=panel,
                    template=template,
                )
                records.append(
                    {
                        "id": f"KSQ001_{template}_{panel}_{source['source_id']}",
                        "card_id": CARD_ID,
                        "candidate_id": CANDIDATE_ID,
                        "run_type": run_type,
                        "model_id": MODEL_ID,
                        "template": template,
                        "panel": panel,
                        "split": source["split"],
                        "source_id": source["source_id"],
                        "source_index": source["source_index"],
                        "country": source["country"],
                        "synthetic_key": source["synthetic_key"],
                        "true_capital": source["true_capital"],
                        "override_capital": source["override_capital"],
                        "lure_capital": source["lure_capital"],
                        "artificial_value": source["artificial_value"],
                        "expected_label": expected_label,
                        "candidate_answers": [
                            source["artificial_value"],
                            source["true_capital"],
                            source["lure_capital"],
                            "UNKNOWN",
                        ],
                        "prompt": prompt,
                    }
                )
    return records


def prompt_occurrences(prompt: str, value: str) -> int:
    return len(re.findall(rf"\b{re.escape(value)}\b", prompt))


def prompt_audit(record: dict[str, Any]) -> dict[str, Any]:
    prompt = record["prompt"]
    panel = record["panel"]
    artificial_occurrences = prompt_occurrences(prompt, record["artificial_value"])
    true_occurrences = prompt_occurrences(prompt, record["true_capital"])
    lure_occurrences = prompt_occurrences(prompt, record["lure_capital"])
    if panel in {"source_local_artificial_lookup", "semantic_prior_lure"}:
        expected_artificial_occurrences = 1
    else:
        expected_artificial_occurrences = 0
    return {
        "artificial_value_occurrences": artificial_occurrences,
        "true_capital_occurrences": true_occurrences,
        "lure_capital_occurrences": lure_occurrences,
        "artificial_value_prompt_count_expected": artificial_occurrences == expected_artificial_occurrences,
        "true_capital_not_prompt_listed": true_occurrences == 0,
        "lure_capital_not_prompt_listed": lure_occurrences == 0,
        "prompt_has_status_lexeme": bool(STATUS_LEXEME_RE.search(prompt)),
        "prompt_has_authority_lexeme": bool(AUTHORITY_RE.search(prompt)),
    }


def structural_check(records: list[dict[str, Any]], templates: tuple[str, ...]) -> dict[str, Any]:
    source_ids = {row["source_id"] for row in records}
    duplicate_ids = [
        row_id for row_id, count in Counter(row["id"] for row in records).items() if count > 1
    ]
    panel_counts = Counter(row["panel"] for row in records)
    template_counts = Counter(row["template"] for row in records)
    split_source_ids: dict[str, set[str]] = {}
    candidate_collision_rows = []
    prompt_audit_failures = []
    for row in records:
        split_source_ids.setdefault(row["split"], set()).add(row["source_id"])
        normalized_candidates = [normalize_text(candidate) for candidate in row["candidate_answers"]]
        if len(set(normalized_candidates)) != len(normalized_candidates):
            candidate_collision_rows.append(row["id"])
        audit = prompt_audit(row)
        if not (
            audit["artificial_value_prompt_count_expected"]
            and audit["true_capital_not_prompt_listed"]
            and audit["lure_capital_not_prompt_listed"]
            and not audit["prompt_has_status_lexeme"]
            and not audit["prompt_has_authority_lexeme"]
        ):
            prompt_audit_failures.append(row["id"])
    expected_rows = len(source_ids) * len(templates) * len(PANELS)
    criteria = {
        "expected_row_count": len(records) == expected_rows,
        "all_panels_present": set(panel_counts) == set(PANELS),
        "all_templates_present": set(template_counts) == set(templates),
        "no_duplicate_record_ids": not duplicate_ids,
        "source_split_disjoint": sum(len(ids) for ids in split_source_ids.values()) == len(source_ids),
        "holdout_sources_present": bool(split_source_ids.get("holdout")),
        "calibration_sources_present": bool(split_source_ids.get("calibration")),
        "candidate_answers_parseable": all(row["candidate_answers"] for row in records),
        "no_candidate_collisions": not candidate_collision_rows,
        "prompt_audit_passed": not prompt_audit_failures,
        "primary_expected_label_ambiguous": all(
            row["expected_label"] == "ambiguous" for row in records if row["panel"] == PRIMARY_PANEL
        ),
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "record_count": len(records),
        "source_count": len(source_ids),
        "panel_counts": dict(sorted(panel_counts.items())),
        "template_counts": dict(sorted(template_counts.items())),
        "split_source_counts": {
            split: len(ids) for split, ids in sorted(split_source_ids.items())
        },
        "expected_rows": expected_rows,
        "duplicate_ids": duplicate_ids[:20],
        "candidate_collision_rows": candidate_collision_rows[:20],
        "prompt_audit_failure_rows": prompt_audit_failures[:20],
    }


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row.get("selected_label", "missing") for row in rows).items()))


def rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parseable = sum(1 for row in rows if row.get("parseable"))
    artificial = sum(1 for row in rows if row.get("selected_label") == "artificial_value")
    real_prior = sum(1 for row in rows if row.get("selected_label") == "real_prior")
    lure = sum(1 for row in rows if row.get("selected_label") == "lure_value")
    unknown = sum(1 for row in rows if row.get("selected_label") == "unknown")
    unparsed = sum(1 for row in rows if row.get("selected_label") == "unparsed")
    candidate_margins = "artificial_minus_real_prior_logit" in rows[0] if rows else False
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
        "unparsed": unparsed,
        "unparsed_rate": rate(unparsed, len(rows)),
    }
    if candidate_margins:
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
    result: dict[str, Any] = {}
    for template in templates:
        rows = [row for row in outputs if row["template"] == template]
        panels = {
            panel: summarize_rows([row for row in rows if row["panel"] == panel])
            for panel in PANELS
        }
        primary = [row for row in rows if row["panel"] == PRIMARY_PANEL]
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


def selection_key(summary: dict[str, Any], template: str, templates: tuple[str, ...]) -> tuple[float, ...]:
    item = summary[template]
    local = item["panels"]["source_local_artificial_lookup"]
    prior = item["panels"]["semantic_prior_direct_control"]
    null = item["panels"]["answer_absent_irrelevant_nulls"]
    primary = item["primary"]
    mixture_count = min(int(primary["artificial_value"]), int(primary["prior_or_lure"]))
    return (
        float(local["artificial_value_rate"]),
        float(prior["real_prior_rate"]),
        float(null["unknown_rate"]),
        float(primary["parseable_rate"]),
        float(mixture_count),
        -templates.index(template),
    )


def select_template(summary: dict[str, Any], templates: tuple[str, ...]) -> dict[str, Any]:
    selected = max(templates, key=lambda template: selection_key(summary, template, templates))
    return {
        "selected_template": selected,
        "selection_key": list(selection_key(summary, selected, templates)),
        "all_selection_keys": {
            template: list(selection_key(summary, template, templates)) for template in templates
        },
        "rule": [
            "max local artificial direct control",
            "max semantic prior direct control",
            "max answer-absent null behavior",
            "max primary parseability",
            "max primary mixture count",
            "earliest template",
        ],
    }


def classify(criteria: dict[str, bool], selected: dict[str, Any]) -> str:
    if not criteria["structural_passed"]:
        return "structural_invalid"
    if not criteria["source_local_artificial_control_passed"]:
        return "familiar_entity_local_lookup_failed"
    if not criteria["semantic_prior_direct_control_passed"]:
        return "semantic_prior_absent"
    if not criteria["answer_absent_null_passed"]:
        return "answer_absent_null_failed"
    if not criteria["primary_parseability_at_least_90p"]:
        return "familiar_entity_conflict_parseability_failed"
    if not criteria["candidate_and_output_margins_reported"]:
        return "familiar_entity_behavior_passed_baselines_missing"
    if not criteria["semantic_prior_competition_present"]:
        primary = selected["primary"]
        if int(primary["artificial_value"]) > 0:
            return "familiar_entity_lookup_key_collapse"
        return "semantic_prior_dominant"
    if not criteria["conflict_neither_branch_above_90p"]:
        return "familiar_entity_one_branch_dominance"
    if criteria["smoke_mode"]:
        return "smoke_behavior_contrast_candidate"
    if not criteria["source_disjoint_holdout_mixture_passed"]:
        return "familiar_entity_holdout_failed"
    return "familiar_entity_prior_counterbalance_behavior_ready"


def summarize(
    records: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    templates: tuple[str, ...],
    full_run: bool,
    score_candidates: bool,
) -> dict[str, Any]:
    structural = structural_check(records, templates)
    by_template = template_summary(outputs, templates)
    selection = select_template(by_template, templates)
    selected = by_template[selection["selected_template"]]
    local = selected["panels"]["source_local_artificial_lookup"]
    prior = selected["panels"]["semantic_prior_direct_control"]
    null = selected["panels"]["answer_absent_irrelevant_nulls"]
    primary = selected["primary"]
    holdout = selected["primary_holdout"]
    selected_rows = [row for row in outputs if row["template"] == selection["selected_template"]]
    prompt_audit_passed = all(
        row["artificial_value_prompt_count_expected"]
        and row["true_capital_not_prompt_listed"]
        and row["lure_capital_not_prompt_listed"]
        and not row["prompt_has_status_lexeme"]
        and not row["prompt_has_authority_lexeme"]
        for row in selected_rows
    )
    artificial_count = int(primary["artificial_value"])
    prior_or_lure_count = int(primary["prior_or_lure"])
    primary_rows = int(primary["rows"])
    primary_max_branch_rate = max(
        float(primary["artificial_value_rate"]),
        float(primary["prior_or_lure_rate"]),
    )
    holdout_mixture = int(holdout["artificial_value"]) > 0 and int(holdout["prior_or_lure"]) > 0
    criteria = {
        "smoke_mode": not full_run,
        "structural_passed": structural["passed"],
        "full_source_count_is_40": structural["source_count"] == 40,
        "source_disjoint_holdout": structural["criteria"]["source_split_disjoint"],
        "selected_prompt_audit_passed": prompt_audit_passed,
        "source_local_artificial_control_passed": float(local["parseable_rate"]) >= 0.90
        and float(local["artificial_value_rate"]) >= 0.85,
        "semantic_prior_direct_control_passed": float(prior["parseable_rate"]) >= 0.90
        and float(prior["real_prior_rate"]) >= 0.75,
        "answer_absent_null_passed": float(null["parseable_rate"]) >= 0.90
        and float(null["unknown_rate"]) >= 0.80
        and float(null["artificial_value_rate"]) <= 0.10
        and float(null["prior_or_lure_rate"]) <= 0.10,
        "primary_parseability_at_least_90p": float(primary["parseable_rate"]) >= 0.90,
        "primary_binary_rows_at_least_40": primary_rows >= 40,
        "semantic_prior_competition_present": artificial_count > 0 and prior_or_lure_count > 0,
        "conflict_neither_branch_above_90p": primary_max_branch_rate <= 0.90,
        "source_disjoint_holdout_mixture_passed": holdout_mixture,
        "candidate_and_output_margins_reported": score_candidates
        and "mean_artificial_minus_real_prior_logit" in primary,
    }
    diagnostic_class = classify(criteria, selected)
    behavior_ready = diagnostic_class == "familiar_entity_prior_counterbalance_behavior_ready"
    behavior_candidate = diagnostic_class in {
        "smoke_behavior_contrast_candidate",
        "familiar_entity_prior_counterbalance_behavior_ready",
    }
    return {
        "structural": structural,
        "by_template": by_template,
        "selection": selection,
        "selected_template_summary": selected,
        "selected_template_rows": selected_rows,
        "criteria": criteria,
        "passed": behavior_ready,
        "behavior_ready": behavior_ready,
        "behavior_candidate": behavior_candidate,
        "signature_ready": False,
        "intervention_ready": False,
        "hidden_state_allowed": False,
        "diagnostic_class": diagnostic_class,
    }


def quiet_summary(summary: dict[str, Any], output_path: Path | None = None) -> dict[str, Any]:
    payload = {
        "diagnostic_class": summary["diagnostic_class"],
        "passed": summary["passed"],
        "behavior_ready": summary["behavior_ready"],
        "behavior_candidate": summary["behavior_candidate"],
        "signature_ready": summary["signature_ready"],
        "hidden_state_allowed": summary["hidden_state_allowed"],
        "criteria": summary["criteria"],
        "selection": summary["selection"],
        "selected_primary": summary["selected_template_summary"]["primary"],
        "selected_controls": {
            panel_name: summary["selected_template_summary"]["panels"][panel_name]
            for panel_name in PANELS
        },
    }
    if output_path is not None:
        payload["output_path"] = str(output_path)
    return payload


def score_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    score_candidates: bool,
    *,
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
            "is_primary_panel": record["panel"] == PRIMARY_PANEL,
            "is_primary_binary": record["panel"] == PRIMARY_PANEL
            and parsed["selected_label"] in {"artificial_value", "real_prior", "lure_value"},
            "is_expected_correct": record["expected_label"] == parsed["selected_label"],
        }
        if score_candidates:
            output.update(next_token_candidate_logits(model, tokenizer, record["prompt"], record))
        outputs.append(output)
        if verbose:
            print(
                f"[{index:03d}/{len(records):03d}] {record['id']} split={record['split']} "
                f"panel={record['panel']} -> {output['selected_label']} "
                f"{str(output['selected_answer'])!r} generated={generated['generated_text']!r}"
            )
    return outputs


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def write_prereg(path: Path) -> None:
    lines = [
        "# KSQ001 Familiar Entity Prior Counterbalance First Run",
        "",
        "Status: behavior-substrate first-run preregistration; no hidden-state work.",
        "",
        "Runner:",
        "",
        "> `code/ksq001_familiar_entity_prior_counterbalance_first_run.py`",
        "",
        "Default result:",
        "",
        "> `results/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE/ksq001_familiar_entity_prior_counterbalance_first_run.json`",
        "",
        "Default 10-source smoke result:",
        "",
        "> `results/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE/ksq001_familiar_entity_prior_counterbalance_smoke_limit10.json`",
        "",
        "Full behavior result:",
        "",
        "> `results/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE/ksq001_familiar_entity_prior_counterbalance_full_behavior.json`",
        "",
        "Status card:",
        "",
        "> `research/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE_FIRST_RUN_STATUS.md`",
        "",
        "## Purpose",
        "",
        "Test whether familiar country names produce measurable competition",
        "between prompt-local artificial city values and learned capital priors",
        "without explicit status labels or source-authority wording.",
        "",
        "## Panels",
        "",
    ]
    for panel_name in PANELS:
        lines.append(f"- `{panel_name}`")
    lines.extend(
        [
            "",
            "## Decision Boundary",
            "",
            "Promote only to behavior-substrate admission if local artificial",
            "lookup, direct semantic-prior recall, answer-absent nulls, familiar",
            "entity conflict mixture, source-disjoint holdout, prompt audit, and",
            "candidate/output baseline reporting pass together.",
            "",
            "Death rule: kill if behavior reduces to local copy, semantic prior",
            "recall, authority wording, parse/answer-shape effects, or null",
            "leakage.",
            "",
            "Forbidden claims:",
            "",
            "- KSQ001 is a mechanism card.",
            "- KSQ001 licenses hidden-state search before the behavior gate passes.",
            "- KSQ001 proves a learned-memory control surface.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def add_structural_section(lines: list[str], structural: dict[str, Any]) -> None:
    lines.extend(
        [
            "## Structural Criteria",
            "",
            "| Criterion | Passed |",
            "| --- | --- |",
        ]
    )
    for key, value in structural["criteria"].items():
        lines.append(f"| `{key}` | `{str(value).lower()}` |")
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- records: `{structural['record_count']}`",
            f"- sources: `{structural['source_count']}`",
            f"- panels: `{json.dumps(structural['panel_counts'], sort_keys=True)}`",
            f"- templates: `{json.dumps(structural['template_counts'], sort_keys=True)}`",
            f"- split source counts: `{json.dumps(structural['split_source_counts'], sort_keys=True)}`",
        ]
    )


def add_behavior_section(lines: list[str], summary: dict[str, Any], result_path: Path) -> None:
    selected = summary["selected_template_summary"]
    criteria = summary["criteria"]
    lines.extend(
        [
            "## Behavior Result",
            "",
            f"- result: `{result_path.as_posix()}`",
            f"- diagnostic class: `{summary['diagnostic_class']}`",
            f"- behavior ready: `{str(summary['behavior_ready']).lower()}`",
            f"- behavior candidate: `{str(summary['behavior_candidate']).lower()}`",
            f"- selected template: `{summary['selection']['selected_template']}`",
            f"- candidate/output margins reported: `{str(criteria['candidate_and_output_margins_reported']).lower()}`",
            "",
            "| Panel | Label counts |",
            "| --- | --- |",
        ]
    )
    for panel_name in PANELS:
        labels = selected["panels"][panel_name]["label_counts"]
        lines.append(f"| `{panel_name}` | `{json.dumps(labels, sort_keys=True)}` |")
    lines.extend(
        [
            "",
            "Interpretation: KSQ001 only becomes behavior-ready if the familiar",
            "entity conflict panel contains both prompt-local artificial values",
            "and learned-prior answers while direct controls and nulls remain",
            "clean. A one-branch outcome is a diagnostic, not a mechanism.",
        ]
    )


def write_status_card(path: Path, result: dict[str, Any], output_path: Path) -> None:
    smoke_result = None
    if "summary" not in result and SMOKE_LIMIT10_RESULT_PATH.exists():
        smoke_result = json.loads(SMOKE_LIMIT10_RESULT_PATH.read_text(encoding="utf-8"))
    full_behavior_result = None
    if "summary" not in result and FULL_BEHAVIOR_RESULT_PATH.exists():
        full_behavior_result = json.loads(FULL_BEHAVIOR_RESULT_PATH.read_text(encoding="utf-8"))
    lines = [
        "# KSQ001 Familiar Entity Prior Counterbalance First-Run Status",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "Runner:",
        "",
        "> `code/ksq001_familiar_entity_prior_counterbalance_first_run.py`",
        "",
        "Result:",
        "",
        f"> `{output_path.as_posix()}`",
        "",
    ]
    if "summary" not in result:
        structural = result["structural"]
        status = "structural_passed" if structural["passed"] else "structural_failed"
        if full_behavior_result is not None:
            status = f"{status}_with_full_behavior"
        elif smoke_result is not None:
            status = f"{status}_with_behavior_smoke"
        lines.extend(
            [
                f"Status: {status}.",
                "",
                "## Verdict",
                "",
            ]
        )
        if structural["passed"] and full_behavior_result is not None:
            lines.append(
                "The structural gate passed, but the full 40-source behavior "
                "run failed admission. Direct local lookup, direct semantic "
                "prior recall, nulls, holdout, and margin reporting survived; "
                "the selected conflict template failed parseability. Hidden-state "
                "work remains forbidden."
            )
        elif structural["passed"] and smoke_result is not None:
            lines.append(
                "The structural gate passed and the 10-source model smoke is a "
                "behavior-contrast candidate. The full 40-source behavior gate "
                "still has not passed, and hidden-state work remains forbidden."
            )
        elif structural["passed"]:
            lines.append("The structural gate passed. Model-scored behavior remains the next gate.")
        else:
            lines.append("The structural gate failed. Do not run model scoring until repaired.")
        lines.append("")
        add_structural_section(lines, structural)
        if smoke_result is not None:
            lines.append("")
            add_behavior_section(lines, smoke_result["summary"], SMOKE_LIMIT10_RESULT_PATH)
        if full_behavior_result is not None:
            lines.append("")
            add_behavior_section(lines, full_behavior_result["summary"], FULL_BEHAVIOR_RESULT_PATH)
    else:
        summary = result["summary"]
        lines.extend(
            [
                f"Status: {summary['diagnostic_class']}.",
                "",
                "## Verdict",
                "",
            ]
        )
        if summary["behavior_ready"]:
            lines.append("The behavior gate passed. This still licenses only a later signature-screen decision.")
        elif summary["behavior_candidate"]:
            lines.append("The run is a behavior candidate or smoke candidate, not a hidden-state license.")
        else:
            lines.append("The behavior gate failed. Hidden-state work remains forbidden.")
        lines.append("")
        add_structural_section(lines, summary["structural"])
        lines.append("")
        add_behavior_section(lines, summary, output_path)
    lines.extend(
        [
            "",
            "## Forbidden Claims",
            "",
            "- KSQ001 is a mechanism card.",
            "- KSQ001 supports intervention.",
            "- KSQ001 found an internal knowledge-control surface.",
            "- Any hidden-state or causal claim follows from this first run alone.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--score-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--score-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-manifest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--status-card", type=Path, default=STATUS_CARD)
    parser.add_argument("--prereg-path", type=Path, default=PREREG_PATH)
    parser.add_argument("--write-status-card", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-prereg", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.time()
    run_type = BEHAVIOR_RUN_TYPE if args.score_model else RUN_TYPE
    records = source_records(args.limit_sources, TEMPLATES, run_type)
    structural = structural_check(records, TEMPLATES)

    if args.write_prereg:
        write_prereg(args.prereg_path)

    if not args.score_model:
        result = {
            "schema_version": 1,
            "card_id": CARD_ID,
            "candidate_id": CANDIDATE_ID,
            "run_type": RUN_TYPE,
            "model_id": args.model_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "purpose": "Structural gate for the KSQ001 familiar-entity prior-counterbalance first run.",
            "templates": list(TEMPLATES),
            "panels": list(PANELS),
            "primary_panel": PRIMARY_PANEL,
            "hidden_state_allowed": False,
            "structural": structural,
            "records": records,
        }
        if args.write_manifest:
            write_json(args.output_path, result)
            if args.write_status_card:
                write_status_card(args.status_card, result, args.output_path)
        print(
            json.dumps(
                {
                    "passed": structural["passed"],
                    "record_count": structural["record_count"],
                    "source_count": structural["source_count"],
                    "criteria": structural["criteria"],
                    "output_path": str(args.output_path) if args.write_manifest else None,
                    "status_card": str(args.status_card) if args.write_status_card else None,
                    "prereg": str(args.prereg_path) if args.write_prereg else None,
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return 0 if structural["passed"] else 1

    if not structural["passed"]:
        print(json.dumps({"passed": False, "diagnostic_class": "structural_invalid", "structural": structural}, indent=2))
        return 1

    model, tokenizer = load_model_and_tokenizer(args.model_id, args.local_files_only)
    outputs = score_records(
        records,
        tokenizer,
        model,
        args.max_new_tokens,
        args.score_candidates,
        verbose=not args.quiet,
    )
    full_run = args.limit_sources is None
    summary = summarize(records, outputs, TEMPLATES, full_run, args.score_candidates)
    result = {
        "schema_version": 1,
        "card_id": CARD_ID,
        "candidate_id": CANDIDATE_ID,
        "run_type": BEHAVIOR_RUN_TYPE,
        "model_id": args.model_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "max_new_tokens": args.max_new_tokens,
        "decoding": {"do_sample": False},
        "score_candidates": args.score_candidates,
        "limit_sources": args.limit_sources,
        "templates": list(TEMPLATES),
        "elapsed_s": time.time() - started,
        "purpose": "Behavior-only KSQ001 familiar-entity prior-counterbalance admission run.",
        "panels": list(PANELS),
        "primary_panel": PRIMARY_PANEL,
        "sources": source_rows(args.limit_sources),
        "hidden_state_allowed": False,
        "structural": structural,
        "records": outputs,
        "summary": summary,
    }
    write_json(args.output_path, result)
    if args.write_status_card:
        write_status_card(args.status_card, result, args.output_path)
    print(json.dumps(quiet_summary(summary, args.output_path), indent=2, ensure_ascii=True))
    return 0 if (summary["passed"] or not full_run) else 2


if __name__ == "__main__":
    raise SystemExit(main())
