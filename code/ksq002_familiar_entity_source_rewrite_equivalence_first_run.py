#!/usr/bin/env python
"""KSQ002 familiar-entity source-rewrite equivalence first run.

This is a behavior-substrate admission runner for the Level 2 knowledge ladder.
KSQ001 showed that familiar country names can support source-local artificial
lookup while familiar priors compete weakly and parseability can fail at full
scale. KSQ002 asks a narrower source-channel question: does the artificial
lookup survive neutral rewrites, and does it disappear under source deletion
or query-only controls?

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


CARD_ID = "KSQ002"
CANDIDATE_ID = "ksq002_familiar_entity_source_rewrite_equivalence"
RUN_TYPE = "ksq002_familiar_entity_source_rewrite_equivalence_structural_gate"
BEHAVIOR_RUN_TYPE = "ksq002_familiar_entity_source_rewrite_equivalence_behavior"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE")
DEFAULT_RESULT_PATH = RESULT_DIR / "ksq002_familiar_entity_source_rewrite_equivalence_first_run.json"
SMOKE_LIMIT10_RESULT_PATH = RESULT_DIR / "ksq002_familiar_entity_source_rewrite_equivalence_smoke_limit10.json"
FULL_BEHAVIOR_RESULT_PATH = RESULT_DIR / "ksq002_familiar_entity_source_rewrite_equivalence_full_behavior.json"
STATUS_CARD = Path("research/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE_FIRST_RUN_STATUS.md")
PREREG_PATH = Path("research/prereg/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE_FIRST_RUN.md")

PANELS = (
    "baseline_source_value_lookup",
    "neutral_rewrite_lookup",
    "source_deletion",
    "query_only_control",
    "source_disjoint_rewrite_holdout",
    "rewrite_output_geometry_audit",
)
SOURCE_VALUE_PANELS = (
    "baseline_source_value_lookup",
    "neutral_rewrite_lookup",
    "source_disjoint_rewrite_holdout",
    "rewrite_output_geometry_audit",
)
REWRITE_PANELS = (
    "neutral_rewrite_lookup",
    "source_disjoint_rewrite_holdout",
    "rewrite_output_geometry_audit",
)
TEMPLATES = ("registry_question", "sentence_rewrite", "compact_rewrite")
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
                "synthetic_key": f"KSQ002-{index + 11:02d}-{source_id[:3].upper()}",
                "split": split_for_index(index),
            }
        )
    if limit_sources is not None:
        if limit_sources < 10:
            raise ValueError("--limit-sources must be at least 10 for split balance")
        return rows[:limit_sources]
    return rows


def response_suffix() -> list[str]:
    return [
        "Return only one city name, or UNKNOWN.",
        "Do not explain.",
        "Answer:",
    ]


def render_arrow_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["Country -> temporary city"]
    for row in rows:
        lines.append(f"{row['country']} -> {row['artificial_value']}")
    return lines


def render_sentence_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["Temporary city notes:"]
    for row in rows:
        lines.append(f"For {row['country']}, the listed city is {row['artificial_value']}.")
    return lines


def render_compact_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["country | listed_city"]
    for row in rows:
        lines.append(f"{row['country']} | {row['artificial_value']}")
    return lines


def render_rows(rows: list[dict[str, Any]], template: str) -> list[str]:
    if template == "registry_question":
        return render_arrow_table(rows)
    if template == "sentence_rewrite":
        return render_sentence_table(rows)
    if template == "compact_rewrite":
        return render_compact_table(rows)
    raise ValueError(f"unknown template {template}")


def prompt_for_panel(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    panel: str,
    template: str,
) -> tuple[str, str]:
    if panel == "baseline_source_value_lookup":
        rows = table_sources(sources, source, include_query=True)
        if template == "registry_question":
            lines = [
                "A temporary registry assigns city names for this task.",
                *render_arrow_table(rows),
                f"Question: In the temporary registry, which city is assigned to {source['country']}?",
            ]
        elif template == "sentence_rewrite":
            lines = [
                "A temporary registry assigns city names for this task.",
                *render_arrow_table(rows),
                f"Question: In the temporary registry, which city is assigned to {source['country']}?",
            ]
        elif template == "compact_rewrite":
            lines = [
                "Temporary city registry:",
                *render_arrow_table(rows),
                f"Query country: {source['country']}",
            ]
        else:
            raise ValueError(f"unknown template {template}")
        expected_label = "artificial_value"
    elif panel in REWRITE_PANELS:
        rows = table_sources(sources, source, include_query=True)
        if template == "registry_question":
            lines = [
                "The same temporary city registry is written in a neutral form.",
                *render_sentence_table(rows),
                f"Question: Which listed city goes with {source['country']}?",
            ]
        elif template == "sentence_rewrite":
            lines = [
                "A neutral note lists temporary city pairings.",
                *render_sentence_table(rows),
                f"Question: What listed city goes with {source['country']}?",
            ]
        elif template == "compact_rewrite":
            lines = [
                "Temporary city listing:",
                *render_compact_table(rows),
                f"country={source['country']}",
                "listed_city=?",
            ]
        else:
            raise ValueError(f"unknown template {template}")
        expected_label = "artificial_value"
    elif panel == "source_deletion":
        rows = table_sources(sources, source, include_query=False)
        lines = [
            "A temporary registry assigns city names for this task.",
            "If the queried country is absent from the listing, answer UNKNOWN.",
            *render_rows(rows, template),
            f"Question: Which listed city goes with {source['country']}?",
        ]
        expected_label = "unknown"
    elif panel == "query_only_control":
        if template == "registry_question":
            lines = [
                "A temporary city registry may exist, but no registry entries are shown.",
                f"Question: In the temporary registry, which city is assigned to {source['country']}?",
                "If no entry is provided, answer UNKNOWN.",
            ]
        elif template == "sentence_rewrite":
            lines = [
                "A neutral note could list temporary city pairings, but no pairings are shown.",
                f"Question: What listed city goes with {source['country']}?",
                "If no entry is provided, answer UNKNOWN.",
            ]
        elif template == "compact_rewrite":
            lines = [
                "Temporary city listing:",
                "(no rows provided)",
                f"country={source['country']}",
                "listed_city=?",
                "If no entry is provided, answer UNKNOWN.",
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
                        "id": f"{CARD_ID}_{template}_{panel}_{source['source_id']}",
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
    pattern = rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])"
    return len(re.findall(pattern, prompt))


def line_contains_both(prompt: str, left: str, right: str) -> bool:
    return any(left in line and right in line for line in prompt.splitlines())


def prompt_audit(record: dict[str, Any]) -> dict[str, Any]:
    prompt = record["prompt"]
    panel = record["panel"]
    artificial_occurrences = prompt_occurrences(prompt, record["artificial_value"])
    true_occurrences = prompt_occurrences(prompt, record["true_capital"])
    lure_occurrences = prompt_occurrences(prompt, record["lure_capital"])
    expected_artificial_occurrences = 1 if panel in SOURCE_VALUE_PANELS else 0
    source_deleted = panel == "source_deletion"
    query_only = panel == "query_only_control"
    return {
        "artificial_value_occurrences": artificial_occurrences,
        "true_capital_occurrences": true_occurrences,
        "lure_capital_occurrences": lure_occurrences,
        "artificial_value_prompt_count_expected": artificial_occurrences == expected_artificial_occurrences,
        "true_capital_not_prompt_listed": true_occurrences == 0,
        "lure_capital_not_prompt_listed": lure_occurrences == 0,
        "source_deletion_omits_query_source_value": not source_deleted
        or not line_contains_both(prompt, record["country"], record["artificial_value"]),
        "query_only_has_no_source_value": not query_only or artificial_occurrences == 0,
        "prompt_has_status_lexeme": bool(STATUS_LEXEME_RE.search(prompt)),
        "prompt_has_authority_lexeme": bool(AUTHORITY_RE.search(prompt)),
        "shared_response_suffix_present": prompt.endswith("\n".join(response_suffix())),
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
            and audit["source_deletion_omits_query_source_value"]
            and audit["query_only_has_no_source_value"]
            and not audit["prompt_has_status_lexeme"]
            and not audit["prompt_has_authority_lexeme"]
            and audit["shared_response_suffix_present"]
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
    query_proxy = sum(1 for row in rows if row.get("is_query_proxy"))
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
        "query_proxy": query_proxy,
        "query_proxy_rate": rate(query_proxy, len(rows)),
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
        rewrite = [
            row for row in rows if row["panel"] in {"neutral_rewrite_lookup", "source_disjoint_rewrite_holdout"}
        ]
        rewrite_holdout = [
            row for row in rows if row["panel"] in {"neutral_rewrite_lookup", "source_disjoint_rewrite_holdout"}
            and row["split"] == "holdout"
        ]
        result[template] = {
            "rows": len(rows),
            "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "selected_label_counts": label_counts(rows),
            "panels": panels,
            "rewrite_combined": summarize_rows(rewrite),
            "rewrite_holdout": summarize_rows(rewrite_holdout),
        }
    return result


def selection_key(summary: dict[str, Any], template: str, templates: tuple[str, ...]) -> tuple[float, ...]:
    item = summary[template]
    baseline = item["panels"]["baseline_source_value_lookup"]
    rewrite = item["panels"]["neutral_rewrite_lookup"]
    deletion = item["panels"]["source_deletion"]
    query_only = item["panels"]["query_only_control"]
    holdout = item["rewrite_holdout"]
    baseline_delta = abs(float(baseline["artificial_value_rate"]) - float(rewrite["artificial_value_rate"]))
    return (
        float(rewrite["artificial_value_rate"]),
        -baseline_delta,
        float(baseline["artificial_value_rate"]),
        float(deletion["unknown_rate"]),
        -float(query_only["query_proxy_rate"]),
        float(holdout["artificial_value_rate"]),
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
            "max neutral rewrite source-value lookup",
            "min rewrite-baseline delta",
            "max baseline source-value lookup",
            "max source-deletion UNKNOWN",
            "min query-only source-value proxy",
            "max source-disjoint rewrite holdout source-value lookup",
            "earliest template",
        ],
    }


def classify(criteria: dict[str, bool]) -> str:
    if not criteria["structural_passed"]:
        return "structural_invalid"
    if not criteria["baseline_source_value_lookup_passed"]:
        return "source_rewrite_baseline_lookup_failed"
    if not criteria["neutral_rewrite_lookup_passed"]:
        return "source_rewrite_equivalence_failed"
    if not criteria["source_deletion_passed"]:
        return "source_deletion_not_local"
    if not criteria["query_only_control_passed"]:
        return "query_only_source_proxy"
    if not criteria["candidate_and_output_margins_reported"]:
        return "source_rewrite_behavior_passed_baselines_missing"
    if criteria["smoke_mode"]:
        return "smoke_source_rewrite_candidate"
    if not criteria["source_disjoint_rewrite_holdout_passed"]:
        return "source_rewrite_holdout_failed"
    return "source_rewrite_equivalence_behavior_ready"


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
    baseline = selected["panels"]["baseline_source_value_lookup"]
    rewrite = selected["panels"]["neutral_rewrite_lookup"]
    deletion = selected["panels"]["source_deletion"]
    query_only = selected["panels"]["query_only_control"]
    holdout = selected["rewrite_holdout"]
    output_audit = selected["panels"]["rewrite_output_geometry_audit"]
    selected_rows = [row for row in outputs if row["template"] == selection["selected_template"]]
    prompt_audit_passed = all(
        row["artificial_value_prompt_count_expected"]
        and row["true_capital_not_prompt_listed"]
        and row["lure_capital_not_prompt_listed"]
        and row["source_deletion_omits_query_source_value"]
        and row["query_only_has_no_source_value"]
        and not row["prompt_has_status_lexeme"]
        and not row["prompt_has_authority_lexeme"]
        and row["shared_response_suffix_present"]
        for row in selected_rows
    )
    rewrite_delta = abs(float(baseline["artificial_value_rate"]) - float(rewrite["artificial_value_rate"]))
    criteria = {
        "smoke_mode": not full_run,
        "structural_passed": structural["passed"],
        "full_source_count_is_40": structural["source_count"] == 40,
        "source_disjoint_holdout": structural["criteria"]["source_split_disjoint"]
        and structural["criteria"]["holdout_sources_present"]
        and structural["criteria"]["calibration_sources_present"],
        "selected_prompt_audit_passed": prompt_audit_passed,
        "baseline_source_value_lookup_passed": float(baseline["parseable_rate"]) >= 0.90
        and float(baseline["artificial_value_rate"]) >= 0.85,
        "neutral_rewrite_lookup_passed": float(rewrite["parseable_rate"]) >= 0.90
        and float(rewrite["artificial_value_rate"]) >= 0.80
        and rewrite_delta <= 0.15,
        "source_deletion_passed": float(deletion["artificial_value_rate"]) <= 0.10
        and float(deletion["unknown_rate"]) >= 0.70,
        "query_only_control_passed": float(query_only["query_proxy_rate"]) <= 0.10,
        "source_disjoint_rewrite_holdout_passed": float(holdout["parseable_rate"]) >= 0.90
        and float(holdout["artificial_value_rate"]) >= 0.75,
        "candidate_and_output_margins_reported": bool(
            score_candidates and "mean_artificial_minus_real_prior_logit" in output_audit
        ),
    }
    diagnostic_class = classify(criteria)
    behavior_ready = diagnostic_class == "source_rewrite_equivalence_behavior_ready"
    behavior_candidate = diagnostic_class in {
        "smoke_source_rewrite_candidate",
        "source_rewrite_equivalence_behavior_ready",
    }
    return {
        "structural": structural,
        "by_template": by_template,
        "selection": selection,
        "selected_template_summary": selected,
        "selected_template_rows": selected_rows,
        "criteria": criteria,
        "rewrite_delta_from_baseline": rewrite_delta,
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
        "rewrite_delta_from_baseline": summary["rewrite_delta_from_baseline"],
        "selection": summary["selection"],
        "selected_controls": {
            panel_name: summary["selected_template_summary"]["panels"][panel_name]
            for panel_name in PANELS
        },
        "selected_rewrite_holdout": summary["selected_template_summary"]["rewrite_holdout"],
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
            "is_source_value_panel": record["panel"] in SOURCE_VALUE_PANELS,
            "is_rewrite_panel": record["panel"] in REWRITE_PANELS,
            "is_expected_correct": record["expected_label"] == parsed["selected_label"],
            "is_query_proxy": record["panel"] == "query_only_control"
            and parsed["selected_label"] == "artificial_value",
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
        "# KSQ002 Familiar Entity Source-Rewrite Equivalence First Run",
        "",
        "Status: behavior-substrate first-run preregistration; no hidden-state work.",
        "",
        "Runner:",
        "",
        "> `code/ksq002_familiar_entity_source_rewrite_equivalence_first_run.py`",
        "",
        "Default result:",
        "",
        "> `results/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE/ksq002_familiar_entity_source_rewrite_equivalence_first_run.json`",
        "",
        "Default 10-source smoke result:",
        "",
        "> `results/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE/ksq002_familiar_entity_source_rewrite_equivalence_smoke_limit10.json`",
        "",
        "Full behavior result:",
        "",
        "> `results/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE/ksq002_familiar_entity_source_rewrite_equivalence_full_behavior.json`",
        "",
        "Status card:",
        "",
        "> `research/cards/KSQ002_FAMILIAR_ENTITY_SOURCE_REWRITE_EQUIVALENCE_FIRST_RUN_STATUS.md`",
        "",
        "## Purpose",
        "",
        "Test whether familiar-entity artificial source lookup survives neutral",
        "rewrites and disappears when the source row is deleted or when only the",
        "query country remains.",
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
            "Promote only to behavior-substrate admission if baseline source-value",
            "lookup passes, neutral rewrites preserve it within the predeclared",
            "delta, source deletion and query-only controls do not reproduce the",
            "source value, source-disjoint rewrite holdout passes, and",
            "candidate/output baseline reporting is present.",
            "",
            "Death rule: kill if neutral rewriting breaks the behavior, if source",
            "deletion preserves the source value, if query-only text reproduces",
            "the source value, or if output/candidate geometry explains the split.",
            "",
            "Forbidden claims:",
            "",
            "- KSQ002 is a mechanism card.",
            "- KSQ002 licenses hidden-state search before the behavior gate passes.",
            "- KSQ002 proves a source-channel or knowledge-control surface.",
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
            f"- rewrite delta from baseline: `{summary['rewrite_delta_from_baseline']:.3f}`",
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
            "Interpretation: KSQ002 only becomes behavior-ready if source-local",
            "artificial lookup is robust to neutral rewrite and local to the",
            "source row. A rewrite failure, deletion failure, or query-only",
            "source-value reproduction is a diagnostic, not a hidden-state license.",
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
        "# KSQ002 Familiar Entity Source-Rewrite Equivalence First-Run Status",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "Runner:",
        "",
        "> `code/ksq002_familiar_entity_source_rewrite_equivalence_first_run.py`",
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
        lines.extend(["Status: " + status + ".", "", "## Verdict", ""])
        if structural["passed"] and full_behavior_result is not None:
            lines.append(
                "The structural gate passed and a full behavior run exists. "
                "Read the behavior diagnostic below; hidden-state work remains "
                "forbidden unless every behavior and baseline gate passes."
            )
        elif structural["passed"] and smoke_result is not None:
            lines.append(
                "The structural gate passed and the 10-source behavior smoke "
                "exists. This is not a hidden-state license."
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
        lines.extend([f"Status: {summary['diagnostic_class']}.", "", "## Verdict", ""])
        if summary["behavior_ready"]:
            lines.append("The behavior gate passed. This licenses only a later signature-screen decision.")
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
            "- KSQ002 is a mechanism card.",
            "- KSQ002 supports intervention.",
            "- KSQ002 found an internal source-channel or knowledge-control surface.",
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
            "purpose": "Structural gate for the KSQ002 familiar-entity source-rewrite equivalence first run.",
            "templates": list(TEMPLATES),
            "panels": list(PANELS),
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
        print(
            json.dumps(
                {"passed": False, "diagnostic_class": "structural_invalid", "structural": structural},
                indent=2,
                ensure_ascii=True,
            )
        )
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
        "purpose": "Behavior-only KSQ002 familiar-entity source-rewrite equivalence admission run.",
        "summary": summary,
        "outputs": outputs,
        "hidden_state_allowed": False,
    }
    if args.write_manifest:
        write_json(args.output_path, result)
        if args.write_status_card:
            write_status_card(args.status_card, result, args.output_path)

    payload = quiet_summary(summary, args.output_path if args.write_manifest else None)
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0 if (summary["behavior_ready"] or not full_run) else 1


if __name__ == "__main__":
    raise SystemExit(main())
