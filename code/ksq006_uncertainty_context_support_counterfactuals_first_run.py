#!/usr/bin/env python
"""KSQ006 context-support counterfactual first run.

This is a behavior-substrate admission runner for the Level 5 knowledge ladder:
context support before any claim about uncertainty, refusal, or correction
mechanisms. It tests whether support-sensitive answer/abstain behavior survives
matched supported, irrelevant, contradicting, insufficient, claim-only, and
context-only controls.

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
from mc007_semi_synthetic_familiar_entity_lookup import generate_answer, normalize_text
from mc016_alphabet_gated_numeric_arbitration import load_model_and_tokenizer


CARD_ID = "KSQ006"
CANDIDATE_ID = "ksq006_uncertainty_context_support_counterfactuals"
RUN_TYPE = "ksq006_uncertainty_context_support_counterfactuals_structural_gate"
BEHAVIOR_RUN_TYPE = "ksq006_uncertainty_context_support_counterfactuals_behavior"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS")
DEFAULT_RESULT_PATH = RESULT_DIR / "ksq006_uncertainty_context_support_counterfactuals_first_run.json"
SMOKE_LIMIT10_RESULT_PATH = (
    RESULT_DIR / "ksq006_uncertainty_context_support_counterfactuals_smoke_limit10.json"
)
FULL_BEHAVIOR_RESULT_PATH = (
    RESULT_DIR / "ksq006_uncertainty_context_support_counterfactuals_full_behavior.json"
)
STATUS_CARD = Path(
    "research/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS_FIRST_RUN_STATUS.md"
)
PREREG_PATH = Path(
    "research/prereg/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS_FIRST_RUN.md"
)

PANELS = (
    "supported_context_rows",
    "irrelevant_context_rows",
    "contradicting_context_rows",
    "insufficient_context_rows",
    "claim_only_and_context_only_controls",
)
CONTROL_SUBTYPES = ("claim_only_control", "context_only_control")
TEMPLATES = ("note_question", "field_form", "compact_record")

STATUS_LEXEME_RE = re.compile(
    r"\b(trusted|untrusted|reliable|unreliable|verified|status|confidence)\b",
    re.IGNORECASE,
)
SUPPORT_WORD_RE = re.compile(r"\b(support|supports|supported|unsupported)\b", re.IGNORECASE)
ABSTAIN_RE = re.compile(
    r"^(unknown|unk|not enough|cannot determine|can't determine|cannot answer|"
    r"not provided|insufficient|i do not know|i don't know)\b",
    re.IGNORECASE,
)


def source_rows(limit_sources: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (source_id, country, true_capital, override_capital, lure_capital) in enumerate(SOURCES):
        normalized = {
            normalize_text(country),
            normalize_text(true_capital),
            normalize_text(override_capital),
            normalize_text(lure_capital),
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
                "false_context_answer": override_capital,
                "irrelevant_city": lure_capital,
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
        "Use the reference notes.",
        "If the requested field is absent or the notes disagree, answer UNKNOWN.",
        "Return only one city name or UNKNOWN.",
        "Do not explain.",
        "Answer:",
    ]


def relation_lines(source: dict[str, Any], panel: str, control_subtype: str | None) -> list[str]:
    country = source["country"]
    true_capital = source["true_capital"]
    false_city = source["false_context_answer"]
    irrelevant_city = source["irrelevant_city"]
    if panel == "supported_context_rows":
        return [
            f"Country: {country}",
            f"Current national capital: {true_capital}",
        ]
    if panel == "irrelevant_context_rows":
        return [
            f"Country: {country}",
            f"Regional rail hub: {false_city}",
        ]
    if panel == "contradicting_context_rows":
        if int(source["source_index"]) % 2 == 0:
            return [
                f"Country: {country}",
                f"Current national capital: {true_capital}",
                f"Current national capital: {false_city}",
            ]
        return [
            f"Country: {country}",
            f"Current national capital: {false_city}",
            f"Current national capital: {true_capital}",
        ]
    if panel == "insufficient_context_rows":
        return [
            f"Country: {country}",
            f"Administrative district mentioned in the file: {irrelevant_city}",
        ]
    if panel == "claim_only_and_context_only_controls":
        if control_subtype == "claim_only_control":
            return [
                f"Claim outside the reference notes: The current national capital of {country} is {true_capital}.",
                "Reference notes: none.",
            ]
        if control_subtype == "context_only_control":
            return [
                f"Country: {country}",
                f"City mentioned in this file: {true_capital}",
            ]
        raise ValueError(f"unknown control subtype {control_subtype}")
    raise ValueError(f"unknown panel {panel}")


def render_notes(lines: list[str], template: str, source: dict[str, Any]) -> list[str]:
    country = source["country"]
    if template == "note_question":
        return [
            "Reference notes:",
            *[f"- {line}" for line in lines],
            f"Question: What is the current national capital of {country}?",
        ]
    if template == "field_form":
        return [
            "Reference notes:",
            *[f"{line}" for line in lines],
            f"Requested entity: {country}",
            "Requested field: current national capital",
        ]
    if template == "compact_record":
        rendered = [line.replace(": ", " | ", 1) for line in lines]
        return [
            "Notes:",
            *rendered,
            f"Query: {country} | current national capital",
        ]
    raise ValueError(f"unknown template {template}")


def expected_label_for_panel(panel: str, control_subtype: str | None) -> str:
    if panel == "supported_context_rows":
        return "supported_answer"
    if panel in {"irrelevant_context_rows", "insufficient_context_rows"}:
        return "abstain"
    if panel == "contradicting_context_rows":
        return "contradiction_detected"
    if panel == "claim_only_and_context_only_controls":
        if control_subtype in CONTROL_SUBTYPES:
            return "control_abstain"
    raise ValueError(f"unknown panel/subtype {panel}/{control_subtype}")


def prompt_for_record(
    source: dict[str, Any],
    *,
    panel: str,
    template: str,
    control_subtype: str | None = None,
) -> tuple[str, str]:
    lines = relation_lines(source, panel, control_subtype)
    prompt_lines = render_notes(lines, template, source)
    return "\n".join([*prompt_lines, *response_suffix()]), expected_label_for_panel(panel, control_subtype)


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
                subtypes: tuple[str | None, ...]
                if panel == "claim_only_and_context_only_controls":
                    subtypes = CONTROL_SUBTYPES
                else:
                    subtypes = (None,)
                for control_subtype in subtypes:
                    prompt, expected_label = prompt_for_record(
                        source,
                        panel=panel,
                        template=template,
                        control_subtype=control_subtype,
                    )
                    suffix = f"_{control_subtype}" if control_subtype else ""
                    records.append(
                        {
                            "id": f"KSQ006_{template}_{panel}{suffix}_{source['source_id']}",
                            "card_id": CARD_ID,
                            "candidate_id": CANDIDATE_ID,
                            "run_type": run_type,
                            "model_id": MODEL_ID,
                            "template": template,
                            "panel": panel,
                            "control_subtype": control_subtype,
                            "split": source["split"],
                            "source_id": source["source_id"],
                            "source_index": source["source_index"],
                            "country": source["country"],
                            "true_capital": source["true_capital"],
                            "false_context_answer": source["false_context_answer"],
                            "irrelevant_city": source["irrelevant_city"],
                            "expected_label": expected_label,
                            "candidate_answers": [
                                source["true_capital"],
                                source["false_context_answer"],
                                source["irrelevant_city"],
                                "UNKNOWN",
                            ],
                            "prompt": prompt,
                        }
                    )
    return records


def candidates(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "answer": record["true_capital"],
            "candidate_label": "true_capital",
            "candidate_type": "supported_or_prior_answer",
        },
        {
            "answer": record["false_context_answer"],
            "candidate_label": "false_context_answer",
            "candidate_type": "unsupported_or_contradicting_answer",
        },
        {
            "answer": record["irrelevant_city"],
            "candidate_label": "irrelevant_city",
            "candidate_type": "irrelevant_context_city",
        },
        {"answer": "UNKNOWN", "candidate_label": "abstain", "candidate_type": "abstention"},
    ]


def selected_label_for_candidate(record: dict[str, Any], candidate_label: str) -> str:
    panel = record["panel"]
    subtype = record.get("control_subtype")
    if panel == "supported_context_rows":
        if candidate_label == "true_capital":
            return "supported_answer"
        if candidate_label == "abstain":
            return "abstain"
        return "wrong_answer"
    if panel == "irrelevant_context_rows":
        if candidate_label == "abstain":
            return "abstain"
        if candidate_label == "false_context_answer":
            return "unsupported_answer"
        if candidate_label == "true_capital":
            return "prior_answer"
        return "irrelevant_answer"
    if panel == "contradicting_context_rows":
        if candidate_label == "abstain":
            return "contradiction_detected"
        if candidate_label == "false_context_answer":
            return "false_accept"
        if candidate_label == "true_capital":
            return "true_answer_despite_contradiction"
        return "wrong_answer"
    if panel == "insufficient_context_rows":
        if candidate_label == "abstain":
            return "abstain"
        return "unsupported_answer"
    if panel == "claim_only_and_context_only_controls":
        if candidate_label == "abstain":
            return "control_abstain"
        if candidate_label == "true_capital" and subtype == "claim_only_control":
            return "claim_only_reproduced_supported"
        if candidate_label == "true_capital" and subtype == "context_only_control":
            return "context_only_reproduced_supported"
        return "control_other_answer"
    raise ValueError(f"unknown panel {panel}")


def strict_parse(record: dict[str, Any], generated_text: str) -> dict[str, Any]:
    stripped = generated_text.strip().lstrip("`'\" ")
    first_line = stripped.splitlines()[0] if stripped else ""
    normalized = normalize_text(first_line)
    if ABSTAIN_RE.search(normalized):
        return {
            "selected_label": selected_label_for_candidate(record, "abstain"),
            "selected_answer": "UNKNOWN",
            "parseable": True,
            "parse_rule": "abstention_prefix",
            "first_line": first_line,
            "normalized_first_line": normalized,
        }
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
            "selected_label": selected_label_for_candidate(record, matches[0]["candidate_label"]),
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


def prompt_occurrences(prompt: str, value: str) -> int:
    return len(re.findall(rf"\b{re.escape(value)}\b", prompt))


def expected_prompt_counts(record: dict[str, Any]) -> dict[str, int]:
    panel = record["panel"]
    subtype = record.get("control_subtype")
    true_count = 0
    false_count = 0
    irrelevant_count = 0
    if panel == "supported_context_rows":
        true_count = 1
    elif panel == "irrelevant_context_rows":
        false_count = 1
    elif panel == "contradicting_context_rows":
        true_count = 1
        false_count = 1
    elif panel == "insufficient_context_rows":
        irrelevant_count = 1
    elif panel == "claim_only_and_context_only_controls":
        if subtype in CONTROL_SUBTYPES:
            true_count = 1
        else:
            raise ValueError(f"unknown control subtype {subtype}")
    else:
        raise ValueError(f"unknown panel {panel}")
    return {
        "true_capital": true_count,
        "false_context_answer": false_count,
        "irrelevant_city": irrelevant_count,
    }


def prompt_audit(record: dict[str, Any]) -> dict[str, Any]:
    prompt = record["prompt"]
    expected = expected_prompt_counts(record)
    true_occurrences = prompt_occurrences(prompt, record["true_capital"])
    false_occurrences = prompt_occurrences(prompt, record["false_context_answer"])
    irrelevant_occurrences = prompt_occurrences(prompt, record["irrelevant_city"])
    suffix = "\n".join(response_suffix())
    support_word_count = len(SUPPORT_WORD_RE.findall(prompt))
    return {
        "true_capital_occurrences": true_occurrences,
        "false_context_answer_occurrences": false_occurrences,
        "irrelevant_city_occurrences": irrelevant_occurrences,
        "true_capital_prompt_count_expected": true_occurrences == expected["true_capital"],
        "false_context_answer_prompt_count_expected": false_occurrences == expected["false_context_answer"],
        "irrelevant_city_prompt_count_expected": irrelevant_occurrences == expected["irrelevant_city"],
        "prompt_has_status_lexeme": bool(STATUS_LEXEME_RE.search(prompt)),
        "support_word_count": support_word_count,
        "prompt_has_support_word": support_word_count > 0,
        "shared_response_suffix_present": suffix in prompt,
    }


def structural_check(records: list[dict[str, Any]], templates: tuple[str, ...]) -> dict[str, Any]:
    source_ids = {row["source_id"] for row in records}
    duplicate_ids = [
        row_id for row_id, count in Counter(row["id"] for row in records).items() if count > 1
    ]
    panel_counts = Counter(row["panel"] for row in records)
    template_counts = Counter(row["template"] for row in records)
    subtype_counts = Counter(str(row["control_subtype"]) for row in records if row["control_subtype"])
    split_source_ids: dict[str, set[str]] = {}
    candidate_collision_rows = []
    prompt_audit_failures = []
    support_word_rows = []
    for row in records:
        split_source_ids.setdefault(row["split"], set()).add(row["source_id"])
        normalized_candidates = [normalize_text(candidate) for candidate in row["candidate_answers"]]
        if len(set(normalized_candidates)) != len(normalized_candidates):
            candidate_collision_rows.append(row["id"])
        audit = prompt_audit(row)
        if audit["prompt_has_support_word"]:
            support_word_rows.append(row["id"])
        if not (
            audit["true_capital_prompt_count_expected"]
            and audit["false_context_answer_prompt_count_expected"]
            and audit["irrelevant_city_prompt_count_expected"]
            and not audit["prompt_has_status_lexeme"]
            and audit["shared_response_suffix_present"]
        ):
            prompt_audit_failures.append(row["id"])
    expected_rows = len(source_ids) * len(templates) * (len(PANELS) + len(CONTROL_SUBTYPES) - 1)
    criteria = {
        "expected_row_count": len(records) == expected_rows,
        "all_panels_present": set(panel_counts) == set(PANELS),
        "all_templates_present": set(template_counts) == set(templates),
        "control_subtypes_present": set(subtype_counts) == set(CONTROL_SUBTYPES),
        "no_duplicate_record_ids": not duplicate_ids,
        "source_split_disjoint": sum(len(ids) for ids in split_source_ids.values()) == len(source_ids),
        "holdout_sources_present": bool(split_source_ids.get("holdout")),
        "calibration_sources_present": bool(split_source_ids.get("calibration")),
        "candidate_answers_parseable": all(row["candidate_answers"] for row in records),
        "no_candidate_collisions": not candidate_collision_rows,
        "prompt_audit_passed": not prompt_audit_failures,
        "no_support_word_prompt_channel": not support_word_rows,
        "shared_requested_mode_suffix": all(
            prompt_audit(row)["shared_response_suffix_present"] for row in records
        ),
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "record_count": len(records),
        "source_count": len(source_ids),
        "panel_counts": dict(sorted(panel_counts.items())),
        "template_counts": dict(sorted(template_counts.items())),
        "control_subtype_counts": dict(sorted(subtype_counts.items())),
        "split_source_counts": {
            split: len(ids) for split, ids in sorted(split_source_ids.items())
        },
        "expected_rows": expected_rows,
        "duplicate_ids": duplicate_ids[:20],
        "candidate_collision_rows": candidate_collision_rows[:20],
        "prompt_audit_failure_rows": prompt_audit_failures[:20],
        "support_word_rows": support_word_rows[:20],
    }


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row.get("selected_label", "missing") for row in rows).items()))


def rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parseable = sum(1 for row in rows if row.get("parseable"))
    supported_answer = sum(1 for row in rows if row.get("selected_label") == "supported_answer")
    abstain = sum(1 for row in rows if row.get("selected_label") == "abstain")
    contradiction_detected = sum(
        1 for row in rows if row.get("selected_label") == "contradiction_detected"
    )
    unsupported_answer = sum(1 for row in rows if row.get("selected_label") == "unsupported_answer")
    false_accept = sum(1 for row in rows if row.get("selected_label") == "false_accept")
    claim_only_reproduced = sum(
        1 for row in rows if row.get("selected_label") == "claim_only_reproduced_supported"
    )
    context_only_reproduced = sum(
        1 for row in rows if row.get("selected_label") == "context_only_reproduced_supported"
    )
    control_abstain = sum(1 for row in rows if row.get("selected_label") == "control_abstain")
    unparsed = sum(1 for row in rows if row.get("selected_label") == "unparsed")
    support_word_count = sum(int(row.get("support_word_count", 0)) for row in rows)
    candidate_margins = "candidate_first_token_logits" in rows[0] if rows else False
    result = {
        "rows": len(rows),
        "label_counts": label_counts(rows),
        "parseable": parseable,
        "parseable_rate": rate(parseable, len(rows)),
        "supported_answer": supported_answer,
        "supported_answer_rate": rate(supported_answer, len(rows)),
        "abstain": abstain,
        "abstain_rate": rate(abstain, len(rows)),
        "contradiction_detected": contradiction_detected,
        "contradiction_detected_rate": rate(contradiction_detected, len(rows)),
        "unsupported_answer": unsupported_answer,
        "unsupported_answer_rate": rate(unsupported_answer, len(rows)),
        "false_accept": false_accept,
        "false_accept_rate": rate(false_accept, len(rows)),
        "claim_only_reproduced_supported": claim_only_reproduced,
        "claim_only_reproduced_supported_rate": rate(claim_only_reproduced, len(rows)),
        "context_only_reproduced_supported": context_only_reproduced,
        "context_only_reproduced_supported_rate": rate(context_only_reproduced, len(rows)),
        "control_reproduced_supported": claim_only_reproduced + context_only_reproduced,
        "control_reproduced_supported_rate": rate(claim_only_reproduced + context_only_reproduced, len(rows)),
        "control_abstain": control_abstain,
        "control_abstain_rate": rate(control_abstain, len(rows)),
        "unparsed": unparsed,
        "unparsed_rate": rate(unparsed, len(rows)),
        "support_word_count": support_word_count,
        "support_word_rate": rate(support_word_count, len(rows)),
    }
    if candidate_margins:
        result["mean_supported_minus_abstain_logit"] = sum(
            float(row["supported_minus_abstain_logit"]) for row in rows
        ) / len(rows)
        result["mean_abstain_minus_supported_logit"] = sum(
            float(row["abstain_minus_supported_logit"]) for row in rows
        ) / len(rows)
        result["mean_false_context_minus_supported_logit"] = sum(
            float(row["false_context_minus_supported_logit"]) for row in rows
        ) / len(rows)
        result["mean_irrelevant_minus_abstain_logit"] = sum(
            float(row["irrelevant_minus_abstain_logit"]) for row in rows
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
        controls = {
            subtype: summarize_rows(
                [
                    row
                    for row in rows
                    if row["panel"] == "claim_only_and_context_only_controls"
                    and row["control_subtype"] == subtype
                ]
            )
            for subtype in CONTROL_SUBTYPES
        }
        holdout_panels = {
            panel: summarize_rows(
                [row for row in rows if row["panel"] == panel and row["split"] == "holdout"]
            )
            for panel in PANELS
        }
        result[template] = {
            "rows": len(rows),
            "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "selected_label_counts": label_counts(rows),
            "panels": panels,
            "controls": controls,
            "holdout_panels": holdout_panels,
        }
    return result


def selection_key(summary: dict[str, Any], template: str, templates: tuple[str, ...]) -> tuple[float, ...]:
    item = summary[template]["panels"]
    supported = item["supported_context_rows"]
    irrelevant = item["irrelevant_context_rows"]
    contradicting = item["contradicting_context_rows"]
    insufficient = item["insufficient_context_rows"]
    controls = item["claim_only_and_context_only_controls"]
    min_parseability = min(float(item[panel]["parseable_rate"]) for panel in PANELS)
    return (
        float(supported["supported_answer_rate"]),
        float(irrelevant["abstain_rate"]),
        float(contradicting["contradiction_detected_rate"]),
        float(insufficient["abstain_rate"]),
        -float(controls["control_reproduced_supported_rate"]),
        min_parseability,
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
            "max supported-context answer",
            "max irrelevant-context abstention",
            "max contradicting-context contradiction detection",
            "max insufficient-context abstention",
            "min claim/context-only supported reproduction",
            "max minimum panel parseability",
            "earliest template",
        ],
    }


def classify(criteria: dict[str, bool]) -> str:
    if not criteria["structural_passed"]:
        return "structural_invalid"
    if not criteria["selected_prompt_audit_passed"]:
        return "prompt_channel_or_support_word_audit_failed"
    if not criteria["supported_context_rows_passed"]:
        return "supported_context_failed"
    if not criteria["irrelevant_context_rows_passed"]:
        return "irrelevant_context_abstention_failed"
    if not criteria["contradicting_context_rows_passed"]:
        return "contradiction_detection_failed"
    if not criteria["insufficient_context_rows_passed"]:
        return "insufficient_context_abstention_failed"
    if not criteria["claim_only_and_context_only_controls_passed"]:
        return "claim_or_context_only_control_failed"
    if not criteria["candidate_and_output_margins_reported"]:
        return "context_support_behavior_passed_baselines_missing"
    if criteria["smoke_mode"]:
        return "smoke_context_support_candidate"
    if not criteria["support_counterfactual_holdout_passed"]:
        return "support_counterfactual_holdout_failed"
    return "context_support_counterfactual_behavior_ready"


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
    panels = selected["panels"]
    supported = panels["supported_context_rows"]
    irrelevant = panels["irrelevant_context_rows"]
    contradicting = panels["contradicting_context_rows"]
    insufficient = panels["insufficient_context_rows"]
    controls = panels["claim_only_and_context_only_controls"]
    holdout = selected["holdout_panels"]
    selected_rows = [row for row in outputs if row["template"] == selection["selected_template"]]
    prompt_audit_passed = all(
        row["true_capital_prompt_count_expected"]
        and row["false_context_answer_prompt_count_expected"]
        and row["irrelevant_city_prompt_count_expected"]
        and not row["prompt_has_status_lexeme"]
        and not row["prompt_has_support_word"]
        and row["shared_response_suffix_present"]
        for row in selected_rows
    )
    holdout_passed = (
        float(holdout["supported_context_rows"]["supported_answer_rate"]) >= 0.75
        and float(holdout["irrelevant_context_rows"]["abstain_rate"]) >= 0.60
        and float(holdout["contradicting_context_rows"]["contradiction_detected_rate"]) >= 0.60
        and float(holdout["insufficient_context_rows"]["abstain_rate"]) >= 0.60
        and float(holdout["claim_only_and_context_only_controls"]["control_reproduced_supported_rate"]) <= 0.25
    )
    criteria = {
        "smoke_mode": not full_run,
        "structural_passed": structural["passed"],
        "full_source_count_is_40": structural["source_count"] == 40,
        "source_disjoint_holdout": structural["criteria"]["source_split_disjoint"],
        "selected_prompt_audit_passed": prompt_audit_passed,
        "supported_context_rows_passed": float(supported["parseable_rate"]) >= 0.90
        and float(supported["supported_answer_rate"]) >= 0.80,
        "irrelevant_context_rows_passed": float(irrelevant["parseable_rate"]) >= 0.90
        and float(irrelevant["abstain_rate"]) >= 0.65
        and float(irrelevant["unsupported_answer_rate"]) <= 0.20,
        "contradicting_context_rows_passed": float(contradicting["parseable_rate"]) >= 0.90
        and float(contradicting["contradiction_detected_rate"]) >= 0.65
        and float(contradicting["false_accept_rate"]) <= 0.20,
        "insufficient_context_rows_passed": float(insufficient["parseable_rate"]) >= 0.90
        and float(insufficient["abstain_rate"]) >= 0.65
        and float(insufficient["support_word_rate"]) < 0.80,
        "claim_only_and_context_only_controls_passed": float(controls["parseable_rate"]) >= 0.90
        and float(controls["control_reproduced_supported_rate"]) <= 0.20,
        "support_counterfactual_holdout_passed": holdout_passed,
        "candidate_and_output_margins_reported": score_candidates
        and "mean_supported_minus_abstain_logit" in supported,
    }
    diagnostic_class = classify(criteria)
    behavior_ready = diagnostic_class == "context_support_counterfactual_behavior_ready"
    behavior_candidate = diagnostic_class in {
        "smoke_context_support_candidate",
        "context_support_counterfactual_behavior_ready",
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


def next_token_candidate_logits(model: Any, tokenizer: Any, prompt: str, record: dict[str, Any]) -> dict[str, Any]:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].detach().float().cpu()
    scores: dict[str, float] = {}
    token_ids: dict[str, int] = {}
    for candidate in candidates(record):
        encoded = tokenizer(" " + candidate["answer"], add_special_tokens=False)["input_ids"]
        if not encoded:
            encoded = tokenizer(candidate["answer"], add_special_tokens=False)["input_ids"]
        first_id = int(encoded[0])
        label = candidate["candidate_label"]
        token_ids[label] = first_id
        scores[label] = float(logits[first_id].item())
    return {
        "candidate_first_token_ids": token_ids,
        "candidate_first_token_logits": scores,
        "supported_minus_abstain_logit": scores["true_capital"] - scores["abstain"],
        "abstain_minus_supported_logit": scores["abstain"] - scores["true_capital"],
        "false_context_minus_supported_logit": scores["false_context_answer"] - scores["true_capital"],
        "irrelevant_minus_abstain_logit": scores["irrelevant_city"] - scores["abstain"],
    }


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
            "is_supported_panel": record["panel"] == "supported_context_rows",
            "is_counterfactual_panel": record["panel"]
            in {
                "irrelevant_context_rows",
                "contradicting_context_rows",
                "insufficient_context_rows",
                "claim_only_and_context_only_controls",
            },
        }
        if score_candidates:
            output.update(next_token_candidate_logits(model, tokenizer, record["prompt"], record))
        outputs.append(output)
        if verbose:
            print(
                f"[{index:03d}/{len(records):03d}] {record['id']} split={record['split']} "
                f"panel={record['panel']} subtype={record['control_subtype']} -> "
                f"{output['selected_label']} {str(output['selected_answer'])!r} "
                f"generated={generated['generated_text']!r}"
            )
    return outputs


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
        "selected_controls": {
            panel_name: summary["selected_template_summary"]["panels"][panel_name]
            for panel_name in PANELS
        },
        "selected_control_subtypes": summary["selected_template_summary"]["controls"],
    }
    if output_path is not None:
        payload["output_path"] = str(output_path)
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def write_prereg(path: Path) -> None:
    lines = [
        "# KSQ006 Context-Support Counterfactuals First Run",
        "",
        "Status: behavior-substrate first-run preregistration; no hidden-state work.",
        "",
        "Runner:",
        "",
        "> `code/ksq006_uncertainty_context_support_counterfactuals_first_run.py`",
        "",
        "Default result:",
        "",
        "> `results/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS/ksq006_uncertainty_context_support_counterfactuals_first_run.json`",
        "",
        "Default 10-source smoke result:",
        "",
        "> `results/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS/ksq006_uncertainty_context_support_counterfactuals_smoke_limit10.json`",
        "",
        "Full behavior result:",
        "",
        "> `results/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS/ksq006_uncertainty_context_support_counterfactuals_full_behavior.json`",
        "",
        "Status card:",
        "",
        "> `research/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS_FIRST_RUN_STATUS.md`",
        "",
        "## Purpose",
        "",
        "Test whether context support can be made counterfactual before",
        "testing factual uncertainty, refusal, or correction signatures. The",
        "run separates supported, irrelevant, contradicting, insufficient,",
        "claim-only, and context-only branches under a shared answer schema.",
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
            "Promote only to behavior-substrate admission if supported rows",
            "answer, irrelevant rows abstain, contradicting rows detect",
            "disagreement, insufficient rows abstain without support-word",
            "leakage, claim-only/context-only controls do not reproduce",
            "supported behavior, source-disjoint holdout passes, and",
            "candidate/output baseline reporting is present.",
            "",
            "Death rule: kill if support language, answer shape, claim-only",
            "text, context-only city mention, caution wording, or output",
            "margin explains the behavior.",
            "",
            "Forbidden claims:",
            "",
            "- KSQ006 is a mechanism card.",
            "- KSQ006 licenses hidden-state search before the behavior gate passes.",
            "- KSQ006 proves a factual uncertainty, refusal, context-support, or correction control surface.",
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
            f"- control subtypes: `{json.dumps(structural['control_subtype_counts'], sort_keys=True)}`",
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
            "Control subtype counts:",
            "",
            "| Control subtype | Label counts |",
            "| --- | --- |",
        ]
    )
    for subtype in CONTROL_SUBTYPES:
        labels = selected["controls"][subtype]["label_counts"]
        lines.append(f"| `{subtype}` | `{json.dumps(labels, sort_keys=True)}` |")
    lines.extend(
        [
            "",
            "Interpretation: KSQ006 only becomes behavior-ready if support",
            "sensitivity cannot be reproduced by claim-only text, context-only",
            "city mention, support-word prompts, answer schema, or output",
            "geometry. A supported-row win with failed counterfactual controls",
            "is a diagnostic, not uncertainty control.",
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
        "# KSQ006 Context-Support Counterfactuals First-Run Status",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "Runner:",
        "",
        "> `code/ksq006_uncertainty_context_support_counterfactuals_first_run.py`",
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
        lines.extend(
            [
                f"Status: {summary['diagnostic_class']}.",
                "",
                "## Verdict",
                "",
            ]
        )
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
            "- KSQ006 is a mechanism card.",
            "- KSQ006 supports intervention.",
            "- KSQ006 found an internal uncertainty, refusal, correction, context-support, or knowledge-control surface.",
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
    parser.add_argument("--max-new-tokens", type=int, default=12)
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
            "purpose": "Structural gate for the KSQ006 context-support counterfactual first run.",
            "templates": list(TEMPLATES),
            "panels": list(PANELS),
            "control_subtypes": list(CONTROL_SUBTYPES),
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
                {
                    "passed": False,
                    "diagnostic_class": "structural_invalid",
                    "structural": structural,
                },
                indent=2,
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
        "purpose": "Behavior-only KSQ006 context-support counterfactual admission run.",
        "panels": list(PANELS),
        "control_subtypes": list(CONTROL_SUBTYPES),
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
