#!/usr/bin/env python
"""KSQ009 schema-specific value lookup.

KSQ008 showed that a neutral evidence repair can suppress many forbidden
channels but still lose positive answering and still accept counted old-schema
`answer_for(entity)=value` rows. KSQ009 narrows the repair:

    Can an explicit ALLOW-row value schema preserve positive lookup while
    rejecting answer_for rows even when they appear in the counted region?

This is a behavior-only substrate test. It can identify a better prompt
contract or a sharper failure class, but it cannot establish uncertainty,
factuality, hidden-state causality, intervention success, or a mechanism.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from ksq007_nonce_evidence_answerability_calibrator import (
    ABSTAIN_RE,
    MODEL_ID,
    STATUS_LEXEME_RE,
    SUPPORT_WORD_RE,
    exact_occurrences,
    generate_answer,
    load_model_and_tokenizer,
    normalize_text,
    rate,
    source_rows,
    write_json,
)


CARD_ID = "KSQ009"
CANDIDATE_ID = "ksq009_schema_specific_value_lookup"
RUN_TYPE = "ksq009_schema_specific_value_lookup_structural_gate"
BEHAVIOR_RUN_TYPE = "ksq009_schema_specific_value_lookup_behavior"
RESULT_DIR = Path("results/cards/KSQ009_SCHEMA_SPECIFIC_VALUE_LOOKUP")
DEFAULT_RESULT_PATH = RESULT_DIR / "ksq009_schema_specific_value_lookup_first_run.json"
SMOKE_LIMIT10_RESULT_PATH = (
    RESULT_DIR / "ksq009_schema_specific_value_lookup_smoke_limit10.json"
)
FULL_BEHAVIOR_RESULT_PATH = (
    RESULT_DIR / "ksq009_schema_specific_value_lookup_full_behavior.json"
)
STATUS_CARD = Path("research/cards/KSQ009_SCHEMA_SPECIFIC_VALUE_LOOKUP_STATUS.md")
PREREG_PATH = Path("research/prereg/KSQ009_SCHEMA_SPECIFIC_VALUE_LOOKUP.md")

PANELS = (
    "exact_allowed_value",
    "absent_allowed_value",
    "unrelated_entity_allowed_value",
    "conflicting_allowed_values",
    "counted_wrong_schema_answer_for",
    "uncounted_wrong_schema_answer_for",
    "uncounted_allowed_value",
    "quoted_allowed_value",
    "allowed_value_vs_uncounted_answer_for_alt",
    "query_only_control",
)
POSITIVE_PANELS = (
    "exact_allowed_value",
    "allowed_value_vs_uncounted_answer_for_alt",
)
CONTROL_PANELS = tuple(panel for panel in PANELS if panel not in POSITIVE_PANELS)
TEMPLATES = ("pipe_rows", "kv_lines", "csv_rows")


def response_suffix() -> list[str]:
    return [
        "Return only the value, or UNKNOWN.",
        "Do not explain.",
        "Answer:",
    ]


def allowed_row(entity: str, value: str, template: str) -> str:
    if template == "pipe_rows":
        return f"ALLOW | {entity} | {value}"
    if template == "kv_lines":
        return f"ALLOW entity={entity} value={value}"
    if template == "csv_rows":
        return f"ALLOW,{entity},{value}"
    raise ValueError(f"unknown template {template}")


def quoted_allowed_row(entity: str, value: str, template: str) -> str:
    return f'QUOTE "{allowed_row(entity, value, template)}"'


def row_payload(source: dict[str, Any], panel: str, template: str) -> dict[str, Any]:
    entity = source["entity"]
    decoy = source["decoy_entity"]
    target = source["target_value"]
    alt = source["alternate_value"]
    if panel == "exact_allowed_value":
        return {
            "counted_rows": [allowed_row(entity, target, template)],
            "other_rows": [],
            "expected_label": "evidence_answer",
        }
    if panel == "absent_allowed_value":
        return {
            "counted_rows": [f"ALLOW entity={entity} region=north"],
            "other_rows": [],
            "expected_label": "abstain",
        }
    if panel == "unrelated_entity_allowed_value":
        return {
            "counted_rows": [allowed_row(decoy, target, template)],
            "other_rows": [],
            "expected_label": "abstain",
        }
    if panel == "conflicting_allowed_values":
        rows = [allowed_row(entity, target, template), allowed_row(entity, alt, template)]
        if int(source["source_index"]) % 2:
            rows = list(reversed(rows))
        return {
            "counted_rows": rows,
            "other_rows": [],
            "expected_label": "abstain",
        }
    if panel == "counted_wrong_schema_answer_for":
        return {
            "counted_rows": [f"answer_for({entity})={target}"],
            "other_rows": [],
            "expected_label": "abstain",
        }
    if panel == "uncounted_wrong_schema_answer_for":
        return {
            "counted_rows": [],
            "other_rows": [f"answer_for({entity})={target}"],
            "expected_label": "control_abstain",
        }
    if panel == "uncounted_allowed_value":
        return {
            "counted_rows": [],
            "other_rows": [allowed_row(entity, target, template)],
            "expected_label": "control_abstain",
        }
    if panel == "quoted_allowed_value":
        return {
            "counted_rows": [],
            "other_rows": [quoted_allowed_row(entity, target, template)],
            "expected_label": "control_abstain",
        }
    if panel == "allowed_value_vs_uncounted_answer_for_alt":
        return {
            "counted_rows": [allowed_row(entity, target, template)],
            "other_rows": [f"answer_for({entity})={alt}"],
            "expected_label": "evidence_answer",
        }
    if panel == "query_only_control":
        return {
            "counted_rows": [],
            "other_rows": [],
            "expected_label": "control_abstain",
        }
    raise ValueError(f"unknown panel {panel}")


def render_prompt(
    source: dict[str, Any],
    *,
    panel: str,
    template: str,
) -> tuple[str, str]:
    payload = row_payload(source, panel, template)
    counted = payload["counted_rows"] or ["(none)"]
    other = payload["other_rows"] or ["(none)"]
    entity = source["entity"]
    if template == "pipe_rows":
        lines = [
            "COUNTED LOOKUP ROWS:",
            *[f"- {row}" for row in counted],
            "OTHER TEXT:",
            *[f"- {row}" for row in other],
            "Use only rows with exactly this form: ALLOW | entity | value.",
            "For the requested entity, return the value field from one matching ALLOW row.",
            "Rows written as answer_for(entity)=value are not ALLOW rows.",
            "OTHER TEXT does not determine the output.",
            "If there is not exactly one matching ALLOW row, return UNKNOWN.",
            f"Requested entity: {entity}",
        ]
    elif template == "kv_lines":
        lines = [
            "COUNTED LOOKUP ROWS:",
            *[f"- {row}" for row in counted],
            "OTHER TEXT:",
            *[f"- {row}" for row in other],
            "Use only rows with both fields: ALLOW entity=<entity> value=<value>.",
            "For the requested entity, return the value field from one matching ALLOW row.",
            "Rows written as answer_for(entity)=value are not ALLOW rows.",
            "OTHER TEXT does not determine the output.",
            "If there is not exactly one matching ALLOW row, return UNKNOWN.",
            f"Requested entity: {entity}",
        ]
    elif template == "csv_rows":
        lines = [
            "COUNTED LOOKUP ROWS:",
            *[f"- {row}" for row in counted],
            "OTHER TEXT:",
            *[f"- {row}" for row in other],
            "Use only rows with comma fields: ALLOW,entity,value.",
            "For the requested entity, return the value field from one matching ALLOW row.",
            "Rows written as answer_for(entity)=value are not ALLOW rows.",
            "OTHER TEXT does not determine the output.",
            "If there is not exactly one matching ALLOW row, return UNKNOWN.",
            f"Requested entity: {entity}",
        ]
    else:
        raise ValueError(f"unknown template {template}")
    return "\n".join([*lines, *response_suffix()]), payload["expected_label"]


def source_records(
    limit_sources: int | None = None,
    templates: tuple[str, ...] = TEMPLATES,
    run_type: str = RUN_TYPE,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in source_rows(limit_sources):
        for template in templates:
            for panel in PANELS:
                prompt, expected_label = render_prompt(source, panel=panel, template=template)
                records.append(
                    {
                        "id": f"{CARD_ID}_{template}_{panel}_{source['source_id']}",
                        "card_id": CARD_ID,
                        "parent_card_id": "KSQ008",
                        "candidate_id": CANDIDATE_ID,
                        "run_type": run_type,
                        "model_id": MODEL_ID,
                        "template": template,
                        "panel": panel,
                        "split": source["split"],
                        "source_id": source["source_id"],
                        "source_index": source["source_index"],
                        "entity": source["entity"],
                        "decoy_entity": source["decoy_entity"],
                        "target_value": source["target_value"],
                        "alternate_value": source["alternate_value"],
                        "lure_value": source["lure_value"],
                        "expected_label": expected_label,
                        "candidate_answers": [
                            source["target_value"],
                            source["alternate_value"],
                            source["lure_value"],
                            "UNKNOWN",
                        ],
                        "prompt": prompt,
                    }
                )
    return records


def candidates(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"answer": record["target_value"], "candidate_label": "target_value"},
        {"answer": record["alternate_value"], "candidate_label": "alternate_value"},
        {"answer": record["lure_value"], "candidate_label": "lure_value"},
        {"answer": "UNKNOWN", "candidate_label": "abstain"},
    ]


def selected_label_for_candidate(record: dict[str, Any], candidate_label: str) -> str:
    panel = record["panel"]
    if candidate_label == "abstain":
        return "control_abstain" if panel in CONTROL_PANELS else "abstain"
    if panel in POSITIVE_PANELS and candidate_label == "target_value":
        return "evidence_answer"
    if panel == "allowed_value_vs_uncounted_answer_for_alt" and candidate_label == "alternate_value":
        return "answer_for_alt_overrode_allowed_value"
    if panel == "absent_allowed_value" and candidate_label == "target_value":
        return "unsupported_target_answered"
    if panel == "unrelated_entity_allowed_value" and candidate_label == "target_value":
        return "unrelated_value_reproduced"
    if panel == "conflicting_allowed_values" and candidate_label in {
        "target_value",
        "alternate_value",
    }:
        return "conflict_value_selected"
    if panel == "counted_wrong_schema_answer_for" and candidate_label == "target_value":
        return "counted_answer_for_reproduced"
    if panel == "uncounted_wrong_schema_answer_for" and candidate_label == "target_value":
        return "uncounted_answer_for_reproduced"
    if panel == "uncounted_allowed_value" and candidate_label == "target_value":
        return "uncounted_allowed_reproduced"
    if panel == "quoted_allowed_value" and candidate_label == "target_value":
        return "quoted_allowed_reproduced"
    if panel == "query_only_control":
        return "query_only_answered"
    return "other_candidate"


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
    for candidate in sorted(
        candidates(record),
        key=lambda row: len(normalize_text(row["answer"])),
        reverse=True,
    ):
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


def expected_prompt_counts(record: dict[str, Any]) -> dict[str, int]:
    panel = record["panel"]
    target_count = 0
    alternate_count = 0
    if panel in {
        "exact_allowed_value",
        "unrelated_entity_allowed_value",
        "counted_wrong_schema_answer_for",
        "uncounted_wrong_schema_answer_for",
        "uncounted_allowed_value",
        "quoted_allowed_value",
    }:
        target_count = 1
    elif panel == "conflicting_allowed_values":
        target_count = 1
        alternate_count = 1
    elif panel == "allowed_value_vs_uncounted_answer_for_alt":
        target_count = 1
        alternate_count = 1
    return {
        "target_value": target_count,
        "alternate_value": alternate_count,
        "lure_value": 0,
    }


def prompt_audit(record: dict[str, Any]) -> dict[str, Any]:
    prompt = record["prompt"]
    expected = expected_prompt_counts(record)
    suffix = "\n".join(response_suffix())
    target_count = exact_occurrences(prompt, record["target_value"])
    alternate_count = exact_occurrences(prompt, record["alternate_value"])
    lure_count = exact_occurrences(prompt, record["lure_value"])
    return {
        "target_value_occurrences": target_count,
        "alternate_value_occurrences": alternate_count,
        "lure_value_occurrences": lure_count,
        "target_value_prompt_count_expected": target_count == expected["target_value"],
        "alternate_value_prompt_count_expected": alternate_count == expected["alternate_value"],
        "lure_value_prompt_count_expected": lure_count == expected["lure_value"],
        "prompt_has_status_lexeme": bool(STATUS_LEXEME_RE.search(prompt)),
        "prompt_has_support_word": bool(SUPPORT_WORD_RE.search(prompt)),
        "shared_response_suffix_present": suffix in prompt,
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
    support_word_rows = []
    status_lexeme_rows = []
    for row in records:
        split_source_ids.setdefault(row["split"], set()).add(row["source_id"])
        normalized_candidates = [normalize_text(candidate) for candidate in row["candidate_answers"]]
        if len(set(normalized_candidates)) != len(normalized_candidates):
            candidate_collision_rows.append(row["id"])
        audit = prompt_audit(row)
        if audit["prompt_has_support_word"]:
            support_word_rows.append(row["id"])
        if audit["prompt_has_status_lexeme"]:
            status_lexeme_rows.append(row["id"])
        if not (
            audit["target_value_prompt_count_expected"]
            and audit["alternate_value_prompt_count_expected"]
            and audit["lure_value_prompt_count_expected"]
            and not audit["prompt_has_status_lexeme"]
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
        "no_support_word_prompt_channel": not support_word_rows,
        "no_status_lexeme_prompt_channel": not status_lexeme_rows,
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
        "split_source_counts": {
            split: len(ids) for split, ids in sorted(split_source_ids.items())
        },
        "expected_rows": expected_rows,
        "duplicate_ids": duplicate_ids[:20],
        "candidate_collision_rows": candidate_collision_rows[:20],
        "prompt_audit_failure_rows": prompt_audit_failures[:20],
        "support_word_rows": support_word_rows[:20],
        "status_lexeme_rows": status_lexeme_rows[:20],
    }


def next_token_candidate_logits(
    model: Any,
    tokenizer: Any,
    prompt: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].detach().float().cpu()
    token_ids: dict[str, int] = {}
    scores: dict[str, float] = {}
    for candidate in candidates(record):
        encoded = tokenizer(" " + candidate["answer"], add_special_tokens=False)["input_ids"]
        if not encoded:
            encoded = tokenizer(candidate["answer"], add_special_tokens=False)["input_ids"]
        first_id = int(encoded[0])
        token_ids[candidate["candidate_label"]] = first_id
        scores[candidate["candidate_label"]] = float(logits[first_id].item())
    return {
        "candidate_first_token_ids": token_ids,
        "candidate_first_token_logits": scores,
        "target_minus_abstain_logit": scores["target_value"] - scores["abstain"],
        "alternate_minus_abstain_logit": scores["alternate_value"] - scores["abstain"],
        "abstain_minus_target_logit": scores["abstain"] - scores["target_value"],
    }


def score_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    score_candidates: bool,
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
        }
        if score_candidates:
            output.update(next_token_candidate_logits(model, tokenizer, record["prompt"], record))
        outputs.append(output)
        print(
            f"[{index:03d}/{len(records):03d}] {record['id']} split={record['split']} "
            f"template={record['template']} panel={record['panel']} -> "
            f"{output['selected_label']} {str(output['selected_answer'])!r} "
            f"generated={generated['generated_text']!r}"
        )
    return outputs


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row.get("selected_label", "missing") for row in rows).items()))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parseable = sum(1 for row in rows if row.get("parseable"))
    evidence_answer = sum(1 for row in rows if row.get("selected_label") == "evidence_answer")
    abstain = sum(1 for row in rows if row.get("selected_label") == "abstain")
    control_abstain = sum(1 for row in rows if row.get("selected_label") == "control_abstain")
    reproduced = sum(
        1
        for row in rows
        if isinstance(row.get("selected_label"), str)
        and row["selected_label"].endswith(("_reproduced", "_answered", "_selected"))
    )
    answer_for_alt_override = sum(
        1 for row in rows if row.get("selected_label") == "answer_for_alt_overrode_allowed_value"
    )
    other = sum(
        1
        for row in rows
        if row.get("selected_label") in {"wrong_candidate", "other_candidate"}
    )
    unparsed = sum(1 for row in rows if row.get("selected_label") == "unparsed")
    candidate_margins = "candidate_first_token_logits" in rows[0] if rows else False
    result = {
        "rows": len(rows),
        "label_counts": label_counts(rows),
        "parseable": parseable,
        "parseable_rate": rate(parseable, len(rows)),
        "evidence_answer": evidence_answer,
        "evidence_answer_rate": rate(evidence_answer, len(rows)),
        "abstain": abstain,
        "abstain_rate": rate(abstain, len(rows)),
        "control_abstain": control_abstain,
        "control_abstain_rate": rate(control_abstain, len(rows)),
        "any_abstain": abstain + control_abstain,
        "any_abstain_rate": rate(abstain + control_abstain, len(rows)),
        "control_reproduced_value": reproduced,
        "control_reproduced_value_rate": rate(reproduced, len(rows)),
        "answer_for_alt_override": answer_for_alt_override,
        "answer_for_alt_override_rate": rate(answer_for_alt_override, len(rows)),
        "other_candidate": other,
        "other_candidate_rate": rate(other, len(rows)),
        "unparsed": unparsed,
        "unparsed_rate": rate(unparsed, len(rows)),
    }
    if candidate_margins:
        result["mean_target_minus_abstain_logit"] = sum(
            float(row["target_minus_abstain_logit"]) for row in rows
        ) / len(rows)
        result["mean_abstain_minus_target_logit"] = sum(
            float(row["abstain_minus_target_logit"]) for row in rows
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
        holdout_panels = {
            panel: summarize_rows(
                [row for row in rows if row["panel"] == panel and row["split"] == "holdout"]
            )
            for panel in PANELS
        }
        control_rows = [row for row in rows if row["panel"] in CONTROL_PANELS]
        result[template] = {
            "rows": len(rows),
            "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "selected_label_counts": label_counts(rows),
            "panels": panels,
            "controls": summarize_rows(control_rows),
            "holdout_panels": holdout_panels,
        }
    return result


def branch_rate(panel_name: str, panel: dict[str, Any]) -> float:
    if panel_name in POSITIVE_PANELS:
        return float(panel["evidence_answer_rate"])
    return float(panel["any_abstain_rate"])


def selection_key(summary: dict[str, Any], template: str, templates: tuple[str, ...]) -> tuple[float, ...]:
    panels = summary[template]["panels"]
    controls = summary[template]["controls"]
    branch_rates = tuple(branch_rate(panel, panels[panel]) for panel in PANELS)
    mixed_panel = panels["allowed_value_vs_uncounted_answer_for_alt"]
    return (
        min(branch_rates),
        *branch_rates,
        -float(controls["control_reproduced_value_rate"]),
        -float(mixed_panel["answer_for_alt_override_rate"]),
        min(float(panels[panel]["parseable_rate"]) for panel in PANELS),
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
            "max weakest branch rate",
            *[f"max {panel} branch rate" for panel in PANELS],
            "min control value reproduction",
            "min answer_for alternate override",
            "max minimum panel parseability",
            "earliest template",
        ],
    }


def panel_passed(panel_name: str, panel: dict[str, Any]) -> bool:
    if panel_name in POSITIVE_PANELS:
        return (
            float(panel["parseable_rate"]) >= 0.90
            and float(panel["evidence_answer_rate"]) >= 0.80
            and float(panel["answer_for_alt_override_rate"]) <= 0.10
        )
    return (
        float(panel["parseable_rate"]) >= 0.90
        and float(panel["any_abstain_rate"]) >= 0.80
        and float(panel["control_reproduced_value_rate"]) <= 0.10
    )


def classify(criteria: dict[str, bool], failed_panels: list[str], selected: dict[str, Any]) -> str:
    if not criteria["structural_passed"]:
        return "structural_invalid"
    if not criteria["selected_prompt_audit_passed"]:
        return "schema_specific_prompt_audit_failed"
    if not criteria["exact_allowed_value_passed"]:
        return "schema_specific_positive_failed"
    if "allowed_value_vs_uncounted_answer_for_alt" in failed_panels:
        mixed = selected["panels"]["allowed_value_vs_uncounted_answer_for_alt"]
        if mixed["label_counts"].get("answer_for_alt_overrode_allowed_value"):
            return "answer_for_competes_with_allowed_value"
        return "schema_specific_mixed_positive_failed"
    failed = set(failed_panels)
    if failed & {"counted_wrong_schema_answer_for", "uncounted_wrong_schema_answer_for"}:
        return "answer_for_schema_specificity_failed"
    if failed & {"uncounted_allowed_value", "quoted_allowed_value"}:
        return "allowed_row_locality_failed"
    if "conflicting_allowed_values" in failed:
        return "schema_specific_conflict_abstention_failed"
    if failed & {"absent_allowed_value", "unrelated_entity_allowed_value", "query_only_control"}:
        return "schema_specific_baseline_failed"
    if not failed_panels and criteria["candidate_and_output_margins_reported"]:
        return "schema_specific_value_lookup_candidate"
    return "schema_specific_value_lookup_unscored_candidate"


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
    holdout = selected["holdout_panels"]
    selected_rows = [row for row in outputs if row["template"] == selection["selected_template"]]
    prompt_audit_passed = all(
        row["target_value_prompt_count_expected"]
        and row["alternate_value_prompt_count_expected"]
        and row["lure_value_prompt_count_expected"]
        and not row["prompt_has_status_lexeme"]
        and not row["prompt_has_support_word"]
        and row["shared_response_suffix_present"]
        for row in selected_rows
    )
    per_panel_pass = {panel: panel_passed(panel, panels[panel]) for panel in PANELS}
    failed_panels = [panel for panel in PANELS if not per_panel_pass.get(panel, False)]
    holdout_passed = all(panel_passed(panel, holdout[panel]) for panel in PANELS)
    answer_for_controls = [
        "counted_wrong_schema_answer_for",
        "uncounted_wrong_schema_answer_for",
        "allowed_value_vs_uncounted_answer_for_alt",
    ]
    locality_controls = ["uncounted_allowed_value", "quoted_allowed_value"]
    abstain_controls = [
        "absent_allowed_value",
        "unrelated_entity_allowed_value",
        "conflicting_allowed_values",
        "query_only_control",
    ]
    criteria = {
        "smoke_mode": not full_run,
        "structural_passed": structural["passed"],
        "full_source_count_is_40": structural["source_count"] == 40,
        "source_disjoint_holdout": structural["criteria"]["source_split_disjoint"],
        "selected_prompt_audit_passed": prompt_audit_passed,
        "exact_allowed_value_passed": per_panel_pass["exact_allowed_value"],
        "allowed_value_vs_uncounted_answer_for_alt_passed": per_panel_pass[
            "allowed_value_vs_uncounted_answer_for_alt"
        ],
        "answer_for_schema_controls_passed": all(
            per_panel_pass[panel] for panel in answer_for_controls
        ),
        "allowed_row_locality_controls_passed": all(
            per_panel_pass[panel] for panel in locality_controls
        ),
        "abstain_controls_passed": all(per_panel_pass[panel] for panel in abstain_controls),
        "all_panels_passed": not failed_panels,
        "source_disjoint_holdout_passed": holdout_passed,
        "candidate_and_output_margins_reported": score_candidates,
    }
    diagnostic_class = classify(criteria, failed_panels, selected)
    behavior_ready = diagnostic_class == "schema_specific_value_lookup_candidate" and full_run
    return {
        "diagnostic_class": diagnostic_class,
        "structural": structural,
        "criteria": criteria,
        "per_panel_pass": per_panel_pass,
        "failed_panels": failed_panels,
        "selection": selection,
        "selected_template_summary": selected,
        "by_template": by_template,
        "schema_specific_decision": {
            "behavior_ready": behavior_ready,
            "signature_screen_allowed": behavior_ready,
            "hidden_state_claim_allowed": False,
            "intervention_allowed": False,
            "mechanism_claim_allowed": False,
            "route_decision": "schema_specific_lookup_admitted_for_margin_screen"
            if behavior_ready
            else "schema_specific_lookup_diagnostic",
            "exported_diagnostic_class": "SCHEMA_SPECIFIC_VALUE_LOOKUP_CANDIDATE"
            if behavior_ready
            else diagnostic_class.upper(),
        },
    }


def build_result(
    records: list[dict[str, Any]],
    outputs: list[dict[str, Any]] | None,
    templates: tuple[str, ...],
    full_run: bool,
    score_candidates: bool,
) -> dict[str, Any]:
    if outputs is None:
        structural = structural_check(records, templates)
        summary = {
            "diagnostic_class": "structural_passed" if structural["passed"] else "structural_failed",
            "structural": structural,
            "criteria": {
                "structural_passed": structural["passed"],
                "full_source_count_is_40": structural["source_count"] == 40,
                "source_disjoint_holdout": structural["criteria"]["source_split_disjoint"],
                "hidden_state_license": False,
            },
            "schema_specific_decision": {
                "behavior_ready": False,
                "signature_screen_allowed": False,
                "hidden_state_claim_allowed": False,
                "intervention_allowed": False,
                "mechanism_claim_allowed": False,
                "route_decision": "structural_only",
                "exported_diagnostic_class": None,
            },
        }
    else:
        summary = summarize(records, outputs, templates, full_run, score_candidates)
    return {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "card_id": CARD_ID,
        "parent_card_id": "KSQ008",
        "candidate_id": CANDIDATE_ID,
        "run_type": BEHAVIOR_RUN_TYPE if outputs is not None else RUN_TYPE,
        "model_id": MODEL_ID,
        "full_run": full_run,
        "templates": list(templates),
        "panels": list(PANELS),
        "summary": summary,
        "records": outputs if outputs is not None else records,
        "allowed_claim": (
            "KSQ009 tests whether an explicit ALLOW-row value schema can preserve "
            "neutral positive lookup while rejecting answer_for rows in counted "
            "and uncounted positions. It is behavior-only."
        ),
        "forbidden_claim": (
            "KSQ009 does not prove uncertainty, factuality, hidden-state causality, "
            "intervention success, or a mechanism."
        ),
    }


def write_prereg(path: Path) -> None:
    lines = [
        "# KSQ009 Schema-Specific Value Lookup",
        "",
        "Status: KSQ008 diagnostic follow-up.",
        "",
        "Runner:",
        "",
        "> `code/ksq009_schema_specific_value_lookup.py`",
        "",
        "Artifacts:",
        "",
        "> `results/cards/KSQ009_SCHEMA_SPECIFIC_VALUE_LOOKUP/ksq009_schema_specific_value_lookup_first_run.json`",
        "",
        "> `results/cards/KSQ009_SCHEMA_SPECIFIC_VALUE_LOOKUP/ksq009_schema_specific_value_lookup_smoke_limit10.json`",
        "",
        "> `results/cards/KSQ009_SCHEMA_SPECIFIC_VALUE_LOOKUP/ksq009_schema_specific_value_lookup_full_behavior.json`",
        "",
        "## Claim Under Test",
        "",
        "KSQ008 failed because neutral positive evidence was weak and counted",
        "old-schema answer_for rows still reproduced values. This packet tests",
        "whether a stricter ALLOW-row value schema can separate positive lookup",
        "from answer_for schema specificity and allowed-row locality.",
        "",
        "## Promotion Rule",
        "",
        "Promote only to behavior-substrate candidate status if exact ALLOW rows",
        "answer, ALLOW rows beat an uncounted answer_for alternate, every",
        "abstention/control panel passes, selected prompt audit passes, source-",
        "disjoint holdout passes, and margins are reported. Hidden-state claims",
        "remain forbidden.",
        "",
        "## Kill / Boundary Rule",
        "",
        "If exact ALLOW rows do not answer, export a positive-parseability",
        "failure. If answer_for rows reproduce values, export schema-specificity",
        "failure. If uncounted or quoted ALLOW rows reproduce values, export",
        "allowed-row locality failure. Treat the typed failure as the datum.",
        "",
        "## Forbidden Claims",
        "",
        "- This is not a mechanism card.",
        "- This is not real uncertainty, refusal, or factual correction.",
        "- Hidden-state probing remains forbidden unless a later full behavior",
        "  run and margin report admit only a signature screen.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_status_card(path: Path, result: dict[str, Any], output_path: Path) -> None:
    summary = result["summary"]
    criteria = summary["criteria"]
    decision = summary["schema_specific_decision"]
    lines = [
        "# KSQ009 Schema-Specific Value Lookup Status",
        "",
        "Date: 2026-07-02",
        "",
        "Runner:",
        "",
        "> `code/ksq009_schema_specific_value_lookup.py`",
        "",
        "Result:",
        "",
        f"> `{output_path.as_posix()}`",
        "",
        f"Status: {summary['diagnostic_class']}.",
        "",
        "## Route Decision",
        "",
        f"- route decision: `{decision['route_decision']}`",
        f"- exported diagnostic class: `{decision['exported_diagnostic_class']}`",
        f"- behavior ready: `{str(decision['behavior_ready']).lower()}`",
        f"- signature screen allowed: `{str(decision['signature_screen_allowed']).lower()}`",
        "",
        "## Gate Criteria",
        "",
        "| Criterion | Passed |",
        "| --- | --- |",
    ]
    for key, value in criteria.items():
        lines.append(f"| `{key}` | `{str(value).lower()}` |")
    if "selection" in summary:
        selected = summary["selection"]["selected_template"]
        lines.extend(
            [
                "",
                "## Selected Template",
                "",
                f"- selected template: `{selected}`",
                f"- selection key: `{json.dumps(summary['selection']['selection_key'])}`",
                f"- failed panels: `{json.dumps(summary['failed_panels'])}`",
                "",
                "## Selected Panels",
                "",
                "| Panel | Rows | Label Counts | Key Rate |",
                "| --- | ---: | --- | ---: |",
            ]
        )
        panels = summary["selected_template_summary"]["panels"]
        for panel_name, panel in panels.items():
            key_rate = (
                panel["evidence_answer_rate"]
                if panel_name in POSITIVE_PANELS
                else panel["any_abstain_rate"]
            )
            lines.append(
                f"| `{panel_name}` | {panel['rows']} | "
                f"`{json.dumps(panel['label_counts'], sort_keys=True)}` | "
                f"{key_rate:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Forbidden Claims",
            "",
            "- KSQ009 is a mechanism card.",
            "- KSQ009 proves real uncertainty, refusal, factual correction, or knowledge control.",
            "- KSQ009 licenses hidden-state intervention or mechanism claims.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--score-candidates", action="store_true")
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--full-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--write-prereg", action="store_true")
    parser.add_argument("--prereg-path", type=Path, default=PREREG_PATH)
    parser.add_argument("--write-status-card", action="store_true")
    parser.add_argument("--status-card", type=Path, default=STATUS_CARD)
    args = parser.parse_args()

    if args.limit_sources is not None and args.full_run:
        raise ValueError("--full-run cannot be combined with --limit-sources")

    output_path = args.output_path
    if output_path is None:
        if args.score and args.limit_sources == 10:
            output_path = SMOKE_LIMIT10_RESULT_PATH
        elif args.score and args.full_run:
            output_path = FULL_BEHAVIOR_RESULT_PATH
        else:
            output_path = DEFAULT_RESULT_PATH

    records = source_records(
        limit_sources=args.limit_sources,
        templates=TEMPLATES,
        run_type=BEHAVIOR_RUN_TYPE if args.score else RUN_TYPE,
    )
    outputs = None
    if args.score:
        model, tokenizer = load_model_and_tokenizer(MODEL_ID, args.local_files_only)
        outputs = score_records(
            records,
            tokenizer,
            model,
            args.max_new_tokens,
            args.score_candidates,
        )
    result = build_result(
        records,
        outputs,
        TEMPLATES,
        full_run=args.full_run,
        score_candidates=args.score_candidates,
    )
    if args.write_prereg:
        write_prereg(args.prereg_path)
    if args.write:
        write_json(output_path, result)
        if args.write_status_card:
            write_status_card(args.status_card, result, output_path)
    print(json.dumps(result["summary"]["criteria"], sort_keys=True))
    if args.score and not all(result["summary"]["criteria"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
