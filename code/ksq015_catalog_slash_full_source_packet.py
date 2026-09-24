#!/usr/bin/env python
"""KSQ015 catalog-slash full-source packet.

KSQ014 split the slash result: catalog-labeled slash text survived a 10-source
locality smoke, while bare slash failed bridge preservation. KSQ015 removes the
failed bare-slash branch and asks whether the catalog-slash survivor becomes a
full-source behavior substrate.

This packet is still behavior-only. A pass can admit only a later signature
screen. It cannot establish hidden-state causality, intervention success,
factuality, uncertainty, or a mechanism.
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


CARD_ID = "KSQ015"
CANDIDATE_ID = "ksq015_catalog_slash_full_source_packet"
RUN_TYPE = "ksq015_catalog_slash_full_source_structural_gate"
BEHAVIOR_RUN_TYPE = "ksq015_catalog_slash_full_source_behavior"
RESULT_DIR = Path("results/cards/KSQ015_CATALOG_SLASH_FULL_SOURCE_PACKET")
DEFAULT_RESULT_PATH = RESULT_DIR / "ksq015_catalog_slash_full_source_first_run.json"
SMOKE_LIMIT10_RESULT_PATH = (
    RESULT_DIR / "ksq015_catalog_slash_full_source_smoke_limit10.json"
)
FULL_BEHAVIOR_RESULT_PATH = (
    RESULT_DIR / "ksq015_catalog_slash_full_source_full_behavior.json"
)
STATUS_CARD = Path("research/cards/KSQ015_CATALOG_SLASH_FULL_SOURCE_PACKET_STATUS.md")
PREREG_PATH = Path("research/prereg/KSQ015_CATALOG_SLASH_FULL_SOURCE_PACKET.md")

PANELS = (
    "exact_bridge",
    "raw_answer_for_alt",
    "catalog_slash_entity_alt",
    "catalog_slash_decoy_alt",
    "catalog_slash_reversed_entity_alt",
    "catalog_slash_entity_only_control",
    "catalog_slash_decoy_only_control",
    "catalog_slash_reversed_entity_only_control",
    "query_only_control",
)
BRIDGE_PANELS = (
    "exact_bridge",
    "raw_answer_for_alt",
    "catalog_slash_entity_alt",
    "catalog_slash_decoy_alt",
    "catalog_slash_reversed_entity_alt",
)
SLASH_BRIDGE_PANELS = (
    "catalog_slash_entity_alt",
    "catalog_slash_decoy_alt",
    "catalog_slash_reversed_entity_alt",
)
CONTROL_PANELS = (
    "catalog_slash_entity_only_control",
    "catalog_slash_decoy_only_control",
    "catalog_slash_reversed_entity_only_control",
    "query_only_control",
)
TEMPLATES = ("tag_rows",)


def response_suffix() -> list[str]:
    return [
        "Return only the value, or UNKNOWN.",
        "Do not explain.",
        "Answer:",
    ]


def code_for(source: dict[str, Any], role: str) -> str:
    index = int(source["source_index"]) + 41
    if role == "target":
        return f"K{index:03d}T"
    if role == "alternate":
        return f"K{index:03d}A"
    if role == "lure":
        return f"K{index:03d}L"
    raise ValueError(f"unknown code role {role}")


def entity_code_row(entity: str, code: str) -> str:
    return f"<ENTITY_CODE> {entity} :: {code}"


def code_value_row(code: str, value: str) -> str:
    return f"<CODE_VALUE> {code} :: {value}"


def bridge_rows(source: dict[str, Any]) -> list[str]:
    return [
        entity_code_row(source["entity"], code_for(source, "target")),
        code_value_row(code_for(source, "target"), source["target_value"]),
    ]


def other_rows(source: dict[str, Any], panel: str) -> list[str]:
    entity = source["entity"]
    decoy = source["decoy_entity"]
    alt = source["alternate_value"]
    if panel == "raw_answer_for_alt":
        return [f"answer_for({entity})={alt}"]
    if panel in {"catalog_slash_entity_alt", "catalog_slash_entity_only_control"}:
        return [f"CATALOG_NOTE {entity} / {alt}"]
    if panel in {"catalog_slash_decoy_alt", "catalog_slash_decoy_only_control"}:
        return [f"CATALOG_NOTE {decoy} / {alt}"]
    if panel in {
        "catalog_slash_reversed_entity_alt",
        "catalog_slash_reversed_entity_only_control",
    }:
        return [f"CATALOG_NOTE {alt} / {entity}"]
    return []


def row_payload(source: dict[str, Any], panel: str) -> dict[str, Any]:
    counted_rows = bridge_rows(source) if panel in BRIDGE_PANELS else []
    return {
        "counted_rows": counted_rows,
        "other_rows": other_rows(source, panel),
        "expected_label": "evidence_answer"
        if panel in BRIDGE_PANELS
        else "control_abstain",
    }


def render_prompt(source: dict[str, Any], *, panel: str, template: str) -> tuple[str, str]:
    if template != "tag_rows":
        raise ValueError(f"unknown template {template}")
    payload = row_payload(source, panel)
    counted = payload["counted_rows"] or ["(none)"]
    other = payload["other_rows"] or ["(none)"]
    entity = source["entity"]
    lines = [
        "COUNTED CODEBOOK ROWS:",
        *[f"- {row}" for row in counted],
        "OTHER TEXT:",
        *[f"- {row}" for row in other],
        "Use counted <ENTITY_CODE> entity :: code rows and counted <CODE_VALUE> code :: value rows.",
        "A complete path has exactly one counted entity-code row for the requested entity and exactly one counted code-value row with the same code.",
        "Return the value from that complete path.",
        "Rows in OTHER TEXT, value banks, metadata rows, decoy pairs, separated mentions, catalog notes, and answer_for strings are not codebook rows.",
        "OTHER TEXT does not determine the output.",
        "If there is not exactly one complete path, return UNKNOWN.",
        f"Requested entity: {entity}",
    ]
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
                        "parent_card_id": "KSQ014",
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
                        "target_code": code_for(source, "target"),
                        "alternate_code": code_for(source, "alternate"),
                        "lure_code": code_for(source, "lure"),
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
    if panel == "raw_answer_for_alt" and candidate_label == "alternate_value":
        return "raw_answer_channel_reproduced"
    if panel == "raw_answer_for_alt" and candidate_label == "target_value":
        return "raw_channel_resisted"
    if panel in SLASH_BRIDGE_PANELS and candidate_label == "target_value":
        return "evidence_answer"
    if panel == "exact_bridge" and candidate_label == "target_value":
        return "evidence_answer"
    if panel in SLASH_BRIDGE_PANELS and candidate_label == "alternate_value":
        return "slash_alt_overrode_bridge"
    if panel in CONTROL_PANELS and candidate_label == "alternate_value":
        return "control_reproduced_value"
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
    return {
        "target_value": 1 if panel in BRIDGE_PANELS else 0,
        "alternate_value": 0 if panel in {"exact_bridge", "query_only_control"} else 1,
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
            and not audit["prompt_has_support_word"]
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
    raw_reproduced = sum(
        1 for row in rows if row.get("selected_label") == "raw_answer_channel_reproduced"
    )
    slash_override = sum(
        1 for row in rows if row.get("selected_label") == "slash_alt_overrode_bridge"
    )
    abstain = sum(1 for row in rows if row.get("selected_label") == "abstain")
    control_abstain = sum(1 for row in rows if row.get("selected_label") == "control_abstain")
    control_reproduced = sum(
        1 for row in rows if row.get("selected_label") == "control_reproduced_value"
    )
    other = sum(
        1
        for row in rows
        if row.get("selected_label") in {"other_candidate", "query_only_answered"}
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
        "raw_answer_channel_reproduced": raw_reproduced,
        "raw_answer_channel_reproduced_rate": rate(raw_reproduced, len(rows)),
        "slash_alt_overrode_bridge": slash_override,
        "slash_alt_override_rate": rate(slash_override, len(rows)),
        "abstain": abstain,
        "abstain_rate": rate(abstain, len(rows)),
        "control_abstain": control_abstain,
        "control_abstain_rate": rate(control_abstain, len(rows)),
        "any_abstain": abstain + control_abstain,
        "any_abstain_rate": rate(abstain + control_abstain, len(rows)),
        "control_reproduced_value": control_reproduced,
        "control_reproduced_value_rate": rate(control_reproduced, len(rows)),
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
        result["mean_alternate_minus_abstain_logit"] = sum(
            float(row["alternate_minus_abstain_logit"]) for row in rows
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
        slash_rows = [row for row in rows if row["panel"] in SLASH_BRIDGE_PANELS]
        result[template] = {
            "rows": len(rows),
            "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "selected_label_counts": label_counts(rows),
            "panels": panels,
            "controls": summarize_rows(control_rows),
            "slash_repairs": summarize_rows(slash_rows),
            "holdout_panels": holdout_panels,
        }
    return result


def panel_passed(panel_name: str, panel: dict[str, Any]) -> bool:
    if panel_name == "raw_answer_for_alt":
        return (
            float(panel["parseable_rate"]) >= 0.90
            and float(panel["raw_answer_channel_reproduced_rate"]) >= 0.80
        )
    if panel_name in BRIDGE_PANELS:
        return (
            float(panel["parseable_rate"]) >= 0.90
            and float(panel["evidence_answer_rate"]) >= 0.80
            and float(panel["slash_alt_override_rate"]) <= 0.10
        )
    return (
        float(panel["parseable_rate"]) >= 0.90
        and float(panel["any_abstain_rate"]) >= 0.80
        and float(panel["control_reproduced_value_rate"]) <= 0.10
        and float(panel["other_candidate_rate"]) <= 0.10
    )


def selection_key(summary: dict[str, Any], template: str, templates: tuple[str, ...]) -> tuple[float, ...]:
    panels = summary[template]["panels"]
    repair_rates = tuple(float(panels[panel]["evidence_answer_rate"]) for panel in SLASH_BRIDGE_PANELS)
    control_rates = tuple(float(panels[panel]["any_abstain_rate"]) for panel in CONTROL_PANELS)
    raw_rate = float(panels["raw_answer_for_alt"]["raw_answer_channel_reproduced_rate"])
    return (
        min(repair_rates),
        min(control_rates),
        float(panels["exact_bridge"]["evidence_answer_rate"]),
        raw_rate,
        *repair_rates,
        *control_rates,
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
            "max weakest catalog slash bridge-answer rate",
            "max weakest catalog slash-only control abstain rate",
            "max exact bridge evidence-answer rate",
            "max raw answer_for positive-control reproduction",
            *[f"max {panel} evidence-answer rate" for panel in SLASH_BRIDGE_PANELS],
            *[f"max {panel} abstain rate" for panel in CONTROL_PANELS],
            "earliest template",
        ],
    }


def classify(criteria: dict[str, bool], failed_panels: list[str], selected: dict[str, Any]) -> str:
    if not criteria["structural_passed"]:
        return "structural_invalid"
    if not criteria["selected_prompt_audit_passed"]:
        return "catalog_slash_prompt_audit_failed"
    if not criteria["exact_bridge_passed"]:
        return "catalog_slash_bridge_positive_failed"
    if not criteria["raw_answer_channel_positive_control_passed"]:
        return "raw_function_assignment_channel_not_reproduced"
    if (
        criteria["full_source_count_is_40"]
        and criteria["catalog_slash_bridge_panels_passed"]
        and criteria["catalog_slash_controls_passed"]
        and criteria["source_disjoint_holdout_passed"]
    ):
        return "catalog_slash_full_source_candidate"
    panels = selected["panels"]
    control_leaks = [
        panel
        for panel in CONTROL_PANELS
        if float(panels[panel]["control_reproduced_value_rate"]) > 0.10
        or float(panels[panel]["other_candidate_rate"]) > 0.10
    ]
    slash_overrides = [
        panel
        for panel in SLASH_BRIDGE_PANELS
        if float(panels[panel]["slash_alt_override_rate"]) > 0.10
    ]
    repair_losses = [
        panel
        for panel in SLASH_BRIDGE_PANELS
        if float(panels[panel]["evidence_answer_rate"]) < 0.80
    ]
    if control_leaks and (slash_overrides or repair_losses):
        return "catalog_slash_control_and_repair_leak"
    if control_leaks:
        return "catalog_slash_control_leak"
    if slash_overrides:
        return "catalog_slash_alt_override"
    if repair_losses:
        return "catalog_slash_bridge_loss"
    if not criteria["source_disjoint_holdout_passed"]:
        return "catalog_slash_holdout_failed"
    return "catalog_slash_parse_or_margin_failure"


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
    criteria = {
        "full_run_mode": full_run,
        "structural_passed": structural["passed"],
        "full_source_count_is_40": structural["source_count"] == 40,
        "source_disjoint_holdout": structural["criteria"]["source_split_disjoint"],
        "selected_prompt_audit_passed": prompt_audit_passed,
        "exact_bridge_passed": per_panel_pass["exact_bridge"],
        "raw_answer_channel_positive_control_passed": per_panel_pass["raw_answer_for_alt"],
        "catalog_slash_bridge_panels_passed": all(
            per_panel_pass[panel] for panel in SLASH_BRIDGE_PANELS
        ),
        "catalog_slash_controls_passed": all(
            per_panel_pass[panel] for panel in CONTROL_PANELS
        ),
        "all_panels_passed": not failed_panels,
        "source_disjoint_holdout_passed": holdout_passed,
        "candidate_and_output_margins_reported": score_candidates,
    }
    diagnostic_class = classify(criteria, failed_panels, selected)
    behavior_ready = diagnostic_class == "catalog_slash_full_source_candidate" and full_run
    return {
        "diagnostic_class": diagnostic_class,
        "structural": structural,
        "criteria": criteria,
        "per_panel_pass": per_panel_pass,
        "failed_panels": failed_panels,
        "selection": selection,
        "selected_template_summary": selected,
        "by_template": by_template,
        "catalog_slash_decision": {
            "behavior_ready": behavior_ready,
            "signature_screen_allowed": behavior_ready,
            "hidden_state_claim_allowed": False,
            "intervention_allowed": False,
            "mechanism_claim_allowed": False,
            "route_decision": "catalog_slash_admitted_for_margin_matched_signature_screen"
            if behavior_ready
            else "catalog_slash_full_source_diagnostic",
            "exported_diagnostic_class": "CATALOG_SLASH_FULL_SOURCE_CANDIDATE"
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
            "catalog_slash_decision": {
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
        "parent_card_id": "KSQ014",
        "candidate_id": CANDIDATE_ID,
        "run_type": BEHAVIOR_RUN_TYPE if outputs is not None else RUN_TYPE,
        "model_id": MODEL_ID,
        "full_run": full_run,
        "templates": list(templates),
        "panels": list(PANELS),
        "summary": summary,
        "records": outputs if outputs is not None else records,
        "allowed_claim": (
            "KSQ015 tests whether the KSQ014 catalog slash locality survivor "
            "becomes a full-source behavior substrate after removing bare "
            "slash. Raw answer_for remains only a positive-control pressure "
            "channel. A pass admits only a later signature screen."
        ),
        "forbidden_claim": (
            "KSQ015 does not prove uncertainty, factuality, hidden-state "
            "causality, intervention success, or a mechanism."
        ),
    }


def write_prereg(path: Path) -> None:
    lines = [
        "# KSQ015 Catalog Slash Full-Source Packet",
        "",
        "Status: KSQ014 diagnostic follow-up.",
        "",
        "Runner:",
        "",
        "> `code/ksq015_catalog_slash_full_source_packet.py`",
        "",
        "Artifacts:",
        "",
        "> `results/cards/KSQ015_CATALOG_SLASH_FULL_SOURCE_PACKET/ksq015_catalog_slash_full_source_first_run.json`",
        "",
        "> `results/cards/KSQ015_CATALOG_SLASH_FULL_SOURCE_PACKET/ksq015_catalog_slash_full_source_smoke_limit10.json`",
        "",
        "> `results/cards/KSQ015_CATALOG_SLASH_FULL_SOURCE_PACKET/ksq015_catalog_slash_full_source_full_behavior.json`",
        "",
        "## Claim Under Test",
        "",
        "KSQ014 showed that catalog-labeled slash text survived a 10-source",
        "locality smoke, while bare slash failed. KSQ015 removes bare slash and",
        "tests whether catalog slash alone survives all 40 sources and the",
        "source-disjoint holdout.",
        "",
        "## Promotion Rule",
        "",
        "Promote only to behavior-substrate status if exact bridge answers, raw",
        "answer_for remains an active positive control, every catalog slash",
        "bridge panel answers the counted bridge rather than the alternate, every",
        "catalog slash-only no-bridge control abstains, query-only abstains,",
        "selected prompt audit passes, source-disjoint holdout passes, and",
        "candidate margins are reported on the full 40-source run.",
        "",
        "A promotion licenses only a later margin-matched signature screen. It",
        "does not license an intervention or mechanism claim.",
        "",
        "## Kill / Boundary Rule",
        "",
        "If raw answer_for no longer reproduces, the packet is not anchored to the",
        "KSQ011-KSQ014 pressure. If catalog slash bridge panels override the",
        "bridge, export catalog slash override. If catalog slash-only controls",
        "reproduce a value, export catalog slash locality leakage. If source-",
        "disjoint holdout fails while aggregate panels pass, export holdout",
        "fragility.",
        "",
        "## Forbidden Claims",
        "",
        "- This is not a mechanism card.",
        "- This is not real uncertainty, refusal, or factual correction.",
        "- A behavior pass would not be a hidden-state or intervention result.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_status_card(path: Path, result: dict[str, Any], output_path: Path) -> None:
    summary = result["summary"]
    criteria = summary["criteria"]
    decision = summary["catalog_slash_decision"]
    lines = [
        "# KSQ015 Catalog Slash Full-Source Packet Status",
        "",
        "Date: 2026-07-02",
        "",
        "Runner:",
        "",
        "> `code/ksq015_catalog_slash_full_source_packet.py`",
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
            if panel_name == "raw_answer_for_alt":
                key_rate = panel["raw_answer_channel_reproduced_rate"]
            elif panel_name in BRIDGE_PANELS:
                key_rate = panel["evidence_answer_rate"]
            else:
                key_rate = panel["any_abstain_rate"]
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
            "- KSQ015 is a mechanism card.",
            "- KSQ015 proves real uncertainty, refusal, factual correction, or knowledge control.",
            "- KSQ015 licenses hidden-state intervention or mechanism claims.",
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
    if args.score and not result["summary"]["criteria"].get("structural_passed", False):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
