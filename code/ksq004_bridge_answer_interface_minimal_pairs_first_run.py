#!/usr/bin/env python
"""KSQ004 bridge answer-interface minimal-pairs first run.

This is a behavior-substrate admission runner for the knowledge bridge ladder.
It tests whether local-table versus learned-fact branch behavior survives when
both branches use the same bare numeric answer interface. The first purpose is
shortcut detection: if the model only looks controlled because branch prompts,
answer shape, side numbers, or final output geometry carry the behavior, the run
must record that as a diagnostic before any hidden-state work is allowed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from mc012_reliability_labeled_numeric_arbitration import (
    ANSWER_RE,
    MODEL_ID,
    base_sources,
    candidate_logprob_payload,
    distractors,
    final_next_token_logits,
    generate_answer,
    response_suffix,
    strict_parse,
    summarize_rows as base_summarize_rows,
    word_occurrences,
)
from mc016_alphabet_gated_numeric_arbitration import load_model_and_tokenizer


CARD_ID = "KSQ004"
CANDIDATE_ID = "ksq004_bridge_answer_interface_minimal_pairs"
RUN_TYPE = "ksq004_bridge_answer_interface_minimal_pairs_structural_gate"
BEHAVIOR_RUN_TYPE = "ksq004_bridge_answer_interface_minimal_pairs_behavior"
RESULT_DIR = Path("results/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS")
DEFAULT_RESULT_PATH = RESULT_DIR / "ksq004_bridge_answer_interface_minimal_pairs_first_run.json"
SMOKE_LIMIT10_RESULT_PATH = RESULT_DIR / "ksq004_bridge_answer_interface_minimal_pairs_smoke_limit10.json"
FULL_BEHAVIOR_RESULT_PATH = RESULT_DIR / "ksq004_bridge_answer_interface_minimal_pairs_full_behavior.json"
STATUS_CARD = Path("research/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS_FIRST_RUN_STATUS.md")
PREREG_PATH = Path("research/prereg/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS_FIRST_RUN.md")

PANELS = (
    "matched_minimal_pairs",
    "local_learned_direct_controls",
    "minimal_pair_conflict",
    "side_answer_leakage",
    "null_and_holdout",
)
PRIMARY_CONFLICT_PANELS = ("minimal_pair_conflict",)
TEMPLATES = ("question_form", "compact_form")
SUBTYPES_BY_PANEL = {
    "matched_minimal_pairs": ("matched_local_contract", "matched_atomic_contract"),
    "local_learned_direct_controls": ("local_direct", "atomic_direct"),
    "minimal_pair_conflict": ("local_branch_conflict", "atomic_branch_conflict"),
    "side_answer_leakage": ("local_side_leakage", "atomic_side_leakage"),
    "null_and_holdout": ("answer_absent_null",),
}
LOCAL_SUBTYPES = {
    "matched_local_contract",
    "local_direct",
    "local_branch_conflict",
    "local_side_leakage",
}
ATOMIC_SUBTYPES = {
    "matched_atomic_contract",
    "atomic_direct",
    "atomic_branch_conflict",
    "atomic_side_leakage",
}
NULL_SUBTYPES = {"answer_absent_null"}
ANSWER_INTERFACE_WORD_RE = re.compile(
    r"\b(choice|option|letter|multiple choice|a/b|a\s+or\s+b)\b",
    re.IGNORECASE,
)


def line_contains_both(text: str, left: str, right: str) -> bool:
    return any(left in line and right in line for line in text.splitlines())


def render_local_table(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    include_query: bool,
    template: str,
) -> list[str]:
    rows = distractors(sources, source, 4)
    visible_rows = rows if include_query else [row for row in rows if row["source_id"] != source["source_id"]]
    if template == "question_form":
        lines = [
            "A local lab table may define temporary element numbers.",
            "The requested relation decides which number to return.",
            "Local lab table:",
        ]
    elif template == "compact_form":
        lines = [
            "Numeric lookup packet.",
            "Relation names may refer to the table or standard chemistry.",
            "Local lab table:",
        ]
    else:
        raise ValueError(f"unknown template {template}")
    for row in visible_rows:
        lines.append(f"{row['element']} has local lab number {row['local_number']}.")
    return lines


def append_query(lines: list[str], source: dict[str, Any], *, relation: str, template: str) -> None:
    if template == "question_form":
        lines.extend(
            [
                f"Requested relation: {relation}.",
                f"Query element: {source['element']}",
            ]
        )
    elif template == "compact_form":
        compact_relation = relation.replace(" ", "_")
        lines.extend(
            [
                f"Relation={compact_relation}",
                f"Element={source['element']}",
            ]
        )
    else:
        raise ValueError(f"unknown template {template}")


def make_prompt(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    panel: str,
    subtype: str,
    template: str,
) -> tuple[str, str]:
    expected_label: str
    if subtype in LOCAL_SUBTYPES:
        expected_label = "local_number"
    elif subtype in ATOMIC_SUBTYPES:
        expected_label = "atomic_number"
    elif subtype in NULL_SUBTYPES:
        expected_label = "unknown"
    else:
        raise ValueError(f"unknown subtype {subtype}")

    if panel == "matched_minimal_pairs":
        if subtype == "matched_local_contract":
            lines = render_local_table(sources, source, include_query=True, template=template)
            append_query(lines, source, relation="local table number", template=template)
        elif subtype == "matched_atomic_contract":
            lines = [
                "No local lab table is active for this item.",
                "Use standard chemistry for the requested relation.",
            ]
            append_query(lines, source, relation="standard atomic number", template=template)
        else:
            raise ValueError(f"bad subtype {subtype} for {panel}")
    elif panel == "local_learned_direct_controls":
        if subtype == "local_direct":
            lines = render_local_table(sources, source, include_query=True, template=template)
            append_query(lines, source, relation="local table number", template=template)
        elif subtype == "atomic_direct":
            lines = [
                "No local lab table is active for this item.",
                "Use standard chemistry for the requested relation.",
            ]
            append_query(lines, source, relation="standard atomic number", template=template)
        else:
            raise ValueError(f"bad subtype {subtype} for {panel}")
    elif panel == "minimal_pair_conflict":
        lines = render_local_table(sources, source, include_query=True, template=template)
        if subtype == "local_branch_conflict":
            append_query(lines, source, relation="local table number", template=template)
        elif subtype == "atomic_branch_conflict":
            append_query(lines, source, relation="standard atomic number", template=template)
        else:
            raise ValueError(f"bad subtype {subtype} for {panel}")
    elif panel == "side_answer_leakage":
        lines = render_local_table(sources, source, include_query=True, template=template)
        lines.extend(
            [
                f"Side note number: {source['lure_atomic_number']}.",
                "The side note number is not the requested relation.",
            ]
        )
        if subtype == "local_side_leakage":
            append_query(lines, source, relation="local table number", template=template)
        elif subtype == "atomic_side_leakage":
            append_query(lines, source, relation="standard atomic number", template=template)
        else:
            raise ValueError(f"bad subtype {subtype} for {panel}")
    elif panel == "null_and_holdout":
        if subtype != "answer_absent_null":
            raise ValueError(f"bad subtype {subtype} for {panel}")
        lines = render_local_table(sources, source, include_query=False, template=template)
        append_query(lines, source, relation="local table number", template=template)
        lines.append("If the query element is absent from the local lab table, return UNKNOWN.")
    else:
        raise ValueError(f"unknown panel {panel}")
    return "\n".join([*lines, *response_suffix()]), expected_label


def source_records(
    limit_sources: int | None = None,
    templates: tuple[str, ...] = TEMPLATES,
    run_type: str = RUN_TYPE,
) -> list[dict[str, Any]]:
    sources = base_sources(limit_sources)
    records: list[dict[str, Any]] = []
    for source in sources:
        for template in templates:
            for panel_name in PANELS:
                for subtype in SUBTYPES_BY_PANEL[panel_name]:
                    prompt, expected_label = make_prompt(
                        sources,
                        source,
                        panel=panel_name,
                        subtype=subtype,
                        template=template,
                    )
                    records.append(
                        {
                            "id": f"{CARD_ID}_{template}_{panel_name}_{subtype}_{source['source_id']}",
                            "card_id": CARD_ID,
                            "candidate_id": CANDIDATE_ID,
                            "run_type": run_type,
                            "model_id": MODEL_ID,
                            "template": template,
                            "panel": panel_name,
                            "subtype": subtype,
                            "split": source["split"],
                            "source_id": source["source_id"],
                            "source_index": source["source_index"],
                            "element": source["element"],
                            "atomic_number": str(source["atomic_number"]),
                            "lure_atomic_number": str(source["lure_atomic_number"]),
                            "local_number": str(source["local_number"]),
                            "expected_label": expected_label,
                            "expected_local_answer": str(source["local_number"]),
                            "expected_real_answer": str(source["atomic_number"]),
                            "expected_null_answer": "UNKNOWN",
                            "side_answer": str(source["lure_atomic_number"]),
                            "prompt": prompt,
                            "candidate_answers": [
                                str(source["local_number"]),
                                str(source["atomic_number"]),
                                str(source["lure_atomic_number"]),
                                "UNKNOWN",
                            ],
                        }
                    )
    return records


def structural_check(records: list[dict[str, Any]], templates: tuple[str, ...]) -> dict[str, Any]:
    source_ids = {row["source_id"] for row in records}
    panels = Counter(row["panel"] for row in records)
    template_counts = Counter(row["template"] for row in records)
    subtype_counts = Counter(row["subtype"] for row in records)
    split_source_ids: dict[str, set[str]] = {}
    for row in records:
        split_source_ids.setdefault(row["split"], set()).add(row["source_id"])

    primary_rows = [row for row in records if row["panel"] in PRIMARY_CONFLICT_PANELS]
    null_rows = [row for row in records if row["subtype"] == "answer_absent_null"]
    matched_rows = [row for row in records if row["panel"] == "matched_minimal_pairs"]
    side_rows = [row for row in records if row["panel"] == "side_answer_leakage"]
    malformed_candidates = [
        row["id"]
        for row in records
        for candidate in row["candidate_answers"]
        if ANSWER_RE.match(candidate) is None
    ]
    candidate_collision_rows = [
        row["id"] for row in records if len(set(row["candidate_answers"])) != len(row["candidate_answers"])
    ]
    primary_real_number_leaks = [
        row["id"] for row in primary_rows if word_occurrences(row["prompt"], row["atomic_number"]) > 0
    ]
    null_query_present = [
        row["id"]
        for row in null_rows
        if line_contains_both(row["prompt"], row["element"], row["local_number"])
    ]
    side_note_missing = [
        row["id"]
        for row in side_rows
        if word_occurrences(row["prompt"], row["lure_atomic_number"]) != 1
    ]
    answer_interface_leaks = [
        row["id"] for row in records if ANSWER_INTERFACE_WORD_RE.search(row["prompt"])
    ]
    answer_suffixes = {"\n".join(row["prompt"].splitlines()[-3:]) for row in records}
    primary_expected = Counter(row["expected_label"] for row in primary_rows)
    matched_expected = Counter(row["expected_label"] for row in matched_rows)
    expected_rows = sum(
        len(source_ids) * len(templates) * len(SUBTYPES_BY_PANEL[panel_name])
        for panel_name in PANELS
    )
    criteria = {
        "expected_row_count": len(records) == expected_rows,
        "all_panels_present": set(panels) == set(PANELS),
        "all_templates_present": set(template_counts) == set(templates),
        "all_subtypes_present": set(subtype_counts) == {
            subtype for subtypes in SUBTYPES_BY_PANEL.values() for subtype in subtypes
        },
        "source_split_disjoint": sum(len(ids) for ids in split_source_ids.values()) == len(source_ids),
        "holdout_sources_present": bool(split_source_ids.get("holdout")),
        "calibration_sources_present": bool(split_source_ids.get("calibration")),
        "primary_expected_labels_balanced": primary_expected["local_number"] == primary_expected["atomic_number"],
        "matched_expected_labels_balanced": matched_expected["local_number"] == matched_expected["atomic_number"],
        "primary_atomic_answer_hidden": not primary_real_number_leaks,
        "answer_absent_omits_query_local_number": not null_query_present,
        "side_panels_have_single_side_number": not side_note_missing,
        "candidate_answers_parseable": not malformed_candidates,
        "no_candidate_collisions": not candidate_collision_rows,
        "single_answer_suffix": len(answer_suffixes) == 1,
        "no_multiple_choice_interface_words": not answer_interface_leaks,
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "record_count": len(records),
        "source_count": len(source_ids),
        "panel_counts": dict(sorted(panels.items())),
        "template_counts": dict(sorted(template_counts.items())),
        "subtype_counts": dict(sorted(subtype_counts.items())),
        "split_source_counts": {
            split: len(ids) for split, ids in sorted(split_source_ids.items())
        },
        "primary_expected_label_counts": dict(sorted(primary_expected.items())),
        "matched_expected_label_counts": dict(sorted(matched_expected.items())),
        "expected_rows": expected_rows,
        "primary_real_number_leak_ids": primary_real_number_leaks[:20],
        "null_query_present_ids": null_query_present[:20],
        "side_note_missing_ids": side_note_missing[:20],
        "malformed_candidate_ids": malformed_candidates[:20],
        "candidate_collision_ids": candidate_collision_rows[:20],
        "answer_interface_leak_ids": answer_interface_leaks[:20],
        "answer_suffix_count": len(answer_suffixes),
    }


def prompt_audit(record: dict[str, Any]) -> dict[str, Any]:
    prompt = record["prompt"]
    primary = record["panel"] in PRIMARY_CONFLICT_PANELS
    null = record["subtype"] == "answer_absent_null"
    side = record["panel"] == "side_answer_leakage"
    return {
        "real_atomic_number_occurrences": word_occurrences(prompt, record["atomic_number"]),
        "lure_atomic_number_occurrences": word_occurrences(prompt, record["lure_atomic_number"]),
        "real_atomic_number_hidden_in_primary": not primary
        or word_occurrences(prompt, record["atomic_number"]) == 0,
        "answer_absent_omits_query_local_number": not null
        or not line_contains_both(prompt, record["element"], record["local_number"]),
        "side_number_count_expected": not side
        or word_occurrences(prompt, record["lure_atomic_number"]) == 1,
        "shared_response_suffix_present": prompt.endswith("\n".join(response_suffix())),
        "no_multiple_choice_interface_words": ANSWER_INTERFACE_WORD_RE.search(prompt) is None,
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
        output = {
            **record,
            "index": index,
            **generated,
            **parsed,
            **prompt_audit(record),
            "is_primary_conflict_panel": record["panel"] in PRIMARY_CONFLICT_PANELS,
            "is_binary_conflict": record["panel"] in PRIMARY_CONFLICT_PANELS
            and parsed["selected_label"] in {"local_number", "atomic_number", "lure_atomic_number"},
            "is_expected_correct": parsed["selected_label"] == record["expected_label"],
            "is_side_answer": parsed["selected_label"] == "lure_atomic_number",
            "is_atomic_or_lure_number": parsed["selected_label"] in {"atomic_number", "lure_atomic_number"},
        }
        if score_candidates:
            output.update(final_next_token_logits(model, tokenizer, record["prompt"], record))
            output.update(candidate_logprob_payload(model, tokenizer, record["prompt"], record))
        outputs.append(output)
        if verbose:
            print(
                f"[{index:03d}/{len(records):03d}] {record['id']} split={record['split']} "
                f"panel={record['panel']} subtype={record['subtype']} -> "
                f"{output['selected_label']} {str(output['selected_answer'])!r} "
                f"generated={generated['generated_text']!r}"
            )
    return outputs


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("selected_label")) for row in rows).items()))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = base_summarize_rows(rows)
    expected_correct = sum(1 for row in rows if row.get("is_expected_correct"))
    side_answer = sum(1 for row in rows if row.get("is_side_answer"))
    result.update(
        {
            "expected_correct": expected_correct,
            "expected_correct_rate": rate(expected_correct, len(rows)),
            "side_answer": side_answer,
            "side_answer_rate": rate(side_answer, len(rows)),
            "expected_label_counts": dict(sorted(Counter(row.get("expected_label") for row in rows).items())),
        }
    )
    return result


def margin_audits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows or "final_local_minus_atomic_number_logit" not in rows[0]:
        return {"reported": False}
    finite_logit = [
        abs(float(row["final_local_minus_atomic_number_logit"]))
        for row in rows
        if math.isfinite(float(row["final_local_minus_atomic_number_logit"]))
    ]
    finite_candidate = [
        abs(float(row["candidate_local_minus_atomic_number_mean_logprob"]))
        for row in rows
        if math.isfinite(float(row["candidate_local_minus_atomic_number_mean_logprob"]))
    ]
    return {
        "reported": True,
        "mean_abs_final_local_atomic_logit_gap": sum(finite_logit) / len(finite_logit)
        if finite_logit
        else None,
        "mean_abs_candidate_local_atomic_logprob_gap": sum(finite_candidate) / len(finite_candidate)
        if finite_candidate
        else None,
    }


def template_summary(outputs: list[dict[str, Any]], templates: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for template in templates:
        rows = [row for row in outputs if row["template"] == template]
        panels = {
            panel_name: summarize_rows([row for row in rows if row["panel"] == panel_name])
            for panel_name in PANELS
        }
        subtypes = {
            subtype: summarize_rows([row for row in rows if row["subtype"] == subtype])
            for subtypes_for_panel in SUBTYPES_BY_PANEL.values()
            for subtype in subtypes_for_panel
        }
        conflict = [row for row in rows if row["panel"] in PRIMARY_CONFLICT_PANELS]
        holdout_conflict = [row for row in conflict if row["split"] == "holdout"]
        result[template] = {
            "rows": len(rows),
            "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "selected_label_counts": label_counts(rows),
            "panels": panels,
            "subtypes": subtypes,
            "primary_conflict": summarize_rows(conflict),
            "primary_conflict_holdout": summarize_rows(holdout_conflict),
            "primary_conflict_margin_audits": margin_audits(conflict),
        }
    return result


def selection_key(summary: dict[str, Any], template: str, templates: tuple[str, ...]) -> tuple[float, ...]:
    item = summary[template]
    subtype = item["subtypes"]
    matched = item["panels"]["matched_minimal_pairs"]
    conflict = item["primary_conflict"]
    side = item["panels"]["side_answer_leakage"]
    null = item["panels"]["null_and_holdout"]
    direct_floor = min(
        float(subtype["local_direct"]["local_number_rate"]),
        float(subtype["atomic_direct"]["atomic_number_rate"]),
    )
    branch_floor = min(
        float(subtype["local_branch_conflict"]["local_number_rate"]),
        float(subtype["atomic_branch_conflict"]["atomic_number_rate"]),
    )
    side_safety = 1.0 - float(side["side_answer_rate"])
    return (
        direct_floor,
        float(matched["expected_correct_rate"]),
        branch_floor,
        float(conflict["expected_correct_rate"]),
        float(null["unknown_rate"]),
        side_safety,
        float(conflict["parseable_rate"]),
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
            "max local and learned direct-control floor",
            "max matched minimal-pair expected correctness",
            "max local and learned conflict-branch floor",
            "max primary conflict expected correctness",
            "max answer-absent null UNKNOWN rate",
            "min side-answer leakage",
            "max primary conflict parseability",
            "earliest template",
        ],
    }


def classify(criteria: dict[str, bool]) -> str:
    if not criteria["structural_passed"]:
        return "structural_invalid"
    if not criteria["selected_prompt_audit_passed"]:
        return "answer_interface_prompt_audit_failed"
    if not criteria["matched_minimal_pairs_passed"]:
        return "bridge_minimal_pair_contract_failed"
    if not criteria["local_learned_direct_controls_passed"]:
        return "bridge_direct_control_failed"
    if not criteria["minimal_pair_conflict_passed"]:
        return "bridge_minimal_pair_contrast_absent"
    if not criteria["side_answer_leakage_passed"]:
        return "answer_interface_branch_shortcut"
    if not criteria["null_and_holdout_passed"]:
        return "answer_interface_null_or_holdout_failed"
    if not criteria["candidate_and_output_margins_reported"]:
        return "bridge_answer_interface_behavior_passed_baselines_missing"
    if criteria["smoke_mode"]:
        return "smoke_answer_interface_candidate"
    return "bridge_answer_interface_behavior_ready"


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
    subtypes = selected["subtypes"]
    matched = panels["matched_minimal_pairs"]
    conflict = selected["primary_conflict"]
    side = panels["side_answer_leakage"]
    null = panels["null_and_holdout"]
    holdout_conflict = selected["primary_conflict_holdout"]
    selected_rows = [row for row in outputs if row["template"] == selection["selected_template"]]
    selected_prompt_audit_passed = all(
        row["real_atomic_number_hidden_in_primary"]
        and row["answer_absent_omits_query_local_number"]
        and row["side_number_count_expected"]
        and row["shared_response_suffix_present"]
        and row["no_multiple_choice_interface_words"]
        for row in selected_rows
    )
    conflict_min_count = 24 if full_run else 6
    holdout_min_rate = 0.65 if full_run else 0.50
    conflict_expected_counts = conflict["expected_label_counts"]
    null_unknown_rate = float(null["unknown_rate"])
    holdout_expected_rate = float(holdout_conflict["expected_correct_rate"])
    criteria = {
        "smoke_mode": not full_run,
        "structural_passed": structural["passed"],
        "full_source_count_is_40": structural["source_count"] == 40,
        "source_disjoint_holdout": structural["criteria"]["source_split_disjoint"]
        and structural["criteria"]["holdout_sources_present"]
        and structural["criteria"]["calibration_sources_present"],
        "selected_prompt_audit_passed": selected_prompt_audit_passed,
        "answer_shape_balance_passed": structural["criteria"]["primary_expected_labels_balanced"]
        and structural["criteria"]["matched_expected_labels_balanced"]
        and structural["criteria"]["single_answer_suffix"],
        "matched_minimal_pairs_passed": float(matched["parseable_rate"]) >= 0.90
        and float(matched["expected_correct_rate"]) >= 0.75,
        "local_learned_direct_controls_passed": float(subtypes["local_direct"]["parseable_rate"]) >= 0.90
        and float(subtypes["local_direct"]["local_number_rate"]) >= 0.75
        and float(subtypes["atomic_direct"]["parseable_rate"]) >= 0.90
        and float(subtypes["atomic_direct"]["atomic_number_rate"]) >= 0.75,
        "minimal_pair_conflict_passed": float(conflict["parseable_rate"]) >= 0.90
        and float(conflict["expected_correct_rate"]) >= 0.70
        and int(conflict_expected_counts.get("local_number", 0)) >= conflict_min_count
        and int(conflict_expected_counts.get("atomic_number", 0)) >= conflict_min_count,
        "side_answer_leakage_passed": float(side["parseable_rate"]) >= 0.90
        and float(side["side_answer_rate"]) <= 0.10,
        "null_and_holdout_passed": null_unknown_rate >= 0.80 and holdout_expected_rate >= holdout_min_rate,
        "candidate_and_output_margins_reported": bool(
            score_candidates and selected["primary_conflict_margin_audits"]["reported"]
        ),
    }
    diagnostic_class = classify(criteria)
    behavior_ready = diagnostic_class == "bridge_answer_interface_behavior_ready"
    behavior_candidate = diagnostic_class in {
        "smoke_answer_interface_candidate",
        "bridge_answer_interface_behavior_ready",
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
        "selected_controls": {
            panel_name: summary["selected_template_summary"]["panels"][panel_name]
            for panel_name in PANELS
        },
        "selected_subtypes": summary["selected_template_summary"]["subtypes"],
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
        "# KSQ004 Bridge Answer-Interface Minimal Pairs First Run",
        "",
        "Status: behavior-substrate first-run preregistration; no hidden-state work.",
        "",
        "Runner:",
        "",
        "> `code/ksq004_bridge_answer_interface_minimal_pairs_first_run.py`",
        "",
        "Default result:",
        "",
        "> `results/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS/ksq004_bridge_answer_interface_minimal_pairs_first_run.json`",
        "",
        "Default 10-source smoke result:",
        "",
        "> `results/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS/ksq004_bridge_answer_interface_minimal_pairs_smoke_limit10.json`",
        "",
        "Full behavior result:",
        "",
        "> `results/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS/ksq004_bridge_answer_interface_minimal_pairs_full_behavior.json`",
        "",
        "Status card:",
        "",
        "> `research/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS_FIRST_RUN_STATUS.md`",
        "",
        "## Purpose",
        "",
        "Test whether bridge-style local-vs-learned behavior survives a shared",
        "bare numeric answer interface. The run is designed to kill answer-shape",
        "and branch-shortcut explanations before hidden-state search.",
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
            "Promote only to behavior-substrate admission if matched minimal",
            "pairs parse under the same answer schema, direct local and learned",
            "controls both pass, conflict minimal pairs produce the requested",
            "branch on non-holdout and holdout rows, side-answer leakage stays",
            "low, answer-absent rows abstain, and candidate/output baseline",
            "reporting is present.",
            "",
            "Death rule: export `ANSWER_INTERFACE_BRANCH_SHORTCUT` if side",
            "answers or answer-shape shortcuts explain the behavior; export",
            "`BRIDGE_MINIMAL_PAIR_CONTRAST_ABSENT` if balancing the answer",
            "interface removes the local-vs-learned contrast.",
            "",
            "Forbidden claims:",
            "",
            "- KSQ004 is a mechanism card.",
            "- KSQ004 licenses hidden-state search before the behavior gate passes.",
            "- KSQ004 proves a learned-memory bridge or answer-interface control surface.",
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
            f"- subtypes: `{json.dumps(structural['subtype_counts'], sort_keys=True)}`",
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
            "| Panel | Label counts | Expected-correct rate |",
            "| --- | --- | --- |",
        ]
    )
    for panel_name in PANELS:
        panel = selected["panels"][panel_name]
        labels = panel["label_counts"]
        lines.append(
            f"| `{panel_name}` | `{json.dumps(labels, sort_keys=True)}` | `{panel['expected_correct_rate']:.3f}` |"
        )
    if "by_template" in summary:
        lines.extend(
            [
                "",
                "Template contrast:",
                "",
                "| Template | Conflict expected-correct | Atomic-branch atomic rate | Holdout expected-correct |",
                "| --- | --- | --- | --- |",
            ]
        )
        for template_name in TEMPLATES:
            template_summary = summary["by_template"][template_name]
            conflict = template_summary["primary_conflict"]
            atomic_branch = template_summary["subtypes"]["atomic_branch_conflict"]
            holdout = template_summary["primary_conflict_holdout"]
            lines.append(
                f"| `{template_name}` | `{conflict['expected_correct_rate']:.3f}` | "
                f"`{atomic_branch['atomic_number_rate']:.3f}` | `{holdout['expected_correct_rate']:.3f}` |"
            )
    lines.extend(
        [
            "",
            "Subtype counts:",
            "",
            "| Subtype | Label counts |",
            "| --- | --- |",
        ]
    )
    for subtype in [
        subtype for subtypes_for_panel in SUBTYPES_BY_PANEL.values() for subtype in subtypes_for_panel
    ]:
        labels = selected["subtypes"][subtype]["label_counts"]
        lines.append(f"| `{subtype}` | `{json.dumps(labels, sort_keys=True)}` |")
    lines.extend(
        [
            "",
            "Interpretation: KSQ004 is a shortcut detector. A pass would only",
            "license a later signature-screen decision. A fail identifies whether",
            "the bridge disappears under matched answer interface, leaks through",
            "side answers, fails null/holdout, or is already output-geometry visible.",
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
        "# KSQ004 Bridge Answer-Interface Minimal Pairs First-Run Status",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "Runner:",
        "",
        "> `code/ksq004_bridge_answer_interface_minimal_pairs_first_run.py`",
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
            "- KSQ004 is a mechanism card.",
            "- KSQ004 supports intervention.",
            "- KSQ004 found an internal bridge, knowledge, or answer-interface control surface.",
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
    parser.add_argument("--max-new-tokens", type=int, default=8)
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
            "purpose": "Structural gate for the KSQ004 answer-interface minimal-pairs first run.",
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
                {
                    "passed": False,
                    "diagnostic_class": "structural_invalid",
                    "structural": structural,
                },
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
        "purpose": "Behavior-only KSQ004 answer-interface minimal-pairs admission run.",
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
