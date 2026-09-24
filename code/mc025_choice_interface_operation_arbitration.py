#!/usr/bin/env python
"""MC025 constrained-choice operation arbitration.

MC024 showed that balanced worked examples repair prompt-local operation
routing but leave the learned atomic branch below gate in an open integer
answer interface, with substantial other-number leakage. MC025 asks the next
narrow diagnostic question: if the answer space is constrained to A/B/C choices
containing the local number, the atomic number, and UNKNOWN, does the operation
rule become reliable?

This is intentionally prompt-visible. Conflict prompts include the atomic
number in the answer options, so a pass would diagnose an answer-interface or
choice-routing bottleneck, not a learned-memory mechanism substrate.
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
    MODEL_ID,
    base_sources,
    distractors,
    generate_answer,
    line_contains_both,
    word_occurrences,
)
from mc016_alphabet_gated_numeric_arbitration import load_model_and_tokenizer
from mc023_query_operation_numeric_arbitration import (
    OPERATION_ASSIGNMENTS,
    OPERATION_CODES,
    RULE_ORDERS,
    append_operation_rules,
    operation_for_label,
    operation_mapping,
    operation_query_line,
)
from mc024_fewshot_operation_numeric_arbitration import operation_example_lines


CARD_ID = "MC025"
RUN_TYPE = "choice_interface_operation_arbitration_structural_gate"
BEHAVIOR_RUN_TYPE = "choice_interface_operation_arbitration_behavior"
RESULT_DIR = Path("results/cards/MC025")
STATUS_CARD = Path("research/cards/MC025_CHOICE_INTERFACE_OPERATION_ARBITRATION_BEHAVIOR_STATUS.md")

PANELS = (
    "synthetic_choice_lookup",
    "familiar_choice_lookup",
    "atomic_choice_control",
    "operation_local_choice_conflict",
    "operation_atomic_choice_conflict",
    "operation_rule_absent_choice_null",
    "answer_absent_choice_null",
)
PRIMARY_CONFLICT_PANELS = ("operation_local_choice_conflict", "operation_atomic_choice_conflict")
TEMPLATES = ("compact_choice_examples",)
OPTION_ORDERS = (
    "local_atomic_unknown",
    "atomic_unknown_local",
    "unknown_local_atomic",
)
CHOICES = ("A", "B", "C")
CHOICE_RE = re.compile(r"^([ABC])[\s\.\):,;!\?]*$", re.IGNORECASE)
STATUS_LEXEME_RE = re.compile(r"\b(trusted|untrusted|reliable|unreliable|status)\b", re.IGNORECASE)


def response_suffix() -> list[str]:
    return [
        "Return only one final token: A, B, or C.",
        "Do not explain.",
        "Answer:",
    ]


def option_order_labels(option_order: str) -> tuple[str, str, str]:
    if option_order == "local_atomic_unknown":
        return ("local_number", "atomic_number", "unknown")
    if option_order == "atomic_unknown_local":
        return ("atomic_number", "unknown", "local_number")
    if option_order == "unknown_local_atomic":
        return ("unknown", "local_number", "atomic_number")
    raise ValueError(f"unknown option order {option_order}")


def option_value(label: str, source: dict[str, Any]) -> str:
    if label == "local_number":
        return str(source["local_number"])
    if label == "atomic_number":
        return str(source["atomic_number"])
    if label == "unknown":
        return "UNKNOWN"
    raise ValueError(f"unknown option label {label}")


def choice_options(source: dict[str, Any], option_order: str) -> tuple[list[str], dict[str, str], dict[str, str]]:
    labels = option_order_labels(option_order)
    lines = ["Answer choices:"]
    choice_to_label: dict[str, str] = {}
    label_to_choice: dict[str, str] = {}
    for choice, label in zip(CHOICES, labels, strict=True):
        choice_to_label[choice] = label
        label_to_choice[label] = choice
        lines.append(f"{choice}. {option_value(label, source)}")
    return lines, choice_to_label, label_to_choice


def local_table_lines(
    rows: list[dict[str, Any]],
    query: dict[str, Any],
    *,
    include_query: bool,
    key_field: str = "element",
) -> list[str]:
    visible_rows = rows if include_query else [row for row in rows if row["source_id"] != query["source_id"]]
    lines = [
        "Rows:",
    ]
    for row in visible_rows:
        lines.append(f"{row[key_field]} has local lab number {row['local_number']}.")
    return lines


def append_rules_and_examples(
    lines: list[str],
    example_rows: list[dict[str, Any]],
    *,
    assignment: str,
    rule_order: str,
) -> None:
    append_operation_rules(lines, assignment=assignment, rule_order=rule_order, template="compact_operation_rules")
    for item in operation_example_lines(example_rows, assignment=assignment)[1:]:
        lines.append(item)


def make_prompt(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    panel: str,
    template: str,
    assignment: str,
    rule_order: str,
    option_order: str,
) -> tuple[str, str, str, dict[str, str], dict[str, str]]:
    if template != "compact_choice_examples":
        raise ValueError(f"unknown template {template}")
    rows = distractors(sources, source, 5)
    example_rows = [row for row in rows if row["source_id"] != source["source_id"]][:2]
    option_lines, choice_to_label, label_to_choice = choice_options(source, option_order)

    if panel == "synthetic_choice_lookup":
        lines = local_table_lines(rows, source, include_query=True, key_field="synthetic_key")
        lines.append(f"Query key: {source['synthetic_key']}")
        lines.append("Choose the local lab number.")
        lines.extend(option_lines)
        return "\n".join([*lines, *response_suffix()]), "local_number", "NONE", choice_to_label, label_to_choice
    if panel == "familiar_choice_lookup":
        lines = local_table_lines(rows, source, include_query=True)
        lines.append(f"Query element: {source['element']}")
        lines.append("Choose the local lab number.")
        lines.extend(option_lines)
        return "\n".join([*lines, *response_suffix()]), "local_number", "NONE", choice_to_label, label_to_choice
    if panel == "atomic_choice_control":
        lines = [
            "No local lab table is active.",
            "Choose the standard atomic number for the queried element.",
            f"Query element: {source['element']}",
        ]
        lines.extend(option_lines)
        return "\n".join([*lines, *response_suffix()]), "atomic_number", "NONE", choice_to_label, label_to_choice
    if panel == "operation_local_choice_conflict":
        operation = operation_for_label("local_number", assignment)
        lines = local_table_lines(rows, source, include_query=True)
        append_rules_and_examples(lines, example_rows, assignment=assignment, rule_order=rule_order)
        lines.append(operation_query_line(operation, source, "compact_operation_rules"))
        lines.append("Apply the query operation, then choose the matching answer.")
        lines.extend(option_lines)
        return "\n".join([*lines, *response_suffix()]), "local_number", operation, choice_to_label, label_to_choice
    if panel == "operation_atomic_choice_conflict":
        operation = operation_for_label("atomic_number", assignment)
        lines = local_table_lines(rows, source, include_query=True)
        append_rules_and_examples(lines, example_rows, assignment=assignment, rule_order=rule_order)
        lines.append(operation_query_line(operation, source, "compact_operation_rules"))
        lines.append("Apply the query operation, then choose the matching answer.")
        lines.extend(option_lines)
        return "\n".join([*lines, *response_suffix()]), "atomic_number", operation, choice_to_label, label_to_choice
    if panel == "operation_rule_absent_choice_null":
        operation = operation_for_label("local_number", assignment)
        lines = local_table_lines(rows, source, include_query=True)
        lines.append("No operation rule or worked example is available for this query.")
        lines.append(operation_query_line(operation, source, "compact_operation_rules"))
        lines.append("If no operation rule is available, choose UNKNOWN.")
        lines.extend(option_lines)
        return "\n".join([*lines, *response_suffix()]), "unknown", operation, choice_to_label, label_to_choice
    if panel == "answer_absent_choice_null":
        operation = operation_for_label("local_number", assignment)
        lines = local_table_lines(rows, source, include_query=False)
        append_rules_and_examples(lines, example_rows, assignment=assignment, rule_order=rule_order)
        lines.append(operation_query_line(operation, source, "compact_operation_rules"))
        lines.append("If the queried element is absent from the table, choose UNKNOWN before applying any operation.")
        lines.extend(option_lines)
        return "\n".join([*lines, *response_suffix()]), "unknown", operation, choice_to_label, label_to_choice
    raise ValueError(f"unknown panel {panel}")


def candidate_answers_for(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "answer": choice,
            "label": choice,
            "candidate_type": record["choice_to_label"][choice],
        }
        for choice in CHOICES
    ]


def source_records(
    limit_sources: int | None = None,
    templates: tuple[str, ...] = TEMPLATES,
    assignments: tuple[str, ...] = OPERATION_ASSIGNMENTS,
    rule_orders: tuple[str, ...] = RULE_ORDERS,
    option_orders: tuple[str, ...] = OPTION_ORDERS,
    run_type: str = RUN_TYPE,
) -> list[dict[str, Any]]:
    sources = base_sources(limit_sources)
    records: list[dict[str, Any]] = []
    for source in sources:
        for template in templates:
            for assignment in assignments:
                for rule_order in rule_orders:
                    for option_order in option_orders:
                        for panel in PANELS:
                            prompt, expected_label, query_operation, choice_to_label, label_to_choice = make_prompt(
                                sources,
                                source,
                                panel=panel,
                                template=template,
                                assignment=assignment,
                                rule_order=rule_order,
                                option_order=option_order,
                            )
                            expected_choice = label_to_choice[expected_label]
                            record = {
                                "id": f"{CARD_ID}_{template}_{assignment}_{rule_order}_{option_order}_{panel}_{source['source_id']}",
                                "card_id": CARD_ID,
                                "run_type": run_type,
                                "model_id": MODEL_ID,
                                "template": template,
                                "operation_assignment": assignment,
                                "rule_order": rule_order,
                                "option_order": option_order,
                                "panel": panel,
                                "split": source["split"],
                                "source_id": source["source_id"],
                                "source_index": source["source_index"],
                                "element": source["element"],
                                "synthetic_key": source["synthetic_key"],
                                "query_operation": query_operation,
                                "atomic_number": str(source["atomic_number"]),
                                "local_number": str(source["local_number"]),
                                "expected_label": expected_label,
                                "expected_choice": expected_choice,
                                "choice_to_label": choice_to_label,
                                "label_to_choice": label_to_choice,
                                "prompt": prompt,
                            }
                            record["candidate_answers"] = candidate_answers_for(record)
                            records.append(record)
    return records


def prompt_audit(record: dict[str, Any]) -> dict[str, Any]:
    prompt = record["prompt"]
    primary = record["panel"] in PRIMARY_CONFLICT_PANELS
    null = record["panel"] == "answer_absent_choice_null"
    return {
        "atomic_number_occurrences": word_occurrences(prompt, record["atomic_number"]),
        "atomic_number_visible_in_conflict_options": primary and word_occurrences(prompt, record["atomic_number"]) >= 1,
        "answer_absent_omits_query_local_row": not null or not line_contains_both(prompt, record["element"], record["local_number"]),
        "prompt_has_status_lexeme": bool(STATUS_LEXEME_RE.search(prompt)),
    }


def structural_check(
    records: list[dict[str, Any]],
    templates: tuple[str, ...],
    assignments: tuple[str, ...],
    rule_orders: tuple[str, ...],
    option_orders: tuple[str, ...],
) -> dict[str, Any]:
    source_ids = {row["source_id"] for row in records}
    expected_count = len(source_ids) * len(templates) * len(assignments) * len(rule_orders) * len(option_orders) * len(PANELS)
    split_by_source: dict[str, set[str]] = {}
    for row in records:
        split_by_source.setdefault(row["source_id"], set()).add(row["split"])
    split_rows = Counter(row["split"] for row in records)
    conflict_rows = [row for row in records if row["panel"] in PRIMARY_CONFLICT_PANELS]
    expected_conflict = Counter(row["expected_label"] for row in conflict_rows)
    operation_conflict = Counter(row["query_operation"] for row in conflict_rows)
    choice_counts_by_expected = {
        expected: Counter(row["expected_choice"] for row in records if row["expected_label"] == expected)
        for expected in ("local_number", "atomic_number", "unknown")
    }
    audits = [prompt_audit(row) for row in records]
    malformed_candidates = [
        f"{row['id']}::{candidate['answer']}"
        for row in records
        for candidate in row["candidate_answers"]
        if candidate["answer"] not in CHOICES
    ]
    option_collisions = [
        row["id"]
        for row in records
        if len(set(row["choice_to_label"].values())) != len(CHOICES)
    ]
    suffixes = {"\n".join(row["prompt"].splitlines()[-3:]) for row in records}
    status_rows = [row["id"] for row, audit in zip(records, audits, strict=True) if audit["prompt_has_status_lexeme"]]
    null_leaks = [
        row["id"]
        for row, audit in zip(records, audits, strict=True)
        if not audit["answer_absent_omits_query_local_row"]
    ]
    conflict_rows_without_atomic_option = [
        row["id"]
        for row, audit in zip(records, audits, strict=True)
        if row["panel"] in PRIMARY_CONFLICT_PANELS and not audit["atomic_number_visible_in_conflict_options"]
    ]
    expected_choice_balanced = all(
        set(counts) == set(CHOICES) and max(counts.values()) - min(counts.values()) <= len(templates) * len(assignments) * len(rule_orders) * len(source_ids)
        for counts in choice_counts_by_expected.values()
    )
    criteria = {
        "expected_row_count": len(records) == expected_count,
        "all_panels_present": set(PANELS) == {row["panel"] for row in records},
        "all_templates_present": set(templates) == {row["template"] for row in records},
        "all_assignments_present": set(assignments) == {row["operation_assignment"] for row in records},
        "all_rule_orders_present": set(rule_orders) == {row["rule_order"] for row in records},
        "all_option_orders_present": set(option_orders) == {row["option_order"] for row in records},
        "source_split_disjoint": all(len(splits) == 1 for splits in split_by_source.values()),
        "holdout_sources_present": any(row["split"] == "holdout" for row in records),
        "calibration_sources_present": any(row["split"] == "calibration" for row in records),
        "atomic_number_visible_in_conflict_options_by_design": not conflict_rows_without_atomic_option,
        "answer_absent_omits_query_local_row": not null_leaks,
        "candidate_choices_parseable": not malformed_candidates,
        "no_option_label_collisions": not option_collisions,
        "single_answer_suffix": len(suffixes) == 1,
        "prompts_have_no_status_lexemes": not status_rows,
        "conflict_expected_balanced": expected_conflict["local_number"] == expected_conflict["atomic_number"],
        "conflict_operations_balanced": operation_conflict["ALPHA"] == operation_conflict["BETA"],
        "expected_choices_balanced": expected_choice_balanced,
    }
    return {
        "passed": all(criteria.values()),
        "record_count": len(records),
        "source_count": len(source_ids),
        "expected_record_count": expected_count,
        "split_row_counts": dict(sorted(split_rows.items())),
        "criteria": criteria,
        "malformed_candidate_rows": malformed_candidates[:20],
        "option_collision_rows": option_collisions[:20],
        "status_lexeme_rows": status_rows[:20],
        "null_leak_rows": null_leaks[:20],
        "conflict_rows_without_atomic_option": conflict_rows_without_atomic_option[:20],
        "conflict_expected_counts": dict(sorted(expected_conflict.items())),
        "conflict_operation_counts": dict(sorted(operation_conflict.items())),
        "expected_choice_counts_by_label": {
            key: dict(sorted(value.items())) for key, value in choice_counts_by_expected.items()
        },
    }


def strict_choice_parse(record: dict[str, Any], generated_text: str) -> dict[str, Any]:
    stripped = generated_text.strip().lstrip("`'\" ")
    first_line = stripped.splitlines()[0].strip().strip("\"'`") if stripped else ""
    match = CHOICE_RE.fullmatch(first_line)
    if not match:
        return {
            "selected_choice": None,
            "selected_label": "unparsed",
            "selected_answer": None,
            "parseable": False,
            "parse_rule": "not_bare_choice",
            "first_line": first_line,
        }
    choice = match.group(1).upper()
    label = record["choice_to_label"][choice]
    return {
        "selected_choice": choice,
        "selected_label": label,
        "selected_answer": option_value(label, record),
        "parseable": True,
        "parse_rule": "strict_bare_choice",
        "first_line": first_line,
    }


def final_choice_logits(model: Any, tokenizer: Any, prompt: str) -> dict[str, Any]:
    import torch
    from mc006_parametric_fact_override_v15_parser_normalized_signature import first_answer_token_id

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].detach().float().cpu()
    token_ids = {choice: first_answer_token_id(tokenizer, choice) for choice in CHOICES}
    scores = {choice: float(logits[token_id].item()) for choice, token_id in token_ids.items()}
    return {
        "choice_first_token_ids": token_ids,
        "final_choice_logits": scores,
    }


def choice_logprob_payload(model: Any, tokenizer: Any, prompt: str) -> dict[str, Any]:
    from mc006_parametric_fact_override_v15_parser_normalized_signature import token_logprob

    scores = {choice: token_logprob(model, tokenizer, prompt, choice) for choice in CHOICES}
    return {"choice_candidate_logprobs": scores}


def margin_payload(record: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    expected_choice = record["expected_choice"]
    wrong_choices = [choice for choice in CHOICES if choice != expected_choice]
    final_logits = output.get("final_choice_logits", {})
    candidate_scores = output.get("choice_candidate_logprobs", {})
    payload: dict[str, Any] = {}
    if final_logits:
        payload["final_expected_choice_minus_best_wrong_logit"] = float(
            final_logits[expected_choice] - max(final_logits[choice] for choice in wrong_choices)
        )
    if candidate_scores:
        expected_mean = float(candidate_scores[expected_choice]["mean_logprob"])
        wrong_mean = max(float(candidate_scores[choice]["mean_logprob"]) for choice in wrong_choices)
        payload["candidate_expected_choice_minus_best_wrong_mean_logprob"] = (
            expected_mean - wrong_mean if math.isfinite(expected_mean) and math.isfinite(wrong_mean) else float("-inf")
        )
    return payload


def score_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    score_candidates: bool,
    verbose: bool,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        generated = generate_answer(model, tokenizer, record["prompt"], max_new_tokens)
        parsed = strict_choice_parse(record, generated["generated_text"])
        output = {
            **record,
            "index": index,
            **generated,
            **parsed,
            **prompt_audit(record),
            "expected_correct": parsed["selected_label"] == record["expected_label"],
            "choice_correct": parsed["selected_choice"] == record["expected_choice"],
        }
        if score_candidates:
            output.update(final_choice_logits(model, tokenizer, record["prompt"]))
            output.update(choice_logprob_payload(model, tokenizer, record["prompt"]))
            output.update(margin_payload(record, output))
        outputs.append(output)
        if verbose:
            print(
                f"[{index:04d}/{len(records):04d}] {record['id']} split={record['split']} "
                f"panel={record['panel']} expected={record['expected_label']}/{record['expected_choice']} "
                f"-> {output['selected_label']}/{output['selected_choice']}"
            )
    return outputs


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("selected_label")) for row in rows).items()))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parseable = sum(1 for row in rows if row.get("parseable"))
    counts = Counter(row.get("selected_label") for row in rows)
    choice_counts = Counter(row.get("selected_choice") for row in rows)
    expected_correct = sum(1 for row in rows if row.get("expected_correct"))
    choice_correct = sum(1 for row in rows if row.get("choice_correct"))
    return {
        "rows": len(rows),
        "label_counts": label_counts(rows),
        "choice_counts": dict(sorted((str(key), value) for key, value in choice_counts.items())),
        "parseable": parseable,
        "parseable_rate": rate(parseable, len(rows)),
        "local_number": counts["local_number"],
        "local_number_rate": rate(counts["local_number"], len(rows)),
        "atomic_number": counts["atomic_number"],
        "atomic_number_rate": rate(counts["atomic_number"], len(rows)),
        "unknown": counts["unknown"],
        "unknown_rate": rate(counts["unknown"], len(rows)),
        "unparsed": counts["unparsed"],
        "unparsed_rate": rate(counts["unparsed"], len(rows)),
        "expected_correct": expected_correct,
        "expected_correct_rate": rate(expected_correct, len(rows)),
        "choice_correct": choice_correct,
        "choice_correct_rate": rate(choice_correct, len(rows)),
        "expected_label_counts": dict(sorted(Counter(str(row.get("expected_label")) for row in rows).items())),
        "expected_choice_counts": dict(sorted(Counter(str(row.get("expected_choice")) for row in rows).items())),
        "query_operation_counts": dict(sorted(Counter(str(row.get("query_operation")) for row in rows).items())),
    }


def margin_audits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows or "candidate_expected_choice_minus_best_wrong_mean_logprob" not in rows[0]:
        return {"reported": False}
    expected_margins = [
        row["candidate_expected_choice_minus_best_wrong_mean_logprob"]
        for row in rows
        if isinstance(row.get("candidate_expected_choice_minus_best_wrong_mean_logprob"), (int, float))
    ]
    return {
        "reported": True,
        "candidate_expected_margin_min": min(expected_margins) if expected_margins else None,
        "candidate_expected_margin_max": max(expected_margins) if expected_margins else None,
    }


def grouped_summaries(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return {value: summarize_rows([row for row in rows if row[key] == value]) for value in sorted({row[key] for row in rows})}


def template_summary(outputs: list[dict[str, Any]], templates: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for template in templates:
        rows = [row for row in outputs if row["template"] == template]
        panels = {panel: summarize_rows([row for row in rows if row["panel"] == panel]) for panel in PANELS}
        conflict_rows = [row for row in rows if row["panel"] in PRIMARY_CONFLICT_PANELS]
        result[template] = {
            "rows": len(rows),
            "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "selected_label_counts": label_counts(rows),
            "panels": panels,
            "primary_conflict": summarize_rows(conflict_rows),
            "primary_conflict_by_operation": grouped_summaries(conflict_rows, "query_operation"),
            "primary_conflict_by_assignment": grouped_summaries(conflict_rows, "operation_assignment"),
            "primary_conflict_by_rule_order": grouped_summaries(conflict_rows, "rule_order"),
            "primary_conflict_by_option_order": grouped_summaries(conflict_rows, "option_order"),
            "margin_audits": margin_audits(rows),
        }
    return result


def selection_key(item: dict[str, Any], template: str, templates: tuple[str, ...]) -> tuple[float, float, float, float, int]:
    panels = item["panels"]
    control_floor = min(
        float(panels["synthetic_choice_lookup"]["local_number_rate"]),
        float(panels["familiar_choice_lookup"]["local_number_rate"]),
        float(panels["atomic_choice_control"]["atomic_number_rate"]),
        float(panels["answer_absent_choice_null"]["unknown_rate"]),
        float(panels["operation_rule_absent_choice_null"]["unknown_rate"]),
    )
    conflict_floor = min(
        float(panels["operation_local_choice_conflict"]["local_number_rate"]),
        float(panels["operation_atomic_choice_conflict"]["atomic_number_rate"]),
    )
    parse_floor = min(float(panel["parseable_rate"]) for panel in panels.values())
    return (
        control_floor,
        conflict_floor,
        float(item["primary_conflict"]["expected_correct_rate"]),
        parse_floor,
        -templates.index(template),
    )


def select_template(by_template: dict[str, Any], templates: tuple[str, ...]) -> dict[str, Any]:
    selected = max(templates, key=lambda template: selection_key(by_template[template], template, templates))
    return {
        "selected_template": selected,
        "selection_key": list(selection_key(by_template[selected], selected, templates)),
        "all_selection_keys": {template: list(selection_key(by_template[template], template, templates)) for template in templates},
        "rule": [
            "max direct-control/null floor",
            "max local/atomic operation floor",
            "max aggregate expected-correct conflict rate",
            "max all-panel parseability floor",
            "earliest template",
        ],
    }


def classify(criteria: dict[str, bool], selected: dict[str, Any]) -> tuple[str, str]:
    if criteria["smoke_mode"]:
        if criteria["choice_operation_behavior_gate_passed"]:
            return "smoke_only", "choice_interface_behavior_passed"
        return "smoke_only", "choice_interface_smoke_failed"
    if criteria["choice_operation_behavior_gate_passed"]:
        return "choice_interface_prompt_visible_passed", "choice_interface_behavior_passed_prompt_visible"
    if not criteria["operation_atomic_choice_conflict_at_least_85p"]:
        return "choice_operation_atomic_branch_failed", "operation_atomic_choice_branch_failed"
    if not criteria["operation_local_choice_conflict_at_least_85p"]:
        return "choice_operation_local_branch_failed", "operation_local_choice_branch_failed"
    if not criteria["all_controls_passed"]:
        return "choice_operation_control_failed", "choice_operation_control_failed"
    return "choice_operation_mixed_failure", "choice_operation_mixed_failure"


def summarize(
    records: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    templates: tuple[str, ...],
    assignments: tuple[str, ...],
    rule_orders: tuple[str, ...],
    option_orders: tuple[str, ...],
    *,
    full_run: bool,
    score_candidates: bool,
) -> dict[str, Any]:
    structural = structural_check(records, templates, assignments, rule_orders, option_orders)
    by_template = template_summary(outputs, templates)
    selection = select_template(by_template, templates)
    selected = by_template[selection["selected_template"]]
    panels = selected["panels"]
    criteria = {
        "smoke_mode": not full_run,
        "structural_passed": structural["passed"],
        "full_source_count_is_40": structural["source_count"] == 40,
        "source_disjoint_holdout": structural["criteria"]["source_split_disjoint"]
        and structural["criteria"]["holdout_sources_present"],
        "selected_prompt_audit_passed": all(
            row["answer_absent_omits_query_local_row"] and not row["prompt_has_status_lexeme"]
            for row in outputs
            if row["template"] == selection["selected_template"]
        ),
        "prompts_have_no_status_lexemes": structural["criteria"]["prompts_have_no_status_lexemes"],
        "atomic_number_visible_in_conflict_options": structural["criteria"]["atomic_number_visible_in_conflict_options_by_design"],
        "synthetic_control_at_least_90p": panels["synthetic_choice_lookup"]["local_number_rate"] >= 0.90,
        "familiar_control_at_least_90p": panels["familiar_choice_lookup"]["local_number_rate"] >= 0.90,
        "atomic_control_at_least_85p": panels["atomic_choice_control"]["atomic_number_rate"] >= 0.85,
        "answer_absent_unknown_at_least_90p": panels["answer_absent_choice_null"]["unknown_rate"] >= 0.90,
        "operation_rule_absent_unknown_at_least_90p": panels["operation_rule_absent_choice_null"]["unknown_rate"] >= 0.90,
        "operation_local_choice_conflict_at_least_85p": panels["operation_local_choice_conflict"]["local_number_rate"] >= 0.85,
        "operation_atomic_choice_conflict_at_least_85p": panels["operation_atomic_choice_conflict"]["atomic_number_rate"] >= 0.85,
        "all_panels_parseable_at_least_95p": min(panel["parseable_rate"] for panel in panels.values()) >= 0.95,
        "candidate_and_output_margins_reported": bool(score_candidates) and selected["margin_audits"]["reported"],
    }
    control_keys = [
        "synthetic_control_at_least_90p",
        "familiar_control_at_least_90p",
        "atomic_control_at_least_85p",
        "answer_absent_unknown_at_least_90p",
        "operation_rule_absent_unknown_at_least_90p",
    ]
    criteria["all_controls_passed"] = all(criteria[key] for key in control_keys)
    behavior_keys = [
        "structural_passed",
        "source_disjoint_holdout",
        "selected_prompt_audit_passed",
        "prompts_have_no_status_lexemes",
        "atomic_number_visible_in_conflict_options",
        "all_controls_passed",
        "operation_local_choice_conflict_at_least_85p",
        "operation_atomic_choice_conflict_at_least_85p",
        "all_panels_parseable_at_least_95p",
        "candidate_and_output_margins_reported",
    ]
    if full_run:
        behavior_keys.append("full_source_count_is_40")
    behavior_gate_passed = all(criteria[key] for key in behavior_keys)
    criteria["choice_operation_behavior_gate_passed"] = behavior_gate_passed
    diagnostic_class, observed = classify(criteria, selected)
    return {
        "structural": structural,
        "by_template": by_template,
        "selection": selection,
        "selected_template_summary": selected,
        "selected_template_rows": [
            row["id"] for row in outputs if row["template"] == selection["selected_template"]
        ],
        "criteria": criteria,
        "observed_failure_pattern": observed,
        "passed": behavior_gate_passed and full_run,
        "behavior_ready": False,
        "behavior_gate_passed": behavior_gate_passed,
        "signature_ready": False,
        "intervention_ready": False,
        "diagnostic_class": diagnostic_class,
    }


def quiet_summary(summary: dict[str, Any], output_path: Path) -> dict[str, Any]:
    selected = summary["selected_template_summary"]
    panels = selected["panels"]
    return {
        "diagnostic_class": summary["diagnostic_class"],
        "observed_failure_pattern": summary["observed_failure_pattern"],
        "passed": summary["passed"],
        "behavior_ready": summary["behavior_ready"],
        "signature_ready": summary["signature_ready"],
        "criteria": summary["criteria"],
        "selection": summary["selection"],
        "selected_controls": {
            panel: panels[panel]
            for panel in (
                "synthetic_choice_lookup",
                "familiar_choice_lookup",
                "atomic_choice_control",
                "answer_absent_choice_null",
                "operation_rule_absent_choice_null",
            )
        },
        "selected_conflicts": {
            panel: panels[panel]
            for panel in PRIMARY_CONFLICT_PANELS
        },
        "output_path": str(output_path),
    }


def write_behavior_status_card(path: Path, result: dict[str, Any], output_path: Path) -> None:
    summary = result["summary"]
    selected = summary["selected_template_summary"]
    criteria = summary["criteria"]
    lines = [
        "# MC025 Choice-Interface Operation Arbitration Behavior Status",
        "",
        f"Status: {summary['diagnostic_class']}.",
        "",
        f"Observed pattern: `{summary['observed_failure_pattern']}`.",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "## Artifact",
        "",
        "- runner:",
        "  `code/mc025_choice_interface_operation_arbitration.py`",
        "- result:",
        f"  `{output_path.as_posix()}`",
        "",
        "## Verdict",
        "",
    ]
    if summary["behavior_gate_passed"]:
        lines.extend(
            [
                "The constrained-choice behavior gate passed.",
                "This is a prompt-visible answer-interface diagnostic because",
                "conflict prompts include the atomic number as an answer option.",
                "It does not justify hidden-state work on learned memory.",
            ]
        )
    elif criteria["smoke_mode"]:
        lines.extend(
            [
                "This is a smoke or partial run. It diagnoses whether constraining",
                "the answer interface repairs operation routing.",
            ]
        )
    else:
        lines.extend(
            [
                "The constrained-choice behavior gate failed.",
                "Hidden-state work remains forbidden for this route.",
            ]
        )
    lines.extend(
        [
            "",
            "## Selected Template",
            "",
            f"- selected template: `{summary['selection']['selected_template']}`",
            f"- selection key: `{json.dumps(summary['selection']['selection_key'])}`",
            "",
            "## Gate Criteria",
            "",
            "| Criterion | Passed |",
            "| --- | --- |",
        ]
    )
    for key, value in criteria.items():
        lines.append(f"| `{key}` | `{str(value).lower()}` |")
    lines.extend(
        [
            "",
            "## Selected Panels",
            "",
            "| Panel | Rows | Parseable | Local Rate | Atomic Rate | Unknown Rate | Expected Correct |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for panel in PANELS:
        item = selected["panels"][panel]
        lines.append(
            f"| `{panel}` | {item['rows']} | {item['parseable_rate']:.3f} | "
            f"{item['local_number_rate']:.3f} | {item['atomic_number_rate']:.3f} | "
            f"{item['unknown_rate']:.3f} | {item['expected_correct_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate Conflict",
            "",
            f"- primary conflict expected-correct rate: {selected['primary_conflict']['expected_correct_rate']:.3f}",
            f"- operation-local choice local rate: {selected['panels']['operation_local_choice_conflict']['local_number_rate']:.3f}",
            f"- operation-atomic choice atomic rate: {selected['panels']['operation_atomic_choice_conflict']['atomic_number_rate']:.3f}",
            "",
            "## Claim Boundary",
            "",
            "MC025 is a prompt-visible answer-interface diagnostic. The atomic",
            "number is present in the choices on conflict rows, so this can only",
            "test whether constrained choices repair routing/output behavior. It",
            "does not establish a learned-memory hidden signature, intervention,",
            "or mechanism card.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--score-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--score-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-manifest", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--artifact-prefix", default="mc025_choice_interface_operation_arbitration_structural")
    parser.add_argument("--status-card", type=Path, default=STATUS_CARD)
    parser.add_argument("--write-status-card", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.time()
    run_type = BEHAVIOR_RUN_TYPE if args.score_model else RUN_TYPE
    records = source_records(args.limit_sources, TEMPLATES, OPERATION_ASSIGNMENTS, RULE_ORDERS, OPTION_ORDERS, run_type)
    structural = structural_check(records, TEMPLATES, OPERATION_ASSIGNMENTS, RULE_ORDERS, OPTION_ORDERS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.write_manifest and not args.score_model:
        result = {
            "schema_version": 1,
            "card_id": CARD_ID,
            "run_type": RUN_TYPE,
            "model_id": args.model_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "purpose": "Structural gate for constrained-choice operation arbitration.",
            "templates": list(TEMPLATES),
            "operation_assignments": list(OPERATION_ASSIGNMENTS),
            "rule_orders": list(RULE_ORDERS),
            "option_orders": list(OPTION_ORDERS),
            "operation_codes": list(OPERATION_CODES),
            "panels": list(PANELS),
            "primary_conflict_panels": list(PRIMARY_CONFLICT_PANELS),
            "prompt_visibility_boundary": "atomic number is visible in answer options on conflict rows by design",
            "structural": structural,
            "records": records,
        }
        output_path = args.output_dir / f"{args.artifact_prefix}_{time.strftime('%Y%m%dT%H%M%S')}.json"
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "passed": structural["passed"],
                    "record_count": structural["record_count"],
                    "source_count": structural["source_count"],
                    "criteria": structural["criteria"],
                    "output_path": str(output_path),
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return 0 if structural["passed"] else 1

    if not args.score_model:
        print(json.dumps(structural, indent=2, ensure_ascii=True))
        return 0 if structural["passed"] else 1

    if not structural["passed"]:
        print(json.dumps({"passed": False, "diagnostic_class": "structural_invalid", "structural": structural}, indent=2))
        return 1

    model, tokenizer = load_model_and_tokenizer(args.model_id, args.local_files_only)
    outputs = score_records(records, tokenizer, model, args.max_new_tokens, args.score_candidates, verbose=not args.quiet)
    full_run = args.limit_sources is None
    summary = summarize(
        records,
        outputs,
        TEMPLATES,
        OPERATION_ASSIGNMENTS,
        RULE_ORDERS,
        OPTION_ORDERS,
        full_run=full_run,
        score_candidates=args.score_candidates,
    )
    prefix = args.artifact_prefix
    if prefix == "mc025_choice_interface_operation_arbitration_structural":
        prefix = "mc025_choice_interface_operation_arbitration_behavior"
    output_path = args.output_dir / f"{prefix}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    result = {
        "schema_version": 1,
        "card_id": CARD_ID,
        "run_type": BEHAVIOR_RUN_TYPE,
        "model_id": args.model_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "max_new_tokens": args.max_new_tokens,
        "decoding": {"do_sample": False},
        "score_candidates": args.score_candidates,
        "limit_sources": args.limit_sources,
        "templates": list(TEMPLATES),
        "operation_assignments": list(OPERATION_ASSIGNMENTS),
        "rule_orders": list(RULE_ORDERS),
        "option_orders": list(OPTION_ORDERS),
        "operation_codes": list(OPERATION_CODES),
        "elapsed_s": time.time() - started,
        "purpose": "Prompt-visible constrained-choice operation arbitration diagnostic before hidden-state work.",
        "prompt_visibility_boundary": "atomic number is visible in answer options on conflict rows by design",
        "panels": list(PANELS),
        "primary_conflict_panels": list(PRIMARY_CONFLICT_PANELS),
        "sources": base_sources(args.limit_sources),
        "structural": structural,
        "records": outputs,
        "summary": summary,
    }
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    if args.write_status_card:
        write_behavior_status_card(args.status_card, result, output_path)
    print(json.dumps(quiet_summary(summary, output_path), indent=2, ensure_ascii=True))
    return 0 if (summary["behavior_gate_passed"] or not full_run) else 2


if __name__ == "__main__":
    raise SystemExit(main())
