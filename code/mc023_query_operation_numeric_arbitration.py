#!/usr/bin/env python
"""MC023 query-operation numeric arbitration.

MC022 showed that semantic row-level answer-source labels are still not a clean
bridge into learned memory. MC023 removes row-level source labels entirely. It
tests whether a query-level operation handle can choose between prompt-local
lab numbers and learned atomic numbers while preserving direct controls, nulls,
source-disjoint splits, and counterbalanced operation assignments.

This is a behavior diagnostic only. A pass would create a behavior substrate
for later hidden-state work; it would not itself be a mechanism claim.
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
    distractors,
    generate_answer,
    line_contains_both,
    word_occurrences,
)
from mc016_alphabet_gated_numeric_arbitration import load_model_and_tokenizer


CARD_ID = "MC023"
RUN_TYPE = "query_operation_numeric_arbitration_structural_gate"
BEHAVIOR_RUN_TYPE = "query_operation_numeric_arbitration_behavior"
RESULT_DIR = Path("results/cards/MC023")
STATUS_CARD = Path("research/cards/MC023_QUERY_OPERATION_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md")

PANELS = (
    "synthetic_numeric_lookup",
    "familiar_entity_numeric_lookup",
    "real_world_atomic_number_control",
    "operation_local_conflict",
    "operation_atomic_conflict",
    "operation_rule_absent_conflict",
    "answer_absent_null",
)
PRIMARY_CONFLICT_PANELS = ("operation_local_conflict", "operation_atomic_conflict")
TEMPLATES = ("query_operation_rules", "compact_operation_rules", "query_first_operation")
OPERATION_ASSIGNMENTS = ("alpha_local_beta_atomic", "alpha_atomic_beta_local")
RULE_ORDERS = ("alpha_first", "beta_first")
OPERATION_CODES = ("ALPHA", "BETA")
STATUS_LEXEME_RE = re.compile(r"\b(trusted|untrusted|reliable|unreliable|status)\b", re.IGNORECASE)


def response_suffix() -> list[str]:
    return [
        "Return only one final token: an integer or UNKNOWN.",
        "Do not explain.",
        "Answer:",
    ]


def operation_mapping(assignment: str) -> dict[str, str]:
    if assignment == "alpha_local_beta_atomic":
        return {"ALPHA": "local_number", "BETA": "atomic_number"}
    if assignment == "alpha_atomic_beta_local":
        return {"ALPHA": "atomic_number", "BETA": "local_number"}
    raise ValueError(f"unknown assignment {assignment}")


def operation_for_label(label: str, assignment: str) -> str:
    mapping = operation_mapping(assignment)
    matches = [operation for operation, mapped_label in mapping.items() if mapped_label == label]
    if len(matches) != 1:
        raise ValueError(f"label {label} not uniquely mapped under {assignment}")
    return matches[0]


def operation_order(rule_order: str) -> tuple[str, str]:
    if rule_order == "alpha_first":
        return ("ALPHA", "BETA")
    if rule_order == "beta_first":
        return ("BETA", "ALPHA")
    raise ValueError(f"unknown rule order {rule_order}")


def local_table_lines(
    rows: list[dict[str, Any]],
    query: dict[str, Any],
    *,
    include_query: bool,
    template: str,
    key_field: str = "element",
) -> list[str]:
    visible_rows = rows if include_query else [row for row in rows if row["source_id"] != query["source_id"]]
    if template == "query_operation_rules":
        lines = [
            "A local lab table may define temporary lab numbers.",
            "Standard chemistry may define atomic numbers.",
            "Element lab-number table:",
        ]
    elif template == "compact_operation_rules":
        lines = [
            "Two numeric systems may conflict.",
            "Rows:",
        ]
    elif template == "query_first_operation":
        lines = [
            "Use the query operation after reading the lab table.",
            "Rows:",
        ]
    else:
        raise ValueError(f"unknown template {template}")
    for row in visible_rows:
        lines.append(f"{row[key_field]} has local lab number {row['local_number']}.")
    return lines


def append_operation_rules(
    lines: list[str],
    *,
    assignment: str,
    rule_order: str,
    template: str,
) -> None:
    mapping = operation_mapping(assignment)
    if template == "query_first_operation":
        lines.append("The query includes one operation code.")
    else:
        lines.append("Operation rules:")
    for operation in operation_order(rule_order):
        target = mapping[operation]
        if target == "local_number":
            lines.append(f"Operation {operation}: return the queried row's local lab number.")
        elif target == "atomic_number":
            lines.append(f"Operation {operation}: return the queried element's standard atomic number.")
        else:
            raise ValueError(f"unknown operation target {target}")


def operation_query_line(operation: str, source: dict[str, Any], template: str) -> str:
    if template == "query_first_operation":
        return f"Query operation: {operation}; query element: {source['element']}"
    return f"Query element: {source['element']}; operation: {operation}"


def make_prompt(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    panel: str,
    template: str,
    assignment: str,
    rule_order: str,
) -> tuple[str, str, str]:
    rows = distractors(sources, source, 4)
    if panel == "synthetic_numeric_lookup":
        lines = local_table_lines(
            rows,
            source,
            include_query=True,
            template=template,
            key_field="synthetic_key",
        )
        lines.append(f"Query key: {source['synthetic_key']}")
        lines.append("Return the local lab number.")
        return "\n".join([*lines, *response_suffix()]), "local_number", "NONE"
    if panel == "familiar_entity_numeric_lookup":
        lines = local_table_lines(rows, source, include_query=True, template=template)
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the local lab number.")
        return "\n".join([*lines, *response_suffix()]), "local_number", "NONE"
    if panel == "real_world_atomic_number_control":
        lines = [
            "No local lab table is active.",
            "Return the standard atomic number for the queried element.",
            f"Query element: {source['element']}",
        ]
        return "\n".join([*lines, *response_suffix()]), "atomic_number", "NONE"
    if panel == "operation_local_conflict":
        operation = operation_for_label("local_number", assignment)
        lines = local_table_lines(rows, source, include_query=True, template=template)
        append_operation_rules(lines, assignment=assignment, rule_order=rule_order, template=template)
        lines.append(operation_query_line(operation, source, template))
        lines.append("Apply the query operation and return the controlling number.")
        return "\n".join([*lines, *response_suffix()]), "local_number", operation
    if panel == "operation_atomic_conflict":
        operation = operation_for_label("atomic_number", assignment)
        lines = local_table_lines(rows, source, include_query=True, template=template)
        append_operation_rules(lines, assignment=assignment, rule_order=rule_order, template=template)
        lines.append(operation_query_line(operation, source, template))
        lines.append("Apply the query operation and return the controlling number.")
        return "\n".join([*lines, *response_suffix()]), "atomic_number", operation
    if panel == "operation_rule_absent_conflict":
        operation = operation_for_label("local_number", assignment)
        lines = local_table_lines(rows, source, include_query=True, template=template)
        lines.append("No operation rule is available for this query.")
        lines.append(operation_query_line(operation, source, template))
        lines.append("If no operation rule is available, return UNKNOWN.")
        return "\n".join([*lines, *response_suffix()]), "unknown", operation
    if panel == "answer_absent_null":
        operation = operation_for_label("local_number", assignment)
        lines = local_table_lines(rows, source, include_query=False, template=template)
        append_operation_rules(lines, assignment=assignment, rule_order=rule_order, template=template)
        lines.append(operation_query_line(operation, source, template))
        lines.append("If the queried element is absent from the table, return UNKNOWN before applying any operation.")
        return "\n".join([*lines, *response_suffix()]), "unknown", operation
    raise ValueError(f"unknown panel {panel}")


def candidate_answers_for(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"answer": record["local_number"], "label": "local_number", "candidate_type": "prompt_local_lab_number"},
        {"answer": record["atomic_number"], "label": "atomic_number", "candidate_type": "real_world_atomic_number"},
        {"answer": "UNKNOWN", "label": "unknown", "candidate_type": "unknown"},
    ]


def source_records(
    limit_sources: int | None = None,
    templates: tuple[str, ...] = TEMPLATES,
    assignments: tuple[str, ...] = OPERATION_ASSIGNMENTS,
    rule_orders: tuple[str, ...] = RULE_ORDERS,
    run_type: str = RUN_TYPE,
) -> list[dict[str, Any]]:
    sources = base_sources(limit_sources)
    records: list[dict[str, Any]] = []
    for source in sources:
        for template in templates:
            for assignment in assignments:
                for rule_order in rule_orders:
                    for panel in PANELS:
                        prompt, expected_label, query_operation = make_prompt(
                            sources,
                            source,
                            panel=panel,
                            template=template,
                            assignment=assignment,
                            rule_order=rule_order,
                        )
                        record = {
                            "id": f"{CARD_ID}_{template}_{assignment}_{rule_order}_{panel}_{source['source_id']}",
                            "card_id": CARD_ID,
                            "run_type": run_type,
                            "model_id": MODEL_ID,
                            "template": template,
                            "operation_assignment": assignment,
                            "rule_order": rule_order,
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
                            "prompt": prompt,
                        }
                        record["candidate_answers"] = candidate_answers_for(record)
                        records.append(record)
    return records


def prompt_audit(record: dict[str, Any]) -> dict[str, Any]:
    prompt = record["prompt"]
    primary = record["panel"] in PRIMARY_CONFLICT_PANELS
    null = record["panel"] == "answer_absent_null"
    return {
        "atomic_number_occurrences": word_occurrences(prompt, record["atomic_number"]),
        "atomic_number_hidden_in_conflict": not primary or word_occurrences(prompt, record["atomic_number"]) == 0,
        "answer_absent_omits_query_local_number": not null or not line_contains_both(prompt, record["element"], record["local_number"]),
        "prompt_has_status_lexeme": bool(STATUS_LEXEME_RE.search(prompt)),
    }


def structural_check(
    records: list[dict[str, Any]],
    templates: tuple[str, ...],
    assignments: tuple[str, ...],
    rule_orders: tuple[str, ...],
) -> dict[str, Any]:
    source_ids = {row["source_id"] for row in records}
    expected_count = len(source_ids) * len(templates) * len(assignments) * len(rule_orders) * len(PANELS)
    split_by_source: dict[str, set[str]] = {}
    for row in records:
        split_by_source.setdefault(row["source_id"], set()).add(row["split"])
    split_rows = Counter(row["split"] for row in records)
    conflict_rows = [row for row in records if row["panel"] in PRIMARY_CONFLICT_PANELS]
    expected_conflict = Counter(row["expected_label"] for row in conflict_rows)
    operation_conflict = Counter(row["query_operation"] for row in conflict_rows)
    audits = [prompt_audit(row) for row in records]
    collisions = [
        row["id"]
        for row in records
        if len({candidate["answer"] for candidate in row["candidate_answers"]}) != len(row["candidate_answers"])
    ]
    malformed_candidates = [
        f"{row['id']}::{candidate['answer']}"
        for row in records
        for candidate in row["candidate_answers"]
        if ANSWER_RE.match(candidate["answer"]) is None
    ]
    suffixes = {"\n".join(row["prompt"].splitlines()[-3:]) for row in records}
    status_rows = [row["id"] for row, audit in zip(records, audits, strict=True) if audit["prompt_has_status_lexeme"]]
    atomic_leaks = [
        row["id"]
        for row, audit in zip(records, audits, strict=True)
        if not audit["atomic_number_hidden_in_conflict"]
    ]
    null_leaks = [
        row["id"]
        for row, audit in zip(records, audits, strict=True)
        if not audit["answer_absent_omits_query_local_number"]
    ]
    criteria = {
        "expected_row_count": len(records) == expected_count,
        "all_panels_present": set(PANELS) == {row["panel"] for row in records},
        "all_templates_present": set(templates) == {row["template"] for row in records},
        "all_assignments_present": set(assignments) == {row["operation_assignment"] for row in records},
        "all_rule_orders_present": set(rule_orders) == {row["rule_order"] for row in records},
        "source_split_disjoint": all(len(splits) == 1 for splits in split_by_source.values()),
        "holdout_sources_present": any(row["split"] == "holdout" for row in records),
        "calibration_sources_present": any(row["split"] == "calibration" for row in records),
        "atomic_number_hidden_in_conflicts": not atomic_leaks,
        "answer_absent_omits_query_local_number": not null_leaks,
        "candidate_answers_parseable": not malformed_candidates,
        "no_candidate_collisions": not collisions,
        "single_answer_suffix": len(suffixes) == 1,
        "prompts_have_no_status_lexemes": not status_rows,
        "conflict_expected_balanced": expected_conflict["local_number"] == expected_conflict["atomic_number"],
        "conflict_operations_balanced": operation_conflict["ALPHA"] == operation_conflict["BETA"],
    }
    return {
        "passed": all(criteria.values()),
        "record_count": len(records),
        "source_count": len(source_ids),
        "expected_record_count": expected_count,
        "split_row_counts": dict(sorted(split_rows.items())),
        "criteria": criteria,
        "candidate_collision_rows": collisions[:20],
        "malformed_candidate_rows": malformed_candidates[:20],
        "status_lexeme_rows": status_rows[:20],
        "atomic_leak_rows": atomic_leaks[:20],
        "null_leak_rows": null_leaks[:20],
        "conflict_expected_counts": dict(sorted(expected_conflict.items())),
        "conflict_operation_counts": dict(sorted(operation_conflict.items())),
    }


def strict_parse(record: dict[str, Any], generated_text: str) -> dict[str, Any]:
    stripped = generated_text.strip().lstrip("`'\" ")
    first_line = stripped.splitlines()[0].strip().strip("\"'`") if stripped else ""
    match = ANSWER_RE.fullmatch(first_line)
    if not match:
        return {
            "selected_label": "unparsed",
            "selected_answer": None,
            "parseable": False,
            "parse_rule": "not_bare_integer_or_unknown",
            "first_line": first_line,
        }
    answer = match.group(1)
    if answer == "UNKNOWN":
        label = "unknown"
    elif answer == record["local_number"]:
        label = "local_number"
    elif answer == record["atomic_number"]:
        label = "atomic_number"
    else:
        label = "other_number"
    return {
        "selected_label": label,
        "selected_answer": answer,
        "parseable": True,
        "parse_rule": "strict_bare_integer_or_unknown",
        "first_line": first_line,
    }


def final_next_token_logits(model: Any, tokenizer: Any, prompt: str, record: dict[str, Any]) -> dict[str, Any]:
    import torch
    from mc006_parametric_fact_override_v15_parser_normalized_signature import first_answer_token_id

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].detach().float().cpu()
    token_ids = {
        candidate["label"]: first_answer_token_id(tokenizer, candidate["answer"])
        for candidate in record["candidate_answers"]
    }
    scores = {label: float(logits[token_id].item()) for label, token_id in token_ids.items()}
    return {
        "candidate_first_token_ids": token_ids,
        "final_next_token_logits": scores,
        "final_local_minus_atomic_number_logit": scores["local_number"] - scores["atomic_number"],
        "final_unknown_minus_local_logit": scores["unknown"] - scores["local_number"],
    }


def candidate_logprob_payload(model: Any, tokenizer: Any, prompt: str, record: dict[str, Any]) -> dict[str, Any]:
    from mc006_parametric_fact_override_v15_parser_normalized_signature import token_logprob

    scores = {
        candidate["label"]: token_logprob(model, tokenizer, prompt, candidate["answer"])
        for candidate in record["candidate_answers"]
    }

    def mean(label: str) -> float:
        value = scores[label]["mean_logprob"]
        return float(value) if math.isfinite(float(value)) else float("-inf")

    return {
        "candidate_logprobs": scores,
        "candidate_local_minus_atomic_number_mean_logprob": mean("local_number") - mean("atomic_number"),
        "candidate_unknown_minus_local_mean_logprob": mean("unknown") - mean("local_number"),
    }


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
        parsed = strict_parse(record, generated["generated_text"])
        output = {
            **record,
            "index": index,
            **generated,
            **parsed,
            **prompt_audit(record),
            "expected_correct": parsed["selected_label"] == record["expected_label"],
        }
        if score_candidates:
            output.update(final_next_token_logits(model, tokenizer, record["prompt"], record))
            output.update(candidate_logprob_payload(model, tokenizer, record["prompt"], record))
        outputs.append(output)
        if verbose:
            print(
                f"[{index:04d}/{len(records):04d}] {record['id']} split={record['split']} "
                f"op={record['query_operation']} panel={record['panel']} "
                f"expected={record['expected_label']} -> {output['selected_label']} "
                f"{str(output['selected_answer'])!r}"
            )
    return outputs


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("selected_label")) for row in rows).items()))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parseable = sum(1 for row in rows if row.get("parseable"))
    counts = Counter(row.get("selected_label") for row in rows)
    expected_correct = sum(1 for row in rows if row.get("expected_correct"))
    return {
        "rows": len(rows),
        "label_counts": label_counts(rows),
        "parseable": parseable,
        "parseable_rate": rate(parseable, len(rows)),
        "local_number": counts["local_number"],
        "local_number_rate": rate(counts["local_number"], len(rows)),
        "atomic_number": counts["atomic_number"],
        "atomic_number_rate": rate(counts["atomic_number"], len(rows)),
        "unknown": counts["unknown"],
        "unknown_rate": rate(counts["unknown"], len(rows)),
        "other_number": counts["other_number"],
        "other_number_rate": rate(counts["other_number"], len(rows)),
        "unparsed": counts["unparsed"],
        "unparsed_rate": rate(counts["unparsed"], len(rows)),
        "expected_correct": expected_correct,
        "expected_correct_rate": rate(expected_correct, len(rows)),
        "expected_label_counts": dict(sorted(Counter(str(row.get("expected_label")) for row in rows).items())),
        "query_operation_counts": dict(sorted(Counter(str(row.get("query_operation")) for row in rows).items())),
    }


def margin_audits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows or "candidate_local_minus_atomic_number_mean_logprob" not in rows[0]:
        return {"reported": False}
    return {"reported": True}


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
            "margin_audits": margin_audits(rows),
        }
    return result


def selection_key(item: dict[str, Any], template: str, templates: tuple[str, ...]) -> tuple[float, float, float, float, int]:
    panels = item["panels"]
    control_floor = min(
        float(panels["synthetic_numeric_lookup"]["local_number_rate"]),
        float(panels["familiar_entity_numeric_lookup"]["local_number_rate"]),
        float(panels["real_world_atomic_number_control"]["atomic_number_rate"]),
        float(panels["answer_absent_null"]["unknown_rate"]),
    )
    conflict_floor = min(
        float(panels["operation_local_conflict"]["local_number_rate"]),
        float(panels["operation_atomic_conflict"]["atomic_number_rate"]),
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


def observed_pattern(selected: dict[str, Any]) -> str:
    panels = selected["panels"]
    if float(panels["real_world_atomic_number_control"]["atomic_number_rate"]) < 0.85:
        return "atomic_control_failed"
    if float(panels["operation_local_conflict"]["local_number_rate"]) < 0.85:
        return "operation_local_branch_failed"
    if float(panels["operation_atomic_conflict"]["atomic_number_rate"]) < 0.85:
        return "operation_atomic_branch_failed"
    return "query_operation_behavior_passed"


def classify(criteria: dict[str, bool], pattern: str) -> str:
    if criteria["smoke_mode"]:
        return "smoke_only"
    if not criteria["structural_passed"]:
        return "structural_invalid"
    if not criteria["selected_prompt_audit_passed"] or not criteria["prompts_have_no_status_lexemes"]:
        return "query_operation_prompt_audit_failed"
    if not criteria["all_controls_passed"]:
        return "query_operation_control_failed"
    if not criteria["operation_local_conflict_at_least_85p"]:
        return "query_operation_local_branch_failed"
    if not criteria["operation_atomic_conflict_at_least_85p"]:
        return "query_operation_atomic_branch_failed"
    if not criteria["candidate_and_output_margins_reported"]:
        return "query_operation_behavior_passed_baselines_missing"
    return "query_operation_behavior_passed"


def summarize(
    records: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    templates: tuple[str, ...],
    assignments: tuple[str, ...],
    rule_orders: tuple[str, ...],
    full_run: bool,
    score_candidates: bool,
) -> dict[str, Any]:
    structural = structural_check(records, templates, assignments, rule_orders)
    by_template = template_summary(outputs, templates)
    selection = select_template(by_template, templates)
    selected = by_template[selection["selected_template"]]
    panels = selected["panels"]
    selected_rows = [row for row in outputs if row["template"] == selection["selected_template"]]
    selected_prompt_audit_passed = all(
        row["atomic_number_hidden_in_conflict"]
        and row["answer_absent_omits_query_local_number"]
        for row in selected_rows
    )
    prompts_have_no_status_lexemes = all(not row["prompt_has_status_lexeme"] for row in selected_rows)
    criteria = {
        "smoke_mode": not full_run,
        "structural_passed": structural["passed"],
        "full_source_count_is_40": structural["source_count"] == 40,
        "source_disjoint_holdout": structural["criteria"]["source_split_disjoint"]
        and structural["criteria"]["holdout_sources_present"]
        and structural["criteria"]["calibration_sources_present"],
        "selected_prompt_audit_passed": selected_prompt_audit_passed,
        "prompts_have_no_status_lexemes": prompts_have_no_status_lexemes,
        "synthetic_control_at_least_90p": float(panels["synthetic_numeric_lookup"]["local_number_rate"]) >= 0.90,
        "familiar_control_at_least_90p": float(panels["familiar_entity_numeric_lookup"]["local_number_rate"]) >= 0.90,
        "atomic_control_at_least_85p": float(panels["real_world_atomic_number_control"]["atomic_number_rate"]) >= 0.85,
        "answer_absent_unknown_at_least_90p": float(panels["answer_absent_null"]["unknown_rate"]) >= 0.90,
        "operation_local_conflict_at_least_85p": float(panels["operation_local_conflict"]["local_number_rate"]) >= 0.85,
        "operation_atomic_conflict_at_least_85p": float(panels["operation_atomic_conflict"]["atomic_number_rate"]) >= 0.85,
        "operation_rule_absent_unknown_at_least_90p": float(panels["operation_rule_absent_conflict"]["unknown_rate"]) >= 0.90,
        "all_panels_parseable_at_least_95p": min(float(panel["parseable_rate"]) for panel in panels.values()) >= 0.95,
        "candidate_and_output_margins_reported": bool(score_candidates and selected["margin_audits"]["reported"]),
    }
    criteria["all_controls_passed"] = (
        criteria["synthetic_control_at_least_90p"]
        and criteria["familiar_control_at_least_90p"]
        and criteria["atomic_control_at_least_85p"]
        and criteria["answer_absent_unknown_at_least_90p"]
        and criteria["operation_rule_absent_unknown_at_least_90p"]
        and criteria["all_panels_parseable_at_least_95p"]
    )
    pattern = observed_pattern(selected)
    diagnostic_class = classify(criteria, pattern)
    behavior_gate_passed = diagnostic_class == "query_operation_behavior_passed"
    return {
        "structural": structural,
        "by_template": by_template,
        "selection": selection,
        "selected_template_summary": selected,
        "selected_template_rows": selected_rows,
        "criteria": criteria,
        "observed_failure_pattern": pattern,
        "passed": behavior_gate_passed,
        "behavior_ready": behavior_gate_passed,
        "behavior_gate_passed": behavior_gate_passed,
        "signature_ready": False,
        "intervention_ready": False,
        "diagnostic_class": diagnostic_class,
    }


def quiet_summary(summary: dict[str, Any], output_path: Path | None = None) -> dict[str, Any]:
    payload = {
        "diagnostic_class": summary["diagnostic_class"],
        "observed_failure_pattern": summary["observed_failure_pattern"],
        "passed": summary["passed"],
        "behavior_ready": summary["behavior_ready"],
        "signature_ready": summary["signature_ready"],
        "criteria": summary["criteria"],
        "selection": summary["selection"],
        "selected_controls": {
            panel: summary["selected_template_summary"]["panels"][panel]
            for panel in (
                "synthetic_numeric_lookup",
                "familiar_entity_numeric_lookup",
                "real_world_atomic_number_control",
                "answer_absent_null",
                "operation_rule_absent_conflict",
            )
        },
        "selected_conflicts": {
            panel: summary["selected_template_summary"]["panels"][panel]
            for panel in PRIMARY_CONFLICT_PANELS
        },
    }
    if output_path is not None:
        payload["output_path"] = str(output_path)
    return payload


def write_behavior_status_card(path: Path, result: dict[str, Any], output_path: Path) -> None:
    summary = result["summary"]
    selected = summary["selected_template_summary"]
    criteria = summary["criteria"]
    lines = [
        "# MC023 Query-Operation Numeric Arbitration Behavior Status",
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
        "  `code/mc023_query_operation_numeric_arbitration.py`",
        "- result:",
        f"  `{output_path.as_posix()}`",
        "",
        "## Verdict",
        "",
    ]
    if summary["behavior_gate_passed"]:
        lines.extend(
            [
                "The query-operation behavior gate passed.",
                "This would justify a separate bridge decision, but it is not",
                "itself a hidden-state mechanism claim.",
            ]
        )
    elif criteria["smoke_mode"]:
        lines.extend(
            [
                "This is a smoke or partial run. It diagnoses whether a",
                "query-level operation handle can repair local-versus-learned",
                "arbitration without row-level source-status text.",
            ]
        )
    else:
        lines.extend(
            [
                "The query-operation behavior gate failed.",
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
            "| Panel | Rows | Parseable | Local Rate | Atomic Rate | Unknown Rate | Other Rate | Expected Correct |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for panel in PANELS:
        item = selected["panels"][panel]
        lines.append(
            f"| `{panel}` | {item['rows']} | {item['parseable_rate']:.3f} | "
            f"{item['local_number_rate']:.3f} | {item['atomic_number_rate']:.3f} | "
            f"{item['unknown_rate']:.3f} | {item['other_number_rate']:.3f} | "
            f"{item['expected_correct_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate Conflict",
            "",
            f"- primary conflict expected-correct rate: {selected['primary_conflict']['expected_correct_rate']:.3f}",
            f"- operation-local local rate: {selected['panels']['operation_local_conflict']['local_number_rate']:.3f}",
            f"- operation-atomic atomic rate: {selected['panels']['operation_atomic_conflict']['atomic_number_rate']:.3f}",
            "",
            "## Claim Boundary",
            "",
            "MC023 is a behavior diagnostic for query-level operation handles over",
            "prompt-local versus learned atomic-number branches. It does not",
            "establish an internal signature, intervention, or mechanism card.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--score-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--score-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-manifest", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--artifact-prefix", default="mc023_query_operation_numeric_arbitration_structural")
    parser.add_argument("--status-card", type=Path, default=STATUS_CARD)
    parser.add_argument("--write-status-card", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.time()
    run_type = BEHAVIOR_RUN_TYPE if args.score_model else RUN_TYPE
    records = source_records(args.limit_sources, TEMPLATES, OPERATION_ASSIGNMENTS, RULE_ORDERS, run_type)
    structural = structural_check(records, TEMPLATES, OPERATION_ASSIGNMENTS, RULE_ORDERS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.write_manifest and not args.score_model:
        result = {
            "schema_version": 1,
            "card_id": CARD_ID,
            "run_type": RUN_TYPE,
            "model_id": args.model_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "purpose": "Structural gate for query-operation numeric arbitration.",
            "templates": list(TEMPLATES),
            "operation_assignments": list(OPERATION_ASSIGNMENTS),
            "rule_orders": list(RULE_ORDERS),
            "operation_codes": list(OPERATION_CODES),
            "panels": list(PANELS),
            "primary_conflict_panels": list(PRIMARY_CONFLICT_PANELS),
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
        full_run=full_run,
        score_candidates=args.score_candidates,
    )
    prefix = args.artifact_prefix
    if prefix == "mc023_query_operation_numeric_arbitration_structural":
        prefix = "mc023_query_operation_numeric_arbitration_behavior"
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
        "operation_codes": list(OPERATION_CODES),
        "elapsed_s": time.time() - started,
        "purpose": "Generated-answer query-operation numeric arbitration diagnostic before hidden-state work.",
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
