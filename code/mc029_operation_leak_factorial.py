#!/usr/bin/env python
"""MC029 factorized operation-leak diagnostic.

MC028 showed that MC027's bare-integer answer interface does not promote to a
full-source bridge. Direct controls and nulls stayed clean, but the learned
atomic operation branch leaked other numbers.

MC029 tests three concrete sources of that leak:

- numeric worked-example outputs;
- query anchoring relative to rules/examples;
- prompt-row salience from query-row position.

This is behavior-substrate work only. It does not license hidden-state probes
unless a variant clears behavior gates and later candidate/output baselines.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import mc027_answer_interface_sweep as mc027
from mc012_reliability_labeled_numeric_arbitration import MODEL_ID, base_sources, distractors, generate_answer
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


CARD_ID = "MC029"
RUN_TYPE = "operation_leak_factorial_structural_gate"
BEHAVIOR_RUN_TYPE = "operation_leak_factorial_behavior"
RESULT_DIR = Path("results/cards/MC029")
STATUS_CARD = Path("research/cards/MC029_OPERATION_LEAK_FACTORIAL_STATUS.md")

PANELS = (
    "atomic_control",
    "operation_local_conflict",
    "operation_atomic_conflict",
    "operation_rule_absent_null",
    "answer_absent_null",
)
PRIMARY_CONFLICT_PANELS = ("operation_local_conflict", "operation_atomic_conflict")

VARIANTS: tuple[dict[str, str], ...] = (
    {
        "template": "baseline_numeric_examples",
        "example_mode": "numeric",
        "query_position": "after_rules_examples",
        "table_order": "query_first",
    },
    {
        "template": "label_examples_no_numbers",
        "example_mode": "label_only",
        "query_position": "after_rules_examples",
        "table_order": "query_first",
    },
    {
        "template": "rules_only",
        "example_mode": "none",
        "query_position": "after_rules_examples",
        "table_order": "query_first",
    },
    {
        "template": "query_before_examples",
        "example_mode": "numeric",
        "query_position": "before_rules_examples",
        "table_order": "query_first",
    },
    {
        "template": "query_row_last",
        "example_mode": "numeric",
        "query_position": "after_rules_examples",
        "table_order": "query_last",
    },
)
TEMPLATES = tuple(variant["template"] for variant in VARIANTS)
VARIANT_BY_TEMPLATE = {variant["template"]: variant for variant in VARIANTS}
STATUS_LEXEME_RE = re.compile(r"\b(trusted|untrusted|reliable|unreliable|status)\b", re.IGNORECASE)
ROW_RE = re.compile(r"^(.+?) has local lab number (\d+)\.$")
EXAMPLE_RE = re.compile(r"^Example: (.+?) with operation (\w+) -> (\d+)\.$")


def response_suffix() -> list[str]:
    return [
        "Return only one final token: an integer or UNKNOWN.",
        "Do not explain.",
        "Answer:",
    ]


def ordered_rows(
    rows: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    include_query: bool,
    table_order: str,
) -> list[dict[str, Any]]:
    if include_query:
        visible = list(rows)
    else:
        visible = [row for row in rows if row["source_id"] != source["source_id"]]
    if not include_query or table_order == "query_first":
        return visible
    if table_order == "query_last":
        others = [row for row in visible if row["source_id"] != source["source_id"]]
        return [*others, source]
    raise ValueError(f"unknown table_order {table_order}")


def local_table_lines(
    rows: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    include_query: bool,
    table_order: str,
) -> list[str]:
    lines = [
        "Two numeric systems may conflict.",
        "Rows:",
    ]
    for row in ordered_rows(rows, source, include_query=include_query, table_order=table_order):
        lines.append(f"{row['element']} has local lab number {row['local_number']}.")
    return lines


def label_example_lines(
    example_rows: list[dict[str, Any]],
    *,
    assignment: str,
) -> list[str]:
    mapping = operation_mapping(assignment)
    local_operation = operation_for_label("local_number", assignment)
    atomic_operation = operation_for_label("atomic_number", assignment)
    local_row = example_rows[0]
    atomic_row = example_rows[1]
    lines = []
    for operation, row in ((local_operation, local_row), (atomic_operation, atomic_row)):
        target = mapping[operation]
        if target == "local_number":
            answer = "the queried row's local lab number"
        elif target == "atomic_number":
            answer = "the queried element's standard atomic number"
        else:
            raise ValueError(f"unknown operation target {target}")
        lines.append(f"Example: {row['element']} with operation {operation} -> {answer}.")
    return lines


def append_rules_and_examples(
    lines: list[str],
    example_rows: list[dict[str, Any]],
    *,
    assignment: str,
    rule_order: str,
    variant: dict[str, str],
) -> None:
    append_operation_rules(lines, assignment=assignment, rule_order=rule_order, template="compact_operation_rules")
    if variant["example_mode"] == "numeric":
        for item in operation_example_lines(example_rows, assignment=assignment)[1:]:
            lines.append(item)
        return
    if variant["example_mode"] == "label_only":
        lines.extend(label_example_lines(example_rows, assignment=assignment))
        return
    if variant["example_mode"] == "none":
        return
    raise ValueError(f"unknown example_mode {variant['example_mode']}")


def add_query_and_rules(
    lines: list[str],
    example_rows: list[dict[str, Any]],
    *,
    source: dict[str, Any],
    operation: str,
    assignment: str,
    rule_order: str,
    variant: dict[str, str],
) -> None:
    query_line = operation_query_line(operation, source, "compact_operation_rules")
    if variant["query_position"] == "before_rules_examples":
        lines.append(query_line)
        append_rules_and_examples(lines, example_rows, assignment=assignment, rule_order=rule_order, variant=variant)
        return
    if variant["query_position"] == "after_rules_examples":
        append_rules_and_examples(lines, example_rows, assignment=assignment, rule_order=rule_order, variant=variant)
        lines.append(query_line)
        return
    raise ValueError(f"unknown query_position {variant['query_position']}")


def make_prompt(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    panel: str,
    variant: dict[str, str],
    assignment: str,
    rule_order: str,
) -> tuple[str, str, str]:
    rows = distractors(sources, source, 5)
    example_rows = [row for row in rows if row["source_id"] != source["source_id"]][:2]
    if len(example_rows) != 2:
        raise ValueError("not enough example rows")

    if panel == "atomic_control":
        lines = [
            "No local lab table is active.",
            "Return the standard atomic number for the queried element.",
            f"Query element: {source['element']}",
        ]
        return "\n".join([*lines, *response_suffix()]), "atomic_number", "NONE"

    if panel == "operation_local_conflict":
        operation = operation_for_label("local_number", assignment)
        lines = local_table_lines(rows, source, include_query=True, table_order=variant["table_order"])
        add_query_and_rules(
            lines,
            example_rows,
            source=source,
            operation=operation,
            assignment=assignment,
            rule_order=rule_order,
            variant=variant,
        )
        lines.append("Apply the query operation and return the controlling number.")
        return "\n".join([*lines, *response_suffix()]), "local_number", operation

    if panel == "operation_atomic_conflict":
        operation = operation_for_label("atomic_number", assignment)
        lines = local_table_lines(rows, source, include_query=True, table_order=variant["table_order"])
        add_query_and_rules(
            lines,
            example_rows,
            source=source,
            operation=operation,
            assignment=assignment,
            rule_order=rule_order,
            variant=variant,
        )
        lines.append("Apply the query operation and return the controlling number.")
        return "\n".join([*lines, *response_suffix()]), "atomic_number", operation

    if panel == "operation_rule_absent_null":
        operation = operation_for_label("local_number", assignment)
        lines = local_table_lines(rows, source, include_query=True, table_order=variant["table_order"])
        lines.append("No operation rule or worked example is available for this query.")
        lines.append(operation_query_line(operation, source, "compact_operation_rules"))
        lines.append("If no operation rule is available, return UNKNOWN.")
        return "\n".join([*lines, *response_suffix()]), "unknown", operation

    if panel == "answer_absent_null":
        operation = operation_for_label("local_number", assignment)
        lines = local_table_lines(rows, source, include_query=False, table_order=variant["table_order"])
        add_query_and_rules(
            lines,
            example_rows,
            source=source,
            operation=operation,
            assignment=assignment,
            rule_order=rule_order,
            variant=variant,
        )
        lines.append("If the queried element is absent from the table, return UNKNOWN before applying any operation.")
        return "\n".join([*lines, *response_suffix()]), "unknown", operation

    raise ValueError(f"unknown panel {panel}")


def label_value(label: str, source: dict[str, Any]) -> str:
    if label == "local_number":
        return str(source["local_number"])
    if label == "atomic_number":
        return str(source["atomic_number"])
    if label == "unknown":
        return "UNKNOWN"
    raise ValueError(f"unknown label {label}")


def candidate_answers_for(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "answer": record["local_number"],
            "label": "local_number",
            "candidate_type": "local_number",
        },
        {
            "answer": record["atomic_number"],
            "label": "atomic_number",
            "candidate_type": "atomic_number",
        },
        {
            "answer": "UNKNOWN",
            "label": "unknown",
            "candidate_type": "unknown",
        },
    ]


def source_records(limit_sources: int | None = None, run_type: str = RUN_TYPE) -> list[dict[str, Any]]:
    sources = base_sources(limit_sources)
    records: list[dict[str, Any]] = []
    for source in sources:
        for variant in VARIANTS:
            for assignment in OPERATION_ASSIGNMENTS:
                for rule_order in RULE_ORDERS:
                    for panel in PANELS:
                        prompt, expected_label, operation = make_prompt(
                            sources,
                            source,
                            panel=panel,
                            variant=variant,
                            assignment=assignment,
                            rule_order=rule_order,
                        )
                        record = {
                            "id": (
                                f"{CARD_ID}_{variant['template']}_{assignment}_{rule_order}_"
                                f"{panel}_{source['source_id']}"
                            ),
                            "card_id": CARD_ID,
                            "run_type": run_type,
                            "model_id": MODEL_ID,
                            "template": variant["template"],
                            "interface_variant": "bare_integer",
                            "option_order": "none",
                            "first_option_label": "none",
                            "example_mode": variant["example_mode"],
                            "query_position": variant["query_position"],
                            "table_order": variant["table_order"],
                            "operation_assignment": assignment,
                            "rule_order": rule_order,
                            "panel": panel,
                            "split": source["split"],
                            "source_id": source["source_id"],
                            "source_index": source["source_index"],
                            "element": source["element"],
                            "synthetic_key": source["synthetic_key"],
                            "query_operation": operation,
                            "atomic_number": str(source["atomic_number"]),
                            "local_number": str(source["local_number"]),
                            "expected_label": expected_label,
                            "expected_answer": label_value(expected_label, source),
                            "label_to_answer": {
                                "local_number": str(source["local_number"]),
                                "atomic_number": str(source["atomic_number"]),
                                "unknown": "UNKNOWN",
                            },
                            "choice_to_label": {},
                            "label_to_choice": {},
                            "prompt": prompt,
                        }
                        record["candidate_answers"] = candidate_answers_for(record)
                        records.append(record)
    return records


def word_occurrences(text: str, value: str) -> int:
    pattern = rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])"
    return len(re.findall(pattern, text))


def prompt_has_query_local_row(record: dict[str, Any]) -> bool:
    return any(
        record["element"] in line and record["local_number"] in line
        for line in record["prompt"].splitlines()
    )


def structural_check(records: list[dict[str, Any]]) -> dict[str, Any]:
    source_ids = {row["source_id"] for row in records}
    expected_count = len(source_ids) * len(VARIANTS) * len(OPERATION_ASSIGNMENTS) * len(RULE_ORDERS) * len(PANELS)
    split_by_source: dict[str, set[str]] = {}
    for row in records:
        split_by_source.setdefault(row["source_id"], set()).add(row["split"])
    conflict_rows = [row for row in records if row["panel"] in PRIMARY_CONFLICT_PANELS]
    split_rows = Counter(row["split"] for row in records)
    template_counts = Counter(row["template"] for row in records)
    expected_conflict = Counter(row["expected_label"] for row in conflict_rows)
    operation_conflict = Counter(row["query_operation"] for row in conflict_rows)
    status_rows = [row["id"] for row in records if STATUS_LEXEME_RE.search(row["prompt"])]
    target_atomic_leaks = [
        row["id"]
        for row in conflict_rows
        if word_occurrences(row["prompt"], row["atomic_number"]) > 0
    ]
    null_leaks = [
        row["id"]
        for row in records
        if row["panel"] == "answer_absent_null" and prompt_has_query_local_row(row)
    ]
    candidate_collisions = [
        row["id"]
        for row in records
        if len({candidate["answer"] for candidate in row["candidate_answers"]}) != len(row["candidate_answers"])
    ]
    criteria = {
        "expected_row_count": len(records) == expected_count,
        "full_source_count_is_40": len(source_ids) == 40,
        "all_panels_present": set(PANELS) == {row["panel"] for row in records},
        "all_templates_present": set(TEMPLATES) == {row["template"] for row in records},
        "all_assignments_present": set(OPERATION_ASSIGNMENTS) == {row["operation_assignment"] for row in records},
        "all_rule_orders_present": set(RULE_ORDERS) == {row["rule_order"] for row in records},
        "source_split_disjoint": all(len(splits) == 1 for splits in split_by_source.values()),
        "holdout_sources_present": any(row["split"] == "holdout" for row in records),
        "calibration_sources_present": any(row["split"] == "calibration" for row in records),
        "target_atomic_number_hidden_in_conflicts": not target_atomic_leaks,
        "answer_absent_omits_query_local_row": not null_leaks,
        "no_candidate_collisions": not candidate_collisions,
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
        "template_counts": dict(sorted(template_counts.items())),
        "criteria": criteria,
        "status_lexeme_rows": status_rows[:20],
        "target_atomic_leak_rows": target_atomic_leaks[:20],
        "null_leak_rows": null_leaks[:20],
        "candidate_collision_rows": candidate_collisions[:20],
        "conflict_expected_counts": dict(sorted(expected_conflict.items())),
        "conflict_operation_counts": dict(sorted(operation_conflict.items())),
    }


def score_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    verbose: bool,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        generated = generate_answer(model, tokenizer, record["prompt"], max_new_tokens)
        parsed = mc027.strict_parse(record, generated["generated_text"])
        output = {
            **record,
            "index": index,
            **generated,
            **parsed,
            "expected_correct": parsed["selected_label"] == record["expected_label"],
        }
        outputs.append(output)
        if verbose:
            print(
                f"[{index:04d}/{len(records):04d}] {record['id']} "
                f"expected={record['expected_label']} -> {output['selected_label']} "
                f"{str(output['selected_answer'])!r}"
            )
    return outputs


def prompt_number_sets(prompt: str) -> tuple[set[str], set[str]]:
    prompt_local_numbers: set[str] = set()
    example_outputs: set[str] = set()
    for line in prompt.splitlines():
        row_match = ROW_RE.match(line)
        if row_match:
            prompt_local_numbers.add(row_match.group(2))
            continue
        example_match = EXAMPLE_RE.match(line)
        if example_match:
            example_outputs.add(example_match.group(3))
    return prompt_local_numbers, example_outputs


def other_number_bucket(row: dict[str, Any], all_atomic: set[str], all_local: set[str]) -> tuple[str, list[str]]:
    selected = str(row["selected_answer"])
    prompt_local_numbers, example_outputs = prompt_number_sets(row["prompt"])
    labels: list[str] = []
    if selected in example_outputs:
        labels.append("worked_example_output")
    if selected in prompt_local_numbers:
        labels.append("prompt_local_row_number")
    if selected in all_local:
        labels.append("bank_local_number")
    if selected in all_atomic:
        labels.append("bank_atomic_number")
    atomic_value = int(row["atomic_number"])
    if selected == str(atomic_value * 2):
        labels.append("double_expected_atomic")
    if selected == str(atomic_value + 10):
        labels.append("expected_atomic_plus_10")
    if not labels:
        labels.append("off_bank_other_number")

    if "worked_example_output" in labels:
        bucket = "worked_example_output"
    elif "prompt_local_row_number" in labels:
        bucket = "prompt_local_row_number"
    elif "bank_local_number" in labels:
        bucket = "bank_local_number"
    elif "bank_atomic_number" in labels:
        bucket = "bank_atomic_number"
    else:
        bucket = "off_bank_other_number"
    return bucket, labels


def variant_other_audit(rows: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_atomic = {row["atomic_number"] for row in all_rows if "atomic_number" in row}
    all_local = {row["local_number"] for row in all_rows if "local_number" in row}
    op_atomic = [row for row in rows if row["panel"] == "operation_atomic_conflict"]
    other_rows = [row for row in op_atomic if row.get("selected_label") == "other_number"]
    bucket_counts: Counter[str] = Counter()
    overlap_counts: Counter[str] = Counter()
    for row in other_rows:
        bucket, labels = other_number_bucket(row, all_atomic, all_local)
        bucket_counts[bucket] += 1
        overlap_counts.update(labels)
    return {
        "operation_atomic_rows": len(op_atomic),
        "other_number_count": len(other_rows),
        "other_number_rate": mc027.rate(len(other_rows), len(op_atomic)),
        "exclusive_bucket_counts": dict(sorted(bucket_counts.items())),
        "overlap_counts": dict(sorted(overlap_counts.items())),
    }


def summarize_variant(rows: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    panels = {panel: mc027.summarize_rows([row for row in rows if row["panel"] == panel]) for panel in PANELS}
    conflict_rows = [row for row in rows if row["panel"] in PRIMARY_CONFLICT_PANELS]
    return {
        "rows": len(rows),
        "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "panels": panels,
        "primary_conflict": mc027.summarize_rows(conflict_rows),
        "primary_conflict_by_operation": mc027.grouped_summaries(conflict_rows, "query_operation"),
        "primary_conflict_by_assignment": mc027.grouped_summaries(conflict_rows, "operation_assignment"),
        "primary_conflict_by_rule_order": mc027.grouped_summaries(conflict_rows, "rule_order"),
        "other_number_audit": variant_other_audit(rows, all_rows),
    }


def variant_gate(panels: dict[str, Any]) -> bool:
    return (
        float(panels["atomic_control"]["atomic_number_rate"]) >= 0.85
        and float(panels["operation_local_conflict"]["local_number_rate"]) >= 0.85
        and float(panels["operation_atomic_conflict"]["atomic_number_rate"]) >= 0.85
        and float(panels["operation_atomic_conflict"]["other_number_rate"]) < 0.10
        and float(panels["operation_rule_absent_null"]["unknown_rate"]) >= 0.90
        and float(panels["answer_absent_null"]["unknown_rate"]) >= 0.90
        and min(float(panel["parseable_rate"]) for panel in panels.values()) >= 0.95
    )


def selection_key(item: dict[str, Any], template: str) -> tuple[float, float, float, float, int]:
    panels = item["panels"]
    control_floor = min(
        float(panels["atomic_control"]["atomic_number_rate"]),
        float(panels["operation_local_conflict"]["local_number_rate"]),
        float(panels["operation_rule_absent_null"]["unknown_rate"]),
        float(panels["answer_absent_null"]["unknown_rate"]),
    )
    atomic_rate = float(panels["operation_atomic_conflict"]["atomic_number_rate"])
    other_rate = float(panels["operation_atomic_conflict"]["other_number_rate"])
    parse_floor = min(float(panel["parseable_rate"]) for panel in panels.values())
    return (
        control_floor,
        atomic_rate,
        -other_rate,
        parse_floor,
        -TEMPLATES.index(template),
    )


def classify(criteria: dict[str, Any], by_template: dict[str, Any]) -> tuple[str, str]:
    if not criteria["structural_passed"]:
        return "operation_leak_factorial_structural_invalid", "structural_invalid"
    if criteria["any_variant_behavior_gate_passed"]:
        if criteria["candidate_and_output_margins_reported"]:
            return "operation_leak_factorial_behavior_ready", "factorial_behavior_ready"
        return (
            "operation_leak_factorial_candidate_baselines_missing",
            "factorial_passed_without_margin_baselines",
        )
    if criteria["rules_only_operation_atomic_rate"] >= 0.85 and criteria["rules_only_answer_absent_unknown_rate"] < 0.90:
        return (
            "operation_leak_factorial_branch_null_tradeoff",
            "rules_only_improves_atomic_branch_but_nulls_fail",
        )
    if criteria["max_operation_atomic_rate"] >= criteria["baseline_operation_atomic_rate"] + 0.05:
        return "operation_leak_factorial_moves_branch_but_not_gate", "factor_moved_branch_not_gate"
    if criteria["min_other_number_rate"] <= 0.10:
        return (
            "operation_leak_factorial_reduces_other_numbers_but_not_gate",
            "other_number_leak_reduced_without_behavior_gate",
        )
    baseline_worked = by_template["baseline_numeric_examples"]["other_number_audit"]["exclusive_bucket_counts"].get(
        "worked_example_output",
        0,
    )
    label_worked = by_template["label_examples_no_numbers"]["other_number_audit"]["exclusive_bucket_counts"].get(
        "worked_example_output",
        0,
    )
    if baseline_worked and label_worked <= baseline_worked / 2:
        return (
            "operation_leak_factorial_worked_example_leak_confirmed",
            "worked_example_outputs_removed_but_gate_still_failed",
        )
    return "operation_leak_factorial_no_repair", "factorial_no_repair"


def summarize(records: list[dict[str, Any]], outputs: list[dict[str, Any]], full_run: bool) -> dict[str, Any]:
    structural = structural_check(records)
    by_template = {
        template: summarize_variant([row for row in outputs if row["template"] == template], outputs)
        for template in TEMPLATES
    }
    variant_gates = {
        template: variant_gate(by_template[template]["panels"])
        for template in TEMPLATES
    }
    selected_template = max(TEMPLATES, key=lambda template: selection_key(by_template[template], template))
    baseline = by_template["baseline_numeric_examples"]["panels"]["operation_atomic_conflict"]
    selected = by_template[selected_template]["panels"]["operation_atomic_conflict"]
    max_atomic_template = max(
        TEMPLATES,
        key=lambda template: float(by_template[template]["panels"]["operation_atomic_conflict"]["atomic_number_rate"]),
    )
    min_other_template = min(
        TEMPLATES,
        key=lambda template: float(by_template[template]["panels"]["operation_atomic_conflict"]["other_number_rate"]),
    )
    rules_only_panels = by_template["rules_only"]["panels"]
    label_audit = by_template["label_examples_no_numbers"]["other_number_audit"]
    baseline_audit = by_template["baseline_numeric_examples"]["other_number_audit"]
    query_before_audit = by_template["query_before_examples"]["other_number_audit"]
    criteria = {
        "smoke_mode": not full_run,
        "structural_passed": structural["passed"],
        "full_source_count_is_40": structural["source_count"] == 40,
        "source_disjoint_holdout": structural["criteria"]["source_split_disjoint"]
        and structural["criteria"]["holdout_sources_present"]
        and structural["criteria"]["calibration_sources_present"],
        "prompts_have_no_status_lexemes": structural["criteria"]["prompts_have_no_status_lexemes"],
        "target_atomic_hidden_in_conflicts": structural["criteria"]["target_atomic_number_hidden_in_conflicts"],
        "answer_absent_omits_query_local_row": structural["criteria"]["answer_absent_omits_query_local_row"],
        "variant_gate_by_template": variant_gates,
        "any_variant_behavior_gate_passed": any(variant_gates.values()),
        "baseline_operation_atomic_rate": float(baseline["atomic_number_rate"]),
        "baseline_other_number_rate": float(baseline["other_number_rate"]),
        "selected_template_operation_atomic_rate": float(selected["atomic_number_rate"]),
        "selected_template_other_number_rate": float(selected["other_number_rate"]),
        "max_operation_atomic_template": max_atomic_template,
        "max_operation_atomic_rate": float(
            by_template[max_atomic_template]["panels"]["operation_atomic_conflict"]["atomic_number_rate"]
        ),
        "min_other_number_template": min_other_template,
        "min_other_number_rate": float(
            by_template[min_other_template]["panels"]["operation_atomic_conflict"]["other_number_rate"]
        ),
        "rules_only_operation_atomic_rate": float(rules_only_panels["operation_atomic_conflict"]["atomic_number_rate"]),
        "rules_only_other_number_rate": float(rules_only_panels["operation_atomic_conflict"]["other_number_rate"]),
        "rules_only_answer_absent_unknown_rate": float(rules_only_panels["answer_absent_null"]["unknown_rate"]),
        "baseline_worked_example_other_count": baseline_audit["exclusive_bucket_counts"].get("worked_example_output", 0),
        "label_examples_worked_example_other_count": label_audit["exclusive_bucket_counts"].get("worked_example_output", 0),
        "query_before_worked_example_other_count": query_before_audit["exclusive_bucket_counts"].get("worked_example_output", 0),
        "candidate_and_output_margins_reported": False,
    }
    diagnostic_class, observed_pattern = classify(criteria, by_template)
    behavior_gate_passed = (
        not criteria["smoke_mode"]
        and criteria["any_variant_behavior_gate_passed"]
        and criteria["candidate_and_output_margins_reported"]
    )
    behavior_candidate = (
        not criteria["smoke_mode"]
        and criteria["any_variant_behavior_gate_passed"]
        and not criteria["candidate_and_output_margins_reported"]
    )
    selected_summary = dict(by_template[selected_template])
    selected_summary["variant_operation_atomic"] = {
        template: by_template[template]["panels"]["operation_atomic_conflict"]
        for template in TEMPLATES
    }
    selected_summary["variant_answer_absent"] = {
        template: by_template[template]["panels"]["answer_absent_null"]
        for template in TEMPLATES
    }
    selected_summary["variant_operation_local"] = {
        template: by_template[template]["panels"]["operation_local_conflict"]
        for template in TEMPLATES
    }
    return {
        "structural": structural,
        "by_template": by_template,
        "selection": {
            "selected_template": selected_template,
            "selection_key": list(selection_key(by_template[selected_template], selected_template)),
            "all_selection_keys": {
                template: list(selection_key(by_template[template], template))
                for template in TEMPLATES
            },
            "rule": [
                "max direct-control/null/local floor",
                "max operation-atomic atomic rate",
                "min operation-atomic other-number rate",
                "max parseability",
                "earliest variant",
            ],
        },
        "selected_template_summary": selected_summary,
        "criteria": criteria,
        "observed_failure_pattern": observed_pattern,
        "passed": behavior_gate_passed,
        "behavior_ready": behavior_gate_passed,
        "behavior_candidate": behavior_candidate,
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
        "behavior_candidate": summary["behavior_candidate"],
        "signature_ready": summary["signature_ready"],
        "criteria": summary["criteria"],
        "selection": summary["selection"],
        "variant_operation_atomic": {
            template: summary["by_template"][template]["panels"]["operation_atomic_conflict"]
            for template in TEMPLATES
        },
        "variant_other_number_audit": {
            template: summary["by_template"][template]["other_number_audit"]
            for template in TEMPLATES
        },
    }
    if output_path is not None:
        payload["output_path"] = str(output_path)
    return payload


def write_behavior_status_card(path: Path, result: dict[str, Any], output_path: Path) -> None:
    summary = result["summary"]
    criteria = summary["criteria"]
    lines = [
        "# MC029 Operation Leak Factorial Status",
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
        "  `code/mc029_operation_leak_factorial.py`",
        "- result:",
        f"  `{output_path.as_posix()}`",
        "",
        "## Verdict",
        "",
        "MC029 is a factorized diagnostic for MC028's full-source other-number leak.",
        "It does not establish a hidden signature, intervention, or mechanism card.",
        "",
        "## Gate Criteria",
        "",
        "| Criterion | Value |",
        "| --- | --- |",
    ]
    for key, value in criteria.items():
        if isinstance(value, dict):
            rendered = json.dumps(value, sort_keys=True)
        elif isinstance(value, float):
            rendered = f"{value:.3f}"
        else:
            rendered = str(value).lower() if isinstance(value, bool) else str(value)
        lines.append(f"| `{key}` | `{rendered}` |")

    lines.extend(
        [
            "",
            "## Variant Comparison",
            "",
            "| Variant | Atomic Control | Local Branch | Atomic Branch | Other Number | Rule Null | Answer Null | Worked-Example Other | Gate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for template in TEMPLATES:
        item = summary["by_template"][template]
        panels = item["panels"]
        audit = item["other_number_audit"]
        worked = audit["exclusive_bucket_counts"].get("worked_example_output", 0)
        gate = criteria["variant_gate_by_template"][template]
        lines.append(
            f"| `{template}` | {panels['atomic_control']['atomic_number_rate']:.3f} | "
            f"{panels['operation_local_conflict']['local_number_rate']:.3f} | "
            f"{panels['operation_atomic_conflict']['atomic_number_rate']:.3f} | "
            f"{panels['operation_atomic_conflict']['other_number_rate']:.3f} | "
            f"{panels['operation_rule_absent_null']['unknown_rate']:.3f} | "
            f"{panels['answer_absent_null']['unknown_rate']:.3f} | {worked} | "
            f"`{str(gate).lower()}` |"
        )

    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "MC029 can only say which prompt-surface factors move the MC028 leak.",
            "It cannot license hidden-state work until a behavior-passing variant",
            "also reports candidate/output margin baselines.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--score-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-manifest", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--artifact-prefix", default="mc029_operation_leak_factorial_structural")
    parser.add_argument("--status-card", type=Path, default=STATUS_CARD)
    parser.add_argument("--write-status-card", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.time()
    run_type = BEHAVIOR_RUN_TYPE if args.score_model else RUN_TYPE
    records = source_records(args.limit_sources, run_type)
    structural = structural_check(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.write_manifest and not args.score_model:
        result = {
            "schema_version": 1,
            "card_id": CARD_ID,
            "run_type": RUN_TYPE,
            "model_id": args.model_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "purpose": "Structural gate for factorized MC028 operation-leak diagnostic.",
            "templates": list(TEMPLATES),
            "variants": list(VARIANTS),
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
    outputs = score_records(records, tokenizer, model, args.max_new_tokens, verbose=not args.quiet)
    full_run = args.limit_sources is None
    summary = summarize(records, outputs, full_run)
    prefix = args.artifact_prefix
    if prefix == "mc029_operation_leak_factorial_structural":
        prefix = "mc029_operation_leak_factorial_behavior"
    output_path = args.output_dir / f"{prefix}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    result = {
        "schema_version": 1,
        "card_id": CARD_ID,
        "run_type": BEHAVIOR_RUN_TYPE,
        "model_id": args.model_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "max_new_tokens": args.max_new_tokens,
        "decoding": {"do_sample": False},
        "score_candidates": False,
        "limit_sources": args.limit_sources,
        "templates": list(TEMPLATES),
        "variants": list(VARIANTS),
        "operation_assignments": list(OPERATION_ASSIGNMENTS),
        "rule_orders": list(RULE_ORDERS),
        "operation_codes": list(OPERATION_CODES),
        "elapsed_s": time.time() - started,
        "purpose": "Factorized diagnostic for MC028 operation-atomic other-number leakage.",
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
    return 0 if (summary["passed"] or not full_run) else 2


if __name__ == "__main__":
    raise SystemExit(main())
