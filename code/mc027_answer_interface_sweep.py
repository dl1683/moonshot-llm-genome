#!/usr/bin/env python
"""MC027 answer-interface sweep for operation arbitration.

MC024/MC025/MC026 showed that output format is not neutral. Open integers,
A/B/C choices, and visible numeric options each changed different parts of the
local-versus-learned bridge. MC027 makes that interface effect the object under
test instead of trying another single prompt repair.

The experiment keeps one compact query-operation substrate with worked examples
and sweeps final-answer interfaces:

- bare integer or UNKNOWN;
- ANSWER=<value>;
- JSON {"answer": value};
- A/B/C choices with balanced option order;
- visible numeric options with balanced option order.

This is behavior-substrate work only. A pass would justify a later hidden-state
route; it would not itself establish a signature, intervention, or mechanism.
"""

from __future__ import annotations

import argparse
import json
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
from mc023_query_operation_numeric_arbitration import (
    OPERATION_ASSIGNMENTS,
    OPERATION_CODES,
    RULE_ORDERS,
    append_operation_rules,
    operation_for_label,
    operation_query_line,
)
from mc024_fewshot_operation_numeric_arbitration import operation_example_lines


CARD_ID = "MC027"
RUN_TYPE = "answer_interface_sweep_structural_gate"
BEHAVIOR_RUN_TYPE = "answer_interface_sweep_behavior"
RESULT_DIR = Path("results/cards/MC027")
STATUS_CARD = Path("research/cards/MC027_ANSWER_INTERFACE_SWEEP_BEHAVIOR_STATUS.md")

PANELS = (
    "familiar_interface_lookup",
    "atomic_interface_control",
    "operation_local_interface_conflict",
    "operation_atomic_interface_conflict",
    "operation_rule_absent_interface_null",
    "answer_absent_interface_null",
)
PRIMARY_CONFLICT_PANELS = ("operation_local_interface_conflict", "operation_atomic_interface_conflict")
INTERFACE_VARIANTS = (
    "bare_integer",
    "answer_prefix",
    "json_answer",
    "letter_choices",
    "numeric_options",
)
OPTION_ORDERS = (
    "local_atomic_unknown",
    "atomic_unknown_local",
    "unknown_local_atomic",
)
NO_OPTION_ORDER = "none"
CHOICES = ("A", "B", "C")
CHOICE_RE = re.compile(r"^([ABC])[\s\.\):,;!\?]*$", re.IGNORECASE)
PREFIX_RE = re.compile(r"^ANSWER\s*=\s*(\d+|UNKNOWN)\s*$", re.IGNORECASE)
STATUS_LEXEME_RE = re.compile(r"\b(trusted|untrusted|reliable|unreliable|status)\b", re.IGNORECASE)


def option_order_labels(option_order: str) -> tuple[str, str, str]:
    if option_order == "local_atomic_unknown":
        return ("local_number", "atomic_number", "unknown")
    if option_order == "atomic_unknown_local":
        return ("atomic_number", "unknown", "local_number")
    if option_order == "unknown_local_atomic":
        return ("unknown", "local_number", "atomic_number")
    raise ValueError(f"unknown option order {option_order}")


def interface_configs() -> tuple[dict[str, str], ...]:
    configs: list[dict[str, str]] = []
    for variant in ("bare_integer", "answer_prefix", "json_answer"):
        configs.append(
            {
                "interface_variant": variant,
                "option_order": NO_OPTION_ORDER,
                "template": variant,
            }
        )
    for variant in ("letter_choices", "numeric_options"):
        for option_order in OPTION_ORDERS:
            configs.append(
                {
                    "interface_variant": variant,
                    "option_order": option_order,
                    "template": f"{variant}_{option_order}",
                }
            )
    return tuple(configs)


INTERFACE_CONFIGS = interface_configs()
TEMPLATES = tuple(config["template"] for config in INTERFACE_CONFIGS)


def label_value(label: str, source: dict[str, Any]) -> str:
    if label == "local_number":
        return str(source["local_number"])
    if label == "atomic_number":
        return str(source["atomic_number"])
    if label == "unknown":
        return "UNKNOWN"
    raise ValueError(f"unknown label {label}")


def local_table_lines(
    rows: list[dict[str, Any]],
    query: dict[str, Any],
    *,
    include_query: bool,
) -> list[str]:
    visible_rows = rows if include_query else [row for row in rows if row["source_id"] != query["source_id"]]
    lines = ["Rows:"]
    for row in visible_rows:
        lines.append(f"{row['element']} has local lab number {row['local_number']}.")
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


def choice_maps(source: dict[str, Any], option_order: str) -> tuple[list[str], dict[str, str], dict[str, str]]:
    labels = option_order_labels(option_order)
    lines = ["Answer choices:"]
    choice_to_label: dict[str, str] = {}
    label_to_choice: dict[str, str] = {}
    for choice, label in zip(CHOICES, labels, strict=True):
        choice_to_label[choice] = label
        label_to_choice[label] = choice
        lines.append(f"{choice}. {label_value(label, source)}")
    return lines, choice_to_label, label_to_choice


def numeric_option_lines(source: dict[str, Any], option_order: str) -> tuple[list[str], dict[str, str]]:
    labels = option_order_labels(option_order)
    label_to_answer = {label: label_value(label, source) for label in labels}
    lines = ["Allowed answers:"]
    for label in labels:
        lines.append(f"- {label_to_answer[label]}")
    return lines, label_to_answer


def interface_payload(
    source: dict[str, Any],
    *,
    interface_variant: str,
    option_order: str,
) -> dict[str, Any]:
    if interface_variant == "bare_integer":
        return {
            "option_lines": [],
            "choice_to_label": {},
            "label_to_choice": {},
            "label_to_answer": {
                "local_number": str(source["local_number"]),
                "atomic_number": str(source["atomic_number"]),
                "unknown": "UNKNOWN",
            },
            "response_suffix": [
                "Return only one final token: an integer or UNKNOWN.",
                "Do not explain.",
                "Answer:",
            ],
        }
    if interface_variant == "answer_prefix":
        return {
            "option_lines": [],
            "choice_to_label": {},
            "label_to_choice": {},
            "label_to_answer": {
                "local_number": f"ANSWER={source['local_number']}",
                "atomic_number": f"ANSWER={source['atomic_number']}",
                "unknown": "ANSWER=UNKNOWN",
            },
            "response_suffix": [
                "Return exactly ANSWER=<integer> or ANSWER=UNKNOWN.",
                "Do not explain.",
                "Answer:",
            ],
        }
    if interface_variant == "json_answer":
        return {
            "option_lines": [],
            "choice_to_label": {},
            "label_to_choice": {},
            "label_to_answer": {
                "local_number": json.dumps({"answer": int(source["local_number"])}, separators=(",", ":")),
                "atomic_number": json.dumps({"answer": int(source["atomic_number"])}, separators=(",", ":")),
                "unknown": json.dumps({"answer": "UNKNOWN"}, separators=(",", ":")),
            },
            "response_suffix": [
                'Return exactly one JSON object like {"answer": 12} or {"answer": "UNKNOWN"}.',
                "Do not explain.",
                "Answer:",
            ],
        }
    if interface_variant == "letter_choices":
        option_lines, choice_to_label, label_to_choice = choice_maps(source, option_order)
        return {
            "option_lines": option_lines,
            "choice_to_label": choice_to_label,
            "label_to_choice": label_to_choice,
            "label_to_answer": label_to_choice,
            "response_suffix": [
                "Return only one final token: A, B, or C.",
                "Do not explain.",
                "Answer:",
            ],
        }
    if interface_variant == "numeric_options":
        option_lines, label_to_answer = numeric_option_lines(source, option_order)
        return {
            "option_lines": option_lines,
            "choice_to_label": {},
            "label_to_choice": {},
            "label_to_answer": label_to_answer,
            "response_suffix": [
                "Return only one final token: one listed integer or UNKNOWN.",
                "Do not explain.",
                "Answer:",
            ],
        }
    raise ValueError(f"unknown interface variant {interface_variant}")


def make_prompt(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    panel: str,
    interface_variant: str,
    option_order: str,
    assignment: str,
    rule_order: str,
) -> tuple[str, str, str, dict[str, Any]]:
    rows = distractors(sources, source, 5)
    example_rows = [row for row in rows if row["source_id"] != source["source_id"]][:2]
    payload = interface_payload(source, interface_variant=interface_variant, option_order=option_order)

    if panel == "familiar_interface_lookup":
        lines = local_table_lines(rows, source, include_query=True)
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the local lab number.")
        lines.extend(payload["option_lines"])
        return "\n".join([*lines, *payload["response_suffix"]]), "local_number", "NONE", payload
    if panel == "atomic_interface_control":
        lines = [
            "No local lab table is active.",
            "Return the standard atomic number for the queried element.",
            f"Query element: {source['element']}",
        ]
        lines.extend(payload["option_lines"])
        return "\n".join([*lines, *payload["response_suffix"]]), "atomic_number", "NONE", payload
    if panel == "operation_local_interface_conflict":
        operation = operation_for_label("local_number", assignment)
        lines = local_table_lines(rows, source, include_query=True)
        append_rules_and_examples(lines, example_rows, assignment=assignment, rule_order=rule_order)
        lines.append(operation_query_line(operation, source, "compact_operation_rules"))
        lines.append("Apply the query operation and return the controlling answer.")
        lines.extend(payload["option_lines"])
        return "\n".join([*lines, *payload["response_suffix"]]), "local_number", operation, payload
    if panel == "operation_atomic_interface_conflict":
        operation = operation_for_label("atomic_number", assignment)
        lines = local_table_lines(rows, source, include_query=True)
        append_rules_and_examples(lines, example_rows, assignment=assignment, rule_order=rule_order)
        lines.append(operation_query_line(operation, source, "compact_operation_rules"))
        lines.append("Apply the query operation and return the controlling answer.")
        lines.extend(payload["option_lines"])
        return "\n".join([*lines, *payload["response_suffix"]]), "atomic_number", operation, payload
    if panel == "operation_rule_absent_interface_null":
        operation = operation_for_label("local_number", assignment)
        lines = local_table_lines(rows, source, include_query=True)
        lines.append("No operation rule or worked example is available for this query.")
        lines.append(operation_query_line(operation, source, "compact_operation_rules"))
        lines.append("If no operation rule is available, return UNKNOWN.")
        lines.extend(payload["option_lines"])
        return "\n".join([*lines, *payload["response_suffix"]]), "unknown", operation, payload
    if panel == "answer_absent_interface_null":
        operation = operation_for_label("local_number", assignment)
        lines = local_table_lines(rows, source, include_query=False)
        append_rules_and_examples(lines, example_rows, assignment=assignment, rule_order=rule_order)
        lines.append(operation_query_line(operation, source, "compact_operation_rules"))
        lines.append("If the queried element is absent from the table, return UNKNOWN before applying any operation.")
        lines.extend(payload["option_lines"])
        return "\n".join([*lines, *payload["response_suffix"]]), "unknown", operation, payload
    raise ValueError(f"unknown panel {panel}")


def candidate_answers_for(record: dict[str, Any]) -> list[dict[str, str]]:
    if record["interface_variant"] == "letter_choices":
        return [
            {
                "answer": choice,
                "label": record["choice_to_label"][choice],
                "candidate_type": record["choice_to_label"][choice],
            }
            for choice in CHOICES
        ]
    return [
        {
            "answer": record["label_to_answer"][label],
            "label": label,
            "candidate_type": label,
        }
        for label in ("local_number", "atomic_number", "unknown")
    ]


def source_records(
    limit_sources: int | None = None,
    configs: tuple[dict[str, str], ...] = INTERFACE_CONFIGS,
    assignments: tuple[str, ...] = OPERATION_ASSIGNMENTS,
    rule_orders: tuple[str, ...] = RULE_ORDERS,
    run_type: str = RUN_TYPE,
) -> list[dict[str, Any]]:
    sources = base_sources(limit_sources)
    records: list[dict[str, Any]] = []
    for source in sources:
        for config in configs:
            interface_variant = config["interface_variant"]
            option_order = config["option_order"]
            template = config["template"]
            for assignment in assignments:
                for rule_order in rule_orders:
                    for panel in PANELS:
                        prompt, expected_label, query_operation, payload = make_prompt(
                            sources,
                            source,
                            panel=panel,
                            interface_variant=interface_variant,
                            option_order=option_order,
                            assignment=assignment,
                            rule_order=rule_order,
                        )
                        first_option_label = (
                            option_order_labels(option_order)[0]
                            if option_order != NO_OPTION_ORDER
                            else "none"
                        )
                        expected_answer = payload["label_to_answer"][expected_label]
                        record = {
                            "id": (
                                f"{CARD_ID}_{template}_{assignment}_{rule_order}_{panel}_"
                                f"{source['source_id']}"
                            ),
                            "card_id": CARD_ID,
                            "run_type": run_type,
                            "model_id": MODEL_ID,
                            "template": template,
                            "interface_variant": interface_variant,
                            "option_order": option_order,
                            "first_option_label": first_option_label,
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
                            "expected_answer": expected_answer,
                            "label_to_answer": payload["label_to_answer"],
                            "choice_to_label": payload["choice_to_label"],
                            "label_to_choice": payload["label_to_choice"],
                            "prompt": prompt,
                        }
                        record["candidate_answers"] = candidate_answers_for(record)
                        records.append(record)
    return records


def prompt_audit(record: dict[str, Any]) -> dict[str, Any]:
    prompt = record["prompt"]
    primary = record["panel"] in PRIMARY_CONFLICT_PANELS
    null = record["panel"] == "answer_absent_interface_null"
    option_interface = record["interface_variant"] in {"letter_choices", "numeric_options"}
    atomic_occurrences = word_occurrences(prompt, record["atomic_number"])
    local_row_absent = not null or not line_contains_both(prompt, record["element"], record["local_number"])
    return {
        "atomic_number_occurrences": atomic_occurrences,
        "atomic_number_visible_in_conflict_options": primary and option_interface and atomic_occurrences >= 1,
        "atomic_number_hidden_in_no_option_conflict": (
            not primary or option_interface or atomic_occurrences == 0
        ),
        "answer_absent_omits_query_local_row": local_row_absent,
        "prompt_has_status_lexeme": bool(STATUS_LEXEME_RE.search(prompt)),
    }


def structural_check(
    records: list[dict[str, Any]],
    configs: tuple[dict[str, str], ...],
    assignments: tuple[str, ...],
    rule_orders: tuple[str, ...],
) -> dict[str, Any]:
    source_ids = {row["source_id"] for row in records}
    expected_count = len(source_ids) * len(configs) * len(assignments) * len(rule_orders) * len(PANELS)
    split_by_source: dict[str, set[str]] = {}
    for row in records:
        split_by_source.setdefault(row["source_id"], set()).add(row["split"])
    split_rows = Counter(row["split"] for row in records)
    conflict_rows = [row for row in records if row["panel"] in PRIMARY_CONFLICT_PANELS]
    expected_conflict = Counter(row["expected_label"] for row in conflict_rows)
    operation_conflict = Counter(row["query_operation"] for row in conflict_rows)
    template_counts = Counter(row["template"] for row in records)
    first_option_counts = Counter(
        row["first_option_label"]
        for row in records
        if row["first_option_label"] != "none"
    )
    audits = [prompt_audit(row) for row in records]
    status_rows = [row["id"] for row, audit in zip(records, audits, strict=True) if audit["prompt_has_status_lexeme"]]
    no_option_atomic_leaks = [
        row["id"]
        for row, audit in zip(records, audits, strict=True)
        if not audit["atomic_number_hidden_in_no_option_conflict"]
    ]
    option_conflict_missing_atomic = [
        row["id"]
        for row, audit in zip(records, audits, strict=True)
        if row["panel"] in PRIMARY_CONFLICT_PANELS
        and row["interface_variant"] in {"letter_choices", "numeric_options"}
        and not audit["atomic_number_visible_in_conflict_options"]
    ]
    null_leaks = [
        row["id"]
        for row, audit in zip(records, audits, strict=True)
        if not audit["answer_absent_omits_query_local_row"]
    ]
    candidate_collisions = [
        row["id"]
        for row in records
        if len({candidate["answer"] for candidate in row["candidate_answers"]}) != len(row["candidate_answers"])
    ]
    suffix_by_template: dict[str, set[str]] = {}
    for row in records:
        suffix_by_template.setdefault(row["template"], set()).add("\n".join(row["prompt"].splitlines()[-3:]))
    criteria = {
        "expected_row_count": len(records) == expected_count,
        "all_panels_present": set(PANELS) == {row["panel"] for row in records},
        "all_templates_present": set(TEMPLATES) == {row["template"] for row in records},
        "all_interface_variants_present": set(INTERFACE_VARIANTS) == {row["interface_variant"] for row in records},
        "all_assignments_present": set(assignments) == {row["operation_assignment"] for row in records},
        "all_rule_orders_present": set(rule_orders) == {row["rule_order"] for row in records},
        "source_split_disjoint": all(len(splits) == 1 for splits in split_by_source.values()),
        "holdout_sources_present": any(row["split"] == "holdout" for row in records),
        "calibration_sources_present": any(row["split"] == "calibration" for row in records),
        "option_conflicts_show_atomic_by_design": not option_conflict_missing_atomic,
        "no_option_conflicts_hide_target_atomic": not no_option_atomic_leaks,
        "answer_absent_omits_query_local_row": not null_leaks,
        "no_candidate_collisions": not candidate_collisions,
        "single_suffix_per_template": all(len(suffixes) == 1 for suffixes in suffix_by_template.values()),
        "prompts_have_no_status_lexemes": not status_rows,
        "conflict_expected_balanced": expected_conflict["local_number"] == expected_conflict["atomic_number"],
        "conflict_operations_balanced": operation_conflict["ALPHA"] == operation_conflict["BETA"],
        "option_first_labels_balanced": len(set(first_option_counts.values())) == 1,
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
        "no_option_atomic_leak_rows": no_option_atomic_leaks[:20],
        "option_conflict_missing_atomic_rows": option_conflict_missing_atomic[:20],
        "null_leak_rows": null_leaks[:20],
        "candidate_collision_rows": candidate_collisions[:20],
        "conflict_expected_counts": dict(sorted(expected_conflict.items())),
        "conflict_operation_counts": dict(sorted(operation_conflict.items())),
        "option_first_label_counts": dict(sorted(first_option_counts.items())),
    }


def parse_numeric_label(record: dict[str, Any], answer: str, parse_rule: str, first_line: str) -> dict[str, Any]:
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
        "selected_choice": None,
        "parseable": True,
        "parse_rule": parse_rule,
        "first_line": first_line,
    }


def strict_parse(record: dict[str, Any], generated_text: str) -> dict[str, Any]:
    stripped = generated_text.strip().lstrip("`'\" ")
    first_line = stripped.splitlines()[0].strip().strip("`") if stripped else ""
    variant = record["interface_variant"]
    if variant in {"bare_integer", "numeric_options"}:
        normalized = first_line.strip().strip("\"'")
        match = ANSWER_RE.fullmatch(normalized)
        if not match:
            return unparsed(first_line, "not_bare_integer_or_unknown")
        return parse_numeric_label(record, match.group(1), "strict_bare_integer_or_unknown", first_line)
    if variant == "answer_prefix":
        match = PREFIX_RE.fullmatch(first_line.strip().strip("\"'"))
        if not match:
            return unparsed(first_line, "not_answer_prefix")
        return parse_numeric_label(record, match.group(1).upper(), "strict_answer_prefix", first_line)
    if variant == "json_answer":
        try:
            parsed = json.loads(first_line)
        except json.JSONDecodeError:
            return unparsed(first_line, "not_json")
        if not isinstance(parsed, dict) or set(parsed) != {"answer"}:
            return unparsed(first_line, "json_missing_single_answer")
        value = parsed["answer"]
        if isinstance(value, int):
            answer = str(value)
        elif isinstance(value, str) and (value == "UNKNOWN" or value.isdigit()):
            answer = value
        else:
            return unparsed(first_line, "json_answer_not_integer_or_unknown")
        return parse_numeric_label(record, answer, "strict_json_answer", first_line)
    if variant == "letter_choices":
        match = CHOICE_RE.fullmatch(first_line.strip().strip("\"'"))
        if not match:
            return unparsed(first_line, "not_bare_choice")
        choice = match.group(1).upper()
        label = record["choice_to_label"][choice]
        return {
            "selected_label": label,
            "selected_answer": record["label_to_answer"][label],
            "selected_choice": choice,
            "parseable": True,
            "parse_rule": "strict_bare_choice",
            "first_line": first_line,
        }
    raise ValueError(f"unknown interface variant {variant}")


def unparsed(first_line: str, parse_rule: str) -> dict[str, Any]:
    return {
        "selected_label": "unparsed",
        "selected_answer": None,
        "selected_choice": None,
        "parseable": False,
        "parse_rule": parse_rule,
        "first_line": first_line,
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
        parsed = strict_parse(record, generated["generated_text"])
        output = {
            **record,
            "index": index,
            **generated,
            **parsed,
            **prompt_audit(record),
            "expected_correct": parsed["selected_label"] == record["expected_label"],
            "first_option_selected": parsed["selected_label"] == record["first_option_label"],
        }
        outputs.append(output)
        if verbose:
            print(
                f"[{index:04d}/{len(records):04d}] {record['id']} "
                f"iface={record['interface_variant']} panel={record['panel']} "
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
    first_option = sum(1 for row in rows if row.get("first_option_selected"))
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
        "first_option_selected": first_option,
        "first_option_selected_rate": rate(first_option, len(rows)),
        "expected_label_counts": dict(sorted(Counter(str(row.get("expected_label")) for row in rows).items())),
        "query_operation_counts": dict(sorted(Counter(str(row.get("query_operation")) for row in rows).items())),
    }


def grouped_summaries(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return {
        str(value): summarize_rows([row for row in rows if row[key] == value])
        for value in sorted({row[key] for row in rows})
    }


def config_summary(outputs: list[dict[str, Any]], templates: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for template in templates:
        rows = [row for row in outputs if row["template"] == template]
        panels = {panel: summarize_rows([row for row in rows if row["panel"] == panel]) for panel in PANELS}
        conflict_rows = [row for row in rows if row["panel"] in PRIMARY_CONFLICT_PANELS]
        result[template] = {
            "rows": len(rows),
            "interface_variant": rows[0]["interface_variant"] if rows else None,
            "option_order": rows[0]["option_order"] if rows else None,
            "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "selected_label_counts": label_counts(rows),
            "panels": panels,
            "primary_conflict": summarize_rows(conflict_rows),
            "primary_conflict_by_operation": grouped_summaries(conflict_rows, "query_operation"),
            "primary_conflict_by_assignment": grouped_summaries(conflict_rows, "operation_assignment"),
            "primary_conflict_by_rule_order": grouped_summaries(conflict_rows, "rule_order"),
        }
    return result


def variant_panel_blocks(outputs: list[dict[str, Any]], panel: str) -> dict[str, Any]:
    return {
        variant: summarize_rows(
            [row for row in outputs if row["interface_variant"] == variant and row["panel"] == panel]
        )
        for variant in INTERFACE_VARIANTS
    }


def variant_overall_blocks(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        variant: summarize_rows([row for row in outputs if row["interface_variant"] == variant])
        for variant in INTERFACE_VARIANTS
    }


def selection_key(item: dict[str, Any], template: str, templates: tuple[str, ...]) -> tuple[float, float, float, float, int]:
    panels = item["panels"]
    control_floor = min(
        float(panels["familiar_interface_lookup"]["local_number_rate"]),
        float(panels["atomic_interface_control"]["atomic_number_rate"]),
        float(panels["answer_absent_interface_null"]["unknown_rate"]),
        float(panels["operation_rule_absent_interface_null"]["unknown_rate"]),
    )
    conflict_floor = min(
        float(panels["operation_local_interface_conflict"]["local_number_rate"]),
        float(panels["operation_atomic_interface_conflict"]["atomic_number_rate"]),
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
            "earliest interface config",
        ],
    }


def interface_gate(panel_blocks: dict[str, Any]) -> bool:
    return (
        float(panel_blocks["familiar_interface_lookup"]["local_number_rate"]) >= 0.90
        and float(panel_blocks["atomic_interface_control"]["atomic_number_rate"]) >= 0.85
        and float(panel_blocks["answer_absent_interface_null"]["unknown_rate"]) >= 0.90
        and float(panel_blocks["operation_rule_absent_interface_null"]["unknown_rate"]) >= 0.90
        and float(panel_blocks["operation_local_interface_conflict"]["local_number_rate"]) >= 0.85
        and float(panel_blocks["operation_atomic_interface_conflict"]["atomic_number_rate"]) >= 0.85
        and min(float(panel["parseable_rate"]) for panel in panel_blocks.values()) >= 0.95
    )


def observed_pattern(criteria: dict[str, Any], selected: dict[str, Any]) -> str:
    if criteria["any_interface_gate_passed"]:
        return "answer_interface_substrate_candidate"
    if criteria["interface_atomic_control_range_at_least_50p"] or criteria["interface_null_range_at_least_50p"]:
        return "answer_interface_strongly_changes_controls"
    if criteria["interface_operation_atomic_range_at_least_30p"]:
        return "answer_interface_changes_learned_branch"
    panels = selected["panels"]
    if float(panels["atomic_interface_control"]["atomic_number_rate"]) < 0.85:
        return "best_interface_atomic_control_failed"
    if float(panels["operation_atomic_interface_conflict"]["atomic_number_rate"]) < 0.85:
        return "best_interface_atomic_branch_failed"
    return "answer_interface_sweep_failed_without_dispersion"


def summarize(
    records: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    configs: tuple[dict[str, str], ...],
    assignments: tuple[str, ...],
    rule_orders: tuple[str, ...],
    full_run: bool,
) -> dict[str, Any]:
    templates = tuple(config["template"] for config in configs)
    structural = structural_check(records, configs, assignments, rule_orders)
    by_template = config_summary(outputs, templates)
    selection = select_template(by_template, templates)
    selected = dict(by_template[selection["selected_template"]])
    selected_rows = [row for row in outputs if row["template"] == selection["selected_template"]]

    interface_panels = {
        variant: {
            panel: summarize_rows(
                [row for row in outputs if row["interface_variant"] == variant and row["panel"] == panel]
            )
            for panel in PANELS
        }
        for variant in INTERFACE_VARIANTS
    }
    selected["interface_variant_overall"] = variant_overall_blocks(outputs)
    selected["interface_variant_familiar_lookup"] = variant_panel_blocks(outputs, "familiar_interface_lookup")
    selected["interface_variant_atomic_control"] = variant_panel_blocks(outputs, "atomic_interface_control")
    selected["interface_variant_operation_local"] = variant_panel_blocks(outputs, "operation_local_interface_conflict")
    selected["interface_variant_operation_atomic"] = variant_panel_blocks(outputs, "operation_atomic_interface_conflict")
    selected["interface_variant_answer_absent"] = variant_panel_blocks(outputs, "answer_absent_interface_null")
    selected["interface_variant_rule_absent"] = variant_panel_blocks(outputs, "operation_rule_absent_interface_null")

    variant_gates = {
        variant: interface_gate(interface_panels[variant])
        for variant in INTERFACE_VARIANTS
    }
    atomic_rates = [
        float(selected["interface_variant_atomic_control"][variant]["atomic_number_rate"])
        for variant in INTERFACE_VARIANTS
    ]
    null_rates = [
        float(selected["interface_variant_answer_absent"][variant]["unknown_rate"])
        for variant in INTERFACE_VARIANTS
    ]
    operation_atomic_rates = [
        float(selected["interface_variant_operation_atomic"][variant]["atomic_number_rate"])
        for variant in INTERFACE_VARIANTS
    ]
    selected_prompt_audit_passed = all(
        row["answer_absent_omits_query_local_row"]
        and (row["interface_variant"] in {"letter_choices", "numeric_options"} or row["atomic_number_hidden_in_no_option_conflict"])
        for row in selected_rows
    )
    prompts_have_no_status_lexemes = all(not row["prompt_has_status_lexeme"] for row in outputs)
    criteria = {
        "smoke_mode": not full_run,
        "structural_passed": structural["passed"],
        "full_source_count_is_40": structural["source_count"] == 40,
        "source_disjoint_holdout": structural["criteria"]["source_split_disjoint"]
        and structural["criteria"]["holdout_sources_present"]
        and structural["criteria"]["calibration_sources_present"],
        "selected_prompt_audit_passed": selected_prompt_audit_passed,
        "prompts_have_no_status_lexemes": prompts_have_no_status_lexemes,
        "any_interface_gate_passed": any(variant_gates.values()),
        "interface_gate_by_variant": variant_gates,
        "interface_atomic_control_range": max(atomic_rates) - min(atomic_rates),
        "interface_null_range": max(null_rates) - min(null_rates),
        "interface_operation_atomic_range": max(operation_atomic_rates) - min(operation_atomic_rates),
        "interface_atomic_control_range_at_least_50p": max(atomic_rates) - min(atomic_rates) >= 0.50,
        "interface_null_range_at_least_50p": max(null_rates) - min(null_rates) >= 0.50,
        "interface_operation_atomic_range_at_least_30p": max(operation_atomic_rates) - min(operation_atomic_rates) >= 0.30,
        "candidate_and_output_margins_reported": False,
    }
    selected_panels = selected["panels"]
    criteria.update(
        {
            "selected_familiar_lookup_at_least_90p": float(selected_panels["familiar_interface_lookup"]["local_number_rate"]) >= 0.90,
            "selected_atomic_control_at_least_85p": float(selected_panels["atomic_interface_control"]["atomic_number_rate"]) >= 0.85,
            "selected_answer_absent_unknown_at_least_90p": float(selected_panels["answer_absent_interface_null"]["unknown_rate"]) >= 0.90,
            "selected_rule_absent_unknown_at_least_90p": float(selected_panels["operation_rule_absent_interface_null"]["unknown_rate"]) >= 0.90,
            "selected_operation_local_at_least_85p": float(selected_panels["operation_local_interface_conflict"]["local_number_rate"]) >= 0.85,
            "selected_operation_atomic_at_least_85p": float(selected_panels["operation_atomic_interface_conflict"]["atomic_number_rate"]) >= 0.85,
            "selected_all_panels_parseable_at_least_95p": min(float(panel["parseable_rate"]) for panel in selected_panels.values()) >= 0.95,
        }
    )
    criteria["all_controls_passed"] = (
        criteria["selected_familiar_lookup_at_least_90p"]
        and criteria["selected_atomic_control_at_least_85p"]
        and criteria["selected_answer_absent_unknown_at_least_90p"]
        and criteria["selected_rule_absent_unknown_at_least_90p"]
        and criteria["selected_all_panels_parseable_at_least_95p"]
    )
    pattern = observed_pattern(criteria, selected)
    behavior_gate_passed = (
        not criteria["smoke_mode"]
        and criteria["any_interface_gate_passed"]
        and criteria["candidate_and_output_margins_reported"]
    )
    diagnostic_class = "smoke_only" if criteria["smoke_mode"] else pattern
    return {
        "structural": structural,
        "by_template": by_template,
        "by_interface_variant": {
            variant: {
                "rows": sum(panel["rows"] for panel in interface_panels[variant].values()),
                "panels": interface_panels[variant],
                "gate_passed": variant_gates[variant],
            }
            for variant in INTERFACE_VARIANTS
        },
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
    selected = summary["selected_template_summary"]
    payload = {
        "diagnostic_class": summary["diagnostic_class"],
        "observed_failure_pattern": summary["observed_failure_pattern"],
        "passed": summary["passed"],
        "behavior_ready": summary["behavior_ready"],
        "signature_ready": summary["signature_ready"],
        "criteria": summary["criteria"],
        "selection": summary["selection"],
        "selected_controls": {
            panel: selected["panels"][panel]
            for panel in (
                "familiar_interface_lookup",
                "atomic_interface_control",
                "answer_absent_interface_null",
                "operation_rule_absent_interface_null",
            )
        },
        "selected_conflicts": {
            panel: selected["panels"][panel]
            for panel in PRIMARY_CONFLICT_PANELS
        },
        "interface_atomic_control": selected["interface_variant_atomic_control"],
        "interface_answer_absent": selected["interface_variant_answer_absent"],
        "interface_operation_atomic": selected["interface_variant_operation_atomic"],
    }
    if output_path is not None:
        payload["output_path"] = str(output_path)
    return payload


def write_behavior_status_card(path: Path, result: dict[str, Any], output_path: Path) -> None:
    summary = result["summary"]
    selected = summary["selected_template_summary"]
    criteria = summary["criteria"]
    lines = [
        "# MC027 Answer-Interface Sweep Behavior Status",
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
        "  `code/mc027_answer_interface_sweep.py`",
        "- result:",
        f"  `{output_path.as_posix()}`",
        "",
        "## Verdict",
        "",
        "MC027 is an answer-interface diagnostic. It measures whether output",
        "format changes the bridge behavior before any hidden-state probe is",
        "allowed.",
        "",
        "## Selected Interface Config",
        "",
        f"- selected config: `{summary['selection']['selected_template']}`",
        f"- selection key: `{json.dumps(summary['selection']['selection_key'])}`",
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
            "## Interface Comparison",
            "",
            "| Interface | Atomic Control Atomic | Answer-Absent UNKNOWN | Operation-Local Local | Operation-Atomic Atomic | Gate Passed |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for variant in INTERFACE_VARIANTS:
        atomic = selected["interface_variant_atomic_control"][variant]
        absent = selected["interface_variant_answer_absent"][variant]
        local = selected["interface_variant_operation_local"][variant]
        op_atomic = selected["interface_variant_operation_atomic"][variant]
        gate = summary["criteria"]["interface_gate_by_variant"][variant]
        lines.append(
            f"| `{variant}` | {atomic['atomic_number_rate']:.3f} | "
            f"{absent['unknown_rate']:.3f} | {local['local_number_rate']:.3f} | "
            f"{op_atomic['atomic_number_rate']:.3f} | `{str(gate).lower()}` |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "MC027 does not establish a signature, intervention, or mechanism card.",
            "It is a behavior-substrate and interface-law diagnostic.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


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
    parser.add_argument("--artifact-prefix", default="mc027_answer_interface_sweep_structural")
    parser.add_argument("--status-card", type=Path, default=STATUS_CARD)
    parser.add_argument("--write-status-card", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.time()
    run_type = BEHAVIOR_RUN_TYPE if args.score_model else RUN_TYPE
    records = source_records(args.limit_sources, INTERFACE_CONFIGS, OPERATION_ASSIGNMENTS, RULE_ORDERS, run_type)
    structural = structural_check(records, INTERFACE_CONFIGS, OPERATION_ASSIGNMENTS, RULE_ORDERS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.write_manifest and not args.score_model:
        result = {
            "schema_version": 1,
            "card_id": CARD_ID,
            "run_type": RUN_TYPE,
            "model_id": args.model_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "purpose": "Structural gate for answer-interface sweep over query-operation arbitration.",
            "templates": list(TEMPLATES),
            "interface_variants": list(INTERFACE_VARIANTS),
            "interface_configs": list(INTERFACE_CONFIGS),
            "operation_assignments": list(OPERATION_ASSIGNMENTS),
            "rule_orders": list(RULE_ORDERS),
            "option_orders": list(OPTION_ORDERS),
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
    summary = summarize(
        records,
        outputs,
        INTERFACE_CONFIGS,
        OPERATION_ASSIGNMENTS,
        RULE_ORDERS,
        full_run=full_run,
    )
    prefix = args.artifact_prefix
    if prefix == "mc027_answer_interface_sweep_structural":
        prefix = "mc027_answer_interface_sweep_behavior"
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
        "interface_variants": list(INTERFACE_VARIANTS),
        "interface_configs": list(INTERFACE_CONFIGS),
        "operation_assignments": list(OPERATION_ASSIGNMENTS),
        "rule_orders": list(RULE_ORDERS),
        "option_orders": list(OPTION_ORDERS),
        "operation_codes": list(OPERATION_CODES),
        "elapsed_s": time.time() - started,
        "purpose": "Answer-interface sweep over query-operation local-versus-learned arbitration.",
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
