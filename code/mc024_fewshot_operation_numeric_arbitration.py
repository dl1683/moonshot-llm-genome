#!/usr/bin/env python
"""MC024 few-shot query-operation numeric arbitration.

MC023 showed that query-level operation handles can preserve direct controls
and nulls while both local and atomic conflict branches remain below gate. This
runner tests the narrow next explanation: maybe the operation handles need
worked examples rather than only a rule statement.

The prompt still has no row-level source-status labels. It adds balanced worked
examples for the operation mapping, then asks for an integer or UNKNOWN. This
is a behavior diagnostic only; a pass would be a substrate for later hidden-
state work, not a mechanism claim.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from mc012_reliability_labeled_numeric_arbitration import MODEL_ID, base_sources, distractors
from mc016_alphabet_gated_numeric_arbitration import load_model_and_tokenizer
from mc023_query_operation_numeric_arbitration import (
    OPERATION_ASSIGNMENTS,
    OPERATION_CODES,
    PANELS,
    PRIMARY_CONFLICT_PANELS,
    RULE_ORDERS,
    append_operation_rules,
    candidate_answers_for,
    operation_for_label,
    operation_mapping,
    operation_query_line,
    quiet_summary,
    response_suffix,
    score_records,
    structural_check,
    summarize,
)


CARD_ID = "MC024"
RUN_TYPE = "fewshot_operation_numeric_arbitration_structural_gate"
BEHAVIOR_RUN_TYPE = "fewshot_operation_numeric_arbitration_behavior"
RESULT_DIR = Path("results/cards/MC024")
STATUS_CARD = Path("research/cards/MC024_FEWSHOT_OPERATION_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md")
TEMPLATES = ("worked_examples_before_rules", "worked_examples_after_rules", "compact_worked_examples")


def local_table_lines(
    rows: list[dict[str, Any]],
    query: dict[str, Any],
    *,
    include_query: bool,
    template: str,
    key_field: str = "element",
) -> list[str]:
    visible_rows = rows if include_query else [row for row in rows if row["source_id"] != query["source_id"]]
    if template == "worked_examples_before_rules":
        lines = [
            "A local lab table may define temporary lab numbers.",
            "Standard chemistry may define atomic numbers.",
            "Element lab-number table:",
        ]
    elif template == "worked_examples_after_rules":
        lines = [
            "Use the operation rules and worked examples to choose one number.",
            "Element lab-number table:",
        ]
    elif template == "compact_worked_examples":
        lines = [
            "Rows:",
        ]
    else:
        raise ValueError(f"unknown template {template}")
    for row in visible_rows:
        lines.append(f"{row[key_field]} has local lab number {row['local_number']}.")
    return lines


def operation_example_lines(
    example_rows: list[dict[str, Any]],
    *,
    assignment: str,
) -> list[str]:
    mapping = operation_mapping(assignment)
    local_operation = operation_for_label("local_number", assignment)
    atomic_operation = operation_for_label("atomic_number", assignment)
    local_row = example_rows[0]
    atomic_row = example_rows[1]
    lines = ["Worked examples:"]
    for operation, row in ((local_operation, local_row), (atomic_operation, atomic_row)):
        target = mapping[operation]
        if target == "local_number":
            answer = row["local_number"]
        elif target == "atomic_number":
            answer = str(row["atomic_number"])
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
    template: str,
) -> None:
    if template == "worked_examples_before_rules":
        lines.extend(operation_example_lines(example_rows, assignment=assignment))
        append_operation_rules(lines, assignment=assignment, rule_order=rule_order, template=template)
        return
    if template == "worked_examples_after_rules":
        append_operation_rules(lines, assignment=assignment, rule_order=rule_order, template=template)
        lines.extend(operation_example_lines(example_rows, assignment=assignment))
        return
    if template == "compact_worked_examples":
        append_operation_rules(lines, assignment=assignment, rule_order=rule_order, template=template)
        for item in operation_example_lines(example_rows, assignment=assignment)[1:]:
            lines.append(item)
        return
    raise ValueError(f"unknown template {template}")


def make_prompt(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    panel: str,
    template: str,
    assignment: str,
    rule_order: str,
) -> tuple[str, str, str]:
    rows = distractors(sources, source, 5)
    example_rows = [row for row in rows if row["source_id"] != source["source_id"]][:2]
    if len(example_rows) != 2:
        raise ValueError("not enough example rows")
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
        append_rules_and_examples(lines, example_rows, assignment=assignment, rule_order=rule_order, template=template)
        lines.append(operation_query_line(operation, source, template))
        lines.append("Apply the query operation and return the controlling number.")
        return "\n".join([*lines, *response_suffix()]), "local_number", operation
    if panel == "operation_atomic_conflict":
        operation = operation_for_label("atomic_number", assignment)
        lines = local_table_lines(rows, source, include_query=True, template=template)
        append_rules_and_examples(lines, example_rows, assignment=assignment, rule_order=rule_order, template=template)
        lines.append(operation_query_line(operation, source, template))
        lines.append("Apply the query operation and return the controlling number.")
        return "\n".join([*lines, *response_suffix()]), "atomic_number", operation
    if panel == "operation_rule_absent_conflict":
        operation = operation_for_label("local_number", assignment)
        lines = local_table_lines(rows, source, include_query=True, template=template)
        lines.append("No operation rule or worked example is available for this query.")
        lines.append(operation_query_line(operation, source, template))
        lines.append("If no operation rule is available, return UNKNOWN.")
        return "\n".join([*lines, *response_suffix()]), "unknown", operation
    if panel == "answer_absent_null":
        operation = operation_for_label("local_number", assignment)
        lines = local_table_lines(rows, source, include_query=False, template=template)
        append_rules_and_examples(lines, example_rows, assignment=assignment, rule_order=rule_order, template=template)
        lines.append(operation_query_line(operation, source, template))
        lines.append("If the queried element is absent from the table, return UNKNOWN before applying any operation.")
        return "\n".join([*lines, *response_suffix()]), "unknown", operation
    raise ValueError(f"unknown panel {panel}")


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


def write_behavior_status_card(path: Path, result: dict[str, Any], output_path: Path) -> None:
    summary = result["summary"]
    selected = summary["selected_template_summary"]
    criteria = summary["criteria"]
    lines = [
        "# MC024 Few-Shot Operation Numeric Arbitration Behavior Status",
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
        "  `code/mc024_fewshot_operation_numeric_arbitration.py`",
        "- result:",
        f"  `{output_path.as_posix()}`",
        "",
        "## Verdict",
        "",
    ]
    if summary["behavior_gate_passed"]:
        lines.extend(
            [
                "The few-shot operation behavior gate passed.",
                "This would justify a separate bridge decision, but it is not",
                "itself a hidden-state mechanism claim.",
            ]
        )
    elif criteria["smoke_mode"]:
        lines.extend(
            [
                "This is a smoke or partial run. It diagnoses whether worked",
                "examples repair query-level operation arbitration without",
                "row-level source-status text.",
            ]
        )
    else:
        lines.extend(
            [
                "The few-shot operation behavior gate failed.",
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
            "MC024 is a behavior diagnostic for worked-example query operation",
            "handles over prompt-local versus learned atomic-number branches. It",
            "does not establish an internal signature, intervention, or mechanism",
            "card.",
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
    parser.add_argument("--artifact-prefix", default="mc024_fewshot_operation_numeric_arbitration_structural")
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
            "purpose": "Structural gate for few-shot query-operation numeric arbitration.",
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
    if prefix == "mc024_fewshot_operation_numeric_arbitration_structural":
        prefix = "mc024_fewshot_operation_numeric_arbitration_behavior"
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
        "purpose": "Generated-answer few-shot query-operation numeric arbitration diagnostic before hidden-state work.",
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
