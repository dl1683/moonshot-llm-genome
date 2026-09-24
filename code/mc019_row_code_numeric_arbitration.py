#!/usr/bin/env python
"""MC019 row-code numeric arbitration behavior gate.

MC018 showed that asking the model to choose a source label can be dominated by
answer-token, source-definition-order, answer-option-order, and local-source
salience biases. MC019 removes the source-label answer interface. It keeps the
answer as an integer or UNKNOWN, but attaches a neutral route code to each local
table row. The prompt tells the model how the queried row's route code maps to
local lab numbers versus standard atomic numbers.

This is still a behavior substrate test. A pass would show that a row-local,
non-status operational code can create local-versus-learned contrast under
order controls. It would not be a mechanism claim.
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


CARD_ID = "MC019"
RUN_TYPE = "row_code_numeric_arbitration_structural_gate"
BEHAVIOR_RUN_TYPE = "row_code_numeric_arbitration_behavior"
RESULT_DIR = Path("results/cards/MC019")
STATUS_CARD = Path("research/cards/MC019_ROW_CODE_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md")

PANELS = (
    "synthetic_numeric_lookup",
    "familiar_entity_numeric_lookup",
    "real_world_atomic_number_control",
    "route_p_local_conflict",
    "route_p_atomic_conflict",
    "route_rule_absent_conflict",
    "answer_absent_null",
)
PRIMARY_CONFLICT_PANELS = ("route_p_local_conflict", "route_p_atomic_conflict")
CONFLICT_PANELS = PRIMARY_CONFLICT_PANELS + ("route_rule_absent_conflict",)
TEMPLATES = ("route_column", "compact_route_column", "query_row_repeated")
DEFINITION_ORDERS = ("p_definition_first", "q_definition_first")
ROUTE_CODES = ("P", "Q")
STATUS_LEXEME_RE = re.compile(r"\b(trusted|untrusted|reliable|unreliable|status)\b", re.IGNORECASE)


def route_code(source: dict[str, Any]) -> str:
    return "P" if source["source_index"] % 2 == 0 else "Q"


def response_suffix() -> list[str]:
    return [
        "Return only one final token: an integer or UNKNOWN.",
        "Do not explain.",
        "Answer:",
    ]


def route_table(
    rows: list[dict[str, Any]],
    query: dict[str, Any],
    *,
    include_query: bool,
    template: str,
) -> list[str]:
    visible_rows = rows if include_query else [row for row in rows if row["source_id"] != query["source_id"]]
    if template == "route_column":
        lines = [
            "A local lab table may define temporary lab numbers.",
            "Standard chemistry may define atomic numbers.",
            "Element lab-number route table:",
        ]
        for row in visible_rows:
            lines.append(
                f"{row['element']} has local lab number {row['local_number']} and route code {route_code(row)}."
            )
        return lines
    if template == "compact_route_column":
        lines = [
            "Two numeric sources may conflict.",
            "Use the queried row's route code before choosing a number.",
            "Rows:",
        ]
        for row in visible_rows:
            lines.append(f"{row['element']} | local {row['local_number']} | route {route_code(row)}")
        return lines
    if template == "query_row_repeated":
        lines = [
            "A local lab table may define temporary lab numbers.",
            "Standard chemistry may define atomic numbers.",
            "Use the queried row repeated below when applying the route rule.",
            "Rows:",
        ]
        for row in visible_rows:
            lines.append(f"{row['element']} | local lab number {row['local_number']} | route code {route_code(row)}")
        return lines
    raise ValueError(f"unknown template {template}")


def basic_local_table(
    rows: list[dict[str, Any]],
    query: dict[str, Any],
    *,
    key_field: str,
    include_query: bool,
    template: str,
) -> list[str]:
    visible_rows = rows if include_query else [row for row in rows if row["source_id"] != query["source_id"]]
    if template == "route_column":
        lines = [
            "A local lab table may define temporary lab numbers.",
            "Standard chemistry may define atomic numbers.",
        ]
    elif template in {"compact_route_column", "query_row_repeated"}:
        lines = [
            "Two numeric sources may conflict.",
            "Use only the requested lookup rule.",
        ]
    else:
        raise ValueError(f"unknown template {template}")
    label = "Synthetic-key lab-number table:" if key_field == "synthetic_key" else "Element lab-number table:"
    lines.append(label)
    for row in visible_rows:
        lines.append(f"{row[key_field]} has local lab number {row['local_number']}.")
    return lines


def append_route_rule(lines: list[str], *, local_route: str, definition_order: str, template: str) -> None:
    if local_route not in ROUTE_CODES:
        raise ValueError(f"unknown local route {local_route}")
    standard_route = "Q" if local_route == "P" else "P"
    if definition_order == "p_definition_first":
        ordered_routes = ("P", "Q")
    elif definition_order == "q_definition_first":
        ordered_routes = ("Q", "P")
    else:
        raise ValueError(f"unknown definition order {definition_order}")

    if template == "route_column":
        lines.append("First find the queried element's route code in the table.")
        for code in ordered_routes:
            if code == local_route:
                lines.append(f"If the queried row has route code {code}, use its local lab number.")
            elif code == standard_route:
                lines.append(f"If the queried row has route code {code}, use the standard atomic number.")
            else:
                raise ValueError(f"unknown route code {code}")
        return

    if template == "compact_route_column":
        for code in ordered_routes:
            if code == local_route:
                lines.append(f"Route {code}: local lab number.")
            elif code == standard_route:
                lines.append(f"Route {code}: standard atomic number.")
            else:
                raise ValueError(f"unknown route code {code}")
        return

    if template == "query_row_repeated":
        lines.append("Apply this route rule to the repeated queried row:")
        for code in ordered_routes:
            if code == local_route:
                lines.append(f"Route code {code} means return the repeated row's local lab number.")
            elif code == standard_route:
                lines.append(f"Route code {code} means return the queried element's standard atomic number.")
            else:
                raise ValueError(f"unknown route code {code}")
        return

    raise ValueError(f"unknown template {template}")


def expected_label_for_route(source: dict[str, Any], *, local_route: str) -> str:
    return "local_number" if route_code(source) == local_route else "atomic_number"


def append_repeated_query_row(lines: list[str], source: dict[str, Any], *, template: str) -> None:
    if template == "query_row_repeated":
        lines.append(
            f"Repeated queried row: {source['element']} | local lab number {source['local_number']} | route code {route_code(source)}"
        )


def make_prompt(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    panel: str,
    template: str,
    definition_order: str,
) -> tuple[str, str]:
    rows = distractors(sources, source, 4)
    if panel == "synthetic_numeric_lookup":
        lines = basic_local_table(rows, source, key_field="synthetic_key", include_query=True, template=template)
        lines.append(f"Query key: {source['synthetic_key']}")
        lines.append("Return the local lab number.")
        return "\n".join([*lines, *response_suffix()]), "local_number"

    if panel == "familiar_entity_numeric_lookup":
        lines = basic_local_table(rows, source, key_field="element", include_query=True, template=template)
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the local lab number.")
        return "\n".join([*lines, *response_suffix()]), "local_number"

    if panel == "real_world_atomic_number_control":
        lines = [
            "No local lab table is active.",
            "Return the standard atomic number for the queried element.",
            f"Query element: {source['element']}",
        ]
        return "\n".join([*lines, *response_suffix()]), "atomic_number"

    if panel == "route_p_local_conflict":
        lines = route_table(rows, source, include_query=True, template=template)
        append_route_rule(lines, local_route="P", definition_order=definition_order, template=template)
        append_repeated_query_row(lines, source, template=template)
        lines.append(f"Query element: {source['element']}")
        lines.append("Apply the queried row's route code and return the controlling number.")
        return "\n".join([*lines, *response_suffix()]), expected_label_for_route(source, local_route="P")

    if panel == "route_p_atomic_conflict":
        lines = route_table(rows, source, include_query=True, template=template)
        append_route_rule(lines, local_route="Q", definition_order=definition_order, template=template)
        append_repeated_query_row(lines, source, template=template)
        lines.append(f"Query element: {source['element']}")
        lines.append("Apply the queried row's route code and return the controlling number.")
        return "\n".join([*lines, *response_suffix()]), expected_label_for_route(source, local_route="Q")

    if panel == "route_rule_absent_conflict":
        lines = route_table(rows, source, include_query=True, template=template)
        append_repeated_query_row(lines, source, template=template)
        lines.append("No route-code rule is available for this query.")
        lines.append("If no route-code rule is available, return UNKNOWN.")
        lines.append(f"Query element: {source['element']}")
        return "\n".join([*lines, *response_suffix()]), "unknown"

    if panel == "answer_absent_null":
        lines = route_table(rows, source, include_query=False, template=template)
        append_route_rule(lines, local_route="P", definition_order=definition_order, template=template)
        lines.append("If the queried element is absent from the table, return UNKNOWN before applying any route rule.")
        lines.append(f"Query element: {source['element']}")
        return "\n".join([*lines, *response_suffix()]), "unknown"

    raise ValueError(f"unknown panel {panel}")


def candidate_answers_for(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"answer": record["local_number"], "label": "local_number", "candidate_type": "prompt_local_lab_number"},
        {"answer": record["atomic_number"], "label": "atomic_number", "candidate_type": "real_world_atomic_number"},
        {"answer": record["lure_atomic_number"], "label": "lure_atomic_number", "candidate_type": "nearby_atomic_number_lure"},
        {"answer": "UNKNOWN", "label": "unknown", "candidate_type": "unknown"},
    ]


def source_records(
    limit_sources: int | None = None,
    templates: tuple[str, ...] = TEMPLATES,
    definition_orders: tuple[str, ...] = DEFINITION_ORDERS,
    run_type: str = RUN_TYPE,
) -> list[dict[str, Any]]:
    sources = base_sources(limit_sources)
    records: list[dict[str, Any]] = []
    for source in sources:
        for template in templates:
            for definition_order in definition_orders:
                for panel in PANELS:
                    prompt, expected_label = make_prompt(
                        sources,
                        source,
                        panel=panel,
                        template=template,
                        definition_order=definition_order,
                    )
                    record = {
                        "id": f"{CARD_ID}_{template}_{definition_order}_{panel}_{source['source_id']}",
                        "card_id": CARD_ID,
                        "run_type": run_type,
                        "model_id": MODEL_ID,
                        "template": template,
                        "definition_order": definition_order,
                        "panel": panel,
                        "split": source["split"],
                        "source_id": source["source_id"],
                        "source_index": source["source_index"],
                        "element": source["element"],
                        "synthetic_key": source["synthetic_key"],
                        "route_code": route_code(source),
                        "atomic_number": str(source["atomic_number"]),
                        "lure_atomic_number": str(source["lure_atomic_number"]),
                        "local_number": str(source["local_number"]),
                        "expected_label": expected_label,
                        "expected_local_answer": str(source["local_number"]),
                        "expected_real_answer": str(source["atomic_number"]),
                        "expected_null_answer": "UNKNOWN",
                        "prompt": prompt,
                    }
                    record["candidate_answers"] = candidate_answers_for(record)
                    records.append(record)
    return records


def prompt_audit(record: dict[str, Any]) -> dict[str, Any]:
    prompt = record["prompt"]
    conflict = record["panel"] in CONFLICT_PANELS
    null = record["panel"] == "answer_absent_null"
    primary = record["panel"] in PRIMARY_CONFLICT_PANELS
    return {
        "real_atomic_number_occurrences": word_occurrences(prompt, record["atomic_number"]),
        "lure_atomic_number_occurrences": word_occurrences(prompt, record["lure_atomic_number"]),
        "real_atomic_number_hidden_in_conflict": not conflict or word_occurrences(prompt, record["atomic_number"]) == 0,
        "lure_atomic_number_hidden_in_conflict": not conflict or word_occurrences(prompt, record["lure_atomic_number"]) == 0,
        "answer_absent_omits_query_local_number": not null or not line_contains_both(prompt, record["element"], record["local_number"]),
        "primary_prompt_has_status_lexeme": bool(primary and STATUS_LEXEME_RE.search(prompt)),
    }


def structural_check(records: list[dict[str, Any]], templates: tuple[str, ...]) -> dict[str, Any]:
    source_ids = {row["source_id"] for row in records}
    expected_count = len(source_ids) * len(templates) * len(DEFINITION_ORDERS) * len(PANELS)
    splits_by_source: dict[str, set[str]] = {}
    for row in records:
        splits_by_source.setdefault(row["source_id"], set()).add(row["split"])
    split_rows = Counter(row["split"] for row in records)
    primary = [row for row in records if row["panel"] in PRIMARY_CONFLICT_PANELS]
    expected_labels = Counter(row["expected_label"] for row in primary)
    route_labels = Counter(row["route_code"] for row in primary)
    expected_by_split: dict[str, Counter[str]] = {}
    route_by_split: dict[str, Counter[str]] = {}
    for row in primary:
        expected_by_split.setdefault(row["split"], Counter())[row["expected_label"]] += 1
        route_by_split.setdefault(row["split"], Counter())[row["route_code"]] += 1
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
    status_rows = [
        row["id"]
        for row, audit in zip(records, audits, strict=True)
        if audit["primary_prompt_has_status_lexeme"]
    ]
    criteria = {
        "expected_row_count": len(records) == expected_count,
        "all_panels_present": set(PANELS) == {row["panel"] for row in records},
        "all_templates_present": set(templates) == {row["template"] for row in records},
        "all_definition_orders_present": set(DEFINITION_ORDERS) == {row["definition_order"] for row in records},
        "source_split_disjoint": all(len(splits) == 1 for splits in splits_by_source.values()),
        "holdout_sources_present": any(row["split"] == "holdout" for row in records),
        "calibration_sources_present": any(row["split"] == "calibration" for row in records),
        "real_atomic_number_hidden_in_conflicts": all(audit["real_atomic_number_hidden_in_conflict"] for audit in audits),
        "lure_atomic_number_hidden_in_conflicts": all(audit["lure_atomic_number_hidden_in_conflict"] for audit in audits),
        "answer_absent_omits_query_local_number": all(audit["answer_absent_omits_query_local_number"] for audit in audits),
        "candidate_answers_parseable": not malformed_candidates,
        "no_candidate_collisions": not collisions,
        "single_answer_suffix": len(suffixes) == 1,
        "primary_prompts_have_no_status_lexemes": not status_rows,
        "primary_expected_labels_balanced": expected_labels["local_number"] == expected_labels["atomic_number"],
        "primary_route_labels_balanced": route_labels["P"] == route_labels["Q"],
        "primary_expected_labels_balanced_by_split": all(
            counts["local_number"] == counts["atomic_number"] for counts in expected_by_split.values()
        ),
        "primary_route_labels_balanced_by_split": all(counts["P"] == counts["Q"] for counts in route_by_split.values()),
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
        "primary_expected_label_counts": dict(sorted(expected_labels.items())),
        "primary_route_label_counts": dict(sorted(route_labels.items())),
        "primary_expected_label_counts_by_split": {
            split: dict(sorted(counts.items())) for split, counts in sorted(expected_by_split.items())
        },
        "primary_route_label_counts_by_split": {
            split: dict(sorted(counts.items())) for split, counts in sorted(route_by_split.items())
        },
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
    elif answer == record["lure_atomic_number"]:
        label = "lure_atomic_number"
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
        "final_local_minus_lure_atomic_number_logit": scores["local_number"] - scores["lure_atomic_number"],
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
        "candidate_local_minus_lure_atomic_number_mean_logprob": mean("local_number") - mean("lure_atomic_number"),
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
            "is_primary_conflict_panel": record["panel"] in PRIMARY_CONFLICT_PANELS,
            "expected_correct": parsed["selected_label"] == record["expected_label"],
        }
        if score_candidates:
            output.update(final_next_token_logits(model, tokenizer, record["prompt"], record))
            output.update(candidate_logprob_payload(model, tokenizer, record["prompt"], record))
        outputs.append(output)
        if verbose:
            print(
                f"[{index:04d}/{len(records):04d}] {record['id']} split={record['split']} "
                f"route={record['route_code']} panel={record['panel']} expected={record['expected_label']} "
                f"-> {output['selected_label']} {str(output['selected_answer'])!r}"
            )
    return outputs


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("selected_label")) for row in rows).items()))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parseable = sum(1 for row in rows if row.get("parseable"))
    local_number = sum(1 for row in rows if row.get("selected_label") == "local_number")
    atomic_number = sum(1 for row in rows if row.get("selected_label") == "atomic_number")
    lure_atomic_number = sum(1 for row in rows if row.get("selected_label") == "lure_atomic_number")
    unknown = sum(1 for row in rows if row.get("selected_label") == "unknown")
    other_number = sum(1 for row in rows if row.get("selected_label") == "other_number")
    unparsed = sum(1 for row in rows if row.get("selected_label") == "unparsed")
    expected_correct = sum(1 for row in rows if row.get("expected_correct"))
    expected_labels = Counter(str(row.get("expected_label")) for row in rows)
    route_labels = Counter(str(row.get("route_code")) for row in rows)
    return {
        "rows": len(rows),
        "label_counts": label_counts(rows),
        "parseable": parseable,
        "parseable_rate": rate(parseable, len(rows)),
        "local_number": local_number,
        "local_number_rate": rate(local_number, len(rows)),
        "atomic_number": atomic_number,
        "atomic_number_rate": rate(atomic_number, len(rows)),
        "lure_atomic_number": lure_atomic_number,
        "lure_atomic_number_rate": rate(lure_atomic_number, len(rows)),
        "unknown": unknown,
        "unknown_rate": rate(unknown, len(rows)),
        "other_number": other_number,
        "other_number_rate": rate(other_number, len(rows)),
        "unparsed": unparsed,
        "unparsed_rate": rate(unparsed, len(rows)),
        "expected_correct": expected_correct,
        "expected_correct_rate": rate(expected_correct, len(rows)),
        "expected_label_counts": dict(sorted(expected_labels.items())),
        "route_label_counts": dict(sorted(route_labels.items())),
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
        conflict = [row for row in rows if row["panel"] in PRIMARY_CONFLICT_PANELS]
        non_holdout = [row for row in conflict if row["split"] != "holdout"]
        holdout = [row for row in conflict if row["split"] == "holdout"]
        expected_local = [row for row in conflict if row["expected_label"] == "local_number"]
        expected_atomic = [row for row in conflict if row["expected_label"] == "atomic_number"]
        result[template] = {
            "rows": len(rows),
            "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "selected_label_counts": label_counts(rows),
            "panels": panels,
            "primary_conflict": summarize_rows(conflict),
            "primary_conflict_non_holdout": summarize_rows(non_holdout),
            "primary_conflict_holdout": summarize_rows(holdout),
            "primary_conflict_expected_local": summarize_rows(expected_local),
            "primary_conflict_expected_atomic": summarize_rows(expected_atomic),
            "primary_conflict_by_definition_order": grouped_summaries(conflict, "definition_order"),
            "primary_conflict_by_route_code": grouped_summaries(conflict, "route_code"),
            "primary_conflict_margin_audits": margin_audits(conflict),
        }
    return result


def selection_key(item: dict[str, Any], template: str, templates: tuple[str, ...]) -> tuple[float, float, float, int, int, int]:
    synthetic = item["panels"]["synthetic_numeric_lookup"]
    familiar = item["panels"]["familiar_entity_numeric_lookup"]
    real = item["panels"]["real_world_atomic_number_control"]
    null = item["panels"]["answer_absent_null"]
    absent_rule = item["panels"]["route_rule_absent_conflict"]
    expected_local = item["primary_conflict_expected_local"]
    expected_atomic = item["primary_conflict_expected_atomic"]
    conflict = item["primary_conflict"]
    non_holdout = item["primary_conflict_non_holdout"]
    holdout = item["primary_conflict_holdout"]
    control_floor = min(
        float(synthetic["local_number_rate"]),
        float(familiar["local_number_rate"]),
        float(real["atomic_number_rate"]),
        float(null["unknown_rate"]),
        float(absent_rule["unknown_rate"]),
    )
    expected_floor = min(
        float(expected_local["local_number_rate"]),
        float(expected_atomic["atomic_number_rate"]),
    )
    non_holdout_balance = min(int(non_holdout["local_number"]), int(non_holdout["atomic_number"]))
    holdout_balance = min(int(holdout["local_number"]), int(holdout["atomic_number"]))
    return (
        control_floor,
        expected_floor,
        float(conflict["expected_correct_rate"]),
        non_holdout_balance,
        holdout_balance,
        -templates.index(template),
    )


def select_template(by_template: dict[str, Any], templates: tuple[str, ...]) -> dict[str, Any]:
    selected = max(templates, key=lambda template: selection_key(by_template[template], template, templates))
    return {
        "selected_template": selected,
        "selection_key": list(selection_key(by_template[selected], selected, templates)),
        "all_selection_keys": {template: list(selection_key(by_template[template], template, templates)) for template in templates},
        "rule": [
            "max direct-control plus null floor",
            "max expected-local/expected-atomic conflict floor",
            "max primary expected-label correctness",
            "max non-holdout local/atomic balance",
            "max holdout local/atomic balance",
            "earliest template",
        ],
    }


def observed_pattern(selected: dict[str, Any]) -> str:
    conflict = selected["primary_conflict"]
    expected_atomic = selected["primary_conflict_expected_atomic"]
    expected_local = selected["primary_conflict_expected_local"]
    if float(conflict["local_number_rate"]) >= 0.90:
        return "row_code_local_number_collapse"
    if float(expected_atomic["atomic_number_rate"]) < 0.50 and float(expected_local["local_number_rate"]) >= 0.85:
        return "row_code_expected_atomic_collapse"
    if float(conflict["expected_correct_rate"]) >= 0.85:
        return "row_code_rule_followed"
    return "mixed_or_unresolved_row_code_failure"


def classify(criteria: dict[str, bool], pattern: str) -> str:
    if criteria["smoke_mode"]:
        return "smoke_only"
    if not criteria["structural_passed"]:
        return "structural_invalid"
    if not criteria["selected_prompt_audit_passed"] or not criteria["primary_prompts_have_no_status_lexemes"]:
        return "row_code_prompt_audit_failed"
    for key in (
        "synthetic_panel_local_at_least_90p",
        "synthetic_panel_parseable_at_least_95p",
        "familiar_panel_local_at_least_90p",
        "familiar_panel_parseable_at_least_95p",
        "real_world_panel_atomic_at_least_85p",
        "real_world_panel_parseable_at_least_95p",
        "route_rule_absent_unknown_at_least_90p",
        "route_rule_absent_parseable_at_least_95p",
        "answer_absent_unknown_at_least_90p",
        "answer_absent_parseable_at_least_95p",
    ):
        if not criteria[key]:
            return f"{key}_failed"
    if not criteria["expected_local_conflict_local_at_least_85p"]:
        return "row_code_expected_local_conflict_failed"
    if not criteria["expected_atomic_conflict_atomic_at_least_85p"]:
        return "row_code_expected_atomic_conflict_failed"
    if not criteria["primary_conflict_expected_correct_at_least_85p"]:
        return pattern
    if not criteria["candidate_and_output_margins_reported"]:
        return "row_code_behavior_passed_baselines_missing"
    return "row_code_behavior_passed"


def summarize(records: list[dict[str, Any]], outputs: list[dict[str, Any]], templates: tuple[str, ...], full_run: bool, score_candidates: bool) -> dict[str, Any]:
    structural = structural_check(records, templates)
    by_template = template_summary(outputs, templates)
    selection = select_template(by_template, templates)
    selected = by_template[selection["selected_template"]]
    synthetic = selected["panels"]["synthetic_numeric_lookup"]
    familiar = selected["panels"]["familiar_entity_numeric_lookup"]
    real = selected["panels"]["real_world_atomic_number_control"]
    null = selected["panels"]["answer_absent_null"]
    absent_rule = selected["panels"]["route_rule_absent_conflict"]
    conflict = selected["primary_conflict"]
    non_holdout = selected["primary_conflict_non_holdout"]
    holdout = selected["primary_conflict_holdout"]
    expected_local = selected["primary_conflict_expected_local"]
    expected_atomic = selected["primary_conflict_expected_atomic"]
    selected_rows = [row for row in outputs if row["template"] == selection["selected_template"]]
    selected_prompt_audit_passed = all(
        row["real_atomic_number_hidden_in_conflict"]
        and row["lure_atomic_number_hidden_in_conflict"]
        and row["answer_absent_omits_query_local_number"]
        for row in selected_rows
    )
    primary_prompts_have_no_status_lexemes = all(not row["primary_prompt_has_status_lexeme"] for row in selected_rows)
    criteria = {
        "smoke_mode": not full_run,
        "structural_passed": structural["passed"],
        "full_source_count_is_40": structural["source_count"] == 40,
        "source_disjoint_holdout": structural["criteria"]["source_split_disjoint"]
        and structural["criteria"]["holdout_sources_present"]
        and structural["criteria"]["calibration_sources_present"],
        "primary_expected_labels_balanced": structural["criteria"]["primary_expected_labels_balanced"],
        "primary_route_labels_balanced": structural["criteria"]["primary_route_labels_balanced"],
        "primary_expected_labels_balanced_by_split": structural["criteria"]["primary_expected_labels_balanced_by_split"],
        "primary_route_labels_balanced_by_split": structural["criteria"]["primary_route_labels_balanced_by_split"],
        "selected_prompt_audit_passed": selected_prompt_audit_passed,
        "primary_prompts_have_no_status_lexemes": primary_prompts_have_no_status_lexemes,
        "synthetic_panel_local_at_least_90p": float(synthetic["local_number_rate"]) >= 0.90,
        "synthetic_panel_parseable_at_least_95p": float(synthetic["parseable_rate"]) >= 0.95,
        "familiar_panel_local_at_least_90p": float(familiar["local_number_rate"]) >= 0.90,
        "familiar_panel_parseable_at_least_95p": float(familiar["parseable_rate"]) >= 0.95,
        "real_world_panel_atomic_at_least_85p": float(real["atomic_number_rate"]) >= 0.85,
        "real_world_panel_parseable_at_least_95p": float(real["parseable_rate"]) >= 0.95,
        "route_rule_absent_unknown_at_least_90p": float(absent_rule["unknown_rate"]) >= 0.90,
        "route_rule_absent_parseable_at_least_95p": float(absent_rule["parseable_rate"]) >= 0.95,
        "answer_absent_unknown_at_least_90p": float(null["unknown_rate"]) >= 0.90,
        "answer_absent_parseable_at_least_95p": float(null["parseable_rate"]) >= 0.95,
        "expected_local_conflict_local_at_least_85p": float(expected_local["local_number_rate"]) >= 0.85,
        "expected_atomic_conflict_atomic_at_least_85p": float(expected_atomic["atomic_number_rate"]) >= 0.85,
        "primary_conflict_expected_correct_at_least_85p": float(conflict["expected_correct_rate"]) >= 0.85,
        "primary_conflict_binary_rows_at_least_40": int(conflict["local_number"]) + int(conflict["atomic_number"]) >= 40,
        "non_holdout_conflict_balance_passed": min(int(non_holdout["local_number"]), int(non_holdout["atomic_number"])) >= 10,
        "holdout_conflict_balance_passed": min(int(holdout["local_number"]), int(holdout["atomic_number"])) >= 4,
        "primary_conflict_parseability_at_least_90p": float(conflict["parseable_rate"]) >= 0.90,
        "candidate_and_output_margins_reported": bool(score_candidates and selected["primary_conflict_margin_audits"]["reported"]),
        "visible_status_label_absent_by_design": primary_prompts_have_no_status_lexemes,
    }
    pattern = observed_pattern(selected)
    diagnostic_class = classify(criteria, pattern)
    behavior_gate_passed = diagnostic_class == "row_code_behavior_passed"
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
        "selected_primary_conflict": summary["selected_template_summary"]["primary_conflict"],
        "selected_controls": {
            panel: summary["selected_template_summary"]["panels"][panel]
            for panel in (
                "synthetic_numeric_lookup",
                "familiar_entity_numeric_lookup",
                "real_world_atomic_number_control",
                "route_rule_absent_conflict",
                "answer_absent_null",
            )
        },
        "selected_expected_local_conflict": summary["selected_template_summary"]["primary_conflict_expected_local"],
        "selected_expected_atomic_conflict": summary["selected_template_summary"]["primary_conflict_expected_atomic"],
        "primary_conflict_by_definition_order": summary["selected_template_summary"]["primary_conflict_by_definition_order"],
        "primary_conflict_by_route_code": summary["selected_template_summary"]["primary_conflict_by_route_code"],
    }
    if output_path is not None:
        payload["output_path"] = str(output_path)
    return payload


def write_behavior_status_card(path: Path, result: dict[str, Any], output_path: Path) -> None:
    summary = result["summary"]
    selected = summary["selected_template_summary"]
    criteria = summary["criteria"]
    lines = [
        "# MC019 Row-Code Numeric Arbitration Behavior Status",
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
        "  `code/mc019_row_code_numeric_arbitration.py`",
        "- result:",
        f"  `{output_path.as_posix()}`",
        "",
        "## Verdict",
        "",
    ]
    if summary["behavior_gate_passed"]:
        lines.extend(
            [
                "The row-code numeric behavior gate passed. This would make",
                "MC019 eligible for hidden-state signature search, but only as",
                "a prompt-visible row-code behavior substrate.",
            ]
        )
    elif criteria["smoke_mode"]:
        lines.extend(
            [
                "This is a smoke or partial run. It is not a full-run verdict,",
                "but the selected-template pattern is usable as a bridge",
                "diagnostic.",
            ]
        )
    else:
        lines.extend(
            [
                "The row-code numeric behavior gate failed. Hidden-state work",
                "remains forbidden for this route.",
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
            "## Selected Controls",
            "",
            "| Panel | Rows | Parseable | Local Rate | Atomic Rate | Unknown Rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for panel in (
        "synthetic_numeric_lookup",
        "familiar_entity_numeric_lookup",
        "real_world_atomic_number_control",
        "route_rule_absent_conflict",
        "answer_absent_null",
    ):
        item = selected["panels"][panel]
        lines.append(
            f"| `{panel}` | {item['rows']} | {item['parseable_rate']:.3f} | "
            f"{item['local_number_rate']:.3f} | {item['atomic_number_rate']:.3f} | {item['unknown_rate']:.3f} |"
        )
    conflict = selected["primary_conflict"]
    lines.extend(
        [
            "",
            "## Primary Conflict",
            "",
            f"- rows: {conflict['rows']}",
            f"- parseable rate: {conflict['parseable_rate']:.3f}",
            f"- local-number rows: {conflict['local_number']}",
            f"- atomic-number rows: {conflict['atomic_number']}",
            f"- expected-correct rows: {conflict['expected_correct']}",
            f"- expected-correct rate: {conflict['expected_correct_rate']:.3f}",
            "",
            "## Claim Boundary",
            "",
            "MC019 is a behavior-substrate test. Passing it would show a",
            "prompt-visible row-code contract can create local-versus-learned",
            "contrast under order controls. It would not by itself establish an",
            "internal signature, intervention, or mechanism card.",
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
    parser.add_argument("--artifact-prefix", default="mc019_row_code_numeric_structural")
    parser.add_argument("--status-card", type=Path, default=STATUS_CARD)
    parser.add_argument("--write-status-card", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.time()
    run_type = BEHAVIOR_RUN_TYPE if args.score_model else RUN_TYPE
    records = source_records(args.limit_sources, TEMPLATES, DEFINITION_ORDERS, run_type)
    structural = structural_check(records, TEMPLATES)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.write_manifest and not args.score_model:
        result = {
            "schema_version": 1,
            "card_id": CARD_ID,
            "run_type": RUN_TYPE,
            "model_id": args.model_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "purpose": "Structural gate for row-code numeric arbitration.",
            "templates": list(TEMPLATES),
            "definition_orders": list(DEFINITION_ORDERS),
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
    summary = summarize(records, outputs, TEMPLATES, full_run=full_run, score_candidates=args.score_candidates)
    prefix = args.artifact_prefix
    if prefix == "mc019_row_code_numeric_structural":
        prefix = "mc019_row_code_numeric_behavior"
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
        "definition_orders": list(DEFINITION_ORDERS),
        "elapsed_s": time.time() - started,
        "purpose": "Generated-answer row-code numeric arbitration behavior scoring before hidden-state work.",
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
