#!/usr/bin/env python
"""MC022 explicit branch-name arbitration.

MC021 showed that opaque route codes are not a clean arbitration substrate:
conditional routing was weak even when both answer branches were visible in the
prompt. MC022 tests the narrow next explanation: maybe the failure is caused by
arbitrary route codes, not by branch arbitration itself.

This runner replaces route codes with semantic answer-source labels:

1. visible-visible: answer source chooses between prompt-visible LOCAL and
   REFERENCE numeric columns;
2. visible-learned: answer source chooses between prompt-visible LOCAL numbers
   and learned ATOMIC numbers.

This is a behavior diagnostic only. No hidden-state search is justified unless
the behavior substrate passes the relevant controls.
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


CARD_ID = "MC022"
RUN_TYPE = "explicit_branch_name_arbitration_structural_gate"
BEHAVIOR_RUN_TYPE = "explicit_branch_name_arbitration_behavior"
RESULT_DIR = Path("results/cards/MC022")
STATUS_CARD = Path("research/cards/MC022_EXPLICIT_BRANCH_NAME_ARBITRATION_BEHAVIOR_STATUS.md")

PANELS = (
    "local_number_control",
    "visible_number_control",
    "atomic_number_control",
    "source_visible_local_conflict",
    "source_visible_reference_conflict",
    "source_learned_local_conflict",
    "source_learned_atomic_conflict",
    "answer_absent_null",
)
VISIBLE_CONFLICT_PANELS = ("source_visible_local_conflict", "source_visible_reference_conflict")
LEARNED_CONFLICT_PANELS = ("source_learned_local_conflict", "source_learned_atomic_conflict")
CONFLICT_PANELS = VISIBLE_CONFLICT_PANELS + LEARNED_CONFLICT_PANELS
TEMPLATES = ("explicit_source_column", "compact_source_column")
RULE_ORDERS = ("local_first", "alternate_first")
SOURCE_LABELS = ("LOCAL", "REFERENCE", "ATOMIC")
STATUS_LEXEME_RE = re.compile(r"\b(trusted|untrusted|reliable|unreliable|status)\b", re.IGNORECASE)


def reference_number(source: dict[str, Any]) -> str:
    return str(201 + source["source_index"])


def response_suffix() -> list[str]:
    return [
        "Return only one final token: an integer or UNKNOWN.",
        "Do not explain.",
        "Answer:",
    ]


def alternate_label(row: dict[str, Any], *, family: str) -> str:
    if family == "visible":
        return "LOCAL" if row["source_index"] % 2 == 0 else "REFERENCE"
    if family == "learned":
        return "LOCAL" if row["source_index"] % 2 == 0 else "ATOMIC"
    raise ValueError(f"unknown family {family}")


def label_for_row(
    row: dict[str, Any],
    query: dict[str, Any],
    *,
    query_label: str,
    family: str,
) -> str:
    if row["source_id"] == query["source_id"]:
        return query_label
    return alternate_label(row, family=family)


def table_lines(
    rows: list[dict[str, Any]],
    query: dict[str, Any],
    *,
    include_query: bool,
    query_label: str,
    family: str,
    template: str,
) -> list[str]:
    visible_rows = rows if include_query else [row for row in rows if row["source_id"] != query["source_id"]]
    if template == "explicit_source_column":
        lines = [
            "A local lab table may define temporary lab numbers.",
            "A visible reference column may define separate prompt numbers.",
            "Standard chemistry may define atomic numbers.",
            "Element answer-source table:",
        ]
        for row in visible_rows:
            source_label = label_for_row(row, query, query_label=query_label, family=family)
            lines.append(
                f"{row['element']} has local lab number {row['local_number']}, "
                f"visible reference number {reference_number(row)}, and answer source {source_label}."
            )
        return lines
    if template == "compact_source_column":
        lines = [
            "Three numeric sources may conflict.",
            "Rows:",
        ]
        for row in visible_rows:
            source_label = label_for_row(row, query, query_label=query_label, family=family)
            lines.append(
                f"{row['element']} | local {row['local_number']} | "
                f"reference {reference_number(row)} | source {source_label}"
            )
        return lines
    raise ValueError(f"unknown template {template}")


def ordered_labels(*, family: str, rule_order: str) -> tuple[str, str]:
    alternate = "REFERENCE" if family == "visible" else "ATOMIC"
    if rule_order == "local_first":
        return ("LOCAL", alternate)
    if rule_order == "alternate_first":
        return (alternate, "LOCAL")
    raise ValueError(f"unknown rule order {rule_order}")


def append_source_rule(lines: list[str], *, family: str, rule_order: str, template: str) -> None:
    ordered = ordered_labels(family=family, rule_order=rule_order)
    if template == "explicit_source_column":
        lines.append("First find the queried element's answer source in the table.")
        for label in ordered:
            if label == "LOCAL":
                lines.append("If the queried row's answer source is LOCAL, return its local lab number.")
            elif label == "REFERENCE":
                lines.append("If the queried row's answer source is REFERENCE, return its visible reference number.")
            elif label == "ATOMIC":
                lines.append("If the queried row's answer source is ATOMIC, return its standard atomic number.")
            else:
                raise ValueError(f"unknown source label {label}")
        return
    if template == "compact_source_column":
        for label in ordered:
            if label == "LOCAL":
                lines.append("Source LOCAL: local lab number.")
            elif label == "REFERENCE":
                lines.append("Source REFERENCE: visible reference number.")
            elif label == "ATOMIC":
                lines.append("Source ATOMIC: standard atomic number.")
            else:
                raise ValueError(f"unknown source label {label}")
        return
    raise ValueError(f"unknown template {template}")


def make_prompt(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    panel: str,
    template: str,
    rule_order: str,
) -> tuple[str, str, str]:
    rows = distractors(sources, source, 4)
    if panel == "local_number_control":
        lines = table_lines(
            rows,
            source,
            include_query=True,
            query_label="LOCAL",
            family="visible",
            template=template,
        )
        lines.append("Return the local lab number for the queried element.")
        lines.append(f"Query element: {source['element']}")
        return "\n".join([*lines, *response_suffix()]), "local_number", "LOCAL"
    if panel == "visible_number_control":
        lines = table_lines(
            rows,
            source,
            include_query=True,
            query_label="REFERENCE",
            family="visible",
            template=template,
        )
        lines.append("Return the visible reference number for the queried element.")
        lines.append(f"Query element: {source['element']}")
        return "\n".join([*lines, *response_suffix()]), "visible_number", "REFERENCE"
    if panel == "atomic_number_control":
        lines = table_lines(
            rows,
            source,
            include_query=True,
            query_label="ATOMIC",
            family="learned",
            template=template,
        )
        lines.append("Ignore local lab numbers and visible reference numbers for this query.")
        lines.append("Return the standard atomic number for the queried element.")
        lines.append(f"Query element: {source['element']}")
        return "\n".join([*lines, *response_suffix()]), "atomic_number", "ATOMIC"
    if panel == "source_visible_local_conflict":
        lines = table_lines(
            rows,
            source,
            include_query=True,
            query_label="LOCAL",
            family="visible",
            template=template,
        )
        append_source_rule(lines, family="visible", rule_order=rule_order, template=template)
        lines.append("Apply the queried row's answer source and return the controlling number.")
        lines.append(f"Query element: {source['element']}")
        return "\n".join([*lines, *response_suffix()]), "local_number", "LOCAL"
    if panel == "source_visible_reference_conflict":
        lines = table_lines(
            rows,
            source,
            include_query=True,
            query_label="REFERENCE",
            family="visible",
            template=template,
        )
        append_source_rule(lines, family="visible", rule_order=rule_order, template=template)
        lines.append("Apply the queried row's answer source and return the controlling number.")
        lines.append(f"Query element: {source['element']}")
        return "\n".join([*lines, *response_suffix()]), "visible_number", "REFERENCE"
    if panel == "source_learned_local_conflict":
        lines = table_lines(
            rows,
            source,
            include_query=True,
            query_label="LOCAL",
            family="learned",
            template=template,
        )
        append_source_rule(lines, family="learned", rule_order=rule_order, template=template)
        lines.append("Apply the queried row's answer source and return the controlling number.")
        lines.append(f"Query element: {source['element']}")
        return "\n".join([*lines, *response_suffix()]), "local_number", "LOCAL"
    if panel == "source_learned_atomic_conflict":
        lines = table_lines(
            rows,
            source,
            include_query=True,
            query_label="ATOMIC",
            family="learned",
            template=template,
        )
        append_source_rule(lines, family="learned", rule_order=rule_order, template=template)
        lines.append("Apply the queried row's answer source and return the controlling number.")
        lines.append(f"Query element: {source['element']}")
        return "\n".join([*lines, *response_suffix()]), "atomic_number", "ATOMIC"
    if panel == "answer_absent_null":
        lines = table_lines(
            rows,
            source,
            include_query=False,
            query_label="LOCAL",
            family="visible",
            template=template,
        )
        append_source_rule(lines, family="visible", rule_order=rule_order, template=template)
        lines.append("If the queried element is absent from the table, return UNKNOWN before applying any source rule.")
        lines.append(f"Query element: {source['element']}")
        return "\n".join([*lines, *response_suffix()]), "unknown", "ABSENT"
    raise ValueError(f"unknown panel {panel}")


def candidate_answers_for(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"answer": record["local_number"], "label": "local_number", "candidate_type": "prompt_local_lab_number"},
        {"answer": record["visible_number"], "label": "visible_number", "candidate_type": "prompt_visible_reference_number"},
        {"answer": record["atomic_number"], "label": "atomic_number", "candidate_type": "real_world_atomic_number"},
        {"answer": "UNKNOWN", "label": "unknown", "candidate_type": "unknown"},
    ]


def source_records(
    limit_sources: int | None = None,
    templates: tuple[str, ...] = TEMPLATES,
    rule_orders: tuple[str, ...] = RULE_ORDERS,
    run_type: str = RUN_TYPE,
) -> list[dict[str, Any]]:
    sources = base_sources(limit_sources)
    records: list[dict[str, Any]] = []
    for source in sources:
        for template in templates:
            for rule_order in rule_orders:
                for panel in PANELS:
                    prompt, expected_label, answer_source_label = make_prompt(
                        sources,
                        source,
                        panel=panel,
                        template=template,
                        rule_order=rule_order,
                    )
                    record = {
                        "id": f"{CARD_ID}_{template}_{rule_order}_{panel}_{source['source_id']}",
                        "card_id": CARD_ID,
                        "run_type": run_type,
                        "model_id": MODEL_ID,
                        "template": template,
                        "rule_order": rule_order,
                        "panel": panel,
                        "split": source["split"],
                        "source_id": source["source_id"],
                        "source_index": source["source_index"],
                        "element": source["element"],
                        "answer_source_label": answer_source_label,
                        "atomic_number": str(source["atomic_number"]),
                        "local_number": str(source["local_number"]),
                        "visible_number": reference_number(source),
                        "expected_label": expected_label,
                        "prompt": prompt,
                    }
                    record["candidate_answers"] = candidate_answers_for(record)
                    records.append(record)
    return records


def prompt_audit(record: dict[str, Any]) -> dict[str, Any]:
    prompt = record["prompt"]
    null = record["panel"] == "answer_absent_null"
    return {
        "atomic_number_occurrences": word_occurrences(prompt, record["atomic_number"]),
        "atomic_number_hidden": word_occurrences(prompt, record["atomic_number"]) == 0,
        "answer_absent_omits_query_local_number": not null or not line_contains_both(prompt, record["element"], record["local_number"]),
        "answer_absent_omits_query_visible_number": not null or not line_contains_both(prompt, record["element"], record["visible_number"]),
        "prompt_has_status_lexeme": bool(STATUS_LEXEME_RE.search(prompt)),
    }


def structural_check(
    records: list[dict[str, Any]],
    templates: tuple[str, ...],
    rule_orders: tuple[str, ...],
) -> dict[str, Any]:
    source_ids = {row["source_id"] for row in records}
    expected_count = len(source_ids) * len(templates) * len(rule_orders) * len(PANELS)
    splits_by_source: dict[str, set[str]] = {}
    for row in records:
        splits_by_source.setdefault(row["source_id"], set()).add(row["split"])
    split_rows = Counter(row["split"] for row in records)
    visible_conflict = [row for row in records if row["panel"] in VISIBLE_CONFLICT_PANELS]
    learned_conflict = [row for row in records if row["panel"] in LEARNED_CONFLICT_PANELS]
    visible_expected = Counter(row["expected_label"] for row in visible_conflict)
    learned_expected = Counter(row["expected_label"] for row in learned_conflict)
    visible_source_labels = Counter(row["answer_source_label"] for row in visible_conflict)
    learned_source_labels = Counter(row["answer_source_label"] for row in learned_conflict)
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
    criteria = {
        "expected_row_count": len(records) == expected_count,
        "all_panels_present": set(PANELS) == {row["panel"] for row in records},
        "all_templates_present": set(templates) == {row["template"] for row in records},
        "all_rule_orders_present": set(rule_orders) == {row["rule_order"] for row in records},
        "source_split_disjoint": all(len(splits) == 1 for splits in splits_by_source.values()),
        "holdout_sources_present": any(row["split"] == "holdout" for row in records),
        "calibration_sources_present": any(row["split"] == "calibration" for row in records),
        "atomic_number_hidden": all(audit["atomic_number_hidden"] for audit in audits),
        "answer_absent_omits_query_local_number": all(audit["answer_absent_omits_query_local_number"] for audit in audits),
        "answer_absent_omits_query_visible_number": all(audit["answer_absent_omits_query_visible_number"] for audit in audits),
        "candidate_answers_parseable": not malformed_candidates,
        "no_candidate_collisions": not collisions,
        "single_answer_suffix": len(suffixes) == 1,
        "prompts_have_no_status_lexemes": not status_rows,
        "visible_conflict_expected_balanced": visible_expected["local_number"] == visible_expected["visible_number"],
        "learned_conflict_expected_balanced": learned_expected["local_number"] == learned_expected["atomic_number"],
        "visible_conflict_source_labels_balanced": visible_source_labels["LOCAL"] == visible_source_labels["REFERENCE"],
        "learned_conflict_source_labels_balanced": learned_source_labels["LOCAL"] == learned_source_labels["ATOMIC"],
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
        "visible_conflict_expected_counts": dict(sorted(visible_expected.items())),
        "learned_conflict_expected_counts": dict(sorted(learned_expected.items())),
        "visible_conflict_source_label_counts": dict(sorted(visible_source_labels.items())),
        "learned_conflict_source_label_counts": dict(sorted(learned_source_labels.items())),
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
    elif answer == record["visible_number"]:
        label = "visible_number"
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
        "final_local_minus_visible_number_logit": scores["local_number"] - scores["visible_number"],
        "final_local_minus_atomic_number_logit": scores["local_number"] - scores["atomic_number"],
        "final_visible_minus_atomic_number_logit": scores["visible_number"] - scores["atomic_number"],
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
        "candidate_local_minus_visible_number_mean_logprob": mean("local_number") - mean("visible_number"),
        "candidate_local_minus_atomic_number_mean_logprob": mean("local_number") - mean("atomic_number"),
        "candidate_visible_minus_atomic_number_mean_logprob": mean("visible_number") - mean("atomic_number"),
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
                f"source={record['answer_source_label']} panel={record['panel']} "
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
        "visible_number": counts["visible_number"],
        "visible_number_rate": rate(counts["visible_number"], len(rows)),
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
        "answer_source_label_counts": dict(sorted(Counter(str(row.get("answer_source_label")) for row in rows).items())),
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
        visible_rows = [row for row in rows if row["panel"] in VISIBLE_CONFLICT_PANELS]
        learned_rows = [row for row in rows if row["panel"] in LEARNED_CONFLICT_PANELS]
        result[template] = {
            "rows": len(rows),
            "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "selected_label_counts": label_counts(rows),
            "panels": panels,
            "visible_conflict": summarize_rows(visible_rows),
            "visible_conflict_by_rule_order": grouped_summaries(visible_rows, "rule_order"),
            "visible_conflict_by_answer_source": grouped_summaries(visible_rows, "answer_source_label"),
            "learned_conflict": summarize_rows(learned_rows),
            "learned_conflict_by_rule_order": grouped_summaries(learned_rows, "rule_order"),
            "learned_conflict_by_answer_source": grouped_summaries(learned_rows, "answer_source_label"),
            "margin_audits": margin_audits(rows),
        }
    return result


def selection_key(item: dict[str, Any], template: str, templates: tuple[str, ...]) -> tuple[float, float, float, float, int]:
    panels = item["panels"]
    control_floor = min(
        float(panels["local_number_control"]["local_number_rate"]),
        float(panels["visible_number_control"]["visible_number_rate"]),
        float(panels["atomic_number_control"]["atomic_number_rate"]),
        float(panels["answer_absent_null"]["unknown_rate"]),
    )
    visible_floor = min(
        float(panels["source_visible_local_conflict"]["local_number_rate"]),
        float(panels["source_visible_reference_conflict"]["visible_number_rate"]),
    )
    learned_floor = min(
        float(panels["source_learned_local_conflict"]["local_number_rate"]),
        float(panels["source_learned_atomic_conflict"]["atomic_number_rate"]),
    )
    parse_floor = min(float(panel["parseable_rate"]) for panel in panels.values())
    return (
        control_floor,
        visible_floor,
        learned_floor,
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
            "max visible-visible source floor",
            "max visible-learned source floor",
            "max all-panel parseability floor",
            "earliest template",
        ],
    }


def observed_pattern(selected: dict[str, Any]) -> str:
    panels = selected["panels"]
    if float(panels["visible_number_control"]["visible_number_rate"]) < 0.90:
        return "visible_reference_control_failed"
    if float(selected["visible_conflict"]["expected_correct_rate"]) < 0.85:
        return "explicit_visible_source_routing_failed"
    if float(selected["learned_conflict"]["expected_correct_rate"]) < 0.85:
        return "explicit_learned_branch_arbitration_failed"
    return "explicit_branch_name_behavior_passed"


def classify(criteria: dict[str, bool], pattern: str) -> str:
    if criteria["smoke_mode"]:
        return "smoke_only"
    if not criteria["structural_passed"]:
        return "structural_invalid"
    if not criteria["selected_prompt_audit_passed"] or not criteria["prompts_have_no_status_lexemes"]:
        return "explicit_branch_name_prompt_audit_failed"
    if not criteria["all_controls_passed"]:
        return "explicit_branch_name_control_failed"
    if pattern == "visible_reference_control_failed":
        return "visible_reference_control_failed"
    if not criteria["visible_conflict_expected_correct_at_least_85p"]:
        return "explicit_visible_source_routing_failed"
    if not criteria["learned_conflict_expected_correct_at_least_85p"]:
        return "explicit_learned_branch_arbitration_failed"
    if not criteria["candidate_and_output_margins_reported"]:
        return "explicit_branch_name_behavior_passed_baselines_missing"
    return "explicit_branch_name_behavior_passed"


def summarize(
    records: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    templates: tuple[str, ...],
    rule_orders: tuple[str, ...],
    full_run: bool,
    score_candidates: bool,
) -> dict[str, Any]:
    structural = structural_check(records, templates, rule_orders)
    by_template = template_summary(outputs, templates)
    selection = select_template(by_template, templates)
    selected = by_template[selection["selected_template"]]
    panels = selected["panels"]
    selected_rows = [row for row in outputs if row["template"] == selection["selected_template"]]
    selected_prompt_audit_passed = all(
        row["atomic_number_hidden"]
        and row["answer_absent_omits_query_local_number"]
        and row["answer_absent_omits_query_visible_number"]
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
        "local_control_at_least_90p": float(panels["local_number_control"]["local_number_rate"]) >= 0.90,
        "visible_control_at_least_90p": float(panels["visible_number_control"]["visible_number_rate"]) >= 0.90,
        "atomic_control_at_least_85p": float(panels["atomic_number_control"]["atomic_number_rate"]) >= 0.85,
        "answer_absent_unknown_at_least_90p": float(panels["answer_absent_null"]["unknown_rate"]) >= 0.90,
        "visible_conflict_expected_correct_at_least_85p": float(selected["visible_conflict"]["expected_correct_rate"]) >= 0.85,
        "learned_conflict_expected_correct_at_least_85p": float(selected["learned_conflict"]["expected_correct_rate"]) >= 0.85,
        "learned_atomic_source_at_least_85p": float(panels["source_learned_atomic_conflict"]["atomic_number_rate"]) >= 0.85,
        "all_panels_parseable_at_least_95p": min(float(panel["parseable_rate"]) for panel in panels.values()) >= 0.95,
        "candidate_and_output_margins_reported": bool(score_candidates and selected["margin_audits"]["reported"]),
    }
    criteria["all_controls_passed"] = (
        criteria["local_control_at_least_90p"]
        and criteria["visible_control_at_least_90p"]
        and criteria["atomic_control_at_least_85p"]
        and criteria["answer_absent_unknown_at_least_90p"]
        and criteria["all_panels_parseable_at_least_95p"]
    )
    pattern = observed_pattern(selected)
    diagnostic_class = classify(criteria, pattern)
    behavior_gate_passed = diagnostic_class == "explicit_branch_name_behavior_passed"
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
            for panel in ("local_number_control", "visible_number_control", "atomic_number_control", "answer_absent_null")
        },
        "selected_visible_conflict": summary["selected_template_summary"]["visible_conflict"],
        "selected_learned_conflict": summary["selected_template_summary"]["learned_conflict"],
        "selected_panels": summary["selected_template_summary"]["panels"],
    }
    if output_path is not None:
        payload["output_path"] = str(output_path)
    return payload


def write_behavior_status_card(path: Path, result: dict[str, Any], output_path: Path) -> None:
    summary = result["summary"]
    selected = summary["selected_template_summary"]
    criteria = summary["criteria"]
    lines = [
        "# MC022 Explicit Branch-Name Arbitration Behavior Status",
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
        "  `code/mc022_explicit_branch_name_arbitration.py`",
        "- result:",
        f"  `{output_path.as_posix()}`",
        "",
        "## Verdict",
        "",
    ]
    if summary["behavior_gate_passed"]:
        lines.extend(
            [
                "The explicit branch-name arbitration behavior gate passed.",
                "This would justify a separate bridge decision, but it is not",
                "itself a hidden-state mechanism claim.",
            ]
        )
    elif criteria["smoke_mode"]:
        lines.extend(
            [
                "This is a smoke or partial run. It diagnoses whether replacing",
                "opaque route codes with semantic answer-source labels repairs",
                "visible-visible or visible-learned branch arbitration.",
            ]
        )
    else:
        lines.extend(
            [
                "The explicit branch-name arbitration behavior gate failed.",
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
            "| Panel | Rows | Parseable | Local Rate | Visible Rate | Atomic Rate | Unknown Rate | Other Rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for panel in PANELS:
        item = selected["panels"][panel]
        lines.append(
            f"| `{panel}` | {item['rows']} | {item['parseable_rate']:.3f} | "
            f"{item['local_number_rate']:.3f} | {item['visible_number_rate']:.3f} | "
            f"{item['atomic_number_rate']:.3f} | {item['unknown_rate']:.3f} | {item['other_number_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate Conflicts",
            "",
            f"- visible conflict expected-correct rate: {selected['visible_conflict']['expected_correct_rate']:.3f}",
            f"- learned conflict expected-correct rate: {selected['learned_conflict']['expected_correct_rate']:.3f}",
            "",
            "## Claim Boundary",
            "",
            "MC022 is a behavior diagnostic for semantic answer-source labels over",
            "prompt-visible versus learned-memory branches. It does not establish",
            "an internal signature, intervention, or mechanism card.",
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
    parser.add_argument("--artifact-prefix", default="mc022_explicit_branch_name_arbitration_structural")
    parser.add_argument("--status-card", type=Path, default=STATUS_CARD)
    parser.add_argument("--write-status-card", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.time()
    run_type = BEHAVIOR_RUN_TYPE if args.score_model else RUN_TYPE
    records = source_records(args.limit_sources, TEMPLATES, RULE_ORDERS, run_type)
    structural = structural_check(records, TEMPLATES, RULE_ORDERS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.write_manifest and not args.score_model:
        result = {
            "schema_version": 1,
            "card_id": CARD_ID,
            "run_type": RUN_TYPE,
            "model_id": args.model_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "purpose": "Structural gate for explicit branch-name arbitration.",
            "templates": list(TEMPLATES),
            "rule_orders": list(RULE_ORDERS),
            "source_labels": list(SOURCE_LABELS),
            "panels": list(PANELS),
            "visible_conflict_panels": list(VISIBLE_CONFLICT_PANELS),
            "learned_conflict_panels": list(LEARNED_CONFLICT_PANELS),
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
    summary = summarize(records, outputs, TEMPLATES, RULE_ORDERS, full_run=full_run, score_candidates=args.score_candidates)
    prefix = args.artifact_prefix
    if prefix == "mc022_explicit_branch_name_arbitration_structural":
        prefix = "mc022_explicit_branch_name_arbitration_behavior"
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
        "rule_orders": list(RULE_ORDERS),
        "source_labels": list(SOURCE_LABELS),
        "elapsed_s": time.time() - started,
        "purpose": "Generated-answer explicit branch-name arbitration diagnostic before hidden-state work.",
        "panels": list(PANELS),
        "visible_conflict_panels": list(VISIBLE_CONFLICT_PANELS),
        "learned_conflict_panels": list(LEARNED_CONFLICT_PANELS),
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
