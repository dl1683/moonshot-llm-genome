#!/usr/bin/env python
"""MC018 counterbalanced neutral selector-label behavior gate.

MC017 changed the answer interface from a number to LOCAL/ATOMIC, but a smoke
run collapsed to LOCAL even when the prompt said no local table was active.
That failure is ambiguous: it could be a real source-selection failure, a
LOCAL-token prior, an answer-list-order prior, or a fixed first-choice habit.

MC018 removes LOCAL/ATOMIC from the answer tokens. The model must answer A, B,
or UNKNOWN, while the prompt counterbalances which source A/B names and whether
the allowed answer list is shown A-first or B-first. This is still a behavior
gate, not a hidden-state mechanism claim.
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
from mc017_selector_token_numeric_arbitration import (
    STATUS_LEXEME_RE,
    append_letter_group_feature,
    is_a_to_m,
    render_local_table,
)


CARD_ID = "MC018"
RUN_TYPE = "counterbalanced_selector_labels_structural_gate"
BEHAVIOR_RUN_TYPE = "counterbalanced_selector_labels_behavior"
RESULT_DIR = Path("results/cards/MC018")
STATUS_CARD = Path("research/cards/MC018_COUNTERBALANCED_SELECTOR_LABELS_BEHAVIOR_STATUS.md")

PANELS = (
    "local_selector_control",
    "atomic_selector_control",
    "local_if_a_m_selector_conflict",
    "local_if_n_z_selector_conflict",
    "selector_rule_absent_conflict",
    "answer_absent_null",
)
PRIMARY_CONFLICT_PANELS = ("local_if_a_m_selector_conflict", "local_if_n_z_selector_conflict")
CONFLICT_PANELS = PRIMARY_CONFLICT_PANELS + ("selector_rule_absent_conflict",)
TEMPLATES = ("explicit_source_labels", "compact_source_labels")
MAPPINGS = ("a_local_b_atomic", "a_atomic_b_local")
OPTION_ORDERS = ("a_first", "b_first")
DEFINITION_ORDERS = ("a_definition_first", "b_definition_first")
CHOICE_RE = re.compile(r"^(A|B|UNKNOWN)[\s\.,;:!\?]*$", re.IGNORECASE)


def source_labels(mapping: str) -> dict[str, str]:
    if mapping == "a_local_b_atomic":
        return {"choice_a": "local_selector", "choice_b": "atomic_selector"}
    if mapping == "a_atomic_b_local":
        return {"choice_a": "atomic_selector", "choice_b": "local_selector"}
    raise ValueError(f"unknown mapping {mapping}")


def choice_for_source(mapping: str, source_label: str) -> str:
    labels = source_labels(mapping)
    for choice, label in labels.items():
        if label == source_label:
            return choice
    raise ValueError(f"source label {source_label} absent from {mapping}")


def choice_token(choice_label: str) -> str:
    return {"choice_a": "A", "choice_b": "B", "unknown": "UNKNOWN"}[choice_label]


def first_choice_label(option_order: str) -> str:
    if option_order == "a_first":
        return "choice_a"
    if option_order == "b_first":
        return "choice_b"
    raise ValueError(f"unknown option order {option_order}")


def answer_suffix(option_order: str) -> list[str]:
    if option_order == "a_first":
        choices = "A, B, or UNKNOWN"
    elif option_order == "b_first":
        choices = "B, A, or UNKNOWN"
    else:
        raise ValueError(f"unknown option order {option_order}")
    return [
        f"Return only one final token: {choices}.",
        "Do not explain.",
        "Answer:",
    ]


def append_mapping(lines: list[str], mapping: str, *, template: str, definition_order: str) -> None:
    labels = source_labels(mapping)

    def desc(label: str) -> str:
        if label == "local_selector":
            return "the local lab table source"
        if label == "atomic_selector":
            return "the standard chemistry source"
        raise ValueError(f"unknown source label {label}")

    if definition_order == "a_definition_first":
        choices = ("choice_a", "choice_b")
    elif definition_order == "b_definition_first":
        choices = ("choice_b", "choice_a")
    else:
        raise ValueError(f"unknown definition order {definition_order}")

    if template == "explicit_source_labels":
        lines.append("Use these source labels:")
        for choice in choices:
            token = "A" if choice == "choice_a" else "B"
            lines.append(f"Source {token} means {desc(labels[choice])}.")
        return
    if template == "compact_source_labels":
        for choice in choices:
            token = "A" if choice == "choice_a" else "B"
            lines.append(f"Source {token} = {desc(labels[choice])}.")
        return
    raise ValueError(f"unknown template {template}")


def append_source_rule(lines: list[str], *, local_on: str, template: str) -> None:
    if template == "explicit_source_labels":
        lines.append("First inspect the first letter of the queried element name.")
        if local_on == "a_m":
            lines.append("If the first letter is from A through M, use the local lab table source.")
            lines.append("If the first letter is from N through Z, use the standard chemistry source.")
        elif local_on == "n_z":
            lines.append("If the first letter is from A through M, use the standard chemistry source.")
            lines.append("If the first letter is from N through Z, use the local lab table source.")
        else:
            raise ValueError(f"unknown local_on {local_on}")
        return
    if template == "compact_source_labels":
        if local_on == "a_m":
            lines.append("Rule: A-M element names use the local lab table source; N-Z names use the standard chemistry source.")
        elif local_on == "n_z":
            lines.append("Rule: A-M element names use the standard chemistry source; N-Z names use the local lab table source.")
        else:
            raise ValueError(f"unknown local_on {local_on}")
        return
    raise ValueError(f"unknown template {template}")


def expected_source_label(source: dict[str, Any], *, local_on: str) -> str:
    a_m = is_a_to_m(source)
    if local_on == "a_m":
        return "local_selector" if a_m else "atomic_selector"
    if local_on == "n_z":
        return "atomic_selector" if a_m else "local_selector"
    raise ValueError(f"unknown local_on {local_on}")


def make_prompt(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    panel: str,
    template: str,
    mapping: str,
    option_order: str,
    definition_order: str,
) -> tuple[str, str, str]:
    rows = distractors(sources, source, 4)
    lines: list[str] = []
    append_mapping(lines, mapping, template=template, definition_order=definition_order)

    if panel == "local_selector_control":
        lines.extend(render_local_table(rows, source, key_field="element", include_query=True, template="selector_rule"))
        lines.append("Active-source instruction: use the local lab table source for this query.")
        lines.append(f"Query element: {source['element']}")
        lines.append("Which source label controls the answer?")
        return "\n".join([*lines, *answer_suffix(option_order)]), "local_selector", choice_for_source(mapping, "local_selector")

    if panel == "atomic_selector_control":
        lines.append("No local lab table is active.")
        lines.append("Active-source instruction: use the standard chemistry source for this query.")
        lines.append(f"Query element: {source['element']}")
        lines.append("Which source label controls the answer?")
        return "\n".join([*lines, *answer_suffix(option_order)]), "atomic_selector", choice_for_source(mapping, "atomic_selector")

    if panel == "local_if_a_m_selector_conflict":
        lines.extend(render_local_table(rows, source, key_field="element", include_query=True, template="selector_rule"))
        append_letter_group_feature(lines, source, template="selector_rule")
        lines.append("Use the selector rule below before choosing a source label.")
        append_source_rule(lines, local_on="a_m", template=template)
        lines.append(f"Query element: {source['element']}")
        lines.append("Which source label controls the answer?")
        expected = expected_source_label(source, local_on="a_m")
        return "\n".join([*lines, *answer_suffix(option_order)]), expected, choice_for_source(mapping, expected)

    if panel == "local_if_n_z_selector_conflict":
        lines.extend(render_local_table(rows, source, key_field="element", include_query=True, template="selector_rule"))
        append_letter_group_feature(lines, source, template="selector_rule")
        lines.append("Use the selector rule below before choosing a source label.")
        append_source_rule(lines, local_on="n_z", template=template)
        lines.append(f"Query element: {source['element']}")
        lines.append("Which source label controls the answer?")
        expected = expected_source_label(source, local_on="n_z")
        return "\n".join([*lines, *answer_suffix(option_order)]), expected, choice_for_source(mapping, expected)

    if panel == "selector_rule_absent_conflict":
        lines.extend(render_local_table(rows, source, key_field="element", include_query=True, template="selector_rule"))
        lines.append("No selector rule is available for choosing between the two sources.")
        lines.append(f"Query element: {source['element']}")
        lines.append("Which source label controls the answer?")
        return "\n".join([*lines, *answer_suffix(option_order)]), "ambiguous", "ambiguous"

    if panel == "answer_absent_null":
        lines.extend(render_local_table(rows, source, key_field="element", include_query=False, template="selector_rule"))
        lines.append("This is a local-table membership check, not a chemistry question.")
        lines.append("If the query element is absent from the local lab table, answer UNKNOWN.")
        lines.append(f"Query element: {source['element']}")
        lines.append("Which source label controls the answer?")
        return "\n".join([*lines, *answer_suffix(option_order)]), "unknown", "unknown"

    raise ValueError(f"unknown panel {panel}")


def candidate_answers_for(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"answer": "A", "label": "choice_a", "candidate_type": record["source_a_label"]},
        {"answer": "B", "label": "choice_b", "candidate_type": record["source_b_label"]},
        {"answer": "UNKNOWN", "label": "unknown", "candidate_type": "unknown"},
    ]


def source_records(
    limit_sources: int | None = None,
    templates: tuple[str, ...] = TEMPLATES,
    mappings: tuple[str, ...] = MAPPINGS,
    option_orders: tuple[str, ...] = OPTION_ORDERS,
    definition_orders: tuple[str, ...] = DEFINITION_ORDERS,
    run_type: str = RUN_TYPE,
) -> list[dict[str, Any]]:
    sources = base_sources(limit_sources)
    records: list[dict[str, Any]] = []
    for source in sources:
        for template in templates:
            for mapping in mappings:
                labels = source_labels(mapping)
                for definition_order in definition_orders:
                    for option_order in option_orders:
                        for panel in PANELS:
                            prompt, expected_label, expected_choice = make_prompt(
                                sources,
                                source,
                                panel=panel,
                                template=template,
                                mapping=mapping,
                                option_order=option_order,
                                definition_order=definition_order,
                            )
                            record = {
                            "id": f"{CARD_ID}_{template}_{mapping}_{definition_order}_{option_order}_{panel}_{source['source_id']}",
                            "card_id": CARD_ID,
                            "run_type": run_type,
                            "model_id": MODEL_ID,
                            "template": template,
                            "mapping": mapping,
                            "definition_order": definition_order,
                            "option_order": option_order,
                            "panel": panel,
                            "split": source["split"],
                            "source_id": source["source_id"],
                            "source_index": source["source_index"],
                            "element": source["element"],
                            "synthetic_key": source["synthetic_key"],
                            "atomic_number": str(source["atomic_number"]),
                            "lure_atomic_number": str(source["lure_atomic_number"]),
                            "local_number": str(source["local_number"]),
                            "source_a_label": labels["choice_a"],
                            "source_b_label": labels["choice_b"],
                            "first_listed_choice": first_choice_label(option_order),
                            "expected_label": expected_label,
                            "expected_choice": expected_choice,
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
    expected_count = len(source_ids) * len(templates) * len(MAPPINGS) * len(DEFINITION_ORDERS) * len(OPTION_ORDERS) * len(PANELS)
    splits_by_source: dict[str, set[str]] = {}
    for row in records:
        splits_by_source.setdefault(row["source_id"], set()).add(row["split"])
    split_rows = Counter(row["split"] for row in records)
    primary = [row for row in records if row["panel"] in PRIMARY_CONFLICT_PANELS]
    expected_sources = Counter(row["expected_label"] for row in primary)
    expected_choices = Counter(row["expected_choice"] for row in primary)
    by_split_sources: dict[str, Counter[str]] = {}
    by_mapping_choices: dict[str, Counter[str]] = {}
    for row in primary:
        by_split_sources.setdefault(row["split"], Counter())[row["expected_label"]] += 1
        by_mapping_choices.setdefault(row["mapping"], Counter())[row["expected_choice"]] += 1
    audits = [prompt_audit(row) for row in records]
    collisions = [
        row["id"]
        for row in records
        if len({candidate["answer"] for candidate in row["candidate_answers"]}) != len(row["candidate_answers"])
    ]
    status_rows = [
        row["id"]
        for row, audit in zip(records, audits, strict=True)
        if audit["primary_prompt_has_status_lexeme"]
    ]
    suffixes_by_order = {
        order: {"\n".join(row["prompt"].splitlines()[-3:]) for row in records if row["option_order"] == order}
        for order in OPTION_ORDERS
    }
    criteria = {
        "expected_row_count": len(records) == expected_count,
        "all_panels_present": set(PANELS) == {row["panel"] for row in records},
        "all_templates_present": set(templates) == {row["template"] for row in records},
        "all_mappings_present": set(MAPPINGS) == {row["mapping"] for row in records},
        "all_definition_orders_present": set(DEFINITION_ORDERS) == {row["definition_order"] for row in records},
        "all_option_orders_present": set(OPTION_ORDERS) == {row["option_order"] for row in records},
        "source_split_disjoint": all(len(splits) == 1 for splits in splits_by_source.values()),
        "holdout_sources_present": any(row["split"] == "holdout" for row in records),
        "calibration_sources_present": any(row["split"] == "calibration" for row in records),
        "real_atomic_number_hidden_in_conflicts": all(audit["real_atomic_number_hidden_in_conflict"] for audit in audits),
        "lure_atomic_number_hidden_in_conflicts": all(audit["lure_atomic_number_hidden_in_conflict"] for audit in audits),
        "answer_absent_omits_query_local_number": all(audit["answer_absent_omits_query_local_number"] for audit in audits),
        "candidate_answers_parseable": all(candidate["answer"] in {"A", "B", "UNKNOWN"} for row in records for candidate in row["candidate_answers"]),
        "no_candidate_collisions": not collisions,
        "single_suffix_per_option_order": all(len(suffixes) == 1 for suffixes in suffixes_by_order.values()),
        "primary_prompts_have_no_status_lexemes": not status_rows,
        "primary_expected_sources_balanced": expected_sources["local_selector"] == expected_sources["atomic_selector"],
        "primary_expected_choices_balanced": expected_choices["choice_a"] == expected_choices["choice_b"],
        "primary_expected_sources_balanced_by_split": all(
            counts["local_selector"] == counts["atomic_selector"] for counts in by_split_sources.values()
        ),
        "primary_expected_choices_balanced_by_mapping": all(
            counts["choice_a"] == counts["choice_b"] for counts in by_mapping_choices.values()
        ),
    }
    return {
        "passed": all(criteria.values()),
        "record_count": len(records),
        "source_count": len(source_ids),
        "expected_record_count": expected_count,
        "split_row_counts": dict(sorted(split_rows.items())),
        "criteria": criteria,
        "candidate_collision_rows": collisions[:20],
        "status_lexeme_rows": status_rows[:20],
        "primary_expected_source_counts": dict(sorted(expected_sources.items())),
        "primary_expected_choice_counts": dict(sorted(expected_choices.items())),
        "primary_expected_source_counts_by_split": {
            split: dict(sorted(counts.items())) for split, counts in sorted(by_split_sources.items())
        },
        "primary_expected_choice_counts_by_mapping": {
            mapping: dict(sorted(counts.items())) for mapping, counts in sorted(by_mapping_choices.items())
        },
    }


def strict_parse(record: dict[str, Any], generated_text: str) -> dict[str, Any]:
    stripped = generated_text.strip().lstrip("`'\" ")
    first_line = stripped.splitlines()[0].strip().strip("\"'`") if stripped else ""
    match = CHOICE_RE.fullmatch(first_line)
    if not match:
        return {
            "selected_choice": None,
            "selected_choice_label": "unparsed",
            "selected_source_label": "unparsed",
            "parseable": False,
            "parse_rule": "not_bare_a_b_or_unknown",
            "first_line": first_line,
        }
    answer = match.group(1).upper()
    if answer == "UNKNOWN":
        choice_label = "unknown"
        source_label = "unknown"
    elif answer == "A":
        choice_label = "choice_a"
        source_label = record["source_a_label"]
    else:
        choice_label = "choice_b"
        source_label = record["source_b_label"]
    return {
        "selected_choice": answer,
        "selected_choice_label": choice_label,
        "selected_source_label": source_label,
        "parseable": True,
        "parse_rule": "strict_bare_a_b_or_unknown",
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
    local_choice = choice_for_source(record["mapping"], "local_selector")
    atomic_choice = choice_for_source(record["mapping"], "atomic_selector")
    return {
        "candidate_first_token_ids": token_ids,
        "final_next_token_logits": scores,
        "final_choice_a_minus_choice_b_logit": scores["choice_a"] - scores["choice_b"],
        "final_unknown_minus_choice_a_logit": scores["unknown"] - scores["choice_a"],
        "final_local_source_minus_atomic_source_logit": scores[local_choice] - scores[atomic_choice],
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

    local_choice = choice_for_source(record["mapping"], "local_selector")
    atomic_choice = choice_for_source(record["mapping"], "atomic_selector")
    return {
        "candidate_logprobs": scores,
        "candidate_choice_a_minus_choice_b_mean_logprob": mean("choice_a") - mean("choice_b"),
        "candidate_unknown_minus_choice_a_mean_logprob": mean("unknown") - mean("choice_a"),
        "candidate_local_source_minus_atomic_source_mean_logprob": mean(local_choice) - mean(atomic_choice),
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
            "is_binary_choice_conflict": record["panel"] in PRIMARY_CONFLICT_PANELS
            and parsed["selected_choice_label"] in {"choice_a", "choice_b"},
            "selected_first_listed_choice": parsed["selected_choice_label"] == record["first_listed_choice"],
            "expected_correct": parsed["selected_source_label"] == record["expected_label"],
            "expected_choice_correct": parsed["selected_choice_label"] == record["expected_choice"],
        }
        if score_candidates:
            output.update(final_next_token_logits(model, tokenizer, record["prompt"], record))
            output.update(candidate_logprob_payload(model, tokenizer, record["prompt"], record))
        outputs.append(output)
        if verbose:
            print(
                f"[{index:04d}/{len(records):04d}] {record['id']} split={record['split']} "
                f"panel={record['panel']} mapping={record['mapping']} order={record['option_order']} "
                f"-> {output['selected_choice_label']}/{output['selected_source_label']} "
                f"expected={record['expected_choice']}/{record['expected_label']} "
                f"generated={generated['generated_text']!r}"
            )
    return outputs


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def label_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parseable = sum(1 for row in rows if row.get("parseable"))
    choice_a = sum(1 for row in rows if row.get("selected_choice_label") == "choice_a")
    choice_b = sum(1 for row in rows if row.get("selected_choice_label") == "choice_b")
    first_listed = sum(1 for row in rows if row.get("selected_first_listed_choice"))
    local = sum(1 for row in rows if row.get("selected_source_label") == "local_selector")
    atomic = sum(1 for row in rows if row.get("selected_source_label") == "atomic_selector")
    unknown = sum(1 for row in rows if row.get("selected_source_label") == "unknown")
    unparsed = sum(1 for row in rows if row.get("selected_source_label") == "unparsed")
    expected_correct = sum(1 for row in rows if row.get("expected_correct"))
    expected_choice_correct = sum(1 for row in rows if row.get("expected_choice_correct"))
    binary_choice = sum(1 for row in rows if row.get("is_binary_choice_conflict"))
    return {
        "rows": len(rows),
        "choice_label_counts": label_counts(rows, "selected_choice_label"),
        "source_label_counts": label_counts(rows, "selected_source_label"),
        "parseable": parseable,
        "parseable_rate": rate(parseable, len(rows)),
        "choice_a": choice_a,
        "choice_a_rate": rate(choice_a, len(rows)),
        "choice_b": choice_b,
        "choice_b_rate": rate(choice_b, len(rows)),
        "first_listed_choice": first_listed,
        "first_listed_choice_rate": rate(first_listed, len(rows)),
        "local_selector": local,
        "local_selector_rate": rate(local, len(rows)),
        "atomic_selector": atomic,
        "atomic_selector_rate": rate(atomic, len(rows)),
        "unknown": unknown,
        "unknown_rate": rate(unknown, len(rows)),
        "unparsed": unparsed,
        "unparsed_rate": rate(unparsed, len(rows)),
        "binary_choice_conflict": binary_choice,
        "expected_correct": expected_correct,
        "expected_correct_rate": rate(expected_correct, len(rows)),
        "expected_choice_correct": expected_choice_correct,
        "expected_choice_correct_rate": rate(expected_choice_correct, len(rows)),
        "expected_source_counts": dict(sorted(Counter(str(row.get("expected_label")) for row in rows).items())),
        "expected_choice_counts": dict(sorted(Counter(str(row.get("expected_choice")) for row in rows).items())),
    }


def margin_audits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows or "candidate_local_source_minus_atomic_source_mean_logprob" not in rows[0]:
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
        expected_local = [row for row in conflict if row["expected_label"] == "local_selector"]
        expected_atomic = [row for row in conflict if row["expected_label"] == "atomic_selector"]
        result[template] = {
            "rows": len(rows),
            "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "selected_choice_counts": label_counts(rows, "selected_choice_label"),
            "selected_source_counts": label_counts(rows, "selected_source_label"),
            "panels": panels,
            "primary_conflict": summarize_rows(conflict),
            "primary_conflict_non_holdout": summarize_rows(non_holdout),
            "primary_conflict_holdout": summarize_rows(holdout),
            "primary_conflict_expected_local": summarize_rows(expected_local),
            "primary_conflict_expected_atomic": summarize_rows(expected_atomic),
            "primary_conflict_by_mapping": grouped_summaries(conflict, "mapping"),
            "primary_conflict_by_definition_order": grouped_summaries(conflict, "definition_order"),
            "primary_conflict_by_option_order": grouped_summaries(conflict, "option_order"),
            "primary_conflict_margin_audits": margin_audits(conflict),
        }
    return result


def selection_key(item: dict[str, Any], template: str, templates: tuple[str, ...]) -> tuple[float, float, float, int, int, float, int]:
    local_control = item["panels"]["local_selector_control"]
    atomic_control = item["panels"]["atomic_selector_control"]
    null = item["panels"]["answer_absent_null"]
    expected_local = item["primary_conflict_expected_local"]
    expected_atomic = item["primary_conflict_expected_atomic"]
    conflict = item["primary_conflict"]
    non_holdout = item["primary_conflict_non_holdout"]
    holdout = item["primary_conflict_holdout"]
    control_floor = min(
        float(local_control["local_selector_rate"]),
        float(atomic_control["atomic_selector_rate"]),
        float(null["unknown_rate"]),
    )
    expected_floor = min(
        float(expected_local["local_selector_rate"]),
        float(expected_atomic["atomic_selector_rate"]),
    )
    conflict_balance = min(int(conflict["choice_a"]), int(conflict["choice_b"]))
    holdout_balance = min(int(holdout["choice_a"]), int(holdout["choice_b"]))
    anti_first_choice_bias = 1.0 - abs(float(conflict["first_listed_choice_rate"]) - 0.5)
    return (
        control_floor,
        expected_floor,
        float(conflict["expected_correct_rate"]),
        conflict_balance,
        holdout_balance,
        anti_first_choice_bias,
        -templates.index(template),
    )


def select_template(by_template: dict[str, Any], templates: tuple[str, ...]) -> dict[str, Any]:
    selected = max(templates, key=lambda template: selection_key(by_template[template], template, templates))
    return {
        "selected_template": selected,
        "selection_key": list(selection_key(by_template[selected], selected, templates)),
        "all_selection_keys": {template: list(selection_key(by_template[template], template, templates)) for template in templates},
        "rule": [
            "max local/atomic/null control floor",
            "max expected-local/expected-atomic conflict floor",
            "max primary source-correct rate",
            "max A/B choice balance",
            "max holdout A/B choice balance",
            "max anti-first-listed-choice-bias",
            "earliest template",
        ],
    }


def observed_pattern(selected: dict[str, Any]) -> str:
    conflict = selected["primary_conflict"]
    atomic_control = selected["panels"]["atomic_selector_control"]
    local_control = selected["panels"]["local_selector_control"]
    if float(conflict["choice_a_rate"]) >= 0.90:
        return "fixed_choice_a_bias"
    if float(conflict["choice_b_rate"]) >= 0.90:
        return "fixed_choice_b_bias"
    if float(conflict["first_listed_choice_rate"]) >= 0.90:
        return "first_listed_option_bias"
    if float(atomic_control["local_selector_rate"]) >= 0.90 and float(local_control["local_selector_rate"]) >= 0.90:
        return "local_source_prior"
    if float(conflict["expected_correct_rate"]) >= 0.85:
        return "source_rule_followed"
    return "mixed_or_unresolved_selector_failure"


def classify(criteria: dict[str, bool], pattern: str) -> str:
    if criteria["smoke_mode"]:
        return "smoke_only"
    if not criteria["structural_passed"]:
        return "structural_invalid"
    if not criteria["selected_prompt_audit_passed"] or not criteria["primary_prompts_have_no_status_lexemes"]:
        return "counterbalanced_selector_prompt_audit_failed"
    for key in (
        "local_selector_control_local_at_least_90p",
        "local_selector_control_parseable_at_least_95p",
        "atomic_selector_control_atomic_at_least_90p",
        "atomic_selector_control_parseable_at_least_95p",
        "answer_absent_panel_unknown_at_least_90p",
        "answer_absent_panel_parseable_at_least_95p",
    ):
        if not criteria[key]:
            return f"{key}_failed"
    if not criteria["expected_local_conflict_local_at_least_85p"]:
        return "counterbalanced_expected_local_conflict_failed"
    if not criteria["expected_atomic_conflict_atomic_at_least_85p"]:
        return "counterbalanced_expected_atomic_conflict_failed"
    if not criteria["primary_conflict_expected_correct_at_least_85p"]:
        return pattern
    if not criteria["primary_conflict_choice_balance_passed"]:
        return "counterbalanced_choice_balance_failed"
    if not criteria["candidate_and_output_margins_reported"]:
        return "counterbalanced_selector_behavior_passed_baselines_missing"
    return "counterbalanced_selector_behavior_passed"


def summarize(records: list[dict[str, Any]], outputs: list[dict[str, Any]], templates: tuple[str, ...], full_run: bool, score_candidates: bool) -> dict[str, Any]:
    structural = structural_check(records, templates)
    by_template = template_summary(outputs, templates)
    selection = select_template(by_template, templates)
    selected = by_template[selection["selected_template"]]
    local_control = selected["panels"]["local_selector_control"]
    atomic_control = selected["panels"]["atomic_selector_control"]
    null = selected["panels"]["answer_absent_null"]
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
        "primary_expected_sources_balanced": structural["criteria"]["primary_expected_sources_balanced"],
        "primary_expected_choices_balanced": structural["criteria"]["primary_expected_choices_balanced"],
        "primary_expected_sources_balanced_by_split": structural["criteria"]["primary_expected_sources_balanced_by_split"],
        "primary_expected_choices_balanced_by_mapping": structural["criteria"]["primary_expected_choices_balanced_by_mapping"],
        "selected_prompt_audit_passed": selected_prompt_audit_passed,
        "primary_prompts_have_no_status_lexemes": primary_prompts_have_no_status_lexemes,
        "local_selector_control_local_at_least_90p": float(local_control["local_selector_rate"]) >= 0.90,
        "local_selector_control_parseable_at_least_95p": float(local_control["parseable_rate"]) >= 0.95,
        "atomic_selector_control_atomic_at_least_90p": float(atomic_control["atomic_selector_rate"]) >= 0.90,
        "atomic_selector_control_parseable_at_least_95p": float(atomic_control["parseable_rate"]) >= 0.95,
        "answer_absent_panel_unknown_at_least_90p": float(null["unknown_rate"]) >= 0.90,
        "answer_absent_panel_parseable_at_least_95p": float(null["parseable_rate"]) >= 0.95,
        "expected_local_conflict_local_at_least_85p": float(expected_local["local_selector_rate"]) >= 0.85,
        "expected_atomic_conflict_atomic_at_least_85p": float(expected_atomic["atomic_selector_rate"]) >= 0.85,
        "primary_conflict_expected_correct_at_least_85p": float(conflict["expected_correct_rate"]) >= 0.85,
        "primary_conflict_binary_rows_at_least_40": int(conflict["binary_choice_conflict"]) >= 40,
        "non_holdout_conflict_choice_balance_passed": min(int(non_holdout["choice_a"]), int(non_holdout["choice_b"])) >= 10,
        "holdout_conflict_choice_balance_passed": min(int(holdout["choice_a"]), int(holdout["choice_b"])) >= 4,
        "primary_conflict_choice_balance_passed": min(int(conflict["choice_a"]), int(conflict["choice_b"])) >= 20,
        "primary_conflict_parseability_at_least_90p": float(conflict["parseable_rate"]) >= 0.90,
        "candidate_and_output_margins_reported": bool(score_candidates and selected["primary_conflict_margin_audits"]["reported"]),
        "visible_status_label_absent_by_design": primary_prompts_have_no_status_lexemes,
    }
    pattern = observed_pattern(selected)
    diagnostic_class = classify(criteria, pattern)
    behavior_gate_passed = diagnostic_class == "counterbalanced_selector_behavior_passed"
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
            for panel in ("local_selector_control", "atomic_selector_control", "selector_rule_absent_conflict", "answer_absent_null")
        },
        "selected_expected_local_conflict": summary["selected_template_summary"]["primary_conflict_expected_local"],
        "selected_expected_atomic_conflict": summary["selected_template_summary"]["primary_conflict_expected_atomic"],
        "primary_conflict_by_mapping": summary["selected_template_summary"]["primary_conflict_by_mapping"],
        "primary_conflict_by_definition_order": summary["selected_template_summary"]["primary_conflict_by_definition_order"],
        "primary_conflict_by_option_order": summary["selected_template_summary"]["primary_conflict_by_option_order"],
    }
    if output_path is not None:
        payload["output_path"] = str(output_path)
    return payload


def write_behavior_status_card(path: Path, result: dict[str, Any], output_path: Path) -> None:
    summary = result["summary"]
    selected = summary["selected_template_summary"]
    criteria = summary["criteria"]
    lines = [
        "# MC018 Counterbalanced Selector Labels Behavior Status",
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
        "  `code/mc018_counterbalanced_selector_labels.py`",
        "- result:",
        f"  `{output_path.as_posix()}`",
        "",
        "## Verdict",
        "",
    ]
    if summary["behavior_gate_passed"]:
        lines.extend(
            [
                "The counterbalanced selector-label behavior gate passed. This",
                "would make MC018 eligible for hidden-state signature search,",
                "but not yet for a mechanism claim.",
            ]
        )
    elif criteria["smoke_mode"]:
        lines.extend(
            [
                "This is a smoke or partial run. Use the observed pattern only",
                "as a diagnostic for whether a full run or prompt repair is worth",
                "doing.",
            ]
        )
    else:
        lines.extend(
            [
                "The counterbalanced selector-label behavior gate failed. Hidden",
                "state work remains forbidden for this route.",
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
    for panel in ("local_selector_control", "atomic_selector_control", "selector_rule_absent_conflict", "answer_absent_null"):
        item = selected["panels"][panel]
        lines.append(
            f"| `{panel}` | {item['rows']} | {item['parseable_rate']:.3f} | "
            f"{item['local_selector_rate']:.3f} | {item['atomic_selector_rate']:.3f} | {item['unknown_rate']:.3f} |"
        )
    conflict = selected["primary_conflict"]
    lines.extend(
        [
            "",
            "## Primary Conflict",
            "",
            f"- rows: {conflict['rows']}",
            f"- parseable rate: {conflict['parseable_rate']:.3f}",
            f"- A rows: {conflict['choice_a']}",
            f"- B rows: {conflict['choice_b']}",
            f"- first-listed choice rate: {conflict['first_listed_choice_rate']:.3f}",
            f"- local-source rows: {conflict['local_selector']}",
            f"- atomic-source rows: {conflict['atomic_selector']}",
            f"- expected-correct rows: {conflict['expected_correct']}",
            f"- expected-correct rate: {conflict['expected_correct_rate']:.3f}",
            "",
            "## Claim Boundary",
            "",
            "MC018 is an answer-interface diagnostic. Passing it would only",
            "show that a neutral, counterbalanced selector behavior substrate",
            "exists. Failing it names the remaining output-interface bias before",
            "any hidden-state claim is attempted.",
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
    parser.add_argument("--score-model", action="store_true")
    parser.add_argument("--no-score-candidates", action="store_true")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--artifact-prefix", default="mc018_counterbalanced_selector_labels_structural")
    args = parser.parse_args()

    run_type = BEHAVIOR_RUN_TYPE if args.score_model else RUN_TYPE
    records = source_records(limit_sources=args.limit_sources, run_type=run_type)
    structural = structural_check(records, TEMPLATES)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if args.write_manifest and not args.score_model:
        artifact = {
            "schema_version": 1,
            "card_id": CARD_ID,
            "run_type": RUN_TYPE,
            "model_id": MODEL_ID,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "purpose": "Structural gate for counterbalanced neutral selector-label behavior.",
            "templates": list(TEMPLATES),
            "mappings": list(MAPPINGS),
            "definition_orders": list(DEFINITION_ORDERS),
            "option_orders": list(OPTION_ORDERS),
            "panels": list(PANELS),
            "primary_conflict_panels": list(PRIMARY_CONFLICT_PANELS),
            "structural": structural,
            "records": records,
        }
        output_path = RESULT_DIR / f"{args.artifact_prefix}_{time.strftime('%Y%m%dT%H%M%S')}.json"
        output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        print(json.dumps({"passed": structural["passed"], **{k: structural[k] for k in ("record_count", "source_count", "criteria")}, "output_path": str(output_path)}, indent=2))
        return 0 if structural["passed"] else 1

    if not args.score_model:
        print(json.dumps(structural, indent=2))
        return 0 if structural["passed"] else 1

    model, tokenizer = load_model_and_tokenizer(args.model_id, args.local_files_only)
    started = time.time()
    outputs = score_records(
        records,
        tokenizer,
        model,
        max_new_tokens=args.max_new_tokens,
        score_candidates=not args.no_score_candidates,
        verbose=not args.quiet,
    )
    elapsed = time.time() - started
    full_run = args.limit_sources is None
    summary = summarize(records, outputs, TEMPLATES, full_run=full_run, score_candidates=not args.no_score_candidates)
    artifact_prefix = args.artifact_prefix
    if artifact_prefix == "mc018_counterbalanced_selector_labels_structural":
        artifact_prefix = "mc018_counterbalanced_selector_labels_behavior"
    output_path = RESULT_DIR / f"{artifact_prefix}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    artifact = {
        "schema_version": 1,
        "card_id": CARD_ID,
        "run_type": BEHAVIOR_RUN_TYPE,
        "model_id": args.model_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "max_new_tokens": args.max_new_tokens,
        "decoding": {"do_sample": False},
        "score_candidates": not args.no_score_candidates,
        "limit_sources": args.limit_sources,
        "templates": list(TEMPLATES),
        "mappings": list(MAPPINGS),
        "definition_orders": list(DEFINITION_ORDERS),
        "option_orders": list(OPTION_ORDERS),
        "elapsed_s": elapsed,
        "purpose": "Generated-answer counterbalanced selector-label behavior scoring before hidden-state work.",
        "panels": list(PANELS),
        "primary_conflict_panels": list(PRIMARY_CONFLICT_PANELS),
        "structural": structural,
        "summary": summary,
        "records": outputs,
    }
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    write_behavior_status_card(STATUS_CARD, artifact, output_path)
    print(json.dumps(quiet_summary(summary, output_path), indent=2))
    return 0 if (summary["behavior_gate_passed"] or not full_run) else 2


if __name__ == "__main__":
    raise SystemExit(main())
