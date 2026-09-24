#!/usr/bin/env python
"""MC017 selector-token numeric arbitration behavior gate.

MC016 showed that a visible non-status alphabet gate still collapsed to local
numeric answers when the model had to emit the controlling number directly.
MC017 changes the answer interface instead:

    choose the controlling source token first: LOCAL or ATOMIC.

The prompt still contains the local lab-number table and hides target/lure
atomic numbers in primary conflict prompts. This is a behavior-gate test of
answer-interface pressure, not a mechanism claim.
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


CARD_ID = "MC017"
RUN_TYPE = "selector_token_numeric_arbitration_structural_gate"
BEHAVIOR_RUN_TYPE = "selector_token_numeric_arbitration_behavior"
RESULT_DIR = Path("results/cards/MC017")
STATUS_CARD = Path("research/cards/MC017_SELECTOR_TOKEN_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md")

PANELS = (
    "synthetic_numeric_lookup",
    "familiar_entity_numeric_lookup",
    "real_world_atomic_number_control",
    "local_selector_control",
    "atomic_selector_control",
    "local_if_a_m_selector_conflict",
    "local_if_n_z_selector_conflict",
    "selector_rule_absent_conflict",
    "answer_absent_null",
)
PRIMARY_CONFLICT_PANELS = ("local_if_a_m_selector_conflict", "local_if_n_z_selector_conflict")
CONFLICT_PANELS = PRIMARY_CONFLICT_PANELS + ("selector_rule_absent_conflict",)
TEMPLATES = ("selector_rule", "compact_selector", "feature_labeled_selector")
STATUS_LEXEME_RE = re.compile(r"\b(trusted|untrusted|reliable|unreliable|status)\b", re.IGNORECASE)
SELECTOR_RE = re.compile(r"^(LOCAL|ATOMIC|UNKNOWN)[\s\.,;:!\?]*$", re.IGNORECASE)


def is_a_to_m(source: dict[str, Any]) -> bool:
    return source["element"][0].upper() <= "M"


def numeric_suffix() -> list[str]:
    return [
        "Return only one final token: an integer or UNKNOWN.",
        "Do not explain.",
        "Answer:",
    ]


def selector_suffix() -> list[str]:
    return [
        "Return only one final token: LOCAL, ATOMIC, or UNKNOWN.",
        "Do not explain.",
        "Answer:",
    ]


def render_local_table(
    rows: list[dict[str, Any]],
    query: dict[str, Any],
    *,
    key_field: str,
    include_query: bool,
    template: str,
) -> list[str]:
    visible_rows = rows if include_query else [row for row in rows if row["source_id"] != query["source_id"]]
    if template == "selector_rule":
        intro = [
            "A local lab table may define temporary lab numbers.",
            "Standard chemistry may define atomic numbers.",
        ]
    elif template == "compact_selector":
        intro = [
            "Two numeric sources may conflict.",
            "Use the selector rule to choose the source before any number.",
        ]
    elif template == "feature_labeled_selector":
        intro = [
            "A local lab table gives temporary lab numbers.",
            "A neutral letter-group feature may decide which source controls.",
        ]
    else:
        raise ValueError(f"unknown template {template}")

    label = "Synthetic-key lab-number table:" if key_field == "synthetic_key" else "Element lab-number table:"
    lines = [*intro, label]
    for row in visible_rows:
        lines.append(f"{row[key_field]} has local lab number {row['local_number']}.")
    return lines


def append_letter_group_feature(lines: list[str], source: dict[str, Any], *, template: str) -> None:
    if template != "feature_labeled_selector":
        return
    group = "A through M" if is_a_to_m(source) else "N through Z"
    lines.append(f"Query letter group: {group}.")


def append_selector_rule(lines: list[str], *, local_on: str, template: str) -> None:
    if local_on not in {"a_m", "n_z"}:
        raise ValueError(f"unknown selector gate {local_on}")
    local_range = "A through M" if local_on == "a_m" else "N through Z"
    opposite_range = "N through Z" if local_on == "a_m" else "A through M"
    if template == "selector_rule":
        lines.extend(
            [
                "First inspect the first letter of the queried element name.",
                f"If the first letter is from {local_range}, the controlling source is LOCAL.",
                f"If the first letter is from {opposite_range}, the controlling source is ATOMIC.",
            ]
        )
    elif template == "compact_selector":
        lines.extend(
            [
                f"Rule: first letter {local_range} -> LOCAL.",
                f"Rule: first letter {opposite_range} -> ATOMIC.",
            ]
        )
    elif template == "feature_labeled_selector":
        lines.extend(
            [
                f"Rule: query letter group {local_range} -> LOCAL.",
                f"Rule: query letter group {opposite_range} -> ATOMIC.",
            ]
        )
    else:
        raise ValueError(f"unknown template {template}")


def expected_selector_label(source: dict[str, Any], *, local_on: str) -> str:
    first_half = is_a_to_m(source)
    local_expected = first_half if local_on == "a_m" else not first_half
    return "local_selector" if local_expected else "atomic_selector"


def make_prompt(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    panel: str,
    template: str,
) -> tuple[str, str, str]:
    rows = distractors(sources, source, 4)
    if panel == "synthetic_numeric_lookup":
        lines = render_local_table(rows, source, key_field="synthetic_key", include_query=True, template=template)
        lines.append(f"Query key: {source['synthetic_key']}")
        lines.append("Return the local lab number.")
        return "\n".join([*lines, *numeric_suffix()]), "local_number", "numeric"
    if panel == "familiar_entity_numeric_lookup":
        lines = render_local_table(rows, source, key_field="element", include_query=True, template=template)
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the local lab number.")
        return "\n".join([*lines, *numeric_suffix()]), "local_number", "numeric"
    if panel == "real_world_atomic_number_control":
        lines = [
            "No local lab table is active.",
            "Return the standard atomic number for the queried element.",
            f"Query element: {source['element']}",
        ]
        return "\n".join([*lines, *numeric_suffix()]), "atomic_number", "numeric"
    if panel == "local_selector_control":
        lines = render_local_table(rows, source, key_field="element", include_query=True, template=template)
        lines.append("For this control panel, the local lab table is the active source.")
        lines.append(f"Query element: {source['element']}")
        lines.append("Which source controls the answer?")
        return "\n".join([*lines, *selector_suffix()]), "local_selector", "selector"
    if panel == "atomic_selector_control":
        lines = [
            "No local lab table is active.",
            "Standard chemistry is the active source.",
            f"Query element: {source['element']}",
            "Which source controls the answer?",
        ]
        return "\n".join([*lines, *selector_suffix()]), "atomic_selector", "selector"
    if panel == "local_if_a_m_selector_conflict":
        lines = render_local_table(rows, source, key_field="element", include_query=True, template=template)
        append_letter_group_feature(lines, source, template=template)
        lines.append("Use the selector rule below before choosing any number.")
        append_selector_rule(lines, local_on="a_m", template=template)
        lines.append(f"Query element: {source['element']}")
        lines.append("Which source controls the answer?")
        return "\n".join([*lines, *selector_suffix()]), expected_selector_label(source, local_on="a_m"), "selector"
    if panel == "local_if_n_z_selector_conflict":
        lines = render_local_table(rows, source, key_field="element", include_query=True, template=template)
        append_letter_group_feature(lines, source, template=template)
        lines.append("Use the selector rule below before choosing any number.")
        append_selector_rule(lines, local_on="n_z", template=template)
        lines.append(f"Query element: {source['element']}")
        lines.append("Which source controls the answer?")
        return "\n".join([*lines, *selector_suffix()]), expected_selector_label(source, local_on="n_z"), "selector"
    if panel == "selector_rule_absent_conflict":
        lines = render_local_table(rows, source, key_field="element", include_query=True, template=template)
        lines.append("No selector rule is available.")
        lines.append("The local lab source and standard chemistry source may conflict.")
        lines.append(f"Query element: {source['element']}")
        lines.append("Which source controls the answer?")
        return "\n".join([*lines, *selector_suffix()]), "ambiguous", "selector"
    if panel == "answer_absent_null":
        lines = render_local_table(rows, source, key_field="element", include_query=False, template=template)
        lines.append(f"Query element: {source['element']}")
        lines.append("This is a local-table membership check, not a chemistry question.")
        lines.append("If the query element is absent from the local lab table, return UNKNOWN.")
        return "\n".join([*lines, *selector_suffix()]), "unknown", "selector"
    raise ValueError(f"unknown panel {panel}")


def candidate_answers_for(record: dict[str, Any]) -> list[dict[str, str]]:
    if record["answer_type"] == "numeric":
        return [
            {"answer": record["local_number"], "label": "local_number", "candidate_type": "prompt_local_lab_number"},
            {"answer": record["atomic_number"], "label": "atomic_number", "candidate_type": "real_world_atomic_number"},
            {"answer": record["lure_atomic_number"], "label": "lure_atomic_number", "candidate_type": "nearby_atomic_number_lure"},
            {"answer": "UNKNOWN", "label": "unknown", "candidate_type": "unknown"},
        ]
    return [
        {"answer": "LOCAL", "label": "local_selector", "candidate_type": "prompt_local_source_selector"},
        {"answer": "ATOMIC", "label": "atomic_selector", "candidate_type": "real_world_source_selector"},
        {"answer": "UNKNOWN", "label": "unknown", "candidate_type": "unknown"},
    ]


def source_records(
    limit_sources: int | None = None,
    templates: tuple[str, ...] = TEMPLATES,
    run_type: str = RUN_TYPE,
) -> list[dict[str, Any]]:
    sources = base_sources(limit_sources)
    records = []
    for source in sources:
        for template in templates:
            for panel in PANELS:
                prompt, expected_label, answer_type = make_prompt(sources, source, panel=panel, template=template)
                record = {
                    "id": f"{CARD_ID}_{template}_{panel}_{source['source_id']}",
                    "card_id": CARD_ID,
                    "run_type": run_type,
                    "model_id": MODEL_ID,
                    "template": template,
                    "panel": panel,
                    "split": source["split"],
                    "source_id": source["source_id"],
                    "source_index": source["source_index"],
                    "element": source["element"],
                    "synthetic_key": source["synthetic_key"],
                    "atomic_number": str(source["atomic_number"]),
                    "lure_atomic_number": str(source["lure_atomic_number"]),
                    "local_number": str(source["local_number"]),
                    "answer_type": answer_type,
                    "expected_label": expected_label,
                    "expected_local_answer": str(source["local_number"]),
                    "expected_real_answer": str(source["atomic_number"]),
                    "expected_null_answer": "UNKNOWN",
                    "prompt": prompt,
                }
                record["candidate_answers"] = candidate_answers_for(record)
                records.append(record)
    return records


def structural_check(records: list[dict[str, Any]], templates: tuple[str, ...]) -> dict[str, Any]:
    source_ids = {row["source_id"] for row in records}
    panels = Counter(row["panel"] for row in records)
    template_counts = Counter(row["template"] for row in records)
    split_source_ids: dict[str, set[str]] = {}
    for row in records:
        split_source_ids.setdefault(row["split"], set()).add(row["source_id"])

    conflict_rows = [row for row in records if row["panel"] in CONFLICT_PANELS]
    primary_rows = [row for row in records if row["panel"] in PRIMARY_CONFLICT_PANELS]
    null_rows = [row for row in records if row["panel"] == "answer_absent_null"]
    primary_expected_label_counts = Counter(row["expected_label"] for row in primary_rows)
    primary_expected_label_counts_by_split = {
        split: Counter(row["expected_label"] for row in primary_rows if row["split"] == split)
        for split in split_source_ids
    }
    real_number_conflict_leaks = [
        row["id"] for row in conflict_rows if word_occurrences(row["prompt"], row["atomic_number"]) > 0
    ]
    lure_number_conflict_leaks = [
        row["id"] for row in conflict_rows if word_occurrences(row["prompt"], row["lure_atomic_number"]) > 0
    ]
    null_query_present = [
        row["id"] for row in null_rows if line_contains_both(row["prompt"], row["element"], row["local_number"])
    ]
    malformed_candidates = [
        row["id"]
        for row in records
        for candidate in row["candidate_answers"]
        if not candidate["answer"] or "\n" in candidate["answer"]
    ]
    candidate_collision_rows = [
        row["id"] for row in records if len({candidate["answer"] for candidate in row["candidate_answers"]}) != len(row["candidate_answers"])
    ]
    status_lexeme_rows = [row["id"] for row in primary_rows if STATUS_LEXEME_RE.search(row["prompt"])]
    suffixes_by_type = {
        answer_type: {"\n".join(row["prompt"].splitlines()[-3:]) for row in records if row["answer_type"] == answer_type}
        for answer_type in {row["answer_type"] for row in records}
    }
    expected_per_template_source = len(PANELS)
    expected_rows = len(source_ids) * len(templates) * expected_per_template_source
    criteria = {
        "expected_row_count": len(records) == expected_rows,
        "all_panels_present": set(panels) == set(PANELS),
        "all_templates_present": set(template_counts) == set(templates),
        "source_split_disjoint": sum(len(ids) for ids in split_source_ids.values()) == len(source_ids),
        "holdout_sources_present": bool(split_source_ids.get("holdout")),
        "calibration_sources_present": bool(split_source_ids.get("calibration")),
        "real_atomic_number_hidden_in_conflicts": not real_number_conflict_leaks,
        "lure_atomic_number_hidden_in_conflicts": not lure_number_conflict_leaks,
        "answer_absent_omits_query_local_number": not null_query_present,
        "candidate_answers_parseable": not malformed_candidates,
        "no_candidate_collisions": not candidate_collision_rows,
        "single_answer_suffix_per_answer_type": all(len(suffixes) == 1 for suffixes in suffixes_by_type.values()),
        "primary_prompts_have_no_status_lexemes": not status_lexeme_rows,
        "primary_expected_labels_balanced": primary_expected_label_counts
        == {"local_selector": len(primary_rows) // 2, "atomic_selector": len(primary_rows) // 2},
        "primary_expected_labels_balanced_by_split": all(
            counts.get("local_selector", 0) == counts.get("atomic_selector", 0)
            and set(counts).issubset({"local_selector", "atomic_selector"})
            for counts in primary_expected_label_counts_by_split.values()
        ),
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "record_count": len(records),
        "source_count": len(source_ids),
        "panel_counts": dict(sorted(panels.items())),
        "template_counts": dict(sorted(template_counts.items())),
        "split_source_counts": {split: len(ids) for split, ids in sorted(split_source_ids.items())},
        "expected_rows": expected_rows,
        "real_number_conflict_leak_ids": real_number_conflict_leaks[:20],
        "lure_number_conflict_leak_ids": lure_number_conflict_leaks[:20],
        "null_query_present_ids": null_query_present[:20],
        "malformed_candidate_ids": malformed_candidates[:20],
        "candidate_collision_ids": candidate_collision_rows[:20],
        "status_lexeme_rows": status_lexeme_rows[:20],
        "suffix_counts_by_answer_type": {key: len(value) for key, value in sorted(suffixes_by_type.items())},
        "primary_expected_label_counts": dict(sorted(primary_expected_label_counts.items())),
        "primary_expected_label_counts_by_split": {
            split: dict(sorted(counts.items())) for split, counts in sorted(primary_expected_label_counts_by_split.items())
        },
    }


def strict_parse(record: dict[str, Any], generated_text: str) -> dict[str, Any]:
    stripped = generated_text.strip().lstrip("`'\" ")
    first_line = stripped.splitlines()[0].strip().strip("\"'`") if stripped else ""
    if record["answer_type"] == "numeric":
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

    match = SELECTOR_RE.fullmatch(first_line)
    if not match:
        return {
            "selected_label": "unparsed",
            "selected_answer": None,
            "parseable": False,
            "parse_rule": "not_bare_selector_or_unknown",
            "first_line": first_line,
        }
    answer = match.group(1).upper()
    label = {
        "LOCAL": "local_selector",
        "ATOMIC": "atomic_selector",
        "UNKNOWN": "unknown",
    }[answer]
    return {
        "selected_label": label,
        "selected_answer": answer,
        "parseable": True,
        "parse_rule": "strict_bare_selector_or_unknown",
        "first_line": first_line,
    }


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
    payload = {
        "candidate_first_token_ids": token_ids,
        "final_next_token_logits": scores,
    }
    if "local_selector" in scores and "atomic_selector" in scores:
        payload["final_local_selector_minus_atomic_selector_logit"] = scores["local_selector"] - scores["atomic_selector"]
        payload["final_unknown_minus_local_selector_logit"] = scores["unknown"] - scores["local_selector"]
    if "local_number" in scores and "atomic_number" in scores:
        payload["final_local_minus_atomic_number_logit"] = scores["local_number"] - scores["atomic_number"]
        payload["final_local_minus_lure_atomic_number_logit"] = scores["local_number"] - scores["lure_atomic_number"]
        payload["final_unknown_minus_local_logit"] = scores["unknown"] - scores["local_number"]
    return payload


def candidate_logprob_payload(model: Any, tokenizer: Any, prompt: str, record: dict[str, Any]) -> dict[str, Any]:
    from mc006_parametric_fact_override_v15_parser_normalized_signature import token_logprob

    scores = {
        candidate["label"]: token_logprob(model, tokenizer, prompt, candidate["answer"])
        for candidate in record["candidate_answers"]
    }

    def mean(label: str) -> float:
        value = scores[label]["mean_logprob"]
        return float(value) if math.isfinite(float(value)) else float("-inf")

    payload = {"candidate_logprobs": scores}
    if "local_selector" in scores and "atomic_selector" in scores:
        payload["candidate_local_selector_minus_atomic_selector_mean_logprob"] = mean("local_selector") - mean("atomic_selector")
        payload["candidate_unknown_minus_local_selector_mean_logprob"] = mean("unknown") - mean("local_selector")
    if "local_number" in scores and "atomic_number" in scores:
        payload["candidate_local_minus_atomic_number_mean_logprob"] = mean("local_number") - mean("atomic_number")
        payload["candidate_local_minus_lure_atomic_number_mean_logprob"] = mean("local_number") - mean("lure_atomic_number")
        payload["candidate_unknown_minus_local_mean_logprob"] = mean("unknown") - mean("local_number")
    return payload


def score_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    score_candidates: bool,
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
            "is_binary_selector_conflict": record["panel"] in PRIMARY_CONFLICT_PANELS
            and parsed["selected_label"] in {"local_selector", "atomic_selector"},
            "expected_correct": parsed["selected_label"] == record["expected_label"],
        }
        if score_candidates:
            output.update(final_next_token_logits(model, tokenizer, record["prompt"], record))
            output.update(candidate_logprob_payload(model, tokenizer, record["prompt"], record))
        outputs.append(output)
        if verbose:
            print(
                f"[{index:04d}/{len(records):04d}] {record['id']} split={record['split']} "
                f"panel={record['panel']} -> {output['selected_label']} "
                f"{str(output['selected_answer'])!r} generated={generated['generated_text']!r}"
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
    local_selector = sum(1 for row in rows if row.get("selected_label") == "local_selector")
    atomic_selector = sum(1 for row in rows if row.get("selected_label") == "atomic_selector")
    unknown = sum(1 for row in rows if row.get("selected_label") == "unknown")
    other_number = sum(1 for row in rows if row.get("selected_label") == "other_number")
    unparsed = sum(1 for row in rows if row.get("selected_label") == "unparsed")
    expected_correct = sum(1 for row in rows if row.get("expected_correct"))
    expected_labels = Counter(str(row.get("expected_label")) for row in rows)
    binary_selector = sum(1 for row in rows if row.get("is_binary_selector_conflict"))
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
        "local_selector": local_selector,
        "local_selector_rate": rate(local_selector, len(rows)),
        "atomic_selector": atomic_selector,
        "atomic_selector_rate": rate(atomic_selector, len(rows)),
        "unknown": unknown,
        "unknown_rate": rate(unknown, len(rows)),
        "other_number": other_number,
        "other_number_rate": rate(other_number, len(rows)),
        "unparsed": unparsed,
        "unparsed_rate": rate(unparsed, len(rows)),
        "binary_selector_conflict": binary_selector,
        "expected_correct": expected_correct,
        "expected_correct_rate": rate(expected_correct, len(rows)),
        "expected_label_counts": dict(sorted(expected_labels.items())),
    }


def margin_audits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows or "candidate_local_selector_minus_atomic_selector_mean_logprob" not in rows[0]:
        return {"reported": False}
    return {"reported": True}


def template_summary(outputs: list[dict[str, Any]], templates: tuple[str, ...]) -> dict[str, Any]:
    result = {}
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
            "selected_label_counts": label_counts(rows),
            "panels": panels,
            "primary_conflict": summarize_rows(conflict),
            "primary_conflict_non_holdout": summarize_rows(non_holdout),
            "primary_conflict_holdout": summarize_rows(holdout),
            "primary_conflict_expected_local": summarize_rows(expected_local),
            "primary_conflict_expected_atomic": summarize_rows(expected_atomic),
            "primary_conflict_margin_audits": margin_audits(conflict),
        }
    return result


def selection_key(item: dict[str, Any], template: str, templates: tuple[str, ...]) -> tuple[float, float, float, int, int, int]:
    synthetic = item["panels"]["synthetic_numeric_lookup"]
    familiar = item["panels"]["familiar_entity_numeric_lookup"]
    real = item["panels"]["real_world_atomic_number_control"]
    local_control = item["panels"]["local_selector_control"]
    atomic_control = item["panels"]["atomic_selector_control"]
    null = item["panels"]["answer_absent_null"]
    conflict = item["primary_conflict"]
    non_holdout = item["primary_conflict_non_holdout"]
    holdout = item["primary_conflict_holdout"]
    expected_local = item["primary_conflict_expected_local"]
    expected_atomic = item["primary_conflict_expected_atomic"]
    control_floor = min(
        float(synthetic["local_number_rate"]),
        float(familiar["local_number_rate"]),
        float(real["atomic_number_rate"]),
        float(local_control["local_selector_rate"]),
        float(atomic_control["atomic_selector_rate"]),
        float(null["unknown_rate"]),
        float(expected_local["local_selector_rate"]),
        float(expected_atomic["atomic_selector_rate"]),
    )
    conflict_balance = min(int(non_holdout["local_selector"]), int(non_holdout["atomic_selector"]))
    holdout_balance = min(int(holdout["local_selector"]), int(holdout["atomic_selector"]))
    return (
        control_floor,
        float(conflict["parseable_rate"]),
        float(conflict["expected_correct_rate"]),
        conflict_balance,
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
            "max numeric controls plus selector controls plus expected selector floor",
            "max primary conflict parseability",
            "max primary expected-label correctness",
            "max non-holdout LOCAL-versus-ATOMIC balance",
            "max holdout LOCAL-versus-ATOMIC balance",
            "earliest template",
        ],
    }


def classify(criteria: dict[str, bool]) -> str:
    if criteria["smoke_mode"]:
        return "smoke_only"
    if not criteria["structural_passed"]:
        return "structural_invalid"
    if not criteria["selected_prompt_audit_passed"] or not criteria["primary_prompts_have_no_status_lexemes"]:
        return "selector_prompt_status_leak_failed"
    for key in (
        "synthetic_panel_a_local_at_least_90p",
        "synthetic_panel_a_parseable_at_least_95p",
        "familiar_panel_b_local_at_least_90p",
        "familiar_panel_b_parseable_at_least_95p",
        "real_world_panel_c_atomic_at_least_85p",
        "real_world_panel_c_parseable_at_least_95p",
        "local_selector_control_local_at_least_90p",
        "local_selector_control_parseable_at_least_95p",
        "atomic_selector_control_atomic_at_least_90p",
        "atomic_selector_control_parseable_at_least_95p",
        "answer_absent_panel_unknown_at_least_90p",
        "answer_absent_panel_parseable_at_least_95p",
    ):
        if not criteria[key]:
            return f"{key}_failed"
    if not criteria["expected_local_conflict_local_selector_at_least_85p"]:
        return "selector_expected_local_conflict_failed"
    if not criteria["expected_atomic_conflict_atomic_selector_at_least_85p"]:
        return "selector_expected_atomic_conflict_failed"
    if not criteria["primary_conflict_expected_correct_at_least_85p"]:
        return "selector_gate_not_followed"
    if not criteria["primary_conflict_binary_rows_at_least_40"]:
        return "selector_conflict_contrast_absent"
    if not criteria["non_holdout_conflict_label_balance_passed"] or not criteria["holdout_conflict_label_balance_passed"]:
        return "selector_conflict_unbalanced"
    if not criteria["candidate_and_output_margins_reported"]:
        return "selector_behavior_passed_baselines_missing"
    return "selector_token_behavior_passed"


def summarize(records: list[dict[str, Any]], outputs: list[dict[str, Any]], templates: tuple[str, ...], full_run: bool, score_candidates: bool) -> dict[str, Any]:
    structural = structural_check(records, templates)
    by_template = template_summary(outputs, templates)
    selection = select_template(by_template, templates)
    selected = by_template[selection["selected_template"]]
    synthetic = selected["panels"]["synthetic_numeric_lookup"]
    familiar = selected["panels"]["familiar_entity_numeric_lookup"]
    real = selected["panels"]["real_world_atomic_number_control"]
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
        "primary_expected_labels_balanced": structural["criteria"]["primary_expected_labels_balanced"],
        "primary_expected_labels_balanced_by_split": structural["criteria"]["primary_expected_labels_balanced_by_split"],
        "selected_prompt_audit_passed": selected_prompt_audit_passed,
        "primary_prompts_have_no_status_lexemes": primary_prompts_have_no_status_lexemes,
        "synthetic_panel_a_local_at_least_90p": float(synthetic["local_number_rate"]) >= 0.90,
        "synthetic_panel_a_parseable_at_least_95p": float(synthetic["parseable_rate"]) >= 0.95,
        "familiar_panel_b_local_at_least_90p": float(familiar["local_number_rate"]) >= 0.90,
        "familiar_panel_b_parseable_at_least_95p": float(familiar["parseable_rate"]) >= 0.95,
        "real_world_panel_c_atomic_at_least_85p": float(real["atomic_number_rate"]) >= 0.85,
        "real_world_panel_c_parseable_at_least_95p": float(real["parseable_rate"]) >= 0.95,
        "local_selector_control_local_at_least_90p": float(local_control["local_selector_rate"]) >= 0.90,
        "local_selector_control_parseable_at_least_95p": float(local_control["parseable_rate"]) >= 0.95,
        "atomic_selector_control_atomic_at_least_90p": float(atomic_control["atomic_selector_rate"]) >= 0.90,
        "atomic_selector_control_parseable_at_least_95p": float(atomic_control["parseable_rate"]) >= 0.95,
        "answer_absent_panel_unknown_at_least_90p": float(null["unknown_rate"]) >= 0.90,
        "answer_absent_panel_parseable_at_least_95p": float(null["parseable_rate"]) >= 0.95,
        "expected_local_conflict_local_selector_at_least_85p": float(expected_local["local_selector_rate"]) >= 0.85,
        "expected_atomic_conflict_atomic_selector_at_least_85p": float(expected_atomic["atomic_selector_rate"]) >= 0.85,
        "primary_conflict_expected_correct_at_least_85p": float(conflict["expected_correct_rate"]) >= 0.85,
        "primary_conflict_binary_rows_at_least_40": int(conflict["binary_selector_conflict"]) >= 40,
        "non_holdout_conflict_local_at_least_10": int(non_holdout["local_selector"]) >= 10,
        "non_holdout_conflict_atomic_at_least_10": int(non_holdout["atomic_selector"]) >= 10,
        "holdout_conflict_local_at_least_4": int(holdout["local_selector"]) >= 4,
        "holdout_conflict_atomic_at_least_4": int(holdout["atomic_selector"]) >= 4,
        "primary_conflict_parseability_at_least_90p": float(conflict["parseable_rate"]) >= 0.90,
        "candidate_and_output_margins_reported": bool(score_candidates and selected["primary_conflict_margin_audits"]["reported"]),
        "visible_status_label_absent_by_design": primary_prompts_have_no_status_lexemes,
    }
    criteria["non_holdout_conflict_label_balance_passed"] = (
        criteria["non_holdout_conflict_local_at_least_10"] and criteria["non_holdout_conflict_atomic_at_least_10"]
    )
    criteria["holdout_conflict_label_balance_passed"] = (
        criteria["holdout_conflict_local_at_least_4"] and criteria["holdout_conflict_atomic_at_least_4"]
    )
    diagnostic_class = classify(criteria)
    behavior_gate_passed = diagnostic_class == "selector_token_behavior_passed"
    return {
        "structural": structural,
        "by_template": by_template,
        "selection": selection,
        "selected_template_summary": selected,
        "selected_template_rows": selected_rows,
        "criteria": criteria,
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
                "local_selector_control",
                "atomic_selector_control",
                "selector_rule_absent_conflict",
                "answer_absent_null",
            )
        },
        "selected_expected_local_conflict": summary["selected_template_summary"]["primary_conflict_expected_local"],
        "selected_expected_atomic_conflict": summary["selected_template_summary"]["primary_conflict_expected_atomic"],
    }
    if output_path is not None:
        payload["output_path"] = str(output_path)
    return payload


def write_behavior_status_card(path: Path, result: dict[str, Any], output_path: Path) -> None:
    summary = result["summary"]
    selected = summary["selected_template_summary"]
    criteria = summary["criteria"]
    lines = [
        "# MC017 Selector-Token Numeric Arbitration Behavior Status",
        "",
        f"Status: {summary['diagnostic_class']}.",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "## Artifact",
        "",
        "- runner:",
        "  `code/mc017_selector_token_numeric_arbitration.py`",
        "- result:",
        f"  `{output_path.as_posix()}`",
        "",
        "## Verdict",
        "",
    ]
    if summary["behavior_gate_passed"]:
        lines.extend(
            [
                "The selector-token behavior gate passed. This is evidence that",
                "the MC016 numeric collapse is at least partly answer-interface",
                "pressure, but it is still not a hidden-state mechanism claim.",
            ]
        )
    elif criteria["smoke_mode"]:
        lines.extend(
            [
                "This is a smoke or partial run. It validates runner plumbing only;",
                "it is not evidence for or against the full MC017 behavior substrate.",
            ]
        )
    else:
        lines.extend(
            [
                "The selector-token behavior gate failed. Hidden-state work remains",
                "forbidden for this route.",
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
            "| Panel | Rows | Parseable | Key Label Rate |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    controls = [
        ("synthetic_numeric_lookup", "local_number_rate"),
        ("familiar_entity_numeric_lookup", "local_number_rate"),
        ("real_world_atomic_number_control", "atomic_number_rate"),
        ("local_selector_control", "local_selector_rate"),
        ("atomic_selector_control", "atomic_selector_rate"),
        ("selector_rule_absent_conflict", "local_selector_rate"),
        ("answer_absent_null", "unknown_rate"),
    ]
    for panel, key in controls:
        item = selected["panels"][panel]
        lines.append(f"| `{panel}` | {item['rows']} | {item['parseable_rate']:.3f} | {item[key]:.3f} |")
    conflict = selected["primary_conflict"]
    lines.extend(
        [
            "",
            "## Primary Conflict",
            "",
            f"- rows: {conflict['rows']}",
            f"- parseable rate: {conflict['parseable_rate']:.3f}",
            f"- LOCAL rows: {conflict['local_selector']}",
            f"- ATOMIC rows: {conflict['atomic_selector']}",
            f"- binary selector rows: {conflict['binary_selector_conflict']}",
            f"- expected-correct rows: {conflict['expected_correct']}",
            f"- expected-correct rate: {conflict['expected_correct_rate']:.3f}",
            "",
            "## Forbidden Claims",
            "",
            "- MC017 is a mechanism card.",
            "- MC017 supports intervention.",
            "- MC017 found an internal knowledge-control surface.",
            "- A selector-token behavior pass would by itself prove a numeric control surface.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--templates", choices=TEMPLATES, nargs="+", default=list(TEMPLATES))
    parser.add_argument("--write-manifest", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--artifact-prefix", default="mc017_selector_token_numeric_structural")
    parser.add_argument("--status-card", type=Path, default=STATUS_CARD)
    parser.add_argument("--score-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--score-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-status-card", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    started = time.time()
    templates = tuple(args.templates)
    run_type = BEHAVIOR_RUN_TYPE if args.score_model else RUN_TYPE
    records = source_records(args.limit_sources, templates, run_type)
    structural = structural_check(records, templates)

    if args.score_model:
        if not structural["passed"]:
            print(json.dumps({"passed": False, "diagnostic_class": "structural_invalid", "structural": structural}, indent=2))
            return 1
        model, tokenizer = load_model_and_tokenizer(args.model_id, args.local_files_only)
        outputs = score_records(records, tokenizer, model, args.max_new_tokens, args.score_candidates, verbose=not args.quiet)
        full_run = args.limit_sources is None and set(templates) == set(TEMPLATES)
        summary = summarize(records, outputs, templates, full_run, args.score_candidates)
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
            "templates": list(templates),
            "elapsed_s": time.time() - started,
            "purpose": "Generated-answer selector-token bridge behavior scoring before any MC017 hidden-state work.",
            "panels": list(PANELS),
            "primary_conflict_panels": list(PRIMARY_CONFLICT_PANELS),
            "sources": base_sources(args.limit_sources),
            "records": outputs,
            "summary": summary,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = args.artifact_prefix
        if prefix == "mc017_selector_token_numeric_structural":
            prefix = "mc017_selector_token_numeric_behavior"
        output_path = args.output_dir / f"{prefix}_{time.strftime('%Y%m%dT%H%M%S')}.json"
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        if args.write_status_card:
            write_behavior_status_card(args.status_card, result, output_path)
        print(json.dumps(quiet_summary(summary, output_path) if args.quiet else {**summary, "output_path": str(output_path)}, indent=2, ensure_ascii=True))
        return 0

    result = {
        "schema_version": 1,
        "card_id": CARD_ID,
        "run_type": RUN_TYPE,
        "model_id": args.model_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": "Structural prompt/channel audit before MC017 generated-answer behavior scoring.",
        "panels": PANELS,
        "primary_conflict_panels": PRIMARY_CONFLICT_PANELS,
        "templates": templates,
        "structural": structural,
        "behavior_gate_passed": False,
        "signature_ready": False,
        "intervention_ready": False,
        "diagnostic_class": "structural_gate_only",
        "records": records,
    }
    output_path = None
    if args.write_manifest:
        args.output_dir.mkdir(parents=True, exist_ok=True)
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
                "output_path": str(output_path) if output_path else None,
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
