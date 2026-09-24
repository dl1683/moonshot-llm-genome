#!/usr/bin/env python
"""MC033 fact-claim bridge closeout.

MC032 showed that replacing checksum validity with cross-table consistency did
not repair learned-branch routing. MC033 is the one preregistered repair pass
from the gap-closure plan: the local row carries a standard-number claim, and
the model must compare that claim against learned atomic-number memory. If the
claim matches the element's standard atomic number, the local lab number
controls. If it differs, the learned atomic number controls.

The design intentionally avoids source-status labels, row codes, operation
handles, worked examples, constrained choices, numeric options, answer-schema
sweeps, absence guards, arithmetic checksum cues, and cross-table consistency.
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
    line_contains_both,
    response_suffix,
    word_occurrences,
)
from mc016_alphabet_gated_numeric_arbitration import load_model_and_tokenizer


CARD_ID = "MC033"
RUN_TYPE = "fact_claim_bridge_structural_gate"
BEHAVIOR_RUN_TYPE = "fact_claim_bridge_behavior"
RESULT_DIR = Path("results/cards/MC033")
STATUS_CARD = Path("research/cards/MC033_FACT_CLAIM_BRIDGE_CLOSEOUT_STATUS.md")

PANELS = (
    "synthetic_numeric_lookup",
    "familiar_entity_numeric_lookup",
    "real_world_atomic_number_control",
    "fact_claim_match_conflict",
    "fact_claim_mismatch_conflict",
    "fact_claim_absent_conflict",
    "answer_absent_null",
)
PRIMARY_CONFLICT_PANELS = ("fact_claim_match_conflict", "fact_claim_mismatch_conflict")
CONFLICT_PANELS = PRIMARY_CONFLICT_PANELS + ("fact_claim_absent_conflict",)
TEMPLATES = ("claim_column", "field_note", "memory_comparison")

STATUS_LEXEME_RE = re.compile(
    r"\b(trusted|untrusted|reliable|unreliable|status)\b",
    re.IGNORECASE,
)
FORBIDDEN_ROUTE_RE = re.compile(
    r"\b(checksum|equation|arithmetic|valid|invalid|cross-check|mirror|second table|two tables)\b",
    re.IGNORECASE,
)


def table_intro(template: str) -> list[str]:
    if template == "claim_column":
        return [
            "A lab notebook may list temporary lab numbers.",
            "Some query rows also include a standard-number claim.",
        ]
    if template == "field_note":
        return [
            "Field notes may list temporary element numbers.",
            "A query note may include one standard-number claim.",
        ]
    if template == "memory_comparison":
        return [
            "A temporary-number ledger is active.",
            "Use learned atomic-number memory only for the comparison rule.",
        ]
    raise ValueError(f"unknown template {template}")


def local_table_lines(
    rows: list[dict[str, Any]],
    query: dict[str, Any],
    *,
    key_field: str,
    include_query: bool,
    template: str,
) -> list[str]:
    visible_rows = (
        rows
        if include_query
        else [row for row in rows if row["source_id"] != query["source_id"]]
    )
    if key_field == "synthetic_key":
        label = "Temporary key table:"
    elif template == "field_note":
        label = "Temporary field rows:"
    elif template == "memory_comparison":
        label = "Temporary-number ledger:"
    else:
        label = "Temporary element table:"
    lines = [*table_intro(template), label]
    for row in visible_rows:
        lines.append(f"{row[key_field]} has local lab number {row['local_number']}.")
    return lines


def fact_claim_line(source: dict[str, Any], claim_number: str, template: str) -> str:
    if template == "claim_column":
        return f"Standard-number claim for {source['element']}: {claim_number}."
    if template == "field_note":
        return f"Query note claims {source['element']} has standard number {claim_number}."
    if template == "memory_comparison":
        return f"Memory-comparison claim for {source['element']}: {claim_number}."
    raise ValueError(f"unknown template {template}")


def append_fact_claim_rule(lines: list[str], template: str) -> None:
    if template == "claim_column":
        lines.extend(
            [
                "If the query claim is the element's standard atomic number, return the local lab number.",
                "If the query claim is not the element's standard atomic number, return the standard atomic number.",
            ]
        )
    elif template == "field_note":
        lines.extend(
            [
                "When the query note's standard-number claim matches learned atomic memory, use the local lab number.",
                "When that claim differs from learned atomic memory, use the standard atomic number.",
            ]
        )
    elif template == "memory_comparison":
        lines.extend(
            [
                "Compare the query claim with learned atomic-number memory.",
                "Same means answer with the local lab number; different means answer with the standard atomic number.",
            ]
        )
    else:
        raise ValueError(f"unknown template {template}")


def make_prompt(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    panel: str,
    template: str,
) -> tuple[str, str, str | None]:
    rows = distractors(sources, source, 4)
    if panel == "synthetic_numeric_lookup":
        lines = local_table_lines(
            rows,
            source,
            key_field="synthetic_key",
            include_query=True,
            template=template,
        )
        lines.append(f"Query key: {source['synthetic_key']}")
        lines.append("Return the local lab number.")
        expected_label = "local_number"
        claim_number = None
    elif panel == "familiar_entity_numeric_lookup":
        lines = local_table_lines(
            rows,
            source,
            key_field="element",
            include_query=True,
            template=template,
        )
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the local lab number.")
        expected_label = "local_number"
        claim_number = None
    elif panel == "real_world_atomic_number_control":
        lines = [
            "No temporary table is active.",
            "Return the standard atomic number for the queried element.",
            f"Query element: {source['element']}",
        ]
        expected_label = "atomic_number"
        claim_number = None
    elif panel == "fact_claim_match_conflict":
        claim_number = str(source["atomic_number"])
        lines = local_table_lines(
            rows,
            source,
            key_field="element",
            include_query=True,
            template=template,
        )
        lines.append(fact_claim_line(source, claim_number, template))
        append_fact_claim_rule(lines, template)
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the controlling number.")
        expected_label = "local_number"
    elif panel == "fact_claim_mismatch_conflict":
        claim_number = str(source["lure_atomic_number"])
        lines = local_table_lines(
            rows,
            source,
            key_field="element",
            include_query=True,
            template=template,
        )
        lines.append(fact_claim_line(source, claim_number, template))
        append_fact_claim_rule(lines, template)
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the controlling number.")
        expected_label = "atomic_number"
    elif panel == "fact_claim_absent_conflict":
        lines = local_table_lines(
            rows,
            source,
            key_field="element",
            include_query=True,
            template=template,
        )
        lines.append("No standard-number claim is provided for the query row.")
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the local lab number for the visible local row.")
        expected_label = "local_number"
        claim_number = None
    elif panel == "answer_absent_null":
        lines = local_table_lines(
            rows,
            source,
            key_field="element",
            include_query=False,
            template=template,
        )
        lines.append(f"Query element: {source['element']}")
        lines.append("This is a local-table membership check, not a chemistry question.")
        lines.append(
            "If the query element is absent from the local table, return UNKNOWN even if you know its standard atomic number."
        )
        expected_label = "unknown"
        claim_number = None
    else:
        raise ValueError(f"unknown panel {panel}")
    return "\n".join([*lines, *response_suffix()]), expected_label, claim_number


def source_records(
    limit_sources: int | None = None,
    templates: tuple[str, ...] = TEMPLATES,
    run_type: str = RUN_TYPE,
) -> list[dict[str, Any]]:
    sources = base_sources(limit_sources)
    records: list[dict[str, Any]] = []
    for source in sources:
        for template in templates:
            for panel in PANELS:
                prompt, expected_label, claim_number = make_prompt(
                    sources,
                    source,
                    panel=panel,
                    template=template,
                )
                records.append(
                    {
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
                        "claim_number": claim_number,
                        "expected_label": expected_label,
                        "expected_local_answer": str(source["local_number"]),
                        "expected_real_answer": str(source["atomic_number"]),
                        "expected_null_answer": "UNKNOWN",
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
    split_source_ids: dict[str, set[str]] = {}
    for row in records:
        split_source_ids.setdefault(row["split"], set()).add(row["source_id"])

    primary_rows = [row for row in records if row["panel"] in PRIMARY_CONFLICT_PANELS]
    mismatch_rows = [row for row in records if row["panel"] == "fact_claim_mismatch_conflict"]
    null_rows = [row for row in records if row["panel"] == "answer_absent_null"]
    expected_rows = len(source_ids) * len(templates) * len(PANELS)
    primary_expected = Counter(row["expected_label"] for row in primary_rows)

    mismatch_real_leaks = [
        row["id"]
        for row in mismatch_rows
        if word_occurrences(row["prompt"], row["atomic_number"]) > 0
    ]
    mismatch_claim_missing = [
        row["id"]
        for row in mismatch_rows
        if word_occurrences(row["prompt"], row["lure_atomic_number"]) != 1
    ]
    null_query_present = [
        row["id"]
        for row in null_rows
        if line_contains_both(row["prompt"], row["element"], row["local_number"])
    ]
    malformed_candidates = [
        row["id"]
        for row in records
        for candidate in row["candidate_answers"]
        if ANSWER_RE.match(candidate) is None
    ]
    candidate_collision_rows = [
        row["id"]
        for row in records
        if len(set(row["candidate_answers"])) != len(row["candidate_answers"])
    ]
    primary_status_rows = [
        row["id"] for row in primary_rows if STATUS_LEXEME_RE.search(row["prompt"])
    ]
    primary_forbidden_route_rows = [
        row["id"] for row in primary_rows if FORBIDDEN_ROUTE_RE.search(row["prompt"])
    ]
    answer_suffixes = {"\n".join(row["prompt"].splitlines()[-3:]) for row in records}

    criteria = {
        "expected_row_count": len(records) == expected_rows,
        "all_panels_present": set(panels) == set(PANELS),
        "all_templates_present": set(template_counts) == set(templates),
        "source_split_disjoint": sum(len(ids) for ids in split_source_ids.values())
        == len(source_ids),
        "holdout_sources_present": bool(split_source_ids.get("holdout")),
        "calibration_sources_present": bool(split_source_ids.get("calibration")),
        "mismatch_real_atomic_number_hidden": not mismatch_real_leaks,
        "mismatch_claim_number_visible_once": not mismatch_claim_missing,
        "answer_absent_omits_query_local_number": not null_query_present,
        "candidate_answers_parseable": not malformed_candidates,
        "no_candidate_collisions": not candidate_collision_rows,
        "single_answer_suffix": len(answer_suffixes) == 1,
        "primary_prompts_have_no_status_lexemes": not primary_status_rows,
        "primary_prompts_avoid_closed_route_lexemes": not primary_forbidden_route_rows,
        "primary_expected_labels_balanced": (
            primary_expected["local_number"] == primary_expected["atomic_number"]
        ),
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "record_count": len(records),
        "source_count": len(source_ids),
        "panel_counts": dict(sorted(panels.items())),
        "template_counts": dict(sorted(template_counts.items())),
        "split_source_counts": {
            split: len(ids) for split, ids in sorted(split_source_ids.items())
        },
        "primary_expected_label_counts": dict(sorted(primary_expected.items())),
        "expected_rows": expected_rows,
        "mismatch_real_leak_ids": mismatch_real_leaks[:20],
        "mismatch_claim_missing_ids": mismatch_claim_missing[:20],
        "null_query_present_ids": null_query_present[:20],
        "malformed_candidate_ids": malformed_candidates[:20],
        "candidate_collision_ids": candidate_collision_rows[:20],
        "primary_status_lexeme_rows": primary_status_rows[:20],
        "primary_forbidden_route_rows": primary_forbidden_route_rows[:20],
        "answer_suffix_count": len(answer_suffixes),
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


def prompt_audit(record: dict[str, Any]) -> dict[str, Any]:
    prompt = record["prompt"]
    primary = record["panel"] in PRIMARY_CONFLICT_PANELS
    mismatch = record["panel"] == "fact_claim_mismatch_conflict"
    null = record["panel"] == "answer_absent_null"
    return {
        "real_atomic_number_occurrences": word_occurrences(prompt, record["atomic_number"]),
        "lure_atomic_number_occurrences": word_occurrences(
            prompt,
            record["lure_atomic_number"],
        ),
        "mismatch_real_atomic_number_hidden": not mismatch
        or word_occurrences(prompt, record["atomic_number"]) == 0,
        "mismatch_claim_number_visible_once": not mismatch
        or word_occurrences(prompt, record["lure_atomic_number"]) == 1,
        "answer_absent_omits_query_local_number": not null
        or not line_contains_both(prompt, record["element"], record["local_number"]),
        "primary_prompt_has_status_lexeme": bool(primary and STATUS_LEXEME_RE.search(prompt)),
        "primary_prompt_has_closed_route_lexeme": bool(
            primary and FORBIDDEN_ROUTE_RE.search(prompt)
        ),
    }


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
            "is_binary_conflict": record["panel"] in PRIMARY_CONFLICT_PANELS
            and parsed["selected_label"] in {"local_number", "atomic_number", "lure_atomic_number"},
            "is_atomic_or_lure_number": parsed["selected_label"]
            in {"atomic_number", "lure_atomic_number"},
            "is_expected_correct": parsed["selected_label"] == record["expected_label"],
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


def rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("selected_label")) for row in rows).items()))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parseable = sum(1 for row in rows if row.get("parseable"))
    local = sum(1 for row in rows if row.get("selected_label") == "local_number")
    atomic = sum(1 for row in rows if row.get("selected_label") == "atomic_number")
    lure = sum(1 for row in rows if row.get("selected_label") == "lure_atomic_number")
    unknown = sum(1 for row in rows if row.get("selected_label") == "unknown")
    other = sum(1 for row in rows if row.get("selected_label") == "other_number")
    unparsed = sum(1 for row in rows if row.get("selected_label") == "unparsed")
    binary_conflict = sum(1 for row in rows if row.get("is_binary_conflict"))
    result = {
        "rows": len(rows),
        "label_counts": label_counts(rows),
        "parseable": parseable,
        "parseable_rate": rate(parseable, len(rows)),
        "local_number": local,
        "local_number_rate": rate(local, len(rows)),
        "atomic_number": atomic,
        "atomic_number_rate": rate(atomic, len(rows)),
        "lure_atomic_number": lure,
        "lure_atomic_number_rate": rate(lure, len(rows)),
        "atomic_or_lure_number": atomic + lure,
        "atomic_or_lure_number_rate": rate(atomic + lure, len(rows)),
        "unknown": unknown,
        "unknown_rate": rate(unknown, len(rows)),
        "other_number": other,
        "other_number_rate": rate(other, len(rows)),
        "unparsed": unparsed,
        "unparsed_rate": rate(unparsed, len(rows)),
        "binary_conflict": binary_conflict,
    }
    for field in (
        "final_local_minus_atomic_number_logit",
        "final_local_minus_lure_atomic_number_logit",
        "candidate_local_minus_atomic_number_mean_logprob",
        "candidate_local_minus_lure_atomic_number_mean_logprob",
    ):
        values = [
            float(row[field])
            for row in rows
            if field in row and math.isfinite(float(row[field]))
        ]
        if values:
            result[f"{field}_mean"] = sum(values) / len(values)
            result[f"{field}_min"] = min(values)
            result[f"{field}_max"] = max(values)
    return result


def margin_audits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows or "final_local_minus_atomic_number_logit" not in rows[0]:
        return {"reported": False}
    return {"reported": True}


def template_summary(outputs: list[dict[str, Any]], templates: tuple[str, ...]) -> dict[str, Any]:
    result = {}
    for template in templates:
        rows = [row for row in outputs if row["template"] == template]
        panels = {
            panel: summarize_rows([row for row in rows if row["panel"] == panel])
            for panel in PANELS
        }
        conflict = [row for row in rows if row["panel"] in PRIMARY_CONFLICT_PANELS]
        non_holdout = [row for row in conflict if row["split"] != "holdout"]
        holdout = [row for row in conflict if row["split"] == "holdout"]
        expected_local = [row for row in conflict if row["expected_label"] == "local_number"]
        expected_atomic = [
            row for row in conflict if row["expected_label"] == "atomic_number"
        ]
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


def selection_key(
    item: dict[str, Any],
    template: str,
    templates: tuple[str, ...],
) -> tuple[float, float, float, int, int, int]:
    synthetic = item["panels"]["synthetic_numeric_lookup"]
    familiar = item["panels"]["familiar_entity_numeric_lookup"]
    real = item["panels"]["real_world_atomic_number_control"]
    null = item["panels"]["answer_absent_null"]
    match = item["panels"]["fact_claim_match_conflict"]
    mismatch = item["panels"]["fact_claim_mismatch_conflict"]
    conflict = item["primary_conflict"]
    non_holdout = item["primary_conflict_non_holdout"]
    holdout = item["primary_conflict_holdout"]
    control_floor = min(
        float(synthetic["local_number_rate"]),
        float(familiar["local_number_rate"]),
        float(real["atomic_number_rate"]),
        float(null["unknown_rate"]),
        float(match["local_number_rate"]),
        float(mismatch["atomic_number_rate"]),
    )
    conflict_balance = min(
        int(non_holdout["local_number"]),
        int(non_holdout["atomic_or_lure_number"]),
    )
    holdout_balance = min(
        int(holdout["local_number"]),
        int(holdout["atomic_or_lure_number"]),
    )
    return (
        control_floor,
        float(conflict["parseable_rate"]),
        -float(mismatch["lure_atomic_number_rate"]),
        conflict_balance,
        holdout_balance,
        -templates.index(template),
    )


def select_template(by_template: dict[str, Any], templates: tuple[str, ...]) -> dict[str, Any]:
    selected = max(
        templates,
        key=lambda template: selection_key(by_template[template], template, templates),
    )
    return {
        "selected_template": selected,
        "selection_key": list(selection_key(by_template[selected], selected, templates)),
        "all_selection_keys": {
            template: list(selection_key(by_template[template], template, templates))
            for template in templates
        },
        "rule": [
            "max direct/control plus fact-claim conflict floor",
            "max primary conflict parseability",
            "min mismatch claimed-number/lure rate",
            "max non-holdout local-versus-atomic/lure balance",
            "max holdout local-versus-atomic/lure balance",
            "earliest template",
        ],
    }


def classify(criteria: dict[str, bool]) -> str:
    if criteria["smoke_mode"]:
        return "smoke_only"
    if not criteria["structural_passed"]:
        return "structural_invalid"
    if not criteria["selected_prompt_audit_passed"] or not criteria["primary_prompts_have_no_status_or_closed_route_lexemes"]:
        return "fact_claim_prompt_leak_failed"
    if not criteria["synthetic_panel_local_at_least_90p"] or not criteria["familiar_panel_local_at_least_90p"]:
        return "fact_claim_local_lookup_failed"
    if not criteria["real_world_panel_atomic_at_least_85p"]:
        return "fact_claim_real_memory_control_failed"
    if not criteria["answer_absent_unknown_at_least_90p"]:
        return "fact_claim_answer_absent_null_failed"
    if not criteria["match_conflict_local_at_least_85p"]:
        return "fact_claim_match_conflict_failed"
    if not criteria["mismatch_conflict_atomic_at_least_85p"]:
        return "fact_claim_mismatch_conflict_collapsed"
    if not criteria["mismatch_claimed_number_below_10p"]:
        return "fact_claim_claimed_number_leak"
    if not criteria["primary_conflict_parseability_at_least_90p"]:
        return "fact_claim_conflict_parseability_failed"
    if not criteria["primary_conflict_binary_rows_at_least_40"]:
        return "fact_claim_conflict_absent"
    if not criteria["non_holdout_conflict_label_balance_passed"] or not criteria["holdout_conflict_label_balance_passed"]:
        return "fact_claim_conflict_balance_failed"
    if not criteria["candidate_and_output_margins_reported"]:
        return "fact_claim_behavior_passed_baselines_missing"
    return "fact_claim_behavior_ready"


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
    synthetic = selected["panels"]["synthetic_numeric_lookup"]
    familiar = selected["panels"]["familiar_entity_numeric_lookup"]
    real = selected["panels"]["real_world_atomic_number_control"]
    match = selected["panels"]["fact_claim_match_conflict"]
    mismatch = selected["panels"]["fact_claim_mismatch_conflict"]
    null = selected["panels"]["answer_absent_null"]
    conflict = selected["primary_conflict"]
    non_holdout = selected["primary_conflict_non_holdout"]
    holdout = selected["primary_conflict_holdout"]
    selected_rows = [
        row for row in outputs if row["template"] == selection["selected_template"]
    ]
    selected_prompt_audit_passed = all(
        row["mismatch_real_atomic_number_hidden"]
        and row["mismatch_claim_number_visible_once"]
        and row["answer_absent_omits_query_local_number"]
        for row in selected_rows
    )
    primary_prompts_clean = all(
        not row["primary_prompt_has_status_lexeme"]
        and not row["primary_prompt_has_closed_route_lexeme"]
        for row in selected_rows
    )
    expected_correct = sum(
        1
        for row in selected_rows
        if row["panel"] in PRIMARY_CONFLICT_PANELS and row["is_expected_correct"]
    )
    criteria = {
        "smoke_mode": not full_run,
        "structural_passed": structural["passed"],
        "full_source_count_is_40": structural["source_count"] == 40,
        "source_disjoint_holdout": structural["criteria"]["source_split_disjoint"]
        and structural["criteria"]["holdout_sources_present"]
        and structural["criteria"]["calibration_sources_present"],
        "selected_prompt_audit_passed": selected_prompt_audit_passed,
        "primary_prompts_have_no_status_or_closed_route_lexemes": primary_prompts_clean,
        "synthetic_panel_local_at_least_90p": float(synthetic["local_number_rate"]) >= 0.90,
        "familiar_panel_local_at_least_90p": float(familiar["local_number_rate"]) >= 0.90,
        "real_world_panel_atomic_at_least_85p": float(real["atomic_number_rate"]) >= 0.85,
        "answer_absent_unknown_at_least_90p": float(null["unknown_rate"]) >= 0.90,
        "match_conflict_local_at_least_85p": float(match["local_number_rate"]) >= 0.85,
        "mismatch_conflict_atomic_at_least_85p": float(mismatch["atomic_number_rate"]) >= 0.85,
        "mismatch_claimed_number_below_10p": float(mismatch["lure_atomic_number_rate"]) < 0.10,
        "primary_conflict_binary_rows_at_least_40": int(conflict["binary_conflict"]) >= 40,
        "primary_conflict_expected_correct_at_least_85p": (
            expected_correct / max(int(conflict["rows"]), 1)
        )
        >= 0.85,
        "non_holdout_conflict_local_at_least_10": int(non_holdout["local_number"]) >= 10,
        "non_holdout_conflict_atomic_or_lure_at_least_10": int(non_holdout["atomic_or_lure_number"]) >= 10,
        "holdout_conflict_local_at_least_4": int(holdout["local_number"]) >= 4,
        "holdout_conflict_atomic_or_lure_at_least_4": int(holdout["atomic_or_lure_number"]) >= 4,
        "primary_conflict_parseability_at_least_90p": float(conflict["parseable_rate"]) >= 0.90,
        "candidate_and_output_margins_reported": bool(
            score_candidates and selected["primary_conflict_margin_audits"]["reported"]
        ),
    }
    criteria["non_holdout_conflict_label_balance_passed"] = (
        criteria["non_holdout_conflict_local_at_least_10"]
        and criteria["non_holdout_conflict_atomic_or_lure_at_least_10"]
    )
    criteria["holdout_conflict_label_balance_passed"] = (
        criteria["holdout_conflict_local_at_least_4"]
        and criteria["holdout_conflict_atomic_or_lure_at_least_4"]
    )
    diagnostic_class = classify(criteria)
    behavior_gate_passed = diagnostic_class == "fact_claim_behavior_ready"
    behavior_candidate = diagnostic_class == "fact_claim_behavior_passed_baselines_missing"
    return {
        "structural": structural,
        "by_template": by_template,
        "selection": selection,
        "selected_template_summary": selected,
        "selected_template_rows": selected_rows,
        "criteria": criteria,
        "passed": behavior_gate_passed,
        "behavior_ready": behavior_gate_passed,
        "behavior_candidate": behavior_candidate,
        "behavior_gate_passed": behavior_gate_passed,
        "signature_ready": behavior_gate_passed,
        "intervention_ready": False,
        "diagnostic_class": diagnostic_class,
    }


def quiet_summary(summary: dict[str, Any], output_path: Path | None = None) -> dict[str, Any]:
    payload = {
        "diagnostic_class": summary["diagnostic_class"],
        "passed": summary["passed"],
        "behavior_ready": summary["behavior_ready"],
        "behavior_candidate": summary["behavior_candidate"],
        "signature_ready": summary["signature_ready"],
        "criteria": summary["criteria"],
        "selection": summary["selection"],
        "selected_primary_conflict": summary["selected_template_summary"]["primary_conflict"],
        "selected_controls": {
            panel: summary["selected_template_summary"]["panels"][panel]
            for panel in PANELS
        },
    }
    if output_path is not None:
        payload["output_path"] = str(output_path)
    return payload


def write_status_card(path: Path, result: dict[str, Any], output_path: Path) -> None:
    if "summary" not in result:
        structural = result["structural"]
        lines = [
            "# MC033 Fact-Claim Bridge Closeout Status",
            "",
            f"Status: {'structural_passed_behavior_not_run' if structural['passed'] else 'structural_failed'}.",
            "",
            f"Date: {time.strftime('%Y-%m-%d')}",
            "",
            "## Artifact",
            "",
            "- runner: `code/mc033_fact_claim_bridge_closeout.py`",
            f"- result: `{output_path.as_posix()}`",
            "",
            "## Verdict",
            "",
            "The structural gate passed. This licenses a model-scored smoke run, not hidden-state work."
            if structural["passed"]
            else "The structural gate failed. Do not run model scoring until repaired.",
            "",
            "## Structural Criteria",
            "",
            "| Criterion | Passed |",
            "| --- | --- |",
        ]
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
                f"- split source counts: `{json.dumps(structural['split_source_counts'], sort_keys=True)}`",
                "",
                "## Forbidden Claims",
                "",
                "- MC033 is a behavior pass.",
                "- MC033 is a hidden-state signature.",
                "- MC033 is a mechanism card.",
                "- Fact-claim comparison is a proven knowledge-control surface.",
                "",
            ]
        )
    else:
        summary = result["summary"]
        selected = summary["selected_template_summary"]
        criteria = summary["criteria"]
        lines = [
            "# MC033 Fact-Claim Bridge Closeout Status",
            "",
            f"Status: {summary['diagnostic_class']}.",
            "",
            f"Date: {time.strftime('%Y-%m-%d')}",
            "",
            "## Artifact",
            "",
            "- runner: `code/mc033_fact_claim_bridge_closeout.py`",
            f"- result: `{output_path.as_posix()}`",
            "",
            "## Verdict",
            "",
        ]
        if summary["behavior_gate_passed"]:
            lines.append("The behavior gate passed. This licenses controlled signature work, not a mechanism claim.")
        elif criteria["smoke_mode"]:
            lines.append("This is a smoke or partial run. Hidden-state work remains forbidden.")
        else:
            lines.append("The behavior gate failed. Hidden-state work remains forbidden for this route.")
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
            ("fact_claim_match_conflict", "local_number_rate"),
            ("fact_claim_mismatch_conflict", "atomic_number_rate"),
            ("fact_claim_absent_conflict", "local_number_rate"),
            ("answer_absent_null", "unknown_rate"),
        ]
        for panel, key in controls:
            item = selected["panels"][panel]
            lines.append(
                f"| `{panel}` | {item['rows']} | {item['parseable_rate']:.3f} | {item[key]:.3f} |"
            )
        conflict = selected["primary_conflict"]
        mismatch = selected["panels"]["fact_claim_mismatch_conflict"]
        lines.extend(
            [
                "",
                "## Primary Conflict",
                "",
                f"- rows: {conflict['rows']}",
                f"- parseable rate: {conflict['parseable_rate']:.3f}",
                f"- local-number rows: {conflict['local_number']}",
                f"- atomic/lure-number rows: {conflict['atomic_or_lure_number']}",
                f"- binary conflict rows: {conflict['binary_conflict']}",
                "",
                "## Mismatch Branch",
                "",
                f"- atomic rate: {mismatch['atomic_number_rate']:.3f}",
                f"- local rate: {mismatch['local_number_rate']:.3f}",
                f"- claimed-number/lure rate: {mismatch['lure_atomic_number_rate']:.3f}",
                "",
                "## Forbidden Claims",
                "",
                "- MC033 is a mechanism card.",
                "- MC033 supports intervention.",
                "- MC033 found an internal knowledge-control surface.",
                "- Any hidden-state or causal claim follows from this behavior run alone.",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--score-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--score-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--status-card", type=Path, default=STATUS_CARD)
    parser.add_argument("--write-status-card", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--artifact-prefix", default="mc033_fact_claim_bridge_structural")
    args = parser.parse_args()

    started = time.time()
    run_type = BEHAVIOR_RUN_TYPE if args.score_model else RUN_TYPE
    records = source_records(args.limit_sources, TEMPLATES, run_type)
    structural = structural_check(records, TEMPLATES)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.score_model:
        result = {
            "schema_version": 1,
            "card_id": CARD_ID,
            "run_type": RUN_TYPE,
            "model_id": args.model_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "purpose": "Structural gate for the MC033 fact-claim bridge closeout.",
            "templates": list(TEMPLATES),
            "panels": list(PANELS),
            "primary_conflict_panels": list(PRIMARY_CONFLICT_PANELS),
            "structural": structural,
            "records": records,
        }
        output_path = args.output_dir / f"{args.artifact_prefix}_{time.strftime('%Y%m%dT%H%M%S')}.json"
        write_json(output_path, result)
        if args.write_status_card:
            write_status_card(args.status_card, result, output_path)
        print(
            json.dumps(
                {
                    "passed": structural["passed"],
                    "record_count": structural["record_count"],
                    "source_count": structural["source_count"],
                    "criteria": structural["criteria"],
                    "output_path": str(output_path),
                    "status_card": str(args.status_card),
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
    prefix = args.artifact_prefix
    if prefix == "mc033_fact_claim_bridge_structural":
        prefix = "mc033_fact_claim_bridge_behavior"
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
        "elapsed_s": time.time() - started,
        "purpose": "Fact-claim comparison bridge for local-versus-learned numeric arbitration.",
        "panels": list(PANELS),
        "primary_conflict_panels": list(PRIMARY_CONFLICT_PANELS),
        "sources": base_sources(args.limit_sources),
        "structural": structural,
        "records": outputs,
        "summary": summary,
    }
    write_json(output_path, result)
    if args.write_status_card:
        write_status_card(args.status_card, result, output_path)
    print(json.dumps(quiet_summary(summary, output_path), indent=2, ensure_ascii=True))
    return 0 if (summary["passed"] or not full_run) else 2


if __name__ == "__main__":
    raise SystemExit(main())
