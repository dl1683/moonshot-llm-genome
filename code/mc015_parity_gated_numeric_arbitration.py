#!/usr/bin/env python
"""MC015 parity-gated numeric arbitration behavior gate.

MC012 created a clean local-versus-learned numeric bridge by showing explicit
trusted/untrusted source-status text. MC013 showed that directly removing that
status channel collapses the learned-fact side. MC014 showed that inferring
source validity from calibration rows still collapses to prompt-local table
dominance. MC015 changes the contract again:

    use a learned atomic-number property as the gate for local table control.

The prompt contains a local lab-number table, but the table controls only for
elements whose standard atomic-number parity matches the rule. The target
atomic number is hidden in conflict prompts. This is a behavior gate only; it
does not claim an internal mechanism.
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
    strict_parse,
    summarize_rows,
    word_occurrences,
)


CARD_ID = "MC015"
RUN_TYPE = "parity_gated_numeric_arbitration_structural_gate"
BEHAVIOR_RUN_TYPE = "parity_gated_numeric_arbitration_behavior"
RESULT_DIR = Path("results/cards/MC015")
STATUS_CARD = Path("research/cards/MC015_PARITY_GATED_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md")

PANELS = (
    "synthetic_numeric_lookup",
    "familiar_entity_numeric_lookup",
    "real_world_atomic_number_control",
    "local_if_even_conflict",
    "local_if_odd_conflict",
    "parity_rule_absent_conflict",
    "answer_absent_null",
)
PRIMARY_CONFLICT_PANELS = ("local_if_even_conflict", "local_if_odd_conflict")
CONFLICT_PANELS = PRIMARY_CONFLICT_PANELS + ("parity_rule_absent_conflict",)
TEMPLATES = ("parity_rule", "compact_parity", "worked_parity")
STATUS_LEXEME_RE = re.compile(r"\b(trusted|untrusted|reliable|unreliable|status)\b", re.IGNORECASE)


def is_even_atomic(source: dict[str, Any]) -> bool:
    return int(source["atomic_number"]) % 2 == 0


def render_local_table(
    rows: list[dict[str, Any]],
    query: dict[str, Any],
    *,
    key_field: str,
    include_query: bool,
    template: str,
) -> list[str]:
    visible_rows = rows if include_query else [row for row in rows if row["source_id"] != query["source_id"]]
    if template == "parity_rule":
        intro = [
            "A local lab table may define temporary lab numbers.",
            "Standard chemistry may define atomic numbers.",
        ]
    elif template == "compact_parity":
        intro = [
            "Two numeric sources may conflict.",
            "Use the parity rule to decide whether the local table controls.",
        ]
    elif template == "worked_parity":
        intro = [
            "A local lab table gives temporary lab numbers.",
            "The parity gate below decides whether that table controls.",
        ]
    else:
        raise ValueError(f"unknown template {template}")

    label = "Synthetic-key lab-number table:" if key_field == "synthetic_key" else "Element lab-number table:"
    lines = [*intro, label]
    for row in visible_rows:
        lines.append(f"{row[key_field]} has local lab number {row['local_number']}.")
    return lines


def parity_example_rows(sources: list[dict[str, Any]], source: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    excluded_numbers = {int(source["atomic_number"]), int(source["lure_atomic_number"])}
    odd = next(
        row
        for row in sources
        if row["source_id"] != source["source_id"]
        and int(row["atomic_number"]) not in excluded_numbers
        and int(row["atomic_number"]) % 2 == 1
    )
    even = next(
        row
        for row in sources
        if row["source_id"] != source["source_id"]
        and int(row["atomic_number"]) not in excluded_numbers
        and int(row["atomic_number"]) % 2 == 0
    )
    return odd, even


def append_parity_rule(
    lines: list[str],
    *,
    local_on: str,
    template: str,
    sources: list[dict[str, Any]] | None = None,
    source: dict[str, Any] | None = None,
) -> None:
    opposite = "odd" if local_on == "even" else "even"
    if template == "parity_rule":
        lines.extend(
            [
                "First determine whether the queried element has an even or odd standard atomic number.",
                f"If the queried element's standard atomic number is {local_on}, use the local lab number.",
                f"If the queried element's standard atomic number is {opposite}, ignore the local lab number.",
                f"For a {opposite} standard atomic number, return the standard atomic number.",
            ]
        )
    elif template == "compact_parity":
        lines.extend(
            [
                f"Rule: {local_on} standard atomic number -> local lab number.",
                f"Rule: {opposite} standard atomic number -> standard atomic number.",
            ]
        )
    elif template == "worked_parity":
        if sources is None or source is None:
            raise ValueError("worked_parity requires example sources")
        odd, even = parity_example_rows(sources, source)
        lines.extend(
            [
                f"Parity example: {odd['element']} has standard atomic number {odd['atomic_number']}, so it is odd.",
                f"Parity example: {even['element']} has standard atomic number {even['atomic_number']}, so it is even.",
                f"Decision rule: if the queried element is {local_on}, answer with the local lab number from the table.",
                f"Decision rule: if the queried element is {opposite}, ignore the table and answer with the standard atomic number.",
            ]
        )
    else:
        raise ValueError(f"unknown template {template}")


def expected_conflict_label(source: dict[str, Any], *, local_on: str) -> str:
    even = is_even_atomic(source)
    local_expected = even if local_on == "even" else not even
    return "local_number" if local_expected else "atomic_number"


def make_prompt(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    panel: str,
    template: str,
) -> tuple[str, str]:
    rows = distractors(sources, source, 4)
    if panel == "synthetic_numeric_lookup":
        lines = render_local_table(rows, source, key_field="synthetic_key", include_query=True, template=template)
        lines.append(f"Query key: {source['synthetic_key']}")
        lines.append("Return the local lab number.")
        expected_label = "local_number"
    elif panel == "familiar_entity_numeric_lookup":
        lines = render_local_table(rows, source, key_field="element", include_query=True, template=template)
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the local lab number.")
        expected_label = "local_number"
    elif panel == "real_world_atomic_number_control":
        lines = [
            "No local lab table is active.",
            "Return the standard atomic number for the queried element.",
            f"Query element: {source['element']}",
        ]
        expected_label = "atomic_number"
    elif panel == "local_if_even_conflict":
        lines = [
            "Decide which numeric source controls before answering.",
        ]
        append_parity_rule(lines, local_on="even", template=template, sources=sources, source=source)
        lines.extend(render_local_table(rows, source, key_field="element", include_query=True, template=template))
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the controlling number.")
        expected_label = expected_conflict_label(source, local_on="even")
    elif panel == "local_if_odd_conflict":
        lines = [
            "Decide which numeric source controls before answering.",
        ]
        append_parity_rule(lines, local_on="odd", template=template, sources=sources, source=source)
        lines.extend(render_local_table(rows, source, key_field="element", include_query=True, template=template))
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the controlling number.")
        expected_label = expected_conflict_label(source, local_on="odd")
    elif panel == "parity_rule_absent_conflict":
        lines = render_local_table(rows, source, key_field="element", include_query=True, template=template)
        lines.append("No parity rule is available.")
        lines.append("The local lab number and standard atomic number may conflict.")
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the controlling number.")
        expected_label = "ambiguous"
    elif panel == "answer_absent_null":
        lines = render_local_table(rows, source, key_field="element", include_query=False, template=template)
        lines.append(f"Query element: {source['element']}")
        lines.append("This is a local-table membership check, not a chemistry question.")
        lines.append("If the query element is absent from the local lab table, return UNKNOWN even if you know its standard atomic number.")
        expected_label = "unknown"
    else:
        raise ValueError(f"unknown panel {panel}")
    return "\n".join([*lines, *response_suffix()]), expected_label


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
                prompt, expected_label = make_prompt(sources, source, panel=panel, template=template)
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
        if ANSWER_RE.match(candidate) is None
    ]
    candidate_collision_rows = [
        row["id"] for row in records if len(set(row["candidate_answers"])) != len(row["candidate_answers"])
    ]
    status_lexeme_rows = [row["id"] for row in primary_rows if STATUS_LEXEME_RE.search(row["prompt"])]
    answer_suffixes = {"\n".join(row["prompt"].splitlines()[-3:]) for row in records}
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
        "single_answer_suffix": len(answer_suffixes) == 1,
        "primary_prompts_have_no_status_lexemes": not status_lexeme_rows,
        "primary_expected_labels_balanced": primary_expected_label_counts == {"local_number": len(primary_rows) // 2, "atomic_number": len(primary_rows) // 2},
        "primary_expected_labels_balanced_by_split": all(
            counts.get("local_number", 0) == counts.get("atomic_number", 0)
            and set(counts).issubset({"local_number", "atomic_number"})
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
        "answer_suffix_count": len(answer_suffixes),
        "primary_expected_label_counts": dict(sorted(primary_expected_label_counts.items())),
        "primary_expected_label_counts_by_split": {
            split: dict(sorted(counts.items())) for split, counts in sorted(primary_expected_label_counts_by_split.items())
        },
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
            "is_atomic_or_lure_number": parsed["selected_label"] in {"atomic_number", "lure_atomic_number"},
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


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("selected_label")) for row in rows).items()))


def summarize_parity_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_rows(rows)
    expected_correct = sum(1 for row in rows if row.get("expected_correct"))
    expected_labels = Counter(str(row.get("expected_label")) for row in rows)
    summary["expected_correct"] = expected_correct
    summary["expected_correct_rate"] = expected_correct / len(rows) if rows else 0.0
    summary["expected_label_counts"] = dict(sorted(expected_labels.items()))
    return summary


def margin_audits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows or "final_local_minus_atomic_number_logit" not in rows[0]:
        return {"reported": False}
    return {"reported": True}


def template_summary(outputs: list[dict[str, Any]], templates: tuple[str, ...]) -> dict[str, Any]:
    result = {}
    for template in templates:
        rows = [row for row in outputs if row["template"] == template]
        panels = {panel: summarize_parity_rows([row for row in rows if row["panel"] == panel]) for panel in PANELS}
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
            "primary_conflict": summarize_parity_rows(conflict),
            "primary_conflict_non_holdout": summarize_parity_rows(non_holdout),
            "primary_conflict_holdout": summarize_parity_rows(holdout),
            "primary_conflict_expected_local": summarize_parity_rows(expected_local),
            "primary_conflict_expected_atomic": summarize_parity_rows(expected_atomic),
            "primary_conflict_margin_audits": margin_audits(conflict),
        }
    return result


def selection_key(item: dict[str, Any], template: str, templates: tuple[str, ...]) -> tuple[float, float, float, int, int, int]:
    synthetic = item["panels"]["synthetic_numeric_lookup"]
    familiar = item["panels"]["familiar_entity_numeric_lookup"]
    real = item["panels"]["real_world_atomic_number_control"]
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
        float(null["unknown_rate"]),
        float(expected_local["local_number_rate"]),
        float(expected_atomic["atomic_number_rate"]),
    )
    conflict_balance = min(int(non_holdout["local_number"]), int(non_holdout["atomic_or_lure_number"]))
    holdout_balance = min(int(holdout["local_number"]), int(holdout["atomic_or_lure_number"]))
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
            "max direct/control plus parity-expected conflict floor",
            "max primary conflict parseability",
            "max primary expected-label correctness",
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
    if not criteria["selected_prompt_audit_passed"] or not criteria["primary_prompts_have_no_status_lexemes"]:
        return "parity_prompt_status_leak_failed"
    if not criteria["synthetic_panel_a_local_at_least_90p"] or not criteria["synthetic_panel_a_parseable_at_least_95p"]:
        return "parity_synthetic_lookup_failed"
    if not criteria["familiar_panel_b_local_at_least_90p"] or not criteria["familiar_panel_b_parseable_at_least_95p"]:
        return "parity_familiar_lookup_failed"
    if not criteria["real_world_panel_c_atomic_at_least_85p"] or not criteria["real_world_panel_c_parseable_at_least_95p"]:
        return "parity_real_memory_control_failed"
    if not criteria["answer_absent_panel_g_unknown_at_least_90p"] or not criteria["answer_absent_panel_g_parseable_at_least_95p"]:
        return "parity_answer_absent_null_failed"
    if not criteria["expected_local_conflict_local_at_least_85p"] or not criteria["expected_local_conflict_parseable_at_least_90p"]:
        return "parity_expected_local_conflict_failed"
    if not criteria["expected_atomic_conflict_atomic_at_least_85p"] or not criteria["expected_atomic_conflict_parseable_at_least_90p"]:
        return "parity_expected_atomic_conflict_failed"
    if not criteria["primary_conflict_parseability_at_least_90p"]:
        return "parity_conflict_parseability_failed"
    if not criteria["primary_conflict_expected_correct_at_least_85p"]:
        return "parity_gate_not_followed"
    if not criteria["primary_conflict_binary_rows_at_least_40"]:
        return "parity_conflict_contrast_absent"
    if not criteria["non_holdout_conflict_label_balance_passed"] or not criteria["holdout_conflict_label_balance_passed"]:
        return "parity_conflict_unbalanced"
    if not criteria["candidate_and_output_margins_reported"]:
        return "parity_behavior_passed_baselines_missing"
    return "parity_gated_behavior_passed"


def summarize(records: list[dict[str, Any]], outputs: list[dict[str, Any]], templates: tuple[str, ...], full_run: bool, score_candidates: bool) -> dict[str, Any]:
    structural = structural_check(records, templates)
    by_template = template_summary(outputs, templates)
    selection = select_template(by_template, templates)
    selected = by_template[selection["selected_template"]]
    synthetic = selected["panels"]["synthetic_numeric_lookup"]
    familiar = selected["panels"]["familiar_entity_numeric_lookup"]
    real = selected["panels"]["real_world_atomic_number_control"]
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
        "answer_absent_panel_g_unknown_at_least_90p": float(null["unknown_rate"]) >= 0.90,
        "answer_absent_panel_g_parseable_at_least_95p": float(null["parseable_rate"]) >= 0.95,
        "expected_local_conflict_local_at_least_85p": float(expected_local["local_number_rate"]) >= 0.85,
        "expected_local_conflict_parseable_at_least_90p": float(expected_local["parseable_rate"]) >= 0.90,
        "expected_atomic_conflict_atomic_at_least_85p": float(expected_atomic["atomic_number_rate"]) >= 0.85,
        "expected_atomic_conflict_parseable_at_least_90p": float(expected_atomic["parseable_rate"]) >= 0.90,
        "primary_conflict_expected_correct_at_least_85p": float(conflict["expected_correct_rate"]) >= 0.85,
        "primary_conflict_binary_rows_at_least_40": int(conflict["binary_conflict"]) >= 40,
        "non_holdout_conflict_local_at_least_10": int(non_holdout["local_number"]) >= 10,
        "non_holdout_conflict_atomic_or_lure_at_least_10": int(non_holdout["atomic_or_lure_number"]) >= 10,
        "holdout_conflict_local_at_least_4": int(holdout["local_number"]) >= 4,
        "holdout_conflict_atomic_or_lure_at_least_4": int(holdout["atomic_or_lure_number"]) >= 4,
        "primary_conflict_parseability_at_least_90p": float(conflict["parseable_rate"]) >= 0.90,
        "candidate_and_output_margins_reported": bool(score_candidates and selected["primary_conflict_margin_audits"]["reported"]),
        "visible_status_label_absent_by_design": primary_prompts_have_no_status_lexemes,
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
    behavior_gate_passed = diagnostic_class == "parity_gated_behavior_passed"
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
        "signature_ready": behavior_gate_passed,
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
            "synthetic_numeric_lookup": summary["selected_template_summary"]["panels"]["synthetic_numeric_lookup"],
            "familiar_entity_numeric_lookup": summary["selected_template_summary"]["panels"]["familiar_entity_numeric_lookup"],
            "real_world_atomic_number_control": summary["selected_template_summary"]["panels"]["real_world_atomic_number_control"],
            "local_if_even_conflict": summary["selected_template_summary"]["panels"]["local_if_even_conflict"],
            "local_if_odd_conflict": summary["selected_template_summary"]["panels"]["local_if_odd_conflict"],
            "parity_rule_absent_conflict": summary["selected_template_summary"]["panels"]["parity_rule_absent_conflict"],
            "answer_absent_null": summary["selected_template_summary"]["panels"]["answer_absent_null"],
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
        "# MC015 Parity-Gated Numeric Arbitration Behavior Status",
        "",
        f"Status: {summary['diagnostic_class']}.",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "## Artifact",
        "",
        "- runner:",
        "  `code/mc015_parity_gated_numeric_arbitration.py`",
        "- result:",
        f"  `{output_path.as_posix()}`",
        "",
        "## Verdict",
        "",
    ]
    if summary["behavior_gate_passed"]:
        lines.extend(
            [
                "The parity-gated behavior gate passed. The behavior table",
                "may be used for a preregistered hidden-signature screen, but this",
                "is not a mechanism card and no intervention is allowed yet.",
            ]
        )
    elif criteria["smoke_mode"]:
        lines.extend(
            [
                "This is a smoke or partial run. It validates runner plumbing only;",
                "it is not evidence for or against the full MC015 behavior substrate.",
            ]
        )
    else:
        lines.extend(
            [
                "The parity-gated behavior gate failed. The target atomic number",
                "is hidden and the visible status label channel is absent, but",
                "the learned parity condition did not produce a clean gated",
                "local-versus-learned conflict table. Hidden-state work remains",
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
        ("local_if_even_conflict", "expected_correct_rate"),
        ("local_if_odd_conflict", "expected_correct_rate"),
        ("parity_rule_absent_conflict", "local_number_rate"),
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
            f"- local-number rows: {conflict['local_number']}",
            f"- atomic/lure-number rows: {conflict['atomic_or_lure_number']}",
            f"- binary conflict rows: {conflict['binary_conflict']}",
            f"- expected-correct rows: {conflict['expected_correct']}",
            f"- expected-correct rate: {conflict['expected_correct_rate']:.3f}",
            "",
            "## Forbidden Claims",
            "",
            "- MC015 is a mechanism card.",
            "- MC015 supports intervention.",
            "- MC015 found an internal knowledge-control surface.",
            "- Any hidden-state or causal claim follows from this behavior run alone.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_model_and_tokenizer(model_id: str, local_files_only: bool) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--templates", choices=TEMPLATES, nargs="+", default=list(TEMPLATES))
    parser.add_argument("--write-manifest", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--artifact-prefix", default="mc015_parity_gated_numeric_structural")
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
            "purpose": "Generated-answer parity-gated numeric bridge behavior scoring before any MC015 hidden-state work.",
            "panels": list(PANELS),
            "primary_conflict_panels": list(PRIMARY_CONFLICT_PANELS),
            "sources": base_sources(args.limit_sources),
            "records": outputs,
            "summary": summary,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = args.artifact_prefix
        if prefix == "mc015_parity_gated_numeric_structural":
            prefix = "mc015_parity_gated_numeric_behavior"
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
        "purpose": "Structural prompt/channel audit before MC015 generated-answer behavior scoring.",
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
