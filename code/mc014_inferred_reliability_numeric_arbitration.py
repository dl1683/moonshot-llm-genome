#!/usr/bin/env python
"""MC014 inferred-reliability numeric arbitration behavior gate.

MC012 created a clean local-versus-learned numeric bridge by showing explicit
trusted/untrusted source-status text. MC013 showed that directly removing that
status channel collapses the learned-fact side. MC014 tests a materially
different route:

    infer source control from calibration facts, not from a status label.

The model sees calibration rows whose reported atomic numbers either match or
contradict standard chemistry. The answer rule is visible, but the source
status is not named. This is still a behavior gate only; it does not claim an
internal mechanism.
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
    ELEMENTS,
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


CARD_ID = "MC014"
RUN_TYPE = "inferred_reliability_numeric_arbitration_structural_gate"
BEHAVIOR_RUN_TYPE = "inferred_reliability_numeric_arbitration_behavior"
RESULT_DIR = Path("results/cards/MC014")
STATUS_CARD = Path("research/cards/MC014_INFERRED_RELIABILITY_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md")

PANELS = (
    "synthetic_numeric_lookup",
    "familiar_entity_numeric_lookup",
    "real_world_atomic_number_control",
    "calibration_consistent_conflict",
    "calibration_inconsistent_conflict",
    "calibration_absent_conflict",
    "answer_absent_null",
)
PRIMARY_CONFLICT_PANELS = ("calibration_consistent_conflict", "calibration_inconsistent_conflict")
CONFLICT_PANELS = PRIMARY_CONFLICT_PANELS + ("calibration_absent_conflict",)
TEMPLATES = ("calibration_rule", "compact_calibration")
STATUS_LEXEME_RE = re.compile(r"\b(trusted|untrusted|reliable|unreliable|status)\b", re.IGNORECASE)


def calibration_candidates(
    source: dict[str, Any],
    *,
    count: int = 3,
) -> list[dict[str, Any]]:
    excluded_atomic_numbers = {int(source["atomic_number"]), int(source["lure_atomic_number"])}
    rows = []
    for index, (source_id, element, atomic_number) in enumerate(ELEMENTS):
        if source_id == source["source_id"] or atomic_number in excluded_atomic_numbers:
            continue
        rows.append(
            {
                "source_id": source_id,
                "source_index": index,
                "element": element,
                "atomic_number": atomic_number,
            }
        )
        if len(rows) == count:
            return rows
    raise ValueError(f"not enough calibration candidates for {source['source_id']}")


def wrong_calibration_number(index: int) -> int:
    return 97 + index


def render_calibration_rows(source: dict[str, Any], *, consistent: bool, template: str) -> list[str]:
    if template == "calibration_rule":
        lines = ["Verification rows from the same source:"]
    elif template == "compact_calibration":
        lines = ["Source verification rows:"]
    else:
        raise ValueError(f"unknown template {template}")
    for index, row in enumerate(calibration_candidates(source)):
        reported = row["atomic_number"] if consistent else wrong_calibration_number(index)
        lines.append(f"{row['element']} is reported as standard atomic number {reported}.")
    return lines


def render_local_table(
    rows: list[dict[str, Any]],
    query: dict[str, Any],
    *,
    key_field: str,
    include_query: bool,
    template: str,
) -> list[str]:
    visible_rows = rows if include_query else [row for row in rows if row["source_id"] != query["source_id"]]
    if template == "calibration_rule":
        intro = [
            "A local lab table may define temporary lab numbers.",
            "Standard chemistry may define atomic numbers.",
        ]
    elif template == "compact_calibration":
        intro = [
            "Two numeric sources may conflict.",
            "Use verification rows to decide whether the local table controls.",
        ]
    else:
        raise ValueError(f"unknown template {template}")

    label = "Synthetic-key lab-number table:" if key_field == "synthetic_key" else "Element lab-number table:"
    lines = [*intro, label]
    for row in visible_rows:
        lines.append(f"{row[key_field]} has local lab number {row['local_number']}.")
    return lines


def append_verification_rule(lines: list[str], *, template: str) -> None:
    if template == "calibration_rule":
        lines.extend(
            [
                "Check the verification rows against standard chemistry before using the local lab table.",
                "If every verification row is correct, the local lab table is valid for this question.",
                "If any verification row is wrong, the local lab table is invalid for this question.",
                "Use a valid local lab table to answer with the local lab number.",
                "For an invalid local lab table, do not copy any local number; return the standard atomic number for the query.",
            ]
        )
    elif template == "compact_calibration":
        lines.extend(
            [
                "Rule: all verification rows correct -> local table valid -> local lab number.",
                "Rule: any verification row wrong -> local table invalid -> standard atomic number.",
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
    elif panel == "calibration_consistent_conflict":
        lines = [
            "Decide whether the local lab table below is valid before answering.",
        ]
        lines.extend(render_calibration_rows(source, consistent=True, template=template))
        append_verification_rule(lines, template=template)
        lines.extend(render_local_table(rows, source, key_field="element", include_query=True, template=template))
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the controlling number.")
        expected_label = "local_number"
    elif panel == "calibration_inconsistent_conflict":
        lines = [
            "Decide whether the local lab table below is valid before answering.",
        ]
        lines.extend(render_calibration_rows(source, consistent=False, template=template))
        append_verification_rule(lines, template=template)
        lines.extend(render_local_table(rows, source, key_field="element", include_query=True, template=template))
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the controlling number.")
        expected_label = "atomic_number"
    elif panel == "calibration_absent_conflict":
        lines = render_local_table(rows, source, key_field="element", include_query=True, template=template)
        lines.append("No verification rows are available.")
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


def margin_audits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows or "final_local_minus_atomic_number_logit" not in rows[0]:
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
        result[template] = {
            "rows": len(rows),
            "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "selected_label_counts": label_counts(rows),
            "panels": panels,
            "primary_conflict": summarize_rows(conflict),
            "primary_conflict_non_holdout": summarize_rows(non_holdout),
            "primary_conflict_holdout": summarize_rows(holdout),
            "primary_conflict_margin_audits": margin_audits(conflict),
        }
    return result


def selection_key(item: dict[str, Any], template: str, templates: tuple[str, ...]) -> tuple[float, float, float, int, int, int]:
    synthetic = item["panels"]["synthetic_numeric_lookup"]
    familiar = item["panels"]["familiar_entity_numeric_lookup"]
    real = item["panels"]["real_world_atomic_number_control"]
    null = item["panels"]["answer_absent_null"]
    consistent = item["panels"]["calibration_consistent_conflict"]
    inconsistent = item["panels"]["calibration_inconsistent_conflict"]
    conflict = item["primary_conflict"]
    non_holdout = item["primary_conflict_non_holdout"]
    holdout = item["primary_conflict_holdout"]
    control_floor = min(
        float(synthetic["local_number_rate"]),
        float(familiar["local_number_rate"]),
        float(real["atomic_number_rate"]),
        float(null["unknown_rate"]),
        float(consistent["local_number_rate"]),
        float(inconsistent["atomic_number_rate"]),
    )
    conflict_balance = min(int(non_holdout["local_number"]), int(non_holdout["atomic_or_lure_number"]))
    holdout_balance = min(int(holdout["local_number"]), int(holdout["atomic_or_lure_number"]))
    return (
        control_floor,
        float(conflict["parseable_rate"]),
        float(conflict["binary_conflict"]),
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
            "max direct/control plus calibration-conflict floor",
            "max primary conflict parseability",
            "max primary binary conflict count",
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
        return "calibration_prompt_status_leak_failed"
    if not criteria["synthetic_panel_a_local_at_least_90p"] or not criteria["synthetic_panel_a_parseable_at_least_95p"]:
        return "calibration_synthetic_lookup_failed"
    if not criteria["familiar_panel_b_local_at_least_90p"] or not criteria["familiar_panel_b_parseable_at_least_95p"]:
        return "calibration_familiar_lookup_failed"
    if not criteria["real_world_panel_c_atomic_at_least_85p"] or not criteria["real_world_panel_c_parseable_at_least_95p"]:
        return "calibration_real_memory_control_failed"
    if not criteria["answer_absent_panel_g_unknown_at_least_90p"] or not criteria["answer_absent_panel_g_parseable_at_least_95p"]:
        return "calibration_answer_absent_null_failed"
    if not criteria["consistent_conflict_local_at_least_85p"] or not criteria["consistent_conflict_parseable_at_least_90p"]:
        return "calibration_consistent_conflict_failed"
    if not criteria["inconsistent_conflict_atomic_at_least_85p"] or not criteria["inconsistent_conflict_parseable_at_least_90p"]:
        return "calibration_inference_conflict_collapsed"
    if not criteria["primary_conflict_parseability_at_least_90p"]:
        return "calibration_conflict_parseability_failed"
    if not criteria["primary_conflict_binary_rows_at_least_40"]:
        return "calibration_inference_conflict_absent"
    if not criteria["non_holdout_conflict_label_balance_passed"] or not criteria["holdout_conflict_label_balance_passed"]:
        return "calibration_inference_conflict_collapsed"
    if not criteria["candidate_and_output_margins_reported"]:
        return "calibration_behavior_passed_baselines_missing"
    return "calibration_inference_behavior_passed"


def summarize(records: list[dict[str, Any]], outputs: list[dict[str, Any]], templates: tuple[str, ...], full_run: bool, score_candidates: bool) -> dict[str, Any]:
    structural = structural_check(records, templates)
    by_template = template_summary(outputs, templates)
    selection = select_template(by_template, templates)
    selected = by_template[selection["selected_template"]]
    synthetic = selected["panels"]["synthetic_numeric_lookup"]
    familiar = selected["panels"]["familiar_entity_numeric_lookup"]
    real = selected["panels"]["real_world_atomic_number_control"]
    null = selected["panels"]["answer_absent_null"]
    consistent = selected["panels"]["calibration_consistent_conflict"]
    inconsistent = selected["panels"]["calibration_inconsistent_conflict"]
    conflict = selected["primary_conflict"]
    non_holdout = selected["primary_conflict_non_holdout"]
    holdout = selected["primary_conflict_holdout"]
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
        "consistent_conflict_local_at_least_85p": float(consistent["local_number_rate"]) >= 0.85,
        "consistent_conflict_parseable_at_least_90p": float(consistent["parseable_rate"]) >= 0.90,
        "inconsistent_conflict_atomic_at_least_85p": float(inconsistent["atomic_number_rate"]) >= 0.85,
        "inconsistent_conflict_parseable_at_least_90p": float(inconsistent["parseable_rate"]) >= 0.90,
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
    behavior_gate_passed = diagnostic_class == "calibration_inference_behavior_passed"
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
            "calibration_consistent_conflict": summary["selected_template_summary"]["panels"]["calibration_consistent_conflict"],
            "calibration_inconsistent_conflict": summary["selected_template_summary"]["panels"]["calibration_inconsistent_conflict"],
            "calibration_absent_conflict": summary["selected_template_summary"]["panels"]["calibration_absent_conflict"],
            "answer_absent_null": summary["selected_template_summary"]["panels"]["answer_absent_null"],
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
        "# MC014 Inferred-Reliability Numeric Arbitration Behavior Status",
        "",
        f"Status: {summary['diagnostic_class']}.",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "## Artifact",
        "",
        "- runner:",
        "  `code/mc014_inferred_reliability_numeric_arbitration.py`",
        "- result:",
        f"  `{output_path.as_posix()}`",
        "",
        "## Verdict",
        "",
    ]
    if summary["behavior_gate_passed"]:
        lines.extend(
            [
                "The inferred-reliability behavior gate passed. The behavior table",
                "may be used for a preregistered hidden-signature screen, but this",
                "is not a mechanism card and no intervention is allowed yet.",
            ]
        )
    elif criteria["smoke_mode"]:
        lines.extend(
            [
                "This is a smoke or partial run. It validates runner plumbing only;",
                "it is not evidence for or against the full MC014 behavior substrate.",
            ]
        )
    else:
        lines.extend(
            [
                "The inferred-reliability behavior gate failed. The visible status",
                "label is absent, but the calibration-consistency prompt did not",
                "produce a clean local-versus-learned conflict table. Hidden-state",
                "work remains forbidden for this route.",
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
        ("calibration_consistent_conflict", "local_number_rate"),
        ("calibration_inconsistent_conflict", "atomic_number_rate"),
        ("calibration_absent_conflict", "local_number_rate"),
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
            "",
            "## Forbidden Claims",
            "",
            "- MC014 is a mechanism card.",
            "- MC014 supports intervention.",
            "- MC014 found an internal knowledge-control surface.",
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
    parser.add_argument("--artifact-prefix", default="mc014_inferred_reliability_numeric_structural")
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
            "purpose": "Generated-answer inferred-reliability numeric bridge behavior scoring before any MC014 hidden-state work.",
            "panels": list(PANELS),
            "primary_conflict_panels": list(PRIMARY_CONFLICT_PANELS),
            "sources": base_sources(args.limit_sources),
            "records": outputs,
            "summary": summary,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = args.artifact_prefix
        if prefix == "mc014_inferred_reliability_numeric_structural":
            prefix = "mc014_inferred_reliability_numeric_behavior"
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
        "purpose": "Structural prompt/channel audit before MC014 generated-answer behavior scoring.",
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
