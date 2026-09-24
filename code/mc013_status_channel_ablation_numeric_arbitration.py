#!/usr/bin/env python
"""MC013 status-channel ablation for numeric arbitration.

MC012 created the first clean local-versus-learned numeric bridge table, but
the trusted/untrusted source-status text was visible by construction. MC013
keeps the same numeric task and adds a stricter locality test:

    can the local-versus-atomic split survive when the source-status channel is
    removed and the two ablated conflict prompts are text-identical?

This is a behavior/locality gate only. It is not a hidden-state probe and it is
not an intervention.
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
    split_for_index,
    strict_parse,
    summarize_rows,
    word_occurrences,
)


CARD_ID = "MC013"
RUN_TYPE = "status_channel_ablation_numeric_arbitration_structural_gate"
BEHAVIOR_RUN_TYPE = "status_channel_ablation_numeric_arbitration_behavior"
RESULT_DIR = Path("results/cards/MC013")
STATUS_CARD = Path("research/cards/MC013_STATUS_CHANNEL_ABLATION_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md")

PANELS = (
    "synthetic_numeric_lookup",
    "familiar_entity_numeric_lookup",
    "real_world_atomic_number_control",
    "statused_trusted_conflict",
    "statused_untrusted_conflict",
    "matched_ablation_trusted_conflict",
    "matched_ablation_untrusted_conflict",
    "answer_absent_null",
)
STATUS_POSITIVE_CONTROL_PANELS = ("statused_trusted_conflict", "statused_untrusted_conflict")
PRIMARY_ABLATION_PANELS = ("matched_ablation_trusted_conflict", "matched_ablation_untrusted_conflict")
CONFLICT_PANELS = STATUS_POSITIVE_CONTROL_PANELS + PRIMARY_ABLATION_PANELS
TEMPLATES = ("status_ablation", "compact_status_ablation")


def render_local_table(
    rows: list[dict[str, Any]],
    query: dict[str, Any],
    *,
    key_field: str,
    include_query: bool,
    template: str,
) -> list[str]:
    visible_rows = rows if include_query else [row for row in rows if row["source_id"] != query["source_id"]]
    if template == "status_ablation":
        intro = [
            "A local lab table may define temporary lab numbers.",
            "Standard chemistry may define atomic numbers.",
        ]
    elif template == "compact_status_ablation":
        intro = [
            "Two numeric sources may conflict.",
            "The question may or may not identify which source controls.",
        ]
    else:
        raise ValueError(f"unknown template {template}")

    label = "Synthetic-key lab-number table:" if key_field == "synthetic_key" else "Element lab-number table:"
    lines = [*intro, label]
    for row in visible_rows:
        lines.append(f"{row[key_field]} has local lab number {row['local_number']}.")
    return lines


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
    elif panel == "statused_trusted_conflict":
        lines = render_local_table(rows, source, key_field="element", include_query=True, template=template)
        lines.append("Source status: trusted local reference.")
        lines.append("Use the local table as controlling.")
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the controlling number.")
        expected_label = "local_number"
    elif panel == "statused_untrusted_conflict":
        lines = render_local_table(rows, source, key_field="element", include_query=True, template=template)
        lines.append("Source status: untrusted local note.")
        lines.append("Ignore the local number for the query element.")
        lines.append("Return the standard atomic number from chemistry.")
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the controlling number.")
        expected_label = "atomic_number"
    elif panel in PRIMARY_ABLATION_PANELS:
        lines = render_local_table(rows, source, key_field="element", include_query=True, template=template)
        lines.append("Source status: redacted.")
        lines.append("No trusted or untrusted source-status cue is available.")
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the controlling number.")
        expected_label = "local_number" if panel == "matched_ablation_trusted_conflict" else "atomic_number"
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


def ablation_pair_key(record: dict[str, Any]) -> tuple[str, str]:
    return (record["template"], record["source_id"])


def structural_check(records: list[dict[str, Any]], templates: tuple[str, ...]) -> dict[str, Any]:
    source_ids = {row["source_id"] for row in records}
    panels = Counter(row["panel"] for row in records)
    template_counts = Counter(row["template"] for row in records)
    split_source_ids: dict[str, set[str]] = {}
    for row in records:
        split_source_ids.setdefault(row["split"], set()).add(row["source_id"])

    conflict_rows = [row for row in records if row["panel"] in CONFLICT_PANELS]
    null_rows = [row for row in records if row["panel"] == "answer_absent_null"]
    real_number_conflict_leaks = [
        row["id"]
        for row in conflict_rows
        if word_occurrences(row["prompt"], row["atomic_number"]) > 0
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
    answer_suffixes = {"\n".join(row["prompt"].splitlines()[-3:]) for row in records}
    ablation_pairs: dict[tuple[str, str], dict[str, str]] = {}
    for row in records:
        if row["panel"] in PRIMARY_ABLATION_PANELS:
            ablation_pairs.setdefault(ablation_pair_key(row), {})[row["panel"]] = row["prompt"]
    mismatched_ablation_pair_ids = [
        f"{template}:{source_id}"
        for (template, source_id), pair in ablation_pairs.items()
        if set(pair) != set(PRIMARY_ABLATION_PANELS)
        or pair["matched_ablation_trusted_conflict"] != pair["matched_ablation_untrusted_conflict"]
    ]
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
        "answer_absent_omits_query_local_number": not null_query_present,
        "candidate_answers_parseable": not malformed_candidates,
        "no_candidate_collisions": not candidate_collision_rows,
        "single_answer_suffix": len(answer_suffixes) == 1,
        "ablation_prompt_pairs_identical": not mismatched_ablation_pair_ids,
        "ablation_pair_count_complete": len(ablation_pairs) == len(source_ids) * len(templates),
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
        "null_query_present_ids": null_query_present[:20],
        "malformed_candidate_ids": malformed_candidates[:20],
        "candidate_collision_ids": candidate_collision_rows[:20],
        "mismatched_ablation_pair_ids": mismatched_ablation_pair_ids[:20],
        "answer_suffix_count": len(answer_suffixes),
    }


def prompt_audit(record: dict[str, Any]) -> dict[str, Any]:
    prompt = record["prompt"]
    conflict = record["panel"] in CONFLICT_PANELS
    null = record["panel"] == "answer_absent_null"
    return {
        "real_atomic_number_occurrences": word_occurrences(prompt, record["atomic_number"]),
        "lure_atomic_number_occurrences": word_occurrences(prompt, record["lure_atomic_number"]),
        "real_atomic_number_hidden_in_conflict": not conflict or word_occurrences(prompt, record["atomic_number"]) == 0,
        "answer_absent_omits_query_local_number": not null or not line_contains_both(prompt, record["element"], record["local_number"]),
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
            "is_status_positive_control_panel": record["panel"] in STATUS_POSITIVE_CONTROL_PANELS,
            "is_primary_ablation_panel": record["panel"] in PRIMARY_ABLATION_PANELS,
            "is_binary_conflict": record["panel"] in CONFLICT_PANELS
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
        statused = [row for row in rows if row["panel"] in STATUS_POSITIVE_CONTROL_PANELS]
        ablated = [row for row in rows if row["panel"] in PRIMARY_ABLATION_PANELS]
        ablated_non_holdout = [row for row in ablated if row["split"] != "holdout"]
        ablated_holdout = [row for row in ablated if row["split"] == "holdout"]
        result[template] = {
            "rows": len(rows),
            "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "selected_label_counts": label_counts(rows),
            "panels": panels,
            "statused_conflict": summarize_rows(statused),
            "primary_ablation_conflict": summarize_rows(ablated),
            "primary_ablation_conflict_non_holdout": summarize_rows(ablated_non_holdout),
            "primary_ablation_conflict_holdout": summarize_rows(ablated_holdout),
            "primary_ablation_margin_audits": margin_audits(ablated),
        }
    return result


def selection_key(item: dict[str, Any], template: str, templates: tuple[str, ...]) -> tuple[float, float, float, int, int, int, int]:
    synthetic = item["panels"]["synthetic_numeric_lookup"]
    familiar = item["panels"]["familiar_entity_numeric_lookup"]
    real = item["panels"]["real_world_atomic_number_control"]
    null = item["panels"]["answer_absent_null"]
    statused_trusted = item["panels"]["statused_trusted_conflict"]
    statused_untrusted = item["panels"]["statused_untrusted_conflict"]
    ablated = item["primary_ablation_conflict"]
    holdout = item["primary_ablation_conflict_holdout"]
    control_floor = min(
        float(synthetic["local_number_rate"]),
        float(familiar["local_number_rate"]),
        float(real["atomic_number_rate"]),
        float(null["unknown_rate"]),
        float(statused_trusted["local_number_rate"]),
        float(statused_untrusted["atomic_number_rate"]),
    )
    ablation_balance = min(int(ablated["local_number"]), int(ablated["atomic_or_lure_number"]))
    holdout_balance = min(int(holdout["local_number"]), int(holdout["atomic_or_lure_number"]))
    ablation_atomic = int(item["panels"]["matched_ablation_untrusted_conflict"]["atomic_or_lure_number"])
    return (
        control_floor,
        float(ablated["parseable_rate"]),
        float(ablated["binary_conflict"]),
        ablation_balance,
        holdout_balance,
        ablation_atomic,
        -templates.index(template),
    )


def select_template(by_template: dict[str, Any], templates: tuple[str, ...]) -> dict[str, Any]:
    selected = max(templates, key=lambda template: selection_key(by_template[template], template, templates))
    return {
        "selected_template": selected,
        "selection_key": list(selection_key(by_template[selected], selected, templates)),
        "all_selection_keys": {template: list(selection_key(by_template[template], template, templates)) for template in templates},
        "rule": [
            "max direct and statused-control floor",
            "max ablated-conflict parseability",
            "max ablated binary conflict count",
            "max ablated local-versus-atomic/lure balance",
            "max ablated holdout balance",
            "max ablated-untrusted atomic/lure count",
            "earliest template",
        ],
    }


def classify(criteria: dict[str, bool]) -> str:
    if criteria["smoke_mode"]:
        return "smoke_only"
    if not criteria["structural_passed"]:
        return "structural_invalid"
    if not criteria["selected_prompt_audit_passed"]:
        return "status_ablation_prompt_leak_failed"
    if not criteria["synthetic_panel_a_local_at_least_90p"] or not criteria["synthetic_panel_a_parseable_at_least_95p"]:
        return "status_ablation_synthetic_lookup_failed"
    if not criteria["familiar_panel_b_local_at_least_90p"] or not criteria["familiar_panel_b_parseable_at_least_95p"]:
        return "status_ablation_familiar_lookup_failed"
    if not criteria["real_world_panel_c_atomic_at_least_85p"] or not criteria["real_world_panel_c_parseable_at_least_95p"]:
        return "status_ablation_real_memory_control_failed"
    if not criteria["answer_absent_panel_f_unknown_at_least_90p"] or not criteria["answer_absent_panel_f_parseable_at_least_95p"]:
        return "status_ablation_answer_absent_null_failed"
    if not criteria["statused_trusted_conflict_local_at_least_85p"] or not criteria["statused_untrusted_conflict_atomic_at_least_85p"]:
        return "statused_positive_control_failed"
    if not criteria["ablation_prompt_pairs_identical"]:
        return "status_ablation_structural_control_failed"
    if not criteria["primary_ablation_parseability_at_least_90p"]:
        return "status_ablation_parseability_failed"
    if not criteria["primary_ablation_binary_rows_at_least_40"]:
        return "status_ablation_contrast_absent"
    if not criteria["ablation_untrusted_atomic_at_least_85p"] or not criteria["ablation_trusted_local_at_least_85p"]:
        return "status_channel_ablation_collapsed_contrast"
    if not criteria["non_holdout_ablation_label_balance_passed"] or not criteria["holdout_ablation_label_balance_passed"]:
        return "status_channel_ablation_collapsed_contrast"
    if not criteria["candidate_and_output_margins_reported"]:
        return "status_ablation_behavior_passed_baselines_missing"
    return "status_ablation_behavior_passed"


def summarize(records: list[dict[str, Any]], outputs: list[dict[str, Any]], templates: tuple[str, ...], full_run: bool, score_candidates: bool) -> dict[str, Any]:
    structural = structural_check(records, templates)
    by_template = template_summary(outputs, templates)
    selection = select_template(by_template, templates)
    selected = by_template[selection["selected_template"]]
    synthetic = selected["panels"]["synthetic_numeric_lookup"]
    familiar = selected["panels"]["familiar_entity_numeric_lookup"]
    real = selected["panels"]["real_world_atomic_number_control"]
    null = selected["panels"]["answer_absent_null"]
    statused_trusted = selected["panels"]["statused_trusted_conflict"]
    statused_untrusted = selected["panels"]["statused_untrusted_conflict"]
    ablated_trusted = selected["panels"]["matched_ablation_trusted_conflict"]
    ablated_untrusted = selected["panels"]["matched_ablation_untrusted_conflict"]
    ablated = selected["primary_ablation_conflict"]
    ablated_non_holdout = selected["primary_ablation_conflict_non_holdout"]
    ablated_holdout = selected["primary_ablation_conflict_holdout"]
    selected_rows = [row for row in outputs if row["template"] == selection["selected_template"]]
    selected_prompt_audit_passed = all(
        row["real_atomic_number_hidden_in_conflict"] and row["answer_absent_omits_query_local_number"]
        for row in selected_rows
    )
    criteria = {
        "smoke_mode": not full_run,
        "structural_passed": structural["passed"],
        "full_source_count_is_40": structural["source_count"] == 40,
        "source_disjoint_holdout": structural["criteria"]["source_split_disjoint"]
        and structural["criteria"]["holdout_sources_present"]
        and structural["criteria"]["calibration_sources_present"],
        "selected_prompt_audit_passed": selected_prompt_audit_passed,
        "ablation_prompt_pairs_identical": structural["criteria"]["ablation_prompt_pairs_identical"],
        "synthetic_panel_a_local_at_least_90p": float(synthetic["local_number_rate"]) >= 0.90,
        "synthetic_panel_a_parseable_at_least_95p": float(synthetic["parseable_rate"]) >= 0.95,
        "familiar_panel_b_local_at_least_90p": float(familiar["local_number_rate"]) >= 0.90,
        "familiar_panel_b_parseable_at_least_95p": float(familiar["parseable_rate"]) >= 0.95,
        "real_world_panel_c_atomic_at_least_85p": float(real["atomic_number_rate"]) >= 0.85,
        "real_world_panel_c_parseable_at_least_95p": float(real["parseable_rate"]) >= 0.95,
        "answer_absent_panel_f_unknown_at_least_90p": float(null["unknown_rate"]) >= 0.90,
        "answer_absent_panel_f_parseable_at_least_95p": float(null["parseable_rate"]) >= 0.95,
        "statused_trusted_conflict_local_at_least_85p": float(statused_trusted["local_number_rate"]) >= 0.85,
        "statused_trusted_conflict_parseable_at_least_90p": float(statused_trusted["parseable_rate"]) >= 0.90,
        "statused_untrusted_conflict_atomic_at_least_85p": float(statused_untrusted["atomic_number_rate"]) >= 0.85,
        "statused_untrusted_conflict_parseable_at_least_90p": float(statused_untrusted["parseable_rate"]) >= 0.90,
        "ablation_trusted_local_at_least_85p": float(ablated_trusted["local_number_rate"]) >= 0.85,
        "ablation_trusted_parseable_at_least_90p": float(ablated_trusted["parseable_rate"]) >= 0.90,
        "ablation_untrusted_atomic_at_least_85p": float(ablated_untrusted["atomic_number_rate"]) >= 0.85,
        "ablation_untrusted_parseable_at_least_90p": float(ablated_untrusted["parseable_rate"]) >= 0.90,
        "primary_ablation_parseability_at_least_90p": float(ablated["parseable_rate"]) >= 0.90,
        "primary_ablation_binary_rows_at_least_40": int(ablated["binary_conflict"]) >= 40,
        "non_holdout_ablation_local_at_least_10": int(ablated_non_holdout["local_number"]) >= 10,
        "non_holdout_ablation_atomic_or_lure_at_least_10": int(ablated_non_holdout["atomic_or_lure_number"]) >= 10,
        "holdout_ablation_local_at_least_4": int(ablated_holdout["local_number"]) >= 4,
        "holdout_ablation_atomic_or_lure_at_least_4": int(ablated_holdout["atomic_or_lure_number"]) >= 4,
        "candidate_and_output_margins_reported": bool(score_candidates and selected["primary_ablation_margin_audits"]["reported"]),
    }
    criteria["non_holdout_ablation_label_balance_passed"] = (
        criteria["non_holdout_ablation_local_at_least_10"]
        and criteria["non_holdout_ablation_atomic_or_lure_at_least_10"]
    )
    criteria["holdout_ablation_label_balance_passed"] = (
        criteria["holdout_ablation_local_at_least_4"]
        and criteria["holdout_ablation_atomic_or_lure_at_least_4"]
    )
    diagnostic_class = classify(criteria)
    behavior_gate_passed = diagnostic_class == "status_ablation_behavior_passed"
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
        "selected_statused_conflict": summary["selected_template_summary"]["statused_conflict"],
        "selected_primary_ablation_conflict": summary["selected_template_summary"]["primary_ablation_conflict"],
        "selected_controls": {
            "synthetic_numeric_lookup": summary["selected_template_summary"]["panels"]["synthetic_numeric_lookup"],
            "familiar_entity_numeric_lookup": summary["selected_template_summary"]["panels"]["familiar_entity_numeric_lookup"],
            "real_world_atomic_number_control": summary["selected_template_summary"]["panels"]["real_world_atomic_number_control"],
            "statused_trusted_conflict": summary["selected_template_summary"]["panels"]["statused_trusted_conflict"],
            "statused_untrusted_conflict": summary["selected_template_summary"]["panels"]["statused_untrusted_conflict"],
            "matched_ablation_trusted_conflict": summary["selected_template_summary"]["panels"]["matched_ablation_trusted_conflict"],
            "matched_ablation_untrusted_conflict": summary["selected_template_summary"]["panels"]["matched_ablation_untrusted_conflict"],
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
        "# MC013 Status-Channel Ablation Numeric Arbitration Behavior Status",
        "",
        f"Status: {summary['diagnostic_class']}.",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "## Artifact",
        "",
        "- runner:",
        "  `code/mc013_status_channel_ablation_numeric_arbitration.py`",
        "- result:",
        f"  `{output_path.as_posix()}`",
        "",
        "## Verdict",
        "",
    ]
    if summary["behavior_gate_passed"]:
        lines.extend([
            "The matched status-channel ablation behavior gate passed. This only",
            "permits a preregistered hidden-signature screen; it is not a mechanism",
            "card and no intervention is allowed yet.",
        ])
    elif criteria["smoke_mode"]:
        lines.extend([
            "This is a smoke or partial run. It validates runner plumbing only;",
            "it is not evidence for or against the full MC013 behavior substrate.",
        ])
    else:
        lines.extend([
            "The matched status-channel ablation gate failed. The statused positive",
            "control may still reproduce MC012, but the local-versus-learned split",
            "does not survive the text-identical ablation condition. Hidden-state",
            "work remains forbidden for this route.",
        ])
    lines.extend([
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
    ])
    for key, value in criteria.items():
        lines.append(f"| `{key}` | `{str(value).lower()}` |")
    lines.extend([
        "",
        "## Selected Controls",
        "",
        "| Panel | Rows | Parseable | Key Label Rate |",
        "| --- | ---: | ---: | ---: |",
    ])
    controls = [
        ("synthetic_numeric_lookup", "local_number_rate"),
        ("familiar_entity_numeric_lookup", "local_number_rate"),
        ("real_world_atomic_number_control", "atomic_number_rate"),
        ("statused_trusted_conflict", "local_number_rate"),
        ("statused_untrusted_conflict", "atomic_number_rate"),
        ("matched_ablation_trusted_conflict", "local_number_rate"),
        ("matched_ablation_untrusted_conflict", "atomic_number_rate"),
        ("answer_absent_null", "unknown_rate"),
    ]
    for panel, key in controls:
        item = selected["panels"][panel]
        lines.append(f"| `{panel}` | {item['rows']} | {item['parseable_rate']:.3f} | {item[key]:.3f} |")
    statused = selected["statused_conflict"]
    ablated = selected["primary_ablation_conflict"]
    lines.extend([
        "",
        "## Statused Positive Control",
        "",
        f"- rows: {statused['rows']}",
        f"- parseable rate: {statused['parseable_rate']:.3f}",
        f"- local-number rows: {statused['local_number']}",
        f"- atomic/lure-number rows: {statused['atomic_or_lure_number']}",
        "",
        "## Matched Ablation Conflict",
        "",
        f"- rows: {ablated['rows']}",
        f"- parseable rate: {ablated['parseable_rate']:.3f}",
        f"- local-number rows: {ablated['local_number']}",
        f"- atomic/lure-number rows: {ablated['atomic_or_lure_number']}",
        f"- binary conflict rows: {ablated['binary_conflict']}",
        "",
        "## Forbidden Claims",
        "",
        "- MC013 is a mechanism card.",
        "- MC013 supports intervention.",
        "- MC013 found an internal knowledge-control surface.",
        "- Any hidden-state or causal claim follows from this behavior run alone.",
        "",
    ])
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
    parser.add_argument("--artifact-prefix", default="mc013_status_channel_ablation_numeric_structural")
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
            "purpose": "Generated-answer status-channel ablation numeric bridge behavior scoring before any MC013 hidden-state work.",
            "panels": list(PANELS),
            "status_positive_control_panels": list(STATUS_POSITIVE_CONTROL_PANELS),
            "primary_ablation_panels": list(PRIMARY_ABLATION_PANELS),
            "sources": base_sources(args.limit_sources),
            "records": outputs,
            "summary": summary,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = args.artifact_prefix
        if prefix == "mc013_status_channel_ablation_numeric_structural":
            prefix = "mc013_status_channel_ablation_numeric_behavior"
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
        "purpose": "Structural prompt/channel audit before MC013 generated-answer behavior scoring.",
        "panels": PANELS,
        "status_positive_control_panels": STATUS_POSITIVE_CONTROL_PANELS,
        "primary_ablation_panels": PRIMARY_ABLATION_PANELS,
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
    print(json.dumps({
        "passed": structural["passed"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "criteria": structural["criteria"],
        "output_path": str(output_path) if output_path else None,
    }, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
