#!/usr/bin/env python
"""KSQ003 statusless evidence-aggregation first run.

This is a behavior-substrate admission runner for the highest-priority
knowledge-substrate candidate. It tests a new bridge class before hidden-state
work: branch selection depends on multiple content evidence notes about the
queried element, not trusted/untrusted labels, row codes, checksums,
cross-table consistency, or row-local atomic-number claims.

The runner is structural by default. Model scoring is optional and remains
behavior-only; it does not license hidden-state signatures or interventions.
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
from mc016_alphabet_gated_numeric_arbitration import load_model_and_tokenizer


CARD_ID = "KSQ003"
CANDIDATE_ID = "ksq003_bridge_statusless_evidence_aggregation"
RUN_TYPE = "ksq003_statusless_evidence_aggregation_structural_gate"
BEHAVIOR_RUN_TYPE = "ksq003_statusless_evidence_aggregation_behavior"
RESULT_DIR = Path("results/cards/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION")
DEFAULT_RESULT_PATH = (
    RESULT_DIR / "ksq003_bridge_statusless_evidence_aggregation_first_run.json"
)
SMOKE_LIMIT10_RESULT_PATH = (
    RESULT_DIR / "ksq003_bridge_statusless_evidence_aggregation_smoke_limit10.json"
)
STATUS_CARD = Path(
    "research/cards/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION_FIRST_RUN_STATUS.md"
)
PREREG_PATH = Path(
    "research/prereg/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION_FIRST_RUN.md"
)

PANELS = (
    "source_local_direct_control",
    "learned_fact_direct_control",
    "all_evidence_fit_conflict",
    "one_evidence_mismatch_conflict",
    "symbol_only_ablation",
    "parity_only_ablation",
    "answer_absent_and_side_null",
)
PRIMARY_CONFLICT_PANELS = (
    "all_evidence_fit_conflict",
    "one_evidence_mismatch_conflict",
)
ABLATION_PANELS = ("symbol_only_ablation", "parity_only_ablation")
TEMPLATES = ("evidence_packet", "field_observations", "compact_fit")

STATUS_LEXEME_RE = re.compile(
    r"\b(trusted|untrusted|reliable|unreliable|status)\b",
    re.IGNORECASE,
)
FORBIDDEN_ROUTE_RE = re.compile(
    r"\b(checksum|equation|arithmetic|cross-table|second table|fact-claim|standard-number claim)\b",
    re.IGNORECASE,
)

SYMBOLS = {
    "hydrogen": "H",
    "helium": "He",
    "lithium": "Li",
    "beryllium": "Be",
    "boron": "B",
    "carbon": "C",
    "nitrogen": "N",
    "oxygen": "O",
    "fluorine": "F",
    "neon": "Ne",
    "sodium": "Na",
    "magnesium": "Mg",
    "aluminum": "Al",
    "silicon": "Si",
    "phosphorus": "P",
    "sulfur": "S",
    "chlorine": "Cl",
    "argon": "Ar",
    "potassium": "K",
    "calcium": "Ca",
    "titanium": "Ti",
    "vanadium": "V",
    "chromium": "Cr",
    "manganese": "Mn",
    "iron": "Fe",
    "cobalt": "Co",
    "nickel": "Ni",
    "copper": "Cu",
    "zinc": "Zn",
    "bromine": "Br",
    "silver": "Ag",
    "tin": "Sn",
    "iodine": "I",
    "barium": "Ba",
    "tungsten": "W",
    "platinum": "Pt",
    "gold": "Au",
    "mercury": "Hg",
    "lead": "Pb",
    "uranium": "U",
}


def source_symbol(source: dict[str, Any]) -> str:
    return SYMBOLS[source["source_id"]]


def source_parity(source: dict[str, Any]) -> str:
    return "even" if int(source["atomic_number"]) % 2 == 0 else "odd"


def wrong_symbol(sources: list[dict[str, Any]], source: dict[str, Any]) -> str:
    for candidate in distractors(sources, source, 8):
        if candidate["source_id"] != source["source_id"]:
            return source_symbol(candidate)
    raise AssertionError("no wrong symbol available")


def wrong_parity(source: dict[str, Any]) -> str:
    return "odd" if source_parity(source) == "even" else "even"


def render_local_table(
    rows: list[dict[str, Any]],
    query: dict[str, Any],
    *,
    include_query: bool,
    template: str,
) -> list[str]:
    visible_rows = (
        rows
        if include_query
        else [row for row in rows if row["source_id"] != query["source_id"]]
    )
    if template == "evidence_packet":
        intro = [
            "A lab table may list temporary lab numbers.",
            "A separate evidence packet may describe the queried element.",
        ]
    elif template == "field_observations":
        intro = [
            "Field observations may accompany a temporary lab-number table.",
            "Use the observations to decide whether the table controls.",
        ]
    elif template == "compact_fit":
        intro = [
            "Two numeric sources may conflict.",
            "Use the fit of the evidence notes to decide the controlling number.",
        ]
    else:
        raise ValueError(f"unknown template {template}")
    lines = [*intro, "Element lab-number table:"]
    for row in visible_rows:
        lines.append(f"{row['element']} has local lab number {row['local_number']}.")
    return lines


def append_evidence_rule(lines: list[str], template: str) -> None:
    if template == "evidence_packet":
        lines.extend(
            [
                "The evidence packet has two notes about the queried element.",
                "If both notes fit the queried element, return the local lab number.",
                "If either note does not fit the queried element, ignore the local lab number.",
                "When ignoring the local lab number, return the standard atomic number.",
                "If the evidence packet is incomplete, return UNKNOWN.",
            ]
        )
    elif template == "field_observations":
        lines.extend(
            [
                "Compare the field observations with the queried element.",
                "Both observations fit -> local lab number.",
                "Any observation does not fit -> standard atomic number.",
                "Only one observation present -> UNKNOWN.",
            ]
        )
    elif template == "compact_fit":
        lines.extend(
            [
                "Rule: both evidence notes fit -> local lab number.",
                "Rule: any evidence note mismatches -> standard atomic number.",
                "Rule: only one evidence note -> UNKNOWN.",
            ]
        )
    else:
        raise ValueError(f"unknown template {template}")


def append_evidence_notes(
    lines: list[str],
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    mode: str,
) -> None:
    if mode == "all_fit":
        notes = [
            f"Evidence note alpha: chemical symbol {source_symbol(source)}.",
            f"Evidence note beta: standard atomic-number parity {source_parity(source)}.",
        ]
    elif mode == "one_mismatch":
        notes = [
            f"Evidence note alpha: chemical symbol {wrong_symbol(sources, source)}.",
            f"Evidence note beta: standard atomic-number parity {wrong_parity(source)}.",
        ]
    elif mode == "symbol_only":
        notes = [f"Evidence note alpha: chemical symbol {source_symbol(source)}."]
    elif mode == "parity_only":
        notes = [
            f"Evidence note alpha: standard atomic-number parity {source_parity(source)}."
        ]
    else:
        raise ValueError(f"unknown evidence mode {mode}")
    lines.append("Evidence notes:")
    lines.extend(notes)


def make_prompt(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    panel: str,
    template: str,
) -> tuple[str, str, str | None]:
    rows = distractors(sources, source, 4)
    evidence_mode: str | None = None
    if panel == "source_local_direct_control":
        lines = render_local_table(rows, source, include_query=True, template=template)
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the local lab number.")
        expected_label = "local_number"
    elif panel == "learned_fact_direct_control":
        lines = [
            "No local lab table is active.",
            "Return the standard atomic number for the queried element.",
            f"Query element: {source['element']}",
        ]
        expected_label = "atomic_number"
    elif panel == "all_evidence_fit_conflict":
        lines = render_local_table(rows, source, include_query=True, template=template)
        append_evidence_notes(lines, sources, source, mode="all_fit")
        append_evidence_rule(lines, template)
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the controlling number.")
        expected_label = "local_number"
        evidence_mode = "all_fit"
    elif panel == "one_evidence_mismatch_conflict":
        lines = render_local_table(rows, source, include_query=True, template=template)
        append_evidence_notes(lines, sources, source, mode="one_mismatch")
        append_evidence_rule(lines, template)
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the controlling number.")
        expected_label = "atomic_number"
        evidence_mode = "one_mismatch"
    elif panel == "symbol_only_ablation":
        lines = render_local_table(rows, source, include_query=True, template=template)
        append_evidence_notes(lines, sources, source, mode="symbol_only")
        append_evidence_rule(lines, template)
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the controlling number.")
        expected_label = "unknown"
        evidence_mode = "symbol_only"
    elif panel == "parity_only_ablation":
        lines = render_local_table(rows, source, include_query=True, template=template)
        append_evidence_notes(lines, sources, source, mode="parity_only")
        append_evidence_rule(lines, template)
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the controlling number.")
        expected_label = "unknown"
        evidence_mode = "parity_only"
    elif panel == "answer_absent_and_side_null":
        lines = render_local_table(rows, source, include_query=False, template=template)
        append_evidence_notes(lines, sources, source, mode="all_fit")
        lines.append(f"Query element: {source['element']}")
        lines.append("This is a local-table membership check.")
        lines.append(
            "If the query element is absent from the local lab table, return UNKNOWN even if evidence notes fit."
        )
        expected_label = "unknown"
        evidence_mode = "all_fit"
    else:
        raise ValueError(f"unknown panel {panel}")
    return "\n".join([*lines, *response_suffix()]), expected_label, evidence_mode


def source_records(
    limit_sources: int | None = None,
    templates: tuple[str, ...] = TEMPLATES,
    run_type: str = RUN_TYPE,
) -> list[dict[str, Any]]:
    sources = base_sources(limit_sources)
    records: list[dict[str, Any]] = []
    for source in sources:
        for template in templates:
            for panel_name in PANELS:
                prompt, expected_label, evidence_mode = make_prompt(
                    sources,
                    source,
                    panel=panel_name,
                    template=template,
                )
                records.append(
                    {
                        "id": f"{CARD_ID}_{template}_{panel_name}_{source['source_id']}",
                        "card_id": CARD_ID,
                        "candidate_id": CANDIDATE_ID,
                        "run_type": run_type,
                        "model_id": MODEL_ID,
                        "template": template,
                        "panel": panel_name,
                        "split": source["split"],
                        "source_id": source["source_id"],
                        "source_index": source["source_index"],
                        "element": source["element"],
                        "element_symbol": source_symbol(source),
                        "atomic_parity": source_parity(source),
                        "synthetic_key": source["synthetic_key"],
                        "atomic_number": str(source["atomic_number"]),
                        "lure_atomic_number": str(source["lure_atomic_number"]),
                        "local_number": str(source["local_number"]),
                        "evidence_mode": evidence_mode,
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
    conflict_rows = [
        row
        for row in records
        if row["panel"] in PRIMARY_CONFLICT_PANELS + ABLATION_PANELS
    ]
    null_rows = [row for row in records if row["panel"] == "answer_absent_and_side_null"]
    expected_rows = len(source_ids) * len(templates) * len(PANELS)
    primary_expected = Counter(row["expected_label"] for row in primary_rows)
    real_number_leaks = [
        row["id"]
        for row in conflict_rows
        if word_occurrences(row["prompt"], row["atomic_number"]) > 0
    ]
    lure_number_leaks = [
        row["id"]
        for row in conflict_rows
        if word_occurrences(row["prompt"], row["lure_atomic_number"]) > 0
    ]
    local_number_evidence_leaks = [
        row["id"]
        for row in conflict_rows
        if word_occurrences(row["prompt"], row["local_number"]) > 1
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
    status_lexeme_rows = [
        row["id"] for row in primary_rows if STATUS_LEXEME_RE.search(row["prompt"])
    ]
    forbidden_route_rows = [
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
        "real_atomic_number_hidden_in_conflicts": not real_number_leaks,
        "lure_atomic_number_hidden_in_conflicts": not lure_number_leaks,
        "local_number_not_used_as_evidence": not local_number_evidence_leaks,
        "answer_absent_omits_query_local_number": not null_query_present,
        "candidate_answers_parseable": not malformed_candidates,
        "no_candidate_collisions": not candidate_collision_rows,
        "single_answer_suffix": len(answer_suffixes) == 1,
        "primary_prompts_have_no_status_lexemes": not status_lexeme_rows,
        "primary_prompts_avoid_closed_route_lexemes": not forbidden_route_rows,
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
        "real_number_leak_ids": real_number_leaks[:20],
        "lure_number_leak_ids": lure_number_leaks[:20],
        "local_number_evidence_leak_ids": local_number_evidence_leaks[:20],
        "null_query_present_ids": null_query_present[:20],
        "malformed_candidate_ids": malformed_candidates[:20],
        "candidate_collision_ids": candidate_collision_rows[:20],
        "status_lexeme_rows": status_lexeme_rows[:20],
        "forbidden_route_rows": forbidden_route_rows[:20],
        "answer_suffix_count": len(answer_suffixes),
    }


def prompt_audit(record: dict[str, Any]) -> dict[str, Any]:
    prompt = record["prompt"]
    conflict = record["panel"] in PRIMARY_CONFLICT_PANELS + ABLATION_PANELS
    primary = record["panel"] in PRIMARY_CONFLICT_PANELS
    null = record["panel"] == "answer_absent_and_side_null"
    return {
        "real_atomic_number_occurrences": word_occurrences(prompt, record["atomic_number"]),
        "lure_atomic_number_occurrences": word_occurrences(
            prompt,
            record["lure_atomic_number"],
        ),
        "real_atomic_number_hidden_in_conflict": not conflict
        or word_occurrences(prompt, record["atomic_number"]) == 0,
        "lure_atomic_number_hidden_in_conflict": not conflict
        or word_occurrences(prompt, record["lure_atomic_number"]) == 0,
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
            and parsed["selected_label"]
            in {"local_number", "atomic_number", "lure_atomic_number"},
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
        panels = {
            panel_name: summarize_rows([row for row in rows if row["panel"] == panel_name])
            for panel_name in PANELS
        }
        conflict = [row for row in rows if row["panel"] in PRIMARY_CONFLICT_PANELS]
        non_holdout = [row for row in conflict if row["split"] != "holdout"]
        holdout = [row for row in conflict if row["split"] == "holdout"]
        expected_local = [
            row for row in conflict if row["expected_label"] == "local_number"
        ]
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


def selection_key(item: dict[str, Any], template: str, templates: tuple[str, ...]) -> tuple[float, float, float, int, int, int]:
    local = item["panels"]["source_local_direct_control"]
    learned = item["panels"]["learned_fact_direct_control"]
    fit = item["panels"]["all_evidence_fit_conflict"]
    mismatch = item["panels"]["one_evidence_mismatch_conflict"]
    symbol_only = item["panels"]["symbol_only_ablation"]
    parity_only = item["panels"]["parity_only_ablation"]
    null = item["panels"]["answer_absent_and_side_null"]
    conflict = item["primary_conflict"]
    non_holdout = item["primary_conflict_non_holdout"]
    holdout = item["primary_conflict_holdout"]
    control_floor = min(
        float(local["local_number_rate"]),
        float(learned["atomic_number_rate"]),
        float(fit["local_number_rate"]),
        float(mismatch["atomic_number_rate"]),
        float(symbol_only["unknown_rate"]),
        float(parity_only["unknown_rate"]),
        float(null["unknown_rate"]),
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
        float(conflict["binary_conflict"]),
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
            "max direct/control plus evidence-conflict floor",
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
    if not criteria["selected_prompt_audit_passed"] or not criteria["primary_prompts_have_no_status_or_closed_route_lexemes"]:
        return "statusless_evidence_prompt_leak_failed"
    if not criteria["source_local_direct_control_passed"]:
        return "statusless_evidence_local_control_failed"
    if not criteria["learned_fact_direct_control_passed"]:
        return "statusless_evidence_learned_control_failed"
    if not criteria["answer_absent_null_passed"]:
        return "statusless_evidence_null_failed"
    if not criteria["all_evidence_fit_local_passed"]:
        return "statusless_evidence_fit_branch_failed"
    if not criteria["one_evidence_mismatch_atomic_passed"]:
        return "statusless_evidence_mismatch_branch_failed"
    if not criteria["single_feature_ablation_unknown_passed"]:
        return "statusless_evidence_visible_channel"
    if not criteria["primary_conflict_parseability_at_least_90p"]:
        return "statusless_evidence_conflict_parseability_failed"
    if not criteria["primary_conflict_expected_correct_at_least_70p"]:
        return "statusless_evidence_expected_branch_failed"
    if not criteria["non_holdout_conflict_label_balance_passed"] or not criteria["holdout_conflict_label_balance_passed"]:
        return "statusless_evidence_conflict_balance_failed"
    if not criteria["candidate_and_output_margins_reported"]:
        return "statusless_evidence_behavior_passed_baselines_missing"
    return "statusless_evidence_behavior_ready"


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
    local = selected["panels"]["source_local_direct_control"]
    learned = selected["panels"]["learned_fact_direct_control"]
    fit = selected["panels"]["all_evidence_fit_conflict"]
    mismatch = selected["panels"]["one_evidence_mismatch_conflict"]
    symbol_only = selected["panels"]["symbol_only_ablation"]
    parity_only = selected["panels"]["parity_only_ablation"]
    null = selected["panels"]["answer_absent_and_side_null"]
    conflict = selected["primary_conflict"]
    non_holdout = selected["primary_conflict_non_holdout"]
    holdout = selected["primary_conflict_holdout"]
    selected_rows = [
        row for row in outputs if row["template"] == selection["selected_template"]
    ]
    selected_prompt_audit_passed = all(
        row["real_atomic_number_hidden_in_conflict"]
        and row["lure_atomic_number_hidden_in_conflict"]
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
        "source_local_direct_control_passed": float(local["parseable_rate"]) >= 0.90
        and float(local["local_number_rate"]) >= 0.85,
        "learned_fact_direct_control_passed": float(learned["parseable_rate"]) >= 0.90
        and float(learned["atomic_number_rate"]) >= 0.75,
        "answer_absent_null_passed": float(null["unknown_rate"]) >= 0.80
        and float(null["atomic_or_lure_number_rate"]) <= 0.10,
        "all_evidence_fit_local_passed": float(fit["local_number_rate"]) >= 0.70,
        "one_evidence_mismatch_atomic_passed": float(mismatch["atomic_number_rate"]) >= 0.70,
        "single_feature_ablation_unknown_passed": float(symbol_only["unknown_rate"]) >= 0.60
        and float(parity_only["unknown_rate"]) >= 0.60,
        "primary_conflict_parseability_at_least_90p": float(conflict["parseable_rate"]) >= 0.90,
        "primary_conflict_binary_rows_at_least_40": int(conflict["binary_conflict"]) >= 40,
        "primary_conflict_expected_correct_at_least_70p": (
            expected_correct / max(int(conflict["rows"]), 1)
        )
        >= 0.70,
        "non_holdout_conflict_local_at_least_24": int(non_holdout["local_number"]) >= 24,
        "non_holdout_conflict_atomic_or_lure_at_least_24": int(
            non_holdout["atomic_or_lure_number"]
        )
        >= 24,
        "holdout_conflict_local_at_least_8": int(holdout["local_number"]) >= 8,
        "holdout_conflict_atomic_or_lure_at_least_8": int(
            holdout["atomic_or_lure_number"]
        )
        >= 8,
        "candidate_and_output_margins_reported": bool(
            score_candidates and selected["primary_conflict_margin_audits"]["reported"]
        ),
    }
    criteria["non_holdout_conflict_label_balance_passed"] = (
        criteria["non_holdout_conflict_local_at_least_24"]
        and criteria["non_holdout_conflict_atomic_or_lure_at_least_24"]
    )
    criteria["holdout_conflict_label_balance_passed"] = (
        criteria["holdout_conflict_local_at_least_8"]
        and criteria["holdout_conflict_atomic_or_lure_at_least_8"]
    )
    diagnostic_class = classify(criteria)
    behavior_gate_passed = diagnostic_class == "statusless_evidence_behavior_ready"
    behavior_candidate = (
        diagnostic_class == "statusless_evidence_behavior_passed_baselines_missing"
    )
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
        "signature_ready": False,
        "intervention_ready": False,
        "hidden_state_allowed": False,
        "diagnostic_class": diagnostic_class,
    }


def quiet_summary(summary: dict[str, Any], output_path: Path | None = None) -> dict[str, Any]:
    payload = {
        "diagnostic_class": summary["diagnostic_class"],
        "passed": summary["passed"],
        "behavior_ready": summary["behavior_ready"],
        "behavior_candidate": summary["behavior_candidate"],
        "signature_ready": summary["signature_ready"],
        "hidden_state_allowed": summary["hidden_state_allowed"],
        "criteria": summary["criteria"],
        "selection": summary["selection"],
        "selected_primary_conflict": summary["selected_template_summary"]["primary_conflict"],
        "selected_controls": {
            panel_name: summary["selected_template_summary"]["panels"][panel_name]
            for panel_name in PANELS
        },
    }
    if output_path is not None:
        payload["output_path"] = str(output_path)
    return payload


def write_prereg(path: Path) -> None:
    lines = [
        "# KSQ003 Bridge Statusless Evidence Aggregation First Run",
        "",
        "Status: behavior-substrate first-run preregistration; no hidden-state work.",
        "",
        "Runner:",
        "",
        "> `code/ksq003_bridge_statusless_evidence_aggregation_first_run.py`",
        "",
        "Default result:",
        "",
        "> `results/cards/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION/ksq003_bridge_statusless_evidence_aggregation_first_run.json`",
        "",
        "Default 10-source smoke result:",
        "",
        "> `results/cards/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION/ksq003_bridge_statusless_evidence_aggregation_smoke_limit10.json`",
        "",
        "Status card:",
        "",
        "> `research/cards/KSQ003_BRIDGE_STATUSLESS_EVIDENCE_AGGREGATION_FIRST_RUN_STATUS.md`",
        "",
        "## Purpose",
        "",
        "Test a statusless evidence-aggregation bridge before any hidden-state",
        "signature work. Branch selection depends on symbol/parity evidence",
        "about the queried element, not source status labels, row codes,",
        "checksums, cross-table consistency, or row-local atomic-number claims.",
        "",
        "## Panels",
        "",
    ]
    for panel_name in PANELS:
        lines.append(f"- `{panel_name}`")
    lines.extend(
        [
            "",
            "## Decision Boundary",
            "",
            "Promote only to behavior-substrate admission if structural checks,",
            "direct controls, conflict mixture, ablations, nulls, source-disjoint",
            "holdout, and candidate/output baselines pass together.",
            "",
            "Death rule: kill the candidate if the learned branch collapses under",
            "table pressure, if evidence features behave like visible status",
            "labels, if nulls fail, or if output/candidate baselines explain the",
            "branch.",
            "",
            "Forbidden claims:",
            "",
            "- KSQ003 is a mechanism card.",
            "- KSQ003 licenses hidden-state search before the behavior gate passes.",
            "- KSQ003 proves a learned-memory bridge.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_status_card(path: Path, result: dict[str, Any], output_path: Path) -> None:
    smoke_result = None
    if "summary" not in result and SMOKE_LIMIT10_RESULT_PATH.exists():
        smoke_result = json.loads(SMOKE_LIMIT10_RESULT_PATH.read_text(encoding="utf-8"))

    lines = [
        "# KSQ003 Bridge Statusless Evidence Aggregation First-Run Status",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "Runner:",
        "",
        "> `code/ksq003_bridge_statusless_evidence_aggregation_first_run.py`",
        "",
        "Result:",
        "",
        f"> `{output_path.as_posix()}`",
        "",
    ]
    if "summary" not in result:
        structural = result["structural"]
        if smoke_result is not None:
            status = (
                "structural_passed_behavior_smoke_failed"
                if structural["passed"]
                and not smoke_result["summary"]["behavior_ready"]
                else "structural_passed_behavior_smoke_passed"
            )
        else:
            status = (
                "structural_passed_behavior_not_run"
                if structural["passed"]
                else "structural_failed"
            )
        lines.extend(
            [
                f"Status: {status}.",
                "",
                "## Verdict",
                "",
            ]
        )
        if smoke_result is not None:
            lines.extend(
                [
                    "The structural gate passed, but the 10-source model smoke",
                    "failed behavior admission: the model followed the local",
                    "table even when evidence mismatched or was incomplete.",
                    "Hidden-state work remains forbidden.",
                ]
            )
        else:
            lines.append(
                "The structural gate passed. This licenses model-scored behavior "
                "admission only, not hidden-state work."
                if structural["passed"]
                else "The structural gate failed. Do not run model scoring until repaired."
            )
        lines.extend(
            [
                "",
                "## Structural Criteria",
                "",
                "| Criterion | Passed |",
                "| --- | --- |",
            ]
        )
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
            ]
        )
        if smoke_result is not None:
            summary = smoke_result["summary"]
            selected = summary["selected_template_summary"]
            criteria = summary["criteria"]
            lines.extend(
                [
                    "",
                    "## Ten-Source Behavior Smoke",
                    "",
                    f"- result: `{SMOKE_LIMIT10_RESULT_PATH.as_posix()}`",
                    f"- diagnostic class: `{summary['diagnostic_class']}`",
                    f"- behavior ready: `{str(summary['behavior_ready']).lower()}`",
                    f"- selected template: `{summary['selection']['selected_template']}`",
                    f"- records: `{summary['structural']['record_count']}`",
                    f"- sources: `{summary['structural']['source_count']}`",
                    f"- candidate/output margins reported: `{str(criteria['candidate_and_output_margins_reported']).lower()}`",
                    "",
                    "| Panel | Label counts |",
                    "| --- | --- |",
                ]
            )
            for panel_name in [
                "source_local_direct_control",
                "learned_fact_direct_control",
                "all_evidence_fit_conflict",
                "one_evidence_mismatch_conflict",
                "symbol_only_ablation",
                "parity_only_ablation",
                "answer_absent_and_side_null",
            ]:
                labels = selected["panels"][panel_name]["label_counts"]
                lines.append(
                    f"| `{panel_name}` | `{json.dumps(labels, sort_keys=True)}` |"
                )
            lines.extend(
                [
                    "",
                    "Smoke interpretation: direct local-number control, learned",
                    "atomic-number control, all-evidence-fit local routing, and",
                    "answer-absent nulls worked in the selected template. The",
                    "one-evidence-mismatch branch and both single-feature",
                    "ablations collapsed to local-number outputs, so KSQ003 is",
                    "not a behavior-ready substrate.",
                ]
            )
    else:
        summary = result["summary"]
        selected = summary["selected_template_summary"]
        criteria = summary["criteria"]
        lines.extend(
            [
                f"Status: {summary['diagnostic_class']}.",
                "",
                "## Verdict",
                "",
            ]
        )
        if summary["behavior_gate_passed"]:
            lines.append("The behavior gate passed. This licenses only the next controlled signature-screen decision.")
        elif criteria["smoke_mode"]:
            lines.append("This is a smoke or partial run. Hidden-state work remains forbidden.")
        else:
            lines.append("The behavior gate failed. Hidden-state work remains forbidden.")
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
            ("source_local_direct_control", "local_number_rate"),
            ("learned_fact_direct_control", "atomic_number_rate"),
            ("all_evidence_fit_conflict", "local_number_rate"),
            ("one_evidence_mismatch_conflict", "atomic_number_rate"),
            ("symbol_only_ablation", "unknown_rate"),
            ("parity_only_ablation", "unknown_rate"),
            ("answer_absent_and_side_null", "unknown_rate"),
        ]
        for panel_name, key in controls:
            item = selected["panels"][panel_name]
            lines.append(
                f"| `{panel_name}` | {item['rows']} | {item['parseable_rate']:.3f} | {item[key]:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Forbidden Claims",
            "",
            "- KSQ003 is a mechanism card.",
            "- KSQ003 supports intervention.",
            "- KSQ003 found an internal knowledge-control surface.",
            "- Any hidden-state or causal claim follows from this first run alone.",
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
    parser.add_argument("--write-manifest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--status-card", type=Path, default=STATUS_CARD)
    parser.add_argument("--prereg-path", type=Path, default=PREREG_PATH)
    parser.add_argument("--write-status-card", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-prereg", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.time()
    run_type = BEHAVIOR_RUN_TYPE if args.score_model else RUN_TYPE
    records = source_records(args.limit_sources, TEMPLATES, run_type)
    structural = structural_check(records, TEMPLATES)

    if args.write_prereg:
        write_prereg(args.prereg_path)

    if not args.score_model:
        result = {
            "schema_version": 1,
            "card_id": CARD_ID,
            "candidate_id": CANDIDATE_ID,
            "run_type": RUN_TYPE,
            "model_id": args.model_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "purpose": "Structural gate for the KSQ003 statusless evidence-aggregation first run.",
            "templates": list(TEMPLATES),
            "panels": list(PANELS),
            "primary_conflict_panels": list(PRIMARY_CONFLICT_PANELS),
            "hidden_state_allowed": False,
            "structural": structural,
            "records": records,
        }
        if args.write_manifest:
            write_json(args.output_path, result)
            if args.write_status_card:
                write_status_card(args.status_card, result, args.output_path)
        print(
            json.dumps(
                {
                    "passed": structural["passed"],
                    "record_count": structural["record_count"],
                    "source_count": structural["source_count"],
                    "criteria": structural["criteria"],
                    "output_path": str(args.output_path) if args.write_manifest else None,
                    "status_card": str(args.status_card) if args.write_status_card else None,
                    "prereg": str(args.prereg_path) if args.write_prereg else None,
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
    result = {
        "schema_version": 1,
        "card_id": CARD_ID,
        "candidate_id": CANDIDATE_ID,
        "run_type": BEHAVIOR_RUN_TYPE,
        "model_id": args.model_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "max_new_tokens": args.max_new_tokens,
        "decoding": {"do_sample": False},
        "score_candidates": args.score_candidates,
        "limit_sources": args.limit_sources,
        "templates": list(TEMPLATES),
        "elapsed_s": time.time() - started,
        "purpose": "Behavior-only statusless evidence-aggregation bridge admission run.",
        "panels": list(PANELS),
        "primary_conflict_panels": list(PRIMARY_CONFLICT_PANELS),
        "sources": base_sources(args.limit_sources),
        "hidden_state_allowed": False,
        "structural": structural,
        "records": outputs,
        "summary": summary,
    }
    write_json(args.output_path, result)
    if args.write_status_card:
        write_status_card(args.status_card, result, args.output_path)
    print(json.dumps(quiet_summary(summary, args.output_path), indent=2, ensure_ascii=True))
    return 0 if (summary["passed"] or not full_run) else 2


if __name__ == "__main__":
    raise SystemExit(main())
