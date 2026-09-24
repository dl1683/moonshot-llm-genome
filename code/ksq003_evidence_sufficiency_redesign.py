#!/usr/bin/env python
"""KSQ003 evidence-sufficiency bridge redesign.

This is the material follow-up to the first KSQ003 statusless evidence run.
The first design asked the model to use a local lab number when evidence fit
and a learned atomic number when evidence mismatched. The smoke run collapsed
mismatch and single-feature ablation rows to the local table.

This redesign changes the contract instead of the wording. The local table is
kept as a visible distractor and direct control, but conflict rows ask for the
standard atomic number only when a two-note identity packet is complete and
consistent with the queried element. Incomplete or contradictory packets must
return UNKNOWN. No hidden-state work is licensed here; this is only a behavior
substrate admission test.
"""

from __future__ import annotations

import argparse
import json
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
from ksq003_bridge_statusless_evidence_aggregation_first_run import (
    source_symbol,
    wrong_symbol,
)


CARD_ID = "KSQ003"
REDESIGN_ID = "ksq003_evidence_sufficiency_redesign"
RUN_TYPE = "ksq003_evidence_sufficiency_redesign_structural_gate"
BEHAVIOR_RUN_TYPE = "ksq003_evidence_sufficiency_redesign_behavior"
RESULT_DIR = Path("results/cards/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN")
DEFAULT_RESULT_PATH = RESULT_DIR / "ksq003_evidence_sufficiency_redesign_first_run.json"
SMOKE_LIMIT10_RESULT_PATH = (
    RESULT_DIR / "ksq003_evidence_sufficiency_redesign_smoke_limit10.json"
)
FULL_BEHAVIOR_RESULT_PATH = (
    RESULT_DIR / "ksq003_evidence_sufficiency_redesign_full_behavior.json"
)
STATUS_CARD = Path("research/cards/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN_STATUS.md")
PREREG_PATH = Path("research/prereg/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN.md")

PANELS = (
    "source_local_direct_control",
    "learned_fact_direct_control",
    "complete_identity_conflict",
    "contradictory_identity_null",
    "symbol_only_ablation",
    "initial_only_ablation",
    "answer_absent_and_side_null",
)
PRIMARY_CONFLICT_PANELS = ("complete_identity_conflict",)
NULL_STRESS_PANELS = (
    "contradictory_identity_null",
    "symbol_only_ablation",
    "initial_only_ablation",
)
TEMPLATES = ("identity_packet", "observation_gate", "compact_identity")

STATUS_LEXEME_RE = re.compile(
    r"\b(trusted|untrusted|reliable|unreliable|status)\b",
    re.IGNORECASE,
)
FORBIDDEN_ROUTE_RE = re.compile(
    r"\b(checksum|equation|arithmetic|cross-table|second table|fact-claim|standard-number claim)\b",
    re.IGNORECASE,
)


def source_initial(source: dict[str, Any]) -> str:
    return str(source["element"])[0].upper()


def wrong_initial(source: dict[str, Any]) -> str:
    return "Z" if source_initial(source) != "Z" else "A"


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
    if template == "identity_packet":
        intro = [
            "A temporary lab table may define local lab numbers.",
            "A separate identity packet may describe the queried element.",
        ]
    elif template == "observation_gate":
        intro = [
            "A lab-number table and a standard-identity observation packet are shown.",
            "Use the requested relation, not table order, to choose the answer.",
        ]
    elif template == "compact_identity":
        intro = [
            "Two numeric answer sources may be present.",
            "The local table is for local-number questions.",
        ]
    else:
        raise ValueError(f"unknown template {template}")
    lines = [*intro, "Element lab-number table:"]
    for row in visible_rows:
        lines.append(f"{row['element']} has local lab number {row['local_number']}.")
    return lines


def append_identity_rule(lines: list[str], template: str) -> None:
    if template == "identity_packet":
        lines.extend(
            [
                "Use the identity packet only if it has both notes.",
                "If both identity notes fit the queried element, return the standard atomic number.",
                "If a note contradicts the queried element, return UNKNOWN.",
                "If only one identity note is present, return UNKNOWN.",
                "For this standard-identity question, do not answer with the local lab number.",
            ]
        )
    elif template == "observation_gate":
        lines.extend(
            [
                "Decision rule for the standard-identity question:",
                "two fitting observations -> standard atomic number",
                "any contradictory observation -> UNKNOWN",
                "one observation only -> UNKNOWN",
                "local lab numbers answer only local-number questions",
            ]
        )
    elif template == "compact_identity":
        lines.extend(
            [
                "Rule: both identity notes fit -> standard atomic number.",
                "Rule: contradiction -> UNKNOWN.",
                "Rule: one identity note -> UNKNOWN.",
                "Rule: never use a local lab number for this standard-identity query.",
            ]
        )
    else:
        raise ValueError(f"unknown template {template}")


def append_identity_notes(
    lines: list[str],
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    mode: str,
) -> None:
    if mode == "complete":
        notes = [
            f"Identity note alpha: chemical symbol {source_symbol(source)}.",
            f"Identity note beta: element name initial {source_initial(source)}.",
        ]
    elif mode == "contradictory":
        notes = [
            f"Identity note alpha: chemical symbol {wrong_symbol(sources, source)}.",
            f"Identity note beta: element name initial {wrong_initial(source)}.",
        ]
    elif mode == "symbol_only":
        notes = [f"Identity note alpha: chemical symbol {source_symbol(source)}."]
    elif mode == "initial_only":
        notes = [f"Identity note alpha: element name initial {source_initial(source)}."]
    else:
        raise ValueError(f"unknown identity-note mode {mode}")
    lines.append("Identity notes:")
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
    elif panel == "complete_identity_conflict":
        lines = render_local_table(rows, source, include_query=True, template=template)
        append_identity_notes(lines, sources, source, mode="complete")
        append_identity_rule(lines, template)
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the standard-identity answer.")
        expected_label = "atomic_number"
        evidence_mode = "complete"
    elif panel == "contradictory_identity_null":
        lines = render_local_table(rows, source, include_query=True, template=template)
        append_identity_notes(lines, sources, source, mode="contradictory")
        append_identity_rule(lines, template)
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the standard-identity answer.")
        expected_label = "unknown"
        evidence_mode = "contradictory"
    elif panel == "symbol_only_ablation":
        lines = render_local_table(rows, source, include_query=True, template=template)
        append_identity_notes(lines, sources, source, mode="symbol_only")
        append_identity_rule(lines, template)
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the standard-identity answer.")
        expected_label = "unknown"
        evidence_mode = "symbol_only"
    elif panel == "initial_only_ablation":
        lines = render_local_table(rows, source, include_query=True, template=template)
        append_identity_notes(lines, sources, source, mode="initial_only")
        append_identity_rule(lines, template)
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the standard-identity answer.")
        expected_label = "unknown"
        evidence_mode = "initial_only"
    elif panel == "answer_absent_and_side_null":
        lines = render_local_table(rows, source, include_query=False, template=template)
        lines.append(f"Query element: {source['element']}")
        lines.append("This is a local-table membership check.")
        lines.append("If the query element is absent from the local table, return UNKNOWN.")
        expected_label = "unknown"
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
                        "id": f"{CARD_ID}_{REDESIGN_ID}_{template}_{panel_name}_{source['source_id']}",
                        "card_id": CARD_ID,
                        "redesign_id": REDESIGN_ID,
                        "run_type": run_type,
                        "model_id": MODEL_ID,
                        "template": template,
                        "panel": panel_name,
                        "split": source["split"],
                        "source_id": source["source_id"],
                        "source_index": source["source_index"],
                        "element": source["element"],
                        "element_symbol": source_symbol(source),
                        "element_initial": source_initial(source),
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
    conflict_rows = [
        row
        for row in records
        if row["panel"] in PRIMARY_CONFLICT_PANELS + NULL_STRESS_PANELS
    ]
    primary_rows = [row for row in records if row["panel"] in PRIMARY_CONFLICT_PANELS]
    null_rows = [row for row in records if row["panel"] == "answer_absent_and_side_null"]
    expected_rows = len(source_ids) * len(templates) * len(PANELS)
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
        row["id"] for row in records if len(set(row["candidate_answers"])) != 4
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
    conflict = record["panel"] in PRIMARY_CONFLICT_PANELS + NULL_STRESS_PANELS
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
    return {
        "reported": True,
        "final_local_minus_atomic_number_logit_mean": mean(
            row["final_local_minus_atomic_number_logit"] for row in rows
        ),
        "candidate_local_minus_atomic_number_mean_logprob_mean": mean(
            row["candidate_local_minus_atomic_number_mean_logprob"] for row in rows
        ),
        "final_unknown_minus_local_logit_mean": mean(
            row["final_unknown_minus_local_logit"] for row in rows
        ),
        "candidate_unknown_minus_local_mean_logprob_mean": mean(
            row["candidate_unknown_minus_local_mean_logprob"] for row in rows
        ),
    }


def mean(values: Any) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def template_summary(outputs: list[dict[str, Any]], templates: tuple[str, ...]) -> dict[str, Any]:
    result = {}
    for template in templates:
        rows = [row for row in outputs if row["template"] == template]
        panels = {
            panel_name: summarize_rows([row for row in rows if row["panel"] == panel_name])
            for panel_name in PANELS
        }
        primary = [row for row in rows if row["panel"] in PRIMARY_CONFLICT_PANELS]
        null_stress = [row for row in rows if row["panel"] in NULL_STRESS_PANELS]
        holdout_primary = [row for row in primary if row["split"] == "holdout"]
        non_holdout_primary = [row for row in primary if row["split"] != "holdout"]
        expected_correct = sum(1 for row in primary if row["is_expected_correct"])
        result[template] = {
            "rows": len(rows),
            "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "selected_label_counts": label_counts(rows),
            "panels": panels,
            "primary_conflict": {
                **summarize_rows(primary),
                "expected_correct": expected_correct,
                "expected_correct_rate": expected_correct / len(primary) if primary else 0.0,
            },
            "primary_conflict_non_holdout": summarize_rows(non_holdout_primary),
            "primary_conflict_holdout": summarize_rows(holdout_primary),
            "null_stress": summarize_rows(null_stress),
            "primary_conflict_margin_audits": margin_audits(primary),
            "null_stress_margin_audits": margin_audits(null_stress),
        }
    return result


def selection_key(item: dict[str, Any], template: str, templates: tuple[str, ...]) -> tuple[float, float, float, float, float, int]:
    local = item["panels"]["source_local_direct_control"]
    learned = item["panels"]["learned_fact_direct_control"]
    complete = item["panels"]["complete_identity_conflict"]
    contradiction = item["panels"]["contradictory_identity_null"]
    symbol_only = item["panels"]["symbol_only_ablation"]
    initial_only = item["panels"]["initial_only_ablation"]
    side_null = item["panels"]["answer_absent_and_side_null"]
    holdout = item["primary_conflict_holdout"]
    control_floor = min(
        float(local["local_number_rate"]),
        float(learned["atomic_number_rate"]),
        float(complete["atomic_number_rate"]),
        float(contradiction["unknown_rate"]),
        float(symbol_only["unknown_rate"]),
        float(initial_only["unknown_rate"]),
        float(side_null["unknown_rate"]),
    )
    return (
        control_floor,
        float(complete["atomic_number_rate"]),
        float(item["null_stress"]["unknown_rate"]),
        float(holdout["atomic_number_rate"]),
        float(item["primary_conflict"]["parseable_rate"]),
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
            "max minimum direct/evidence/null control rate",
            "max complete-conflict atomic-number rate",
            "max null-stress UNKNOWN rate",
            "max holdout complete-conflict atomic-number rate",
            "max complete-conflict parseability",
            "earliest template",
        ],
    }


def classify(criteria: dict[str, bool]) -> str:
    if criteria["smoke_mode"]:
        return "smoke_only"
    if not criteria["structural_passed"]:
        return "structural_invalid"
    if not criteria["selected_prompt_audit_passed"] or not criteria["primary_prompts_have_no_status_or_closed_route_lexemes"]:
        return "evidence_sufficiency_prompt_leak_failed"
    if not criteria["source_local_direct_control_passed"]:
        return "evidence_sufficiency_local_control_failed"
    if not criteria["learned_fact_direct_control_passed"]:
        return "evidence_sufficiency_learned_control_failed"
    if not criteria["answer_absent_null_passed"]:
        return "evidence_sufficiency_answer_absent_null_failed"
    if not criteria["complete_identity_conflict_atomic_passed"]:
        return "evidence_sufficiency_learned_branch_failed"
    if not criteria["contradictory_identity_unknown_passed"]:
        return "evidence_sufficiency_contradiction_not_rejected"
    if not criteria["single_feature_ablation_unknown_passed"]:
        return "evidence_sufficiency_partial_evidence_visible_channel"
    if not criteria["holdout_complete_identity_atomic_passed"]:
        return "evidence_sufficiency_holdout_failed"
    if not criteria["candidate_and_output_margins_reported"]:
        return "evidence_sufficiency_behavior_passed_baselines_missing"
    return "evidence_sufficiency_behavior_ready"


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
    complete = selected["panels"]["complete_identity_conflict"]
    contradiction = selected["panels"]["contradictory_identity_null"]
    symbol_only = selected["panels"]["symbol_only_ablation"]
    initial_only = selected["panels"]["initial_only_ablation"]
    side_null = selected["panels"]["answer_absent_and_side_null"]
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
        "answer_absent_null_passed": float(side_null["unknown_rate"]) >= 0.80
        and float(side_null["local_number_rate"]) <= 0.10,
        "complete_identity_conflict_atomic_passed": float(complete["parseable_rate"]) >= 0.90
        and float(complete["atomic_number_rate"]) >= 0.70
        and float(complete["local_number_rate"]) <= 0.10,
        "contradictory_identity_unknown_passed": float(contradiction["unknown_rate"]) >= 0.70
        and float(contradiction["local_number_rate"]) <= 0.10,
        "single_feature_ablation_unknown_passed": float(symbol_only["unknown_rate"]) >= 0.60
        and float(initial_only["unknown_rate"]) >= 0.60
        and float(symbol_only["local_number_rate"]) <= 0.10
        and float(initial_only["local_number_rate"]) <= 0.10,
        "holdout_complete_identity_atomic_passed": int(holdout["rows"]) > 0
        and float(holdout["atomic_number_rate"]) >= 0.70
        and float(holdout["local_number_rate"]) <= 0.10,
        "candidate_and_output_margins_reported": bool(
            score_candidates and selected["primary_conflict_margin_audits"]["reported"]
        ),
    }
    diagnostic_class = classify(criteria)
    behavior_gate_passed = diagnostic_class == "evidence_sufficiency_behavior_ready"
    behavior_candidate = (
        diagnostic_class == "evidence_sufficiency_behavior_passed_baselines_missing"
    )
    redesign_decision = {
        "promotion_rule_passed": behavior_gate_passed,
        "kill_rule_triggered": (
            not criteria["smoke_mode"]
            and float(complete["local_number_rate"]) > 0.30
            and (
                float(contradiction["local_number_rate"]) > 0.30
                or float(symbol_only["local_number_rate"]) > 0.30
                or float(initial_only["local_number_rate"]) > 0.30
            )
        ),
        "route_decision": (
            "admit_behavior_substrate_only"
            if behavior_gate_passed
            else "kill_same_family_statusless_evidence_aggregation"
            if not criteria["smoke_mode"]
            and (
                float(complete["local_number_rate"]) > 0.30
                or float(contradiction["local_number_rate"]) > 0.30
                or float(symbol_only["local_number_rate"]) > 0.30
                or float(initial_only["local_number_rate"]) > 0.30
            )
            else "continue_or_redesign_after_smoke"
            if criteria["smoke_mode"]
            else "bound_evidence_sufficiency_without_hidden_state"
        ),
        "redesign_verdict": diagnostic_class,
        "exported_diagnostic_class": (
            "STATUSLESS_EVIDENCE_LOCAL_TABLE_DOMINANCE"
            if (
                float(complete["local_number_rate"]) > 0.30
                or float(contradiction["local_number_rate"]) > 0.30
                or float(symbol_only["local_number_rate"]) > 0.30
                or float(initial_only["local_number_rate"]) > 0.30
            )
            else "STATUSLESS_EVIDENCE_SUFFICIENCY_BOUNDARY"
        ),
        "behavior_ready": behavior_gate_passed,
        "signature_screen_allowed": behavior_gate_passed,
        "hidden_state_claim_allowed": False,
        "intervention_allowed": False,
        "mechanism_claim_allowed": False,
    }
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
        "hidden_state_allowed": False,
        "diagnostic_class": diagnostic_class,
        "redesign_decision": redesign_decision,
    }


def quiet_summary(summary: dict[str, Any], output_path: Path | None = None) -> dict[str, Any]:
    payload = {
        "diagnostic_class": summary["diagnostic_class"],
        "passed": summary["passed"],
        "behavior_ready": summary["behavior_ready"],
        "signature_ready": summary["signature_ready"],
        "hidden_state_allowed": summary["hidden_state_allowed"],
        "redesign_decision": summary["redesign_decision"],
        "criteria": summary["criteria"],
        "selection": summary["selection"],
        "selected_primary_conflict": summary["selected_template_summary"]["primary_conflict"],
        "selected_null_stress": summary["selected_template_summary"]["null_stress"],
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
        "# KSQ003 Evidence Sufficiency Redesign",
        "",
        "Status: material redesign preregistration; no hidden-state work.",
        "",
        "Runner:",
        "",
        "> `code/ksq003_evidence_sufficiency_redesign.py`",
        "",
        "Default artifacts:",
        "",
        f"> `{DEFAULT_RESULT_PATH.as_posix()}`",
        "",
        f"> `{SMOKE_LIMIT10_RESULT_PATH.as_posix()}`",
        "",
        f"> `{FULL_BEHAVIOR_RESULT_PATH.as_posix()}`",
        "",
        "## Claim Under Test",
        "",
        "A statusless evidence-sufficiency contract can prevent local-table",
        "dominance without trusted/untrusted labels: complete identity evidence",
        "routes to the learned atomic-number branch, contradictory or incomplete",
        "evidence routes to UNKNOWN, and direct local lookup remains clean.",
        "",
        "## Promotion Rule",
        "",
        "Promote only to behavior substrate if direct local lookup, direct learned",
        "atomic recall, complete-identity conflict, contradiction nulls,",
        "single-feature ablations, answer-absent nulls, source-disjoint holdout,",
        "and candidate/output baseline reporting all pass.",
        "",
        "## Kill Rule",
        "",
        "Kill same-family statusless evidence aggregation if complete conflict,",
        "contradiction, or single-feature ablation rows again collapse to local",
        "lab-number answers in the full run.",
        "",
        "## Forbidden Claims",
        "",
        "- This preregistration does not license hidden-state probing.",
        "- This redesign is not a mechanism card.",
        "- A passing behavior substrate would still require signature and intervention gates.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_status_card(path: Path, result: dict[str, Any], output_path: Path) -> None:
    structural = result["structural"]
    summary = result.get("summary")
    lines = [
        "# KSQ003 Evidence Sufficiency Redesign Status",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "Runner:",
        "",
        "> `code/ksq003_evidence_sufficiency_redesign.py`",
        "",
        "Result:",
        "",
        f"> `{output_path.as_posix()}`",
        "",
    ]
    if summary is None:
        lines.extend(
            [
                "Status: structural_passed_behavior_not_run."
                if structural["passed"]
                else "Status: structural_failed.",
                "",
                "## Structural Criteria",
                "",
                "| Criterion | Passed |",
                "| --- | --- |",
            ]
        )
        for key, value in structural["criteria"].items():
            lines.append(f"| `{key}` | `{str(value).lower()}` |")
    else:
        selected = summary["selected_template_summary"]
        decision = summary["redesign_decision"]
        lines.extend(
            [
                f"Status: {summary['diagnostic_class']}.",
                "",
                "## Verdict",
                "",
            ]
        )
        if summary["behavior_ready"]:
            lines.extend(
                [
                    "The evidence-sufficiency redesign passed as a behavior",
                    "substrate only. It may support a later hidden-state",
                    "signature screen, but it is not a mechanism card and no",
                    "intervention has been run.",
                ]
            )
        elif summary["criteria"]["smoke_mode"]:
            lines.extend(
                [
                    "This is a smoke run. It only decides whether the material",
                    "redesign is worth a full behavior run.",
                ]
            )
        else:
            lines.extend(
                [
                    "The evidence-sufficiency redesign did not pass behavior",
                    "admission. Hidden-state work remains forbidden.",
                ]
            )
        lines.extend(
            [
                "",
                "## Route Decision",
                "",
                f"- route decision: `{decision['route_decision']}`",
                f"- exported diagnostic class: `{decision['exported_diagnostic_class']}`",
                f"- behavior ready: `{str(decision['behavior_ready']).lower()}`",
                f"- signature screen allowed: `{str(decision['signature_screen_allowed']).lower()}`",
                "",
                "## Gate Criteria",
                "",
                "| Criterion | Passed |",
                "| --- | --- |",
            ]
        )
        for key, value in summary["criteria"].items():
            lines.append(f"| `{key}` | `{str(value).lower()}` |")
        lines.extend(
            [
                "",
                "## Selected Template",
                "",
                f"- selected template: `{summary['selection']['selected_template']}`",
                f"- selection key: `{json.dumps(summary['selection']['selection_key'])}`",
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
            ("complete_identity_conflict", "atomic_number_rate"),
            ("contradictory_identity_null", "unknown_rate"),
            ("symbol_only_ablation", "unknown_rate"),
            ("initial_only_ablation", "unknown_rate"),
            ("answer_absent_and_side_null", "unknown_rate"),
        ]
        for panel, key in controls:
            item = selected["panels"][panel]
            lines.append(
                f"| `{panel}` | {item['rows']} | {item['parseable_rate']:.3f} | {item[key]:.3f} |"
            )
        primary = selected["primary_conflict"]
        null_stress = selected["null_stress"]
        lines.extend(
            [
                "",
                "## Primary Conflict",
                "",
                f"- rows: `{primary['rows']}`",
                f"- atomic-number rows: `{primary['atomic_number']}`",
                f"- local-number rows: `{primary['local_number']}`",
                f"- parseable rate: `{primary['parseable_rate']:.3f}`",
                "",
                "## Null Stress",
                "",
                f"- rows: `{null_stress['rows']}`",
                f"- unknown rows: `{null_stress['unknown']}`",
                f"- local-number rows: `{null_stress['local_number']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Forbidden Claims",
            "",
            "- KSQ003 is a mechanism card.",
            "- KSQ003 supports intervention.",
            "- KSQ003 found an internal knowledge-control surface.",
            "- Any hidden-state or causal claim follows from this redesign alone.",
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
            "redesign_id": REDESIGN_ID,
            "run_type": RUN_TYPE,
            "model_id": args.model_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "purpose": "Structural gate for the KSQ003 evidence-sufficiency redesign.",
            "templates": list(TEMPLATES),
            "panels": list(PANELS),
            "primary_conflict_panels": list(PRIMARY_CONFLICT_PANELS),
            "null_stress_panels": list(NULL_STRESS_PANELS),
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
        "redesign_id": REDESIGN_ID,
        "run_type": BEHAVIOR_RUN_TYPE,
        "model_id": args.model_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "max_new_tokens": args.max_new_tokens,
        "decoding": {"do_sample": False},
        "score_candidates": args.score_candidates,
        "limit_sources": args.limit_sources,
        "templates": list(TEMPLATES),
        "elapsed_s": time.time() - started,
        "purpose": "Behavior-only evidence-sufficiency bridge redesign admission run.",
        "panels": list(PANELS),
        "primary_conflict_panels": list(PRIMARY_CONFLICT_PANELS),
        "null_stress_panels": list(NULL_STRESS_PANELS),
        "sources": base_sources(args.limit_sources),
        "hidden_state_allowed": False,
        "structural": structural,
        "records": outputs,
        "summary": summary,
        "redesign_decision": summary["redesign_decision"],
    }
    write_json(args.output_path, result)
    if args.write_status_card:
        write_status_card(args.status_card, result, args.output_path)
    print(json.dumps(quiet_summary(summary, args.output_path), indent=2, ensure_ascii=True))
    return 0 if (summary["passed"] or not full_run) else 2


if __name__ == "__main__":
    raise SystemExit(main())
