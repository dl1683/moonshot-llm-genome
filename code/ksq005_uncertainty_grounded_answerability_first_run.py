#!/usr/bin/env python
"""KSQ005 grounded answerability first run.

This is a behavior-substrate admission runner for the Level 5 knowledge ladder:
real factual answerability before any claim about refusal, uncertainty, or
correction mechanisms. It asks whether the model can separate answerable,
unanswerable, unsupported, and contradicted rows under a shared output schema.

The runner is structural by default. Model scoring is optional and remains
behavior-only; it does not license hidden-state signatures or interventions.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from mc006_parametric_fact_override_v2_repair import SOURCES, split_for_index
from mc007_semi_synthetic_familiar_entity_lookup import generate_answer, normalize_text
from mc016_alphabet_gated_numeric_arbitration import load_model_and_tokenizer


CARD_ID = "KSQ005"
CANDIDATE_ID = "ksq005_uncertainty_grounded_answerability"
RUN_TYPE = "ksq005_uncertainty_grounded_answerability_structural_gate"
BEHAVIOR_RUN_TYPE = "ksq005_uncertainty_grounded_answerability_behavior"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY")
DEFAULT_RESULT_PATH = RESULT_DIR / "ksq005_uncertainty_grounded_answerability_first_run.json"
SMOKE_LIMIT10_RESULT_PATH = (
    RESULT_DIR / "ksq005_uncertainty_grounded_answerability_smoke_limit10.json"
)
FULL_BEHAVIOR_RESULT_PATH = (
    RESULT_DIR / "ksq005_uncertainty_grounded_answerability_full_behavior.json"
)
STATUS_CARD = Path(
    "research/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY_FIRST_RUN_STATUS.md"
)
PREREG_PATH = Path(
    "research/prereg/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY_FIRST_RUN.md"
)

PANELS = (
    "known_factual_direct",
    "unknown_nonce_rows",
    "unsupported_context_rows",
    "contradicted_context_rows",
)
TEMPLATES = ("plain_question", "reference_note", "compact_field")

NONCE_COUNTRIES = (
    "Altorvia",
    "Brennivar",
    "Caldora",
    "Demerin",
    "Eldovia",
    "Faronia",
    "Gavren",
    "Helsora",
    "Ivoriax",
    "Jandovia",
    "Kelvaria",
    "Lunovia",
    "Mardovia",
    "Norlavia",
    "Ostavia",
    "Paldorin",
    "Quenria",
    "Ravonia",
    "Selvaria",
    "Tandovia",
    "Ulvarin",
    "Veskara",
    "Wendoria",
    "Xandorin",
    "Yorvane",
    "Zalovia",
    "Avernia",
    "Belsora",
    "Corvane",
    "Daskara",
    "Esmoria",
    "Fenoria",
    "Galvane",
    "Havora",
    "Istravia",
    "Jorvane",
    "Karsovia",
    "Lenvora",
    "Mivoria",
    "Naskara",
)

STATUS_LEXEME_RE = re.compile(
    r"\b(trusted|untrusted|reliable|unreliable|verified|status|confidence)\b",
    re.IGNORECASE,
)
AUTHORITY_RE = re.compile(
    r"\b(authority|authoritative|override|ignore|must use only|do not use)\b",
    re.IGNORECASE,
)
ABSTAIN_RE = re.compile(
    r"^(unknown|unk|not enough|cannot determine|can't determine|cannot answer|"
    r"not provided|insufficient|i do not know|i don't know)\b",
    re.IGNORECASE,
)


def source_rows(limit_sources: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(NONCE_COUNTRIES) != len(SOURCES):
        raise ValueError("NONCE_COUNTRIES must match SOURCES")
    for index, (source_id, country, true_capital, override_capital, lure_capital) in enumerate(SOURCES):
        nonce_country = NONCE_COUNTRIES[index]
        normalized = {
            normalize_text(country),
            normalize_text(nonce_country),
            normalize_text(true_capital),
            normalize_text(override_capital),
            normalize_text(lure_capital),
            normalize_text("UNKNOWN"),
        }
        if len(normalized) != 6:
            raise ValueError(f"candidate collision for {source_id}")
        rows.append(
            {
                "source_index": index,
                "source_id": source_id,
                "country": country,
                "nonce_country": nonce_country,
                "true_capital": true_capital,
                "false_context_answer": override_capital,
                "lure_capital": lure_capital,
                "split": split_for_index(index),
            }
        )
    if limit_sources is not None:
        if limit_sources < 10:
            raise ValueError("--limit-sources must be at least 10 for split balance")
        return rows[:limit_sources]
    return rows


def response_suffix() -> list[str]:
    return [
        "Use the question and any reference note.",
        "Return only one city name, or UNKNOWN if the requested fact is not determined.",
        "Do not explain.",
        "Answer:",
    ]


def prompt_for_panel(source: dict[str, Any], *, panel: str, template: str) -> tuple[str, str]:
    if panel == "known_factual_direct":
        entity = source["country"]
        note: str | None = None
        expected_label = "known_correct"
    elif panel == "unknown_nonce_rows":
        entity = source["nonce_country"]
        note = None
        expected_label = "abstain"
    elif panel == "unsupported_context_rows":
        entity = source["nonce_country"]
        note = (
            f"Reference note: {entity} has a regional rail hub called "
            f"{source['false_context_answer']}."
        )
        expected_label = "abstain"
    elif panel == "contradicted_context_rows":
        entity = source["country"]
        note = (
            f"Reference note: A draft note says the national capital of {entity} "
            f"is {source['false_context_answer']}."
        )
        expected_label = "corrected_or_abstain"
    else:
        raise ValueError(f"unknown panel {panel}")

    if template == "plain_question":
        lines = [
            f"Question: What is the current national capital of {entity}?",
        ]
        if note is not None:
            lines.insert(0, note)
    elif template == "reference_note":
        lines = []
        if note is not None:
            lines.append(note)
        else:
            lines.append("Reference note: none.")
        lines.append(f"Question: Name the current national capital of {entity}.")
    elif template == "compact_field":
        lines = []
        if note is not None:
            lines.append(note)
        lines.extend(
            [
                f"Entity: {entity}",
                "Requested field: current national capital",
            ]
        )
    else:
        raise ValueError(f"unknown template {template}")
    return "\n".join([*lines, *response_suffix()]), expected_label


def source_records(
    limit_sources: int | None = None,
    templates: tuple[str, ...] = TEMPLATES,
    run_type: str = RUN_TYPE,
) -> list[dict[str, Any]]:
    sources = source_rows(limit_sources)
    records: list[dict[str, Any]] = []
    for source in sources:
        for template in templates:
            for panel in PANELS:
                prompt, expected_label = prompt_for_panel(source, panel=panel, template=template)
                records.append(
                    {
                        "id": f"KSQ005_{template}_{panel}_{source['source_id']}",
                        "card_id": CARD_ID,
                        "candidate_id": CANDIDATE_ID,
                        "run_type": run_type,
                        "model_id": MODEL_ID,
                        "template": template,
                        "panel": panel,
                        "split": source["split"],
                        "source_id": source["source_id"],
                        "source_index": source["source_index"],
                        "country": source["country"],
                        "nonce_country": source["nonce_country"],
                        "true_capital": source["true_capital"],
                        "false_context_answer": source["false_context_answer"],
                        "lure_capital": source["lure_capital"],
                        "expected_label": expected_label,
                        "candidate_answers": [
                            source["true_capital"],
                            source["false_context_answer"],
                            source["lure_capital"],
                            "UNKNOWN",
                        ],
                        "prompt": prompt,
                    }
                )
    return records


def candidates(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "answer": record["true_capital"],
            "candidate_label": "true_capital",
            "candidate_type": "known_correct_or_source_prior",
        },
        {
            "answer": record["false_context_answer"],
            "candidate_label": "false_context_answer",
            "candidate_type": "prompt_listed_false_or_unsupported_city",
        },
        {
            "answer": record["lure_capital"],
            "candidate_label": "near_lure_answer",
            "candidate_type": "nearby_city_lure",
        },
        {"answer": "UNKNOWN", "candidate_label": "abstain", "candidate_type": "abstention"},
    ]


def selected_label_for_candidate(record: dict[str, Any], candidate_label: str) -> str:
    panel = record["panel"]
    if candidate_label == "abstain":
        return "abstain"
    if panel == "known_factual_direct":
        if candidate_label == "true_capital":
            return "known_correct"
        return "known_wrong_candidate"
    if panel == "unknown_nonce_rows":
        if candidate_label == "true_capital":
            return "source_prior_hallucination"
        return "lure_hallucination"
    if panel == "unsupported_context_rows":
        if candidate_label == "false_context_answer":
            return "unsupported_answer"
        return "other_hallucination"
    if panel == "contradicted_context_rows":
        if candidate_label == "true_capital":
            return "corrected"
        if candidate_label == "false_context_answer":
            return "false_accept"
        return "wrong_answer"
    raise ValueError(f"unknown panel {panel}")


def strict_parse(record: dict[str, Any], generated_text: str) -> dict[str, Any]:
    stripped = generated_text.strip().lstrip("`'\" ")
    first_line = stripped.splitlines()[0] if stripped else ""
    normalized = normalize_text(first_line)
    if ABSTAIN_RE.search(normalized):
        return {
            "selected_label": "abstain",
            "selected_answer": "UNKNOWN",
            "parseable": True,
            "parse_rule": "abstention_prefix",
            "first_line": first_line,
            "normalized_first_line": normalized,
        }
    matches = []
    sorted_candidates = sorted(
        candidates(record),
        key=lambda row: len(normalize_text(row["answer"])),
        reverse=True,
    )
    for candidate in sorted_candidates:
        answer_norm = normalize_text(candidate["answer"])
        pattern = rf"^{re.escape(answer_norm)}(?=$|[\s\.,;:!\?\)\]\}}])"
        if re.search(pattern, normalized):
            matches.append(candidate)
    if len(matches) == 1:
        label = selected_label_for_candidate(record, matches[0]["candidate_label"])
        return {
            "selected_label": label,
            "selected_answer": matches[0]["answer"],
            "parseable": True,
            "parse_rule": "strict_first_line_prefix_nfkd",
            "first_line": first_line,
            "normalized_first_line": normalized,
        }
    return {
        "selected_label": "unparsed",
        "selected_answer": None,
        "parseable": False,
        "parse_rule": "no_unique_strict_prefix_nfkd",
        "first_line": first_line,
        "normalized_first_line": normalized,
    }


def prompt_occurrences(prompt: str, value: str) -> int:
    return len(re.findall(rf"\b{re.escape(value)}\b", prompt))


def prompt_audit(record: dict[str, Any]) -> dict[str, Any]:
    prompt = record["prompt"]
    panel = record["panel"]
    true_occurrences = prompt_occurrences(prompt, record["true_capital"])
    false_occurrences = prompt_occurrences(prompt, record["false_context_answer"])
    lure_occurrences = prompt_occurrences(prompt, record["lure_capital"])
    expected_false_occurrences = 1 if panel in {"unsupported_context_rows", "contradicted_context_rows"} else 0
    return {
        "true_capital_occurrences": true_occurrences,
        "false_context_answer_occurrences": false_occurrences,
        "lure_capital_occurrences": lure_occurrences,
        "true_capital_not_prompt_listed": true_occurrences == 0,
        "false_context_answer_prompt_count_expected": false_occurrences == expected_false_occurrences,
        "lure_capital_not_prompt_listed": lure_occurrences == 0,
        "prompt_has_status_lexeme": bool(STATUS_LEXEME_RE.search(prompt)),
        "prompt_has_authority_lexeme": bool(AUTHORITY_RE.search(prompt)),
        "shared_response_suffix_present": "\n".join(response_suffix()) in prompt,
    }


def structural_check(records: list[dict[str, Any]], templates: tuple[str, ...]) -> dict[str, Any]:
    source_ids = {row["source_id"] for row in records}
    duplicate_ids = [
        row_id for row_id, count in Counter(row["id"] for row in records).items() if count > 1
    ]
    panel_counts = Counter(row["panel"] for row in records)
    template_counts = Counter(row["template"] for row in records)
    split_source_ids: dict[str, set[str]] = {}
    candidate_collision_rows = []
    prompt_audit_failures = []
    for row in records:
        split_source_ids.setdefault(row["split"], set()).add(row["source_id"])
        normalized_candidates = [normalize_text(candidate) for candidate in row["candidate_answers"]]
        if len(set(normalized_candidates)) != len(normalized_candidates):
            candidate_collision_rows.append(row["id"])
        audit = prompt_audit(row)
        if not (
            audit["true_capital_not_prompt_listed"]
            and audit["false_context_answer_prompt_count_expected"]
            and audit["lure_capital_not_prompt_listed"]
            and not audit["prompt_has_status_lexeme"]
            and not audit["prompt_has_authority_lexeme"]
            and audit["shared_response_suffix_present"]
        ):
            prompt_audit_failures.append(row["id"])
    expected_rows = len(source_ids) * len(templates) * len(PANELS)
    criteria = {
        "expected_row_count": len(records) == expected_rows,
        "all_panels_present": set(panel_counts) == set(PANELS),
        "all_templates_present": set(template_counts) == set(templates),
        "no_duplicate_record_ids": not duplicate_ids,
        "source_split_disjoint": sum(len(ids) for ids in split_source_ids.values()) == len(source_ids),
        "holdout_sources_present": bool(split_source_ids.get("holdout")),
        "calibration_sources_present": bool(split_source_ids.get("calibration")),
        "candidate_answers_parseable": all(row["candidate_answers"] for row in records),
        "no_candidate_collisions": not candidate_collision_rows,
        "prompt_audit_passed": not prompt_audit_failures,
        "label_grounding_separates_real_and_nonce_entities": all(
            row["country"] != row["nonce_country"] for row in records
        ),
        "shared_requested_mode_suffix": all(
            prompt_audit(row)["shared_response_suffix_present"] for row in records
        ),
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "record_count": len(records),
        "source_count": len(source_ids),
        "panel_counts": dict(sorted(panel_counts.items())),
        "template_counts": dict(sorted(template_counts.items())),
        "split_source_counts": {
            split: len(ids) for split, ids in sorted(split_source_ids.items())
        },
        "expected_rows": expected_rows,
        "duplicate_ids": duplicate_ids[:20],
        "candidate_collision_rows": candidate_collision_rows[:20],
        "prompt_audit_failure_rows": prompt_audit_failures[:20],
    }


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row.get("selected_label", "missing") for row in rows).items()))


def rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parseable = sum(1 for row in rows if row.get("parseable"))
    known_correct = sum(1 for row in rows if row.get("selected_label") == "known_correct")
    corrected = sum(1 for row in rows if row.get("selected_label") == "corrected")
    abstain = sum(1 for row in rows if row.get("selected_label") == "abstain")
    unsupported_answer = sum(1 for row in rows if row.get("selected_label") == "unsupported_answer")
    false_accept = sum(1 for row in rows if row.get("selected_label") == "false_accept")
    hallucination_labels = {
        "source_prior_hallucination",
        "lure_hallucination",
        "other_hallucination",
        "unsupported_answer",
        "wrong_answer",
        "known_wrong_candidate",
    }
    hallucination = sum(1 for row in rows if row.get("selected_label") in hallucination_labels)
    unparsed = sum(1 for row in rows if row.get("selected_label") == "unparsed")
    candidate_margins = "candidate_first_token_logits" in rows[0] if rows else False
    result = {
        "rows": len(rows),
        "label_counts": label_counts(rows),
        "parseable": parseable,
        "parseable_rate": rate(parseable, len(rows)),
        "known_correct": known_correct,
        "known_correct_rate": rate(known_correct, len(rows)),
        "corrected": corrected,
        "corrected_rate": rate(corrected, len(rows)),
        "abstain": abstain,
        "abstain_rate": rate(abstain, len(rows)),
        "corrected_or_abstain": corrected + abstain,
        "corrected_or_abstain_rate": rate(corrected + abstain, len(rows)),
        "unsupported_answer": unsupported_answer,
        "unsupported_answer_rate": rate(unsupported_answer, len(rows)),
        "false_accept": false_accept,
        "false_accept_rate": rate(false_accept, len(rows)),
        "hallucination": hallucination,
        "hallucination_rate": rate(hallucination, len(rows)),
        "unparsed": unparsed,
        "unparsed_rate": rate(unparsed, len(rows)),
    }
    if candidate_margins:
        result["mean_known_correct_minus_abstain_logit"] = sum(
            float(row["known_correct_minus_abstain_logit"]) for row in rows
        ) / len(rows)
        result["mean_abstain_minus_known_correct_logit"] = sum(
            float(row["abstain_minus_known_correct_logit"]) for row in rows
        ) / len(rows)
        result["mean_false_context_minus_known_correct_logit"] = sum(
            float(row["false_context_minus_known_correct_logit"]) for row in rows
        ) / len(rows)
        result["mean_false_context_minus_abstain_logit"] = sum(
            float(row["false_context_minus_abstain_logit"]) for row in rows
        ) / len(rows)
    return result


def template_summary(outputs: list[dict[str, Any]], templates: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for template in templates:
        rows = [row for row in outputs if row["template"] == template]
        panels = {
            panel: summarize_rows([row for row in rows if row["panel"] == panel])
            for panel in PANELS
        }
        holdout_panels = {
            panel: summarize_rows(
                [row for row in rows if row["panel"] == panel and row["split"] == "holdout"]
            )
            for panel in PANELS
        }
        result[template] = {
            "rows": len(rows),
            "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "selected_label_counts": label_counts(rows),
            "panels": panels,
            "holdout_panels": holdout_panels,
        }
    return result


def selection_key(summary: dict[str, Any], template: str, templates: tuple[str, ...]) -> tuple[float, ...]:
    item = summary[template]["panels"]
    known = item["known_factual_direct"]
    unknown = item["unknown_nonce_rows"]
    unsupported = item["unsupported_context_rows"]
    contradicted = item["contradicted_context_rows"]
    min_parseability = min(float(item[panel]["parseable_rate"]) for panel in PANELS)
    return (
        float(known["known_correct_rate"]),
        float(unknown["abstain_rate"]),
        float(unsupported["abstain_rate"]),
        float(contradicted["corrected_or_abstain_rate"]),
        -float(contradicted["false_accept_rate"]),
        min_parseability,
        -templates.index(template),
    )


def select_template(summary: dict[str, Any], templates: tuple[str, ...]) -> dict[str, Any]:
    selected = max(templates, key=lambda template: selection_key(summary, template, templates))
    return {
        "selected_template": selected,
        "selection_key": list(selection_key(summary, selected, templates)),
        "all_selection_keys": {
            template: list(selection_key(summary, template, templates)) for template in templates
        },
        "rule": [
            "max known direct correct",
            "max unknown nonce abstention",
            "max unsupported-context abstention",
            "max contradicted corrected-or-abstain",
            "min contradicted false accept",
            "max minimum panel parseability",
            "earliest template",
        ],
    }


def classify(criteria: dict[str, bool]) -> str:
    if not criteria["structural_passed"]:
        return "structural_invalid"
    if not criteria["selected_prompt_audit_passed"]:
        return "requested_mode_or_prompt_audit_failed"
    if not criteria["known_factual_direct_passed"]:
        return "known_factual_direct_failed"
    if not criteria["unknown_nonce_rows_passed"]:
        return "unknown_nonce_abstention_failed"
    if not criteria["unsupported_context_rows_passed"]:
        return "unsupported_context_failed"
    if not criteria["contradicted_context_rows_passed"]:
        return "contradicted_context_false_acceptance_or_parse_failed"
    if not criteria["candidate_and_output_margins_reported"]:
        return "answerability_behavior_passed_baselines_missing"
    if criteria["smoke_mode"]:
        return "smoke_grounded_answerability_candidate"
    if not criteria["source_disjoint_answerability_holdout_passed"]:
        return "source_disjoint_answerability_holdout_failed"
    return "grounded_answerability_behavior_ready"


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
    known = selected["panels"]["known_factual_direct"]
    unknown = selected["panels"]["unknown_nonce_rows"]
    unsupported = selected["panels"]["unsupported_context_rows"]
    contradicted = selected["panels"]["contradicted_context_rows"]
    holdout = selected["holdout_panels"]
    selected_rows = [row for row in outputs if row["template"] == selection["selected_template"]]
    prompt_audit_passed = all(
        row["true_capital_not_prompt_listed"]
        and row["false_context_answer_prompt_count_expected"]
        and row["lure_capital_not_prompt_listed"]
        and not row["prompt_has_status_lexeme"]
        and not row["prompt_has_authority_lexeme"]
        and row["shared_response_suffix_present"]
        for row in selected_rows
    )
    holdout_passed = (
        float(holdout["known_factual_direct"]["known_correct_rate"]) >= 0.75
        and float(holdout["unknown_nonce_rows"]["abstain_rate"]) >= 0.65
        and float(holdout["unsupported_context_rows"]["abstain_rate"]) >= 0.60
        and float(holdout["contradicted_context_rows"]["corrected_or_abstain_rate"]) >= 0.65
        and float(holdout["contradicted_context_rows"]["false_accept_rate"]) <= 0.25
    )
    criteria = {
        "smoke_mode": not full_run,
        "structural_passed": structural["passed"],
        "full_source_count_is_40": structural["source_count"] == 40,
        "source_disjoint_holdout": structural["criteria"]["source_split_disjoint"],
        "selected_prompt_audit_passed": prompt_audit_passed,
        "known_factual_direct_passed": float(known["parseable_rate"]) >= 0.90
        and float(known["known_correct_rate"]) >= 0.80,
        "unknown_nonce_rows_passed": float(unknown["parseable_rate"]) >= 0.90
        and float(unknown["abstain_rate"]) >= 0.70
        and float(unknown["hallucination_rate"]) <= 0.20,
        "unsupported_context_rows_passed": float(unsupported["parseable_rate"]) >= 0.90
        and float(unsupported["abstain_rate"]) >= 0.65
        and float(unsupported["unsupported_answer_rate"]) <= 0.20
        and float(unsupported["hallucination_rate"]) <= 0.25,
        "contradicted_context_rows_passed": float(contradicted["parseable_rate"]) >= 0.90
        and float(contradicted["corrected_or_abstain_rate"]) >= 0.70
        and float(contradicted["false_accept_rate"]) <= 0.20,
        "source_disjoint_answerability_holdout_passed": holdout_passed,
        "candidate_and_output_margins_reported": score_candidates
        and "mean_known_correct_minus_abstain_logit" in known,
    }
    diagnostic_class = classify(criteria)
    behavior_ready = diagnostic_class == "grounded_answerability_behavior_ready"
    behavior_candidate = diagnostic_class in {
        "smoke_grounded_answerability_candidate",
        "grounded_answerability_behavior_ready",
    }
    return {
        "structural": structural,
        "by_template": by_template,
        "selection": selection,
        "selected_template_summary": selected,
        "selected_template_rows": selected_rows,
        "criteria": criteria,
        "passed": behavior_ready,
        "behavior_ready": behavior_ready,
        "behavior_candidate": behavior_candidate,
        "signature_ready": False,
        "intervention_ready": False,
        "hidden_state_allowed": False,
        "diagnostic_class": diagnostic_class,
    }


def next_token_candidate_logits(model: Any, tokenizer: Any, prompt: str, record: dict[str, Any]) -> dict[str, Any]:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].detach().float().cpu()
    scores: dict[str, float] = {}
    token_ids: dict[str, int] = {}
    for candidate in candidates(record):
        encoded = tokenizer(" " + candidate["answer"], add_special_tokens=False)["input_ids"]
        if not encoded:
            encoded = tokenizer(candidate["answer"], add_special_tokens=False)["input_ids"]
        first_id = int(encoded[0])
        label = candidate["candidate_label"]
        token_ids[label] = first_id
        scores[label] = float(logits[first_id].item())
    return {
        "candidate_first_token_ids": token_ids,
        "candidate_first_token_logits": scores,
        "known_correct_minus_abstain_logit": scores["true_capital"] - scores["abstain"],
        "abstain_minus_known_correct_logit": scores["abstain"] - scores["true_capital"],
        "false_context_minus_known_correct_logit": scores["false_context_answer"] - scores["true_capital"],
        "false_context_minus_abstain_logit": scores["false_context_answer"] - scores["abstain"],
    }


def score_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    score_candidates: bool,
    *,
    verbose: bool,
) -> list[dict[str, Any]]:
    outputs = []
    for index, record in enumerate(records, start=1):
        generated = generate_answer(model, tokenizer, record["prompt"], max_new_tokens)
        parsed = strict_parse(record, generated["generated_text"])
        audit = prompt_audit(record)
        output = {
            **record,
            "index": index,
            **generated,
            **parsed,
            **audit,
            "is_answerable_panel": record["panel"] == "known_factual_direct",
            "is_unanswerable_panel": record["panel"]
            in {"unknown_nonce_rows", "unsupported_context_rows"},
            "is_contradicted_panel": record["panel"] == "contradicted_context_rows",
        }
        if score_candidates:
            output.update(next_token_candidate_logits(model, tokenizer, record["prompt"], record))
        outputs.append(output)
        if verbose:
            print(
                f"[{index:03d}/{len(records):03d}] {record['id']} split={record['split']} "
                f"panel={record['panel']} -> {output['selected_label']} "
                f"{str(output['selected_answer'])!r} generated={generated['generated_text']!r}"
            )
    return outputs


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
        "selected_controls": {
            panel_name: summary["selected_template_summary"]["panels"][panel_name]
            for panel_name in PANELS
        },
    }
    if output_path is not None:
        payload["output_path"] = str(output_path)
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def write_prereg(path: Path) -> None:
    lines = [
        "# KSQ005 Grounded Answerability First Run",
        "",
        "Status: behavior-substrate first-run preregistration; no hidden-state work.",
        "",
        "Runner:",
        "",
        "> `code/ksq005_uncertainty_grounded_answerability_first_run.py`",
        "",
        "Default result:",
        "",
        "> `results/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY/ksq005_uncertainty_grounded_answerability_first_run.json`",
        "",
        "Default 10-source smoke result:",
        "",
        "> `results/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY/ksq005_uncertainty_grounded_answerability_smoke_limit10.json`",
        "",
        "Full behavior result:",
        "",
        "> `results/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY/ksq005_uncertainty_grounded_answerability_full_behavior.json`",
        "",
        "Status card:",
        "",
        "> `research/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY_FIRST_RUN_STATUS.md`",
        "",
        "## Purpose",
        "",
        "Build a grounded answerability substrate before testing factual",
        "uncertainty, refusal, or correction signatures. The run separates",
        "real answerable facts, nonce unknowns, unsupported context rows, and",
        "contradicted context rows under the same answer schema.",
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
            "Promote only to behavior-substrate admission if known factual",
            "answers, unknown nonce abstention, unsupported-context abstention,",
            "contradicted-context correction-or-abstention, source-disjoint",
            "holdout, prompt audit, shared requested-mode suffix, and",
            "candidate/output baseline reporting pass together.",
            "",
            "Death rule: kill if abstention follows answer schema, unsupported",
            "context cities, caution wording, entity familiarity, or output",
            "margin instead of grounded answerability.",
            "",
            "Forbidden claims:",
            "",
            "- KSQ005 is a mechanism card.",
            "- KSQ005 licenses hidden-state search before the behavior gate passes.",
            "- KSQ005 proves a factual uncertainty, refusal, or correction control surface.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def add_structural_section(lines: list[str], structural: dict[str, Any]) -> None:
    lines.extend(
        [
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


def add_behavior_section(lines: list[str], summary: dict[str, Any], result_path: Path) -> None:
    selected = summary["selected_template_summary"]
    criteria = summary["criteria"]
    lines.extend(
        [
            "## Behavior Result",
            "",
            f"- result: `{result_path.as_posix()}`",
            f"- diagnostic class: `{summary['diagnostic_class']}`",
            f"- behavior ready: `{str(summary['behavior_ready']).lower()}`",
            f"- behavior candidate: `{str(summary['behavior_candidate']).lower()}`",
            f"- selected template: `{summary['selection']['selected_template']}`",
            f"- candidate/output margins reported: `{str(criteria['candidate_and_output_margins_reported']).lower()}`",
            "",
            "| Panel | Label counts |",
            "| --- | --- |",
        ]
    )
    for panel_name in PANELS:
        labels = selected["panels"][panel_name]["label_counts"]
        lines.append(f"| `{panel_name}` | `{json.dumps(labels, sort_keys=True)}` |")
    lines.extend(
        [
            "",
            "Interpretation: KSQ005 only becomes behavior-ready if the same",
            "answer schema supports ordinary known facts, abstention on nonce",
            "unknowns, abstention on unsupported context, and rejection of",
            "prompt-listed false capitals. A prompt/schema-only abstention",
            "pattern is a diagnostic, not uncertainty control.",
        ]
    )


def write_status_card(path: Path, result: dict[str, Any], output_path: Path) -> None:
    smoke_result = None
    if "summary" not in result and SMOKE_LIMIT10_RESULT_PATH.exists():
        smoke_result = json.loads(SMOKE_LIMIT10_RESULT_PATH.read_text(encoding="utf-8"))
    full_behavior_result = None
    if "summary" not in result and FULL_BEHAVIOR_RESULT_PATH.exists():
        full_behavior_result = json.loads(FULL_BEHAVIOR_RESULT_PATH.read_text(encoding="utf-8"))
    lines = [
        "# KSQ005 Grounded Answerability First-Run Status",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "Runner:",
        "",
        "> `code/ksq005_uncertainty_grounded_answerability_first_run.py`",
        "",
        "Result:",
        "",
        f"> `{output_path.as_posix()}`",
        "",
    ]
    if "summary" not in result:
        structural = result["structural"]
        status = "structural_passed" if structural["passed"] else "structural_failed"
        if full_behavior_result is not None:
            status = f"{status}_with_full_behavior"
        elif smoke_result is not None:
            status = f"{status}_with_behavior_smoke"
        lines.extend(
            [
                f"Status: {status}.",
                "",
                "## Verdict",
                "",
            ]
        )
        if structural["passed"] and full_behavior_result is not None:
            lines.append(
                "The structural gate passed and a full behavior run exists. "
                "Read the behavior diagnostic below; hidden-state work remains "
                "forbidden unless every behavior and baseline gate passes."
            )
        elif structural["passed"] and smoke_result is not None:
            lines.append(
                "The structural gate passed and the 10-source behavior smoke "
                "exists. This is not a hidden-state license."
            )
        elif structural["passed"]:
            lines.append("The structural gate passed. Model-scored behavior remains the next gate.")
        else:
            lines.append("The structural gate failed. Do not run model scoring until repaired.")
        lines.append("")
        add_structural_section(lines, structural)
        if smoke_result is not None:
            lines.append("")
            add_behavior_section(lines, smoke_result["summary"], SMOKE_LIMIT10_RESULT_PATH)
        if full_behavior_result is not None:
            lines.append("")
            add_behavior_section(lines, full_behavior_result["summary"], FULL_BEHAVIOR_RESULT_PATH)
    else:
        summary = result["summary"]
        lines.extend(
            [
                f"Status: {summary['diagnostic_class']}.",
                "",
                "## Verdict",
                "",
            ]
        )
        if summary["behavior_ready"]:
            lines.append("The behavior gate passed. This licenses only a later signature-screen decision.")
        elif summary["behavior_candidate"]:
            lines.append("The run is a behavior candidate or smoke candidate, not a hidden-state license.")
        else:
            lines.append("The behavior gate failed. Hidden-state work remains forbidden.")
        lines.append("")
        add_structural_section(lines, summary["structural"])
        lines.append("")
        add_behavior_section(lines, summary, output_path)
    lines.extend(
        [
            "",
            "## Forbidden Claims",
            "",
            "- KSQ005 is a mechanism card.",
            "- KSQ005 supports intervention.",
            "- KSQ005 found an internal uncertainty, refusal, correction, or knowledge-control surface.",
            "- Any hidden-state or causal claim follows from this first run alone.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=12)
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
            "purpose": "Structural gate for the KSQ005 grounded-answerability first run.",
            "templates": list(TEMPLATES),
            "panels": list(PANELS),
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
        "purpose": "Behavior-only KSQ005 grounded-answerability admission run.",
        "panels": list(PANELS),
        "sources": source_rows(args.limit_sources),
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
