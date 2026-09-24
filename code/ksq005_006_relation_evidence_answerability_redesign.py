#!/usr/bin/env python
"""KSQ005/KSQ006 relation-evidence answerability redesign.

This is the material second-wave redesign for the real uncertainty work order.
The first KSQ005/KSQ006 runs showed that correction-looking rows can pass while
unknown, unsupported, claim-only, and city-mention controls fail. This runner
tests a stricter behavior substrate: only exact relation rows of the form
``capital_of(entity)=city`` determine a grounded capital answer. Claim-only and
mention-only controls are still visible, but they are outside the determining
relation grammar.

This is behavior-only. Passing this runner would license at most a later
signature screen under a prompt-contract-visible substrate, not an internal
uncertainty mechanism.
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
from ksq005_uncertainty_grounded_answerability_first_run import NONCE_COUNTRIES


CARD_ID = "KSQ005_006"
REDESIGN_ID = "ksq005_006_relation_evidence_answerability_redesign"
RUN_TYPE = "ksq005_006_relation_evidence_answerability_structural_gate"
BEHAVIOR_RUN_TYPE = "ksq005_006_relation_evidence_answerability_behavior"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY")
DEFAULT_RESULT_PATH = RESULT_DIR / "ksq005_006_relation_evidence_answerability_first_run.json"
SMOKE_LIMIT10_RESULT_PATH = (
    RESULT_DIR / "ksq005_006_relation_evidence_answerability_smoke_limit10.json"
)
FULL_BEHAVIOR_RESULT_PATH = (
    RESULT_DIR / "ksq005_006_relation_evidence_answerability_full_behavior.json"
)
STATUS_CARD = Path(
    "research/cards/KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY_STATUS.md"
)
PREREG_PATH = Path(
    "research/prereg/KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY.md"
)

PANELS = (
    "known_factual_direct",
    "supported_relation_rows",
    "unknown_nonce_absent_rows",
    "unsupported_relation_rows",
    "contradictory_relation_rows",
    "claim_only_control",
    "mention_only_control",
)
CONTROL_PANELS = ("claim_only_control", "mention_only_control")
TEMPLATES = ("relation_rows", "ledger_form", "compact_relation")

STATUS_LEXEME_RE = re.compile(
    r"\b(trusted|untrusted|reliable|unreliable|verified|status|confidence)\b",
    re.IGNORECASE,
)
SUPPORT_WORD_RE = re.compile(r"\b(support|supports|supported|unsupported)\b", re.IGNORECASE)
ABSTAIN_RE = re.compile(
    r"^(unknown|unk|not enough|cannot determine|can't determine|cannot answer|"
    r"not provided|insufficient|i do not know|i don't know)\b",
    re.IGNORECASE,
)


def source_rows(limit_sources: int | None = None) -> list[dict[str, Any]]:
    if len(NONCE_COUNTRIES) != len(SOURCES):
        raise ValueError("NONCE_COUNTRIES must match SOURCES")
    rows: list[dict[str, Any]] = []
    for index, (source_id, country, true_capital, false_city, lure_city) in enumerate(SOURCES):
        nonce_country = NONCE_COUNTRIES[index]
        normalized = {
            normalize_text(country),
            normalize_text(nonce_country),
            normalize_text(true_capital),
            normalize_text(false_city),
            normalize_text(lure_city),
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
                "false_context_answer": false_city,
                "lure_capital": lure_city,
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
        "Return only one city name, or UNKNOWN.",
        "Do not explain.",
        "Answer:",
    ]


def relation_block(source: dict[str, Any], panel: str) -> tuple[list[str], str, str]:
    country = source["country"]
    nonce = source["nonce_country"]
    true_capital = source["true_capital"]
    false_city = source["false_context_answer"]
    if panel == "known_factual_direct":
        return [], country, "known_correct"
    if panel == "supported_relation_rows":
        return [f"REL capital_of({country})={true_capital}"], country, "supported_answer"
    if panel == "unknown_nonce_absent_rows":
        return [f"REL region_tag({nonce})=northern"], nonce, "abstain"
    if panel == "unsupported_relation_rows":
        return [f"REL rail_hub({nonce})={false_city}"], nonce, "abstain"
    if panel == "contradictory_relation_rows":
        if int(source["source_index"]) % 2 == 0:
            return [
                f"REL capital_of({country})={true_capital}",
                f"REL capital_of({country})={false_city}",
            ], country, "abstain"
        return [
            f"REL capital_of({country})={false_city}",
            f"REL capital_of({country})={true_capital}",
        ], country, "abstain"
    if panel == "claim_only_control":
        return [f"CLAIM capital_of({country})={true_capital}"], country, "control_abstain"
    if panel == "mention_only_control":
        return [f"REL mentioned_city({country})={true_capital}"], country, "control_abstain"
    raise ValueError(f"unknown panel {panel}")


def render_prompt(
    source: dict[str, Any],
    *,
    panel: str,
    template: str,
) -> tuple[str, str]:
    rows, query_entity, expected_label = relation_block(source, panel)
    if panel == "known_factual_direct":
        lines = [
            "Reference note: none.",
            f"Question: Name the current national capital of {query_entity}.",
        ]
    elif template == "relation_rows":
        lines = [
            "Relation rows:",
            *[f"- {row}" for row in rows],
            "Decision rule: use exactly one matching row that begins REL capital_of(entity)=city.",
            "If no matching REL capital_of row exists, or if matching rows disagree, answer UNKNOWN.",
            "CLAIM rows and other REL predicates do not determine the requested capital.",
            f"Query: capital_of({query_entity})",
        ]
    elif template == "ledger_form":
        lines = [
            "Ledger:",
            *rows,
            "Only a single non-conflicting REL capital_of relation determines the requested capital.",
            "CLAIM rows and non-capital REL rows do not determine the requested capital.",
            f"Requested relation: capital_of({query_entity})",
        ]
    elif template == "compact_relation":
        lines = [
            "Rows:",
            *rows,
            "Use one exact REL capital_of row; absent or conflicting rows mean UNKNOWN.",
            "Ignore CLAIM rows and REL rows with another predicate.",
            f"Ask: capital_of({query_entity})",
        ]
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
                prompt, expected_label = render_prompt(source, panel=panel, template=template)
                records.append(
                    {
                        "id": f"{CARD_ID}_{REDESIGN_ID}_{template}_{panel}_{source['source_id']}",
                        "card_id": CARD_ID,
                        "redesign_id": REDESIGN_ID,
                        "run_type": run_type,
                        "model_id": MODEL_ID,
                        "template": template,
                        "panel": panel,
                        "split": source["split"],
                        "source_id": source["source_id"],
                        "source_index": source["source_index"],
                        "country": source["country"],
                        "nonce_country": source["nonce_country"],
                        "query_entity": source["country"]
                        if panel
                        in {
                            "known_factual_direct",
                            "supported_relation_rows",
                            "contradictory_relation_rows",
                            "claim_only_control",
                            "mention_only_control",
                        }
                        else source["nonce_country"],
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
            "candidate_type": "known_or_relation_capital",
        },
        {
            "answer": record["false_context_answer"],
            "candidate_label": "false_context_answer",
            "candidate_type": "conflicting_or_noncapital_city",
        },
        {
            "answer": record["lure_capital"],
            "candidate_label": "lure_capital",
            "candidate_type": "near_lure_city",
        },
        {"answer": "UNKNOWN", "candidate_label": "abstain", "candidate_type": "abstention"},
    ]


def selected_label_for_candidate(record: dict[str, Any], candidate_label: str) -> str:
    panel = record["panel"]
    if panel == "known_factual_direct":
        if candidate_label == "true_capital":
            return "known_correct"
        if candidate_label == "abstain":
            return "abstain"
        return "known_wrong_candidate"
    if panel == "supported_relation_rows":
        if candidate_label == "true_capital":
            return "supported_answer"
        if candidate_label == "abstain":
            return "abstain"
        return "wrong_answer"
    if panel in {"unknown_nonce_absent_rows", "unsupported_relation_rows", "contradictory_relation_rows"}:
        if candidate_label == "abstain":
            return "abstain"
        if panel == "unsupported_relation_rows" and candidate_label == "false_context_answer":
            return "unsupported_answer"
        if panel == "contradictory_relation_rows" and candidate_label == "false_context_answer":
            return "false_accept"
        if candidate_label == "true_capital":
            return "prior_or_true_answer"
        return "other_answer"
    if panel in CONTROL_PANELS:
        if candidate_label == "abstain":
            return "control_abstain"
        if candidate_label == "true_capital" and panel == "claim_only_control":
            return "claim_only_reproduced_supported"
        if candidate_label == "true_capital" and panel == "mention_only_control":
            return "mention_only_reproduced_supported"
        return "control_other_answer"
    raise ValueError(f"unknown panel {panel}")


def strict_parse(record: dict[str, Any], generated_text: str) -> dict[str, Any]:
    stripped = generated_text.strip().lstrip("`'\" ")
    first_line = stripped.splitlines()[0] if stripped else ""
    normalized = normalize_text(first_line)
    if ABSTAIN_RE.search(normalized):
        return {
            "selected_label": selected_label_for_candidate(record, "abstain"),
            "selected_answer": "UNKNOWN",
            "parseable": True,
            "parse_rule": "abstention_prefix",
            "first_line": first_line,
            "normalized_first_line": normalized,
        }
    matches = []
    for candidate in sorted(
        candidates(record),
        key=lambda row: len(normalize_text(row["answer"])),
        reverse=True,
    ):
        answer_norm = normalize_text(candidate["answer"])
        pattern = rf"^{re.escape(answer_norm)}(?=$|[\s\.,;:!\?\)\]\}}])"
        if re.search(pattern, normalized):
            matches.append(candidate)
    if len(matches) == 1:
        return {
            "selected_label": selected_label_for_candidate(record, matches[0]["candidate_label"]),
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


def exact_occurrences(prompt: str, value: str) -> int:
    pattern = rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])"
    return len(re.findall(pattern, prompt))


def expected_prompt_counts(record: dict[str, Any]) -> dict[str, int]:
    panel = record["panel"]
    true_count = 0
    false_count = 0
    lure_count = 0
    if panel in {"supported_relation_rows", "claim_only_control", "mention_only_control"}:
        true_count = 1
    elif panel == "unsupported_relation_rows":
        false_count = 1
    elif panel == "contradictory_relation_rows":
        true_count = 1
        false_count = 1
    return {
        "true_capital": true_count,
        "false_context_answer": false_count,
        "lure_capital": lure_count,
    }


def prompt_audit(record: dict[str, Any]) -> dict[str, Any]:
    prompt = record["prompt"]
    expected = expected_prompt_counts(record)
    true_occurrences = exact_occurrences(prompt, record["true_capital"])
    false_occurrences = exact_occurrences(prompt, record["false_context_answer"])
    lure_occurrences = exact_occurrences(prompt, record["lure_capital"])
    suffix = "\n".join(response_suffix())
    return {
        "true_capital_occurrences": true_occurrences,
        "false_context_answer_occurrences": false_occurrences,
        "lure_capital_occurrences": lure_occurrences,
        "true_capital_prompt_count_expected": true_occurrences == expected["true_capital"],
        "false_context_answer_prompt_count_expected": false_occurrences == expected["false_context_answer"],
        "lure_capital_prompt_count_expected": lure_occurrences == expected["lure_capital"],
        "prompt_has_status_lexeme": bool(STATUS_LEXEME_RE.search(prompt)),
        "prompt_has_support_word": bool(SUPPORT_WORD_RE.search(prompt)),
        "shared_response_suffix_present": suffix in prompt,
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
    support_word_rows = []
    for row in records:
        split_source_ids.setdefault(row["split"], set()).add(row["source_id"])
        normalized_candidates = [normalize_text(candidate) for candidate in row["candidate_answers"]]
        if len(set(normalized_candidates)) != len(normalized_candidates):
            candidate_collision_rows.append(row["id"])
        audit = prompt_audit(row)
        if audit["prompt_has_support_word"]:
            support_word_rows.append(row["id"])
        if not (
            audit["true_capital_prompt_count_expected"]
            and audit["false_context_answer_prompt_count_expected"]
            and audit["lure_capital_prompt_count_expected"]
            and not audit["prompt_has_status_lexeme"]
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
        "no_support_word_prompt_channel": not support_word_rows,
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
        "support_word_rows": support_word_rows[:20],
    }


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row.get("selected_label", "missing") for row in rows).items()))


def rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parseable = sum(1 for row in rows if row.get("parseable"))
    known_correct = sum(1 for row in rows if row.get("selected_label") == "known_correct")
    supported_answer = sum(1 for row in rows if row.get("selected_label") == "supported_answer")
    abstain = sum(1 for row in rows if row.get("selected_label") == "abstain")
    control_abstain = sum(1 for row in rows if row.get("selected_label") == "control_abstain")
    unsupported_answer = sum(1 for row in rows if row.get("selected_label") == "unsupported_answer")
    false_accept = sum(1 for row in rows if row.get("selected_label") == "false_accept")
    claim_only_reproduced = sum(
        1 for row in rows if row.get("selected_label") == "claim_only_reproduced_supported"
    )
    mention_only_reproduced = sum(
        1 for row in rows if row.get("selected_label") == "mention_only_reproduced_supported"
    )
    prior_or_true = sum(1 for row in rows if row.get("selected_label") == "prior_or_true_answer")
    other_answer = sum(
        1
        for row in rows
        if row.get("selected_label")
        in {"known_wrong_candidate", "wrong_answer", "other_answer", "control_other_answer"}
    )
    unparsed = sum(1 for row in rows if row.get("selected_label") == "unparsed")
    candidate_margins = "candidate_first_token_logits" in rows[0] if rows else False
    result = {
        "rows": len(rows),
        "label_counts": label_counts(rows),
        "parseable": parseable,
        "parseable_rate": rate(parseable, len(rows)),
        "known_correct": known_correct,
        "known_correct_rate": rate(known_correct, len(rows)),
        "supported_answer": supported_answer,
        "supported_answer_rate": rate(supported_answer, len(rows)),
        "abstain": abstain,
        "abstain_rate": rate(abstain, len(rows)),
        "control_abstain": control_abstain,
        "control_abstain_rate": rate(control_abstain, len(rows)),
        "unknown_or_control_abstain": abstain + control_abstain,
        "unknown_or_control_abstain_rate": rate(abstain + control_abstain, len(rows)),
        "unsupported_answer": unsupported_answer,
        "unsupported_answer_rate": rate(unsupported_answer, len(rows)),
        "false_accept": false_accept,
        "false_accept_rate": rate(false_accept, len(rows)),
        "claim_only_reproduced_supported": claim_only_reproduced,
        "claim_only_reproduced_supported_rate": rate(claim_only_reproduced, len(rows)),
        "mention_only_reproduced_supported": mention_only_reproduced,
        "mention_only_reproduced_supported_rate": rate(mention_only_reproduced, len(rows)),
        "control_reproduced_supported": claim_only_reproduced + mention_only_reproduced,
        "control_reproduced_supported_rate": rate(claim_only_reproduced + mention_only_reproduced, len(rows)),
        "prior_or_true_answer": prior_or_true,
        "prior_or_true_answer_rate": rate(prior_or_true, len(rows)),
        "other_answer": other_answer,
        "other_answer_rate": rate(other_answer, len(rows)),
        "unparsed": unparsed,
        "unparsed_rate": rate(unparsed, len(rows)),
    }
    if candidate_margins:
        result["mean_true_minus_abstain_logit"] = sum(
            float(row["true_minus_abstain_logit"]) for row in rows
        ) / len(rows)
        result["mean_abstain_minus_true_logit"] = sum(
            float(row["abstain_minus_true_logit"]) for row in rows
        ) / len(rows)
        result["mean_false_minus_abstain_logit"] = sum(
            float(row["false_minus_abstain_logit"]) for row in rows
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
        control_rows = [row for row in rows if row["panel"] in CONTROL_PANELS]
        result[template] = {
            "rows": len(rows),
            "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "selected_label_counts": label_counts(rows),
            "panels": panels,
            "controls": summarize_rows(control_rows),
            "holdout_panels": holdout_panels,
        }
    return result


def selection_key(summary: dict[str, Any], template: str, templates: tuple[str, ...]) -> tuple[float, ...]:
    item = summary[template]["panels"]
    controls = summary[template]["controls"]
    known = item["known_factual_direct"]
    supported = item["supported_relation_rows"]
    unknown = item["unknown_nonce_absent_rows"]
    unsupported = item["unsupported_relation_rows"]
    contradiction = item["contradictory_relation_rows"]
    min_parseability = min(float(item[panel]["parseable_rate"]) for panel in PANELS)
    return (
        float(known["known_correct_rate"]),
        float(supported["supported_answer_rate"]),
        float(unknown["abstain_rate"]),
        float(unsupported["abstain_rate"]),
        float(contradiction["abstain_rate"]),
        float(controls["control_abstain_rate"]),
        -float(controls["control_reproduced_supported_rate"]),
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
            "max known factual direct correct",
            "max relation-supported answer",
            "max unknown nonce absent abstention",
            "max unsupported relation abstention",
            "max contradictory relation abstention",
            "max claim/mention control abstention",
            "min claim/mention supported reproduction",
            "max minimum panel parseability",
            "earliest template",
        ],
    }


def classify(criteria: dict[str, bool]) -> str:
    if not criteria["structural_passed"]:
        return "structural_invalid"
    if not criteria["selected_prompt_audit_passed"]:
        return "relation_evidence_prompt_audit_failed"
    if not criteria["known_factual_direct_passed"]:
        return "relation_evidence_known_direct_failed"
    if not criteria["supported_relation_rows_passed"]:
        return "relation_evidence_supported_failed"
    if not criteria["unknown_nonce_absent_rows_passed"]:
        return "relation_evidence_unknown_nonce_failed"
    if not criteria["unsupported_relation_rows_passed"]:
        return "relation_evidence_unsupported_failed"
    if not criteria["contradictory_relation_rows_passed"]:
        return "relation_evidence_contradiction_failed"
    if not criteria["claim_only_and_mention_only_controls_passed"]:
        return "relation_evidence_claim_or_mention_control_failed"
    if not criteria["candidate_and_output_margins_reported"]:
        return "relation_evidence_behavior_passed_baselines_missing"
    if criteria["smoke_mode"]:
        return "smoke_relation_evidence_candidate"
    if not criteria["source_disjoint_answerability_holdout_passed"]:
        return "relation_evidence_holdout_failed"
    return "relation_evidence_answerability_behavior_ready"


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
    panels = selected["panels"]
    controls = selected["controls"]
    known = panels["known_factual_direct"]
    supported = panels["supported_relation_rows"]
    unknown = panels["unknown_nonce_absent_rows"]
    unsupported = panels["unsupported_relation_rows"]
    contradiction = panels["contradictory_relation_rows"]
    holdout = selected["holdout_panels"]
    selected_rows = [row for row in outputs if row["template"] == selection["selected_template"]]
    prompt_audit_passed = all(
        row["true_capital_prompt_count_expected"]
        and row["false_context_answer_prompt_count_expected"]
        and row["lure_capital_prompt_count_expected"]
        and not row["prompt_has_status_lexeme"]
        and not row["prompt_has_support_word"]
        and row["shared_response_suffix_present"]
        for row in selected_rows
    )
    holdout_passed = (
        float(holdout["known_factual_direct"]["known_correct_rate"]) >= 0.75
        and float(holdout["supported_relation_rows"]["supported_answer_rate"]) >= 0.75
        and float(holdout["unknown_nonce_absent_rows"]["abstain_rate"]) >= 0.75
        and float(holdout["unsupported_relation_rows"]["abstain_rate"]) >= 0.75
        and float(holdout["contradictory_relation_rows"]["abstain_rate"]) >= 0.75
        and float(holdout["claim_only_control"]["control_abstain_rate"]) >= 0.75
        and float(holdout["mention_only_control"]["control_abstain_rate"]) >= 0.75
    )
    criteria = {
        "smoke_mode": not full_run,
        "structural_passed": structural["passed"],
        "full_source_count_is_40": structural["source_count"] == 40,
        "source_disjoint_holdout": structural["criteria"]["source_split_disjoint"],
        "selected_prompt_audit_passed": prompt_audit_passed,
        "known_factual_direct_passed": float(known["parseable_rate"]) >= 0.90
        and float(known["known_correct_rate"]) >= 0.75,
        "supported_relation_rows_passed": float(supported["parseable_rate"]) >= 0.90
        and float(supported["supported_answer_rate"]) >= 0.80,
        "unknown_nonce_absent_rows_passed": float(unknown["parseable_rate"]) >= 0.90
        and float(unknown["abstain_rate"]) >= 0.80
        and float(unknown["prior_or_true_answer_rate"]) <= 0.10,
        "unsupported_relation_rows_passed": float(unsupported["parseable_rate"]) >= 0.90
        and float(unsupported["abstain_rate"]) >= 0.80
        and float(unsupported["unsupported_answer_rate"]) <= 0.10,
        "contradictory_relation_rows_passed": float(contradiction["parseable_rate"]) >= 0.90
        and float(contradiction["abstain_rate"]) >= 0.80
        and float(contradiction["false_accept_rate"]) <= 0.10
        and float(contradiction["prior_or_true_answer_rate"]) <= 0.10,
        "claim_only_and_mention_only_controls_passed": float(controls["parseable_rate"]) >= 0.90
        and float(controls["control_abstain_rate"]) >= 0.80
        and float(controls["control_reproduced_supported_rate"]) <= 0.10,
        "source_disjoint_answerability_holdout_passed": holdout_passed,
        "candidate_and_output_margins_reported": score_candidates
        and "mean_true_minus_abstain_logit" in supported,
    }
    diagnostic_class = classify(criteria)
    behavior_ready = diagnostic_class == "relation_evidence_answerability_behavior_ready"
    behavior_candidate = diagnostic_class in {
        "smoke_relation_evidence_candidate",
        "relation_evidence_answerability_behavior_ready",
    }
    redesign_decision = {
        "promotion_rule_passed": behavior_ready,
        "kill_rule_triggered": (
            not criteria["smoke_mode"]
            and criteria["known_factual_direct_passed"]
            and criteria["supported_relation_rows_passed"]
            and (
                not criteria["unknown_nonce_absent_rows_passed"]
                or not criteria["unsupported_relation_rows_passed"]
                or not criteria["claim_only_and_mention_only_controls_passed"]
            )
        ),
        "route_decision": (
            "admit_behavior_substrate_only"
            if behavior_ready
            else "kill_current_uncertainty_route"
            if not criteria["smoke_mode"]
            and criteria["known_factual_direct_passed"]
            and criteria["supported_relation_rows_passed"]
            and (
                not criteria["unknown_nonce_absent_rows_passed"]
                or not criteria["unsupported_relation_rows_passed"]
                or not criteria["claim_only_and_mention_only_controls_passed"]
            )
            else "continue_or_redesign_after_smoke"
            if criteria["smoke_mode"]
            else "bound_relation_evidence_answerability_without_hidden_state"
        ),
        "redesign_verdict": diagnostic_class,
        "exported_diagnostic_class": (
            "RELATION_EVIDENCE_BEHAVIOR_SUBSTRATE"
            if behavior_ready
            else "RELATION_EVIDENCE_ANSWERABILITY_BOUNDARY"
        ),
        "behavior_ready": behavior_ready,
        "signature_screen_allowed": behavior_ready,
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
        "passed": behavior_ready,
        "behavior_ready": behavior_ready,
        "behavior_candidate": behavior_candidate,
        "signature_ready": behavior_ready,
        "intervention_ready": False,
        "hidden_state_allowed": False,
        "diagnostic_class": diagnostic_class,
        "redesign_decision": redesign_decision,
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
        "true_minus_abstain_logit": scores["true_capital"] - scores["abstain"],
        "abstain_minus_true_logit": scores["abstain"] - scores["true_capital"],
        "false_minus_abstain_logit": scores["false_context_answer"] - scores["abstain"],
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
            "is_answerable_panel": record["panel"]
            in {"known_factual_direct", "supported_relation_rows"},
            "is_abstention_panel": record["panel"]
            in {
                "unknown_nonce_absent_rows",
                "unsupported_relation_rows",
                "contradictory_relation_rows",
                "claim_only_control",
                "mention_only_control",
            },
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
        "redesign_decision": summary["redesign_decision"],
        "criteria": summary["criteria"],
        "selection": summary["selection"],
        "selected_controls": {
            panel_name: summary["selected_template_summary"]["panels"][panel_name]
            for panel_name in PANELS
        },
        "selected_control_summary": summary["selected_template_summary"]["controls"],
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
        "# KSQ005/KSQ006 Relation-Evidence Answerability Redesign",
        "",
        "Status: material second-wave redesign; no hidden-state work.",
        "",
        "Runner:",
        "",
        "> `code/ksq005_006_relation_evidence_answerability_redesign.py`",
        "",
        "Artifacts:",
        "",
        f"> `{DEFAULT_RESULT_PATH.as_posix()}`",
        "",
        f"> `{SMOKE_LIMIT10_RESULT_PATH.as_posix()}`",
        "",
        f"> `{FULL_BEHAVIOR_RESULT_PATH.as_posix()}`",
        "",
        "## Claim Under Test",
        "",
        "Real answerability may require a visible relation-evidence grammar before",
        "unknown, unsupported, contradiction, claim-only, and mention-only controls",
        "pass together. The tested contract is exact `capital_of(entity)=city`",
        "relations. This is a behavior contract, not an internal uncertainty claim.",
        "",
        "## Promotion Rule",
        "",
        "Admit only if known factual direct answers, relation-supported answers,",
        "unknown nonce abstention, unsupported relation abstention, contradictory",
        "relation abstention, claim-only controls, mention-only controls, source-",
        "disjoint holdout, prompt audit, parser, and candidate/output baselines all",
        "pass together.",
        "",
        "## Kill Rule",
        "",
        "Kill the current uncertainty route if the redesign fixes supported or",
        "correction-looking rows while still failing unknown, unsupported,",
        "claim-only, or mention-only controls.",
        "",
        "## Forbidden Claims",
        "",
        "- This redesign is not a mechanism card.",
        "- This redesign does not prove uncertainty, refusal, or correction control.",
        "- Hidden-state probing remains forbidden until behavior admission passes.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_status_card(path: Path, result: dict[str, Any], output_path: Path) -> None:
    structural = result["structural"]
    summary = result.get("summary")
    lines = [
        "# KSQ005/KSQ006 Relation-Evidence Answerability Status",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "Runner:",
        "",
        "> `code/ksq005_006_relation_evidence_answerability_redesign.py`",
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
                "## Selected Panels",
                "",
                "| Panel | Rows | Label Counts | Key Rate |",
                "| --- | ---: | --- | ---: |",
            ]
        )
        key_rates = {
            "known_factual_direct": "known_correct_rate",
            "supported_relation_rows": "supported_answer_rate",
            "unknown_nonce_absent_rows": "abstain_rate",
            "unsupported_relation_rows": "abstain_rate",
            "contradictory_relation_rows": "abstain_rate",
            "claim_only_control": "control_abstain_rate",
            "mention_only_control": "control_abstain_rate",
        }
        for panel_name in PANELS:
            item = selected["panels"][panel_name]
            key = key_rates[panel_name]
            lines.append(
                f"| `{panel_name}` | {item['rows']} | "
                f"`{json.dumps(item['label_counts'], sort_keys=True)}` | "
                f"{item[key]:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Forbidden Claims",
            "",
            "- KSQ005/KSQ006 is a mechanism card.",
            "- KSQ005/KSQ006 supports intervention.",
            "- KSQ005/KSQ006 found an internal uncertainty, refusal, correction, context-support, or knowledge-control surface.",
            "- Any hidden-state or causal claim follows from this redesign alone.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


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
            "purpose": "Structural gate for the KSQ005/KSQ006 relation-evidence answerability redesign.",
            "templates": list(TEMPLATES),
            "panels": list(PANELS),
            "control_panels": list(CONTROL_PANELS),
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
        "purpose": "Behavior-only relation-evidence answerability redesign admission run.",
        "panels": list(PANELS),
        "control_panels": list(CONTROL_PANELS),
        "sources": source_rows(args.limit_sources),
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
