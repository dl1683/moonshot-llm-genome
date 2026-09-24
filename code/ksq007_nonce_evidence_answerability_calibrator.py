#!/usr/bin/env python
"""KSQ007 nonce-evidence answerability calibrator.

KSQ005/KSQ006 showed that real capital priors and mere city mentions can
overwhelm answerability controls. This runner removes those world-prior and
city-mention pressures while preserving the same core question:

    Can the model answer when one exact evidence row exists, and abstain when
    evidence is absent, unrelated, contradictory, claim-only, or mention-only?

This is a behavior-only calibrator. A pass would show that the previous
relation-evidence failure is specific to familiar fact pressure or city-value
salience. It would not prove uncertainty, refusal, factuality, or an internal
knowledge-control mechanism.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from mc006_parametric_fact_override_v2_repair import split_for_index
from mc007_semi_synthetic_familiar_entity_lookup import generate_answer, normalize_text
from mc016_alphabet_gated_numeric_arbitration import load_model_and_tokenizer


CARD_ID = "KSQ007"
CANDIDATE_ID = "ksq007_nonce_evidence_answerability_calibrator"
RUN_TYPE = "ksq007_nonce_evidence_answerability_structural_gate"
BEHAVIOR_RUN_TYPE = "ksq007_nonce_evidence_answerability_behavior"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/KSQ007_NONCE_EVIDENCE_ANSWERABILITY")
DEFAULT_RESULT_PATH = RESULT_DIR / "ksq007_nonce_evidence_answerability_first_run.json"
SMOKE_LIMIT10_RESULT_PATH = (
    RESULT_DIR / "ksq007_nonce_evidence_answerability_smoke_limit10.json"
)
FULL_BEHAVIOR_RESULT_PATH = (
    RESULT_DIR / "ksq007_nonce_evidence_answerability_full_behavior.json"
)
STATUS_CARD = Path("research/cards/KSQ007_NONCE_EVIDENCE_ANSWERABILITY_STATUS.md")
PREREG_PATH = Path("research/prereg/KSQ007_NONCE_EVIDENCE_ANSWERABILITY.md")

PANELS = (
    "exact_evidence_rows",
    "absent_evidence_rows",
    "unrelated_entity_rows",
    "conflicting_evidence_rows",
    "claim_only_control",
    "mention_only_control",
    "query_only_control",
)
CONTROL_PANELS = (
    "unrelated_entity_rows",
    "claim_only_control",
    "mention_only_control",
    "query_only_control",
)
ABSTAIN_PANELS = tuple(panel for panel in PANELS if panel != "exact_evidence_rows")
TEMPLATES = ("evidence_rows", "ledger_form", "compact_evidence")

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

ENTITY_STEMS = (
    "Aster",
    "Brom",
    "Cald",
    "Demer",
    "Eld",
    "Faron",
    "Galen",
    "Hestor",
    "Iven",
    "Joran",
    "Kelv",
    "Lumar",
    "Merek",
    "Norel",
    "Orin",
    "Prax",
    "Quen",
    "Rovel",
    "Saren",
    "Tovin",
    "Ulmar",
    "Vesk",
    "Wyrn",
    "Xand",
    "Yorin",
    "Zarek",
    "Aldor",
    "Bex",
    "Cyr",
    "Dov",
    "Erin",
    "Fex",
    "Gor",
    "Hale",
    "Irix",
    "Jex",
    "Korr",
    "Lorn",
    "Mav",
    "Nix",
)
VALUE_BANK = (
    "Virel",
    "Norva",
    "Calen",
    "Darsk",
    "Elvar",
    "Fend",
    "Glim",
    "Havon",
    "Istral",
    "Jorva",
    "Kelm",
    "Lorix",
    "Mavon",
    "Neral",
    "Orven",
    "Peld",
    "Quorin",
    "Ravik",
    "Seld",
    "Teral",
    "Ulven",
    "Varn",
    "Wexel",
    "Xorin",
    "Yelv",
    "Zoril",
    "Arven",
    "Bral",
    "Cevin",
    "Dovin",
    "Eral",
    "Faln",
    "Gevor",
    "Harin",
    "Ivenor",
    "Jask",
    "Kavon",
    "Lurin",
    "Meral",
    "Nask",
)
ALT_VALUE_BANK = (
    "Pavian",
    "Orel",
    "Nivor",
    "Meld",
    "Lask",
    "Korin",
    "Javel",
    "Irdan",
    "Helm",
    "Gavik",
    "Faronel",
    "Evin",
    "Dask",
    "Caldor",
    "Brel",
    "Aven",
    "Zendal",
    "Yarv",
    "Xeld",
    "Warin",
    "Vald",
    "Urel",
    "Tavon",
    "Sorin",
    "Reld",
    "Qavik",
    "Peldor",
    "Oskar",
    "Narin",
    "Mavel",
    "Ledor",
    "Kask",
    "Jorin",
    "Iral",
    "Havenor",
    "Geld",
    "Favel",
    "Eldor",
    "Darin",
    "Corv",
)
LURE_VALUE_BANK = (
    "Amber",
    "Beryl",
    "Copper",
    "Denim",
    "Ebony",
    "Frost",
    "Graph",
    "Hazel",
    "Indigo",
    "Jade",
    "Khaki",
    "Lilac",
    "Magenta",
    "Nickel",
    "Ochre",
    "Pearl",
    "Quartz",
    "Ruby",
    "Silver",
    "Topaz",
    "Umber",
    "Violet",
    "White",
    "Xenon",
    "Yellow",
    "Zinc",
    "Azure",
    "Bronze",
    "Cobalt",
    "Drift",
    "Ember",
    "Flint",
    "Gold",
    "Harbor",
    "Ivory",
    "Jet",
    "Kite",
    "Linen",
    "Marble",
    "Navy",
)


def source_rows(limit_sources: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, stem in enumerate(ENTITY_STEMS):
        source_id = f"nonce_{index + 1:02d}"
        entity = f"ENT-{stem}-{index + 41:02d}"
        decoy_entity = f"ENT-DECOY-{stem}-{index + 81:02d}"
        target_value = VALUE_BANK[index]
        alternate_value = ALT_VALUE_BANK[index]
        lure_value = LURE_VALUE_BANK[index]
        normalized = {
            normalize_text(entity),
            normalize_text(decoy_entity),
            normalize_text(target_value),
            normalize_text(alternate_value),
            normalize_text(lure_value),
            normalize_text("UNKNOWN"),
        }
        if len(normalized) != 6:
            raise ValueError(f"candidate collision for {source_id}")
        rows.append(
            {
                "source_index": index,
                "source_id": source_id,
                "entity": entity,
                "decoy_entity": decoy_entity,
                "target_value": target_value,
                "alternate_value": alternate_value,
                "lure_value": lure_value,
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
        "Return only the slot value, or UNKNOWN.",
        "Do not explain.",
        "Answer:",
    ]


def evidence_block(source: dict[str, Any], panel: str) -> tuple[list[str], str]:
    entity = source["entity"]
    decoy = source["decoy_entity"]
    target = source["target_value"]
    alt = source["alternate_value"]
    if panel == "exact_evidence_rows":
        return [f"EVIDENCE answer_for({entity})={target}"], "evidence_answer"
    if panel == "absent_evidence_rows":
        return [f"EVIDENCE region_tag({entity})=north"], "abstain"
    if panel == "unrelated_entity_rows":
        return [f"EVIDENCE answer_for({decoy})={target}"], "abstain"
    if panel == "conflicting_evidence_rows":
        if int(source["source_index"]) % 2 == 0:
            return [
                f"EVIDENCE answer_for({entity})={target}",
                f"EVIDENCE answer_for({entity})={alt}",
            ], "abstain"
        return [
            f"EVIDENCE answer_for({entity})={alt}",
            f"EVIDENCE answer_for({entity})={target}",
        ], "abstain"
    if panel == "claim_only_control":
        return [f"CLAIM answer_for({entity})={target}"], "control_abstain"
    if panel == "mention_only_control":
        return [f"MENTION {entity} {target}"], "control_abstain"
    if panel == "query_only_control":
        return [], "control_abstain"
    raise ValueError(f"unknown panel {panel}")


def render_prompt(
    source: dict[str, Any],
    *,
    panel: str,
    template: str,
) -> tuple[str, str]:
    rows, expected_label = evidence_block(source, panel)
    entity = source["entity"]
    if template == "evidence_rows":
        lines = [
            "Rows:",
            *[f"- {row}" for row in rows],
            "Use exactly one matching EVIDENCE answer_for(entity)=value row.",
            "If no matching row exists, or if matching rows disagree, answer UNKNOWN.",
            "CLAIM rows, MENTION rows, and other predicates do not determine the answer.",
            f"Query: answer_for({entity})",
        ]
    elif template == "ledger_form":
        lines = [
            "Ledger:",
            *rows,
            "A single non-conflicting EVIDENCE answer_for row determines the requested value.",
            "Other rows do not determine the requested value.",
            f"Requested slot: answer_for({entity})",
        ]
    elif template == "compact_evidence":
        lines = [
            "Data:",
            *rows,
            "One exact EVIDENCE answer_for row gives the value; absent or disagreeing rows mean UNKNOWN.",
            "Ignore CLAIM, MENTION, and non-answer rows.",
            f"Ask: answer_for({entity})",
        ]
    else:
        raise ValueError(f"unknown template {template}")
    return "\n".join([*lines, *response_suffix()]), expected_label


def source_records(
    limit_sources: int | None = None,
    templates: tuple[str, ...] = TEMPLATES,
    run_type: str = RUN_TYPE,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in source_rows(limit_sources):
        for template in templates:
            for panel in PANELS:
                prompt, expected_label = render_prompt(source, panel=panel, template=template)
                records.append(
                    {
                        "id": f"{CARD_ID}_{template}_{panel}_{source['source_id']}",
                        "card_id": CARD_ID,
                        "candidate_id": CANDIDATE_ID,
                        "run_type": run_type,
                        "model_id": MODEL_ID,
                        "template": template,
                        "panel": panel,
                        "split": source["split"],
                        "source_id": source["source_id"],
                        "source_index": source["source_index"],
                        "entity": source["entity"],
                        "decoy_entity": source["decoy_entity"],
                        "target_value": source["target_value"],
                        "alternate_value": source["alternate_value"],
                        "lure_value": source["lure_value"],
                        "expected_label": expected_label,
                        "candidate_answers": [
                            source["target_value"],
                            source["alternate_value"],
                            source["lure_value"],
                            "UNKNOWN",
                        ],
                        "prompt": prompt,
                    }
                )
    return records


def candidates(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"answer": record["target_value"], "candidate_label": "target_value"},
        {"answer": record["alternate_value"], "candidate_label": "alternate_value"},
        {"answer": record["lure_value"], "candidate_label": "lure_value"},
        {"answer": "UNKNOWN", "candidate_label": "abstain"},
    ]


def selected_label_for_candidate(record: dict[str, Any], candidate_label: str) -> str:
    panel = record["panel"]
    if panel == "exact_evidence_rows":
        if candidate_label == "target_value":
            return "evidence_answer"
        if candidate_label == "abstain":
            return "abstain"
        return "wrong_candidate"
    if candidate_label == "abstain":
        return "control_abstain" if panel in CONTROL_PANELS else "abstain"
    if panel == "unrelated_entity_rows" and candidate_label == "target_value":
        return "unrelated_value_reproduced"
    if panel == "conflicting_evidence_rows" and candidate_label in {
        "target_value",
        "alternate_value",
    }:
        return "conflict_value_selected"
    if panel == "claim_only_control" and candidate_label == "target_value":
        return "claim_only_reproduced"
    if panel == "mention_only_control" and candidate_label == "target_value":
        return "mention_only_reproduced"
    if panel == "query_only_control":
        return "query_only_answered"
    return "other_candidate"


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
    if panel in {"exact_evidence_rows", "unrelated_entity_rows", "claim_only_control", "mention_only_control"}:
        target = 1
    elif panel == "conflicting_evidence_rows":
        target = 1
    else:
        target = 0
    alternate = 1 if panel == "conflicting_evidence_rows" else 0
    lure = 0
    return {
        "target_value": target,
        "alternate_value": alternate,
        "lure_value": lure,
    }


def prompt_audit(record: dict[str, Any]) -> dict[str, Any]:
    prompt = record["prompt"]
    expected = expected_prompt_counts(record)
    suffix = "\n".join(response_suffix())
    target_count = exact_occurrences(prompt, record["target_value"])
    alternate_count = exact_occurrences(prompt, record["alternate_value"])
    lure_count = exact_occurrences(prompt, record["lure_value"])
    return {
        "target_value_occurrences": target_count,
        "alternate_value_occurrences": alternate_count,
        "lure_value_occurrences": lure_count,
        "target_value_prompt_count_expected": target_count == expected["target_value"],
        "alternate_value_prompt_count_expected": alternate_count == expected["alternate_value"],
        "lure_value_prompt_count_expected": lure_count == expected["lure_value"],
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
            audit["target_value_prompt_count_expected"]
            and audit["alternate_value_prompt_count_expected"]
            and audit["lure_value_prompt_count_expected"]
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


def next_token_candidate_logits(
    model: Any,
    tokenizer: Any,
    prompt: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].detach().float().cpu()
    token_ids: dict[str, int] = {}
    scores: dict[str, float] = {}
    for candidate in candidates(record):
        encoded = tokenizer(" " + candidate["answer"], add_special_tokens=False)["input_ids"]
        if not encoded:
            encoded = tokenizer(candidate["answer"], add_special_tokens=False)["input_ids"]
        first_id = int(encoded[0])
        token_ids[candidate["candidate_label"]] = first_id
        scores[candidate["candidate_label"]] = float(logits[first_id].item())
    return {
        "candidate_first_token_ids": token_ids,
        "candidate_first_token_logits": scores,
        "target_minus_abstain_logit": scores["target_value"] - scores["abstain"],
        "alternate_minus_abstain_logit": scores["alternate_value"] - scores["abstain"],
        "abstain_minus_target_logit": scores["abstain"] - scores["target_value"],
    }


def score_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    score_candidates: bool,
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
        }
        if score_candidates:
            output.update(next_token_candidate_logits(model, tokenizer, record["prompt"], record))
        outputs.append(output)
        print(
            f"[{index:03d}/{len(records):03d}] {record['id']} split={record['split']} "
            f"template={record['template']} panel={record['panel']} -> "
            f"{output['selected_label']} {str(output['selected_answer'])!r} "
            f"generated={generated['generated_text']!r}"
        )
    return outputs


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row.get("selected_label", "missing") for row in rows).items()))


def rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parseable = sum(1 for row in rows if row.get("parseable"))
    evidence_answer = sum(1 for row in rows if row.get("selected_label") == "evidence_answer")
    abstain = sum(1 for row in rows if row.get("selected_label") == "abstain")
    control_abstain = sum(1 for row in rows if row.get("selected_label") == "control_abstain")
    reproduced = sum(
        1
        for row in rows
        if row.get("selected_label")
        in {
            "unrelated_value_reproduced",
            "claim_only_reproduced",
            "mention_only_reproduced",
            "query_only_answered",
        }
    )
    conflict_value = sum(1 for row in rows if row.get("selected_label") == "conflict_value_selected")
    other = sum(
        1
        for row in rows
        if row.get("selected_label") in {"wrong_candidate", "other_candidate"}
    )
    unparsed = sum(1 for row in rows if row.get("selected_label") == "unparsed")
    candidate_margins = "candidate_first_token_logits" in rows[0] if rows else False
    result = {
        "rows": len(rows),
        "label_counts": label_counts(rows),
        "parseable": parseable,
        "parseable_rate": rate(parseable, len(rows)),
        "evidence_answer": evidence_answer,
        "evidence_answer_rate": rate(evidence_answer, len(rows)),
        "abstain": abstain,
        "abstain_rate": rate(abstain, len(rows)),
        "control_abstain": control_abstain,
        "control_abstain_rate": rate(control_abstain, len(rows)),
        "any_abstain": abstain + control_abstain,
        "any_abstain_rate": rate(abstain + control_abstain, len(rows)),
        "control_reproduced_value": reproduced,
        "control_reproduced_value_rate": rate(reproduced, len(rows)),
        "conflict_value_selected": conflict_value,
        "conflict_value_selected_rate": rate(conflict_value, len(rows)),
        "other_candidate": other,
        "other_candidate_rate": rate(other, len(rows)),
        "unparsed": unparsed,
        "unparsed_rate": rate(unparsed, len(rows)),
    }
    if candidate_margins:
        result["mean_target_minus_abstain_logit"] = sum(
            float(row["target_minus_abstain_logit"]) for row in rows
        ) / len(rows)
        result["mean_abstain_minus_target_logit"] = sum(
            float(row["abstain_minus_target_logit"]) for row in rows
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
        abstain_rows = [row for row in rows if row["panel"] in ABSTAIN_PANELS]
        result[template] = {
            "rows": len(rows),
            "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "selected_label_counts": label_counts(rows),
            "panels": panels,
            "controls": summarize_rows(control_rows),
            "abstain_panels": summarize_rows(abstain_rows),
            "holdout_panels": holdout_panels,
        }
    return result


def selection_key(summary: dict[str, Any], template: str, templates: tuple[str, ...]) -> tuple[float, ...]:
    panels = summary[template]["panels"]
    controls = summary[template]["controls"]
    exact = panels["exact_evidence_rows"]
    absent = panels["absent_evidence_rows"]
    unrelated = panels["unrelated_entity_rows"]
    conflict = panels["conflicting_evidence_rows"]
    claim = panels["claim_only_control"]
    mention = panels["mention_only_control"]
    query = panels["query_only_control"]
    min_parseability = min(float(panels[panel]["parseable_rate"]) for panel in PANELS)
    branch_rates = (
        float(exact["evidence_answer_rate"]),
        float(absent["abstain_rate"]),
        float(unrelated["control_abstain_rate"]),
        float(conflict["abstain_rate"]),
        float(claim["control_abstain_rate"]),
        float(mention["control_abstain_rate"]),
        float(query["control_abstain_rate"]),
    )
    return (
        min(branch_rates),
        *branch_rates,
        -float(controls["control_reproduced_value_rate"]),
        -float(conflict["conflict_value_selected_rate"]),
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
            "max weakest branch rate",
            "max exact-evidence answer rate",
            "max absent-evidence abstention",
            "max unrelated-entity abstention",
            "max contradiction abstention",
            "max claim-only abstention",
            "max mention-only abstention",
            "max query-only abstention",
            "min control value reproduction",
            "min conflict value selection",
            "max minimum panel parseability",
            "earliest template",
        ],
    }


def classify(criteria: dict[str, bool]) -> str:
    if not criteria["structural_passed"]:
        return "structural_invalid"
    if not criteria["selected_prompt_audit_passed"]:
        return "nonce_evidence_prompt_audit_failed"
    if not criteria["exact_evidence_rows_passed"]:
        return "nonce_evidence_answer_rows_failed"
    if not criteria["absent_evidence_rows_passed"]:
        return "nonce_evidence_absent_rows_failed"
    if not criteria["unrelated_entity_rows_passed"]:
        return "nonce_evidence_unrelated_entity_failed"
    if not criteria["conflicting_evidence_rows_passed"]:
        return "nonce_evidence_conflict_failed"
    if not criteria["claim_only_and_mention_only_controls_passed"]:
        return "nonce_evidence_claim_or_mention_control_failed"
    if not criteria["query_only_control_passed"]:
        return "nonce_evidence_query_only_failed"
    if not criteria["candidate_and_output_margins_reported"]:
        return "nonce_evidence_behavior_passed_baselines_missing"
    if criteria["smoke_mode"]:
        return "smoke_nonce_evidence_answerability_candidate"
    if not criteria["source_disjoint_answerability_holdout_passed"]:
        return "nonce_evidence_holdout_failed"
    return "nonce_evidence_answerability_behavior_ready"


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
    holdout = selected["holdout_panels"]
    selected_rows = [row for row in outputs if row["template"] == selection["selected_template"]]
    prompt_audit_passed = all(
        row["target_value_prompt_count_expected"]
        and row["alternate_value_prompt_count_expected"]
        and row["lure_value_prompt_count_expected"]
        and not row["prompt_has_status_lexeme"]
        and not row["prompt_has_support_word"]
        and row["shared_response_suffix_present"]
        for row in selected_rows
    )
    controls = selected["controls"]
    claim_mention = [
        *[row for row in outputs if row["template"] == selection["selected_template"] and row["panel"] == "claim_only_control"],
        *[row for row in outputs if row["template"] == selection["selected_template"] and row["panel"] == "mention_only_control"],
    ]
    claim_mention_summary = summarize_rows(claim_mention)
    holdout_passed = (
        float(holdout["exact_evidence_rows"]["evidence_answer_rate"]) >= 0.75
        and float(holdout["absent_evidence_rows"]["abstain_rate"]) >= 0.75
        and float(holdout["unrelated_entity_rows"]["control_abstain_rate"]) >= 0.75
        and float(holdout["conflicting_evidence_rows"]["abstain_rate"]) >= 0.75
        and float(holdout["claim_only_control"]["control_abstain_rate"]) >= 0.75
        and float(holdout["mention_only_control"]["control_abstain_rate"]) >= 0.75
        and float(holdout["query_only_control"]["control_abstain_rate"]) >= 0.75
    )
    criteria = {
        "smoke_mode": not full_run,
        "structural_passed": structural["passed"],
        "full_source_count_is_40": structural["source_count"] == 40,
        "source_disjoint_holdout": structural["criteria"]["source_split_disjoint"],
        "selected_prompt_audit_passed": prompt_audit_passed,
        "exact_evidence_rows_passed": float(panels["exact_evidence_rows"]["parseable_rate"]) >= 0.90
        and float(panels["exact_evidence_rows"]["evidence_answer_rate"]) >= 0.80,
        "absent_evidence_rows_passed": float(panels["absent_evidence_rows"]["parseable_rate"]) >= 0.90
        and float(panels["absent_evidence_rows"]["abstain_rate"]) >= 0.80,
        "unrelated_entity_rows_passed": float(panels["unrelated_entity_rows"]["parseable_rate"]) >= 0.90
        and float(panels["unrelated_entity_rows"]["control_abstain_rate"]) >= 0.80
        and float(panels["unrelated_entity_rows"]["control_reproduced_value_rate"]) <= 0.10,
        "conflicting_evidence_rows_passed": float(panels["conflicting_evidence_rows"]["parseable_rate"]) >= 0.90
        and float(panels["conflicting_evidence_rows"]["abstain_rate"]) >= 0.80
        and float(panels["conflicting_evidence_rows"]["conflict_value_selected_rate"]) <= 0.10,
        "claim_only_and_mention_only_controls_passed": float(claim_mention_summary["parseable_rate"]) >= 0.90
        and float(claim_mention_summary["control_abstain_rate"]) >= 0.80
        and float(claim_mention_summary["control_reproduced_value_rate"]) <= 0.10,
        "query_only_control_passed": float(panels["query_only_control"]["parseable_rate"]) >= 0.90
        and float(panels["query_only_control"]["control_abstain_rate"]) >= 0.80,
        "source_disjoint_answerability_holdout_passed": holdout_passed,
        "candidate_and_output_margins_reported": score_candidates,
    }
    diagnostic_class = classify(criteria)
    behavior_ready = diagnostic_class == "nonce_evidence_answerability_behavior_ready"
    return {
        "diagnostic_class": diagnostic_class,
        "structural": structural,
        "criteria": criteria,
        "selection": selection,
        "selected_template_summary": selected,
        "by_template": by_template,
        "calibrator_decision": {
            "behavior_ready": behavior_ready,
            "signature_screen_allowed": False,
            "hidden_state_claim_allowed": False,
            "intervention_allowed": False,
            "mechanism_claim_allowed": False,
            "route_decision": "calibrate_answerability_axis_only"
            if behavior_ready
            else "nonce_evidence_answerability_failed",
            "exported_diagnostic_class": "NONCE_EVIDENCE_ANSWERABILITY_CALIBRATOR"
            if behavior_ready
            else "NONCE_EVIDENCE_ANSWERABILITY_BOUNDARY",
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")


def build_result(
    records: list[dict[str, Any]],
    outputs: list[dict[str, Any]] | None,
    templates: tuple[str, ...],
    full_run: bool,
    score_candidates: bool,
) -> dict[str, Any]:
    if outputs is None:
        structural = structural_check(records, templates)
        summary = {
            "diagnostic_class": "structural_passed" if structural["passed"] else "structural_failed",
            "structural": structural,
            "criteria": {
                "structural_passed": structural["passed"],
                "full_source_count_is_40": structural["source_count"] == 40,
                "source_disjoint_holdout": structural["criteria"]["source_split_disjoint"],
                "hidden_state_license": False,
            },
            "calibrator_decision": {
                "behavior_ready": False,
                "signature_screen_allowed": False,
                "hidden_state_claim_allowed": False,
                "intervention_allowed": False,
                "mechanism_claim_allowed": False,
                "route_decision": "structural_only",
                "exported_diagnostic_class": None,
            },
        }
    else:
        summary = summarize(records, outputs, templates, full_run, score_candidates)
    return {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "card_id": CARD_ID,
        "candidate_id": CANDIDATE_ID,
        "run_type": BEHAVIOR_RUN_TYPE if outputs is not None else RUN_TYPE,
        "model_id": MODEL_ID,
        "full_run": full_run,
        "templates": list(templates),
        "panels": list(PANELS),
        "summary": summary,
        "records": outputs if outputs is not None else records,
        "allowed_claim": (
            "KSQ007 calibrates whether answerability can be made stable when "
            "world-prior and real-city pressure are removed. It is behavior-only "
            "and cannot establish real uncertainty or an internal mechanism."
        ),
        "forbidden_claim": (
            "KSQ007 does not prove factuality, refusal, uncertainty, knowledge "
            "control, hidden-state causality, or intervention success."
        ),
    }


def write_prereg(path: Path) -> None:
    lines = [
        "# KSQ007 Nonce-Evidence Answerability Calibrator",
        "",
        "Status: post-KSQ005/006 behavior-only calibration work order.",
        "",
        "Runner:",
        "",
        "> `code/ksq007_nonce_evidence_answerability_calibrator.py`",
        "",
        "Artifacts:",
        "",
        "> `results/cards/KSQ007_NONCE_EVIDENCE_ANSWERABILITY/ksq007_nonce_evidence_answerability_first_run.json`",
        "",
        "> `results/cards/KSQ007_NONCE_EVIDENCE_ANSWERABILITY/ksq007_nonce_evidence_answerability_smoke_limit10.json`",
        "",
        "> `results/cards/KSQ007_NONCE_EVIDENCE_ANSWERABILITY/ksq007_nonce_evidence_answerability_full_behavior.json`",
        "",
        "## Claim Under Test",
        "",
        "The KSQ005/006 relation-evidence failure may be caused by familiar",
        "capital priors and city-value salience rather than by answerability",
        "grammar itself. KSQ007 removes real-world priors with nonce entities",
        "and nonce slot values while preserving exact evidence, absent evidence,",
        "unrelated entity, contradiction, claim-only, mention-only, query-only,",
        "holdout, prompt-audit, and candidate/output controls.",
        "",
        "## Promotion Rule",
        "",
        "Promote only to calibrator status if exact evidence rows answer, all",
        "absent/unrelated/conflicting/control rows abstain, source-disjoint",
        "holdout passes, prompt audit passes, and candidate/output margins are",
        "reported. A pass still does not license hidden-state work.",
        "",
        "## Kill Rule",
        "",
        "Kill this nonce-evidence route if it cannot jointly preserve exact",
        "evidence answering and abstention on unrelated, contradiction, claim,",
        "mention, and query-only controls.",
        "",
        "## Forbidden Claims",
        "",
        "- This is not a mechanism card.",
        "- This is not real uncertainty, refusal, or factual correction.",
        "- Hidden-state probing remains forbidden from this result alone.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_status_card(path: Path, result: dict[str, Any], output_path: Path) -> None:
    summary = result["summary"]
    criteria = summary["criteria"]
    decision = summary["calibrator_decision"]
    lines = [
        "# KSQ007 Nonce-Evidence Answerability Status",
        "",
        "Date: 2026-07-02",
        "",
        "Runner:",
        "",
        "> `code/ksq007_nonce_evidence_answerability_calibrator.py`",
        "",
        "Result:",
        "",
        f"> `{output_path.as_posix()}`",
        "",
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
    for key, value in criteria.items():
        lines.append(f"| `{key}` | `{str(value).lower()}` |")
    if "selection" in summary:
        selected = summary["selection"]["selected_template"]
        lines.extend(
            [
                "",
                "## Selected Template",
                "",
                f"- selected template: `{selected}`",
                f"- selection key: `{json.dumps(summary['selection']['selection_key'])}`",
                "",
                "## Selected Panels",
                "",
                "| Panel | Rows | Label Counts | Key Rate |",
                "| --- | ---: | --- | ---: |",
            ]
        )
        panels = summary["selected_template_summary"]["panels"]
        for panel_name, panel in panels.items():
            if panel_name == "exact_evidence_rows":
                key_rate = panel["evidence_answer_rate"]
            elif panel_name in CONTROL_PANELS:
                key_rate = panel["control_abstain_rate"]
            else:
                key_rate = panel["abstain_rate"]
            lines.append(
                f"| `{panel_name}` | {panel['rows']} | "
                f"`{json.dumps(panel['label_counts'], sort_keys=True)}` | "
                f"{key_rate:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Forbidden Claims",
            "",
            "- KSQ007 is a mechanism card.",
            "- KSQ007 proves real uncertainty, refusal, factual correction, or knowledge control.",
            "- KSQ007 licenses hidden-state probing or intervention.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--score-candidates", action="store_true")
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--full-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--write-prereg", action="store_true")
    parser.add_argument("--prereg-path", type=Path, default=PREREG_PATH)
    parser.add_argument("--write-status-card", action="store_true")
    parser.add_argument("--status-card", type=Path, default=STATUS_CARD)
    args = parser.parse_args()

    if args.limit_sources is not None and args.full_run:
        raise ValueError("--full-run cannot be combined with --limit-sources")

    output_path = args.output_path
    if output_path is None:
        if args.score and args.limit_sources == 10:
            output_path = SMOKE_LIMIT10_RESULT_PATH
        elif args.score and args.full_run:
            output_path = FULL_BEHAVIOR_RESULT_PATH
        else:
            output_path = DEFAULT_RESULT_PATH

    records = source_records(
        limit_sources=args.limit_sources,
        templates=TEMPLATES,
        run_type=BEHAVIOR_RUN_TYPE if args.score else RUN_TYPE,
    )
    outputs = None
    if args.score:
        model, tokenizer = load_model_and_tokenizer(MODEL_ID, args.local_files_only)
        outputs = score_records(
            records,
            tokenizer,
            model,
            args.max_new_tokens,
            args.score_candidates,
        )
    result = build_result(
        records,
        outputs,
        TEMPLATES,
        full_run=args.full_run,
        score_candidates=args.score_candidates,
    )
    if args.write_prereg:
        write_prereg(args.prereg_path)
    if args.write:
        write_json(output_path, result)
        if args.write_status_card:
            write_status_card(args.status_card, result, output_path)
    print(json.dumps(result["summary"]["criteria"], sort_keys=True))
    if args.score and not all(result["summary"]["criteria"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
