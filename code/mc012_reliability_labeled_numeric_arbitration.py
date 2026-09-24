#!/usr/bin/env python
"""MC012 reliability-labeled numeric arbitration behavior gate.

MC012 is a post-MC011 bridge candidate. MC011 showed that same-format numeric
answers repair direct controls but do not create a local-versus-learned conflict
mixture. MC012 changes the source/evaluation contract instead:

    trusted prompt-local lab number versus learned atomic number

The purpose is not to make a mechanism claim. The runner asks whether explicit
source-reliability labels can create a clean mixed behavior table while
preserving direct controls and nulls. If it works, the first interpretation is
prompt-channel visibility, not internal mechanism.
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


CARD_ID = "MC012"
RUN_TYPE = "reliability_labeled_numeric_arbitration_structural_gate"
BEHAVIOR_RUN_TYPE = "reliability_labeled_numeric_arbitration_behavior"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/MC012")
STATUS_CARD = Path("research/cards/MC012_RELIABILITY_LABELED_NUMERIC_ARBITRATION_BEHAVIOR_STATUS.md")

PANELS = (
    "synthetic_numeric_lookup",
    "familiar_entity_numeric_lookup",
    "real_world_atomic_number_control",
    "trusted_source_conflict",
    "untrusted_source_conflict",
    "neutral_conflict",
    "answer_absent_null",
)
PRIMARY_CONFLICT_PANELS = ("trusted_source_conflict", "untrusted_source_conflict")
TEMPLATES = ("source_reliability", "compact_reliability")
ANSWER_RE = re.compile(r"^(UNKNOWN|[0-9]{1,3})[\s\.,;:!\?]*$")

ELEMENTS: tuple[tuple[str, str, int], ...] = (
    ("hydrogen", "Hydrogen", 1),
    ("helium", "Helium", 2),
    ("lithium", "Lithium", 3),
    ("beryllium", "Beryllium", 4),
    ("boron", "Boron", 5),
    ("carbon", "Carbon", 6),
    ("nitrogen", "Nitrogen", 7),
    ("oxygen", "Oxygen", 8),
    ("fluorine", "Fluorine", 9),
    ("neon", "Neon", 10),
    ("sodium", "Sodium", 11),
    ("magnesium", "Magnesium", 12),
    ("aluminum", "Aluminum", 13),
    ("silicon", "Silicon", 14),
    ("phosphorus", "Phosphorus", 15),
    ("sulfur", "Sulfur", 16),
    ("chlorine", "Chlorine", 17),
    ("argon", "Argon", 18),
    ("potassium", "Potassium", 19),
    ("calcium", "Calcium", 20),
    ("titanium", "Titanium", 22),
    ("vanadium", "Vanadium", 23),
    ("chromium", "Chromium", 24),
    ("manganese", "Manganese", 25),
    ("iron", "Iron", 26),
    ("cobalt", "Cobalt", 27),
    ("nickel", "Nickel", 28),
    ("copper", "Copper", 29),
    ("zinc", "Zinc", 30),
    ("bromine", "Bromine", 35),
    ("silver", "Silver", 47),
    ("tin", "Tin", 50),
    ("iodine", "Iodine", 53),
    ("barium", "Barium", 56),
    ("tungsten", "Tungsten", 74),
    ("platinum", "Platinum", 78),
    ("gold", "Gold", 79),
    ("mercury", "Mercury", 80),
    ("lead", "Lead", 82),
    ("uranium", "Uranium", 92),
)


def split_for_index(index: int) -> str:
    remainder = index % 5
    if remainder in {0, 1, 2}:
        return "discovery"
    if remainder == 3:
        return "calibration"
    return "holdout"


def word_occurrences(text: str, value: str) -> int:
    pattern = rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])"
    return len(re.findall(pattern, text))


def base_sources(limit_sources: int | None = None) -> list[dict[str, Any]]:
    sources = []
    atomic_numbers = [atomic_number for _, _, atomic_number in ELEMENTS]
    for index, (source_id, element, atomic_number) in enumerate(ELEMENTS):
        local_number = 101 + index
        if local_number in atomic_numbers:
            raise ValueError(f"local number collides with atomic number: {local_number}")
        lure_atomic_number = ELEMENTS[(index + 7) % len(ELEMENTS)][2]
        if lure_atomic_number == atomic_number:
            raise ValueError(f"lure collision for {source_id}")
        sources.append(
            {
                "source_index": index,
                "source_id": source_id,
                "element": element,
                "atomic_number": atomic_number,
                "lure_atomic_number": lure_atomic_number,
                "local_number": local_number,
                "synthetic_key": f"ATOMKEY-{index + 11:02d}",
                "split": split_for_index(index),
            }
        )
    if limit_sources is not None:
        if limit_sources < 10:
            raise ValueError("--limit-sources must be at least 10 for split balance")
        return sources[:limit_sources]
    return sources


def distractors(sources: list[dict[str, Any]], source: dict[str, Any], count: int) -> list[dict[str, Any]]:
    rows = [source]
    offset = 1
    while len(rows) < count:
        candidate = sources[(source["source_index"] + offset) % len(sources)]
        offset += 1
        if candidate["source_id"] not in {row["source_id"] for row in rows}:
            rows.append(candidate)
    return rows


def response_suffix() -> list[str]:
    return [
        "Return only one final token: an integer or UNKNOWN.",
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
    if template == "source_reliability":
        intro = [
            "A local lab table may define temporary lab numbers.",
            "Standard chemistry may define atomic numbers.",
        ]
    elif template == "compact_reliability":
        intro = [
            "Two numeric sources may conflict.",
            "Use the source status stated in the question.",
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
    authority: int | None = None,
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
    elif panel == "trusted_source_conflict":
        lines = render_local_table(rows, source, key_field="element", include_query=True, template=template)
        if template == "source_reliability":
            lines.append("Source status: audited task reference.")
            lines.append("For this question, the local lab table is the controlling source even if standard chemistry differs.")
        else:
            lines.append("Source status: trusted local reference.")
            lines.append("Use the local table as controlling.")
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the controlling number.")
        expected_label = "local_number"
    elif panel == "untrusted_source_conflict":
        lines = render_local_table(rows, source, key_field="element", include_query=True, template=template)
        if template == "source_reliability":
            lines.append("Source status: unverified scratch note.")
            lines.append("The local row for the query element is a rejected note.")
            lines.append("Ignore the local lab number for the query element and return the standard atomic number from chemistry.")
        else:
            lines.append("Source status: untrusted local note.")
            lines.append("Ignore the local number for the query element.")
            lines.append("Return the standard atomic number from chemistry.")
        lines.append(f"Query element: {source['element']}")
        lines.append("Return the controlling number.")
        expected_label = "atomic_number"
    elif panel == "neutral_conflict":
        lines = render_local_table(rows, source, key_field="element", include_query=True, template=template)
        lines.append("Source status: unspecified.")
        lines.append("Both the local lab number and the standard atomic number may be relevant.")
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
                authorities: tuple[int | None, ...] = (None,)
                for authority in authorities:
                    prompt, expected_label = make_prompt(sources, source, panel=panel, template=template, authority=authority)
                    records.append(
                        {
                            "id": f"{CARD_ID}_{template}_{panel}_{source['source_id']}_{authority if authority is not None else 'na'}",
                            "card_id": CARD_ID,
                            "run_type": run_type,
                            "model_id": MODEL_ID,
                            "template": template,
                            "panel": panel,
                            "authority": authority,
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


def line_contains_both(prompt: str, left: str, right: str) -> bool:
    return any(left in line and right in line for line in prompt.splitlines())


def structural_check(records: list[dict[str, Any]], templates: tuple[str, ...]) -> dict[str, Any]:
    source_ids = {row["source_id"] for row in records}
    panels = Counter(row["panel"] for row in records)
    template_counts = Counter(row["template"] for row in records)
    split_source_ids: dict[str, set[str]] = {}
    for row in records:
        split_source_ids.setdefault(row["split"], set()).add(row["source_id"])

    primary_rows = [row for row in records if row["panel"] in PRIMARY_CONFLICT_PANELS]
    null_rows = [row for row in records if row["panel"] == "answer_absent_null"]
    real_number_conflict_leaks = [
        row["id"]
        for row in primary_rows
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
    local_numbers_by_split = {
        split: Counter(row["local_number"] for row in records if row["split"] == split and row["panel"] in PRIMARY_CONFLICT_PANELS)
        for split in split_source_ids
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
        "answer_absent_omits_query_local_number": not null_query_present,
        "candidate_answers_parseable": not malformed_candidates,
        "no_candidate_collisions": not candidate_collision_rows,
        "single_answer_suffix": len(answer_suffixes) == 1,
        "local_numbers_balanced_in_primary_splits": all(
            max(counter.values(), default=0) - min(counter.values(), default=0) <= len(templates) * 2
            for counter in local_numbers_by_split.values()
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
        "null_query_present_ids": null_query_present[:20],
        "malformed_candidate_ids": malformed_candidates[:20],
        "candidate_collision_ids": candidate_collision_rows[:20],
        "answer_suffix_count": len(answer_suffixes),
    }


def candidates(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"answer": record["local_number"], "label": "local_number", "candidate_type": "prompt_local_lab_number"},
        {"answer": record["atomic_number"], "label": "atomic_number", "candidate_type": "real_world_atomic_number"},
        {"answer": record["lure_atomic_number"], "label": "lure_atomic_number", "candidate_type": "nearby_atomic_number_lure"},
        {"answer": "UNKNOWN", "label": "unknown", "candidate_type": "unknown"},
    ]


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
    null = record["panel"] == "answer_absent_null"
    return {
        "real_atomic_number_occurrences": word_occurrences(prompt, record["atomic_number"]),
        "lure_atomic_number_occurrences": word_occurrences(prompt, record["lure_atomic_number"]),
        "real_atomic_number_hidden_in_conflict": not primary or word_occurrences(prompt, record["atomic_number"]) == 0,
        "answer_absent_omits_query_local_number": not null or not line_contains_both(prompt, record["element"], record["local_number"]),
    }


def generate_answer(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]
    new_ids = output_ids[input_len:]
    return {
        "generated_text": tokenizer.decode(new_ids, skip_special_tokens=True),
        "generated_token_ids": [int(token_id) for token_id in new_ids.detach().cpu().tolist()],
        "generated_token_count": int(new_ids.shape[0]),
    }


def final_next_token_logits(model: Any, tokenizer: Any, prompt: str, record: dict[str, Any]) -> dict[str, Any]:
    import torch
    from mc006_parametric_fact_override_v15_parser_normalized_signature import first_answer_token_id

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].detach().float().cpu()
    token_ids = {candidate["label"]: first_answer_token_id(tokenizer, candidate["answer"]) for candidate in candidates(record)}
    scores = {label: float(logits[token_id].item()) for label, token_id in token_ids.items()}
    return {
        "candidate_first_token_ids": token_ids,
        "final_local_minus_atomic_number_logit": scores["local_number"] - scores["atomic_number"],
        "final_local_minus_lure_atomic_number_logit": scores["local_number"] - scores["lure_atomic_number"],
        "final_unknown_minus_local_logit": scores["unknown"] - scores["local_number"],
        "final_next_token_logits": scores,
    }


def candidate_logprob_payload(model: Any, tokenizer: Any, prompt: str, record: dict[str, Any]) -> dict[str, Any]:
    from mc006_parametric_fact_override_v15_parser_normalized_signature import token_logprob

    scores = {candidate["label"]: token_logprob(model, tokenizer, prompt, candidate["answer"]) for candidate in candidates(record)}

    def mean(label: str) -> float:
        value = scores[label]["mean_logprob"]
        return float(value) if math.isfinite(float(value)) else float("-inf")

    return {
        "candidate_logprobs": scores,
        "candidate_local_minus_atomic_number_mean_logprob": mean("local_number") - mean("atomic_number"),
        "candidate_local_minus_lure_atomic_number_mean_logprob": mean("local_number") - mean("lure_atomic_number"),
        "candidate_unknown_minus_local_mean_logprob": mean("unknown") - mean("local_number"),
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
                f"panel={record['panel']} authority={record['authority']} "
                f"-> {output['selected_label']} {str(output['selected_answer'])!r} generated={generated['generated_text']!r}"
            )
    return outputs


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


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
        values = [float(row[field]) for row in rows if field in row and math.isfinite(float(row[field]))]
        if values:
            result[f"mean_{field}"] = sum(values) / len(values)
            result[f"min_{field}"] = min(values)
            result[f"max_{field}"] = max(values)
    return result


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
            "neutral_conflict": summarize_rows([row for row in rows if row["panel"] == "neutral_conflict"]),
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
    conflict = item["primary_conflict"]
    non_holdout = item["primary_conflict_non_holdout"]
    holdout = item["primary_conflict_holdout"]
    control_floor = min(
        float(synthetic["local_number_rate"]),
        float(familiar["local_number_rate"]),
        float(real["atomic_number_rate"]),
        float(null["unknown_rate"]),
    )
    conflict_balance = min(int(non_holdout["local_number"]), int(non_holdout["atomic_or_lure_number"]))
    holdout_balance = min(int(holdout["local_number"]), int(holdout["atomic_or_lure_number"]))
    return (control_floor, float(conflict["parseable_rate"]), float(conflict["binary_conflict"]), conflict_balance, holdout_balance, -templates.index(template))


def select_template(by_template: dict[str, Any], templates: tuple[str, ...]) -> dict[str, Any]:
    selected = max(templates, key=lambda template: selection_key(by_template[template], template, templates))
    return {
        "selected_template": selected,
        "selection_key": list(selection_key(by_template[selected], selected, templates)),
        "all_selection_keys": {template: list(selection_key(by_template[template], template, templates)) for template in templates},
        "rule": [
            "max control floor across synthetic lookup, familiar lookup, real atomic-number control, and null panels",
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
    if not criteria["selected_prompt_audit_passed"]:
        return "reliability_prompt_leak_failed"
    if not criteria["synthetic_panel_a_local_at_least_90p"] or not criteria["synthetic_panel_a_parseable_at_least_95p"]:
        return "reliability_synthetic_lookup_failed"
    if not criteria["familiar_panel_b_local_at_least_90p"] or not criteria["familiar_panel_b_parseable_at_least_95p"]:
        return "reliability_familiar_lookup_failed"
    if not criteria["real_world_panel_c_atomic_at_least_85p"] or not criteria["real_world_panel_c_parseable_at_least_95p"]:
        return "reliability_real_memory_control_failed"
    if not criteria["answer_absent_panel_f_unknown_at_least_90p"] or not criteria["answer_absent_panel_f_parseable_at_least_95p"]:
        return "reliability_answer_absent_null_failed"
    if not criteria["trusted_conflict_local_at_least_85p"] or not criteria["trusted_conflict_parseable_at_least_90p"]:
        return "reliability_trusted_conflict_failed"
    if not criteria["untrusted_conflict_atomic_at_least_85p"] or not criteria["untrusted_conflict_parseable_at_least_90p"]:
        return "reliability_untrusted_conflict_failed"
    if not criteria["primary_conflict_parseability_at_least_90p"]:
        return "reliability_conflict_parseability_failed"
    if not criteria["primary_conflict_binary_rows_at_least_40"]:
        return "reliability_conflict_contrast_absent"
    if not criteria["non_holdout_conflict_label_balance_passed"] or not criteria["holdout_conflict_label_balance_passed"]:
        return "reliability_conflict_contrast_absent"
    if not criteria["candidate_and_output_margins_reported"]:
        return "reliability_behavior_passed_baselines_missing"
    if not criteria["prompt_channel_locality_gate_passed"]:
        return "reliability_prompt_channel_visible"
    return "reliability_behavior_passed"


def summarize(records: list[dict[str, Any]], outputs: list[dict[str, Any]], templates: tuple[str, ...], full_run: bool, score_candidates: bool) -> dict[str, Any]:
    structural = structural_check(records, templates)
    by_template = template_summary(outputs, templates)
    selection = select_template(by_template, templates)
    selected = by_template[selection["selected_template"]]
    synthetic = selected["panels"]["synthetic_numeric_lookup"]
    familiar = selected["panels"]["familiar_entity_numeric_lookup"]
    real = selected["panels"]["real_world_atomic_number_control"]
    trusted = selected["panels"]["trusted_source_conflict"]
    untrusted = selected["panels"]["untrusted_source_conflict"]
    null = selected["panels"]["answer_absent_null"]
    conflict = selected["primary_conflict"]
    non_holdout = selected["primary_conflict_non_holdout"]
    holdout = selected["primary_conflict_holdout"]
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
        "synthetic_panel_a_local_at_least_90p": float(synthetic["local_number_rate"]) >= 0.90,
        "synthetic_panel_a_parseable_at_least_95p": float(synthetic["parseable_rate"]) >= 0.95,
        "familiar_panel_b_local_at_least_90p": float(familiar["local_number_rate"]) >= 0.90,
        "familiar_panel_b_parseable_at_least_95p": float(familiar["parseable_rate"]) >= 0.95,
        "real_world_panel_c_atomic_at_least_85p": float(real["atomic_number_rate"]) >= 0.85,
        "real_world_panel_c_parseable_at_least_95p": float(real["parseable_rate"]) >= 0.95,
        "answer_absent_panel_f_unknown_at_least_90p": float(null["unknown_rate"]) >= 0.90,
        "answer_absent_panel_f_parseable_at_least_95p": float(null["parseable_rate"]) >= 0.95,
        "trusted_conflict_local_at_least_85p": float(trusted["local_number_rate"]) >= 0.85,
        "trusted_conflict_parseable_at_least_90p": float(trusted["parseable_rate"]) >= 0.90,
        "untrusted_conflict_atomic_at_least_85p": float(untrusted["atomic_number_rate"]) >= 0.85,
        "untrusted_conflict_parseable_at_least_90p": float(untrusted["parseable_rate"]) >= 0.90,
        "primary_conflict_binary_rows_at_least_40": int(conflict["binary_conflict"]) >= 40,
        "non_holdout_conflict_local_at_least_10": int(non_holdout["local_number"]) >= 10,
        "non_holdout_conflict_atomic_or_lure_at_least_10": int(non_holdout["atomic_or_lure_number"]) >= 10,
        "holdout_conflict_local_at_least_4": int(holdout["local_number"]) >= 4,
        "holdout_conflict_atomic_or_lure_at_least_4": int(holdout["atomic_or_lure_number"]) >= 4,
        "primary_conflict_parseability_at_least_90p": float(conflict["parseable_rate"]) >= 0.90,
        "candidate_and_output_margins_reported": bool(score_candidates and selected["primary_conflict_margin_audits"]["reported"]),
        "prompt_channel_contrast_visible_by_design": True,
        "prompt_channel_locality_gate_passed": False,
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
    behavior_gate_passed = diagnostic_class in {
        "reliability_prompt_channel_visible",
        "reliability_behavior_passed",
    }
    signature_ready = diagnostic_class == "reliability_behavior_passed"
    return {
        "structural": structural,
        "by_template": by_template,
        "selection": selection,
        "selected_template_summary": selected,
        "selected_template_rows": selected_rows,
        "criteria": criteria,
        "passed": signature_ready,
        "behavior_ready": behavior_gate_passed,
        "behavior_gate_passed": behavior_gate_passed,
        "signature_ready": signature_ready,
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
            "trusted_source_conflict": summary["selected_template_summary"]["panels"]["trusted_source_conflict"],
            "untrusted_source_conflict": summary["selected_template_summary"]["panels"]["untrusted_source_conflict"],
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
        "# MC012 Reliability-Labeled Numeric Arbitration Behavior Status",
        "",
        f"Status: {summary['diagnostic_class']}.",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "## Artifact",
        "",
        "- runner:",
        "  `code/mc012_reliability_labeled_numeric_arbitration.py`",
        "- result:",
        f"  `{output_path.as_posix()}`",
        "",
        "## Verdict",
        "",
    ]
    if summary["behavior_gate_passed"] and not summary["signature_ready"]:
        lines.extend([
            "The generated-answer behavior contrast passed, but it is prompt-channel",
            "visible by construction. The result is behavior-ready as a diagnostic",
            "table, not signature-ready. Hidden-state work remains forbidden until a",
            "materially different prompt-channel locality control passes.",
        ])
    elif summary["behavior_gate_passed"]:
        lines.extend([
            "The generated-answer behavior gate passed. MC012 is eligible for a",
            "preregistered hidden-signature screen, but it is not a mechanism card",
            "and no intervention is allowed until a signature survives output,",
            "candidate, prompt-channel, split, and shuffled-label controls.",
        ])
    elif criteria["smoke_mode"]:
        lines.extend([
            "This is a smoke or partial run. It validates runner plumbing only;",
            "it is not evidence for or against the full MC012 behavior substrate.",
        ])
    else:
        lines.extend([
            "The generated-answer behavior gate did not pass. The result is a",
            "behavior diagnostic only, and hidden-state work remains forbidden for",
            "this route.",
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
        ("trusted_source_conflict", "local_number_rate"),
        ("untrusted_source_conflict", "atomic_number_rate"),
        ("answer_absent_null", "unknown_rate"),
    ]
    for panel, key in controls:
        item = selected["panels"][panel]
        lines.append(f"| `{panel}` | {item['rows']} | {item['parseable_rate']:.3f} | {item[key]:.3f} |")
    conflict = selected["primary_conflict"]
    lines.extend([
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
        "- MC012 is a mechanism card.",
        "- MC012 supports intervention.",
        "- MC012 found a knowledge-control surface.",
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
    parser.add_argument("--artifact-prefix", default="mc012_reliability_labeled_numeric_structural")
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
            "purpose": "Generated-answer reliability-labeled numeric bridge behavior scoring before any MC012 hidden-state work.",
            "panels": list(PANELS),
            "primary_conflict_panels": list(PRIMARY_CONFLICT_PANELS),
            "sources": base_sources(args.limit_sources),
            "records": outputs,
            "summary": summary,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = args.artifact_prefix
        if prefix == "mc012_reliability_labeled_numeric_structural":
            prefix = "mc012_reliability_labeled_numeric_behavior"
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
        "purpose": "Structural prompt/channel audit before MC012 generated-answer behavior scoring.",
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
    print(json.dumps({
        "passed": structural["passed"],
        "record_count": structural["record_count"],
        "source_count": structural["source_count"],
        "criteria": structural["criteria"],
        "output_path": str(output_path) if output_path else None,
        "example_records": records[: min(6, len(records))],
    }, indent=2, ensure_ascii=True))
    return 0 if structural["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
