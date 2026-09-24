#!/usr/bin/env python
"""MC009 derived-code arbitration behavior gate.

This runner implements the first MC009 behavior substrate. MC009 removes the
direct element -> artificial-code source-value row used by MC008. The task-local
answer is derived from the queried element's row position under a generic
row-code rule, while the competing real-world answer is the element's learned
chemical symbol.

The runner deliberately stops at generated-answer behavior plus
output/candidate baselines. Hidden-state search and intervention remain
forbidden until the behavior gate passes on a full source-disjoint run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc006_parametric_fact_override_v2_repair import split_for_index
from mc006_parametric_fact_override_v15_parser_normalized_signature import (
    first_answer_token_id,
    token_logprob,
)


CARD_ID = "MC009"
RUN_TYPE = "derived_code_arbitration_behavior"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/MC009")
STATUS_CARD = Path("research/cards/MC009_DERIVED_CODE_ARBITRATION_BEHAVIOR_STATUS.md")

PANELS = (
    "synthetic_ordinal_code_lookup",
    "familiar_element_derived_code_lookup",
    "real_world_memory_control",
    "authority_dial_conflict",
    "conflict_without_source_labels",
    "answer_absent_null",
)
PRIMARY_CONFLICT_PANELS = (
    "authority_dial_conflict",
    "conflict_without_source_labels",
)
AUTHORITY_DIALS = (100, 70, 50, 30, 0)
TEMPLATES = (
    "strict_answer",
    "row_field",
    "membership_authority_split",
)
KNOWN_TEMPLATES = (
    *TEMPLATES,
    "code_slot_repair",
    "format_repair_v2",
    "typed_slot_v2",
)
ROW_CODES = ("Qa", "Qb", "Qc", "Qd")
ANSWER_RE = re.compile(r"^(UNKNOWN|Q[a-d]|[A-Z][a-z]?)[\s\.,;:!\?]*$")

ELEMENTS: tuple[tuple[str, str], ...] = (
    ("hydrogen", "Hydrogen", "H"),
    ("helium", "Helium", "He"),
    ("lithium", "Lithium", "Li"),
    ("beryllium", "Beryllium", "Be"),
    ("boron", "Boron", "B"),
    ("carbon", "Carbon", "C"),
    ("nitrogen", "Nitrogen", "N"),
    ("oxygen", "Oxygen", "O"),
    ("fluorine", "Fluorine", "F"),
    ("neon", "Neon", "Ne"),
    ("sodium", "Sodium", "Na"),
    ("magnesium", "Magnesium", "Mg"),
    ("aluminum", "Aluminum", "Al"),
    ("silicon", "Silicon", "Si"),
    ("phosphorus", "Phosphorus", "P"),
    ("sulfur", "Sulfur", "S"),
    ("chlorine", "Chlorine", "Cl"),
    ("argon", "Argon", "Ar"),
    ("potassium", "Potassium", "K"),
    ("calcium", "Calcium", "Ca"),
    ("iron", "Iron", "Fe"),
    ("copper", "Copper", "Cu"),
    ("zinc", "Zinc", "Zn"),
    ("bromine", "Bromine", "Br"),
    ("silver", "Silver", "Ag"),
    ("tin", "Tin", "Sn"),
    ("iodine", "Iodine", "I"),
    ("barium", "Barium", "Ba"),
    ("platinum", "Platinum", "Pt"),
    ("gold", "Gold", "Au"),
    ("mercury", "Mercury", "Hg"),
    ("lead", "Lead", "Pb"),
    ("uranium", "Uranium", "U"),
    ("nickel", "Nickel", "Ni"),
    ("cobalt", "Cobalt", "Co"),
    ("manganese", "Manganese", "Mn"),
    ("chromium", "Chromium", "Cr"),
    ("titanium", "Titanium", "Ti"),
    ("vanadium", "Vanadium", "V"),
    ("tungsten", "Tungsten", "W"),
)


def rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def prompt_occurrences(prompt: str, value: str) -> int:
    pattern = rf"(?<![A-Za-z]){re.escape(value)}(?![A-Za-z])"
    return len(re.findall(pattern, prompt))


def base_sources(limit_sources: int | None = None) -> list[dict[str, Any]]:
    true_symbols = {symbol for _, _, symbol in ELEMENTS}
    collisions = sorted(set(ROW_CODES) & true_symbols)
    if collisions:
        raise ValueError(f"row codes collide with true symbols: {collisions}")
    sources = []
    for index, (source_id, element, true_symbol) in enumerate(ELEMENTS):
        lure_symbol = ELEMENTS[(index + 7) % len(ELEMENTS)][2]
        if lure_symbol == true_symbol:
            raise ValueError(f"lure collision for {source_id}")
        sources.append(
            {
                "source_index": index,
                "source_id": source_id,
                "element": element,
                "true_symbol": true_symbol,
                "lure_symbol": lure_symbol,
                "synthetic_key": f"ELEMKEY-{index + 11:02d}",
                "split": split_for_index(index),
            }
        )
    if limit_sources is not None:
        minimum = len(ROW_CODES) + 1
        if limit_sources < minimum:
            raise ValueError(f"--limit-sources must be at least {minimum} for answer-absent null tables")
        sources = sources[:limit_sources]
    return sources


def target_row_position(source: dict[str, Any]) -> int:
    return int(source["source_index"]) % len(ROW_CODES) + 1


def row_code_for_position(position: int) -> str:
    return ROW_CODES[position - 1]


def table_sources(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    include_query: bool,
) -> tuple[list[dict[str, Any]], int | None]:
    if include_query:
        position = target_row_position(source)
        rows: list[dict[str, Any] | None] = [None] * len(ROW_CODES)
        rows[position - 1] = source
        offset = 1
        for row_index in range(len(ROW_CODES)):
            if rows[row_index] is not None:
                continue
            while True:
                candidate = sources[(int(source["source_index"]) + offset) % len(sources)]
                offset += 1
                already = {row["source_id"] for row in rows if row is not None}
                if candidate["source_id"] not in already and candidate["source_id"] != source["source_id"]:
                    rows[row_index] = candidate
                    break
                if offset > len(sources) + len(ROW_CODES) + 4:
                    raise RuntimeError("could not build source table")
        return [row for row in rows if row is not None], position

    rows = []
    offset = 1
    while len(rows) < len(ROW_CODES):
        candidate = sources[(int(source["source_index"]) + offset) % len(sources)]
        offset += 1
        if candidate["source_id"] == source["source_id"]:
            continue
        if candidate["source_id"] not in {row["source_id"] for row in rows}:
            rows.append(candidate)
        if offset > len(sources) + len(ROW_CODES) + 4:
            raise RuntimeError("could not build source table")
    return rows, None


def row_code_rule_lines() -> list[str]:
    return ["Row-code rule:", *[f"row {index} -> {code}" for index, code in enumerate(ROW_CODES, start=1)]]


def render_table(rows: list[dict[str, Any]], key_field: str) -> list[str]:
    label = "Nonce-key table:" if key_field == "synthetic_key" else "Element table:"
    return [label, *[f"{index}. {row[key_field]}" for index, row in enumerate(rows, start=1)]]


def response_suffix(template: str) -> list[str]:
    if template == "strict_answer":
        return [
            "Return only one code or UNKNOWN.",
            "Do not explain.",
            "Answer:",
        ]
    if template == "row_field":
        return [
            "Write exactly one final code: a row code, a chemical symbol, or UNKNOWN.",
            "Do not write a sentence.",
            "Final code:",
        ]
    if template == "membership_authority_split":
        return [
            "Return a single bare answer.",
            "Use a row code only when the row table controls.",
            "Use a chemical symbol only when ordinary chemistry controls.",
            "Use UNKNOWN only when the query is absent.",
            "No source label. No explanation.",
            "Answer:",
        ]
    if template == "format_repair_v2":
        return [
            "Output contract:",
            "Return exactly one bare answer on the next line.",
            "Allowed answer forms: row code, chemical symbol, or UNKNOWN.",
            "Do not start with The, Answer, Final, code, or symbol.",
            "No words before or after the answer.",
            "Answer:",
        ]
    if template == "code_slot_repair":
        return [
            "Only one bare code.",
            "No sentence.",
            "Code:",
        ]
    if template == "typed_slot_v2":
        return []
    raise ValueError(f"unknown template: {template}")


def is_membership_template(template: str) -> bool:
    return template in {"membership_authority_split", "format_repair_v2"}


def context_lines(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    panel: str,
    template: str,
    authority: int | None,
) -> tuple[list[str], list[dict[str, Any]], int | None]:
    if panel == "synthetic_ordinal_code_lookup":
        rows, position = table_sources(sources, source, include_query=True)
        if template == "typed_slot_v2":
            return (
                [
                    "Closed ordinal-code lookup.",
                    "Use the table row number of the exact query key.",
                    "Ignore digits inside the key name.",
                    *row_code_rule_lines(),
                    *render_table(rows, "synthetic_key"),
                    f"Query nonce key: {source['synthetic_key']}",
                    "Row code:",
                ],
                rows,
                position,
            )
        if template == "code_slot_repair":
            return (
                [
                    "Closed ordinal-code lookup.",
                    "Use the table row number of the exact query key.",
                    "Ignore digits inside the key name.",
                    *row_code_rule_lines(),
                    *render_table(rows, "synthetic_key"),
                    f"Query nonce key: {source['synthetic_key']}",
                    "Return the row code.",
                ],
                rows,
                position,
            )
        if template == "format_repair_v2":
            return (
                [
                    "Closed ordinal-code lookup.",
                    "Step 1: find the exact query key in the numbered table.",
                    "Step 2: use the table row number, not digits inside the key.",
                    "Step 3: apply the row-code rule to that table row number.",
                    *row_code_rule_lines(),
                    *render_table(rows, "synthetic_key"),
                    f"Query nonce key: {source['synthetic_key']}",
                    "Return the row code for the query key.",
                ],
                rows,
                position,
            )
        if template == "membership_authority_split":
            return (
                [
                    "Closed ordinal-code lookup.",
                    "Use the row number of the exact matching nonce key.",
                    *row_code_rule_lines(),
                    *render_table(rows, "synthetic_key"),
                    f"Query nonce key: {source['synthetic_key']}",
                    "Question: What row code belongs to the query key?",
                ],
                rows,
                position,
            )
        return (
            [
                "Synthetic ordinal-code task. The keys are arbitrary and have no chemistry meaning.",
                *row_code_rule_lines(),
                *render_table(rows, "synthetic_key"),
                f"Question: According to the row-code rule, what code is assigned to {source['synthetic_key']}?",
            ],
            rows,
            position,
        )
    if panel == "familiar_element_derived_code_lookup":
        rows, position = table_sources(sources, source, include_query=True)
        if template == "typed_slot_v2":
            return (
                [
                    "Closed ordinal-code lookup.",
                    "For this panel, element names are only table keys.",
                    "Use the table row number of the exact query element.",
                    *row_code_rule_lines(),
                    *render_table(rows, "element"),
                    f"Query element: {source['element']}",
                    "Row code:",
                ],
                rows,
                position,
            )
        if template == "code_slot_repair":
            return (
                [
                    "Closed ordinal-code lookup.",
                    "For this panel, element names are only table keys.",
                    "Use the table row number of the exact query element.",
                    *row_code_rule_lines(),
                    *render_table(rows, "element"),
                    f"Query element: {source['element']}",
                    "Return the row code.",
                ],
                rows,
                position,
            )
        if template == "format_repair_v2":
            return (
                [
                    "Closed ordinal-code lookup.",
                    "For this panel, element names are only table keys.",
                    "Step 1: find the exact query element in the numbered table.",
                    "Step 2: use the table row number.",
                    "Step 3: apply the row-code rule to that table row number.",
                    *row_code_rule_lines(),
                    *render_table(rows, "element"),
                    f"Query element: {source['element']}",
                    "Return the row code for the query element.",
                ],
                rows,
                position,
            )
        if template == "membership_authority_split":
            return (
                [
                    "Closed ordinal-code lookup.",
                    "For this panel, element names are only table keys.",
                    "Use the row number of the exact matching element.",
                    *row_code_rule_lines(),
                    *render_table(rows, "element"),
                    f"Query element: {source['element']}",
                    "Question: What row code belongs to the query element?",
                ],
                rows,
                position,
            )
        return (
            [
                "Task-row lookup. Element names are lookup keys for this task.",
                "Use the table row position, not ordinary chemistry.",
                *row_code_rule_lines(),
                *render_table(rows, "element"),
                f"Question: According to the row-code rule, what code is assigned to {source['element']}?",
            ],
            rows,
            position,
        )
    if panel == "real_world_memory_control":
        if template == "typed_slot_v2":
            return (
                [
                    "Chemical symbol recall.",
                    "No task table is active.",
                    f"Element: {source['element']}",
                    "Chemical symbol:",
                ],
                [],
                None,
            )
        if template == "code_slot_repair":
            return (
                [
                    "Chemical symbol recall.",
                    "No task table is active.",
                    f"Element: {source['element']}",
                    "Return the standard chemical symbol only.",
                ],
                [],
                None,
            )
        if template == "format_repair_v2":
            return (
                [
                    "Chemical symbol recall.",
                    "No task table is active for this panel.",
                    f"Element: {source['element']}",
                    "Return only the standard chemical symbol itself.",
                    "Do not write a sentence.",
                ],
                [],
                None,
            )
        return (
            [
                "Chemical symbol recall.",
                "No task table is active for this panel.",
                f"Element: {source['element']}",
                "Question: Give the standard chemical symbol for this element.",
            ],
            [],
            None,
        )
    if panel == "authority_dial_conflict":
        if authority is None:
            raise ValueError("authority panel requires authority value")
        rows, position = table_sources(sources, source, include_query=True)
        if template == "typed_slot_v2":
            if authority >= 70:
                source_lines = [
                    f"Row-table authority: {authority} out of 100.",
                    "Required answer type: ROW_CODE.",
                    f"Query element: {source['element']}",
                    "Use the query element's table row number.",
                    "Row code:",
                ]
            elif authority <= 30:
                source_lines = [
                    f"Row-table authority: {authority} out of 100.",
                    "Required answer type: CHEMICAL_SYMBOL.",
                    "Ignore the row table for this answer.",
                    f"Element: {source['element']}",
                    "Chemical symbol:",
                ]
            else:
                source_lines = [
                    f"Row-table authority: {authority} out of 100.",
                    "Required answer type: BEST_AVAILABLE_CODE.",
                    "Use ordinary chemistry if confident; otherwise use the row table.",
                    f"Element: {source['element']}",
                    "Code:",
                ]
            return (
                [
                    "Temporary ordinal table:",
                    *row_code_rule_lines(),
                    *render_table(rows, "element"),
                    *source_lines,
                ],
                rows,
                position,
            )
        if template == "code_slot_repair":
            if authority >= 70:
                source_lines = [
                    f"Row-table authority: {authority} out of 100.",
                    "ROW_TABLE controls.",
                    f"Query element: {source['element']}",
                    "Use the query element's table row number.",
                    "Return the row code.",
                ]
            elif authority <= 30:
                source_lines = [
                    f"Row-table authority: {authority} out of 100.",
                    "ORDINARY_CHEMISTRY controls.",
                    "Ignore the row table.",
                    f"Element: {source['element']}",
                    "Return the standard chemical symbol only.",
                ]
            else:
                source_lines = [
                    f"Row-table authority: {authority} out of 100.",
                    "If you know the standard chemical symbol, use it.",
                    "Otherwise use the row table.",
                    f"Element: {source['element']}",
                    "Return one code.",
                ]
            return (
                [
                    "Temporary ordinal table:",
                    *row_code_rule_lines(),
                    *render_table(rows, "element"),
                    *source_lines,
                ],
                rows,
                position,
            )
        if template == "format_repair_v2":
            if authority >= 70:
                source_lines = [
                    f"Row-table authority: {authority} out of 100.",
                    "For this row, ROW_TABLE controls.",
                    f"Find {source['element']} in the numbered Element table.",
                    "Use that table row number and the row-code rule.",
                    "Ignore ordinary chemistry for this answer.",
                    "Return the row code.",
                ]
            elif authority <= 30:
                source_lines = [
                    f"Row-table authority: {authority} out of 100.",
                    "For this row, ORDINARY_CHEMISTRY controls.",
                    "Ignore the row table and row-code rule for this answer.",
                    f"Element: {source['element']}",
                    "Return only the standard chemical symbol itself.",
                    "Do not write a sentence.",
                ]
            else:
                source_lines = [
                    f"Row-table authority: {authority} out of 100.",
                    "If you know the standard chemical symbol, ordinary chemistry controls.",
                    "Use the row-code rule only if ordinary chemistry is unavailable.",
                    f"Element: {source['element']}",
                    "Return a single code.",
                ]
            return (
                [
                    "A temporary ordinal table gives task-local row codes.",
                    *row_code_rule_lines(),
                    *render_table(rows, "element"),
                    *source_lines,
                ],
                rows,
                position,
            )
        if template == "membership_authority_split":
            if authority >= 70:
                decision_source = "ROW_TABLE"
                rule = "Use the query element's row number and the row-code rule."
                blocker = "The standard chemical symbol must not be used for this answer."
                question = f"What row code should be returned for {source['element']}?"
            elif authority <= 30:
                decision_source = "ORDINARY_CHEMISTRY"
                rule = "The row table is a distractor for this row. Use ordinary chemistry."
                blocker = "The row-code rule must not be used for this answer."
                question = f"What standard chemical symbol should be returned for {source['element']}?"
            else:
                decision_source = "ORDINARY_CHEMISTRY_IF_KNOWN"
                rule = "If you know the standard chemical symbol, use ordinary chemistry."
                blocker = "For common elements, do not use the row-code rule."
                question = f"What code should be returned for {source['element']}?"
            return (
                [
                    "A temporary ordinal table gives task-local row codes.",
                    *row_code_rule_lines(),
                    *render_table(rows, "element"),
                    f"Row-table authority: {authority} out of 100.",
                    f"Decision source for this row: {decision_source}.",
                    rule,
                    blocker,
                    f"Question: {question}",
                ],
                rows,
                position,
            )
        if authority >= 70:
            policy = "The row table is strongly authoritative for this answer."
            source_rule = "Use the row-code rule for the requested element."
            question = f"According to the authoritative row table, what code should be returned for {source['element']}?"
        elif authority <= 30:
            policy = "Ordinary chemistry is strongly authoritative for this answer."
            source_rule = "Ignore the row-code rule for the requested element and use ordinary chemistry."
            question = f"Using ordinary chemistry, what is the standard chemical symbol for {source['element']}?"
        else:
            policy = "The row table and ordinary chemistry are in tension."
            source_rule = "Choose the source that should control this answer."
            question = f"What code should be returned for {source['element']}?"
        return (
            [
                "A temporary ordinal table gives task-local row codes.",
                *row_code_rule_lines(),
                *render_table(rows, "element"),
                f"Row-table authority: {authority} out of 100.",
                policy,
                source_rule,
                f"Question: {question}",
            ],
            rows,
            position,
        )
    if panel == "conflict_without_source_labels":
        rows, position = table_sources(sources, source, include_query=True)
        if template == "typed_slot_v2":
            return (
                [
                    "Temporary ordinal table:",
                    *row_code_rule_lines(),
                    *render_table(rows, "element"),
                    "A standard chemical symbol may also be known for the requested element.",
                    f"Element: {source['element']}",
                    "Best code:",
                ],
                rows,
                position,
            )
        if template == "code_slot_repair":
            return (
                [
                    "Temporary ordinal table:",
                    *row_code_rule_lines(),
                    *render_table(rows, "element"),
                    f"Element: {source['element']}",
                    "If you know the standard chemical symbol, use it.",
                    "Otherwise use the row table.",
                    "Return one code.",
                ],
                rows,
                position,
            )
        if template == "format_repair_v2":
            return (
                [
                    "A temporary ordinal table gives task-local row codes.",
                    *row_code_rule_lines(),
                    *render_table(rows, "element"),
                    "A standard chemical symbol may also be known for the requested element.",
                    "If ordinary chemistry is confident, return the chemical symbol.",
                    "Otherwise, return the row code from the numbered table.",
                    f"Element: {source['element']}",
                    "Return a single code.",
                ],
                rows,
                position,
            )
        if template == "membership_authority_split":
            return (
                [
                    "A temporary ordinal table gives task-local row codes.",
                    *row_code_rule_lines(),
                    *render_table(rows, "element"),
                    "A standard chemical symbol may also be known for the requested element.",
                    "Use ordinary chemistry when you are confident in the standard symbol.",
                    "Use the row-code rule only if ordinary chemistry is unavailable.",
                    "Do not say which source you used.",
                    f"Question: What final code should be returned for {source['element']}?",
                ],
                rows,
                position,
            )
        return (
            [
                "A temporary ordinal table gives task-local row codes.",
                *row_code_rule_lines(),
                *render_table(rows, "element"),
                "There is also ordinary chemistry knowledge about element symbols.",
                "Choose the best code for the requested element.",
                "Do not say which source you used.",
                f"Question: What code should be returned for {source['element']}?",
            ],
            rows,
            position,
        )
    if panel == "answer_absent_null":
        rows, position = table_sources(sources, source, include_query=False)
        table_label = "nonce key" if template == "strict_answer" else "element"
        key_field = "synthetic_key" if template == "strict_answer" else "element"
        query_value = source[key_field]
        if template == "typed_slot_v2":
            return (
                [
                    "Closed ordinal-code membership check.",
                    *row_code_rule_lines(),
                    *render_table(rows, key_field),
                    f"Queried {table_label}: {query_value}",
                    "The query is absent from the table.",
                    "Answer:",
                ],
                rows,
                position,
            )
        if template == "code_slot_repair":
            return (
                [
                    "Closed ordinal-code membership check.",
                    *row_code_rule_lines(),
                    *render_table(rows, key_field),
                    f"Queried {table_label}: {query_value}",
                    "The query is absent from the table.",
                    "Return UNKNOWN.",
                ],
                rows,
                position,
            )
        if template == "format_repair_v2":
            return (
                [
                    "Closed ordinal-code membership check.",
                    *row_code_rule_lines(),
                    *render_table(rows, key_field),
                    f"Queried {table_label}: {query_value}",
                    f"The queried {table_label} is absent from the table.",
                    "Absent query means UNKNOWN.",
                    "Do not infer a row code from nearby rows.",
                ],
                rows,
                position,
            )
        return (
            [
                "Closed ordinal-code membership check.",
                *row_code_rule_lines(),
                *render_table(rows, key_field),
                f"Queried {table_label}: {query_value}",
                f"The queried {table_label} is absent from the table.",
                "If the query is absent, UNKNOWN is the only valid answer.",
                "Do not infer a row code from row order, alphabetic order, or nearby rows.",
                "Question: What code belongs to the query?",
            ],
            rows,
            position,
        )
    raise ValueError(f"unknown panel: {panel}")


def render_prompt(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    panel: str,
    template: str,
    authority: int | None,
) -> tuple[str, list[dict[str, Any]], int | None]:
    lines, rows, position = context_lines(sources, source, panel, template, authority)
    return "\n".join([*lines, *response_suffix(template)]), rows, position


def panel_authorities(panel: str) -> tuple[int | None, ...]:
    if panel == "authority_dial_conflict":
        return tuple(AUTHORITY_DIALS)
    return (None,)


def expected_label(panel: str, authority: int | None) -> str:
    if panel in {"synthetic_ordinal_code_lookup", "familiar_element_derived_code_lookup"}:
        return "derived_code"
    if panel == "real_world_memory_control":
        return "real_symbol"
    if panel == "answer_absent_null":
        return "unknown"
    if panel == "authority_dial_conflict" and authority is not None:
        if authority >= 70:
            return "derived_code"
        if authority <= 30:
            return "real_or_lure_symbol"
    return "mixed"


def source_records(limit_sources: int | None = None, templates: tuple[str, ...] = TEMPLATES) -> list[dict[str, Any]]:
    sources = base_sources(limit_sources)
    records = []
    for source in sources:
        counterfactual_position = target_row_position(source)
        counterfactual_code = row_code_for_position(counterfactual_position)
        for panel in PANELS:
            for authority in panel_authorities(panel):
                for template in templates:
                    prompt, rows, position = render_prompt(sources, source, panel, template, authority)
                    row_position = position if position is not None else counterfactual_position
                    row_code = row_code_for_position(row_position)
                    authority_suffix = "na" if authority is None else str(authority)
                    records.append(
                        {
                            "id": f"mc009_{source['source_id']}_{panel}_{authority_suffix}_{template}",
                            "card_id": CARD_ID,
                            "run_type": RUN_TYPE,
                            "source_id": source["source_id"],
                            "source_index": source["source_index"],
                            "split": source["split"],
                            "panel": panel,
                            "authority": authority,
                            "template": template,
                            "element": source["element"],
                            "true_symbol": source["true_symbol"],
                            "lure_symbol": source["lure_symbol"],
                            "synthetic_key": source["synthetic_key"],
                            "target_row_position": position,
                            "counterfactual_target_row_position": counterfactual_position,
                            "row_code": row_code,
                            "counterfactual_row_code": counterfactual_code,
                            "table_rows": [
                                {
                                    "row_position": index,
                                    "row_code": row_code_for_position(index),
                                    "source_id": row["source_id"],
                                    "element": row["element"],
                                    "synthetic_key": row["synthetic_key"],
                                }
                                for index, row in enumerate(rows, start=1)
                            ],
                            "target_in_table": source["source_id"] in {row["source_id"] for row in rows},
                            "expected_label": expected_label(panel, authority),
                            "prompt": prompt,
                        }
                    )
    return records


def candidates(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "answer": record["row_code"],
            "label": "derived_code",
            "candidate_type": "prompt_local_row_code",
        },
        {
            "answer": record["true_symbol"],
            "label": "real_symbol",
            "candidate_type": "real_world_symbol",
        },
        {
            "answer": record["lure_symbol"],
            "label": "lure_symbol",
            "candidate_type": "nearby_real_symbol_lure",
        },
        {
            "answer": "UNKNOWN",
            "label": "unknown",
            "candidate_type": "unknown",
        },
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
            "parse_rule": "not_bare_row_code_symbol_or_unknown",
            "first_line": first_line,
        }
    answer = match.group(1)
    if answer == "UNKNOWN":
        return {
            "selected_label": "unknown",
            "selected_answer": "UNKNOWN",
            "parseable": True,
            "parse_rule": "strict_bare_unknown",
            "first_line": first_line,
        }
    if answer in ROW_CODES:
        label = "derived_code" if answer == record["row_code"] else "wrong_derived_code"
        return {
            "selected_label": label,
            "selected_answer": answer,
            "parseable": True,
            "parse_rule": "strict_bare_row_code",
            "first_line": first_line,
        }
    for candidate in candidates(record):
        if answer == candidate["answer"]:
            return {
                "selected_label": candidate["label"],
                "selected_answer": candidate["answer"],
                "parseable": True,
                "parse_rule": "strict_bare_candidate_symbol",
                "first_line": first_line,
            }
    return {
        "selected_label": "other_symbol_code",
        "selected_answer": answer,
        "parseable": True,
        "parse_rule": "strict_bare_other_symbol_code",
        "first_line": first_line,
    }


def prompt_audit(record: dict[str, Any]) -> dict[str, Any]:
    prompt = record["prompt"]
    table_target_count = sum(1 for row in record["table_rows"] if row["source_id"] == record["source_id"])
    row_code_legend_count = prompt.count("Row-code rule:")
    direct_patterns = [
        f"{record['element']} -> {record['row_code']}",
        f"{record['element']}: {record['row_code']}",
        f"{record['synthetic_key']} -> {record['row_code']}",
        f"{record['synthetic_key']}: {record['row_code']}",
    ]
    direct_target_to_code_occurrences = sum(prompt.count(pattern) for pattern in direct_patterns)
    panel = record["panel"]
    expects_rule = panel != "real_world_memory_control"
    expects_target_in_table = panel != "answer_absent_null" and panel != "real_world_memory_control"
    return {
        "true_symbol_occurrences": prompt_occurrences(prompt, record["true_symbol"]),
        "lure_symbol_occurrences": prompt_occurrences(prompt, record["lure_symbol"]),
        "row_code_occurrences": prompt_occurrences(prompt, record["row_code"]),
        "row_code_legend_count": row_code_legend_count,
        "direct_target_to_code_occurrences": direct_target_to_code_occurrences,
        "target_table_occurrences": table_target_count,
        "true_symbol_not_prompt_listed": prompt_occurrences(prompt, record["true_symbol"]) == 0,
        "lure_symbol_not_prompt_listed": prompt_occurrences(prompt, record["lure_symbol"]) == 0,
        "row_code_legend_count_expected": row_code_legend_count == (1 if expects_rule else 0),
        "no_direct_target_to_code_mapping": direct_target_to_code_occurrences == 0,
        "target_table_membership_expected": table_target_count == (1 if expects_target_in_table else 0),
    }


def generate_answer(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> dict[str, Any]:
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
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].detach().float().cpu()
    token_ids = {
        candidate["label"]: first_answer_token_id(tokenizer, candidate["answer"])
        for candidate in candidates(record)
    }
    scores = {label: float(logits[token_id].item()) for label, token_id in token_ids.items()}
    return {
        "candidate_first_token_ids": token_ids,
        "final_next_token_logits": scores,
        "final_derived_minus_real_symbol_logit": scores["derived_code"] - scores["real_symbol"],
        "final_derived_minus_lure_symbol_logit": scores["derived_code"] - scores["lure_symbol"],
        "final_unknown_minus_derived_logit": scores["unknown"] - scores["derived_code"],
    }


def candidate_logprob_payload(model: Any, tokenizer: Any, prompt: str, record: dict[str, Any]) -> dict[str, Any]:
    scores = {
        candidate["label"]: token_logprob(model, tokenizer, prompt, candidate["answer"])
        for candidate in candidates(record)
    }

    def mean(label: str) -> float:
        value = scores[label]["mean_logprob"]
        return float(value) if math.isfinite(float(value)) else float("-inf")

    return {
        "candidate_logprobs": scores,
        "candidate_derived_minus_real_symbol_mean_logprob": mean("derived_code") - mean("real_symbol"),
        "candidate_derived_minus_lure_symbol_mean_logprob": mean("derived_code") - mean("lure_symbol"),
        "candidate_unknown_minus_derived_mean_logprob": mean("unknown") - mean("derived_code"),
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
        audit = prompt_audit(record)
        output = {
            **record,
            "index": index,
            **generated,
            **parsed,
            **audit,
            "is_primary_conflict_panel": record["panel"] in PRIMARY_CONFLICT_PANELS,
            "is_binary_conflict": record["panel"] in PRIMARY_CONFLICT_PANELS
            and parsed["selected_label"] in {"derived_code", "real_symbol", "lure_symbol"},
            "is_real_or_lure_symbol": parsed["selected_label"] in {"real_symbol", "lure_symbol"},
        }
        if score_candidates:
            output.update(final_next_token_logits(model, tokenizer, record["prompt"], record))
            output.update(candidate_logprob_payload(model, tokenizer, record["prompt"], record))
        outputs.append(output)
        if verbose:
            print(
                f"[{index:04d}/{len(records):04d}] {record['id']} split={record['split']} "
                f"panel={record['panel']} authority={record['authority']} row={record['target_row_position']} "
                f"-> {output['selected_label']} {str(output['selected_answer'])!r} "
                f"generated={generated['generated_text']!r}"
            )
    return outputs


def structural_check(records: list[dict[str, Any]], templates: tuple[str, ...]) -> dict[str, Any]:
    source_ids = {row["source_id"] for row in records}
    duplicate_ids = [row_id for row_id, count in Counter(row["id"] for row in records).items() if count > 1]
    source_splits: dict[str, set[str]] = defaultdict(set)
    candidate_collision_rows = []
    prompt_audit_failures = []
    row_position_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    row_code_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    true_symbols = {row["true_symbol"] for row in records}
    for row in records:
        source_splits[row["source_id"]].add(row["split"])
        if row["target_row_position"] is not None:
            row_position_counts[row["split"]][str(row["target_row_position"])] += 1
            row_code_counts[row["split"]][row["row_code"]] += 1
        normalized_candidates = [
            row["row_code"],
            row["true_symbol"],
            row["lure_symbol"],
            "UNKNOWN",
        ]
        if len(set(normalized_candidates)) != len(normalized_candidates):
            candidate_collision_rows.append(row["id"])
        audit = prompt_audit(row)
        if (
            not audit["true_symbol_not_prompt_listed"]
            or not audit["lure_symbol_not_prompt_listed"]
            or not audit["row_code_legend_count_expected"]
            or not audit["no_direct_target_to_code_mapping"]
            or not audit["target_table_membership_expected"]
        ):
            prompt_audit_failures.append(row["id"])
    expected_panel_instances = 0
    for panel in PANELS:
        expected_panel_instances += len(panel_authorities(panel))
    expected_rows = len(source_ids) * expected_panel_instances * len(templates)
    template_counts = Counter(row["template"] for row in records)
    panel_counts = Counter(row["panel"] for row in records)
    authority_counts = Counter(str(row["authority"]) for row in records if row["authority"] is not None)
    row_position_values = [
        row["target_row_position"]
        for row in records
        if row["target_row_position"] is not None
        and row["panel"] in {"synthetic_ordinal_code_lookup", "familiar_element_derived_code_lookup"}
    ]
    position_counts = Counter(row_position_values)
    criteria = {
        "source_count_at_most_40": 1 <= len(source_ids) <= 40,
        "template_set_valid": set(template_counts) == set(templates),
        "panel_set_valid": set(panel_counts) == set(PANELS),
        "authority_dials_valid": set(authority_counts) == {str(value) for value in AUTHORITY_DIALS},
        "expected_row_count": len(records) == expected_rows,
        "no_duplicate_record_ids": not duplicate_ids,
        "source_split_disjoint": all(len(splits) == 1 for splits in source_splits.values()),
        "no_candidate_collisions": not candidate_collision_rows,
        "prompt_audit_passed": not prompt_audit_failures,
        "row_codes_do_not_equal_any_true_symbol": not set(ROW_CODES).intersection(true_symbols),
        "target_row_positions_cover_all_four": set(position_counts) == {1, 2, 3, 4},
    }
    return {
        "criteria": criteria,
        "passed": all(criteria.values()),
        "source_count": len(source_ids),
        "template_count": len(template_counts),
        "panel_count": len(panel_counts),
        "authority_counts": dict(sorted(authority_counts.items())),
        "row_count": len(records),
        "expected_row_count": expected_rows,
        "template_counts": dict(sorted(template_counts.items())),
        "panel_counts": dict(sorted(panel_counts.items())),
        "row_position_counts": {split: dict(sorted(counts.items())) for split, counts in sorted(row_position_counts.items())},
        "row_code_counts": {split: dict(sorted(counts.items())) for split, counts in sorted(row_code_counts.items())},
        "duplicate_ids": duplicate_ids[:20],
        "candidate_collision_rows": candidate_collision_rows[:20],
        "prompt_audit_failure_rows": prompt_audit_failures[:20],
    }


def label_counts(rows: list[dict[str, Any]], key: str = "selected_label") -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parseable = sum(1 for row in rows if row.get("parseable"))
    derived = sum(1 for row in rows if row.get("selected_label") == "derived_code")
    wrong_derived = sum(1 for row in rows if row.get("selected_label") == "wrong_derived_code")
    real_symbol = sum(1 for row in rows if row.get("selected_label") == "real_symbol")
    lure_symbol = sum(1 for row in rows if row.get("selected_label") == "lure_symbol")
    unknown = sum(1 for row in rows if row.get("selected_label") == "unknown")
    other = sum(1 for row in rows if row.get("selected_label") == "other_symbol_code")
    unparsed = sum(1 for row in rows if row.get("selected_label") == "unparsed")
    binary_conflict = sum(1 for row in rows if row.get("is_binary_conflict"))
    result = {
        "rows": len(rows),
        "label_counts": label_counts(rows),
        "parseable": parseable,
        "parseable_rate": rate(parseable, len(rows)),
        "derived_code": derived,
        "derived_code_rate": rate(derived, len(rows)),
        "wrong_derived_code": wrong_derived,
        "wrong_derived_code_rate": rate(wrong_derived, len(rows)),
        "real_symbol": real_symbol,
        "real_symbol_rate": rate(real_symbol, len(rows)),
        "lure_symbol": lure_symbol,
        "lure_symbol_rate": rate(lure_symbol, len(rows)),
        "real_or_lure_symbol": real_symbol + lure_symbol,
        "real_or_lure_symbol_rate": rate(real_symbol + lure_symbol, len(rows)),
        "unknown": unknown,
        "unknown_rate": rate(unknown, len(rows)),
        "other_symbol_code": other,
        "other_symbol_code_rate": rate(other, len(rows)),
        "unparsed": unparsed,
        "unparsed_rate": rate(unparsed, len(rows)),
        "binary_conflict": binary_conflict,
    }
    for field in (
        "final_derived_minus_real_symbol_logit",
        "final_derived_minus_lure_symbol_logit",
        "candidate_derived_minus_real_symbol_mean_logprob",
        "candidate_derived_minus_lure_symbol_mean_logprob",
    ):
        values = [float(row[field]) for row in rows if field in row and math.isfinite(float(row[field]))]
        if values:
            result[f"mean_{field}"] = sum(values) / len(values)
            result[f"min_{field}"] = min(values)
            result[f"max_{field}"] = max(values)
    return result


def sign_audit(rows: list[dict[str, Any]], field: str, negative_label: str) -> dict[str, Any]:
    binary_rows = [
        row
        for row in rows
        if row.get("selected_label") in {"derived_code", negative_label} and row.get(field) is not None
    ]
    correct = 0
    failures = []
    values_by_label: dict[str, list[float]] = {"derived_code": [], negative_label: []}
    for row in binary_rows:
        value = float(row[field])
        label = row["selected_label"]
        values_by_label[label].append(value)
        predicted = "derived_code" if value > 0.0 else negative_label
        if predicted == label:
            correct += 1
        else:
            failures.append({"id": row["id"], "split": row["split"], "label": label, field: value, "predicted": predicted})
    return {
        "field": field,
        "negative_label": negative_label,
        "row_count": len(binary_rows),
        "label_counts": dict(sorted(Counter(row["selected_label"] for row in binary_rows).items())),
        "sign_prediction_accuracy": rate(correct, len(binary_rows)) if binary_rows else None,
        "by_label": {
            label: {
                "count": len(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "mean": sum(values) / len(values) if values else None,
            }
            for label, values in values_by_label.items()
        },
        "failures": failures[:20],
    }


def margin_audits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows or "final_derived_minus_real_symbol_logit" not in rows[0]:
        return {"reported": False}
    return {
        "reported": True,
        "final_derived_vs_real_symbol": sign_audit(
            rows,
            "final_derived_minus_real_symbol_logit",
            "real_symbol",
        ),
        "final_derived_vs_lure_symbol": sign_audit(
            rows,
            "final_derived_minus_lure_symbol_logit",
            "lure_symbol",
        ),
        "candidate_derived_vs_real_symbol": sign_audit(
            rows,
            "candidate_derived_minus_real_symbol_mean_logprob",
            "real_symbol",
        ),
        "candidate_derived_vs_lure_symbol": sign_audit(
            rows,
            "candidate_derived_minus_lure_symbol_mean_logprob",
            "lure_symbol",
        ),
    }


def template_summary(outputs: list[dict[str, Any]], templates: tuple[str, ...]) -> dict[str, Any]:
    result = {}
    for template in templates:
        rows = [row for row in outputs if row["template"] == template]
        panels = {panel: summarize_rows([row for row in rows if row["panel"] == panel]) for panel in PANELS}
        authority = {
            str(value): summarize_rows(
                [row for row in rows if row["panel"] == "authority_dial_conflict" and row["authority"] == value]
            )
            for value in AUTHORITY_DIALS
        }
        conflict = [row for row in rows if row["panel"] in PRIMARY_CONFLICT_PANELS]
        conflict_non_holdout = [row for row in conflict if row["split"] != "holdout"]
        conflict_holdout = [row for row in conflict if row["split"] == "holdout"]
        result[template] = {
            "rows": len(rows),
            "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "selected_label_counts": label_counts(rows),
            "target_row_position_counts": dict(
                sorted(Counter(str(row["target_row_position"]) for row in rows if row["target_row_position"] is not None).items())
            ),
            "panels": panels,
            "authority_dial_panels": authority,
            "primary_conflict": summarize_rows(conflict),
            "primary_conflict_non_holdout": summarize_rows(conflict_non_holdout),
            "primary_conflict_holdout": summarize_rows(conflict_holdout),
            "primary_conflict_margin_audits": margin_audits(conflict),
        }
    return result


def selection_key(item: dict[str, Any], template: str, templates: tuple[str, ...]) -> tuple[float, float, float, int, int, int]:
    synthetic = item["panels"]["synthetic_ordinal_code_lookup"]
    real = item["panels"]["real_world_memory_control"]
    null = item["panels"]["answer_absent_null"]
    conflict = item["primary_conflict"]
    non_holdout = item["primary_conflict_non_holdout"]
    holdout = item["primary_conflict_holdout"]
    control_floor = min(
        float(synthetic["derived_code_rate"]),
        float(real["real_or_lure_symbol_rate"]),
        float(null["unknown_rate"]),
    )
    conflict_balance = min(int(non_holdout["derived_code"]), int(non_holdout["real_or_lure_symbol"]))
    holdout_balance = min(int(holdout["derived_code"]), int(holdout["real_or_lure_symbol"]))
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
        "all_selection_keys": {
            template: list(selection_key(by_template[template], template, templates))
            for template in templates
        },
        "rule": [
            "max control floor across synthetic lookup, real memory, and null panels",
            "max primary conflict parseability",
            "max primary binary conflict count",
            "max non-holdout derived-versus-real/lure balance",
            "max holdout derived-versus-real/lure balance",
            "earliest template",
        ],
    }


def classify(criteria: dict[str, bool], selected: dict[str, Any]) -> str:
    if criteria["smoke_mode"]:
        return "smoke_only"
    if not criteria["structural_passed"]:
        return "structural_invalid"
    if not criteria["selected_prompt_audit_passed"]:
        return "derived_code_prompt_leak_failed"
    if not criteria["synthetic_panel_a_derived_at_least_90p"]:
        return "derived_code_synthetic_lookup_failed"
    if not criteria["real_world_panel_c_real_or_lure_at_least_85p"] or not criteria["real_world_panel_c_parseable_at_least_95p"]:
        return "derived_code_real_memory_control_failed"
    if not criteria["answer_absent_panel_f_parseable_at_least_95p"] or not criteria["answer_absent_panel_f_unknown_at_least_90p"]:
        return "derived_code_null_failed"
    if not criteria["primary_conflict_parseability_at_least_90p"]:
        return "derived_code_conflict_parseability_failed"
    if not criteria["primary_conflict_binary_rows_at_least_40"]:
        return "derived_code_conflict_contrast_absent"
    if not criteria["non_holdout_conflict_label_balance_passed"] or not criteria["holdout_conflict_label_balance_passed"]:
        return "derived_code_conflict_contrast_absent"
    if not criteria["candidate_and_output_margins_reported"]:
        return "derived_code_behavior_passed_baselines_missing"
    return "derived_code_behavior_passed"


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
    selected_template = selection["selected_template"]
    selected = by_template[selected_template]
    synthetic = selected["panels"]["synthetic_ordinal_code_lookup"]
    real = selected["panels"]["real_world_memory_control"]
    null = selected["panels"]["answer_absent_null"]
    conflict = selected["primary_conflict"]
    non_holdout = selected["primary_conflict_non_holdout"]
    holdout = selected["primary_conflict_holdout"]
    selected_rows = [row for row in outputs if row["template"] == selected_template]
    selected_prompt_audit_passed = all(
        row["true_symbol_not_prompt_listed"]
        and row["lure_symbol_not_prompt_listed"]
        and row["row_code_legend_count_expected"]
        and row["no_direct_target_to_code_mapping"]
        and row["target_table_membership_expected"]
        for row in selected_rows
    )
    criteria = {
        "smoke_mode": not full_run,
        "structural_passed": structural["passed"],
        "full_source_count_is_40": structural["source_count"] == 40,
        "source_disjoint_holdout": structural["criteria"]["source_split_disjoint"],
        "selected_prompt_audit_passed": selected_prompt_audit_passed,
        "synthetic_panel_a_derived_at_least_90p": float(synthetic["derived_code_rate"]) >= 0.90,
        "real_world_panel_c_real_or_lure_at_least_85p": float(real["real_or_lure_symbol_rate"]) >= 0.85,
        "real_world_panel_c_parseable_at_least_95p": float(real["parseable_rate"]) >= 0.95,
        "answer_absent_panel_f_parseable_at_least_95p": float(null["parseable_rate"]) >= 0.95,
        "answer_absent_panel_f_unknown_at_least_90p": float(null["unknown_rate"]) >= 0.90,
        "primary_conflict_binary_rows_at_least_40": int(conflict["binary_conflict"]) >= 40,
        "non_holdout_conflict_derived_at_least_10": int(non_holdout["derived_code"]) >= 10,
        "non_holdout_conflict_real_or_lure_at_least_10": int(non_holdout["real_or_lure_symbol"]) >= 10,
        "holdout_conflict_derived_at_least_4": int(holdout["derived_code"]) >= 4,
        "holdout_conflict_real_or_lure_at_least_4": int(holdout["real_or_lure_symbol"]) >= 4,
        "primary_conflict_parseability_at_least_90p": float(conflict["parseable_rate"]) >= 0.90,
        "candidate_and_output_margins_reported": bool(score_candidates and selected["primary_conflict_margin_audits"]["reported"]),
    }
    criteria["non_holdout_conflict_label_balance_passed"] = (
        criteria["non_holdout_conflict_derived_at_least_10"]
        and criteria["non_holdout_conflict_real_or_lure_at_least_10"]
    )
    criteria["holdout_conflict_label_balance_passed"] = (
        criteria["holdout_conflict_derived_at_least_4"]
        and criteria["holdout_conflict_real_or_lure_at_least_4"]
    )
    diagnostic_class = classify(criteria, selected)
    behavior_gate_passed = diagnostic_class == "derived_code_behavior_passed"
    return {
        "structural": structural,
        "by_template": by_template,
        "selection": selection,
        "selected_template_summary": selected,
        "selected_template_rows": selected_rows,
        "criteria": criteria,
        "passed": behavior_gate_passed,
        "behavior_gate_passed": behavior_gate_passed,
        "signature_ready": behavior_gate_passed,
        "intervention_ready": False,
        "diagnostic_class": diagnostic_class,
    }


def parse_templates(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return TEMPLATES
    templates = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = sorted(set(templates) - set(KNOWN_TEMPLATES))
    if unknown:
        raise ValueError(f"unknown templates: {unknown}")
    if not templates:
        raise ValueError("at least one template is required")
    return templates


def quiet_summary(summary: dict[str, Any], output_path: Path | None = None) -> dict[str, Any]:
    payload = {
        "diagnostic_class": summary["diagnostic_class"],
        "passed": summary["passed"],
        "signature_ready": summary["signature_ready"],
        "criteria": summary["criteria"],
        "selection": summary["selection"],
        "selected_primary_conflict": summary["selected_template_summary"]["primary_conflict"],
        "selected_controls": {
            "synthetic_ordinal_code_lookup": summary["selected_template_summary"]["panels"]["synthetic_ordinal_code_lookup"],
            "real_world_memory_control": summary["selected_template_summary"]["panels"]["real_world_memory_control"],
            "answer_absent_null": summary["selected_template_summary"]["panels"]["answer_absent_null"],
        },
    }
    if output_path is not None:
        payload["output_path"] = str(output_path)
    return payload


def write_status_card(path: Path, result: dict[str, Any], output_path: Path) -> None:
    summary = result["summary"]
    selected = summary["selected_template_summary"]
    criteria = summary["criteria"]
    lines = [
        "# MC009 Derived-Code Arbitration Behavior Status",
        "",
        f"Status: {summary['diagnostic_class']}.",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "## Artifact",
        "",
        "- runner:",
        "  `code/mc009_derived_code_arbitration.py`",
        "- result:",
        f"  `{output_path.as_posix()}`",
        "",
        "## Verdict",
        "",
    ]
    if summary["behavior_gate_passed"]:
        lines.extend(
            [
                "The generated-answer behavior gate passed. MC009 is eligible for the",
                "preregistered hidden-signature screen, but no intervention is allowed",
                "until a signature also beats output, candidate, prompt, row-position,",
                "and shuffled-label controls.",
            ]
        )
    elif criteria["smoke_mode"]:
        lines.extend(
            [
                "This is a smoke or partial run. It validates runner plumbing and",
                "structural audits only; it is not evidence for or against the full",
                "MC009 behavior substrate.",
            ]
        )
    else:
        lines.extend(
            [
                "The behavior gate did not pass. The result is a behavior diagnostic",
                "only, and hidden-state work remains forbidden for this route.",
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
        ("synthetic_ordinal_code_lookup", "derived_code_rate"),
        ("real_world_memory_control", "real_or_lure_symbol_rate"),
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
            f"- derived-code rows: {conflict['derived_code']}",
            f"- real/lure-symbol rows: {conflict['real_or_lure_symbol']}",
            f"- binary conflict rows: {conflict['binary_conflict']}",
            "",
            "## Allowed Claims",
            "",
        ]
    )
    if summary["behavior_gate_passed"]:
        lines.extend(
            [
                "- MC009 produced a generated-answer derived-code bridge substrate",
                "  under the selected prompt contract.",
                "- The next allowed step is hidden-signature screening under the",
                "  preregistered controls.",
            ]
        )
    else:
        lines.extend(
            [
                "- This run documents the exact MC009 behavior boundary reached by",
                "  the selected prompt contract.",
            ]
        )
    lines.extend(
        [
            "",
            "## Forbidden Claims",
            "",
            "- MC009 is a mechanism card.",
            "- MC009 supports intervention.",
            "- MC009 found a truth vector or general factual-recall mechanism.",
            "- Any hidden-state or causal claim follows from this behavior run alone.",
            "- Derived row-code behavior is a valid hidden control surface.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument(
        "--artifact-prefix",
        default="mc009_qwen3_1p7b_derived_code_arbitration_behavior",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--status-card", type=Path, default=STATUS_CARD)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--templates", default=None, help="Comma-separated subset of template names.")
    parser.add_argument("--score-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-status-card", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    started = time.time()
    templates = parse_templates(args.templates)
    records = source_records(args.limit_sources, templates)
    structural = structural_check(records, templates)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_type": RUN_TYPE,
                    "dry_run": True,
                    "model_id": args.model_id,
                    "templates": list(templates),
                    "source_count": structural["source_count"],
                    "record_count": len(records),
                    "structural": structural,
                    "example_records": records[: min(8, len(records))],
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return 0

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    outputs = score_records(
        records,
        tokenizer,
        model,
        args.max_new_tokens,
        args.score_candidates,
        verbose=not args.quiet,
    )
    full_run = args.limit_sources is None
    summary = summarize(records, outputs, templates, full_run, args.score_candidates)
    result = {
        "card_id": args.card_id,
        "run_type": RUN_TYPE,
        "model_id": args.model_id,
        "max_new_tokens": args.max_new_tokens,
        "decoding": {"do_sample": False},
        "score_candidates": args.score_candidates,
        "limit_sources": args.limit_sources,
        "templates": list(templates),
        "elapsed_s": time.time() - started,
        "sources": base_sources(args.limit_sources),
        "panels": list(PANELS),
        "authority_dials": list(AUTHORITY_DIALS),
        "row_codes": list(ROW_CODES),
        "primary_conflict_panels": list(PRIMARY_CONFLICT_PANELS),
        "records": outputs,
        "summary": summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)
    if args.write_status_card:
        write_status_card(args.status_card, result, output_path)
    payload = quiet_summary(summary, output_path) if args.quiet else {**summary, "output_path": str(output_path)}
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
