#!/usr/bin/env python
"""MC007 semi-synthetic familiar-entity lookup behavior gate."""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc006_parametric_fact_override_v2_repair import SOURCES, split_for_index


CARD_ID = "MC007"
RUN_TYPE = "semi_synthetic_familiar_entity_lookup_behavior"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/MC007")


PANELS = (
    "synthetic_control_lookup",
    "familiar_entity_lookup",
    "familiar_entity_conflict",
    "answer_absent_null",
)

TEMPLATES = (
    "plain_table",
    "strict_task_city",
    "not_geography",
    "lookup_only",
)

ARTIFICIAL_VALUES = (
    "Doveton",
    "Larkspur",
    "Norhaven",
    "Bramley",
    "Viremont",
    "Calford",
    "Eldwick",
    "Mariton",
    "Fenwick",
    "Oakmere",
    "Riverton",
    "Ashvale",
    "Belgrove",
    "Corland",
    "Dunwick",
    "Fairholt",
    "Glenford",
    "Harborin",
    "Ivydale",
    "Junemont",
    "Kingsley",
    "Lorvale",
    "Mereford",
    "Northley",
    "Orchard",
    "Pineholt",
    "Quinford",
    "Rosewick",
    "Stoneby",
    "Trenford",
    "Umberley",
    "Valeport",
    "Westmere",
    "Yardley",
    "Zephyrn",
    "Alderin",
    "Brookley",
    "Cairnton",
    "Dalehurst",
    "Emberton",
)


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", stripped.strip().lower())


def base_sources(limit_sources: int | None = None) -> list[dict[str, Any]]:
    sources = []
    for index, (source_id, country, true_capital, override_capital, lure_capital) in enumerate(SOURCES):
        artificial_value = ARTIFICIAL_VALUES[index]
        forbidden = {
            normalize_text(true_capital),
            normalize_text(override_capital),
            normalize_text(lure_capital),
        }
        if normalize_text(artificial_value) in forbidden:
            raise ValueError(f"artificial value collision for {source_id}: {artificial_value}")
        sources.append(
            {
                "source_index": index,
                "source_id": source_id,
                "country": country,
                "true_capital": true_capital,
                "override_capital": override_capital,
                "lure_capital": lure_capital,
                "artificial_value": artificial_value,
                "synthetic_key": f"KEY-{index + 11:02d}-{source_id[:3].upper()}",
                "split": split_for_index(index),
            }
        )
    if limit_sources is not None:
        if limit_sources < 1:
            raise ValueError("--limit-sources must be positive")
        sources = sources[:limit_sources]
    return sources


def table_sources(sources: list[dict[str, Any]], source: dict[str, Any], include_query: bool) -> list[dict[str, Any]]:
    if include_query:
        selected = [source]
        offset = 1
    else:
        selected = []
        offset = 1
    while len(selected) < 4:
        candidate = sources[(int(source["source_index"]) + offset) % len(sources)]
        if candidate["source_id"] != source["source_id"]:
            selected.append(candidate)
        offset += 1
        if offset > len(sources) + 4:
            raise RuntimeError("could not build table sources")
    return selected[:4]


def render_table(rows: list[dict[str, Any]], key_field: str) -> list[str]:
    label = "Code key" if key_field == "synthetic_key" else "Country"
    return [f"{label} -> task city"] + [
        f"{row[key_field]} -> {row['artificial_value']}"
        for row in rows
    ]


def render_prompt(
    sources: list[dict[str, Any]],
    source: dict[str, Any],
    panel: str,
    template: str,
) -> str:
    if panel == "synthetic_control_lookup":
        key_field = "synthetic_key"
        rows = table_sources(sources, source, include_query=True)
        query = source["synthetic_key"]
        question = f"According to the table, what is the task city for {query}?"
    elif panel in {"familiar_entity_lookup", "familiar_entity_conflict"}:
        key_field = "country"
        rows = table_sources(sources, source, include_query=True)
        query = source["country"]
        question = f"According to the table, what is the task city for {query}?"
    elif panel == "answer_absent_null":
        key_field = "country"
        rows = table_sources(sources, source, include_query=False)
        query = source["country"]
        question = f"According to the table, what is the task city for {query}?"
    else:
        raise ValueError(f"unknown panel: {panel}")

    table = render_table(rows, key_field)
    if template == "plain_table":
        lines = [
            "Use the mapping table below for this task.",
            "",
            *table,
            "",
            f"Question: {question}",
            "Answer with only the city.",
            "Answer:",
        ]
    elif template == "strict_task_city":
        lines = [
            "This is a task-city lookup. The table is the only authority.",
            "",
            *table,
            "",
            f"Question: {question}",
            "Return exactly one city from the table, or UNKNOWN if the key is absent.",
            "Answer:",
        ]
    elif template == "not_geography":
        lines = [
            "This is not a geography question. Familiar names are just lookup keys.",
            "Use only the artificial task-city table.",
            "",
            *table,
            "",
            f"Question: {question}",
            "Output only the task city. If the key is missing, output UNKNOWN.",
            "Answer:",
        ]
    elif template == "lookup_only":
        lines = [
            "Lookup task:",
            *table,
            f"Query: {query}",
            "Output:",
        ]
    else:
        raise ValueError(f"unknown template: {template}")

    if panel == "familiar_entity_conflict":
        lines.insert(0, "Do not answer with real-world capitals.")
    if panel == "answer_absent_null":
        lines.insert(0, "If the queried key is absent from the table, answer UNKNOWN.")
    return "\n".join(lines)


def source_records(limit_sources: int | None = None, templates: tuple[str, ...] = TEMPLATES) -> list[dict[str, Any]]:
    sources = base_sources(limit_sources)
    records = []
    for source in sources:
        for panel in PANELS:
            for template in templates:
                expected_label = "unknown" if panel == "answer_absent_null" else "artificial_value"
                records.append(
                    {
                        "id": f"mc007_{source['source_id']}_{panel}_{template}",
                        "card_id": CARD_ID,
                        "run_type": RUN_TYPE,
                        "source_id": source["source_id"],
                        "split": source["split"],
                        "panel": panel,
                        "template": template,
                        "country": source["country"],
                        "synthetic_key": source["synthetic_key"],
                        "true_capital": source["true_capital"],
                        "override_capital": source["override_capital"],
                        "lure_capital": source["lure_capital"],
                        "artificial_value": source["artificial_value"],
                        "expected_label": expected_label,
                        "prompt": render_prompt(sources, source, panel, template),
                    }
                )
    return records


def candidates(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "answer": record["artificial_value"],
            "label": "artificial_value",
            "candidate_type": "prompt_local_value",
        },
        {
            "answer": record["true_capital"],
            "label": "real_prior",
            "candidate_type": "real_world_prior",
        },
        {
            "answer": record["lure_capital"],
            "label": "lure_value",
            "candidate_type": "nearby_real_lure",
        },
        {"answer": "UNKNOWN", "label": "unknown", "candidate_type": "unknown"},
    ]


def strict_parse(record: dict[str, Any], generated_text: str) -> dict[str, Any]:
    stripped = generated_text.strip().lstrip("`'\" ")
    first_line = stripped.splitlines()[0] if stripped else ""
    normalized = normalize_text(first_line)
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
        return {
            "selected_label": matches[0]["label"],
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
    pattern = rf"\b{re.escape(value)}\b"
    return len(re.findall(pattern, prompt))


def prompt_audit(record: dict[str, Any]) -> dict[str, Any]:
    artificial_occurrences = prompt_occurrences(record["prompt"], record["artificial_value"])
    true_occurrences = prompt_occurrences(record["prompt"], record["true_capital"])
    if record["panel"] == "answer_absent_null":
        artificial_expected = 0
    else:
        artificial_expected = 1
    return {
        "artificial_value_occurrences": artificial_occurrences,
        "true_capital_occurrences": true_occurrences,
        "artificial_value_prompt_count_expected": artificial_occurrences == artificial_expected,
        "true_capital_not_prompt_listed": true_occurrences == 0,
    }


def next_token_candidate_logits(model: Any, tokenizer: Any, prompt: str, record: dict[str, Any]) -> dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].detach().float().cpu()
    scores = {}
    token_ids = {}
    for candidate in candidates(record):
        encoded = tokenizer(" " + candidate["answer"], add_special_tokens=False)["input_ids"]
        if not encoded:
            encoded = tokenizer(candidate["answer"], add_special_tokens=False)["input_ids"]
        first_id = int(encoded[0])
        token_ids[candidate["label"]] = first_id
        scores[candidate["label"]] = float(logits[first_id].item())
    return {
        "candidate_first_token_ids": token_ids,
        "candidate_first_token_logits": scores,
        "artificial_minus_real_prior_logit": scores["artificial_value"] - scores["real_prior"],
        "artificial_minus_lure_logit": scores["artificial_value"] - scores["lure_value"],
        "unknown_minus_artificial_logit": scores["unknown"] - scores["artificial_value"],
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
            "is_primary_panel": record["panel"] in {"familiar_entity_lookup", "familiar_entity_conflict"},
            "is_primary_binary": record["panel"] in {"familiar_entity_lookup", "familiar_entity_conflict"}
            and parsed["selected_label"] in {"artificial_value", "real_prior", "lure_value"},
        }
        if score_candidates:
            output.update(next_token_candidate_logits(model, tokenizer, record["prompt"], record))
        outputs.append(output)
        print(
            f"[{index:03d}/{len(records):03d}] {record['id']} split={record['split']} "
            f"panel={record['panel']} -> {output['selected_label']} "
            f"{str(output['selected_answer'])!r} generated={generated['generated_text']!r}"
        )
    return outputs


def structural_check(records: list[dict[str, Any]], templates: tuple[str, ...]) -> dict[str, Any]:
    source_ids = {row["source_id"] for row in records}
    duplicate_ids = [row_id for row_id, count in Counter(row["id"] for row in records).items() if count > 1]
    source_splits: dict[str, set[str]] = {}
    collision_rows = []
    prompt_audit_failures = []
    for row in records:
        source_splits.setdefault(row["source_id"], set()).add(row["split"])
        normalized_candidates = [
            normalize_text(row["artificial_value"]),
            normalize_text(row["true_capital"]),
            normalize_text(row["override_capital"]),
            normalize_text(row["lure_capital"]),
            normalize_text("UNKNOWN"),
        ]
        if len(set(normalized_candidates)) != len(normalized_candidates):
            collision_rows.append(row["id"])
        audit = prompt_audit(row)
        if not audit["artificial_value_prompt_count_expected"] or not audit["true_capital_not_prompt_listed"]:
            prompt_audit_failures.append(row["id"])
    expected_rows = len(source_ids) * len(PANELS) * len(templates)
    template_counts = Counter(row["template"] for row in records)
    panel_counts = Counter(row["panel"] for row in records)
    criteria = {
        "source_count_at_most_40": 1 <= len(source_ids) <= 40,
        "template_set_valid": set(template_counts) == set(templates),
        "panel_set_valid": set(panel_counts) == set(PANELS),
        "expected_row_count": len(records) == expected_rows,
        "no_duplicate_record_ids": not duplicate_ids,
        "source_split_disjoint": all(len(splits) == 1 for splits in source_splits.values()),
        "no_candidate_collisions": not collision_rows,
        "prompt_audit_passed": not prompt_audit_failures,
    }
    return {
        "criteria": criteria,
        "passed": all(criteria.values()),
        "source_count": len(source_ids),
        "template_count": len(template_counts),
        "panel_count": len(panel_counts),
        "row_count": len(records),
        "expected_row_count": expected_rows,
        "template_counts": dict(sorted(template_counts.items())),
        "panel_counts": dict(sorted(panel_counts.items())),
        "duplicate_ids": duplicate_ids,
        "candidate_collision_rows": collision_rows,
        "prompt_audit_failure_rows": prompt_audit_failures,
    }


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["selected_label"] for row in rows).items()))


def rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parseable = sum(1 for row in rows if row.get("parseable"))
    artificial = sum(1 for row in rows if row.get("selected_label") == "artificial_value")
    real_prior = sum(1 for row in rows if row.get("selected_label") == "real_prior")
    lure = sum(1 for row in rows if row.get("selected_label") == "lure_value")
    unknown = sum(1 for row in rows if row.get("selected_label") == "unknown")
    primary_binary = sum(1 for row in rows if row.get("is_primary_binary"))
    result = {
        "rows": len(rows),
        "label_counts": label_counts(rows),
        "parseable": parseable,
        "parseable_rate": rate(parseable, len(rows)),
        "artificial_value": artificial,
        "artificial_value_rate": rate(artificial, len(rows)),
        "real_prior": real_prior,
        "real_prior_rate": rate(real_prior, len(rows)),
        "lure_value": lure,
        "lure_value_rate": rate(lure, len(rows)),
        "unknown": unknown,
        "unknown_rate": rate(unknown, len(rows)),
        "primary_binary": primary_binary,
    }
    if rows and "artificial_minus_real_prior_logit" in rows[0]:
        result["mean_artificial_minus_real_prior_logit"] = sum(
            float(row["artificial_minus_real_prior_logit"]) for row in rows
        ) / len(rows)
        result["mean_artificial_minus_lure_logit"] = sum(
            float(row["artificial_minus_lure_logit"]) for row in rows
        ) / len(rows)
        result["mean_unknown_minus_artificial_logit"] = sum(
            float(row["unknown_minus_artificial_logit"]) for row in rows
        ) / len(rows)
    return result


def template_summary(outputs: list[dict[str, Any]], templates: tuple[str, ...]) -> dict[str, Any]:
    result = {}
    for template in templates:
        rows = [row for row in outputs if row["template"] == template]
        panels = {panel: summarize_rows([row for row in rows if row["panel"] == panel]) for panel in PANELS}
        primary = [
            row
            for row in rows
            if row["panel"] in {"familiar_entity_lookup", "familiar_entity_conflict"}
        ]
        primary_non_holdout = [row for row in primary if row["split"] != "holdout"]
        primary_holdout = [row for row in primary if row["split"] == "holdout"]
        result[template] = {
            "rows": len(rows),
            "split_row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "selected_label_counts": label_counts(rows),
            "panels": panels,
            "primary": summarize_rows(primary),
            "primary_non_holdout": summarize_rows(primary_non_holdout),
            "primary_holdout": summarize_rows(primary_holdout),
        }
    return result


def selection_key(summary: dict[str, Any], template: str, templates: tuple[str, ...]) -> tuple[float, float, int, int]:
    item = summary[template]
    panel_a_rate = float(item["panels"]["synthetic_control_lookup"]["artificial_value_rate"])
    primary_parseable = float(item["primary"]["parseable_rate"])
    primary_artificial = int(item["primary"]["artificial_value"])
    template_index = templates.index(template)
    return (panel_a_rate, primary_parseable, primary_artificial, -template_index)


def select_template(summary: dict[str, Any], templates: tuple[str, ...]) -> dict[str, Any]:
    selected = max(templates, key=lambda template: selection_key(summary, template, templates))
    return {
        "selected_template": selected,
        "selection_key": list(selection_key(summary, selected, templates)),
        "all_selection_keys": {
            template: list(selection_key(summary, template, templates))
            for template in templates
        },
        "rule": [
            "max Panel A artificial-value adherence",
            "max primary Panel B/C parseability",
            "max primary artificial-value adherence",
            "earliest template",
        ],
    }


def classify(criteria: dict[str, bool], selected: dict[str, Any]) -> str:
    if criteria["smoke_mode"]:
        return "smoke_only"
    if not criteria["structural_passed"]:
        return "structural_invalid"
    if not criteria["panel_a_artificial_adherence_at_least_90p"]:
        return "synthetic_control_failed"
    if not criteria["primary_parseability_at_least_90p"]:
        return "behavior_substrate_failed"
    if not criteria["primary_binary_rows_at_least_40"]:
        return "behavior_substrate_failed"
    if not criteria["selected_prompt_audit_passed"]:
        return "prompt_leak_failed"
    if criteria["contrast_balance_passed"]:
        return "behavior_contrast_substrate_passed"
    if int(selected["primary"]["real_prior"]) + int(selected["primary"]["lure_value"]) == 0:
        return "source_value_behavior_passed_contrast_absent"
    if int(selected["primary"]["artificial_value"]) < int(selected["primary"]["real_prior"]) + int(
        selected["primary"]["lure_value"]
    ):
        return "semantic_prior_dominant"
    return "behavior_substrate_failed"


def summarize(
    records: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    templates: tuple[str, ...],
    full_run: bool,
) -> dict[str, Any]:
    structural = structural_check(records, templates)
    by_template = template_summary(outputs, templates)
    selection = select_template(by_template, templates)
    selected_template = selection["selected_template"]
    selected = by_template[selected_template]
    primary = selected["primary"]
    non_holdout = selected["primary_non_holdout"]
    holdout = selected["primary_holdout"]
    prior_or_lure = int(primary["real_prior"]) + int(primary["lure_value"])
    non_holdout_prior_or_lure = int(non_holdout["real_prior"]) + int(non_holdout["lure_value"])
    holdout_prior_or_lure = int(holdout["real_prior"]) + int(holdout["lure_value"])
    contrast_present = prior_or_lure > 0 and int(primary["artificial_value"]) > 0
    contrast_balance_passed = (
        contrast_present
        and int(non_holdout["artificial_value"]) >= 10
        and non_holdout_prior_or_lure >= 10
        and int(holdout["artificial_value"]) >= 4
        and holdout_prior_or_lure >= 4
    )
    selected_rows = [row for row in outputs if row["template"] == selected_template]
    criteria = {
        "smoke_mode": not full_run,
        "structural_passed": structural["passed"],
        "full_source_count_is_40": structural["source_count"] == 40,
        "primary_binary_rows_at_least_40": int(primary["primary_binary"]) >= 40,
        "primary_parseability_at_least_90p": float(primary["parseable_rate"]) >= 0.90,
        "selected_prompt_audit_passed": all(
            row["artificial_value_prompt_count_expected"] and row["true_capital_not_prompt_listed"]
            for row in selected_rows
        ),
        "panel_a_artificial_adherence_at_least_90p": float(
            selected["panels"]["synthetic_control_lookup"]["artificial_value_rate"]
        )
        >= 0.90,
        "contrast_present": contrast_present,
        "contrast_balance_passed": contrast_balance_passed,
        "holdout_source_disjoint": structural["criteria"]["source_split_disjoint"],
    }
    diagnostic_class = classify(criteria, selected)
    return {
        "structural": structural,
        "by_template": by_template,
        "selection": selection,
        "selected_template_summary": selected,
        "selected_template_rows": selected_rows,
        "criteria": criteria,
        "passed": diagnostic_class
        in {
            "behavior_contrast_substrate_passed",
            "source_value_behavior_passed_contrast_absent",
        },
        "signature_ready": diagnostic_class == "behavior_contrast_substrate_passed",
        "diagnostic_class": diagnostic_class,
    }


def parse_templates(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return TEMPLATES
    templates = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = sorted(set(templates) - set(TEMPLATES))
    if unknown:
        raise ValueError(f"unknown templates: {unknown}")
    if not templates:
        raise ValueError("at least one template is required")
    return templates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument(
        "--artifact-prefix",
        default="mc007_qwen3_1p7b_semi_synthetic_familiar_entity_lookup_behavior",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--templates", default=None, help="Comma-separated subset of template names.")
    parser.add_argument("--score-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
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
                    "example_records": records[: min(4, len(records))],
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

    outputs = score_records(records, tokenizer, model, args.max_new_tokens, args.score_candidates)
    full_run = args.limit_sources is None and set(templates) == set(TEMPLATES)
    summary = summarize(records, outputs, templates, full_run)
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
        "records": outputs,
        "summary": summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)
    print(json.dumps({**summary, "output_path": str(output_path)}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
