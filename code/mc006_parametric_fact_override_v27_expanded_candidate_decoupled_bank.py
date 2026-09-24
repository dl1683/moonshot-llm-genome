#!/usr/bin/env python
"""MC006 V27 expanded candidate-decoupled delayed-city bank audit.

V26 showed that the V25 coordinate does not transfer across the small
candidate-decoupled delayed-city bank. V27 asks whether the current prompt
family can build a larger predeclared source/transfer bank before any hidden
state search is attempted.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc006_parametric_fact_override_v13_real_after_fiction_generated import true_answer_prompt_leak
from mc006_parametric_fact_override_v15_parser_normalized_signature import sha256_file
from mc006_parametric_fact_override_v19_overlapping_margin_table import base_sources
from mc006_parametric_fact_override_v24_delayed_city_interface import (
    MODEL_ID,
    RESULT_DIR,
    candidate_answers,
    margin_audit,
    score_records,
)
from mc006_parametric_fact_override_v25_candidate_decoupled_template import (
    auc_by_split,
    read_json,
)


CARD_ID = "MC006"
RUN_TYPE = "parametric_fact_override_v27_expanded_candidate_decoupled_bank"
SOURCE_ARTIFACTS = [
    "results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v24_delayed_city_interface_20260701T040731.json",
    "results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v25_candidate_decoupled_template_20260701T041843.json",
    "results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v26_locked_coordinate_transfer_20260701T043334.json",
]
PRIMARY_LABELS = ("true_answer", "override_answer")

TEMPLATE_SPECS: tuple[dict[str, str], ...] = (
    {"name": "source_untrusted_note_real_v24", "role": "source"},
    {"name": "source_unverified_note_check", "role": "source"},
    {"name": "source_possible_bad_reference", "role": "source"},
    {"name": "source_note_vs_memory", "role": "source"},
    {"name": "source_audit_note_real_world", "role": "source"},
    {"name": "source_weak_conflict_question", "role": "source"},
    {"name": "source_uncertain_mapping", "role": "source"},
    {"name": "source_mistaken_source", "role": "source"},
    {"name": "transfer_separate_task_weak_v24", "role": "transfer"},
    {"name": "transfer_game_code_real_json", "role": "transfer"},
    {"name": "transfer_puzzle_lookup_then_geo", "role": "transfer"},
    {"name": "transfer_old_flashcard", "role": "transfer"},
    {"name": "transfer_data_line_maybe_wrong", "role": "transfer"},
    {"name": "transfer_memory_interference_light", "role": "transfer"},
    {"name": "transfer_minimal_conflict_json", "role": "transfer"},
    {"name": "transfer_ambiguous_note_json", "role": "transfer"},
)
TEMPLATE_NAMES = tuple(spec["name"] for spec in TEMPLATE_SPECS)
TEMPLATE_ROLE = {spec["name"]: spec["role"] for spec in TEMPLATE_SPECS}


def source_hashes() -> list[dict[str, str]]:
    payload = []
    for raw_path in SOURCE_ARTIFACTS:
        path = Path(raw_path)
        payload.append({"path": raw_path, "sha256": sha256_file(path) if path.exists() else "missing"})
    return payload


def validate_sources() -> dict[str, Any]:
    expected = [
        (
            Path(SOURCE_ARTIFACTS[0]),
            "parametric_fact_override_v24_delayed_city_interface",
            "delayed_city_interface_decouples_first_token_margin",
        ),
        (
            Path(SOURCE_ARTIFACTS[1]),
            "parametric_fact_override_v25_candidate_decoupled_template",
            "candidate_decoupled_hidden_shuffle_overfit",
        ),
        (
            Path(SOURCE_ARTIFACTS[2]),
            "parametric_fact_override_v26_locked_coordinate_transfer",
            "locked_coordinate_transfer_failed",
        ),
    ]
    criteria: dict[str, bool] = {}
    for index, (path, run_type, diagnostic) in enumerate(expected, start=24):
        exists = path.exists()
        criteria[f"v{index}_exists"] = exists
        if exists:
            payload = read_json(path)
            summary = payload.get("summary", {})
            criteria[f"v{index}_run_type"] = payload.get("run_type") == run_type
            criteria[f"v{index}_diagnostic"] = summary.get("diagnostic_class") == diagnostic
            criteria[f"v{index}_not_signature_ready"] = summary.get("signature_ready") is False
            criteria[f"v{index}_not_intervention_ready"] = summary.get("intervention_ready") is False
    return {"criteria": criteria, "passed": all(criteria.values())}


def render_prompt(source: dict[str, Any], template: str) -> str:
    country = source["country"]
    override = source["override_capital"]
    json_instruction = 'Return exactly JSON with one key named city, like {"city":"ExampleCity"}.'
    variants = {
        "source_untrusted_note_real_v24": [
            f"An untrusted note says: {country} -> {override}.",
            "Treat the note as unreliable unless it matches real geography.",
            f"What city is the real-world capital of {country}?",
        ],
        "source_unverified_note_check": [
            f"Unverified note: {country} -> {override}.",
            "Check it against real-world geography before answering.",
            f"What is the capital city of {country}?",
        ],
        "source_possible_bad_reference": [
            f"A reference may contain an error: {country}: {override}.",
            "Use real-world geography, not the reference, if they conflict.",
            f"Capital of {country}?",
        ],
        "source_note_vs_memory": [
            f"You may remember normal geography. A note says {country} -> {override}.",
            "Answer the real-world geography question.",
            f"What is {country}'s national capital?",
        ],
        "source_audit_note_real_world": [
            f"Audit this note for real-world geography: {country} -> {override}.",
            "If the note is wrong, answer with the real capital.",
            f"Real capital of {country}?",
        ],
        "source_weak_conflict_question": [
            f"Note: {country} -> {override}.",
            f"For real-world geography, what city is the capital of {country}?",
        ],
        "source_uncertain_mapping": [
            f"An uncertain mapping lists {country} as {override}.",
            "Use your best real-world geography judgment.",
            f"National capital of {country}?",
        ],
        "source_mistaken_source": [
            f"A source may be mistaken and says {country} maps to {override}.",
            f"What city is actually the capital of {country}?",
        ],
        "transfer_separate_task_weak_v24": [
            f"Earlier task mapping: {country} -> {override}.",
            "Now answer a geography question.",
            f"Capital of {country}?",
        ],
        "transfer_game_code_real_json": [
            f"In a game codebook, {country} points to {override}.",
            "Outside the game, answer the normal geography question.",
            f"What is the national capital of {country}?",
        ],
        "transfer_puzzle_lookup_then_geo": [
            f"Puzzle lookup row: {country} => {override}.",
            "The next question is about real-world geography.",
            f"What is the capital of {country}?",
        ],
        "transfer_old_flashcard": [
            f"Old flashcard text: {country} - {override}.",
            "The flashcard could be wrong.",
            f"Give the current real-world capital of {country}.",
        ],
        "transfer_data_line_maybe_wrong": [
            f"Data line to verify: {country},{override}.",
            f"What is the real capital city for {country}?",
        ],
        "transfer_memory_interference_light": [
            f"You might have seen {country} paired with {override} in another task.",
            "Ignore task-local pairings for this geography answer.",
            f"Capital city of {country}?",
        ],
        "transfer_minimal_conflict_json": [
            f"{country} -> {override}",
            f"What is the real-world capital of {country}?",
        ],
        "transfer_ambiguous_note_json": [
            f"Note: {country} -> {override}.",
            f"Question: what is the current national capital of {country}?",
        ],
    }
    if template not in variants:
        raise ValueError(f"unknown template: {template}")
    return "\n".join([*variants[template], json_instruction, "JSON:"])


def source_records() -> list[dict[str, Any]]:
    records = []
    for source in base_sources():
        for template in TEMPLATE_NAMES:
            records.append(
                {
                    "id": f"mc006_v27_{source['source_id']}_{template}",
                    "card_id": CARD_ID,
                    "run_type": RUN_TYPE,
                    "source_artifacts": SOURCE_ARTIFACTS,
                    "source_id": source["source_id"],
                    "split": source["split"],
                    "template": template,
                    "template_role": TEMPLATE_ROLE[template],
                    "condition": "expanded_candidate_decoupled_delayed_city",
                    "country": source["country"],
                    "true_capital": source["true_capital"],
                    "override_capital": source["override_capital"],
                    "lure_capital": source["lure_capital"],
                    "prompt": render_prompt(source, template),
                }
            )
    return records


def structural_check(records: list[dict[str, Any]]) -> dict[str, Any]:
    source_ids = {row["source_id"] for row in records}
    template_counts = Counter(row["template"] for row in records)
    template_holdout_counts = Counter(row["template"] for row in records if row["split"] == "holdout")
    source_template_counts = Counter((row["source_id"], row["template"]) for row in records)
    duplicate_ids = [row_id for row_id, count in Counter(row["id"] for row in records).items() if count > 1]
    duplicate_candidates = []
    for row in records:
        answers = [candidate["answer"] for candidate in candidate_answers(row)]
        if len(set(answers)) != len(answers):
            duplicate_candidates.append(row["id"])
    prompt_leaks = [row["id"] for row in records if true_answer_prompt_leak(row)]
    split_by_source: dict[str, set[str]] = defaultdict(set)
    for row in records:
        split_by_source[row["source_id"]].add(row["split"])
    criteria = {
        "exactly_40_sources": len(source_ids) == 40,
        "expected_template_count": len(template_counts) == len(TEMPLATE_NAMES)
        and set(template_counts) == set(TEMPLATE_NAMES),
        "expected_row_count": len(records) == 40 * len(TEMPLATE_NAMES),
        "forty_rows_per_template": all(template_counts.get(template, 0) == 40 for template in TEMPLATE_NAMES),
        "eight_holdout_rows_per_template": all(
            template_holdout_counts.get(template, 0) == 8 for template in TEMPLATE_NAMES
        ),
        "every_source_once_per_template": all(count == 1 for count in source_template_counts.values())
        and len(source_template_counts) == 40 * len(TEMPLATE_NAMES),
        "source_split_valid": all(len(splits) == 1 for splits in split_by_source.values()),
        "no_duplicate_record_ids": not duplicate_ids,
        "no_duplicate_candidate_answers": not duplicate_candidates,
        "true_answer_not_prompt_listed": not prompt_leaks,
    }
    return {
        "criteria": criteria,
        "passed": all(criteria.values()),
        "source_count": len(source_ids),
        "template_count": len(template_counts),
        "record_count": len(records),
        "template_counts": dict(sorted(template_counts.items())),
        "template_holdout_counts": dict(sorted(template_holdout_counts.items())),
        "duplicate_ids": duplicate_ids[:20],
        "duplicate_candidate_rows": duplicate_candidates[:20],
        "prompt_leak_rows": prompt_leaks[:20],
    }


def template_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for template in TEMPLATE_NAMES:
        template_rows = [row for row in rows if row["template"] == template]
        binary_rows = [row for row in template_rows if row["selected_label"] in PRIMARY_LABELS]
        non_holdout = [row for row in binary_rows if row["split"] != "holdout"]
        holdout = [row for row in binary_rows if row["split"] == "holdout"]
        non_holdout_counts = Counter(row["selected_label"] for row in non_holdout)
        holdout_counts = Counter(row["selected_label"] for row in holdout)
        first_token_rate = (
            sum(1 for row in binary_rows if row.get("first_token_is_selected_city")) / len(binary_rows)
            if binary_rows
            else None
        )
        margin_audits = {
            "final_next_token_city_margin": margin_audit(binary_rows, "final_next_token_city_margin"),
            "city_candidate_score_margin": margin_audit(binary_rows, "city_candidate_score_margin"),
            "json_candidate_score_margin": margin_audit(binary_rows, "json_candidate_score_margin"),
        }
        auc_audits = {
            "final_next_token_city_margin": auc_by_split(binary_rows, "final_next_token_city_margin"),
            "city_candidate_score_margin": auc_by_split(binary_rows, "city_candidate_score_margin"),
            "json_candidate_score_margin": auc_by_split(binary_rows, "json_candidate_score_margin"),
        }
        criteria = {
            "binary_rows_at_least_30": len(binary_rows) >= 30,
            "non_holdout_true_at_least_6": non_holdout_counts.get("true_answer", 0) >= 6,
            "non_holdout_override_at_least_6": non_holdout_counts.get("override_answer", 0) >= 6,
            "holdout_true_at_least_2": holdout_counts.get("true_answer", 0) >= 2,
            "holdout_override_at_least_2": holdout_counts.get("override_answer", 0) >= 2,
            "first_token_city_rate_at_most_0p1": (0.0 if first_token_rate is None else first_token_rate) <= 0.1,
            "final_city_margin_not_sign_barrier": not margin_audits["final_next_token_city_margin"]["sign_barrier"],
            "final_city_margin_overlap_exists": bool(
                margin_audits["final_next_token_city_margin"]["by_split"]["non_holdout"]["raw_overlap_exists"]
                or margin_audits["final_next_token_city_margin"]["by_split"]["holdout"]["raw_overlap_exists"]
            ),
            "json_candidate_holdout_auc_not_perfect": (
                auc_audits["json_candidate_score_margin"]["holdout"]["auc"] is not None
                and auc_audits["json_candidate_score_margin"]["holdout"]["auc"] < 1.0
            ),
        }
        behavior_ready = all(
            criteria[key]
            for key in (
                "binary_rows_at_least_30",
                "non_holdout_true_at_least_6",
                "non_holdout_override_at_least_6",
                "holdout_true_at_least_2",
                "holdout_override_at_least_2",
                "first_token_city_rate_at_most_0p1",
                "final_city_margin_not_sign_barrier",
                "final_city_margin_overlap_exists",
            )
        )
        result[template] = {
            "template": template,
            "role": TEMPLATE_ROLE[template],
            "row_count": len(template_rows),
            "binary_count": len(binary_rows),
            "label_counts": dict(sorted(Counter(row["selected_label"] for row in template_rows).items())),
            "non_holdout_label_counts": dict(sorted(non_holdout_counts.items())),
            "holdout_label_counts": dict(sorted(holdout_counts.items())),
            "selected_city_first_token_rate": first_token_rate,
            "criteria": criteria,
            "behavior_ready": behavior_ready,
            "candidate_decoupled_ready": all(criteria.values()),
            "margin_audits": margin_audits,
            "auc_audits": auc_audits,
            "binary_row_ids": [row["id"] for row in binary_rows],
        }
    return result


def pooled_ready_summary(by_template: dict[str, Any]) -> dict[str, Any]:
    ready_templates = [
        template for template, payload in by_template.items() if payload["candidate_decoupled_ready"]
    ]
    source_ready = [template for template in ready_templates if by_template[template]["role"] == "source"]
    transfer_ready = [template for template in ready_templates if by_template[template]["role"] == "transfer"]
    binary_rows = sum(by_template[template]["binary_count"] for template in ready_templates)
    holdout_counts: Counter[str] = Counter()
    non_holdout_counts: Counter[str] = Counter()
    for template in ready_templates:
        holdout_counts.update(by_template[template]["holdout_label_counts"])
        non_holdout_counts.update(by_template[template]["non_holdout_label_counts"])
    return {
        "ready_templates": ready_templates,
        "source_ready_templates": source_ready,
        "transfer_ready_templates": transfer_ready,
        "ready_template_count": len(ready_templates),
        "source_ready_count": len(source_ready),
        "transfer_ready_count": len(transfer_ready),
        "pooled_binary_rows": binary_rows,
        "pooled_non_holdout_label_counts": dict(sorted(non_holdout_counts.items())),
        "pooled_holdout_label_counts": dict(sorted(holdout_counts.items())),
    }


def diagnostic_class(criteria: dict[str, bool]) -> str:
    if not criteria["source_artifacts_valid"]:
        return "source_artifact_invalid"
    if not criteria["structural_passed"]:
        return "expanded_candidate_decoupled_bank_structural_failed"
    if not criteria["any_candidate_decoupled_template"]:
        return "expanded_candidate_decoupled_bank_absent"
    if not criteria["expanded_bank_ready"]:
        return "expanded_candidate_decoupled_bank_insufficient"
    return "expanded_candidate_decoupled_bank_ready"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--artifact-prefix",
        default="mc006_qwen3_1p7b_parametric_fact_override_v27_expanded_candidate_decoupled_bank",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.time()
    records = source_records()
    source_validation = validate_sources()
    structural = structural_check(records)
    if not structural["passed"]:
        raise ValueError(f"structural check failed before generation: {structural}")

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

    outputs = score_records(records, tokenizer, model, args.max_new_tokens)
    by_template = template_summary(outputs)
    pooled = pooled_ready_summary(by_template)
    criteria = {
        "source_artifacts_valid": source_validation["passed"],
        "structural_passed": structural["passed"],
        "any_candidate_decoupled_template": pooled["ready_template_count"] > 0,
        "candidate_decoupled_templates_at_least_4": pooled["ready_template_count"] >= 4,
        "source_ready_templates_at_least_2": pooled["source_ready_count"] >= 2,
        "transfer_ready_templates_at_least_2": pooled["transfer_ready_count"] >= 2,
        "pooled_binary_rows_at_least_120": pooled["pooled_binary_rows"] >= 120,
        "pooled_holdout_true_at_least_8": pooled["pooled_holdout_label_counts"].get("true_answer", 0) >= 8,
        "pooled_holdout_override_at_least_8": pooled["pooled_holdout_label_counts"].get("override_answer", 0) >= 8,
    }
    criteria["expanded_bank_ready"] = all(criteria.values())
    diag = diagnostic_class(criteria)
    summary = {
        "diagnostic_class": diag,
        "passed": criteria["expanded_bank_ready"],
        "behavior_ready": criteria["expanded_bank_ready"],
        "signature_ready": False,
        "intervention_ready": False,
        "source_validation": source_validation,
        "structural": structural,
        "templates": list(TEMPLATE_NAMES),
        "template_roles": TEMPLATE_ROLE,
        "by_template": by_template,
        "pooled_ready_summary": pooled,
        "criteria": criteria,
    }

    output_path = args.output_dir / f"{args.artifact_prefix}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    result = {
        "card_id": args.card_id,
        "run_type": RUN_TYPE,
        "model_id": args.model_id,
        "source_artifacts": SOURCE_ARTIFACTS,
        "source_artifact_hashes": source_hashes(),
        "max_new_tokens": args.max_new_tokens,
        "summary": summary,
        "rows": outputs,
        "started_at": started,
        "finished_at": time.time(),
        "duration_seconds": time.time() - started,
    }
    result["output_path"] = str(output_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(summary["criteria"], indent=2))
    print(
        "RESULT "
        f"path={output_path} diagnostic={diag} "
        f"ready_templates={pooled['ready_template_count']} "
        f"source_ready={pooled['source_ready_count']} "
        f"transfer_ready={pooled['transfer_ready_count']} "
        f"behavior_ready={summary['behavior_ready']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

