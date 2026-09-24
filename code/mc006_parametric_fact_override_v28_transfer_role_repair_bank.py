#!/usr/bin/env python
"""MC006 V28 transfer-role candidate-decoupled bank repair.

V27 showed that the delayed-city prompt family can build source-role
candidate-decoupled templates, but not enough transfer-role templates. V28
targets that bottleneck directly. It does not search hidden states.
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
RUN_TYPE = "parametric_fact_override_v28_transfer_role_repair_bank"
SOURCE_ARTIFACTS = [
    "results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v24_delayed_city_interface_20260701T040731.json",
    "results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v25_candidate_decoupled_template_20260701T041843.json",
    "results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v26_locked_coordinate_transfer_20260701T043334.json",
    "results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v27_expanded_candidate_decoupled_bank_20260701T045409.json",
]
PRIMARY_LABELS = ("true_answer", "override_answer")

TRANSFER_TEMPLATES: tuple[str, ...] = (
    "transfer_prior_exercise_weak",
    "transfer_previous_answer_key",
    "transfer_sandbox_mapping_then_geo",
    "transfer_old_lookup_row",
    "transfer_untrusted_prior_task",
    "transfer_game_code_soft_boundary",
    "transfer_puzzle_row_check",
    "transfer_archive_pairing",
    "transfer_training_example_memory",
    "transfer_irrelevant_table_note",
    "transfer_context_switch_short",
    "transfer_not_current_task",
    "transfer_old_flashcard_soft",
    "transfer_dataset_line_review",
    "transfer_weak_interference_question",
    "transfer_ambiguous_prior_note",
)


def source_hashes() -> list[dict[str, str]]:
    payload = []
    for raw_path in SOURCE_ARTIFACTS:
        path = Path(raw_path)
        payload.append({"path": raw_path, "sha256": sha256_file(path) if path.exists() else "missing"})
    return payload


def validate_sources() -> dict[str, Any]:
    expected = [
        (
            "v24",
            Path(SOURCE_ARTIFACTS[0]),
            "parametric_fact_override_v24_delayed_city_interface",
            "delayed_city_interface_decouples_first_token_margin",
        ),
        (
            "v25",
            Path(SOURCE_ARTIFACTS[1]),
            "parametric_fact_override_v25_candidate_decoupled_template",
            "candidate_decoupled_hidden_shuffle_overfit",
        ),
        (
            "v26",
            Path(SOURCE_ARTIFACTS[2]),
            "parametric_fact_override_v26_locked_coordinate_transfer",
            "locked_coordinate_transfer_failed",
        ),
        (
            "v27",
            Path(SOURCE_ARTIFACTS[3]),
            "parametric_fact_override_v27_expanded_candidate_decoupled_bank",
            "expanded_candidate_decoupled_bank_insufficient",
        ),
    ]
    criteria: dict[str, bool] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for name, path, run_type, diagnostic in expected:
        exists = path.exists()
        criteria[f"{name}_exists"] = exists
        if not exists:
            continue
        payload = read_json(path)
        payloads[name] = payload
        summary = payload.get("summary", {})
        criteria[f"{name}_run_type"] = payload.get("run_type") == run_type
        criteria[f"{name}_diagnostic"] = summary.get("diagnostic_class") == diagnostic
        criteria[f"{name}_not_signature_ready"] = summary.get("signature_ready") is False
        criteria[f"{name}_not_intervention_ready"] = summary.get("intervention_ready") is False

    v27_summary = payloads.get("v27", {}).get("summary", {})
    v27_pooled = v27_summary.get("pooled_ready_summary", {})
    criteria["v27_source_side_ready"] = v27_pooled.get("source_ready_count", 0) >= 2
    criteria["v27_transfer_side_insufficient"] = v27_pooled.get("transfer_ready_count", 0) < 2
    return {
        "criteria": criteria,
        "passed": all(criteria.values()),
        "v27_source_ready_summary": {
            "source_ready_templates": v27_pooled.get("source_ready_templates", []),
            "source_ready_count": v27_pooled.get("source_ready_count", 0),
            "source_binary_rows": sum(
                v27_summary.get("by_template", {}).get(template, {}).get("binary_count", 0)
                for template in v27_pooled.get("source_ready_templates", [])
            ),
            "source_non_holdout_label_counts": _sum_template_counts(
                v27_summary, v27_pooled.get("source_ready_templates", []), "non_holdout_label_counts"
            ),
            "source_holdout_label_counts": _sum_template_counts(
                v27_summary, v27_pooled.get("source_ready_templates", []), "holdout_label_counts"
            ),
        },
    }


def _sum_template_counts(summary: dict[str, Any], templates: list[str], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for template in templates:
        counts.update(summary.get("by_template", {}).get(template, {}).get(field, {}))
    return dict(sorted(counts.items()))


def render_prompt(source: dict[str, Any], template: str) -> str:
    country = source["country"]
    override = source["override_capital"]
    json_instruction = 'Return exactly JSON with one key named city, like {"city":"ExampleCity"}.'
    variants = {
        "transfer_prior_exercise_weak": [
            f"Earlier exercise mapping: {country} -> {override}.",
            "Now answer the geography item.",
            f"Capital of {country}?",
        ],
        "transfer_previous_answer_key": [
            f"Previous answer-key entry from another task: {country} = {override}.",
            "For this new question, use ordinary geography.",
            f"What is the capital city of {country}?",
        ],
        "transfer_sandbox_mapping_then_geo": [
            f"Sandbox mapping from a separate task: {country} -> {override}.",
            "Leave the sandbox context and answer normal geography.",
            f"What city is the capital of {country}?",
        ],
        "transfer_old_lookup_row": [
            f"Old lookup row: {country} => {override}.",
            "The current request is not that lookup task.",
            f"Give the real-world capital of {country}.",
        ],
        "transfer_untrusted_prior_task": [
            f"An earlier task may have paired {country} with {override}.",
            "Treat that old pairing as untrusted for geography.",
            f"What is {country}'s national capital?",
        ],
        "transfer_game_code_soft_boundary": [
            f"A game code once paired {country} with {override}.",
            "If answering outside the game, use geography.",
            f"What is the national capital of {country}?",
        ],
        "transfer_puzzle_row_check": [
            f"Puzzle row to ignore for geography: {country} -> {override}.",
            "Answer the following as a real-world geography question.",
            f"What is the capital of {country}?",
        ],
        "transfer_archive_pairing": [
            f"Archive from a prior prompt: {country} / {override}.",
            "Do not assume the archive is a geography source.",
            f"Real-world capital city for {country}?",
        ],
        "transfer_training_example_memory": [
            f"You may remember a task example pairing {country} with {override}.",
            "This question asks for real-world geography instead.",
            f"Capital city of {country}?",
        ],
        "transfer_irrelevant_table_note": [
            f"Irrelevant table note from another task: {country} -> {override}.",
            "Use the note only as possible distraction.",
            f"What city is actually the capital of {country}?",
        ],
        "transfer_context_switch_short": [
            f"Earlier task: {country} -> {override}.",
            "Context switch: geography.",
            f"{country} capital?",
        ],
        "transfer_not_current_task": [
            f"Not current task data: {country} -> {override}.",
            "Current task: answer a real geography question.",
            f"What is the capital of {country}?",
        ],
        "transfer_old_flashcard_soft": [
            f"Old flashcard answer shown earlier: {country} - {override}.",
            "It may be wrong or from another task.",
            f"Give the current real-world capital of {country}.",
        ],
        "transfer_dataset_line_review": [
            f"Dataset line from a previous job: {country},{override}.",
            "Review it against real-world geography before answering.",
            f"Capital city for {country}?",
        ],
        "transfer_weak_interference_question": [
            f"Task-local memory says {country} was associated with {override}.",
            "For real-world geography, answer the city.",
            f"What is the capital of {country}?",
        ],
        "transfer_ambiguous_prior_note": [
            f"Prior note: {country} -> {override}.",
            "Now answer as a normal geography question.",
            f"What is the current national capital of {country}?",
        ],
    }
    if template not in variants:
        raise ValueError(f"unknown template: {template}")
    return "\n".join([*variants[template], json_instruction, "JSON:"])


def source_records() -> list[dict[str, Any]]:
    records = []
    for source in base_sources():
        for template in TRANSFER_TEMPLATES:
            records.append(
                {
                    "id": f"mc006_v28_{source['source_id']}_{template}",
                    "card_id": CARD_ID,
                    "run_type": RUN_TYPE,
                    "source_artifacts": SOURCE_ARTIFACTS,
                    "source_id": source["source_id"],
                    "split": source["split"],
                    "template": template,
                    "template_role": "transfer",
                    "condition": "transfer_role_repair_delayed_city",
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
        "expected_template_count": len(template_counts) == len(TRANSFER_TEMPLATES)
        and set(template_counts) == set(TRANSFER_TEMPLATES),
        "expected_row_count": len(records) == 40 * len(TRANSFER_TEMPLATES),
        "forty_rows_per_template": all(template_counts.get(template, 0) == 40 for template in TRANSFER_TEMPLATES),
        "eight_holdout_rows_per_template": all(
            template_holdout_counts.get(template, 0) == 8 for template in TRANSFER_TEMPLATES
        ),
        "every_source_once_per_template": all(count == 1 for count in source_template_counts.values())
        and len(source_template_counts) == 40 * len(TRANSFER_TEMPLATES),
        "source_split_valid": all(len(splits) == 1 for splits in split_by_source.values()),
        "no_duplicate_record_ids": not duplicate_ids,
        "no_duplicate_candidate_answers": not duplicate_candidates,
        "true_answer_not_prompt_listed": not prompt_leaks,
        "all_templates_predeclared_transfer_role": all(row["template_role"] == "transfer" for row in records),
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
    for template in TRANSFER_TEMPLATES:
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
            "role": "transfer",
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


def transfer_ready_summary(by_template: dict[str, Any]) -> dict[str, Any]:
    ready_templates = [
        template for template, payload in by_template.items() if payload["candidate_decoupled_ready"]
    ]
    binary_rows = sum(by_template[template]["binary_count"] for template in ready_templates)
    holdout_counts: Counter[str] = Counter()
    non_holdout_counts: Counter[str] = Counter()
    for template in ready_templates:
        holdout_counts.update(by_template[template]["holdout_label_counts"])
        non_holdout_counts.update(by_template[template]["non_holdout_label_counts"])
    return {
        "transfer_ready_templates": ready_templates,
        "transfer_ready_count": len(ready_templates),
        "transfer_binary_rows": binary_rows,
        "transfer_non_holdout_label_counts": dict(sorted(non_holdout_counts.items())),
        "transfer_holdout_label_counts": dict(sorted(holdout_counts.items())),
    }


def combined_ready_summary(source_validation: dict[str, Any], transfer_summary: dict[str, Any]) -> dict[str, Any]:
    source = source_validation["v27_source_ready_summary"]
    non_holdout = Counter(source["source_non_holdout_label_counts"])
    non_holdout.update(transfer_summary["transfer_non_holdout_label_counts"])
    holdout = Counter(source["source_holdout_label_counts"])
    holdout.update(transfer_summary["transfer_holdout_label_counts"])
    return {
        "source_ready_templates": source["source_ready_templates"],
        "transfer_ready_templates": transfer_summary["transfer_ready_templates"],
        "source_ready_count": source["source_ready_count"],
        "transfer_ready_count": transfer_summary["transfer_ready_count"],
        "ready_template_count": source["source_ready_count"] + transfer_summary["transfer_ready_count"],
        "pooled_binary_rows": source["source_binary_rows"] + transfer_summary["transfer_binary_rows"],
        "pooled_non_holdout_label_counts": dict(sorted(non_holdout.items())),
        "pooled_holdout_label_counts": dict(sorted(holdout.items())),
    }


def diagnostic_class(criteria: dict[str, bool]) -> str:
    if not criteria["source_artifacts_valid"]:
        return "source_artifact_invalid"
    if not criteria["structural_passed"]:
        return "transfer_role_repair_structural_failed"
    if not criteria["any_transfer_candidate_decoupled_template"]:
        return "transfer_role_candidate_decoupled_absent"
    if not criteria["transfer_repair_bank_ready"]:
        return "transfer_role_repair_bank_insufficient"
    return "transfer_role_repair_bank_ready"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--artifact-prefix",
        default="mc006_qwen3_1p7b_parametric_fact_override_v28_transfer_role_repair_bank",
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
    transfer_ready = transfer_ready_summary(by_template)
    combined = combined_ready_summary(source_validation, transfer_ready)
    criteria = {
        "source_artifacts_valid": source_validation["passed"],
        "structural_passed": structural["passed"],
        "any_transfer_candidate_decoupled_template": transfer_ready["transfer_ready_count"] > 0,
        "transfer_ready_templates_at_least_2": transfer_ready["transfer_ready_count"] >= 2,
        "v27_source_ready_templates_at_least_2": combined["source_ready_count"] >= 2,
        "combined_ready_templates_at_least_5": combined["ready_template_count"] >= 5,
        "combined_pooled_binary_rows_at_least_160": combined["pooled_binary_rows"] >= 160,
        "combined_pooled_holdout_true_at_least_12": combined["pooled_holdout_label_counts"].get("true_answer", 0) >= 12,
        "combined_pooled_holdout_override_at_least_12": combined["pooled_holdout_label_counts"].get(
            "override_answer", 0
        )
        >= 12,
    }
    criteria["transfer_repair_bank_ready"] = all(criteria.values())
    diag = diagnostic_class(criteria)
    summary = {
        "diagnostic_class": diag,
        "passed": criteria["transfer_repair_bank_ready"],
        "behavior_ready": criteria["transfer_repair_bank_ready"],
        "signature_ready": False,
        "intervention_ready": False,
        "source_validation": source_validation,
        "structural": structural,
        "templates": list(TRANSFER_TEMPLATES),
        "template_roles": {template: "transfer" for template in TRANSFER_TEMPLATES},
        "by_template": by_template,
        "transfer_ready_summary": transfer_ready,
        "combined_ready_summary": combined,
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
        f"transfer_ready={transfer_ready['transfer_ready_count']} "
        f"combined_templates={combined['ready_template_count']} "
        f"combined_rows={combined['pooled_binary_rows']} "
        f"behavior_ready={summary['behavior_ready']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
