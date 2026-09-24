#!/usr/bin/env python
"""MC006 V23 final-margin sign-barrier audit.

V18-V22 repeatedly failed to produce strict final-output margin overlap. This
offline diagnostic tests whether that is just poor prompt search, or a property
of the greedy generated-answer interface: when the parsed binary answer starts
with the true capital token, the true-minus-override final margin is positive;
when it starts with the override token, the margin is non-positive.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from mc006_parametric_fact_override_v15_parser_normalized_signature import (
    first_answer_token_id,
    sha256_file,
)


CARD_ID = "MC006"
RUN_TYPE = "parametric_fact_override_v23_final_margin_sign_barrier"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/MC006")

DEFAULT_V14_ARTIFACT = (
    RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v14_parser_normalized_20260701T000538.json"
)
DEFAULT_V18_ARTIFACT = (
    RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v18_margin_matched_leadtime_20260701T022343.json"
)
DEFAULT_V19_ARTIFACT = (
    RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v19_overlapping_margin_table_20260701T023921.json"
)
DEFAULT_V20_ARTIFACT = (
    RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v20_strict_overlap_selection_audit_20260701T025043.json"
)
DEFAULT_V21_ARTIFACT = (
    RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v21_pair_matched_leadtime_20260701T030507.json"
)
DEFAULT_V22_ARTIFACT = (
    RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v22_source_path_leadtime_curve_20260701T032420.json"
)

PRIMARY_LABELS = ("true_answer", "override_answer")
EXPECTED = {
    "v14": ("parametric_fact_override_v14_parser_normalized", "parser_normalized_generated_substrate_passed"),
    "v18": ("parametric_fact_override_v18_margin_matched_leadtime", "global_margin_separation_blocks_matching"),
    "v19": ("parametric_fact_override_v19_overlapping_margin_table", "non_holdout_candidate_margin_overlap_failed"),
    "v20": ("parametric_fact_override_v20_strict_overlap_selection_audit", "strict_final_margin_overlap_absent"),
    "v21": ("parametric_fact_override_v21_pair_matched_leadtime", "approximate_pair_matching_failed_margin_baselines"),
    "v22": ("parametric_fact_override_v22_source_path_leadtime_curve", "source_path_final_margin_shadow"),
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_artifact(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    expected_run_type, expected_diagnostic = EXPECTED[name]
    summary = payload.get("summary", {})
    criteria = {
        "expected_run_type": payload.get("run_type") == expected_run_type,
        "expected_diagnostic_class": summary.get("diagnostic_class") == expected_diagnostic,
    }
    if name == "v14":
        criteria["source_passed"] = summary.get("passed") is True
    else:
        criteria["not_signature_ready"] = summary.get("signature_ready") is False
    return {"criteria": criteria, "passed": all(criteria.values())}


def label_for_row(row: dict[str, Any]) -> str | None:
    label = row.get("selected_label", row.get("label_name"))
    return label if label in PRIMARY_LABELS else None


def split_for_row(row: dict[str, Any]) -> str:
    return str(row.get("split", "unknown"))


def range_payload(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def margin_scope(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    binary_rows = [row for row in rows if label_for_row(row) and row.get(field) is not None]
    values_by_label: dict[str, list[float]] = {label: [] for label in PRIMARY_LABELS}
    sign_correct = 0
    sign_failures = []
    for row in binary_rows:
        label = label_for_row(row)
        assert label is not None
        margin = float(row[field])
        values_by_label[label].append(margin)
        predicted = "true_answer" if margin > 0.0 else "override_answer"
        if predicted == label:
            sign_correct += 1
        else:
            sign_failures.append(
                {
                    "id": row.get("id"),
                    "split": split_for_row(row),
                    "label": label,
                    "margin": margin,
                    "predicted": predicted,
                }
            )

    true_values = values_by_label["true_answer"]
    override_values = values_by_label["override_answer"]
    if true_values and override_values:
        overlap_low = max(min(true_values), min(override_values))
        overlap_high = min(max(true_values), max(override_values))
        overlap_exists = overlap_high >= overlap_low
        raw_gap = 0.0 if overlap_exists else overlap_low - overlap_high
    else:
        overlap_exists = False
        raw_gap = None

    closest = sorted(
        [
            {
                "id": row.get("id"),
                "split": split_for_row(row),
                "label": label_for_row(row),
                "margin": float(row[field]),
            }
            for row in binary_rows
        ],
        key=lambda item: abs(float(item["margin"])),
    )[:12]

    return {
        "row_count": len(binary_rows),
        "label_counts": dict(sorted(Counter(label_for_row(row) for row in binary_rows).items())),
        "by_label": {label: range_payload(values) for label, values in values_by_label.items()},
        "true_rows_positive": sum(1 for value in true_values if value > 0.0),
        "true_rows_nonpositive": sum(1 for value in true_values if value <= 0.0),
        "override_rows_nonpositive": sum(1 for value in override_values if value <= 0.0),
        "override_rows_positive": sum(1 for value in override_values if value > 0.0),
        "sign_prediction_accuracy": (sign_correct / len(binary_rows)) if binary_rows else None,
        "sign_failures": sign_failures,
        "raw_overlap_exists": overlap_exists,
        "raw_gap": raw_gap,
        "closest_to_zero": closest,
    }


def margin_audit(name: str, rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    by_split = {}
    for split in ["all", "non_holdout", "discovery", "calibration", "holdout"]:
        if split == "all":
            split_rows = rows
        elif split == "non_holdout":
            split_rows = [row for row in rows if split_for_row(row) != "holdout"]
        else:
            split_rows = [row for row in rows if split_for_row(row) == split]
        by_split[split] = margin_scope(split_rows, field)
    all_scope = by_split["all"]
    return {
        "name": name,
        "field": field,
        "by_split": by_split,
        "sign_barrier": bool(
            all_scope["row_count"] > 0
            and all_scope["sign_prediction_accuracy"] == 1.0
            and all_scope["true_rows_nonpositive"] == 0
            and all_scope["override_rows_positive"] == 0
            and not all_scope["raw_overlap_exists"]
        ),
    }


def selected_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list(payload.get("summary", {}).get("selected_template_rows", []))


def binary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if label_for_row(row)]


def token_alignment(
    name: str,
    rows: list[dict[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    checked = []
    failures = []
    for row in binary_rows(rows):
        generated_ids = row.get("generated_token_ids") or []
        if not generated_ids:
            failures.append({"id": row.get("id"), "reason": "missing_generated_token_ids"})
            continue
        label = label_for_row(row)
        assert label is not None
        selected_answer = row["true_capital"] if label == "true_answer" else row["override_capital"]
        selected_token_id = first_answer_token_id(tokenizer, selected_answer)
        generated_first_token_id = int(generated_ids[0])
        aligned = generated_first_token_id == selected_token_id
        item = {
            "id": row.get("id"),
            "split": split_for_row(row),
            "label": label,
            "selected_answer": selected_answer,
            "generated_first_token_id": generated_first_token_id,
            "selected_answer_first_token_id": selected_token_id,
            "generated_first_token": tokenizer.decode([generated_first_token_id]),
            "selected_answer_first_token": tokenizer.decode([selected_token_id]),
            "aligned": aligned,
        }
        checked.append(item)
        if not aligned:
            failures.append(item)
    return {
        "name": name,
        "checked_rows": len(checked),
        "aligned_rows": sum(1 for item in checked if item["aligned"]),
        "alignment_accuracy": (
            sum(1 for item in checked if item["aligned"]) / len(checked)
            if checked
            else None
        ),
        "failures": failures[:20],
    }


def classify(criteria: dict[str, bool]) -> str:
    if not criteria["source_artifacts_valid"]:
        return "source_artifact_invalid"
    if criteria["v18_final_sign_barrier"] and criteria["v19_final_sign_barrier"]:
        return "greedy_final_margin_sign_barrier"
    if criteria["v19_final_raw_overlap_exists"]:
        return "final_margin_overlap_possible"
    return "final_margin_overlap_absent_unexplained"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--v14-artifact", type=Path, default=DEFAULT_V14_ARTIFACT)
    parser.add_argument("--v18-artifact", type=Path, default=DEFAULT_V18_ARTIFACT)
    parser.add_argument("--v19-artifact", type=Path, default=DEFAULT_V19_ARTIFACT)
    parser.add_argument("--v20-artifact", type=Path, default=DEFAULT_V20_ARTIFACT)
    parser.add_argument("--v21-artifact", type=Path, default=DEFAULT_V21_ARTIFACT)
    parser.add_argument("--v22-artifact", type=Path, default=DEFAULT_V22_ARTIFACT)
    parser.add_argument(
        "--artifact-prefix",
        default="mc006_qwen3_1p7b_parametric_fact_override_v23_final_margin_sign_barrier",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.time()
    paths = {
        "v14": args.v14_artifact,
        "v18": args.v18_artifact,
        "v19": args.v19_artifact,
        "v20": args.v20_artifact,
        "v21": args.v21_artifact,
        "v22": args.v22_artifact,
    }
    payloads = {name: read_json(path) for name, path in paths.items()}
    source_validation = {
        name: validate_artifact(name, payload)
        for name, payload in payloads.items()
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )

    v18_rows = payloads["v18"]["row_scores"]
    v19_rows = payloads["v19"]["records"]
    v19_selected_rows = selected_rows(payloads["v19"])
    v14_selected_rows = selected_rows(payloads["v14"])

    margin_audits = {
        "v18_v14_selected_final_next_token_margin": margin_audit(
            "v18_v14_selected_final_next_token_margin",
            v18_rows,
            "final_next_token_output_margin_raw",
        ),
        "v19_all_binary_final_next_token_margin": margin_audit(
            "v19_all_binary_final_next_token_margin",
            v19_rows,
            "final_next_token_margin",
        ),
        "v19_selected_template_final_next_token_margin": margin_audit(
            "v19_selected_template_final_next_token_margin",
            v19_selected_rows,
            "final_next_token_margin",
        ),
        "v19_all_binary_candidate_score_margin": margin_audit(
            "v19_all_binary_candidate_score_margin",
            v19_rows,
            "candidate_score_margin",
        ),
    }

    token_alignments = {
        "v14_selected_template": token_alignment("v14_selected_template", v14_selected_rows, tokenizer),
        "v19_selected_template": token_alignment("v19_selected_template", v19_selected_rows, tokenizer),
        "v19_all_binary": token_alignment("v19_all_binary", v19_rows, tokenizer),
    }

    criteria = {
        "source_artifacts_valid": all(item["passed"] for item in source_validation.values()),
        "v18_final_sign_barrier": margin_audits[
            "v18_v14_selected_final_next_token_margin"
        ]["sign_barrier"],
        "v19_final_sign_barrier": margin_audits[
            "v19_all_binary_final_next_token_margin"
        ]["sign_barrier"],
        "v19_selected_final_sign_barrier": margin_audits[
            "v19_selected_template_final_next_token_margin"
        ]["sign_barrier"],
        "v19_final_raw_overlap_exists": margin_audits[
            "v19_all_binary_final_next_token_margin"
        ]["by_split"]["all"]["raw_overlap_exists"],
        "v19_candidate_raw_overlap_exists": margin_audits[
            "v19_all_binary_candidate_score_margin"
        ]["by_split"]["all"]["raw_overlap_exists"],
        "v14_selected_first_token_alignment_perfect": token_alignments[
            "v14_selected_template"
        ]["alignment_accuracy"]
        == 1.0,
        "v19_all_first_token_alignment_perfect": token_alignments[
            "v19_all_binary"
        ]["alignment_accuracy"]
        == 1.0,
    }
    diagnostic_class = classify(criteria)

    summary = {
        "model_id": args.model_id,
        "source_artifacts": {name: str(path) for name, path in paths.items()},
        "source_artifact_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "source_validation": source_validation,
        "primary_question": (
            "Does strict final-output margin overlap fail because generated "
            "binary labels are coupled to the first generated answer token?"
        ),
        "margin_audits": margin_audits,
        "token_alignments": token_alignments,
        "criteria": criteria,
        "diagnostic_supported": diagnostic_class == "greedy_final_margin_sign_barrier",
        "signature_ready": False,
        "intervention_ready": False,
        "passed": False,
        "diagnostic_class": diagnostic_class,
        "claim_boundary": (
            "V23 does not prove a knowledge mechanism. It diagnoses the output "
            "interface: strict final-margin overlap is not a good default gate "
            "for the current greedy binary generated-answer rows because the "
            "parsed label is coupled to the first generated answer token."
        ),
    }
    result = {
        "card_id": args.card_id,
        "run_type": RUN_TYPE,
        "model_id": args.model_id,
        "elapsed_s": time.time() - started,
        "summary": summary,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)

    compact = {
        "diagnostic_class": diagnostic_class,
        "diagnostic_supported": summary["diagnostic_supported"],
        "signature_ready": summary["signature_ready"],
        "intervention_ready": summary["intervention_ready"],
        "criteria": criteria,
        "v18_final_all": margin_audits[
            "v18_v14_selected_final_next_token_margin"
        ]["by_split"]["all"],
        "v19_final_all": margin_audits[
            "v19_all_binary_final_next_token_margin"
        ]["by_split"]["all"],
        "v19_candidate_all": margin_audits[
            "v19_all_binary_candidate_score_margin"
        ]["by_split"]["all"],
        "token_alignments": token_alignments,
        "output_path": str(output_path),
    }
    print(json.dumps(compact, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
