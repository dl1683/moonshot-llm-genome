#!/usr/bin/env python
"""MC006 V25 candidate-score-decoupled template audit.

V24 fixed the direct first-token city-margin barrier but selected a template
whose JSON-completion candidate-score margin still reached 1.000 holdout AUC.
V25 asks whether the V24 row bank contains a behavior-ready template where that
full-completion candidate baseline is not perfect, and if so whether a hidden
monitor survives output/candidate controls and shuffled-label selection.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc003_delayed_copy_signature import percentile
from mc006_parametric_fact_override_v15_parser_normalized_signature import (
    sha256_file,
)
from mc006_parametric_fact_override_v24_delayed_city_interface import (
    MODEL_ID,
    RESULT_DIR,
    TEMPLATES,
    collect_features_and_position_margins,
    margin_audit,
    scalar_baseline,
    score_hidden_candidates,
    split_masks,
)


CARD_ID = "MC006"
RUN_TYPE = "parametric_fact_override_v25_candidate_decoupled_template"
DEFAULT_V24_ARTIFACT = (
    RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v24_delayed_city_interface_20260701T040731.json"
)
V24_RUN_TYPE = "parametric_fact_override_v24_delayed_city_interface"
V24_DIAGNOSTIC_CLASS = "delayed_city_interface_decouples_first_token_margin"
PRIMARY_LABELS = ("true_answer", "override_answer")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def auc_score(values: list[float], labels: list[int]) -> float | None:
    pos = [value for value, label in zip(values, labels) if label == 1]
    neg = [value for value, label in zip(values, labels) if label == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0
    for pos_value in pos:
        for neg_value in neg:
            total += 1
            if pos_value > neg_value:
                wins += 1.0
            elif pos_value == neg_value:
                wins += 0.5
    return wins / total if total else None


def label_int(row: dict[str, Any]) -> int:
    if row["selected_label"] == "true_answer":
        return 1
    if row["selected_label"] == "override_answer":
        return 0
    raise ValueError(f"non-binary row: {row.get('id')}")


def split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split == "all":
        return rows
    if split == "non_holdout":
        return [row for row in rows if row["split"] != "holdout"]
    return [row for row in rows if row["split"] == split]


def auc_by_split(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    payload = {}
    for split in ("all", "non_holdout", "discovery", "calibration", "holdout"):
        selected = split_rows(rows, split)
        values = [float(row[field]) for row in selected]
        labels = [label_int(row) for row in selected]
        payload[split] = {
            "row_count": len(selected),
            "label_counts": dict(sorted(Counter(row["selected_label"] for row in selected).items())),
            "auc": auc_score(values, labels),
        }
    return payload


def validate_v24(v24: dict[str, Any]) -> dict[str, Any]:
    summary = v24.get("summary", {})
    criteria = {
        "expected_run_type": v24.get("run_type") == V24_RUN_TYPE,
        "expected_diagnostic_class": summary.get("diagnostic_class") == V24_DIAGNOSTIC_CLASS,
        "v24_behavior_ready": summary.get("behavior_ready") is True,
        "v24_not_signature_ready": summary.get("signature_ready") is False,
        "v24_intervention_not_ready": summary.get("intervention_ready") is False,
    }
    return {"criteria": criteria, "passed": all(criteria.values())}


def template_candidate(row_bank: list[dict[str, Any]], template: str) -> dict[str, Any]:
    template_rows = [row for row in row_bank if row["template"] == template]
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
    return {
        "template": template,
        "row_count": len(template_rows),
        "binary_count": len(binary_rows),
        "label_counts": dict(sorted(Counter(row["selected_label"] for row in template_rows).items())),
        "non_holdout_label_counts": dict(sorted(non_holdout_counts.items())),
        "holdout_label_counts": dict(sorted(holdout_counts.items())),
        "selected_city_first_token_rate": first_token_rate,
        "criteria": criteria,
        "behavior_ready": all(
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
        ),
        "candidate_decoupled_ready": all(criteria.values()),
        "margin_audits": margin_audits,
        "auc_audits": auc_audits,
        "binary_rows": binary_rows,
    }


def selection_key(item: dict[str, Any], template: str) -> tuple[int, int, int, float, float, int, int]:
    json_holdout_auc = item["auc_audits"]["json_candidate_score_margin"]["holdout"]["auc"]
    city_holdout_auc = item["auc_audits"]["city_candidate_score_margin"]["holdout"]["auc"]
    final_holdout_auc = item["auc_audits"]["final_next_token_city_margin"]["holdout"]["auc"]
    finite_json_gap = 1.0 - float(json_holdout_auc) if json_holdout_auc is not None else -1.0
    best_holdout_control = max(
        [float(value) for value in (json_holdout_auc, city_holdout_auc, final_holdout_auc) if value is not None]
        or [1.0]
    )
    holdout_rows = sum(item["holdout_label_counts"].values())
    return (
        int(item["candidate_decoupled_ready"]),
        int(item["behavior_ready"]),
        int(json_holdout_auc is not None),
        finite_json_gap,
        -best_holdout_control,
        holdout_rows,
        -TEMPLATES.index(template),
    )


def select_template(template_payloads: dict[str, Any]) -> dict[str, Any]:
    selected = max(TEMPLATES, key=lambda template: selection_key(template_payloads[template], template))
    return {
        "selected_template": selected,
        "selection_key": list(selection_key(template_payloads[selected], selected)),
        "all_selection_keys": {
            template: list(selection_key(template_payloads[template], template))
            for template in TEMPLATES
        },
    }


def shuffled_selection_null(
    features_by_position: dict[str, dict[str, np.ndarray]],
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
    runs: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    null_records = []
    base_labels = labels.copy()
    for index in range(runs):
        shuffled = base_labels.copy()
        for split in ("discovery", "holdout"):
            indices = np.flatnonzero(masks[split]).tolist()
            values = [int(shuffled[item]) for item in indices]
            rng.shuffle(values)
            for item, value in zip(indices, values):
                shuffled[item] = value
        _, selected = score_hidden_candidates(features_by_position, shuffled, masks)
        assert selected is not None
        null_records.append(
            {
                "run": index,
                "selected": selected["name"],
                "discovery_auc": selected["discovery_auc"],
                "holdout_auc": selected["holdout_auc"],
            }
        )
    discovery = [float(row["discovery_auc"] or 0.0) for row in null_records]
    holdout = [float(row["holdout_auc"] or 0.0) for row in null_records]
    return {
        "runs": runs,
        "seed": seed,
        "discovery_auc_p95": percentile(discovery, 0.95),
        "holdout_auc_p95": percentile(holdout, 0.95),
        "max_discovery_auc": max(discovery) if discovery else None,
        "max_holdout_auc": max(holdout) if holdout else None,
        "records": null_records,
    }


def hidden_screen(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    shuffle_runs: int,
    shuffle_seed: int,
) -> dict[str, Any]:
    labels = np.asarray([label_int(row) for row in rows], dtype=np.int64)
    masks = split_masks(rows)
    if len(set(labels[masks["discovery"]].tolist())) < 2:
        return {"ran": False, "reason": "discovery_single_class"}
    if len(set(labels[masks["holdout"]].tolist())) < 2:
        return {"ran": False, "reason": "holdout_single_class"}

    features, position_margins, scored_rows, mapping = collect_features_and_position_margins(rows, tokenizer, model)
    candidates, selected_hidden = score_hidden_candidates(features, labels, masks)
    assert selected_hidden is not None
    global_baselines = {
        "final_next_token_city_margin": scalar_baseline(
            [row["final_next_token_city_margin"] for row in scored_rows],
            labels,
            masks,
            "final_next_token_city_margin",
        ),
        "city_candidate_score_margin": scalar_baseline(
            [row["city_candidate_score_margin"] for row in scored_rows],
            labels,
            masks,
            "city_candidate_score_margin",
        ),
        "json_candidate_score_margin": scalar_baseline(
            [row["json_candidate_score_margin"] for row in scored_rows],
            labels,
            masks,
            "json_candidate_score_margin",
        ),
    }
    position_baselines = {
        position: scalar_baseline(
            position_margins[position]["next_token_city_margin"],
            labels,
            masks,
            f"{position}_next_token_city_margin",
        )
        for position in position_margins
    }
    shuffled_null = shuffled_selection_null(features, labels, masks, shuffle_runs, shuffle_seed)
    best_control_holdout = max(
        float(item["holdout_auc"] or 0.0)
        for item in list(global_baselines.values()) + list(position_baselines.values())
    )
    selected_holdout = float(selected_hidden["holdout_auc"] or 0.0)
    selected_discovery = float(selected_hidden["discovery_auc"] or 0.0)
    shuffle_p95 = float(shuffled_null["holdout_auc_p95"] or 0.0)
    signature_gate = bool(
        selected_discovery >= 0.85
        and selected_holdout >= 0.75
        and selected_holdout > best_control_holdout
        and selected_holdout > shuffle_p95
    )
    return {
        "ran": True,
        "mapping": mapping,
        "row_count": len(rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "label_counts": dict(sorted(Counter(row["selected_label"] for row in rows).items())),
        "selected_hidden": selected_hidden,
        "top_hidden_candidates": sorted(
            candidates,
            key=lambda row: (
                -1.0 if row["discovery_auc"] is None else float(row["discovery_auc"]),
                -1.0 if row["holdout_auc"] is None else float(row["holdout_auc"]),
            ),
            reverse=True,
        )[:20],
        "global_baselines": global_baselines,
        "position_baselines": position_baselines,
        "best_control_holdout_auc": best_control_holdout,
        "shuffled_selection_null": shuffled_null,
        "signature_gate": signature_gate,
    }


def diagnostic_class(criteria: dict[str, bool], hidden: dict[str, Any]) -> str:
    if not criteria["v24_source_valid"]:
        return "source_artifact_invalid"
    if not criteria["candidate_decoupled_template_found"]:
        return "candidate_decoupled_template_absent"
    if not hidden.get("ran"):
        return "candidate_decoupled_template_hidden_not_run"
    if criteria["signature_gate"]:
        return "candidate_decoupled_hidden_signature_candidate"
    if criteria["hidden_beats_candidate_controls"] and not criteria["hidden_beats_shuffle_null"]:
        return "candidate_decoupled_hidden_shuffle_overfit"
    if not criteria["hidden_beats_candidate_controls"]:
        return "candidate_decoupled_hidden_not_above_controls"
    return "candidate_decoupled_hidden_failed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--v24-artifact", type=Path, default=DEFAULT_V24_ARTIFACT)
    parser.add_argument(
        "--artifact-prefix",
        default="mc006_qwen3_1p7b_parametric_fact_override_v25_candidate_decoupled_template",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-runs", type=int, default=128)
    parser.add_argument("--shuffle-seed", type=int, default=25006)
    args = parser.parse_args()

    started = time.time()
    v24 = read_json(args.v24_artifact)
    v24_validation = validate_v24(v24)
    row_bank = list(v24["rows"])
    template_payloads = {
        template: template_candidate(row_bank, template)
        for template in TEMPLATES
    }
    selection = select_template(template_payloads)
    selected_template = selection["selected_template"]
    selected_payload = template_payloads[selected_template]
    selected_rows = selected_payload["binary_rows"]

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

    hidden = (
        hidden_screen(selected_rows, tokenizer, model, args.shuffle_runs, args.shuffle_seed)
        if selected_payload["candidate_decoupled_ready"]
        else {"ran": False, "reason": "candidate_decoupled_template_not_ready"}
    )
    selected_hidden = hidden.get("selected_hidden") or {}
    best_control = float(hidden.get("best_control_holdout_auc") or 0.0)
    shuffle_p95 = float((hidden.get("shuffled_selection_null") or {}).get("holdout_auc_p95") or 0.0)
    selected_holdout = float(selected_hidden.get("holdout_auc") or 0.0)
    criteria = {
        "v24_source_valid": v24_validation["passed"],
        "candidate_decoupled_template_found": bool(selected_payload["candidate_decoupled_ready"]),
        "selected_template_behavior_ready": bool(selected_payload["behavior_ready"]),
        "selected_json_candidate_holdout_auc_not_perfect": bool(
            selected_payload["criteria"]["json_candidate_holdout_auc_not_perfect"]
        ),
        "hidden_screen_ran": bool(hidden.get("ran")),
        "hidden_beats_candidate_controls": bool(hidden.get("ran") and selected_holdout > best_control),
        "hidden_beats_shuffle_null": bool(hidden.get("ran") and selected_holdout > shuffle_p95),
        "signature_gate": bool(hidden.get("signature_gate")),
    }
    diag = diagnostic_class(criteria, hidden)
    summary = {
        "diagnostic_class": diag,
        "passed": criteria["signature_gate"],
        "diagnostic_supported": criteria["candidate_decoupled_template_found"],
        "signature_ready": criteria["signature_gate"],
        "intervention_ready": criteria["signature_gate"],
        "v24_validation": v24_validation,
        "selection": selection,
        "selected_template_summary": {
            key: value for key, value in selected_payload.items() if key != "binary_rows"
        },
        "criteria": criteria,
    }

    output_path = args.output_dir / f"{args.artifact_prefix}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    result = {
        "card_id": args.card_id,
        "run_type": RUN_TYPE,
        "model_id": args.model_id,
        "source_artifacts": [str(args.v24_artifact)],
        "source_artifact_hashes": [{"path": str(args.v24_artifact), "sha256": sha256_file(args.v24_artifact)}],
        "summary": summary,
        "hidden_screen": hidden,
        "by_template": {
            template: {key: value for key, value in payload.items() if key != "binary_rows"}
            for template, payload in template_payloads.items()
        },
        "selected_binary_rows": selected_rows,
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
        f"selected_template={selected_template} "
        f"signature_ready={summary['signature_ready']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
