#!/usr/bin/env python
"""MC006 V17 causal stress test for the V16 pre-output signature."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc006_parametric_fact_override_v14_parser_normalized import strict_parse_nfkd
from mc006_parametric_fact_override_v15_parser_normalized_signature import (
    first_answer_token_id,
    read_json,
    sha256_file,
    validate_source,
)
from mc006_parametric_fact_override_v16_pre_output_position_signature import (
    PRIMARY_LABELS,
    structural_check,
    token_positions_for_prompt,
)


CARD_ID = "MC006"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/MC006")
DEFAULT_V14_ARTIFACT = (
    RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v14_parser_normalized_20260701T000538.json"
)
DEFAULT_V16_ARTIFACT = (
    RESULT_DIR / "mc006_qwen3_1p7b_parametric_fact_override_v16_pre_output_position_signature_20260701T002113.json"
)
RUN_TYPE = "parametric_fact_override_v17_pre_output_steering_stress"
SELECTED_POSITION = "after_mapping_line"
SELECTED_LAYER = 4
WRONG_POSITION = "after_return_line"
WRONG_LAYER = 16
DEFAULT_DOSES = (1.0, 2.0, 4.0)


def validate_v16(v16: dict[str, Any], v14_sha256: str) -> None:
    if v16.get("run_type") != "parametric_fact_override_v16_pre_output_position_signature":
        raise ValueError(f"unexpected V16 run_type: {v16.get('run_type')!r}")
    summary = v16.get("summary", {})
    if summary.get("diagnostic_class") != "leadtime_signal_supported_but_output_global_confounded":
        raise ValueError(f"unexpected V16 diagnostic class: {summary.get('diagnostic_class')!r}")
    if not summary.get("leadtime_supported"):
        raise ValueError("V16 lead-time support was not true")
    if summary.get("source_artifact_sha256") != v14_sha256:
        raise ValueError("V16 source hash does not match V14 artifact")
    selected = summary.get("selected_pre_output_candidate", {})
    if selected.get("position") != SELECTED_POSITION or int(selected.get("layer", -1)) != SELECTED_LAYER:
        raise ValueError(f"unexpected V16 selected candidate: {selected!r}")


def build_rows(source_result: dict[str, Any], tokenizer: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    side_rows = []
    selected_template = source_result["summary"]["selection"]["selected_template"]
    for source_row in source_result["summary"]["selected_template_rows"]:
        prompt_ids = tokenizer(source_row["prompt"], add_special_tokens=False)["input_ids"]
        is_primary = source_row["selected_label"] in PRIMARY_LABELS
        row = {
            "id": source_row["id"].replace("_v13_", "_v17_"),
            "source_row_id": source_row["id"],
            "source_id": source_row["source_id"],
            "split": source_row["split"],
            "template": selected_template,
            "condition": source_row["condition"],
            "country": source_row["country"],
            "true_capital": source_row["true_capital"],
            "override_capital": source_row["override_capital"],
            "lure_capital": source_row["lure_capital"],
            "source_selected_label": source_row["selected_label"],
            "source_selected_answer": source_row["selected_answer"],
            "selected_label": source_row["selected_label"],
            "selected_answer": source_row["selected_answer"],
            "source_first_line": source_row.get("first_line"),
            "source_generated_text": source_row.get("generated_text"),
            "label_changed_by_normalization": bool(source_row.get("label_changed_by_normalization")),
            "is_primary": is_primary,
            "binary_label": 1 if source_row["selected_label"] == "true_answer" else 0 if is_primary else None,
            "label_name": source_row["selected_label"] if is_primary else None,
            "prompt": source_row["prompt"],
            "prompt_token_count": len(prompt_ids),
        }
        if is_primary:
            rows.append(row)
        else:
            side_rows.append(row)
    rows.sort(key=lambda item: (item["split"], item["source_id"]))
    side_rows.sort(key=lambda item: (item["split"], item["source_id"]))
    return rows, side_rows


def collect_vectors(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    layer: int,
    position: str,
) -> np.ndarray:
    vectors = []
    for index, row in enumerate(rows, start=1):
        positions = token_positions_for_prompt(tokenizer, row["prompt"])
        token_index = int(positions[position]["token_index"])
        inputs = tokenizer(
            row["prompt"],
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        vectors.append(out.hidden_states[layer + 1][0, token_index, :].detach().float().cpu().numpy())
        print(f"[v17 direction] {index:03d}/{len(rows):03d} {row['id']}")
    return np.stack(vectors, axis=0).astype(np.float32)


def build_signature_delta(
    vectors: np.ndarray,
    rows: list[dict[str, Any]],
    dose: float,
    random_seed: int,
) -> dict[str, Any]:
    labels = np.array([int(row["binary_label"]) for row in rows], dtype=np.int64)
    train_mask = np.array([row["split"] != "holdout" for row in rows], dtype=bool)
    x_train = vectors[train_mask]
    y_train = labels[train_mask]
    mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = x_train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    z_train = (x_train - mean) / std
    pos = z_train[y_train == 1]
    neg = z_train[y_train == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("V17 direction requires both labels in non-holdout rows")
    raw_direction = pos.mean(axis=0) - neg.mean(axis=0)
    direction_norm = float(np.linalg.norm(raw_direction))
    if direction_norm < 1e-12:
        raise ValueError("zero V17 direction")
    unit_z = (raw_direction / direction_norm).astype(np.float32)
    gradient = (unit_z.reshape(1, -1) / std).reshape(-1).astype(np.float32)
    denom = float(np.dot(gradient, gradient))
    if denom < 1e-12:
        raise ValueError("zero V17 raw-space gradient")
    delta = (float(dose) * gradient / denom).astype(np.float32)
    rng = np.random.default_rng(random_seed)
    random_delta = rng.normal(size=delta.shape).astype(np.float32)
    random_delta = random_delta / max(float(np.linalg.norm(random_delta)), 1e-12)
    random_delta = random_delta * float(np.linalg.norm(delta))
    return {
        "dose": float(dose),
        "delta": delta,
        "random_delta": random_delta.astype(np.float32),
        "direction_norm": direction_norm,
        "delta_norm": float(np.linalg.norm(delta)),
        "random_delta_norm": float(np.linalg.norm(random_delta)),
        "expected_signature_score_shift": float(np.dot(gradient, delta)),
    }


def arm_specs(doses: list[float]) -> list[dict[str, Any]]:
    specs = [{"arm": "baseline", "dose": 0.0, "kind": "none", "layer": None, "position": None, "sign": 0.0}]
    for dose in doses:
        specs.extend(
            [
                {
                    "arm": "plus_selected",
                    "dose": dose,
                    "kind": "signature",
                    "layer": SELECTED_LAYER,
                    "position": SELECTED_POSITION,
                    "sign": 1.0,
                },
                {
                    "arm": "minus_selected",
                    "dose": dose,
                    "kind": "signature",
                    "layer": SELECTED_LAYER,
                    "position": SELECTED_POSITION,
                    "sign": -1.0,
                },
                {
                    "arm": "random_selected",
                    "dose": dose,
                    "kind": "random",
                    "layer": SELECTED_LAYER,
                    "position": SELECTED_POSITION,
                    "sign": 1.0,
                },
                {
                    "arm": "plus_wrong_position",
                    "dose": dose,
                    "kind": "signature",
                    "layer": SELECTED_LAYER,
                    "position": WRONG_POSITION,
                    "sign": 1.0,
                },
                {
                    "arm": "plus_wrong_layer",
                    "dose": dose,
                    "kind": "signature",
                    "layer": WRONG_LAYER,
                    "position": SELECTED_POSITION,
                    "sign": 1.0,
                },
            ]
        )
    return specs


def install_delta_hook(
    model: Any,
    layer: int,
    token_index: int,
    delta: torch.Tensor,
) -> Any:
    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
            if int(hidden.shape[1]) <= token_index:
                return output
            patched = hidden.clone()
            patched[:, token_index, :] = patched[:, token_index, :] + delta
            return (patched,) + output[1:]
        if int(output.shape[1]) <= token_index:
            return output
        patched = output.clone()
        patched[:, token_index, :] = patched[:, token_index, :] + delta
        return patched

    return model.model.layers[layer].register_forward_hook(hook)


def true_minus_override_margin(row: dict[str, Any], tokenizer: Any, logits: torch.Tensor) -> float:
    true_token = first_answer_token_id(tokenizer, row["true_capital"])
    override_token = first_answer_token_id(tokenizer, row["override_capital"])
    return float(logits[true_token] - logits[override_token])


def run_generation_arm(
    row: dict[str, Any],
    tokenizer: Any,
    model: Any,
    spec: dict[str, Any],
    deltas_by_dose: dict[float, dict[str, Any]],
    max_new_tokens: int,
) -> dict[str, Any]:
    handle = None
    if spec["kind"] != "none":
        delta_key = "random_delta" if spec["kind"] == "random" else "delta"
        delta_np = deltas_by_dose[float(spec["dose"])][delta_key] * float(spec["sign"])
        delta = torch.tensor(
            delta_np,
            device=model.device,
            dtype=next(model.parameters()).dtype,
        )
        token_index = int(token_positions_for_prompt(tokenizer, row["prompt"])[spec["position"]]["token_index"])
        handle = install_delta_hook(model, int(spec["layer"]), token_index, delta)
    try:
        inputs = tokenizer(
            row["prompt"],
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)
        input_len = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
    finally:
        if handle is not None:
            handle.remove()
    new_tokens = generated.sequences[0, input_len:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    parsed = strict_parse_nfkd({**row, "generated_text": generated_text})
    if not generated.scores:
        margin = float("nan")
    else:
        margin = true_minus_override_margin(row, tokenizer, generated.scores[0][0].detach().float().cpu())
    return {
        "row_id": row["id"],
        "source_id": row["source_id"],
        "split": row["split"],
        "is_primary": row["is_primary"],
        "source_selected_label": row["source_selected_label"],
        "source_selected_answer": row["source_selected_answer"],
        "arm": spec["arm"],
        "dose": float(spec["dose"]),
        "kind": spec["kind"],
        "layer": spec["layer"],
        "position": spec["position"],
        "sign": float(spec["sign"]),
        "generated_text": generated_text,
        "generated_token_ids": [int(token_id) for token_id in new_tokens.detach().cpu().tolist()],
        "true_minus_override_first_token_margin": margin,
        **parsed,
    }


def add_deltas(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_by_row = {
        row["row_id"]: row
        for row in outputs
        if row["arm"] == "baseline"
    }
    enriched = []
    for row in outputs:
        baseline = baseline_by_row[row["row_id"]]
        margin_delta = float(
            row["true_minus_override_first_token_margin"]
            - baseline["true_minus_override_first_token_margin"]
        )
        enriched.append(
            {
                **row,
                "baseline_selected_label": baseline["selected_label"],
                "baseline_selected_answer": baseline["selected_answer"],
                "margin_delta_vs_baseline": margin_delta,
                "label_changed_vs_baseline": row["selected_label"] != baseline["selected_label"],
                "label_changed_vs_source": row["selected_label"] != row["source_selected_label"],
            }
        )
    return enriched


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "mean_margin": None,
            "mean_margin_delta": None,
            "label_counts": {},
            "label_changes_vs_baseline": 0,
            "predicted_direction_changes": 0,
        }
    predicted_changes = 0
    for row in rows:
        if row["arm"] == "plus_selected" and row["baseline_selected_label"] == "override_answer":
            predicted_changes += int(row["selected_label"] == "true_answer")
        if row["arm"] == "minus_selected" and row["baseline_selected_label"] == "true_answer":
            predicted_changes += int(row["selected_label"] == "override_answer")
    return {
        "n": len(rows),
        "mean_margin": float(np.mean([row["true_minus_override_first_token_margin"] for row in rows])),
        "mean_margin_delta": float(np.mean([row["margin_delta_vs_baseline"] for row in rows])),
        "median_margin_delta": float(np.median([row["margin_delta_vs_baseline"] for row in rows])),
        "label_counts": dict(sorted(Counter(row["selected_label"] for row in rows).items())),
        "label_changes_vs_baseline": sum(1 for row in rows if row["label_changed_vs_baseline"]),
        "label_changes_vs_source": sum(1 for row in rows if row["label_changed_vs_source"]),
        "predicted_direction_changes": predicted_changes,
        "first_lines": {
            row["row_id"]: row.get("first_line")
            for row in rows
            if row["label_changed_vs_baseline"]
        },
    }


def build_summary(outputs: list[dict[str, Any]], primary_rows: list[dict[str, Any]], side_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in outputs:
        if row["is_primary"] and row["split"] == "holdout":
            subset = "holdout_primary"
        elif row["is_primary"] and row["split"] != "holdout":
            subset = "non_holdout_primary"
        elif not row["is_primary"]:
            subset = "side_rows"
        else:
            subset = "other"
        grouped[(row["arm"], float(row["dose"]), subset)].append(row)
    arm_summaries = {
        f"{arm}@{dose:g}/{subset}": summarize_rows(rows)
        for (arm, dose, subset), rows in sorted(grouped.items())
    }
    baseline_holdout = grouped[("baseline", 0.0, "holdout_primary")]
    baseline_reproduction = {
        "matches": sum(row["selected_label"] == row["source_selected_label"] for row in baseline_holdout),
        "total": len(baseline_holdout),
        "mismatches": [
            {
                "row_id": row["row_id"],
                "source_label": row["source_selected_label"],
                "baseline_label": row["selected_label"],
                "first_line": row.get("first_line"),
            }
            for row in baseline_holdout
            if row["selected_label"] != row["source_selected_label"]
        ],
    }
    dose_summaries = {}
    for dose in sorted({float(row["dose"]) for row in outputs if float(row["dose"]) > 0.0}):
        plus = arm_summaries[f"plus_selected@{dose:g}/holdout_primary"]
        minus = arm_summaries[f"minus_selected@{dose:g}/holdout_primary"]
        random_summary = arm_summaries[f"random_selected@{dose:g}/holdout_primary"]
        wrong_position = arm_summaries[f"plus_wrong_position@{dose:g}/holdout_primary"]
        wrong_layer = arm_summaries[f"plus_wrong_layer@{dose:g}/holdout_primary"]
        selected_abs = (
            abs(float(plus["mean_margin_delta"]))
            + abs(float(minus["mean_margin_delta"]))
        ) / 2.0
        control_abs = max(
            abs(float(random_summary["mean_margin_delta"])),
            abs(float(wrong_position["mean_margin_delta"])),
            abs(float(wrong_layer["mean_margin_delta"])),
        )
        side_plus = arm_summaries[f"plus_selected@{dose:g}/side_rows"]
        side_minus = arm_summaries[f"minus_selected@{dose:g}/side_rows"]
        dose_summaries[str(dose)] = {
            "dose": dose,
            "plus_mean_margin_delta": plus["mean_margin_delta"],
            "minus_mean_margin_delta": minus["mean_margin_delta"],
            "selected_abs_mean_effect": selected_abs,
            "max_control_abs_mean_effect": control_abs,
            "control_gap": selected_abs - control_abs,
            "plus_predicted_direction_changes": plus["predicted_direction_changes"],
            "minus_predicted_direction_changes": minus["predicted_direction_changes"],
            "total_predicted_direction_changes": plus["predicted_direction_changes"] + minus["predicted_direction_changes"],
            "plus_side_label_changes": side_plus["label_changes_vs_baseline"],
            "minus_side_label_changes": side_minus["label_changes_vs_baseline"],
            "max_side_label_changes": max(
                side_plus["label_changes_vs_baseline"],
                side_minus["label_changes_vs_baseline"],
            ),
        }
    best_dose = max(
        dose_summaries.values(),
        key=lambda row: (
            float(row["control_gap"]),
            float(row["selected_abs_mean_effect"]),
            -float(row["dose"]),
        ),
    )
    return {
        "primary_row_count": len(primary_rows),
        "side_row_count": len(side_rows),
        "baseline_reproduction": baseline_reproduction,
        "arm_summaries": arm_summaries,
        "dose_summaries": dose_summaries,
        "best_dose": best_dose,
    }


def classify(summary: dict[str, Any], source_checks_passed: bool) -> tuple[str, dict[str, bool], dict[str, bool]]:
    best = summary["best_dose"]
    baseline = summary["baseline_reproduction"]
    causal_criteria = {
        "source_checks_passed": source_checks_passed,
        "baseline_reproduces_at_least_6_of_8_holdout_labels": baseline["matches"] >= 6
        and baseline["total"] == 8,
        "plus_selected_increases_margin_by_0p25": float(best["plus_mean_margin_delta"]) >= 0.25,
        "minus_selected_decreases_margin_by_0p25": float(best["minus_mean_margin_delta"]) <= -0.25,
        "selected_abs_effect_beats_controls_by_0p25": float(best["control_gap"]) >= 0.25,
        "has_predicted_direction_generation_change_or_margin_only": True,
    }
    side_effect_passed = int(best["max_side_label_changes"]) <= 3
    has_generation_change = int(best["total_predicted_direction_changes"]) >= 1
    mechanism_criteria = {
        **causal_criteria,
        "has_predicted_direction_generation_change": has_generation_change,
        "side_rows_no_broad_parse_corruption": side_effect_passed,
        "not_global_output_confounded": False,
    }
    if not causal_criteria["source_checks_passed"]:
        label = "source_artifact_invalid"
    elif not causal_criteria["baseline_reproduces_at_least_6_of_8_holdout_labels"]:
        label = "baseline_reproduction_failed"
    elif not (
        causal_criteria["plus_selected_increases_margin_by_0p25"]
        and causal_criteria["minus_selected_decreases_margin_by_0p25"]
    ):
        label = "intervention_failed"
    elif not causal_criteria["selected_abs_effect_beats_controls_by_0p25"]:
        label = "control_matched"
    elif not side_effect_passed:
        label = "side_effect_failed"
    elif has_generation_change:
        label = "causal_stress_positive_global_output_confounded"
    else:
        label = "margin_only_causal_effect_global_output_confounded"
    if all(mechanism_criteria.values()):
        label = "pre_output_steering_mechanism_supported"
    return label, causal_criteria, mechanism_criteria


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--v14-artifact", type=Path, default=DEFAULT_V14_ARTIFACT)
    parser.add_argument("--v16-artifact", type=Path, default=DEFAULT_V16_ARTIFACT)
    parser.add_argument(
        "--artifact-prefix",
        default="mc006_qwen3_1p7b_parametric_fact_override_v17_pre_output_steering_stress",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--doses", type=float, nargs="+", default=list(DEFAULT_DOSES))
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=26017)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.time()
    v14_sha = sha256_file(args.v14_artifact)
    v16_sha = sha256_file(args.v16_artifact)
    v14 = read_json(args.v14_artifact)
    v16 = read_json(args.v16_artifact)
    validate_source(v14)
    validate_v16(v16, v14_sha)

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("V17 requires a fast tokenizer for offset mapping")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    primary_rows, side_rows = build_rows(v14, tokenizer)
    structural = structural_check(primary_rows, side_rows)
    if not structural["passed"]:
        raise ValueError(f"V17 structural check failed: {structural}")
    vectors = collect_vectors(primary_rows, tokenizer, model, SELECTED_LAYER, SELECTED_POSITION)
    deltas_by_dose = {
        float(dose): build_signature_delta(
            vectors,
            primary_rows,
            float(dose),
            args.random_seed + int(float(dose) * 1000),
        )
        for dose in args.doses
    }

    rows_for_generation = primary_rows + side_rows
    outputs = []
    specs = arm_specs([float(dose) for dose in args.doses])
    total = len(rows_for_generation) * len(specs)
    counter = 0
    for spec in specs:
        for row in rows_for_generation:
            counter += 1
            output = run_generation_arm(
                row,
                tokenizer,
                model,
                spec,
                deltas_by_dose,
                args.max_new_tokens,
            )
            outputs.append(output)
            print(
                f"[v17 arm] {counter:03d}/{total:03d} {spec['arm']}@{float(spec['dose']):g} "
                f"{row['id']} -> {output['selected_label']} "
                f"margin={output['true_minus_override_first_token_margin']:+.3f}"
            )
    outputs = add_deltas(outputs)
    run_summary = build_summary(outputs, primary_rows, side_rows)
    diagnostic_class, causal_criteria, mechanism_criteria = classify(run_summary, True)
    summary = {
        "model_id": args.model_id,
        "v14_artifact": str(args.v14_artifact),
        "v14_artifact_sha256": v14_sha,
        "v16_artifact": str(args.v16_artifact),
        "v16_artifact_sha256": v16_sha,
        "source_checks_passed": True,
        "selected_position": SELECTED_POSITION,
        "selected_layer": SELECTED_LAYER,
        "wrong_position_control": WRONG_POSITION,
        "wrong_layer_control": WRONG_LAYER,
        "doses": [float(dose) for dose in args.doses],
        "delta_summaries": {
            str(dose): {
                key: value
                for key, value in payload.items()
                if key not in {"delta", "random_delta"}
            }
            for dose, payload in deltas_by_dose.items()
        },
        "structural": structural,
        **run_summary,
        "causal_criteria": causal_criteria,
        "mechanism_criteria": mechanism_criteria,
        "causal_stress_supported": diagnostic_class
        in {
            "causal_stress_positive_global_output_confounded",
            "margin_only_causal_effect_global_output_confounded",
            "pre_output_steering_mechanism_supported",
        },
        "passed": diagnostic_class == "pre_output_steering_mechanism_supported",
        "diagnostic_class": diagnostic_class,
    }
    result = {
        "card_id": args.card_id,
        "run_type": RUN_TYPE,
        "model_id": args.model_id,
        "elapsed_s": time.time() - started,
        "summary": summary,
        "records": [{key: value for key, value in row.items() if key != "prompt"} for row in rows_for_generation],
        "outputs": outputs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)
    print(json.dumps({**summary, "output_path": str(output_path)}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
