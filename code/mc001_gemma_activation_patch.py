#!/usr/bin/env python
"""MC001G matched activation replacement gate for Gemma 2 2B."""

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

from mc001_gemma_prehint_margin_discovery import attach_item_margins, matched_rows
from mc001_gemma_repair_discovery import collect_rows, load_json, select_clean_items, split_clean_items
from mc001_logit_smoke import letter_token_ids, render_for_score
from mc001_qwen3_smoke import classify


MATCH_CONDITIONS = ("wrong_disclaimed", "wrong_unsure")


def item_id(row: dict[str, Any]) -> str:
    return row["id"].split("__", 1)[0]


def prepare_rows(repair_rows: list[dict[str, Any]], bin_width: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clean_items = select_clean_items(repair_rows)
    split_items = split_clean_items(clean_items)
    primary_rows = collect_rows(repair_rows, clean_items, split_items)
    primary_rows = [row for row in primary_rows if row["condition"] in MATCH_CONDITIONS]
    enriched = attach_item_margins(primary_rows, repair_rows, bin_width)
    matched = matched_rows(enriched)
    discovery = [row for row in matched if row["split"] == "discovery"]
    holdout = [{**row, "eval_group": "matched_holdout"} for row in matched if row["split"] == "holdout"]

    by_item_condition = {(item_id(row), row["condition"]): row for row in repair_rows}
    holdout_items = sorted({row["item_id"] for row in holdout})
    locality_base: list[dict[str, Any]] = []
    for item in holdout_items:
        for condition in ["no_hint", "correct_hint"]:
            row = by_item_condition[(item, condition)]
            locality_base.append({**row, "item_id": item, "split": "holdout", "eval_group": f"locality_{condition}"})
    locality = attach_item_margins(locality_base, repair_rows, bin_width)

    return discovery, holdout + locality


def score_logits(
    logits: torch.Tensor,
    option_ids: dict[str, list[int]],
    correct_answer: str,
    wrong_answer: str,
) -> tuple[str, str, dict[str, float], float]:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    scores = {
        letter: max(float(log_probs[token_id].item()) for token_id in token_ids)
        for letter, token_ids in option_ids.items()
    }
    parsed = max(scores, key=scores.get)
    label = classify(parsed, correct_answer, wrong_answer)
    return parsed, label, scores, scores[correct_answer] - scores[wrong_answer]


def score_and_collect(
    model: Any,
    tokenizer: Any,
    option_ids: dict[str, list[int]],
    rows: list[dict[str, Any]],
    render_mode: str,
    layers: list[int],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
    hiddens: dict[int, dict[str, np.ndarray]] = {layer: {} for layer in layers}
    outputs: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        rendered = render_for_score(tokenizer, row["prompt"], render_mode)
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            model_outputs = model(**inputs, output_hidden_states=True)
        logits = model_outputs.logits[0, -1]
        parsed, label, scores, margin = score_logits(logits, option_ids, row["correct_answer"], row["wrong_answer"])
        for layer in layers:
            hiddens[layer][row["id"]] = model_outputs.hidden_states[layer + 1][0, -1].float().cpu().numpy()
        outputs.append(
            {
                **row,
                "eval_index": index,
                "rendered_prompt": rendered,
                "source_label": row["label"],
                "parsed_answer": parsed,
                "label": label,
                "option_logprobs": scores,
                "correct_minus_wrong_logprob": margin,
            }
        )
    return outputs, hiddens


def choose_donor(
    target: dict[str, Any],
    donors_by_label: dict[str, list[dict[str, Any]]],
    label: str,
) -> tuple[dict[str, Any], str]:
    pool = donors_by_label[label]
    exact_letter = [
        row for row in pool
        if row["condition"] == target["condition"]
        and row["no_hint_margin_bin"] == target["no_hint_margin_bin"]
        and row["correct_answer"] == target["correct_answer"]
        and row["wrong_answer"] == target["wrong_answer"]
    ]
    if exact_letter:
        return exact_letter[0], "condition_bin_letters"

    exact_condition_bin = [
        row for row in pool
        if row["condition"] == target["condition"]
        and row["no_hint_margin_bin"] == target["no_hint_margin_bin"]
    ]
    if exact_condition_bin:
        return exact_condition_bin[0], "condition_bin"

    exact_bin = [row for row in pool if row["no_hint_margin_bin"] == target["no_hint_margin_bin"]]
    if exact_bin:
        return exact_bin[0], "bin"

    same_condition = [row for row in pool if row["condition"] == target["condition"]]
    if same_condition:
        nearest = min(same_condition, key=lambda row: (abs(row["no_hint_margin_bin"] - target["no_hint_margin_bin"]), row["id"]))
        return nearest, "condition_nearest_bin"

    nearest = min(pool, key=lambda row: (abs(row["no_hint_margin_bin"] - target["no_hint_margin_bin"]), row["id"]))
    return nearest, "nearest_bin"


def donor_plan(discovery_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    donors_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(discovery_rows, key=lambda row: (row["label"], row["condition"], row["no_hint_margin_bin"], row["id"])):
        donors_by_label[row["label"]].append(row)
    for label in ["truth_following", "user_agreement_error"]:
        if not donors_by_label[label]:
            raise RuntimeError(f"no discovery donors for {label}")

    plan: dict[str, dict[str, Any]] = {}
    for target in eval_rows:
        truth_donor, truth_rule = choose_donor(target, donors_by_label, "truth_following")
        agreement_donor, agreement_rule = choose_donor(target, donors_by_label, "user_agreement_error")
        plan[target["id"]] = {
            "truth_following": {"id": truth_donor["id"], "rule": truth_rule},
            "user_agreement_error": {"id": agreement_donor["id"], "rule": agreement_rule},
        }
    return plan


def score_with_replacement(
    model: Any,
    tokenizer: Any,
    option_ids: dict[str, list[int]],
    rows: list[dict[str, Any]],
    render_mode: str,
    layer: int | None,
    vectors_by_row_id: dict[str, np.ndarray] | None,
    arm_name: str,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    target_layer = model.model.layers[layer] if layer is not None else None
    for index, row in enumerate(rows, start=1):
        hook_handle = None
        if target_layer is not None and vectors_by_row_id is not None:
            replacement = torch.tensor(vectors_by_row_id[row["id"]])

            def hook_fn(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
                if isinstance(output, tuple):
                    hidden = output[0].clone()
                    hidden[:, -1, :] = replacement.to(device=hidden.device, dtype=hidden.dtype)
                    return (hidden, *output[1:])
                hidden = output.clone()
                hidden[:, -1, :] = replacement.to(device=hidden.device, dtype=hidden.dtype)
                return hidden

            hook_handle = target_layer.register_forward_hook(hook_fn)

        rendered = render_for_score(tokenizer, row["prompt"], render_mode)
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        try:
            with torch.inference_mode():
                logits = model(**inputs).logits[0, -1]
        finally:
            if hook_handle is not None:
                hook_handle.remove()
        parsed, label, scores, margin = score_logits(logits, option_ids, row["correct_answer"], row["wrong_answer"])
        outputs.append(
            {
                **row,
                "eval_index": index,
                "arm": arm_name,
                "source_label": row["label"],
                "parsed_answer": parsed,
                "label": label,
                "option_logprobs": scores,
                "correct_minus_wrong_logprob": margin,
            }
        )
    return outputs


def build_replacement_vectors(
    arm: dict[str, Any],
    eval_rows: list[dict[str, Any]],
    donor_lookup: dict[str, dict[str, Any]],
    donor_hiddens: dict[int, dict[str, np.ndarray]],
    target_hiddens: dict[int, dict[str, np.ndarray]],
) -> dict[str, np.ndarray] | None:
    if arm["kind"] == "baseline":
        return None
    if arm["kind"] == "self":
        return {row["id"]: target_hiddens[arm["layer"]][row["id"]] for row in eval_rows}
    if arm["kind"] == "donor":
        donor_label = arm["donor_label"]
        return {
            row["id"]: donor_hiddens[arm["layer"]][donor_lookup[row["id"]][donor_label]["id"]]
            for row in eval_rows
        }
    raise ValueError(f"unknown arm kind: {arm['kind']}")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(row["label"] for row in rows)
    source_labels = Counter(row["source_label"] for row in rows)
    parsed = Counter(row["parsed_answer"] for row in rows)
    margins = [float(row["correct_minus_wrong_logprob"]) for row in rows]
    return {
        "n": len(rows),
        "source_labels": dict(sorted(source_labels.items())),
        "labels": dict(sorted(labels.items())),
        "parsed_answers": dict(sorted(parsed.items())),
        "mean_margin": float(np.mean(margins)) if margins else None,
        "min_margin": float(np.min(margins)) if margins else None,
        "max_margin": float(np.max(margins)) if margins else None,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["eval_group"]].append(row)

    summary: dict[str, Any] = {}
    for group, group_rows in sorted(groups.items()):
        by_source = {
            label: summarize_rows([row for row in group_rows if row["source_label"] == label])
            for label in sorted({row["source_label"] for row in group_rows})
        }
        summary[group] = {
            "all": summarize_rows(group_rows),
            "by_source_label": by_source,
        }
    return summary


def compare_to_baseline(baseline: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_by_id = {row["id"]: row for row in baseline}
    changed = [row for row in rows if row["label"] != baseline_by_id[row["id"]]["label"]]

    def count_truth(group: str, source_label: str) -> int:
        return sum(
            row["eval_group"] == group
            and row["source_label"] == source_label
            and row["label"] == "truth_following"
            for row in rows
        )

    def baseline_truth(group: str, source_label: str) -> int:
        return sum(
            row["eval_group"] == group
            and row["source_label"] == source_label
            and row["label"] == "truth_following"
            for row in baseline
        )

    return {
        "changed_label_n": len(changed),
        "changed_label_ids": [
            {
                "id": row["id"],
                "eval_group": row["eval_group"],
                "source_label": row["source_label"],
                "baseline_label": baseline_by_id[row["id"]]["label"],
                "patched_label": row["label"],
                "baseline_answer": baseline_by_id[row["id"]]["parsed_answer"],
                "patched_answer": row["parsed_answer"],
            }
            for row in changed
        ],
        "matched_holdout_agreement_truth_delta": count_truth("matched_holdout", "user_agreement_error")
        - baseline_truth("matched_holdout", "user_agreement_error"),
        "matched_holdout_truth_truth_delta": count_truth("matched_holdout", "truth_following")
        - baseline_truth("matched_holdout", "truth_following"),
        "locality_no_hint_truth_delta": count_truth("locality_no_hint", "truth_following")
        - baseline_truth("locality_no_hint", "truth_following"),
        "locality_correct_hint_truth_delta": count_truth("locality_correct_hint", "truth_following")
        - baseline_truth("locality_correct_hint", "truth_following"),
    }


def donor_rule_summary(donor_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for label in ["truth_following", "user_agreement_error"]:
        rules = Counter(row[label]["rule"] for row in donor_lookup.values())
        donors = Counter(row[label]["id"] for row in donor_lookup.values())
        summary[label] = {
            "rules": dict(sorted(rules.items())),
            "unique_donor_count": len(donors),
            "most_common_donors": donors.most_common(5),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/gemma-2-2b")
    parser.add_argument("--repair-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cards/MC001G"))
    parser.add_argument("--artifact-prefix", default="mc001g_gemma2_2b_repair")
    parser.add_argument("--render-mode", choices=["raw", "chat"], default="raw")
    parser.add_argument("--bin-width", type=float, default=0.5)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--logistic-layer", type=int, default=20)
    parser.add_argument("--wrong-layer", type=int, default=13)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    repair = load_json(args.repair_result)
    discovery_rows, eval_rows = prepare_rows(repair["records"], args.bin_width)
    layers = sorted({args.layer, args.logistic_layer, args.wrong_layer})

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    option_ids = letter_token_ids(tokenizer)

    started = time.time()
    scored_discovery, donor_hiddens = score_and_collect(model, tokenizer, option_ids, discovery_rows, args.render_mode, layers)
    scored_eval, target_hiddens = score_and_collect(model, tokenizer, option_ids, eval_rows, args.render_mode, layers)
    donor_lookup = donor_plan(scored_discovery, scored_eval)

    arms = [
        {"name": "baseline", "kind": "baseline", "layer": None},
        {"name": f"layer{args.layer}_self_replace", "kind": "self", "layer": args.layer},
        {"name": f"layer{args.layer}_truth_donor_replace", "kind": "donor", "layer": args.layer, "donor_label": "truth_following"},
        {"name": f"layer{args.layer}_agreement_donor_replace", "kind": "donor", "layer": args.layer, "donor_label": "user_agreement_error"},
        {
            "name": f"layer{args.logistic_layer}_truth_donor_replace",
            "kind": "donor",
            "layer": args.logistic_layer,
            "donor_label": "truth_following",
        },
        {"name": f"wrong_layer{args.wrong_layer}_truth_donor_replace", "kind": "donor", "layer": args.wrong_layer, "donor_label": "truth_following"},
    ]

    arm_outputs: dict[str, Any] = {}
    baseline_records = scored_eval
    for arm in arms:
        if arm["kind"] == "baseline":
            rows = scored_eval
        else:
            vectors = build_replacement_vectors(arm, scored_eval, donor_lookup, donor_hiddens, target_hiddens)
            rows = score_with_replacement(
                model,
                tokenizer,
                option_ids,
                scored_eval,
                args.render_mode,
                arm["layer"],
                vectors,
                arm["name"],
            )
        arm_outputs[arm["name"]] = {
            "arm": arm,
            "summary": summarize(rows),
            "comparison_to_baseline": compare_to_baseline(baseline_records, rows),
            "records": rows,
        }
        print(f"completed {arm['name']}")

    elapsed = time.time() - started
    result = {
        "card_id": "MC001G",
        "run_type": f"{args.artifact_prefix}_matched_activation_patch",
        "model_id": args.model_id,
        "repair_result": str(args.repair_result),
        "render_mode": args.render_mode,
        "bin_width": args.bin_width,
        "layers": {"primary": args.layer, "logistic": args.logistic_layer, "wrong": args.wrong_layer},
        "match_conditions": list(MATCH_CONDITIONS),
        "discovery_row_count": len(scored_discovery),
        "eval_row_count": len(scored_eval),
        "eval_group_counts": dict(sorted(Counter(row["eval_group"] for row in scored_eval).items())),
        "donor_rule_summary": donor_rule_summary(donor_lookup),
        "donor_lookup": donor_lookup,
        "elapsed_s": elapsed,
        "discovery_rows": scored_discovery,
        "arms": arm_outputs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_matched_activation_patch_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)

    compact = {
        "eval_group_counts": result["eval_group_counts"],
        "donor_rule_summary": result["donor_rule_summary"],
        "arms": {
            name: {
                "summary": arm["summary"],
                "comparison_to_baseline": arm["comparison_to_baseline"],
            }
            for name, arm in arm_outputs.items()
        },
        "output_path": str(output_path),
    }
    print(json.dumps(compact, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
