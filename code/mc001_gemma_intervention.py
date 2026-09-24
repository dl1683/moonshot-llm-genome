#!/usr/bin/env python
"""MC001G matched dense intervention gate for Gemma 2 2B."""

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
from mc001_gemma_repair_discovery import collect_rows, label_value, load_json, select_clean_items, split_clean_items
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
    holdout = [row for row in matched if row["split"] == "holdout"]

    by_item_condition = {(item_id(row), row["condition"]): row for row in repair_rows}
    holdout_items = sorted({row["item_id"] for row in holdout})
    locality: list[dict[str, Any]] = []
    for item in holdout_items:
        for condition in ["no_hint", "correct_hint"]:
            base = by_item_condition[(item, condition)]
            locality.append({**base, "item_id": item, "split": "holdout", "eval_group": f"locality_{condition}"})

    return discovery, [{**row, "eval_group": "matched_holdout"} for row in holdout] + locality


def score_rows(
    model: Any,
    tokenizer: Any,
    option_ids: dict[str, list[int]],
    rows: list[dict[str, Any]],
    render_mode: str,
    layer_index: int | None = None,
    vector: torch.Tensor | None = None,
    alpha: float = 0.0,
    collect_layer: int | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    captured: list[np.ndarray] = []

    hook_handle = None
    if layer_index is not None and vector is not None and alpha != 0.0:
        target_layer = model.model.layers[layer_index]

        def hook_fn(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            if isinstance(output, tuple):
                hidden = output[0].clone()
                hidden[:, -1, :] += alpha * vector.to(device=hidden.device, dtype=hidden.dtype)
                return (hidden, *output[1:])
            hidden = output.clone()
            hidden[:, -1, :] += alpha * vector.to(device=hidden.device, dtype=hidden.dtype)
            return hidden

        hook_handle = target_layer.register_forward_hook(hook_fn)

    outputs: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(rows, start=1):
            rendered = render_for_score(tokenizer, row["prompt"], render_mode)
            inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                model_outputs = model(**inputs, output_hidden_states=collect_layer is not None)
            if collect_layer is not None:
                captured.append(model_outputs.hidden_states[collect_layer + 1][0, -1].float().cpu().numpy())
            logits = model_outputs.logits[0, -1].float()
            log_probs = torch.log_softmax(logits, dim=-1)
            scores = {
                letter: max(float(log_probs[token_id].item()) for token_id in token_ids)
                for letter, token_ids in option_ids.items()
            }
            parsed = max(scores, key=scores.get)
            label = classify(parsed, row["correct_answer"], row["wrong_answer"])
            outputs.append(
                {
                    **row,
                    "eval_index": index,
                    "rendered_prompt": rendered,
                    "parsed_answer": parsed,
                    "label": label,
                    "option_logprobs": scores,
                    "correct_minus_wrong_logprob": scores[row["correct_answer"]] - scores[row["wrong_answer"]],
                }
            )
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    hidden_array = np.stack(captured, axis=0).astype(np.float32) if captured else None
    return outputs, hidden_array


def build_direction(train_rows: list[dict[str, Any]], train_hidden: np.ndarray) -> tuple[np.ndarray, float]:
    y = np.array([label_value(row["label"]) for row in train_rows], dtype=np.int64)
    truth_mean = train_hidden[y == 1].mean(axis=0)
    agree_mean = train_hidden[y == 0].mean(axis=0)
    direction = truth_mean - agree_mean
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise RuntimeError("zero intervention direction")
    return direction.astype(np.float32), norm


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["eval_group"]].append(row)

    def summarize_rows(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        labels = Counter(row["label"] for row in group_rows)
        parsed = Counter(row["parsed_answer"] for row in group_rows)
        margins = [float(row["correct_minus_wrong_logprob"]) for row in group_rows]
        return {
            "n": len(group_rows),
            "labels": dict(sorted(labels.items())),
            "parsed_answers": dict(sorted(parsed.items())),
            "mean_margin": float(np.mean(margins)) if margins else None,
            "min_margin": float(np.min(margins)) if margins else None,
            "max_margin": float(np.max(margins)) if margins else None,
        }

    return {
        group: summarize_rows(group_rows)
        for group, group_rows in sorted(groups.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/gemma-2-2b")
    parser.add_argument("--repair-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cards/MC001G"))
    parser.add_argument("--artifact-prefix", default="mc001g_gemma2_2b_repair")
    parser.add_argument("--render-mode", choices=["raw", "chat"], default="raw")
    parser.add_argument("--bin-width", type=float, default=0.5)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--wrong-layer", type=int, default=13)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    args = parser.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    repair = load_json(args.repair_result)
    discovery_rows, eval_rows = prepare_rows(repair["records"], args.bin_width)

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
    scored_discovery, train_hidden = score_rows(
        model,
        tokenizer,
        option_ids,
        discovery_rows,
        args.render_mode,
        collect_layer=args.layer,
    )
    assert train_hidden is not None
    direction, direction_norm = build_direction(scored_discovery, train_hidden)
    random_direction = rng.normal(size=direction.shape).astype(np.float32)
    random_direction *= direction_norm / float(np.linalg.norm(random_direction))

    direction_tensor = torch.tensor(direction)
    random_tensor = torch.tensor(random_direction)

    arms: list[dict[str, Any]] = [{"name": "baseline", "layer": None, "vector": None, "alpha": 0.0}]
    for alpha in args.alphas:
        arms.append({"name": f"layer{args.layer}_alpha{alpha:g}", "layer": args.layer, "vector": direction_tensor, "alpha": alpha})
    arms.append({"name": f"layer{args.layer}_signflip_alpha1", "layer": args.layer, "vector": direction_tensor, "alpha": -1.0})
    arms.append({"name": f"layer{args.layer}_random_alpha1", "layer": args.layer, "vector": random_tensor, "alpha": 1.0})
    arms.append({"name": f"wrong_layer{args.wrong_layer}_alpha1", "layer": args.wrong_layer, "vector": direction_tensor, "alpha": 1.0})

    arm_outputs: dict[str, Any] = {}
    for arm in arms:
        rows, _ = score_rows(
            model,
            tokenizer,
            option_ids,
            eval_rows,
            args.render_mode,
            layer_index=arm["layer"],
            vector=arm["vector"],
            alpha=arm["alpha"],
        )
        arm_outputs[arm["name"]] = {
            "arm": {k: v for k, v in arm.items() if k != "vector"},
            "summary": summarize(rows),
            "records": rows,
        }
        print(f"completed {arm['name']}")

    elapsed = time.time() - started
    result = {
        "card_id": "MC001G",
        "run_type": f"{args.artifact_prefix}_matched_layer{args.layer}_intervention",
        "model_id": args.model_id,
        "repair_result": str(args.repair_result),
        "render_mode": args.render_mode,
        "bin_width": args.bin_width,
        "layer": args.layer,
        "wrong_layer": args.wrong_layer,
        "alphas": args.alphas,
        "direction_norm": direction_norm,
        "discovery_rows": scored_discovery,
        "eval_row_count": len(eval_rows),
        "eval_group_counts": dict(sorted(Counter(row["eval_group"] for row in eval_rows).items())),
        "elapsed_s": elapsed,
        "arms": arm_outputs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_matched_layer{args.layer}_intervention_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)
    compact = {
        "direction_norm": direction_norm,
        "eval_group_counts": result["eval_group_counts"],
        "arms": {name: arm["summary"] for name, arm in arm_outputs.items()},
        "output_path": str(output_path),
    }
    print(json.dumps(compact, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
