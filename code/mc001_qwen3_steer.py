#!/usr/bin/env python
"""Activation-direction smoke intervention for MC-001 Qwen3-0.6B."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc001_qwen3_smoke import classify, parse_answer, summarize


MODEL_ID = "Qwen/Qwen3-0.6B"
RESULT_DIR = Path("results/cards/MC001")


def latest_smoke_result() -> Path:
    candidates = sorted(RESULT_DIR.glob("mc001_qwen3_0p6b_smoke_factual_ladder_*.json"))
    if not candidates:
        raise FileNotFoundError("no factual_ladder smoke result found")
    return candidates[-1]


def base_id(record_id: str) -> str:
    return record_id.split("__", 1)[0]


def load_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for record in data["records"]:
        if record["condition"] in {"no_hint", "correct_hint"}:
            continue
        if record["label"] == "other_error":
            continue
        rows.append(record)
    return rows


def split_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = sorted({base_id(record["id"]) for record in records})
    train_groups = set(groups[::2])
    train = [record for record in records if base_id(record["id"]) in train_groups]
    eval_rows = [record for record in records if base_id(record["id"]) not in train_groups]
    return train, eval_rows


def load_model(model_id: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def layer_module_for_hidden_index(model: Any, hidden_index: int) -> Any:
    if hidden_index <= 0:
        raise ValueError("hidden_index must be >= 1 to map onto a transformer block output")
    return model.model.layers[hidden_index - 1]


def extract_layer_vectors(records: list[dict[str, Any]], tokenizer: Any, model: Any, hidden_index: int) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for index, record in enumerate(records, start=1):
        inputs = tokenizer(record["rendered_prompt"], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        vec = out.hidden_states[hidden_index][0, -1, :].detach().float().cpu().numpy()
        vectors.append(vec)
        print(f"[direction {index:03d}/{len(records):03d}] {record['id']}")
    return np.stack(vectors, axis=0)


def build_direction(train: list[dict[str, Any]], tokenizer: Any, model: Any, hidden_index: int) -> dict[str, Any]:
    vectors = extract_layer_vectors(train, tokenizer, model, hidden_index)
    y = np.array([1 if record["label"] == "user_agreement_error" else 0 for record in train])
    truth = vectors[y == 0]
    agree = vectors[y == 1]
    if len(truth) == 0 or len(agree) == 0:
        raise ValueError("train split needs both truth-following and agreement examples")
    raw = truth.mean(axis=0) - agree.mean(axis=0)
    raw_norm = float(np.linalg.norm(raw))
    unit = raw / max(raw_norm, 1e-8)
    median_hidden_norm = float(np.median(np.linalg.norm(vectors, axis=1)))
    return {
        "direction": unit.astype("float32"),
        "raw_norm": raw_norm,
        "median_hidden_norm": median_hidden_norm,
        "train_positive_rate": float(y.mean()),
        "train_n": len(train),
    }


def generate_with_steer(
    record: dict[str, Any],
    tokenizer: Any,
    model: Any,
    hidden_index: int,
    direction: torch.Tensor,
    alpha: float,
    median_hidden_norm: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    delta = direction.to(model.device, dtype=model.dtype) * (alpha * median_hidden_norm)

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0].clone()
            hidden[:, -1, :] = hidden[:, -1, :] + delta
            return (hidden,) + output[1:]
        hidden = output.clone()
        hidden[:, -1, :] = hidden[:, -1, :] + delta
        return hidden

    handle = layer_module_for_hidden_index(model, hidden_index).register_forward_hook(hook)
    try:
        inputs = tokenizer(record["rendered_prompt"], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()
    new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
    parsed = parse_answer(completion, record["answer_kind"], record["question"])
    label = classify(parsed, record["correct_answer"], record["wrong_answer"])
    return {
        **record,
        "alpha": alpha,
        "completion": completion,
        "parsed_answer": parsed,
        "label": label,
    }


def summarize_by_alpha(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["alpha"])].append(row)
    return {alpha: summarize(group_rows) for alpha, group_rows in sorted(grouped.items(), key=lambda kv: float(kv[0]))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=None)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--hidden-index", type=int, default=7)
    parser.add_argument("--alphas", default="-0.2,-0.1,0,0.1,0.2,0.4")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    args = parser.parse_args()

    source = args.result or latest_smoke_result()
    records = load_records(source)
    train, eval_rows = split_records(records)
    tokenizer, model = load_model(args.model_id)
    direction_info = build_direction(train, tokenizer, model, args.hidden_index)
    direction = torch.tensor(direction_info["direction"])
    alphas = [float(value) for value in args.alphas.split(",") if value.strip()]

    started = time.time()
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        for index, record in enumerate(eval_rows, start=1):
            row = generate_with_steer(
                record,
                tokenizer,
                model,
                args.hidden_index,
                direction,
                alpha,
                direction_info["median_hidden_norm"],
                args.max_new_tokens,
            )
            rows.append(row)
            print(f"[alpha {alpha:+.2f} {index:03d}/{len(eval_rows):03d}] {record['id']} -> {row['parsed_answer']!r} {row['label']}")

    output = {
        "card_id": "MC-001",
        "run_type": "qwen3_0p6b_activation_direction_smoke",
        "model_id": args.model_id,
        "source_result": str(source),
        "hidden_index": args.hidden_index,
        "hook_layer_index": args.hidden_index - 1,
        "alphas": alphas,
        "train_n": len(train),
        "eval_n_per_alpha": len(eval_rows),
        "direction": {k: v for k, v in direction_info.items() if k != "direction"},
        "elapsed_s": time.time() - started,
        "summary_by_alpha": summarize_by_alpha(rows),
        "records": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"mc001_qwen3_0p6b_steer_factual_ladder_{stamp}.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(output["summary_by_alpha"], indent=2, ensure_ascii=True))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
