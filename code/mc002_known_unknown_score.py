#!/usr/bin/env python
"""MC002 known-vs-unknown answer-text scoring gate."""

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

from mc002_known_unknown_smoke import (
    CARD_ID,
    MODEL_ID,
    RESULT_DIR,
    Source,
    audit,
    iter_manifest_records,
    render_for_generation,
    structural_check,
)


MANIFEST_DIR = Path("data/cards/MC002")


def entity_name(question: str) -> str:
    match = re.search(r"called\s+([^?]+)\?", question)
    if match:
        return match.group(1).strip()
    match = re.search(r"capital of\s+([^?]+)\?", question)
    if match:
        name = match.group(1).strip()
        return re.sub(r"^(the country|the)\s+", "", name, flags=re.IGNORECASE)
    return question.strip(" ?")


def candidate_answers(record: dict[str, Any]) -> list[dict[str, str]]:
    candidates = [
        {"answer": "UNKNOWN", "label": "abstain", "candidate_type": "unknown"},
        {
            "answer": record["lure_answer"],
            "label": "known_wrong_lure" if record["entity_type"] == "real_country" else "lure_hallucination",
            "candidate_type": "lure",
        },
        {
            "answer": entity_name(record["question"]),
            "label": "known_other_answer" if record["entity_type"] == "real_country" else "other_hallucination",
            "candidate_type": "entity_name",
        },
    ]
    if record["entity_type"] == "real_country":
        candidates.append(
            {
                "answer": record["expected_answer"],
                "label": "known_correct",
                "candidate_type": "expected",
            }
        )
    return candidates


def write_manifest(path: Path, model_id: str, card_id: str) -> list[dict[str, Any]]:
    records = iter_manifest_records(model_id, card_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    return records


def token_logprob(
    model: Any,
    tokenizer: Any,
    rendered_prompt: str,
    answer: str,
) -> dict[str, Any]:
    full_text = f"{rendered_prompt} {answer}"
    prompt_ids = tokenizer(rendered_prompt, return_tensors="pt")["input_ids"][0]
    full_inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    full_ids = full_inputs["input_ids"][0]
    start = int(prompt_ids.shape[0])
    if start >= int(full_ids.shape[0]):
        return {
            "answer": answer,
            "token_count": 0,
            "sum_logprob": -math.inf,
            "mean_logprob": -math.inf,
        }

    with torch.inference_mode():
        logits = model(**full_inputs).logits[0]

    token_ids = full_ids[start:]
    logprobs = torch.log_softmax(logits[start - 1 : -1], dim=-1)
    selected = logprobs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
    sum_logprob = float(selected.sum().detach().cpu())
    token_count = int(token_ids.shape[0])
    return {
        "answer": answer,
        "token_count": token_count,
        "sum_logprob": sum_logprob,
        "mean_logprob": sum_logprob / token_count,
    }


def score_records(
    records: list[dict[str, Any]],
    model_id: str,
    render_mode: str,
) -> list[dict[str, Any]]:
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

    outputs: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        rendered = render_for_generation(tokenizer, record["prompt"], render_mode)
        scored_candidates = []
        for candidate in candidate_answers(record):
            score = token_logprob(model, tokenizer, rendered, candidate["answer"])
            scored_candidates.append({**candidate, **score})
        winner = max(
            scored_candidates,
            key=lambda candidate: (candidate["mean_logprob"], candidate["sum_logprob"], candidate["candidate_type"]),
        )
        output = {
            **record,
            "index": index,
            "rendered_prompt": rendered,
            "label": winner["label"],
            "selected_answer": winner["answer"],
            "selected_candidate_type": winner["candidate_type"],
            "candidate_scores": sorted(scored_candidates, key=lambda row: row["mean_logprob"], reverse=True),
        }
        outputs.append(output)
        print(
            f"[{index:03d}/{len(records):03d}] {record['id']} -> "
            f"{winner['label']} :: {winner['answer']!r} mean_logprob={winner['mean_logprob']:.4f}"
        )
    return outputs


def counter(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"n": len(rows), "label_counts": counter(rows, "label")}


def summarize(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    conditions = sorted({row["condition"] for row in outputs})
    entity_types = sorted({row["entity_type"] for row in outputs})
    by_condition = {
        condition: summarize_rows([row for row in outputs if row["condition"] == condition])
        for condition in conditions
    }
    by_entity_condition = {
        f"{entity_type}::{condition}": summarize_rows(
            [row for row in outputs if row["entity_type"] == entity_type and row["condition"] == condition]
        )
        for entity_type in entity_types
        for condition in conditions
    }
    return {
        "overall": summarize_rows(outputs),
        "by_condition": by_condition,
        "by_entity_condition": by_entity_condition,
        "audit": audit(outputs),
    }


def scoring_structural_check(records: list[dict[str, Any]]) -> dict[str, Any]:
    base = structural_check(records)
    candidate_counts = Counter(len(candidate_answers(record)) for record in records)
    tokenless = [
        record["id"]
        for record in records
        for candidate in candidate_answers(record)
        if not candidate["answer"].strip()
    ]
    checks = {
        **base["checks"],
        "real_records_have_four_candidates": candidate_counts.get(4, 0) == 160,
        "nonce_records_have_three_candidates": candidate_counts.get(3, 0) == 160,
        "candidate_answers_present": not tokenless,
    }
    return {
        **base,
        "checks": checks,
        "passed": all(checks.values()),
        "candidate_count_distribution": dict(sorted(candidate_counts.items())),
        "tokenless_candidates": tokenless,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc002_gemma2_2b_known_unknown_scoring")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--render-mode", choices=["raw", "chat"], default="raw")
    parser.add_argument("--limit", type=int, default=0, help="0 means all records")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    manifest = args.manifest or (MANIFEST_DIR / f"{args.artifact_prefix}_manifest.jsonl")
    records = write_manifest(manifest, args.model_id, args.card_id)
    structure = scoring_structural_check(records)
    if not structure["passed"]:
        print(json.dumps(structure, indent=2, ensure_ascii=True))
        return 2
    if args.limit:
        records = records[: args.limit]
    if args.manifest_only:
        print(json.dumps({"manifest": str(manifest), "records": len(records), "structural_check": structure}, indent=2))
        return 0

    started = time.time()
    outputs = score_records(records, args.model_id, args.render_mode)
    elapsed = time.time() - started
    summary = summarize(outputs)
    result = {
        "card_id": args.card_id,
        "run_type": f"{args.artifact_prefix}_answer_scoring",
        "model_id": args.model_id,
        "render_mode": args.render_mode,
        "manifest": str(manifest),
        "scoring": "mean_logprob_per_candidate_token",
        "limit": args.limit or None,
        "elapsed_s": elapsed,
        "structural_check": structure,
        "summary": summary,
        "records": outputs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_score_{args.render_mode}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)

    print(json.dumps({**summary, "output_path": str(output_path)}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
