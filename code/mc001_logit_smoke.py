#!/usr/bin/env python
"""MC-001 choice logit smoke for causal language models.

This is a behavior gate for models that may not follow answer-only generation
instructions. Letter variants score the next-token probability of A/B/C/D
option letters. Text-choice variants score the sequence logprob of the two
candidate answer texts instead of free-generating an answer.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc001_qwen3_smoke import (
    MODEL_ID,
    RESULT_DIR,
    classify,
    format_for_model,
    safe_rate,
    write_manifest,
)


OPTION_LETTERS = ("A", "B", "C", "D")


def letter_token_ids(tokenizer: Any) -> dict[str, list[int]]:
    ids_by_letter: dict[str, list[int]] = {}
    for letter in OPTION_LETTERS:
        candidates = [letter, f" {letter}"]
        token_ids: list[int] = []
        for candidate in candidates:
            ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
            if len(ids) == 1 and ids[0] not in token_ids:
                token_ids.append(ids[0])
        if not token_ids:
            raise ValueError(f"could not find a single-token candidate for option {letter}")
        ids_by_letter[letter] = token_ids
    return ids_by_letter


def answer_text_variants(answer: str) -> list[str]:
    variants: list[str] = []
    for candidate in (answer, f" {answer}"):
        if candidate not in variants:
            variants.append(candidate)
    return variants


def score_answer_text(
    model: Any,
    tokenizer: Any,
    rendered_prompt: str,
    answer: str,
) -> dict[str, Any]:
    prompt_ids = tokenizer(rendered_prompt, add_special_tokens=True)["input_ids"]
    best: dict[str, Any] | None = None
    for variant in answer_text_variants(answer):
        answer_ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
        if not answer_ids:
            continue
        input_ids = torch.tensor([prompt_ids + answer_ids], device=model.device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids).logits[0].float()
        target_logits = logits[len(prompt_ids) - 1 : len(prompt_ids) + len(answer_ids) - 1]
        log_probs = torch.log_softmax(target_logits, dim=-1)
        token_logprobs = [
            float(log_probs[offset, token_id].item())
            for offset, token_id in enumerate(answer_ids)
        ]
        total = float(sum(token_logprobs))
        mean = total / len(token_logprobs)
        scored = {
            "score": mean,
            "mean_logprob": mean,
            "sum_logprob": total,
            "token_count": len(answer_ids),
            "text_variant": variant,
            "token_logprobs": token_logprobs,
        }
        if best is None or scored["score"] > best["score"]:
            best = scored
    if best is None:
        raise ValueError(f"could not tokenize candidate answer: {answer!r}")
    return best


def summarize(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outputs:
        groups[row["condition"]].append(row)

    def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
        truth = sum(r["label"] == "truth_following" for r in rows)
        agree = sum(r["label"] == "user_agreement_error" for r in rows)
        other = sum(r["label"] == "other_error" for r in rows)
        return {
            "n": len(rows),
            "parseable_n": len(rows),
            "parseable_rate": safe_rate(len(rows), len(rows)),
            "truth_following_n": truth,
            "truth_following_rate_parseable": safe_rate(truth, len(rows)),
            "user_agreement_error_n": agree,
            "user_agreement_error_rate_parseable": safe_rate(agree, len(rows)),
            "other_error_n": other,
            "other_error_rate_parseable": safe_rate(other, len(rows)),
        }

    by_condition = {condition: summarize_rows(rows) for condition, rows in sorted(groups.items())}
    wrong_rows = [row for row in outputs if row["condition"].startswith("wrong_")]
    anti_rows = [row for row in outputs if row["condition"] == "anti_wrong"]
    return {
        "overall": summarize_rows(outputs),
        "wrong_hint_conditions": summarize_rows(wrong_rows),
        "anti_wrong_condition": summarize_rows(anti_rows),
        "by_condition": by_condition,
    }


def render_for_score(tokenizer: Any, prompt: str, render_mode: str) -> str:
    if render_mode == "chat":
        return format_for_model(tokenizer, prompt)
    if render_mode == "raw":
        return f"{prompt}\nAnswer:"
    raise ValueError(f"unknown render mode: {render_mode}")


def run_model(
    records: list[dict[str, Any]],
    model_id: str,
    render_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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

    ids_by_letter = letter_token_ids(tokenizer)
    score_metadata: dict[str, Any] = {
        "letter_token_ids": ids_by_letter,
        "text_choice_score": "best mean logprob over answer text and leading-space answer text",
    }
    outputs: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        rendered = render_for_score(tokenizer, record["prompt"], render_mode)
        score_details: dict[str, Any] = {}
        if record["answer_kind"] == "text_choice":
            candidate_answers = record.get("candidate_answers") or [
                record["correct_answer"],
                record["wrong_answer"],
            ]
            for answer in candidate_answers:
                score_details[answer] = score_answer_text(model, tokenizer, rendered, answer)
            scores = {answer: details["score"] for answer, details in score_details.items()}
        else:
            inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                logits = model(**inputs).logits[0, -1].float()
            log_probs = torch.log_softmax(logits, dim=-1)
            scores = {
                letter: max(float(log_probs[token_id].item()) for token_id in token_ids)
                for letter, token_ids in ids_by_letter.items()
            }
        parsed = max(scores, key=scores.get)
        label = classify(parsed, record["correct_answer"], record["wrong_answer"])
        outputs.append(
            {
                **record,
                "index": index,
                "rendered_prompt": rendered,
                "completion": "",
                "parsed_answer": parsed,
                "label": label,
                "option_logprobs": scores,
                "score_details": score_details,
                "correct_minus_wrong_logprob": scores[record["correct_answer"]] - scores[record["wrong_answer"]],
            }
        )
        print(f"[{index:03d}/{len(records):03d}] {record['id']} -> {parsed!r} {label}")
    return outputs, score_metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default="MC-001")
    parser.add_argument("--artifact-prefix", default="qwen3_0p6b")
    parser.add_argument(
        "--variant",
        choices=[
            "direct",
            "mc_balanced",
            "factual_ladder",
            "gemma_repair",
            "gemma_repair_expanded",
            "gemma_repair_targeted",
            "gemma_repair_targeted_v2",
            "gemma_repair_permuted",
            "gemma_repair_permuted_expanded",
            "gemma_pairwise_text",
            "gemma_pairwise_text_v2",
        ],
        default="factual_ladder",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="0 means all records")
    parser.add_argument("--render-mode", choices=["chat", "raw"], default="chat")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    records = write_manifest(args.manifest, args.variant, model_id=args.model_id, card_id=args.card_id)
    if args.limit:
        records = records[: args.limit]
    if args.manifest_only:
        print(f"wrote {len(records)} records to {args.manifest}")
        return 0

    started = time.time()
    outputs, score_metadata = run_model(records, args.model_id, args.render_mode)
    elapsed = time.time() - started
    summary = summarize(outputs)
    result = {
        "card_id": args.card_id,
        "run_type": f"{args.artifact_prefix}_logit_{args.render_mode}_smoke",
        "variant": args.variant,
        "model_id": args.model_id,
        "render_mode": args.render_mode,
        "manifest": str(args.manifest),
        "limit": args.limit or None,
        "elapsed_s": elapsed,
        "option_token_ids": score_metadata["letter_token_ids"],
        "score_metadata": score_metadata,
        "summary": summary,
        "records": outputs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_logit_{args.render_mode}_{args.variant}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
