#!/usr/bin/env python
"""Head/layer localization audit for MC-001 Qwen3-0.6B.

V9 showed that masking hint source tokens strongly reduces wrong-hint
agreement. V10 asks whether that source-token effect can be localized to a
specific attention layer/head path rather than a coarse full-source ablation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

from mc001_qwen3_controlled import (
    CARD_ID,
    MANIFEST_DIR,
    MODEL_ID,
    RESULT_DIR,
    classify,
    generate_one,
    load_model,
    option_scores,
    option_token_ids,
    parse_answer,
    summarize_many,
    summarize_rows,
)
from mc001_qwen3_controlled_v2 import (
    add_margin_metadata,
    iter_records,
    margin_bin_counts,
    write_manifest,
)
from mc001_qwen3_controlled_v3 import (
    answer_distribution,
    rows_digest,
    score_cell,
    summarize_margin_bins,
)
from mc001_qwen3_controlled_v6_prefill_controls import validation_filters
from mc001_qwen3_controlled_v9_attention_source_ablation import (
    line_span,
    matched_positions,
    random_matched_positions,
    render_row as render_v9_row,
    token_positions_for_span,
)


ValidationFilter = Callable[[dict[str, Any]], bool]


SOURCE_GROUPS = [
    "hint_answer",
    "hint_non_answer",
    "hint_line",
    "answer_instruction_matched_hint_answer",
    "random_matched_hint_answer",
]


def set_eager_attention(model: Any) -> None:
    """Use explicit causal masks so per-head source masks preserve causality."""
    model.config._attn_implementation = "eager"
    if hasattr(model, "model"):
        model.model.config._attn_implementation = "eager"
        for layer in model.model.layers:
            layer.self_attn.config._attn_implementation = "eager"


@contextmanager
def tokenizer_padding_side(tokenizer: Any, side: str) -> Iterable[None]:
    old_side = tokenizer.padding_side
    tokenizer.padding_side = side
    try:
        yield
    finally:
        tokenizer.padding_side = old_side


def augment_path_positions(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    rendered = row["rendered_prompt"]
    positions = {key: list(value) for key, value in row["path_positions"].items()}
    hint_line = set(positions.get("hint_line", []))
    hint_answer = set(positions.get("hint_answer", []))
    positions["hint_non_answer"] = sorted(hint_line - hint_answer)

    answer_span = line_span(rendered, "Answer with only one letter:")
    question_span = line_span(rendered, "Question:")
    answer_instruction = token_positions_for_span(tokenizer, rendered, *answer_span) if answer_span else []
    question = token_positions_for_span(tokenizer, rendered, *question_span) if question_span else []
    all_content = sorted(set(question + answer_instruction + positions.get("hint_line", [])))
    positions["answer_instruction_matched_hint_answer"] = matched_positions(answer_instruction, len(hint_answer))
    positions["random_matched_hint_answer"] = random_matched_positions(
        row["id"],
        all_content,
        hint_line,
        len(hint_answer),
    )
    return {
        **row,
        "variant": "controlled_v10_head_localization",
        "path_positions": positions,
        "path_position_counts": {key: len(value) for key, value in positions.items()},
    }


def render_row(tokenizer: Any, record: dict[str, Any]) -> dict[str, Any]:
    return augment_path_positions(tokenizer, render_v9_row(tokenizer, record))


def shifted_positions_for_batch(rows: list[dict[str, Any]], source_group: str, seq_lens: list[int], max_len: int) -> list[list[int]]:
    shifted: list[list[int]] = []
    for row, seq_len in zip(rows, seq_lens):
        pad = max_len - seq_len
        shifted.append([int(pos) + pad for pos in row["path_positions"].get(source_group, [])])
    return shifted


def install_source_mask_hook(
    model: Any,
    layer_index: int,
    head_indices: list[int],
    source_positions_by_batch: list[list[int]],
) -> Any:
    heads = [int(head) for head in head_indices]

    def hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None:
            raise RuntimeError("V10 requires eager attention with an explicit causal attention_mask")
        if attention_mask.shape[1] == 1:
            mask = attention_mask.expand(
                attention_mask.shape[0],
                module.config.num_attention_heads,
                attention_mask.shape[-2],
                attention_mask.shape[-1],
            ).clone()
        else:
            mask = attention_mask.clone()
        neg = torch.finfo(mask.dtype).min
        query_len = mask.shape[-2]
        key_len = mask.shape[-1]
        query_indices = [query_len - 1] if query_len > 1 else [0]
        for batch_index, positions in enumerate(source_positions_by_batch):
            valid = [int(pos) for pos in positions if 0 <= int(pos) < key_len]
            if not valid:
                continue
            for head in heads:
                for query_index in query_indices:
                    mask[batch_index, head, query_index, valid] = neg
        kwargs["attention_mask"] = mask
        return args, kwargs

    return model.model.layers[layer_index].self_attn.register_forward_pre_hook(hook, with_kwargs=True)


def next_token_features_from_logits(
    logits: torch.Tensor,
    token_ids: dict[str, list[int]],
    row: dict[str, Any],
) -> dict[str, Any]:
    scores = option_scores(logits.detach().float(), token_ids)
    pred = max(scores, key=scores.get)
    return {
        "option_scores": scores,
        "next_token_answer": pred,
        "next_token_label": classify(pred, row["correct_answer"], row["wrong_answer"]),
        "correct_minus_wrong_logit": scores[row["correct_answer"]] - scores[row["wrong_answer"]],
    }


def batched_next_token_features(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    token_ids: dict[str, list[int]],
    batch_size: int,
    source_group: str | None = None,
    layer_index: int | None = None,
    head_indices: list[int] | None = None,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    with tokenizer_padding_side(tokenizer, "left"):
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            texts = [row["rendered_prompt"] for row in batch]
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
            handle = None
            if source_group is not None and layer_index is not None and head_indices is not None:
                seq_lens = [int(value) for value in inputs["attention_mask"].sum(dim=1).detach().cpu().tolist()]
                max_len = int(inputs["input_ids"].shape[-1])
                shifted = shifted_positions_for_batch(batch, source_group, seq_lens, max_len)
                handle = install_source_mask_hook(model, layer_index, head_indices, shifted)
            try:
                with torch.inference_mode():
                    out = model(**inputs, use_cache=False, logits_to_keep=1)
            finally:
                if handle is not None:
                    handle.remove()
            for index, row in enumerate(batch):
                features[row["id"]] = next_token_features_from_logits(out.logits[index, -1, :], token_ids, row)
    return features


def generate_with_source_mask(
    row: dict[str, Any],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    source_group: str,
    layer_index: int,
    head_indices: list[int],
) -> dict[str, Any]:
    source_positions = [list(row["path_positions"].get(source_group, []))]
    handle = install_source_mask_hook(model, layer_index, head_indices, source_positions)
    try:
        inputs = tokenizer(row["rendered_prompt"], return_tensors="pt").to(model.device)
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
    parsed = parse_answer(completion, row)
    label = classify(parsed, row["correct_answer"], row["wrong_answer"])
    return {"completion": completion, "parsed_answer": parsed, "label": label}


def summarize_filtered(arm_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for filter_name, predicate in validation_filters().items():
        output[filter_name] = {
            arm: summarize_many([row for row in rows if predicate(row)])
            for arm, rows in sorted(arm_rows.items())
        }
    output["hard_plus_side_eval"] = {
        arm: summarize_many(rows)
        for arm, rows in sorted(arm_rows.items())
    }
    return output


def summarize_screen_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_rows(rows)
    deltas = [float(row["correct_minus_wrong_delta"]) for row in rows]
    summary["mean_correct_minus_wrong_delta"] = float(np.mean(deltas)) if deltas else None
    summary["median_correct_minus_wrong_delta"] = float(np.median(deltas)) if deltas else None
    summary["source_position_nonzero_n"] = sum(int(row.get("source_position_count", 0)) > 0 for row in rows)
    return summary


def screen_rows_for_features(
    rows: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    source_group: str,
    layer_index: int,
    head_indices: list[int],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        feature = features[row["id"]]
        margin = float(feature["correct_minus_wrong_logit"])
        out.append(
            {
                **row,
                "arm": "next_token_screen",
                "source_group": source_group,
                "mask_layer": layer_index,
                "mask_heads": head_indices,
                "source_position_count": len(row["path_positions"].get(source_group, [])),
                "option_scores": feature["option_scores"],
                "next_token_answer": feature["next_token_answer"],
                "parsed_answer": feature["next_token_answer"],
                "label": feature["next_token_label"],
                "correct_minus_wrong_logit": margin,
                "correct_minus_wrong_delta": margin - float(row["baseline_correct_minus_wrong_logit"]),
            }
        )
    return out


def screen_score(summary: dict[str, Any]) -> tuple[float, float, float]:
    truth = float(summary["truth_following_n"])
    agreement = float(summary["user_agreement_error_n"])
    delta = float(summary["mean_correct_minus_wrong_delta"] or 0.0)
    return (truth, -agreement, delta)


def run_head_screen(
    hard_rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    token_ids: dict[str, list[int]],
    batch_size: int,
    source_group: str,
    max_layers: int = 0,
    max_heads: int = 0,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    num_layers = max_layers or model.config.num_hidden_layers
    num_heads = max_heads or model.config.num_attention_heads
    for layer_index in range(num_layers):
        for head_index in range(num_heads):
            features = batched_next_token_features(
                hard_rows,
                tokenizer,
                model,
                token_ids,
                batch_size,
                source_group=source_group,
                layer_index=layer_index,
                head_indices=[head_index],
            )
            rows = screen_rows_for_features(hard_rows, features, source_group, layer_index, [head_index])
            summary = summarize_screen_rows(rows)
            records.append(
                {
                    "source_group": source_group,
                    "mask_layer": layer_index,
                    "mask_heads": [head_index],
                    "summary": summary,
                }
            )
            print(
                f"[v10 head screen L{layer_index:02d} H{head_index:02d} {source_group}] "
                f"truth={summary['truth_following_n']}/{summary['n']} "
                f"agree={summary['user_agreement_error_n']} "
                f"delta={summary['mean_correct_minus_wrong_delta']:.3f}"
            )
    return records


def run_layer_group_screen(
    hard_rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    token_ids: dict[str, list[int]],
    batch_size: int,
    source_groups: list[str],
    max_layers: int = 0,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    all_heads = list(range(model.config.num_attention_heads))
    for source_group in source_groups:
        for layer_index in range(max_layers or model.config.num_hidden_layers):
            features = batched_next_token_features(
                hard_rows,
                tokenizer,
                model,
                token_ids,
                batch_size,
                source_group=source_group,
                layer_index=layer_index,
                head_indices=all_heads,
            )
            rows = screen_rows_for_features(hard_rows, features, source_group, layer_index, all_heads)
            summary = summarize_screen_rows(rows)
            records.append(
                {
                    "source_group": source_group,
                    "mask_layer": layer_index,
                    "mask_heads": all_heads,
                    "summary": summary,
                }
            )
            print(
                f"[v10 layer screen L{layer_index:02d} all-heads {source_group}] "
                f"truth={summary['truth_following_n']}/{summary['n']} "
                f"agree={summary['user_agreement_error_n']} "
                f"delta={summary['mean_correct_minus_wrong_delta']:.3f}"
            )
    return records


def select_top_heads(records: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    ranked = sorted(records, key=lambda record: screen_score(record["summary"]), reverse=True)
    return ranked[:top_n]


def select_top_layers_by_group(records: list[dict[str, Any]], top_n_per_group: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["source_group"]].append(record)
    selected: list[dict[str, Any]] = []
    for source_group, group_records in sorted(grouped.items()):
        ranked = sorted(group_records, key=lambda record: screen_score(record["summary"]), reverse=True)
        selected.extend(ranked[:top_n_per_group])
    return selected


def spec_name(spec: dict[str, Any]) -> str:
    heads = spec["mask_heads"]
    if len(heads) == 1:
        head_part = f"H{heads[0]:02d}"
    else:
        head_part = "all_heads"
    return f"{spec['source_group']}__L{spec['mask_layer']:02d}__{head_part}"


def dedupe_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int, tuple[int, ...]]] = set()
    out: list[dict[str, Any]] = []
    for spec in specs:
        key = (spec["source_group"], int(spec["mask_layer"]), tuple(int(head) for head in spec["mask_heads"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(spec)
    return out


def run_generation_validation(
    eval_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    arm_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        base = baseline_by_id[row["id"]]
        arm_rows["baseline"].append({**row, **{key: base[key] for key in ["completion", "parsed_answer", "label"]}, "arm": "baseline"})
    for spec in specs:
        arm_name = spec_name(spec)
        for index, row in enumerate(eval_rows, start=1):
            result = generate_with_source_mask(
                row,
                tokenizer,
                model,
                max_new_tokens,
                spec["source_group"],
                int(spec["mask_layer"]),
                [int(head) for head in spec["mask_heads"]],
            )
            arm_row = {
                **row,
                **result,
                "arm": arm_name,
                "source_group": spec["source_group"],
                "mask_layer": int(spec["mask_layer"]),
                "mask_heads": [int(head) for head in spec["mask_heads"]],
                "source_position_count": len(row["path_positions"].get(spec["source_group"], [])),
            }
            arm_rows[arm_name].append(arm_row)
            print(f"[v10 gen {arm_name} {index:03d}/{len(eval_rows):03d}] {row['id']} -> {arm_row['parsed_answer']!r} {arm_row['label']}")
    return {
        "arm_settings": [{"name": "baseline", "kind": "baseline"}] + [
            {
                "name": spec_name(spec),
                "kind": "targeted_source_mask",
                "source_group": spec["source_group"],
                "mask_layer": int(spec["mask_layer"]),
                "mask_heads": [int(head) for head in spec["mask_heads"]],
                "screen_summary": spec.get("summary"),
            }
            for spec in specs
        ],
        "summary_by_arm": {key: summarize_many(value) for key, value in sorted(arm_rows.items())},
        "summary_by_arm_margin_bin": {key: summarize_margin_bins(value) for key, value in sorted(arm_rows.items())},
        "summary_filtered": summarize_filtered(arm_rows),
        "score_by_arm": {key: score_cell(value) for key, value in sorted(arm_rows.items())},
        "answer_distribution_by_arm": {key: answer_distribution(value) for key, value in sorted(arm_rows.items())},
        "records": [row for rows in arm_rows.values() for row in rows],
    }


def split_condition_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(f"{row['split']}::{row['condition']}" for row in rows).items()))


def position_count_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({key for row in rows for key in row["path_position_counts"]})
    output: dict[str, Any] = {}
    for group in groups:
        counts = [int(row["path_position_counts"].get(group, 0)) for row in rows]
        output[group] = {
            "min": min(counts) if counts else None,
            "max": max(counts) if counts else None,
            "mean": float(np.mean(counts)) if counts else None,
            "nonzero_n": sum(count > 0 for count in counts),
        }
    return output


def compact_screen_record(record: dict[str, Any]) -> dict[str, Any]:
    summary = record["summary"]
    return {
        "source_group": record["source_group"],
        "mask_layer": int(record["mask_layer"]),
        "mask_heads": [int(head) for head in record["mask_heads"]],
        "truth_following_n": int(summary["truth_following_n"]),
        "user_agreement_error_n": int(summary["user_agreement_error_n"]),
        "other_error_n": int(summary["other_error_n"]),
        "mean_correct_minus_wrong_delta": summary["mean_correct_minus_wrong_delta"],
        "median_correct_minus_wrong_delta": summary["median_correct_minus_wrong_delta"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DIR / "mc001_qwen3_0p6b_controlled_v10_head_localization_manifest.jsonl")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-heads", type=int, default=8)
    parser.add_argument("--top-layers-per-group", type=int, default=2)
    parser.add_argument("--max-screen-layers", type=int, default=0)
    parser.add_argument("--max-screen-heads", type=int, default=0)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--screen-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(10)
    random.seed(10)
    np.random.seed(10)

    records = iter_records()
    digest = write_manifest(args.manifest, records)
    if args.manifest_only:
        print(f"wrote {len(records)} records to {args.manifest}")
        print(f"sha256={digest}")
        return 0

    tokenizer, model = load_model(args.model_id)
    set_eager_attention(model)
    token_ids = option_token_ids(tokenizer)
    started = time.time()

    rendered_rows = [render_row(tokenizer, record) for record in records]
    baseline_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rendered_rows, start=1):
        result = generate_one(row, tokenizer, model, args.max_new_tokens)
        baseline_rows.append({**row, **result})
        print(f"[v10 baseline {index:03d}/{len(rendered_rows):03d}] {row['id']} -> {result['parsed_answer']!r} {result['label']}")

    baseline_features = batched_next_token_features(
        baseline_rows,
        tokenizer,
        model,
        token_ids,
        args.batch_size,
    )
    baseline_rows = add_margin_metadata(baseline_rows, baseline_features)

    validation_rows = [
        row
        for row in baseline_rows
        if row["split"] in {"calibration", "holdout", "paraphrase_holdout"}
    ]
    hard_rows = [
        row
        for row in validation_rows
        if row["condition"].startswith("wrong_") and row["baseline_margin_bin"] == "agreement_favored"
    ]
    side_rows = [
        row
        for row in validation_rows
        if row["condition"] in {"no_hint", "correct_hint", "anti_wrong"}
    ]
    eval_rows_by_id = {row["id"]: row for row in hard_rows + side_rows}
    eval_rows = list(eval_rows_by_id.values())

    print(f"[v10 selection] validation={len(validation_rows)} hard={len(hard_rows)} side={len(side_rows)} eval={len(eval_rows)}")

    head_screen = run_head_screen(
        hard_rows,
        tokenizer,
        model,
        token_ids,
        args.batch_size,
        source_group="hint_answer",
        max_layers=args.max_screen_layers,
        max_heads=args.max_screen_heads,
    )
    layer_group_screen = run_layer_group_screen(
        hard_rows,
        tokenizer,
        model,
        token_ids,
        args.batch_size,
        source_groups=SOURCE_GROUPS,
        max_layers=args.max_screen_layers,
    )

    top_head_specs = select_top_heads(head_screen, args.top_heads)
    top_layer_specs = select_top_layers_by_group(layer_group_screen, args.top_layers_per_group)
    selected_specs = dedupe_specs(top_head_specs + top_layer_specs)

    generation = None
    if not args.screen_only:
        generation = run_generation_validation(
            eval_rows,
            baseline_rows,
            selected_specs,
            tokenizer,
            model,
            args.max_new_tokens,
        )

    discovery = [row for row in baseline_rows if row["split"] == "discovery"]
    calibration = [row for row in baseline_rows if row["split"] == "calibration"]
    holdout = [row for row in baseline_rows if row["split"] == "holdout"]
    paraphrase = [row for row in baseline_rows if row["split"] == "paraphrase_holdout"]

    output = {
        "card_id": CARD_ID,
        "run_type": "qwen3_0p6b_controlled_v10_head_localization",
        "model_id": args.model_id,
        "attention_implementation": "eager",
        "manifest": str(args.manifest),
        "manifest_sha256": digest,
        "records_digest": rows_digest(records),
        "records_n": len(records),
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "elapsed_s": time.time() - started,
        "baseline_summary": summarize_many(baseline_rows),
        "position_count_summary": position_count_summary(baseline_rows),
        "candidate_counts": {
            "discovery": len(discovery),
            "calibration": len(calibration),
            "holdout": len(holdout),
            "paraphrase_holdout": len(paraphrase),
            "validation_rows": len(validation_rows),
            "hard_agreement_favored_wrong_hint_rows": len(hard_rows),
            "side_effect_rows": len(side_rows),
            "generation_eval_rows": len(eval_rows),
        },
        "candidate_margin_bin_counts": {
            "discovery": margin_bin_counts(discovery),
            "calibration": margin_bin_counts(calibration),
            "holdout": margin_bin_counts(holdout),
            "paraphrase_holdout": margin_bin_counts(paraphrase),
            "validation_wrong_hint": margin_bin_counts([
                row
                for row in validation_rows
                if row["condition"].startswith("wrong_") and row["label"] in {"truth_following", "user_agreement_error"}
            ]),
        },
        "selection_rule": {
            "hard_rows": "validation split, wrong-hint condition, baseline_margin_bin == agreement_favored",
            "side_rows": "validation split, condition in no_hint/correct_hint/anti_wrong",
            "split_condition_counts": split_condition_counts(eval_rows),
        },
        "head_screen": {
            "source_group": "hint_answer",
            "records": head_screen,
            "top": [compact_screen_record(record) for record in top_head_specs],
        },
        "layer_group_screen": {
            "source_groups": SOURCE_GROUPS,
            "records": layer_group_screen,
            "top": [compact_screen_record(record) for record in top_layer_specs],
        },
        "selected_generation_specs": [compact_screen_record(record) for record in selected_specs],
        "generation_validation": generation,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"mc001_qwen3_0p6b_controlled_v10_head_localization_{timestamp}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output_path": str(output_path),
        "baseline": output["baseline_summary"]["overall"],
        "candidate_counts": output["candidate_counts"],
        "head_screen_top": output["head_screen"]["top"],
        "layer_group_screen_top": output["layer_group_screen"]["top"],
        "generation_available": generation is not None,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
