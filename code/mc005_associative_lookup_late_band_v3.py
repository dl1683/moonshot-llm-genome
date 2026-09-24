#!/usr/bin/env python
"""MC005 V3 late-band and layout-holdout source-edge audit."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc001_qwen3_controlled_v10_head_localization import (
    install_source_mask_hook,
    set_eager_attention,
    tokenizer_padding_side,
)
from mc005_associative_lookup_source_edge import (
    CARD_ID,
    KEY_WORDS,
    MODEL_ID,
    RESULT_DIR,
    VALUE_WORDS,
    first_token_id,
    score_rows,
    summarize_arm,
    valid_words,
)


BANDS = {
    "early_0_6": list(range(0, 7)),
    "mid_7_13": list(range(7, 14)),
    "signature_14_18": list(range(14, 19)),
    "late_19_23": list(range(19, 24)),
    "late_24_27": list(range(24, 28)),
    "late_20_26": list(range(20, 27)),
}

LATE_BANDS = {"late_19_23", "late_24_27", "late_20_26"}
EARLIER_BANDS = {"early_0_6", "mid_7_13", "signature_14_18"}

HEAD_GROUPS = {
    "all_heads": None,
    "lower_half_heads": list(range(0, 8)),
    "upper_half_heads": list(range(8, 16)),
    "even_heads": list(range(0, 16, 2)),
    "odd_heads": list(range(1, 16, 2)),
}


def split_and_layout(row_index: int) -> tuple[str, str]:
    mod = row_index % 4
    if mod in {0, 1}:
        return "discovery", "dash_colon"
    if mod == 2:
        return "holdout", "dash_colon"
    return "layout_holdout", "arrow"


def render_prompt(pairs: list[tuple[str, str]], query_key: str, layout: str) -> tuple[str, list[dict[str, Any]]]:
    value_spans: list[dict[str, Any]] = []
    if layout == "dash_colon":
        lines = ["Reference pairs:"]
        cursor = len(lines[0]) + 1
        for pair_index, (key, value) in enumerate(pairs):
            line = f"- {key}: {value}"
            value_start = cursor + line.index(value)
            value_spans.append({"pair_index": pair_index, "value": value, "char_start": value_start})
            lines.append(line)
            cursor += len(line) + 1
        query_line = f"- {query_key}:"
        lines.append(query_line)
        return "\n".join(lines), value_spans
    if layout == "arrow":
        lines = ["Lookup table:"]
        cursor = len(lines[0]) + 1
        for pair_index, (key, value) in enumerate(pairs):
            line = f"{key} -> {value}"
            value_start = cursor + line.index(value)
            value_spans.append({"pair_index": pair_index, "value": value, "char_start": value_start})
            lines.append(line)
            cursor += len(line) + 1
        query_line = f"{query_key} ->"
        lines.append(query_line)
        return "\n".join(lines), value_spans
    raise ValueError(f"unknown layout {layout!r}")


def token_position_for_char(offsets: list[list[int]], char_index: int) -> int:
    candidates = [index for index, (left, right) in enumerate(offsets) if left <= char_index < right]
    if not candidates:
        raise ValueError(f"could not find token for char index {char_index}")
    return int(candidates[0])


def final_query_position(prompt: str, offsets: list[list[int]], query_key: str, layout: str) -> int:
    needle = f"- {query_key}:" if layout == "dash_colon" else f"{query_key} ->"
    char_index = prompt.rindex(needle) + needle.index(query_key)
    return token_position_for_char(offsets, char_index)


def build_rows_v3(tokenizer: Any, row_count: int, pair_count: int, seed: int) -> list[dict[str, Any]]:
    keys = valid_words(tokenizer, KEY_WORDS)
    values = valid_words(tokenizer, VALUE_WORDS)
    if len(keys) < pair_count or len(values) < pair_count:
        raise ValueError(f"not enough single-token words: keys={len(keys)} values={len(values)}")

    rows: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for row_index in range(row_count):
        local = random.Random(rng.randint(0, 10_000_000) + row_index)
        row_keys = local.sample(keys, pair_count)
        row_values = local.sample(values, pair_count)
        pairs = list(zip(row_keys, row_values, strict=True))
        query_pair_index = row_index % pair_count
        distractor_pair_index = (query_pair_index + 1 + (row_index % (pair_count - 1))) % pair_count
        if distractor_pair_index == query_pair_index:
            distractor_pair_index = (query_pair_index + 1) % pair_count
        other_value_indices = [
            index for index in range(pair_count) if index not in {query_pair_index, distractor_pair_index}
        ]
        random_value_index = other_value_indices[row_index % len(other_value_indices)]
        query_key, target_value = pairs[query_pair_index]
        _, distractor_value = pairs[distractor_pair_index]
        split, layout = split_and_layout(row_index)
        prompt, value_spans = render_prompt(pairs, query_key, layout)
        offset_enc = tokenizer(prompt, return_offsets_mapping=True)
        offsets = [[int(left), int(right)] for left, right in offset_enc["offset_mapping"]]
        input_ids = [int(value) for value in offset_enc["input_ids"]]
        value_positions: dict[str, int] = {}
        for span in value_spans:
            pair_index = int(span["pair_index"])
            value = str(span["value"])
            pos = token_position_for_char(offsets, int(span["char_start"]))
            value_positions[str(pair_index)] = int(pos)
        rows.append(
            {
                "id": f"mc005_v3_row_{row_index:03d}",
                "source_id": f"lookup_v3_{row_index:03d}",
                "split": split,
                "layout": layout,
                "pairs": [{"key": key, "value": value} for key, value in pairs],
                "pair_count": pair_count,
                "query_pair_index": query_pair_index,
                "distractor_pair_index": distractor_pair_index,
                "random_value_index": random_value_index,
                "query_key": query_key,
                "target_value": target_value,
                "distractor_value": distractor_value,
                "random_value": pairs[random_value_index][1],
                "target_token_id": first_token_id(tokenizer, target_value),
                "distractor_token_id": first_token_id(tokenizer, distractor_value),
                "positions": {
                    "target_value": value_positions[str(query_pair_index)],
                    "distractor_value": value_positions[str(distractor_pair_index)],
                    "random_value": value_positions[str(random_value_index)],
                    "final_query_key": final_query_position(prompt, offsets, query_key, layout),
                },
                "target_before_distractor": query_pair_index < distractor_pair_index,
                "query_pair_low_half": query_pair_index < (pair_count / 2),
                "rendered_prompt": prompt,
                "token_length": len(input_ids),
            }
        )
    return rows


def shifted_positions_for_batch(rows: list[dict[str, Any]], source_key: str, seq_lens: list[int], max_len: int) -> list[list[int]]:
    shifted: list[list[int]] = []
    for row, seq_len in zip(rows, seq_lens, strict=True):
        shifted.append([int(row["positions"][source_key]) + (max_len - seq_len)])
    return shifted


def score_rows_band(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    band_layers: list[int],
    source_key: str,
    head_indices: list[int] | None = None,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    heads = head_indices or list(range(model.config.num_attention_heads))
    with tokenizer_padding_side(tokenizer, "left"):
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            inputs = tokenizer([row["rendered_prompt"] for row in batch], return_tensors="pt", padding=True).to(
                model.device
            )
            seq_lens = [int(value) for value in inputs["attention_mask"].sum(dim=1).detach().cpu().tolist()]
            max_len = int(inputs["input_ids"].shape[-1])
            shifted = shifted_positions_for_batch(batch, source_key, seq_lens, max_len)
            handles = [
                install_source_mask_hook(model, int(layer), heads, shifted)
                for layer in band_layers
            ]
            try:
                with torch.inference_mode():
                    out = model(**inputs, use_cache=False, logits_to_keep=1)
            finally:
                for handle in handles:
                    handle.remove()
            for index, row in enumerate(batch):
                logits = out.logits[index, -1, :].detach().float()
                target_logit = float(logits[int(row["target_token_id"])])
                distractor_logit = float(logits[int(row["distractor_token_id"])])
                margin = target_logit - distractor_logit
                features[row["id"]] = {
                    "target_logit": target_logit,
                    "distractor_logit": distractor_logit,
                    "target_minus_distractor_margin": margin,
                    "target_wins": margin > 0.0,
                }
    return features


def audit_band_source(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    baseline: dict[str, dict[str, Any]],
    batch_size: int,
    source_key: str,
) -> dict[str, Any]:
    results = {}
    for name, layers in BANDS.items():
        arm = score_rows_band(rows, tokenizer, model, batch_size, layers, source_key)
        results[name] = {"layers": layers, **summarize_arm(rows, baseline, arm)}
        print(
            f"[{rows[0]['split'] if rows else 'empty'} {source_key} {name}] "
            f"mean_delta={results[name]['mean_delta']:.4f} "
            f"win_loss={results[name]['target_win_loss']}"
        )
    return results


def audit_head_groups(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    baseline: dict[str, dict[str, Any]],
    batch_size: int,
    selected_band: str,
) -> dict[str, Any]:
    results = {}
    for name, group_heads in HEAD_GROUPS.items():
        arm = score_rows_band(
            rows,
            tokenizer,
            model,
            batch_size,
            BANDS[selected_band],
            "target_value",
            head_indices=group_heads,
        )
        results[name] = {
            "heads": group_heads if group_heads is not None else "all",
            **summarize_arm(rows, baseline, arm),
        }
        print(
            f"[{rows[0]['split'] if rows else 'empty'} head-group {selected_band} {name}] "
            f"mean_delta={results[name]['mean_delta']:.4f} "
            f"win_loss={results[name]['target_win_loss']}"
        )
    return results


def clean_split_rows(clean_rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in clean_rows if row["split"] == split]


def select_band(discovery_target_results: dict[str, Any]) -> str:
    return min(
        discovery_target_results,
        key=lambda name: (
            discovery_target_results[name]["mean_delta"],
            -discovery_target_results[name]["target_win_loss"],
        ),
    )


def split_criteria(
    selected_target: dict[str, Any],
    selected_distractor: dict[str, Any],
    selected_random: dict[str, Any],
    earlier_controls: dict[str, Any],
    prefix: str,
) -> dict[str, bool]:
    return {
        f"{prefix}_selected_target_reduces_margin_by_1_0": selected_target["mean_delta"] <= -1.0,
        f"{prefix}_selected_target_flips_at_least_3": selected_target["target_win_loss"] >= 3,
        f"{prefix}_selected_target_beats_distractor_by_0_50": selected_target["mean_delta"]
        <= selected_distractor["mean_delta"] - 0.50,
        f"{prefix}_selected_target_beats_random_by_0_50": selected_target["mean_delta"]
        <= selected_random["mean_delta"] - 0.50,
        f"{prefix}_selected_target_beats_earlier_bands_by_0_50": all(
            selected_target["mean_delta"] <= control["mean_delta"] - 0.50
            for control in earlier_controls.values()
        ),
    }


def evaluate(
    rows: list[dict[str, Any]],
    clean_rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    discovery_target_results: dict[str, Any],
    holdout_results: dict[str, dict[str, Any]],
    layout_results: dict[str, dict[str, Any]],
    head_group_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    discovery_rows = clean_split_rows(clean_rows, "discovery")
    holdout_rows = clean_split_rows(clean_rows, "holdout")
    layout_rows = clean_split_rows(clean_rows, "layout_holdout")
    selected = select_band(discovery_target_results)

    holdout_target = holdout_results["target"][selected]
    holdout_distractor = holdout_results["distractor"][selected]
    holdout_random = holdout_results["random"][selected]
    layout_target = layout_results["target"][selected]
    layout_distractor = layout_results["distractor"][selected]
    layout_random = layout_results["random"][selected]
    holdout_earlier = {name: holdout_results["target"][name] for name in EARLIER_BANDS}
    layout_earlier = {name: layout_results["target"][name] for name in EARLIER_BANDS}

    criteria = {
        "clean_discovery_rows_at_least_36": len(discovery_rows) >= 36,
        "clean_holdout_rows_at_least_16": len(holdout_rows) >= 16,
        "clean_layout_holdout_rows_at_least_16": len(layout_rows) >= 16,
        "selected_band_is_late": selected in LATE_BANDS,
    }
    criteria.update(
        split_criteria(holdout_target, holdout_distractor, holdout_random, holdout_earlier, "holdout")
    )
    criteria.update(
        split_criteria(layout_target, layout_distractor, layout_random, layout_earlier, "layout")
    )

    behavior_target_wins = sum(1 for row in rows if baseline[row["id"]]["target_wins"])
    return {
        "model_behavior": {
            "rows": len(rows),
            "target_wins": behavior_target_wins,
            "clean_rows": len(clean_rows),
            "discovery_clean_rows": len(discovery_rows),
            "holdout_clean_rows": len(holdout_rows),
            "layout_holdout_clean_rows": len(layout_rows),
        },
        "candidate_bands": {name: layers for name, layers in BANDS.items()},
        "selected_band": selected,
        "selected_band_layers": BANDS[selected],
        "discovery_target_results": discovery_target_results,
        "holdout_results": holdout_results,
        "layout_holdout_results": layout_results,
        "selected_holdout": {
            "target": holdout_target,
            "distractor": holdout_distractor,
            "random": holdout_random,
            "earlier_target_controls": holdout_earlier,
        },
        "selected_layout_holdout": {
            "target": layout_target,
            "distractor": layout_distractor,
            "random": layout_random,
            "earlier_target_controls": layout_earlier,
        },
        "head_group_results": head_group_results,
        "criteria": criteria,
        "passed": all(criteria.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_associative_lookup_late_band_v3")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--row-count", type=int, default=96)
    parser.add_argument("--pair-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

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
    set_eager_attention(model)
    model.eval()

    started = time.time()
    rows = build_rows_v3(tokenizer, args.row_count, args.pair_count, args.seed)
    baseline = score_rows(rows, tokenizer, model, args.batch_size)
    for row in rows:
        row["baseline"] = baseline[row["id"]]
        row["clean"] = bool(baseline[row["id"]]["target_wins"])
    clean_rows = [row for row in rows if row["clean"]]
    split_counts = {split: len(clean_split_rows(clean_rows, split)) for split in ("discovery", "holdout", "layout_holdout")}
    print(f"[behavior] clean={len(clean_rows)}/{len(rows)} splits={split_counts}")

    discovery_rows = clean_split_rows(clean_rows, "discovery")
    holdout_rows = clean_split_rows(clean_rows, "holdout")
    layout_rows = clean_split_rows(clean_rows, "layout_holdout")
    discovery_target = audit_band_source(discovery_rows, tokenizer, model, baseline, args.batch_size, "target_value")
    selected = select_band(discovery_target)
    print(f"[selection] selected_band={selected} layers={BANDS[selected]}")

    holdout_results = {
        "target": audit_band_source(holdout_rows, tokenizer, model, baseline, args.batch_size, "target_value"),
        "distractor": audit_band_source(holdout_rows, tokenizer, model, baseline, args.batch_size, "distractor_value"),
        "random": audit_band_source(holdout_rows, tokenizer, model, baseline, args.batch_size, "random_value"),
    }
    layout_results = {
        "target": audit_band_source(layout_rows, tokenizer, model, baseline, args.batch_size, "target_value"),
        "distractor": audit_band_source(layout_rows, tokenizer, model, baseline, args.batch_size, "distractor_value"),
        "random": audit_band_source(layout_rows, tokenizer, model, baseline, args.batch_size, "random_value"),
    }
    head_group_results = {
        "holdout": audit_head_groups(holdout_rows, tokenizer, model, baseline, args.batch_size, selected),
        "layout_holdout": audit_head_groups(layout_rows, tokenizer, model, baseline, args.batch_size, selected),
    }

    summary = evaluate(
        rows,
        clean_rows,
        baseline,
        discovery_target,
        holdout_results,
        layout_results,
        head_group_results,
    )
    elapsed = time.time() - started
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_late_band_v3",
        "model_id": args.model_id,
        "row_count": args.row_count,
        "pair_count": args.pair_count,
        "seed": args.seed,
        "elapsed_s": elapsed,
        "summary": summary,
        "rows": rows,
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
