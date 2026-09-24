#!/usr/bin/env python
"""Cumulative layer-band source-mask audit for MC-001 Qwen3-0.6B.

V10 found only weak single-head and single-layer localization for the V9
hint-source effect. V11 asks whether a compact layer band, rather than one
isolated layer/head, recovers the coarse source-token intervention.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    option_token_ids,
    parse_answer,
    summarize_many,
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
from mc001_qwen3_controlled_v10_head_localization import (
    batched_next_token_features,
    install_source_mask_hook,
    position_count_summary,
    render_row,
    set_eager_attention,
    split_condition_counts,
    summarize_filtered,
)


@dataclass(frozen=True)
class Band:
    name: str
    layers: tuple[int, ...]


@dataclass(frozen=True)
class SourceBandSpec:
    source_group: str
    band: Band


def clipped_range(start: int, stop_inclusive: int, num_layers: int) -> tuple[int, ...]:
    return tuple(layer for layer in range(start, stop_inclusive + 1) if 0 <= layer < num_layers)


def layer_bands(num_layers: int) -> list[Band]:
    """Fixed MC-001 bands, clipped defensively for non-28-layer smoke targets."""
    candidates = [
        ("all_L00_L27", clipped_range(0, 27, num_layers)),
        ("early_L00_L06", clipped_range(0, 6, num_layers)),
        ("middle_L07_L13", clipped_range(7, 13, num_layers)),
        ("upper_L14_L20", clipped_range(14, 20, num_layers)),
        ("late_L21_L27", clipped_range(21, 27, num_layers)),
        ("single_L22", clipped_range(22, 22, num_layers)),
        ("pair_L21_L22", clipped_range(21, 22, num_layers)),
        ("triad_L17_L21_L22", tuple(layer for layer in (17, 21, 22) if layer < num_layers)),
        ("band_L17_L22", clipped_range(17, 22, num_layers)),
        ("band_L14_L22", clipped_range(14, 22, num_layers)),
        ("core_L20_L23", clipped_range(20, 23, num_layers)),
        ("top8_L20_L27", clipped_range(20, 27, num_layers)),
    ]
    seen: set[tuple[int, ...]] = set()
    bands: list[Band] = []
    for name, layers in candidates:
        if not layers or layers in seen:
            continue
        seen.add(layers)
        bands.append(Band(name, layers))
    return bands


def make_specs(num_layers: int) -> list[SourceBandSpec]:
    bands = {band.name: band for band in layer_bands(num_layers)}
    specs: list[SourceBandSpec] = []

    for band in bands.values():
        specs.append(SourceBandSpec("hint_answer", band))

    diagnostic_band_names = [
        "all_L00_L27",
        "late_L21_L27",
        "band_L17_L22",
        "band_L14_L22",
        "top8_L20_L27",
    ]
    for name in diagnostic_band_names:
        if name in bands:
            specs.append(SourceBandSpec("hint_line", bands[name]))

    non_answer_band_names = ["all_L00_L27", "late_L21_L27", "band_L17_L22"]
    for name in non_answer_band_names:
        if name in bands:
            specs.append(SourceBandSpec("hint_non_answer", bands[name]))

    control_band_names = [
        "all_L00_L27",
        "late_L21_L27",
        "band_L17_L22",
        "core_L20_L23",
        "top8_L20_L27",
    ]
    for source_group in ["answer_instruction_matched_hint_answer", "random_matched_hint_answer"]:
        for name in control_band_names:
            if name in bands:
                specs.append(SourceBandSpec(source_group, bands[name]))

    return specs


def spec_name(spec: SourceBandSpec) -> str:
    return f"{spec.source_group}__{spec.band.name}"


def install_layer_band_hooks(
    model: Any,
    source_positions: list[list[int]],
    layers: tuple[int, ...],
    heads: list[int],
) -> list[Any]:
    return [
        install_source_mask_hook(model, int(layer), heads, source_positions)
        for layer in layers
    ]


def generate_with_layer_band_mask(
    row: dict[str, Any],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    spec: SourceBandSpec,
    heads: list[int],
) -> dict[str, Any]:
    source_positions = [list(row["path_positions"].get(spec.source_group, []))]
    handles: list[Any] = []
    try:
        handles = install_layer_band_hooks(model, source_positions, spec.band.layers, heads)
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
        for handle in handles:
            handle.remove()
    new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
    parsed = parse_answer(completion, row)
    label = classify(parsed, row["correct_answer"], row["wrong_answer"])
    return {"completion": completion, "parsed_answer": parsed, "label": label}


def run_generation_validation(
    eval_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    specs: list[SourceBandSpec],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    heads = list(range(model.config.num_attention_heads))
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    arm_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        base = baseline_by_id[row["id"]]
        arm_rows["baseline"].append({
            **row,
            **{key: base[key] for key in ["completion", "parsed_answer", "label"]},
            "arm": "baseline",
        })

    for spec in specs:
        arm_name = spec_name(spec)
        for index, row in enumerate(eval_rows, start=1):
            result = generate_with_layer_band_mask(
                row,
                tokenizer,
                model,
                max_new_tokens,
                spec,
                heads,
            )
            arm_row = {
                **row,
                **result,
                "arm": arm_name,
                "source_group": spec.source_group,
                "layer_band": spec.band.name,
                "mask_layers": list(spec.band.layers),
                "mask_heads": heads,
                "source_position_count": len(row["path_positions"].get(spec.source_group, [])),
            }
            arm_rows[arm_name].append(arm_row)
            print(f"[v11 gen {arm_name} {index:03d}/{len(eval_rows):03d}] {row['id']} -> {arm_row['parsed_answer']!r} {arm_row['label']}")

    return {
        "arm_settings": [{"name": "baseline", "kind": "baseline"}] + [
            {
                "name": spec_name(spec),
                "kind": "layer_band_source_mask",
                "source_group": spec.source_group,
                "layer_band": spec.band.name,
                "mask_layers": list(spec.band.layers),
                "mask_heads": heads,
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


def compact_arm_summary(generation: dict[str, Any] | None, section: str = "wrong_hint_agreement_favored") -> list[dict[str, Any]]:
    if generation is None:
        return []
    out: list[dict[str, Any]] = []
    filtered = generation["summary_filtered"][section]
    for arm, value in filtered.items():
        overall = value["overall"] if isinstance(value, dict) and "overall" in value else value
        out.append({
            "arm": arm,
            "n": int(overall["n"]),
            "parseable_n": int(overall["parseable_n"]),
            "truth_following_n": int(overall["truth_following_n"]),
            "user_agreement_error_n": int(overall["user_agreement_error_n"]),
            "other_error_n": int(overall["other_error_n"]),
            "truth_following_rate_parseable": overall["truth_following_rate_parseable"],
            "user_agreement_error_rate_parseable": overall["user_agreement_error_rate_parseable"],
            "other_error_rate_parseable": overall["other_error_rate_parseable"],
        })
    return sorted(
        out,
        key=lambda row: (
            row["truth_following_n"],
            -row["user_agreement_error_n"],
            -row["other_error_n"],
            row["arm"],
        ),
        reverse=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DIR / "mc001_qwen3_0p6b_controlled_v11_layer_band_source_manifest.jsonl")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-arms", type=int, default=0)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(11)
    random.seed(11)
    np.random.seed(11)

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
        print(f"[v11 baseline {index:03d}/{len(rendered_rows):03d}] {row['id']} -> {result['parsed_answer']!r} {result['label']}")

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
    print(f"[v11 selection] validation={len(validation_rows)} hard={len(hard_rows)} side={len(side_rows)} eval={len(eval_rows)}")

    specs = make_specs(model.config.num_hidden_layers)
    if args.max_arms:
        specs = specs[: args.max_arms]
    print("[v11 arms]")
    for spec in specs:
        print(f"  {spec_name(spec)} layers={list(spec.band.layers)}")

    generation = run_generation_validation(
        eval_rows,
        baseline_rows,
        specs,
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
        "run_type": "qwen3_0p6b_controlled_v11_layer_band_source",
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
        "layer_bands": [
            {"name": band.name, "layers": list(band.layers)}
            for band in layer_bands(model.config.num_hidden_layers)
        ],
        "selected_specs": [
            {
                "name": spec_name(spec),
                "source_group": spec.source_group,
                "layer_band": spec.band.name,
                "mask_layers": list(spec.band.layers),
            }
            for spec in specs
        ],
        "generation_validation": generation,
        "hard_bin_top": compact_arm_summary(generation, "wrong_hint_agreement_favored"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"mc001_qwen3_0p6b_controlled_v11_layer_band_source_{timestamp}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output_path": str(output_path),
        "baseline": output["baseline_summary"]["overall"],
        "candidate_counts": output["candidate_counts"],
        "hard_bin_top": output["hard_bin_top"][:12],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
