#!/usr/bin/env python
"""MC005 V15 fine localization diagnostic for the Qwen3-1.7B Response surface."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc001_qwen3_controlled_v10_head_localization import (
    install_source_mask_hook,
    set_eager_attention,
    tokenizer_padding_side,
)
from mc005_associative_lookup_reliability_v4 import SELECTED_BAND, SELECTED_LAYERS
from mc005_associative_lookup_response_marker_v9 import (
    MARKER_TEXT,
    Scenario,
    arm_is_clean,
    arm_is_weak,
    build_rows,
)
from mc005_associative_lookup_source_edge import CARD_ID, MODEL_ID, RESULT_DIR, summarize_arm


DISCOVERY_SEEDS = [17, 23]
LOOKUP_HOLDOUT_SEED = 31
NULL_HOLDOUT_SEEDS = [37, 41]
PAIR_COUNT = 16
LOOKUP_ROW_COUNT = 64
NULL_ROW_COUNT = 64
LOOKUP_ARM_KEYS = ("target_value", "distractor_value", "random_value")
NULL_ARM_KEYS = (
    "source_value",
    "non_source_control_value",
    "earlier_neutral_colon",
    "final_label",
    "final_colon",
)


@dataclass(frozen=True)
class Candidate:
    name: str
    layers: tuple[int, ...]
    heads: tuple[int, ...] | None
    selectable: bool


FULL_CANDIDATE = Candidate("full_l20_26_all", tuple(SELECTED_LAYERS), None, False)

COMPACT_CANDIDATES = [
    Candidate(f"single_l{layer}_all", (layer,), None, True)
    for layer in SELECTED_LAYERS
] + [
    Candidate("slice_l20_22_all", (20, 21, 22), None, True),
    Candidate("slice_l23_24_all", (23, 24), None, True),
    Candidate("slice_l25_26_all", (25, 26), None, True),
    Candidate("full_l20_26_lower_heads", tuple(SELECTED_LAYERS), tuple(range(0, 8)), True),
    Candidate("full_l20_26_upper_heads", tuple(SELECTED_LAYERS), tuple(range(8, 16)), True),
    Candidate("full_l20_26_even_heads", tuple(SELECTED_LAYERS), tuple(range(0, 16, 2)), True),
    Candidate("full_l20_26_odd_heads", tuple(SELECTED_LAYERS), tuple(range(1, 16, 2)), True),
]


def candidate_payload(candidate: Candidate, model: Any | None = None) -> dict[str, Any]:
    if candidate.heads is None:
        heads: str | list[int] = "all" if model is None else list(range(model.config.num_attention_heads))
    else:
        heads = list(candidate.heads)
    return {
        "name": candidate.name,
        "layers": list(candidate.layers),
        "heads": heads,
        "selectable": candidate.selectable,
    }


def shifted_positions_for_batch(
    rows: list[dict[str, Any]],
    source_key: str,
    seq_lens: list[int],
    max_len: int,
) -> list[list[int]]:
    shifted: list[list[int]] = []
    for row, seq_len in zip(rows, seq_lens, strict=True):
        shifted.append([int(row["positions"][source_key]) + (max_len - seq_len)])
    return shifted


def next_token_features(logits: torch.Tensor, row: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    target_logit = float(logits[int(row["target_token_id"])])
    distractor_logit = float(logits[int(row["distractor_token_id"])])
    margin = target_logit - distractor_logit
    greedy_token_id = int(torch.argmax(logits).item())
    if greedy_token_id == int(row["target_token_id"]):
        greedy_label = "target"
    elif greedy_token_id == int(row["distractor_token_id"]):
        greedy_label = "distractor"
    else:
        greedy_label = "other"
    return {
        "target_logit": target_logit,
        "distractor_logit": distractor_logit,
        "target_minus_distractor_margin": margin,
        "target_wins": margin > 0.0,
        "greedy_token_id": greedy_token_id,
        "greedy_token": tokenizer.decode([greedy_token_id]),
        "greedy_label": greedy_label,
    }


def score_rows_path(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    candidate: Candidate | None = None,
    source_key: str | None = None,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    if candidate is not None:
        heads = list(candidate.heads) if candidate.heads is not None else list(range(model.config.num_attention_heads))
    else:
        heads = []
    with tokenizer_padding_side(tokenizer, "left"):
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            inputs = tokenizer([row["rendered_prompt"] for row in batch], return_tensors="pt", padding=True).to(
                model.device
            )
            handles = []
            if candidate is not None and source_key is not None:
                seq_lens = [int(value) for value in inputs["attention_mask"].sum(dim=1).detach().cpu().tolist()]
                max_len = int(inputs["input_ids"].shape[-1])
                shifted = shifted_positions_for_batch(batch, source_key, seq_lens, max_len)
                handles = [
                    install_source_mask_hook(model, int(layer), heads, shifted)
                    for layer in candidate.layers
                ]
            try:
                with torch.inference_mode():
                    out = model(**inputs, use_cache=False, logits_to_keep=1)
            finally:
                for handle in handles:
                    handle.remove()
            for index, row in enumerate(batch):
                features[row["id"]] = next_token_features(out.logits[index, -1, :].detach().float(), row, tokenizer)
    return features


def add_arm_label(summary: dict[str, Any]) -> dict[str, Any]:
    summary["abs_target_win_change"] = abs(int(summary["target_win_loss"]))
    if arm_is_clean(summary):
        summary["arm_label"] = "clean"
    elif arm_is_weak(summary):
        summary["arm_label"] = "weak"
    else:
        summary["arm_label"] = "side_effect"
    return summary


def summarize_path_arm(
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    arm: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return add_arm_label(summarize_arm(rows, baseline, arm))


def build_lookup_rows_for_seed(tokenizer: Any, seed: int, split: str, row_count: int) -> list[dict[str, Any]]:
    scenario = Scenario(
        "lookup_pair16_response",
        "lookup",
        PAIR_COUNT,
        LOOKUP_ARM_KEYS,
        row_count=row_count,
    )
    rows = build_rows(tokenizer, scenario, seed, "mc005_v15")
    for row in rows:
        row["split"] = split
    return rows


def build_null_rows_for_seed(tokenizer: Any, seed: int, split: str, row_count: int) -> list[dict[str, Any]]:
    scenario = Scenario(
        "answer_absent_pair16_response_null",
        "null",
        PAIR_COUNT,
        NULL_ARM_KEYS,
        row_count=row_count,
    )
    rows = build_rows(tokenizer, scenario, seed, "mc005_v15")
    for row in rows:
        row["split"] = split
    return rows


def classify_null(rows: list[dict[str, Any]], baseline: dict[str, dict[str, Any]], arm_summaries: dict[str, Any]) -> str:
    baseline_floor = int(0.75 * len(rows))
    baseline_clean_rows = sum(1 for row in rows if baseline[row["id"]]["target_wins"])
    if baseline_clean_rows < baseline_floor:
        return "invalid_baseline"
    arms = list(arm_summaries.values())
    if all(arm_is_clean(arm) for arm in arms):
        return "clean_null"
    if all(arm_is_weak(arm) for arm in arms):
        return "weak_null"
    return "side_effect"


def baseline_summary(rows: list[dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> dict[str, Any]:
    margins = [float(baseline[row["id"]]["target_minus_distractor_margin"]) for row in rows]
    counts = {"target": 0, "distractor": 0, "other": 0}
    for row in rows:
        counts[str(baseline[row["id"]]["greedy_label"])] += 1
    return {
        "rows": len(rows),
        "target_wins": sum(1 for row in rows if baseline[row["id"]]["target_wins"]),
        "mean_margin": sum(margins) / max(1, len(margins)),
        "greedy_counts": counts,
    }


def select_candidate(discovery_results: dict[str, dict[str, Any]]) -> str:
    selectable = [
        (name, summary)
        for name, summary in discovery_results.items()
        if summary["candidate"]["selectable"]
    ]
    return min(
        selectable,
        key=lambda item: (
            float(item[1]["target_value"]["mean_delta"]),
            -int(item[1]["target_value"]["target_win_loss"]),
            item[0],
        ),
    )[0]


def effect_share(selected_target: dict[str, Any], full_target: dict[str, Any]) -> float | None:
    full_mag = abs(float(full_target["mean_delta"]))
    if full_mag <= 1e-12:
        return None
    return abs(float(selected_target["mean_delta"])) / full_mag


def classify_diagnostic(criteria: dict[str, bool], share: float | None) -> str:
    if all(criteria.values()):
        return "compact_localization_supported"
    if not criteria["selected_target_mean_delta_at_most_minus_1"] or not criteria["selected_target_win_loss_at_least_3"]:
        return "no_compact_effect"
    if not criteria["selected_target_beats_distractor_and_random"]:
        return "control_failed"
    if not criteria["selected_null_holdouts_clean"]:
        return "null_failed"
    if share is None or not criteria["selected_effect_share_at_least_0p60"]:
        return "full_band_required"
    return "weak_compact_localization"


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_response_marker_v15_fine_localization")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lookup-row-count", type=int, default=LOOKUP_ROW_COUNT)
    parser.add_argument("--null-row-count", type=int, default=NULL_ROW_COUNT)
    parser.add_argument("--discovery-seeds", default=",".join(str(seed) for seed in DISCOVERY_SEEDS))
    parser.add_argument("--lookup-holdout-seed", type=int, default=LOOKUP_HOLDOUT_SEED)
    parser.add_argument("--null-holdout-seeds", default=",".join(str(seed) for seed in NULL_HOLDOUT_SEEDS))
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    discovery_seeds = parse_ints(args.discovery_seeds)
    null_holdout_seeds = parse_ints(args.null_holdout_seeds)
    candidates = [FULL_CANDIDATE] + COMPACT_CANDIDATES
    candidate_by_name = {candidate.name: candidate for candidate in candidates}

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
    discovery_rows = [
        row
        for seed in discovery_seeds
        for row in build_lookup_rows_for_seed(tokenizer, seed, "lookup_discovery", args.lookup_row_count)
    ]
    lookup_holdout_rows = build_lookup_rows_for_seed(
        tokenizer,
        args.lookup_holdout_seed,
        "lookup_holdout",
        args.lookup_row_count,
    )
    null_holdout_rows_by_seed = {
        seed: build_null_rows_for_seed(tokenizer, seed, "answer_absent_null_holdout", args.null_row_count)
        for seed in null_holdout_seeds
    }

    discovery_baseline = score_rows_path(discovery_rows, tokenizer, model, args.batch_size)
    lookup_holdout_baseline = score_rows_path(lookup_holdout_rows, tokenizer, model, args.batch_size)

    discovery_results: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        target_arm = score_rows_path(
            discovery_rows,
            tokenizer,
            model,
            args.batch_size,
            candidate,
            "target_value",
        )
        discovery_results[candidate.name] = {
            "candidate": candidate_payload(candidate, model),
            "target_value": summarize_path_arm(discovery_rows, discovery_baseline, target_arm),
        }
        target = discovery_results[candidate.name]["target_value"]
        print(
            f"[v15 discovery {candidate.name}] "
            f"mean_delta={target['mean_delta']:.4f} win_loss={target['target_win_loss']}"
        )

    selected_name = select_candidate(discovery_results)
    selected = candidate_by_name[selected_name]
    print(f"[v15 selected] {selected_name}")

    lookup_holdout: dict[str, dict[str, Any]] = {}
    for label, candidate in (("selected", selected), ("full", FULL_CANDIDATE)):
        lookup_holdout[label] = {"candidate": candidate_payload(candidate, model), "arms": {}}
        for arm_key in LOOKUP_ARM_KEYS:
            arm = score_rows_path(
                lookup_holdout_rows,
                tokenizer,
                model,
                args.batch_size,
                candidate,
                arm_key,
            )
            lookup_holdout[label]["arms"][arm_key] = summarize_path_arm(lookup_holdout_rows, lookup_holdout_baseline, arm)
        target = lookup_holdout[label]["arms"]["target_value"]
        print(
            f"[v15 holdout {label}] "
            f"target_mean_delta={target['mean_delta']:.4f} win_loss={target['target_win_loss']}"
        )

    null_holdout: dict[int, dict[str, Any]] = {}
    for seed, rows in null_holdout_rows_by_seed.items():
        baseline = score_rows_path(rows, tokenizer, model, args.batch_size)
        arm_summaries = {}
        for arm_key in NULL_ARM_KEYS:
            arm = score_rows_path(rows, tokenizer, model, args.batch_size, selected, arm_key)
            arm_summaries[arm_key] = summarize_path_arm(rows, baseline, arm)
        label = classify_null(rows, baseline, arm_summaries)
        null_holdout[seed] = {
            "seed": seed,
            "label": label,
            "baseline": baseline_summary(rows, baseline),
            "baseline_floor": int(0.75 * len(rows)),
            "arms": arm_summaries,
        }
        print(
            f"[v15 null seed={seed}] label={label} "
            + " ".join(
                f"{key}={arm_summaries[key]['mean_delta']:.4f}/"
                f"{arm_summaries[key]['target_win_loss']}/"
                f"{arm_summaries[key]['arm_label']}"
                for key in NULL_ARM_KEYS
            )
        )

    selected_target = lookup_holdout["selected"]["arms"]["target_value"]
    selected_distractor = lookup_holdout["selected"]["arms"]["distractor_value"]
    selected_random = lookup_holdout["selected"]["arms"]["random_value"]
    full_target = lookup_holdout["full"]["arms"]["target_value"]
    share = effect_share(selected_target, full_target)
    criteria = {
        "selected_target_mean_delta_at_most_minus_1": float(selected_target["mean_delta"]) <= -1.0,
        "selected_target_win_loss_at_least_3": int(selected_target["target_win_loss"]) >= 3,
        "selected_target_beats_distractor_and_random": (
            float(selected_target["mean_delta"]) <= float(selected_distractor["mean_delta"]) - 0.50
            and float(selected_target["mean_delta"]) <= float(selected_random["mean_delta"]) - 0.50
        ),
        "selected_effect_share_at_least_0p60": share is not None and share >= 0.60,
        "selected_null_holdouts_clean": all(item["label"] == "clean_null" for item in null_holdout.values()),
    }

    suite_summary = {
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "marker": MARKER_TEXT,
        "model_id": args.model_id,
        "pair_count": PAIR_COUNT,
        "discovery_seeds": discovery_seeds,
        "lookup_holdout_seed": args.lookup_holdout_seed,
        "null_holdout_seeds": null_holdout_seeds,
        "lookup_row_count_per_seed": args.lookup_row_count,
        "null_row_count_per_seed": args.null_row_count,
        "candidate_count": len(candidates),
        "selectable_candidate_count": len(COMPACT_CANDIDATES),
        "selected_candidate": candidate_payload(selected, model),
        "full_candidate": candidate_payload(FULL_CANDIDATE, model),
        "lookup_holdout_baseline": baseline_summary(lookup_holdout_rows, lookup_holdout_baseline),
        "selected_effect_share_of_full_holdout": share,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "diagnostic_class": classify_diagnostic(criteria, share),
    }

    elapsed = time.time() - started
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_response_marker_v15_fine_localization",
        "model_id": args.model_id,
        "selected_band": SELECTED_BAND,
        "selected_layers": SELECTED_LAYERS,
        "elapsed_s": elapsed,
        "summary": suite_summary,
        "discovery_results": discovery_results,
        "lookup_holdout": lookup_holdout,
        "null_holdout": null_holdout,
        "rows": discovery_rows + lookup_holdout_rows + [
            row for rows in null_holdout_rows_by_seed.values() for row in rows
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)
    print(json.dumps({**suite_summary, "output_path": str(output_path)}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
