#!/usr/bin/env python
"""MC001G Gemma Scope sparse-feature discovery on matched Gemma rows."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

from mc001_gemma_prehint_margin_discovery import attach_item_margins, baseline_auc, matched_rows, summarize_match
from mc001_gemma_repair_discovery import collect_rows, label_value, load_json, orient_auc, select_clean_items, split_clean_items
from mc001_logit_smoke import letter_token_ids, render_for_score
from mc001_qwen3_smoke import classify


MATCH_CONDITIONS = ("wrong_disclaimed", "wrong_unsure")
DEFAULT_SAE_REPO = "google/gemma-scope-2b-pt-res"
DEFAULT_SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
DEFAULT_SAE_IDS = {
    14: "layer_14/width_16k/canonical",
    20: "layer_20/width_16k/canonical",
}


def prepare_rows(repair_rows: list[dict[str, Any]], bin_width: float) -> list[dict[str, Any]]:
    clean_items = select_clean_items(repair_rows)
    split_items = split_clean_items(clean_items)
    primary_rows = collect_rows(repair_rows, clean_items, split_items)
    primary_rows = [row for row in primary_rows if row["condition"] in MATCH_CONDITIONS]
    enriched = attach_item_margins(primary_rows, repair_rows, bin_width)
    return matched_rows(enriched)


def score_rows_and_collect_hidden(
    rows: list[dict[str, Any]],
    model_id: str,
    render_mode: str,
    layers: list[int],
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
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
    option_ids = letter_token_ids(tokenizer)

    hidden_by_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    scored_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        rendered = render_for_score(tokenizer, row["prompt"], render_mode)
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits[0, -1].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        scores = {
            letter: max(float(log_probs[token_id].item()) for token_id in token_ids)
            for letter, token_ids in option_ids.items()
        }
        parsed = max(scores, key=scores.get)
        for layer in layers:
            hidden_by_layer[layer].append(outputs.hidden_states[layer + 1][0, -1].float().cpu().numpy())
        scored_rows.append(
            {
                **row,
                "index": index,
                "rendered_prompt": rendered,
                "parsed_answer_recomputed": parsed,
                "label_recomputed": classify(parsed, row["correct_answer"], row["wrong_answer"]),
                "option_logprobs_recomputed": scores,
                "correct_minus_wrong_logprob_recomputed": scores[row["correct_answer"]] - scores[row["wrong_answer"]],
            }
        )
        print(f"[{index:03d}/{len(rows):03d}] {row['id']} split={row['split']} label={row['label']}")

    return scored_rows, {
        layer: np.stack(layer_rows, axis=0).astype(np.float32)
        for layer, layer_rows in hidden_by_layer.items()
    }


def load_sae(release: str, sae_id: str) -> SAE:
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device="cpu")
    sae.eval()
    return sae


def sae_metadata(sae: SAE) -> dict[str, Any]:
    cfg = getattr(sae, "cfg", None)
    return {
        "class": type(sae).__name__,
        "d_in": int(getattr(cfg, "d_in", 0) or 0),
        "d_sae": int(getattr(cfg, "d_sae", 0) or 0),
        "normalize_activations": str(getattr(cfg, "normalize_activations", None)),
        "threshold_shape": list(getattr(sae, "threshold").shape) if hasattr(sae, "threshold") else None,
    }


def encode_with_sae(hidden: np.ndarray, sae: SAE) -> tuple[np.ndarray, np.ndarray]:
    hidden_tensor = torch.tensor(hidden.astype(np.float32), dtype=torch.float32)
    with torch.inference_mode():
        acts = sae.encode(hidden_tensor)
        recon = sae.decode(acts)
    return acts.cpu().numpy().astype(np.float32), recon.cpu().numpy().astype(np.float32)


def reconstruction_stats(hidden: np.ndarray, recon: np.ndarray) -> dict[str, Any]:
    residual = hidden.astype(np.float32) - recon
    mse_by_row = np.mean(residual * residual, axis=1)
    hidden_energy = np.mean(hidden.astype(np.float32) * hidden.astype(np.float32), axis=1)
    denom = np.linalg.norm(hidden, axis=1) * np.linalg.norm(recon, axis=1)
    cosine = np.sum(hidden * recon, axis=1) / np.maximum(denom, 1e-12)
    return {
        "mean_mse": float(np.mean(mse_by_row)),
        "mean_relative_mse_to_hidden_energy": float(np.mean(mse_by_row / np.maximum(hidden_energy, 1e-12))),
        "mean_cosine": float(np.mean(cosine)),
        "min_cosine": float(np.min(cosine)),
        "max_cosine": float(np.max(cosine)),
    }


def activation_stats(acts: np.ndarray) -> dict[str, Any]:
    l0 = np.count_nonzero(acts > 0, axis=1)
    return {
        "n_rows": int(acts.shape[0]),
        "width": int(acts.shape[1]),
        "mean_l0": float(np.mean(l0)),
        "std_l0": float(np.std(l0)),
        "min_l0": int(np.min(l0)),
        "max_l0": int(np.max(l0)),
        "mean_positive_activation": float(np.mean(acts[acts > 0])) if np.any(acts > 0) else 0.0,
    }


def auc_by_feature(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUC requires both binary classes")
    diff = pos[:, None, :] - neg[None, :, :]
    return ((diff > 0).sum(axis=(0, 1)) + 0.5 * (diff == 0).sum(axis=(0, 1))) / (len(pos) * len(neg))


def auc_1d(scores: np.ndarray, y: np.ndarray) -> float:
    return float(auc_by_feature(scores.reshape(-1, 1), y)[0])


def feature_record(
    feature: int,
    rank: int,
    acts: np.ndarray,
    rows: list[dict[str, Any]],
    y: np.ndarray,
    train_mask: np.ndarray,
    train_auc_raw: np.ndarray,
    train_auc_oriented: np.ndarray,
    holdout_auc_oriented: np.ndarray,
    orientation: np.ndarray,
) -> dict[str, Any]:
    train = train_mask
    holdout = ~train_mask
    values = acts[:, feature]
    train_truth = values[train & (y == 1)]
    train_agree = values[train & (y == 0)]
    holdout_truth = values[holdout & (y == 1)]
    holdout_agree = values[holdout & (y == 0)]
    top_rows = sorted(
        [
            {
                "id": row["id"],
                "split": row["split"],
                "label": row["label"],
                "condition": row["condition"],
                "activation": float(values[index]),
            }
            for index, row in enumerate(rows)
            if values[index] > 0
        ],
        key=lambda row: (-row["activation"], row["id"]),
    )[:8]
    return {
        "rank": int(rank),
        "feature": int(feature),
        "orientation": int(orientation[feature]),
        "discovery_auc_raw": float(train_auc_raw[feature]),
        "discovery_auc_oriented": float(train_auc_oriented[feature]),
        "holdout_auc_oriented": float(holdout_auc_oriented[feature]),
        "discovery_active_n": int(np.count_nonzero(values[train] > 0)),
        "holdout_active_n": int(np.count_nonzero(values[holdout] > 0)),
        "discovery_truth_mean": float(np.mean(train_truth)),
        "discovery_agreement_mean": float(np.mean(train_agree)),
        "holdout_truth_mean": float(np.mean(holdout_truth)),
        "holdout_agreement_mean": float(np.mean(holdout_agree)),
        "top_activating_rows": top_rows,
    }


def sparse_combo_auc(
    acts: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    feature_ids: list[int],
    orientation: np.ndarray,
) -> dict[str, Any]:
    oriented = acts[:, feature_ids] * orientation[feature_ids].reshape(1, -1)
    train_values = oriented[train_mask]
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    scores = ((oriented - mean) / std).sum(axis=1)
    return {
        **orient_auc(scores[train_mask], y[train_mask], scores[~train_mask], y[~train_mask]),
        "feature_ids": [int(feature) for feature in feature_ids],
    }


def summarize_feature_layer(
    acts: np.ndarray,
    rows: list[dict[str, Any]],
    y: np.ndarray,
    train_mask: np.ndarray,
    min_train_active: int,
    top_n: int,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    train = train_mask
    holdout = ~train_mask
    train_acts = acts[train]
    holdout_acts = acts[holdout]
    y_train = y[train]
    y_holdout = y[holdout]

    train_auc_raw = auc_by_feature(train_acts, y_train)
    holdout_auc_raw = auc_by_feature(holdout_acts, y_holdout)
    orientation = np.where(train_auc_raw >= 0.5, 1.0, -1.0).astype(np.float32)
    train_auc_oriented = np.maximum(train_auc_raw, 1.0 - train_auc_raw)
    holdout_auc_oriented = np.where(orientation > 0, holdout_auc_raw, 1.0 - holdout_auc_raw)

    train_active = np.count_nonzero(train_acts > 0, axis=0)
    train_nonconstant = np.ptp(train_acts, axis=0) > 0
    eligible = np.where((train_active >= min_train_active) & train_nonconstant)[0]
    if len(eligible) == 0:
        raise RuntimeError("no eligible SAE features")
    mean_diff = np.abs(train_acts[y_train == 1].mean(axis=0) - train_acts[y_train == 0].mean(axis=0))
    ordered = sorted(
        eligible.tolist(),
        key=lambda feature: (-float(train_auc_oriented[feature]), -float(mean_diff[feature]), int(feature)),
    )
    top_features = [
        feature_record(
            feature,
            rank + 1,
            acts,
            rows,
            y,
            train_mask,
            train_auc_raw,
            train_auc_oriented,
            holdout_auc_oriented,
            orientation,
        )
        for rank, feature in enumerate(ordered[:top_n])
    ]

    combo_metrics = {}
    for k in [1, 4, 8, 16]:
        take = ordered[: min(k, len(ordered))]
        combo_metrics[f"top_{len(take)}"] = sparse_combo_auc(acts, y, train_mask, take, orientation)

    null_rows: list[dict[str, Any]] = []
    for perm_index in range(permutations):
        permuted_train_y = rng.permutation(y_train)
        perm_train_auc = auc_by_feature(train_acts, permuted_train_y)
        perm_orientation = np.where(perm_train_auc >= 0.5, 1.0, -1.0).astype(np.float32)
        perm_train_auc_oriented = np.maximum(perm_train_auc, 1.0 - perm_train_auc)
        perm_ordered = sorted(
            eligible.tolist(),
            key=lambda feature: (-float(perm_train_auc_oriented[feature]), -float(mean_diff[feature]), int(feature)),
        )
        feature = perm_ordered[0]
        holdout_auc = holdout_auc_raw[feature] if perm_orientation[feature] > 0 else 1.0 - holdout_auc_raw[feature]
        null_rows.append(
            {
                "permutation": int(perm_index),
                "feature": int(feature),
                "discovery_auc_oriented": float(perm_train_auc_oriented[feature]),
                "holdout_auc_oriented": float(holdout_auc),
            }
        )

    null_holdouts = np.array([row["holdout_auc_oriented"] for row in null_rows], dtype=np.float32)
    rank1_holdout = top_features[0]["holdout_auc_oriented"]
    return {
        "eligible_feature_count": int(len(eligible)),
        "min_train_active": int(min_train_active),
        "top_features": top_features,
        "sparse_combo_metrics": combo_metrics,
        "label_shuffle_null": {
            "permutations": int(permutations),
            "mean_rank1_holdout_auc": float(np.mean(null_holdouts)) if len(null_holdouts) else None,
            "max_rank1_holdout_auc": float(np.max(null_holdouts)) if len(null_holdouts) else None,
            "p_ge_observed_rank1_holdout": float(np.mean(null_holdouts >= rank1_holdout)) if len(null_holdouts) else None,
            "top_examples": sorted(null_rows, key=lambda row: (-row["holdout_auc_oriented"], row["feature"]))[:10],
        },
    }


def counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/gemma-2-2b")
    parser.add_argument("--repair-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cards/MC001G"))
    parser.add_argument("--artifact-prefix", default="mc001g_gemma2_2b_repair")
    parser.add_argument("--render-mode", choices=["raw", "chat"], default="raw")
    parser.add_argument("--bin-width", type=float, default=0.5)
    parser.add_argument("--sae-repo", default=DEFAULT_SAE_REPO)
    parser.add_argument("--sae-release", default=DEFAULT_SAE_RELEASE)
    parser.add_argument("--layers", type=int, nargs="+", default=[14, 20])
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-train-active", type=int, default=2)
    parser.add_argument("--permutations", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    repair = load_json(args.repair_result)
    selected_rows = prepare_rows(repair["records"], args.bin_width)
    y = np.array([label_value(row["label"]) for row in selected_rows], dtype=np.int64)
    split_mask = np.array([row["split"] == "discovery" for row in selected_rows], dtype=bool)
    if len(set(y[split_mask])) != 2 or len(set(y[~split_mask])) != 2:
        raise RuntimeError("matched discovery and holdout splits must both contain truth and agreement labels")

    for layer in args.layers:
        if layer not in DEFAULT_SAE_IDS:
            raise ValueError(f"no default SAE ID registered for layer {layer}")

    started = time.time()
    scored_rows, hidden_by_layer = score_rows_and_collect_hidden(selected_rows, args.model_id, args.render_mode, args.layers)
    for row in scored_rows:
        row["correct_minus_wrong_logprob"] = row["correct_minus_wrong_logprob_recomputed"]
        row["parsed_answer"] = row["parsed_answer_recomputed"]

    baselines = baseline_auc(scored_rows, y, split_mask)
    layers: dict[str, Any] = {}
    for layer in args.layers:
        sae_id = DEFAULT_SAE_IDS[layer]
        print(f"loading SAE layer {layer}: {args.sae_release} / {sae_id}")
        sae = load_sae(args.sae_release, sae_id)
        acts, recon = encode_with_sae(hidden_by_layer[layer], sae)
        layers[str(layer)] = {
            "sae_repo": args.sae_repo,
            "sae_release": args.sae_release,
            "sae_id": sae_id,
            "sae_metadata": sae_metadata(sae),
            "activation_stats": activation_stats(acts),
            "reconstruction_stats": reconstruction_stats(hidden_by_layer[layer], recon),
            "feature_metrics": summarize_feature_layer(
                acts,
                scored_rows,
                y,
                split_mask,
                args.min_train_active,
                args.top_n,
                args.permutations,
                rng,
            ),
        }
        del sae, acts, recon

    elapsed = time.time() - started
    result = {
        "card_id": "MC001G",
        "run_type": f"{args.artifact_prefix}_gemma_scope_sparse_discovery",
        "model_id": args.model_id,
        "repair_result": str(args.repair_result),
        "render_mode": args.render_mode,
        "bin_width": args.bin_width,
        "match_conditions": list(MATCH_CONDITIONS),
        "selected_row_count": len(scored_rows),
        "selected_label_counts": counts(scored_rows, "label"),
        "split_counts": counts(scored_rows, "split"),
        "match_summary": summarize_match(scored_rows),
        "baselines": baselines,
        "elapsed_s": elapsed,
        "records": scored_rows,
        "layers": layers,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"{args.artifact_prefix}_gemma_scope_sparse_discovery_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)

    compact = {
        "selected_label_counts": result["selected_label_counts"],
        "split_counts": result["split_counts"],
        "baselines": result["baselines"],
        "layers": {
            layer: {
                "activation_stats": layer_result["activation_stats"],
                "reconstruction_stats": layer_result["reconstruction_stats"],
                "rank1": layer_result["feature_metrics"]["top_features"][0],
                "sparse_combo_metrics": layer_result["feature_metrics"]["sparse_combo_metrics"],
                "label_shuffle_null": layer_result["feature_metrics"]["label_shuffle_null"],
            }
            for layer, layer_result in layers.items()
        },
        "output_path": str(output_path),
    }
    print(json.dumps(compact, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
