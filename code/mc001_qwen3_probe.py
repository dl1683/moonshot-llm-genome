#!/usr/bin/env python
"""Layerwise hidden-state probe for MC-001 Qwen3 smoke outputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-0.6B"
RESULT_DIR = Path("results/cards/MC001")


def latest_smoke_result() -> Path:
    candidates = sorted(RESULT_DIR.glob("mc001_qwen3_0p6b_smoke_factual_ladder_*.json"))
    if not candidates:
        raise FileNotFoundError("no factual_ladder smoke result found")
    return candidates[-1]


def base_id(record_id: str) -> str:
    return record_id.split("__", 1)[0]


def load_probe_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for record in data["records"]:
        condition = record["condition"]
        if condition in {"no_hint", "correct_hint"}:
            continue
        if record["label"] == "other_error":
            continue
        rows.append(record)
    return rows


def extract_hidden(records: list[dict[str, Any]], model_id: str) -> list[np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    layer_rows: list[list[np.ndarray]] | None = None
    for index, record in enumerate(records, start=1):
        inputs = tokenizer(record["rendered_prompt"], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        hidden_states = out.hidden_states
        if layer_rows is None:
            layer_rows = [[] for _ in hidden_states]
        for layer_index, hidden in enumerate(hidden_states):
            vec = hidden[0, -1, :].detach().float().cpu().numpy()
            layer_rows[layer_index].append(vec)
        print(f"[{index:03d}/{len(records):03d}] extracted {record['id']}")
    assert layer_rows is not None
    return [np.stack(rows, axis=0) for rows in layer_rows]


def score_features(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    cv = GroupKFold(n_splits=5)
    aucs: list[float] = []
    accs: list[float] = []
    for train_idx, test_idx in cv.split(x, y, groups):
        if len(set(y[test_idx])) < 2 or len(set(y[train_idx])) < 2:
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
        )
        clf.fit(x[train_idx], y[train_idx])
        prob = clf.predict_proba(x[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        aucs.append(float(roc_auc_score(y[test_idx], prob)))
        accs.append(float(accuracy_score(y[test_idx], pred)))
    return {
        "folds": len(aucs),
        "auc_mean": None if not aucs else float(np.mean(aucs)),
        "auc_values": aucs,
        "accuracy_mean": None if not accs else float(np.mean(accs)),
        "accuracy_values": accs,
    }


def score_condition_baseline(records: list[dict[str, Any]], y: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    conditions = np.array([[record["condition"]] for record in records])
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    x = encoder.fit_transform(conditions)
    result = score_features(x, y, groups)
    result["features"] = list(encoder.get_feature_names_out(["condition"]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=None)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    args = parser.parse_args()

    result_path = args.result or latest_smoke_result()
    records = load_probe_rows(result_path)
    y = np.array([1 if record["label"] == "user_agreement_error" else 0 for record in records])
    groups = np.array([base_id(record["id"]) for record in records])

    started = time.time()
    layers = extract_hidden(records, args.model_id)
    layer_scores = []
    for layer_index, x in enumerate(layers):
        score = score_features(x, y, groups)
        score["layer_index"] = layer_index
        layer_scores.append(score)
        print(f"layer {layer_index}: auc={score['auc_mean']} acc={score['accuracy_mean']}")
    condition_baseline = score_condition_baseline(records, y, groups)
    best = max(
        (s for s in layer_scores if s["auc_mean"] is not None),
        key=lambda s: s["auc_mean"],
    )
    output = {
        "card_id": "MC-001",
        "run_type": "qwen3_0p6b_hidden_probe_smoke",
        "model_id": args.model_id,
        "source_result": str(result_path),
        "n_records": len(records),
        "positive_rate": float(y.mean()),
        "elapsed_s": time.time() - started,
        "target_label": "user_agreement_error",
        "condition_baseline": condition_baseline,
        "best_layer": best,
        "layers": layer_scores,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_path = args.output_dir / f"mc001_qwen3_0p6b_probe_factual_ladder_{stamp}.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"condition_baseline": condition_baseline, "best_layer": best}, indent=2))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
