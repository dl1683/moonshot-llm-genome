#!/usr/bin/env python
"""MC005 V23 offline row-heterogeneity diagnostic over the V22 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CARD_ID = "MC005"
MODEL_ID = "Qwen/Qwen3-1.7B"
RESULT_DIR = Path("results/cards/MC005")
DEFAULT_V22_PATH = RESULT_DIR / "mc005_qwen3_1p7b_response_marker_v22_row_interaction_20260630T201924.json"

PAIR_OMITTED_LAYER = {
    "slice_l24_25_all": 26,
    "slice_l24_26_all_pair": 25,
    "slice_l25_26_all": 24,
}
PRIORITY_FAMILIES = [
    "best_pair_winner",
    "best_pair_omitted_layer",
    "target_source_position",
    "query_pair_index",
    "baseline_margin_band",
    "target_value",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def margin_band_by_rank(diagnostics: list[dict[str, Any]]) -> dict[str, str]:
    parent_rows = [row for row in diagnostics if row["parent_effect_row"]]
    ordered = sorted(parent_rows, key=lambda row: (float(row["baseline_margin"]), row["row_id"]))
    labels = ["q1_low", "q2_mid_low", "q3_mid_high", "q4_high"]
    bands: dict[str, str] = {}
    n = len(ordered)
    for index, row in enumerate(ordered):
        label = labels[min(3, int(index * 4 / max(1, n)))]
        bands[row["row_id"]] = label
    for row in diagnostics:
        bands.setdefault(row["row_id"], "non_parent_effect")
    return bands


def best_pair(row_diag: dict[str, Any]) -> tuple[str | None, float | None]:
    shares = {
        name: value
        for name, value in row_diag["pair_negative_effect_shares"].items()
        if value is not None
    }
    if not shares:
        return None, None
    name = max(sorted(shares), key=lambda key: float(shares[key]))
    return name, float(shares[name])


def failure_reasons(row_diag: dict[str, Any]) -> list[str]:
    if not row_diag["parent_effect_row"] or row_diag["all_three_margin_row"]:
        return []
    reasons = []
    if not row_diag["all_pairs_below_0p60"]:
        reasons.append("pair_share_recovered")
    if float(row_diag["singles_residual"]) > -1.0:
        reasons.append("single_residual_weak")
    if not row_diag["all_pair_plus_single_superadditive"]:
        reasons.append("pair_plus_single_residual_weak")
    return reasons or ["unclassified"]


def enrich_rows(v22: dict[str, Any]) -> list[dict[str, Any]]:
    lookup_rows = {
        row["id"]: row
        for row in v22["rows"]
        if row.get("mode") == "lookup"
    }
    bands = margin_band_by_rank(v22["row_diagnostics"])
    enriched = []
    for diag in v22["row_diagnostics"]:
        row = lookup_rows[diag["row_id"]]
        winner, winner_share = best_pair(diag)
        target_position = int(row["positions"]["target_value"])
        final_label_position = int(row["positions"]["final_label"])
        reasons = failure_reasons(diag)
        enriched.append(
            {
                "row_id": diag["row_id"],
                "seed": int(row["seed"]),
                "query_pair_index": int(row["query_pair_index"]),
                "target_source_position": target_position,
                "distance_to_final_label": final_label_position - target_position,
                "baseline_margin_band": bands[diag["row_id"]],
                "target_value": str(row["target_value"]),
                "target_token_id": int(row["target_token_id"]),
                "query_key": str(row["query_key"]),
                "best_pair_winner": winner,
                "best_pair_omitted_layer": PAIR_OMITTED_LAYER.get(winner),
                "best_pair_share": winner_share,
                "baseline_margin": float(diag["baseline_margin"]),
                "parent_delta": float(diag["parent_delta"]),
                "singles_residual": float(diag["singles_residual"]),
                "parent_effect_row": bool(diag["parent_effect_row"]),
                "all_three_margin_row": bool(diag["all_three_margin_row"]),
                "parent_flip_pair_resistant": bool(diag["parent_flip_pair_resistant"]),
                "failure_reasons": reasons,
            }
        )
    return enriched


def stratum_summary(group_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    parent_rows = [row for row in rows if row["parent_effect_row"]]
    parent_count = len(parent_rows)
    all_three_count = sum(1 for row in parent_rows if row["all_three_margin_row"])
    flip_resistant_count = sum(1 for row in parent_rows if row["parent_flip_pair_resistant"])
    best_pair_shares = [
        float(row["best_pair_share"])
        for row in parent_rows
        if row["best_pair_share"] is not None
    ]
    singles_residuals = [float(row["singles_residual"]) for row in parent_rows]
    return {
        "group": str(group_name),
        "row_count": len(rows),
        "parent_effect_row_count": parent_count,
        "all_three_margin_row_count": all_three_count,
        "all_three_fraction_of_parent_effect_rows": (
            all_three_count / parent_count if parent_count else None
        ),
        "parent_flip_pair_resistant_count": flip_resistant_count,
        "median_best_pair_share": median(best_pair_shares),
        "median_singles_residual": median(singles_residuals),
        "mean_parent_delta": mean([float(row["parent_delta"]) for row in parent_rows]),
        "mean_baseline_margin": mean([float(row["baseline_margin"]) for row in parent_rows]),
    }


def build_strata(enriched_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    strata = {}
    for family in PRIORITY_FAMILIES:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in enriched_rows:
            value = row[family]
            groups[str(value)].append(row)
        table = [stratum_summary(group, rows) for group, rows in groups.items()]
        table.sort(
            key=lambda item: (
                -int(item["parent_effect_row_count"]),
                item["group"],
            )
        )
        strata[family] = table
    return strata


def eligible_contrast(table: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in table
        if int(row["parent_effect_row_count"]) >= 16
        and row["all_three_fraction_of_parent_effect_rows"] is not None
    ]
    if len(eligible) < 2:
        return {
            "eligible_group_count": len(eligible),
            "contrast": None,
            "strong": False,
            "min_group": None,
            "max_group": None,
        }
    min_group = min(eligible, key=lambda row: float(row["all_three_fraction_of_parent_effect_rows"]))
    max_group = max(eligible, key=lambda row: float(row["all_three_fraction_of_parent_effect_rows"]))
    contrast = (
        float(max_group["all_three_fraction_of_parent_effect_rows"])
        - float(min_group["all_three_fraction_of_parent_effect_rows"])
    )
    return {
        "eligible_group_count": len(eligible),
        "contrast": contrast,
        "strong": contrast >= 0.25,
        "min_group": min_group,
        "max_group": max_group,
    }


def choose_diagnostic_label(contrasts: dict[str, dict[str, Any]]) -> tuple[str, str | None]:
    structural = [
        "best_pair_winner",
        "best_pair_omitted_layer",
        "target_source_position",
        "query_pair_index",
    ]
    for family in structural:
        if contrasts[family]["strong"]:
            return "best_pair_position_heterogeneity", family
    if contrasts["baseline_margin_band"]["strong"]:
        return "margin_band_heterogeneity", "baseline_margin_band"
    if contrasts["target_value"]["strong"]:
        return "token_identity_heterogeneity", "target_value"
    return "diffuse_heterogeneity", None


def failure_reason_counts(enriched_rows: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counter: Counter[str] = Counter()
    combination_counter: Counter[str] = Counter()
    non_all_three_parent_rows = [
        row for row in enriched_rows
        if row["parent_effect_row"] and not row["all_three_margin_row"]
    ]
    for row in non_all_three_parent_rows:
        reasons = row["failure_reasons"]
        for reason in reasons:
            reason_counter[reason] += 1
        combination_counter["+".join(reasons)] += 1
    return {
        "non_all_three_parent_effect_row_count": len(non_all_three_parent_rows),
        "single_reason_counts": dict(sorted(reason_counter.items())),
        "combination_counts": dict(sorted(combination_counter.items())),
    }


def validate_v22(v22: dict[str, Any]) -> bool:
    summary = v22.get("summary", {})
    return (
        v22.get("run_type") == "associative_lookup_response_marker_v22_row_interaction"
        and summary.get("diagnostic_class") == "mean_only_interaction"
        and "row_diagnostics" in v22
        and "rows" in v22
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-v22", type=Path, default=DEFAULT_V22_PATH)
    parser.add_argument("--card-id", default=CARD_ID)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--artifact-prefix", default="mc005_qwen3_1p7b_response_marker_v23_row_heterogeneity")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    args = parser.parse_args()

    started = time.time()
    source_sha256 = sha256_file(args.source_v22)
    v22 = read_json(args.source_v22)
    valid_source = validate_v22(v22)

    if valid_source:
        enriched = enrich_rows(v22)
        strata = build_strata(enriched)
        contrasts = {
            family: eligible_contrast(table)
            for family, table in strata.items()
        }
        diagnostic_label, dominant_family = choose_diagnostic_label(contrasts)
        failures = failure_reason_counts(enriched)
    else:
        enriched = []
        strata = {}
        contrasts = {}
        diagnostic_label = "invalid_source_artifact"
        dominant_family = None
        failures = {}

    v22_summary = v22.get("summary", {})
    row_summary = v22_summary.get("row_summary", {})
    elapsed = time.time() - started
    summary = {
        "model_id": args.model_id,
        "source_v22_path": str(args.source_v22),
        "source_v22_sha256": source_sha256,
        "valid_source_artifact": valid_source,
        "v22_diagnostic_class": v22_summary.get("diagnostic_class"),
        "v22_parent_effect_row_count": row_summary.get("parent_effect_row_count"),
        "v22_all_three_margin_row_count": row_summary.get("all_three_margin_row_count"),
        "v22_all_three_fraction": row_summary.get("all_three_margin_row_fraction_of_parent_effect_rows"),
        "v22_parent_flip_pair_resistant_row_count": row_summary.get("parent_flip_pair_resistant_row_count"),
        "enriched_lookup_row_count": len(enriched),
        "diagnostic_label": diagnostic_label,
        "dominant_family": dominant_family,
        "contrast_rule": {
            "minimum_parent_effect_rows_per_group": 16,
            "minimum_all_three_fraction_range": 0.25,
        },
    }
    result = {
        "card_id": args.card_id,
        "run_type": "associative_lookup_response_marker_v23_row_heterogeneity",
        "model_id": args.model_id,
        "elapsed_s": elapsed,
        "summary": summary,
        "headline_contrasts": contrasts,
        "failure_reason_counts": failures,
        "strata": strata,
        "enriched_rows": enriched,
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
