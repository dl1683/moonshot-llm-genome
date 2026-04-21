"""Refit power-law C(k) = c0 * k^p across all atlas cells INCLUDING DiT.

Joins: existing 18 cells (6 systems x 3 depths) from results/gate2/ck_power_fit.json
       + 3 new DiT cells from results/cross_arch/atlas_rows_n*_imagenet_seed*_only_dit-xl-2-256.json

Writes a new summary at results/gate2/ck_power_fit_with_dit.json reporting
whether DiT joins the cluster (p within the existing CV band).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent


def fit_power_law(ks, Cs):
    lks = np.log(np.asarray(ks, dtype=float))
    lcs = np.log(np.asarray(Cs, dtype=float))
    p, log_c0 = np.polyfit(lks, lcs, 1)
    pred = p * lks + log_c0
    ss_res = float(np.sum((lcs - pred) ** 2))
    ss_tot = float(np.sum((lcs - lcs.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(p), float(log_c0), float(np.exp(log_c0)), r2


def main():
    # Existing per-cell fits (already computed)
    existing = json.loads((_ROOT / "results/gate2/ck_power_fit.json").read_text())
    cells = dict(existing.get("per_cell", {}))

    # Find DiT rows
    dit_files = sorted(glob.glob(str(
        _ROOT / "results/cross_arch/atlas_rows_n*_imagenet_seed*_only_dit-xl-2-256.json")))
    dit_added = 0
    for p in dit_files:
        d = json.loads(Path(p).read_text())
        for row in d["rows"]:
            per_k = row["clustering_C_per_k"]
            ks = [int(k) for k, v in per_k.items() if v is not None]
            Cs = [per_k[str(k)] for k in ks]
            if len(ks) < 5:
                continue
            p_slope, log_c0, c0, r2 = fit_power_law(ks, Cs)
            cell_key = f"dit-xl-2-256||depth{row['k_normalized']:.2f}||seed{row.get('seed')}"
            cells[cell_key] = {
                "p_slope": p_slope,
                "log_c0_intercept": log_c0,
                "c0_implied": c0,
                "R2": r2,
                "k_grid": ks,
                "C_observed": Cs,
                "seed": row.get("seed"),
            }
            dit_added += 1

    # New summary
    ps = np.array([v["p_slope"] for v in cells.values()])
    c0s = np.array([v["c0_implied"] for v in cells.values()])
    r2s = np.array([v["R2"] for v in cells.values()])
    summary = {
        "n_cells": len(cells),
        "n_dit_added": dit_added,
        "p_mean": float(ps.mean()),
        "p_std": float(ps.std(ddof=1)),
        "p_cv_pct": float(100 * ps.std(ddof=1) / abs(ps.mean())),
        "p_min": float(ps.min()),
        "p_max": float(ps.max()),
        "c0_mean": float(c0s.mean()),
        "c0_std": float(c0s.std(ddof=1)),
        "R2_mean": float(r2s.mean()),
    }

    # DiT-specific cells
    dit_ps = [v["p_slope"] for k, v in cells.items() if k.startswith("dit-")]
    dit_c0s = [v["c0_implied"] for k, v in cells.items() if k.startswith("dit-")]
    dit_r2s = [v["R2"] for k, v in cells.items() if k.startswith("dit-")]
    summary["dit_p_values"] = dit_ps
    summary["dit_c0_values"] = dit_c0s
    summary["dit_R2_values"] = dit_r2s
    if dit_ps:
        non_dit = [v["p_slope"] for k, v in cells.items() if not k.startswith("dit-")]
        non_dit = np.array(non_dit)
        mu = non_dit.mean()
        sd = non_dit.std(ddof=1)
        z_scores = [(p - mu) / sd for p in dit_ps]
        summary["dit_z_scores_vs_existing_cluster"] = z_scores
        summary["existing_cluster_mu"] = float(mu)
        summary["existing_cluster_sd"] = float(sd)
        summary["dit_joins_cluster"] = all(abs(z) < 3.0 for z in z_scores)

    out = {"purpose": "power-law fit C(k) = c0 * k^p across 6 architectures + DiT",
           "summary": summary,
           "per_cell": cells}
    out_path = _ROOT / "results/gate2/ck_power_fit_with_dit.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
