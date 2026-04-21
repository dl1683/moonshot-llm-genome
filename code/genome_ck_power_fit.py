"""Candidate v2 functional form: C(X, k) ≈ c_0 · k^p + residual.

The locked v1 derivation (Laplace-Beltrami) predicted DECREASING C(k);
observed C(k) INCREASES monotonically on all 5 systems. We need a new
candidate functional form.

Visual inspection of the k-sweep (genome_015) shows nearly-log-linear
growth of C with k across the 3-130 range on all 5 systems. If C(k) =
c_0 · k^p, then log C = log c_0 + p log k — a linear regression gives
(p, c_0).

This script fits log C vs log k per (system, depth) cell, reports:
  - slope p (the exponent)
  - intercept log c_0
  - R² of the linear fit
  - whether p is approximately constant ACROSS systems (cross-arch
    universality of the v2 form).

If p ≈ const ≈ 0.17 across systems AND R² > 0.95: v2 derivation candidate
has legs. If p varies wildly or fits are poor: the power law is also
wrong and we need a third form.

CPU-only, <10s wall-clock. Uses existing atlas data from the wider
k-sweep.
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_knn_rows() -> dict:
    """Return {(system, depth) -> {k -> [C values across seeds]}}."""
    root = Path(__file__).parent.parent / "results" / "cross_arch"
    grid: dict = defaultdict(lambda: defaultdict(list))
    for p in sorted(root.glob("atlas_rows_n2000_c4_seed*_only_*.json")):
        d = json.loads(p.read_text())
        for r in d.get("rows", []):
            if r.get("primitive_id") != "knn_clustering":
                continue
            est = r["estimator"]
            if not est.startswith("knn_k"):
                continue
            k = int(est[5:])
            depth = round(float(r["k_normalized"]), 2)
            sys_key = r["system_key"]
            grid[(sys_key, depth)][k].append(float(r["value"]))
    return grid


def _fit_power(ks: np.ndarray, Cs: np.ndarray) -> dict:
    """Linear regression in log-log. Returns slope + intercept + R²."""
    logk = np.log(ks)
    logC = np.log(Cs)
    slope, intercept = np.polyfit(logk, logC, 1)
    y_pred = slope * logk + intercept
    ss_res = np.sum((logC - y_pred) ** 2)
    ss_tot = np.sum((logC - logC.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return {
        "p_slope": float(slope),
        "log_c0_intercept": float(intercept),
        "c0_implied": float(np.exp(intercept)),
        "R2": float(r2),
    }


def fit_all_systems():
    grid = _load_knn_rows()
    out: dict = {"purpose": "power-law fit C(k) = c0 * k^p", "per_cell": {}}
    print(f"{'system':30s} {'depth':>6s} {'n_k':>4s} {'p':>8s} {'c0':>8s} {'R²':>8s}")
    p_values = []
    for (sys_, depth), by_k in sorted(grid.items()):
        ks = []
        Cs = []
        for k, vals in sorted(by_k.items()):
            ks.append(k)
            Cs.append(np.mean(vals))
        if len(ks) < 3:
            continue
        ks_a = np.array(ks)
        Cs_a = np.array(Cs)
        r = _fit_power(ks_a, Cs_a)
        r["k_grid"] = [int(k) for k in ks]
        r["C_observed"] = [float(c) for c in Cs]
        key = f"{sys_}||depth{depth}"
        out["per_cell"][key] = r
        p_values.append(r["p_slope"])
        print(f"  {sys_:28s} {depth:6.2f} {len(ks):4d} "
              f"{r['p_slope']:8.4f} {r['c0_implied']:8.4f} {r['R2']:8.4f}")

    p_arr = np.array(p_values)
    print()
    print(f"Summary of p (exponent) across all (system, depth) cells:")
    print(f"  n cells     : {len(p_arr)}")
    print(f"  p mean      : {p_arr.mean():.4f}")
    print(f"  p std       : {p_arr.std(ddof=1):.4f}")
    print(f"  p CV        : {100*p_arr.std(ddof=1)/p_arr.mean():.2f}%")
    print(f"  p range     : [{p_arr.min():.4f}, {p_arr.max():.4f}]")
    print()
    print(f"Verdict: p is cross-architecture universal if CV < 10%")
    print(f"         and per-cell R^2 > 0.95 (clean log-linearity).")

    out["summary"] = {
        "n_cells": len(p_arr),
        "p_mean": float(p_arr.mean()),
        "p_std": float(p_arr.std(ddof=1)),
        "p_cv_pct": float(100 * p_arr.std(ddof=1) / p_arr.mean()),
        "p_min": float(p_arr.min()),
        "p_max": float(p_arr.max()),
        "R2_mean": float(np.mean([r["R2"] for r in out["per_cell"].values()])),
    }

    out_path = Path(__file__).parent.parent / "results" / "gate2" / "ck_power_fit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nout: {out_path}")


if __name__ == "__main__":
    fit_all_systems()
