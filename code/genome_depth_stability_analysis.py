"""Analyze depth-variation in power-law exponent p within each architecture.

The 27-cell atlas (results/gate2/ck_power_fit_with_dit.json) has p values
at 3 sentinel depths per system. This script computes within-system depth
spread to see whether the cross-architecture convergence coexists with
within-system depth-variation.

If within-system CV is comparable to between-system CV: p is a global
architecture invariant across depths.
If within-system CV is much smaller: depth-specific convergence.
If within-system CV is larger than between-system: something is wrong.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent


def main():
    data = json.loads((_ROOT / "results/gate2/ck_power_fit_with_dit.json").read_text())
    per_cell = data.get("per_cell", {})

    per_system = {}
    for key, v in per_cell.items():
        # key format: "system||depthX.XX[||seedY]"
        parts = key.split("||")
        sys_name = parts[0]
        per_system.setdefault(sys_name, []).append({
            "key": key, "p": v["p_slope"], "c_0": v["c0_implied"],
            "R2": v["R2"],
        })

    print(f"{'system':>32s} {'n':>3s} {'p_mean':>8s} {'p_std':>8s} {'p_CV%':>8s} "
          f"{'p_range':>16s}")
    summary = {}
    for sys_name, cells in per_system.items():
        ps = np.array([c["p"] for c in cells])
        m = ps.mean()
        s = ps.std(ddof=1) if ps.size > 1 else 0.0
        cv = 100 * s / abs(m) if m != 0 else 0.0
        rng = f"[{ps.min():.3f}, {ps.max():.3f}]"
        print(f"  {sys_name:>32s} {len(cells):3d} {m:8.3f} {s:8.3f} {cv:8.2f} {rng:>16s}")
        summary[sys_name] = {
            "n_cells": len(cells), "p_mean": float(m), "p_std": float(s),
            "p_cv_pct": float(cv), "p_min": float(ps.min()),
            "p_max": float(ps.max()),
        }

    all_ps = np.array([c["p"] for cells in per_system.values() for c in cells])
    between_system_means = np.array([s["p_mean"] for s in summary.values()])
    print(f"\n  between-system p mean spread: "
          f"{between_system_means.min():.3f} - {between_system_means.max():.3f} "
          f"(spread {between_system_means.max()-between_system_means.min():.3f})")
    print(f"  within-system p CV range: "
          f"{min(s['p_cv_pct'] for s in summary.values()):.2f}% - "
          f"{max(s['p_cv_pct'] for s in summary.values()):.2f}%")
    print(f"  all 27 cells: mean {all_ps.mean():.3f}, std {all_ps.std(ddof=1):.3f}, "
          f"CV {100*all_ps.std(ddof=1)/all_ps.mean():.2f}%")

    out = {"per_system": summary,
           "overall": {"n_cells": int(all_ps.size),
                       "p_mean": float(all_ps.mean()),
                       "p_std": float(all_ps.std(ddof=1)),
                       "p_cv_pct": float(100 * all_ps.std(ddof=1) / all_ps.mean())}}
    out_path = _ROOT / "results/gate2/depth_stability_analysis.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
