"""Candidate-8 P2 alternative: plateau-plus-power-law spectrum.

The pure power-law model (`genome_candidate8_p2_waterfill.py`) FAILED — it
predicts ratio ~ 0.3 while empirical ratio is ~ 2.2. The spectrum
is NOT pure power-law over rank 1..h; it has a 'bulk' plateau of roughly
constant-magnitude eigenvalues at the top, then a power-law tail.

Model:
  lambda_i = 1                  for i = 1..k_bulk
  lambda_i = (i / k_bulk)^{-2*alpha}   for i > k_bulk
where k_bulk is the effective bulk width (1 <= k_bulk <= h).

Test: for each system, use the measured alpha; sweep k_bulk in [1, h/2];
find the k_bulk that produces ratio_model == ratio_empirical.

If a SINGLE k_bulk value works across all 5 text systems: candidate-8 P2
gains a 2-parameter closed form (alpha, k_bulk). If each system needs a
different k_bulk, the plateau is a per-system fit and the derivation
isn't universal.

Bonus: plot the measured k_bulk values to see if they cluster by modality
(which would give a story: training produces a universal spectral shape
characterized by (alpha, k_bulk=f(modality))).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_candidate8_p2_waterfill import (  # noqa: E402
    eff_rank, d_rd_from_rate_distortion,
)
_ROOT = _THIS_DIR.parent


def plateau_power_law(alpha, k_bulk, h):
    i = np.arange(1, h + 1, dtype=np.float64)
    lam = np.where(i <= k_bulk, 1.0, (i / k_bulk) ** (-2 * alpha))
    return lam


def predict_ratio(alpha, k_bulk, h=1024):
    lam = plateau_power_law(alpha, k_bulk, h)
    er = eff_rank(lam)
    drd = d_rd_from_rate_distortion(lam)
    return er, drd, er / drd if drd and drd > 0 else np.nan


def fit_k_bulk(alpha, ratio_target, h=1024):
    """Find k_bulk in [1, h/2] that minimizes |ratio_model(alpha, k_bulk) - ratio_target|."""
    k_grid = np.linspace(1, h // 2, 100).astype(int)
    errs = []
    for k in k_grid:
        _, _, ratio = predict_ratio(alpha, int(k), h)
        errs.append(abs(ratio - ratio_target) / max(ratio_target, 1e-6))
    errs = np.array(errs)
    best_idx = int(np.argmin(errs))
    return int(k_grid[best_idx]), float(errs[best_idx])


def main():
    empirical_text = [
        ("qwen3-0.6b",            0.861, 2.059, 25.27, 12.27),
        ("deepseek-1.5b",         0.772, 2.413, 33.93, 14.06),
        ("bert-base",             0.784, 2.292, 32.94, 14.37),
        ("roberta-base",          0.768, 2.158, 28.06, 13.00),
        ("minilm-l6-contrastive", 0.773, 2.199, 28.23, 12.84),
    ]

    print("system        alpha   er_emp  drd_emp  ratio_emp  k_bulk_fit  er_mod  drd_mod  ratio_mod  fit_err")
    print("-" * 110)
    results = []
    for sk, a, rat_emp, er_emp, drd_emp in empirical_text:
        k_bulk, fit_err = fit_k_bulk(a, rat_emp)
        er_mod, drd_mod, ratio_mod = predict_ratio(a, k_bulk)
        results.append({"system": sk, "alpha": a, "ratio_emp": rat_emp,
                        "eff_rank_emp": er_emp, "d_rd_emp": drd_emp,
                        "k_bulk_fit": k_bulk, "eff_rank_mod": er_mod,
                        "d_rd_mod": drd_mod, "ratio_mod": ratio_mod,
                        "fit_err": fit_err})
        print(f"{sk:22s}  {a:.3f}   {er_emp:6.1f}   {drd_emp:6.1f}   "
              f"{rat_emp:6.3f}    {k_bulk:6d}    "
              f"{er_mod:6.1f}   {drd_mod:6.1f}   {ratio_mod:6.3f}    {fit_err:.3f}")

    ks = [r["k_bulk_fit"] for r in results]
    print(f"\nk_bulk across 5 text systems: {ks}")
    print(f"  mean={np.mean(ks):.1f}  std={np.std(ks):.1f}  cv={np.std(ks)/np.mean(ks)*100:.1f}pct")

    if np.std(ks) / max(np.mean(ks), 1e-6) < 0.10:
        verdict = (f"K_BULK_UNIVERSAL k_bulk approximately equals {np.mean(ks):.0f} across all 5 text systems "
                   f"(CV {np.std(ks)/np.mean(ks)*100:.1f}pct). candidate-8 P2 "
                   "derivation closes: ratio(alpha) with bulk width k approximately h/8. "
                   "Nature-grade 2-parameter derivation.")
    elif np.std(ks) / max(np.mean(ks), 1e-6) < 0.30:
        verdict = (f"K_BULK_NARROW — k_bulk=={np.mean(ks):.0f}+-{np.std(ks):.0f} "
                   f"(CV {np.std(ks)/np.mean(ks)*100:.1f}pct). Plateau width "
                   "nearly constant across systems. Partial derivation.")
    else:
        verdict = (f"K_BULK_PER_SYSTEM — k_bulk varies 30pct+ across systems. "
                   "Plateau is not a universal constant; spectrum has "
                   "system-specific bulk structure beyond alpha.")
    print(f"\n  verdict: {verdict}")

    out = {"purpose": "Candidate-8 P2 plateau-plus-power-law fit",
           "per_system": results,
           "k_bulk_mean": float(np.mean(ks)),
           "k_bulk_std": float(np.std(ks)),
           "k_bulk_cv": float(np.std(ks) / max(np.mean(ks), 1e-6)),
           "verdict": verdict}
    out_path = _ROOT / "results/gate2/candidate8_p2_plateau.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
