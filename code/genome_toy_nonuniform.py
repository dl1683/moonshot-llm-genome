"""Candidate-7: does NON-uniform density per axis produce c = n_axes?

Candidate-6 Step 5 falsified on uniform-per-axis product manifolds (genome_054).
Synthesis §mechanism-open: possibility (b) is training-induced non-generic
structure. Trained networks are known to have HEAVY-TAILED activation
distributions (power-law, Pareto, log-normal). Maybe heavy-tailed density
per axis is what produces c = n_axes scaling.

Test: replace uniform [0,1] per axis with a power-law density (Pareto tail)
and see if c tracks n_axes.

Per-axis density: x_i = |y_i|^{-1/alpha} where y_i ~ N(0,1), alpha = 1.5
(heavy-tail scaling exponent similar to trained-network activation reports).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_primitives import knn_clustering_coefficient  # noqa: E402
from genome_rate_distortion_probe import rate_distortion_dim, fit_power_law  # noqa: E402

_ROOT = _THIS_DIR.parent
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]


def generate_nonuniform_cloud(n_axes, n_points=2000, h=128, alpha=1.5, noise=0.01, seed=42):
    """Each axis is heavy-tailed (Pareto-like). Embedded in R^h via random orthonormal."""
    rng = np.random.default_rng(seed + n_axes)
    # Heavy-tailed per-axis samples: Pareto tail
    # x ~ standard_t or Pareto
    raw = rng.standard_t(df=2.5, size=(n_points, n_axes)).astype(np.float32)
    # Clip extreme tails
    raw = np.clip(raw, -10, 10)
    # Standardize
    raw = (raw - raw.mean(axis=0)) / (raw.std(axis=0) + 1e-6)
    # Embed
    A = rng.standard_normal((h, n_axes)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    X = raw @ Q.T
    X += noise * rng.standard_normal(X.shape).astype(np.float32)
    return X


def main():
    results = []
    t0 = time.time()
    for n_axes in (1, 2, 3, 4, 5, 8):
        X = generate_nonuniform_cloud(n_axes=n_axes, n_points=2000, h=128, alpha=1.5)
        Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
        p, c0, r2 = fit_power_law(K_GRID, Cs)
        rd = rate_distortion_dim(X)
        c = p * rd["d_rd"]
        rel_err = abs(c - n_axes) / n_axes
        passes = rel_err < 0.20
        print(f"  n_axes={n_axes}: p={p:.3f}  d_rd={rd['d_rd']:.2f}  c={c:.2f}  "
              f"rel_err={rel_err:.3f}  {'PASS' if passes else 'MISS'}")
        results.append({"n_axes": n_axes, "p": p, "d_rd": rd["d_rd"],
                         "c": c, "rel_err": rel_err, "passes": passes})

    passes = sum(r["passes"] for r in results)
    print(f"\n=== CANDIDATE-7: HEAVY-TAILED PRODUCT MANIFOLD ===")
    print(f"  pass rate: {passes}/{len(results)} axes-counts within 20% of n_axes")
    if passes >= len(results) * 0.7:
        verdict = "HEAVY_TAIL_VALIDATES — non-uniform density produces c = n_axes"
    elif passes >= len(results) * 0.3:
        verdict = "HEAVY_TAIL_PARTIAL — density non-uniformity helps but not universal"
    else:
        verdict = "HEAVY_TAIL_FALSIFIED — density alone not the missing ingredient"
    print(f"  verdict: {verdict}")

    out = {"purpose": "Candidate-7: heavy-tailed product manifold c test",
           "per_n_axes": results, "pass_count": passes, "verdict": verdict}
    out_path = _ROOT / "results/gate2/toy_nonuniform_c.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
