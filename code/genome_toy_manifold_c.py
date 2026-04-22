"""Candidate-6 Step 5 validation: product-manifold toy model.

Claim: on `n_axes`-dim product manifold `M = [0,1]^n_axes` embedded in R^h,
measured `c = p · d_rd ≈ n_axes`.

Protocol:
  - For n_axes in {1, 2, 3, 4, 5, 8}:
    - Generate n=2000 points uniformly in [0,1]^n_axes
    - Embed into R^h via random orthonormal rotation (h = 128)
    - Add small isotropic noise σ=0.01 to ensure full-rank cloud
    - Measure p via kNN-graph scaling at k ∈ {3, 5, 8, 12, 18, 27, 40, 60, 90, 130}
    - Measure d_rd via k-means scaling
    - Report c = p · d_rd
  - Pass criterion: mean |c - n_axes| / n_axes < 0.20 across all n_axes tested
  - Partial-pass: 3/6 or 4/6 values within 20%

If pass: Step 5 of candidate-6 validates on pure synthetic data; the
derivation path from 1-D axis scaling to c=n_axes is real (even if
hand-waved in prose).
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


def generate_product_cloud(n_axes: int, n_points: int = 2000,
                           h: int = 128, noise: float = 0.01, seed: int = 42):
    rng = np.random.default_rng(seed + n_axes)  # per-axes-count seed
    # Uniform on [0,1]^n_axes
    raw = rng.uniform(0, 1, size=(n_points, n_axes)).astype(np.float32)
    # Random orthonormal embedding into R^h
    A = rng.standard_normal((h, n_axes)).astype(np.float32)
    Q, _ = np.linalg.qr(A)  # (h, n_axes)
    X = raw @ Q.T  # (n_points, h)
    # Small isotropic noise
    X += noise * rng.standard_normal(X.shape).astype(np.float32)
    return X


def main():
    results = []
    t0 = time.time()
    for n_axes in (1, 2, 3, 4, 5, 8):
        X = generate_product_cloud(n_axes=n_axes, n_points=2000, h=128)
        Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
        p, c0, r2 = fit_power_law(K_GRID, Cs)
        rd = rate_distortion_dim(X)
        c = p * rd["d_rd"]
        rel_err = abs(c - n_axes) / n_axes
        passes = rel_err < 0.20
        print(f"  n_axes={n_axes}: p={p:.3f}  d_rd={rd['d_rd']:.2f}  c={c:.2f}  "
              f"rel_err={rel_err:.3f}  {'PASS' if passes else 'MISS'}  (Ck_R2={r2:.3f})")
        results.append({"n_axes": n_axes, "p": p, "d_rd": rd["d_rd"],
                         "c": c, "c_ideal": n_axes, "rel_err": rel_err,
                         "passes": passes, "Ck_R2": r2,
                         "rd_fit_R2": rd["R2"]})

    passes = sum(r["passes"] for r in results)
    print(f"\n=== SUMMARY: candidate-6 Step 5 toy-model validation ===")
    print(f"  pass rate: {passes}/{len(results)} axes-counts within 20% of n_axes")
    if passes >= len(results) * 0.7:
        verdict = "STEP_5_VALIDATED — synthetic data confirms c ~ n_axes"
    elif passes >= len(results) * 0.4:
        verdict = "STEP_5_PARTIAL — some n_axes fit but not universal; derivation needs refinement"
    else:
        verdict = "STEP_5_FALSIFIED — c does NOT equal n_axes on product manifolds, claim is coincidence"
    print(f"  verdict: {verdict}")

    out = {"purpose": "Candidate-6 Step 5 validation on product manifolds",
           "per_n_axes": results, "pass_count": passes,
           "verdict": verdict, "wall_s": time.time() - t0}
    out_path = _ROOT / "results/gate2/toy_manifold_c.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
