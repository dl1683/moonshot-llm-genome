"""Block-bootstrap empirical SE on kNN-10 via fresh extractions.

Codex R8 follow-up to the SE sanity check (`genome_se_sanity.py`): the
earlier script estimated empirical SE from cross-seed spread (3 seeds),
which gives only 1 degree of freedom per (system, depth) cell. For the
paper we want a *proper* bootstrap SE estimate — many resamples of the
same stimulus family, compute C on each, take std of the C values.

This runs fresh extraction of kNN-10 on B independent resampled clouds
per (system, depth) cell and reports bootstrap SE alongside analytic SE.
CPU-heavy but trivially parallelizable over cells (24 cores available).

Strategy: for each existing per-system atlas file that stores n=2000
results, we re-extract only the CPU-level primitives (not GPU activation
extraction — we use the saved row-level per-point-C statistics where
available). Since our atlas rows only store the scalar C and analytic SE
not the raw cloud, we use an alternate strategy: re-extract the kNN
coefficient on a synthetic manifold of matched n and d_int estimate,
then compare to the observed atlas values to back out the expected SE.

Actually the cleanest test: just run block-bootstrap on fresh Gaussian
clouds at the REAL ambient dimensions of each system, since the random-
Gaussian baseline is already computed and a solid null. This calibrates
the bootstrap-vs-analytic SE gap in a REFERENCE regime where we know
the answer.

Usage:
    python code/genome_bootstrap_se.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


def _single_cloud_knn10(seed: int, n: int, h: int) -> float:
    """Worker: compute C(X, k=10) on one Gaussian cloud."""
    from genome_primitives import knn_clustering_coefficient
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, h)).astype(np.float32)
    m = knn_clustering_coefficient(X, k=10)
    return float(m.value)


def bootstrap_se_at(n: int, h: int, B: int = 50,
                    max_workers: int = 20) -> dict:
    """Run B independent Gaussian clouds at (n, h). Return bootstrap SE
    + analytic SE from one cloud for comparison.
    """
    from genome_primitives import knn_clustering_coefficient
    # 1 reference cloud to report analytic SE
    ref = _single_cloud_knn10(seed=42, n=n, h=h)
    rng_ref = np.random.default_rng(42)
    X_ref = rng_ref.standard_normal((n, h)).astype(np.float32)
    m_ref = knn_clustering_coefficient(X_ref, k=10)
    analytic_se = float(m_ref.se)

    # B fresh bootstraps
    seeds = list(range(1000, 1000 + B))
    values: list[float] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_single_cloud_knn10, s, n, h): s for s in seeds}
        for f in as_completed(futures):
            values.append(f.result())
    elapsed = time.time() - t0
    empirical_se = float(np.std(values, ddof=1))
    return {
        "n": n, "h": h, "B": B,
        "analytic_se_from_one_cloud": analytic_se,
        "empirical_bootstrap_se": empirical_se,
        "ratio": empirical_se / (analytic_se + 1e-12),
        "C_mean_across_B": float(np.mean(values)),
        "C_std_across_B": empirical_se,
        "C_min": float(np.min(values)),
        "C_max": float(np.max(values)),
        "wall_clock_seconds": round(elapsed, 2),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", "--bootstraps", type=int, default=50)
    ap.add_argument("-n", type=int, default=2000)
    ap.add_argument("--h-list", type=int, nargs="+",
                    default=[384, 768, 1024, 1536])
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    print(f"Bootstrapping kNN-10 SE at n={args.n}, h in {args.h_list}, "
          f"B={args.bootstraps}, workers={args.workers}")
    out_dir = _THIS_DIR.parent / "results" / "gate1"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "n": args.n, "B": args.bootstraps,
                     "per_h": {}}
    for h in args.h_list:
        print(f"\n=== n={args.n}, h={h} ===")
        r = bootstrap_se_at(args.n, h, B=args.bootstraps,
                            max_workers=args.workers)
        print(f"  analytic SE (1 cloud): {r['analytic_se_from_one_cloud']:.5f}")
        print(f"  empirical SE (B={args.bootstraps} clouds): {r['empirical_bootstrap_se']:.5f}")
        print(f"  ratio empirical / analytic: {r['ratio']:.3f}")
        print(f"  C mean [min, max]: {r['C_mean_across_B']:.4f} "
              f"[{r['C_min']:.4f}, {r['C_max']:.4f}]")
        print(f"  wall-clock: {r['wall_clock_seconds']:.1f}s")
        results["per_h"][str(h)] = r

    out_path = out_dir / "bootstrap_se_random_gaussian.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nout: {out_path}")
