"""Gate-1 G1.2 rotation-invariance test for kNN-10 clustering coefficient.

Per the LOCKED prereg `research/prereg/genome_knn_k10_portability_2026-04-21.md`
§3, kNN-10 should be mathematically invariant under orthogonal rotations
of the point cloud (rotations preserve Euclidean distances). This is
asserted in §3.1 of the paper; this script empirically verifies it.

Test protocol:
  - Generate a realistic Gaussian cloud at n=2000, h=768 (matches several
    of our actual trained-network ambient dims).
  - Apply B random Haar-measure orthogonal rotations Q ∈ O(h) and compute
    C(QX, k=10) for each.
  - Verify that the spread across rotations is numerically ≤ float32
    precision — anything larger signals a bug in the primitive.

Success criterion: max_across_rotations(|ΔC|) / |C| < 1e-4 (float32 kNN
tie-breaking tolerance).

CPU-only, ≤1 min wall-clock at n=2000.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from genome_primitives import knn_clustering_coefficient  # noqa: E402


def haar_orthogonal(h: int, rng: np.random.Generator) -> np.ndarray:
    """Sample Q ∈ O(h) from Haar measure via QR of Gaussian."""
    G = rng.standard_normal((h, h))
    Q, R = np.linalg.qr(G)
    # Fix signs so Q is uniformly distributed on O(h)
    d = np.sign(np.diagonal(R))
    d[d == 0] = 1
    Q = Q * d
    return Q.astype(np.float32)


def g12_rotation_test(n: int = 2000, h: int = 768, B: int = 10,
                       seed: int = 42) -> dict:
    """Run rotation-invariance test. Returns C values across B rotations
    + relative spread."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, h)).astype(np.float32)
    # Baseline
    C_base = float(knn_clustering_coefficient(X, k=10).value)
    # Apply B rotations
    C_rot = []
    for b in range(B):
        Q = haar_orthogonal(h, rng)
        XR = X @ Q
        C_rot.append(float(knn_clustering_coefficient(XR, k=10).value))
    delta_max = float(np.max(np.abs(np.array(C_rot) - C_base)))
    rel_spread = delta_max / (C_base + 1e-12)
    return {
        "n": n, "h": h, "B": B,
        "C_baseline": C_base,
        "C_across_rotations": C_rot,
        "max_abs_delta": delta_max,
        "relative_spread": rel_spread,
        "float32_tol": 1e-4,
        "verdict": "PASS" if rel_spread < 1e-4 else
                   "LOOSE_PASS" if rel_spread < 1e-3 else "FAIL",
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=2000)
    ap.add_argument("--h-list", type=int, nargs="+",
                    default=[384, 768, 1024, 1536])
    ap.add_argument("-B", "--rotations", type=int, default=10)
    args = ap.parse_args()

    out_dir = _THIS_DIR.parent / "results" / "gate1"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
               "n": args.n, "B": args.rotations, "per_h": {}}
    print(f"G1.2 rotation invariance test  n={args.n}  B={args.rotations}")
    for h in args.h_list:
        t0 = time.time()
        r = g12_rotation_test(n=args.n, h=h, B=args.rotations)
        elapsed = time.time() - t0
        print(f"  h={h:5d}  C_base={r['C_baseline']:.5f}  "
              f"max_abs_dC={r['max_abs_delta']:.2e}  "
              f"rel={r['relative_spread']:.2e}  verdict={r['verdict']}  "
              f"({elapsed:.1f}s)")
        results["per_h"][str(h)] = r

    out_path = out_dir / "g12_rotation_invariance.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nout: {out_path}")
