"""SE sanity check for kNN-10 clustering coefficient (Codex R8 action #5).

The Gate-1 machinery computes analytic SE as:
    SE(C) ≈ std(C_i) / √n
treating per-point clustering values C_i as independent samples.

R8 kill-shot (research integrity): on a kNN graph, the C_i values are
coupled (each point's neighbor set overlaps with its neighbors'
neighbor sets), so effective sample size is < n. If true, the analytic
SE UNDERESTIMATES the real variance, and every Gate-1 "pass" at
δ=0.10 is partially statistical luck.

This script estimates the ratio empirical_SE / analytic_SE via a
2-part test:

PART A (synthetic baseline). On a random Gaussian cloud of matched n and
h (where points are genuinely iid), the analytic formula should be
accurate. We verify this — if the synthetic ratio >> 1 the formula
itself is wrong; if ≈ 1 the formula is correct under iid.

PART B (empirical chunked SE on the existing atlas). For each system
whose atlas we have, we re-extract the point cloud at one depth, then:
  - compute analytic SE from the primitive
  - compute empirical SE by splitting the cloud into B disjoint blocks
    of size n_block, computing C on each, taking std(C_blocks).
    Scale back to full-n: empirical_SE ≈ std(C_blocks) / √B.
  - report ratio empirical / analytic.

Per CLAUDE.md §6.1 this is a dedicated kill-shot script for the SE
assumption, justifying a new file.

Usage:
    python code/genome_se_sanity.py --synthetic
    python code/genome_se_sanity.py --system qwen3-0.6b --n 2000
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


def synthetic_sanity(n_trials: int = 20,
                     n: int = 2000, h: int = 768,
                     k: int = 10, n_block: int = 200,
                     seed: int = 42) -> dict:
    """PART A: synthetic Gaussian baseline.

    Points are iid → analytic SE should be accurate. We verify with
    block-bootstrap. Multiple trials to get a distribution.
    """
    rng = np.random.default_rng(seed)
    ratios = []
    analytic_ses = []
    empirical_ses = []
    for trial in range(n_trials):
        X = rng.standard_normal((n, h)).astype(np.float32)
        # Analytic SE (primitive output)
        m = knn_clustering_coefficient(X, k=k)
        analytic_se = float(m.se)
        analytic_ses.append(analytic_se)
        # Empirical SE via disjoint block means
        assert n % n_block == 0, f"n={n} not divisible by block size {n_block}"
        B = n // n_block
        block_means = []
        for b in range(B):
            Xb = X[b * n_block:(b + 1) * n_block]
            mb = knn_clustering_coefficient(Xb, k=k)
            block_means.append(mb.value)
        # std of block means ≈ std of C-at-block-size
        std_block = float(np.std(block_means, ddof=1))
        # Convert to std-of-mean-at-full-n. Under iid, std(C at block size)
        # = std(C_i) / √n_block. Then std(C at full n) = std(C_i) / √n =
        # std(C at block size) × √(n_block / n) = std_block × √(1/B).
        empirical_se = std_block * np.sqrt(1.0 / B)
        empirical_ses.append(empirical_se)
        ratio = empirical_se / (analytic_se + 1e-12)
        ratios.append(ratio)

    return {
        "n_trials": n_trials,
        "n": n, "h": h, "k": k, "n_block": n_block,
        "analytic_se_mean": float(np.mean(analytic_ses)),
        "analytic_se_std": float(np.std(analytic_ses, ddof=1)),
        "empirical_se_mean": float(np.mean(empirical_ses)),
        "empirical_se_std": float(np.std(empirical_ses, ddof=1)),
        "ratio_mean": float(np.mean(ratios)),
        "ratio_std": float(np.std(ratios, ddof=1)),
        "ratio_min": float(np.min(ratios)),
        "ratio_max": float(np.max(ratios)),
    }


def _load_atlas_for_system(system_key: str) -> list[dict]:
    """Load all atlas rows for a single system across seeds."""
    root = Path(__file__).parent.parent / "results" / "cross_arch"
    rows: list[dict] = []
    for p in sorted(root.glob("atlas_rows_n2000_c4_seed*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        for r in d.get("rows", []):
            if r["system_key"] == system_key:
                rows.append(r)
    return rows


def real_cloud_sanity(system_key: str, n_block: int = 200, k: int = 10) -> dict:
    """PART B: chunked-SE sanity on an EXISTING extraction.

    We don't have the full point cloud persisted, so instead we exploit
    the following: for each (seed, depth) cell, the Gate-1 G1.3 machinery
    is comparing cell values ACROSS seeds. The relative spread of C across
    the 3 seeds IS an empirical SE estimate, and we can compare it to the
    analytic SE per cell.

    If empirical_SE >> analytic_SE, the analytic formula undershoots —
    meaning Gate-1 margins are too tight.
    """
    from collections import defaultdict
    rows = _load_atlas_for_system(system_key)
    # Filter to kNN-k10 only
    rows = [r for r in rows if r.get("primitive_id") == "knn_clustering"
            and r.get("estimator") == f"knn_k{k}"]
    if not rows:
        return {"error": f"no kNN-k{k} rows for {system_key}"}

    # Group by (depth, seed)
    by_cell = defaultdict(dict)
    for r in rows:
        depth = round(float(r["k_normalized"]), 2)
        seed = int(r["seed"])
        by_cell[depth][seed] = (float(r["value"]), float(r["se"]))

    per_depth_results = []
    for depth, per_seed in sorted(by_cell.items()):
        seeds = sorted(per_seed.keys())
        if len(seeds) < 2:
            continue
        vals = np.array([per_seed[s][0] for s in seeds])
        ses = np.array([per_seed[s][1] for s in seeds])
        # empirical SE = std across seeds (each seed is a re-sampled cloud at
        # same n=2000). Under iid, std_across_seeds ≈ SE_per_cloud. If
        # std_across_seeds >> analytic SE, the primitive's SE is wrong.
        empirical_se = float(np.std(vals, ddof=1))
        mean_analytic_se = float(np.mean(ses))
        per_depth_results.append({
            "depth": depth,
            "n_seeds": len(seeds),
            "C_mean": float(np.mean(vals)),
            "C_per_seed": [float(v) for v in vals],
            "empirical_se_across_seeds": empirical_se,
            "analytic_se_mean": mean_analytic_se,
            "ratio": empirical_se / (mean_analytic_se + 1e-12),
        })

    if not per_depth_results:
        return {"error": "no multi-seed cells"}

    ratios = [d["ratio"] for d in per_depth_results]
    return {
        "system_key": system_key,
        "k": k,
        "per_depth": per_depth_results,
        "mean_ratio": float(np.mean(ratios)),
        "max_ratio": float(np.max(ratios)),
        "min_ratio": float(np.min(ratios)),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true",
                    help="run synthetic Gaussian baseline (Part A)")
    ap.add_argument("--all-systems", action="store_true",
                    help="run Part B across all available systems")
    ap.add_argument("--system", type=str, default=None,
                    help="run Part B on a single system")
    ap.add_argument("--n-trials", type=int, default=5)
    args = ap.parse_args()

    out_dir = _THIS_DIR.parent / "results" / "gate1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "se_sanity.json"
    payload: dict = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    if args.synthetic or not any([args.all_systems, args.system]):
        print("=== PART A: synthetic Gaussian baseline ===")
        res_a = synthetic_sanity(n_trials=args.n_trials)
        payload["part_a_synthetic"] = res_a
        print(f"  n_trials={res_a['n_trials']}, n={res_a['n']}, h={res_a['h']}, k={res_a['k']}")
        print(f"  analytic SE (mean ± std): {res_a['analytic_se_mean']:.5f} ± {res_a['analytic_se_std']:.5f}")
        print(f"  empirical SE (mean ± std): {res_a['empirical_se_mean']:.5f} ± {res_a['empirical_se_std']:.5f}")
        print(f"  ratio empirical/analytic: mean {res_a['ratio_mean']:.3f}, "
              f"range [{res_a['ratio_min']:.3f}, {res_a['ratio_max']:.3f}]")
        print(f"  interpretation: ratio ~1 under iid means analytic formula correct; "
              f"ratio > 1.5 means undershoot")

    if args.system or args.all_systems:
        print()
        print("=== PART B: real-cloud chunked SE via cross-seed spread ===")
        systems_to_test = ([args.system] if args.system else
                           ["qwen3-0.6b", "rwkv-4-169m", "dinov2-small",
                            "falcon-h1-0.5b", "deepseek-r1-distill-qwen-1.5b"])
        b_results: dict = {}
        for sys_ in systems_to_test:
            print(f"  [{sys_}]")
            res_b = real_cloud_sanity(sys_)
            b_results[sys_] = res_b
            if "error" in res_b:
                print(f"    ERROR: {res_b['error']}")
                continue
            print(f"    mean ratio (empirical / analytic): {res_b['mean_ratio']:.3f}  "
                  f"max {res_b['max_ratio']:.3f}  min {res_b['min_ratio']:.3f}")
        payload["part_b_real_cloud"] = b_results

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\nout: {out_path}")
