"""Pilot for v2 derivation framework C (heavy-tailed NN-degree).

Per `research/derivations/power_law_v2_candidates.md` framework C: if the
k-NN in-degree distribution on the pooled hidden-state point cloud is
heavy-tailed with tail exponent alpha, then scale-free-graph theory predicts
C(k) ~ k^((3-alpha)/(alpha-1)), which would explain the observed p~=0.17
exponent.

Pilot design (fast, CPU-dominated given point cloud is small):
  1. Extract Qwen3-0.6B pooled mid-depth hidden states for n=1000 C4 stimuli
     (one re-extraction; same pipeline as cross_arch).
  2. For each k in {10, 30, 60}, compute the k-NN in-degree distribution
     (how many points list x as one of their k nearest).
  3. Fit a power-law tail to the upper half of the degree distribution
     (MLE per Clauset-Shalizi-Newman 2009).
  4. If alpha is well-defined (R^2 of log-log tail > 0.9, finite support),
     compute predicted p_C = (3 - alpha)/(alpha - 1), compare to empirical
     p for Qwen3 at this depth (0.14-0.18 per results/gate2/ck_power_fit.json).

Output: results/gate2/nn_degree_pilot_qwen3.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_extractor import extract_trajectory  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent


def nn_in_degree(X: np.ndarray, k: int) -> np.ndarray:
    """For each point, count how many other points list it in their k-NN set."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=1).fit(X)
    _, idxs = nn.kneighbors(X, return_distance=True)
    idxs = idxs[:, 1:]  # drop self-column
    in_deg = np.bincount(idxs.ravel(), minlength=X.shape[0])
    return in_deg


def fit_power_tail(degrees: np.ndarray) -> dict:
    """Fit a power law to the upper tail of degree distribution.

    Log-log linear regression on the complementary CDF: log P(X>=d) = -a*log d + const.
    Uses the top-half (d >= median) to avoid small-d noise.
    """
    d = np.asarray(degrees)
    d = d[d > 0]
    if d.size < 20:
        return {"alpha": None, "reason": "too-few-points"}
    d_sorted = np.sort(d)
    median = np.median(d_sorted)
    tail = d_sorted[d_sorted >= median]
    if tail.size < 10:
        return {"alpha": None, "reason": "too-few-tail"}
    # Empirical CCDF
    unique = np.unique(tail)
    ccdf = np.array([np.mean(tail >= u) for u in unique])
    if np.any(ccdf <= 0):
        mask = ccdf > 0
        unique = unique[mask]
        ccdf = ccdf[mask]
    if unique.size < 5:
        return {"alpha": None, "reason": "collapsed-tail"}
    x = np.log(unique.astype(float))
    y = np.log(ccdf)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # For P(X>=d) ~ d^{-(alpha-1)}, slope of log CCDF = -(alpha-1), so alpha = 1 - slope
    alpha = float(1.0 - slope)
    return {
        "alpha": alpha,
        "slope_ccdf": float(slope),
        "r2_loglog": float(r2),
        "n_tail": int(tail.size),
        "n_unique": int(unique.size),
        "median": float(median),
        "max": int(tail.max()),
    }


def main():
    t0 = time.time()
    print(f"[{time.time()-t0:.1f}s] loading Qwen3-0.6B fp16...")
    sys_obj = load_system("Qwen/Qwen3-0.6B", quant="fp16", untrained=False,
                          device="cuda")
    n_layers = sys_obj.n_hidden_layers()
    mid_idx = n_layers // 2  # ~depth 0.5
    print(f"[{time.time()-t0:.1f}s] Qwen3 {n_layers} layers, sampling mid {mid_idx}")

    print(f"[{time.time()-t0:.1f}s] loading 1000 C4-clean stimuli seed=42...")
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= 1000:
            break
    print(f"[{time.time()-t0:.1f}s] got {len(sents)} sentences")

    print(f"[{time.time()-t0:.1f}s] extracting pooled hidden states at layer {mid_idx}...")
    traj = extract_trajectory(
        model=sys_obj.model, tokenizer=sys_obj.tokenizer,
        texts=sents, layer_indices=[mid_idx],
        pooling="seq_mean", device="cuda",
        system_key="qwen3-0.6b", class_id=1,
        quantization="fp16",
        stimulus_version="c4_clean.v1.seed42.n1000",
        seed=42, batch_size=16,
        max_length=256,
    )
    X = traj.layers[0].X.astype(np.float32)
    print(f"[{time.time()-t0:.1f}s] X shape {X.shape}")

    results = {}
    for k in (10, 30, 60):
        in_deg = nn_in_degree(X, k=k)
        fit = fit_power_tail(in_deg)
        # For scale-free-graph relation: C_pred = k^((3-alpha)/(alpha-1))
        p_pred = None
        if fit.get("alpha") is not None and fit["alpha"] > 1.1:
            a = fit["alpha"]
            p_pred = (3 - a) / (a - 1)
        results[f"k={k}"] = {
            "in_degree_stats": {
                "mean": float(in_deg.mean()),
                "std": float(in_deg.std(ddof=1)),
                "max": int(in_deg.max()),
                "min": int(in_deg.min()),
                "skew": float(((in_deg - in_deg.mean()) ** 3).mean()
                              / (in_deg.std(ddof=1) ** 3 + 1e-12)),
            },
            "tail_fit": fit,
            "predicted_p_from_framework_C": p_pred,
        }
        r2 = fit.get("r2_loglog")
        r2_s = f"{r2:.3f}" if isinstance(r2, float) else "nan"
        print(f"  k={k}: alpha={fit.get('alpha')}, R2={r2_s}, p_pred={p_pred}")

    # Empirical Qwen3 p at mid-depth from existing fit
    existing = json.loads((_ROOT / "results/gate2/ck_power_fit.json").read_text())
    qwen3_mid_p = None
    for cell, v in existing.get("per_cell", {}).items():
        if cell.startswith("qwen3-0.6b") and "depth0.5" in cell:
            qwen3_mid_p = v.get("p_slope")
            break

    out = {
        "purpose": "Pilot C: heavy-tailed NN in-degree distribution",
        "system": "qwen3-0.6b",
        "depth_idx": mid_idx,
        "depth_normalized": mid_idx / (n_layers - 1),
        "n_stimuli": int(X.shape[0]),
        "hidden_dim": int(X.shape[1]),
        "per_k": results,
        "empirical_p_at_qwen3_mid": qwen3_mid_p,
        "wall_clock_s": time.time() - t0,
    }
    out_path = _ROOT / "results/gate2/nn_degree_pilot_qwen3.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[{time.time()-t0:.1f}s] wrote {out_path}")


if __name__ == "__main__":
    main()
