"""Pilot for v2 derivation framework B (doubling-dimension ratio).

Per `research/derivations/power_law_v2_candidates.md` framework B:

    C(X, k) ~ k^p  with  p = (h - d_db) / d_db

where h is ambient (hidden) dimension and d_db is the doubling dimension.

Practical doubling-dim estimator (k-NN distance scaling):
For a point on a d_db-dim manifold, the k-th-NN distance r_k scales as
r_k ~ k^{1/d_db}. So log(r_k) vs log(k) has slope 1/d_db.
We estimate d_db from the slope at k in {8, 16, 32, 64, 128}.

Pilot on 3 systems x 1 depth (same as framework A pilot, n=1000 C4 seed 42):
  - Qwen3-0.6B mid
  - RWKV-4-169M mid
  - DeepSeek-R1-Distill-Qwen-1.5B mid
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


def doubling_dim_knn_scaling(X: np.ndarray, k_grid=(8, 16, 32, 64, 128)) -> dict:
    """Estimate doubling dim from log-log slope of median r_k vs k."""
    from sklearn.neighbors import NearestNeighbors
    max_k = max(k_grid)
    nn = NearestNeighbors(n_neighbors=max_k + 1, algorithm="auto",
                          n_jobs=1).fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)
    # dists[:, 0] is self-distance 0; take k-th neighbor = dists[:, k]
    r_k_median = np.array([np.median(dists[:, k]) for k in k_grid])
    log_k = np.log(np.asarray(k_grid, dtype=float))
    log_r = np.log(r_k_median)
    slope, intercept = np.polyfit(log_k, log_r, 1)
    pred = slope * log_k + intercept
    ss_res = float(np.sum((log_r - pred) ** 2))
    ss_tot = float(np.sum((log_r - log_r.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # slope = 1/d_db, so d_db = 1/slope
    d_db = 1.0 / slope if slope > 1e-6 else float("inf")
    return {
        "d_db": float(d_db),
        "slope_log_rk_vs_log_k": float(slope),
        "R2_fit": float(r2),
        "k_grid": list(k_grid),
        "r_k_median": r_k_median.tolist(),
    }


def load_empirical_p(system_key: str, target_depth: float = 0.52) -> float | None:
    p_file = _ROOT / "results/gate2/ck_power_fit.json"
    if not p_file.exists():
        return None
    d = json.loads(p_file.read_text())
    best = None
    best_gap = float("inf")
    for cell, v in d.get("per_cell", {}).items():
        if cell.startswith(system_key):
            try:
                depth = float(cell.split("depth")[1])
            except Exception:
                continue
            gap = abs(depth - target_depth)
            if gap < best_gap:
                best_gap = gap
                best = v.get("p_slope")
    return best


def run_one(hf_id: str, system_key: str, n: int, seed: int):
    t0 = time.time()
    print(f"\n=== {system_key} ===")
    sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    n_layers = sys_obj.n_hidden_layers()
    mid = n_layers // 2

    sents = []
    for rec in c4_clean_v1(seed=seed, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= n:
            break

    traj = extract_trajectory(
        model=sys_obj.model, tokenizer=sys_obj.tokenizer,
        texts=sents, layer_indices=[mid], pooling="seq_mean",
        device="cuda", system_key=system_key, class_id=0,
        quantization="fp16",
        stimulus_version=f"c4_clean.v1.seed{seed}.n{n}",
        seed=seed, batch_size=16, max_length=256,
    )
    X = traj.layers[0].X.astype(np.float32)
    h = X.shape[1]

    db = doubling_dim_knn_scaling(X)
    d_db = db["d_db"]
    p_pred = (h - d_db) / d_db if d_db > 0 else float("nan")
    p_emp = load_empirical_p(system_key)
    rel_err = None
    if p_emp is not None and abs(p_emp) > 1e-9:
        rel_err = abs(p_pred - p_emp) / abs(p_emp)

    print(f"  h={h}  d_db={d_db:.3f}  p_pred={p_pred:.3f}  p_emp={p_emp}  rel_err={rel_err}")

    sys_obj.unload()
    torch.cuda.empty_cache()

    return {
        "system_key": system_key, "hf_id": hf_id,
        "depth_idx": mid, "n_stimuli": int(X.shape[0]),
        "hidden_dim": h,
        "d_db": d_db,
        "knn_scaling_R2": db["R2_fit"],
        "p_predicted_framework_B": p_pred,
        "p_empirical": p_emp,
        "rel_error": rel_err,
        "elapsed_s": time.time() - t0,
    }


def main():
    systems = [
        ("Qwen/Qwen3-0.6B", "qwen3-0.6b"),
        ("RWKV/rwkv-4-169m-pile", "rwkv-4-169m"),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
         "deepseek-r1-distill-qwen-1.5b"),
    ]
    results = []
    for hf_id, system_key in systems:
        try:
            results.append(run_one(hf_id, system_key, n=1000, seed=42))
        except Exception as e:
            print(f"FAILED on {system_key}: {type(e).__name__}: {e}")
            results.append({"system_key": system_key,
                            "error": f"{type(e).__name__}: {e}"})

    print("\n=== SUMMARY: FRAMEWORK B ===")
    print(f"{'system':>32s} {'h':>6s} {'d_db':>8s} {'p_pred':>8s} {'p_emp':>8s} {'rel_err':>8s}")
    passes = 0
    tested = 0
    for r in results:
        if "error" in r:
            print(f"  {r['system_key']:>32s}  ERROR: {r['error']}")
            continue
        print(f"  {r['system_key']:>32s} {r['hidden_dim']:6d} "
              f"{r['d_db']:8.3f} {r['p_predicted_framework_B']:8.3f} "
              f"{r['p_empirical']:8.3f} {r['rel_error']:8.3f}")
        tested += 1
        if r["rel_error"] is not None and r["rel_error"] < 0.20:
            passes += 1
    verdict = "FRAMEWORK_B_SUPPORTED" if passes >= 2 else "FRAMEWORK_B_FALSIFIED"
    print(f"\nPasses (|rel_err| < 0.20): {passes} of {tested}. -> {verdict}")

    out = {"purpose": "Pilot B: doubling-dimension for v2 derivation",
           "per_system": results,
           "passes_20pct_criterion": passes,
           "n_systems_tested": tested,
           "verdict": verdict}
    out_path = _ROOT / "results/gate2/doubling_dim_pilot.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
