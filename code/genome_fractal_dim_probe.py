"""Pilot for v2 derivation framework A (fractal d_2 / d_int gap).

Per `research/derivations/power_law_v2_candidates.md` framework A:

    C(X, k) ~ k^p  with  p = d_2 / d_int - 1

where d_2 is the fractal correlation dimension (Grassberger-Procaccia),
d_int is the intrinsic (pointwise) dimension (TwoNN or MLE).

Pilot on 3 systems x 1 depth:
  - Qwen3-0.6B mid
  - RWKV-4-169M mid
  - DeepSeek-R1-Distill-Qwen-1.5B mid

Each extracts n=1000 pooled hidden states on C4 seed 42, then computes:
  - d_int via TwoNN (existing genome_primitives.twonn_id)
  - d_2 via Grassberger-Procaccia correlation integral:
      C_2(r) = (2 / n(n-1)) * sum_{i<j} 1[||x_i - x_j|| <= r]
    Slope of log C_2(r) vs log r in the scaling regime = d_2.
  - empirical p (from results/gate2/ck_power_fit.json)
  - predicted p_A = d_2 / d_int - 1
  - |predicted - empirical| / empirical

Framework A passes if |delta| / empirical < 0.20 on >=2/3 systems.
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
from genome_primitives import twonn_id  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent


def grassberger_procaccia_d2(X: np.ndarray, n_r_bins: int = 40,
                              tail_frac_lo: float = 0.05,
                              tail_frac_hi: float = 0.50) -> dict:
    """Estimate correlation dimension d_2 via GP log-log slope.

    Computes all pairwise distances, then the correlation integral C_2(r)
    at 40 log-spaced r values in [min_dist, median_dist]. Fits a line in
    the scaling regime (between tail_frac_lo and tail_frac_hi of the
    log-r range).
    """
    n = X.shape[0]
    # All pairwise sq-distances via chunked compute to bound memory.
    # For n=1000, full matrix is 8MB — fine.
    from scipy.spatial.distance import pdist
    D = pdist(X, metric="euclidean")  # shape (n*(n-1)/2,)
    D = D[D > 0]  # drop zeros (should be none from distinct points)

    r_min = np.percentile(D, 1.0)
    r_max = np.percentile(D, 75.0)
    rs = np.exp(np.linspace(np.log(r_min), np.log(r_max), n_r_bins))
    # Correlation integral — fraction of pairs within r
    C2 = np.array([np.mean(D <= r) for r in rs])
    # Scaling-regime fit
    log_r = np.log(rs)
    log_C = np.log(C2 + 1e-12)
    lo = int(n_r_bins * tail_frac_lo)
    hi = int(n_r_bins * tail_frac_hi)
    slope, intercept = np.polyfit(log_r[lo:hi], log_C[lo:hi], 1)
    pred = slope * log_r[lo:hi] + intercept
    ss_res = float(np.sum((log_C[lo:hi] - pred) ** 2))
    ss_tot = float(np.sum((log_C[lo:hi] - log_C[lo:hi].mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "d_2": float(slope),
        "R2_fit": float(r2),
        "r_min": float(r_min), "r_max": float(r_max),
        "n_pairs": int(D.size),
        "fit_range_indices": [lo, hi],
    }


def load_empirical_p(system_key: str, target_depth: float = 0.52) -> float | None:
    """Look up the empirical p value for the closest matching depth."""
    p_file = _ROOT / "results/gate2/ck_power_fit.json"
    if not p_file.exists():
        return None
    d = json.loads(p_file.read_text())
    best = None
    best_gap = float("inf")
    for cell, v in d.get("per_cell", {}).items():
        if cell.startswith(system_key):
            # cell is "system||depthX.XX"
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
    print(f"\n=== {system_key} ({hf_id}) ===")
    sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    n_layers = sys_obj.n_hidden_layers()
    mid = n_layers // 2
    print(f"[{time.time()-t0:.1f}s] {n_layers} layers, sampling mid {mid}")

    sents = []
    for rec in c4_clean_v1(seed=seed, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= n:
            break
    print(f"[{time.time()-t0:.1f}s] {len(sents)} C4 stimuli")

    traj = extract_trajectory(
        model=sys_obj.model, tokenizer=sys_obj.tokenizer,
        texts=sents, layer_indices=[mid], pooling="seq_mean",
        device="cuda", system_key=system_key, class_id=0,
        quantization="fp16",
        stimulus_version=f"c4_clean.v1.seed{seed}.n{n}",
        seed=seed, batch_size=16, max_length=256,
    )
    X = traj.layers[0].X.astype(np.float32)
    print(f"[{time.time()-t0:.1f}s] cloud {X.shape}")

    # d_int via TwoNN
    m_twonn = twonn_id(X)
    d_int = m_twonn.value
    print(f"[{time.time()-t0:.1f}s] TwoNN d_int = {d_int:.3f}")

    # d_2 via Grassberger-Procaccia
    gp = grassberger_procaccia_d2(X)
    d_2 = gp["d_2"]
    print(f"[{time.time()-t0:.1f}s] GP d_2 = {d_2:.3f} (R^2 fit = {gp['R2_fit']:.3f})")

    # Framework A prediction
    p_pred = d_2 / d_int - 1
    p_emp = load_empirical_p(system_key)
    print(f"  framework A: p_pred = {p_pred:.3f}")
    print(f"  empirical p = {p_emp}")
    rel_err = None
    if p_emp is not None and abs(p_emp) > 1e-9:
        rel_err = abs(p_pred - p_emp) / abs(p_emp)
        print(f"  |predicted - empirical| / empirical = {rel_err:.3f}")

    sys_obj.unload()
    torch.cuda.empty_cache()

    return {
        "system_key": system_key, "hf_id": hf_id,
        "depth_idx": mid, "n_stimuli": int(X.shape[0]),
        "hidden_dim": int(X.shape[1]),
        "d_int_twonn": float(d_int),
        "d_2_gp": float(d_2),
        "gp_fit_R2": float(gp["R2_fit"]),
        "p_predicted_framework_A": float(p_pred),
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
            r = run_one(hf_id, system_key, n=1000, seed=42)
            results.append(r)
        except Exception as e:
            print(f"FAILED on {system_key}: {type(e).__name__}: {e}")
            results.append({"system_key": system_key,
                            "error": f"{type(e).__name__}: {e}"})

    print("\n=== SUMMARY: FRAMEWORK A ===")
    print(f"{'system':>35s} {'d_int':>8s} {'d_2':>8s} {'p_pred':>8s} {'p_emp':>8s} {'rel_err':>8s}")
    passes = 0
    tested = 0
    for r in results:
        if "error" in r:
            print(f"  {r['system_key']:>35s}  ERROR: {r['error']}")
            continue
        print(f"  {r['system_key']:>35s} {r['d_int_twonn']:8.3f} "
              f"{r['d_2_gp']:8.3f} {r['p_predicted_framework_A']:8.3f} "
              f"{r['p_empirical']:8.3f} {r['rel_error']:8.3f}")
        tested += 1
        if r["rel_error"] is not None and r["rel_error"] < 0.20:
            passes += 1

    verdict = "FRAMEWORK_A_SUPPORTED" if passes >= 2 else "FRAMEWORK_A_FALSIFIED"
    print(f"\nPasses (|rel_err| < 0.20): {passes} of {tested}. -> {verdict}")

    out = {"purpose": "Pilot A: fractal d_2/d_int for v2 derivation",
           "per_system": results,
           "passes_20pct_criterion": passes,
           "n_systems_tested": tested,
           "verdict": verdict}
    out_path = _ROOT / "results/gate2/fractal_dim_pilot.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
