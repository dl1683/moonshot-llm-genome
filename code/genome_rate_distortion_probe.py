"""Pilot for v2 derivation framework D (rate-distortion).

Per research/derivations/power_law_v2_candidates.md framework D:
the manifesto framing of intelligence as multi-resolution coding
(Equitz-Cover, Rimoldi). The point cloud at layer ell is an encoder
output; C(X, k) at varying k samples the local-coherence-vs-scale
trade-off of that encoder.

Practical R(D) estimator: k-means at K clusters gives average
within-cluster squared distance = D; code length R = log2(K).
Slope of log D vs log K at intermediate K regimes the rate-distortion
dimension d_rd:
    R(D) ~ -d_rd * log D + const  =>  d/d(log D) [log K] = -d_rd
(equivalently, D scales as K^(-2/d_rd) asymptotically).

Framework D prediction: does d_rd connect to p?

Test on Qwen3-0.6B / RWKV-4-169M / DeepSeek-R1-Distill-Qwen-1.5B
trained vs untrained, mid-depth, n=1000.

Pass criterion: if |predicted_p - empirical_p| / empirical_p < 0.20
on >= 2/3 systems, framework D is a viable Gate-2 G2.3 re-derivation
candidate.
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
from genome_primitives import knn_clustering_coefficient  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]


def rate_distortion_dim(X: np.ndarray, n_codebook_sizes: int = 8) -> dict:
    """Estimate rate-distortion dimension via k-means at multiple K.

    Slope of log(D) vs log(K) in the scaling regime gives d_rd from
    D ~ K^(-2/d_rd). Use K values spaced log-uniformly.
    """
    from sklearn.cluster import KMeans
    n = X.shape[0]
    K_vals = np.unique(np.round(np.exp(np.linspace(
        np.log(4), np.log(min(128, n // 4)), n_codebook_sizes))).astype(int))
    Ds = []
    for K in K_vals:
        km = KMeans(n_clusters=int(K), n_init=3, random_state=0)
        km.fit(X)
        # Average within-cluster squared distance
        D = float(km.inertia_ / n)
        Ds.append(D)
    Ds = np.array(Ds)
    # D = K^(-2/d_rd) => log D = -(2/d_rd) log K + const
    log_K = np.log(K_vals.astype(float))
    log_D = np.log(Ds)
    slope, intercept = np.polyfit(log_K, log_D, 1)
    pred = slope * log_K + intercept
    ss_res = float(np.sum((log_D - pred) ** 2))
    ss_tot = float(np.sum((log_D - log_D.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # slope = -2/d_rd  =>  d_rd = -2/slope
    d_rd = -2.0 / slope if abs(slope) > 1e-6 else float("inf")
    return {
        "d_rd": float(d_rd),
        "slope_logD_vs_logK": float(slope),
        "R2": float(r2),
        "K_values": K_vals.tolist(),
        "D_values": Ds.tolist(),
    }


def fit_power_law(ks, Cs):
    lks = np.log(np.asarray(ks, dtype=float))
    lcs = np.log(np.asarray(Cs, dtype=float))
    p, log_c0 = np.polyfit(lks, lcs, 1)
    pred = p * lks + log_c0
    ss_res = float(np.sum((lcs - pred) ** 2))
    ss_tot = float(np.sum((lcs - lcs.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(p), float(np.exp(log_c0)), r2


def run_one(hf_id: str, system_key: str, untrained: bool = False,
            n: int = 1000, seed: int = 42):
    tag = "UNTRAINED" if untrained else "TRAINED"
    print(f"\n=== {system_key} [{tag}] ===")
    t0 = time.time()
    sys_obj = load_system(hf_id, quant="fp16", untrained=untrained, device="cuda")
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

    # Empirical p from kNN power-law
    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p_emp, c0_emp, r2_ck = fit_power_law(K_GRID, Cs)

    # Rate-distortion dimension
    rd = rate_distortion_dim(X)
    d_rd = rd["d_rd"]

    # Candidate framework D predictions to test
    p_pred_inverse = 2.0 / d_rd if d_rd > 0 else float("nan")
    p_pred_inverse_half = 1.0 / d_rd if d_rd > 0 else float("nan")
    p_pred_rd_based = d_rd / (2 * X.shape[1]) if X.shape[1] > 0 else float("nan")

    rel_err_inverse = abs(p_pred_inverse - p_emp) / abs(p_emp) if abs(p_emp) > 1e-6 else float("inf")
    rel_err_inverse_half = abs(p_pred_inverse_half - p_emp) / abs(p_emp) if abs(p_emp) > 1e-6 else float("inf")

    print(f"  empirical p = {p_emp:.3f}  c_0 = {c0_emp:.3f}  R^2 = {r2_ck:.4f}")
    print(f"  rate-distortion: d_rd = {d_rd:.3f}  fit R^2 = {rd['R2']:.3f}")
    print(f"    candidate p_pred = 2/d_rd = {p_pred_inverse:.3f}  rel_err = {rel_err_inverse:.3f}")
    print(f"    candidate p_pred = 1/d_rd = {p_pred_inverse_half:.3f}  rel_err = {rel_err_inverse_half:.3f}")

    sys_obj.unload()
    torch.cuda.empty_cache()
    return {
        "system_key": system_key, "hf_id": hf_id, "untrained": untrained,
        "depth_idx": mid, "n_stimuli": int(X.shape[0]),
        "hidden_dim": int(X.shape[1]),
        "empirical_p": p_emp, "empirical_c0": c0_emp, "Ck_R2": r2_ck,
        "d_rd": d_rd, "rd_fit_R2": rd["R2"],
        "candidate_p_pred_inverse": p_pred_inverse,
        "candidate_p_pred_inverse_half": p_pred_inverse_half,
        "rel_err_inverse": rel_err_inverse,
        "rel_err_inverse_half": rel_err_inverse_half,
        "elapsed_s": time.time() - t0,
    }


def main():
    targets = [
        ("Qwen/Qwen3-0.6B", "qwen3-0.6b"),
        ("RWKV/rwkv-4-169m-pile", "rwkv-4-169m"),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
         "deepseek-r1-distill-qwen-1.5b"),
    ]
    results = []
    for hf_id, sk in targets:
        results.append(run_one(hf_id, sk, untrained=False))

    print("\n=== SUMMARY: FRAMEWORK D (rate-distortion) trained ===")
    print(f"{'system':>32s} {'d_rd':>8s} {'p_emp':>8s} {'p=2/d_rd':>10s} {'p=1/d_rd':>10s} {'rel_err':>8s}")
    passes_inverse = 0
    passes_inverse_half = 0
    for r in results:
        print(f"  {r['system_key']:>32s} {r['d_rd']:8.3f} {r['empirical_p']:8.3f} "
              f"{r['candidate_p_pred_inverse']:10.3f} {r['candidate_p_pred_inverse_half']:10.3f} "
              f"{min(r['rel_err_inverse'], r['rel_err_inverse_half']):8.3f}")
        if r["rel_err_inverse"] < 0.20:
            passes_inverse += 1
        if r["rel_err_inverse_half"] < 0.20:
            passes_inverse_half += 1

    if passes_inverse >= 2:
        verdict = f"FRAMEWORK_D_SUPPORTED (p=2/d_rd, {passes_inverse}/3 systems pass 20% criterion)"
    elif passes_inverse_half >= 2:
        verdict = f"FRAMEWORK_D_SUPPORTED (p=1/d_rd, {passes_inverse_half}/3 systems pass 20% criterion)"
    else:
        verdict = "FRAMEWORK_D_FALSIFIED"
    print(f"\n  {verdict}")

    out = {"purpose": "Framework D pilot: rate-distortion for v2 derivation",
           "per_system": results,
           "passes_p_eq_2_over_drd": passes_inverse,
           "passes_p_eq_1_over_drd": passes_inverse_half,
           "verdict": verdict}
    out_path = _ROOT / "results/gate2/rate_distortion_pilot.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
