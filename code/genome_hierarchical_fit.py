"""Gate-2 G2.3 hierarchical fit: pooled vs per-system test of the locked
derivation's functional form for kNN-k10 clustering coefficient.

Per `research/prereg/genome_knn_k10_hierarchical_2026-04-21.md` and the
LOCKED derivation `research/derivations/knn_clustering_universality.md`:

    C(X, k) = alpha_d * (1 - beta_d * kappa * k^(2/d_int))_+ + eps

Under Level-1 universality, alpha_d and beta_d should be UNIVERSAL
constants (shared across all architectures at a given ambient dim d).
This script fits:

  H0 (pooled):    alpha_d, beta_d shared; kappa_i, d_int,i per-system.
  H1 (per-system): all 4 params per-system.
  H2 (family):    pool alpha_d, beta_d within an arch family.

Then compares by BIC. Prereg kill criterion: `BIC(H1) - BIC(H0) > 10`
required for Level-1 functional-universality claim.

Smoke-testable on the existing Gate-1 atlas: uses measured `C(X, k)`
at k ∈ {5, 10} (only 2 k-values so far — G2.3 lock needs the extended
k-sweep {5, 10, 20, 30}, but 2 points is enough to verify the fit
pipeline converges before locking).

Usage:
    python code/genome_hierarchical_fit.py --smoke
    python code/genome_hierarchical_fit.py --full
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


def _load_atlas() -> list[dict]:
    """Load Gate-1 atlas rows for all 5 Batch-1 classes."""
    rows = []
    root = Path(__file__).parent.parent / "results" / "cross_arch"

    for seed in (42, 123, 456):
        main = root / f"atlas_rows_n2000_c4_seed{seed}.json"
        if main.exists():
            rows.extend(json.loads(main.read_text())["rows"])
        deepseek = root / f"atlas_rows_n2000_c4_seed{seed}_only_deepseek-r1-distill-qwen-1.5b.json"
        if deepseek.exists():
            rows.extend(json.loads(deepseek.read_text())["rows"])
        falcon_n4k = root / f"atlas_rows_n4000_c4_seed{seed}_only_falcon-h1-0.5b.json"
        if falcon_n4k.exists():
            # Replace n=2000 Falcon rows with n=4000 (more precise)
            rows = [r for r in rows if r["system_key"] != "falcon-h1-0.5b"]
            rows.extend(json.loads(falcon_n4k.read_text())["rows"])
    return rows


def _extract_knn_grid(rows: list[dict]) -> dict:
    """Build (system, depth, k) -> list[(value, se)] from atlas rows.

    k is encoded in estimator name ("knn_k5", "knn_k10", etc.).
    """
    grid: dict[tuple[str, float, int], list[tuple[float, float]]] = {}
    for r in rows:
        if r["primitive_id"] != "knn_clustering":
            continue
        est = r["estimator"]
        if not est.startswith("knn_k"):
            continue
        k = int(est[5:])
        key = (r["system_key"], round(r["k_normalized"], 2), k)
        grid.setdefault(key, []).append((r["value"], r["se"]))
    return grid


def _id_per_system(rows: list[dict]) -> dict[str, float]:
    """Mean TwoNN intrinsic dimension per system, for d_int initial guess."""
    dints = {}
    for r in rows:
        if r["primitive_id"] == "intrinsic_dim" and r["estimator"] == "twonn":
            dints.setdefault(r["system_key"], []).append(r["value"])
    return {s: float(np.mean([v for v in vals if v == v])) for s, vals in dints.items()}


def _neg_log_likelihood_pooled(params, data):
    """H0 pooled: params = [log_alpha, log_beta, {log_kappa_i}, {log_dint_i}].

    data: list of (system_idx, k, C_mean, C_se)
    Uses Gaussian residual assumption with per-point SE as sigma.
    """
    n_sys = max(d[0] for d in data) + 1
    log_alpha, log_beta = params[0], params[1]
    log_kappas = params[2:2 + n_sys]
    log_dints = params[2 + n_sys:2 + 2 * n_sys]
    alpha = math.exp(log_alpha)
    beta = math.exp(log_beta)

    nll = 0.0
    for (sys_idx, k, C_obs, C_se) in data:
        kappa = math.exp(log_kappas[sys_idx])
        dint = math.exp(log_dints[sys_idx])
        inner = 1.0 - beta * kappa * (k ** (2.0 / dint))
        C_pred = alpha * max(inner, 0.0)
        sigma = max(C_se, 1e-4)
        nll += 0.5 * ((C_obs - C_pred) / sigma) ** 2 + math.log(sigma)
    return nll


def _neg_log_likelihood_per_system(params, data):
    """H1 per-system: params = [log_alpha_i, log_beta_i, log_kappa_i, log_dint_i] per system."""
    n_sys = max(d[0] for d in data) + 1
    per = 4
    nll = 0.0
    for (sys_idx, k, C_obs, C_se) in data:
        la, lb, lk, ld = params[sys_idx * per: (sys_idx + 1) * per]
        alpha, beta = math.exp(la), math.exp(lb)
        kappa = math.exp(lk)
        dint = math.exp(ld)
        inner = 1.0 - beta * kappa * (k ** (2.0 / dint))
        C_pred = alpha * max(inner, 0.0)
        sigma = max(C_se, 1e-4)
        nll += 0.5 * ((C_obs - C_pred) / sigma) ** 2 + math.log(sigma)
    return nll


def fit(grid: dict, id_hints: dict[str, float], verbose: bool = True) -> dict:
    """Fit H0 (pooled) and H1 (per-system), return BIC comparison."""
    systems = sorted({s for (s, _, _) in grid.keys()})
    sys_to_idx = {s: i for i, s in enumerate(systems)}
    n_sys = len(systems)

    # Flatten grid → data list, each observation = (system_idx, k, C_mean, C_se)
    data = []
    for (s, depth, k), vals in grid.items():
        sys_idx = sys_to_idx[s]
        for (v, se) in vals:
            if not math.isfinite(v):
                continue
            data.append((sys_idx, k, v, se if se > 0 else 0.001))
    N = len(data)
    if verbose:
        print(f"[fit] N={N} observations, {n_sys} systems: {systems}")

    # Initial guesses.
    log_alpha0 = math.log(0.33)          # observed C ~0.3
    log_beta0 = math.log(0.1)            # guess
    log_kappas0 = [math.log(1.0)] * n_sys
    log_dints0 = [math.log(max(id_hints.get(s, 20.0), 1.0)) for s in systems]

    x0_h0 = np.array([log_alpha0, log_beta0] + log_kappas0 + log_dints0)
    res_h0 = minimize(_neg_log_likelihood_pooled, x0_h0, args=(data,),
                      method="L-BFGS-B", options={"maxiter": 500})
    nll_h0 = float(res_h0.fun)
    k_h0 = 2 + 2 * n_sys  # 2 pooled + 2*n_sys per-system

    x0_h1 = np.array([[log_alpha0, log_beta0, lk, ld] for lk, ld in
                      zip(log_kappas0, log_dints0)]).flatten()
    res_h1 = minimize(_neg_log_likelihood_per_system, x0_h1, args=(data,),
                      method="L-BFGS-B", options={"maxiter": 500})
    nll_h1 = float(res_h1.fun)
    k_h1 = 4 * n_sys

    bic_h0 = 2 * nll_h0 + k_h0 * math.log(N)
    bic_h1 = 2 * nll_h1 + k_h1 * math.log(N)
    delta_bic = bic_h1 - bic_h0  # positive => H0 (pooled) preferred

    verdict = "POOLED_UNIVERSAL" if delta_bic > 10 else \
              "PER_SYSTEM_PREFERRED" if delta_bic < -10 else \
              "INDETERMINATE"

    # Extract H0 param estimates
    alpha_hat = math.exp(res_h0.x[0])
    beta_hat = math.exp(res_h0.x[1])
    kappa_hat = {s: math.exp(res_h0.x[2 + i]) for i, s in enumerate(systems)}
    dint_hat = {s: math.exp(res_h0.x[2 + n_sys + i]) for i, s in enumerate(systems)}

    return {
        "N_observations": N,
        "systems": systems,
        "nll_h0_pooled": nll_h0,
        "nll_h1_per_system": nll_h1,
        "k_h0": k_h0,
        "k_h1": k_h1,
        "bic_h0_pooled": bic_h0,
        "bic_h1_per_system": bic_h1,
        "delta_bic": delta_bic,
        "delta_bic_verdict": verdict,
        "h0_fit": {
            "alpha_d": alpha_hat,
            "beta_d": beta_hat,
            "kappa_per_system": kappa_hat,
            "dint_per_system": dint_hat,
        },
        "h0_converged": bool(res_h0.success),
        "h1_converged": bool(res_h1.success),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="quick test on existing k={5,10} data only")
    ap.add_argument("--full", action="store_true",
                    help="requires extended k-sweep to k={5,10,20,30}")
    args = ap.parse_args()

    rows = _load_atlas()
    print(f"[main] loaded {len(rows)} atlas rows")
    grid = _extract_knn_grid(rows)
    print(f"[main] {len(grid)} (system, depth, k) cells")
    id_hints = _id_per_system(rows)
    print(f"[main] d_int hints: {id_hints}")

    result = fit(grid, id_hints)

    out_dir = Path(__file__).parent.parent / "results" / "gate2"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke or not args.full else "full"
    out = out_dir / f"hierarchical_fit_{suffix}.json"
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, default=float)

    print()
    print(f"=== G2.3 HIERARCHICAL FIT VERDICT ===")
    print(f"N_observations  : {result['N_observations']}")
    print(f"BIC pooled (H0) : {result['bic_h0_pooled']:.2f}")
    print(f"BIC per-sys (H1): {result['bic_h1_per_system']:.2f}")
    print(f"Delta-BIC       : {result['delta_bic']:+.2f}")
    print(f"  (> +10 means POOLED UNIVERSAL strongly preferred)")
    print(f"  (< -10 means PER-SYSTEM strongly preferred)")
    print(f"Verdict         : {result['delta_bic_verdict']}")
    print()
    print(f"H0 pooled estimates:")
    print(f"  alpha_d = {result['h0_fit']['alpha_d']:.4f}")
    print(f"  beta_d  = {result['h0_fit']['beta_d']:.4f}")
    print(f"  kappa per system: {result['h0_fit']['kappa_per_system']}")
    print(f"  d_int per system: {result['h0_fit']['dint_per_system']}")
    print(f"out: {out}")
