"""Candidate-8 P3 closed-form derivation: Marchenko-Pastur + low-rank
signal spike model.

Prior attempts:
 - P2 pure power-law (genome_065): FAILED, predicts ratio 0.3 vs
   observed 2.2 (7x too small).
 - P2 plateau + power-law tail: k_bulk=48 universal across 5 text
   systems (CV 4.2pct) but predicted ratio 1.5 vs observed 2.2 (30pct off).

New model: the trained activation covariance is a SUM of a deterministic
low-rank signal and Marchenko-Pastur noise bulk. Specifically:

    lambda_i = { S + noise_spike  for i = 1..k
               { noise_i          for i > k

where noise_i follows the MP distribution with aspect ratio
gamma = n/h (sample count / feature count) and total variance sigma^2.

This is the standard spiked-covariance model from random matrix theory.
For a PCA on a trained activation cloud:
 - k signal eigenvalues sit above the MP bulk edge
 - (h - k) noise eigenvalues form an MP bulk
 - The signal strength S controls how far above the edge they sit

Closed forms (Baik-Ben Arous-Peche):
 - MP bulk edges: lambda_+- = sigma^2 (1 +- sqrt(gamma))^2
 - Signal eigenvalue lies at: S + sigma^2 gamma / S for S > sigma^2 sqrt(gamma)

Numerical procedure:
 - Parameters (S, sigma, k, h, n): sample S+MP spectrum
 - eff_rank, d_rd_waterfill, ratio from the spectrum
 - For each empirical (alpha, eff_rank, d_rd, c) tuple, find (S, sigma,
   k) minimizing bridge rel_err. If a UNIVERSAL (S, sigma/c_empirical,
   k/h) emerges across 7/8 systems, candidate-8 derivation closes.

Pure CPU, ~2min.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_candidate8_p2_waterfill import (  # noqa: E402
    eff_rank, d_rd_from_rate_distortion,
)

_ROOT = _THIS_DIR.parent


def mp_spectrum(gamma, sigma, h):
    """Sample h eigenvalues from the Marchenko-Pastur density.

    MP density (gamma <= 1):
        p(x) = sqrt((lambda_+ - x)(x - lambda_-)) / (2 pi sigma^2 gamma x)
        for x in [lambda_-, lambda_+], else 0
    We generate h eigenvalues deterministically via inverse CDF sampling.
    """
    lam_plus = sigma ** 2 * (1 + gamma ** 0.5) ** 2
    lam_minus = sigma ** 2 * (1 - gamma ** 0.5) ** 2 if gamma <= 1 else 0.0
    # Sample at uniform quantiles to get a deterministic spectrum
    u = (np.arange(1, h + 1, dtype=np.float64) - 0.5) / h
    # numerical inverse-CDF of MP: bisection per quantile
    xs = np.linspace(lam_minus + 1e-10, lam_plus - 1e-10, 5000)
    pdf = np.sqrt(np.maximum((lam_plus - xs) * (xs - lam_minus), 0.0)) / (
        2 * np.pi * sigma ** 2 * gamma * xs + 1e-12)
    cdf = np.cumsum(pdf); cdf = cdf / cdf[-1]
    return np.interp(u, cdf, xs)


def spiked_covariance_spectrum(S, sigma, k, h, gamma):
    """Build a spiked-covariance spectrum: k signal eigenvalues at S +
    sigma^2 * gamma / S (Baik-Ben Arous-Peche position), plus (h-k) MP
    bulk eigenvalues."""
    if k >= h or k < 0:
        raise ValueError(f"k must be in [0, h); got {k}")
    # Bulk
    bulk = mp_spectrum(gamma, sigma, h - k)
    # Signal eigenvalues
    if S > sigma ** 2 * gamma ** 0.5:
        signal_val = S + sigma ** 2 * gamma / S
    else:
        signal_val = sigma ** 2 * (1 + gamma ** 0.5) ** 2
    signal = np.full(k, signal_val)
    # Combine, sort descending
    lam = np.concatenate([signal, bulk])
    lam = np.sort(lam)[::-1]
    return lam


def predict_ratio_mp_lowrank(S, sigma, k, h, gamma):
    lam = spiked_covariance_spectrum(S, sigma, k, h, gamma)
    er = eff_rank(lam)
    drd = d_rd_from_rate_distortion(lam)
    if drd is None or not np.isfinite(drd) or drd <= 0:
        return er, drd, float("nan")
    return er, drd, er / drd


def fit_system(er_obs, ratio_obs, h=1024, gamma=1.0):
    """Find (S, sigma, k) that best matches (er_obs, ratio_obs)."""
    # Grid search then refine
    best = None
    for k in [16, 24, 32, 48, 64, 96, 128]:
        for S_ratio in [2, 4, 8, 16, 32]:
            for sigma in [0.3, 0.5, 0.7, 1.0]:
                S = S_ratio * sigma ** 2
                try:
                    er, drd, ratio = predict_ratio_mp_lowrank(S, sigma, k, h, gamma)
                    if not np.isfinite(ratio):
                        continue
                    err = (abs(er - er_obs) / max(er_obs, 1e-6)) ** 2 + \
                          (abs(ratio - ratio_obs) / max(ratio_obs, 1e-6)) ** 2
                    if best is None or err < best["err"]:
                        best = {"S": S, "sigma": sigma, "k": k,
                                "eff_rank": er, "d_rd": drd, "ratio": ratio,
                                "err": err}
                except Exception:
                    continue
    return best


def main():
    scorecard = [
        {"system": "qwen3-0.6b",                  "eff_rank": 25.27, "d_rd_kmeans": 12.27, "c": 1.889, "h": 1024, "n": 1000},
        {"system": "deepseek-r1-distill-qwen-1.5b","eff_rank": 33.93, "d_rd_kmeans": 14.06, "c": 2.410, "h": 1536, "n": 1000},
        {"system": "bert-base-uncased",           "eff_rank": 32.94, "d_rd_kmeans": 14.37, "c": 2.653, "h":  768, "n": 1000},
        {"system": "roberta-base",                "eff_rank": 28.06, "d_rd_kmeans": 13.00, "c": 2.250, "h":  768, "n": 1000},
        {"system": "minilm-l6-contrastive",       "eff_rank": 28.23, "d_rd_kmeans": 12.84, "c": 2.027, "h":  384, "n": 1000},
        {"system": "dinov2-small",                "eff_rank": 26.45, "d_rd_kmeans":  9.82, "c": 2.242, "h":  384, "n":  500},
        {"system": "clip-text-b32",               "eff_rank": 51.10, "d_rd_kmeans": 16.04, "c": 2.975, "h":  512, "n": 1000},
        {"system": "clip-vision-b32",             "eff_rank": 22.30, "d_rd_kmeans": 10.38, "c": 2.447, "h":  768, "n":  500},
    ]

    print(f"{'system':<30s} {'er_obs':>7s} {'c_obs':>7s} {'k_fit':>6s} "
          f"{'S':>7s} {'sigma':>6s} {'er_mod':>7s} {'ratio_mod':>9s} {'err_pct':>8s}")
    print("-" * 100)
    rows = []
    for r in scorecard:
        gamma = r["n"] / r["h"]
        fit = fit_system(r["eff_rank"], r["c"], h=r["h"], gamma=gamma)
        if fit is None:
            print(f"{r['system']:<30s} FIT_FAILED")
            continue
        ratio_err = abs(fit["ratio"] - r["c"]) / max(r["c"], 1e-6)
        rows.append({**r, **fit, "ratio_err": float(ratio_err)})
        print(f"{r['system']:<30s} {r['eff_rank']:7.2f} {r['c']:7.3f} "
              f"{fit['k']:6d} {fit['S']:7.3f} {fit['sigma']:6.2f} "
              f"{fit['eff_rank']:7.2f} {fit['ratio']:9.3f} {ratio_err*100:7.2f}pct")

    if not rows:
        print("\n  verdict: ALL_FITS_FAILED"); return

    # Universality analysis: is there a consistent (S, sigma, k/h) ratio?
    ks = [r["k"] for r in rows]
    khs = [r["k"] / r["h"] for r in rows]
    sigmas = [r["sigma"] for r in rows]
    S_sigma2 = [r["S"] / r["sigma"] ** 2 for r in rows]
    print(f"\n  k_fit values:           {ks}")
    print(f"  k/h (rel bulk width):   mean={np.mean(khs):.3f}  std={np.std(khs):.3f}  CV={np.std(khs)/np.mean(khs)*100:.1f}pct")
    print(f"  sigma (noise scale):    mean={np.mean(sigmas):.3f}  std={np.std(sigmas):.3f}")
    print(f"  S/sigma^2 (SNR):        mean={np.mean(S_sigma2):.2f}  std={np.std(S_sigma2):.2f}")
    print(f"  mean ratio_err:         {np.mean([r['ratio_err'] for r in rows])*100:.2f}pct")

    # A universal verdict if k/h is consistent + errs are < 15pct
    mean_err = np.mean([r["ratio_err"] for r in rows]) * 100
    kh_cv = np.std(khs) / np.mean(khs) * 100 if np.mean(khs) > 0 else 100
    if mean_err < 15 and kh_cv < 40:
        verdict = (f"P3_MP_LOWRANK_CLOSES - spiked-covariance model fits "
                   f"all systems with mean rel_err {mean_err:.1f}pct and "
                   f"k/h consistent at CV {kh_cv:.0f}pct. Candidate-8 "
                   "reduces to a 3-parameter (S, sigma, k) spiked-cov fit.")
    elif mean_err < 25:
        verdict = (f"P3_PARTIAL - MP+spike fit gives mean rel_err "
                   f"{mean_err:.1f}pct, k/h CV {kh_cv:.0f}pct. Closer than "
                   "plateau-only model but not yet tight.")
    else:
        verdict = (f"P3_MISS - MP+spike model fits badly. Model class "
                   "insufficient; need Wishart-with-structured-signal "
                   "or operator-level model.")
    print(f"\n  verdict: {verdict}")

    out = {"purpose": "Candidate-8 P3 derivation: Marchenko-Pastur + low-rank spike",
           "per_system": rows,
           "k_fit_cv_pct": kh_cv,
           "mean_ratio_err_pct": mean_err,
           "verdict": verdict}
    out_path = _ROOT / "results/gate2/candidate8_p3_mp_lowrank.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
