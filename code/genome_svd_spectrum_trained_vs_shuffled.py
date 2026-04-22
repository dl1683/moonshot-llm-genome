"""Derivation step: characterize trained-vs-shuffled covariance spectra.

genome_056 established c lives in training-specific joint inter-dim
structure (trained c=1.89, shuffled c=12.2). Mechanism claim needs the
specific structural signature that training imposes and shuffle destroys.

This probe compares the singular value spectrum of:
  - trained activation covariance
  - marginal-shuffled version (same per-dim variance, destroyed joint)
  - random Gaussian of same (n, h)

Hypothesis: trained covariance has FAST SINGULAR VALUE DECAY (power-law
or exponential) compressing effective rank to ~d_rd. Shuffled covariance
is closer to diagonal (every singular value ~= per-dim variance). Gap in
decay rate IS the training-specific joint structure.

If trained singular values follow σ_i ~ i^{-α} with α giving effective
rank ~d_rd, random matrix theory predicts c = p·d_rd relations that a
future derivation can exploit.
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


def spectrum(X):
    Xc = X - X.mean(axis=0)
    # Singular values of Xc / sqrt(n-1) = sqrt eigenvalues of cov
    s = np.linalg.svd(Xc, compute_uv=False) / np.sqrt(X.shape[0] - 1)
    return s.astype(np.float64)


def fit_power_tail(s, tail_frac_lo=0.05, tail_frac_hi=0.5):
    """Fit power-law decay σ_i ~ i^{-α} in the tail."""
    ranks = np.arange(1, len(s) + 1)
    lo = max(1, int(len(s) * tail_frac_lo))
    hi = int(len(s) * tail_frac_hi)
    logr = np.log(ranks[lo:hi])
    logs = np.log(s[lo:hi] + 1e-12)
    slope, intercept = np.polyfit(logr, logs, 1)
    return {"alpha": float(-slope), "intercept": float(intercept),
             "fit_range": (lo, hi)}


def effective_rank(s):
    """Participation ratio of eigenvalues = (sum σ²)² / sum σ⁴."""
    s2 = s ** 2
    return float(s2.sum() ** 2 / (s2 * s2).sum() / (s2.sum() ** 2 / s2.sum() ** 2) ** 0 + 1e-12) if s2.sum() > 0 else 0.0


def eff_rank(s):
    s2 = s ** 2
    total = s2.sum()
    if total <= 0:
        return 0.0
    return float(total ** 2 / (s2 ** 2).sum())


def main():
    t0 = time.time()
    print(f"[{time.time()-t0:.1f}s] loading Qwen3 trained mid-depth...")
    sys_obj = load_system("Qwen/Qwen3-0.6B", quant="fp16", untrained=False, device="cuda")
    mid = sys_obj.n_hidden_layers() // 2
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= 1000:
            break
    traj = extract_trajectory(
        model=sys_obj.model, tokenizer=sys_obj.tokenizer,
        texts=sents, layer_indices=[mid], pooling="seq_mean",
        device="cuda", system_key="qwen3-0.6b", class_id=1,
        quantization="fp16", stimulus_version="c4_clean.v1.seed42.n1000",
        seed=42, batch_size=16, max_length=256,
    )
    X_trained = traj.layers[0].X.astype(np.float32)
    sys_obj.unload(); torch.cuda.empty_cache()
    print(f"[{time.time()-t0:.1f}s] cloud {X_trained.shape}")

    s_trained = spectrum(X_trained)
    rng = np.random.default_rng(42)
    X_shuf = X_trained.copy()
    for j in range(X_shuf.shape[1]):
        rng.shuffle(X_shuf[:, j])
    s_shuffled = spectrum(X_shuf)
    mu, sd = X_trained.mean(axis=0), X_trained.std(axis=0)
    X_gauss = rng.normal(loc=mu, scale=sd, size=X_trained.shape).astype(np.float32)
    s_gauss = spectrum(X_gauss)

    fit_t = fit_power_tail(s_trained)
    fit_s = fit_power_tail(s_shuffled)
    fit_g = fit_power_tail(s_gauss)

    er_t = eff_rank(s_trained)
    er_s = eff_rank(s_shuffled)
    er_g = eff_rank(s_gauss)

    print(f"\n  Trained:      alpha={fit_t['alpha']:.3f}  eff_rank={er_t:.1f}  top-10 sigma = {[round(float(x),2) for x in s_trained[:10]]}")
    print(f"  Shuffled:     alpha={fit_s['alpha']:.3f}  eff_rank={er_s:.1f}")
    print(f"  Gaussian:     alpha={fit_g['alpha']:.3f}  eff_rank={er_g:.1f}")

    print(f"\n  Compare eff_rank to d_rd (k-means scaling): trained d_rd=12.3, shuffled=32.4")
    print(f"  Compare alpha: trained has steeper decay -> concentrated joint structure")

    out = {
        "trained": {"alpha": fit_t["alpha"], "eff_rank": er_t,
                    "top10_sigma": s_trained[:10].tolist()},
        "shuffled": {"alpha": fit_s["alpha"], "eff_rank": er_s,
                     "top10_sigma": s_shuffled[:10].tolist()},
        "gaussian": {"alpha": fit_g["alpha"], "eff_rank": er_g,
                     "top10_sigma": s_gauss[:10].tolist()},
    }
    out_path = _ROOT / "results/gate2/svd_spectrum_trained_vs_shuffled.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
