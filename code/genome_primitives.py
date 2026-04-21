"""Batch-1 measurement primitives for the Neural Genome atlas.

Implements the three Batch-1 primitives (per `research/atlas_tl_session.md`
section 3c + prereg `genome_id_portability_2026-04-21.md`):

- Intrinsic dimension via TwoNN (Facco et al. 2017) and MLE (Levina & Bickel 2004).
- Participation ratio (centered vs uncentered for G1.4).
- kNN-5 clustering coefficient (k=5 vs k=10 for G1.4).

All primitives return a `Measurement` carrying value + analytical SE + metadata.
Sufficient statistics are also returned so they can be persisted per section 2.5.6f.

Windows + CUDA: sklearn n_jobs=1 when GPU context active.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors


# -------------------- Contracts --------------------

@dataclasses.dataclass
class Measurement:
    """One primitive's value at one (system, layer, pooling, quant) tuple."""

    primitive_id: str
    value: float
    se: float                   # analytical standard error
    estimator: str              # e.g. "twonn", "mle", "pr_centered", ...
    n_points: int
    metadata: dict[str, Any]    # sufficient statistics for re-derivation


# -------------------- TwoNN intrinsic dimension --------------------

def _twonn_log_mu(X: np.ndarray) -> np.ndarray:
    """Compute the Pareto log-ratios log(r2/r1) for TwoNN estimator.

    Returns the sufficient statistic `log_mu` of shape (n,) that fully
    determines the TwoNN MLE via `d_hat = n / sum(log_mu)`.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be (n, d); got shape {X.shape}")
    n, _ = X.shape
    if n < 3:
        raise ValueError(f"TwoNN requires n >= 3; got n={n}")

    nn = NearestNeighbors(n_neighbors=3, algorithm="auto", n_jobs=1).fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)
    # dists[:, 0] is self (zero); take 1st and 2nd non-self neighbors.
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    # Guard against duplicates (r1 == 0); drop those points.
    keep = r1 > 1e-12
    r1 = r1[keep]
    r2 = r2[keep]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_mu = np.log(r2) - np.log(r1)
    log_mu = log_mu[np.isfinite(log_mu) & (log_mu > 0)]
    return log_mu


def twonn_id(X: np.ndarray) -> Measurement:
    """TwoNN intrinsic dimension (Facco et al. 2017).

    d_hat = n / sum(log_mu) where mu = r2/r1.
    Analytical SE: d / sqrt(n) (Pareto MLE asymptotics).
    """
    log_mu = _twonn_log_mu(X)
    n = int(log_mu.size)
    if n < 3:
        return Measurement("intrinsic_dim", float("nan"), float("nan"),
                           estimator="twonn", n_points=n,
                           metadata={"reason": "too-few-points"})
    d_hat = float(n / log_mu.sum())
    se = float(d_hat / np.sqrt(n))
    return Measurement(
        primitive_id="intrinsic_dim",
        value=d_hat,
        se=se,
        estimator="twonn",
        n_points=n,
        metadata={
            "log_mu_sum": float(log_mu.sum()),
            "log_mu_mean": float(log_mu.mean()),
            "n_used": n,
        },
    )


def mle_id(X: np.ndarray, k: int = 10) -> Measurement:
    """Levina-Bickel MLE intrinsic-dimension estimator (G1.4 estimator variant).

    d_hat(k) = [mean_i (1/(k-1)) sum_{j=1..k-1} log(r_k(i) / r_j(i))]^{-1}
    """
    if X.ndim != 2:
        raise ValueError(f"X must be (n, d); got shape {X.shape}")
    n, _ = X.shape
    if n < k + 1:
        return Measurement("intrinsic_dim", float("nan"), float("nan"),
                           estimator="mle", n_points=n,
                           metadata={"reason": "too-few-points", "k": k})

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=1).fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)
    # Drop self-column (column 0).
    dists = dists[:, 1:]  # shape (n, k)
    rk = dists[:, -1:]    # shape (n, 1)
    # Use r1..r_{k-1} in the denominator; skip rk itself (log(1) = 0 contributes zero).
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratios = np.log(rk) - np.log(dists[:, :-1])  # shape (n, k-1)
    per_point = log_ratios.mean(axis=1)  # shape (n,)
    per_point = per_point[np.isfinite(per_point) & (per_point > 0)]
    if per_point.size < 3:
        return Measurement("intrinsic_dim", float("nan"), float("nan"),
                           estimator="mle", n_points=n,
                           metadata={"reason": "all-degenerate", "k": k})
    d_hat = float(1.0 / per_point.mean())
    se = float(d_hat / np.sqrt(per_point.size))
    return Measurement(
        primitive_id="intrinsic_dim",
        value=d_hat,
        se=se,
        estimator=f"mle_k{k}",
        n_points=int(per_point.size),
        metadata={"per_point_mean": float(per_point.mean()), "k": k},
    )


# -------------------- Participation ratio --------------------

def participation_ratio(X: np.ndarray, centered: bool = True) -> Measurement:
    """Participation ratio PR = (sum(lambda))^2 / sum(lambda^2) where lambda
    are covariance eigenvalues. G1.4 estimator pair: centered vs uncentered.

    Analytical SE via delta method on the sum-of-eigenvalues statistic.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be (n, d); got shape {X.shape}")
    n, d = X.shape
    if n < 3:
        return Measurement("participation_ratio", float("nan"), float("nan"),
                           estimator=f"pr_{'centered' if centered else 'uncentered'}",
                           n_points=n, metadata={"reason": "too-few-points"})

    if centered:
        Xc = X - X.mean(axis=0, keepdims=True)
    else:
        Xc = X
    # Covariance eigenvalues via SVD of Xc (faster than eigendecomp for n > d
    # or d > n).
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = (s ** 2) / max(n - 1, 1)
    s1 = float(lam.sum())
    s2 = float((lam ** 2).sum())
    if s2 <= 0:
        return Measurement("participation_ratio", float("nan"), float("nan"),
                           estimator=f"pr_{'centered' if centered else 'uncentered'}",
                           n_points=n, metadata={"reason": "zero-variance"})
    pr = (s1 ** 2) / s2

    # Analytical SE: delta method on PR wrt eigenvalues.
    # Var(PR) ~ sum_i (dPR/dlam_i)^2 * Var(lam_i); Var(lam_i) = 2 lam_i^2 / (n-1).
    # dPR/dlam_i = 2 s1 / s2 - 2 s1^2 lam_i / s2^2.
    dpr = 2.0 * s1 / s2 - 2.0 * (s1 ** 2) * lam / (s2 ** 2)
    var_pr = float(np.sum((dpr ** 2) * 2.0 * (lam ** 2) / max(n - 1, 1)))
    se = float(np.sqrt(max(var_pr, 0.0)))

    top_k = min(100, lam.size)
    return Measurement(
        primitive_id="participation_ratio",
        value=float(pr),
        se=se,
        estimator=f"pr_{'centered' if centered else 'uncentered'}",
        n_points=n,
        metadata={
            "sum_lam": s1,
            "sum_lam2": s2,
            "top_eigenvalues": lam[:top_k].tolist(),
            "centered": centered,
        },
    )


# -------------------- kNN clustering coefficient --------------------

def knn_clustering_coefficient(X: np.ndarray, k: int = 5) -> Measurement:
    """Mean per-point k-NN graph clustering coefficient (P1.3).

    For each point i: C(i) = (# edges among kNN_k(i)) / C(k, 2).
    Returns mean_i C(i) with O(1/n) analytical SE via the sample-mean CLT.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be (n, d); got shape {X.shape}")
    n, _ = X.shape
    if n < k + 2:
        return Measurement("knn_clustering", float("nan"), float("nan"),
                           estimator=f"knn_k{k}", n_points=n,
                           metadata={"reason": "too-few-points", "k": k})

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=1).fit(X)
    _, idxs = nn.kneighbors(X, return_distance=True)
    # Drop self-column.
    neigh = idxs[:, 1:]  # shape (n, k)

    # Build a boolean adjacency for "is j in kNN(i)" quickly.
    adj = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), k)
    cols = neigh.reshape(-1)
    adj[rows, cols] = True

    # For each i, count edges j<->m among its k neighbors where j in neigh[i], m in neigh[i].
    per_point = np.zeros(n, dtype=np.float64)
    denom = (k * (k - 1)) / 2.0 if k >= 2 else 1.0
    for i in range(n):
        ns = neigh[i]
        # Edges among ns: (m, j) where adj[m, j] or adj[j, m].
        sub = adj[np.ix_(ns, ns)] | adj[np.ix_(ns, ns)].T
        # Count undirected edges, exclude diagonal.
        np.fill_diagonal(sub, False)
        per_point[i] = sub.sum() / 2.0 / denom

    c_mean = float(per_point.mean())
    # SE = std / sqrt(n). Use ddof=1 for unbiased variance.
    c_std = float(per_point.std(ddof=1)) if n >= 2 else 0.0
    se = float(c_std / np.sqrt(n))

    return Measurement(
        primitive_id="knn_clustering",
        value=c_mean,
        se=se,
        estimator=f"knn_k{k}",
        n_points=n,
        metadata={
            "c_std": c_std,
            "per_point_summary": {
                "mean": c_mean,
                "std": c_std,
                "min": float(per_point.min()),
                "max": float(per_point.max()),
            },
            "k": k,
        },
    )


# -------------------- CLI smoke --------------------

if __name__ == "__main__":
    # Synthetic smoke: sample 500 points from a 5-d manifold embedded in R^20.
    rng = np.random.default_rng(42)
    d_true = 5
    d_ambient = 20
    n = 500
    Z = rng.standard_normal((n, d_true))
    A = rng.standard_normal((d_true, d_ambient))
    X = Z @ A  # linear embedding, so intrinsic dim should be ~5.

    id_twonn = twonn_id(X)
    id_mle = mle_id(X, k=10)
    pr_c = participation_ratio(X, centered=True)
    pr_u = participation_ratio(X, centered=False)
    cluster_k5 = knn_clustering_coefficient(X, k=5)
    cluster_k10 = knn_clustering_coefficient(X, k=10)

    for m in (id_twonn, id_mle, pr_c, pr_u, cluster_k5, cluster_k10):
        print(f"  {m.primitive_id:22s} {m.estimator:14s} "
              f"value={m.value:.4f} se={m.se:.4f} n={m.n_points}")
    # Both ID estimators should be ~5 (the true intrinsic dim).
    print(f"OK: synthetic 5-d manifold in R^20 -> "
          f"TwoNN={id_twonn.value:.2f}, MLE={id_mle.value:.2f} (expect ~5)")
