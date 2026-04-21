"""Ablation schemes for Gate-2 G2.4 causal-ablation probe.

Per `research/prereg/genome_knn_k10_causal_2026-04-21.md` §4, the causal
test for kNN-k10 requires three ablation schemes applied at variable
strength λ to hidden-state activations `X ∈ R^{n × h}`:

  1. topk    — project out the span of each point's top-10 nearest-neighbor
               tangent approximation. This is the coordinate-defined subspace
               that kNN-k10 measures; ablating it tests causal specificity.
  2. random  — project out a uniformly-random 10-dim subspace (Haar measure
               on the Stiefel manifold of R^h). Negative-control: ablation of
               arbitrary 10-dim subspaces should have smaller downstream loss
               impact than the top-k-neighbor subspace.
  3. pca     — project out the top-10 global PCA components of the batch.
               Negative-control: the top-k-neighbor subspace is LOCAL; if
               global PCA ablation has the same effect, kNN-k10 is not
               capturing local structure specifically.

Each scheme takes `X` and scheme-specific parameters and returns
`(X_ablated, projection_operator_info)`. The ablation formula per §4b:

    X_ablated = X - λ * (P @ X.T).T

where P is the projector onto the ablation subspace. λ ∈ [0, 1].

This module is pure numpy — no torch, no GPU — so it is unit-testable and
usable both for mock tests and for preparing hooks that the real causal-probe
module will install on forward passes.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


# -------------------- Subspace projectors --------------------

def _project_out_subspace(X: np.ndarray, basis: np.ndarray,
                          lam: float) -> np.ndarray:
    """Ablate the subspace spanned by `basis` from X with strength `lam`.

    X:     (n, h) point cloud
    basis: (k, h) row-wise basis vectors (will be orthonormalized via QR)
    lam:   scalar in [0, 1]; 0 = no-op, 1 = full removal of subspace component

    Returns X' = X - lam * P(X), where P projects onto span(basis).
    """
    if basis.shape[0] == 0:
        return X.copy()
    # Row-wise basis -> column-wise for QR.
    Q, _ = np.linalg.qr(basis.T)  # Q: (h, k) orthonormal cols
    XQ = X @ Q                     # (n, k)
    X_proj = XQ @ Q.T              # (n, h)
    return X - lam * X_proj


def ablate_random_10d(X: np.ndarray, lam: float, *,
                      k_dim: int = 10,
                      rng: np.random.Generator | None = None
                      ) -> tuple[np.ndarray, dict]:
    """Ablation scheme (2): random 10-dim subspace of R^h, per-point SAME.

    Haar-random basis drawn via QR decomposition of Gaussian matrix.
    Same basis applied to all points -> tests "arbitrary 10-dim subspace".
    """
    n, h = X.shape
    rng = rng or np.random.default_rng()
    G = rng.standard_normal((h, k_dim)).astype(X.dtype)
    Q, _ = np.linalg.qr(G)  # Q: (h, k_dim)
    XQ = X @ Q
    X_proj = XQ @ Q.T
    X_ablated = X - lam * X_proj
    return X_ablated, {"scheme": "random", "k_dim": k_dim, "lam": lam,
                       "subspace_rank": k_dim}


def ablate_pca_10d(X: np.ndarray, lam: float, *,
                   k_dim: int = 10,
                   centered: bool = True) -> tuple[np.ndarray, dict]:
    """Ablation scheme (3): top-10 PCA components of the batch.

    Uses thin SVD on centered X. Global subspace — same for all points.
    """
    n, h = X.shape
    X_center = X.mean(axis=0, keepdims=True) if centered else np.zeros((1, h))
    Xc = X - X_center
    # Thin SVD: Xc = U S V^T, V is (h, min(n,h)) with column PCs.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V_top = Vt[:k_dim].T  # (h, k_dim) top PC directions
    XQ = Xc @ V_top
    X_proj = XQ @ V_top.T
    X_ablated = X - lam * X_proj
    return X_ablated, {"scheme": "pca", "k_dim": k_dim, "lam": lam,
                       "subspace_rank": min(k_dim, len(S)),
                       "explained_variance_top_k": float(
                           (S[:k_dim] ** 2).sum() / (S ** 2).sum())}


def ablate_topk_neighbors(X: np.ndarray, lam: float, *,
                          k: int = 10) -> tuple[np.ndarray, dict]:
    """Ablation scheme (1): top-k nearest-neighbor tangent subspace per point.

    Per-point: find k nearest neighbors N_k(x_i), form tangent vectors
    v_{i,j} = x_{j} - x_i, and ablate the subspace these span.

    Per-point subspace varies -> this is the LOCAL ablation scheme, and the
    one the derivation predicts should have the largest functional impact.

    Returns X' where each row x_i' = x_i - lam * P_i(x_i), P_i = projector
    onto span{x_{j_m} - x_i : m=1..k}.
    """
    n, h = X.shape
    # Euclidean kNN. n_neighbors = k+1 because 0-th NN is self.
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean",
                          algorithm="auto", n_jobs=1)
    nn.fit(X)
    _dist, idx = nn.kneighbors(X, return_distance=True)

    X_ablated = np.empty_like(X)
    effective_ranks = np.empty(n, dtype=np.int32)
    for i in range(n):
        neighbors = X[idx[i, 1:k+1]]           # (k, h), skip self
        tangent = neighbors - X[i:i+1]          # (k, h)
        # QR on (h, k) to orthonormalize.
        Q, _ = np.linalg.qr(tangent.T)          # (h, rank)
        x_i = X[i]
        x_proj = Q @ (Q.T @ x_i)                # (h,)
        X_ablated[i] = x_i - lam * x_proj
        effective_ranks[i] = Q.shape[1]

    info = {
        "scheme": "topk",
        "k": k,
        "lam": lam,
        "per_point_effective_rank_mean": float(effective_ranks.mean()),
        "per_point_effective_rank_min": int(effective_ranks.min()),
        "per_point_effective_rank_max": int(effective_ranks.max()),
    }
    return X_ablated, info


# -------------------- Sanity self-test --------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n, h = 200, 64
    X = rng.standard_normal((n, h)).astype(np.float32)

    for scheme, fn in (
        ("topk", lambda X, lam: ablate_topk_neighbors(X, lam, k=10)),
        ("random", lambda X, lam: ablate_random_10d(X, lam, rng=np.random.default_rng(0))),
        ("pca", lambda X, lam: ablate_pca_10d(X, lam)),
    ):
        # lam=0 should be a no-op.
        X0, _ = fn(X, 0.0)
        assert np.allclose(X0, X, atol=1e-6), f"{scheme} lam=0 not no-op"
        # lam=1 should move X by non-trivial amount.
        X1, info = fn(X, 1.0)
        delta = np.linalg.norm(X1 - X, "fro") / (np.linalg.norm(X, "fro") + 1e-9)
        print(f"{scheme:10s} lam=1 relative Frobenius shift: {delta:.4f}  info: {info}")
        assert delta > 1e-3, f"{scheme} lam=1 did not meaningfully perturb X"

    print("ALL OK")
