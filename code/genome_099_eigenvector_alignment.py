"""Adversarial #4: eigenvector alignment across trained ML systems.

Codex flagged: invariant only talks about eigenvalues, not eigenvectors.
Our null catalog (genome_093) showed aux-loss matching eigenvalues does NOT
install capability. That suggests capability may live in direction identity.

If systems share the same eigenvalue STRUCTURE (spectrum shape) but have
DIFFERENT top eigenvector directions when compared via CKA / subspace overlap,
then the invariant is epiphenomenal: shape universal, direction-identity
specific.

This probe:
 1. Extract pooled mid-depth activations for 5 text systems
 2. For each system, compute top-30 right singular vectors V in R^(h x 30)
 3. For each pair (A, B), compute max subspace overlap using projection norms:
    - For systems with different hidden dims, compute overlap in the
      DATA-SIDE singular space (left singular vectors U in R^(n x 30))
      which all share the same n dimension (the C4 sentences)
    - overlap(A,B) = ||U_A^T U_B||_F / sqrt(k) for k singular directions

High overlap (near 1) → same directions → universal subspace
Low overlap (near 1/sqrt(h)) → random orthogonal → direction identity is specific

Key interpretation:
  If overlap is high across systems, spectrum shape universality reflects
  shared semantic subspace - invariant is meaningful.
  If overlap is low, spectrum is shape-universal but direction-specific;
  the invariant measures noise-level-convergence, not shared semantics.
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


SYSTEMS = [
    ("qwen3-0.6b",                   "Qwen/Qwen3-0.6B"),
    ("deepseek-r1-distill-qwen-1.5b","deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("bert-base-uncased",            "bert-base-uncased"),
    ("roberta-base",                 "FacebookAI/roberta-base"),
    ("minilm-l6-contrastive",        "sentence-transformers/all-MiniLM-L6-v2"),
]


def extract_top_k_left_singular(X, k=30):
    """Return top-k left singular vectors of centered X ∈ R^(n, h).
    U has shape (n, k), shared across systems (same stimuli)."""
    Xc = X - X.mean(axis=0)
    # full_matrices=False: economy SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :k], S[:k]


def subspace_overlap(U_A, U_B):
    """||U_A^T U_B||_F^2 / k — Frobenius projection squared norm (CKA-like)."""
    k = U_A.shape[1]
    M = U_A.T @ U_B
    return float(np.linalg.norm(M, "fro") ** 2 / k)


def main():
    t0 = time.time()
    N = 800
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= N:
            break

    Us = {}
    for sys_key, hf_id in SYSTEMS:
        print(f"\n[{time.time()-t0:.1f}s] ===== {sys_key} =====")
        try:
            sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
        except Exception as e:
            print(f"  FAIL load: {e}"); continue
        mid = max(1, sys_obj.n_hidden_layers() // 2)
        try:
            traj = extract_trajectory(
                model=sys_obj.model, tokenizer=sys_obj.tokenizer,
                texts=sents, layer_indices=[mid], pooling="seq_mean",
                device="cuda", system_key=sys_key, class_id=1,
                quantization="fp16",
                stimulus_version="c4_clean.v1.seed42.n800",
                seed=42, batch_size=16, max_length=256,
            )
            X = traj.layers[0].X.astype(np.float32)
        except Exception as e:
            print(f"  FAIL extract: {e}")
            sys_obj.unload(); torch.cuda.empty_cache(); continue
        sys_obj.unload(); torch.cuda.empty_cache()
        U, S = extract_top_k_left_singular(X, k=30)
        Us[sys_key] = U
        print(f"  X shape {X.shape}, top-30 left singvec shape {U.shape}")

    systems = list(Us.keys())
    n = len(systems)
    overlap_matrix = np.zeros((n, n))
    for i, a in enumerate(systems):
        for j, b in enumerate(systems):
            overlap_matrix[i, j] = subspace_overlap(Us[a], Us[b])

    # Baseline for comparison: if two random orthogonal 30-dim subspaces in R^n,
    # expected overlap = 30/n. At n=800 that's 30/800 = 0.0375. Overlap ≈ 0.04
    # means "no shared structure". Overlap ≈ 1 means "same subspace".
    n_stim = 800
    random_baseline = 30.0 / n_stim

    print("\n\n=== TOP-30 LEFT SINGULAR-VECTOR SUBSPACE OVERLAP MATRIX ===")
    print(f"  random-baseline (no shared structure): {random_baseline:.4f}")
    print(f"  identity (same subspace):              1.0")
    print()
    print(f"  {'':32s}", end="")
    for s in systems:
        print(f" {s[:12]:>13s}", end="")
    print()
    for i, a in enumerate(systems):
        print(f"  {a:30s}  ", end="")
        for j in range(n):
            v = overlap_matrix[i, j]
            print(f"{v:>13.4f}", end="")
        print()

    # Off-diagonal average
    off_diag = []
    for i in range(n):
        for j in range(i + 1, n):
            off_diag.append(overlap_matrix[i, j])
    if off_diag:
        mean_off = float(np.mean(off_diag))
        print(f"\n  mean off-diagonal overlap: {mean_off:.4f}")
        print(f"  ratio to random baseline:  {mean_off/random_baseline:.2f}x")
        if mean_off < random_baseline * 3:
            print("  VERDICT: direction identity is LARGELY UNSHARED across systems")
        elif mean_off > 0.5:
            print("  VERDICT: direction identity is LARGELY SHARED across systems")
        else:
            print("  VERDICT: direction identity is PARTIALLY shared")

    out = {"systems": systems, "overlap_matrix": overlap_matrix.tolist(),
           "random_baseline": random_baseline,
           "mean_off_diag": float(np.mean(off_diag)) if off_diag else None}
    out_path = _ROOT / "results/gate2/eigenvector_alignment.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
