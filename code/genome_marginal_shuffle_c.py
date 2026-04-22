"""Joint-structure attribution test:

Extract Qwen3-0.6B trained mid-depth activations (matches existing genome_044
baseline). Measure c = p·d_rd on:

  (a) trained cloud as-is
  (b) marginal-shuffled cloud: each feature dim independently permuted across
      stimuli, which preserves per-dim 1D histogram but destroys JOINT
      inter-dim structure

If c_trained >> c_shuffled, the c value in trained networks depends on
TRAINING-SPECIFIC joint structure — density non-uniformity per-axis alone
(which shuffle preserves) is not enough. This would rule out the simplest
'density explains c' alternatives and point at joint informational axis
alignment as the training-specific ingredient.
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
from genome_rate_distortion_probe import rate_distortion_dim, fit_power_law  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]


def compute_cpdrd(X):
    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p, _, r2 = fit_power_law(K_GRID, Cs)
    rd = rate_distortion_dim(X)
    return p, rd["d_rd"], p * rd["d_rd"], r2


def main():
    t0 = time.time()
    print(f"[{time.time()-t0:.1f}s] loading Qwen3-0.6B trained, extracting mid-depth...")
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
    print(f"[{time.time()-t0:.1f}s] trained cloud {X_trained.shape}")

    # Measure trained
    p_t, drd_t, c_t, r2_t = compute_cpdrd(X_trained)
    print(f"  TRAINED as-is:          p={p_t:.3f}  d_rd={drd_t:.2f}  c={c_t:.2f}  Ck_R2={r2_t:.3f}")

    # Marginal-shuffle: for each column (dim), permute the entries across rows
    rng = np.random.default_rng(42)
    X_shuf = X_trained.copy()
    for j in range(X_shuf.shape[1]):
        rng.shuffle(X_shuf[:, j])
    p_s, drd_s, c_s, r2_s = compute_cpdrd(X_shuf)
    print(f"  MARGINAL-SHUFFLED:      p={p_s:.3f}  d_rd={drd_s:.2f}  c={c_s:.2f}  Ck_R2={r2_s:.3f}")

    # Gaussian-match: preserve per-dim mean + std, resample as independent Gaussians
    mu = X_trained.mean(axis=0)
    sd = X_trained.std(axis=0)
    X_gauss = rng.normal(loc=mu, scale=sd, size=X_trained.shape).astype(np.float32)
    p_g, drd_g, c_g, r2_g = compute_cpdrd(X_gauss)
    print(f"  GAUSSIAN-MARGINAL-MATCH: p={p_g:.3f}  d_rd={drd_g:.2f}  c={c_g:.2f}  Ck_R2={r2_g:.3f}")

    print(f"\n=== SUMMARY ===")
    if c_t / max(c_s, 0.01) > 1.5:
        verdict = "JOINT_STRUCTURE_DOMINATES — trained c > shuffled c by >1.5x, training-specific joint structure"
    elif abs(c_t - c_s) / c_t < 0.15:
        verdict = "MARGINAL_EXPLAINS — shuffle preserves c, per-dim distribution carries it"
    else:
        verdict = "MIXED — partial joint-structure contribution"
    print(f"  c_trained / c_shuffled = {c_t / max(c_s, 0.01):.2f}x")
    print(f"  c_trained / c_gaussian = {c_t / max(c_g, 0.01):.2f}x")
    print(f"  verdict: {verdict}")

    out = {
        "trained": {"p": p_t, "d_rd": drd_t, "c": c_t, "Ck_R2": r2_t},
        "shuffled": {"p": p_s, "d_rd": drd_s, "c": c_s, "Ck_R2": r2_s},
        "gaussian_marginal": {"p": p_g, "d_rd": drd_g, "c": c_g, "Ck_R2": r2_g},
        "c_ratio_trained_over_shuffled": c_t / max(c_s, 0.01),
        "c_ratio_trained_over_gaussian": c_t / max(c_g, 0.01),
        "verdict": verdict,
    }
    out_path = _ROOT / "results/gate2/marginal_shuffle_c.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
