"""Measure d_rd on DiT-XL/2-256 mid-depth pooled features; compute c = p * d_rd.

Tests candidate-4 prediction in research/derivations/c_integer_derivation_attempt.md:
  DiT stimulus dim = 2D spatial + 1D class-identity (+ maybe 1D noise-timestep)
  Candidate-4 predicts: c ≈ 3 (or 4 if noise-time axis counts)

Reuses the DiT probe infrastructure from code/genome_dit_probe.py — loads
DiT-XL/2-256, VAE-encodes 1000 ImageNet-val stimuli seed 42, noise at t=250,
forward with null class, extracts mid-depth pooled point cloud, measures
p, d_rd, c.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DiTTransformer2DModel

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_dit_probe import (  # noqa: E402
    SENTINEL_DEPTHS, K_GRID, FIXED_TIMESTEP, NULL_CLASS, DIT_SCALING,
    load_stimuli_imagenet, encode_with_vae, extract_dit_features,
)
from genome_primitives import knn_clustering_coefficient  # noqa: E402
from genome_rate_distortion_probe import rate_distortion_dim, fit_power_law  # noqa: E402

_ROOT = _THIS_DIR.parent


def main():
    t0 = time.time()
    n = 1000
    seed = 42
    print(f"[{time.time()-t0:.1f}s] loading DiT-XL/2-256 + VAE...")
    device = "cuda"
    transformer = DiTTransformer2DModel.from_pretrained(
        "facebook/DiT-XL-2-256", subfolder="transformer",
        torch_dtype=torch.float16).to(device).eval()
    vae = AutoencoderKL.from_pretrained(
        "facebook/DiT-XL-2-256", subfolder="vae",
        torch_dtype=torch.float16).to(device).eval()

    n_layers = len(transformer.transformer_blocks)
    mid_block = int(round(0.52 * (n_layers - 1)))  # mid-depth (14 for 28 layers)
    print(f"[{time.time()-t0:.1f}s] {n_layers} blocks, sampling mid block {mid_block}")

    print(f"[{time.time()-t0:.1f}s] loading {n} ImageNet images seed={seed}...")
    images = load_stimuli_imagenet(seed, n)
    print(f"[{time.time()-t0:.1f}s] got {len(images)} images")

    print(f"[{time.time()-t0:.1f}s] VAE-encoding...")
    latents = encode_with_vae(vae, images, device)
    print(f"[{time.time()-t0:.1f}s] latents {tuple(latents.shape)}")

    print(f"[{time.time()-t0:.1f}s] DiT forward (t={FIXED_TIMESTEP}, null class)...")
    features, _ = extract_dit_features(transformer, latents, [mid_block], device, seed=seed)
    X = features[mid_block].astype(np.float32)
    print(f"[{time.time()-t0:.1f}s] mid-depth cloud {X.shape}")

    # Power-law fit + d_rd
    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p, c0, r2 = fit_power_law(K_GRID, Cs)
    rd = rate_distortion_dim(X)
    c = p * rd["d_rd"]
    print(f"\n  DiT-XL/2-256 mid-depth:")
    print(f"    p = {p:.3f}")
    print(f"    d_rd = {rd['d_rd']:.2f}  (rd_fit_R2={rd['R2']:.3f})")
    print(f"    c = p*d_rd = {c:.2f}")
    print(f"    C(k) R^2 = {r2:.3f}")

    # Candidate-4 prediction check
    prediction = 3.0  # 2D spatial + 1D class-identity (noise timestep fixed, ignored)
    alt_prediction = 4.0  # if noise timestep counts
    rel_err_3 = abs(c - prediction) / prediction
    rel_err_4 = abs(c - alt_prediction) / alt_prediction

    print(f"\n  Candidate-4 predictions:")
    print(f"    c=3 (2D spatial + 1 identity): rel_err = {rel_err_3:.3f}")
    print(f"    c=4 (+ noise timestep):        rel_err = {rel_err_4:.3f}")
    nearest = 3.0 if rel_err_3 < rel_err_4 else 4.0
    pass_threshold = min(rel_err_3, rel_err_4) < 0.20
    verdict = (
        f"CANDIDATE4_SUPPORTED (c={c:.2f} within 20% of {nearest:.0f})"
        if pass_threshold else
        f"CANDIDATE4_FALSIFIED (c={c:.2f} not within 20% of either 3 or 4)"
    )
    print(f"  verdict: {verdict}")

    out = {
        "system": "dit-xl-2-256", "stimulus_bank": "ImageNet-val", "n": n, "seed": seed,
        "mid_block": mid_block, "timestep": FIXED_TIMESTEP, "null_class": NULL_CLASS,
        "p": p, "c_0": c0, "Ck_R2": r2,
        "d_rd": rd["d_rd"], "rd_fit_R2": rd["R2"],
        "c_invariant": c,
        "candidate_4_prediction_3": 3.0, "rel_err_vs_3": rel_err_3,
        "candidate_4_prediction_4": 4.0, "rel_err_vs_4": rel_err_4,
        "verdict": verdict,
    }
    out_path = _ROOT / "results/gate2/dit_c_invariant.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
