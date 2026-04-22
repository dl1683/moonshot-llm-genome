"""Stimulus intrinsic-dim sweep — rung-2 derivation experiment.

Per research/prereg/genome_stim_dim_sweep_2026-04-21.md (LOCKED).

Primary test: on DINOv2-small mid-depth, compute c = p * d_rd under:
  - c_natural      : 1000 ImageNet-val images seed 42
  - c_1d_stripes   : 1000 single-row-replicated 1D-structure images
  - c_iid_noise    : 1000 IID uniform-noise images

Pre-reg prediction: c_1d_stripes < c_natural by >=0.3; c_iid_noise even
lower OR breaks power-law (R^2 < 0.90).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_extractor import extract_vision_trajectory  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from genome_primitives import knn_clustering_coefficient  # noqa: E402
from genome_rate_distortion_probe import rate_distortion_dim, fit_power_law  # noqa: E402
from stimulus_banks import imagenet_val_v1  # noqa: E402

_ROOT = _THIS_DIR.parent
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]


def make_1d_stripe_image(pil: Image.Image, size: int = 224) -> Image.Image:
    """Collapse a natural image to an approximate 1D structure by picking a
    single row and tiling it vertically."""
    arr = np.asarray(pil.convert("RGB").resize((size, size), Image.BICUBIC))
    # Use middle row
    row = arr[size // 2:size // 2 + 1, :, :]  # (1, size, 3)
    tiled = np.repeat(row, size, axis=0)  # (size, size, 3)
    return Image.fromarray(tiled)


def make_iid_noise_image(rng: np.random.Generator, size: int = 224) -> Image.Image:
    arr = (rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8))
    return Image.fromarray(arr)


def extract_c(model_sys, images, condition_name: str, seed: int = 42):
    n_layers = model_sys.n_hidden_layers()
    mid = n_layers // 2
    traj = extract_vision_trajectory(
        model=model_sys.model, image_processor=model_sys.image_processor,
        images=images, layer_indices=[mid], pooling="cls_or_mean",
        device="cuda", system_key=model_sys.system_key, class_id=6,
        quantization="fp16",
        stimulus_version=f"stim_dim_{condition_name}.seed{seed}.n{len(images)}",
        seed=seed, batch_size=16,
    )
    X = traj.layers[0].X.astype(np.float32)
    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p, c0, r2 = fit_power_law(K_GRID, Cs)
    rd = rate_distortion_dim(X)
    c_invariant = p * rd["d_rd"]
    return {
        "condition": condition_name, "n": int(X.shape[0]), "h": int(X.shape[1]),
        "p": p, "c_0": c0, "Ck_R2": r2,
        "d_rd": rd["d_rd"], "rd_fit_R2": rd["R2"],
        "c_invariant": c_invariant,
    }


def run_system(hf_id: str, sk: str, seed: int = 42, n: int = 1000):
    print(f"\n=== {sk} [vision] ===")
    t0 = time.time()
    sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    rng = np.random.default_rng(seed)

    # Load natural images once
    natural = []
    for rec in imagenet_val_v1(seed=seed, n_samples=n):
        natural.append(rec["image"])
        if len(natural) >= n:
            break
    print(f"[{time.time()-t0:.1f}s] got {len(natural)} natural stimuli")

    print(f"[{time.time()-t0:.1f}s] condition natural...")
    r_nat = extract_c(sys_obj, natural, "natural", seed)
    print(f"  natural: p={r_nat['p']:.3f}  d_rd={r_nat['d_rd']:.2f}  c={r_nat['c_invariant']:.2f}  R^2={r_nat['Ck_R2']:.3f}")

    # Build 1D-stripe images
    stripes = [make_1d_stripe_image(img) for img in natural]
    print(f"[{time.time()-t0:.1f}s] condition 1d_stripes...")
    r_stripes = extract_c(sys_obj, stripes, "1d_stripes", seed)
    print(f"  stripes: p={r_stripes['p']:.3f}  d_rd={r_stripes['d_rd']:.2f}  c={r_stripes['c_invariant']:.2f}  R^2={r_stripes['Ck_R2']:.3f}")

    # Build IID noise images
    noise = [make_iid_noise_image(rng) for _ in range(n)]
    print(f"[{time.time()-t0:.1f}s] condition iid_noise...")
    r_noise = extract_c(sys_obj, noise, "iid_noise", seed)
    print(f"  noise:   p={r_noise['p']:.3f}  d_rd={r_noise['d_rd']:.2f}  c={r_noise['c_invariant']:.2f}  R^2={r_noise['Ck_R2']:.3f}")

    sys_obj.unload(); torch.cuda.empty_cache()
    return {"system": sk, "hf_id": hf_id,
            "conditions": {"natural": r_nat, "1d_stripes": r_stripes, "iid_noise": r_noise},
            "elapsed_s": time.time() - t0}


def verdict(r):
    c_nat = r["conditions"]["natural"]["c_invariant"]
    c_str = r["conditions"]["1d_stripes"]["c_invariant"]
    c_noise = r["conditions"]["iid_noise"]["c_invariant"]
    r2_str = r["conditions"]["1d_stripes"]["Ck_R2"]
    # Rules from prereg
    rule1 = 2.63 <= c_nat <= 3.95  # replicates prior vision c band
    rule2 = (c_nat - c_str) >= 0.3  # shift prediction
    rule3 = (c_str - c_noise) > 0 or r2_str < 0.90  # floor
    rule4 = 1.7 <= c_str <= 2.3  # optional tight
    if rule1 and rule2 and rule3 and rule4:
        return "STRONGLY_SUPPORTED"
    if rule1 and rule2 and rule3:
        return "SUPPORTED"
    if (c_nat - c_str) > 0:
        return "PARTIAL_direction_correct_magnitude_small"
    return "FALSIFIED"


def main():
    results = [run_system("facebook/dinov2-small", "dinov2-small")]

    print("\n=== VERDICT ===")
    for r in results:
        v = verdict(r)
        c_nat = r["conditions"]["natural"]["c_invariant"]
        c_str = r["conditions"]["1d_stripes"]["c_invariant"]
        c_noise = r["conditions"]["iid_noise"]["c_invariant"]
        shift = c_nat - c_str
        print(f"  {r['system']}: c(nat)={c_nat:.2f}  c(1d)={c_str:.2f}  c(noise)={c_noise:.2f}  shift={shift:+.2f}")
        print(f"    verdict: {v}")

    out = {"purpose": "Stimulus intrinsic-dim sweep — does c shift when vision forced 1D?",
           "per_system": results,
           "verdicts": {r["system"]: verdict(r) for r in results}}
    out_path = _ROOT / "results/gate2/stim_dim_sweep.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
