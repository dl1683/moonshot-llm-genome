"""Expand d_rd coverage: Qwen3-1.7B (text) + DiT-XL/2-256 (vision, on ImageNet).

Combined with existing data (Qwen3-0.6B / RWKV / DeepSeek text, DINOv2 /
I-JEPA vision), this gives 4 text + 3 vision points on the invariant
c = p × d_rd.

Hypothesis: c is modality-stratified:
  text:   c ≈ 2.0  =>  p = 2 / d_rd
  vision: c ≈ 3.0  =>  p = 3 / d_rd

If c is tight within each modality across ≥3 systems each, the
training-manifold invariant is c(modality).
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
from genome_extractor import extract_trajectory, extract_vision_trajectory  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from genome_primitives import knn_clustering_coefficient  # noqa: E402
from genome_rate_distortion_probe import rate_distortion_dim, fit_power_law  # noqa: E402
from stimulus_banks import c4_clean_v1, imagenet_val_v1  # noqa: E402

_ROOT = _THIS_DIR.parent
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]


def run_text(hf_id, sk):
    print(f"\n=== {sk} [TEXT] ===")
    sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    n_layers = sys_obj.n_hidden_layers()
    mid = n_layers // 2
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= 1000:
            break
    traj = extract_trajectory(
        model=sys_obj.model, tokenizer=sys_obj.tokenizer,
        texts=sents, layer_indices=[mid], pooling="seq_mean",
        device="cuda", system_key=sk, class_id=1, quantization="fp16",
        stimulus_version="c4_clean.v1.seed42.n1000", seed=42,
        batch_size=16, max_length=256,
    )
    X = traj.layers[0].X.astype(np.float32)
    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p, c0, r2 = fit_power_law(K_GRID, Cs)
    rd = rate_distortion_dim(X)
    sys_obj.unload(); torch.cuda.empty_cache()
    c = p * rd["d_rd"]
    print(f"  p={p:.3f}  d_rd={rd['d_rd']:.2f}  c=p*d_rd={c:.2f}")
    return {"system_key": sk, "modality": "text", "p": p, "d_rd": rd["d_rd"],
            "c_product": c, "Ck_R2": r2}


def run_vision(hf_id, sk):
    print(f"\n=== {sk} [VISION] ===")
    sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    n_layers = sys_obj.n_hidden_layers()
    mid = n_layers // 2
    images = []
    for rec in imagenet_val_v1(seed=42, n_samples=1000):
        images.append(rec["image"])
        if len(images) >= 1000:
            break
    traj = extract_vision_trajectory(
        model=sys_obj.model, image_processor=sys_obj.image_processor,
        images=images, layer_indices=[mid], pooling="cls_or_mean",
        device="cuda", system_key=sk, class_id=6, quantization="fp16",
        stimulus_version="imagenet_val.v1.seed42.n1000", seed=42,
        batch_size=16,
    )
    X = traj.layers[0].X.astype(np.float32)
    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p, c0, r2 = fit_power_law(K_GRID, Cs)
    rd = rate_distortion_dim(X)
    sys_obj.unload(); torch.cuda.empty_cache()
    c = p * rd["d_rd"]
    print(f"  p={p:.3f}  d_rd={rd['d_rd']:.2f}  c=p*d_rd={c:.2f}")
    return {"system_key": sk, "modality": "vision", "p": p, "d_rd": rd["d_rd"],
            "c_product": c, "Ck_R2": r2}


def main():
    results = []
    # Add Qwen3-1.7B to text
    try:
        results.append(run_text("Qwen/Qwen3-1.7B", "qwen3-1.7b"))
    except Exception as e:
        print(f"FAILED Qwen3-1.7B text: {e}")
    # CLIP-vision to vision (3rd vision point)
    try:
        results.append(run_vision("openai/clip-vit-base-patch32", "clip-vit-b32-image"))
    except Exception as e:
        print(f"FAILED CLIP vision: {e}")

    # Seed the existing data from earlier runs
    existing = {
        "qwen3-0.6b":    {"modality": "text", "p": 0.154, "d_rd": 12.27},
        "rwkv-4-169m":   {"modality": "text", "p": 0.171, "d_rd": 11.40},
        "deepseek-r1-distill-qwen-1.5b": {"modality": "text", "p": 0.171, "d_rd": 14.06},
        "dinov2-small":  {"modality": "vision", "p": 0.219, "d_rd": 13.51},
        "ijepa-vith14":  {"modality": "vision", "p": 0.192, "d_rd": 13.69},
    }
    for sk, v in existing.items():
        v["system_key"] = sk
        v["c_product"] = v["p"] * v["d_rd"]
        results.append(v)

    print(f"\n=== CROSS-SYSTEM c = p * d_rd ===")
    print(f"{'system':>30s} {'modality':>8s} {'p':>7s} {'d_rd':>7s} {'c':>7s}")
    by_mod = {}
    for r in results:
        mod = r["modality"]
        print(f"  {r['system_key']:>30s} {mod:>8s} {r['p']:7.3f} {r['d_rd']:7.2f} {r['c_product']:7.2f}")
        by_mod.setdefault(mod, []).append(r["c_product"])

    print("\nPer-modality c statistics:")
    for mod, cs in by_mod.items():
        arr = np.array(cs)
        print(f"  {mod}: n={len(cs)}  c mean={arr.mean():.2f}  std={arr.std(ddof=1) if len(cs)>1 else 0:.2f}  range=[{arr.min():.2f}, {arr.max():.2f}]")

    out = {"purpose": "Is c = p * d_rd a modality-stratified constant?",
           "per_system": results,
           "per_modality_summary": {
               mod: {"n": len(cs), "c_mean": float(np.array(cs).mean()),
                     "c_std": float(np.array(cs).std(ddof=1)) if len(cs) > 1 else 0.0}
               for mod, cs in by_mod.items()
           }}
    out_path = _ROOT / "results/gate2/drd_c_invariant_cross_modality.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
