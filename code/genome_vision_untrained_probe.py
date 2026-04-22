"""Vision-side negative control: DINOv2 + CLIP-vision untrained twins.

The genome_028 / 029 / 030 training-convergence finding covers 3 TEXT
systems. This probe extends to 2 VISION systems (DINOv2-small and
CLIP-ViT-B/32 image branch) at mid-depth under ImageNet-val n=1000.

Tests whether cross-architecture convergence-to-p~=0.17 also holds in
the vision modality, or if vision architectures have their own
random-init -> trained mapping.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoConfig, AutoImageProcessor, AutoModel, CLIPVisionModel,
)

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_extractor import extract_vision_trajectory  # noqa: E402
from genome_primitives import knn_clustering_coefficient  # noqa: E402
from stimulus_banks import imagenet_val_v1  # noqa: E402

_ROOT = _THIS_DIR.parent
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]


def fit_power_law(ks, Cs):
    lks = np.log(np.asarray(ks, dtype=float))
    lcs = np.log(np.asarray(Cs, dtype=float))
    p, log_c0 = np.polyfit(lks, lcs, 1)
    pred = p * lks + log_c0
    ss_res = float(np.sum((lcs - pred) ** 2))
    ss_tot = float(np.sum((lcs - lcs.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(p), float(np.exp(log_c0)), r2


def run_one(hf_id: str, system_key: str, untrained: bool, n: int = 1000,
            seed: int = 42, model_cls_name: str = "AutoModel"):
    tag = "UNTRAINED" if untrained else "TRAINED"
    print(f"\n=== {system_key} [{tag}] ===")
    t0 = time.time()
    image_processor = AutoImageProcessor.from_pretrained(hf_id)
    if model_cls_name == "CLIPVisionModel":
        from transformers import CLIPVisionModel as MCLS
        if untrained:
            config = AutoConfig.from_pretrained(hf_id)
            # For CLIP wrapper we want just the vision_config side
            vc = config.vision_config if hasattr(config, "vision_config") else config
            model = MCLS(vc).to(dtype=torch.float16, device="cuda").eval()
        else:
            model = MCLS.from_pretrained(hf_id, torch_dtype=torch.float16).to("cuda").eval()
    else:
        if untrained:
            config = AutoConfig.from_pretrained(hf_id)
            model = AutoModel.from_config(config, torch_dtype=torch.float16).to("cuda").eval()
        else:
            model = AutoModel.from_pretrained(hf_id, torch_dtype=torch.float16).to("cuda").eval()

    config = model.config
    n_layers = None
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(config, attr):
            v = getattr(config, attr)
            if isinstance(v, int) and v > 0:
                n_layers = v
                break
    if n_layers is None and hasattr(config, "vision_config"):
        n_layers = getattr(config.vision_config, "num_hidden_layers", None)
    mid = n_layers // 2

    images = []
    for rec in imagenet_val_v1(seed=seed, n_samples=n):
        images.append(rec["image"])
        if len(images) >= n:
            break
    print(f"[{time.time()-t0:.1f}s] {len(images)} images, mid layer {mid}")

    traj = extract_vision_trajectory(
        model=model, image_processor=image_processor,
        images=images, layer_indices=[mid], pooling="cls_or_mean",
        device="cuda", system_key=system_key, class_id=6,
        quantization="fp16",
        stimulus_version="imagenet_val.v1.seed42.n1000",
        seed=seed, batch_size=16,
    )
    X = traj.layers[0].X.astype(np.float32)
    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p, c0, r2 = fit_power_law(K_GRID, Cs)
    diffs = np.diff(Cs)
    monotone = bool(np.all(diffs >= 0) or np.all(diffs <= 0))
    print(f"  C = {[round(c, 3) for c in Cs]}")
    print(f"  {tag}: p={p:.3f}  c_0={c0:.3f}  R^2={r2:.4f}  monotone={monotone}")
    del model
    torch.cuda.empty_cache()
    return {
        "system_key": system_key, "hf_id": hf_id, "untrained": untrained,
        "depth_idx": mid, "n_stimuli": int(X.shape[0]),
        "hidden_dim": int(X.shape[1]),
        "k_grid": K_GRID, "C_values": Cs,
        "p_slope": p, "c_0": c0, "R2": r2, "monotone": monotone,
        "elapsed_s": time.time() - t0,
    }


def main():
    targets = [
        ("facebook/dinov2-small", "dinov2-small", "AutoModel"),
        ("openai/clip-vit-base-patch32", "clip-vit-b32-image", "CLIPVisionModel"),
    ]
    results = []
    for hf_id, sk, cls_name in targets:
        for untrained in (False, True):
            try:
                results.append(run_one(hf_id, sk, untrained=untrained,
                                        model_cls_name=cls_name))
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"FAILED on {sk} untrained={untrained}: {type(e).__name__}: {e}")
                results.append({"system_key": sk, "untrained": untrained,
                                "error": f"{type(e).__name__}: {e}"})

    print("\n=== SUMMARY: VISION trained vs untrained ===")
    print(f"{'system':>22s} {'tag':>10s} {'p':>8s} {'c_0':>8s} {'R^2':>8s}")
    for r in results:
        if "error" in r:
            print(f"  {r['system_key']:>22s}  ERROR: {r['error']}")
            continue
        tag = "untrained" if r["untrained"] else "trained"
        print(f"  {r['system_key']:>22s} {tag:>10s} {r['p_slope']:8.3f} "
              f"{r['c_0']:8.3f} {r['R2']:8.4f}")

    out = {"purpose": "Vision negative control: DINOv2 + CLIP trained vs untrained",
           "per_cell": results}
    out_path = _ROOT / "results/gate2/vision_untrained_power_law.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
