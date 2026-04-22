"""Held-out cross-modality test of framework D (p = 2/d_rd).

genome_036 supported the relation on 3 text systems (Qwen3 / RWKV /
DeepSeek, mean rel_err 8.5%). To avoid a text-only finding, test on the
vision half of the bestiary: DINOv2-small and I-JEPA-ViT-H/14.

Vision systems have higher empirical p (0.20-0.22 vs text 0.16-0.17). If
p = 2/d_rd holds, d_rd should be correspondingly smaller (~9-10 vs ~12-14
on text). If the relation breaks on vision, the framework is text-local
and we report it as such.
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
from genome_extractor import extract_vision_trajectory  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from genome_primitives import knn_clustering_coefficient  # noqa: E402
from genome_rate_distortion_probe import rate_distortion_dim, fit_power_law  # noqa: E402
from stimulus_banks import imagenet_val_v1  # noqa: E402

_ROOT = _THIS_DIR.parent
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]


def run_one(hf_id: str, system_key: str, n: int = 1000, seed: int = 42):
    print(f"\n=== {system_key} [TRAINED vision] ===")
    t0 = time.time()
    sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    n_layers = sys_obj.n_hidden_layers()
    mid = n_layers // 2

    images = []
    for rec in imagenet_val_v1(seed=seed, n_samples=n):
        images.append(rec["image"])
        if len(images) >= n:
            break

    traj = extract_vision_trajectory(
        model=sys_obj.model, image_processor=sys_obj.image_processor,
        images=images, layer_indices=[mid], pooling="cls_or_mean",
        device="cuda", system_key=system_key, class_id=6,
        quantization="fp16",
        stimulus_version=f"imagenet_val.v1.seed{seed}.n{n}",
        seed=seed, batch_size=16,
    )
    X = traj.layers[0].X.astype(np.float32)
    print(f"  cloud shape: {X.shape}")

    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p_emp, c0_emp, r2_ck = fit_power_law(K_GRID, Cs)
    rd = rate_distortion_dim(X)
    d_rd = rd["d_rd"]
    p_pred = 2.0 / d_rd if d_rd > 0 else float("nan")
    rel_err = abs(p_pred - p_emp) / abs(p_emp) if abs(p_emp) > 1e-6 else float("inf")

    print(f"  empirical p = {p_emp:.3f}  R^2 = {r2_ck:.4f}")
    print(f"  d_rd = {d_rd:.3f}  fit R^2 = {rd['R2']:.3f}")
    print(f"  p_pred = 2/d_rd = {p_pred:.3f}  rel_err = {rel_err:.3f}")

    sys_obj.unload()
    torch.cuda.empty_cache()
    return {
        "system_key": system_key, "hf_id": hf_id,
        "depth_idx": mid, "n_stimuli": int(X.shape[0]),
        "hidden_dim": int(X.shape[1]),
        "empirical_p": p_emp, "empirical_c0": c0_emp, "Ck_R2": r2_ck,
        "d_rd": d_rd, "rd_fit_R2": rd["R2"],
        "p_predicted_framework_D": p_pred,
        "rel_err": rel_err,
        "elapsed_s": time.time() - t0,
    }


def main():
    targets = [
        ("facebook/dinov2-small", "dinov2-small"),
        ("facebook/ijepa_vith14_1k", "ijepa-vith14"),
    ]
    results = []
    for hf_id, sk in targets:
        try:
            results.append(run_one(hf_id, sk))
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({"system_key": sk, "error": f"{type(e).__name__}: {e}"})

    print("\n=== SUMMARY: FRAMEWORK D held-out on vision systems ===")
    print(f"{'system':>20s} {'d_rd':>8s} {'p_emp':>8s} {'p_pred':>8s} {'rel_err':>8s}")
    passes = 0
    tested = 0
    for r in results:
        if "error" in r:
            print(f"  {r['system_key']:>20s}  ERROR: {r['error']}")
            continue
        tested += 1
        print(f"  {r['system_key']:>20s} {r['d_rd']:8.3f} {r['empirical_p']:8.3f} "
              f"{r['p_predicted_framework_D']:8.3f} {r['rel_err']:8.3f}")
        if r["rel_err"] < 0.20:
            passes += 1

    overall_status = ("SUPPORTED_ACROSS_MODALITIES" if passes == tested and tested >= 2
                      else "PARTIAL" if passes > 0
                      else "TEXT_LOCAL")
    print(f"\n  vision held-out pass rate: {passes}/{tested}")
    print(f"  combined (text + vision) verdict: {overall_status}")

    out = {"purpose": "Held-out cross-modality test of framework D p = 2/d_rd",
           "per_system": results,
           "vision_passes": passes, "vision_tested": tested,
           "overall_status": overall_status}
    out_path = _ROOT / "results/gate2/rate_distortion_vision_holdout.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
