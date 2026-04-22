"""Vision-only candidate-8 bridge probe — complements text-side genome_060.

DINOv2-small only (Batch-1 vision anchor). Tests whether
`c = p * d_rd` approximately equals `eff_rank / d_rd` on image
activations. If yes, candidate-8 bridge extends cross-modality.
Merges result into results/gate2/svd_bridge_multimodel.json on exit.
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


def spectrum(X):
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False) / np.sqrt(X.shape[0] - 1)
    return s.astype(np.float64)


def eff_rank(s):
    s2 = s ** 2
    total = s2.sum()
    return 0.0 if total <= 0 else float(total ** 2 / (s2 ** 2).sum())


def fit_alpha_tail(s, lo_frac=0.05, hi_frac=0.5):
    r = np.arange(1, len(s) + 1)
    lo = max(1, int(len(s) * lo_frac))
    hi = int(len(s) * hi_frac)
    slope, _ = np.polyfit(np.log(r[lo:hi]), np.log(s[lo:hi] + 1e-12), 1)
    return float(-slope)


def main():
    seed = 42
    n = 500
    t0 = time.time()
    print(f"[{time.time()-t0:.1f}s] loading imagenet samples...")
    imgs = []
    for rec in imagenet_val_v1(seed=seed, n_samples=n):
        imgs.append(rec["image"])
        if len(imgs) >= n:
            break
    print(f"  {len(imgs)} images")

    hf_id = "facebook/dinov2-small"
    sk = "dinov2-small"
    print(f"[{time.time()-t0:.1f}s] loading {sk}...")
    sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    mid = sys_obj.n_hidden_layers() // 2
    traj = extract_vision_trajectory(
        model=sys_obj.model, image_processor=sys_obj.image_processor,
        images=imgs, layer_indices=[mid], pooling="cls_or_mean",
        device="cuda", system_key=sk, class_id=6,
        quantization="fp16",
        stimulus_version=f"imagenet_val.v1.seed{seed}.n{n}",
        seed=seed, batch_size=16,
    )
    X = traj.layers[0].X.astype(np.float32)
    sys_obj.unload(); torch.cuda.empty_cache()

    s = spectrum(X)
    er = eff_rank(s)
    alpha = fit_alpha_tail(s)
    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p, _, _ = fit_power_law(K_GRID, Cs)
    rd = rate_distortion_dim(X)
    c = p * rd["d_rd"]
    ratio = er / rd["d_rd"]
    rel_err = abs(ratio - c) / c
    verdict = "PASS" if rel_err < 0.15 else "FAIL"
    print(f"\n  {sk}: c={c:.3f}  eff_rank={er:.1f}  d_rd={rd['d_rd']:.2f}  "
          f"alpha={alpha:.3f}  ratio={ratio:.3f}  rel_err={rel_err:.3f}  {verdict}")

    # Merge into existing bridge JSON
    main_json = _ROOT / "results/gate2/svd_bridge_multimodel.json"
    record = {"system": sk, "modality": "vision",
              "c": float(c), "p": float(p), "d_rd": float(rd["d_rd"]),
              "eff_rank": er, "alpha": alpha, "ratio": ratio,
              "rel_err_ratio_vs_c": rel_err}
    try:
        existing = json.loads(main_json.read_text())
    except Exception:
        existing = {"purpose": "Candidate-8 bridge",
                    "per_system": [], "verdict": "IN_PROGRESS"}
    # Replace any prior DINOv2 record
    existing["per_system"] = [r for r in existing["per_system"]
                              if r.get("system") != sk]
    existing["per_system"].append(record)

    # Recompute global verdict
    passes = 0; tested = 0
    for r in existing["per_system"]:
        if "error" in r:
            continue
        tested += 1
        if r.get("rel_err_ratio_vs_c", 1) < 0.15:
            passes += 1
    rate = passes / max(tested, 1)
    if rate >= 0.8:
        existing["verdict"] = (f"CANDIDATE_8_SUPPORTED - {passes}/{tested}")
    elif rate >= 0.5:
        existing["verdict"] = (f"CANDIDATE_8_PARTIAL - {passes}/{tested}")
    else:
        existing["verdict"] = (f"CANDIDATE_8_FALSIFIED - only {passes}/{tested}")

    main_json.write_text(json.dumps(existing, indent=2))
    print(f"\n  merged vision row into {main_json}")
    print(f"  updated verdict: {existing['verdict']}")


if __name__ == "__main__":
    main()
