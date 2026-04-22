"""Multi-system trained 3-stim-seed sweep to complete the trained-side
robustness check. genome_032 did Qwen3; this extends to RWKV + DeepSeek
+ DINOv2 so the 'trained cluster is tight per seed' claim is validated
across modalities.

3 stim_seeds x 4 systems x n=1000, mid-depth. ~15 min total.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_extractor import extract_trajectory, extract_vision_trajectory  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from genome_primitives import knn_clustering_coefficient  # noqa: E402
from stimulus_banks import c4_clean_v1, imagenet_val_v1  # noqa: E402

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


def run_one(hf_id: str, system_key: str, modality: str, stim_seed: int,
            n: int = 1000):
    t0 = time.time()
    sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    n_layers = sys_obj.n_hidden_layers()
    mid = n_layers // 2

    if modality == "text":
        sents = []
        for rec in c4_clean_v1(seed=stim_seed, n_samples=5000):
            sents.append(rec["text"])
            if len(sents) >= n:
                break
        traj = extract_trajectory(
            model=sys_obj.model, tokenizer=sys_obj.tokenizer,
            texts=sents, layer_indices=[mid], pooling="seq_mean",
            device="cuda", system_key=system_key, class_id=0,
            quantization="fp16",
            stimulus_version=f"c4_clean.v1.seed{stim_seed}.n{n}",
            seed=stim_seed, batch_size=16, max_length=256,
        )
    else:
        images = []
        for rec in imagenet_val_v1(seed=stim_seed, n_samples=n):
            images.append(rec["image"])
            if len(images) >= n:
                break
        traj = extract_vision_trajectory(
            model=sys_obj.model, image_processor=sys_obj.image_processor,
            images=images, layer_indices=[mid], pooling="cls_or_mean",
            device="cuda", system_key=system_key, class_id=0,
            quantization="fp16",
            stimulus_version=f"imagenet_val.v1.seed{stim_seed}.n{n}",
            seed=stim_seed, batch_size=16,
        )
    X = traj.layers[0].X.astype(np.float32)
    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p, c0, r2 = fit_power_law(K_GRID, Cs)
    print(f"  {system_key:>20s} stim_seed={stim_seed}: p={p:.3f} c_0={c0:.3f} R^2={r2:.4f}  t={time.time()-t0:.1f}s")
    sys_obj.unload()
    return {"system_key": system_key, "modality": modality,
            "stim_seed": stim_seed, "p": p, "c_0": c0, "R2": r2,
            "n": int(X.shape[0]), "h": int(X.shape[1])}


def main():
    targets = [
        ("RWKV/rwkv-4-169m-pile", "rwkv-4-169m", "text"),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
         "deepseek-r1-distill-qwen-1.5b", "text"),
        ("facebook/dinov2-small", "dinov2-small", "vision"),
    ]
    results = []
    for hf_id, sk, mod in targets:
        print(f"\n=== {sk} [{mod}] ===")
        for stim_seed in (42, 123, 456):
            try:
                results.append(run_one(hf_id, sk, mod, stim_seed))
            except Exception as e:
                import traceback; traceback.print_exc()
                results.append({"system_key": sk, "stim_seed": stim_seed,
                                "error": f"{type(e).__name__}: {e}"})

    per_system = {}
    for r in results:
        if "error" in r:
            continue
        per_system.setdefault(r["system_key"], []).append(r)

    print("\n=== SUMMARY: trained 3-stim-seed robustness ===")
    print(f"{'system':>32s} {'p_mean':>8s} {'p_std':>8s} {'CV%':>6s} {'p_range':>16s}")
    summary = {}
    for sk, rs in per_system.items():
        ps = np.array([r["p"] for r in rs])
        m = ps.mean(); s = ps.std(ddof=1) if ps.size > 1 else 0.0
        cv = 100 * s / abs(m) if m != 0 else 0.0
        rng = f"[{ps.min():.3f}, {ps.max():.3f}]"
        print(f"  {sk:>32s} {m:8.3f} {s:8.3f} {cv:6.2f} {rng:>16s}")
        summary[sk] = {"p_mean": float(m), "p_std": float(s), "p_cv_pct": float(cv),
                       "p_range": [float(ps.min()), float(ps.max())]}

    out = {"purpose": "3-stim-seed robustness for trained RWKV + DeepSeek + DINOv2",
           "per_run": results, "per_system_summary": summary}
    out_path = _ROOT / "results/gate2/trained_stim_seed_sweep_all.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
