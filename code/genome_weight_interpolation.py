"""Weight-space linear interpolation sweep.

W(α) = α · W_trained + (1 - α) · W_untrained,   α ∈ {0, 0.25, 0.5, 0.75, 1.0}

For each α, measure:
  - Mid-depth pooled activation geometry (p, d_rd, c)
  - NLL on 500 C4-clean stimuli seed 42

Question: does the c = p * d_rd invariant emerge LINEARLY with α, or with a
threshold / phase-transition? Does capability (NLL) emerge linearly too, or
with a different curve?

If geometry and NLL both emerge linearly with α: weight-space is smooth/linear.
If geometry emerges linearly but NLL has a threshold: capability is nonlinear
  in weight composition.
If neither emerges linearly: interpolation lives in a region that isn't either
  model (expected for deep networks — Garipov et al mode connectivity).

Protocol: load trained + untrained, create blended state_dict, overwrite
untrained model's weights with blended, forward pass, restore weights.

Cheapest rung-3 capability-substrate test remaining — 5 alphas × 10s each.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_extractor import extract_trajectory  # noqa: E402
from genome_geometry_transfusion import measure_nll, fit_power_law  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from genome_primitives import knn_clustering_coefficient  # noqa: E402
from genome_rate_distortion_probe import rate_distortion_dim  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]


def main():
    hf_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    sk = "deepseek-r1-distill-qwen-1.5b"
    seed = 42
    n = 500
    t0 = time.time()

    sents = []
    for rec in c4_clean_v1(seed=seed, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= n:
            break
    print(f"[{time.time()-t0:.1f}s] {len(sents)} stim")

    # Load trained and capture its state_dict
    print(f"[{time.time()-t0:.1f}s] loading TRAINED...")
    sys_t = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    sd_trained = {k: v.detach().clone() for k, v in sys_t.model.state_dict().items()}
    sys_t.unload(); torch.cuda.empty_cache()

    # Load untrained and capture state_dict
    print(f"[{time.time()-t0:.1f}s] loading UNTRAINED...")
    sys_u = load_system(hf_id, quant="fp16", untrained=True, device="cuda")
    sd_untrained = {k: v.detach().clone() for k, v in sys_u.model.state_dict().items()}

    n_layers = sys_u.n_hidden_layers()
    mid = n_layers // 2

    def set_blended(alpha):
        """Overwrite model's weights with α·trained + (1-α)·untrained."""
        blended = {}
        for k in sd_untrained:
            if k in sd_trained and sd_trained[k].shape == sd_untrained[k].shape:
                if sd_trained[k].is_floating_point():
                    blended[k] = (alpha * sd_trained[k].float()
                                  + (1 - alpha) * sd_untrained[k].float()
                                  ).to(sd_untrained[k].dtype)
                else:
                    # integer tensors (like token embedding indices if any): use trained
                    blended[k] = sd_trained[k].clone()
            else:
                blended[k] = sd_untrained[k].clone()
        sys_u.model.load_state_dict(blended, strict=False)

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []
    for alpha in alphas:
        print(f"\n[{time.time()-t0:.1f}s] alpha = {alpha}...")
        set_blended(alpha)
        nll_a, _ = measure_nll(sys_u.model, sys_u.tokenizer, sents)
        traj_a = extract_trajectory(
            model=sys_u.model, tokenizer=sys_u.tokenizer,
            texts=sents, layer_indices=[mid], pooling="seq_mean",
            device="cuda", system_key=sk + f"_alpha{int(alpha*100)}", class_id=2,
            quantization="fp16",
            stimulus_version=f"alpha{alpha}.seed{seed}.n{n}", seed=seed,
            batch_size=16, max_length=256,
        )
        X_a = traj_a.layers[0].X.astype(np.float32)
        Cs_a = [float(knn_clustering_coefficient(X_a, k=k).value) for k in K_GRID]
        p_a, _, r2 = fit_power_law(K_GRID, Cs_a)
        rd_a = rate_distortion_dim(X_a)
        c_a = p_a * rd_a["d_rd"]
        print(f"  a={alpha}: p={p_a:.3f}  d_rd={rd_a['d_rd']:.2f}  c={c_a:.2f}  NLL={nll_a:.3f}")
        results.append({"alpha": alpha, "p": p_a, "d_rd": rd_a["d_rd"],
                        "c": c_a, "NLL": nll_a, "Ck_R2": r2})

    sys_u.unload(); torch.cuda.empty_cache()

    # Verdict
    nlls = [r["NLL"] for r in results]
    cs = [r["c"] for r in results]
    drs = [r["d_rd"] for r in results]

    # Linearity check: fit linear model to NLL vs alpha, measure R^2
    a = np.array(alphas); nll_arr = np.array(nlls); c_arr = np.array(cs)
    nll_slope, nll_int = np.polyfit(a, nll_arr, 1)
    nll_pred = nll_slope * a + nll_int
    nll_r2 = 1 - np.sum((nll_arr - nll_pred)**2) / np.sum((nll_arr - nll_arr.mean())**2)
    c_slope, c_int = np.polyfit(a, c_arr, 1)
    c_pred = c_slope * a + c_int
    c_r2 = 1 - np.sum((c_arr - c_pred)**2) / np.sum((c_arr - c_arr.mean())**2)

    print(f"\n=== WEIGHT-INTERPOLATION SUMMARY ===")
    print(f"{'alpha':>6s} {'d_rd':>8s} {'c':>7s} {'NLL':>8s}")
    for r in results:
        print(f"  {r['alpha']:6.2f} {r['d_rd']:8.2f} {r['c']:7.2f} {r['NLL']:8.3f}")
    print(f"\n  NLL vs alpha linear fit R^2 = {nll_r2:.3f}")
    print(f"  c   vs alpha linear fit R^2 = {c_r2:.3f}")

    if nll_r2 > 0.9 and c_r2 > 0.9:
        verdict = "LINEAR_IN_WEIGHT — both geometry and NLL linear in α"
    elif nll_r2 > 0.9:
        verdict = "NLL_LINEAR_GEOMETRY_NONLINEAR — capability transitions smoothly, geometry has structure"
    elif c_r2 > 0.9:
        verdict = "GEOMETRY_LINEAR_NLL_NONLINEAR — geometry is the simple quantity, capability has threshold"
    else:
        verdict = "BOTH_NONLINEAR — intermediate weights live in a distinct regime (mode-connectivity-like)"
    print(f"  verdict: {verdict}")

    out = {"purpose": "Weight-space linear interpolation W(α) = α·W_trained + (1-α)·W_untrained",
           "alphas": alphas, "per_alpha": results,
           "nll_linear_R2": float(nll_r2), "c_linear_R2": float(c_r2),
           "verdict": verdict}
    out_path = _ROOT / "results/gate2/weight_interpolation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
