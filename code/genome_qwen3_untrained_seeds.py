"""Multi-seed untrained Qwen3 probe.

The genome_028 Qwen3 UNTRAINED result was a U-shape in C(k) (not a power
law): values [0.330, 0.337, 0.317, 0.308, 0.304, 0.301, 0.294, 0.321,
0.358, 0.377] — drops from k=3 to k=40, then rises. R^2 of log-linear fit
therefore collapsed to 0.110. This probe tests whether the U-shape is
stable across different random-init seeds or a single-seed artifact.

Runs Qwen3-0.6B untrained at 3 different torch random seeds (42, 123, 456),
n=1000 C4 stimuli (fixed seed 42), mid-depth. Reports per-seed C(k)
shape and log-linear fit R^2 + p.

If all 3 seeds give R^2 < 0.8, the U-shape is a structural property of
random-init Qwen3 architecture (not a seed outlier).
If R^2 varies widely across seeds, the genome_028 Qwen3 result was one
draw from a bigger distribution — de-emphasize in paper.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_extractor import extract_trajectory  # noqa: E402
from genome_primitives import knn_clustering_coefficient  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

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


def run_one(torch_seed: int, stim_seed: int = 42, n: int = 1000):
    print(f"\n=== torch_seed={torch_seed} ===")
    t0 = time.time()
    torch.manual_seed(torch_seed)
    hf_id = "Qwen/Qwen3-0.6B"
    config = AutoConfig.from_pretrained(hf_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
    model = model.to("cuda").eval()
    print(f"[{time.time()-t0:.1f}s] model ready")

    n_layers = config.num_hidden_layers
    mid = n_layers // 2
    sents = []
    for rec in c4_clean_v1(seed=stim_seed, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= n:
            break

    traj = extract_trajectory(
        model=model, tokenizer=tokenizer,
        texts=sents, layer_indices=[mid], pooling="seq_mean",
        device="cuda", system_key="qwen3-0.6b", class_id=1,
        quantization="fp16",
        stimulus_version=f"c4_clean.v1.seed{stim_seed}.n{n}",
        seed=stim_seed, batch_size=16, max_length=256,
    )
    X = traj.layers[0].X.astype(np.float32)
    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p, c0, r2 = fit_power_law(K_GRID, Cs)
    # Also check monotonicity
    diffs = np.diff(Cs)
    monotone = bool(np.all(diffs >= 0) or np.all(diffs <= 0))
    print(f"  C = {[round(c, 3) for c in Cs]}")
    print(f"  p={p:.3f}  c_0={c0:.3f}  R^2={r2:.4f}  monotone={monotone}")
    del model
    torch.cuda.empty_cache()
    return {"torch_seed": torch_seed, "k_grid": K_GRID, "C_values": Cs,
            "p_slope": p, "c_0": c0, "R2": r2, "monotone": monotone,
            "elapsed_s": time.time() - t0}


def main():
    results = []
    for ts in (42, 123, 456):
        results.append(run_one(torch_seed=ts))

    print("\n=== SUMMARY: Qwen3 UNTRAINED multi-seed ===")
    print(f"{'torch_seed':>12s} {'p':>8s} {'c_0':>8s} {'R^2':>8s} {'monotone':>10s}")
    r2s = []
    ps = []
    for r in results:
        r2s.append(r["R2"])
        ps.append(r["p_slope"])
        print(f"  {r['torch_seed']:12d} {r['p_slope']:8.3f} {r['c_0']:8.3f} "
              f"{r['R2']:8.4f} {str(r['monotone']):>10s}")

    r2_arr = np.array(r2s)
    p_arr = np.array(ps)
    print(f"\n  R^2 range: [{r2_arr.min():.3f}, {r2_arr.max():.3f}]")
    print(f"  p   range: [{p_arr.min():.3f}, {p_arr.max():.3f}]")
    if r2_arr.max() < 0.8:
        print(f"  verdict: U-shape is ROBUST — Qwen3 random init structurally fails power law")
    elif r2_arr.min() > 0.9:
        print(f"  verdict: genome_028 was a seed outlier — Qwen3 random init DOES produce power law")
    else:
        print(f"  verdict: seed-variable — Qwen3 random init is borderline")

    out = {
        "purpose": "Qwen3 UNTRAINED 3-seed robustness of genome_028 U-shape finding",
        "per_seed": results,
        "R2_range": [float(r2_arr.min()), float(r2_arr.max())],
        "p_range": [float(p_arr.min()), float(p_arr.max())],
    }
    out_path = _ROOT / "results/gate2/qwen3_untrained_seeds.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
