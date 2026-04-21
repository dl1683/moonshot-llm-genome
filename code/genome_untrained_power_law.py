"""Negative control: does the power law `C(X, k) = c_0 · k^p` also hold on
UNTRAINED (random-init) twins?

If YES + same exponent: the power law is architecture-driven, not learned-
geometry. Manifesto claim weakens.
If YES but different exponent: the TRAINING shapes the exponent.
If NO (power law doesn't fit untrained): clustering structure is a training
signature.

Tests 3 systems (Qwen3 / RWKV / DeepSeek) each at untrained=True and
untrained=False, mid-depth, n=1000 C4 seed 42, same log-spaced k-grid.

Per genome_loaders: untrained=True initializes random weights from the
same config → same architecture, no training.
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


def run_one(hf_id: str, system_key: str, untrained: bool, n: int, seed: int):
    tag = "UNTRAINED" if untrained else "TRAINED"
    print(f"\n=== {system_key} [{tag}] ===")
    t0 = time.time()
    sys_obj = load_system(hf_id, quant="fp16", untrained=untrained,
                          device="cuda")
    n_layers = sys_obj.n_hidden_layers()
    mid = n_layers // 2

    sents = []
    for rec in c4_clean_v1(seed=seed, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= n:
            break

    traj = extract_trajectory(
        model=sys_obj.model, tokenizer=sys_obj.tokenizer,
        texts=sents, layer_indices=[mid], pooling="seq_mean",
        device="cuda", system_key=system_key, class_id=0,
        quantization="fp16",
        stimulus_version=f"c4_clean.v1.seed{seed}.n{n}",
        seed=seed, batch_size=16, max_length=256,
    )
    X = traj.layers[0].X.astype(np.float32)

    Cs = []
    for k in K_GRID:
        if X.shape[0] <= k + 1:
            Cs.append(float("nan"))
            continue
        Cs.append(float(knn_clustering_coefficient(X, k=k).value))
    p_slope, c0, r2 = fit_power_law(K_GRID, Cs)
    print(f"  {tag} C(k=8..12)={np.interp(10, K_GRID, Cs):.3f}  p={p_slope:.3f}  c_0={c0:.3f}  R^2={r2:.4f}")

    sys_obj.unload()
    torch.cuda.empty_cache()

    return {
        "system_key": system_key, "hf_id": hf_id, "untrained": untrained,
        "depth_idx": mid, "n_stimuli": int(X.shape[0]),
        "hidden_dim": int(X.shape[1]),
        "k_grid": K_GRID, "C_values": Cs,
        "p_slope": p_slope, "c_0": c0, "R2": r2,
        "elapsed_s": time.time() - t0,
    }


def main():
    systems = [
        ("Qwen/Qwen3-0.6B", "qwen3-0.6b"),
        ("RWKV/rwkv-4-169m-pile", "rwkv-4-169m"),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
         "deepseek-r1-distill-qwen-1.5b"),
    ]
    results = []
    for hf_id, system_key in systems:
        for untrained in (False, True):
            try:
                results.append(run_one(hf_id, system_key, untrained,
                                        n=1000, seed=42))
            except Exception as e:
                print(f"FAILED on {system_key} untrained={untrained}: "
                      f"{type(e).__name__}: {e}")
                results.append({"system_key": system_key,
                                "untrained": untrained,
                                "error": f"{type(e).__name__}: {e}"})

    print("\n=== SUMMARY: TRAINED vs UNTRAINED POWER LAW ===")
    print(f"{'system':>32s} {'tag':>10s} {'p':>8s} {'c_0':>8s} {'R^2':>8s}")
    for r in results:
        if "error" in r:
            print(f"  {r['system_key']:>32s}  ERROR: {r['error']}")
            continue
        tag = "untrained" if r["untrained"] else "trained"
        print(f"  {r['system_key']:>32s} {tag:>10s} {r['p_slope']:8.3f} "
              f"{r['c_0']:8.3f} {r['R2']:8.4f}")

    # Deltas
    print("\nDeltas (trained - untrained):")
    paired = {}
    for r in results:
        if "error" in r:
            continue
        paired.setdefault(r["system_key"], {})[r["untrained"]] = r
    for sk, pair in paired.items():
        if True in pair and False in pair:
            t = pair[False]  # trained
            u = pair[True]   # untrained
            print(f"  {sk}: dp={t['p_slope']-u['p_slope']:+.3f} "
                  f"dc0={t['c_0']-u['c_0']:+.3f} dR2={t['R2']-u['R2']:+.4f}")

    out = {"purpose": "Power-law negative control: trained vs untrained",
           "per_cell": results}
    out_path = _ROOT / "results/gate2/untrained_power_law.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
