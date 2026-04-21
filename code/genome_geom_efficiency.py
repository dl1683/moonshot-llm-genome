"""Strategic MINOR-ADJUSTMENT from 2026-04-21 Codex strategic check:
the Geometry → Efficiency probe.

Thesis: if C(X, k) = c_0 · k^p is a real cross-architecture coordinate AND
the manifesto claim (intelligence = geometry, not scale) has content, then
degrading a single model's geometry via weight compression should both
(a) perturb (c_0, p) away from the trained-network reference band, and
(b) correlate with capability degradation measured independently.

If Δ(c_0, p) predicts Δ(capability) → the coordinate is an early-warning
signal for compression: you quantize until geometry drifts, you stop, you
have quality. If no correlation → the coordinate is descriptive-only, not
useful for efficiency-gating.

Minimum experiment (this script):
  - Qwen3-0.6B loaded at FP16 and Q8 (extendable to Q4 if bnb supports).
  - For each quantization, extract activations on the same 500-stimulus C4
    batch, compute kNN-k clustering at k ∈ {3, 5, 10, 20, 30}, fit the
    (c_0, p) power law.
  - For each quantization, compute next-token NLL on the same batch as a
    capability proxy.
  - Report (quant, c_0, p, NLL) tuples + cross-quant deltas.

CPU for the fit, GPU for the extraction + NLL.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from stimulus_banks import c4_clean_v1  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from genome_extractor import extract_trajectory, sentinel_layer_indices  # noqa: E402
from genome_primitives import knn_clustering_coefficient  # noqa: E402


def fit_power_law(ks: np.ndarray, Cs: np.ndarray) -> dict:
    logk = np.log(ks)
    logC = np.log(Cs)
    slope, intercept = np.polyfit(logk, logC, 1)
    y_pred = slope * logk + intercept
    ss_res = np.sum((logC - y_pred) ** 2)
    ss_tot = np.sum((logC - logC.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return {"p_slope": float(slope), "c0": float(np.exp(intercept)), "R2": float(r2)}


def measure_nll(model, tokenizer, texts, device, max_length=128, batch_size=16):
    import torch
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attn == 0] = -100
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            ntok = (labels != -100).sum().item()
            total_nll += float(out.loss.item()) * ntok
            total_tokens += ntok
    return total_nll / max(total_tokens, 1), total_tokens


def run_quant_cell(quant: str, *, n: int, seed: int, max_length: int,
                   device: str = "cuda") -> dict:
    import torch
    print(f"\n=== QUANT={quant} ===")
    hf_id = "Qwen/Qwen3-0.6B"
    sys_obj = load_system(hf_id, quant=quant, untrained=False, device=device)
    print(f"  loaded {hf_id} at {quant}")

    texts = [it["text"] for it in c4_clean_v1(seed=seed, n_samples=n)]
    print(f"  {len(texts)} C4 stimuli loaded")

    # NLL baseline (capability proxy)
    nll_mean, ntok = measure_nll(sys_obj.model, sys_obj.tokenizer, texts,
                                  device=device, max_length=max_length)
    print(f"  NLL per token = {nll_mean:.4f}  (n_tokens={ntok})")

    # Extract at mid-depth only (L=28 → layer 14)
    n_layers = sys_obj.n_hidden_layers()
    mid_layer = n_layers // 2
    traj = extract_trajectory(
        sys_obj.model, sys_obj.tokenizer, texts,
        layer_indices=[mid_layer], pooling="seq_mean",
        max_length=max_length, device=device,
        system_key=sys_obj.system_key, class_id=sys_obj.class_id,
        quantization=quant,
        stimulus_version=f"c4_clean.len256.v1.seed{seed}.n{len(texts)}",
        seed=seed, batch_size=32,
    )
    X = traj.layers[0].X
    print(f"  mid-depth cloud shape: {X.shape}")

    # k-sweep
    k_grid = [3, 5, 10, 20, 30]
    Cs = []
    for k in k_grid:
        m = knn_clustering_coefficient(X, k=k)
        Cs.append(m.value)
        print(f"  kNN-{k:2d}  C = {m.value:.4f}")
    fit = fit_power_law(np.array(k_grid), np.array(Cs))
    print(f"  power-law fit: c_0 = {fit['c0']:.4f}, p = {fit['p_slope']:.4f}, R² = {fit['R2']:.4f}")

    sys_obj.unload()
    return {
        "quantization": quant,
        "nll_per_token": nll_mean,
        "k_grid": k_grid,
        "C_values": Cs,
        "c_0": fit["c0"],
        "p_slope": fit["p_slope"],
        "R2": fit["R2"],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--n-sentences", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--quants", type=str, nargs="+", default=["fp16", "q8"])
    args = ap.parse_args()

    t0 = time.time()
    results = []
    for q in args.quants:
        try:
            r = run_quant_cell(q, n=args.n_sentences, seed=args.seed,
                               max_length=args.max_length)
            results.append(r)
        except Exception as e:
            print(f"SKIP quant={q}: {type(e).__name__}: {e}")
            results.append({"quantization": q, "error": f"{type(e).__name__}: {e}"})

    print()
    print("=== SUMMARY: GEOMETRY → EFFICIENCY ===")
    print(f"{'quant':>8s} {'c_0':>10s} {'p':>10s} {'NLL':>10s} {'R²':>8s}")
    for r in results:
        if "error" in r:
            print(f"  {r['quantization']:>8s}  ERROR: {r['error']}")
            continue
        print(f"  {r['quantization']:>8s} {r['c_0']:10.4f} {r['p_slope']:10.4f} "
              f"{r['nll_per_token']:10.4f} {r['R2']:8.4f}")

    # Cross-quant deltas if 2+ successful
    ok = [r for r in results if "error" not in r]
    if len(ok) >= 2:
        base = ok[0]
        print()
        print("Deltas vs baseline (quant={}):".format(base["quantization"]))
        for r in ok[1:]:
            dC = r["c_0"] - base["c_0"]
            dp = r["p_slope"] - base["p_slope"]
            dNLL = r["nll_per_token"] - base["nll_per_token"]
            rel_dNLL = dNLL / base["nll_per_token"] * 100
            print(f"  {r['quantization']:>8s} -> Δc_0={dC:+.4f} Δp={dp:+.4f} "
                  f"ΔNLL={dNLL:+.4f} ({rel_dNLL:+.1f}%)")

    out = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
           "n_sentences": args.n_sentences, "seed": args.seed,
           "results": results,
           "total_wall_clock_seconds": round(time.time() - t0, 2)}
    out_path = _THIS_DIR.parent / "results" / "gate2" / "geom_efficiency.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nout: {out_path}")
