"""Genome_103: is the invariant universal at EVERY layer depth, or only at mid?

All prior invariant measurements used mid-depth only. This probe sweeps
every layer of 3 systems and computes sqrt(er)*alpha per layer. Then
reports CV across systems per-layer.

If tight (CV<10pct) at ALL layers → invariant is architecturally-intrinsic,
not a mid-depth sweet spot.
If tight only at mid, honest scope to 'mid-depth attractor'.
If tight for some LAYERS only (say layers 5-20 of 28), that's a new
universal - 'trained-LM mid-depth band' as a physical region.
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
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent

SYSTEMS = [
    ("qwen3-0.6b", "Qwen/Qwen3-0.6B"),
    ("deepseek-r1-distill-qwen-1.5b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("qwen3-1.7b", "Qwen/Qwen3-1.7B"),
]


def spectrum(X):
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False) / np.sqrt(max(X.shape[0] - 1, 1))
    return s.astype(np.float64)


def stats(s):
    s2 = s ** 2
    er = float(s2.sum() ** 2 / (s2 ** 2).sum()) if s2.sum() > 0 else 0.0
    h = len(s)
    r = np.arange(1, h + 1)
    lo, hi = max(1, int(h * 0.05)), int(h * 0.5)
    slope, _ = np.polyfit(np.log(r[lo:hi]), np.log(s[lo:hi] + 1e-12), 1)
    alpha = float(-slope)
    return {"eff_rank": er, "alpha": alpha,
            "sqrt_er_alpha": float(np.sqrt(er) * alpha)}


def main():
    t0 = time.time()
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=3000):
        sents.append(rec["text"])
        if len(sents) >= 800:
            break

    per_system_per_layer = {}
    for label, hf_id in SYSTEMS:
        print(f"\n[{time.time()-t0:.1f}s] ===== {label} =====")
        try:
            sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
        except Exception as e:
            print(f"  FAIL load: {e}"); continue
        n_layers = sys_obj.n_hidden_layers()
        layer_indices = list(range(n_layers))
        try:
            traj = extract_trajectory(
                model=sys_obj.model, tokenizer=sys_obj.tokenizer,
                texts=sents, layer_indices=layer_indices, pooling="seq_mean",
                device="cuda", system_key=label, class_id=1,
                quantization="fp16",
                stimulus_version="c4_clean.v1.seed42.n800",
                seed=42, batch_size=16, max_length=256,
            )
        except Exception as e:
            print(f"  FAIL extract: {e}")
            sys_obj.unload(); torch.cuda.empty_cache(); continue
        per_layer = []
        for L in layer_indices:
            X = traj.layers[L].X.astype(np.float32)
            s = spectrum(X)
            st = stats(s)
            per_layer.append({"layer": L, **st})
        sys_obj.unload(); torch.cuda.empty_cache()
        per_system_per_layer[label] = per_layer
        # Print summary at some depths
        for L in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
            st = per_layer[L]
            print(f"  layer {L:3d}/{n_layers}  eff_rank={st['eff_rank']:6.2f}  "
                  f"alpha={st['alpha']:.3f}  sqrt(er)*a={st['sqrt_er_alpha']:.3f}")

    # Across-system CV per normalized depth
    print(f"\n\n=== CV across systems at each normalized depth (0 = first, 1 = last) ===")
    n_bins = 10
    bin_stats = [[] for _ in range(n_bins)]
    for label, per_layer in per_system_per_layer.items():
        n_layers = len(per_layer)
        for i, entry in enumerate(per_layer):
            # Normalize depth: i / (n_layers-1) in [0, 1]
            frac = i / max(n_layers - 1, 1)
            b = min(int(frac * n_bins), n_bins - 1)
            bin_stats[b].append(entry["sqrt_er_alpha"])
    print(f"  {'bin':8s} {'range':12s} {'N':>3s} {'mean':>7s} {'std':>7s} {'CV%':>6s}")
    for b in range(n_bins):
        vals = bin_stats[b]
        if not vals: continue
        m, s = float(np.mean(vals)), float(np.std(vals))
        cv = 100*s/m if m else 0
        lo = b / n_bins; hi = (b + 1) / n_bins
        mark = "  <-- TIGHT" if 0 < cv < 10 else ("  <-- LOOSE" if cv > 15 else "")
        print(f"  {b:<8d} {f'[{lo:.1f},{hi:.1f}]':12s} {len(vals):>3d} {m:>7.3f} {s:>7.3f} {cv:>6.2f}%{mark}")

    out = {"per_system_per_layer": per_system_per_layer}
    out_path = _ROOT / "results/gate2/layer_depth_sweep.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
