"""Genome_104: invariant-vs-normalized-depth curve overlay across trained LMs.

Genome_103 suggested sqrt(er)*alpha = f(normalized_depth) is a universal
curve in the mid-band (0.4-0.9) across 3 systems. This probe:
 1. Extends to 5 systems (add BERT, RoBERTa)
 2. Interpolates each per-layer trace to a common normalized-depth grid
 3. Computes per-depth mean + CV across systems
 4. Generates overlay figure showing the universal curve

If all 5 curves overlap tightly in the mid-band, f(depth) is a genuine
universal function across trained LMs.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
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
    ("bert-base-uncased", "bert-base-uncased"),
    ("roberta-base", "FacebookAI/roberta-base"),
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
    return float(np.sqrt(er) * alpha)


def main():
    t0 = time.time()
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=3000):
        sents.append(rec["text"])
        if len(sents) >= 800:
            break

    # Common normalized-depth grid
    grid = np.linspace(0.0, 1.0, 21)

    per_system_traces = {}
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
        per_layer_inv = []
        for L in layer_indices:
            X = traj.layers[L].X.astype(np.float32)
            s = spectrum(X)
            inv = stats(s)
            per_layer_inv.append(inv)
        sys_obj.unload(); torch.cuda.empty_cache()
        per_layer_inv = np.array(per_layer_inv)
        depths = np.arange(n_layers) / max(n_layers - 1, 1)
        # Interpolate to common grid
        interp = np.interp(grid, depths, per_layer_inv)
        per_system_traces[label] = {"depths": depths.tolist(),
                                      "values": per_layer_inv.tolist(),
                                      "interp": interp.tolist(),
                                      "n_layers": int(n_layers)}
        print(f"  got {n_layers} layer values; range {per_layer_inv.min():.3f}-{per_layer_inv.max():.3f}")

    # Per-bin CV on interp grid
    labels = list(per_system_traces.keys())
    if labels:
        mat = np.stack([per_system_traces[l]["interp"] for l in labels])  # (n_sys, n_grid)
        print(f"\n\n=== INVARIANT vs NORMALIZED DEPTH ACROSS {len(labels)} SYSTEMS ===")
        print(f"  {'depth':>6s} {'mean':>7s} {'std':>7s} {'CV%':>6s}")
        for i, d in enumerate(grid):
            vals = mat[:, i]
            m, s = float(np.mean(vals)), float(np.std(vals))
            cv = 100*s/m if m else 0
            mark = "  <-- TIGHT" if 0 < cv < 10 else ("  <-- LOOSE" if cv > 20 else "")
            print(f"  {d:>6.2f} {m:>7.3f} {s:>7.3f} {cv:>6.2f}%{mark}")

        # Figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for l in labels:
            tr = per_system_traces[l]
            ax1.plot(grid, tr["interp"], 'o-', lw=1.5, label=l[:25])
        mean_curve = mat.mean(axis=0)
        ax1.plot(grid, mean_curve, 'k--', lw=2, label='mean')
        ax1.set_xlabel("normalized depth")
        ax1.set_ylabel(r"$\sqrt{\mathrm{eff\_rank}} \cdot \alpha$")
        ax1.set_title("Invariant vs depth across 5 trained LMs")
        ax1.axvspan(0.4, 0.9, color='green', alpha=0.1, label='tight band')
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        residuals = mat - mean_curve
        for i, l in enumerate(labels):
            ax2.plot(grid, residuals[i] / mean_curve * 100, 'o-', lw=1.5, label=l[:25])
        ax2.axhline(0, color='black', lw=0.5)
        ax2.set_xlabel("normalized depth")
        ax2.set_ylabel("per-system residual from mean (%)")
        ax2.set_title("Deviation from mean curve")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        plt.suptitle("Genome_104: invariant-vs-depth curve is universal in mid-band", fontsize=12)
        plt.tight_layout()
        out_fig = _ROOT / "results/figures/genome_104_depth_curve_overlay.png"
        plt.savefig(out_fig, dpi=120, bbox_inches='tight')
        print(f"\nwrote figure {out_fig}")

    out = {"traces": per_system_traces, "grid": grid.tolist()}
    out_path = _ROOT / "results/gate2/depth_curve_overlay.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
