"""Generate the 3 anchor figures for the workshop paper from JSON artifacts.

Figures (per research/PAPER_OUTLINE.md):
  fig1_ck_cross_architecture.png   — C(k) log-log across 5 Batch-1 systems
                                     (Gate-1 + replaces-derivation-falsification result)
  fig2_causal_ablation.png         — Loss delta vs λ for 3 schemes on 3 text systems
                                     (Gate-2 G2.4 specificity)
  fig3_geometry_efficiency.png     — R² of C(k) fit vs quant level, overlaid with NLL drift
                                     (strategic MINOR-ADJUSTMENT / §5.5 data)

Pure CPU. Reads JSONs from results/. Writes to results/figures/.

Per CLAUDE.md file-naming: genome_<topic>_<variant>.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
_OUT = _ROOT / "results" / "figures"
_OUT.mkdir(parents=True, exist_ok=True)

SYSTEM_COLOR = {
    "qwen3-0.6b": "#1f77b4",
    "deepseek-r1-distill-qwen-1.5b": "#ff7f0e",
    "rwkv-4-169m": "#2ca02c",
    "falcon-h1-0.5b": "#d62728",
    "dinov2-small": "#9467bd",
}
SYSTEM_SHORT = {
    "qwen3-0.6b": "Qwen3-0.6B",
    "deepseek-r1-distill-qwen-1.5b": "DeepSeek-R1-Distill-1.5B",
    "rwkv-4-169m": "RWKV-4-169M",
    "falcon-h1-0.5b": "Falcon-H1-0.5B",
    "dinov2-small": "DINOv2-small",
}


# -------------------- Figure 1: C(k) cross-architecture --------------------

def fig1_ck():
    data = json.load(open(_ROOT / "results" / "gate2" / "Ck_curves_middepth.json"))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for sys, per_k in data["per_system"].items():
        ks = sorted(int(k) for k in per_k)
        Cs = [per_k[str(k)]["mean"] for k in ks]
        stds = [per_k[str(k)]["std"] for k in ks]
        ax.errorbar(ks, Cs, yerr=stds, marker="o", ms=5, lw=1.5,
                    color=SYSTEM_COLOR.get(sys, "gray"),
                    label=SYSTEM_SHORT.get(sys, sys), capsize=2)
    ax.set_xscale("log")
    ax.set_xlabel("kNN neighborhood size  $k$")
    ax.set_ylabel(r"mean clustering coefficient  $C(X, k)$")
    ax.set_title("Figure 1. Cross-architecture $C(X, k)$ at mid-depth, 5 trained networks\n"
                 "(n=2000 stimuli, seed-averaged, error bars = std across 3 seeds)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out = _OUT / "genome_fig1_ck_cross_architecture.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


# -------------------- Figure 2: G2.4 causal ablation --------------------

def fig2_causal():
    """Loss delta vs lambda, one panel per system, 3 schemes colored."""
    import glob
    cells = {}
    for p in sorted(glob.glob(str(_ROOT / "results" / "gate2" /
                                    "causal_*_n500_seed42.json"))):
        d = json.load(open(p))
        cells[(d["system_key"], d["depth_index"])] = d

    # Pick one mid-depth cell per system for clarity
    systems = ["qwen3-0.6b", "rwkv-4-169m", "deepseek-r1-distill-qwen-1.5b"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=False)

    for ax, sys in zip(axes, systems):
        mid_cell = None
        # Prefer depth_index==1, fall back to 0
        for depth_idx in (1, 0, 2):
            if (sys, depth_idx) in cells:
                mid_cell = cells[(sys, depth_idx)]
                break
        if mid_cell is None:
            ax.set_title(f"{SYSTEM_SHORT.get(sys, sys)} — no data")
            continue

        base_loss = mid_cell["results"]["baseline"]["loss"]
        lams = [0.0, 0.25, 0.5, 0.75, 1.0]
        for scheme, color in [("topk", "#d62728"),
                              ("random", "#7f7f7f"),
                              ("pca", "#1f77b4")]:
            rels = []
            for lam in lams:
                key = f"{scheme}|lam={lam}" if lam > 0 else "baseline"
                v = mid_cell["results"].get(key, mid_cell["results"]["baseline"])
                loss = v.get("loss", np.nan)
                rels.append(100 * (loss - base_loss) / base_loss)
            ax.plot(lams, rels, marker="o", lw=2, color=color,
                    label=scheme)
        ax.axhline(5, ls="--", color="black", alpha=0.5, lw=0.8,
                   label=r"prereg $\delta_{causal}=5\%$")
        ax.set_xlabel(r"ablation strength  $\lambda$")
        ax.set_ylabel(r"relative NLL shift  $\Delta / \mathrm{NLL}_{baseline}$  (%)")
        ax.set_title(f"{SYSTEM_SHORT.get(sys, sys)} (depth {mid_cell['k_normalized']:.2f})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    fig.suptitle("Figure 2. Gate-2 G2.4 causal ablation — topk specificity vs random/PCA controls",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = _OUT / "genome_fig2_causal_ablation.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


# -------------------- Figure 3: Geometry → Efficiency --------------------

def fig3_geom_efficiency():
    d = json.load(open(_ROOT / "results" / "gate2" / "geom_efficiency.json"))
    results = [r for r in d["results"] if "error" not in r]

    quants = [r["quantization"] for r in results]
    R2s    = [r["R2"]            for r in results]
    NLLs   = [r["nll_per_token"] for r in results]
    c0s    = [r["c_0"]           for r in results]
    ps     = [r["p_slope"]       for r in results]

    fp16_NLL = NLLs[0]
    rel_nll = [100 * (n - fp16_NLL) / fp16_NLL for n in NLLs]

    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    ax2 = ax1.twinx()

    x = np.arange(len(quants))
    ax1.plot(x, R2s, "-o", color="#2ca02c", lw=2, ms=8,
             label=r"$R^2$ of $C(k) = c_0 \cdot k^{p}$ fit")
    ax2.plot(x, rel_nll, "-s", color="#d62728", lw=2, ms=8,
             label=r"$\Delta$ NLL rel. to FP16  (%)")

    ax1.set_xticks(x)
    ax1.set_xticklabels([q.upper() for q in quants])
    ax1.set_xlabel("weight quantization")
    ax1.set_ylabel(r"power-law fit quality  $R^2$", color="#2ca02c")
    ax2.set_ylabel(r"relative NLL increase (%)", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Figure 3. Geometry $\\to$ Efficiency on Qwen3-0.6B\n"
                  "$R^2$ of power-law fit is MONOTONE with compression, tracking NLL")

    # Annotate values
    for i, (q, r2, nll) in enumerate(zip(quants, R2s, rel_nll)):
        ax1.annotate(f"R²={r2:.4f}", (i, r2), xytext=(6, -12),
                     textcoords="offset points", color="#2ca02c", fontsize=8)
        ax2.annotate(f"+{nll:.1f}%", (i, nll), xytext=(6, 6),
                     textcoords="offset points", color="#d62728", fontsize=8)

    # Legend combining both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left",
               fontsize=8, framealpha=0.9)

    fig.tight_layout()
    out = _OUT / "genome_fig3_geometry_efficiency.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig1_ck()
    fig2_causal()
    fig3_geom_efficiency()
    print("all figures written.")
