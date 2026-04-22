"""Figure 4: training as convergence — trained vs untrained power-law
exponent p across 3 text architectures.

Two-panel figure:
  (a) per-system log-log C(k) curves, trained (solid) vs untrained (dashed)
  (b) exponent p on a number line showing trained cluster vs untrained spread

Data source: results/gate2/untrained_power_law.json
Writes: results/figures/genome_fig4_training_convergence.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_OUT = _ROOT / "results" / "figures"
_OUT.mkdir(parents=True, exist_ok=True)

SYSTEM_COLOR = {
    "qwen3-0.6b": "#1f77b4",
    "rwkv-4-169m": "#2ca02c",
    "deepseek-r1-distill-qwen-1.5b": "#ff7f0e",
}
SYSTEM_SHORT = {
    "qwen3-0.6b": "Qwen3-0.6B",
    "rwkv-4-169m": "RWKV-4-169M",
    "deepseek-r1-distill-qwen-1.5b": "DeepSeek-R1-Distill-1.5B",
}


def main():
    data = json.loads(
        (_ROOT / "results/gate2/untrained_power_law.json").read_text())
    by_system = {}
    for cell in data["per_cell"]:
        if "error" in cell:
            continue
        by_system.setdefault(cell["system_key"], {})[cell["untrained"]] = cell

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3),
                                    gridspec_kw={"width_ratios": [1.2, 1]})

    # Panel a: log-log C(k) curves per system, trained solid + untrained dashed
    for sk, pair in by_system.items():
        color = SYSTEM_COLOR.get(sk, "gray")
        label_base = SYSTEM_SHORT.get(sk, sk)
        if False in pair:
            t = pair[False]
            ax1.loglog(t["k_grid"], t["C_values"], "-o",
                       color=color, ms=4, lw=1.8,
                       label=f"{label_base} trained (p={t['p_slope']:.3f}, R²={t['R2']:.3f})")
        if True in pair:
            u = pair[True]
            ax1.loglog(u["k_grid"], u["C_values"], "--s",
                       color=color, ms=4, lw=1.5, alpha=0.75,
                       label=f"{label_base} random-init (p={u['p_slope']:.3f}, R²={u['R2']:.3f})")
    ax1.set_xlabel("kNN neighborhood size  $k$")
    ax1.set_ylabel(r"$C(X, k)$")
    ax1.set_title("(a)  $C(X, k)$  curves: trained vs random-init twin")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="lower right", fontsize=7, framealpha=0.92)

    # Panel b: number line showing p values, trained cluster vs untrained spread
    trained_ps = []
    untrained_ps = []
    labels = []
    for sk, pair in by_system.items():
        if False in pair:
            trained_ps.append(pair[False]["p_slope"])
        if True in pair:
            untrained_ps.append(pair[True]["p_slope"])
        labels.append(SYSTEM_SHORT.get(sk, sk))

    # Load multi-seed untrained data for error bars
    multi = _ROOT / "results/gate2/untrained_3seed_rwkv_deepseek.json"
    multi_seed = {}
    if multi.exists():
        mdata = json.loads(multi.read_text())
        summ = mdata.get("per_system_summary", {})
        for sk, s in summ.items():
            multi_seed[sk] = (s["p_mean"], s["p_std"])
    qmulti = _ROOT / "results/gate2/qwen3_untrained_seeds.json"
    if qmulti.exists():
        qd = json.loads(qmulti.read_text())
        ps = np.array([r["p_slope"] for r in qd.get("per_seed", [])])
        if ps.size:
            multi_seed["qwen3-0.6b"] = (float(ps.mean()), float(ps.std(ddof=1)))

    # Load trained 3-stim-seed data for trained error bars
    trained_multi = {}
    tmf = _ROOT / "results/gate2/trained_stim_seed_sweep_all.json"
    if tmf.exists():
        tdata = json.loads(tmf.read_text())
        for sk, s in tdata.get("per_system_summary", {}).items():
            trained_multi[sk] = (s["p_mean"], s["p_std"])
    q32 = _ROOT / "results/gate2/qwen3_trained_seed_sweep.json"
    if q32.exists():
        qd = json.loads(q32.read_text())
        trained_multi["qwen3-0.6b"] = (qd["p_mean"], qd["p_std"])

    # Horizontal strip plot: trained on one row, untrained on another
    for i, sk in enumerate(by_system):
        color = SYSTEM_COLOR.get(sk, "gray")
        if sk in trained_multi:
            pm, ps_ = trained_multi[sk]
            ax2.errorbar([pm], [1], xerr=[ps_], fmt="o", color=color,
                         ms=9, capsize=5, mec="black", mew=0.6, zorder=3,
                         lw=1.5)
            ax2.text(pm, 1.08, SYSTEM_SHORT.get(sk, sk)[:10], ha="center",
                     fontsize=7, color=color)
        elif False in by_system[sk]:
            p = by_system[sk][False]["p_slope"]
            ax2.scatter([p], [1], color=color, s=140, marker="o",
                        edgecolor="black", lw=0.6, zorder=3)
            ax2.text(p, 1.08, SYSTEM_SHORT.get(sk, sk)[:10], ha="center",
                     fontsize=7, color=color)
        if sk in multi_seed:
            # 3-seed mean + std error bar for untrained
            pm, ps_ = multi_seed[sk]
            ax2.errorbar([pm], [0], xerr=[ps_], fmt="s", color=color,
                         ms=9, capsize=5, mec="black", mew=0.6, zorder=3,
                         alpha=0.85, lw=1.5)
            ax2.text(pm, -0.12, SYSTEM_SHORT.get(sk, sk)[:10], ha="center",
                     fontsize=7, color=color)
        elif True in by_system[sk]:
            p = by_system[sk][True]["p_slope"]
            ax2.scatter([p], [0], color=color, s=140, marker="s",
                        edgecolor="black", lw=0.6, zorder=3, alpha=0.8)
            ax2.text(p, -0.12, SYSTEM_SHORT.get(sk, sk)[:10], ha="center",
                     fontsize=7, color=color)

    # Shaded trained-cluster band (mean ± std across full atlas 27 cells ≈ 0.179 ± 0.022)
    ax2.axvspan(0.179 - 0.022, 0.179 + 0.022, alpha=0.12, color="#888888",
                label=r"trained atlas band  $p=0.179\pm0.022$  (27 cells)")
    ax2.axvline(0.179, color="#555555", lw=0.6, ls=":", alpha=0.6)

    ax2.set_xlim(-0.05, 0.42)
    ax2.set_ylim(-0.4, 1.4)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["random-init", "trained"])
    ax2.set_xlabel(r"power-law exponent  $p$")
    ax2.set_title("(b)  trained $p$ cluster vs random-init $p$ spread")
    ax2.grid(True, axis="x", alpha=0.25)
    ax2.legend(loc="upper left", fontsize=7, framealpha=0.92)

    # Annotate the spread-factor
    t_arr = np.array(trained_ps)
    u_arr = np.array(untrained_ps)
    t_spread = (t_arr.max() - t_arr.min()) if t_arr.size > 1 else 0
    u_spread = (u_arr.max() - u_arr.min()) if u_arr.size > 1 else 0
    ax2.annotate("untrained 9-cell p range = 0.37\n"
                 "trained 3-system p range = 0.017\n"
                 r"compression ratio $\approx$ 22$\times$",
                 xy=(0.38, 1.0), xytext=(0.22, 0.55),
                 fontsize=7, ha="left",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fff8dc", ec="#888", alpha=0.9))

    fig.suptitle("Figure 4. Training is a convergence operation: "
                 r"random-init $p$ (9 cells, 3 seeds/sys) spans 22$\times$, trained $p$ collapses to 1.1$\times$",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = _OUT / "genome_fig4_training_convergence.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
