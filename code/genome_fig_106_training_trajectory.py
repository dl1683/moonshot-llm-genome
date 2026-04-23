"""Figure: invariant convergence during Pythia-410m training.

X = log(training_step + 1), Y = sqrt(er)*alpha. Expect monotone-ish
convergence from random-init-like 9+ toward trained attractor 4.27.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent


def main():
    data = json.load(open(_ROOT / "results/gate2/pythia_training_trajectory.json"))
    rows = data["rows"]
    # Parse step count from revision string "step{N}"
    xs = []
    ys = []
    labels = []
    for r in rows:
        rev = r["revision"]
        step = int(rev.replace("step", ""))
        xs.append(max(step, 1))
        ys.append(r["sqrt_er_alpha"])
        labels.append(rev)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, ys, 'o-', lw=2, ms=10, color='C0', label='Pythia-410m')
    for x, y, l in zip(xs, ys, labels):
        ax.annotate(f"{l}\n{y:.2f}", (x, y), xytext=(8, 5),
                    textcoords='offset points', fontsize=8)
    ax.axhline(data.get("attractor_ref", 4.27), color='green', ls='--', lw=1.5,
               label='trained attractor (3√2 ≈ 4.24)')
    ax.axhline(7.49, color='red', ls=':', lw=1, alpha=0.6,
               label='random-init mean (genome_097)')
    ax.set_xscale("log")
    ax.set_xlabel("training step (log scale)")
    ax.set_ylabel(r"$\sqrt{\mathrm{eff\_rank}}\cdot\alpha$")
    ax.set_title("Invariant convergence during Pythia-410m training")
    ax.grid(alpha=0.3)
    ax.legend()
    out = _ROOT / "results/figures/genome_106_pythia_trajectory.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
