"""Overlay the normalized-variance CDFs of 5 trained ML systems.

If genome_096's universality claim is right, these 5 curves should be
literally indistinguishable on the CDF axis — differing by less than 1%
at q25, 0.42% at q50, 0.13% at q75.

One figure: curves overlaid on a single axis (normalized rank vs CDF).
Plus a residual-vs-mean panel showing how tight the spread really is.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent


def main():
    d = json.load(open(_ROOT / "results/gate2/spectrum_dump_analysis.json"))
    spectra = d["spectra"]  # {sys: [s_1, s_2, ..., s_h]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Build common normalized-rank grid for interpolation
    rank_grid = np.linspace(0.01, 1.0, 200)
    cdfs = {}
    for sys_key, s_list in spectra.items():
        s = np.array(s_list, dtype=np.float64)
        s2 = s ** 2
        norm = s2 / s2.sum()
        cum = np.cumsum(norm)
        h = len(s)
        ranks = np.arange(1, h + 1) / h
        # Interpolate onto common grid
        cdfs[sys_key] = np.interp(rank_grid, ranks, cum)

    # Plot individual CDFs
    for sys_key, curve in cdfs.items():
        ax1.plot(rank_grid, curve, lw=1.5, label=sys_key)
    ax1.axhline(0.984, color='red', ls=':', lw=1, alpha=0.5, label='q50 universal ≈ 0.984')
    ax1.axhline(0.932, color='orange', ls=':', lw=1, alpha=0.5, label='q25 universal ≈ 0.932')
    ax1.axhline(0.997, color='purple', ls=':', lw=1, alpha=0.5, label='q75 universal ≈ 0.997')
    ax1.axvline(0.5, color='gray', ls='--', lw=0.5, alpha=0.5)
    ax1.axvline(0.25, color='gray', ls='--', lw=0.5, alpha=0.5)
    ax1.axvline(0.75, color='gray', ls='--', lw=0.5, alpha=0.5)
    ax1.set_xlabel('normalized rank (i / h)')
    ax1.set_ylabel('cumulative variance fraction')
    ax1.set_title('Normalized-variance CDFs across 5 trained ML systems')
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.01)

    # Residuals from mean CDF
    all_curves = np.stack(list(cdfs.values()))
    mean_curve = all_curves.mean(axis=0)
    for sys_key, curve in cdfs.items():
        ax2.plot(rank_grid, (curve - mean_curve) * 100, lw=1.5, label=sys_key)
    ax2.axhline(0, color='black', lw=0.5)
    ax2.set_xlabel('normalized rank (i / h)')
    ax2.set_ylabel('CDF residual (%) from 5-system mean')
    ax2.set_title('Per-system deviation from universal CDF')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 1)

    plt.suptitle('The universal CDF of trained ML activation spectra (genome_096)', fontsize=13)
    plt.tight_layout()
    out = _ROOT / "results/figures/genome_096_universal_cdf.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
