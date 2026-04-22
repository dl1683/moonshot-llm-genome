"""Generate trajectory figure for genome_089 invariant tracking.

Two-panel plot:
  left: eff_rank vs training step (with teacher line + lesion line)
  right: sqrt(eff_rank)*alpha vs training step
  both: annotate rep-count and teacher target
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent


def main():
    data = json.load(open(_ROOT / "results/gate2/invariant_trajectory.json"))
    traj = data["trajectory"]
    teacher = data["teacher_invariant"]

    steps = [t["step"] for t in traj]
    er = [t["eff_rank"] for t in traj]
    inv = [t["sqrt_er_alpha"] for t in traj]
    rep = [t["rep"] for t in traj]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(steps, er, 'o-', color='C0', lw=2, markersize=8, label='student')
    ax1.axhline(teacher["eff_rank"], color='green', ls='--', lw=1.5, label=f'teacher ({teacher["eff_rank"]:.1f})')
    ax1.set_xlabel('training step')
    ax1.set_ylabel('eff_rank (mid-depth activations)')
    ax1.set_title('Mode-diversity trajectory during lesion recovery')
    ax1.grid(alpha=0.3)
    for s, e, r in zip(steps, er, rep):
        ax1.annotate(f'rep={r}/5', (s, e), xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=8, color='C3' if r >= 3 else 'C2')
    ax1.legend()

    ax2.plot(steps, inv, 'o-', color='C1', lw=2, markersize=8, label='student')
    ax2.axhline(teacher["sqrt_er_alpha"], color='green', ls='--', lw=1.5,
                label=f'teacher ({teacher["sqrt_er_alpha"]:.2f})')
    ax2.axhline(4.243, color='red', ls=':', lw=1, label='3√2 = 4.243 (n=800 attractor)')
    ax2.set_xlabel('training step')
    ax2.set_ylabel(r'$\sqrt{\mathrm{eff\_rank}} \cdot \alpha$')
    ax2.set_title('Spectral invariant trajectory')
    ax2.grid(alpha=0.3)
    for s, v, r in zip(steps, inv, rep):
        ax2.annotate(f'rep={r}/5', (s, v), xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=8, color='C3' if r >= 3 else 'C2')
    ax2.legend()

    plt.suptitle('Capability recovery = mode-diversity recovery (genome_089)', fontsize=13)
    plt.tight_layout()
    out_path = _ROOT / "results/figures/genome_089_invariant_trajectory.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
