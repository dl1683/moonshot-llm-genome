"""Figure: control vs aux A/B comparison for genome_090.

Two panels side by side: eff_rank trajectory and rep-count trajectory,
control (orange) vs aux (blue). Annotate teacher targets.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent


def main():
    data = json.load(open(_ROOT / "results/gate2/geometry_aux_recovery.json"))
    teacher = data["teacher_invariant"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    for name, color, key in [("control (γ=0)", "C1", "control"),
                              ("aux (γ=1e-3)", "C0", "aux")]:
        log = data[key]["log"]
        steps = [e["step"] for e in log]
        er = [e["eff_rank"] for e in log]
        inv = [e["sqrt_er_alpha"] for e in log]
        rep = [e["rep"] for e in log]
        nll = [e["val_nll"] for e in log]
        ax1.plot(steps, er, 'o-', color=color, lw=2, markersize=8, label=name)
        ax2.plot(steps, inv, 'o-', color=color, lw=2, markersize=8, label=name)
        ax3.plot(steps, nll, 'o-', color=color, lw=2, markersize=8, label=name)

    ax1.axhline(teacher["eff_rank"], color='green', ls='--', lw=1.5, label=f'teacher')
    ax1.set_xlabel('training step'); ax1.set_ylabel('eff_rank')
    ax1.set_title('Mode diversity recovery')
    ax1.grid(alpha=0.3); ax1.legend()

    ax2.axhline(teacher["sqrt_er_alpha"], color='green', ls='--', lw=1.5, label='teacher')
    ax2.set_xlabel('training step'); ax2.set_ylabel(r'$\sqrt{\mathrm{eff\_rank}}\cdot\alpha$')
    ax2.set_title('Spectral invariant trajectory')
    ax2.grid(alpha=0.3); ax2.legend()

    ax3.axhline(data["teacher_nll"], color='green', ls='--', lw=1.5, label='teacher')
    ax3.set_xlabel('training step'); ax3.set_ylabel('val NLL')
    ax3.set_title('Language-model NLL')
    ax3.grid(alpha=0.3); ax3.legend()

    plt.suptitle('Geometry-aux-loss A/B test: does matching teacher mode diversity accelerate recovery?',
                 fontsize=12)
    plt.tight_layout()
    out_path = _ROOT / "results/figures/genome_090_aux_recovery_ab.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
