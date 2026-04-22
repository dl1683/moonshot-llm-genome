"""Figure: genome_093 shows aux loss controls spectrum but NOT capability.

Four panels: eff_rank, alpha, sqrt(er)*alpha, NLL — each control vs aux.
Key story: aux drives all 3 spectral measures toward teacher; NLL is NOT
improved (in fact slightly worse).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent


def main():
    data = json.load(open(_ROOT / "results/gate2/buffered_aux_recovery.json"))
    teacher = data["teacher_invariant"]
    tnll = data["teacher_nll"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for name, color, key in [("control (γ=0)", "C1", "control"),
                              ("aux buffered (γ=1e-2)", "C0", "aux")]:
        log = data[key]["log"]
        steps = [e["step"] for e in log]
        er = [e["eff_rank"] for e in log]
        a = [e["alpha"] for e in log]
        inv = [e["sqrt_er_alpha"] for e in log]
        nll = [e["val_nll"] for e in log]

        axes[0,0].plot(steps, er, 'o-', color=color, lw=2, ms=7, label=name)
        axes[0,1].plot(steps, a, 'o-', color=color, lw=2, ms=7, label=name)
        axes[1,0].plot(steps, inv, 'o-', color=color, lw=2, ms=7, label=name)
        axes[1,1].plot(steps, nll, 'o-', color=color, lw=2, ms=7, label=name)

    axes[0,0].axhline(teacher["eff_rank"], color='green', ls='--', label=f'teacher ({teacher["eff_rank"]:.1f})')
    axes[0,0].set_ylabel('eff_rank'); axes[0,0].set_title('Effective rank (mode diversity)')
    axes[0,0].set_xlabel('step'); axes[0,0].grid(alpha=0.3); axes[0,0].legend()

    axes[0,1].axhline(teacher["alpha"], color='green', ls='--', label=f'teacher ({teacher["alpha"]:.2f})')
    axes[0,1].set_ylabel(r'$\alpha$ (spectrum tail slope)')
    axes[0,1].set_title('Power-law tail slope')
    axes[0,1].set_xlabel('step'); axes[0,1].grid(alpha=0.3); axes[0,1].legend()

    axes[1,0].axhline(teacher["sqrt_er_alpha"], color='green', ls='--', label=f'teacher ({teacher["sqrt_er_alpha"]:.2f})')
    axes[1,0].set_ylabel(r'$\sqrt{\mathrm{eff\_rank}}\cdot\alpha$')
    axes[1,0].set_title('Invariant (trained attractor ≈ 3√2 at large n)')
    axes[1,0].set_xlabel('step'); axes[1,0].grid(alpha=0.3); axes[1,0].legend()

    axes[1,1].axhline(tnll, color='green', ls='--', label=f'teacher ({tnll:.2f})')
    axes[1,1].set_ylabel('val NLL')
    axes[1,1].set_title('Language-model NLL (capability)')
    axes[1,1].set_xlabel('step'); axes[1,1].grid(alpha=0.3); axes[1,1].legend()

    plt.suptitle('genome_093: aux loss controls spectrum (top row) but does NOT improve NLL (bottom-right)',
                 fontsize=12)
    plt.tight_layout()
    out_path = _ROOT / "results/figures/genome_093_aux_spectrum_knob.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
