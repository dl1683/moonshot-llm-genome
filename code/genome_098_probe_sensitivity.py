"""Adversarial #2: measure-choice sensitivity of the invariant.

Codex flag: alpha fit depends on window [lo, hi] for the log-log slope.
eff_rank depends on centering. Pooling, layer choice, truncation all vary.
If sqrt(er)*alpha CV balloons from 5% to >15% under natural variations,
it's a probe-specific number, not an intrinsic law.

This is CPU-only: operates on the raw spectra saved in
results/gate2/spectrum_dump_analysis.json (which has {system: [s_1, ..., s_h]}).

Variations tested:
  A. fit window: (5%,50%), (10%,40%), (2%,30%), (5%,25%), (10%,75%), (20%,80%)
  B. truncation: use full spectrum vs. top-512 vs. top-256
  C. squared-vs-unsquared slope definition (σ vs σ²)

For each variation, compute CV across 5 systems. Report variations
where CV > 10%.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent


def fit_alpha(s, lo_frac, hi_frac):
    h = len(s)
    lo = max(1, int(h * lo_frac))
    hi = max(lo + 1, int(h * hi_frac))
    r = np.arange(1, h + 1)
    slope, _ = np.polyfit(np.log(r[lo:hi]), np.log(s[lo:hi] + 1e-20), 1)
    return float(-slope)


def compute_invariant(s, lo=0.05, hi=0.5, truncate_to=None, use_sigma2=False):
    if truncate_to is not None:
        s = s[:truncate_to]
    s2 = s ** 2
    er = float(s2.sum() ** 2 / (s2 ** 2).sum()) if s2.sum() > 0 else 0.0
    alpha = fit_alpha(s, lo, hi)
    if use_sigma2:
        alpha = alpha * 2  # slope of sigma^2 is 2x slope of sigma
    return {"eff_rank": er, "alpha": alpha,
            "sqrt_er_alpha": float(np.sqrt(er) * alpha),
            "er_alpha2": float(er * alpha ** 2)}


def main():
    d = json.load(open(_ROOT / "results/gate2/spectrum_dump_analysis.json"))
    spectra = {k: np.array(v, dtype=np.float64) for k, v in d["spectra"].items()}

    variations = [
        ("baseline [5%,50%]",      dict(lo=0.05, hi=0.50)),
        ("narrow [10%,40%]",       dict(lo=0.10, hi=0.40)),
        ("early [2%,30%]",         dict(lo=0.02, hi=0.30)),
        ("early-narrow [5%,25%]",  dict(lo=0.05, hi=0.25)),
        ("wide [10%,75%]",         dict(lo=0.10, hi=0.75)),
        ("late [20%,80%]",         dict(lo=0.20, hi=0.80)),
        ("very-late [30%,95%]",    dict(lo=0.30, hi=0.95)),
        ("top-512 truncation",     dict(lo=0.05, hi=0.50, truncate_to=512)),
        ("top-256 truncation",     dict(lo=0.05, hi=0.50, truncate_to=256)),
        ("top-128 truncation",     dict(lo=0.05, hi=0.50, truncate_to=128)),
    ]

    print(f"{'variation':35s} {'mean':>7s} {'std':>7s} {'CV%':>6s}   {'er_a2 mean':>10s} {'CV%':>6s}")
    for name, kwargs in variations:
        invs = []
        er_a2s = []
        for sys_key, s in spectra.items():
            result = compute_invariant(s, **kwargs)
            invs.append(result["sqrt_er_alpha"])
            er_a2s.append(result["er_alpha2"])
        m = np.mean(invs); sd = np.std(invs)
        cv = 100 * sd / m if m != 0 else 0.0
        m2 = np.mean(er_a2s); sd2 = np.std(er_a2s)
        cv2 = 100 * sd2 / m2 if m2 != 0 else 0.0
        mark = "  <-- TIGHT" if 0 < cv < 7 else ("  <-- LOOSE" if cv > 15 else "")
        print(f"{name:35s} {m:>7.3f} {sd:>7.3f} {cv:>6.2f}%  {m2:>10.3f} {cv2:>6.2f}%{mark}")

    # Also compute "per-system" table for the baseline to compare
    print(f"\nPer-system (baseline [5%,50%]):")
    for sys_key, s in spectra.items():
        r = compute_invariant(s)
        print(f"  {sys_key:30s}  sqrt(er)*a={r['sqrt_er_alpha']:.3f}  er*a^2={r['er_alpha2']:.3f}")


if __name__ == "__main__":
    main()
