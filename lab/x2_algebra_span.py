"""X2 — THE PRODUCT-ALGEBRA SPAN PROBE (neighbor courtesy, INBOX note 6;
extends the x-series after x1's range census. Also a lab structure question:
is any trained layer a structurally special object under products?)

WHY (note 6, verbatim intent): "a cheap curiosity measurement is how many
significant singular values the span of {W, W^2, W^3, W^T W, ...} has at
depth 4-6 - a layer whose products stay in a low-dimensional span would be
structurally special." The lab's stake: the shape/height law says trained
objects carry conserved STRUCTURE; a small closed product algebra would be
the strongest structural statement a single matrix could make.

INSTRUMENT (pre-registered against a scale artifact): raw powers of a
trained matrix with spectral radius < 1 vanish geometrically, which would
fake low span dimension as a NORM artifact, not closure. So every product
is normalized to unit Frobenius norm BEFORE stacking: the span question
becomes one of DIRECTIONS. Controls: (a) random-init matrices at the SAME
measured std as each trained matrix (vanishing behavior matched); (b) a
POSITIVE control with minimal polynomial degree 2 (symmetric, two distinct
eigenvalues: powers stay in span{W, I} exactly) — the probe must see
closure when closure exists. The eigenvalue magnitudes |lambda| are
recorded alongside: direction-collapse at depth k is governed by
(|lambda_2|/|lambda_1|)^k, so the eigengap prior is part of the read.

MATERIAL: runs/checkpoints/e131_consolidated_e113.pt (2.74M, 6-layer,
d=192). Square objects: the six attn.c_proj (192x192) and the six MLP
block composites mlp.2 @ mlp.0 (192x192 — the map the block actually
computes, up to the nonlinearity between them).

BARS (frozen before compute):
  SPECIAL   — any trained square object whose depth-6 product span has
              effective dimension <= 4 of 8 at the 1e-3 relative
              threshold, while its matched-std random control sits at
              full rank (8 of 8) — a small closed algebra, named.
  GENERIC   — trained spans fill like their matched random controls
              (no small closed algebra in any trained layer); the
              honest negative recorded, with eigengap context.
  PARTIAL   — any object between (e.g. collapse in the composites but
              not the projections) — mapped verbatim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
torch.set_num_threads(4)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import run_dir, save_json

CKPT = Path(__file__).resolve().parent.parent / "runs/checkpoints/e131_consolidated_e113.pt"
THRESHOLDS = [1e-2, 1e-3, 1e-6]
PRODUCTS = ["W", "W^2", "W^3", "W^4", "W^5", "W^6", "W^T W", "W W^T"]


def product_family(W: torch.Tensor) -> torch.Tensor:
    """The 8 depth<=6 products, each unit-Frobenius-normalized (float64)."""
    mats = [W]
    P = W.clone()
    for _ in range(5):
        P = P @ W
        mats.append(P.clone())
    mats.append(W.T @ W)
    mats.append(W @ W.T)
    cols = []
    for M in mats:
        n = torch.linalg.norm(M.double())
        cols.append(M.double().reshape(-1) / n)
    return torch.stack(cols, dim=1)  # (n^2, 8)


def span_svd(W: torch.Tensor) -> dict:
    cols = product_family(W).numpy().astype(np.float64)
    # unit columns by construction; singular values of the (n^2, 8) stack
    s = np.linalg.svd(cols, compute_uv=False)
    rel = s / s[0]
    ev = torch.linalg.eigvals(W.double())
    mags = torch.sort(ev.abs(), descending=True).values.numpy()
    return {
        "spectrum_rel": rel.tolist(),
        "eff_dim": {f"{t:.0e}": int((rel > t).sum()) for t in THRESHOLDS},
        "eig_mags_top8": mags[:8].tolist(),
        "eig_ratio_2_1": float(mags[1] / mags[0]) if mags[0] > 0 else None,
    }


def main() -> None:
    rd = run_dir("x2")
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = blob["model"]

    objects: dict[str, torch.Tensor] = {}
    for i in range(6):
        objects[f"h.{i}.attn.c_proj"] = sd[f"h.{i}.attn.c_proj.weight"].float()
        objects[f"h.{i}.mlp_composite"] = (
            sd[f"h.{i}.mlp.2.weight"] @ sd[f"h.{i}.mlp.0.weight"]
        ).float()

    g = torch.Generator().manual_seed(0)
    # positive control: symmetric, exactly two distinct eigenvalues ->
    # minimal polynomial degree 2 -> every power in span{W, I}
    Q = torch.linalg.qr(torch.randn(192, 192, generator=g))[0]
    pos = Q @ torch.diag(torch.tensor([2.0] * 96 + [0.5] * 96)) @ Q.T
    objects["control_positive_2eig"] = pos

    results: dict[str, dict] = {}
    for name, W in objects.items():
        std = float(W.std())
        r = span_svd(W)
        if name.startswith("control_positive"):
            ctrl = None
        else:
            Wr = torch.randn(W.shape, generator=g) * std
            ctrl = span_svd(Wr)
        results[name] = {"std": std, "trained": r, "random_control": ctrl}
        save_json(rd / "metrics.json", {"experiment": "x2_product_algebra_span",
                                        "status": "PROGRESSIVE", "results": results})

    # adjudicate
    verdicts = {}
    for name, r in results.items():
        if name.startswith("control_positive"):
            d = r["trained"]["eff_dim"]["1e-03"]
            verdicts[name] = f"instrument check: eff_dim(1e-3)={d} (must be <=3 of 8 to see closure)"
            continue
        d_t = r["trained"]["eff_dim"]["1e-03"]
        d_r = r["random_control"]["eff_dim"]["1e-03"]
        verdicts[name] = f"trained {d_t}/8 vs random {d_r}/8 at 1e-3"
    special = [n for n, r in results.items()
               if not n.startswith("control_positive")
               and r["trained"]["eff_dim"]["1e-03"] <= 4
               and r["random_control"]["eff_dim"]["1e-03"] >= 7]
    verdict = "SPECIAL" if special else "GENERIC"
    if not special and any(
        r["trained"]["eff_dim"]["1e-03"] != r["random_control"]["eff_dim"]["1e-03"]
        for n, r in results.items() if not n.startswith("control_positive")
    ):
        verdict = "PARTIAL"

    save_json(rd / "metrics.json", {
        "experiment": "x2_product_algebra_span",
        "status": "DONE",
        "bars": ["SPECIAL", "GENERIC", "PARTIAL"],
        "verdict": verdict,
        "special_objects": special,
        "per_object_verdicts": verdicts,
        "instrument": "8 products depth<=6, unit-Frobenius-normalized, float64 SVD; "
                      "matched-std random controls; 2-eigenvalue positive control",
        "results": results,
        "checkpoint": str(CKPT),
    })

    # PNG: relative spectra
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, r in results.items():
        spec = r["trained"]["spectrum_rel"]
        sty = "--" if name.startswith("control_positive") else "-"
        ax.plot(range(1, 9), spec, sty, marker="o", ms=3, label=name)
        if r["random_control"]:
            ax.plot(range(1, 9), r["random_control"]["spectrum_rel"], ":",
                    marker="x", ms=3, alpha=0.5)
    ax.set_yscale("log")
    ax.axhline(1e-3, color="k", lw=0.6, ls="--", alpha=0.6)
    ax.set_xlabel("singular value index (of the 8-product span)")
    ax.set_ylabel("sigma / sigma_1")
    ax.set_title(f"X2 product-algebra span — e131 layers vs matched-random (dotted)\nverdict: {verdict}")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(rd / "span_spectra.png", dpi=140)
    print(json.dumps({"verdict": verdict, "special": special,
                      "positive_control": verdicts["control_positive_2eig"]}, indent=2))


if __name__ == "__main__":
    main()
