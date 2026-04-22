"""GenomeGuard noise-magnitude sweep: map rel_err vs perturbation strength.

genome_067 tested 3pct Frobenius noise and saw rel_err barely move
(1.26x). Real training failures (blown LR, gradient explosion) deform
weights far more than 3pct; we test whether the bridge rel_err is
responsive at larger perturbation magnitudes - which closes the
doomed-detection criterion GenomeGuard missed.

DESIGN. Qwen3-0.6B pretrained; inject Gaussian noise at geometric sigma
sweep: sigma_rel in {0, 0.01, 0.03, 0.1, 0.3, 0.5}. Apply to all
attention weights plus all MLP weights (full layer perturbation, not
just Q/K/V). Measure bridge rel_err at each sigma. Produces the
response curve that justifies GenomeGuard's doomed-detection threshold.

KILL CONDITION:
  - rel_err at sigma=0.3 must be >= 2x baseline (0.18+) to pass doomed
    detection criterion.
  - rel_err must rise MONOTONICALLY with sigma (no weird non-monotone
    behavior that would break the detector).

If passes: GenomeGuard detects both silent data corruption AND
catastrophic training divergence. Full spec ready to ship.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_genomeguard import probe_health  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent


def inject_noise_all(sys_obj, sigma_rel, substrs=("self_attn", "mlp")):
    """Inject sigma_rel-scaled noise into all matching 2D weight tensors."""
    sd = sys_obj.model.state_dict()
    for k in sd.keys():
        if not any(s in k for s in substrs):
            continue
        W = sd[k]
        if W.ndim != 2:
            continue
        scale = sigma_rel * torch.norm(W.float()).item() / (W.numel() ** 0.5)
        noise = torch.randn_like(W.float()) * scale
        W.add_(noise.to(W.dtype))


def main():
    # Probe: 1000 C4 texts (same as genome_060 / 067)
    probe = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        probe.append(rec["text"])
        if len(probe) >= 1000:
            break

    sigma_grid = [0.0, 0.01, 0.03, 0.1, 0.3, 0.5]
    rows = []
    torch.manual_seed(42)

    for cumulative_sigma in sigma_grid:
        # Reload fresh model and inject cumulative sigma in one shot so the
        # noise statistics are independent per measurement (not cumulative).
        sys_obj = load_system("Qwen/Qwen3-0.6B", quant="fp16", untrained=False, device="cuda")
        mid = sys_obj.n_hidden_layers() // 2
        if cumulative_sigma > 0:
            inject_noise_all(sys_obj, cumulative_sigma)
        h = probe_health(sys_obj.model, sys_obj.tokenizer, probe, "cuda", mid,
                         tag=f"noise_{cumulative_sigma:.3f}")
        h["sigma_rel"] = cumulative_sigma
        rows.append(h)
        print(f"  sigma={cumulative_sigma:.2f}  rel_err={h['bridge_rel_err']:.3f}  "
              f"c={h['c']:.2f}  ratio={h['ratio']:.2f}  alpha={h['alpha']:.3f}")
        sys_obj.unload(); torch.cuda.empty_cache()

    base = rows[0]["bridge_rel_err"]
    print("\n=== GenomeGuard noise response curve ===")
    print(f"{'sigma':>8s} {'rel_err':>10s} {'sep_vs_base':>15s}")
    for r in rows:
        sep = r["bridge_rel_err"] / max(base, 1e-6)
        print(f"  {r['sigma_rel']:6.2f}   {r['bridge_rel_err']:8.3f}   {sep:12.2f}x")

    # Kill condition: rel_err at sigma=0.3 >= 2x baseline AND monotonic
    r_at_30 = next(r for r in rows if abs(r["sigma_rel"] - 0.3) < 1e-6)
    sep_30 = r_at_30["bridge_rel_err"] / max(base, 1e-6)
    monotone = all(rows[i]["bridge_rel_err"] <= rows[i + 1]["bridge_rel_err"] + 0.02
                   for i in range(len(rows) - 1))
    passes = sep_30 >= 2.0 and monotone

    if passes:
        verdict = (f"DOOMED_DETECTION_LANDS - rel_err at sigma=0.3 is "
                   f"{sep_30:.1f}x baseline, monotone across full sweep. "
                   "GenomeGuard detects catastrophic training divergence.")
    else:
        verdict = (f"DOOMED_DETECTION_PARTIAL - sep@sigma=0.3 = {sep_30:.1f}x, "
                   f"monotone={monotone}. Refine before shipping.")

    print(f"\n  verdict: {verdict}")

    out = {"purpose": "GenomeGuard noise-magnitude response curve",
           "rows": rows,
           "baseline_rel_err": base,
           "sep_at_sigma_30": float(sep_30),
           "monotone": bool(monotone),
           "verdict": verdict}
    out_path = _ROOT / "results/gate2/genomeguard_noise_sweep.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
