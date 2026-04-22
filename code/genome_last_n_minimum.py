"""Minimum sufficient atlas: how few last-layers carry the genome signal?

genome_079 showed layers 21-27 (last 7) recover 52.7pct of capability.
This script finds the minimum subset:
  last-1, last-2, last-3, last-5, last-7, last-10, last-14, all-28.

Cheapest data for the minimum-sufficient atlas framing. Partner demo
becomes "N KB of activations carries half a 600M-param model's
capability" - smaller N is a stronger claim.
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
from genome_capability_patch_k48 import (  # noqa: E402
    extract_mid_activations, lesion_midblock,
)
from genome_capability_patch_k48_v2 import MeanShiftHook  # noqa: E402
from genome_geometry_transfusion import measure_nll  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent


def run_with_patches(hf_id, sents, atlas, student_atlas, patch_layers, n_layers, label):
    sys_s = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    sd = sys_s.model.state_dict()
    for L in range(n_layers):
        lesion_midblock(sd, f"model.layers.{L}.")
    sys_s.model.load_state_dict(sd, strict=False)
    handles = []
    for L in patch_layers:
        shift = (atlas[L] - student_atlas[L]).astype(np.float32)
        hook = MeanShiftHook(shift)
        h = sys_s.model.model.layers[L].register_forward_hook(hook)
        handles.append(h)
    try:
        nll, _ = measure_nll(sys_s.model, sys_s.tokenizer, sents)
    except Exception as e:
        nll = float("nan"); print(f"  {label} ERR: {e}")
    for h in handles:
        h.remove()
    sys_s.unload(); torch.cuda.empty_cache()
    return float(nll)


def main():
    hf_id = "Qwen/Qwen3-0.6B"
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= 300:
            break
    t0 = time.time()

    sys_t = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    n_layers = sys_t.n_hidden_layers()
    nll_teacher, _ = measure_nll(sys_t.model, sys_t.tokenizer, sents)
    atlas = {L: extract_mid_activations(sys_t, sents, L, f"t{L}").mean(axis=0).astype(np.float32)
             for L in range(n_layers)}
    sys_t.unload(); torch.cuda.empty_cache()
    print(f"  teacher NLL={nll_teacher:.3f}")

    sys_s = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    sd = sys_s.model.state_dict()
    for L in range(n_layers):
        lesion_midblock(sd, f"model.layers.{L}.")
    sys_s.model.load_state_dict(sd, strict=False)
    nll_lesion, _ = measure_nll(sys_s.model, sys_s.tokenizer, sents)
    student_atlas = {L: extract_mid_activations(sys_s, sents, L, f"s{L}").mean(axis=0).astype(np.float32)
                      for L in range(n_layers)}
    sys_s.unload(); torch.cuda.empty_cache()
    print(f"  lesion NLL={nll_lesion:.3f}")

    gap = nll_lesion - nll_teacher
    ns = [1, 2, 3, 5, 7, 10, 14, 28]
    rows = []
    for n in ns:
        layers = list(range(n_layers - n, n_layers))
        print(f"\n-- last-{n} (layers {layers[0]}..{layers[-1]}) --")
        nll = run_with_patches(hf_id, sents, atlas, student_atlas, layers, n_layers, f"last-{n}")
        fg = (nll_lesion - nll) / max(gap, 1e-6)
        size_kb = n * 1024 * 4 / 1024
        rows.append({"regime": f"last-{n}", "n_patched": n,
                     "first_layer": layers[0], "last_layer": layers[-1],
                     "nll": nll, "fg_closed": float(fg), "atlas_kb": size_kb})
        print(f"  NLL={nll:.3f}  fg={fg:+.3f}  atlas_size={size_kb:.0f} KB")

    print("\n=== MINIMUM SUFFICIENT ATLAS (last-N sweep) ===")
    print(f"{'regime':<10s} {'n':>3s} {'nll':>7s} {'fg_closed':>10s} {'kb':>5s}")
    for r in rows:
        print(f"  {r['regime']:<10s} {r['n_patched']:3d} {r['nll']:6.3f}  {r['fg_closed']:+9.3f}  {r['atlas_kb']:4.0f}")

    # Find minimum n with fg >= 50pct
    min_n_50 = next((r["n_patched"] for r in rows if r["fg_closed"] >= 0.50), None)
    min_n_30 = next((r["n_patched"] for r in rows if r["fg_closed"] >= 0.30), None)
    min_n_15 = next((r["n_patched"] for r in rows if r["fg_closed"] >= 0.15), None)

    print(f"\n  min last-N with fg >=50pct: {min_n_50}  ({min_n_50*4 if min_n_50 else None} KB)")
    print(f"  min last-N with fg >=30pct: {min_n_30}  ({min_n_30*4 if min_n_30 else None} KB)")
    print(f"  min last-N with fg >=15pct: {min_n_15}  ({min_n_15*4 if min_n_15 else None} KB)")

    out = {"teacher_nll": nll_teacher, "lesion_nll": nll_lesion,
           "per_regime": rows,
           "min_n_50pct": min_n_50, "min_n_30pct": min_n_30, "min_n_15pct": min_n_15}
    out_path = _ROOT / "results/gate2/last_n_minimum.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
