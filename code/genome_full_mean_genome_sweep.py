"""Neural Genome sweep: fraction-of-layers-patched vs capability recovery.

genome_078 showed:
  Full 28-layer atlas: fg_closed = 0.49
  Half atlas (first 14): fg_closed = -0.01

Linear interpolation suggests the recovery is NOT linear in fraction
patched. This script tests 5 patching regimes to characterize the
recovery curve and answer: which layers carry the capability?

Regimes (lesion ALL 28 layers; vary WHICH layers are patched):
  (A) none      — no patch (baseline lesion NLL)
  (B) first-7   — patch layers 0..6
  (C) last-7    — patch layers 21..27
  (D) first-14  — patch layers 0..13 (genome_078 half)
  (E) last-14   — patch layers 14..27
  (F) all-28    — patch layers 0..27 (genome_078 full)
  (G) alternate — patch even-indexed layers only (14 layers)
  (H) mid-14    — patch layers 7..20

Tells us whether:
 - The genome recovery depends symmetrically on position
 - Front-loaded or back-loaded or middle-loaded
 - Coverage-density matters vs specific layer positions
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


def run_with_patches(hf_id, sents, atlas, student_atlas, patch_layers, n_layers,
                     label):
    """Reload fully-lesioned student and apply mean-shift hooks on the
    specified subset of layers. Return NLL."""
    sys_s = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    sd = sys_s.model.state_dict()
    for layer_idx in range(n_layers):
        lesion_midblock(sd, f"model.layers.{layer_idx}.")
    sys_s.model.load_state_dict(sd, strict=False)
    handles = []
    for layer_idx in patch_layers:
        shift = (atlas[layer_idx] - student_atlas[layer_idx]).astype(np.float32)
        hook = MeanShiftHook(shift)
        h = sys_s.model.model.layers[layer_idx].register_forward_hook(hook)
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

    # Teacher atlas
    print(f"[{time.time()-t0:.1f}s] teacher atlas...")
    sys_t = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    n_layers = sys_t.n_hidden_layers()
    nll_teacher, _ = measure_nll(sys_t.model, sys_t.tokenizer, sents)
    atlas = {}
    for L in range(n_layers):
        atlas[L] = extract_mid_activations(sys_t, sents, L, f"t{L}").mean(axis=0).astype(np.float32)
    sys_t.unload(); torch.cuda.empty_cache()
    print(f"  teacher NLL = {nll_teacher:.3f}, atlas {len(atlas)} layers")

    # Student atlas (from fully-lesioned model - same random seed)
    print(f"[{time.time()-t0:.1f}s] student atlas (from full-lesion)...")
    sys_s = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    sd = sys_s.model.state_dict()
    for L in range(n_layers):
        lesion_midblock(sd, f"model.layers.{L}.")
    sys_s.model.load_state_dict(sd, strict=False)
    nll_lesion, _ = measure_nll(sys_s.model, sys_s.tokenizer, sents)
    student_atlas = {}
    for L in range(n_layers):
        student_atlas[L] = extract_mid_activations(sys_s, sents, L, f"s{L}").mean(axis=0).astype(np.float32)
    sys_s.unload(); torch.cuda.empty_cache()
    print(f"  lesion NLL = {nll_lesion:.3f}, gap = {nll_lesion - nll_teacher:.3f}")

    regimes = {
        "none":      [],
        "first-7":   list(range(0, 7)),
        "last-7":    list(range(21, 28)),
        "first-14":  list(range(0, 14)),
        "last-14":   list(range(14, 28)),
        "all-28":    list(range(0, 28)),
        "alternate": list(range(0, 28, 2)),
        "mid-14":    list(range(7, 21)),
    }

    rows = []
    for name, layers in regimes.items():
        print(f"\n-- {name}: patching {len(layers)} layers --")
        nll = run_with_patches(hf_id, sents, atlas, student_atlas, layers, n_layers, name)
        gap = nll_lesion - nll_teacher
        fg = (nll_lesion - nll) / max(gap, 1e-6)
        rows.append({"regime": name, "n_patched": len(layers),
                     "layers": layers, "nll": nll,
                     "fg_closed": float(fg)})
        print(f"  NLL = {nll:.3f}  fg_closed = {fg:+.3f}")

    print("\n=== NEURAL GENOME RECOVERY CURVE ===")
    print(f"{'regime':<10s} {'n_patched':>10s} {'NLL':>8s} {'fg_closed':>10s}")
    for r in rows:
        print(f"  {r['regime']:<10s}  {r['n_patched']:9d}  {r['nll']:7.3f}  {r['fg_closed']:+9.3f}")

    out = {"teacher_nll": nll_teacher,
           "lesion_nll": nll_lesion,
           "per_regime": rows,
           "n_layers": n_layers,
           "atlas_kb": n_layers * 1024 * 4 / 1024}
    out_path = _ROOT / "results/gate2/full_mean_genome_sweep.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
