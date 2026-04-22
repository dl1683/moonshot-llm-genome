"""THE NEURAL GENOME: per-layer mean-activation atlas of a trained model.

If a single 1024-dim mean-shift can recover 60pct of capability at ONE
lesioned layer (genome_074 / 075), what if we compute the per-layer
mean for EVERY layer and use those 28 x 1024 scalars = 28,672 scalars
= ~115 KB as the "genome" of the model?

EXPERIMENT.
 1. Compute teacher's per-layer mean-activation signature: a dict of
    {layer_idx: mean_activation_vec_h} for all 28 layers of Qwen3-0.6B.
    Store it. ~115 KB of floats. This IS the model's "atlas coordinates."
 2. Lesion EVERY transformer layer of a student Qwen3-0.6B copy (all 28
    layers have their weights scrambled). This should catastrophically
    break capability.
 3. Install layer-wise mean-shift hooks on every layer using the
    teacher atlas.
 4. Measure NLL: teacher, fully-lesioned, full-genome-patched.

KILL: fraction_gap_closed < 0.05. Means the per-layer mean atlas
doesn't capture capability at scale.

PASS (landmark): fg_closed >= 0.15. Means a 115 KB vector table
captures enough signal from the 600M-param teacher to recover some
capability in a fully-lesioned student. Direct "model genome" demo.

STRONGER PASS (outreach-grade): fg_closed >= 0.30. Means the mean
atlas is a meaningful compression of model capability.

Per CLAUDE.md sec 0.05 scope lock: CS / AI / math only. This advances
'map the learning of every AI model + model surgery' directly.
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


def main():
    hf_id = "Qwen/Qwen3-0.6B"
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= 300:
            break
    t0 = time.time()

    # -------- (1) Build teacher atlas --------
    print(f"[{time.time()-t0:.1f}s] building teacher atlas (per-layer mean-activation)...")
    sys_t = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    n_layers = sys_t.n_hidden_layers()
    nll_teacher, _ = measure_nll(sys_t.model, sys_t.tokenizer, sents)
    print(f"  teacher NLL = {nll_teacher:.3f}")
    atlas = {}
    for layer_idx in range(n_layers):
        acts = extract_mid_activations(sys_t, sents, layer_idx, f"teacher_L{layer_idx}")
        atlas[layer_idx] = acts.mean(axis=0).astype(np.float32)
    sys_t.unload(); torch.cuda.empty_cache()
    # Size in bytes
    total_scalars = sum(v.size for v in atlas.values())
    total_bytes = total_scalars * 4
    print(f"  atlas: {n_layers} layers x {len(next(iter(atlas.values())))} dim = "
          f"{total_scalars} scalars = {total_bytes/1024:.1f} KB")

    # -------- (2) Student with ALL layers lesioned --------
    print(f"[{time.time()-t0:.1f}s] building student with ALL layers lesioned...")
    sys_s = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    sd = sys_s.model.state_dict()
    total_lesioned = 0
    for layer_idx in range(n_layers):
        total_lesioned += lesion_midblock(sd, f"model.layers.{layer_idx}.")
    sys_s.model.load_state_dict(sd, strict=False)
    print(f"  lesioned {total_lesioned} matmul weights across {n_layers} layers")

    # Extract student's per-layer means to compute the shift
    print(f"[{time.time()-t0:.1f}s] extracting student per-layer means for shift computation...")
    student_atlas = {}
    for layer_idx in range(n_layers):
        acts = extract_mid_activations(sys_s, sents, layer_idx, f"student_L{layer_idx}")
        student_atlas[layer_idx] = acts.mean(axis=0).astype(np.float32)

    nll_lesion, _ = measure_nll(sys_s.model, sys_s.tokenizer, sents)
    print(f"  lesion NLL = {nll_lesion:.3f}  gap = {nll_lesion - nll_teacher:.3f}")

    # -------- (3) Install per-layer mean-shift hooks --------
    print(f"[{time.time()-t0:.1f}s] installing per-layer mean-shift hooks...")
    handles = []
    for layer_idx in range(n_layers):
        shift = (atlas[layer_idx] - student_atlas[layer_idx]).astype(np.float32)
        hook = MeanShiftHook(shift)
        h = sys_s.model.model.layers[layer_idx].register_forward_hook(hook)
        handles.append(h)

    # -------- (4) Measure full-genome-patched NLL --------
    nll_patched, _ = measure_nll(sys_s.model, sys_s.tokenizer, sents)
    for h in handles:
        h.remove()
    print(f"  full-genome-patched NLL = {nll_patched:.3f}")

    sys_s.unload(); torch.cuda.empty_cache()

    # -------- (5) Partial-atlas variant: patch FIRST HALF only --------
    print(f"\n[{time.time()-t0:.1f}s] partial-atlas variant (first half only)...")
    sys_s2 = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    sd = sys_s2.model.state_dict()
    for layer_idx in range(n_layers):
        lesion_midblock(sd, f"model.layers.{layer_idx}.")
    sys_s2.model.load_state_dict(sd, strict=False)
    half = n_layers // 2
    handles = []
    for layer_idx in range(half):
        shift = (atlas[layer_idx] - student_atlas[layer_idx]).astype(np.float32)
        hook = MeanShiftHook(shift)
        h = sys_s2.model.model.layers[layer_idx].register_forward_hook(hook)
        handles.append(h)
    nll_half_patched, _ = measure_nll(sys_s2.model, sys_s2.tokenizer, sents)
    for h in handles:
        h.remove()
    sys_s2.unload(); torch.cuda.empty_cache()

    # -------- (6) Report --------
    gap = nll_lesion - nll_teacher
    fg_full = (nll_lesion - nll_patched) / max(gap, 1e-6)
    fg_half = (nll_lesion - nll_half_patched) / max(gap, 1e-6)

    print(f"\n=== THE NEURAL GENOME: MEAN-ATLAS CAPABILITY RECOVERY ===")
    print(f"  teacher NLL:                        {nll_teacher:.3f}")
    print(f"  all-layers-lesion NLL:              {nll_lesion:.3f}")
    print(f"  full-atlas-patched NLL ({n_layers} layers): {nll_patched:.3f}  fg_closed={fg_full:+.3f}")
    print(f"  half-atlas-patched NLL ({half} layers):    {nll_half_patched:.3f}  fg_closed={fg_half:+.3f}")
    print(f"  atlas storage size:                 {total_bytes/1024:.1f} KB")

    if fg_full >= 0.30:
        verdict = (f"NEURAL_GENOME_LANDS_HARD - a {total_bytes/1024:.0f} KB mean-"
                   f"activation atlas recovers {fg_full*100:.0f}pct of capability "
                   f"in a fully-lesioned model. Model genome is literally 115 KB.")
    elif fg_full >= 0.15:
        verdict = (f"NEURAL_GENOME_LANDS - {fg_full*100:.0f}pct recovery. A "
                   f"{total_bytes/1024:.0f} KB vector table carries "
                   "meaningful capability signal.")
    elif fg_full >= 0.05:
        verdict = f"PARTIAL - {fg_full*100:.0f}pct recovery from atlas."
    else:
        verdict = "ATLAS_NULL - per-layer mean atlas does not carry enough signal."

    print(f"\n  verdict: {verdict}")

    out = {"purpose": "THE NEURAL GENOME: per-layer mean-activation capability atlas",
           "teacher_nll": float(nll_teacher),
           "lesion_nll_all_layers": float(nll_lesion),
           "full_atlas_patched_nll": float(nll_patched),
           "half_atlas_patched_nll": float(nll_half_patched),
           "fg_closed_full": float(fg_full),
           "fg_closed_half": float(fg_half),
           "n_layers": int(n_layers),
           "atlas_bytes": int(total_bytes),
           "verdict": verdict}
    out_path = _ROOT / "results/gate2/full_mean_genome.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
