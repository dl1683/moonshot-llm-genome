"""Partial-lesion capability patch: lesion ONLY the last 7 layers, patch with
atlas, measure both NLL recovery AND coherent generation.

genome_083 showed that lesioning ALL 28 layers and patching with atlas
restores NLL but collapses generation to degenerate repetition. This
probes a more realistic surgery scenario:

 - Early/middle layers (0..20) remain PRETRAINED and intact.
 - Only the last 7 layers (21..27) are lesioned.
 - Atlas patch applied to the same last 7 layers.

If conditioning machinery in early/middle layers is preserved, the
atlas-patched last 7 layers should be able to produce coherent token
predictions. Tests whether the atlas surgery is useful when the rest
of the model is intact (e.g., patching a partially-corrupted
checkpoint).

Measures: NLL + 5 prompt completions, same as genome_083.
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
from genome_atlas_qualitative_samples import PROMPTS, generate  # noqa: E402
from genome_capability_patch_k48 import extract_mid_activations, lesion_midblock  # noqa: E402
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

    # Teacher atlas for the LAST 7 layers
    print(f"[{time.time()-t0:.1f}s] teacher atlas (last 7 layers)...")
    sys_t = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    n_layers = sys_t.n_hidden_layers()
    last_7 = list(range(n_layers - 7, n_layers))
    nll_teacher, _ = measure_nll(sys_t.model, sys_t.tokenizer, sents)
    atlas = {L: extract_mid_activations(sys_t, sents, L, f"t{L}").mean(axis=0).astype(np.float32)
             for L in last_7}
    teacher_out = {p: generate(sys_t.model, sys_t.tokenizer, p) for p in PROMPTS}
    for p, c in teacher_out.items():
        safe_p = p.encode("ascii", "backslashreplace").decode("ascii")
        safe_c = c.encode("ascii", "backslashreplace").decode("ascii")
        print(f"  [TEACHER] {safe_p!r} -> {safe_c!r}")
    sys_t.unload(); torch.cuda.empty_cache()

    # Student with ONLY last-7 lesioned
    print(f"\n[{time.time()-t0:.1f}s] student with last-7 lesion...")
    sys_s = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    sd = sys_s.model.state_dict()
    for L in last_7:
        lesion_midblock(sd, f"model.layers.{L}.")
    sys_s.model.load_state_dict(sd, strict=False)
    nll_lesion, _ = measure_nll(sys_s.model, sys_s.tokenizer, sents)
    student_atlas = {L: extract_mid_activations(sys_s, sents, L, f"s{L}").mean(axis=0).astype(np.float32)
                      for L in last_7}
    # Generation with lesion, no patch
    lesion_out = {}
    for p in PROMPTS:
        lesion_out[p] = generate(sys_s.model, sys_s.tokenizer, p)
        safe_p = p.encode("ascii", "backslashreplace").decode("ascii")
        safe_c = lesion_out[p].encode("ascii", "backslashreplace").decode("ascii")
        print(f"  [LAST7-LESION] {safe_p!r} -> {safe_c!r}")
    print(f"  last-7-lesion NLL={nll_lesion:.3f}")
    sys_s.unload(); torch.cuda.empty_cache()

    # Patched student: last-7 lesioned + atlas applied
    print(f"\n[{time.time()-t0:.1f}s] student with last-7 lesion + atlas patch...")
    sys_p = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    sd = sys_p.model.state_dict()
    for L in last_7:
        lesion_midblock(sd, f"model.layers.{L}.")
    sys_p.model.load_state_dict(sd, strict=False)
    handles = []
    for L in last_7:
        shift = (atlas[L] - student_atlas[L]).astype(np.float32)
        hook = MeanShiftHook(shift)
        h = sys_p.model.model.layers[L].register_forward_hook(hook)
        handles.append(h)
    nll_patched, _ = measure_nll(sys_p.model, sys_p.tokenizer, sents)
    # Generation with atlas
    patched_out = {}
    for p in PROMPTS:
        patched_out[p] = generate(sys_p.model, sys_p.tokenizer, p)
        safe_p = p.encode("ascii", "backslashreplace").decode("ascii")
        safe_c = patched_out[p].encode("ascii", "backslashreplace").decode("ascii")
        print(f"  [LAST7-ATLAS] {safe_p!r} -> {safe_c!r}")
    for h in handles:
        h.remove()
    sys_p.unload(); torch.cuda.empty_cache()

    gap = nll_lesion - nll_teacher
    fg = (nll_lesion - nll_patched) / max(gap, 1e-6)

    print(f"\n=== PARTIAL-LESION (last-7 only) + ATLAS PATCH ===")
    print(f"  teacher NLL:      {nll_teacher:.3f}")
    print(f"  last-7-lesion NLL:{nll_lesion:.3f}  gap={gap:.3f}")
    print(f"  atlas-patched NLL:{nll_patched:.3f}  fg_closed={fg:+.3f}")

    out = {"teacher_nll": nll_teacher,
           "last7_lesion_nll": nll_lesion,
           "patched_nll": nll_patched,
           "fraction_gap_closed": float(fg),
           "teacher_completions": {p: c for p, c in teacher_out.items()},
           "lesioned_completions": {p: c for p, c in lesion_out.items()},
           "patched_completions": {p: c for p, c in patched_out.items()}}
    # ASCII-safe JSON write
    p = Path(_ROOT / "results/gate2/atlas_partial_lesion.json")
    p.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
