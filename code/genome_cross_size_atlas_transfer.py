"""Cross-size atlas transfer: does Qwen3-0.6B's genome patch a lesioned
Qwen3-1.7B? (Different hidden sizes, same tokenizer family.)

Setup:
 - Teacher: Qwen3-0.6B (h=1024, 28 layers)     — atlas source
 - Student: Qwen3-1.7B (h=2048, 28 layers)     — lesion recipient

Since hidden sizes differ, the teacher's 1024-d mean atlas cannot be
directly added to the student's 2048-d activations. We fit a CHEAP
linear projection P (1024 -> 2048) from one probe batch, then use
P(teacher_mean) as the shift for the student.

The projection P is fit by matching the SECOND-MOMENT statistics of
student activations to teacher via closed-form least squares (no
training loop).

Kill: fg_closed < 0.05 on Qwen3-1.7B full-lesion. Positive: cross-size
transfer is a real thing.

This test is on CS/AI scope only (no biology). Partner angle: Liquid AI
/ Martian / anybody doing cross-model capability reasoning.
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


def fit_projection(X_src, X_tgt):
    """Fit P in R^{h_src x h_tgt} such that X_src @ P approximately equals X_tgt.
    Closed form (ridge-regularized pinv to keep numerical stability).
    """
    # Center both
    mu_s = X_src.mean(axis=0, keepdims=True)
    mu_t = X_tgt.mean(axis=0, keepdims=True)
    A = X_src - mu_s    # (n, h_src)
    B = X_tgt - mu_t    # (n, h_tgt)
    # Ridge-regularized pseudoinverse: P = (A^T A + lambda I)^-1 A^T B
    lam = 1e-2 * float(np.trace(A.T @ A)) / max(A.shape[1], 1)
    AtA = A.T @ A + lam * np.eye(A.shape[1], dtype=A.dtype)
    AtB = A.T @ B
    P = np.linalg.solve(AtA, AtB).astype(np.float32)
    # Sanity: if any NaN, fall back to identity-padded mean-only
    if not np.all(np.isfinite(P)):
        P = np.zeros((A.shape[1], B.shape[1]), dtype=np.float32)
    return P


def main():
    teacher_id = "Qwen/Qwen3-0.6B"
    student_id = "Qwen/Qwen3-1.7B"
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= 300:
            break
    t0 = time.time()

    # Teacher atlas
    print(f"[{time.time()-t0:.1f}s] TEACHER Qwen3-0.6B...")
    sys_t = load_system(teacher_id, quant="fp16", untrained=False, device="cuda")
    n_layers_t = sys_t.n_hidden_layers()
    atlas_t = {}
    for L in range(n_layers_t):
        atlas_t[L] = extract_mid_activations(sys_t, sents, L, f"t{L}").astype(np.float32)
    sys_t.unload(); torch.cuda.empty_cache()
    h_teacher = atlas_t[0].shape[1]
    print(f"  teacher h={h_teacher}, layers={n_layers_t}")

    # Reference: student pretrained + student lesioned
    print(f"\n[{time.time()-t0:.1f}s] STUDENT Qwen3-1.7B pretrained ref...")
    sys_s_ref = load_system(student_id, quant="fp16", untrained=False, device="cuda")
    n_layers_s = sys_s_ref.n_hidden_layers()
    nll_student_pretrained, _ = measure_nll(sys_s_ref.model, sys_s_ref.tokenizer, sents)
    # Grab per-layer student activations to fit projections per layer
    student_acts_unlesioned = {}
    for L in range(n_layers_s):
        student_acts_unlesioned[L] = extract_mid_activations(sys_s_ref, sents, L,
                                                               f"s_unl_{L}").astype(np.float32)
    sys_s_ref.unload(); torch.cuda.empty_cache()
    h_student = student_acts_unlesioned[0].shape[1]
    print(f"  student h={h_student}, layers={n_layers_s}, pretrained NLL={nll_student_pretrained:.3f}")

    # If layer count differs, map teacher layers to student layers proportionally
    layer_map = {}
    for L in range(n_layers_s):
        L_t = int(round(L * (n_layers_t - 1) / max(n_layers_s - 1, 1)))
        layer_map[L] = L_t

    # Fit projection per student-layer using PRETRAINED student activations.
    projections = {}
    for L in range(n_layers_s):
        L_t = layer_map[L]
        P = fit_projection(atlas_t[L_t], student_acts_unlesioned[L])
        projections[L] = P

    # Student fully lesioned
    print(f"\n[{time.time()-t0:.1f}s] STUDENT lesion all layers...")
    sys_s = load_system(student_id, quant="fp16", untrained=False, device="cuda")
    sd = sys_s.model.state_dict()
    for L in range(n_layers_s):
        lesion_midblock(sd, f"model.layers.{L}.")
    sys_s.model.load_state_dict(sd, strict=False)
    nll_lesion, _ = measure_nll(sys_s.model, sys_s.tokenizer, sents)
    # Extract lesioned student means for shift computation
    student_atlas_lesion = {}
    for L in range(n_layers_s):
        student_atlas_lesion[L] = extract_mid_activations(sys_s, sents, L,
                                                            f"s_les_{L}").mean(axis=0).astype(np.float32)
    print(f"  lesion NLL = {nll_lesion:.3f}")
    print(f"  gap vs student pretrained = {nll_lesion - nll_student_pretrained:.3f}")

    # Install cross-size atlas hooks: shift = P @ teacher_mean - student_lesion_mean
    handles = []
    for L in range(n_layers_s):
        L_t = layer_map[L]
        teacher_mean_t = atlas_t[L_t].mean(axis=0).astype(np.float32)  # (h_teacher,)
        projected = teacher_mean_t @ projections[L]                     # (h_student,)
        shift = (projected - student_atlas_lesion[L]).astype(np.float32)
        hook = MeanShiftHook(shift)
        h = sys_s.model.model.layers[L].register_forward_hook(hook)
        handles.append(h)
    nll_patched, _ = measure_nll(sys_s.model, sys_s.tokenizer, sents)
    for h in handles:
        h.remove()
    sys_s.unload(); torch.cuda.empty_cache()

    gap = nll_lesion - nll_student_pretrained
    fg_closed = (nll_lesion - nll_patched) / max(gap, 1e-6)

    print("\n=== CROSS-SIZE ATLAS TRANSFER (Qwen3-0.6B -> Qwen3-1.7B) ===")
    print(f"  student pretrained NLL:   {nll_student_pretrained:.3f}")
    print(f"  student all-lesion NLL:   {nll_lesion:.3f}  gap={gap:.3f}")
    print(f"  cross-size patched NLL:   {nll_patched:.3f}  fg_closed={fg_closed:+.3f}")

    if fg_closed >= 0.15:
        verdict = (f"CROSS_SIZE_TRANSFER_LANDS - Qwen3-0.6B atlas via "
                   f"projection recovers {fg_closed*100:.0f}pct of Qwen3-"
                   f"1.7B capability. Cross-size genome is a real effect.")
    elif fg_closed >= 0.05:
        verdict = f"PARTIAL - {fg_closed*100:.0f}pct cross-size recovery."
    else:
        verdict = ("CROSS_SIZE_NULL - projection-based cross-size transfer "
                   "failed.")
    print(f"\n  verdict: {verdict}")

    out = {"purpose": "Cross-size atlas transfer Qwen3-0.6B -> Qwen3-1.7B",
           "teacher_id": teacher_id, "student_id": student_id,
           "teacher_h": int(h_teacher), "student_h": int(h_student),
           "n_layers_teacher": int(n_layers_t),
           "n_layers_student": int(n_layers_s),
           "student_pretrained_nll": float(nll_student_pretrained),
           "lesion_nll": float(nll_lesion),
           "patched_nll": float(nll_patched),
           "fg_closed": float(fg_closed),
           "verdict": verdict}
    out_path = _ROOT / "results/gate2/cross_size_atlas_transfer.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
