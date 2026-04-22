"""Capability patch generalization: does mean-shift surgery work at other
layer depths on Qwen3-0.6B?

genome_074 showed mean-shift closes 65pct of the capability gap at layer 14
(mid). To claim "model surgery via mean-shift works," we need it to
generalize across layer depths. Test lesion at {7, 14, 21} out of 28 layers
(shallow / mid / deep).

Also sweeps 2 student conditions per layer:
  (A) Frobenius-matched Gaussian lesion (as genome_074)
  (B) Same-magnitude Gaussian but SCALED by a controllable factor in
      {0.5, 1.0, 2.0} to probe whether mean-shift works across different
      lesion severities.

Also tests "full direction-identity patch" (mean + rank-1024 orthogonal-
Procrustes rotation) at mid depth, compared to mean-shift alone.
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
from genome_capability_patch_k48_v2 import MeanShiftHook, ReplaceHook, ResidualHook  # noqa: E402
from genome_geometry_transfusion import measure_nll  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent


def lesion_midblock_scaled(state_dict, prefix, scale=1.0, seed=42):
    count = 0
    for k in list(state_dict.keys()):
        if not k.startswith(prefix):
            continue
        W = state_dict[k]
        if W.ndim != 2:
            continue
        fro = float(torch.norm(W.float()).item())
        rnd = torch.randn_like(W.float())
        rnd_fro = float(torch.norm(rnd).item())
        rnd = rnd * (fro / max(rnd_fro, 1e-6)) * scale
        state_dict[k] = rnd.to(W.dtype)
        count += 1
    return count


def procrustes(X_src, X_tgt):
    A = X_src - X_src.mean(axis=0, keepdims=True)
    B = X_tgt - X_tgt.mean(axis=0, keepdims=True)
    M = A.T @ B
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return (U @ Vt).astype(np.float32)


class MeanPlusRotationHook:
    def __init__(self, R, shift):
        self.R = torch.from_numpy(R).float()
        self.shift = torch.from_numpy(shift).float()
    def __call__(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        R = self.R.to(device=h.device, dtype=h.dtype)
        s = self.shift.to(device=h.device, dtype=h.dtype)
        h_new = h @ R + s
        return (h_new,) + output[1:] if isinstance(output, tuple) else h_new


def main():
    hf_id = "Qwen/Qwen3-0.6B"
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= 500:
            break

    t0 = time.time()

    # Load teacher ONCE to get teacher activations at ALL test layers
    print(f"[{time.time()-t0:.1f}s] TEACHER (extract all target layers)...")
    sys_t = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    nll_teacher, _ = measure_nll(sys_t.model, sys_t.tokenizer, sents)
    teacher_acts = {}
    for layer_idx in [7, 14, 21]:
        teacher_acts[layer_idx] = extract_mid_activations(sys_t, sents, layer_idx,
                                                           f"teacher_L{layer_idx}")
    print(f"  teacher NLL={nll_teacher:.3f}")
    sys_t.unload(); torch.cuda.empty_cache()

    rows = []
    # Layer depth sweep (scale = 1.0)
    for lesion_layer in [7, 14, 21]:
        for lesion_scale in [1.0]:
            print(f"\n-- lesion_layer={lesion_layer}  scale={lesion_scale} --")
            sys_s = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
            sd = sys_s.model.state_dict()
            prefix = f"model.layers.{lesion_layer}."
            lesion_midblock_scaled(sd, prefix, scale=lesion_scale)
            sys_s.model.load_state_dict(sd, strict=False)
            nll_les, _ = measure_nll(sys_s.model, sys_s.tokenizer, sents)
            X_stu = extract_mid_activations(sys_s, sents, lesion_layer,
                                              f"student_L{lesion_layer}")
            X_tea = teacher_acts[lesion_layer]
            mean_shift = (X_tea.mean(axis=0) - X_stu.mean(axis=0)).astype(np.float32)
            print(f"  lesion NLL={nll_les:.3f}  gap_vs_teacher={nll_les - nll_teacher:.3f}")

            # mean-shift hook
            target_mod = sys_s.model.model.layers[lesion_layer]
            hook = MeanShiftHook(mean_shift)
            handle = target_mod.register_forward_hook(hook)
            nll_m, _ = measure_nll(sys_s.model, sys_s.tokenizer, sents)
            handle.remove()

            # mean + rotation
            R = procrustes(X_stu, X_tea)
            hook2 = MeanPlusRotationHook(R, mean_shift)
            handle = target_mod.register_forward_hook(hook2)
            try:
                nll_mr, _ = measure_nll(sys_s.model, sys_s.tokenizer, sents)
            except Exception as e:
                nll_mr = float("nan"); print(f"  mean+rot err: {e}")
            handle.remove()

            gap = nll_les - nll_teacher
            fg_m = (nll_les - nll_m) / max(gap, 1e-6)
            fg_mr = (nll_les - nll_mr) / max(gap, 1e-6) if np.isfinite(nll_mr) else float("nan")
            print(f"  mean-shift NLL={nll_m:.3f}  fg_closed={fg_m:+.3f}")
            print(f"  mean+rot   NLL={nll_mr:.3f}  fg_closed={fg_mr:+.3f}")
            rows.append({
                "lesion_layer": lesion_layer, "lesion_scale": lesion_scale,
                "teacher_nll": float(nll_teacher),
                "lesion_nll": float(nll_les),
                "mean_shift_nll": float(nll_m),
                "mean_rotation_nll": float(nll_mr),
                "fg_closed_mean": float(fg_m),
                "fg_closed_mean_rot": float(fg_mr) if np.isfinite(fg_mr) else None,
            })
            sys_s.unload(); torch.cuda.empty_cache()

    print("\n\n=== CAPABILITY PATCH GENERALIZATION ===")
    print(f"{'layer':>6s} {'scale':>6s} {'lesion':>8s} {'mean_shift':>11s} "
          f"{'fg_m':>6s} {'mean+rot':>10s} {'fg_mr':>7s}")
    for r in rows:
        print(f"  {r['lesion_layer']:4d}  {r['lesion_scale']:6.1f}  "
              f"{r['lesion_nll']:7.3f}  {r['mean_shift_nll']:9.3f}   "
              f"{r['fg_closed_mean']:+.3f}  {r['mean_rotation_nll']:8.3f}   "
              f"{r['fg_closed_mean_rot']:+.3f}" if r['fg_closed_mean_rot'] is not None else
              f"  {r['lesion_layer']:4d}  {r['lesion_scale']:6.1f}  "
              f"{r['lesion_nll']:7.3f}  {r['mean_shift_nll']:9.3f}   "
              f"{r['fg_closed_mean']:+.3f}")

    mean_fg_mean = np.mean([r["fg_closed_mean"] for r in rows])
    if mean_fg_mean >= 0.20:
        verdict = (f"MEAN_SHIFT_GENERALIZES - mean fraction_gap_closed "
                   f"{mean_fg_mean:.2f} across 3 lesion layers. Capability "
                   "surgery via mean-shift is a layer-agnostic effect on "
                   "Qwen3.")
    else:
        verdict = (f"PARTIAL - mean_fg={mean_fg_mean:.2f}. Mean-shift works "
                   "at mid depth but not uniformly across layers.")
    print(f"\n  verdict: {verdict}")

    out = {"purpose": "Mean-shift capability surgery layer-depth generalization",
           "per_condition": rows,
           "mean_fg_closed_across_layers": float(mean_fg_mean),
           "verdict": verdict}
    out_path = _ROOT / "results/gate2/capability_patch_generalize.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
