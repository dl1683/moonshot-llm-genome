"""Capability-patch richer adapter class: mean + diagonal std.

genome_074 / genome_075 showed mean-shift (1024 params) recovers ~60pct
of capability gap on a lesioned Qwen3 mid block. mean+rotation (full
orthogonal Procrustes at block boundary) HURTS because rotation
disrupts residual stream.

This experiment tests a MIDDLE GROUND adapter class: mean + per-dim
STD rescale. 2 x 1024 = 2048 params total. Hook form:

    h_new = (h - mean_student) * (std_teacher / std_student) + mean_teacher

which is "z-score normalize student, then shift to teacher distribution."
It preserves within-sample pairwise cosine structure (unlike rotation)
while matching both first and second moments.

If this beats mean-only, we have an even tighter capability-surgery
adapter class. If it doesn't beat mean-only, mean-shift is the
unique right adapter and adding variance hurts (probably because
dividing by student std amplifies lesion noise).

Tests at 3 layer depths to match genome_075.
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


class MeanStdHook:
    """h_new = (h - mu_s) * (std_t / std_s) + mu_t."""
    def __init__(self, mu_s, mu_t, std_s, std_t, eps=1e-4):
        self.mu_s = torch.from_numpy(mu_s).float()
        self.mu_t = torch.from_numpy(mu_t).float()
        self.scale = torch.from_numpy(std_t / (std_s + eps)).float()

    def __call__(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        mu_s = self.mu_s.to(device=h.device, dtype=h.dtype)
        mu_t = self.mu_t.to(device=h.device, dtype=h.dtype)
        scale = self.scale.to(device=h.device, dtype=h.dtype)
        h_new = (h - mu_s) * scale + mu_t
        return (h_new,) + output[1:] if isinstance(output, tuple) else h_new


def main():
    hf_id = "Qwen/Qwen3-0.6B"
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= 500:
            break
    t0 = time.time()

    # Teacher
    sys_t = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    nll_teacher, _ = measure_nll(sys_t.model, sys_t.tokenizer, sents)
    teacher_acts = {}
    for layer_idx in [7, 14, 21]:
        teacher_acts[layer_idx] = extract_mid_activations(
            sys_t, sents, layer_idx, f"teacher_L{layer_idx}")
    sys_t.unload(); torch.cuda.empty_cache()
    print(f"  teacher NLL = {nll_teacher:.3f}")

    rows = []
    for lesion_layer in [7, 14, 21]:
        print(f"\n-- lesion_layer={lesion_layer} --")
        sys_s = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
        sd = sys_s.model.state_dict()
        prefix = f"model.layers.{lesion_layer}."
        lesion_midblock(sd, prefix)
        sys_s.model.load_state_dict(sd, strict=False)
        nll_les, _ = measure_nll(sys_s.model, sys_s.tokenizer, sents)
        X_s = extract_mid_activations(sys_s, sents, lesion_layer, f"stu_L{lesion_layer}")
        X_t = teacher_acts[lesion_layer]
        mu_s = X_s.mean(axis=0).astype(np.float32)
        mu_t = X_t.mean(axis=0).astype(np.float32)
        std_s = X_s.std(axis=0).astype(np.float32)
        std_t = X_t.std(axis=0).astype(np.float32)
        mean_shift = mu_t - mu_s

        target_mod = sys_s.model.model.layers[lesion_layer]

        # Mean-only baseline
        hook_m = MeanShiftHook(mean_shift)
        handle = target_mod.register_forward_hook(hook_m)
        nll_m, _ = measure_nll(sys_s.model, sys_s.tokenizer, sents)
        handle.remove()

        # Mean + std
        hook_ms = MeanStdHook(mu_s, mu_t, std_s, std_t)
        handle = target_mod.register_forward_hook(hook_ms)
        try:
            nll_ms, _ = measure_nll(sys_s.model, sys_s.tokenizer, sents)
        except Exception as e:
            nll_ms = float("nan"); print(f"  mean+std err: {e}")
        handle.remove()

        gap = nll_les - nll_teacher
        fg_m = (nll_les - nll_m) / max(gap, 1e-6)
        fg_ms = (nll_les - nll_ms) / max(gap, 1e-6) if np.isfinite(nll_ms) else float("nan")
        print(f"  lesion NLL = {nll_les:.3f}  gap = {gap:.3f}")
        print(f"  mean-only NLL = {nll_m:.3f}  fg = {fg_m:+.3f}")
        print(f"  mean+std  NLL = {nll_ms:.3f}  fg = {fg_ms:+.3f}")
        rows.append({
            "lesion_layer": lesion_layer,
            "teacher_nll": nll_teacher,
            "lesion_nll": nll_les,
            "mean_only_nll": nll_m, "fg_mean": float(fg_m),
            "mean_std_nll": nll_ms, "fg_mean_std": float(fg_ms) if np.isfinite(fg_ms) else None,
        })
        sys_s.unload(); torch.cuda.empty_cache()

    print("\n=== MEAN + DIAGONAL STD CAPABILITY PATCH ===")
    fg_m_mean = np.mean([r["fg_mean"] for r in rows])
    fg_ms_valid = [r["fg_mean_std"] for r in rows if r["fg_mean_std"] is not None]
    fg_ms_mean = np.mean(fg_ms_valid) if fg_ms_valid else None
    print(f"  mean(fg_mean_only): {fg_m_mean:.3f}")
    print(f"  mean(fg_mean+std):  {fg_ms_mean}")

    if fg_ms_mean is not None and fg_ms_mean > fg_m_mean + 0.05:
        verdict = (f"MEAN+STD BEATS MEAN-ONLY by {(fg_ms_mean-fg_m_mean)*100:.1f}pp. "
                   "Tighter adapter class: mean + per-dim std scaling recovers "
                   "more than mean-only. 2K params.")
    elif fg_ms_mean is not None and abs(fg_ms_mean - fg_m_mean) <= 0.05:
        verdict = "MEAN+STD NEUTRAL - no gain over mean-only. Keep mean-only (1K params)."
    elif fg_ms_mean is not None:
        verdict = f"MEAN+STD HURTS by {(fg_m_mean-fg_ms_mean)*100:.1f}pp. Std rescale amplifies lesion noise; mean-only wins."
    else:
        verdict = "MEAN+STD CRASHED"
    print(f"\n  verdict: {verdict}")

    out = {"per_layer": rows,
           "mean_fg_mean_only": float(fg_m_mean),
           "mean_fg_mean_plus_std": float(fg_ms_mean) if fg_ms_mean is not None else None,
           "verdict": verdict}
    out_path = _ROOT / "results/gate2/capability_patch_mean_std.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
