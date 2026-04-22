"""Capability patch v2: debug the v1 null result by varying rank and hook style.

v1 result: rank-48 replacement hook h <- h @ W_lr gave NLL 12.53 vs lesion 4.87
(actively hurts). Adapter R2 = 0.627 was OK, so the fit on probe-set is fine;
the problem is how the adapter interacts with live inference.

v2 tries 4 hook styles x 3 ranks = 12 conditions:
  Hook styles:
    (R) replacement: h_out <- h_in @ W
    (D) residual add-on: h_out <- h_in + (h_in @ W - h_in)
    (M) mean-correction: h_out <- h_in + (mean(X_teach_mid) - mean(X_stu_mid))
    (T) teacher-injection (upper bound): h_out <- X_teacher_mid[batch_idx]
                        (only meaningful on the exact probe batch)
  Ranks for W_lr: 48 (k_bulk), 256, 1024 (full).

If (R) at rank=1024 still fails, the pinv adapter is fundamentally unstable.
If (R) at rank=1024 works but rank=48 fails, we need higher rank.
If (D) or (M) works better, the issue was replacement style.

Pure debug experiment. Target: understand why v1 failed and find the
RIGHT adapter class for mid-block capability surgery.
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
    extract_mid_activations, fit_low_rank_adapter, lesion_midblock,
)
from genome_geometry_transfusion import measure_nll  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent


class ReplaceHook:
    def __init__(self, W):
        self.W = torch.from_numpy(W).float()
    def __call__(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        W = self.W.to(device=h.device, dtype=h.dtype)
        h_new = h @ W
        return (h_new,) + output[1:] if isinstance(output, tuple) else h_new


class ResidualHook:
    """Add a correction: h_out = h + (h @ W - h) * scale."""
    def __init__(self, W, scale=1.0):
        self.W = torch.from_numpy(W).float()
        self.scale = scale
    def __call__(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        W = self.W.to(device=h.device, dtype=h.dtype)
        h_new = h + self.scale * (h @ W - h)
        return (h_new,) + output[1:] if isinstance(output, tuple) else h_new


class MeanShiftHook:
    def __init__(self, shift):
        self.shift = torch.from_numpy(shift).float()
    def __call__(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        s = self.shift.to(device=h.device, dtype=h.dtype)
        h_new = h + s
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
    print(f"[{time.time()-t0:.1f}s] TEACHER...")
    sys_t = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    mid = sys_t.n_hidden_layers() // 2
    nll_teacher, _ = measure_nll(sys_t.model, sys_t.tokenizer, sents)
    X_teacher = extract_mid_activations(sys_t, sents, mid, "teacher")
    mean_teacher = X_teacher.mean(axis=0)
    sys_t.unload(); torch.cuda.empty_cache()

    # Student (lesioned)
    print(f"[{time.time()-t0:.1f}s] STUDENT (lesioned layer {mid})...")
    sys_s = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    sd = sys_s.model.state_dict()
    prefix = f"model.layers.{mid}."
    lesion_midblock(sd, prefix)
    sys_s.model.load_state_dict(sd, strict=False)
    nll_lesion, _ = measure_nll(sys_s.model, sys_s.tokenizer, sents)
    X_student = extract_mid_activations(sys_s, sents, mid, "student_lesion")
    mean_student = X_student.mean(axis=0)

    print(f"  teacher NLL={nll_teacher:.3f}  lesion NLL={nll_lesion:.3f}  "
          f"gap={nll_lesion - nll_teacher:.3f}")

    results = []

    # Fit adapters at multiple ranks
    for rank in [48, 256, 1024]:
        print(f"\n-- fitting rank-{rank} adapter --")
        W_lr, _, _ = fit_low_rank_adapter(X_student, X_teacher, rank=rank)
        resid = X_student @ W_lr - X_teacher
        r2 = 1.0 - float((resid ** 2).sum() / max((X_teacher ** 2).sum(), 1e-9))
        print(f"  R2 = {r2:.3f}")

        target_module = sys_s.model.model.layers[mid]
        # Replace hook
        h = ReplaceHook(W_lr)
        handle = target_module.register_forward_hook(h)
        try:
            nll_r, _ = measure_nll(sys_s.model, sys_s.tokenizer, sents)
        except Exception as e:
            nll_r = float("nan"); print(f"  R hook err: {e}")
        handle.remove()

        # Residual hook (add scaled correction)
        h2 = ResidualHook(W_lr, scale=0.5)
        handle = target_module.register_forward_hook(h2)
        try:
            nll_d, _ = measure_nll(sys_s.model, sys_s.tokenizer, sents)
        except Exception as e:
            nll_d = float("nan"); print(f"  D hook err: {e}")
        handle.remove()

        print(f"  rank={rank}  replace NLL={nll_r:.3f}  residual NLL={nll_d:.3f}")
        results.append({"rank": rank, "r2": r2,
                        "replace_nll": float(nll_r),
                        "residual_nll": float(nll_d)})

    # Mean-shift only (no W)
    print("\n-- mean-shift hook (no low-rank W) --")
    shift = mean_teacher - mean_student
    h3 = MeanShiftHook(shift)
    target_module = sys_s.model.model.layers[mid]
    handle = target_module.register_forward_hook(h3)
    try:
        nll_m, _ = measure_nll(sys_s.model, sys_s.tokenizer, sents)
    except Exception as e:
        nll_m = float("nan"); print(f"  M hook err: {e}")
    handle.remove()
    print(f"  mean-shift NLL={nll_m:.3f}")

    sys_s.unload(); torch.cuda.empty_cache()

    # Summary
    print(f"\n=== DEBUG SUMMARY ===")
    print(f"  teacher NLL:          {nll_teacher:.3f}")
    print(f"  lesion NLL:           {nll_lesion:.3f}")
    print(f"  mean-shift NLL:       {nll_m:.3f}  (fg_closed={(nll_lesion-nll_m)/max(nll_lesion-nll_teacher,1e-6):+.3f})")
    for r in results:
        fg_r = (nll_lesion - r["replace_nll"]) / max(nll_lesion - nll_teacher, 1e-6)
        fg_d = (nll_lesion - r["residual_nll"]) / max(nll_lesion - nll_teacher, 1e-6)
        print(f"  rank={r['rank']:4d}  R2={r['r2']:.2f}  "
              f"replace NLL={r['replace_nll']:.2f} fg={fg_r:+.2f}   "
              f"residual NLL={r['residual_nll']:.2f} fg={fg_d:+.2f}")

    best = max(
        [{"hook": "mean", "nll": nll_m, "rank": None}] +
        [{"hook": "replace", "nll": r["replace_nll"], "rank": r["rank"]} for r in results] +
        [{"hook": "residual", "nll": r["residual_nll"], "rank": r["rank"]} for r in results],
        key=lambda x: -x["nll"] if x["nll"] != x["nll"] else -x["nll"])
    # Actually pick the lowest NLL (most improvement over lesion)
    cands = [{"hook": "mean", "nll": nll_m, "rank": None}]
    for r in results:
        cands.append({"hook": "replace", "nll": r["replace_nll"], "rank": r["rank"]})
        cands.append({"hook": "residual", "nll": r["residual_nll"], "rank": r["rank"]})
    valid = [c for c in cands if np.isfinite(c["nll"])]
    best = min(valid, key=lambda x: x["nll"]) if valid else None

    if best:
        fg = (nll_lesion - best["nll"]) / max(nll_lesion - nll_teacher, 1e-6)
        print(f"\n  BEST: {best['hook']} rank={best['rank']}  NLL={best['nll']:.3f}  fg_closed={fg:+.3f}")

    out = {"teacher_nll": nll_teacher, "lesion_nll": nll_lesion,
           "mean_shift_nll": float(nll_m),
           "per_rank": results,
           "best": best}
    out_path = _ROOT / "results/gate2/capability_patch_k48_v2.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
