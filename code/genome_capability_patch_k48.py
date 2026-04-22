"""Codex home-run A2: capability PATCH on a lesioned pretrained model via
a rank-48 adapter.

STATED PROJECT END GOAL (CLAUDE.md sec 0.05 scope lock): MODEL SURGERY -
transfer a capability from Model A into Model B without retraining.
This experiment is the most direct test of that end goal.

PROTOCOL.
 1. Teacher = pretrained Qwen3-0.6B (unperturbed).
 2. Student = pretrained Qwen3-0.6B with ONE mid-depth transformer block
    (layer `mid`) randomized (weights replaced with same-shape Gaussian
    matched to original per-tensor Frobenius). Creates a lesion at a
    specific layer.
 3. Baseline NLL: teacher (low), student (elevated, because mid block is
    scrambled).
 4. Fit a RANK-48 ADAPTER W = A B^T (A: h x 48, B: h x 48) on a 1k C4
    probe bank, minimising ||A @ B^T @ h_student_mid - h_teacher_mid||.
    Training via SVD of residual across probes: cheap, closed-form,
    no gradient backprop needed.
 5. Insert the adapter at the student's mid-layer output: h <- A (B^T h).
 6. Re-measure student NLL with adapter in place.
 7. Fraction gap closed: (NLL_lesion - NLL_patched) / (NLL_lesion - NLL_teacher).

KILL: fraction_gap_closed < 0.05 OR NLL_patched >= NLL_lesion. Means a
rank-48 linear adapter targeting direction-identity alignment cannot
patch a lesioned block back toward the trained behavior.

PASS: fraction_gap_closed >= 0.20 OR bridge rel_err restored to within
2x of teacher baseline. Demonstrates capability surgery via a small
adapter informed by k_bulk bulk-width universal.

Why k=48 (not full rank): because the universal bulk width measured
across 5 trained text systems is k_bulk approximately h/22 = 48 at
h=1024. We test whether this principled rank is enough.
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
from genome_extractor import extract_trajectory  # noqa: E402
from genome_geometry_transfusion import measure_nll  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent


def lesion_midblock(state_dict, prefix, seed=42):
    """Randomize all matmul weights in the given layer prefix, preserving
    per-tensor Frobenius norm and dtype/device."""
    rng = np.random.default_rng(seed)
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
        rnd = rnd * (fro / max(rnd_fro, 1e-6))
        state_dict[k] = rnd.to(W.dtype)
        count += 1
    return count


def fit_low_rank_adapter(X_in, X_target, rank=48):
    """Fit A, B such that X_in @ B @ A^T approximately equals X_target,
    with A in R^{h x rank}, B in R^{h x rank}. Closed form via SVD of
    least-squares solution.

    min ||X_in @ W - X_target||_F  with W = B A^T (rank <= rank).
    Let U, S, Vt = SVD(X_in^+ @ X_target); W = sum top-rank.
    """
    X_in_p = np.linalg.pinv(X_in)
    W_full = X_in_p @ X_target           # (h_in, h_out)
    U, S, Vt = np.linalg.svd(W_full, full_matrices=False)
    r = min(rank, len(S))
    W_lr = U[:, :r] @ np.diag(S[:r]) @ Vt[:r]
    # Also return (B, A) factorization: W_lr = B A^T where
    # B = U @ diag(sqrt(S)), A = Vt^T @ diag(sqrt(S))
    B = U[:, :r] @ np.diag(np.sqrt(S[:r]))
    A = Vt[:r].T @ np.diag(np.sqrt(S[:r]))
    return W_lr, B, A


def extract_mid_activations(sys_obj, sents, mid, tag):
    traj = extract_trajectory(
        model=sys_obj.model, tokenizer=sys_obj.tokenizer,
        texts=sents, layer_indices=[mid], pooling="seq_mean",
        device="cuda", system_key=tag, class_id=1,
        quantization="fp16",
        stimulus_version=tag, seed=42,
        batch_size=16, max_length=256,
    )
    return traj.layers[0].X.astype(np.float32)


class AdapterHook:
    """Forward hook that replaces a layer's output with W_lr @ output."""
    def __init__(self, W_lr):
        self.W = torch.from_numpy(W_lr).float()

    def __call__(self, module, input, output):
        # output may be a tuple (hidden, ...) for transformer blocks
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        W = self.W.to(device=h.device, dtype=h.dtype)
        h_new = h @ W
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new


def main():
    hf_id = "Qwen/Qwen3-0.6B"
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= 500:
            break

    t0 = time.time()

    # (1) Teacher: pretrained Qwen3-0.6B
    print(f"[{time.time()-t0:.1f}s] loading TEACHER (pretrained Qwen3-0.6B)...")
    sys_t = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    mid = sys_t.n_hidden_layers() // 2
    nll_teacher, _ = measure_nll(sys_t.model, sys_t.tokenizer, sents)
    X_teacher_mid = extract_mid_activations(sys_t, sents, mid, "teacher")
    print(f"  teacher NLL={nll_teacher:.3f}, X_teacher shape={X_teacher_mid.shape}")
    sys_t.unload(); torch.cuda.empty_cache()

    # (2) Student: pretrained Qwen3-0.6B with mid block LESIONED
    print(f"[{time.time()-t0:.1f}s] loading STUDENT with lesion at layer {mid}...")
    sys_s = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    sd = sys_s.model.state_dict()
    prefix = f"model.layers.{mid}."
    n_lesioned = lesion_midblock(sd, prefix)
    sys_s.model.load_state_dict(sd, strict=False)
    print(f"  lesioned {n_lesioned} matmul weights in block at layer {mid}")
    nll_lesion, _ = measure_nll(sys_s.model, sys_s.tokenizer, sents)
    X_student_mid = extract_mid_activations(sys_s, sents, mid, "student_lesion")
    print(f"  lesioned NLL={nll_lesion:.3f}, X_student shape={X_student_mid.shape}")

    # (3) Fit rank-48 adapter from student's mid activations to teacher's.
    print(f"[{time.time()-t0:.1f}s] fitting rank-48 adapter...")
    W_lr, B, A = fit_low_rank_adapter(X_student_mid, X_teacher_mid, rank=48)
    # Quality metric: residual variance
    resid = X_student_mid @ W_lr - X_teacher_mid
    r2 = 1.0 - float((resid ** 2).sum() / max((X_teacher_mid ** 2).sum(), 1e-9))
    print(f"  adapter R2 (aligned vs target) = {r2:.3f}")

    # (4) Install the adapter as a forward hook at the mid-layer output.
    print(f"[{time.time()-t0:.1f}s] installing adapter hook at mid layer...")
    hook = AdapterHook(W_lr)
    target_module = sys_s.model.model.layers[mid]
    handle = target_module.register_forward_hook(hook)
    nll_patched, _ = measure_nll(sys_s.model, sys_s.tokenizer, sents)
    handle.remove()
    print(f"  patched NLL={nll_patched:.3f}")

    sys_s.unload(); torch.cuda.empty_cache()

    # (5) Report
    gap_total = nll_lesion - nll_teacher
    gap_closed = nll_lesion - nll_patched
    fraction = gap_closed / max(gap_total, 1e-6)

    print(f"\n=== CAPABILITY PATCH (rank-48 adapter at lesioned layer {mid}) ===")
    print(f"  teacher NLL:     {nll_teacher:.3f}")
    print(f"  lesion NLL:      {nll_lesion:.3f}")
    print(f"  patched NLL:     {nll_patched:.3f}")
    print(f"  fraction_gap_closed: {fraction:+.3f}")
    print(f"  adapter R2:          {r2:.3f}")

    if fraction >= 0.20 and r2 >= 0.5:
        verdict = (f"CAPABILITY_PATCH_LANDS - rank-48 adapter restores "
                   f"{fraction*100:.0f}pct of the lesion NLL gap with R2 "
                   f"{r2:.2f}. Model surgery via rank-48 adapter works.")
    elif fraction >= 0.05:
        verdict = (f"PARTIAL_PATCH - {fraction*100:.0f}pct gap closed. "
                   "Adapter helps but doesn't fully restore capability.")
    else:
        verdict = (f"PATCH_NULL - fraction_gap_closed={fraction:+.3f}, R2="
                   f"{r2:.3f}. Rank-48 linear adapter insufficient for "
                   "mid-block capability restoration.")
    print(f"\n  verdict: {verdict}")

    out = {"purpose": "Codex home-run A2: capability patch via rank-48 adapter (model surgery)",
           "teacher_nll": float(nll_teacher),
           "lesion_nll": float(nll_lesion),
           "patched_nll": float(nll_patched),
           "fraction_gap_closed": float(fraction),
           "adapter_R2": float(r2),
           "lesion_layer": int(mid),
           "n_lesioned_matrices": int(n_lesioned),
           "adapter_rank": 48,
           "verdict": verdict}
    out_path = _ROOT / "results/gate2/capability_patch_k48.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
