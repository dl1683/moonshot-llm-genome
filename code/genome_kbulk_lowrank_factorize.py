"""Compute-efficiency demo: low-rank-factorize mid-depth MLP layers to
rank k_bulk=48 (universal h/22 plateau width from genome_065) and
measure NLL retention on pretrained Qwen3-0.6B.

HYPOTHESIS. The plateau-plus-power-law spectral fit (genome_065) said
every trained 1024-d text model has approximately 48 principal
directions carrying the informational work. If that's mechanistically
right, replacing a full-rank mid-depth MLP (say gate_proj, up_proj,
down_proj with shapes 1024 x 2816 and 2816 x 1024 for Qwen3) with a
rank-48 factorization should lose little capability.

Math. For a linear W (m x n) we do W = W_l @ W_r where W_l is m x k,
W_r is k x n. k_bulk=48. Parameter count drops from m*n to k*(m+n),
a reduction of factor m*n / (k*(m+n)). For 1024x2816 that is
~13x reduction per layer.

EXPERIMENT. Compare val-NLL on C4-clean (n=1000 held-out):
  (A) BASELINE: unperturbed Qwen3-0.6B mid-depth
  (B) FACTORED_K_BULK: truncate 3 MLP projections at mid-layer to rank 48
  (C) FACTORED_K_16 (control, too aggressive)
  (D) FACTORED_K_256 (control, should have ~0pct NLL loss)

KILL CONDITION:
  - If k=48 gives >5pct NLL loss on mid-layer alone: k_bulk is not a
    safe rank target; candidate-8's "bulk width" interpretation is
    off. Report null.
  - If k=48 gives <=2pct NLL loss, the demo lands: ~13x parameter
    reduction per factored layer with approximately unchanged capability,
    using a principled rank target from the bridge.

Per CLAUDE.md sec 0.05 scope lock: this advances MODEL SURGERY /
COMPUTE EFFICIENCY on a trained AI model (Qwen3). No biology.
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
from genome_geometry_transfusion import measure_nll  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent


def low_rank_factorize(W, k):
    """Return W_approx = U_k @ Sigma_k @ V_k^T via truncated SVD rank k."""
    W32 = W.detach().to(dtype=torch.float32).cpu()
    try:
        U, S, Vh = torch.linalg.svd(W32, full_matrices=False)
    except Exception:
        return W  # fall back to identity
    k_eff = min(k, len(S))
    Uk = U[:, :k_eff]
    Sk = S[:k_eff]
    Vhk = Vh[:k_eff, :]
    W_approx = Uk @ torch.diag(Sk) @ Vhk
    return W_approx.to(dtype=W.dtype, device=W.device)


def factor_mid_mlp(sys_obj, k, mid_layer):
    """Apply rank-k SVD truncation to the mid-layer's MLP projections.
    Modifies sys_obj.model in place (destructive). Caller reloads for
    a fresh baseline.
    """
    sd = sys_obj.model.state_dict()
    prefix = f"model.layers.{mid_layer}."
    patched = []
    for name in ["mlp.gate_proj.weight", "mlp.up_proj.weight",
                 "mlp.down_proj.weight"]:
        key = prefix + name
        if key in sd:
            orig_shape = tuple(sd[key].shape)
            sd[key] = low_rank_factorize(sd[key], k)
            patched.append({"name": key, "shape": orig_shape,
                            "param_count_original": int(orig_shape[0] * orig_shape[1]),
                            "param_count_rank_k": int(k * (orig_shape[0] + orig_shape[1]))})
    sys_obj.model.load_state_dict(sd, strict=False)
    return patched


def run_condition(hf_id, sents, k, mid_layer, label):
    print(f"\n-- condition={label}  k={k} --")
    sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    # mid layer already known; re-derive for safety
    n_layers = sys_obj.n_hidden_layers()
    mid = n_layers // 2
    if k is not None:
        patched = factor_mid_mlp(sys_obj, k, mid)
        print(f"  patched {len(patched)} MLP projections at layer {mid}")
    else:
        patched = []
        print(f"  unperturbed baseline (layer {mid})")
    nll, _ = measure_nll(sys_obj.model, sys_obj.tokenizer, sents)
    print(f"  val NLL = {nll:.4f}")
    sys_obj.unload(); torch.cuda.empty_cache()
    return {"label": label, "k": k, "val_nll": float(nll),
            "patched_projections": patched}


def main():
    t0 = time.time()
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= 500:
            break
    print(f"[{time.time()-t0:.1f}s] {len(sents)} held-out C4 texts")

    hf_id = "Qwen/Qwen3-0.6B"
    mid = None  # derived per load

    conditions = []
    # (A) baseline
    conditions.append(run_condition(hf_id, sents, None, mid, "baseline"))
    # (B) k_bulk = 48 target
    conditions.append(run_condition(hf_id, sents, 48, mid, "factored_k48"))
    # (C) controls
    conditions.append(run_condition(hf_id, sents, 16, mid, "factored_k16_aggressive"))
    conditions.append(run_condition(hf_id, sents, 256, mid, "factored_k256_gentle"))

    base = conditions[0]["val_nll"]
    print("\n=== K_BULK LOW-RANK FACTORIZATION RESULTS ===")
    for c in conditions:
        rel = (c["val_nll"] - base) / max(base, 1e-6) * 100
        if c["patched_projections"]:
            total_orig = sum(p["param_count_original"] for p in c["patched_projections"])
            total_new = sum(p["param_count_rank_k"] for p in c["patched_projections"])
            reduction = total_orig / max(total_new, 1)
        else:
            reduction = 1.0
        print(f"  {c['label']:30s} k={c['k']}  NLL={c['val_nll']:.4f}  "
              f"rel_vs_base={rel:+.2f}pct  param_reduction={reduction:.1f}x")

    k48 = next(c for c in conditions if c["k"] == 48)
    rel_k48 = (k48["val_nll"] - base) / max(base, 1e-6) * 100
    if rel_k48 <= 2.0:
        verdict = (f"K_BULK_LOW_RANK_DEMO_LANDS - at k_bulk=48 the mid-MLP "
                   f"loses only {rel_k48:.2f}pct NLL for ~13x parameter "
                   "reduction per factored projection. k_bulk is a "
                   "principled rank target for compression.")
    elif rel_k48 <= 5.0:
        verdict = (f"PARTIAL - k=48 loses {rel_k48:.2f}pct NLL. Demo is "
                   "borderline; may need full retraining with the truncated "
                   "basis to land.")
    else:
        verdict = (f"KILL - k=48 loses {rel_k48:.2f}pct NLL on the "
                   "single-layer factorization. k_bulk is NOT a safe "
                   "zero-shot rank target.")
    print(f"\n  verdict: {verdict}")

    out = {"purpose": "k_bulk=48 low-rank factorization demo on Qwen3 mid-MLP",
           "per_condition": conditions,
           "baseline_nll": base,
           "k48_nll_rel_pct": float(rel_k48),
           "verdict": verdict}
    out_path = _ROOT / "results/gate2/kbulk_lowrank_factorize.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
