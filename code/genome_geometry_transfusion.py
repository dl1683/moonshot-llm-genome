"""Rung-3 paradigm-shift experiment: GEOMETRY TRANSFUSION.

Hypothesis: trained-manifold representational geometry is not just descriptive
but causally generative of capability. If we can TRANSPLANT a trained model's
geometric structure into a random-init model (without any gradient updates)
and observe capability emergence, geometry is the substrate.

Protocol (DeepSeek-R1-Distill-Qwen-1.5B, text, C4-clean n=1000 seed 42):

1. Load TRAINED model. Extract mid-depth pooled activations on 1000 C4
   sentences -> X_trained (n, h).
2. Load UNTRAINED twin (same arch, random weights). Extract mid-depth
   activations on the SAME 1000 sentences -> X_untrained.
3. Compute transfusion operator: whiten(X_untrained) then recolor with
   covariance of X_trained:
       T(x) = mean_trained + L_trained @ L_untrained^{-1} @ (x - mean_untrained)
   where L_* is the Cholesky factor of Cov_*.
4. Apply T inside the untrained model via a forward hook at mid-depth (on
   the untrained model's hidden state, post-layernorm).
5. Measure:
   (a) Geometry: compute p, d_rd, c on the transfused mid-depth cloud.
       Check if these shift toward the trained band (p~0.17, d_rd~14, c~2.4).
   (b) NLL on same 1000 stimuli, transfused forward pass vs untrained
       baseline. Check if NLL drops substantially.

Outcomes:
  - GEOMETRY + CAPABILITY SHIFT: both (a) moves toward trained AND (b) drops
    NLL substantially (>= 5 percent relative drop) -> paradigm-shift landing.
  - GEOMETRY-ONLY: (a) moves toward trained, (b) does not help NLL -> geometry
    is a correlate, not the causal substrate.
  - NEITHER: transfusion operator implementation wrong OR invariant is higher
    order than 2nd-moment (covariance) structure.
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
from genome_loaders import load_system  # noqa: E402
from genome_primitives import knn_clustering_coefficient  # noqa: E402
from genome_rate_distortion_probe import rate_distortion_dim, fit_power_law  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]


def measure_nll(model, tokenizer, texts, device="cuda", max_length=128, batch_size=16):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attn == 0] = -100
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            ntok = (labels != -100).sum().item()
            total_nll += float(out.loss.item()) * ntok
            total_tokens += ntok
    return total_nll / max(total_tokens, 1), total_tokens


def compute_transfusion_op(X_src: np.ndarray, X_dst: np.ndarray):
    """Compute affine operator T that maps X_src distribution to X_dst distribution.

    Whiten X_src, recolor with X_dst covariance.
    Returns (mean_src, L_src_inv, L_dst, mean_dst) as torch tensors for hook use.
    """
    mu_src = X_src.mean(axis=0)
    mu_dst = X_dst.mean(axis=0)
    Xc_src = X_src - mu_src
    Xc_dst = X_dst - mu_dst
    cov_src = Xc_src.T @ Xc_src / (X_src.shape[0] - 1)
    cov_dst = Xc_dst.T @ Xc_dst / (X_dst.shape[0] - 1)
    # Regularize slightly for numerical stability
    eps = 1e-4
    cov_src += eps * np.eye(cov_src.shape[0])
    cov_dst += eps * np.eye(cov_dst.shape[0])
    L_src = np.linalg.cholesky(cov_src)
    L_dst = np.linalg.cholesky(cov_dst)
    L_src_inv = np.linalg.solve(L_src, np.eye(L_src.shape[0]))
    return mu_src, L_src_inv, L_dst, mu_dst


def transfuse(x: torch.Tensor, mu_src, L_src_inv, L_dst, mu_dst):
    """x shape (batch, seq, h) — apply affine transfusion per token."""
    orig_dtype = x.dtype
    x32 = x.to(torch.float32)
    # whiten
    w = (x32 - mu_src) @ L_src_inv.T
    # recolor
    y = w @ L_dst.T + mu_dst
    return y.to(orig_dtype)


def main():
    hf_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    sk = "deepseek-r1-distill-qwen-1.5b"
    seed = 42
    n = 1000
    t0 = time.time()

    sents = []
    for rec in c4_clean_v1(seed=seed, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= n:
            break
    print(f"[{time.time()-t0:.1f}s] {len(sents)} C4 stimuli")

    # Step 1: trained extract
    print(f"[{time.time()-t0:.1f}s] loading TRAINED {sk}...")
    sys_t = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    n_layers = sys_t.n_hidden_layers()
    mid = n_layers // 2
    traj_t = extract_trajectory(
        model=sys_t.model, tokenizer=sys_t.tokenizer,
        texts=sents, layer_indices=[mid], pooling="seq_mean",
        device="cuda", system_key=sk, class_id=2, quantization="fp16",
        stimulus_version=f"c4_clean.v1.seed{seed}.n{n}", seed=seed,
        batch_size=16, max_length=256,
    )
    X_trained = traj_t.layers[0].X.astype(np.float32)
    # Trained baseline metrics
    Cs_t = [float(knn_clustering_coefficient(X_trained, k=k).value) for k in K_GRID]
    p_t, c0_t, r2_t = fit_power_law(K_GRID, Cs_t)
    rd_t = rate_distortion_dim(X_trained)
    c_trained = p_t * rd_t["d_rd"]
    print(f"  TRAINED    p={p_t:.3f}  d_rd={rd_t['d_rd']:.2f}  c={c_trained:.2f}")
    nll_trained, _ = measure_nll(sys_t.model, sys_t.tokenizer, sents)
    print(f"  TRAINED NLL = {nll_trained:.4f}")
    sys_t.unload(); torch.cuda.empty_cache()

    # Step 2: untrained extract
    print(f"[{time.time()-t0:.1f}s] loading UNTRAINED {sk}...")
    sys_u = load_system(hf_id, quant="fp16", untrained=True, device="cuda")
    traj_u = extract_trajectory(
        model=sys_u.model, tokenizer=sys_u.tokenizer,
        texts=sents, layer_indices=[mid], pooling="seq_mean",
        device="cuda", system_key=sk, class_id=2, quantization="fp16",
        stimulus_version=f"c4_clean.v1.seed{seed}.n{n}", seed=seed,
        batch_size=16, max_length=256,
    )
    X_untrained = traj_u.layers[0].X.astype(np.float32)
    Cs_u = [float(knn_clustering_coefficient(X_untrained, k=k).value) for k in K_GRID]
    p_u, c0_u, r2_u = fit_power_law(K_GRID, Cs_u)
    rd_u = rate_distortion_dim(X_untrained)
    c_untrained = p_u * rd_u["d_rd"]
    print(f"  UNTRAINED  p={p_u:.3f}  d_rd={rd_u['d_rd']:.2f}  c={c_untrained:.2f}")
    nll_untrained, _ = measure_nll(sys_u.model, sys_u.tokenizer, sents)
    print(f"  UNTRAINED NLL = {nll_untrained:.4f}")

    # Step 3: transfusion op (map untrained -> trained distribution)
    print(f"[{time.time()-t0:.1f}s] computing transfusion operator (whiten+recolor)...")
    mu_src, L_src_inv, L_dst, mu_dst = compute_transfusion_op(X_untrained, X_trained)
    dev = "cuda"
    mu_src_t = torch.from_numpy(mu_src).to(dev).float()
    L_src_inv_t = torch.from_numpy(L_src_inv).to(dev).float()
    L_dst_t = torch.from_numpy(L_dst).to(dev).float()
    mu_dst_t = torch.from_numpy(mu_dst).to(dev).float()

    # Step 4: hook-apply T inside untrained model at mid-depth
    print(f"[{time.time()-t0:.1f}s] installing transfusion hook on layer {mid}...")

    def _transformer_blocks(model):
        for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
            obj = model
            ok = True
            for attr in path.split("."):
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    ok = False; break
            if ok:
                return obj
        raise RuntimeError("cannot find transformer blocks")

    blocks = _transformer_blocks(sys_u.model)
    hook_handle = None

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        y = transfuse(h, mu_src_t, L_src_inv_t, L_dst_t, mu_dst_t)
        if isinstance(output, tuple):
            return (y,) + output[1:]
        return y

    hook_handle = blocks[mid].register_forward_hook(hook)

    try:
        # Step 5a: measure transfused geometry via fresh extraction
        traj_trans = extract_trajectory(
            model=sys_u.model, tokenizer=sys_u.tokenizer,
            texts=sents, layer_indices=[mid], pooling="seq_mean",
            device="cuda", system_key=sk + "_transfused", class_id=2,
            quantization="fp16",
            stimulus_version=f"transfused.seed{seed}.n{n}", seed=seed,
            batch_size=16, max_length=256,
        )
        X_trans = traj_trans.layers[0].X.astype(np.float32)
        Cs_tr = [float(knn_clustering_coefficient(X_trans, k=k).value) for k in K_GRID]
        p_tr, c0_tr, r2_tr = fit_power_law(K_GRID, Cs_tr)
        rd_tr = rate_distortion_dim(X_trans)
        c_trans = p_tr * rd_tr["d_rd"]
        print(f"  TRANSFUSED p={p_tr:.3f}  d_rd={rd_tr['d_rd']:.2f}  c={c_trans:.2f}")

        # Step 5b: measure transfused NLL
        nll_trans, _ = measure_nll(sys_u.model, sys_u.tokenizer, sents)
        print(f"  TRANSFUSED NLL = {nll_trans:.4f}")
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    sys_u.unload(); torch.cuda.empty_cache()

    # Verdict
    p_shift_fraction = ((p_u - p_tr) / (p_u - p_t)) if abs(p_u - p_t) > 1e-6 else 0.0
    c_shift_fraction = ((c_untrained - c_trans) / (c_untrained - c_trained)) if abs(c_untrained - c_trained) > 1e-6 else 0.0
    nll_drop_rel = (nll_untrained - nll_trans) / nll_untrained

    print(f"\n=== VERDICT ===")
    print(f"  p:    untrained={p_u:.3f}  transfused={p_tr:.3f}  trained={p_t:.3f}  "
          f"fraction_shift_toward_trained={p_shift_fraction:.2f}")
    print(f"  c:    untrained={c_untrained:.2f}  transfused={c_trans:.2f}  trained={c_trained:.2f}  "
          f"fraction_shift_toward_trained={c_shift_fraction:.2f}")
    print(f"  NLL:  untrained={nll_untrained:.3f}  transfused={nll_trans:.3f}  "
          f"trained={nll_trained:.3f}")
    print(f"        relative drop untrained->transfused = {100*nll_drop_rel:+.1f}%")

    if c_shift_fraction > 0.5 and nll_drop_rel > 0.05:
        verdict = "GEOMETRY_AND_CAPABILITY_SHIFT_paradigm_indicator"
    elif c_shift_fraction > 0.5:
        verdict = "GEOMETRY_ONLY_shift_no_capability"
    elif nll_drop_rel > 0.05:
        verdict = "CAPABILITY_SHIFT_WITHOUT_CLEAR_GEOMETRY_shift"
    else:
        verdict = "NEITHER_shift_operator_may_be_wrong"
    print(f"  verdict: {verdict}")

    out = {
        "purpose": "Geometry transfusion — does trained geometry transplanted into untrained cause capability?",
        "trained": {"p": p_t, "d_rd": rd_t["d_rd"], "c": c_trained, "NLL": nll_trained},
        "untrained": {"p": p_u, "d_rd": rd_u["d_rd"], "c": c_untrained, "NLL": nll_untrained},
        "transfused": {"p": p_tr, "d_rd": rd_tr["d_rd"], "c": c_trans, "NLL": nll_trans},
        "shifts": {
            "p_fraction_toward_trained": p_shift_fraction,
            "c_fraction_toward_trained": c_shift_fraction,
            "nll_relative_drop": nll_drop_rel,
        },
        "verdict": verdict,
    }
    out_path = _ROOT / "results/gate2/geometry_transfusion.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
