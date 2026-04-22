"""Reverse-direction causal test: PCA-compress trained activations at mid-depth
and measure d_rd + NLL as a function of the retained rank k.

If geometry is CAUSAL for capability, we expect a joint curve where d_rd
drops and NLL climbs together as k shrinks. If geometry is only a correlate,
d_rd can drop a lot while NLL stays low (capability survives small rank).

Protocol (DeepSeek-R1-Distill-Qwen-1.5B, trained, C4-clean n=1000 seed 42):
  1. Forward once, extract mid-depth activations X (b*s, h).
  2. Compute PCA basis from X (top-h components).
  3. For k in {1024(full), 512, 256, 128, 64, 32, 16, 8, 4}:
     - Hook mid-depth: project activation onto top-k PCA subspace, reconstruct,
       hook-apply during inference.
     - Measure d_rd on the reconstructed pooled cloud + NLL on the same stim.
  4. Plot joint curve d_rd-vs-NLL as k varies.

If NLL climbs cleanly with d_rd drop across all k, geometry is causal for
capability. If NLL is flat until k < some threshold, d_rd has a "capability
cliff" at specific rank.
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
from genome_geometry_transfusion import measure_nll, fit_power_law  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from genome_primitives import knn_clustering_coefficient  # noqa: E402
from genome_rate_distortion_probe import rate_distortion_dim  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]


def main():
    hf_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    sk = "deepseek-r1-distill-qwen-1.5b"
    seed = 42
    n = 500  # faster — this is 9 PCA points
    t0 = time.time()

    sents = []
    for rec in c4_clean_v1(seed=seed, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= n:
            break
    print(f"[{time.time()-t0:.1f}s] {len(sents)} C4 stimuli")

    print(f"[{time.time()-t0:.1f}s] loading TRAINED {sk}...")
    sys_t = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    n_layers = sys_t.n_hidden_layers()
    mid = n_layers // 2

    # Extract token-level activations for PCA basis
    token_acts = []
    def cap_hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        token_acts.append(h.detach().to(torch.float32).cpu())
    blocks = None
    for path in ("model.layers", "transformer.h"):
        obj = sys_t.model
        ok = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False; break
        if ok:
            blocks = obj; break
    h_t = blocks[mid].register_forward_hook(cap_hook)
    try:
        with torch.no_grad():
            for i in range(0, len(sents), 16):
                chunk = sents[i:i + 16]
                enc = sys_t.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256)
                _ = sys_t.model(input_ids=enc["input_ids"].to("cuda"), attention_mask=enc["attention_mask"].to("cuda"))
    finally:
        h_t.remove()

    all_toks = torch.cat([t.reshape(-1, t.shape[-1]) for t in token_acts], dim=0).numpy()
    mu_tok = all_toks.mean(axis=0)
    X_c = all_toks - mu_tok
    # SVD basis — use truncated since full h=1536 is big
    # subsample for speed
    rng = np.random.default_rng(seed)
    if X_c.shape[0] > 20000:
        idx = rng.choice(X_c.shape[0], 20000, replace=False)
        X_c_sub = X_c[idx]
    else:
        X_c_sub = X_c
    print(f"[{time.time()-t0:.1f}s] PCA SVD on {X_c_sub.shape[0]} tokens, h={X_c_sub.shape[1]}...")
    U, S, Vt = np.linalg.svd(X_c_sub, full_matrices=False)
    # Vt rows are principal components, shape (h, h)
    print(f"[{time.time()-t0:.1f}s] PCA done. top singular values: {S[:5]}")

    mu_t = torch.from_numpy(mu_tok).to("cuda").float()
    V_full = torch.from_numpy(Vt).to("cuda").float()  # (h, h), rows are PCs

    nll_full, _ = measure_nll(sys_t.model, sys_t.tokenizer, sents)
    print(f"  TRAINED full (no hook) NLL = {nll_full:.4f}")
    traj_full = extract_trajectory(
        model=sys_t.model, tokenizer=sys_t.tokenizer,
        texts=sents, layer_indices=[mid], pooling="seq_mean",
        device="cuda", system_key=sk + "_full", class_id=2, quantization="fp16",
        stimulus_version=f"pca_full.seed{seed}.n{n}", seed=seed,
        batch_size=16, max_length=256,
    )
    X_full = traj_full.layers[0].X.astype(np.float32)
    Cs_full = [float(knn_clustering_coefficient(X_full, k=kk).value) for kk in K_GRID]
    p_full, _, _ = fit_power_law(K_GRID, Cs_full)
    rd_full = rate_distortion_dim(X_full)
    c_full = p_full * rd_full["d_rd"]
    print(f"  TRAINED full: p={p_full:.3f}  d_rd={rd_full['d_rd']:.2f}  c={c_full:.2f}")

    # Sweep k
    ks_sweep = [4, 8, 16, 32, 64, 128, 256, 512]
    results = []
    for k_pca in ks_sweep:
        print(f"\n[{time.time()-t0:.1f}s] k_pca = {k_pca}...")
        V_k = V_full[:k_pca]  # (k_pca, h)

        def proj_hook(module, inputs, output, V_k=V_k, mu_t=mu_t):
            h = output[0] if isinstance(output, tuple) else output
            od = h.dtype
            h32 = h.to(torch.float32)
            # Center, project to top-k, reconstruct
            hc = h32 - mu_t
            coeffs = hc @ V_k.T  # (b, s, k)
            recon = coeffs @ V_k + mu_t  # (b, s, h)
            y = recon.to(od)
            if isinstance(output, tuple):
                return (y,) + output[1:]
            return y

        handle = blocks[mid].register_forward_hook(proj_hook)
        try:
            nll_k, _ = measure_nll(sys_t.model, sys_t.tokenizer, sents)
            traj_k = extract_trajectory(
                model=sys_t.model, tokenizer=sys_t.tokenizer,
                texts=sents, layer_indices=[mid], pooling="seq_mean",
                device="cuda", system_key=sk + f"_pca{k_pca}", class_id=2,
                quantization="fp16",
                stimulus_version=f"pca{k_pca}.seed{seed}.n{n}", seed=seed,
                batch_size=16, max_length=256,
            )
            X_k = traj_k.layers[0].X.astype(np.float32)
            Cs_k = [float(knn_clustering_coefficient(X_k, k=kk).value) for kk in K_GRID]
            p_k, _, r2_k = fit_power_law(K_GRID, Cs_k)
            rd_k = rate_distortion_dim(X_k)
            c_k = p_k * rd_k["d_rd"]
            rel_nll = (nll_k - nll_full) / nll_full
            print(f"  k={k_pca}: p={p_k:.3f}  d_rd={rd_k['d_rd']:.2f}  c={c_k:.2f}  NLL={nll_k:.3f}  rel_dNLL={100*rel_nll:+.1f}%")
            results.append({
                "k_pca": k_pca, "p": p_k, "d_rd": rd_k["d_rd"], "c": c_k,
                "R2_ck": r2_k, "NLL": nll_k, "rel_dNLL_pct": 100 * rel_nll,
            })
        finally:
            handle.remove()

    sys_t.unload(); torch.cuda.empty_cache()

    out = {
        "purpose": "PCA-compression sweep on trained model — causal direction test",
        "trained_full": {"p": p_full, "d_rd": rd_full["d_rd"], "c": c_full, "NLL": nll_full},
        "pca_sweep": results,
    }
    out_path = _ROOT / "results/gate2/pca_sweep_causal.json"
    out_path.write_text(json.dumps(out, indent=2))

    # Concise summary
    print(f"\n=== CAUSAL PCA SWEEP SUMMARY ===")
    print(f"  {'k_pca':>6s} {'d_rd':>8s} {'c':>7s} {'NLL':>8s} {'rel_dNLL%':>10s}")
    print(f"  {'full':>6s} {rd_full['d_rd']:8.2f} {c_full:7.2f} {nll_full:8.3f} {'(baseline)':>10s}")
    for r in results:
        print(f"  {r['k_pca']:6d} {r['d_rd']:8.2f} {r['c']:7.2f} {r['NLL']:8.3f} {r['rel_dNLL_pct']:+10.1f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
