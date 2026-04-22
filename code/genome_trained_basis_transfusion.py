"""Forward transfusion using TRAINED FEATURE DIRECTIONS only (no scale/mean transfer).

genome_045 showed that at same rank, PCA-trained-axes preserve capability
much more than random axes. This asks the reverse: if we FORCE untrained
activations to live in the trained top-k PCA subspace (per-token projection
onto trained basis V), do we get capability back?

If NLL drops substantially -> directions are the transferable substrate.
If NLL stays at untrained-level -> capability requires the specific trained
activation VALUES at those axes, not just the axes themselves (i.e., needs
weight transfer, not activation post-processing).

Protocol (DeepSeek-R1-Distill-Qwen-1.5B):
  1. Extract trained mid-depth activations -> PCA basis V_trained.
  2. Hook untrained mid-depth: project each token's activation onto V_trained
     top-k (for k in {64, 256, 512, 1024}), reconstruct.
  3. Measure NLL per k + geometry (d_rd, c).
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
    n = 500
    t0 = time.time()

    sents = []
    for rec in c4_clean_v1(seed=seed, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= n:
            break
    print(f"[{time.time()-t0:.1f}s] {len(sents)} stim")

    # Extract trained PCA basis
    print(f"[{time.time()-t0:.1f}s] loading TRAINED to build V_trained...")
    sys_t = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    n_layers = sys_t.n_hidden_layers()
    mid = n_layers // 2
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

    token_acts = []
    def cap_hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        token_acts.append(h.detach().to(torch.float32).cpu())
    h_cap = blocks[mid].register_forward_hook(cap_hook)
    try:
        with torch.no_grad():
            for i in range(0, len(sents), 16):
                chunk = sents[i:i + 16]
                enc = sys_t.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256)
                _ = sys_t.model(input_ids=enc["input_ids"].to("cuda"), attention_mask=enc["attention_mask"].to("cuda"))
    finally:
        h_cap.remove()

    all_toks = torch.cat([t.reshape(-1, t.shape[-1]) for t in token_acts], dim=0).numpy()
    mu_trained = all_toks.mean(axis=0)
    Xc = all_toks - mu_trained
    rng = np.random.default_rng(seed)
    if Xc.shape[0] > 20000:
        Xc = Xc[rng.choice(Xc.shape[0], 20000, replace=False)]
    print(f"[{time.time()-t0:.1f}s] SVD on {Xc.shape[0]} trained tokens...")
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V_full = torch.from_numpy(Vt).to("cuda").float()
    mu_tr_t = torch.from_numpy(mu_trained).to("cuda").float()
    h = Xc.shape[1]
    print(f"[{time.time()-t0:.1f}s] V_trained shape {V_full.shape}")
    sys_t.unload(); torch.cuda.empty_cache()

    # Load untrained
    print(f"[{time.time()-t0:.1f}s] loading UNTRAINED...")
    sys_u = load_system(hf_id, quant="fp16", untrained=True, device="cuda")
    blocks_u = None
    for path in ("model.layers", "transformer.h"):
        obj = sys_u.model
        ok = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False; break
        if ok:
            blocks_u = obj; break

    nll_untrained, _ = measure_nll(sys_u.model, sys_u.tokenizer, sents)
    print(f"  UNTRAINED baseline NLL = {nll_untrained:.4f}")

    results = []
    for k_test in (64, 256, 512, 1024):
        V_k = V_full[:k_test]

        def proj_hook(module, inputs, output, V_k=V_k, mu_tr_t=mu_tr_t):
            hh = output[0] if isinstance(output, tuple) else output
            od = hh.dtype
            h32 = hh.to(torch.float32)
            # Project onto trained top-k basis, reconstruct, shift mean to trained mean
            # Note: we do NOT subtract untrained mean — we are forcing the activation
            # into the trained-basis span AND trained-mean location
            hc = h32 - h32.mean(dim=(0, 1), keepdim=True)  # center per batch
            coeffs = hc @ V_k.T
            recon = coeffs @ V_k + mu_tr_t
            y = recon.to(od)
            if isinstance(output, tuple):
                return (y,) + output[1:]
            return y

        handle = blocks_u[mid].register_forward_hook(proj_hook)
        try:
            nll_k, _ = measure_nll(sys_u.model, sys_u.tokenizer, sents)
            traj_k = extract_trajectory(
                model=sys_u.model, tokenizer=sys_u.tokenizer,
                texts=sents, layer_indices=[mid], pooling="seq_mean",
                device="cuda", system_key=sk + f"_basis_k{k_test}", class_id=2,
                quantization="fp16",
                stimulus_version=f"basis_k{k_test}.seed{seed}.n{n}", seed=seed,
                batch_size=16, max_length=256,
            )
            X_k = traj_k.layers[0].X.astype(np.float32)
            Cs_k = [float(knn_clustering_coefficient(X_k, k=kk).value) for kk in K_GRID]
            p_k, _, _ = fit_power_law(K_GRID, Cs_k)
            rd_k = rate_distortion_dim(X_k)
            c_k = p_k * rd_k["d_rd"]
            drop_rel = (nll_untrained - nll_k) / nll_untrained
            print(f"  k={k_test}: p={p_k:.3f} d_rd={rd_k['d_rd']:.2f} c={c_k:.2f} NLL={nll_k:.3f} rel_drop={100*drop_rel:+.1f}%")
            results.append({"k": k_test, "p": p_k, "d_rd": rd_k["d_rd"],
                            "c": c_k, "NLL": nll_k, "rel_drop_pct": 100 * drop_rel})
        finally:
            handle.remove()

    sys_u.unload(); torch.cuda.empty_cache()

    # Verdict
    best_drop = max((r["rel_drop_pct"] for r in results), default=0)
    if best_drop > 20:
        verdict = "BASIS_TRANSFER_WORKS — trained directions recover substantial capability"
    elif best_drop > 5:
        verdict = "BASIS_TRANSFER_PARTIAL — trained directions give marginal capability"
    else:
        verdict = "BASIS_TRANSFER_FAILS — directions alone insufficient, need trained activation values at those axes"
    print(f"\n  verdict: {verdict}")
    print(f"  best NLL drop across k = {best_drop:+.1f}%")

    out = {"purpose": "Forward transfusion of TRAINED directions only",
           "untrained_NLL": nll_untrained, "per_k": results, "verdict": verdict}
    out_path = _ROOT / "results/gate2/trained_basis_transfusion.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
