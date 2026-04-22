"""Stronger transfusion variant: k-means codebook transplant.

genome_042 showed that covariance-level transfusion moves geometry 61% of
the way toward trained but doesn't transfer capability. The null tells us
the invariant is higher-order than 2nd moments. This iteration tests
a nonlinear piecewise transfer: fit k-means on trained activations, snap
untrained activations to the nearest trained codebook entry at inference.

Protocol (DeepSeek-R1-Distill-Qwen-1.5B):

1. Extract mid-depth activations on 1000 C4 stimuli from TRAINED model.
   Fit k-means with K=256 codebook entries -> centroids (256, h).
2. Hook mid-depth of UNTRAINED twin: at forward time, for each token
   position's activation vector, find nearest centroid in L2 and replace
   the activation with that centroid.
3. Measure (a) geometry on the per-token-snapped cloud, (b) NLL with
   snapped activations flowing to downstream layers.

Result categories:
  - CODEBOOK + CAPABILITY SHIFT: NLL drops >= 5 percent relative.
  - CODEBOOK GEOMETRY-ONLY: geometry moves toward trained but NLL stays high.
  - NEITHER.
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
    n = 1000
    K = 256  # codebook size
    t0 = time.time()

    sents = []
    for rec in c4_clean_v1(seed=seed, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= n:
            break
    print(f"[{time.time()-t0:.1f}s] {len(sents)} C4 stimuli")

    # Step 1: extract trained activations including token-level (not just pooled)
    print(f"[{time.time()-t0:.1f}s] loading TRAINED, extracting token-level hidden states at mid layer...")
    sys_t = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    n_layers = sys_t.n_hidden_layers()
    mid = n_layers // 2

    # We need TOKEN-level activations to build the codebook. Hook captures per-token.
    token_acts = []
    def cap_hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        # h: (batch, seq, h); take non-pad tokens — we'll filter later if needed
        token_acts.append(h.detach().to(torch.float32).cpu())

    blocks_t = None
    for path in ("model.layers", "transformer.h"):
        obj = sys_t.model
        ok = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False; break
        if ok:
            blocks_t = obj; break
    handle_t = blocks_t[mid].register_forward_hook(cap_hook)
    try:
        with torch.no_grad():
            for i in range(0, len(sents), 16):
                chunk = sents[i:i + 16]
                enc = sys_t.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256)
                _ = sys_t.model(input_ids=enc["input_ids"].to("cuda"), attention_mask=enc["attention_mask"].to("cuda"))
    finally:
        handle_t.remove()

    # Concatenate and subsample
    all_toks = torch.cat([t.reshape(-1, t.shape[-1]) for t in token_acts], dim=0).numpy()
    # subsample to ≤50000 tokens for k-means speed
    rng = np.random.default_rng(seed)
    if all_toks.shape[0] > 50000:
        idx = rng.choice(all_toks.shape[0], 50000, replace=False)
        all_toks = all_toks[idx]
    print(f"[{time.time()-t0:.1f}s] {all_toks.shape[0]} tokens for codebook fit")

    # Trained baseline NLL
    nll_trained, _ = measure_nll(sys_t.model, sys_t.tokenizer, sents)
    print(f"  TRAINED NLL = {nll_trained:.4f}")
    sys_t.unload(); torch.cuda.empty_cache()

    # Step 2: fit k-means
    print(f"[{time.time()-t0:.1f}s] fitting k-means K={K}...")
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=K, n_init=3, random_state=seed)
    km.fit(all_toks)
    centroids = km.cluster_centers_.astype(np.float32)  # (K, h)
    print(f"[{time.time()-t0:.1f}s] codebook ready, inertia={km.inertia_:.0f}")

    # Step 3: load untrained, install snap hook
    print(f"[{time.time()-t0:.1f}s] loading UNTRAINED + installing snap hook...")
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

    centroids_t = torch.from_numpy(centroids).to("cuda")  # float32
    # Precompute centroid norms for efficient nearest-neighbor: ||c||^2
    c_sq = (centroids_t * centroids_t).sum(dim=1)  # (K,)

    # Untrained baseline NLL first
    nll_untrained, _ = measure_nll(sys_u.model, sys_u.tokenizer, sents)
    print(f"  UNTRAINED NLL (pre-snap) = {nll_untrained:.4f}")

    def snap_hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        orig_dtype = h.dtype
        h32 = h.to(torch.float32)  # (b, s, h)
        # For each token, find nearest centroid
        # distances^2 = ||x||^2 + ||c||^2 - 2 x·c
        x = h32.reshape(-1, h32.shape[-1])  # (b*s, h)
        x_sq = (x * x).sum(dim=1, keepdim=True)  # (b*s, 1)
        xc = x @ centroids_t.T  # (b*s, K)
        d2 = x_sq + c_sq.unsqueeze(0) - 2 * xc  # (b*s, K)
        idx = d2.argmin(dim=1)  # (b*s,)
        snapped = centroids_t[idx].reshape(h32.shape).to(orig_dtype)
        if isinstance(output, tuple):
            return (snapped,) + output[1:]
        return snapped

    handle_u = blocks_u[mid].register_forward_hook(snap_hook)
    try:
        # Measure snapped-geometry via fresh extraction
        print(f"[{time.time()-t0:.1f}s] measuring snapped geometry + NLL...")
        traj_snap = extract_trajectory(
            model=sys_u.model, tokenizer=sys_u.tokenizer,
            texts=sents, layer_indices=[mid], pooling="seq_mean",
            device="cuda", system_key=sk + "_snapped", class_id=2,
            quantization="fp16",
            stimulus_version=f"snap.seed{seed}.n{n}", seed=seed,
            batch_size=16, max_length=256,
        )
        X_snap = traj_snap.layers[0].X.astype(np.float32)
        Cs_s = [float(knn_clustering_coefficient(X_snap, k=k).value) for k in K_GRID]
        p_s, c0_s, r2_s = fit_power_law(K_GRID, Cs_s)
        rd_s = rate_distortion_dim(X_snap)
        c_snap = p_s * rd_s["d_rd"]
        print(f"  SNAPPED     p={p_s:.3f}  d_rd={rd_s['d_rd']:.2f}  c={c_snap:.2f}  R^2={r2_s:.3f}")

        nll_snap, _ = measure_nll(sys_u.model, sys_u.tokenizer, sents)
        print(f"  SNAPPED NLL = {nll_snap:.4f}")
    finally:
        handle_u.remove()
    sys_u.unload(); torch.cuda.empty_cache()

    # Verdict
    drop_rel = (nll_untrained - nll_snap) / nll_untrained
    if c_snap < 6.0 and drop_rel > 0.05:
        verdict = "CODEBOOK_AND_CAPABILITY"
    elif c_snap < 6.0:
        verdict = "CODEBOOK_GEOMETRY_ONLY"
    elif drop_rel > 0.05:
        verdict = "CAPABILITY_WITHOUT_GEOMETRY"
    else:
        verdict = "NEITHER"
    print(f"\n  verdict: {verdict}")
    print(f"  NLL: untrained={nll_untrained:.3f}  snapped={nll_snap:.3f}  trained={nll_trained:.3f}  rel_drop={100*drop_rel:+.1f}%")

    out = {
        "purpose": "k-means codebook transfusion — piecewise nonlinear transplant",
        "K_codebook": K,
        "trained_NLL": float(nll_trained),
        "untrained_NLL": float(nll_untrained),
        "snapped": {"p": p_s, "d_rd": rd_s["d_rd"], "c": c_snap, "NLL": float(nll_snap)},
        "nll_relative_drop": float(drop_rel),
        "verdict": verdict,
    }
    out_path = _ROOT / "results/gate2/codebook_transfusion.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
