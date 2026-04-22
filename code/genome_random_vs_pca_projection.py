"""Is it the RANK or the TRAINED DIRECTIONS that matter?

genome_044 showed PCA-compression of trained activations to rank k destroys
both d_rd and NLL monotonically. Open question: is this because (a) the
dimensional collapse itself is what destroys capability (rank matters,
which directions don't), or (b) the trained principal directions specifically
are what matters (random-direction subspaces at same k would destroy
MORE capability).

Protocol (DeepSeek-R1-Distill-Qwen-1.5B, trained, mid-depth, C4-clean
n=500 seed 42):

For each k in {16, 64, 256}:
  - PCA projection: project onto top-k PCA components of trained activations,
    reconstruct. (Already measured in genome_044 — reload.)
  - RANDOM projection: project onto a random orthonormal k-subspace of R^h,
    reconstruct.
  - Measure d_rd + NLL under RANDOM-projection hook.

Compare per-k NLL under the two projections.

Pre-registered prediction:
  If feature-direction identity matters: RANDOM projection destroys NLL MORE
  than PCA projection at the same k.
  If only rank matters: RANDOM and PCA destroy NLL equivalently at same k.

Verdict categories:
  - DIRECTION_MATTERS: random NLL > PCA NLL + 10% at >=2/3 k values tested
  - RANK_ONLY: random NLL - PCA NLL < 10% at all k values
  - MIXED: intermediate
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

    # Load existing PCA sweep data to avoid re-running
    pca_data = json.loads((_ROOT / "results/gate2/pca_sweep_causal.json").read_text())
    pca_lookup = {r["k_pca"]: r for r in pca_data["pca_sweep"]}

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

    # Get trained mid-depth mean
    token_acts = []
    def cap_hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        token_acts.append(h.detach().to(torch.float32).cpu())
    h_cap = blocks[mid].register_forward_hook(cap_hook)
    try:
        with torch.no_grad():
            for i in range(0, min(len(sents), 100), 16):
                chunk = sents[i:i + 16]
                enc = sys_t.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256)
                _ = sys_t.model(input_ids=enc["input_ids"].to("cuda"), attention_mask=enc["attention_mask"].to("cuda"))
    finally:
        h_cap.remove()
    all_toks = torch.cat([t.reshape(-1, t.shape[-1]) for t in token_acts], dim=0).numpy()
    mu = all_toks.mean(axis=0)
    h = all_toks.shape[1]
    mu_t = torch.from_numpy(mu).to("cuda").float()
    print(f"[{time.time()-t0:.1f}s] h={h}, mu computed")

    nll_full, _ = measure_nll(sys_t.model, sys_t.tokenizer, sents)
    print(f"  TRAINED full NLL = {nll_full:.4f}")

    # Random subspace projection for each k
    rng = np.random.default_rng(seed + 1)  # different seed for projection matrix
    results = []
    for k_test in (16, 64, 256):
        # Sample random orthonormal k-subspace of R^h
        A = rng.standard_normal((h, k_test)).astype(np.float32)
        # QR for orthonormal basis
        Q, _ = np.linalg.qr(A)  # (h, k)
        V_rand = torch.from_numpy(Q.T).to("cuda").float()  # (k, h) — rows are orthonormal directions

        def rand_hook(module, inputs, output, V_rand=V_rand, mu_t=mu_t):
            hh = output[0] if isinstance(output, tuple) else output
            od = hh.dtype
            h32 = hh.to(torch.float32)
            hc = h32 - mu_t
            coeffs = hc @ V_rand.T
            recon = coeffs @ V_rand + mu_t
            y = recon.to(od)
            if isinstance(output, tuple):
                return (y,) + output[1:]
            return y

        handle = blocks[mid].register_forward_hook(rand_hook)
        try:
            nll_rand, _ = measure_nll(sys_t.model, sys_t.tokenizer, sents)
            traj_r = extract_trajectory(
                model=sys_t.model, tokenizer=sys_t.tokenizer,
                texts=sents, layer_indices=[mid], pooling="seq_mean",
                device="cuda", system_key=sk + f"_rand{k_test}", class_id=2,
                quantization="fp16",
                stimulus_version=f"rand_proj_{k_test}.seed{seed}.n{n}", seed=seed,
                batch_size=16, max_length=256,
            )
            X_r = traj_r.layers[0].X.astype(np.float32)
            Cs_r = [float(knn_clustering_coefficient(X_r, k=kk).value) for kk in K_GRID]
            p_r, _, _ = fit_power_law(K_GRID, Cs_r)
            rd_r = rate_distortion_dim(X_r)
            c_r = p_r * rd_r["d_rd"]
        finally:
            handle.remove()

        pca_r = pca_lookup.get(k_test, {})
        pca_nll = pca_r.get("NLL", float("nan"))
        pca_d_rd = pca_r.get("d_rd", float("nan"))
        pca_c = pca_r.get("c", float("nan"))
        gap_nll = nll_rand - pca_nll
        gap_rel = 100 * gap_nll / pca_nll if pca_nll > 0 else float("nan")
        print(f"  k={k_test}: RANDOM p={p_r:.3f} d_rd={rd_r['d_rd']:.2f} c={c_r:.2f} NLL={nll_rand:.3f}")
        print(f"           vs PCA   p={pca_r.get('p', float('nan')):.3f} d_rd={pca_d_rd:.2f} c={pca_c:.2f} NLL={pca_nll:.3f}")
        print(f"           NLL gap (random - PCA) = {gap_nll:+.3f} ({gap_rel:+.1f}% of PCA)")

        results.append({
            "k": k_test,
            "random": {"p": p_r, "d_rd": rd_r["d_rd"], "c": c_r, "NLL": nll_rand},
            "pca": {"p": pca_r.get("p"), "d_rd": pca_d_rd, "c": pca_c, "NLL": pca_nll},
            "nll_gap_rand_minus_pca": gap_nll,
            "nll_gap_pct": gap_rel,
        })

    sys_t.unload(); torch.cuda.empty_cache()

    # Verdict
    passes_direction = sum(1 for r in results if r["nll_gap_pct"] is not None and r["nll_gap_pct"] > 10.0)
    passes_rank_only = sum(1 for r in results if r["nll_gap_pct"] is not None and abs(r["nll_gap_pct"]) < 10.0)
    if passes_direction >= 2:
        verdict = "DIRECTION_MATTERS — trained feature-directions specifically cause capability"
    elif passes_rank_only >= 3:
        verdict = "RANK_ONLY — d_rd alone is the causal quantity regardless of direction"
    else:
        verdict = "MIXED — some directions matter but rank is dominant"
    print(f"\n  verdict: {verdict}")

    out = {"purpose": "Random-orthogonal vs PCA projection at same rank",
           "trained_full_NLL": nll_full, "h": h,
           "per_k": results, "verdict": verdict}
    out_path = _ROOT / "results/gate2/random_vs_pca_projection.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
