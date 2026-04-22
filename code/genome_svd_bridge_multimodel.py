"""Candidate-8 universality probe: does eff_rank / d_rd ≈ c hold across
all candidate-5-fitting systems?

genome_057 on Qwen3-0.6B found eff_rank/d_rd = 25.27/12.27 = 2.06 at
trained mid-depth, C4 n=1000 — intriguingly close to the measured
c=p*d_rd=1.89. If this bridge holds across the 11 candidate-5-fitting
systems at C4, then:

    c ≈ eff_rank / d_rd    (candidate-8)

becomes a two-geometric-measurement equation (not a fit). Then the
subsequent derivation reduces to computing eff_rank(α) and d_rd(α)
from the same singular-value spectrum with power-law decay α — a
tractable random-matrix problem.

This script measures the bridge for each available system under fp16,
C4 seed 42, n=1000. Vision systems use imagenet_val_v1. Outputs:

    results/gate2/svd_bridge_multimodel.json
      - per-system: c, eff_rank, d_rd, ratio, alpha
      - summary: rel_err distribution, pass rate at ≤15 percent.

See `research/derivations/candidate_8_spectral_bridge.md` for full
motivation and the next steps if this passes or fails.
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
from stimulus_banks import c4_clean_v1, imagenet_val_v1  # noqa: E402

_ROOT = _THIS_DIR.parent
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]


def spectrum(X):
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False) / np.sqrt(X.shape[0] - 1)
    return s.astype(np.float64)


def eff_rank(s):
    s2 = s ** 2
    total = s2.sum()
    return 0.0 if total <= 0 else float(total ** 2 / (s2 ** 2).sum())


def fit_alpha_tail(s, lo_frac=0.05, hi_frac=0.5):
    r = np.arange(1, len(s) + 1)
    lo = max(1, int(len(s) * lo_frac))
    hi = int(len(s) * hi_frac)
    slope, _ = np.polyfit(np.log(r[lo:hi]), np.log(s[lo:hi] + 1e-12), 1)
    return float(-slope)


# RWKV and Falcon-H1 intentionally excluded: Windows custom-kernel issues per
# user directive "we can drop rwkv and mamba from our list if they keep being
# a problem" + COMPUTE.md Windows constraints. The bridge hypothesis is a
# geometric identity, so excluding SSM/hybrid classes at this test-of-
# universality step does not invalidate candidate-8 on transformer/encoder
# classes. Can add back via a separate Linux-compatible run later.
TEXT_SYSTEMS = [
    ("Qwen/Qwen3-0.6B", "qwen3-0.6b", 1),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
     "deepseek-r1-distill-qwen-1.5b", 2),
    ("bert-base-uncased", "bert-base-uncased", 7),
    ("FacebookAI/roberta-base", "roberta-base", 7),
    ("sentence-transformers/all-MiniLM-L6-v2", "minilm-l6-contrastive", 8),
]
# Vision handled separately because stimulus bank is different.
VISION_SYSTEMS = [
    ("facebook/dinov2-small", "dinov2-small", 6),
]


def run_one_text(hf_id, sk, cid, n=1000, seed=42):
    print(f"\n=== {sk} (text) ===")
    t0 = time.time()
    sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    mid = sys_obj.n_hidden_layers() // 2
    sents = []
    for rec in c4_clean_v1(seed=seed, n_samples=5 * n):
        sents.append(rec["text"])
        if len(sents) >= n:
            break
    traj = extract_trajectory(
        model=sys_obj.model, tokenizer=sys_obj.tokenizer,
        texts=sents, layer_indices=[mid], pooling="seq_mean",
        device="cuda", system_key=sk, class_id=cid,
        quantization="fp16",
        stimulus_version=f"c4_clean.v1.seed{seed}.n{n}",
        seed=seed, batch_size=16, max_length=256,
    )
    X = traj.layers[0].X.astype(np.float32)
    sys_obj.unload(); torch.cuda.empty_cache()
    s = spectrum(X)
    er = eff_rank(s)
    alpha = fit_alpha_tail(s)
    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p, _, _ = fit_power_law(K_GRID, Cs)
    rd = rate_distortion_dim(X)
    c = p * rd["d_rd"]
    ratio = er / rd["d_rd"]
    print(f"  c={c:.3f}  eff_rank={er:.1f}  d_rd={rd['d_rd']:.2f}  "
          f"alpha={alpha:.3f}  ratio={ratio:.3f}  rel_err={abs(ratio-c)/c:.3f}  "
          f"(t={time.time()-t0:.1f}s)")
    return {"system": sk, "modality": "text", "c": float(c),
            "p": float(p), "d_rd": float(rd["d_rd"]),
            "eff_rank": er, "alpha": alpha, "ratio": ratio,
            "rel_err_ratio_vs_c": abs(ratio - c) / c}


def run_one_vision(hf_id, sk, cid, n=500, seed=42):
    print(f"\n=== {sk} (vision) ===")
    t0 = time.time()
    sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    mid = sys_obj.n_hidden_layers() // 2
    imgs = []
    for rec in imagenet_val_v1(seed=seed, n_samples=n):
        imgs.append(rec["image"])
        if len(imgs) >= n:
            break
    traj = extract_trajectory(
        model=sys_obj.model, tokenizer=None,
        texts=None, images=imgs, layer_indices=[mid], pooling="cls_or_mean",
        device="cuda", system_key=sk, class_id=cid,
        quantization="fp16",
        stimulus_version=f"imagenet_val.v1.seed{seed}.n{n}",
        seed=seed, batch_size=16, max_length=None,
    )
    X = traj.layers[0].X.astype(np.float32)
    sys_obj.unload(); torch.cuda.empty_cache()
    s = spectrum(X)
    er = eff_rank(s)
    alpha = fit_alpha_tail(s)
    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p, _, _ = fit_power_law(K_GRID, Cs)
    rd = rate_distortion_dim(X)
    c = p * rd["d_rd"]
    ratio = er / rd["d_rd"]
    print(f"  c={c:.3f}  eff_rank={er:.1f}  d_rd={rd['d_rd']:.2f}  "
          f"alpha={alpha:.3f}  ratio={ratio:.3f}  rel_err={abs(ratio-c)/c:.3f}  "
          f"(t={time.time()-t0:.1f}s)")
    return {"system": sk, "modality": "vision", "c": float(c),
            "p": float(p), "d_rd": float(rd["d_rd"]),
            "eff_rank": er, "alpha": alpha, "ratio": ratio,
            "rel_err_ratio_vs_c": abs(ratio - c) / c}


def main():
    results = []
    for hf, sk, cid in TEXT_SYSTEMS:
        try:
            results.append(run_one_text(hf, sk, cid))
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({"system": sk, "error": str(e)})
    for hf, sk, cid in VISION_SYSTEMS:
        try:
            results.append(run_one_vision(hf, sk, cid))
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({"system": sk, "error": str(e)})

    print("\n=== CANDIDATE-8 PASS RATE ===")
    passes = 0; tested = 0
    for r in results:
        if "error" in r:
            continue
        tested += 1
        ok = r["rel_err_ratio_vs_c"] < 0.15
        tag = "PASS" if ok else "FAIL"
        if ok:
            passes += 1
        print(f"  {r['system']}: c={r['c']:.2f}  ratio={r['ratio']:.2f}  "
              f"rel_err={r['rel_err_ratio_vs_c']:.3f}  {tag}")
    if tested > 0:
        rate = passes / tested
        if rate >= 0.8:
            verdict = (f"CANDIDATE_8_SUPPORTED — {passes}/{tested} systems "
                       f"fit eff_rank/d_rd approximately equals c within 15pct. "
                       "Bridge is a universal geometric identity, not a Qwen3 coincidence.")
        elif rate >= 0.5:
            verdict = (f"CANDIDATE_8_PARTIAL — {passes}/{tested} pass. "
                       "Bridge holds on some systems; investigate failures.")
        else:
            verdict = (f"CANDIDATE_8_FALSIFIED — only {passes}/{tested} pass. "
                       "Bridge is NOT universal; Qwen3 was a coincidence.")
    else:
        verdict = "NO_SYSTEMS_MEASURED"
    print(f"\n  verdict: {verdict}")

    out = {"purpose": "Candidate-8 bridge: eff_rank/d_rd approximately equals c across scorecard systems",
           "per_system": results,
           "verdict": verdict}
    out_path = _ROOT / "results/gate2/svd_bridge_multimodel.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
