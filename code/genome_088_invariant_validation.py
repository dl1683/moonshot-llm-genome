"""Validate the sqrt(eff_rank)*alpha ≈ 3*sqrt(2) invariant.

An 8-system retrospective scan of the candidate-8 bridge data found that
sqrt(eff_rank) * alpha ≈ 4.25 (CV 4.6%) — tighter than the bridge itself
(CV 14.7%). At mean 4.247, this is 3*sqrt(2) = 4.2426 to 0.1%.

The relation eff_rank ≈ 18/alpha² uses two INDEPENDENT functionals of the
same SVD spectrum (eff_rank is participation ratio, alpha is tail slope).
If the invariant also holds on shuffled / random-init / Gaussian spectra,
it's an RMT artifact. If it holds ONLY on trained spectra, it's the
derivation-grade spectral fingerprint of training.

Probe: for each system in {Qwen3-0.6B, DeepSeek-R1-Distill-1.5B, BERT-
base, RoBERTa-base, MiniLM-L6} — so 5 text systems — compute alpha and
eff_rank on three spectra:

  1. TRAINED: actual activation cloud under C4.
  2. SHUFFLED: same marginals, joint structure destroyed.
  3. GAUSSIAN: iid Gaussian with the same per-dim mean/std.

Compare sqrt(eff_rank)*alpha across the three conditions. If trained
clusters tight around 4.25 and shuffled/Gaussian cluster at a different
value, the invariant is training-specific.

Also logs: alpha*c, eff_rank/d_rd bridge estimates where computable.
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
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent


SYSTEMS = [
    ("qwen3-0.6b",                   "Qwen/Qwen3-0.6B"),
    ("deepseek-r1-distill-qwen-1.5b","deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("bert-base-uncased",            "bert-base-uncased"),
    ("roberta-base",                 "FacebookAI/roberta-base"),
    ("minilm-l6-contrastive",        "sentence-transformers/all-MiniLM-L6-v2"),
]


def spectrum(X):
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False) / np.sqrt(max(X.shape[0] - 1, 1))
    return s.astype(np.float64)


def fit_power_tail(s, lo=0.05, hi=0.5):
    r = np.arange(1, len(s) + 1)
    a = max(1, int(len(s) * lo))
    b = int(len(s) * hi)
    lr = np.log(r[a:b])
    ls = np.log(s[a:b] + 1e-12)
    slope, intercept = np.polyfit(lr, ls, 1)
    return float(-slope), float(intercept)


def eff_rank(s):
    s2 = s ** 2
    tot = s2.sum()
    return float(tot ** 2 / (s2 ** 2).sum()) if tot > 0 else 0.0


def main():
    t0 = time.time()
    N_SENTS = 800
    MAX_LEN = 256
    BATCH = 16

    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= N_SENTS:
            break

    rows = []
    for sys_key, hf_id in SYSTEMS:
        print(f"\n[{time.time()-t0:.1f}s] ===== {sys_key} =====")
        try:
            sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
        except Exception as e:
            print(f"  FAIL load: {e}")
            continue
        mid = max(1, sys_obj.n_hidden_layers() // 2)
        try:
            traj = extract_trajectory(
                model=sys_obj.model, tokenizer=sys_obj.tokenizer,
                texts=sents, layer_indices=[mid], pooling="seq_mean",
                device="cuda", system_key=sys_key, class_id=1,
                quantization="fp16",
                stimulus_version=f"c4_clean.v1.seed42.n{N_SENTS}",
                seed=42, batch_size=BATCH, max_length=MAX_LEN,
            )
            X = traj.layers[0].X.astype(np.float32)
        except Exception as e:
            print(f"  FAIL extract: {e}")
            try:
                sys_obj.unload(); torch.cuda.empty_cache()
            except Exception:
                pass
            continue
        sys_obj.unload(); torch.cuda.empty_cache()
        print(f"  cloud {X.shape}")

        s_tr = spectrum(X)
        rng = np.random.default_rng(42)
        Xs = X.copy()
        for j in range(Xs.shape[1]):
            rng.shuffle(Xs[:, j])
        s_sh = spectrum(Xs)
        mu, sd = X.mean(axis=0), X.std(axis=0)
        Xg = rng.normal(loc=mu, scale=sd, size=X.shape).astype(np.float32)
        s_gs = spectrum(Xg)

        for cond, s in [("trained", s_tr), ("shuffled", s_sh), ("gaussian", s_gs)]:
            alpha, _ = fit_power_tail(s)
            er = eff_rank(s)
            inv = np.sqrt(er) * alpha
            rows.append({
                "system": sys_key, "condition": cond,
                "alpha": alpha, "eff_rank": er,
                "sqrt_er_alpha": float(inv),
                "er_alpha2": float(er * alpha ** 2),
                "n_samples": X.shape[0], "h": X.shape[1],
            })
            print(f"  {cond:9s}  alpha={alpha:.3f}  eff_rank={er:6.2f}  "
                  f"sqrt(er)*alpha={inv:.3f}  er*alpha^2={er*alpha**2:.3f}")

    print("\n\n=== SUMMARY (sqrt_er_alpha across conditions) ===")
    for cond in ("trained", "shuffled", "gaussian"):
        vals = [r["sqrt_er_alpha"] for r in rows if r["condition"] == cond]
        if vals:
            m, s = np.mean(vals), np.std(vals)
            cv = 100*s/m if m > 0 else 0
            print(f"  {cond:9s}  N={len(vals)}  mean={m:.3f}  std={s:.3f}  CV={cv:.2f}%")

    print("\n=== SUMMARY (eff_rank * alpha^2 across conditions) ===")
    for cond in ("trained", "shuffled", "gaussian"):
        vals = [r["er_alpha2"] for r in rows if r["condition"] == cond]
        if vals:
            m, s = np.mean(vals), np.std(vals)
            cv = 100*s/m if m > 0 else 0
            print(f"  {cond:9s}  N={len(vals)}  mean={m:.3f}  std={s:.3f}  CV={cv:.2f}%")

    out_path = _ROOT / "results/gate2/invariant_validation.json"
    out_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
