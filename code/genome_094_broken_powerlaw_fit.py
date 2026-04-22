"""Fit broken-power-law to empirical trained-ML spectra (replacement for genome_091).

Broken-power-law model:
  sigma^2 = i^(-2*a1)                      for i <= k_brk
         = (k_brk/i)^(2*a2) * k_brk^(-2*a1)  for i > k_brk
(continuous at i = k_brk)

Numerical search on population spectrum hinted that (k_brk=24, a1=0.4, a2=0.8)
reproduces the empirical invariant sqrt(er)*alpha ≈ 4.27 and er*alpha^2 ≈ 18.3
at the 1% level.

This probe fits the 3-parameter model to each of 5 text systems' actual
mid-depth singular spectra and reports:
  - best-fit (k_brk, a1, a2) per system
  - log-log R^2 per system
  - CV of k_brk and a2 across systems
  - comparison of predicted eff_rank vs empirical

If k_brk clusters tightly (CV < 20%) AND k_brk ≈ k_bulk/2 where k_bulk=48
(genome_047 universal), the broken-power-law is the empirical spectrum shape
and the invariant is explained.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import curve_fit

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


def broken_model_log(i, logA, k_brk, a1, a2):
    """log(sigma^2) under the broken-power-law, parameterized continuously.
    sigma^2 = A * f(i) where f is the broken-power-law form.
    """
    i = np.asarray(i, dtype=np.float64)
    k = max(k_brk, 1.0)
    # head: i^(-2*a1); tail: (k/i)^(2*a2) * k^(-2*a1)
    head = -2 * a1 * np.log(i)
    tail = -2 * a1 * np.log(k) + 2 * a2 * np.log(k) - 2 * a2 * np.log(i)
    log_f = np.where(i <= k, head, tail)
    return logA + log_f


def fit_broken(s):
    s2 = s ** 2
    i = np.arange(1, len(s) + 1, dtype=np.float64)
    y = np.log(s2 + 1e-20)
    try:
        (logA, k_brk, a1, a2), cov = curve_fit(
            broken_model_log, i, y,
            p0=[float(y[0]), 24.0, 0.4, 0.8],
            bounds=([-50, 2, 0.0, 0.1], [50, 200, 2.0, 3.5]),
            maxfev=50000,
        )
    except Exception as e:
        return {"err": str(e)}
    y_pred = broken_model_log(i, logA, k_brk, a1, a2)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    s2_pred = np.exp(y_pred)
    er_pred = float(s2_pred.sum() ** 2 / (s2_pred ** 2).sum())
    er_emp = float(s2.sum() ** 2 / (s2 ** 2).sum())
    return {"logA": float(logA), "k_brk": float(k_brk),
             "a1_head": float(a1), "a2_tail": float(a2),
             "r2": float(r2), "er_pred": er_pred, "er_emp": er_emp}


def main():
    t0 = time.time()
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= 800:
            break

    rows = []
    for sys_key, hf_id in SYSTEMS:
        print(f"\n[{time.time()-t0:.1f}s] ===== {sys_key} =====")
        try:
            sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
        except Exception as e:
            print(f"  FAIL load: {e}"); continue
        mid = max(1, sys_obj.n_hidden_layers() // 2)
        try:
            traj = extract_trajectory(
                model=sys_obj.model, tokenizer=sys_obj.tokenizer,
                texts=sents, layer_indices=[mid], pooling="seq_mean",
                device="cuda", system_key=sys_key, class_id=1,
                quantization="fp16",
                stimulus_version="c4_clean.v1.seed42.n800",
                seed=42, batch_size=16, max_length=256,
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
        s = spectrum(X)
        fit = fit_broken(s)
        row = {"system": sys_key, "n": int(X.shape[0]), "h": int(X.shape[1]), **fit}
        rows.append(row)
        if "err" not in fit:
            print(f"  broken fit: k_brk={fit['k_brk']:.1f}  a1={fit['a1_head']:.3f}  "
                  f"a2={fit['a2_tail']:.3f}  R2={fit['r2']:.4f}  "
                  f"er_pred={fit['er_pred']:.2f}  er_emp={fit['er_emp']:.2f}")

    print("\n=== SUMMARY (broken power-law fits) ===")
    keys = ["k_brk", "a1_head", "a2_tail", "r2", "er_pred", "er_emp"]
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if vals:
            m, s = float(np.mean(vals)), float(np.std(vals))
            cv = 100*s/m if m != 0 else 0
            print(f"  {k:10s}  N={len(vals)}  mean={m:.3f}  std={s:.3f}  CV={cv:.1f}%")

    # Connection check: k_brk vs k_bulk=48 from genome_047
    ks = [r["k_brk"] for r in rows if "k_brk" in r]
    if ks:
        ratio = float(np.mean(ks)) / 48.0
        print(f"\n  k_brk / k_bulk (48) ratio: {ratio:.3f}  (target: 0.5)")

    out_path = _ROOT / "results/gate2/broken_powerlaw_fit.json"
    out_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
