"""Fit shifted power-law sigma^2 = A * (i + k_head)^(-2*alpha) to empirical spectra.

Hypothesis: trained ML activation spectra have a universal k_head ≈ 5
(the "head" size beyond which power-law decay kicks in). Combined
with alpha ≈ 0.78, this shape analytically produces the candidate
invariant eff_rank * alpha^2 ≈ 18 and sqrt(eff_rank) * alpha ≈ 3*sqrt(2).

Probe: for each of 5 text systems (genome_088), extract mid-depth
activation spectrum, then fit sigma^2 = A * (i + k)^(-2*alpha) by
least squares in log-log space. Report:
  - fitted k_head per system (target: ~5 for all)
  - fitted alpha per system
  - fitted A per system
  - residual R^2
  - predicted eff_rank from fitted shape vs empirical eff_rank

If k_head clusters universally (CV < 20%) around 5, we have a
concrete spectral-shape law and a derivation path for the invariant.
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


def shifted_model(i, logA, k, alpha):
    """log(sigma^2) = logA - 2*alpha*log(i + k)"""
    return logA - 2 * alpha * np.log(i + k)


def fit_shifted(s):
    s2 = s ** 2
    i = np.arange(1, len(s) + 1, dtype=np.float64)
    # Fit in log space; weight by s^2 (top eigenvalues matter more)
    y = np.log(s2 + 1e-20)
    try:
        (logA, k, alpha), cov = curve_fit(
            shifted_model, i, y,
            p0=[0.0, 5.0, 0.8],
            bounds=([-20, 0, 0.1], [20, 200, 3.0]),
            maxfev=20000,
        )
    except Exception as e:
        return {"err": str(e)}
    y_pred = shifted_model(i, logA, k, alpha)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # Predicted eff_rank from the fitted shape
    s2_pred = np.exp(y_pred)
    er_pred = float(s2_pred.sum() ** 2 / (s2_pred ** 2).sum())
    er_emp = float(s2.sum() ** 2 / (s2 ** 2).sum())
    return {"logA": float(logA), "k_head": float(k),
             "alpha_true": float(alpha), "r2": float(r2),
             "er_pred": er_pred, "er_emp": er_emp}


def fit_pure_powerlaw(s, lo_frac=0.05, hi_frac=0.5):
    h = len(s)
    lo = max(1, int(h * lo_frac)); hi = int(h * hi_frac)
    r = np.arange(1, h + 1)
    slope, _ = np.polyfit(np.log(r[lo:hi]), np.log(s[lo:hi] + 1e-12), 1)
    return float(-slope)


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
        alpha_pure = fit_pure_powerlaw(s)
        fit = fit_shifted(s)
        row = {"system": sys_key, "n": int(X.shape[0]), "h": int(X.shape[1]),
                "alpha_pure_fit": alpha_pure, **fit}
        rows.append(row)
        print(f"  pure alpha-fit: alpha={alpha_pure:.3f}")
        if "err" not in fit:
            print(f"  shifted fit A*(i+k)^(-2a): k_head={fit['k_head']:.2f}  "
                  f"alpha_true={fit['alpha_true']:.3f}  R2={fit['r2']:.4f}  "
                  f"er_pred={fit['er_pred']:.2f}  er_emp={fit['er_emp']:.2f}")
        else:
            print(f"  fit failed: {fit['err']}")

    print("\n=== SUMMARY (shifted power-law fits) ===")
    ks = [r["k_head"] for r in rows if "k_head" in r]
    atrues = [r["alpha_true"] for r in rows if "alpha_true" in r]
    apures = [r["alpha_pure_fit"] for r in rows if "alpha_pure_fit" in r]
    if ks:
        km, ks_ = float(np.mean(ks)), float(np.std(ks))
        am, as_ = float(np.mean(atrues)), float(np.std(atrues))
        apm, aps_ = float(np.mean(apures)), float(np.std(apures))
        print(f"  k_head      N={len(ks)} mean={km:.2f}  std={ks_:.2f}  CV={100*ks_/km:.1f}%")
        print(f"  alpha_true  N={len(atrues)} mean={am:.3f}  std={as_:.3f}  CV={100*as_/am:.1f}%")
        print(f"  alpha_pure  N={len(apures)} mean={apm:.3f}  std={aps_:.3f}  CV={100*aps_/apm:.1f}%")

    out_path = _ROOT / "results/gate2/shifted_powerlaw_fit.json"
    out_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
