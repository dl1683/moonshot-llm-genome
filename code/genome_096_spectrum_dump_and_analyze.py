"""Dump raw trained spectra across 5 text systems, then non-parametric analysis.

Both parametric shape candidates (shifted-PL genome_091, broken-PL genome_094)
were falsified. The invariant sqrt(er)·α ≈ 3√2 is real; its derivation via a
simple shape is ruled out. This probe does NON-PARAMETRIC analysis instead:
extract raw singular values for all 5 text systems, save them, then look for
a universal STRUCTURAL feature that produces the invariant.

Features computed per spectrum:
  - cumulative variance fraction at quantiles {5, 10, 25, 50, 75} of eigenvalue index
  - log-spaced eigenvalue density
  - Shannon entropy of normalized variance (equivalent to participation ratio in log space)
  - ratio σ_k / σ_1 at k ∈ {10, 30, 100, 300}
  - cumulative sum up-to participation-ratio eff_rank

If some feature cluster cross-system at low CV AND relates algebraically to
eff_rank·α² = 18, that's the derivation candidate.
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


def analyze_spectrum(s, label):
    s2 = s ** 2
    total = s2.sum()
    normalized = s2 / max(total, 1e-20)
    h = len(s)

    # Standard stats
    er = float(total ** 2 / (s2 ** 2).sum()) if total > 0 else 0.0
    r = np.arange(1, h + 1)
    lo, hi = max(1, int(h * 0.05)), int(h * 0.5)
    slope, _ = np.polyfit(np.log(r[lo:hi]), np.log(s[lo:hi] + 1e-12), 1)
    alpha = float(-slope)
    inv = np.sqrt(er) * alpha
    ea2 = er * alpha ** 2

    # Cumulative variance fraction at quantiles
    cum = np.cumsum(normalized)
    q_idx = {q: int(h * q / 100) for q in (1, 5, 10, 25, 50, 75)}
    cum_at_q = {f"cum_var_at_q{q}": float(cum[q_idx[q] - 1]) for q in q_idx}

    # Shannon entropy of normalized variance
    p = normalized[normalized > 0]
    shannon = float(-(p * np.log(p)).sum())
    # effective dim from entropy: exp(shannon)
    er_shannon = float(np.exp(shannon))

    # Ratio sigma_k / sigma_1 at specific k
    sigma_ratios = {f"sigma_ratio_k{k}": float(s[k-1]/s[0]) if k-1 < h and s[0] > 0 else 0.0
                    for k in (10, 30, 100, 300)}

    # Index at which cumulative variance reaches 50%, 80%, 95%
    idx_at_var = {}
    for v in (0.5, 0.8, 0.95):
        ix = int(np.searchsorted(cum, v)) + 1
        idx_at_var[f"idx_at_{int(v*100)}pct_var"] = ix

    # Top-k cumulative variance eff_rank:
    # At what k does truncating to top-k reproduce eff_rank?
    cumtotal = np.cumsum(s2)
    cumsq = np.cumsum(s2 ** 2)
    er_topk = cumtotal ** 2 / np.maximum(cumsq, 1e-20)
    # Find k where er_topk(k) ≈ full eff_rank (e.g., 95% of it)
    target = er * 0.95
    k_for_95er = int(np.searchsorted(er_topk, target)) + 1

    return {
        "label": label, "h": int(h),
        "eff_rank": er, "alpha": alpha, "sqrt_er_alpha": inv, "er_alpha2": ea2,
        **cum_at_q,
        "shannon": shannon, "exp_shannon": er_shannon,
        **sigma_ratios,
        **idx_at_var,
        "k_for_95er": k_for_95er,
    }


def main():
    t0 = time.time()
    N = 800
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= N:
            break

    all_rows = []
    spectra = {}  # save raw for potential reuse
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
            sys_obj.unload(); torch.cuda.empty_cache(); continue
        sys_obj.unload(); torch.cuda.empty_cache()
        s = spectrum(X)
        spectra[sys_key] = s.tolist()
        row = analyze_spectrum(s, sys_key)
        all_rows.append(row)
        print(f"  eff_rank={row['eff_rank']:.2f}  alpha={row['alpha']:.3f}  "
              f"inv={row['sqrt_er_alpha']:.3f}  er*a2={row['er_alpha2']:.3f}")
        print(f"  cum_var q5={row['cum_var_at_q5']:.3f}  q10={row['cum_var_at_q10']:.3f}  "
              f"q25={row['cum_var_at_q25']:.3f}")
        print(f"  idx_at_50pct={row['idx_at_50pct_var']}  80pct={row['idx_at_80pct_var']}  "
              f"95pct={row['idx_at_95pct_var']}  k_for_95er={row['k_for_95er']}")
        print(f"  shannon={row['shannon']:.3f}  exp_shannon={row['exp_shannon']:.2f}")

    print("\n\n=== CROSS-SYSTEM STATS (target: low-CV features) ===")
    if not all_rows:
        print("no rows"); return
    for key in all_rows[0]:
        if key in ("label", "h"): continue
        vals = [r[key] for r in all_rows]
        m = np.mean(vals)
        s_ = np.std(vals)
        cv = 100*s_/m if m != 0 else 0
        mark = "  <-- TIGHT" if 0 < cv < 10 else ""
        print(f"  {key:30s}  mean={m:>9.3f}  std={s_:>7.3f}  CV={cv:>6.2f}%{mark}")

    out = {"rows": all_rows, "spectra": spectra}
    out_path = _ROOT / "results/gate2/spectrum_dump_analysis.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
