"""Adversarial test of the 'trained-specific invariant' claim.

Codex adversarial review flagged:
 #1 The 'untrained' control (shuffled/iid-Gaussian of the trained cloud) is NOT
    the same as a random-init network with the same architecture. If random-init
    sits near sqrt(er)*alpha = 4.24, the invariant is an architecture artifact.
 #5 If fine-tuning (SFT/RLHF/distill) shifts the constant while capability stays,
    the claim is 'pretraining signature' not 'capability geometry'.

This probe measures invariant on:
 - Qwen3-0.6B trained (baseline)
 - Qwen3-0.6B RANDOM-INIT (same architecture, weights replaced by N(0, 0.02)
   following the model's init scheme)
 - DeepSeek-R1-Distill-Qwen-1.5B (distilled from a reasoning teacher)
 - BERT-base vs BERT-random-init

If random-init gives sqrt(er)*alpha ~= 4.24, #1 confirmed → story collapses.
If fine-tuned gives a different value than base, #5 confirmed → reframe.
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


def spectrum(X):
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False) / np.sqrt(max(X.shape[0] - 1, 1))
    return s.astype(np.float64)


def stats(s):
    s2 = s ** 2
    er = float(s2.sum() ** 2 / (s2 ** 2).sum()) if s2.sum() > 0 else 0.0
    h = len(s)
    r = np.arange(1, h + 1)
    lo, hi = max(1, int(h * 0.05)), int(h * 0.5)
    slope, _ = np.polyfit(np.log(r[lo:hi]), np.log(s[lo:hi] + 1e-12), 1)
    alpha = float(-slope)
    return {"eff_rank": er, "alpha": alpha,
            "sqrt_er_alpha": float(np.sqrt(er) * alpha),
            "er_alpha2": float(er * alpha ** 2)}


# Each entry: (label, hf_id, untrained_flag)
SYSTEMS = [
    ("qwen3-0.6b-trained",   "Qwen/Qwen3-0.6B", False),
    ("qwen3-0.6b-randinit",  "Qwen/Qwen3-0.6B", True),
    ("bert-trained",         "bert-base-uncased", False),
    ("bert-randinit",        "bert-base-uncased", True),
    ("deepseek-r1-distill",  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", False),
    ("roberta-trained",      "FacebookAI/roberta-base", False),
    ("roberta-randinit",     "FacebookAI/roberta-base", True),
]


def main():
    t0 = time.time()
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= 800:
            break

    rows = []
    for label, hf_id, untrained in SYSTEMS:
        print(f"\n[{time.time()-t0:.1f}s] ===== {label} (untrained={untrained}) =====")
        try:
            sys_obj = load_system(hf_id, quant="fp16", untrained=untrained, device="cuda")
        except Exception as e:
            print(f"  FAIL load: {e}")
            continue
        mid = max(1, sys_obj.n_hidden_layers() // 2)
        try:
            traj = extract_trajectory(
                model=sys_obj.model, tokenizer=sys_obj.tokenizer,
                texts=sents, layer_indices=[mid], pooling="seq_mean",
                device="cuda", system_key=label, class_id=1,
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
        st = stats(s)
        st.update({"label": label, "untrained": untrained, "n": int(X.shape[0]), "h": int(X.shape[1])})
        rows.append(st)
        print(f"  eff_rank={st['eff_rank']:.2f}  alpha={st['alpha']:.3f}  "
              f"sqrt(er)*alpha={st['sqrt_er_alpha']:.3f}  er*alpha^2={st['er_alpha2']:.3f}")

    print("\n\n=== ADVERSARIAL COMPARISON ===")
    print(f"{'label':30s} {'sqrt(er)*a':>12s} {'er*a^2':>10s}  {'note':s}")
    for r in rows:
        note = "RANDOM-INIT" if r["untrained"] else "TRAINED"
        print(f"{r['label']:30s} {r['sqrt_er_alpha']:>12.3f} {r['er_alpha2']:>10.3f}  {note}")

    # Compute split stats
    trained_inv = [r["sqrt_er_alpha"] for r in rows if not r["untrained"]]
    randinit_inv = [r["sqrt_er_alpha"] for r in rows if r["untrained"]]
    if trained_inv and randinit_inv:
        tm, ts = float(np.mean(trained_inv)), float(np.std(trained_inv))
        rm, rs = float(np.mean(randinit_inv)), float(np.std(randinit_inv))
        sep_sigma = (rm - tm) / ts if ts > 0 else float('inf')
        print(f"\n  trained:   N={len(trained_inv)}  mean={tm:.3f}  std={ts:.3f}")
        print(f"  randinit:  N={len(randinit_inv)}  mean={rm:.3f}  std={rs:.3f}")
        print(f"  separation: {sep_sigma:.2f} sigma of trained distribution")
        print(f"  ref: 3*sqrt(2) = {np.sqrt(18):.3f}")

    out_path = _ROOT / "results/gate2/adversarial_baselines.json"
    out_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
