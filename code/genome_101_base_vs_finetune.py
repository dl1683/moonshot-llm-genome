"""Adversarial #5 closure: does SFT/RLHF shift the invariant?

Codex: if fine-tuning variants move the constant while capability stays,
the invariant is a 'pretraining dynamics signature' not 'capability geometry'.

Test pairs (same backbone, different training regime):
 - Qwen3-0.6B-base (no fine-tune)
 - Qwen3-1.7B (available; check Instruct vs base if registry has both)
 - DeepSeek-R1-Distill-1.5B (distilled from R1)
 - Falcon-H1-0.5B-Instruct (if in registry)

If all four land in [4.0, 4.5], invariant survives SFT/RLHF/distill.
If Instruct variants drift by > 0.5, reframe.
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
    ("qwen3-0.6b-base",            "Qwen/Qwen3-0.6B"),
    ("qwen3-1.7b",                 "Qwen/Qwen3-1.7B"),
    ("deepseek-r1-distill-qwen-1.5b","deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("falcon-h1-0.5b-instruct",    "tiiuae/Falcon-H1-0.5B-Instruct"),
]


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


def main():
    t0 = time.time()
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= 800:
            break

    rows = []
    for label, hf_id in SYSTEMS:
        print(f"\n[{time.time()-t0:.1f}s] ===== {label} =====")
        try:
            sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
        except Exception as e:
            print(f"  FAIL load: {e}"); continue
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
        st.update({"label": label, "h": int(X.shape[1])})
        rows.append(st)
        print(f"  eff_rank={st['eff_rank']:.2f}  alpha={st['alpha']:.3f}  "
              f"sqrt(er)*a={st['sqrt_er_alpha']:.3f}  er*a2={st['er_alpha2']:.3f}")

    print("\n\n=== BASE vs FINETUNE ===")
    for r in rows:
        print(f"  {r['label']:35s}  sqrt(er)*a={r['sqrt_er_alpha']:.3f}")
    invs = [r["sqrt_er_alpha"] for r in rows]
    if invs:
        m, s = float(np.mean(invs)), float(np.std(invs))
        cv = 100*s/m if m else 0
        print(f"\n  all N={len(invs)}  mean={m:.3f}  std={s:.3f}  CV={cv:.2f}%")

    out = {"rows": rows}
    out_path = _ROOT / "results/gate2/base_vs_finetune.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
