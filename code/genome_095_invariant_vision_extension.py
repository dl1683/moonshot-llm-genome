"""Extend sqrt(eff_rank)*alpha invariant to vision and cross-modal systems.

genome_088 (N=5 text) established:
  Trained:  sqrt(er)*alpha = 4.268  CV 5.09%
  Shuffled: 5.472  CV 17.0%
  Gaussian: 5.463  CV 17.3%
  (5.5 sigma trained-vs-untrained separation, matches 3*sqrt(2) to 0.6%)

If this invariant is truly universal across trained ML, it should hold
at similar CV on vision and cross-modal systems too. This probe runs
4 additional systems under their appropriate stimulus banks:
  - DINOv2-small (vision, ImageNet)
  - CLIP-vision-B/32 (cross-modal, ImageNet)
  - CLIP-text-B/32 (cross-modal, COCO captions or C4)
  - Falcon-H1-0.5B (text hybrid)

Plus untrained shuffle controls.

Target: CV stays < 7% when invariant values across 9 systems (5 text
+ 4 new) are computed. Trained/shuffled separation remains > 3 sigma.

If yes - invariant crosses modality, strongest cross-class universality
result to date.
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
from stimulus_banks import c4_clean_v1, imagenet_val_v1  # noqa: E402

_ROOT = _THIS_DIR.parent


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
    slope, _ = np.polyfit(lr, ls, 1)
    return float(-slope)


def eff_rank(s):
    s2 = s ** 2
    tot = s2.sum()
    return float(tot ** 2 / (s2 ** 2).sum()) if tot > 0 else 0.0


def main():
    t0 = time.time()
    N = 800
    MAX_LEN_TEXT = 256

    text_sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        text_sents.append(rec["text"])
        if len(text_sents) >= N:
            break

    # Get imagenet images for vision systems
    try:
        img_items = []
        for rec in imagenet_val_v1(seed=42, n_samples=N):
            img_items.append(rec)
            if len(img_items) >= N:
                break
    except Exception as e:
        print(f"imagenet load failed: {e}; skipping vision probe")
        img_items = None

    systems = [
        ("dinov2-small", "facebook/dinov2-small", "vision"),
        ("clip-vit-base-patch32-vision", "openai/clip-vit-base-patch32", "vision_align"),
        ("clip-vit-base-patch32-text", "openai/clip-vit-base-patch32", "text_align"),
        ("falcon-h1-0.5b", "tiiuae/Falcon-H1-0.5B-Instruct", "text"),
    ]

    rows = []
    for sys_key, hf_id, modality in systems:
        print(f"\n[{time.time()-t0:.1f}s] ===== {sys_key} ({modality}) =====")
        try:
            sys_obj = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
        except Exception as e:
            print(f"  FAIL load: {e}")
            continue
        mid = max(1, sys_obj.n_hidden_layers() // 2)
        texts = text_sents if modality.startswith("text") else img_items
        if texts is None:
            print("  no stimuli available, skip")
            sys_obj.unload(); torch.cuda.empty_cache(); continue
        try:
            kwargs = dict(
                model=sys_obj.model, tokenizer=sys_obj.tokenizer,
                texts=texts, layer_indices=[mid], pooling="seq_mean",
                device="cuda", system_key=sys_key, class_id=1,
                quantization="fp16",
                stimulus_version=("imagenet1k_val.v1.seed42.n800" if modality.startswith("vision") else "c4_clean.v1.seed42.n800"),
                seed=42, batch_size=16, max_length=MAX_LEN_TEXT,
            )
            traj = extract_trajectory(**kwargs)
            X = traj.layers[0].X.astype(np.float32)
        except Exception as e:
            print(f"  FAIL extract: {e}")
            sys_obj.unload(); torch.cuda.empty_cache(); continue
        sys_obj.unload(); torch.cuda.empty_cache()
        print(f"  cloud {X.shape}")

        s_tr = spectrum(X)
        rng = np.random.default_rng(42)
        Xs = X.copy()
        for j in range(Xs.shape[1]):
            rng.shuffle(Xs[:, j])
        s_sh = spectrum(Xs)

        for cond, s in [("trained", s_tr), ("shuffled", s_sh)]:
            alpha = fit_power_tail(s)
            er = eff_rank(s)
            inv = np.sqrt(er) * alpha
            rows.append({
                "system": sys_key, "modality": modality, "condition": cond,
                "alpha": alpha, "eff_rank": er,
                "sqrt_er_alpha": float(inv),
                "er_alpha2": float(er * alpha ** 2),
                "n": X.shape[0], "h": X.shape[1],
            })
            print(f"  {cond:9s}  alpha={alpha:.3f}  eff_rank={er:6.2f}  "
                  f"sqrt(er)*alpha={inv:.3f}  er*alpha^2={er*alpha**2:.3f}")

    print("\n\n=== SUMMARY (vision extension) ===")
    for cond in ("trained", "shuffled"):
        vals = [r["sqrt_er_alpha"] for r in rows if r["condition"] == cond]
        if vals:
            m, s = float(np.mean(vals)), float(np.std(vals))
            cv = 100*s/m if m else 0
            print(f"  {cond:9s}  N={len(vals)}  mean={m:.3f}  std={s:.3f}  CV={cv:.2f}%")

    out = {"rows": rows,
           "reference_3sqrt2": float(np.sqrt(18)),
           "reference_genome_088_text_N5": {"trained_mean": 4.268, "trained_cv_pct": 5.09}}
    out_path = _ROOT / "results/gate2/invariant_vision_extension.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
