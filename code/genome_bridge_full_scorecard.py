"""Extend candidate-8 bridge test to the FULL candidate-5 scorecard.

genome_060 covered 5 text systems (Qwen3, DeepSeek, BERT, RoBERTa, MiniLM).
genome_063 added DINOv2 vision (fail 20pct, overall 5/6 PASS).

Remaining scorecard systems to test:
  - CLIP-text-B/32     (text + 1 alignment, predicted c=3)
  - CLIP-vision-B/32   (vision + 1 alignment, predicted c=4)

Falcon-H1-0.5B, I-JEPA, DiT-XL/2-256 excluded per:
  - Falcon-H1 / RWKV: Windows custom-kernel issues
  - DiT / I-JEPA: separate extractor paths, will add in a later session

Output merges into results/gate2/svd_bridge_multimodel.json.

If CLIP branches land at ratios near 3 and 4 respectively, AND the
ratio matches the candidate-5 prediction BETTER than kNN-based c
matches (it often does per DINOv2), then the "ratio is more
fundamental than c" claim graduates from speculation to support.
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
from genome_extractor import extract_trajectory, extract_vision_trajectory  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from genome_primitives import knn_clustering_coefficient  # noqa: E402
from genome_rate_distortion_probe import rate_distortion_dim, fit_power_law  # noqa: E402
from stimulus_banks import c4_clean_v1, imagenet_val_v1  # noqa: E402

_ROOT = _THIS_DIR.parent
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]


def spectrum(X):
    Xc = X - X.mean(axis=0)
    return (np.linalg.svd(Xc, compute_uv=False) / np.sqrt(X.shape[0] - 1)).astype(np.float64)


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


def analyze(X):
    s = spectrum(X)
    er = eff_rank(s)
    alpha = fit_alpha_tail(s)
    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p, _, _ = fit_power_law(K_GRID, Cs)
    rd = rate_distortion_dim(X)
    c = p * rd["d_rd"]
    ratio = er / rd["d_rd"]
    return {"c": float(c), "p": float(p), "d_rd": float(rd["d_rd"]),
            "eff_rank": er, "alpha": alpha, "ratio": ratio,
            "rel_err_ratio_vs_c": abs(ratio - c) / max(c, 1e-6)}


def run_clip_text(n=1000, seed=42):
    """CLIP-text-B/32 text tower — extract text embeddings."""
    from transformers import CLIPModel, CLIPProcessor
    print("\n=== clip-text-b32 ===")
    t0 = time.time()
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).cuda().eval()
    sents = []
    for rec in c4_clean_v1(seed=seed, n_samples=5 * n):
        sents.append(rec["text"])
        if len(sents) >= n:
            break
    reps = []
    batch = 16
    with torch.no_grad():
        for i in range(0, n, batch):
            inputs = processor(text=sents[i:i + batch], return_tensors="pt",
                               padding=True, truncation=True, max_length=77).to("cuda")
            # Use mid-depth hidden state from text transformer (layer n//2)
            out = model.text_model(**inputs, output_hidden_states=True)
            hs = out.hidden_states  # tuple of (embed_out, layer1_out, ...)
            mid = len(hs) // 2
            # pool with attention mask
            h = hs[mid]
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            reps.append(pooled.float().cpu().numpy())
    X = np.concatenate(reps, axis=0)
    del model; torch.cuda.empty_cache()
    r = analyze(X)
    r["system"] = "clip-text-b32"; r["modality"] = "text_aligned"
    print(f"  c={r['c']:.3f}  eff_rank={r['eff_rank']:.1f}  d_rd={r['d_rd']:.2f}  "
          f"alpha={r['alpha']:.3f}  ratio={r['ratio']:.3f}  rel_err={r['rel_err_ratio_vs_c']:.3f}  (t={time.time()-t0:.1f}s)")
    return r


def run_clip_vision(n=500, seed=42):
    """CLIP-vision-B/32 vision tower — extract image embeddings."""
    from transformers import CLIPModel, CLIPProcessor
    print("\n=== clip-vision-b32 ===")
    t0 = time.time()
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).cuda().eval()
    imgs = []
    for rec in imagenet_val_v1(seed=seed, n_samples=n):
        imgs.append(rec["image"])
        if len(imgs) >= n:
            break
    reps = []
    batch = 16
    with torch.no_grad():
        for i in range(0, n, batch):
            inputs = processor(images=imgs[i:i + batch], return_tensors="pt").to("cuda")
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
            out = model.vision_model(**inputs, output_hidden_states=True)
            hs = out.hidden_states
            mid = len(hs) // 2
            # CLIP vision has CLS token at index 0
            pooled = hs[mid][:, 0, :]
            reps.append(pooled.float().cpu().numpy())
    X = np.concatenate(reps, axis=0)
    del model; torch.cuda.empty_cache()
    r = analyze(X)
    r["system"] = "clip-vision-b32"; r["modality"] = "vision_aligned"
    print(f"  c={r['c']:.3f}  eff_rank={r['eff_rank']:.1f}  d_rd={r['d_rd']:.2f}  "
          f"alpha={r['alpha']:.3f}  ratio={r['ratio']:.3f}  rel_err={r['rel_err_ratio_vs_c']:.3f}  (t={time.time()-t0:.1f}s)")
    return r


def main():
    new_rows = []
    for fn in [run_clip_text, run_clip_vision]:
        try:
            new_rows.append(fn())
        except Exception as e:
            import traceback; traceback.print_exc()
            new_rows.append({"system": fn.__name__, "error": str(e)})

    # Merge into bridge JSON
    main_json = _ROOT / "results/gate2/svd_bridge_multimodel.json"
    existing = json.loads(main_json.read_text())
    # Remove any prior rows for these systems
    keep = [r for r in existing["per_system"]
            if r.get("system") not in {"clip-text-b32", "clip-vision-b32"}]
    existing["per_system"] = keep + new_rows

    passes = 0; tested = 0
    for r in existing["per_system"]:
        if "error" in r:
            continue
        tested += 1
        if r.get("rel_err_ratio_vs_c", 1) < 0.15:
            passes += 1
    rate = passes / max(tested, 1)
    if rate >= 0.8:
        existing["verdict"] = f"CANDIDATE_8_SUPPORTED - {passes}/{tested}"
    elif rate >= 0.5:
        existing["verdict"] = f"CANDIDATE_8_PARTIAL - {passes}/{tested}"
    else:
        existing["verdict"] = f"CANDIDATE_8_FALSIFIED - only {passes}/{tested}"
    main_json.write_text(json.dumps(existing, indent=2))
    print(f"\nUpdated {main_json} with verdict: {existing['verdict']}")

    # Candidate-5 prediction check per row
    pred = {"clip-text-b32": 3.0, "clip-vision-b32": 4.0}
    print("\n=== candidate-5 predictions (ratio vs c against predicted integer) ===")
    for r in new_rows:
        if "error" in r:
            continue
        p = pred.get(r["system"])
        if p is None:
            continue
        c_err = abs(r["c"] - p) / p
        r_err = abs(r["ratio"] - p) / p
        winner = "ratio" if r_err < c_err else "c"
        print(f"  {r['system']}: c={r['c']:.2f} (err {c_err:.2%})  "
              f"ratio={r['ratio']:.2f} (err {r_err:.2%})  {winner} wins")


if __name__ == "__main__":
    main()
