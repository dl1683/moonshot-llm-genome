"""Measure c = p * d_rd on CLIP-text (text branch of CLIP-ViT-B/32).

CLIP-text is architecturally a text transformer (like GPT/BERT) but trained
with contrastive objective aligning text embeddings to vision embeddings.

Two hypotheses this disambiguates:
  H1 (modality): c follows modality of processing. CLIP-text is text-processing,
                  so c ≈ 2 (text band).
  H2 (alignment): c follows alignment target. CLIP-text is aligned to vision,
                  so c gets pulled toward vision ≈ 3.

Comparison:
  - Prior text bands: Qwen3, RWKV, DeepSeek, Qwen3-1.7B at c ≈ [1.89, 2.40].
  - Prior vision bands: DINOv2, I-JEPA, CLIP-vision at c ≈ [2.63, 3.95].
  - CLIP-vision c = 3.95 (highest in vision band — may be alignment inflation).

Predict:
  c_clip_text ≈ 2.0 → H1 supported (modality is the driver)
  c_clip_text ≈ 2.5-3.0 → intermediate / alignment-adjacent
  c_clip_text ≈ 3.0+ → H2 supported (alignment is the driver)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, CLIPTextModel

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_primitives import knn_clustering_coefficient  # noqa: E402
from genome_rate_distortion_probe import rate_distortion_dim, fit_power_law  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]


def main():
    hf_id = "openai/clip-vit-base-patch32"
    t0 = time.time()
    print(f"[{time.time()-t0:.1f}s] loading CLIP text encoder...")
    model = CLIPTextModel.from_pretrained(hf_id, torch_dtype=torch.float16).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(hf_id)

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    mid = n_layers // 2
    print(f"[{time.time()-t0:.1f}s] CLIP-text: {n_layers} layers, sampling mid {mid}, h={cfg.hidden_size}")

    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= 1000:
            break
    print(f"[{time.time()-t0:.1f}s] {len(sents)} C4 stimuli")

    # Extract with hook at mid layer
    mid_acts = []
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        mid_acts.append(h.detach().to(torch.float32).cpu())
    handle = model.text_model.encoder.layers[mid].register_forward_hook(hook)
    try:
        with torch.no_grad():
            for i in range(0, len(sents), 16):
                batch = sents[i:i + 16]
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=77)
                _ = model(input_ids=enc["input_ids"].to("cuda"),
                          attention_mask=enc["attention_mask"].to("cuda"))
    finally:
        handle.remove()

    # seq_mean pool
    pooled = []
    for h in mid_acts:
        pooled.append(h.mean(dim=1).numpy())
    X = np.concatenate(pooled, axis=0).astype(np.float32)
    print(f"[{time.time()-t0:.1f}s] cloud shape {X.shape}")

    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p, c0, r2 = fit_power_law(K_GRID, Cs)
    rd = rate_distortion_dim(X)
    c = p * rd["d_rd"]
    print(f"\n  CLIP-text mid-depth:")
    print(f"    p = {p:.3f}")
    print(f"    d_rd = {rd['d_rd']:.2f}")
    print(f"    c = p*d_rd = {c:.2f}")

    # Hypotheses
    h1_value, h2_value = 2.0, 3.0
    rel_err_h1 = abs(c - h1_value) / h1_value
    rel_err_h2 = abs(c - h2_value) / h2_value
    if rel_err_h1 < 0.15 and rel_err_h1 < rel_err_h2:
        verdict = f"H1_MODALITY (c={c:.2f} near 2.0, text-band): c is modality-of-processing"
    elif rel_err_h2 < 0.15 and rel_err_h2 < rel_err_h1:
        verdict = f"H2_ALIGNMENT (c={c:.2f} near 3.0, vision-band): c is alignment-target"
    elif 2.3 < c < 2.9:
        verdict = f"INTERMEDIATE (c={c:.2f}): partial alignment effect"
    else:
        verdict = f"OUT_OF_BAND (c={c:.2f}): unexpected"
    print(f"  rel_err vs H1 (modality,c=2): {rel_err_h1:.3f}")
    print(f"  rel_err vs H2 (alignment,c=3): {rel_err_h2:.3f}")
    print(f"  verdict: {verdict}")

    out = {
        "system": "clip-vit-b32-text-branch", "n": int(X.shape[0]), "h": int(X.shape[1]),
        "p": p, "d_rd": rd["d_rd"], "c_invariant": c,
        "rel_err_vs_H1_modality_c_2": rel_err_h1,
        "rel_err_vs_H2_alignment_c_3": rel_err_h2,
        "verdict": verdict,
    }
    out_path = _ROOT / "results/gate2/clip_text_c.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
