"""Diffusion class probe: does DiT-XL/2-256 (class-conditional ImageNet
diffusion transformer) join the cross-architecture C(k) cluster?

Closes the architecture-agnosticism gap Codex flagged strategically: current
bestiary is sequence models + image encoders, but not a genuinely
non-next-token-time generative-prediction system. Diffusion models denoise by
predicting added noise from noised input — a training objective fundamentally
distinct from CLM / MLM / contrastive / self-distillation / JEPA.

Pipeline:
    PIL image -> VAE encode (4ch 32x32 latent) -> scale 0.18215 -> add noise
    at t=250 (midway through 1000-step DDPM) -> DiT with null class label ->
    hook per-block residual output -> seq_mean pool over 256 spatial tokens
    -> point cloud (n_stimuli, 1152).

Writes: results/cross_arch/atlas_rows_n{N}_imagenet_seed{seed}_only_dit-xl-2-256.json

Hidden state dim = attention_head_dim * num_attention_heads = 72 * 16 = 1152.
Depth = 28 transformer blocks. Sentinel depths {0.26, 0.52, 0.74} -> blocks
{7, 14, 21}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DiTTransformer2DModel

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_primitives import knn_clustering_coefficient  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
DIT_SCALING = 0.18215  # standard DDPM/DiT VAE latent scale

SENTINEL_DEPTHS = (0.26, 0.52, 0.74)
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]
FIXED_TIMESTEP = 250  # middle-ish of DDPM 1000 schedule
NULL_CLASS = 1000  # DiT uses 1001 embeds (0-999 + null)


def pil_to_tensor(img, size=256):
    """PIL RGB -> normalized tensor (1, 3, H, W) in [-1, 1]."""
    from PIL import Image
    img = img.convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return (t - IMAGENET_MEAN) / IMAGENET_STD


def load_stimuli_imagenet(seed: int, n: int) -> list:
    """Load n ImageNet-val PIL images using existing stimulus_banks path."""
    from stimulus_banks import imagenet_val_v1
    out = []
    for rec in imagenet_val_v1(seed=seed, n_samples=n):
        out.append(rec["image"])
        if len(out) >= n:
            break
    return out


def encode_with_vae(vae, images, device, batch_size=16):
    """Stream-encode PIL images through VAE -> latents (N, 4, 32, 32)."""
    out = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        ts = torch.cat([pil_to_tensor(im) for im in batch], dim=0).to(device, dtype=torch.float16)
        with torch.no_grad():
            enc = vae.encode(ts).latent_dist.sample() * DIT_SCALING
        out.append(enc.detach())
        del ts, enc
        torch.cuda.empty_cache()
    return torch.cat(out, dim=0)


def extract_dit_features(transformer, latents, sentinel_blocks, device, batch_size=32, seed=42):
    """Forward batch of latents through DiT at fixed timestep + null class,
    hook sentinel-block residual outputs, pool seq_mean over the 256 tokens.

    Returns: dict{block_idx: np.ndarray (N, hidden_dim)}
    """
    blocks = transformer.transformer_blocks
    n_layers = len(blocks)
    captured = {i: [] for i in sentinel_blocks}
    scratch = {}

    def make_hook(idx):
        def hook(module, _inp, output):  # noqa: ARG001
            h = output[0] if isinstance(output, tuple) else output
            # h shape: (batch, 256, 1152)
            h32 = h.detach().to(torch.float32)
            pooled = h32.mean(dim=1).cpu().numpy()
            scratch[idx] = pooled
        return hook

    hooks = [blocks[i].register_forward_hook(make_hook(i))
             for i in sentinel_blocks]

    try:
        g = torch.Generator(device=device).manual_seed(seed)
        N = latents.shape[0]
        for start in range(0, N, batch_size):
            chunk = latents[start:start + batch_size]
            # Deterministic noise at fixed t=250
            noise = torch.randn(chunk.shape, generator=g, device=device,
                                dtype=chunk.dtype)
            # Simple noising (DDPM alpha_bar at t=250 roughly 0.85; this is a
            # probe, not generation — fixed noisy-latent produces a well-defined
            # point cloud regardless of exact schedule match)
            alpha_bar = 0.85
            noisy = alpha_bar ** 0.5 * chunk + (1 - alpha_bar) ** 0.5 * noise
            t = torch.full((chunk.shape[0],), FIXED_TIMESTEP,
                           dtype=torch.long, device=device)
            class_labels = torch.full((chunk.shape[0],), NULL_CLASS,
                                      dtype=torch.long, device=device)
            scratch.clear()
            with torch.no_grad():
                _ = transformer(hidden_states=noisy, timestep=t,
                                class_labels=class_labels, return_dict=False)
            for i, arr in scratch.items():
                captured[i].append(arr)
            torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()

    return {i: np.concatenate(v, axis=0) for i, v in captured.items()}, n_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"[{time.time()-t0:.1f}s] loading DiT-XL/2-256 + VAE to {device} fp16...")

    transformer = DiTTransformer2DModel.from_pretrained(
        "facebook/DiT-XL-2-256", subfolder="transformer",
        torch_dtype=torch.float16).to(device).eval()
    vae = AutoencoderKL.from_pretrained(
        "facebook/DiT-XL-2-256", subfolder="vae",
        torch_dtype=torch.float16).to(device).eval()

    n_layers = len(transformer.transformer_blocks)
    sentinel_blocks = [int(round(d * (n_layers - 1))) for d in SENTINEL_DEPTHS]
    print(f"[{time.time()-t0:.1f}s] {n_layers} DiT blocks; sentinel blocks {sentinel_blocks}")
    print(f"[{time.time()-t0:.1f}s] loading {args.n} ImageNet-val images seed={args.seed}...")
    images = load_stimuli_imagenet(args.seed, args.n)
    print(f"[{time.time()-t0:.1f}s] got {len(images)} stimuli")

    print(f"[{time.time()-t0:.1f}s] VAE-encoding to latents...")
    latents = encode_with_vae(vae, images, device)
    print(f"[{time.time()-t0:.1f}s] latents shape {tuple(latents.shape)}")

    print(f"[{time.time()-t0:.1f}s] DiT forward pass (fixed t={FIXED_TIMESTEP}, null class)...")
    features, n_layers = extract_dit_features(
        transformer, latents, sentinel_blocks, device, seed=args.seed)

    # Measure C(k) and build atlas rows
    rows = []
    for block_idx in sentinel_blocks:
        X = features[block_idx].astype(np.float32)
        k_norm = block_idx / (n_layers - 1)
        print(f"  block {block_idx} (depth {k_norm:.2f}): X shape {X.shape}")
        per_k = {}
        for k in K_GRID:
            if X.shape[0] <= k + 1:
                per_k[str(k)] = None
                continue
            C = float(knn_clustering_coefficient(X, k=k).value)
            per_k[str(k)] = C
        rows.append({
            "system_key": "dit-xl-2-256",
            "hf_id": "facebook/DiT-XL-2-256",
            "class_id": 11,  # diffusion transformer
            "class_name": "class-conditional diffusion transformer (DiT)",
            "modality": "vision",
            "stimulus_version": "vision.imagenet1k_val.v1",
            "scope_label": "vision:imagenet1k_val.v1:seq_mean",
            "tokenizer": "DiT_VAE_latent",
            "quantization": "fp16",
            "untrained": False,
            "seed": args.seed,
            "n_stimuli": int(X.shape[0]),
            "depth_index": block_idx,
            "k_normalized": round(k_norm, 4),
            "hidden_dim": int(X.shape[1]),
            "clustering_C_per_k": per_k,
            "timestep": FIXED_TIMESTEP,
            "null_class": NULL_CLASS,
        })

    out_path = args.out or str(
        _THIS_DIR.parent / "results" / "cross_arch"
        / f"atlas_rows_n{args.n}_imagenet_seed{args.seed}_only_dit-xl-2-256.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(
        {"rows": rows,
         "meta": {"model": "facebook/DiT-XL-2-256",
                  "n_stimuli": args.n, "seed": args.seed,
                  "timestep": FIXED_TIMESTEP, "null_class": NULL_CLASS,
                  "k_grid": K_GRID, "sentinel_blocks": sentinel_blocks}},
        indent=2))

    elapsed = time.time() - t0
    print(f"[{elapsed:.1f}s] done; wrote {out_path}")
    for r in rows:
        ks = r["clustering_C_per_k"]
        k10 = ks.get("12") or ks.get("8")
        print(f"  depth {r['k_normalized']:.2f}: C(k=8..12) -> {k10}")


if __name__ == "__main__":
    main()
