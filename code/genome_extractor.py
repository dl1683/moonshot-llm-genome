"""Hook-based activation extractor for the Neural Genome Batch-1 pipeline.

Produces a `PointCloudTrajectory` (per `research/atlas_tl_session.md` 2b
contract) given a loaded system and a stimulus batch. Streams layer-by-layer
to avoid holding full residual-stream tensors in VRAM simultaneously.

Returns point clouds at requested depths only (sentinel-depth selection per
prereg `research/prereg/genome_id_portability_2026-04-21.md` 5 + atlas_tl_session.md 2.5.6b).

Windows + CUDA constraints: no DataLoader multiprocessing; single forward pass.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import numpy as np  # noqa: F401 (used inside hook closure after rewrite)
import torch


# -------------------- Data contracts --------------------

@dataclasses.dataclass
class PointCloudLayer:
    """One layer's point cloud in a trajectory.

    For seq_mean pooling: one point per input sentence (n = n_sentences).
    For per_token pooling: one point per token sampled (n = n_tokens_subsample).
    """

    k_index: int                # raw layer index in the model (0..L-1)
    k_normalized: float         # ell / L, in [0, 1]
    pooling: str                # "seq_mean" or "per_token_subsample"
    X: np.ndarray               # shape (n, d), float32
    point_kind: str             # "token" or "sentence"


@dataclasses.dataclass
class PointCloudTrajectory:
    """Full trajectory for one system + stimulus batch + pooling."""

    system_key: str
    class_id: int
    quantization: str
    index_kind: str             # "layer" for transformer/SSM
    pooling: str
    stimulus_version: str
    seed: int
    layers: list[PointCloudLayer]
    n_stimuli: int              # number of input sentences

    def get_layer(self, k_normalized: float, tol: float = 0.02) -> PointCloudLayer:
        """Look up the layer closest to the requested normalized depth."""
        best = None
        best_dist = float("inf")
        for lyr in self.layers:
            d = abs(lyr.k_normalized - k_normalized)
            if d < best_dist:
                best_dist = d
                best = lyr
        if best is None or best_dist > tol:
            available = [lyr.k_normalized for lyr in self.layers]
            raise KeyError(
                f"no layer within tol={tol} of k_normalized={k_normalized}; "
                f"available: {available}"
            )
        return best


# -------------------- Depth selection --------------------

SENTINEL_DEPTHS: tuple[float, ...] = (0.25, 0.50, 0.75)


def sentinel_layer_indices(n_layers: int) -> list[int]:
    """Map sentinel depths {0.25, 0.50, 0.75} to layer indices for a model
    with n_layers layers. Rounds to nearest integer.
    """
    return [int(round(d * (n_layers - 1))) for d in SENTINEL_DEPTHS]


def full_curve_layer_indices(n_layers: int, n_points: int = 21) -> list[int]:
    """Sample n_points layer indices uniformly across the depth axis (full
    descriptive curve). Per `atlas_tl_session.md` 2.5.6b: descriptive only.
    """
    if n_points >= n_layers:
        return list(range(n_layers))
    xs = np.linspace(0, n_layers - 1, n_points)
    return [int(round(x)) for x in xs]


# -------------------- Extraction --------------------

def _transformer_blocks(model: Any) -> list[Any]:
    """Walk the model and return the per-layer block list, regardless of
    transformer/SSM/hybrid naming convention.
    """
    for path in [
        "model.layers",              # Qwen3, Llama
        "backbone.layers",           # Mamba2-hf
        "transformer.h",             # GPT-2 style
        "model.decoder.layers",      # some BART-ish
        "rwkv.blocks",               # RWKV
        "model.rwkv.blocks",         # RWKV wrapped
        "gpt_neox.layers",           # Pythia / GPT-NeoX
        "model.blocks",              # Falcon-H1 variants
        "encoder.layer",             # DINOv2, ViT, BERT encoders
        "vit.encoder.layer",         # ViT variants with outer wrapper
        "dinov2.encoder.layer",      # DINOv2 direct
        "blocks",                    # bare ViT (timm-style wrapped)
    ]:
        obj = model
        ok = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, (torch.nn.ModuleList, list)):
            return list(obj)
    raise AttributeError(
        f"could not locate layer list on model of type {type(model).__name__}; "
        f"tried model.layers, backbone.layers, transformer.h, model.decoder.layers"
    )


def extract_trajectory(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    layer_indices: list[int],
    pooling: str = "seq_mean",
    max_length: int = 256,
    device: str = "cuda",
    system_key: str,
    class_id: int,
    quantization: str,
    stimulus_version: str,
    seed: int,
    batch_size: int = 64,
) -> PointCloudTrajectory:
    """Forward-pass texts through the model in MICRO-BATCHES, hook selected
    layers, return a PointCloudTrajectory of pooled point-clouds.

    Micro-batching (batch_size sentences per forward pass) keeps peak VRAM
    bounded regardless of total n_sentences. Pooled outputs are accumulated
    per-layer across batches and concatenated at the end.

    - pooling="seq_mean": one point per input sentence (batch_size × d per pass).
    - pooling="per_token_subsample": flattens tokens, subsamples upstream.
    """
    if pooling not in ("seq_mean", "per_token_subsample"):
        raise ValueError(f"unsupported pooling {pooling!r}")

    blocks = _transformer_blocks(model)
    n_layers = len(blocks)
    layer_idx_set = set(layer_indices)
    # Accumulate pooled arrays across micro-batches; concatenate at end.
    batch_captured: dict[int, list[np.ndarray]] = {i: [] for i in layer_indices}

    # Hooks write into a per-batch scratch dict; outer loop moves scratch into batch_captured.
    scratch: dict[int, np.ndarray] = {}
    hooks = []
    for i, block in enumerate(blocks):
        if i not in layer_idx_set:
            continue

        def make_hook(layer_i: int):
            def hook(module, _inputs, output):  # noqa: ARG001
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                mask = _current_attention_mask[0]  # captured via closure
                h32 = h.detach().to(torch.float32)
                if pooling == "seq_mean":
                    mask_f = mask.unsqueeze(-1).to(torch.float32)
                    lengths = mask.sum(dim=1, keepdim=True).to(torch.float32).clamp(min=1)
                    pooled = (h32 * mask_f).sum(dim=1) / lengths
                    arr = pooled.cpu().numpy()
                else:
                    batch_, seq, dim = h32.shape
                    flat = h32.reshape(batch_ * seq, dim)
                    mflat = mask.reshape(batch_ * seq).bool()
                    arr = flat[mflat].cpu().numpy()
                finite_mask = np.all(np.isfinite(arr), axis=1)
                if not finite_mask.all():
                    arr = arr[finite_mask]
                scratch[layer_i] = arr
            return hook
        hooks.append(block.register_forward_hook(make_hook(i)))

    # Single-element list so closure can rebind per-batch attention mask.
    _current_attention_mask: list[Any] = [None]

    try:
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = tokenizer(
                chunk, padding=True, truncation=True, max_length=max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            _current_attention_mask[0] = attention_mask

            scratch.clear()
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False, output_hidden_states=False)
            for i, arr in scratch.items():
                batch_captured[i].append(arr)
            # Release VRAM before next micro-batch.
            del input_ids, attention_mask, enc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()

    captured: dict[int, np.ndarray] = {
        i: np.concatenate(arrs, axis=0) if arrs else np.zeros((0, 0), dtype=np.float32)
        for i, arrs in batch_captured.items()
    }

    # Assemble PointCloudTrajectory in the declared layer_indices order.
    layers: list[PointCloudLayer] = []
    for i in sorted(layer_indices):
        if i not in captured:
            raise RuntimeError(
                f"hook for layer {i} never fired; model forward path may have "
                f"skipped that layer"
            )
        point_kind = "sentence" if pooling == "seq_mean" else "token"
        layers.append(PointCloudLayer(
            k_index=i,
            k_normalized=i / max(n_layers - 1, 1),
            pooling=pooling,
            X=captured[i],
            point_kind=point_kind,
        ))

    return PointCloudTrajectory(
        system_key=system_key,
        class_id=class_id,
        quantization=quantization,
        index_kind="layer",
        pooling=pooling,
        stimulus_version=stimulus_version,
        seed=seed,
        layers=layers,
        n_stimuli=len(texts),
    )


# -------------------- Vision extraction --------------------

def extract_vision_trajectory(
    model: Any,
    image_processor: Any,
    images: list,             # list of PIL Images
    *,
    layer_indices: list[int],
    pooling: str = "cls_or_mean",
    device: str = "cuda",
    system_key: str,
    class_id: int,
    quantization: str,
    stimulus_version: str,
    seed: int,
    batch_size: int = 32,
) -> PointCloudTrajectory:
    """Forward-pass a batch of PIL images through a vision encoder (DINOv2 /
    ViT), hook the selected blocks, return a PointCloudTrajectory.

    Pooling for vision:
      - "cls_or_mean": take CLS token if present (index 0) else mean of patch tokens.
      - "patch_mean": mean across patch tokens (excluding CLS if present).
    Default produces one point per image (same semantics as "seq_mean" for text).
    """
    if pooling not in ("cls_or_mean", "patch_mean"):
        raise ValueError(f"unsupported vision pooling {pooling!r}")

    blocks = _transformer_blocks(model)
    n_layers = len(blocks)
    layer_idx_set = set(layer_indices)
    batch_captured: dict[int, list[np.ndarray]] = {i: [] for i in layer_indices}
    scratch: dict[int, np.ndarray] = {}

    hooks = []
    for i, block in enumerate(blocks):
        if i not in layer_idx_set:
            continue

        def make_hook(layer_i: int):
            def hook(module, _inputs, output):  # noqa: ARG001
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                h32 = h.detach().to(torch.float32)
                if pooling == "cls_or_mean":
                    pooled = h32.mean(dim=1)
                else:
                    pooled = h32[:, 1:, :].mean(dim=1) if h32.shape[1] > 1 \
                        else h32.mean(dim=1)
                arr = pooled.cpu().numpy()
                finite_mask = np.all(np.isfinite(arr), axis=1)
                if not finite_mask.all():
                    arr = arr[finite_mask]
                scratch[layer_i] = arr
            return hook
        hooks.append(block.register_forward_hook(make_hook(i)))

    try:
        model_dtype = next(model.parameters()).dtype
        for start in range(0, len(images), batch_size):
            chunk = images[start:start + batch_size]
            proc_out = image_processor(images=chunk, return_tensors="pt")
            pixel_values = proc_out["pixel_values"].to(device)
            if pixel_values.dtype != model_dtype:
                pixel_values = pixel_values.to(model_dtype)
            scratch.clear()
            with torch.no_grad():
                _ = model(pixel_values=pixel_values)
            for i, arr in scratch.items():
                batch_captured[i].append(arr)
            del pixel_values, proc_out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()

    captured: dict[int, np.ndarray] = {
        i: np.concatenate(arrs, axis=0) if arrs else np.zeros((0, 0), dtype=np.float32)
        for i, arrs in batch_captured.items()
    }

    layers: list[PointCloudLayer] = []
    for i in sorted(layer_indices):
        if i not in captured:
            raise RuntimeError(
                f"hook for layer {i} never fired; model forward path may have "
                f"skipped that block")
        layers.append(PointCloudLayer(
            k_index=i,
            k_normalized=i / max(n_layers - 1, 1),
            pooling=pooling,
            X=captured[i],
            point_kind="image",
        ))

    return PointCloudTrajectory(
        system_key=system_key,
        class_id=class_id,
        quantization=quantization,
        index_kind="layer",
        pooling=pooling,
        stimulus_version=stimulus_version,
        seed=seed,
        layers=layers,
        n_stimuli=len(images),
    )


# -------------------- CLI smoke --------------------

if __name__ == "__main__":
    # Check the module imports cleanly and sentinel-depth mapping is sane.
    for L in (12, 24, 32):
        print(f"L={L}: sentinel layers={sentinel_layer_indices(L)}, "
              f"full-curve layers (21 pts)={full_curve_layer_indices(L, 21)[:5]}... "
              f"({len(full_curve_layer_indices(L, 21))} total)")
    print(f"SENTINEL_DEPTHS = {SENTINEL_DEPTHS}")
    print("OK: genome_extractor imports cleanly + sentinel-layer math sane")
