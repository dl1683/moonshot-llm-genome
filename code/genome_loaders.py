"""Model loaders for the Neural Genome Batch-1 pipeline.

Loads the three Batch-1 anchor systems + their random-init twins in FP16 or
Q8 quantization, returning a uniform `LoadedSystem` record. Per
`research/atlas_tl_session.md` 3b and 2.5.6 G1.5.

Windows + CUDA constraints (per CLAUDE.md 6): num_workers=0, pin_memory=False,
ASCII-only source, per-model native tokenizer.

Usage:
    from genome_loaders import load_system, SYSTEM_IDS

    sys = load_system("Qwen/Qwen3-0.6B", quant="fp16", untrained=False,
                      device="cuda")
    # sys.model, sys.tokenizer, sys.metadata

Trained models come from the canonical registry at ../../models/MODEL_DIRECTORY.md.
Untrained twins are random-init using the config from the trained HF ID — same
architecture, fresh weights.
"""

from __future__ import annotations

import dataclasses
import gc
from typing import Any

import torch
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# -------------------- Canonical system IDs for Batch 1 --------------------

SYSTEM_IDS: dict[str, dict[str, Any]] = {
    "qwen3-0.6b": {
        "hf_id": "Qwen/Qwen3-0.6B",
        "class_id": 1,
        "class_name": "autoregressive LLM",
        "approx_params": 600_000_000,
        "modality": "text",
    },
    "rwkv-4-169m": {
        "hf_id": "RWKV/rwkv-4-169m-pile",
        "class_id": 3,
        "class_name": "linear-attention recurrent (RWKV)",
        "approx_params": 169_000_000,
        "modality": "text",
        # SUBSTITUTE for state-spaces/mamba2-370m which is Windows-blocked:
        # Mamba/Mamba2 HF wrappers require the mamba-ssm + causal-conv1d CUDA
        # kernels, which have no prebuilt Windows wheels and fail source build
        # (bare_metal_version NameError at compile). RWKV-4 is HF-native, uses
        # linear-attention recurrence (class_id=3 per SYSTEM_BESTIARY), and
        # serves the same "non-transformer recurrent/state-space" role in
        # Batch-1's ≥3-class gate test.
    },
    # Skipped for now on Windows (same mamba-ssm kernel blocker):
    # "mamba2-370m": state-spaces/mamba2-370m (needs mamba-ssm CUDA kernels)
    # Flag: revisit on Linux or once prebuilt Windows wheel available.
    "falcon-h1-0.5b": {
        "hf_id": "tiiuae/Falcon-H1-0.5B-Instruct",
        "class_id": 4,
        "class_name": "hybrid (transformer + Mamba2)",
        "approx_params": 500_000_000,
        "trust_remote_code": True,  # Falcon-H1 requires custom modeling code
        "modality": "text",
    },
    "deepseek-r1-distill-qwen-1.5b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "class_id": 2,
        "class_name": "reasoning (distilled-from-R1)",
        "approx_params": 1_500_000_000,
        "modality": "text",
        # Reasoning class added 2026-04-21 to push Batch-1 bestiary from 4
        # classes (1,3,4,6) -> 5 classes (1,2,3,4,6), hitting the Level-1
        # threshold per UNIVERSALITY_LEVELS.md. Same qwen-family tokenizer so
        # fits the same text F.
    },
    "dinov2-small": {
        "hf_id": "facebook/dinov2-small",
        "class_id": 6,
        "class_name": "vision encoder (ViT)",
        "approx_params": 22_000_000,
        "modality": "vision",
        # DINOv2 is a self-supervised ViT — no causal LM head. Use AutoModel.
        # Added per strategic-adversarial Codex directive (2026-04-21):
        # "add 1 non-language class immediately (vision encoder cleanest)."
        # Single move satisfies both ">=3 classes that actually run" and
        # "1st non-language class" per manifesto anti-drift rule.
    },
    # -------------------- Batch 2: encoder / contrastive / multilingual
    # Added 2026-04-21 per user-directed scope expansion ("we've only
    # tested decoder models"). Each class probes a distinct training-objective
    # or modality axis the Batch-1 bestiary is blind to.
    "bert-base-uncased": {
        "hf_id": "bert-base-uncased",
        "class_id": 7,
        "class_name": "masked-LM encoder (MLM objective)",
        "approx_params": 110_000_000,
        "modality": "text",
        # BERT is encoder-only — no causal head. Tests training-objective
        # invariance of kNN-k10: MLM vs autoregressive CLM. Uses AutoModel.
        "uses_causal_lm": False,
    },
    "minilm-l6-contrastive": {
        "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
        "class_id": 8,
        "class_name": "contrastive text encoder (sentence transformer)",
        "approx_params": 22_000_000,
        "modality": "text",
        # Contrastive training objective distinct from MLM and CLM. If kNN-k10
        # passes here, universality spans 3 distinct text training objectives.
        "uses_causal_lm": False,
    },
    "clip-vit-b32-image": {
        "hf_id": "openai/clip-vit-base-patch32",
        "class_id": 10,
        "class_name": "contrastive vision encoder (CLIP image branch)",
        "approx_params": 151_000_000,
        "modality": "vision",
        # CLIP vision branch: contrastive supervision (image-text pairs) vs
        # DINOv2 self-supervision. If both pass, kNN-k10 is vision-training-
        # objective-invariant. Uses CLIPVisionModel via AutoModel + its own
        # image processor.
        "uses_causal_lm": False,
        "vision_model_class": "CLIPVisionModel",
    },
    # -------------------- Batch 3: predictive-only architectures --------------------
    # Added 2026-04-21 per strategic Codex directive: current bestiary is
    # sequence models + image encoders; genuinely non-next-token-time
    # generative-prediction systems (JEPA, diffusion) are the cleanest
    # architecture-agnostic stress test we haven't run.
    "ijepa-vitb16": {
        "hf_id": "facebook/ijepa_vitb16_1k",
        "class_id": 9,
        "class_name": "predictive-masked vision encoder (I-JEPA)",
        "approx_params": 86_000_000,
        "modality": "vision",
        # I-JEPA: Image-based Joint Embedding Predictive Architecture. Trained
        # by predicting target-block features from context-block features in
        # the latent embedding space (LeCun / Assran et al. 2023). Encoder-only
        # ViT, so architecturally it fits the existing extractor, but the
        # TRAINING OBJECTIVE is fundamentally different from DINOv2 (self-
        # distillation), CLIP (contrastive), BERT (masked-input-reconstruction),
        # or any CLM. If kNN-k10 passes on I-JEPA too, the cross-architecture
        # portability spans 6 training objectives.
        "uses_causal_lm": False,
    },
}


@dataclasses.dataclass
class LoadedSystem:
    """One loaded model + tokenizer-or-image-processor + metadata.

    Held in memory; .unload() should be called before loading the next one
    unless running multiple concurrently (allowed for the three sub-1B models
    per COMPUTE.md section 2).
    """

    system_key: str             # e.g. "qwen3-0.6b" or "dinov2-small"
    hf_id: str
    class_id: int
    class_name: str
    modality: str               # "text" or "vision"
    untrained: bool             # True -> random-init twin
    quant: str                  # "fp16" or "q8"
    device: str                 # "cuda" or "cpu"
    model: Any                  # the HF model
    tokenizer: Any              # text: HF tokenizer; vision: None
    image_processor: Any        # vision: HF image processor; text: None
    config: Any                 # the HF config (for layer-count etc.)

    def n_hidden_layers(self) -> int:
        """Return depth L for the system. Handles transformer + Mamba + hybrid +
        multi-modal configs (CLIP) that nest vision_config / text_config."""
        # Try top-level attrs first.
        cfg = self.config
        for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
            if hasattr(cfg, attr):
                val = getattr(cfg, attr)
                if isinstance(val, int) and val > 0:
                    return val
        # For multi-modal configs (CLIP, BLIP, etc.), look inside the relevant
        # sub-config matching our modality.
        sub_attr = "vision_config" if self.modality == "vision" else "text_config"
        if hasattr(cfg, sub_attr):
            sub = getattr(cfg, sub_attr)
            for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
                if hasattr(sub, attr):
                    val = getattr(sub, attr)
                    if isinstance(val, int) and val > 0:
                        return val
        raise AttributeError(
            f"cannot determine layer depth for {self.hf_id}: "
            f"config has none of num_hidden_layers / n_layer / num_layers / n_layers "
            f"(also checked {sub_attr} sub-config)"
        )

    def unload(self) -> None:
        """Free VRAM. Call between systems if not running concurrent."""
        del self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# -------------------- Quantization config --------------------

def _quantization_config(quant: str) -> BitsAndBytesConfig | None:
    if quant == "fp16":
        return None
    if quant == "q8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quant == "q4":
        # NF4 quantization with fp16 compute dtype — standard bnb 4-bit config.
        # Added 2026-04-21 for the Geometry-Efficiency probe (strategic Codex
        # directive): need Q4 data point to test whether (c_0, p) geometry
        # drift predicts NLL drift when capability visibly degrades.
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError(f"unsupported quant {quant!r}; expected 'fp16'/'q8'/'q4'")


def _torch_dtype(quant: str) -> torch.dtype:
    if quant == "fp16":
        return torch.float16
    if quant in ("q8", "q4"):
        # bnb quantized weights load at fp16 then quantize internally.
        return torch.float16
    raise ValueError(f"unsupported quant {quant!r}")


# -------------------- Loading --------------------

def load_system(hf_id: str, *, quant: str = "fp16", untrained: bool = False,
                device: str = "cuda") -> LoadedSystem:
    """Load a Batch-1 system. Trained = pretrained weights from HF hub.
    Untrained = random-init using the same config (for negative control).

    Parameters:
        hf_id    : canonical HuggingFace ID from SYSTEM_IDS
        quant    : "fp16" or "q8"
        untrained: True -> random-init twin; False -> trained weights
        device   : "cuda" or "cpu"

    Raises:
        ValueError if hf_id not in registry
        RuntimeError if CUDA requested but unavailable
    """
    system_key = _resolve_system_key(hf_id)
    meta = SYSTEM_IDS[system_key]
    trust_remote = bool(meta.get("trust_remote_code", False))
    modality = meta.get("modality", "text")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    tokenizer = None
    image_processor = None
    if modality == "text":
        tokenizer_source = meta.get("tokenizer_fallback", hf_id)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source, trust_remote_code=trust_remote)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source, trust_remote_code=False)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
    elif modality == "vision":
        image_processor = AutoImageProcessor.from_pretrained(
            hf_id, trust_remote_code=trust_remote)
    else:
        raise ValueError(f"unsupported modality {modality!r}")

    config = AutoConfig.from_pretrained(hf_id, trust_remote_code=trust_remote)

    quant_cfg = _quantization_config(quant)
    dtype = _torch_dtype(quant)

    # Pick model class. Text systems with a causal LM head use
    # AutoModelForCausalLM (the default); encoder-only text systems (BERT,
    # sentence-transformers, etc.) set `uses_causal_lm: False` and go
    # through AutoModel. Vision systems use AutoModel unless a specific
    # vision_model_class is declared (e.g. CLIPVisionModel for CLIP's
    # vision branch).
    uses_causal_lm = bool(meta.get("uses_causal_lm", modality == "text"))
    vision_model_class_name = meta.get("vision_model_class")
    if modality == "vision" and vision_model_class_name == "CLIPVisionModel":
        from transformers import CLIPVisionModel  # noqa: PLC0415
        model_cls = CLIPVisionModel
    elif uses_causal_lm:
        model_cls = AutoModelForCausalLM
    else:
        model_cls = AutoModel

    if untrained:
        # Random-init: instantiate from config only, skip pretrained weights.
        # bnb quantization on random weights is not the comparison we want; force fp16.
        if quant != "fp16":
            raise ValueError(
                "untrained twins must be loaded at fp16 (quantizing random "
                "weights does not give a meaningful negative control)"
            )
        model = model_cls.from_config(
            config, torch_dtype=dtype, trust_remote_code=trust_remote)
        model = model.to(device)
    else:
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": trust_remote,
        }
        if quant_cfg is not None:
            load_kwargs["quantization_config"] = quant_cfg
            # bitsandbytes places weights on its chosen device automatically.
        else:
            load_kwargs["device_map"] = device if device == "cuda" else None

        model = model_cls.from_pretrained(hf_id, **load_kwargs)
        if quant_cfg is None and device == "cuda":
            model = model.to(device)

    model.eval()

    return LoadedSystem(
        system_key=system_key,
        hf_id=hf_id,
        class_id=meta["class_id"],
        class_name=meta["class_name"],
        modality=modality,
        untrained=untrained,
        quant=quant,
        device=device,
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        config=config,
    )


def _resolve_system_key(hf_id: str) -> str:
    for key, meta in SYSTEM_IDS.items():
        if meta["hf_id"].lower() == hf_id.lower():
            return key
    raise ValueError(
        f"unknown system {hf_id!r}; allowed Batch-1 systems: "
        f"{[m['hf_id'] for m in SYSTEM_IDS.values()]}"
    )


# -------------------- CLI for quick sanity --------------------

if __name__ == "__main__":
    import sys

    # Smoke: list systems and check CUDA availability; do NOT actually load
    # models (that would download multi-GB weights). Intended to verify the
    # module imports cleanly.
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Torch version : {torch.__version__}")
    print(f"Registered Batch-1 systems:")
    for key, meta in SYSTEM_IDS.items():
        print(f"  {key:20s} hf_id={meta['hf_id']:45s} "
              f"class={meta['class_id']} ({meta['class_name']})")
    print(f"OK: {len(SYSTEM_IDS)} systems registered")
    sys.exit(0)
