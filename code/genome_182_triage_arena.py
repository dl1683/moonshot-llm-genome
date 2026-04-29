"""
genome_182_triage_arena.py

Blinded Training Triage Arena: cross-architecture geometry diagnostic.

Pre-reg: research/prereg/genome_182_triage_arena_2026-04-29.md (LOCKED)

72 cells = 2 architectures (Qwen3-arch, GPT-2-arch) x 3 arms x 12 seeds.
Leave-one-architecture-out CV. 9 baselines + combined telemetry.
Two co-primary geometry models (full + reference-free).

PASS: BOTH co-primary models on BOTH LOAO folds beat best non-geometry
baseline by >=25% MSE reduction, paired seed-block bootstrap 95% CI > 0,
held-out R^2 >= 0.20, shuffled-geometry p <= 0.01, bad-run AUROC >= 0.75.

Outputs:
  - results/genome_182_triage_arena.json
  - results/cache/genome_182_features/*.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import genome_180_forecast as g180

OUT_PATH = ROOT / "results" / "genome_182_triage_arena.json"
CACHE_DIR = ROOT / "results" / "cache" / "genome_182_features"

SEEDS = list(range(12))  # 0..11
SEQ_LEN = 256
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
TRAIN_STEPS = 3600
LR = 3e-4
LR_WARMUP_STEPS = 200
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
LOG_EVERY = 200
EVAL_EVERY = 600

TRAJECTORY_STEPS = [10, 20, 40, 60, 80, 108, 200, 500]

N_TRAIN_WINDOWS = 8192
N_C4_VAL_WINDOWS = 1000
PROBE_WINDOWS = 16
C4_TRAIN_SEED = 182001
C4_VAL_SEED = 182301
PROBE_SEED = 182180

BOOTSTRAP_N = 10_000
SHUFFLED_GEOMETRY_ITERS = 1000
PASS_MSE_REDUCTION = 0.25
WEAK_PASS_MSE_REDUCTION = 0.10
RANDOM_STATE = 182

HIDDEN_DIM = 768
N_LAYERS = 8
N_HEADS = 12
FFN_DIM = 2048

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FORWARD_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


RIDGE_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


# ---------------------------------------------------------------------------
# Architecture configs
# ---------------------------------------------------------------------------

ARCH_CONFIGS = {
    "qwen3": {
        "type": "qwen3",
        "hidden_size": HIDDEN_DIM,
        "num_hidden_layers": N_LAYERS,
        "num_attention_heads": N_HEADS,
        "num_key_value_heads": 6,  # GQA
        "intermediate_size": FFN_DIM,
        "vocab_size": 151936,
        "max_position_embeddings": SEQ_LEN + 64,
        "tie_word_embeddings": True,
    },
    "gpt2": {
        "type": "gpt2",
        "n_embd": HIDDEN_DIM,
        "n_layer": N_LAYERS,
        "n_head": N_HEADS,
        "n_inner": FFN_DIM,
        "vocab_size": 50257,
        "n_positions": SEQ_LEN + 64,
        "tie_word_embeddings": True,
    },
}

PHASE2_ARCH_CONFIGS = {
    "falcon_h1": {
        "type": "falcon_h1",
        "hidden_size": HIDDEN_DIM,
        "num_hidden_layers": N_LAYERS,
        "num_attention_heads": N_HEADS,
        "num_key_value_heads": 2,
        "intermediate_size": FFN_DIM,
        "vocab_size": 32784,
        "max_position_embeddings": SEQ_LEN + 64,
        "tie_word_embeddings": False,
    },
}


ARM_LABELS = ["scratch_ce", "seq_kd_full", "embed_anchor"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def print_flush(msg: str):
    print(msg, flush=True)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context():
    if torch.cuda.is_available():
        return torch.amp.autocast("cuda", dtype=FORWARD_DTYPE)
    return nullcontext()


def warmup_lr(step_0indexed: int) -> float:
    if step_0indexed < LR_WARMUP_STEPS:
        return LR * (step_0indexed + 1) / LR_WARMUP_STEPS
    return LR


def param_count(model) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"n_total_params": total, "n_trainable_params": trainable}


def causal_ce_loss(logits: torch.Tensor, labels: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous().float()
    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.shape)
    return (loss_per_token * shift_mask).sum() / shift_mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def make_model(arch: str, seed: int):
    set_seed(seed)
    cfg_dict = ARCH_CONFIGS[arch]
    arch_type = cfg_dict["type"]

    if arch_type == "qwen3":
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(
            hidden_size=cfg_dict["hidden_size"],
            num_hidden_layers=cfg_dict["num_hidden_layers"],
            num_attention_heads=cfg_dict["num_attention_heads"],
            num_key_value_heads=cfg_dict["num_key_value_heads"],
            intermediate_size=cfg_dict["intermediate_size"],
            vocab_size=cfg_dict["vocab_size"],
            max_position_embeddings=cfg_dict["max_position_embeddings"],
            tie_word_embeddings=cfg_dict["tie_word_embeddings"],
        )
        model = Qwen3ForCausalLM(cfg).to(torch.bfloat16).to(DEVICE)
    elif arch_type == "gpt2":
        from transformers import GPT2Config, GPT2LMHeadModel
        cfg = GPT2Config(
            n_embd=cfg_dict["n_embd"],
            n_layer=cfg_dict["n_layer"],
            n_head=cfg_dict["n_head"],
            n_inner=cfg_dict["n_inner"],
            vocab_size=cfg_dict["vocab_size"],
            n_positions=cfg_dict["n_positions"],
            tie_word_embeddings=cfg_dict["tie_word_embeddings"],
        )
        model = GPT2LMHeadModel(cfg).to(torch.bfloat16).to(DEVICE)
    elif arch_type == "falcon_h1":
        from transformers import FalconH1Config, FalconH1ForCausalLM
        cfg = FalconH1Config(
            hidden_size=cfg_dict["hidden_size"],
            num_hidden_layers=cfg_dict["num_hidden_layers"],
            num_attention_heads=cfg_dict["num_attention_heads"],
            num_key_value_heads=cfg_dict["num_key_value_heads"],
            intermediate_size=cfg_dict["intermediate_size"],
            vocab_size=cfg_dict["vocab_size"],
            max_position_embeddings=cfg_dict["max_position_embeddings"],
            tie_word_embeddings=cfg_dict["tie_word_embeddings"],
        )
        model = FalconH1ForCausalLM(cfg).to(torch.bfloat16).to(DEVICE)
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")
    return model


def get_tokenizer(arch: str):
    from transformers import AutoTokenizer
    if arch == "qwen3":
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    elif arch == "gpt2":
        tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    elif arch == "falcon_h1":
        tok = AutoTokenizer.from_pretrained("tiiuae/Falcon-H1-0.5B-Base", trust_remote_code=True)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


# ---------------------------------------------------------------------------
# Shared-vocab-subset anchor for cross-architecture embed_anchor arm
# ---------------------------------------------------------------------------

def build_shared_vocab_map(qwen_tok, gpt2_tok) -> dict[int, int]:
    """Map GPT-2 token IDs to Qwen3 token IDs for shared vocabulary tokens.

    Returns {gpt2_id: qwen3_id} for tokens whose decoded text matches exactly.
    """
    shared = {}
    for gpt2_id in range(gpt2_tok.vocab_size):
        try:
            text = gpt2_tok.decode([gpt2_id])
        except Exception:
            continue
        qwen_ids = qwen_tok.encode(text, add_special_tokens=False)
        if len(qwen_ids) == 1:
            roundtrip = qwen_tok.decode(qwen_ids)
            if roundtrip == text:
                shared[gpt2_id] = qwen_ids[0]
    return shared


def load_qwen3_donor():
    """Load frozen Qwen3-0.6B as the anchor donor."""
    sys.path.insert(0, str(ROOT.parent / "models"))
    try:
        from registry import resolve as _resolve_model
        model_id = _resolve_model("qwen3-0.6b").get("hf_id", "Qwen/Qwen3-0.6B")
    except Exception:
        model_id = "Qwen/Qwen3-0.6B"
    from transformers import AutoModelForCausalLM
    donor = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    donor.eval()
    for p in donor.parameters():
        p.requires_grad_(False)
    return donor


def snapshot_donor_embed_lm_head(donor) -> dict[str, torch.Tensor]:
    """Extract embed_tokens and lm_head weights from Qwen3 donor in FP32."""
    params = {}
    for name, p in donor.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            params[name] = p.detach().to(torch.float32).clone()
    return params


def cross_arch_anchor_loss(
    recipient,
    donor_embed_params: dict[str, torch.Tensor],
    arch: str,
    shared_vocab_map: dict[int, int] | None = None,
) -> torch.Tensor:
    """Compute Frobenius anchor loss on embed/lm_head only.

    For same-architecture (Qwen3 recipient): direct param matching by name+shape.
    For cross-architecture (GPT-2 recipient): shared-vocab-subset matching using
    the 42K+ shared tokens between GPT-2 and Qwen3 tokenizers.
    """
    total = torch.zeros((), device=DEVICE, dtype=torch.float32)
    matched = 0

    if arch == "qwen3":
        for name, p in recipient.named_parameters():
            if name not in donor_embed_params:
                continue
            dp = donor_embed_params[name]
            if p.shape == dp.shape:
                total = total + ((p.to(torch.float32) - dp) ** 2).sum()
                matched += 1
            elif p.ndim == 2 and dp.ndim == 2 and p.shape[0] == dp.shape[0]:
                min_dim = min(p.shape[1], dp.shape[1])
                total = total + ((p.to(torch.float32)[:, :min_dim] - dp[:, :min_dim]) ** 2).sum()
                matched += 1
            elif p.ndim == 2 and dp.ndim == 2 and p.shape[1] == dp.shape[1]:
                min_rows = min(p.shape[0], dp.shape[0])
                total = total + ((p.to(torch.float32)[:min_rows] - dp[:min_rows]) ** 2).sum()
                matched += 1
    elif arch == "gpt2" and shared_vocab_map is not None:
        recip_embed = None
        donor_embed = None
        for name, p in recipient.named_parameters():
            if name == "transformer.wte.weight":
                recip_embed = p
                break
        for dname, dp in donor_embed_params.items():
            if "embed_tokens" in dname:
                donor_embed = dp
                break
        if recip_embed is not None and donor_embed is not None:
            gpt2_ids = sorted(shared_vocab_map.keys())
            qwen_ids = [shared_vocab_map[g] for g in gpt2_ids]
            gpt2_idx = torch.tensor(gpt2_ids, dtype=torch.long, device=DEVICE)
            qwen_idx = torch.tensor(qwen_ids, dtype=torch.long, device=DEVICE)
            r_rows = recip_embed[gpt2_idx].to(torch.float32)
            d_rows = donor_embed.to(DEVICE)[qwen_idx]
            if r_rows.shape[1] == d_rows.shape[1]:
                total = total + ((r_rows - d_rows) ** 2).sum()
                matched += len(gpt2_ids)
            else:
                min_dim = min(r_rows.shape[1], d_rows.shape[1])
                total = total + ((r_rows[:, :min_dim] - d_rows[:, :min_dim]) ** 2).sum()
                matched += len(gpt2_ids)

    if matched == 0:
        raise RuntimeError(f"cross_arch_anchor_loss: no parameters matched for arch={arch}")
    return total


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_c4_pools(tok, n_train: int, n_val: int, seq_len: int) -> dict:
    """Load and tokenize C4 data pools for training and validation.

    Train data from C4 train split; val/probe from C4 validation split
    to avoid prefix overlap with teacher-generated texts (Codex cycle 84 fix).
    """
    from datasets import load_dataset

    ds_train = load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=True)
    train_texts = []
    for item in ds_train:
        train_texts.append(item["text"])
        if len(train_texts) >= n_train + 512:
            break
    rng = np.random.default_rng(C4_TRAIN_SEED)
    rng.shuffle(train_texts)
    train_texts = train_texts[:n_train]

    ds_val = load_dataset("allenai/c4", "en", split="validation", streaming=True, trust_remote_code=True)
    val_probe_texts = []
    for item in ds_val:
        val_probe_texts.append(item["text"])
        if len(val_probe_texts) >= n_val + PROBE_WINDOWS + 128:
            break
    rng_val = np.random.default_rng(C4_VAL_SEED)
    rng_val.shuffle(val_probe_texts)
    val_texts = val_probe_texts[:n_val]
    probe_texts = val_probe_texts[n_val:n_val + PROBE_WINDOWS]

    def tokenize_batch(text_list, max_windows):
        all_ids = []
        all_mask = []
        for text in text_list:
            enc = tok(text, truncation=True, max_length=seq_len,
                      padding="max_length", return_tensors="pt")
            all_ids.append(enc["input_ids"])
            all_mask.append(enc["attention_mask"])
            if len(all_ids) >= max_windows:
                break
        return torch.cat(all_ids, dim=0)[:max_windows], torch.cat(all_mask, dim=0)[:max_windows]

    train_ids, train_mask = tokenize_batch(train_texts, n_train)
    val_ids, val_mask = tokenize_batch(val_texts, n_val)
    probe_ids, probe_mask = tokenize_batch(probe_texts, PROBE_WINDOWS)

    probe_batch = {
        "input_ids": probe_ids,
        "attention_mask": probe_mask,
    }

    return {
        "train_ids": train_ids,
        "train_mask": train_mask,
        "val_ids": val_ids,
        "val_mask": val_mask,
        "probe_batch": probe_batch,
    }


def generate_teacher_texts(n_texts: int) -> list[str]:
    """Generate teacher texts from Qwen3-0.6B for seq_kd_full arm."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(ROOT.parent / "models"))
    try:
        from registry import resolve as _resolve_model
        model_id = _resolve_model("qwen3-0.6b").get("hf_id", "Qwen/Qwen3-0.6B")
    except Exception:
        model_id = "Qwen/Qwen3-0.6B"

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    model.config.pad_token_id = tok.pad_token_id
    model.eval()

    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=True)
    seed_texts = []
    for item in ds:
        seed_texts.append(item["text"][:200])
        if len(seed_texts) >= n_texts + 64:
            break

    texts = []
    batch_size = 8
    for i in range(0, min(len(seed_texts), n_texts), batch_size):
        batch = seed_texts[i:i + batch_size]
        enc = tok(batch, truncation=True, max_length=64,
                  padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=192,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                pad_token_id=tok.pad_token_id,
            )
        for seq in out:
            texts.append(tok.decode(seq, skip_special_tokens=True))
        if len(texts) >= n_texts:
            break

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return texts[:n_texts]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_nll(model, val_ids, val_mask) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    for i in range(0, val_ids.shape[0], EVAL_BATCH_SIZE):
        ids = val_ids[i:i + EVAL_BATCH_SIZE].to(DEVICE)
        mask = val_mask[i:i + EVAL_BATCH_SIZE].to(DEVICE)
        with autocast_context():
            out = model(input_ids=ids, attention_mask=mask, use_cache=False)
        logits = out.logits[:, :-1, :]
        targets = ids[:, 1:]
        tmask = mask[:, 1:].float()
        loss_per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view(targets.shape)
        n_tokens = tmask.sum().item()
        total_loss += (loss_per_token * tmask).sum().item()
        total_tokens += n_tokens
        preds = logits.argmax(dim=-1)
        correct += ((preds == targets) * tmask).sum().item()
    model.train()
    nll = total_loss / max(total_tokens, 1)
    acc = correct / max(total_tokens, 1)
    return {"nll": nll, "top1_acc": acc}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

AGNOSTIC_FEATURE_NAMES = [
    "early_loss",
    "mid_spectral_alpha",
    "mid_participation_ratio",
    "mid_sqrt_pr_alpha",
    "depth_alpha_drift",
    "depth_pr_drift",
    "depth_sqrt_pr_alpha_drift",
    "twonn_intrinsic_dim",
    "knn10_clustering_coeff",
    "gradient_noise_scale",
    "grad_norm_mean",
    "grad_norm_var",
    "curvature_top_eigen_proxy",
    "hidden_norm_early_late_ratio",
    "hidden_var_early_late_ratio",
    "norm_param_early_late_ratio",
]

PURE_GEOMETRY_FEATURE_NAMES = [
    "mid_spectral_alpha",
    "mid_participation_ratio",
    "mid_sqrt_pr_alpha",
    "depth_alpha_drift",
    "depth_pr_drift",
    "depth_sqrt_pr_alpha_drift",
    "twonn_intrinsic_dim",
    "knn10_clustering_coeff",
    "hidden_norm_early_late_ratio",
    "hidden_var_early_late_ratio",
]

MANIFOLD_ONLY_FEATURE_NAMES = [
    "mid_spectral_alpha",
    "mid_participation_ratio",
    "mid_sqrt_pr_alpha",
    "depth_alpha_drift",
    "depth_pr_drift",
    "depth_sqrt_pr_alpha_drift",
    "twonn_intrinsic_dim",
    "knn10_clustering_coeff",
]

PURE_TELEMETRY_FEATURE_NAMES = [
    "early_loss",
    "gradient_noise_scale",
    "grad_norm_mean",
    "grad_norm_var",
    "curvature_top_eigen_proxy",
    "norm_param_early_late_ratio",
]

SHESHA_FEATURE_NAMES = [
    "shesha_feature_split",
    "shesha_sample_split",
    "shesha_anchor_stability",
]

QWEN_REF_FEATURE_NAMES = [
    "hidden_to_qwen_ref_pca64_procrustes_residual",
    "hidden_to_qwen_ref_pca64_rsa_distance",
    "embed_to_qwen_ref_pca64_procrustes_residual",
    "embed_to_qwen_ref_pca64_rsa_distance",
    "lm_head_to_qwen_ref_pca64_procrustes_residual",
    "lm_head_to_qwen_ref_pca64_rsa_distance",
    "embed_reference_rows_used",
    "lm_head_reference_rows_used",
]


def _load_qwen3_reference_geometry(qwen_tok) -> dict[str, Any]:
    """Load Qwen3-0.6B reference geometry arrays for Model A features."""
    from transformers import AutoModelForCausalLM
    sys.path.insert(0, str(ROOT.parent / "models"))
    try:
        from registry import resolve as _resolve_model
        model_id = _resolve_model("qwen3-0.6b").get("hf_id", "Qwen/Qwen3-0.6B")
    except Exception:
        model_id = "Qwen/Qwen3-0.6B"

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    ref_model.eval()
    if hasattr(ref_model.config, "use_cache"):
        ref_model.config.use_cache = False

    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True,
                      trust_remote_code=True)
    probe_texts = []
    for item in ds:
        probe_texts.append(item["text"])
        if len(probe_texts) >= PROBE_WINDOWS:
            break
    enc = qwen_tok(probe_texts, truncation=True, max_length=SEQ_LEN,
                   padding="max_length", return_tensors="pt")
    probe_ids = enc["input_ids"][:PROBE_WINDOWS]
    probe_mask = enc["attention_mask"][:PROBE_WINDOWS]

    with torch.no_grad():
        out = ref_model(
            input_ids=probe_ids.to(DEVICE),
            attention_mask=probe_mask.to(DEVICE),
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states = out.hidden_states
    mid_idx = len(hidden_states) // 2
    mid_h = hidden_states[mid_idx].float().cpu().numpy()
    mask_np = probe_mask.numpy()
    rows = []
    for b in range(mid_h.shape[0]):
        valid = mask_np[b].sum()
        rows.append(mid_h[b, :valid])
    ref_hidden = np.concatenate(rows, axis=0)[:g180.FEATURE_MAX_POINTS]

    ref_embed = None
    for name, p in ref_model.named_parameters():
        if "embed_tokens" in name:
            ref_embed = p.detach().float().cpu().numpy()[:g180.EMBED_MAX_ROWS]
            break

    ref_head = None
    for name, p in ref_model.named_parameters():
        if "lm_head" in name:
            ref_head = p.detach().float().cpu().numpy()[:g180.EMBED_MAX_ROWS]
            break

    del ref_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "reference_hidden": ref_hidden,
        "reference_embedding": ref_embed,
        "reference_lm_head": ref_head,
    }


def extract_features_for_cell(model, probe_batch, arch: str,
                               include_qwen_ref: bool = True) -> dict[str, float]:
    """Extract geometry features at the feature step.

    Uses g180.extract_features for the core computation, then filters
    to agnostic-only or full feature set.
    """
    layer_indices = [1, 1 + N_LAYERS // 2, N_LAYERS]
    features = g180.extract_features(model, probe_batch, layer_indices)
    if not include_qwen_ref:
        features = {k: v for k, v in features.items()
                    if not any(ref in k for ref in ["qwen_ref", "reference_rows"])}
    optional_prefixes = ("shesha_",)
    bad = [k for k, v in features.items()
           if not math.isfinite(float(v)) and not k.startswith(optional_prefixes)]
    if bad:
        raise RuntimeError(f"non-finite features: {bad}")
    return features


# ---------------------------------------------------------------------------
# Training loop for a single cell
# ---------------------------------------------------------------------------

def train_cell(
    *,
    arch: str,
    arm: str,
    seed: int,
    pools: dict,
    teacher_pools: dict | None = None,
    donor_embed_params: dict[str, torch.Tensor] | None = None,
    shared_vocab_map: dict[int, int] | None = None,
    qwen_ref_probe: dict | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Train one cell and return result dict with features + final NLL."""
    train_steps = 20 if smoke else TRAIN_STEPS
    feature_step = max(1, int(math.ceil(0.03 * train_steps)))

    cell_id = f"{arch}_{arm}_s{seed}"
    print_flush(f"\n=== Cell {cell_id} ===")
    t_cell = time.time()

    set_seed(seed)
    model = make_model(arch, seed)
    counts = param_count(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )

    train_ids = pools["train_ids"]
    train_mask = pools["train_mask"]
    n_examples = train_ids.shape[0]

    if arm == "seq_kd_full" and teacher_pools is not None:
        train_ids = teacher_pools["train_ids"]
        train_mask = teacher_pools["train_mask"]
        n_examples = train_ids.shape[0]

    rng = np.random.default_rng(seed)
    schedule = rng.integers(0, n_examples, size=(train_steps, TRAIN_BATCH_SIZE), dtype=np.int64)

    anchor_lambda_constant = 0.0
    if arm == "embed_anchor" and donor_embed_params is not None:
        anchor_lambda_constant = 0.01
        print_flush(f"    anchor: lambda={anchor_lambda_constant} (constant, no decay; matches g165)")

    initial_metrics = evaluate_nll(model, pools["val_ids"], pools["val_mask"])
    print_flush(f"    params={counts['n_total_params']/1e6:.2f}M "
                f"step=0 c4_nll={initial_metrics['nll']:.4f}")

    train_log = []
    trajectory_losses = {}
    features = None
    early_loss = float("nan")

    model.train()
    t_train = time.time()

    for step in range(1, train_steps + 1):
        current_lr = warmup_lr(step - 1)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        batch_idx = torch.as_tensor(schedule[step - 1], dtype=torch.long)
        ids = train_ids[batch_idx].to(DEVICE)
        mask = train_mask[batch_idx].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            ce = causal_ce_loss(logits, ids, mask)

        loss = ce

        if arm == "embed_anchor" and donor_embed_params is not None:
            anchor = cross_arch_anchor_loss(
                model, donor_embed_params, arch, shared_vocab_map
            )
            loss = loss + anchor_lambda_constant * anchor

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step} cell={cell_id}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP, error_if_nonfinite=True)
        optimizer.step()

        early_loss = float(ce.detach().float().cpu().item())

        if step in TRAJECTORY_STEPS or (smoke and step in [5, 10, 15, 20]):
            traj_subset = min(128, pools["val_ids"].shape[0])
            traj_nll = evaluate_nll(
                model, pools["val_ids"][:traj_subset], pools["val_mask"][:traj_subset]
            )
            trajectory_losses[step] = traj_nll["nll"]
            model.train()

        if step == feature_step and features is None:
            probe = dict(pools["probe_batch"])
            probe["early_loss"] = early_loss
            if qwen_ref_probe is not None:
                probe.update(qwen_ref_probe)
            features = extract_features_for_cell(model, probe, arch)
            model.train()

        if step % LOG_EVERY == 0 or step == train_steps:
            row = {"step": step, "lr": current_lr, "ce_loss": early_loss,
                   "elapsed_s": time.time() - t_train}
            if step % EVAL_EVERY == 0 or step == train_steps:
                row.update(evaluate_nll(model, pools["val_ids"], pools["val_mask"]))
                model.train()
                print_flush(f"    step={step:4d} ce={early_loss:.4f} "
                            f"c4_nll={row.get('nll', 'N/A'):.4f} ({row['elapsed_s']:.0f}s)")
            train_log.append(row)

    if features is None:
        probe = dict(pools["probe_batch"])
        probe["early_loss"] = early_loss
        if qwen_ref_probe is not None:
            probe.update(qwen_ref_probe)
        features = extract_features_for_cell(model, probe, arch)

    final_metrics = evaluate_nll(model, pools["val_ids"], pools["val_mask"])
    if not math.isfinite(final_metrics["nll"]):
        raise RuntimeError(f"non-finite final NLL cell={cell_id}: {final_metrics['nll']}")
    wallclock_s = time.time() - t_cell

    result = {
        "cell_id": cell_id,
        "arch": arch,
        "arm": arm,
        "seed": seed,
        "train_steps": train_steps,
        "feature_step": feature_step,
        "n_total_params": counts["n_total_params"],
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "final_nll": float(final_metrics["nll"]),
        "early_loss": float(features.get("early_loss", early_loss)),
        "trajectory_losses": trajectory_losses,
        "features": features,
        "train_log": train_log,
        "wallclock_s": wallclock_s,
    }

    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print_flush(f"    DONE {cell_id}: final_nll={result['final_nll']:.4f} "
                f"({wallclock_s:.0f}s)")
    return result


# ---------------------------------------------------------------------------
# Labels (normalized per prereg)
# ---------------------------------------------------------------------------

def compute_normalized_labels(cells: list[dict]) -> list[dict]:
    """Compute label = (scratch_final - arm_final) / scratch_final.

    Returns list of dicts with 'label' field added.
    """
    scratch_by = {}
    for c in cells:
        if c["arm"] == "scratch_ce":
            scratch_by[(c["arch"], c["seed"])] = c["final_nll"]

    labeled = []
    for c in cells:
        if c["arm"] == "scratch_ce":
            continue
        key = (c["arch"], c["seed"])
        if key not in scratch_by:
            continue
        scratch_nll = scratch_by[key]
        if scratch_nll < 1e-8:
            continue
        label = (scratch_nll - c["final_nll"]) / scratch_nll
        labeled.append({**c, "label": label})
    return labeled


# ---------------------------------------------------------------------------
# Analysis: Ridge with CV alpha selection
# ---------------------------------------------------------------------------

def fit_ridge_cv(X_train, y_train, alpha_grid=None):
    """Fit Ridge with 5-fold CV for alpha selection on train data only."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    alpha_grid = alpha_grid or RIDGE_ALPHA_GRID
    best_alpha = 1.0
    best_score = float("inf")
    kf = KFold(n_splits=min(5, len(y_train)), shuffle=True, random_state=RANDOM_STATE)
    for alpha in alpha_grid:
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            m = Ridge(alpha=alpha)
            m.fit(X_train[train_idx], y_train[train_idx])
            pred = m.predict(X_train[val_idx])
            scores.append(float(np.mean((y_train[val_idx] - pred) ** 2)))
        mean_score = np.mean(scores)
        if mean_score < best_score:
            best_score = mean_score
            best_alpha = alpha
    model = Ridge(alpha=best_alpha)
    model.fit(X_train, y_train)
    model._selected_alpha = best_alpha
    return model


def feature_matrix(cells: list[dict], feature_names: list[str],
                    impute_medians: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build (n_cells, n_features) matrix from cell dicts.

    Returns (X, medians) where medians can be passed to subsequent calls
    for test-set imputation using train statistics only.
    """
    X = np.zeros((len(cells), len(feature_names)), dtype=np.float64)
    for i, c in enumerate(cells):
        feats = c.get("features", {})
        for j, fname in enumerate(feature_names):
            val = feats.get(fname, float("nan"))
            X[i, j] = float(val) if val is not None else float("nan")
    if impute_medians is None:
        medians = np.nanmedian(X, axis=0)
    else:
        medians = impute_medians
    nan_mask = np.isnan(X)
    if nan_mask.any():
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = medians[j] if np.isfinite(medians[j]) else 0.0
    return X, medians


def standardize(X_train, X_test):
    """Z-score standardize using train stats only."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-12] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _baseline_features_raw(cells: list[dict], baseline_type: str) -> np.ndarray:
    """Build raw feature matrix (may contain NaN). No imputation."""
    n = len(cells)
    if baseline_type == "scalar_early_loss":
        return np.array([[c["early_loss"]] for c in cells], dtype=np.float64)
    elif baseline_type == "trajectory":
        steps = TRAJECTORY_STEPS
        X = np.zeros((n, len(steps) + 2), dtype=np.float64)
        for i, c in enumerate(cells):
            traj = c.get("trajectory_losses", {})
            for j, s in enumerate(steps):
                X[i, j] = traj.get(s, traj.get(str(s), float("nan")))
            valid_steps = [(s, X[i, j]) for j, s in enumerate(steps)
                           if np.isfinite(X[i, j]) and s <= 108]
            if len(valid_steps) >= 3:
                log_s = np.log(np.array([v[0] for v in valid_steps], dtype=np.float64))
                nll_v = np.array([v[1] for v in valid_steps], dtype=np.float64)
                coeffs = np.polyfit(log_s, nll_v, 2)
                X[i, -2] = coeffs[1]
                X[i, -1] = coeffs[0]
            elif len(valid_steps) >= 2:
                log_s = np.log(np.array([v[0] for v in valid_steps], dtype=np.float64))
                nll_v = np.array([v[1] for v in valid_steps], dtype=np.float64)
                X[i, -2] = np.polyfit(log_s, nll_v, 1)[0]
                X[i, -1] = 0.0
        return X
    elif baseline_type == "gradient_stats":
        X = np.zeros((n, 3), dtype=np.float64)
        for i, c in enumerate(cells):
            feats = c.get("features", {})
            X[i, 0] = feats.get("gradient_noise_scale", float("nan"))
            X[i, 1] = feats.get("grad_norm_mean", float("nan"))
            X[i, 2] = feats.get("grad_norm_var", float("nan"))
        return X
    elif baseline_type == "arm_labels":
        arms = ARM_LABELS
        X = np.zeros((n, len(arms)), dtype=np.float64)
        for i, c in enumerate(cells):
            for j, a in enumerate(arms):
                X[i, j] = 1.0 if c["arm"] == a else 0.0
        return X
    elif baseline_type == "delayed_loss":
        X = np.zeros((n, 2), dtype=np.float64)
        for i, c in enumerate(cells):
            traj = c.get("trajectory_losses", {})
            X[i, 0] = traj.get(200, traj.get("200", float("nan")))
            X[i, 1] = traj.get(500, traj.get("500", float("nan")))
        return X
    elif baseline_type == "within_arm_residual":
        arms = ARM_LABELS
        steps = TRAJECTORY_STEPS
        X = np.zeros((n, len(arms) + len(steps) + 2), dtype=np.float64)
        for i, c in enumerate(cells):
            for j, a in enumerate(arms):
                X[i, j] = 1.0 if c["arm"] == a else 0.0
            traj = c.get("trajectory_losses", {})
            for j, s in enumerate(steps):
                X[i, len(arms) + j] = traj.get(s, traj.get(str(s), float("nan")))
            valid_steps = [(s, X[i, len(arms) + j]) for j, s in enumerate(steps)
                           if np.isfinite(X[i, len(arms) + j]) and s <= 108]
            if len(valid_steps) >= 3:
                log_s = np.log(np.array([v[0] for v in valid_steps], dtype=np.float64))
                nll_v = np.array([v[1] for v in valid_steps], dtype=np.float64)
                coeffs = np.polyfit(log_s, nll_v, 2)
                X[i, -2] = coeffs[1]
                X[i, -1] = coeffs[0]
            elif len(valid_steps) >= 2:
                log_s = np.log(np.array([v[0] for v in valid_steps], dtype=np.float64))
                nll_v = np.array([v[1] for v in valid_steps], dtype=np.float64)
                X[i, -2] = np.polyfit(log_s, nll_v, 1)[0]
                X[i, -1] = 0.0
        return X
    elif baseline_type == "combined_telemetry":
        parts = [_baseline_features_raw(cells, bt)
                 for bt in ["arm_labels", "trajectory", "gradient_stats", "delayed_loss"]]
        return np.hstack(parts)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


def baseline_features(cells: list[dict], baseline_type: str,
                      impute_medians: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build feature matrix with NaN imputation.

    Returns (X, medians) where medians can be passed for test-set imputation
    using train-only statistics.
    """
    X = _baseline_features_raw(cells, baseline_type)
    if impute_medians is None:
        medians = np.nanmedian(X, axis=0)
    else:
        medians = impute_medians
    nan_mask = np.isnan(X)
    if nan_mask.any():
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = medians[j] if np.isfinite(medians[j]) else 0.0
    return X, medians


BASELINE_TYPES = [
    "scalar_early_loss",
    "trajectory",
    "gradient_stats",
    "arm_labels",
    "delayed_loss",
    "within_arm_residual",
    "combined_telemetry",
]


# ---------------------------------------------------------------------------
# Leave-one-architecture-out evaluation
# ---------------------------------------------------------------------------

def loao_evaluate(
    cells: list[dict],
    geometry_feature_names: list[str],
    model_label: str,
) -> dict[str, Any]:
    """Run leave-one-architecture-out evaluation for one geometry model.

    Returns per-fold and aggregate results including all baseline comparisons.
    """
    archs = sorted(set(c["arch"] for c in cells))
    if len(archs) < 2:
        raise ValueError(f"Need >= 2 architectures for LOAO, got {archs}")

    folds = {}
    for test_arch in archs:
        train_cells = [c for c in cells if c["arch"] != test_arch]
        test_cells = [c for c in cells if c["arch"] == test_arch]

        y_train = np.array([c["label"] for c in train_cells], dtype=np.float64)
        y_test = np.array([c["label"] for c in test_cells], dtype=np.float64)

        X_train_geo, train_medians = feature_matrix(train_cells, geometry_feature_names)
        X_test_geo, _ = feature_matrix(test_cells, geometry_feature_names, impute_medians=train_medians)
        X_train_geo_s, X_test_geo_s = standardize(X_train_geo, X_test_geo)

        geo_model = fit_ridge_cv(X_train_geo_s, y_train)
        geo_pred = geo_model.predict(X_test_geo_s)
        geo_mse = float(np.mean((y_test - geo_pred) ** 2))
        var_y = float(np.var(y_test)) * len(y_test) if len(y_test) > 1 else 1.0
        ss_res = float(np.sum((y_test - geo_pred) ** 2))
        ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
        geo_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

        baseline_results = {}
        best_baseline_mse = float("inf")
        best_baseline_name = None
        best_baseline_pred = None

        for bt in BASELINE_TYPES:
            X_train_b, train_b_medians = baseline_features(train_cells, bt)
            X_test_b, _ = baseline_features(test_cells, bt, impute_medians=train_b_medians)
            X_train_b_s, X_test_b_s = standardize(X_train_b, X_test_b)
            b_model = fit_ridge_cv(X_train_b_s, y_train)
            b_pred = b_model.predict(X_test_b_s)
            b_mse = float(np.mean((y_test - b_pred) ** 2))
            b_r2 = 1.0 - float(np.sum((y_test - b_pred)**2)) / ss_tot if ss_tot > 1e-12 else float("nan")
            baseline_results[bt] = {"mse": b_mse, "r2": b_r2, "alpha": b_model._selected_alpha}
            if b_mse < best_baseline_mse:
                best_baseline_mse = b_mse
                best_baseline_name = bt
                best_baseline_pred = b_pred

        arm_mean_pred = np.zeros(len(test_cells), dtype=np.float64)
        arm_means = {}
        for a in ARM_LABELS:
            train_arm_labels = [c["label"] for c in train_cells if c["arm"] == a]
            arm_means[a] = float(np.mean(train_arm_labels)) if train_arm_labels else 0.0
        for i, c in enumerate(test_cells):
            arm_mean_pred[i] = arm_means.get(c["arm"], 0.0)
        arm_mean_mse = float(np.mean((y_test - arm_mean_pred) ** 2))
        baseline_results["arm_mean"] = {"mse": arm_mean_mse}
        if arm_mean_mse < best_baseline_mse:
            best_baseline_mse = arm_mean_mse
            best_baseline_name = "arm_mean"
            best_baseline_pred = arm_mean_pred

        mse_reduction = (best_baseline_mse - geo_mse) / best_baseline_mse if best_baseline_mse > 1e-12 else float("nan")

        test_seed_ids = np.array([c["seed"] for c in test_cells])
        paired_boot = paired_bootstrap_mse(y_test, best_baseline_pred, geo_pred, test_seed_ids)

        shuffled_p = shuffled_geometry_test(
            X_train_geo_s, X_test_geo_s, y_train, y_test, geo_mse
        )

        auroc = bad_run_auroc(y_test, geo_pred, best_baseline_pred)

        kill_result = simulated_kill(y_test, geo_pred, test_cells)

        folds[test_arch] = {
            "n_train": len(train_cells),
            "n_test": len(test_cells),
            "geometry_mse": geo_mse,
            "geometry_r2": geo_r2,
            "geometry_alpha": geo_model._selected_alpha,
            "best_baseline_name": best_baseline_name,
            "best_baseline_mse": best_baseline_mse,
            "mse_reduction_vs_best": mse_reduction,
            "paired_bootstrap": paired_boot,
            "shuffled_geometry_p": shuffled_p,
            "bad_run_auroc": auroc,
            "simulated_kill": kill_result,
            "baselines": baseline_results,
        }

    return {
        "model_label": model_label,
        "feature_names": geometry_feature_names,
        "folds": folds,
    }


def paired_bootstrap_mse(y, pred_base, pred_geo, seed_ids, n_boot=BOOTSTRAP_N):
    """Paired seed-block bootstrap for MSE improvement.

    Samples by seed block (preserving all arms per seed) per prereg spec.
    seed_ids: array of seed identifiers, one per cell.
    """
    rng = np.random.default_rng(RANDOM_STATE + 7)
    unique_seeds = np.unique(seed_ids)
    n_seeds = len(unique_seeds)
    seed_to_idx = {s: np.where(seed_ids == s)[0] for s in unique_seeds}

    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sampled_seeds = rng.choice(unique_seeds, size=n_seeds, replace=True)
        idx = np.concatenate([seed_to_idx[s] for s in sampled_seeds])
        base_mse = np.mean((y[idx] - pred_base[idx]) ** 2)
        geo_mse = np.mean((y[idx] - pred_geo[idx]) ** 2)
        deltas[i] = base_mse - geo_mse
    return {
        "mean": float(deltas.mean()),
        "ci95_lo": float(np.percentile(deltas, 2.5)),
        "ci95_hi": float(np.percentile(deltas, 97.5)),
        "p_gt_0": float(np.mean(deltas > 0)),
    }


def shuffled_geometry_test(X_train, X_test, y_train, y_test, real_mse,
                           n_iter=SHUFFLED_GEOMETRY_ITERS):
    """Permutation test: shuffle geometry features across rows."""
    rng = np.random.default_rng(RANDOM_STATE + 13)
    n_better = 0
    for _ in range(n_iter):
        perm = rng.permutation(len(X_train))
        X_shuf = X_train[perm]
        m = fit_ridge_cv(X_shuf, y_train)
        pred = m.predict(X_test)
        shuf_mse = float(np.mean((y_test - pred) ** 2))
        if shuf_mse <= real_mse:
            n_better += 1
    return float(n_better / n_iter)


def bad_run_auroc(y_true, geo_pred, base_pred, threshold_quantile=0.3):
    """AUROC for detecting bad runs (bottom 30% by true gain)."""
    if len(y_true) < 4:
        return {"geometry": float("nan"), "baseline": float("nan"), "delta": float("nan")}
    threshold = np.quantile(y_true, threshold_quantile)
    bad_label = (y_true <= threshold).astype(int)
    if bad_label.sum() == 0 or bad_label.sum() == len(bad_label):
        return {"geometry": float("nan"), "baseline": float("nan"), "delta": float("nan")}
    from sklearn.metrics import roc_auc_score
    geo_score = -geo_pred
    base_score = -base_pred
    geo_auroc = float(roc_auc_score(bad_label, geo_score))
    base_auroc = float(roc_auc_score(bad_label, base_score))
    return {"geometry": geo_auroc, "baseline": base_auroc, "delta": geo_auroc - base_auroc}


def simulated_kill(y_true, geo_pred, cells, kill_fraction=0.3):
    """Simulate killing bottom 30% predicted runs."""
    n = len(y_true)
    n_kill = max(1, int(n * kill_fraction))
    order = np.argsort(geo_pred)
    killed_idx = set(order[:n_kill].tolist())
    survived_idx = [i for i in range(n) if i not in killed_idx]

    total_gain = float(np.sum(y_true))
    survived_gain = float(np.sum(y_true[survived_idx]))
    gain_retained = survived_gain / total_gain if abs(total_gain) > 1e-12 else float("nan")
    compute_saved = float(n_kill / n)

    true_bottom = set(np.argsort(y_true)[:n_kill].tolist())
    precision = len(killed_idx & true_bottom) / n_kill if n_kill > 0 else 0.0

    return {
        "n_killed": n_kill,
        "n_survived": len(survived_idx),
        "compute_saved_fraction": compute_saved,
        "gain_retained_fraction": gain_retained,
        "kill_precision": precision,
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

CO_PRIMARY_MODELS = {"model_a_full_geometry", "model_b_reference_free"}


def compute_verdict(loao_results: dict) -> dict[str, Any]:
    """Determine PASS / WEAK PASS / FAIL per prereg criteria.

    Only co-primary models (A, B) gate the verdict. C/D/E are exploratory.
    """
    all_pass = True
    any_weak = False
    details = {}

    for model_label, result in loao_results.items():
        is_primary = model_label in CO_PRIMARY_MODELS
        folds = result["folds"]
        fold_pass = True
        fold_weak = False
        for arch, fold in folds.items():
            mse_red = fold["mse_reduction_vs_best"]
            ci_lo = fold["paired_bootstrap"]["ci95_lo"]
            r2 = fold["geometry_r2"]
            shuf_p = fold["shuffled_geometry_p"]
            auroc = fold["bad_run_auroc"]["geometry"]
            auroc_delta = fold["bad_run_auroc"]["delta"]
            kill_save = fold["simulated_kill"]["compute_saved_fraction"]
            kill_retain = fold["simulated_kill"]["gain_retained_fraction"]

            fold_checks = {
                "mse_reduction_ge_25pct": mse_red >= 0.25,
                "ci_lo_gt_0": ci_lo > 0,
                "r2_ge_0.20": r2 >= 0.20,
                "shuffled_p_le_0.01": shuf_p <= 0.01,
                "auroc_ge_0.75": auroc >= 0.75 if not np.isnan(auroc) else False,
                "auroc_delta_ge_0.05": auroc_delta >= 0.05 if not np.isnan(auroc_delta) else False,
                "kill_saves_ge_20pct": kill_save >= 0.20,
                "kill_retains_ge_90pct": kill_retain >= 0.90 if not np.isnan(kill_retain) else False,
            }
            all_checks_pass = all(fold_checks.values())
            weak_checks = ci_lo > 0 and (0.10 <= mse_red < 0.25 or not all_checks_pass)

            if not all_checks_pass:
                fold_pass = False
            if weak_checks:
                fold_weak = True

            details[f"{model_label}_{arch}"] = {
                "checks": fold_checks,
                "all_pass": all_checks_pass,
                "mse_reduction": mse_red,
                "r2": r2,
                "shuffled_p": shuf_p,
            }

        if is_primary:
            if not fold_pass:
                all_pass = False
            if fold_weak:
                any_weak = True

    missing_primary = CO_PRIMARY_MODELS - set(loao_results)
    if missing_primary:
        return {
            "verdict": "FAIL",
            "reason": f"missing co-primary models: {sorted(missing_primary)}",
            "details": details,
        }

    if all_pass:
        verdict = "PASS"
    elif any_weak:
        verdict = "WEAK PASS"
    else:
        verdict = "FAIL"

    return {"verdict": verdict, "details": details}


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------

def save_incremental(out_path: Path, data: dict):
    """Atomic incremental save."""
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str, allow_nan=False)
    tmp.replace(out_path)


def load_existing(out_path: Path) -> dict | None:
    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Shesha post-hoc augmentation
# ---------------------------------------------------------------------------

def shesha_augment_main():
    """Replay each cell to feature_step, extract Shesha features, re-run analysis."""
    print_flush("=== Shesha Augmentation Mode ===")
    existing = load_existing(OUT_PATH)
    if not existing or "cells" not in existing:
        print_flush("ERROR: no g182 results found. Run main experiment first.")
        return

    cells = existing["cells"]
    need_shesha = [c for c in cells if "shesha_feature_split" not in c.get("features", {})]
    print_flush(f"  {len(cells)} cells total, {len(need_shesha)} need Shesha features")

    if not need_shesha:
        print_flush("  All cells already have Shesha features. Running analysis only.")
    else:
        tok_qwen = get_tokenizer("qwen3")
        tok_gpt2 = get_tokenizer("gpt2")
        toks = {"qwen3": tok_qwen, "gpt2": tok_gpt2}

        pools_by_arch = {}
        for arch in ARCH_CONFIGS:
            pools_by_arch[arch] = load_c4_pools(toks[arch], N_TRAIN_WINDOWS, N_C4_VAL_WINDOWS, SEQ_LEN)

        donor_embed = None
        shared_vocab = None
        has_anchor_cells = any(c["arm"] == "embed_anchor" for c in need_shesha)
        if has_anchor_cells:
            donor = load_qwen3_donor()
            donor_embed = snapshot_donor_embed_lm_head(donor)
            tok_qwen_local = get_tokenizer("qwen3")
            tok_gpt2_local = get_tokenizer("gpt2")
            shared_vocab = build_shared_vocab_map(tok_qwen_local, tok_gpt2_local)
            del donor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        teacher_pools_by_arch = {}
        has_kd_cells = any(c["arm"] == "seq_kd_full" for c in need_shesha)
        if has_kd_cells:
            teacher_cache_path = CACHE_DIR / "teacher_texts.json"
            if teacher_cache_path.exists():
                with open(teacher_cache_path, encoding="utf-8") as f:
                    teacher_texts = json.load(f)
                print_flush(f"  Loaded {len(teacher_texts)} cached teacher texts for replay")
            else:
                raise RuntimeError(
                    f"Teacher text cache not found at {teacher_cache_path}. "
                    "Run main experiment first to generate and cache teacher texts. "
                    "Regenerating would produce different texts (sampling-based), "
                    "making Shesha features non-comparable."
                )
            for arch in ARCH_CONFIGS:
                tok = toks[arch]
                enc_list = [tok(t, truncation=True, max_length=SEQ_LEN,
                               padding="max_length", return_tensors="pt")
                            for t in teacher_texts]
                t_ids = torch.cat([e["input_ids"] for e in enc_list], dim=0)[:N_TRAIN_WINDOWS]
                t_mask = torch.cat([e["attention_mask"] for e in enc_list], dim=0)[:N_TRAIN_WINDOWS]
                teacher_pools_by_arch[arch] = {"train_ids": t_ids, "train_mask": t_mask}

        feature_step = max(1, int(math.ceil(0.03 * TRAIN_STEPS)))
        replayed = 0
        for i, c in enumerate(need_shesha):
            arch, arm, seed = c["arch"], c["arm"], c["seed"]
            cell_id = c["cell_id"]
            print_flush(f"\n  [{i+1}/{len(need_shesha)}] Replaying {cell_id} to step {feature_step}...")

            set_seed(seed)
            model = make_model(arch, seed)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY,
            )

            pools = pools_by_arch[arch]
            train_ids = pools["train_ids"]
            train_mask = pools["train_mask"]
            n_examples = train_ids.shape[0]

            if arm == "seq_kd_full" and arch in teacher_pools_by_arch:
                train_ids = teacher_pools_by_arch[arch]["train_ids"]
                train_mask = teacher_pools_by_arch[arch]["train_mask"]
                n_examples = train_ids.shape[0]

            rng = np.random.default_rng(seed)
            schedule = rng.integers(0, n_examples, size=(TRAIN_STEPS, TRAIN_BATCH_SIZE), dtype=np.int64)

            anchor_lambda = 0.01 if (arm == "embed_anchor" and donor_embed is not None) else 0.0

            model.train()
            for step in range(1, feature_step + 1):
                current_lr = warmup_lr(step - 1)
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
                batch_idx = torch.as_tensor(schedule[step - 1], dtype=torch.long)
                ids = train_ids[batch_idx].to(DEVICE)
                mask = train_mask[batch_idx].to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context():
                    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                    ce = causal_ce_loss(logits, ids, mask)
                loss = ce
                if anchor_lambda > 0:
                    anchor = cross_arch_anchor_loss(model, donor_embed, arch, shared_vocab)
                    loss = loss + anchor_lambda * anchor
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            model.eval()
            probe = dict(pools["probe_batch"])
            probe["early_loss"] = 0.0
            layer_indices = [1, 1 + N_LAYERS // 2, N_LAYERS]
            hidden_states, attention_mask_out, _ = g180._model_hidden_states(model, probe)
            chosen = g180._select_hidden_indices(len(hidden_states), layer_indices)
            mid_idx = chosen[len(chosen) // 2]
            mid = g180._hidden_cloud(hidden_states[mid_idx], attention_mask_out)

            shesha_feats = g180._shesha_features(mid)
            print_flush(f"    Shesha: {shesha_feats}")

            c["features"].update(shesha_feats)
            replayed += 1

            del model, optimizer
            gc.collect()
            torch.cuda.empty_cache()

        existing["cells"] = cells
        save_incremental(OUT_PATH, existing)
        print_flush(f"\n  Saved augmented results to {OUT_PATH}")

    labeled = compute_normalized_labels(cells)
    print_flush(f"\n  {len(labeled)} labeled cells (scratch excluded)")

    loao_results = {}
    for model_label, feat_names in [
        ("model_a_full_geometry", AGNOSTIC_FEATURE_NAMES + QWEN_REF_FEATURE_NAMES),
        ("model_b_reference_free", AGNOSTIC_FEATURE_NAMES),
        ("model_c_pure_geometry", PURE_GEOMETRY_FEATURE_NAMES),
        ("model_c_prime_manifold_only", MANIFOLD_ONLY_FEATURE_NAMES),
        ("model_d_pure_telemetry", PURE_TELEMETRY_FEATURE_NAMES),
        ("model_e_shesha", SHESHA_FEATURE_NAMES),
    ]:
        has_feats = all(
            any(math.isfinite(float(c["features"].get(fn, float("nan")))) for fn in feat_names)
            for c in labeled
        )
        if not has_feats:
            print_flush(f"\n  SKIP {model_label}: missing features for some cells")
            continue
        print_flush(f"\n--- LOAO evaluation: {model_label} ({len(feat_names)} features) ---")
        result = loao_evaluate(labeled, feat_names, model_label)
        loao_results[model_label] = result
        for a, fold in result["folds"].items():
            print_flush(f"    fold={a}: geo_mse={fold['geometry_mse']:.6f} "
                        f"best_base={fold['best_baseline_name']}={fold['best_baseline_mse']:.6f} "
                        f"reduction={fold['mse_reduction_vs_best']:.1%} "
                        f"R2={fold['geometry_r2']:.3f}")

    if loao_results:
        verdict = compute_verdict(loao_results)
        print_flush(f"\n*** VERDICT: {verdict['verdict']} ***")
        existing["shesha_augmented_loao"] = loao_results
        existing["shesha_augmented_verdict"] = verdict
        save_incremental(OUT_PATH, existing)
        print_flush(f"Saved augmented analysis to {OUT_PATH}")


# ---------------------------------------------------------------------------
# Re-analysis (uses saved cells, no training)
# ---------------------------------------------------------------------------

def reanalyze_main():
    """Re-run 5-model LOAO analysis on saved cells without retraining."""
    print_flush("=== Re-analysis Mode (5-model LOAO on saved cells) ===")
    existing = load_existing(OUT_PATH)
    if not existing or "cells" not in existing:
        print_flush("ERROR: no g182 results found. Run main experiment first.")
        return

    cells = existing["cells"]
    print_flush(f"  {len(cells)} cells loaded")

    labeled = compute_normalized_labels(cells)
    print_flush(f"  {len(labeled)} labeled cells (scratch excluded)")

    loao_results = {}
    for model_label, feat_names in [
        ("model_a_full_geometry", AGNOSTIC_FEATURE_NAMES + QWEN_REF_FEATURE_NAMES),
        ("model_b_reference_free", AGNOSTIC_FEATURE_NAMES),
        ("model_c_pure_geometry", PURE_GEOMETRY_FEATURE_NAMES),
        ("model_c_prime_manifold_only", MANIFOLD_ONLY_FEATURE_NAMES),
        ("model_d_pure_telemetry", PURE_TELEMETRY_FEATURE_NAMES),
        ("model_e_shesha", SHESHA_FEATURE_NAMES),
    ]:
        has_feats = all(
            any(math.isfinite(float(c["features"].get(fn, float("nan")))) for fn in feat_names)
            for c in labeled
        )
        if not has_feats:
            print_flush(f"\n  SKIP {model_label}: missing features for some cells")
            continue
        print_flush(f"\n--- LOAO evaluation: {model_label} ({len(feat_names)} features) ---")
        result = loao_evaluate(labeled, feat_names, model_label)
        loao_results[model_label] = result
        for arch, fold in result["folds"].items():
            print_flush(f"    fold={arch}: geo_mse={fold['geometry_mse']:.6f} "
                        f"best_base={fold['best_baseline_name']}={fold['best_baseline_mse']:.6f} "
                        f"reduction={fold['mse_reduction_vs_best']:.1%} "
                        f"R2={fold['geometry_r2']:.3f}")

    if loao_results:
        verdict = compute_verdict(loao_results)
        print_flush(f"\n*** VERDICT: {verdict['verdict']} ***")
        existing["reanalysis_loao"] = loao_results
        existing["reanalysis_verdict"] = verdict
        existing["reanalysis_timestamp"] = now_utc()
        save_incremental(OUT_PATH, existing)
        print_flush(f"Saved re-analysis to {OUT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="g182 Blinded Training Triage Arena")
    parser.add_argument("--smoke", action="store_true", help="Smoke test (20 steps per cell)")
    parser.add_argument("--stage1-only", action="store_true", help="Run 48-cell stage only")
    parser.add_argument("--max-cells", type=int, default=0,
                        help="Stop after N new cells (0=unlimited). For 4h session: ~11 cells.")
    parser.add_argument("--shesha-augment", action="store_true",
                        help="Post-hoc: replay cells to step 108, add Shesha features, re-run analysis")
    parser.add_argument("--reanalyze", action="store_true",
                        help="Re-run 5-model LOAO analysis on saved cells (no training)")
    args = parser.parse_args()

    if args.reanalyze:
        reanalyze_main()
        return
    if args.shesha_augment:
        shesha_augment_main()
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print_flush(f"g182 Triage Arena started at {now_utc()}")
    print_flush(f"  smoke={args.smoke}  stage1_only={args.stage1_only}")
    print_flush(f"  device={DEVICE}  dtype={FORWARD_DTYPE}")

    existing = load_existing(OUT_PATH)
    completed_cells = {}
    if existing and "cells" in existing:
        for c in existing["cells"]:
            completed_cells[c["cell_id"]] = c
        print_flush(f"  Resuming: {len(completed_cells)} cells already done")

    seeds = SEEDS[:2] if args.smoke else SEEDS
    stage1_arms = ["scratch_ce", "seq_kd_full"]
    stage2_arms = ["embed_anchor"]
    archs = list(ARCH_CONFIGS.keys())

    print_flush(f"\n--- Loading Qwen3 donor for anchor arm ---")
    donor = load_qwen3_donor()
    donor_embed_params = snapshot_donor_embed_lm_head(donor)
    del donor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print_flush(f"\n--- Building shared vocab map (GPT-2 <-> Qwen3) ---")
    qwen_tok = get_tokenizer("qwen3")
    gpt2_tok = get_tokenizer("gpt2")
    shared_vocab_map = build_shared_vocab_map(qwen_tok, gpt2_tok)
    print_flush(f"    Shared tokens: {len(shared_vocab_map)} "
                f"({100*len(shared_vocab_map)/gpt2_tok.vocab_size:.1f}% of GPT-2 vocab)")

    print_flush(f"\n--- Generating teacher texts for seq_kd_full arm ---")
    teacher_cache_path = CACHE_DIR / "teacher_texts.json"
    if teacher_cache_path.exists():
        with open(teacher_cache_path, encoding="utf-8") as f:
            teacher_texts = json.load(f)
        print_flush(f"    Loaded {len(teacher_texts)} cached teacher texts")
    else:
        n_teacher = 96 if args.smoke else N_TRAIN_WINDOWS + 512
        teacher_texts = generate_teacher_texts(n_teacher)
        teacher_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(teacher_cache_path, "w", encoding="utf-8") as f:
            json.dump(teacher_texts, f)
        print_flush(f"    Generated and cached {len(teacher_texts)} teacher texts")

    print_flush(f"\n--- Loading Qwen3 reference geometry for Model A features ---")
    qwen_ref_probe = _load_qwen3_reference_geometry(qwen_tok)
    print_flush(f"    Reference geometry loaded (hidden shape: "
                f"{qwen_ref_probe.get('reference_hidden', np.array([])).shape})")

    all_cells = []
    new_cells_trained = 0
    max_cells = args.max_cells if args.max_cells > 0 else float("inf")
    t_start = time.time()

    arch_pools = {}
    arch_teacher_pools = {}
    for arch in archs:
        tok = get_tokenizer(arch)
        print_flush(f"\n=== Loading data for {arch} (vocab={tok.vocab_size}) ===")
        n_train = 96 if args.smoke else N_TRAIN_WINDOWS
        n_val = 64 if args.smoke else N_C4_VAL_WINDOWS
        pools = load_c4_pools(tok, n_train, n_val, SEQ_LEN)
        arch_pools[arch] = pools

        teacher_enc = []
        for text in teacher_texts:
            enc = tok(text, truncation=True, max_length=SEQ_LEN,
                      padding="max_length", return_tensors="pt")
            teacher_enc.append(enc)
        teacher_ids = torch.cat([e["input_ids"] for e in teacher_enc], dim=0)[:n_train]
        teacher_mask = torch.cat([e["attention_mask"] for e in teacher_enc], dim=0)[:n_train]
        arch_teacher_pools[arch] = {"train_ids": teacher_ids, "train_mask": teacher_mask}

    hit_limit = False
    for arch in archs:
        if hit_limit:
            break
        print_flush(f"\n=== Stage 1: {arch} (scratch_ce + seq_kd_full) ===")
        pools = arch_pools[arch]
        teacher_pools = arch_teacher_pools[arch]

        for arm in stage1_arms:
            if hit_limit:
                break
            for seed in seeds:
                cell_id = f"{arch}_{arm}_s{seed}"
                if cell_id in completed_cells:
                    all_cells.append(completed_cells[cell_id])
                    print_flush(f"  SKIP {cell_id} (already done)")
                    continue

                if new_cells_trained >= max_cells:
                    print_flush(f"  MAX CELLS ({int(max_cells)}) reached, pausing.")
                    hit_limit = True
                    break
                result = train_cell(
                    arch=arch, arm=arm, seed=seed,
                    pools=pools,
                    teacher_pools=teacher_pools if arm == "seq_kd_full" else None,
                    qwen_ref_probe=qwen_ref_probe,
                    smoke=args.smoke,
                )
                all_cells.append(result)
                new_cells_trained += 1
                save_incremental(OUT_PATH, {
                    "genome": "182",
                    "timestamp_utc": now_utc(),
                    "cells": all_cells,
                    "status": "running",
                })

    if not args.stage1_only and not hit_limit:
        for arch in archs:
            if hit_limit:
                break
            print_flush(f"\n=== Stage 2: {arch} (embed_anchor) ===")
            pools = arch_pools[arch]

            for seed in seeds:
                cell_id = f"{arch}_embed_anchor_s{seed}"
                if cell_id in completed_cells:
                    all_cells.append(completed_cells[cell_id])
                    print_flush(f"  SKIP {cell_id} (already done)")
                    continue

                if new_cells_trained >= max_cells:
                    print_flush(f"  MAX CELLS ({int(max_cells)}) reached, pausing.")
                    hit_limit = True
                    break
                result = train_cell(
                    arch=arch, arm="embed_anchor", seed=seed,
                    pools=pools,
                    donor_embed_params=donor_embed_params,
                    shared_vocab_map=shared_vocab_map if arch == "gpt2" else None,
                    qwen_ref_probe=qwen_ref_probe,
                    smoke=args.smoke,
                )
                all_cells.append(result)
                new_cells_trained += 1
                save_incremental(OUT_PATH, {
                    "genome": "182",
                    "timestamp_utc": now_utc(),
                    "cells": all_cells,
                    "status": "running",
                })

    if hit_limit:
        print_flush(f"\n=== {len(all_cells)} cells ({new_cells_trained} new). "
                    f"PARTIAL run (--max-cells={int(max_cells)}). Saving and exiting. ===")
        save_incremental(OUT_PATH, {
            "genome": "182",
            "timestamp_utc": now_utc(),
            "cells": all_cells,
            "n_cells": len(all_cells),
            "new_cells_trained": new_cells_trained,
            "status": "partial",
        })
        print_flush(f"Saved partial: {OUT_PATH}")
        return

    print_flush(f"\n=== All {len(all_cells)} cells ({new_cells_trained} new). "
                f"Running analysis... ===")

    labeled = compute_normalized_labels(all_cells)
    print_flush(f"    {len(labeled)} labeled cells (scratch excluded, used as denominator)")

    loao_results = {}
    for model_label, feat_names in [
        ("model_a_full_geometry", AGNOSTIC_FEATURE_NAMES + QWEN_REF_FEATURE_NAMES),
        ("model_b_reference_free", AGNOSTIC_FEATURE_NAMES),
        ("model_c_pure_geometry", PURE_GEOMETRY_FEATURE_NAMES),
        ("model_c_prime_manifold_only", MANIFOLD_ONLY_FEATURE_NAMES),
        ("model_d_pure_telemetry", PURE_TELEMETRY_FEATURE_NAMES),
        ("model_e_shesha", SHESHA_FEATURE_NAMES),
    ]:
        print_flush(f"\n--- LOAO evaluation: {model_label} ({len(feat_names)} features) ---")
        result = loao_evaluate(labeled, feat_names, model_label)
        loao_results[model_label] = result
        for arch, fold in result["folds"].items():
            print_flush(f"    fold={arch}: geo_mse={fold['geometry_mse']:.6f} "
                        f"best_base={fold['best_baseline_name']}={fold['best_baseline_mse']:.6f} "
                        f"reduction={fold['mse_reduction_vs_best']:.1%} "
                        f"R2={fold['geometry_r2']:.3f}")

    verdict = compute_verdict(loao_results)
    print_flush(f"\n*** VERDICT: {verdict['verdict']} ***")

    total_time = time.time() - t_start
    final = {
        "genome": "182",
        "timestamp_utc": now_utc(),
        "prereg": "research/prereg/genome_182_triage_arena_2026-04-29.md",
        "cells": all_cells,
        "n_cells": len(all_cells),
        "labeled_cells": len(labeled),
        "loao_results": loao_results,
        "verdict": verdict,
        "total_wallclock_s": total_time,
        "status": "complete",
    }
    save_incremental(OUT_PATH, final)
    print_flush(f"\nSaved: {OUT_PATH}")
    print_flush(f"Total wallclock: {total_time/3600:.1f}h")


if __name__ == "__main__":
    main()
