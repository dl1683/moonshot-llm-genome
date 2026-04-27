"""genome_168_rebasin_zero_step_transplant.py

Zero-step donor-block transplant with per-layer permutation re-basin and
RMSNorm scale refit, per
research/prereg/genome_168_rebasin_zero_step_transplant_2026-04-27.md.

Protocol summary
----------------
1. Load pretrained Qwen3-0.6B donor (frozen, BF16, GPU).
2. For each recipient seed in [42, 7, 13]:
   a. Load a random-init Qwen3-0.6B-architecture recipient.
   b. Draw a calibration C4 slice disjoint from eval by seed.
   c. Capture per-layer block input, post-attention residual, and block output
      activations on the calibration slice for donor and recipient.
   d. Fit one shared within-head permutation per layer with Hungarian on the
      donor/recipient block-output correlation matrix.
   e. Fit scalar RMSNorm refits for input_layernorm and post_attention_layernorm
      from recipient-vs-donor activation magnitudes.
   f. Build transplant arms by copying donor transformer blocks into the
      recipient with no alignment, permutation only, norm-refit only, or both.
   g. Evaluate each arm at step 0 / 10 / 50 on C4-val and WikiText-val.
3. Save JSON results and a compressed NPZ of alignment transforms.

Mathematics
-----------
Let P_l map recipient coordinates into donor coordinates at layer l:

    x_donor ~= P_l x_recipient

For a donor block F_l, the donor block expressed in the recipient gauge is:

    F_l^recipient(x) = P_l^{-1} F_l^donor(P_l x)

For linear maps this becomes the usual conjugation:

    W_in_recipient  = W_in_donor  P_l
    W_out_recipient = P_l^{-1} W_out_donor

RMSNorm is permutation-equivariant, unlike a general orthogonal rotation:

    P^{-1} RMSNorm(P x; gamma) = RMSNorm(x; P^{-1} gamma)

So a pure permutation can be absorbed exactly into the RMSNorm gamma vector.
The additional norm-refit arm multiplies the donor gamma by a scalar activity
ratio fit on the calibration set:

    gamma_refit = gamma_transplanted * (||h_recipient||_rms / ||h_donor||_rms)

Qwen3-0.6B uses GQA with:
  hidden_size = 1024
  head_dim = 128
  num_attention_heads = 16
  num_key_value_heads = 8

The residual hidden size therefore factors as 8 groups of 128. To satisfy the
locked "no cross-head permutation" constraint, this implementation fits one
128-way permutation per layer and repeats it across:
  - the 8 hidden residual groups (1024 = 8 x 128)
  - the 16 query heads
  - the 8 key/value heads

This keeps q/k/v/o internally consistent while never mixing channels across
head boundaries.
"""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.optimize import linear_sum_assignment
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_JSON = RESULTS_DIR / "genome_168_rebasin_zero_step_transplant.json"
ALIGNMENT_NPZ = RESULTS_DIR / "genome_168_rebasin_zero_step_transplant_alignments.npz"

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

SEEDS = [42, 7, 13]
SEQ_LEN = 256
CALIB_BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
TRAIN_STEPS = 50
EVAL_STEPS = [0, 10, 50]

CALIB_WINDOWS = 48          # 48 * 256 = 12,288 calibration tokens.
TRAIN_WINDOWS = 512         # 512 windows supports 50 x 8 draws with replacement.
N_C4_EVAL_WINDOWS = 200
# Reduced from 200 -> 150: wikitext-103 validation only yields ~186 windows of
# length 256 at strict-no-truncation per the sampler, so 200 fails. 150 is well
# under the available pool. Codex-written; user-fix 2026-04-27.
N_WIKI_EVAL_WINDOWS = 150
N_BOOT = 10000
LR = 3e-4
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1

C4_CALIB_SEED_BASE = 16800
C4_TRAIN_SEED_BASE = 16850
C4_EVAL_SEED = 168900
WIKI_EVAL_SEED = 168901

PASS_ZERO_STEP_GAIN_NATS = 0.8
PASS_ZERO_STEP_MIN_5X_FLOOR_NATS = 0.5
PASS_STEP50_GAIN_NATS = 0.4
FAIL_ZERO_STEP_GAIN_NATS = 0.3
FAIL_STEP50_GAIN_NATS = 0.1

ALIGNMENT_ARMS = ["permutation_only", "norm_refit_only", "permutation_plus_norm_refit"]
ALL_ARMS = [
    "random_init_only",
    "identity",
    "permutation_only",
    "norm_refit_only",
    "permutation_plus_norm_refit",
    "raw_copy",
    "donor_full",
]

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


@dataclass
class LayerAlignment:
    layer_idx: int
    head_perm: np.ndarray
    head_perm_inv: np.ndarray
    hidden_perm: np.ndarray
    hidden_perm_inv: np.ndarray
    input_scale: float
    post_scale: float
    mean_output_corr: float
    min_output_corr: float
    max_output_corr: float


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_local_config() -> Any:
    cfg = AutoConfig.from_pretrained(
        MODEL_ID,
        local_files_only=True,
        trust_remote_code=False,
    )
    # Force eager attention for stable hook behavior across environments.
    if hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = "eager"
    return cfg


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def load_trained_donor():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=DTYPE,
    ).to(DEVICE).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_random_init(seed: int):
    cfg = _load_local_config()
    set_seed(seed)
    model = AutoModelForCausalLM.from_config(cfg)
    model = model.to(DEVICE).to(DTYPE)
    return model


def get_model_meta(model) -> dict[str, int]:
    cfg = model.config
    hidden_size = int(cfg.hidden_size)
    head_dim = int(getattr(cfg, "head_dim", hidden_size // cfg.num_attention_heads))
    n_q_heads = int(cfg.num_attention_heads)
    n_kv_heads = int(cfg.num_key_value_heads)
    hidden_groups = hidden_size // head_dim
    if hidden_groups * head_dim != hidden_size:
        raise ValueError(
            f"hidden_size={hidden_size} is not divisible by head_dim={head_dim}"
        )
    return {
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "n_layers": int(cfg.num_hidden_layers),
        "n_q_heads": n_q_heads,
        "n_kv_heads": n_kv_heads,
        "hidden_groups": hidden_groups,
        "intermediate_size": int(cfg.intermediate_size),
    }


def _stream_dataset_texts(dataset_name: str, config_name: str, split: str, seed: int):
    ds = load_dataset(
        dataset_name,
        config_name,
        split=split,
        streaming=True,
        trust_remote_code=False,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    for record in ds:
        text = record.get("text", "")
        if isinstance(text, str) and text.strip():
            yield text


def sample_token_windows(
    tok,
    *,
    dataset_name: str,
    config_name: str,
    split: str,
    seed: int,
    n_windows: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample exact-length token windows from a streamed text dataset.

    Each accepted example contributes one contiguous window of length seq_len.
    We keep one window per source text to reduce within-document dependence.
    """
    rng = np.random.default_rng(seed)
    windows: list[np.ndarray] = []
    for text in _stream_dataset_texts(dataset_name, config_name, split, seed):
        token_ids = tok(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        if len(token_ids) < seq_len:
            continue
        if len(token_ids) == seq_len:
            start = 0
        else:
            start = int(rng.integers(0, len(token_ids) - seq_len + 1))
        window = np.asarray(token_ids[start:start + seq_len], dtype=np.int64)
        if window.shape[0] != seq_len:
            continue
        windows.append(window)
        if len(windows) >= n_windows:
            break
    if len(windows) < n_windows:
        raise RuntimeError(
            f"only sampled {len(windows)} / {n_windows} windows from "
            f"{dataset_name}:{config_name}:{split} seed={seed}"
        )
    input_ids = torch.tensor(np.stack(windows, axis=0), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return input_ids, attention_mask


def load_c4_windows(tok, *, split: str, seed: int, n_windows: int):
    return sample_token_windows(
        tok,
        dataset_name="allenai/c4",
        config_name="en",
        split=split,
        seed=seed,
        n_windows=n_windows,
        seq_len=SEQ_LEN,
    )


def load_wikitext_windows(tok, *, split: str, seed: int, n_windows: int):
    return sample_token_windows(
        tok,
        dataset_name="Salesforce/wikitext",
        config_name="wikitext-103-raw-v1",
        split=split,
        seed=seed,
        n_windows=n_windows,
        seq_len=SEQ_LEN,
    )


class CalibrationStats:
    def __init__(self, meta: dict[str, int]):
        n_layers = meta["n_layers"]
        hidden_size = meta["hidden_size"]
        head_dim = meta["head_dim"]
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_count = [0 for _ in range(n_layers)]
        self.output_sum_d = [np.zeros(head_dim, dtype=np.float64) for _ in range(n_layers)]
        self.output_sum_r = [np.zeros(head_dim, dtype=np.float64) for _ in range(n_layers)]
        self.output_sumsq_d = [np.zeros(head_dim, dtype=np.float64) for _ in range(n_layers)]
        self.output_sumsq_r = [np.zeros(head_dim, dtype=np.float64) for _ in range(n_layers)]
        self.output_cross = [np.zeros((head_dim, head_dim), dtype=np.float64) for _ in range(n_layers)]
        self.input_sum_sq_d = [np.zeros(hidden_size, dtype=np.float64) for _ in range(n_layers)]
        self.input_sum_sq_r = [np.zeros(hidden_size, dtype=np.float64) for _ in range(n_layers)]
        self.post_sum_sq_d = [np.zeros(hidden_size, dtype=np.float64) for _ in range(n_layers)]
        self.post_sum_sq_r = [np.zeros(hidden_size, dtype=np.float64) for _ in range(n_layers)]
        self.input_count = [0 for _ in range(n_layers)]
        self.post_count = [0 for _ in range(n_layers)]

    def update_layer(
        self,
        layer_idx: int,
        donor_input: np.ndarray,
        recip_input: np.ndarray,
        donor_post: np.ndarray,
        recip_post: np.ndarray,
        donor_output: np.ndarray,
        recip_output: np.ndarray,
    ) -> None:
        # Scalar norm refit uses full hidden-state RMS.
        self.input_sum_sq_d[layer_idx] += np.square(donor_input, dtype=np.float64).sum(axis=0)
        self.input_sum_sq_r[layer_idx] += np.square(recip_input, dtype=np.float64).sum(axis=0)
        self.post_sum_sq_d[layer_idx] += np.square(donor_post, dtype=np.float64).sum(axis=0)
        self.post_sum_sq_r[layer_idx] += np.square(recip_post, dtype=np.float64).sum(axis=0)
        self.input_count[layer_idx] += donor_input.shape[0]
        self.post_count[layer_idx] += donor_post.shape[0]

        # Fit one shared 128-way permutation by pooling over the 8 hidden groups.
        donor_grouped = donor_output.reshape(-1, self.hidden_size // self.head_dim, self.head_dim)
        recip_grouped = recip_output.reshape(-1, self.hidden_size // self.head_dim, self.head_dim)
        donor_collapsed = donor_grouped.reshape(-1, self.head_dim)
        recip_collapsed = recip_grouped.reshape(-1, self.head_dim)
        self.output_sum_d[layer_idx] += donor_collapsed.sum(axis=0)
        self.output_sum_r[layer_idx] += recip_collapsed.sum(axis=0)
        self.output_sumsq_d[layer_idx] += np.square(donor_collapsed, dtype=np.float64).sum(axis=0)
        self.output_sumsq_r[layer_idx] += np.square(recip_collapsed, dtype=np.float64).sum(axis=0)
        self.output_cross[layer_idx] += donor_collapsed.T @ recip_collapsed
        self.output_count[layer_idx] += donor_collapsed.shape[0]


def _register_post_attention_hooks(model):
    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        def _hook(module, inputs, idx=layer_idx):
            captured[idx] = inputs[0].detach().float().cpu()
        hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(_hook))
    return captured, hooks


@torch.no_grad()
def collect_batch_activations(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    post_inputs, hooks = _register_post_attention_hooks(model)
    try:
        out = model(
            input_ids=input_ids.to(DEVICE),
            attention_mask=attention_mask.to(DEVICE),
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = [h.detach().float().cpu() for h in out.hidden_states]
    finally:
        for hook in hooks:
            hook.remove()
    return hidden_states, post_inputs


def collect_alignment_stats(
    donor,
    recipient,
    calib_ids: torch.Tensor,
    calib_mask: torch.Tensor,
    meta: dict[str, int],
) -> CalibrationStats:
    stats = CalibrationStats(meta)
    n_layers = meta["n_layers"]
    for start in range(0, calib_ids.size(0), CALIB_BATCH_SIZE):
        end = start + CALIB_BATCH_SIZE
        ids = calib_ids[start:end]
        mask = calib_mask[start:end]
        donor_hidden, donor_post = collect_batch_activations(donor, ids, mask)
        recip_hidden, recip_post = collect_batch_activations(recipient, ids, mask)

        for layer_idx in range(n_layers):
            donor_input = donor_hidden[layer_idx].reshape(-1, meta["hidden_size"]).numpy()
            recip_input = recip_hidden[layer_idx].reshape(-1, meta["hidden_size"]).numpy()
            donor_post_resid = donor_post[layer_idx].reshape(-1, meta["hidden_size"]).numpy()
            recip_post_resid = recip_post[layer_idx].reshape(-1, meta["hidden_size"]).numpy()
            donor_output = donor_hidden[layer_idx + 1].reshape(-1, meta["hidden_size"]).numpy()
            recip_output = recip_hidden[layer_idx + 1].reshape(-1, meta["hidden_size"]).numpy()
            stats.update_layer(
                layer_idx,
                donor_input,
                recip_input,
                donor_post_resid,
                recip_post_resid,
                donor_output,
                recip_output,
            )
        del donor_hidden, donor_post, recip_hidden, recip_post
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return stats


def invert_perm(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm), dtype=perm.dtype)
    return inv


def build_hidden_perm(head_perm: np.ndarray, hidden_groups: int) -> np.ndarray:
    offsets = np.arange(hidden_groups, dtype=np.int64)[:, None] * head_perm.shape[0]
    return (offsets + head_perm[None, :]).reshape(-1)


def fit_alignments(stats: CalibrationStats, meta: dict[str, int]) -> list[LayerAlignment]:
    alignments: list[LayerAlignment] = []
    eps = 1e-8
    if meta["head_dim"] % 2 != 0:
        raise ValueError(f"head_dim={meta['head_dim']} must be even for RoPE-pair-safe permutations")
    for layer_idx in range(meta["n_layers"]):
        n = stats.output_count[layer_idx]
        if n <= 0:
            raise RuntimeError(f"layer {layer_idx}: no output observations collected")
        mean_d = stats.output_sum_d[layer_idx] / n
        mean_r = stats.output_sum_r[layer_idx] / n
        var_d = stats.output_sumsq_d[layer_idx] / n - np.square(mean_d)
        var_r = stats.output_sumsq_r[layer_idx] / n - np.square(mean_r)
        cov = stats.output_cross[layer_idx] / n - np.outer(mean_d, mean_r)
        denom = np.sqrt(np.maximum(var_d, eps))[:, None] * np.sqrt(np.maximum(var_r, eps))[None, :]
        corr = cov / np.maximum(denom, eps)

        # RoPE couples adjacent hidden coordinates into 2-d rotation pairs.
        # We therefore fit the Hungarian permutation over 64 pair slots and
        # expand each matched pair back to consecutive coordinates.
        n_pairs = meta["head_dim"] // 2
        pair_score = np.zeros((n_pairs, n_pairs), dtype=np.float64)
        for donor_pair in range(n_pairs):
            d0 = 2 * donor_pair
            d1 = d0 + 1
            for recip_pair in range(n_pairs):
                r0 = 2 * recip_pair
                r1 = r0 + 1
                pair_score[donor_pair, recip_pair] = corr[d0, r0] + corr[d1, r1]
        row_ind, col_ind = linear_sum_assignment(-pair_score)
        if not np.array_equal(row_ind, np.arange(n_pairs)):
            raise RuntimeError(f"layer {layer_idx}: Hungarian pair rows are not contiguous")
        head_perm_list: list[int] = []
        for recip_pair in col_ind.tolist():
            head_perm_list.extend([2 * recip_pair, 2 * recip_pair + 1])
        head_perm = np.asarray(head_perm_list, dtype=np.int64)
        head_perm_inv = invert_perm(head_perm)
        hidden_perm = build_hidden_perm(head_perm, meta["hidden_groups"])
        hidden_perm_inv = invert_perm(hidden_perm)

        matched_corr = corr[np.arange(meta["head_dim"]), head_perm]
        input_count = max(stats.input_count[layer_idx], 1)
        post_count = max(stats.post_count[layer_idx], 1)
        donor_input_rms = float(np.sqrt(stats.input_sum_sq_d[layer_idx].sum() / (input_count * meta["hidden_size"])))
        recip_input_rms = float(np.sqrt(stats.input_sum_sq_r[layer_idx].sum() / (input_count * meta["hidden_size"])))
        donor_post_rms = float(np.sqrt(stats.post_sum_sq_d[layer_idx].sum() / (post_count * meta["hidden_size"])))
        recip_post_rms = float(np.sqrt(stats.post_sum_sq_r[layer_idx].sum() / (post_count * meta["hidden_size"])))

        input_scale = recip_input_rms / max(donor_input_rms, eps)
        post_scale = recip_post_rms / max(donor_post_rms, eps)

        alignments.append(
            LayerAlignment(
                layer_idx=layer_idx,
                head_perm=head_perm,
                head_perm_inv=head_perm_inv,
                hidden_perm=hidden_perm,
                hidden_perm_inv=hidden_perm_inv,
                input_scale=input_scale,
                post_scale=post_scale,
                mean_output_corr=float(matched_corr.mean()),
                min_output_corr=float(matched_corr.min()),
                max_output_corr=float(matched_corr.max()),
            )
        )
    return alignments


def _index_select_cpu_safe(tensor: torch.Tensor, dim: int, indices: np.ndarray) -> torch.Tensor:
    idx = torch.as_tensor(indices, device=tensor.device, dtype=torch.long)
    return tensor.index_select(dim, idx)


def permute_rows_grouped(weight: torch.Tensor, num_groups: int, group_size: int, perm: np.ndarray) -> torch.Tensor:
    shape = weight.shape
    view = weight.reshape(num_groups, group_size, shape[1])
    view = _index_select_cpu_safe(view, 1, perm)
    return view.reshape(shape)


def permute_cols_grouped(weight: torch.Tensor, num_groups: int, group_size: int, perm: np.ndarray) -> torch.Tensor:
    shape = weight.shape
    view = weight.reshape(shape[0], num_groups, group_size)
    view = _index_select_cpu_safe(view, 2, perm)
    return view.reshape(shape)


def transplant_one_layer(
    donor_layer,
    recip_layer,
    alignment: LayerAlignment,
    meta: dict[str, int],
    *,
    use_perm: bool,
    use_norm_refit: bool,
) -> None:
    hidden_perm = alignment.hidden_perm if use_perm else None
    hidden_perm_inv = alignment.hidden_perm_inv if use_perm else None
    head_perm = alignment.head_perm if use_perm else None
    head_perm_inv = alignment.head_perm_inv if use_perm else None

    q_heads = meta["n_q_heads"]
    kv_heads = meta["n_kv_heads"]
    head_dim = meta["head_dim"]

    with torch.no_grad():
        # RMSNorm gamma vectors.
        input_gamma = donor_layer.input_layernorm.weight.detach().clone()
        post_gamma = donor_layer.post_attention_layernorm.weight.detach().clone()
        if use_perm:
            input_gamma = _index_select_cpu_safe(input_gamma, 0, hidden_perm_inv)
            post_gamma = _index_select_cpu_safe(post_gamma, 0, hidden_perm_inv)
        if use_norm_refit:
            input_gamma = input_gamma * input_gamma.new_tensor(alignment.input_scale)
            post_gamma = post_gamma * post_gamma.new_tensor(alignment.post_scale)
        recip_layer.input_layernorm.weight.copy_(input_gamma.to(recip_layer.input_layernorm.weight.dtype))
        recip_layer.post_attention_layernorm.weight.copy_(post_gamma.to(recip_layer.post_attention_layernorm.weight.dtype))

        # Shared q/k norm gamma over the 128-d head basis.
        q_norm = donor_layer.self_attn.q_norm.weight.detach().clone()
        k_norm = donor_layer.self_attn.k_norm.weight.detach().clone()
        if use_perm:
            q_norm = _index_select_cpu_safe(q_norm, 0, head_perm_inv)
            k_norm = _index_select_cpu_safe(k_norm, 0, head_perm_inv)
        recip_layer.self_attn.q_norm.weight.copy_(q_norm.to(recip_layer.self_attn.q_norm.weight.dtype))
        recip_layer.self_attn.k_norm.weight.copy_(k_norm.to(recip_layer.self_attn.k_norm.weight.dtype))

        # q_proj: output rows are per-query-head internal channels; input cols are hidden channels.
        q_proj = donor_layer.self_attn.q_proj.weight.detach().clone()
        if use_perm:
            q_proj = _index_select_cpu_safe(q_proj, 1, hidden_perm)
            q_proj = permute_rows_grouped(q_proj, q_heads, head_dim, head_perm_inv)
        recip_layer.self_attn.q_proj.weight.copy_(q_proj.to(recip_layer.self_attn.q_proj.weight.dtype))

        # k_proj / v_proj: same internal 128-d permutation, but 8 KV heads.
        k_proj = donor_layer.self_attn.k_proj.weight.detach().clone()
        v_proj = donor_layer.self_attn.v_proj.weight.detach().clone()
        if use_perm:
            k_proj = _index_select_cpu_safe(k_proj, 1, hidden_perm)
            v_proj = _index_select_cpu_safe(v_proj, 1, hidden_perm)
            k_proj = permute_rows_grouped(k_proj, kv_heads, head_dim, head_perm_inv)
            v_proj = permute_rows_grouped(v_proj, kv_heads, head_dim, head_perm_inv)
        recip_layer.self_attn.k_proj.weight.copy_(k_proj.to(recip_layer.self_attn.k_proj.weight.dtype))
        recip_layer.self_attn.v_proj.weight.copy_(v_proj.to(recip_layer.self_attn.v_proj.weight.dtype))

        # o_proj: input cols are per-query-head internal channels; output rows are hidden channels.
        o_proj = donor_layer.self_attn.o_proj.weight.detach().clone()
        if use_perm:
            o_proj = permute_cols_grouped(o_proj, q_heads, head_dim, head_perm)
            o_proj = _index_select_cpu_safe(o_proj, 0, hidden_perm_inv)
        recip_layer.self_attn.o_proj.weight.copy_(o_proj.to(recip_layer.self_attn.o_proj.weight.dtype))

        # MLP input/output hidden basis changes. Intermediate channels stay in donor order.
        gate_proj = donor_layer.mlp.gate_proj.weight.detach().clone()
        up_proj = donor_layer.mlp.up_proj.weight.detach().clone()
        down_proj = donor_layer.mlp.down_proj.weight.detach().clone()
        if use_perm:
            gate_proj = _index_select_cpu_safe(gate_proj, 1, hidden_perm)
            up_proj = _index_select_cpu_safe(up_proj, 1, hidden_perm)
            down_proj = _index_select_cpu_safe(down_proj, 0, hidden_perm_inv)
        recip_layer.mlp.gate_proj.weight.copy_(gate_proj.to(recip_layer.mlp.gate_proj.weight.dtype))
        recip_layer.mlp.up_proj.weight.copy_(up_proj.to(recip_layer.mlp.up_proj.weight.dtype))
        recip_layer.mlp.down_proj.weight.copy_(down_proj.to(recip_layer.mlp.down_proj.weight.dtype))


def transplant_transformer_blocks(
    donor,
    recipient,
    alignments: list[LayerAlignment],
    meta: dict[str, int],
    *,
    use_perm: bool,
    use_norm_refit: bool,
) -> None:
    for alignment in alignments:
        donor_layer = donor.model.layers[alignment.layer_idx]
        recip_layer = recipient.model.layers[alignment.layer_idx]
        transplant_one_layer(
            donor_layer,
            recip_layer,
            alignment,
            meta,
            use_perm=use_perm,
            use_norm_refit=use_norm_refit,
        )


@torch.no_grad()
def eval_perplexity_top1(model, eval_ids: torch.Tensor, eval_mask: torch.Tensor) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_top1 = 0
    for start in range(0, eval_ids.size(0), EVAL_BATCH_SIZE):
        end = start + EVAL_BATCH_SIZE
        ids = eval_ids[start:end].to(DEVICE)
        mask = eval_mask[start:end].to(DEVICE)
        out = model(input_ids=ids, attention_mask=mask, use_cache=False)
        logits = out.logits[:, :-1].contiguous()
        labels = ids[:, 1:].contiguous()
        valid = mask[:, 1:].contiguous().bool()
        labels_for_loss = labels.masked_fill(~valid, -100)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels_for_loss.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss += float(loss.item())
        total_tokens += int(valid.sum().item())
        preds = logits.argmax(dim=-1)
        correct_top1 += int(((preds == labels) & valid).sum().item())
    model.train()
    return {
        "nll": total_loss / max(total_tokens, 1),
        "top1_acc": correct_top1 / max(total_tokens, 1),
        "n_tokens": float(total_tokens),
    }


def evaluate_arm(model, c4_eval, wiki_eval) -> dict[str, dict[str, float]]:
    c4_ids, c4_mask = c4_eval
    wiki_ids, wiki_mask = wiki_eval
    return {
        "c4": eval_perplexity_top1(model, c4_ids, c4_mask),
        "wikitext": eval_perplexity_top1(model, wiki_ids, wiki_mask),
    }


def make_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )


def train_steps_in_place(
    model,
    optimizer,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    rng: np.random.Generator,
    n_steps: int,
) -> None:
    model.train()
    for _ in range(n_steps):
        idx = rng.integers(0, train_ids.size(0), size=TRAIN_BATCH_SIZE)
        ids = train_ids[idx].to(DEVICE)
        mask = train_mask[idx].to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=ids, attention_mask=mask, use_cache=False)
        logits = out.logits[:, :-1].contiguous()
        labels = ids[:, 1:].contiguous()
        valid = mask[:, 1:].contiguous().bool()
        labels = labels.masked_fill(~valid, -100)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def run_recipient_trajectory(
    arm_name: str,
    *,
    donor,
    alignments: list[LayerAlignment],
    meta: dict[str, int],
    seed: int,
    train_data,
    c4_eval,
    wiki_eval,
) -> dict[str, Any]:
    model = load_random_init(seed)
    if arm_name == "raw_copy" or arm_name == "identity":
        transplant_transformer_blocks(
            donor, model, alignments, meta, use_perm=False, use_norm_refit=False
        )
    elif arm_name == "permutation_only":
        transplant_transformer_blocks(
            donor, model, alignments, meta, use_perm=True, use_norm_refit=False
        )
    elif arm_name == "norm_refit_only":
        transplant_transformer_blocks(
            donor, model, alignments, meta, use_perm=False, use_norm_refit=True
        )
    elif arm_name == "permutation_plus_norm_refit":
        transplant_transformer_blocks(
            donor, model, alignments, meta, use_perm=True, use_norm_refit=True
        )
    elif arm_name == "random_init_only":
        pass
    else:
        raise ValueError(f"unknown recipient arm: {arm_name}")

    steps: dict[str, Any] = {}
    steps["0"] = evaluate_arm(model, c4_eval, wiki_eval)
    if arm_name != "donor_full":
        train_ids, train_mask = train_data
        optimizer = make_optimizer(model)
        rng = np.random.default_rng(seed + 10)
        train_steps_in_place(model, optimizer, train_ids, train_mask, rng=rng, n_steps=10)
        steps["10"] = evaluate_arm(model, c4_eval, wiki_eval)
        train_steps_in_place(model, optimizer, train_ids, train_mask, rng=rng, n_steps=40)
        steps["50"] = evaluate_arm(model, c4_eval, wiki_eval)
        del optimizer
    n_params = int(sum(p.numel() for p in model.parameters()))
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"n_params": n_params, "steps": steps}


def bootstrap_mean_ci(values: list[float], *, n_boot: int = N_BOOT, seed: int = 0) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    n = len(arr)
    for i in range(n_boot):
        boots[i] = arr[rng.integers(0, n, size=n)].mean()
    return {
        "mean": float(arr.mean()),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
    }


def summarize_alignment(alignments: list[LayerAlignment]) -> dict[str, Any]:
    return {
        f"layer_{align.layer_idx:02d}": {
            "input_scale": align.input_scale,
            "post_scale": align.post_scale,
            "mean_output_corr": align.mean_output_corr,
            "min_output_corr": align.min_output_corr,
            "max_output_corr": align.max_output_corr,
        }
        for align in alignments
    }


def save_alignments_npz(per_seed_alignments: dict[int, list[LayerAlignment]]) -> str:
    arrays: dict[str, np.ndarray] = {}
    for seed, alignments in per_seed_alignments.items():
        for align in alignments:
            prefix = f"seed{seed}_layer{align.layer_idx:02d}"
            arrays[f"{prefix}_head_perm"] = align.head_perm.astype(np.int64)
            arrays[f"{prefix}_head_perm_inv"] = align.head_perm_inv.astype(np.int64)
            arrays[f"{prefix}_hidden_perm"] = align.hidden_perm.astype(np.int64)
            arrays[f"{prefix}_hidden_perm_inv"] = align.hidden_perm_inv.astype(np.int64)
            arrays[f"{prefix}_input_scale"] = np.asarray([align.input_scale], dtype=np.float32)
            arrays[f"{prefix}_post_scale"] = np.asarray([align.post_scale], dtype=np.float32)
    np.savez_compressed(ALIGNMENT_NPZ, **arrays)
    return str(ALIGNMENT_NPZ)


def build_analysis(results_by_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    per_arm = {}
    for arm in ["identity", "raw_copy", *ALIGNMENT_ARMS]:
        zero_deltas = []
        step50_deltas = []
        for seed in SEEDS:
            random0 = results_by_seed[seed]["arms"]["random_init_only"]["steps"]["0"]["c4"]["nll"]
            random50 = results_by_seed[seed]["arms"]["random_init_only"]["steps"]["50"]["c4"]["nll"]
            arm0 = results_by_seed[seed]["arms"][arm]["steps"]["0"]["c4"]["nll"]
            arm50 = results_by_seed[seed]["arms"][arm]["steps"]["50"]["c4"]["nll"]
            zero_deltas.append(random0 - arm0)
            step50_deltas.append(random50 - arm50)
        zero_stats = bootstrap_mean_ci(zero_deltas, seed=20260427 + len(arm))
        step50_stats = bootstrap_mean_ci(step50_deltas, seed=20260527 + len(arm))
        per_arm[arm] = {
            "zero_step_gain_vs_random_init_nats": zero_stats,
            "step50_gain_vs_random_init_nats": step50_stats,
            "seed_deltas": {
                "step0": [float(x) for x in zero_deltas],
                "step50": [float(x) for x in step50_deltas],
            },
            "passes_prereg": (
                zero_stats["mean"] >= PASS_ZERO_STEP_GAIN_NATS
                and zero_stats["ci_lo"] > 0.0
                and zero_stats["mean"] >= PASS_ZERO_STEP_MIN_5X_FLOOR_NATS
                and step50_stats["mean"] >= PASS_STEP50_GAIN_NATS
            ),
        }

    best_arm = max(
        ALIGNMENT_ARMS,
        key=lambda arm: per_arm[arm]["zero_step_gain_vs_random_init_nats"]["mean"],
    )
    best = per_arm[best_arm]
    if best["passes_prereg"]:
        verdict = (
            f"PASS: {best_arm} zero-step C4 gain = "
            f"{best['zero_step_gain_vs_random_init_nats']['mean']:+.3f} nats "
            f"[{best['zero_step_gain_vs_random_init_nats']['ci_lo']:+.3f}, "
            f"{best['zero_step_gain_vs_random_init_nats']['ci_hi']:+.3f}], "
            f"step50 gain = {best['step50_gain_vs_random_init_nats']['mean']:+.3f} nats."
        )
        outcome = "PASS"
    else:
        decisive_fail = (
            best["zero_step_gain_vs_random_init_nats"]["mean"] < FAIL_ZERO_STEP_GAIN_NATS
            or best["zero_step_gain_vs_random_init_nats"]["ci_lo"] <= 0.0
            or best["step50_gain_vs_random_init_nats"]["mean"] < FAIL_STEP50_GAIN_NATS
        )
        outcome = "FAIL"
        verdict = (
            f"FAIL: best aligned arm {best_arm} zero-step C4 gain = "
            f"{best['zero_step_gain_vs_random_init_nats']['mean']:+.3f} nats "
            f"[{best['zero_step_gain_vs_random_init_nats']['ci_lo']:+.3f}, "
            f"{best['zero_step_gain_vs_random_init_nats']['ci_hi']:+.3f}], "
            f"step50 gain = {best['step50_gain_vs_random_init_nats']['mean']:+.3f} nats. "
            f"{'Decisive prereg fail thresholds fire.' if decisive_fail else 'Misses prereg PASS thresholds.'}"
        )
    return {
        "per_arm_c4_gain_vs_random_init": per_arm,
        "best_alignment_arm": best_arm,
        "outcome": outcome,
        "verdict": verdict,
        "criteria": {
            "pass_zero_step_gain_nats": PASS_ZERO_STEP_GAIN_NATS,
            "pass_zero_step_5x_floor_nats": PASS_ZERO_STEP_MIN_5X_FLOOR_NATS,
            "pass_step50_gain_nats": PASS_STEP50_GAIN_NATS,
            "fail_zero_step_gain_nats": FAIL_ZERO_STEP_GAIN_NATS,
            "fail_step50_gain_nats": FAIL_STEP50_GAIN_NATS,
        },
    }


def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"genome_168: re-basin zero-step transplant on {MODEL_ID}")
    print(f"  device={DEVICE} dtype={DTYPE} seeds={SEEDS}")

    tok = load_tokenizer()
    donor = load_trained_donor()
    meta = get_model_meta(donor)
    print(
        "  hidden={hidden_size} head_dim={head_dim} q_heads={n_q_heads} "
        "kv_heads={n_kv_heads} layers={n_layers}".format(**meta)
    )

    print("Loading held-out eval windows...")
    c4_eval = load_c4_windows(tok, split="validation", seed=C4_EVAL_SEED, n_windows=N_C4_EVAL_WINDOWS)
    wiki_eval = load_wikitext_windows(tok, split="validation", seed=WIKI_EVAL_SEED, n_windows=N_WIKI_EVAL_WINDOWS)
    print(f"  C4 eval: {tuple(c4_eval[0].shape)}")
    print(f"  Wiki eval: {tuple(wiki_eval[0].shape)}")

    print("Evaluating donor_full once...")
    donor_metrics = evaluate_arm(donor, c4_eval, wiki_eval)
    donor_arm = {
        "n_params": int(sum(p.numel() for p in donor.parameters())),
        "steps": {
            "0": donor_metrics,
            "10": donor_metrics,
            "50": donor_metrics,
        },
        "note": "donor_full is frozen upper bound; metrics repeat across step labels.",
    }

    results_by_seed: dict[int, dict[str, Any]] = {}
    per_seed_alignments: dict[int, list[LayerAlignment]] = {}

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        calib_seed = C4_CALIB_SEED_BASE + seed
        train_seed = C4_TRAIN_SEED_BASE + seed
        print(f"  calibration seed={calib_seed} train seed={train_seed}")

        calib_ids, calib_mask = load_c4_windows(
            tok,
            split="train",
            seed=calib_seed,
            n_windows=CALIB_WINDOWS,
        )
        train_data = load_c4_windows(
            tok,
            split="train",
            seed=train_seed,
            n_windows=TRAIN_WINDOWS,
        )
        print(f"  calibration windows: {tuple(calib_ids.shape)}")
        print(f"  train windows: {tuple(train_data[0].shape)}")

        recipient_for_fit = load_random_init(seed).eval()
        print("  collecting donor/recipient calibration activations...")
        stats = collect_alignment_stats(donor, recipient_for_fit, calib_ids, calib_mask, meta)
        alignments = fit_alignments(stats, meta)
        per_seed_alignments[seed] = alignments
        del recipient_for_fit
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"  mean matched corr across layers: "
            f"{np.mean([a.mean_output_corr for a in alignments]):.4f}"
        )

        seed_results = {
            "alignment_summary": summarize_alignment(alignments),
            "arms": {},
        }

        # Unique recipient trajectories.
        for arm_name in [
            "random_init_only",
            "raw_copy",
            "permutation_only",
            "norm_refit_only",
            "permutation_plus_norm_refit",
        ]:
            print(f"  running arm={arm_name}")
            arm_result = run_recipient_trajectory(
                arm_name,
                donor=donor,
                alignments=alignments,
                meta=meta,
                seed=seed,
                train_data=train_data,
                c4_eval=c4_eval,
                wiki_eval=wiki_eval,
            )
            step0_c4 = arm_result["steps"]["0"]["c4"]
            step50_c4 = arm_result["steps"]["50"]["c4"]
            print(
                f"    step0 C4 NLL={step0_c4['nll']:.4f} top1={100*step0_c4['top1_acc']:.2f}% | "
                f"step50 C4 NLL={step50_c4['nll']:.4f} top1={100*step50_c4['top1_acc']:.2f}%"
            )
            seed_results["arms"][arm_name] = arm_result

        # identity is intentionally the same raw donor block copy as raw_copy.
        seed_results["arms"]["identity"] = {
            **seed_results["arms"]["raw_copy"],
            "shared_with": "raw_copy",
            "note": "identity is intentionally identical to raw_copy; separate label kept per prereg.",
        }
        seed_results["arms"]["donor_full"] = donor_arm
        results_by_seed[seed] = seed_results

    alignment_npz = save_alignments_npz(per_seed_alignments)
    analysis = build_analysis(results_by_seed)

    out = {
        "genome": 168,
        "name": "rebasin_zero_step_transplant",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "config": {
            "seeds": SEEDS,
            "seq_len": SEQ_LEN,
            "calibration_batch_size": CALIB_BATCH_SIZE,
            "calibration_windows": CALIB_WINDOWS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "train_windows": TRAIN_WINDOWS,
            "train_steps": TRAIN_STEPS,
            "eval_steps": EVAL_STEPS,
            "n_c4_eval_windows": N_C4_EVAL_WINDOWS,
            "n_wikitext_eval_windows": N_WIKI_EVAL_WINDOWS,
            "lr": LR,
            "betas": list(BETAS),
            "weight_decay": WEIGHT_DECAY,
            "c4_calibration_seed_base": C4_CALIB_SEED_BASE,
            "c4_train_seed_base": C4_TRAIN_SEED_BASE,
            "c4_eval_seed": C4_EVAL_SEED,
            "wikitext_eval_seed": WIKI_EVAL_SEED,
            "head_respecting_permutation": {
                "head_dim": meta["head_dim"],
                "hidden_groups": meta["hidden_groups"],
                "query_heads": meta["n_q_heads"],
                "key_value_heads": meta["n_kv_heads"],
                "fit": "single RoPE-pair-safe within-head permutation per layer, repeated across residual groups and attention heads",
            },
            "transplant_scope": "all transformer blocks only; recipient embeddings/lm_head/final norm stay at random init",
        },
        "alignment_npz": alignment_npz,
        "results_by_seed": results_by_seed,
        "analysis": analysis,
        "verdict": analysis["verdict"],
        "elapsed_s": time.time() - t0,
    }
    RESULTS_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved JSON: {RESULTS_JSON}")
    print(f"Saved NPZ:  {ALIGNMENT_NPZ}")
    print(f"Verdict:    {analysis['verdict']}")


if __name__ == "__main__":
    main()