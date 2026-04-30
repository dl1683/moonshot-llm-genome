"""
genome_197_output_interface_canary_arena.py

Prereg: research/prereg/genome_197_output_interface_canary_arena_2026-04-30.md

Tests whether step-0/step-50 lm_head geometry predicts final 5000-step NLL
across 10 deliberately varied output-interface initializations, beating a
scalar step-50 early-loss baseline.

10 conditions x 3 seeds = 30 cells. Untied model, lm_head init only, no anchors.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats as sp_stats

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import genome_165_annealed_donor as g165
import genome_167_kd_canonical as g167
import genome_188_tokenizer_flow_bridge as g188
import genome_191_string_match_decomposition as g191

OUT_PATH = ROOT / "results" / "genome_197_output_interface_canary_arena.json"

SEEDS = [42, 7, 13]
TRAIN_STEPS = 5000
FEATURE_STEP = 50
LOG_EVERY = 100
EVAL_EVERY = 500
DEVICE = g165.DEVICE
SCAFFOLD_SEED_ORTHO = 19701
SCAFFOLD_SEED_COV = 19702
SCAFFOLD_SEED_ETF = 19703
FEATURE_ROW_SAMPLE = 8192

CONDITIONS = [
    "trained_qwen3",
    "frequency_scaled",
    "orthogonal_scaffold",
    "covariance_scaffold",
    "identity_axis",
    "neural_collapse_etf",
    "random_gaussian",
    "trained_random_directions",
    "trained_shuffled",
    "anti_frequency_scaled",
]

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


def print_flush(msg: str) -> None:
    print(msg, flush=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------- lm_head condition constructors ----------

def _row_norms(m: np.ndarray) -> np.ndarray:
    return np.linalg.norm(m, axis=1, keepdims=True).clip(min=1e-12)


def _unit_directions(m: np.ndarray) -> np.ndarray:
    return m / _row_norms(m)


def _rescale_to_fro(m: np.ndarray, target_fro: float) -> np.ndarray:
    fro = float(np.linalg.norm(m, "fro"))
    if fro < 1e-12:
        return m
    return m * (target_fro / fro)


def _fill_unmatched_gaussian(out: np.ndarray, matched_mask: np.ndarray,
                             matched_norm_mean: float, rng_seed: int) -> None:
    rng = np.random.RandomState(rng_seed)
    unmatched_ids = np.where(~matched_mask[:out.shape[0]])[0]
    for i in unmatched_ids:
        v = rng.randn(out.shape[1]).astype(np.float32)
        out[i] = v / max(np.linalg.norm(v), 1e-12) * matched_norm_mean


def build_trained_qwen3(
    trained_lm_head: np.ndarray, matched_mask: np.ndarray,
    scratch_fro: float, gpt2_vocab: int, embed_dim: int,
) -> np.ndarray:
    out = np.zeros((gpt2_vocab, embed_dim), dtype=np.float32)
    out[matched_mask] = trained_lm_head[matched_mask]
    matched_norm_mean = float(_row_norms(trained_lm_head[matched_mask]).mean())
    _fill_unmatched_gaussian(out, matched_mask, matched_norm_mean, 44444)
    return _rescale_to_fro(out, scratch_fro)


def build_frequency_scaled(
    trained_lm_head: np.ndarray, matched_mask: np.ndarray,
    token_freqs: np.ndarray, scratch_fro: float, gpt2_vocab: int, embed_dim: int,
) -> np.ndarray:
    rng = np.random.RandomState(55555)
    dirs = _unit_directions(trained_lm_head)
    out = np.zeros((gpt2_vocab, embed_dim), dtype=np.float32)
    freq_norms = np.log1p(token_freqs[:gpt2_vocab]).astype(np.float32)
    freq_norms = freq_norms / freq_norms.mean().clip(min=1e-12) * float(_row_norms(trained_lm_head[matched_mask]).mean())
    for i in range(gpt2_vocab):
        if matched_mask[i]:
            out[i] = dirs[i] * freq_norms[i]
        else:
            rand_dir = rng.randn(embed_dim).astype(np.float32)
            out[i] = rand_dir / max(np.linalg.norm(rand_dir), 1e-12) * freq_norms[i]
    return _rescale_to_fro(out, scratch_fro)


def build_orthogonal_scaffold(
    trained_lm_head: np.ndarray, matched_mask: np.ndarray,
    scratch_fro: float, gpt2_vocab: int, embed_dim: int,
) -> np.ndarray:
    rng = np.random.RandomState(SCAFFOLD_SEED_ORTHO)
    Z = rng.randn(embed_dim, embed_dim).astype(np.float64)
    Q, _ = np.linalg.qr(Z)
    Q = Q.astype(np.float32)
    out = np.zeros((gpt2_vocab, embed_dim), dtype=np.float32)
    out[matched_mask] = trained_lm_head[matched_mask] @ Q
    matched_norm_mean = float(_row_norms(out[matched_mask]).mean())
    _fill_unmatched_gaussian(out, matched_mask, matched_norm_mean, 44445)
    return _rescale_to_fro(out, scratch_fro)


def build_covariance_scaffold(
    trained_lm_head: np.ndarray, matched_mask: np.ndarray,
    scratch_fro: float, gpt2_vocab: int, embed_dim: int,
) -> np.ndarray:
    rng = np.random.RandomState(SCAFFOLD_SEED_COV)
    matched_rows = trained_lm_head[matched_mask]
    mu = matched_rows.mean(axis=0).astype(np.float64)
    cov = np.cov(matched_rows.astype(np.float64), rowvar=False)
    cov += 1e-6 * np.eye(embed_dim, dtype=np.float64)
    out = rng.multivariate_normal(mu, cov, size=gpt2_vocab).astype(np.float32)
    matched_norm_mean = float(_row_norms(matched_rows).mean())
    norms = _row_norms(out)
    out = out / norms * matched_norm_mean
    return _rescale_to_fro(out, scratch_fro)


def build_identity_axis(
    scratch_fro: float, gpt2_vocab: int, embed_dim: int,
    token_freqs: np.ndarray,
) -> np.ndarray:
    freq_order = np.argsort(-token_freqs[:gpt2_vocab])
    out = np.zeros((gpt2_vocab, embed_dim), dtype=np.float32)
    signs = np.array([1.0, -1.0], dtype=np.float32)
    for rank, tid in enumerate(freq_order):
        axis = rank % embed_dim
        sign = signs[(rank // embed_dim) % 2]
        out[tid, axis] = sign
    return _rescale_to_fro(out, scratch_fro)


def build_neural_collapse_etf(
    scratch_fro: float, gpt2_vocab: int, embed_dim: int,
) -> np.ndarray:
    rng = np.random.RandomState(SCAFFOLD_SEED_ETF)
    Z = rng.randn(gpt2_vocab, embed_dim).astype(np.float64)
    U, _, Vt = np.linalg.svd(Z, full_matrices=False)
    k = min(gpt2_vocab, embed_dim)
    etf = U[:, :k] @ Vt[:k, :]
    etf = etf.astype(np.float32)
    norms = _row_norms(etf)
    etf = etf / norms
    jitter = rng.randn(gpt2_vocab, embed_dim).astype(np.float32) * 0.01
    etf = etf + jitter
    return _rescale_to_fro(etf, scratch_fro)


def build_random_gaussian(
    scratch_fro: float, gpt2_vocab: int, embed_dim: int, seed: int,
) -> np.ndarray:
    rng = np.random.RandomState(seed + 99999)
    out = rng.randn(gpt2_vocab, embed_dim).astype(np.float32)
    return _rescale_to_fro(out, scratch_fro)


def build_trained_random_directions(
    trained_lm_head: np.ndarray, matched_mask: np.ndarray,
    scratch_fro: float, gpt2_vocab: int, embed_dim: int, seed: int,
) -> np.ndarray:
    rng = np.random.RandomState(seed + 88888)
    norms = _row_norms(trained_lm_head)
    dirs = rng.randn(gpt2_vocab, embed_dim).astype(np.float32)
    dirs = dirs / _row_norms(dirs)
    out = dirs * norms[:gpt2_vocab]
    return _rescale_to_fro(out, scratch_fro)


def build_trained_shuffled(
    trained_lm_head: np.ndarray, matched_mask: np.ndarray,
    scratch_fro: float, gpt2_vocab: int, embed_dim: int, seed: int,
) -> np.ndarray:
    rng = np.random.RandomState(seed + 77777)
    out = np.zeros((gpt2_vocab, embed_dim), dtype=np.float32)
    matched_ids = np.where(matched_mask)[0]
    matched_rows = trained_lm_head[matched_ids].copy()
    perm = rng.permutation(len(matched_ids))
    out[matched_ids] = matched_rows[perm]
    matched_norm_mean = float(_row_norms(matched_rows).mean())
    _fill_unmatched_gaussian(out, matched_mask, matched_norm_mean, seed + 44446)
    return _rescale_to_fro(out, scratch_fro)


def build_anti_frequency_scaled(
    scratch_fro: float, gpt2_vocab: int, embed_dim: int,
    token_freqs: np.ndarray, seed: int,
) -> np.ndarray:
    rng = np.random.RandomState(seed + 66666)
    inv_freq = 1.0 / np.log1p(token_freqs[:gpt2_vocab] + 1).astype(np.float32)
    inv_freq = inv_freq / inv_freq.mean().clip(min=1e-12)
    dirs = rng.randn(gpt2_vocab, embed_dim).astype(np.float32)
    dirs = dirs / _row_norms(dirs)
    out = dirs * inv_freq[:, None]
    return _rescale_to_fro(out, scratch_fro)


# ---------- Geometry feature extraction ----------

def extract_geometry_features(
    W: np.ndarray,
    row_sample_idx: np.ndarray,
    token_freqs: np.ndarray,
    trained_ref: np.ndarray | None = None,
    scaffold_refs: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    W_s = W[row_sample_idx].astype(np.float64)
    feats: dict[str, float] = {}

    # Spectral
    X = W_s - W_s.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    s = s[np.isfinite(s) & (s > 1e-12)]
    if s.size >= 3:
        lam = s**2 / max(X.shape[0] - 1, 1)
        pr = float(np.sum(lam)**2 / np.sum(lam**2)) if np.sum(lam**2) > 0 else float("nan")
        feats["stable_rank"] = float(np.sum(s**2) / (s[0]**2)) if s[0] > 0 else float("nan")
        feats["participation_ratio"] = pr
        lam_norm = lam / lam.sum()
        feats["effective_rank_entropy"] = float(-np.sum(lam_norm * np.log(lam_norm + 1e-30)))
        feats["top_sv_mass"] = float(s[0]**2 / np.sum(s**2))
        feats["condition_number"] = float(s[0] / s[-1]) if s[-1] > 1e-12 else float("nan")

        tail_start = min(5, max(0, s.size // 5))
        tail_stop = min(s.size, max(tail_start + 3, 64))
        ranks = np.arange(tail_start + 1, tail_stop + 1, dtype=np.float64)
        y = np.log(s[tail_start:tail_stop])
        if y.size >= 3 and np.isfinite(y).all():
            slope = float(np.polyfit(np.log(ranks), y, deg=1)[0])
            feats["tail_alpha"] = -slope
        else:
            feats["tail_alpha"] = float("nan")
        alpha = feats["tail_alpha"]
        feats["sqrt_pr_alpha"] = float(math.sqrt(max(pr, 0)) * alpha) if math.isfinite(pr) and math.isfinite(alpha) else float("nan")
    else:
        for k in ["tail_alpha", "stable_rank", "participation_ratio", "effective_rank_entropy", "top_sv_mass", "condition_number", "sqrt_pr_alpha"]:
            feats[k] = float("nan")

    # Row norms
    norms = np.linalg.norm(W_s, axis=1)
    feats["norm_mean"] = float(norms.mean())
    feats["norm_std"] = float(norms.std())
    feats["norm_cv"] = float(norms.std() / norms.mean()) if norms.mean() > 1e-12 else float("nan")
    feats["norm_skew"] = float(sp_stats.skew(norms))
    norm_p = norms / norms.sum().clip(min=1e-30)
    feats["norm_entropy"] = float(-np.sum(norm_p * np.log(norm_p + 1e-30)))
    sorted_norms = np.sort(norms)
    n = len(sorted_norms)
    idx = np.arange(1, n + 1)
    feats["norm_gini"] = float((2 * np.sum(idx * sorted_norms) / (n * np.sum(sorted_norms))) - (n + 1) / n) if np.sum(sorted_norms) > 0 else float("nan")

    # Frequency-norm correlation
    sample_freqs = token_freqs[row_sample_idx]
    if np.std(sample_freqs) > 0 and np.std(norms) > 0:
        feats["spearman_norm_freq"] = float(sp_stats.spearmanr(norms, sample_freqs).statistic)
    else:
        feats["spearman_norm_freq"] = float("nan")

    freq_sorted = np.argsort(-sample_freqs)
    top_q = max(1, n // 5)
    feats["freq_rare_norm_ratio"] = float(norms[freq_sorted[-top_q:]].mean() / norms[freq_sorted[:top_q]].mean()) if norms[freq_sorted[:top_q]].mean() > 1e-12 else float("nan")

    # Angular features (deterministic random subsample for speed)
    ang_rng = np.random.RandomState(12345)
    ang_n = min(2048, W_s.shape[0])
    ang_idx = ang_rng.choice(W_s.shape[0], size=ang_n, replace=False) if W_s.shape[0] > ang_n else np.arange(W_s.shape[0])
    W_ang = W_s[ang_idx]
    W_unit = W_ang / np.linalg.norm(W_ang, axis=1, keepdims=True).clip(min=1e-12)
    gram = W_unit @ W_unit.T
    np.fill_diagonal(gram, 0.0)
    upper = gram[np.triu_indices(ang_n, k=1)]
    feats["mean_pairwise_cosine"] = float(upper.mean())
    feats["std_pairwise_cosine"] = float(upper.std())
    feats["max_coherence"] = float(np.abs(upper).max())
    feats["gram_offdiag_fro"] = float(np.sqrt(np.sum(upper**2)))
    feats["angular_spread"] = float(np.arccos(np.clip(upper.mean(), -1, 1)))

    # kNN features (deterministic random subsample)
    from sklearn.neighbors import NearestNeighbors
    knn_rng = np.random.RandomState(54321)
    knn_n = min(4096, W_s.shape[0])
    knn_idx = knn_rng.choice(W_s.shape[0], size=knn_n, replace=False) if W_s.shape[0] > knn_n else np.arange(W_s.shape[0])
    W_knn = W_s[knn_idx]
    k = 10
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=1).fit(W_knn)
    dists, idxs = nn.kneighbors(W_knn)
    nn_cos = []
    for i in range(knn_n):
        for j in idxs[i, 1:]:
            c = float(np.dot(W_knn[i], W_knn[j]) / (np.linalg.norm(W_knn[i]) * np.linalg.norm(W_knn[j]) + 1e-12))
            nn_cos.append(c)
    nn_cos_arr = np.array(nn_cos)
    feats["nn_cosine_mean"] = float(nn_cos_arr.mean())
    feats["nn_cosine_p95"] = float(np.percentile(nn_cos_arr, 95))

    # Mutual nearest neighbor fraction
    neigh_sets = [set(idxs[i, 1:].tolist()) for i in range(knn_n)]
    mutual = sum(1 for i in range(knn_n) for j in idxs[i, 1:] if i in neigh_sets[j])
    feats["mutual_nn_frac"] = float(mutual / (knn_n * k))

    # kNN clustering coefficient (fraction of neighbor pairs that are mutual neighbors)
    tri_count = 0
    possible = 0
    for i in range(knn_n):
        nbrs = neigh_sets[i]
        for a in nbrs:
            for b in nbrs:
                if a < b:
                    possible += 1
                    if b in neigh_sets[a]:
                        tri_count += 1
    feats["knn_clustering_coeff"] = float(tri_count / possible) if possible > 0 else float("nan")

    # Frequency-bucket neighbor purity
    knn_sample_freqs = token_freqs[row_sample_idx][:knn_n] if len(row_sample_idx) >= knn_n else token_freqs[row_sample_idx]
    freq_ranks = np.argsort(-knn_sample_freqs)
    bucket_size = max(1, knn_n // 5)
    bucket_labels = np.zeros(knn_n, dtype=np.int32)
    for b in range(5):
        start = b * bucket_size
        end = min((b + 1) * bucket_size, knn_n)
        bucket_labels[freq_ranks[start:end]] = b
    purity_sum = 0
    for i in range(knn_n):
        same_bucket = sum(1 for j in idxs[i, 1:] if j < knn_n and bucket_labels[j] == bucket_labels[i])
        purity_sum += same_bucket / k
    feats["freq_bucket_purity"] = float(purity_sum / knn_n)

    # Reference distances
    if trained_ref is not None:
        ref_s = trained_ref[row_sample_idx].astype(np.float64)
        ref_n = min(2048, W_s.shape[0], ref_s.shape[0])
        A = W_s[:ref_n] - W_s[:ref_n].mean(axis=0, keepdims=True)
        B = ref_s[:ref_n] - ref_s[:ref_n].mean(axis=0, keepdims=True)
        na, nb = np.linalg.norm(A), np.linalg.norm(B)
        if na > 1e-12 and nb > 1e-12:
            An, Bn = A / na, B / nb
            U, _, Vt = np.linalg.svd(An.T @ Bn, full_matrices=False)
            feats["procrustes_residual"] = float(np.linalg.norm(An @ (U @ Vt) - Bn))
        else:
            feats["procrustes_residual"] = float("nan")

        sa = np.linalg.svd(A, compute_uv=False)
        sb = np.linalg.svd(B, compute_uv=False)
        k_spec = min(len(sa), len(sb), 64)
        feats["spectral_wasserstein"] = float(np.sum(np.abs(sa[:k_spec] - sb[:k_spec])))

        from scipy.spatial.distance import pdist
        da = pdist(A[:min(512, ref_n)])
        db = pdist(B[:min(512, ref_n)])
        if np.std(da) > 1e-12 and np.std(db) > 1e-12:
            feats["rsa_distance"] = float(1.0 - np.corrcoef(da, db)[0, 1])
        else:
            feats["rsa_distance"] = float("nan")
        # Row-norm KL divergence to trained reference
        ref_norms = np.linalg.norm(ref_s, axis=1)
        n_bins = 50
        all_norms = np.concatenate([norms, ref_norms])
        bins = np.linspace(all_norms.min() - 1e-6, all_norms.max() + 1e-6, n_bins + 1)
        p_hist = np.histogram(norms, bins=bins)[0].astype(np.float64) + 1
        q_hist = np.histogram(ref_norms, bins=bins)[0].astype(np.float64) + 1
        p_hist /= p_hist.sum()
        q_hist /= q_hist.sum()
        feats["norm_kl_to_trained"] = float(np.sum(p_hist * np.log(p_hist / q_hist)))
    else:
        feats["procrustes_residual"] = float("nan")
        feats["spectral_wasserstein"] = float("nan")
        feats["rsa_distance"] = float("nan")
        feats["norm_kl_to_trained"] = float("nan")

    # Scaffold distances
    if scaffold_refs is not None:
        for name, ref_arr in scaffold_refs.items():
            ref_s = ref_arr[row_sample_idx].astype(np.float64)
            feats[f"scaffold_fro_{name}"] = float(np.linalg.norm(W_s - ref_s, "fro"))
    else:
        for name in ["etf", "identity", "covariance"]:
            feats[f"scaffold_fro_{name}"] = float("nan")

    return feats


# ---------- Model construction ----------

def make_untied_model(tok_gpt2, seed: int):
    from transformers import Qwen3ForCausalLM
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=len(tok_gpt2),
        hidden_size=1024,
        num_hidden_layers=8,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=2816,
        max_position_embeddings=g188.SEQ_LEN + 64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        head_dim=64,
        rope_theta=10000.0,
        use_cache=False,
    )
    model = Qwen3ForCausalLM(cfg).to(device=DEVICE)
    model.config.pad_token_id = tok_gpt2.pad_token_id
    return model


# ---------- Training cell ----------

def train_cell(
    condition: str,
    seed: int,
    tok_gpt2,
    lm_head_init: np.ndarray,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
    token_freqs: np.ndarray,
    row_sample_idx: np.ndarray,
    trained_ref: np.ndarray | None,
    scaffold_refs: dict[str, np.ndarray] | None = None,
    *,
    n_steps: int = TRAIN_STEPS,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = make_untied_model(tok_gpt2, seed)

    # Record scratch lm_head for dynamics features
    scratch_head = model.lm_head.weight.detach().cpu().numpy().copy()

    # Inject lm_head init
    head_t = torch.from_numpy(lm_head_init).to(model.lm_head.weight.device, dtype=model.lm_head.weight.dtype)
    with torch.no_grad():
        model.lm_head.weight.copy_(head_t)
    del head_t

    # Step-0 geometry features
    step0_W = model.lm_head.weight.detach().cpu().numpy().copy()
    step0_feats = extract_geometry_features(step0_W, row_sample_idx, token_freqs, trained_ref, scaffold_refs)
    step0_feats = {f"s0_{k}": v for k, v in step0_feats.items()}

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=g188.LR, betas=g188.BETAS, weight_decay=g188.WEIGHT_DECAY,
    )

    n_train = train_ids.shape[0]
    trajectory = {}
    step50_feats: dict[str, float] = {}
    delta_feats: dict[str, float] = {}
    dynamics: dict[str, float] = {}
    early_losses: dict[int, float] = {}
    EARLY_EVAL_STEPS = {10, 25}
    t0 = time.time()

    for step in range(1, n_steps + 1):
        idx = torch.randint(0, n_train, (g188.BATCH_SIZE,))
        batch_ids = train_ids[idx].to(DEVICE)
        batch_mask = train_mask[idx].to(DEVICE)

        with torch.amp.autocast("cuda", dtype=g188.FORWARD_DTYPE):
            out = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_ids)
            loss = out.loss

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step} cond={condition} seed={seed}")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), g188.GRAD_CLIP)
        optimizer.step()

        if step % LOG_EVERY == 0:
            print_flush(f"    step {step}/{n_steps} loss={loss.item():.4f}")

        if step in EARLY_EVAL_STEPS:
            model.eval()
            with torch.no_grad():
                nll_early = g188._eval_nll(model, val_ids, val_mask)
            early_losses[step] = float(nll_early)
            trajectory[str(step)] = float(nll_early)
            model.train()

        if step == FEATURE_STEP:
            # Step-50 geometry features
            model.eval()
            with torch.no_grad():
                val_nll_50 = g188._eval_nll(model, val_ids, val_mask)
            trajectory[str(step)] = float(val_nll_50)
            print_flush(f"    eval step={step} val_nll={val_nll_50:.4f}")

            step50_W = model.lm_head.weight.detach().cpu().numpy().copy()
            step50_feats = extract_geometry_features(step50_W, row_sample_idx, token_freqs, trained_ref, scaffold_refs)
            step50_feats = {f"s50_{k}": v for k, v in step50_feats.items()}

            # Dynamics features
            update_vec = step50_W - step0_W
            dynamics = {
                "update_fro": float(np.linalg.norm(update_vec, "fro")),
                "update_cosine": float(
                    np.sum(step50_W * step0_W) /
                    (np.linalg.norm(step50_W, "fro") * np.linalg.norm(step0_W, "fro") + 1e-12)
                ),
            }

            # Delta features
            delta_feats = {}
            for k in step0_feats:
                s0_key = k
                s50_key = k.replace("s0_", "s50_")
                if s50_key in step50_feats:
                    s0_v = step0_feats[s0_key]
                    s50_v = step50_feats[s50_key]
                    if math.isfinite(s0_v) and math.isfinite(s50_v):
                        delta_feats[f"delta_{k[3:]}"] = s50_v - s0_v
                    else:
                        delta_feats[f"delta_{k[3:]}"] = float("nan")
            del step50_W
            model.train()

        if (step % EVAL_EVERY == 0 and step != FEATURE_STEP) or step == n_steps:
            model.eval()
            with torch.no_grad():
                val_nll = g188._eval_nll(model, val_ids, val_mask)
            trajectory[str(step)] = float(val_nll)
            if step % EVAL_EVERY == 0:
                print_flush(f"    eval step={step} val_nll={val_nll:.4f}")
            model.train()

    model.eval()
    with torch.no_grad():
        final_nll = g188._eval_nll(model, val_ids, val_mask)

    all_features = {}
    all_features.update(step0_feats)
    all_features.update(step50_feats)
    all_features.update(delta_feats)
    all_features.update(dynamics)
    all_features["early_loss_50"] = float(trajectory.get(str(FEATURE_STEP), float("nan")))

    # Loss slope from steps 10/25/50 (diagnostic, not locked comparator)
    early_losses[FEATURE_STEP] = float(trajectory.get(str(FEATURE_STEP), float("nan")))
    slope_steps = sorted(early_losses.keys())
    if len(slope_steps) >= 2 and all(math.isfinite(early_losses[s]) for s in slope_steps):
        xs = np.array(slope_steps, dtype=np.float64)
        ys = np.array([early_losses[s] for s in slope_steps], dtype=np.float64)
        all_features["early_loss_slope"] = float(np.polyfit(xs, ys, deg=1)[0])
    else:
        all_features["early_loss_slope"] = float("nan")

    result = {
        "condition": condition,
        "seed": seed,
        "final_val_nll": float(final_nll),
        "trajectory": trajectory,
        "geometry_features": all_features,
        "wallclock_s": time.time() - t0,
    }
    del model, optimizer, scratch_head
    cleanup_cuda()
    return result


# ---------- Ridge prediction ----------

def run_prediction_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    results = payload.get("results", {})
    rows = []
    for cond in CONDITIONS:
        if cond not in results:
            continue
        for s in SEEDS:
            key = str(s)
            if key not in results[cond]:
                continue
            cell = results[cond][key]
            if "geometry_features" not in cell:
                continue
            rows.append({
                "condition": cond,
                "seed": s,
                "final_nll": cell["final_val_nll"],
                "features": cell["geometry_features"],
            })

    if len(rows) != len(CONDITIONS) * len(SEEDS):
        return {"status": "insufficient_data", "n_rows": len(rows),
                "expected": len(CONDITIONS) * len(SEEDS)}

    def _safe_finite(v):
        return v is not None and isinstance(v, (int, float)) and math.isfinite(v)

    # Build feature matrix (geometry only, no early_loss)
    all_feat_keys = sorted(set(
        k for r in rows for k in r["features"]
        if k != "early_loss_50" and _safe_finite(r["features"].get(k))
    ))
    geom_keys = [k for k in all_feat_keys if k != "early_loss_50"]

    n = len(rows)
    X_geom = np.zeros((n, len(geom_keys)), dtype=np.float64)
    X_loss = np.zeros((n, 1), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    cond_labels = []
    seed_labels = []

    for i, r in enumerate(rows):
        y[i] = r["final_nll"]
        cond_labels.append(r["condition"])
        seed_labels.append(r["seed"])
        for j, k in enumerate(geom_keys):
            v = r["features"].get(k)
            X_geom[i, j] = float(v) if _safe_finite(v) else 0.0
        v_loss = r["features"].get("early_loss_50")
        X_loss[i, 0] = float(v_loss) if _safe_finite(v_loss) else 0.0

    # NaN column filter
    valid_cols = np.all(np.isfinite(X_geom), axis=0)
    X_geom = X_geom[:, valid_cols]
    geom_keys = [k for k, v in zip(geom_keys, valid_cols) if v]

    unique_conds = sorted(set(cond_labels))
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    # Leave-one-condition-out CV
    geom_errors = []
    loss_errors = []
    geom_wins = 0

    for held_out in unique_conds:
        train_mask = np.array([c != held_out for c in cond_labels])
        test_mask = ~train_mask
        assert test_mask.sum() == len(SEEDS), f"Expected {len(SEEDS)} test rows for {held_out}, got {test_mask.sum()}"

        X_tr_g, X_te_g = X_geom[train_mask], X_geom[test_mask]
        X_tr_l, X_te_l = X_loss[train_mask], X_loss[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]

        scaler_g = StandardScaler().fit(X_tr_g)
        X_tr_gs = scaler_g.transform(X_tr_g)
        X_te_gs = scaler_g.transform(X_te_g)

        scaler_l = StandardScaler().fit(X_tr_l)
        X_tr_ls = scaler_l.transform(X_tr_l)
        X_te_ls = scaler_l.transform(X_te_l)

        ridge_g = RidgeCV(alphas=alphas).fit(X_tr_gs, y_tr)
        ridge_l = RidgeCV(alphas=alphas).fit(X_tr_ls, y_tr)

        pred_g = ridge_g.predict(X_te_gs)
        pred_l = ridge_l.predict(X_te_ls)

        mse_g = float(np.mean((y_te - pred_g) ** 2))
        mse_l = float(np.mean((y_te - pred_l) ** 2))
        geom_errors.append(mse_g)
        loss_errors.append(mse_l)
        if mse_g < mse_l:
            geom_wins += 1

    geom_mse = float(np.mean(geom_errors))
    loss_mse = float(np.mean(loss_errors))
    mse_reduction = 1.0 - geom_mse / loss_mse if loss_mse > 0 else float("nan")

    # Out-of-fold R2 (using LOCO predictions, NOT in-sample)
    oof_preds_g = np.full(n, np.nan)
    for fold_idx, held_out in enumerate(unique_conds):
        test_mask = np.array([c == held_out for c in cond_labels])
        train_mask_f = ~test_mask
        X_tr_g = X_geom[train_mask_f]
        X_te_g = X_geom[test_mask]
        y_tr_f = y[train_mask_f]
        sc = StandardScaler().fit(X_tr_g)
        ridge_oof = RidgeCV(alphas=alphas).fit(sc.transform(X_tr_g), y_tr_f)
        oof_preds_g[test_mask] = ridge_oof.predict(sc.transform(X_te_g))
    ss_res_oof = float(np.sum((y - oof_preds_g) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_oof = 1.0 - ss_res_oof / ss_tot if ss_tot > 0 else float("nan")

    # Bootstrap CI on MSE reduction
    rng = np.random.RandomState(42)
    boot_reductions = []
    n_boot = 1000
    for _ in range(n_boot):
        boot_idx = rng.choice(len(unique_conds), size=len(unique_conds), replace=True)
        boot_geom = np.mean([geom_errors[i] for i in boot_idx])
        boot_loss = np.mean([loss_errors[i] for i in boot_idx])
        if boot_loss > 0:
            boot_reductions.append(1.0 - boot_geom / boot_loss)
    ci_lower = float(np.percentile(boot_reductions, 2.5))
    ci_upper = float(np.percentile(boot_reductions, 97.5))

    # Seed-stratified permutation test
    unique_seeds = sorted(set(seed_labels))
    seed_arr = np.array(seed_labels)
    perm_reductions = []
    n_perm = 1000
    for p in range(n_perm):
        perm_rng = np.random.RandomState(p)
        perm_idx = np.arange(n)
        for sd in unique_seeds:
            mask = seed_arr == sd
            within = np.where(mask)[0]
            perm_idx[within] = within[perm_rng.permutation(len(within))]
        X_perm = X_geom[perm_idx]
        perm_errors = []
        for held_out in unique_conds:
            train_mask = np.array([c != held_out for c in cond_labels])
            test_mask = ~train_mask
            X_tr = X_perm[train_mask]
            X_te = X_perm[test_mask]
            y_tr_p, y_te_p = y[train_mask], y[test_mask]
            sc = StandardScaler().fit(X_tr)
            ridge_p = RidgeCV(alphas=alphas).fit(sc.transform(X_tr), y_tr_p)
            pred_p = ridge_p.predict(sc.transform(X_te))
            perm_errors.append(float(np.mean((y_te_p - pred_p) ** 2)))
        perm_mse = float(np.mean(perm_errors))
        if loss_mse > 0:
            perm_reductions.append(1.0 - perm_mse / loss_mse)
    exceedance = sum(1 for pr in perm_reductions if pr >= mse_reduction)
    perm_p = float((exceedance + 1) / (n_perm + 1))

    # NLL range check
    nll_range = float(y.max() - y.min())

    # Verdict
    if nll_range < 0.10:
        verdict = "FAIL_NO_SIGNAL"
    elif mse_reduction >= 0.25 and ci_lower > 0 and r2_oof >= 0.35 and perm_p <= 0.05 and geom_wins >= 8:
        verdict = "PASS_CANARY"
    elif mse_reduction >= 0.10 and ci_lower >= 0 and perm_p <= 0.10:
        verdict = "WEAK_PASS"
    elif mse_reduction < 0.10 or ci_lower < 0:
        verdict = "FAIL_LOSS_ONLY"
    else:
        verdict = "AMBIGUOUS"

    # Step-0-only ablation (addresses adversarial attack #4: step-50 dynamics leak)
    s0_keys = [k for k in geom_keys if k.startswith("s0_")]
    s0_geom_mse = float("nan")
    if len(s0_keys) >= 3:
        s0_col_idx = [geom_keys.index(k) for k in s0_keys]
        X_s0 = X_geom[:, s0_col_idx]
        s0_errors = []
        for held_out in unique_conds:
            train_mask_s = np.array([c != held_out for c in cond_labels])
            test_mask_s = ~train_mask_s
            sc_s0 = StandardScaler().fit(X_s0[train_mask_s])
            r_s0 = RidgeCV(alphas=alphas).fit(sc_s0.transform(X_s0[train_mask_s]), y[train_mask_s])
            pred_s0 = r_s0.predict(sc_s0.transform(X_s0[test_mask_s]))
            s0_errors.append(float(np.mean((y[test_mask_s] - pred_s0) ** 2)))
        s0_geom_mse = float(np.mean(s0_errors))

    # Norm-only ablation (addresses adversarial attack #1: norm/condition decoding)
    norm_keys = [k for k in geom_keys if "norm" in k.lower()]
    norm_geom_mse = float("nan")
    if len(norm_keys) >= 2:
        norm_col_idx = [geom_keys.index(k) for k in norm_keys]
        X_norm = X_geom[:, norm_col_idx]
        norm_errors = []
        for held_out in unique_conds:
            train_mask_n = np.array([c != held_out for c in cond_labels])
            test_mask_n = ~train_mask_n
            sc_n = StandardScaler().fit(X_norm[train_mask_n])
            r_n = RidgeCV(alphas=alphas).fit(sc_n.transform(X_norm[train_mask_n]), y[train_mask_n])
            pred_n = r_n.predict(sc_n.transform(X_norm[test_mask_n]))
            norm_errors.append(float(np.mean((y[test_mask_n] - pred_n) ** 2)))
        norm_geom_mse = float(np.mean(norm_errors))

    return {
        "status": "complete",
        "verdict": verdict,
        "geom_mse": geom_mse,
        "loss_mse": loss_mse,
        "mse_reduction": mse_reduction,
        "mse_reduction_ci": [ci_lower, ci_upper],
        "r2_oof": r2_oof,
        "perm_p": perm_p,
        "geom_wins": geom_wins,
        "n_conditions": len(unique_conds),
        "n_cells": n,
        "nll_range": nll_range,
        "n_geom_features": len(geom_keys),
        "geom_feature_names": geom_keys,
        "s0_only_mse": s0_geom_mse,
        "norm_only_mse": norm_geom_mse,
    }


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    smoke = args.smoke
    n_steps = 50 if smoke else TRAIN_STEPS
    seeds = [42] if smoke else SEEDS
    conditions = ["trained_qwen3", "trained_shuffled"] if smoke else CONDITIONS
    run_out_path = OUT_PATH.with_name(OUT_PATH.stem + "_smoke.json") if smoke else OUT_PATH

    print_flush(f"=== g197 Output-Interface Canary Arena ===")
    print_flush(f"  smoke={smoke}, steps={n_steps}, seeds={seeds}, conditions={len(conditions)}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok_qwen = AutoTokenizer.from_pretrained(g188.QWEN_MODEL_ID)
    tok_gpt2 = AutoTokenizer.from_pretrained(g188.GPT2_MODEL_ID)
    if tok_gpt2.pad_token is None:
        tok_gpt2.pad_token = tok_gpt2.eos_token

    gpt2_vocab = len(tok_gpt2)
    print_flush(f"  GPT-2 vocab: {gpt2_vocab}")

    print_flush("\n--- Loading data ---")
    train_ids, train_mask, _ = g167.load_c4_windows(
        tok_gpt2, split="train", seed=g188.C4_TRAIN_SEED, n_windows=g188.N_TRAIN_WINDOWS,
    )
    train_hashes = g167.collect_13gram_hashes(train_ids, train_mask)
    val_ids, val_mask, _ = g167.load_c4_windows(
        tok_gpt2, split="train", seed=g188.C4_VAL_SEED, n_windows=g188.N_C4_VAL_WINDOWS,
        forbidden_hashes=train_hashes,
    )
    print_flush(f"  Train: {train_ids.shape}, Val: {val_ids.shape}")

    # Token frequencies
    token_freqs = np.zeros(gpt2_vocab, dtype=np.float64)
    for tid in train_ids.reshape(-1).tolist():
        if tid < gpt2_vocab:
            token_freqs[tid] += 1

    print_flush("\n--- Loading Qwen3 trained lm_head ---")
    qwen_model = AutoModelForCausalLM.from_pretrained(g188.QWEN_MODEL_ID, torch_dtype=torch.float32)
    trained_lm_head_full = qwen_model.lm_head.weight.detach().cpu().numpy().copy()
    embed_dim = trained_lm_head_full.shape[1]
    del qwen_model
    cleanup_cuda()
    print_flush(f"  Trained lm_head: {trained_lm_head_full.shape}")

    print_flush("\n--- Building string-match lm_head ---")
    trained_lm_head, matched_mask = g191.build_string_match_with_mask(
        tok_qwen, tok_gpt2, trained_lm_head_full, gpt2_vocab, embed_dim,
    )
    n_matched = int(matched_mask.sum())
    print_flush(f"  Matched: {n_matched}")

    # Scratch model Frobenius norm for rescaling
    torch.manual_seed(42)
    scratch_model = make_untied_model(tok_gpt2, 42)
    scratch_fro = float(np.linalg.norm(scratch_model.lm_head.weight.detach().cpu().numpy(), "fro"))
    del scratch_model
    cleanup_cuda()
    print_flush(f"  Scratch lm_head Fro: {scratch_fro:.1f}")

    # Row sample index (fixed)
    rng_sample = np.random.RandomState(42)
    freq_top = np.argsort(-token_freqs)[:min(4096, gpt2_vocab)]
    remaining = np.setdiff1d(np.arange(gpt2_vocab), freq_top)
    rand_pick = rng_sample.choice(remaining, size=min(4096, len(remaining)), replace=False)
    row_sample_idx = np.sort(np.concatenate([freq_top, rand_pick]))
    print_flush(f"  Row sample: {len(row_sample_idx)} rows")

    # Precompute scaffold references for distance features
    scaffold_refs = {
        "etf": build_neural_collapse_etf(scratch_fro, gpt2_vocab, embed_dim),
        "identity": build_identity_axis(scratch_fro, gpt2_vocab, embed_dim, token_freqs),
        "covariance": build_covariance_scaffold(trained_lm_head, matched_mask, scratch_fro, gpt2_vocab, embed_dim),
    }

    # Build condition heads
    def get_head(cond: str, seed: int) -> np.ndarray:
        if cond == "trained_qwen3":
            return build_trained_qwen3(trained_lm_head, matched_mask, scratch_fro, gpt2_vocab, embed_dim)
        elif cond == "frequency_scaled":
            return build_frequency_scaled(trained_lm_head, matched_mask, token_freqs, scratch_fro, gpt2_vocab, embed_dim)
        elif cond == "orthogonal_scaffold":
            return build_orthogonal_scaffold(trained_lm_head, matched_mask, scratch_fro, gpt2_vocab, embed_dim)
        elif cond == "covariance_scaffold":
            return build_covariance_scaffold(trained_lm_head, matched_mask, scratch_fro, gpt2_vocab, embed_dim)
        elif cond == "identity_axis":
            return build_identity_axis(scratch_fro, gpt2_vocab, embed_dim, token_freqs)
        elif cond == "neural_collapse_etf":
            return build_neural_collapse_etf(scratch_fro, gpt2_vocab, embed_dim)
        elif cond == "random_gaussian":
            return build_random_gaussian(scratch_fro, gpt2_vocab, embed_dim, seed)
        elif cond == "trained_random_directions":
            return build_trained_random_directions(trained_lm_head, matched_mask, scratch_fro, gpt2_vocab, embed_dim, seed)
        elif cond == "trained_shuffled":
            return build_trained_shuffled(trained_lm_head, matched_mask, scratch_fro, gpt2_vocab, embed_dim, seed)
        elif cond == "anti_frequency_scaled":
            return build_anti_frequency_scaled(scratch_fro, gpt2_vocab, embed_dim, token_freqs, seed)
        else:
            raise ValueError(f"Unknown condition: {cond}")

    # Resume
    if not args.no_resume and run_out_path.exists():
        payload = json.loads(run_out_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "genome": 197,
            "name": "output_interface_canary_arena",
            "timestamp_utc_started": now_utc(),
            "config": {
                "train_steps": n_steps,
                "seeds": seeds,
                "conditions": conditions,
                "n_matched": n_matched,
                "scratch_fro": scratch_fro,
                "feature_step": FEATURE_STEP,
                "row_sample_size": len(row_sample_idx),
            },
            "results": {},
            "summary": {},
            "verdict": "INCOMPLETE",
        }

    t_start = time.time()

    def _sanitize_nans(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize_nans(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_nans(v) for v in obj]
        return obj

    def save():
        payload["timestamp_utc_last_write"] = now_utc()
        payload["elapsed_s"] = time.time() - t_start
        tmp = run_out_path.with_suffix(run_out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(_sanitize_nans(payload), indent=2, default=str, allow_nan=False), encoding="utf-8")
        os.replace(tmp, run_out_path)

    for cond in conditions:
        payload["results"].setdefault(cond, {})

        for seed in seeds:
            key = str(seed)
            if key in payload["results"][cond] and not args.no_resume:
                cell = payload["results"][cond][key]
                if isinstance(cell, dict) and "final_val_nll" in cell:
                    print_flush(f"\n  Skipping {cond}/seed={seed} (done)")
                    continue

            print_flush(f"\n  === {cond} seed={seed} ===")
            head = get_head(cond, seed)
            result = train_cell(
                condition=cond,
                seed=seed,
                tok_gpt2=tok_gpt2,
                lm_head_init=head,
                train_ids=train_ids,
                train_mask=train_mask,
                val_ids=val_ids,
                val_mask=val_mask,
                token_freqs=token_freqs,
                row_sample_idx=row_sample_idx,
                trained_ref=trained_lm_head,
                scaffold_refs=scaffold_refs,
                n_steps=n_steps,
            )
            payload["results"][cond][key] = result
            save()
            print_flush(f"  {cond} seed={seed} nll={result['final_val_nll']:.4f} ({result['wallclock_s']:.0f}s)")

    # Run prediction analysis
    print_flush("\n--- Running prediction analysis ---")
    summary = run_prediction_analysis(payload)
    payload["summary"] = summary
    payload["verdict"] = summary.get("verdict", "INCOMPLETE")
    save()

    print_flush(f"\n*** g197 VERDICT: {summary.get('verdict', '?')} ***")
    for key in ["mse_reduction", "r2_oof", "perm_p", "geom_wins", "nll_range", "s0_only_mse", "norm_only_mse"]:
        if key in summary:
            print_flush(f"  {key}: {summary[key]}")


if __name__ == "__main__":
    main()
