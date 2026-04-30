"""
genome_183_corpus_derived_init.py

Cycle 138 rescue experiment after g186 FAIL.

Question: Can corpus-derived interface priors (PPMI SVD of C4 co-occurrence)
replace trained-model donors for the +0.513 nat embed/lm_head anchor effect?

Stage A (primary, <=4h):
  1. scratch_ce -- baseline
  2. trained_anchor -- Qwen3 embed+lm_head anchor (g181b reference)
  3. ppmi_svd_anchor -- PPMI co-occurrence SVD from C4, anchor at lambda=0.01

Stage B (conditional on ppmi_svd >= 0.15 nats vs scratch):
  4. frequency_anchor -- random orthonormal dirs scaled by unigram log-freq
  5. random_structured_anchor -- random orthonormal scaled to Qwen3 embed Frobenius norm
  6. ppmi_svd_shuffled_rows -- same SVD but token-to-row permuted (adversarial)
  7. covariance_matched_anchor -- match trained embed covariance (trained-stat control)
  8. spectral_matched_anchor -- match trained embed spectrum (trained-stat control)

PASS: ppmi_svd recovers >= 50% of trained_anchor gap (+0.257 nats), beats scratch 3/3 seeds
FAIL: ppmi_svd < 50% of trained_anchor gap

Prereg: research/prereg/genome_183_corpus_derived_init_2026-04-30.md
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field
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

import genome_165_annealed_donor as g165
import genome_167_kd_canonical as g167
import genome_181a_tokenizer_isolation as g181a

OUT_PATH = ROOT / "results" / "genome_183_corpus_derived_init.json"

SEEDS = [42, 7, 13]
SEQ_LEN = g165.SEQ_LEN
BATCH_SIZE = g165.BATCH_SIZE
TRAIN_STEPS = 5000
N_TRAIN_WINDOWS = 16384
N_C4_VAL_WINDOWS = 256
LR = g165.LR
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
LOG_EVERY = 100
EVAL_EVERY = 500
C4_TRAIN_SEED = g165.C4_TRAIN_SEED
C4_VAL_SEED = g165.C4_VAL_SEED
ANCHOR_LAMBDA = 0.01
N_BOOT = 10_000
DEVICE = g165.DEVICE
FORWARD_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

COOC_WINDOW = 5
COOC_MIN_COUNT = 5
COOC_MAX_VOCAB = 50_000
SVD_DIM = None  # set dynamically from model config

STAGE_A_ARMS = ["scratch_ce", "trained_anchor", "ppmi_svd_anchor"]
STAGE_B_ARMS = [
    "frequency_anchor", "random_structured_anchor",
    "ppmi_svd_shuffled_rows",
    "covariance_matched_anchor", "spectral_matched_anchor",
]
ALL_ARMS = STAGE_A_ARMS + STAGE_B_ARMS

STAGE_B_GATE_NATS = 0.15
PASS_RECOVERY_FRAC = 0.50
G181B_TRAINED_GAP = 0.513

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


def autocast_context():
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


# ---------- Corpus preprocessing ----------

def collect_unigram_counts(
    train_ids: torch.Tensor, train_mask: torch.Tensor, vocab_size: int,
) -> np.ndarray:
    counts = np.zeros(vocab_size, dtype=np.int64)
    ids_np = train_ids.numpy().ravel()
    mask_np = train_mask.numpy().ravel()
    valid = ids_np[mask_np > 0]
    for tok_id in valid:
        if 0 <= tok_id < vocab_size:
            counts[tok_id] += 1
    return counts


def build_cooccurrence_sparse(
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    vocab_size: int,
    window: int = COOC_WINDOW,
    min_count: int = COOC_MIN_COUNT,
    max_vocab: int = COOC_MAX_VOCAB,
) -> tuple[Any, np.ndarray, np.ndarray]:
    from scipy import sparse

    unigram = collect_unigram_counts(train_ids, train_mask, vocab_size)
    top_ids = np.argsort(-unigram)[:max_vocab]
    top_set = set(top_ids.tolist())
    id_to_idx = {int(tid): i for i, tid in enumerate(top_ids)}

    rows, cols, vals = [], [], []
    ids_np = train_ids.numpy()
    mask_np = train_mask.numpy()

    for seq_i in range(ids_np.shape[0]):
        seq = ids_np[seq_i]
        msk = mask_np[seq_i]
        valid_pos = np.where(msk > 0)[0]
        for pi, pos in enumerate(valid_pos):
            center = int(seq[pos])
            if center not in top_set:
                continue
            ci = id_to_idx[center]
            start = max(0, pi - window)
            end = min(len(valid_pos), pi + window + 1)
            for pj in range(start, end):
                if pj == pi:
                    continue
                ctx = int(seq[valid_pos[pj]])
                if ctx not in top_set:
                    continue
                cj = id_to_idx[ctx]
                rows.append(ci)
                cols.append(cj)
                vals.append(1.0)

    n = len(top_ids)
    cooc = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    cooc = cooc.tocsr()
    cooc = cooc + cooc.T
    cooc.data[cooc.data < min_count] = 0
    cooc.eliminate_zeros()

    print_flush(f"  Co-occurrence: {n}x{n} sparse, {cooc.nnz} nonzero entries")
    return cooc, top_ids, unigram


def ppmi_from_cooccurrence(
    cooc, unigram: np.ndarray, top_ids: np.ndarray,
) -> Any:
    from scipy import sparse

    row_sums = np.array(cooc.sum(axis=1)).flatten()
    total = row_sums.sum()
    if total < 1:
        raise RuntimeError("Empty co-occurrence matrix")

    cooc_coo = cooc.tocoo()
    r, c, v = cooc_coo.row, cooc_coo.col, cooc_coo.data

    p_ij = v / total
    p_i = row_sums[r] / total
    p_j = row_sums[c] / total

    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(p_ij / (p_i * p_j))
    pmi[~np.isfinite(pmi)] = 0.0
    ppmi_vals = np.maximum(pmi, 0.0)

    ppmi = sparse.coo_matrix((ppmi_vals, (r, c)), shape=cooc.shape, dtype=np.float64)
    ppmi = ppmi.tocsr()
    ppmi.eliminate_zeros()
    print_flush(f"  PPMI matrix: {ppmi.shape[0]}x{ppmi.shape[1]}, {ppmi.nnz} nonzero")
    return ppmi


def truncated_svd_embeddings(
    ppmi, dim: int = SVD_DIM, seed: int = 42,
) -> np.ndarray:
    from scipy.sparse.linalg import svds
    k = min(dim, ppmi.shape[0] - 1, ppmi.shape[1] - 1)
    U, S, _ = svds(ppmi.astype(np.float64), k=k, random_state=seed)
    idx = np.argsort(-S)
    U = U[:, idx]
    S = S[idx]
    embeddings = U * np.sqrt(S)[np.newaxis, :]
    print_flush(f"  SVD embeddings: {embeddings.shape}, top-5 singular values: {S[:5]}")
    return embeddings.astype(np.float32)


def expand_to_full_vocab(
    top_ids: np.ndarray, vectors: np.ndarray,
    vocab_size: int, embed_dim: int, seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    full = rng.standard_normal((vocab_size, embed_dim)).astype(np.float32)
    scale = np.linalg.norm(vectors, axis=1).mean()
    full *= scale / (np.linalg.norm(full, axis=1, keepdims=True) + 1e-8)
    for i, tid in enumerate(top_ids):
        if tid < vocab_size and i < vectors.shape[0]:
            full[tid] = vectors[i]
    return full


def make_frequency_embeddings(
    unigram: np.ndarray, vocab_size: int, embed_dim: int, seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((embed_dim, embed_dim)).astype(np.float32))
    dirs = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    for i in range(vocab_size):
        dirs[i] = Q[i % embed_dim]
    log_freq = np.log1p(unigram.astype(np.float32))
    log_freq = log_freq / (log_freq.max() + 1e-8)
    dirs *= log_freq[:, np.newaxis]
    return dirs


def make_random_structured_embeddings(
    target_fro_norm: float, vocab_size: int, embed_dim: int, seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((embed_dim, embed_dim)).astype(np.float32))
    full = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    for i in range(vocab_size):
        full[i] = Q[i % embed_dim]
    current_fro = np.linalg.norm(full)
    if current_fro > 1e-8:
        full *= target_fro_norm / current_fro
    return full


def make_shuffled_svd_embeddings(
    svd_full: np.ndarray, seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 9999)
    shuffled = svd_full.copy()
    perm = rng.permutation(shuffled.shape[0])
    shuffled = shuffled[perm]
    return shuffled


def make_covariance_matched_embeddings(
    trained_embed: np.ndarray, vocab_size: int, embed_dim: int, seed: int = 42,
) -> np.ndarray:
    cov = np.cov(trained_embed.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0)
    sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((vocab_size, embed_dim)).astype(np.float32)
    matched = z @ sqrt_cov.astype(np.float32)
    return matched


def make_spectral_matched_embeddings(
    trained_embed: np.ndarray, vocab_size: int, embed_dim: int, seed: int = 42,
) -> np.ndarray:
    _, S, _ = np.linalg.svd(trained_embed, full_matrices=False)
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((vocab_size, embed_dim)).astype(np.float32))
    k = min(len(S), embed_dim)
    diag = np.zeros(embed_dim, dtype=np.float32)
    diag[:k] = S[:k].astype(np.float32)
    matched = Q @ np.diag(diag)
    return matched


# ---------- Interface injection ----------

def inject_embed_weights(model, embed_array: np.ndarray) -> None:
    embed_tensor = torch.from_numpy(embed_array).to(dtype=torch.bfloat16, device=DEVICE)
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(embed_tensor)
        if hasattr(model, "lm_head") and model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
            model.lm_head.weight.copy_(embed_tensor)


def build_custom_anchor_pairs(
    recipient: torch.nn.Module,
    target_embed: np.ndarray,
) -> list[tuple[str, torch.nn.Parameter, torch.Tensor]]:
    target_tensor = torch.from_numpy(target_embed).to(dtype=torch.float32, device=DEVICE)
    pairs = []
    for name, param in recipient.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            if param.shape == target_tensor.shape:
                pairs.append((name, param, target_tensor))
    if not pairs:
        raise RuntimeError("No embed/lm_head params found for custom anchor")
    return pairs


# ---------- Training loop ----------

def train_cell(
    arm_label: str,
    seed: int,
    anchor_pairs: list[tuple[str, torch.nn.Parameter, torch.Tensor]],
    anchor_lambda: float,
    custom_embed: np.ndarray | None,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
    n_steps: int = TRAIN_STEPS,
    eval_every: int = EVAL_EVERY,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    recipient = g165.load_random_init(seed)
    if hasattr(recipient.config, "use_cache"):
        recipient.config.use_cache = False

    if custom_embed is not None:
        inject_embed_weights(recipient, custom_embed)

    optimizer = torch.optim.AdamW(
        recipient.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY,
    )
    rng = np.random.default_rng(seed)
    train_schedule = rng.integers(0, int(train_ids.shape[0]), size=(n_steps, BATCH_SIZE), dtype=np.int64)

    actual_pairs = []
    if anchor_pairs:
        for name_template, _, target_tensor in anchor_pairs:
            for name, param in recipient.named_parameters():
                if name == name_template and param.shape == target_tensor.shape:
                    actual_pairs.append((name, param, target_tensor))

    trajectory = []
    initial_metrics = g181a.evaluate_nll(recipient, val_ids, val_mask)
    trajectory.append({"step": 0, **initial_metrics})
    print_flush(f"    {arm_label} seed={seed} step=0 nll={initial_metrics['nll']:.4f}")

    t0 = time.time()
    recipient.train()
    for step in range(1, n_steps + 1):
        batch_indices = train_schedule[step - 1]
        ids = train_ids[torch.as_tensor(batch_indices, dtype=torch.long)].to(DEVICE)
        mask = train_mask[torch.as_tensor(batch_indices, dtype=torch.long)].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            logits = recipient(input_ids=ids, attention_mask=mask, use_cache=False).logits
            ce_loss = g167.causal_ce_loss(logits, ids, mask)

        if not torch.isfinite(ce_loss):
            raise RuntimeError(f"non-finite CE loss at step {step} arm={arm_label} seed={seed}")

        ce_loss.backward()

        if actual_pairs and anchor_lambda > 0.0:
            with torch.no_grad():
                coeff = 2.0 * anchor_lambda
                for _, param, donor_tensor in actual_pairs:
                    if param.grad is None:
                        continue
                    param.grad.add_(param.detach().to(donor_tensor.dtype) - donor_tensor, alpha=coeff)

        torch.nn.utils.clip_grad_norm_(recipient.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % LOG_EVERY == 0 or step == n_steps:
            row = {"step": step, "ce_loss": float(ce_loss.item()), "elapsed_s": time.time() - t0}
            if step % eval_every == 0 or step == n_steps:
                row.update(g181a.evaluate_nll(recipient, val_ids, val_mask))
                print_flush(f"    {arm_label} seed={seed} step={step} ce={row['ce_loss']:.4f} nll={row['nll']:.4f} ({row['elapsed_s']:.0f}s)")
            elif step % (LOG_EVERY * 5) == 0:
                print_flush(f"    {arm_label} seed={seed} step={step} ce={row['ce_loss']:.4f} ({row['elapsed_s']:.0f}s)")
            trajectory.append(row)

    final_metrics = trajectory[-1]
    if "nll" not in final_metrics:
        final_metrics = {"step": n_steps, **g181a.evaluate_nll(recipient, val_ids, val_mask)}
        trajectory.append(final_metrics)

    result = {
        "seed": seed, "arm_label": arm_label,
        "anchor_lambda": anchor_lambda,
        "initial_nll": float(initial_metrics["nll"]),
        "final_nll": float(final_metrics["nll"]),
        "final_top1_acc": float(final_metrics.get("top1_acc", 0)),
        "trajectory": trajectory,
        "wallclock_s": time.time() - t0,
    }
    del recipient, optimizer
    cleanup_cuda()
    return result


# ---------- Verdict ----------

def compute_verdict(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload["results"]
    arms_done = {arm for arm in results if results[arm]}
    stage_a_done = all(
        str(s) in results.get(arm, {}) for arm in STAGE_A_ARMS for s in SEEDS
    )
    if not stage_a_done:
        return {"status": "incomplete", "stage": "A"}

    scratch_nlls = [float(results["scratch_ce"][str(s)]["final_nll"]) for s in SEEDS]
    trained_nlls = [float(results["trained_anchor"][str(s)]["final_nll"]) for s in SEEDS]
    ppmi_nlls = [float(results["ppmi_svd_anchor"][str(s)]["final_nll"]) for s in SEEDS]

    trained_gaps = [scratch_nlls[i] - trained_nlls[i] for i in range(len(SEEDS))]
    ppmi_gaps = [scratch_nlls[i] - ppmi_nlls[i] for i in range(len(SEEDS))]

    mean_trained_gap = float(np.mean(trained_gaps))
    mean_ppmi_gap = float(np.mean(ppmi_gaps))
    ppmi_beats_scratch_all = all(g > 0 for g in ppmi_gaps)
    recovery_frac = mean_ppmi_gap / mean_trained_gap if abs(mean_trained_gap) > 1e-8 else 0.0

    rng = np.random.default_rng(183000)
    arr = np.asarray(ppmi_gaps, dtype=np.float64)
    boot_means = np.empty(N_BOOT, dtype=np.float64)
    for i in range(N_BOOT):
        boot_means[i] = arr[rng.integers(0, len(arr), size=len(arr))].mean()
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    p1_pass = mean_ppmi_gap >= PASS_RECOVERY_FRAC * G181B_TRAINED_GAP
    p2_pass = ppmi_beats_scratch_all

    best_control_gap = 0.0
    best_control_name = "scratch_ce"
    for arm in ["random_structured_anchor", "frequency_anchor"]:
        if arm in results and results[arm]:
            arm_nlls = [float(results[arm][str(s)]["final_nll"]) for s in SEEDS if str(s) in results[arm]]
            if len(arm_nlls) == len(SEEDS):
                arm_gap = float(np.mean([scratch_nlls[i] - arm_nlls[i] for i in range(len(SEEDS))]))
                if arm_gap > best_control_gap:
                    best_control_gap = arm_gap
                    best_control_name = arm
    p3_pass = (mean_ppmi_gap - best_control_gap) >= 0.10

    stage_b_gate = mean_ppmi_gap >= STAGE_B_GATE_NATS

    if p1_pass and p2_pass and p3_pass:
        if recovery_frac >= 0.80:
            verdict = "STRONG_PASS"
        else:
            verdict = "PASS"
    elif p2_pass and not p1_pass:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    summary = {
        "verdict": verdict,
        "stage_a_complete": True,
        "stage_b_gate": stage_b_gate,
        "mean_trained_gap": mean_trained_gap,
        "mean_ppmi_gap": mean_ppmi_gap,
        "recovery_fraction": recovery_frac,
        "ppmi_beats_scratch_all_seeds": ppmi_beats_scratch_all,
        "trained_gaps_per_seed": {str(s): trained_gaps[i] for i, s in enumerate(SEEDS)},
        "ppmi_gaps_per_seed": {str(s): ppmi_gaps[i] for i, s in enumerate(SEEDS)},
        "ppmi_ci_95": [ci_lo, ci_hi],
        "p1_recovery_pass": p1_pass,
        "p2_all_seeds_pass": p2_pass,
        "p3_beats_control": p3_pass,
        "best_non_corpus_control": best_control_name,
        "best_control_gap": best_control_gap,
        "scratch_mean_nll": float(np.mean(scratch_nlls)),
        "trained_mean_nll": float(np.mean(trained_nlls)),
        "ppmi_mean_nll": float(np.mean(ppmi_nlls)),
    }

    for arm in STAGE_B_ARMS:
        if arm in results and results[arm]:
            arm_nlls = [float(results[arm][str(s)]["final_nll"]) for s in SEEDS if str(s) in results[arm]]
            if arm_nlls:
                arm_gaps = [scratch_nlls[i] - arm_nlls[i] for i in range(min(len(SEEDS), len(arm_nlls)))]
                summary[f"{arm}_mean_gap"] = float(np.mean(arm_gaps))
                summary[f"{arm}_per_seed"] = {str(SEEDS[i]): arm_gaps[i] for i in range(len(arm_gaps))}

    return summary


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Genome 183 corpus-derived init.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--stage-b", action="store_true", help="Run Stage B arms")
    parser.add_argument("--smoke", action="store_true", help="Smoke test: 1 seed, 50 steps")
    parser.add_argument("--confound-check", action="store_true",
                        help="Run ppmi_svd_anchor_no_init (anchor-only, no weight injection) seed 42")
    args = parser.parse_args()

    smoke = args.smoke
    seeds = [42] if smoke else SEEDS
    train_steps = 50 if smoke else TRAIN_STEPS

    if args.confound_check:
        arms_to_run = ["ppmi_svd_anchor_no_init"]
        seeds = [42]
    elif args.stage_b:
        arms_to_run = STAGE_B_ARMS
    else:
        arms_to_run = STAGE_A_ARMS

    print_flush(f"=== genome_183 corpus-derived init ({now_utc()}) ===")
    print_flush(f"  model={g165._MODEL_ID}")
    print_flush(f"  device={DEVICE} dtype={FORWARD_DTYPE}")
    print_flush(f"  arms={arms_to_run}")
    print_flush(f"  seeds={seeds} steps={train_steps}")
    if smoke:
        print_flush("  *** SMOKE TEST MODE ***")

    t_start = time.time()

    if not args.no_resume and OUT_PATH.exists():
        payload = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    else:
        payload = {
            "genome": "183",
            "name": "corpus_derived_init",
            "timestamp_utc_started": now_utc(),
            "model_id": g165._MODEL_ID,
            "device": DEVICE,
            "config": {
                "seeds": seeds,
                "train_steps": train_steps,
                "anchor_lambda": ANCHOR_LAMBDA,
                "cooc_window": COOC_WINDOW,
                "svd_dim": "model_embed_dim",
                "cooc_max_vocab": COOC_MAX_VOCAB,
            },
            "preprocessing": {},
            "results": {},
            "summary": {},
            "verdict": "INCOMPLETE",
        }
    payload.setdefault("results", {})
    payload.setdefault("preprocessing", {})

    def save(incremental=True):
        payload["timestamp_utc_last_write"] = now_utc()
        payload["elapsed_s"] = time.time() - t_start
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUT_PATH.with_suffix(OUT_PATH.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        os.replace(tmp, OUT_PATH)

    # --- Load tokenizer and data ---
    donor_model, tok = g165.load_trained_donor()
    vocab_size = donor_model.config.vocab_size
    embed_dim = donor_model.config.hidden_size

    trained_embed_np = donor_model.model.embed_tokens.weight.detach().cpu().float().numpy()
    trained_fro_norm = float(np.linalg.norm(trained_embed_np))
    payload["preprocessing"]["trained_embed_fro_norm"] = trained_fro_norm
    payload["preprocessing"]["vocab_size"] = vocab_size
    payload["preprocessing"]["embed_dim"] = embed_dim

    donor_params_cpu = g181a.snapshot_params_cpu(donor_model)
    del donor_model
    cleanup_cuda()
    donor_params_device = g181a.stage_params_to_device(donor_params_cpu)
    del donor_params_cpu
    cleanup_cuda()

    print_flush("  Loading C4 data...")
    train_ids, train_mask, val_ids, val_mask, data_meta = g181a.load_main_data(tok) if hasattr(g181a, 'load_main_data') else _load_data(tok)
    payload["data"] = data_meta
    save()

    # --- Build corpus-derived embeddings ---
    print_flush("  Building corpus embeddings...")
    cooc, top_ids, unigram = build_cooccurrence_sparse(train_ids, train_mask, vocab_size)
    ppmi = ppmi_from_cooccurrence(cooc, unigram, top_ids)
    svd_vecs = truncated_svd_embeddings(ppmi, dim=embed_dim, seed=42)
    svd_full = expand_to_full_vocab(top_ids, svd_vecs, vocab_size, embed_dim, seed=42)
    payload["preprocessing"]["cooc_nnz"] = int(cooc.nnz)
    payload["preprocessing"]["ppmi_nnz"] = int(ppmi.nnz)
    payload["preprocessing"]["top_vocab_size"] = len(top_ids)

    freq_embed = make_frequency_embeddings(unigram, vocab_size, embed_dim, seed=42)
    rand_struct_embed = make_random_structured_embeddings(trained_fro_norm, vocab_size, embed_dim, seed=42)
    shuffled_svd = make_shuffled_svd_embeddings(svd_full, seed=42)
    cov_matched = make_covariance_matched_embeddings(trained_embed_np, vocab_size, embed_dim, seed=42)
    spec_matched = make_spectral_matched_embeddings(trained_embed_np, vocab_size, embed_dim, seed=42)

    del cooc, ppmi, svd_vecs
    gc.collect()

    payload["preprocessing"]["svd_embed_fro_norm"] = float(np.linalg.norm(svd_full))
    payload["preprocessing"]["freq_embed_fro_norm"] = float(np.linalg.norm(freq_embed))
    payload["preprocessing"]["preprocessing_time_s"] = time.time() - t_start
    save()

    embed_map = {
        "ppmi_svd_anchor": svd_full,
        "frequency_anchor": freq_embed,
        "random_structured_anchor": rand_struct_embed,
        "ppmi_svd_shuffled_rows": shuffled_svd,
        "covariance_matched_anchor": cov_matched,
        "spectral_matched_anchor": spec_matched,
    }

    eval_every = 25 if smoke else EVAL_EVERY
    actual_steps = train_steps

    # --- Train cells ---
    for arm_label in arms_to_run:
        payload["results"].setdefault(arm_label, {})
        for seed in seeds:
            if str(seed) in payload["results"][arm_label]:
                cell = payload["results"][arm_label][str(seed)]
                if isinstance(cell, dict) and cell.get("final_nll") is not None:
                    print_flush(f"  SKIP {arm_label} seed={seed} (done)")
                    continue

            print_flush(f"\n--- arm={arm_label} seed={seed} ---")

            if arm_label == "scratch_ce":
                anchor_pairs = []
                anchor_lam = 0.0
                custom_embed = None
            elif arm_label == "trained_anchor":
                anchor_pairs = g181a.build_anchor_pairs(
                    g165.load_random_init(seed), donor_params_device, "embed_lm_head",
                )
                cleanup_cuda()
                anchor_lam = ANCHOR_LAMBDA
                custom_embed = None
            elif arm_label == "ppmi_svd_anchor_no_init":
                custom_embed_arr = embed_map.get("ppmi_svd_anchor")
                if custom_embed_arr is None:
                    print_flush(f"  WARNING: no ppmi_svd embed for confound check, skipping")
                    continue
                dummy_model = g165.load_random_init(seed)
                anchor_pairs = build_custom_anchor_pairs(dummy_model, custom_embed_arr)
                del dummy_model
                cleanup_cuda()
                anchor_lam = ANCHOR_LAMBDA
                custom_embed = None
            else:
                custom_embed_arr = embed_map.get(arm_label)
                if custom_embed_arr is None:
                    print_flush(f"  WARNING: no embed for {arm_label}, skipping")
                    continue
                dummy_model = g165.load_random_init(seed)
                anchor_pairs = build_custom_anchor_pairs(dummy_model, custom_embed_arr)
                del dummy_model
                cleanup_cuda()
                anchor_lam = ANCHOR_LAMBDA
                custom_embed = custom_embed_arr

            result = train_cell(
                arm_label=arm_label,
                seed=seed,
                anchor_pairs=anchor_pairs,
                anchor_lambda=anchor_lam,
                custom_embed=custom_embed,
                train_ids=train_ids,
                train_mask=train_mask,
                val_ids=val_ids,
                val_mask=val_mask,
                n_steps=actual_steps,
                eval_every=eval_every,
            )
            payload["results"][arm_label][str(seed)] = result

            summary = compute_verdict(payload)
            payload["summary"] = summary
            payload["verdict"] = summary.get("verdict", "INCOMPLETE")
            save()

            print_flush(f"  {arm_label} seed={seed} final_nll={result['final_nll']:.4f} ({result['wallclock_s']:.0f}s)")

    # --- Final summary ---
    summary = compute_verdict(payload)
    payload["summary"] = summary
    payload["verdict"] = summary.get("verdict", "INCOMPLETE")
    payload["status"] = "completed" if summary.get("stage_a_complete") else "running"
    save()

    print_flush(f"\n*** g183 VERDICT: {summary.get('verdict', '?')} ***")
    tg = summary.get('mean_trained_gap')
    pg = summary.get('mean_ppmi_gap')
    rf = summary.get('recovery_fraction')
    if tg is not None and pg is not None:
        print_flush(f"  trained_gap={tg:.4f} ppmi_gap={pg:.4f}")
    if rf is not None:
        print_flush(f"  recovery={rf:.1%}")
    print_flush(f"  stage_b_gate={'PASS' if summary.get('stage_b_gate') else 'FAIL/INCOMPLETE'}")


def _load_data(tok):
    train_ids, train_mask, train_meta = g167.load_c4_windows(
        tok, split="train", seed=C4_TRAIN_SEED, n_windows=N_TRAIN_WINDOWS,
    )
    train_hashes = g167.collect_13gram_hashes(train_ids, train_mask)
    val_ids, val_mask, val_meta = g167.load_c4_windows(
        tok, split="train", seed=C4_VAL_SEED, n_windows=N_C4_VAL_WINDOWS,
        forbidden_hashes=train_hashes,
    )
    meta = {
        "train": train_meta, "c4_val": val_meta,
        "train_seed": C4_TRAIN_SEED, "val_seed": C4_VAL_SEED,
        "train_shape": list(train_ids.shape), "val_shape": list(val_ids.shape),
    }
    return train_ids, train_mask, val_ids, val_mask, meta


if __name__ == "__main__":
    main()
