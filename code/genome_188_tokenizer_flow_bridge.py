"""
genome_188_tokenizer_flow_bridge.py

Codex design gate: codex_outputs/g188_tokenizer_flow_bridge_design_gate_20260430.md
Prereg: research/prereg/genome_188_tokenizer_flow_bridge_2026-04-30.md

Tests whether trained Qwen3 interface embeddings can be transcoded through a
sparse tokenizer-alignment graph (character-offset overlap + Sinkhorn balancing)
into GPT-2 tokenizer space, producing anchor targets that recover a meaningful
fraction of the within-family trained-anchor effect (+0.513 nats, g181b).

8 arms: scratch_ce, flow_bridge_init_anchor, flow_anchor_only, flow_init_only,
char_overlap_no_ot, direct_string_match_anchor, flow_shuffled_qwen_rows,
flow_random_source. All GPT-2-tokenizer Qwen3-arch, 3 seeds.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import genome_165_annealed_donor as g165
import genome_167_kd_canonical as g167

OUT_PATH = ROOT / "results" / "genome_188_tokenizer_flow_bridge.json"
CACHE_DIR = ROOT / "results" / "cache" / "genome_188_tokenizer_flow_bridge"

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

ALIGNMENT_N_SPANS = 50_000
TOP_QWEN_TOKENS = 20_000
TOP_GPT2_TOKENS = 20_000
SINKHORN_ITERS = 100
SINKHORN_EPS = 0.1

QWEN_MODEL_ID = "Qwen/Qwen3-0.6B"
GPT2_MODEL_ID = "openai-community/gpt2"

STAGE_A_ARMS = [
    "scratch_ce",
    "flow_bridge_init_anchor",
    "char_overlap_no_ot",
    "direct_string_match_anchor",
    "flow_shuffled_qwen_rows",
    "flow_random_source",
]
STAGE_B_ARMS = [
    "flow_anchor_only",
    "flow_init_only",
]
STAGE_B_GATE_NATS = 0.05

PASS_NATS = 0.12
PARTIAL_NATS = 0.05
STRONG_PASS_NATS = 0.20
BEATS_CHAR_OVERLAP = 0.04
BEATS_STRING_MATCH = 0.05
BEATS_SHUFFLED_RANDOM = 0.08

G181B_TRAINED_GAP = 0.513


def print_flush(msg: str) -> None:
    print(msg, flush=True)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------- Tokenizer alignment ----------

def tokenize_with_offsets(text: str, tokenizer) -> list[tuple[int, int, int]]:
    """Return list of (token_id, char_start, char_end) for each token."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    result = []
    for tid, (start, end) in zip(token_ids, offsets):
        result.append((tid, start, end))
    return result


def build_offset_alignment_edges(
    texts: list[str],
    tok_src,
    tok_tgt,
    top_src_ids: set[int],
    top_tgt_ids: set[int],
) -> sparse.coo_matrix:
    """Build sparse bipartite alignment graph via character-offset overlap.

    For each text span, tokenize with both tokenizers. Two tokens (src_i, tgt_j)
    get an edge weight proportional to their character overlap in that span.
    Only tokens in top_src_ids/top_tgt_ids are included.

    Returns COO matrix of shape (max_src_id+1, max_tgt_id+1).
    """
    from collections import defaultdict
    edge_counts = defaultdict(float)

    for text in texts:
        src_toks = tokenize_with_offsets(text, tok_src)
        tgt_toks = tokenize_with_offsets(text, tok_tgt)

        for s_id, s_start, s_end in src_toks:
            if s_id not in top_src_ids or s_start >= s_end:
                continue
            for t_id, t_start, t_end in tgt_toks:
                if t_id not in top_tgt_ids or t_start >= t_end:
                    continue
                overlap = max(0, min(s_end, t_end) - max(s_start, t_start))
                if overlap > 0:
                    edge_counts[(s_id, t_id)] += overlap

    if not edge_counts:
        raise ValueError("No alignment edges found!")

    rows, cols, vals = [], [], []
    for (si, ti), w in edge_counts.items():
        rows.append(si)
        cols.append(ti)
        vals.append(w)

    max_src = max(rows) + 1
    max_tgt = max(cols) + 1
    mat = sparse.coo_matrix(
        (np.array(vals, dtype=np.float64), (np.array(rows), np.array(cols))),
        shape=(max_src, max_tgt),
    )
    return mat.tocsr()


def sparse_sinkhorn_balance(
    alignment: sparse.csr_matrix,
    eps: float = SINKHORN_EPS,
    n_iter: int = SINKHORN_ITERS,
) -> sparse.csr_matrix:
    """Sinkhorn-like row/column balancing on a sparse alignment matrix.

    Applies iterative row and column normalization to produce a doubly-stochastic-like
    coupling. Not a true OT solve but computationally tractable on sparse graphs.
    """
    coo = alignment.tocoo().astype(np.float64)
    vals = coo.data.copy()
    rows = coo.row
    cols = coo.col
    shape = coo.shape

    vals = np.exp(-vals / (eps * vals.max() + 1e-12))
    vals = np.maximum(vals, 1e-20)

    for _ in range(n_iter):
        mat = sparse.coo_matrix((vals, (rows, cols)), shape=shape).tocsr()
        row_sums = np.array(mat.sum(axis=1)).flatten()
        row_sums = np.maximum(row_sums, 1e-20)
        for idx in range(len(vals)):
            vals[idx] /= row_sums[rows[idx]]

        mat = sparse.coo_matrix((vals, (rows, cols)), shape=shape).tocsc()
        col_sums = np.array(mat.sum(axis=0)).flatten()
        col_sums = np.maximum(col_sums, 1e-20)
        for idx in range(len(vals)):
            vals[idx] /= col_sums[cols[idx]]

    plan = sparse.coo_matrix((vals, (rows, cols)), shape=shape).tocsr()
    return plan


def barycentric_target_embeddings(
    plan: sparse.csr_matrix,
    src_embeddings: np.ndarray,
    tgt_vocab_size: int,
    tgt_embed_dim: int,
    seed: int = 42,
) -> np.ndarray:
    """Compute target embeddings as weighted sum of source embeddings via OT plan.

    For each target token j, embedding = sum_i plan[i,j] * src_embeddings[i] / sum_i plan[i,j].
    Tokens with no plan edges get fallback embeddings (mean of covered tokens).
    """
    plan_csc = plan.tocsc()
    result = np.zeros((tgt_vocab_size, tgt_embed_dim), dtype=np.float32)
    covered = np.zeros(tgt_vocab_size, dtype=bool)

    for j in range(min(tgt_vocab_size, plan_csc.shape[1])):
        col_start = plan_csc.indptr[j]
        col_end = plan_csc.indptr[j + 1]
        if col_start == col_end:
            continue
        src_ids = plan_csc.indices[col_start:col_end]
        weights = plan_csc.data[col_start:col_end].astype(np.float64)

        valid = src_ids < src_embeddings.shape[0]
        if not valid.any():
            continue

        src_ids = src_ids[valid]
        weights = weights[valid]
        w_sum = weights.sum()
        if w_sum < 1e-20:
            continue

        weights /= w_sum
        emb = np.zeros(tgt_embed_dim, dtype=np.float64)
        for k in range(len(src_ids)):
            emb += weights[k] * src_embeddings[src_ids[k]].astype(np.float64)
        result[j] = emb.astype(np.float32)
        covered[j] = True

    if covered.any():
        mean_emb = result[covered].mean(axis=0)
        result[~covered] = mean_emb

    return result


def direct_string_match_embeddings(
    tok_src,
    tok_tgt,
    src_embeddings: np.ndarray,
    tgt_vocab_size: int,
    tgt_embed_dim: int,
) -> np.ndarray:
    """Map source embeddings to target vocabulary by exact string match."""
    src_vocab = tok_src.get_vocab()
    tgt_vocab = tok_tgt.get_vocab()
    src_inv = {v: k for k, v in src_vocab.items()}

    result = np.zeros((tgt_vocab_size, tgt_embed_dim), dtype=np.float32)
    matched = np.zeros(tgt_vocab_size, dtype=bool)

    for token_str, tgt_id in tgt_vocab.items():
        if tgt_id >= tgt_vocab_size:
            continue
        if token_str in src_vocab:
            src_id = src_vocab[token_str]
            if src_id < src_embeddings.shape[0]:
                result[tgt_id] = src_embeddings[src_id]
                matched[tgt_id] = True

    if matched.any():
        mean_emb = result[matched].mean(axis=0)
        result[~matched] = mean_emb

    n_matched = matched.sum()
    print_flush(f"  String match: {n_matched}/{tgt_vocab_size} tokens matched ({100*n_matched/tgt_vocab_size:.1f}%)")
    return result


def char_overlap_embeddings(
    alignment: sparse.csr_matrix,
    src_embeddings: np.ndarray,
    tgt_vocab_size: int,
    tgt_embed_dim: int,
) -> np.ndarray:
    """Map source embeddings to target using raw character-overlap weights (no OT)."""
    return barycentric_target_embeddings(
        alignment, src_embeddings, tgt_vocab_size, tgt_embed_dim,
    )


def normalize_to_fro_norm(embeddings: np.ndarray, target_norm: float) -> np.ndarray:
    """Scale embedding matrix to match target Frobenius norm."""
    current = np.linalg.norm(embeddings)
    if current < 1e-8:
        return embeddings
    return embeddings * (target_norm / current)


# ---------- Training ----------

def make_gpt2_qwen3_model(tok_gpt2, seed: int):
    """Create a Qwen3-arch model with GPT-2 vocab size, matching Qwen3-0.6B hidden_size."""
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
        max_position_embeddings=SEQ_LEN + 64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        head_dim=64,
        rope_theta=10000.0,
        use_cache=False,
        bos_token_id=tok_gpt2.bos_token_id if tok_gpt2.bos_token_id is not None else tok_gpt2.eos_token_id,
        eos_token_id=tok_gpt2.eos_token_id,
        pad_token_id=tok_gpt2.pad_token_id if tok_gpt2.pad_token_id is not None else tok_gpt2.eos_token_id,
    )
    if hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = "eager"
    model = Qwen3ForCausalLM(cfg)
    model.tie_weights()
    model.to(DEVICE)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def build_anchor_targets(
    custom_embed: np.ndarray,
) -> list[tuple[str, torch.Tensor]]:
    """Create (param_name, target) pairs for embed_in + lm_head anchor.

    Returns parameter names so targets are resolved against the actual training
    model, not a throwaway dummy.
    """
    target = torch.from_numpy(custom_embed).to(dtype=torch.float32)
    return [
        ("model.embed_tokens.weight", target.clone()),
        ("lm_head.weight", target.clone()),
    ]


def train_cell(
    arm_label: str,
    seed: int,
    tok_gpt2,
    anchor_targets: list | None,
    anchor_lambda: float,
    custom_embed: np.ndarray | None,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
    n_steps: int = TRAIN_STEPS,
    eval_every: int = EVAL_EVERY,
) -> dict[str, Any]:
    """Train a single cell with GPT-2-tokenizer Qwen3-arch model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = make_gpt2_qwen3_model(tok_gpt2, seed)

    if custom_embed is not None:
        emb_t = torch.from_numpy(custom_embed).to(model.model.embed_tokens.weight.device,
                                                    dtype=model.model.embed_tokens.weight.dtype)
        model.model.embed_tokens.weight.data.copy_(emb_t)
        if hasattr(model, "lm_head") and not model.config.tie_word_embeddings:
            model.lm_head.weight.data.copy_(emb_t)

    actual_anchor_pairs = []
    if anchor_targets:
        param_dict = dict(model.named_parameters())
        for param_name, target_tensor in anchor_targets:
            if param_name in param_dict and param_dict[param_name].shape == target_tensor.shape:
                actual_anchor_pairs.append(
                    (param_dict[param_name], target_tensor.to(DEVICE, dtype=param_dict[param_name].dtype))
                )

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY,
    )

    n_train = train_ids.shape[0]
    trajectory = {}
    t0 = time.time()

    for step in range(1, n_steps + 1):
        idx = torch.randint(0, n_train, (BATCH_SIZE,))
        batch_ids = train_ids[idx].to(DEVICE)
        batch_mask = train_mask[idx].to(DEVICE)

        with torch.amp.autocast("cuda", dtype=FORWARD_DTYPE):
            out = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_ids)
            loss = out.loss

        if actual_anchor_pairs:
            anchor_loss = torch.tensor(0.0, device=DEVICE)
            for param, target in actual_anchor_pairs:
                anchor_loss = anchor_loss + F.mse_loss(param, target)
            loss = loss + anchor_lambda * anchor_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % LOG_EVERY == 0:
            print_flush(f"    step {step}/{n_steps} loss={loss.item():.4f}")

        if step % eval_every == 0 or step == n_steps:
            model.eval()
            with torch.no_grad():
                val_nll = _eval_nll(model, val_ids, val_mask)
            trajectory[str(step)] = float(val_nll)
            if step % eval_every == 0:
                print_flush(f"    eval step={step} val_nll={val_nll:.4f}")
            model.train()

    model.eval()
    with torch.no_grad():
        final_nll = _eval_nll(model, val_ids, val_mask)
    wallclock = time.time() - t0

    del model, optimizer
    cleanup_cuda()

    return {
        "arm_label": arm_label,
        "seed": seed,
        "final_nll": float(final_nll),
        "trajectory": trajectory,
        "wallclock_s": float(wallclock),
    }


def _eval_nll(model, val_ids, val_mask, batch_size=16) -> float:
    """Evaluate mean NLL on validation set."""
    total_loss = 0.0
    total_tokens = 0
    n = val_ids.shape[0]
    for i in range(0, n, batch_size):
        b_ids = val_ids[i:i+batch_size].to(DEVICE)
        b_mask = val_mask[i:i+batch_size].to(DEVICE)
        with torch.amp.autocast("cuda", dtype=FORWARD_DTYPE):
            out = model(input_ids=b_ids, attention_mask=b_mask, labels=b_ids)
        n_tok = b_mask.sum().item()
        total_loss += out.loss.item() * n_tok
        total_tokens += n_tok
    return total_loss / max(total_tokens, 1)


# ---------- Verdict ----------

def compute_verdict(payload: dict[str, Any]) -> dict[str, Any]:
    """Compute experiment verdict from results."""
    results = payload["results"]
    stage_a_done = all(
        str(s) in results.get(arm, {}) for arm in STAGE_A_ARMS for s in SEEDS
    )
    if not stage_a_done:
        return {"status": "incomplete", "stage": "A"}

    scratch_nlls = [float(results["scratch_ce"][str(s)]["final_nll"]) for s in SEEDS]
    flow_nlls = [float(results["flow_bridge_init_anchor"][str(s)]["final_nll"]) for s in SEEDS]

    flow_gaps = [scratch_nlls[i] - flow_nlls[i] for i in range(len(SEEDS))]
    mean_flow_gap = float(np.mean(flow_gaps))
    flow_beats_scratch_all = all(g > 0 for g in flow_gaps)

    char_nlls = [float(results["char_overlap_no_ot"][str(s)]["final_nll"]) for s in SEEDS]
    string_nlls = [float(results["direct_string_match_anchor"][str(s)]["final_nll"]) for s in SEEDS]
    shuf_nlls = [float(results["flow_shuffled_qwen_rows"][str(s)]["final_nll"]) for s in SEEDS]
    rand_nlls = [float(results["flow_random_source"][str(s)]["final_nll"]) for s in SEEDS]

    char_gaps = [scratch_nlls[i] - char_nlls[i] for i in range(len(SEEDS))]
    string_gaps = [scratch_nlls[i] - string_nlls[i] for i in range(len(SEEDS))]
    shuf_gaps = [scratch_nlls[i] - shuf_nlls[i] for i in range(len(SEEDS))]
    rand_gaps = [scratch_nlls[i] - rand_nlls[i] for i in range(len(SEEDS))]

    mean_char_gap = float(np.mean(char_gaps))
    mean_string_gap = float(np.mean(string_gaps))
    mean_shuf_gap = float(np.mean(shuf_gaps))
    mean_rand_gap = float(np.mean(rand_gaps))

    p1_pass = mean_flow_gap >= PASS_NATS and flow_beats_scratch_all
    p2_pass = (mean_flow_gap - mean_char_gap) >= BEATS_CHAR_OVERLAP
    p3_pass = (mean_flow_gap - mean_string_gap) >= BEATS_STRING_MATCH
    p4_shuf = (mean_flow_gap - mean_shuf_gap) >= BEATS_SHUFFLED_RANDOM and all(
        flow_gaps[i] > shuf_gaps[i] for i in range(len(SEEDS))
    )
    p4_rand = (mean_flow_gap - mean_rand_gap) >= BEATS_SHUFFLED_RANDOM and all(
        flow_gaps[i] > rand_gaps[i] for i in range(len(SEEDS))
    )
    p4_pass = p4_shuf and p4_rand

    if p1_pass and p2_pass and p3_pass and p4_pass:
        if mean_flow_gap >= STRONG_PASS_NATS:
            verdict = "STRONG_PASS"
        else:
            verdict = "PASS"
    elif mean_flow_gap >= PARTIAL_NATS and flow_beats_scratch_all:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    stage_b_gate = mean_flow_gap >= STAGE_B_GATE_NATS

    rng = np.random.default_rng(188000)
    arr = np.asarray(flow_gaps, dtype=np.float64)
    boot_means = np.empty(N_BOOT, dtype=np.float64)
    for i in range(N_BOOT):
        boot_means[i] = arr[rng.integers(0, len(arr), size=len(arr))].mean()
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    summary = {
        "verdict": verdict,
        "stage_a_complete": True,
        "stage_b_gate": stage_b_gate,
        "mean_flow_gap": mean_flow_gap,
        "mean_char_gap": mean_char_gap,
        "mean_string_gap": mean_string_gap,
        "mean_shuf_gap": mean_shuf_gap,
        "mean_rand_gap": mean_rand_gap,
        "flow_gaps_per_seed": {str(s): flow_gaps[i] for i, s in enumerate(SEEDS)},
        "flow_ci_95": [ci_lo, ci_hi],
        "p1_beats_scratch": p1_pass,
        "p2_beats_char_overlap": p2_pass,
        "p3_beats_string_match": p3_pass,
        "p4_beats_shuffled_random": p4_pass,
        "scratch_mean_nll": float(np.mean(scratch_nlls)),
        "flow_mean_nll": float(np.mean(flow_nlls)),
        "recovery_fraction": mean_flow_gap / G181B_TRAINED_GAP if G181B_TRAINED_GAP > 0 else 0.0,
    }

    for arm in STAGE_B_ARMS:
        if arm in results and results[arm]:
            arm_nlls = [float(results[arm][str(s)]["final_nll"]) for s in SEEDS if str(s) in results[arm]]
            if arm_nlls:
                arm_gaps = [scratch_nlls[i] - arm_nlls[i] for i in range(min(len(SEEDS), len(arm_nlls)))]
                summary[f"{arm}_mean_gap"] = float(np.mean(arm_gaps))

    return summary


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage-b", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    smoke = args.smoke
    seeds = [42] if smoke else SEEDS
    train_steps = 50 if smoke else TRAIN_STEPS
    eval_every_steps = 25 if smoke else EVAL_EVERY

    if args.stage_b:
        arms_to_run = STAGE_B_ARMS
    else:
        arms_to_run = STAGE_A_ARMS

    print_flush(f"=== g188 Tokenizer-Flow Bridge ===")
    print_flush(f"  smoke={smoke}, seeds={seeds}, steps={train_steps}")
    print_flush(f"  arms={arms_to_run}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_PATH.exists() and not args.no_resume:
        payload = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    else:
        payload = {
            "genome": 188,
            "name": "tokenizer_flow_bridge",
            "timestamp_utc_started": datetime.now(timezone.utc).isoformat(),
            "source_model_id": QWEN_MODEL_ID,
            "target_tokenizer_id": GPT2_MODEL_ID,
            "device": str(DEVICE),
            "config": {
                "seeds": seeds,
                "train_steps": train_steps,
                "anchor_lambda": ANCHOR_LAMBDA,
                "alignment_n_spans": ALIGNMENT_N_SPANS,
                "top_qwen_tokens": TOP_QWEN_TOKENS,
                "top_gpt2_tokens": TOP_GPT2_TOKENS,
                "sinkhorn_iters": SINKHORN_ITERS,
                "sinkhorn_eps": SINKHORN_EPS,
            },
            "preprocessing": {},
            "results": {},
            "summary": {},
            "verdict": "INCOMPLETE",
        }

    def save():
        payload["timestamp_utc_last_write"] = datetime.now(timezone.utc).isoformat()
        payload["elapsed_s"] = time.time() - t_start
        OUT_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    t_start = time.time()
    for arm in arms_to_run:
        if arm not in payload["results"]:
            payload["results"][arm] = {}

    # --- Step 1: Load tokenizers ---
    print_flush("\n--- Loading tokenizers ---")
    from transformers import AutoTokenizer
    tok_qwen = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
    tok_gpt2 = AutoTokenizer.from_pretrained(GPT2_MODEL_ID)
    if tok_gpt2.pad_token is None:
        tok_gpt2.pad_token = tok_gpt2.eos_token
        tok_gpt2.pad_token_id = tok_gpt2.eos_token_id
    qwen_vocab_size = tok_qwen.vocab_size
    gpt2_vocab_size = len(tok_gpt2)
    print_flush(f"  Qwen vocab: {qwen_vocab_size}, GPT-2 vocab: {gpt2_vocab_size}")

    # --- Step 2: Get trained Qwen3 embeddings ---
    print_flush("\n--- Loading trained Qwen3 embeddings ---")
    trained_model, _ = g165.load_trained_donor(tok_qwen)
    trained_embed = trained_model.model.embed_tokens.weight.detach().float().cpu().numpy().copy()
    embed_dim = trained_embed.shape[1]
    trained_fro_norm = float(np.linalg.norm(trained_embed))
    print_flush(f"  Trained embed: {trained_embed.shape}, Fro norm: {trained_fro_norm:.1f}")
    del trained_model
    cleanup_cuda()

    # --- Step 3: Load C4 spans for alignment ---
    print_flush("\n--- Loading C4 spans for alignment ---")
    alignment_cache = CACHE_DIR / "alignment_edges.npz"
    plan_cache = CACHE_DIR / "flow_plan_topk.npz"

    if alignment_cache.exists() and plan_cache.exists():
        print_flush("  Loading cached alignment + plan...")
        loader = np.load(str(alignment_cache), allow_pickle=True)
        alignment = sparse.csr_matrix(
            (loader["data"], loader["indices"], loader["indptr"]),
            shape=tuple(loader["shape"]),
        )
        loader2 = np.load(str(plan_cache), allow_pickle=True)
        plan = sparse.csr_matrix(
            (loader2["data"], loader2["indices"], loader2["indptr"]),
            shape=tuple(loader2["shape"]),
        )
        print_flush(f"  Alignment: {alignment.shape}, {alignment.nnz} edges")
        print_flush(f"  Plan: {plan.shape}, {plan.nnz} entries")
    else:
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=True)
        texts = []
        for i, ex in enumerate(ds):
            if i >= ALIGNMENT_N_SPANS:
                break
            texts.append(ex["text"][:512])  # truncate to 512 chars
        print_flush(f"  Loaded {len(texts)} C4 spans")

        top_qwen_ids = set(range(min(TOP_QWEN_TOKENS, qwen_vocab_size)))
        top_gpt2_ids = set(range(min(TOP_GPT2_TOKENS, gpt2_vocab_size)))

        print_flush("  Building alignment edges...")
        alignment = build_offset_alignment_edges(texts, tok_qwen, tok_gpt2, top_qwen_ids, top_gpt2_ids)
        print_flush(f"  Alignment: {alignment.shape}, {alignment.nnz} edges")

        np.savez_compressed(
            str(alignment_cache),
            data=alignment.data, indices=alignment.indices,
            indptr=alignment.indptr, shape=alignment.shape,
        )

        print_flush("  Running sparse Sinkhorn balancing...")
        plan = sparse_sinkhorn_balance(alignment, eps=SINKHORN_EPS, n_iter=SINKHORN_ITERS)
        print_flush(f"  Plan: {plan.shape}, {plan.nnz} entries")

        np.savez_compressed(
            str(plan_cache),
            data=plan.data, indices=plan.indices,
            indptr=plan.indptr, shape=plan.shape,
        )

        del texts
        gc.collect()

    payload["preprocessing"]["alignment_edges"] = int(alignment.nnz)
    payload["preprocessing"]["plan_entries"] = int(plan.nnz)
    payload["preprocessing"]["trained_embed_fro_norm"] = trained_fro_norm

    # --- Step 4: Build all embedding targets ---
    print_flush("\n--- Building embedding targets ---")

    flow_bridge_embed = barycentric_target_embeddings(
        plan, trained_embed, gpt2_vocab_size, embed_dim,
    )
    flow_bridge_embed = normalize_to_fro_norm(flow_bridge_embed, trained_fro_norm)
    print_flush(f"  flow_bridge: Fro={np.linalg.norm(flow_bridge_embed):.1f}")

    char_overlap_embed = char_overlap_embeddings(
        alignment, trained_embed, gpt2_vocab_size, embed_dim,
    )
    char_overlap_embed = normalize_to_fro_norm(char_overlap_embed, trained_fro_norm)

    string_match_embed = direct_string_match_embeddings(
        tok_qwen, tok_gpt2, trained_embed, gpt2_vocab_size, embed_dim,
    )
    string_match_embed = normalize_to_fro_norm(string_match_embed, trained_fro_norm)

    rng = np.random.default_rng(188001)
    shuffled_embed = trained_embed.copy()
    perm = rng.permutation(shuffled_embed.shape[0])
    shuffled_embed = shuffled_embed[perm]
    flow_shuffled_embed = barycentric_target_embeddings(
        plan, shuffled_embed, gpt2_vocab_size, embed_dim,
    )
    flow_shuffled_embed = normalize_to_fro_norm(flow_shuffled_embed, trained_fro_norm)

    random_source = rng.standard_normal(trained_embed.shape).astype(np.float32)
    random_source = normalize_to_fro_norm(random_source, trained_fro_norm)
    flow_random_embed = barycentric_target_embeddings(
        plan, random_source, gpt2_vocab_size, embed_dim,
    )
    flow_random_embed = normalize_to_fro_norm(flow_random_embed, trained_fro_norm)

    embed_map = {
        "flow_bridge_init_anchor": flow_bridge_embed,
        "flow_anchor_only": flow_bridge_embed,
        "flow_init_only": flow_bridge_embed,
        "char_overlap_no_ot": char_overlap_embed,
        "direct_string_match_anchor": string_match_embed,
        "flow_shuffled_qwen_rows": flow_shuffled_embed,
        "flow_random_source": flow_random_embed,
    }

    payload["preprocessing"]["preprocessing_time_s"] = time.time() - t_start
    save()

    # --- Step 5: Load GPT-2-tokenizer training data ---
    # NOTE: This experiment uses a GPT-2-tokenizer Qwen3-arch model
    # The data loading uses GPT-2 tokenizer
    print_flush("\n--- Loading GPT-2-tokenizer training data ---")
    train_ids, train_mask, train_meta = g167.load_c4_windows(
        tok_gpt2, split="train", seed=C4_TRAIN_SEED, n_windows=N_TRAIN_WINDOWS,
    )
    train_hashes = g167.collect_13gram_hashes(train_ids, train_mask)
    val_ids, val_mask, val_meta = g167.load_c4_windows(
        tok_gpt2, split="train", seed=C4_VAL_SEED, n_windows=N_C4_VAL_WINDOWS,
        forbidden_hashes=train_hashes,
    )
    print_flush(f"  Train: {train_ids.shape}, Val: {val_ids.shape}")

    # --- Step 6: Train cells ---
    print_flush("\n--- Training cells ---")
    actual_steps = train_steps

    for arm_label in arms_to_run:
        for seed in seeds:
            key = str(seed)
            if key in payload["results"].get(arm_label, {}) and not args.no_resume:
                print_flush(f"  Skipping {arm_label}/seed={seed} (already done)")
                continue

            print_flush(f"\n  === {arm_label} seed={seed} ===")

            anchor_targets = None
            custom_embed = None

            if arm_label == "scratch_ce":
                pass  # no anchor, no custom embed
            elif arm_label == "flow_bridge_init_anchor":
                emb = embed_map[arm_label]
                custom_embed = emb
                anchor_targets = build_anchor_targets(emb)
            elif arm_label == "flow_anchor_only":
                emb = embed_map[arm_label]
                custom_embed = None  # no init injection
                anchor_targets = build_anchor_targets(emb)
            elif arm_label == "flow_init_only":
                custom_embed = embed_map[arm_label]
                anchor_targets = None  # no anchor
            elif arm_label in embed_map:
                emb = embed_map[arm_label]
                custom_embed = emb
                anchor_targets = build_anchor_targets(emb)
            else:
                print_flush(f"  WARNING: unknown arm {arm_label}, skipping")
                continue

            result = train_cell(
                arm_label=arm_label,
                seed=seed,
                tok_gpt2=tok_gpt2,
                anchor_targets=anchor_targets,
                anchor_lambda=ANCHOR_LAMBDA if anchor_targets else 0.0,
                custom_embed=custom_embed,
                train_ids=train_ids,
                train_mask=train_mask,
                val_ids=val_ids,
                val_mask=val_mask,
                n_steps=actual_steps,
                eval_every=eval_every_steps,
            )
            payload["results"][arm_label][key] = result

            summary = compute_verdict(payload)
            payload["summary"] = summary
            payload["verdict"] = summary.get("verdict", "INCOMPLETE")
            save()

            print_flush(f"  {arm_label} seed={seed} final_nll={result['final_nll']:.4f} ({result['wallclock_s']:.0f}s)")

    summary = compute_verdict(payload)
    payload["summary"] = summary
    payload["verdict"] = summary.get("verdict", "INCOMPLETE")
    payload["status"] = "completed" if summary.get("stage_a_complete") else "running"
    save()

    print_flush(f"\n*** g188 VERDICT: {summary.get('verdict', '?')} ***")
    fg = summary.get("mean_flow_gap")
    if fg is not None:
        print_flush(f"  flow_gap={fg:+.4f} recovery={summary.get('recovery_fraction', 0):.1%}")
    print_flush(f"  stage_b_gate={'PASS' if summary.get('stage_b_gate') else 'FAIL/INCOMPLETE'}")


if __name__ == "__main__":
    main()
