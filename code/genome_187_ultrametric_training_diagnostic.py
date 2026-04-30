"""
genome_187_ultrametric_training_diagnostic.py

Measures ultrametric convergence of token embedding geometry across
Pythia training checkpoints. Novel gap: nobody has measured this on
text LLMs (only protein LMs in arXiv 2512.20926).

Primary metric: angular-distance triplet slack (not raw violation count).
Controls: random-init, Gaussian, row-norm-matched, spectral-matched.
Extra: cophenetic correlation, spectral alpha, PR, norm-freq correlation,
embed_in vs embed_out trajectories.

Codex design gate: codex_outputs/g187_ultrametric_design_gate_20260430.md
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

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent

OUT_PATH = ROOT / "results" / "genome_187_ultrametric_training_diagnostic.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
}

CHECKPOINTS = [
    "step0", "step1", "step2", "step4", "step8", "step16", "step32", "step64",
    "step128", "step256", "step512", "step1000", "step2000", "step4000",
    "step8000", "step16000", "step32000", "step64000", "step128000", "step143000",
]

TOKEN_SUBSET_SIZE = 10_000
CCC_SUBSET_SIZE = 3_000
N_TRIPLETS = 500_000
SMOKE_TRIPLETS = 10_000
SMOKE_CHECKPOINTS = ["step0", "step1000", "step64000", "step143000"]

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def print_flush(msg: str) -> None:
    print(msg, flush=True)


def load_token_subset(model_id: str, n: int = TOKEN_SUBSET_SIZE) -> np.ndarray:
    """Deterministic top-N tokens by rank (proxy for frequency)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    vocab_size = tok.vocab_size
    ids = np.arange(min(n, vocab_size))
    return ids


def load_pythia_embedding(
    model_id: str, revision: str, matrix: str = "embed_in",
) -> np.ndarray:
    """Load embedding matrix from a Pythia checkpoint."""
    from transformers import GPTNeoXForCausalLM
    model = GPTNeoXForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.float32,
    )
    assert not model.config.tie_word_embeddings, "Pythia should NOT tie embeddings"
    if matrix == "embed_in":
        w = model.gpt_neox.embed_in.weight.detach().cpu().numpy()
    elif matrix == "embed_out":
        w = model.embed_out.weight.detach().cpu().numpy()
    else:
        raise ValueError(f"Unknown matrix: {matrix}")
    del model
    gc.collect()
    return w


def normalize_embeddings(emb: np.ndarray) -> np.ndarray:
    """L2-normalize each row."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return emb / norms


def pairwise_angular_distance(emb_norm: np.ndarray) -> np.ndarray:
    """Angular distance matrix from L2-normalized embeddings."""
    cos_sim = emb_norm @ emb_norm.T
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return np.arccos(cos_sim) / np.pi


def sample_triplet_slack(
    dist: np.ndarray, n_triplets: int, seed: int = 42,
) -> dict[str, float]:
    """Compute ultrametric triplet slack statistics."""
    rng = np.random.RandomState(seed)
    n = dist.shape[0]
    idx = rng.randint(0, n, size=(n_triplets, 3))
    i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
    d_ij = dist[i, j]
    d_ik = dist[i, k]
    d_jk = dist[j, k]
    triplet_dists = np.stack([d_ij, d_ik, d_jk], axis=1)
    triplet_dists.sort(axis=1)
    d_min, d_mid, d_max = triplet_dists[:, 0], triplet_dists[:, 1], triplet_dists[:, 2]
    slack = (d_max - d_mid) / np.maximum(d_max, 1e-12)
    return {
        "mean_slack": float(np.mean(slack)),
        "median_slack": float(np.median(slack)),
        "p90_slack": float(np.percentile(slack, 90)),
        "violation_rate_tau_0.01": float(np.mean(slack < 0.01)),
        "violation_rate_tau_0.05": float(np.mean(slack < 0.05)),
        "n_triplets": int(n_triplets),
    }


def compute_cophenetic_ccc(
    dist: np.ndarray, subset_size: int = CCC_SUBSET_SIZE, seed: int = 42,
) -> dict[str, float]:
    """Cophenetic correlation coefficient on a subset."""
    from scipy.cluster.hierarchy import linkage, cophenet
    from scipy.spatial.distance import squareform

    n = dist.shape[0]
    if n > subset_size:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, subset_size, replace=False)
        sub_dist = dist[np.ix_(idx, idx)]
    else:
        sub_dist = dist

    condensed = squareform(sub_dist, checks=False)

    results = {}
    for method in ["average", "complete"]:
        Z = linkage(condensed, method=method)
        ccc, _ = cophenet(Z, condensed)
        results[f"ccc_{method}"] = float(ccc)
    return results


def compute_spectral_stats(emb: np.ndarray) -> dict[str, float]:
    """Spectral alpha, participation ratio, stable rank, top-PC variance."""
    cov = np.cov(emb.T)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    eigvals = np.maximum(eigvals, 0)
    total_var = eigvals.sum()
    if total_var < 1e-12:
        return {"spectral_alpha": 0, "participation_ratio": 0, "stable_rank": 0, "top_pc_var_frac": 0}

    pr = (eigvals.sum() ** 2) / (eigvals ** 2).sum()
    stable_rank = eigvals.sum() / eigvals[0] if eigvals[0] > 0 else 0
    top_pc_frac = eigvals[0] / total_var

    ranks = np.arange(1, len(eigvals) + 1)
    valid = eigvals > 1e-12
    if valid.sum() >= 5:
        log_r = np.log(ranks[valid])
        log_e = np.log(eigvals[valid])
        coeffs = np.polyfit(log_r, log_e, 1)
        alpha = -coeffs[0]
    else:
        alpha = 0.0

    return {
        "spectral_alpha": float(alpha),
        "participation_ratio": float(pr),
        "stable_rank": float(stable_rank),
        "top_pc_var_frac": float(top_pc_frac),
    }


def compute_norm_stats(emb: np.ndarray, token_ids: np.ndarray) -> dict[str, float]:
    """Row norm statistics and norm-frequency correlation."""
    norms = np.linalg.norm(emb, axis=1)
    log_rank = np.log(token_ids + 1)
    if len(norms) > 2:
        from scipy.stats import spearmanr
        corr, p = spearmanr(log_rank, norms)
    else:
        corr, p = 0.0, 1.0
    return {
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
        "norm_freq_spearman_r": float(corr),
        "norm_freq_spearman_p": float(p),
    }


def make_random_controls(
    emb: np.ndarray, seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate control embeddings matched to shape/statistics."""
    rng = np.random.RandomState(seed)
    n, d = emb.shape
    gaussian = rng.randn(n, d).astype(np.float32)

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norm_matched = rng.randn(n, d).astype(np.float32)
    norm_matched = norm_matched / np.maximum(np.linalg.norm(norm_matched, axis=1, keepdims=True), 1e-12) * norms

    U, s, Vt = np.linalg.svd(emb, full_matrices=False)
    Q, _ = np.linalg.qr(rng.randn(d, d).astype(np.float32))
    spec_matched = (rng.randn(n, len(s)).astype(np.float32) @ np.diag(s) @ Q[:len(s), :])

    return {
        "gaussian": gaussian,
        "norm_matched": norm_matched,
        "spectral_matched": spec_matched,
    }


def analyze_checkpoint(
    model_id: str, revision: str, matrix: str,
    token_ids: np.ndarray, n_triplets: int,
) -> dict[str, Any]:
    """Full analysis of one (model, checkpoint, matrix) cell."""
    t0 = time.time()

    emb_full = load_pythia_embedding(model_id, revision, matrix)
    emb = emb_full[token_ids]

    emb_norm = normalize_embeddings(emb)
    dist = pairwise_angular_distance(emb_norm)

    result: dict[str, Any] = {}
    result["triplet_slack"] = sample_triplet_slack(dist, n_triplets)
    result["cophenetic"] = compute_cophenetic_ccc(dist)
    result["spectral"] = compute_spectral_stats(emb)
    result["norms"] = compute_norm_stats(emb, token_ids)

    controls = make_random_controls(emb)
    result["controls"] = {}
    for ctrl_name, ctrl_emb in controls.items():
        ctrl_norm = normalize_embeddings(ctrl_emb)
        ctrl_dist = pairwise_angular_distance(ctrl_norm)
        result["controls"][ctrl_name] = sample_triplet_slack(ctrl_dist, n_triplets)
        del ctrl_dist, ctrl_norm
    del controls

    result["wallclock_s"] = time.time() - t0
    del dist, emb_norm, emb, emb_full
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model", default="pythia-160m", choices=list(MODELS.keys()))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    model_id = MODELS[args.model]
    model_key = args.model
    checkpoints = SMOKE_CHECKPOINTS if args.smoke else CHECKPOINTS
    n_triplets = SMOKE_TRIPLETS if args.smoke else N_TRIPLETS

    print_flush(f"genome_187 ultrametric training diagnostic")
    print_flush(f"  model={model_id}")
    print_flush(f"  checkpoints={len(checkpoints)}")
    print_flush(f"  n_triplets={n_triplets}")

    if not args.no_resume and OUT_PATH.exists():
        payload = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    else:
        payload = {
            "genome": "187",
            "name": "ultrametric_training_diagnostic",
            "timestamp_utc_started": now_utc(),
            "models": {},
        }

    payload.setdefault("models", {})

    token_ids = load_token_subset(model_id, TOKEN_SUBSET_SIZE)
    print_flush(f"  token_subset: {len(token_ids)} tokens")

    payload["models"].setdefault(model_key, {})
    model_results = payload["models"][model_key]

    for matrix in ["embed_in", "embed_out"]:
        model_results.setdefault(matrix, {})
        for ckpt in checkpoints:
            if ckpt in model_results[matrix]:
                cell = model_results[matrix][ckpt]
                if isinstance(cell, dict) and cell.get("triplet_slack"):
                    print_flush(f"  SKIP {model_key}/{matrix}/{ckpt} (done)")
                    continue

            print_flush(f"\n--- {model_key}/{matrix}/{ckpt} ---")
            try:
                result = analyze_checkpoint(
                    model_id, ckpt, matrix, token_ids, n_triplets,
                )
                model_results[matrix][ckpt] = result
                print_flush(
                    f"  mean_slack={result['triplet_slack']['mean_slack']:.4f}"
                    f"  ccc_avg={result['cophenetic'].get('ccc_average', 0):.4f}"
                    f"  wall={result['wallclock_s']:.0f}s"
                )
            except Exception as e:
                print_flush(f"  ERROR: {e}")
                model_results[matrix][ckpt] = {"error": str(e)}

            payload["timestamp_utc_last_write"] = now_utc()
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = OUT_PATH.with_suffix(OUT_PATH.suffix + ".tmp")
            tmp.write_text(
                json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8",
            )
            os.replace(tmp, OUT_PATH)

    print_flush("\nDone.")


if __name__ == "__main__":
    main()
