"""
genome_180_forecast.py

Genome Forecast / Diagnostic experiment.

Question
--------
At <=3% of training compute, do early tokenizer/embed and activation-geometry
features predict paired final C4 NLL gain better than early validation loss
alone?

Primary split
-------------
Train on Qwen-family / Qwen-tokenizer completed cells and hold out g173
Llama-family cells. RWKV/Falcon-H1 feature-only sanity rows can be added later,
but are not supervised labels in g180 v0.

Outputs
-------
  - results/genome_180_forecast.json
  - results/cache/genome_180_forecast_features/*.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

OUT_PATH = ROOT / "results" / "genome_180_forecast.json"
CACHE_DIR = ROOT / "results" / "cache" / "genome_180_forecast_features"

RESULT_JSON_PATHS = {
    "g165": ROOT / "results" / "genome_165_annealed_donor.json",
    "g167": ROOT / "results" / "genome_167_kd_canonical.json",
    "g172": ROOT / "results" / "genome_172_kd_warmup_cutoff.json",
    "g174": ROOT / "results" / "genome_174_donor_specificity_control.json",
    "g177": ROOT / "results" / "genome_177_matched_alt_donor.json",
    "g173": ROOT / "results" / "genome_173_cross_arch_flop_cashout.json",
    "g181a": ROOT / "results" / "genome_181a_tokenizer_isolation.json",
}

DEFAULT_SEEDS = [42, 7, 13]
PROBE_WINDOWS = 16
PROBE_SEQ_LEN = 256
FEATURE_MAX_POINTS = 1024
RSA_MAX_POINTS = 256
EMBED_MAX_ROWS = 4096
BOOTSTRAP_N = 10_000
PASS_MSE_REDUCTION = 0.25
ACTIONABLE_GAIN_NATS = 0.50
STOP_RECOMMEND_GAIN_THRESHOLD = 0.0
RANDOM_STATE = 180

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


@dataclass(frozen=True)
class CellSpec:
    source: str
    arm: str
    seed: int
    scratch_arm: str
    result_key: str
    result_section: tuple[str, ...]
    final_steps: int
    family: str
    split: str
    protocol: str
    replay: dict[str, Any] = field(default_factory=dict)

    @property
    def target_steps(self) -> int:
        return max(1, int(math.ceil(0.03 * float(self.final_steps))))

    @property
    def cell_id(self) -> str:
        raw = f"{self.source}:{'/'.join(self.result_section)}:{self.arm}:{self.seed}"
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
        return safe.strip("_")


@dataclass
class EarlyCheckpoint:
    model: Any
    early_loss: float
    target_steps: int
    metadata: dict[str, Any]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=True), encoding="utf-8")
    os.replace(tmp, path)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_results_jsons(paths: Mapping[str, Path] = RESULT_JSON_PATHS) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, path in paths.items():
        if path.exists():
            out[key] = read_json(path)
    missing = sorted(set(paths) - set(out))
    if missing:
        print(f"  missing result JSONs skipped: {', '.join(missing)}")
    return out


def cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def autocast_context():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    except Exception:
        pass
    return nullcontext()


def device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def set_seed(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_cell_spec(arm_spec: CellSpec | Mapping[str, Any]) -> CellSpec:
    if isinstance(arm_spec, CellSpec):
        return arm_spec
    data = dict(arm_spec)
    if "result_section" in data and not isinstance(data["result_section"], tuple):
        data["result_section"] = tuple(data["result_section"])
    return CellSpec(**data)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _to_numpy(x: Any) -> np.ndarray:
    # Handle torch tensors (incl. CUDA tensors) via duck-typing first to avoid
    # falling through to np.asarray on a device tensor.
    cpu_fn = getattr(x, "cpu", None)
    if callable(cpu_fn):
        detached = x.detach() if hasattr(x, "detach") else x
        return detached.float().cpu().numpy()
    return np.asarray(x)


def _numeric_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    if not math.isfinite(out):
        return float("nan")
    return out


def _sample_rows(X: np.ndarray, max_rows: int, seed: int = RANDOM_STATE) -> np.ndarray:
    if X.shape[0] <= max_rows:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_rows, replace=False)
    idx.sort()
    return X[idx]


def _hidden_cloud(hidden: Any, attention_mask: Any | None, max_points: int = FEATURE_MAX_POINTS) -> np.ndarray:
    H = _to_numpy(hidden).astype(np.float64, copy=False)
    if H.ndim == 3:
        if attention_mask is not None:
            mask = _to_numpy(attention_mask).astype(bool).reshape(-1)
            H = H.reshape(-1, H.shape[-1])[mask]
        else:
            H = H.reshape(-1, H.shape[-1])
    if H.ndim != 2:
        raise ValueError(f"hidden cloud must be 2-D or 3-D; got shape {H.shape}")
    H = H[np.isfinite(H).all(axis=1)]
    return _sample_rows(H, max_points)


def _input_batch_for_model(probe_batch: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    dev = device()
    out: dict[str, Any] = {}
    for key in ("input_ids", "attention_mask", "labels"):
        if key in probe_batch and probe_batch[key] is not None:
            value = probe_batch[key]
            if isinstance(value, torch.Tensor):
                out[key] = value.to(dev)
            else:
                out[key] = torch.as_tensor(value, device=dev)
    if "labels" not in out and "input_ids" in out:
        out["labels"] = out["input_ids"]
    return out


def _select_hidden_indices(n_hidden_states: int, layer_indices: Sequence[int]) -> list[int]:
    if layer_indices:
        raw = list(layer_indices)
    else:
        raw = [1, max(1, n_hidden_states // 2), n_hidden_states - 1]
    out: list[int] = []
    for idx in raw:
        if idx < 0:
            idx = n_hidden_states + idx
        idx = max(0, min(n_hidden_states - 1, int(idx)))
        if idx not in out:
            out.append(idx)
    if not out:
        out = [n_hidden_states - 1]
    return out


def _model_hidden_states(model: Any, probe_batch: Mapping[str, Any]) -> tuple[tuple[Any, ...], Any | None, float]:
    import torch

    batch = _input_batch_for_model(probe_batch)
    attention_mask = batch.get("attention_mask")
    labels = batch.pop("labels", None)
    was_training = bool(getattr(model, "training", False))
    model.eval()
    with torch.no_grad():
        with autocast_context():
            out = model(
                **batch,
                labels=labels,
                output_hidden_states=True,
                use_cache=False,
            )
    if was_training:
        model.train()
    early_loss = float(out.loss.detach().float().cpu().item()) if getattr(out, "loss", None) is not None else float("nan")
    hidden_states = tuple(out.hidden_states)
    return hidden_states, attention_mask, early_loss


def _spectral_features(X: np.ndarray) -> dict[str, float]:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or min(X.shape) < 3:
        return {"alpha": float("nan"), "participation_ratio": float("nan"), "sqrt_pr_alpha": float("nan")}
    X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    s = s[np.isfinite(s) & (s > 1e-12)]
    if s.size < 3:
        return {"alpha": float("nan"), "participation_ratio": float("nan"), "sqrt_pr_alpha": float("nan")}
    lam = (s ** 2) / max(X.shape[0] - 1, 1)
    denom = float(np.sum(lam ** 2))
    pr = float((np.sum(lam) ** 2) / denom) if denom > 0.0 else float("nan")
    tail_start = min(5, max(0, s.size // 5))
    tail_stop = min(s.size, max(tail_start + 3, 64))
    ranks = np.arange(tail_start + 1, tail_stop + 1, dtype=np.float64)
    y = np.log(s[tail_start:tail_stop])
    if y.size >= 3 and np.isfinite(y).all():
        slope = float(np.polyfit(np.log(ranks), y, deg=1)[0])
        alpha = -slope
    else:
        alpha = float("nan")
    return {
        "alpha": float(alpha),
        "participation_ratio": float(pr),
        "sqrt_pr_alpha": float(math.sqrt(max(pr, 0.0)) * alpha) if math.isfinite(pr) and math.isfinite(alpha) else float("nan"),
    }


def _twonn_id(X: np.ndarray) -> float:
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 4:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=3, n_jobs=1).fit(X)
    dists, _ = nn.kneighbors(X)
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    keep = r1 > 1e-12
    log_mu = np.log(r2[keep]) - np.log(r1[keep])
    log_mu = log_mu[np.isfinite(log_mu) & (log_mu > 0)]
    if log_mu.size < 3:
        return float("nan")
    return float(log_mu.size / np.sum(log_mu))


def _knn_clustering(X: np.ndarray, k: int = 10) -> float:
    try:
        from genome_primitives import knn_clustering_coefficient

        return float(knn_clustering_coefficient(X, k=k).value)
    except Exception:
        pass

    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if X.ndim != 2 or n < k + 2:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=1).fit(X)
    _, idxs = nn.kneighbors(X)
    neigh = idxs[:, 1:]
    adj = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), k)
    adj[rows, neigh.reshape(-1)] = True
    denom = (k * (k - 1)) / 2.0
    vals = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ns = neigh[i]
        sub = adj[np.ix_(ns, ns)] | adj[np.ix_(ns, ns)].T
        np.fill_diagonal(sub, False)
        vals[i] = (sub.sum() / 2.0) / denom
    return float(vals.mean())


def _pca_scores(X: np.ndarray, n_components: int = 64) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"PCA input must be 2-D; got {X.shape}")
    X = X - X.mean(axis=0, keepdims=True)
    k = min(n_components, X.shape[0] - 1, X.shape[1])
    if k < 1:
        return np.zeros((X.shape[0], 1), dtype=np.float64)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    scores = U[:, :k] * S[:k]
    if k < n_components:
        pad = np.zeros((scores.shape[0], n_components - k), dtype=scores.dtype)
        scores = np.concatenate([scores, pad], axis=1)
    return scores


def _procrustes_residual(A: np.ndarray, B: np.ndarray) -> float:
    n = min(A.shape[0], B.shape[0])
    d = min(A.shape[1], B.shape[1])
    if n < 3 or d < 1:
        return float("nan")
    A = A[:n, :d] - A[:n, :d].mean(axis=0, keepdims=True)
    B = B[:n, :d] - B[:n, :d].mean(axis=0, keepdims=True)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return float("nan")
    A = A / norm_a
    B = B / norm_b
    U, _, Vt = np.linalg.svd(A.T @ B, full_matrices=False)
    R = U @ Vt
    return float(np.linalg.norm(A @ R - B))


def _pairwise_distance_vector(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    gram = X @ X.T
    sq = np.diag(gram)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0)
    iu = np.triu_indices(X.shape[0], k=1)
    return np.sqrt(D2[iu])


def _rsa_distance(A: np.ndarray, B: np.ndarray) -> float:
    n = min(A.shape[0], B.shape[0], RSA_MAX_POINTS)
    if n < 4:
        return float("nan")
    A = A[:n] - A[:n].mean(axis=0, keepdims=True)
    B = B[:n] - B[:n].mean(axis=0, keepdims=True)
    da = _pairwise_distance_vector(A)
    db = _pairwise_distance_vector(B)
    if np.std(da) <= 1e-12 or np.std(db) <= 1e-12:
        return float("nan")
    corr = float(np.corrcoef(da, db)[0, 1])
    return float(1.0 - corr)


def _reference_array(probe_batch: Mapping[str, Any], key: str) -> np.ndarray | None:
    value = probe_batch.get(key)
    if value is None:
        return None
    if isinstance(value, dict):
        if "mid" in value:
            value = value["mid"]
        elif value:
            value = next(iter(value.values()))
        else:
            return None
    arr = _to_numpy(value)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2:
        return None
    arr = arr[np.isfinite(arr).all(axis=1)]
    return arr


def _procrustes_rsa_features(X: np.ndarray, ref: np.ndarray | None, prefix: str) -> dict[str, float]:
    if ref is None or X.shape[0] < 4 or ref.shape[0] < 4:
        return {
            f"{prefix}_pca64_procrustes_residual": float("nan"),
            f"{prefix}_pca64_rsa_distance": float("nan"),
        }
    n = min(X.shape[0], ref.shape[0], FEATURE_MAX_POINTS)
    A = _pca_scores(X[:n], n_components=64)
    B = _pca_scores(ref[:n], n_components=64)
    return {
        f"{prefix}_pca64_procrustes_residual": _procrustes_residual(A, B),
        f"{prefix}_pca64_rsa_distance": _rsa_distance(A, B),
    }


def _embedding_weight(model: Any) -> np.ndarray | None:
    try:
        emb = model.get_input_embeddings()
        if emb is None or not hasattr(emb, "weight"):
            return None
        W = _to_numpy(emb.weight)
        if W.ndim != 2:
            return None
        return W
    except Exception:
        return None


def _lm_head_weight(model: Any) -> np.ndarray | None:
    try:
        head = getattr(model, "lm_head", None)
        if head is None or not hasattr(head, "weight"):
            return None
        W = _to_numpy(head.weight)
        if W.ndim != 2:
            return None
        return W
    except Exception:
        return None


def _embedding_reference_features(model: Any, probe_batch: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    current = _embedding_weight(model)
    ref = _reference_array(probe_batch, "reference_embedding")
    if current is not None and ref is not None:
        n = min(current.shape[0], ref.shape[0], EMBED_MAX_ROWS)
        out.update(_procrustes_rsa_features(current[:n], ref[:n], "embed_to_qwen_ref"))
        out["embed_reference_rows_used"] = float(n)
    else:
        out.update(_procrustes_rsa_features(np.zeros((0, 1)), None, "embed_to_qwen_ref"))
        out["embed_reference_rows_used"] = float("nan")

    current_head = _lm_head_weight(model)
    ref_head = _reference_array(probe_batch, "reference_lm_head")
    if current_head is not None and ref_head is not None:
        n = min(current_head.shape[0], ref_head.shape[0], EMBED_MAX_ROWS)
        out.update(_procrustes_rsa_features(current_head[:n], ref_head[:n], "lm_head_to_qwen_ref"))
        out["lm_head_reference_rows_used"] = float(n)
    else:
        out.update(_procrustes_rsa_features(np.zeros((0, 1)), None, "lm_head_to_qwen_ref"))
        out["lm_head_reference_rows_used"] = float("nan")
    return out


def _gradient_noise_scale(model: Any, probe_batch: Mapping[str, Any], n_microbatches: int = 4) -> dict[str, float]:
    import torch

    if "input_ids" not in probe_batch:
        return {"gradient_noise_scale": float("nan"), "grad_norm_mean": float("nan"), "grad_norm_var": float("nan")}
    ids = probe_batch["input_ids"]
    if not isinstance(ids, torch.Tensor):
        ids = torch.as_tensor(ids)
    batch_n = int(ids.shape[0])
    if batch_n < 1:
        return {"gradient_noise_scale": float("nan"), "grad_norm_mean": float("nan"), "grad_norm_var": float("nan")}
    chunks = np.array_split(np.arange(batch_n), min(n_microbatches, batch_n))
    was_training = bool(getattr(model, "training", False))
    model.train()
    norms: list[float] = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        mb: dict[str, Any] = {}
        for key in ("input_ids", "attention_mask"):
            if key in probe_batch and probe_batch[key] is not None:
                value = probe_batch[key]
                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor(value)
                mb[key] = value[chunk].to(device())
        labels = mb["input_ids"]
        model.zero_grad(set_to_none=True)
        with autocast_context():
            out = model(**mb, labels=labels, use_cache=False)
        loss = out.loss
        if not torch.isfinite(loss):
            continue
        loss.backward()
        total = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total += float(param.grad.detach().float().pow(2).sum().item())
        norms.append(math.sqrt(max(total, 0.0)))
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    if len(norms) < 2:
        return {"gradient_noise_scale": float("nan"), "grad_norm_mean": float("nan"), "grad_norm_var": float("nan")}
    arr = np.asarray(norms, dtype=np.float64)
    mean = float(arr.mean())
    var = float(arr.var(ddof=1))
    return {
        "gradient_noise_scale": float(var / max(mean * mean, 1e-12)),
        "grad_norm_mean": mean,
        "grad_norm_var": var,
    }


def _last_block_lm_head_params(model: Any) -> list[Any]:
    layer_idxs: list[int] = []
    for name, _ in model.named_parameters():
        for pat in (r"model\.layers\.(\d+)\.", r"layers\.(\d+)\.", r"transformer\.h\.(\d+)\."):
            match = re.search(pat, name)
            if match:
                layer_idxs.append(int(match.group(1)))
                break
    last_idx = max(layer_idxs) if layer_idxs else None
    params: list[Any] = []
    for name, param in model.named_parameters():
        if not getattr(param, "requires_grad", False):
            continue
        include = "lm_head" in name
        if last_idx is not None:
            include = include or f"layers.{last_idx}." in name or f"model.layers.{last_idx}." in name
        if include:
            params.append(param)
    if not params:
        params = [p for p in model.parameters() if getattr(p, "requires_grad", False)][-4:]
    return params


def _normalize_vectors(vectors: Sequence[Any]) -> list[Any]:
    import torch

    total = torch.zeros((), device=device(), dtype=torch.float32)
    for v in vectors:
        total = total + v.detach().float().pow(2).sum()
    scale = torch.sqrt(total).clamp_min(1e-12)
    return [v / scale.to(v.device, dtype=v.dtype) for v in vectors]


def _curvature_top_eigen_proxy(model: Any, probe_batch: Mapping[str, Any], n_iter: int = 4, time_limit_s: float = 30.0) -> float:
    import torch

    params = _last_block_lm_head_params(model)
    if not params:
        return float("nan")
    batch = _input_batch_for_model(probe_batch)
    labels = batch.pop("labels", batch.get("input_ids"))
    was_training = bool(getattr(model, "training", False))
    model.eval()
    rng = torch.Generator(device=device())
    rng.manual_seed(RANDOM_STATE)
    vectors = [torch.randn(p.shape, generator=rng, device=p.device, dtype=p.dtype) for p in params]
    vectors = _normalize_vectors(vectors)
    eig = float("nan")
    t0 = time.time()
    try:
        for _ in range(n_iter):
            if time.time() - t0 > time_limit_s:
                break
            model.zero_grad(set_to_none=True)
            with autocast_context():
                out = model(**batch, labels=labels, use_cache=False)
            loss = out.loss
            grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
            dot = torch.zeros((), device=device(), dtype=torch.float32)
            for grad, vec in zip(grads, vectors):
                if grad is not None:
                    dot = dot + (grad.float() * vec.float()).sum()
            hvps = torch.autograd.grad(dot, params, allow_unused=True)
            dense_hvps = [torch.zeros_like(p) if h is None else h.detach() for h, p in zip(hvps, params)]
            eig_tensor = torch.zeros((), device=device(), dtype=torch.float32)
            for vec, hvp in zip(vectors, dense_hvps):
                eig_tensor = eig_tensor + (vec.float() * hvp.float()).sum()
            eig = float(eig_tensor.detach().cpu().item())
            vectors = _normalize_vectors(dense_hvps)
    except Exception:
        eig = float("nan")
    finally:
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()
    return eig


def _norm_variance_depth_ratios(
    hidden_by_idx: Mapping[int, np.ndarray],
    model: Any,
) -> dict[str, float]:
    if not hidden_by_idx:
        return {
            "hidden_norm_early_late_ratio": float("nan"),
            "hidden_var_early_late_ratio": float("nan"),
            "norm_param_early_late_ratio": float("nan"),
        }
    ordered = sorted(hidden_by_idx)
    early = hidden_by_idx[ordered[0]]
    late = hidden_by_idx[ordered[-1]]
    early_norm = float(np.mean(np.linalg.norm(early, axis=1)))
    late_norm = float(np.mean(np.linalg.norm(late, axis=1)))
    early_var = float(np.var(early))
    late_var = float(np.var(late))

    layer_norms: dict[int, list[float]] = {}
    for name, param in model.named_parameters():
        if param.ndim != 1 or "norm" not in name.lower():
            continue
        match = re.search(r"layers\.(\d+)\.", name)
        if not match:
            continue
        idx = int(match.group(1))
        layer_norms.setdefault(idx, []).append(float(np.linalg.norm(_to_numpy(param))))
    if layer_norms:
        first_key = min(layer_norms)
        last_key = max(layer_norms)
        first = float(np.median(layer_norms[first_key]))
        last = float(np.median(layer_norms[last_key]))
        norm_ratio = last / max(first, 1e-12)
    else:
        norm_ratio = float("nan")
    return {
        "hidden_norm_early_late_ratio": late_norm / max(early_norm, 1e-12),
        "hidden_var_early_late_ratio": late_var / max(early_var, 1e-12),
        "norm_param_early_late_ratio": norm_ratio,
    }


def extract_features(model, probe_batch, layer_indices) -> dict[str, float]:
    """Compute the eight cycle-66 feature groups for one early checkpoint.

    Required scalar outputs are intentionally simple and Ridge-friendly. The
    Qwen reference arrays are supplied through probe_batch by main(), which
    keeps this function's public signature stable.
    """

    hidden_states, attention_mask, observed_loss = _model_hidden_states(model, probe_batch)
    chosen = _select_hidden_indices(len(hidden_states), layer_indices)
    mid_idx = chosen[len(chosen) // 2]

    hidden_by_idx: dict[int, np.ndarray] = {}
    spectral_by_idx: dict[int, dict[str, float]] = {}
    for idx in chosen:
        X = _hidden_cloud(hidden_states[idx], attention_mask, max_points=FEATURE_MAX_POINTS)
        hidden_by_idx[idx] = X
        spectral_by_idx[idx] = _spectral_features(X)

    mid = hidden_by_idx[mid_idx]
    mid_spec = spectral_by_idx[mid_idx]
    xs = np.linspace(0.0, 1.0, num=len(chosen), dtype=np.float64)

    def _slope(key: str) -> float:
        ys = np.asarray([spectral_by_idx[idx][key] for idx in chosen], dtype=np.float64)
        keep = np.isfinite(ys)
        if keep.sum() < 2:
            return float("nan")
        return float(np.polyfit(xs[keep], ys[keep], deg=1)[0])

    features: dict[str, float] = {
        "early_loss": _numeric_or_nan(probe_batch.get("early_loss", observed_loss)),
        "mid_spectral_alpha": mid_spec["alpha"],
        "mid_participation_ratio": mid_spec["participation_ratio"],
        "mid_sqrt_pr_alpha": mid_spec["sqrt_pr_alpha"],
        "depth_alpha_drift": _slope("alpha"),
        "depth_pr_drift": _slope("participation_ratio"),
        "depth_sqrt_pr_alpha_drift": _slope("sqrt_pr_alpha"),
        "twonn_intrinsic_dim": _twonn_id(mid),
        "knn10_clustering_coeff": _knn_clustering(mid, k=10),
    }
    features.update(_procrustes_rsa_features(mid, _reference_array(probe_batch, "reference_hidden"), "hidden_to_qwen_ref"))
    features.update(_embedding_reference_features(model, probe_batch))
    features.update(_gradient_noise_scale(model, probe_batch, n_microbatches=4))
    features["curvature_top_eigen_proxy"] = _curvature_top_eigen_proxy(model, probe_batch, n_iter=4, time_limit_s=30.0)
    features.update(_norm_variance_depth_ratios(hidden_by_idx, model))
    return {key: _numeric_or_nan(value) for key, value in features.items()}


# ---------------------------------------------------------------------------
# Labels and known-cell inventory
# ---------------------------------------------------------------------------


def _section(root: Mapping[str, Any], path: Sequence[str]) -> Any:
    cur: Any = root
    for part in path:
        cur = cur[part]
    return cur


def _cell_payload(results_jsons: Mapping[str, Any], cell: CellSpec | Mapping[str, Any], arm: str | None = None) -> Any:
    spec = _as_cell_spec(cell)
    root = results_jsons[spec.source]
    section = _section(root, spec.result_section)
    return section[arm or spec.arm][str(spec.seed)]


def _final_c4_nll_from_payload(payload: Any) -> float:
    if isinstance(payload, list):
        if not payload:
            raise ValueError("empty trajectory")
        return float(payload[-1]["nll"])
    if isinstance(payload, dict):
        if "final_metrics" in payload:
            return float(payload["final_metrics"]["c4_val"]["nll"])
        if "final_nll" in payload:
            return float(payload["final_nll"])
        if "trajectory" in payload and payload["trajectory"]:
            rows = [row for row in payload["trajectory"] if "nll" in row]
            if rows:
                return float(rows[-1]["nll"])
        if "nll" in payload:
            return float(payload["nll"])
    raise KeyError(f"could not extract final C4 NLL from payload keys={list(payload) if isinstance(payload, dict) else type(payload)}")


def compute_label(arm, seed, results_jsons) -> float:
    """Return paired final C4 NLL gain vs same-seed scratch baseline.

    `arm` can be a CellSpec or a mapping with the same fields. `seed` is kept
    as an explicit argument for the required API and must match the cell seed.
    """

    spec = _as_cell_spec(arm)
    if int(seed) != int(spec.seed):
        spec = CellSpec(**{**asdict(spec), "seed": int(seed)})
    scratch_nll = _final_c4_nll_from_payload(_cell_payload(results_jsons, spec, spec.scratch_arm))
    arm_nll = _final_c4_nll_from_payload(_cell_payload(results_jsons, spec, spec.arm))
    return float(scratch_nll - arm_nll)


def _early_loss_from_existing(results_jsons: Mapping[str, Any], spec: CellSpec) -> float:
    try:
        payload = _cell_payload(results_jsons, spec)
    except Exception:
        return float("nan")
    rows: list[Mapping[str, Any]] = []
    if isinstance(payload, list):
        rows = [row for row in payload if isinstance(row, dict)]
    elif isinstance(payload, dict):
        if "trajectory" in payload:
            rows = [row for row in payload["trajectory"] if isinstance(row, dict)]
        elif "train_log" in payload:
            rows = [row for row in payload["train_log"] if isinstance(row, dict)]
    if not rows:
        return float("nan")
    target = spec.target_steps
    candidate = min(rows, key=lambda row: abs(int(row.get("step", target)) - target))
    for key in ("nll", "ce_loss", "total_loss"):
        if key in candidate and candidate[key] is not None:
            return float(candidate[key])
    return float("nan")


def _result_arms(payload: Mapping[str, Any], path: Sequence[str]) -> list[str]:
    section = _section(payload, path)
    return list(section.keys())


def build_known_cells(results_jsons: Mapping[str, dict[str, Any]]) -> list[CellSpec]:
    cells: list[CellSpec] = []

    def add_cells(
        source: str,
        result_section: tuple[str, ...],
        scratch_arm: str,
        final_steps: int,
        family: str,
        split: str,
        protocol: str,
        replay_defaults: Mapping[str, Any] | None = None,
    ) -> None:
        if source not in results_jsons:
            return
        arms = _result_arms(results_jsons[source], result_section)
        seeds = sorted({int(seed) for arm in arms for seed in _section(results_jsons[source], result_section)[arm].keys()})
        for arm in arms:
            for seed in seeds:
                if str(seed) not in _section(results_jsons[source], result_section).get(arm, {}):
                    continue
                replay = dict(replay_defaults or {})
                replay.setdefault("arm_label", arm)
                cells.append(
                    CellSpec(
                        source=source,
                        arm=arm,
                        seed=seed,
                        scratch_arm=scratch_arm,
                        result_key=source,
                        result_section=result_section,
                        final_steps=final_steps,
                        family=family,
                        split=split,
                        protocol=protocol,
                        replay=replay,
                    )
                )

    add_cells("g165", ("results",), "scratch_baseline", 500, "qwen_anchor", "train", "qwen_anchor")
    add_cells("g167", ("results",), "scratch_ce", 6000, "qwen_vocab_minimal_llama", "train", "minimal_kd")
    add_cells("g172", ("results",), "scratch_ce", 6000, "qwen_vocab_minimal_llama", "train", "minimal_kd_schedule")
    add_cells("g174", ("part_a", "results"), "scratch_baseline", 500, "qwen_anchor", "train", "qwen_anchor_nulls")
    add_cells("g174", ("part_b", "results"), "scratch_ce", 6000, "qwen_vocab_minimal_llama", "train", "minimal_kd_nulls")
    # Filter out g174 PART B random_teacher cells: replay path triggers CUDA
    # scatter-gather assert (random teacher topk indices land in padded vocab
    # range, OOB on student lm_head). These cells are null controls; their
    # forecast labels are not load-bearing. Drop to avoid context corruption.
    cells = [c for c in cells if "kd_random_teacher" not in c.arm]
    add_cells("g177", ("results",), "scratch_baseline", 500, "qwen_anchor", "train", "qwen_anchor_alt_donor")
    add_cells("g181a", ("results",), "scratch_ce", 2000, "qwen_anchor", "train", "tokenizer_isolation")

    if "g173" in results_jsons:
        arms = _result_arms(results_jsons["g173"], ("results",))
        for arm in arms:
            if arm.endswith("_llama"):
                scratch = "scratch_ce_llama"
                family = "llama_family"
                split = "test"
            elif arm.endswith("_qwen_arch"):
                scratch = "scratch_ce_qwen_arch"
                family = "qwen_arch_control"
                split = "train"
            else:
                continue
            for seed_text in results_jsons["g173"]["results"][arm].keys():
                seed = int(seed_text)
                cells.append(
                    CellSpec(
                        source="g173",
                        arm=arm,
                        seed=seed,
                        scratch_arm=scratch,
                        result_key="g173",
                        result_section=("results",),
                        final_steps=3600,
                        family=family,
                        split=split,
                        protocol="cross_arch_kd",
                        replay={"arm_label": arm},
                    )
                )
    return cells


# ---------------------------------------------------------------------------
# Early replay
# ---------------------------------------------------------------------------


def _cache_path_for_cell(spec: CellSpec) -> Path:
    digest = hashlib.sha1(spec.cell_id.encode("utf-8")).hexdigest()[:10]
    return CACHE_DIR / f"{spec.cell_id}_{digest}.json"


def _anchor_subset_for_cell(spec: CellSpec) -> str:
    if spec.source == "g181a":
        if spec.arm == "full_anchor":
            return "all"
        if spec.arm == "embed_lm_head_only_anchor":
            return "embed_lm_head"
        if spec.arm == "no_embed_lm_head_anchor":
            return "no_embed_lm_head"
        return "none"
    if "attn_only" in spec.arm:
        return "attn"
    return "all" if "anchor" in spec.arm else "none"


def _anchor_lambda_for_cell(spec: CellSpec, results_jsons: Mapping[str, Any] | None = None) -> float:
    if "scratch" in spec.arm or spec.arm == "scratch_ce":
        return 0.0
    if results_jsons is None:
        lazy: dict[str, Any] = {}
        for key in ("g177", "g181a"):
            path = RESULT_JSON_PATHS.get(key)
            if path is not None and path.exists():
                try:
                    lazy[key] = read_json(path)
                except Exception:
                    pass
        results_jsons = lazy
    if spec.source == "g181a" and results_jsons and "g181a" in results_jsons:
        diag = results_jsons["g181a"].get("anchor_diagnostics", {}).get("lambda_and_frobenius", {})
        by_seed = diag.get("by_seed", {}).get(str(spec.seed), {})
        if spec.arm == "embed_lm_head_only_anchor":
            return float(by_seed.get("lambda_embed_lm_head_only_anchor", 0.0))
        if spec.arm == "no_embed_lm_head_anchor":
            return float(by_seed.get("lambda_no_embed_lm_head_anchor", 0.0))
        if spec.arm == "full_anchor":
            return 0.01
    if spec.source == "g177" and results_jsons and "g177" in results_jsons:
        diag = results_jsons["g177"].get("anchor_diagnostics", {}).get(spec.arm, {})
        by_seed = diag.get("by_seed", {}).get(str(spec.seed), {})
        if "actual_lambda_0" in by_seed:
            return float(by_seed["actual_lambda_0"])
    if spec.source == "g165":
        match = re.search(r"lam([0-9.eE+-]+)", spec.arm)
        if match:
            return float(match.group(1))
        if "attn_only" in spec.arm:
            return 1.3e-3
    if spec.source in {"g174", "g177"}:
        return 0.01
    return 0.0


def _load_anchor_params_for_cell(spec: CellSpec):
    import genome_165_annealed_donor as g165

    if "scratch" in spec.arm or spec.arm == "scratch_ce":
        return None
    if spec.source == "g174":
        import genome_174_donor_specificity_control as g174

        if spec.arm == "anchor_random_donor":
            return g174.load_random_anchor_params(g174.RANDOM_DONOR_SEED)
        if spec.arm == "anchor_permuted_donor":
            trained, _ = g174.load_trained_anchor_params()
            params, _ = g174.build_permuted_anchor_params(trained, seed=g174.PERMUTED_DONOR_SEED)
            return params
    if spec.source == "g177" and spec.arm.startswith("anchor_alt_donor_seed_"):
        import genome_177_matched_alt_donor as g177

        donor_seed = int(spec.arm.rsplit("_", 1)[-1])
        params, _ = g177.load_state_npz(g177.alt_npz_path(donor_seed))
        return params
    donor, _ = g165.load_trained_donor()
    try:
        if hasattr(donor.config, "use_cache"):
            donor.config.use_cache = False
        return g165.snapshot_donor_params(donor)
    finally:
        del donor
        cleanup_cuda()


def _build_anchor_pairs(model: Any, params: Mapping[str, Any], subset: str) -> list[tuple[str, Any, Any]]:
    import torch

    embed_names = {"model.embed_tokens.weight", "lm_head.weight", "model.lm_head.weight"}
    pairs = []
    for name, param in model.named_parameters():
        if name not in params:
            continue
        target = params[name]
        if tuple(param.shape) != tuple(target.shape):
            continue
        if subset == "attn" and ".self_attn." not in name:
            continue
        if subset == "embed_lm_head" and name not in embed_names:
            continue
        if subset == "no_embed_lm_head" and name in embed_names:
            continue
        if subset == "none":
            continue
        if not isinstance(target, torch.Tensor):
            target = torch.as_tensor(target)
        pairs.append((name, param, target.to(param.device, dtype=torch.float32)))
    return pairs


def _lambda_schedule_from_arm(arm_label: str, lam0: float, step: int) -> float:
    if "hardcut" in arm_label or "hard_cut_step1" in arm_label:
        return lam0 if step == 1 else 0.0
    if "step" in arm_label and "hardcut" not in arm_label:
        return lam0 if step < 25 else 0.0
    if "linear" in arm_label:
        return lam0 * max(0.0, 1.0 - step / 50.0)
    if "exponential" in arm_label:
        return lam0 * math.exp(-step / 10.0)
    return lam0


def _load_anchor_train_data():
    import genome_165_annealed_donor as g165
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(g165._MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = g165.load_c4_texts(g165.C4_TRAIN_SEED, max(32 * g165.SEQ_LEN, PROBE_WINDOWS * PROBE_SEQ_LEN))
    ids, mask = g165.tokenize_block(tok, texts, g165.SEQ_LEN)
    return tok, ids, mask


def _replay_anchor_cell(spec: CellSpec, target_steps: int, results_jsons: Mapping[str, Any] | None) -> EarlyCheckpoint:
    import torch
    import genome_165_annealed_donor as g165

    tok, train_ids, train_mask = _load_anchor_train_data()
    set_seed(spec.seed)
    model = g165.load_random_init(spec.seed)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=g165.LR, betas=(0.9, 0.95))
    rng = np.random.default_rng(spec.seed)
    schedule = rng.integers(0, int(train_ids.shape[0]), size=(target_steps, g165.BATCH_SIZE), dtype=np.int64)
    lam0 = _anchor_lambda_for_cell(spec, results_jsons)
    subset = _anchor_subset_for_cell(spec)
    anchor_pairs: list[tuple[str, Any, Any]] = []
    if lam0 > 0.0 and subset != "none":
        params = _load_anchor_params_for_cell(spec)
        if params is None:
            raise RuntimeError(f"{spec.cell_id}: anchor params missing")
        anchor_pairs = _build_anchor_pairs(model, params, subset)

    early_loss = float("nan")
    model.train()
    for step in range(1, target_steps + 1):
        idx = schedule[step - 1]
        ids = train_ids[idx].to(device())
        mask = train_mask[idx].to(device())
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            out = model(input_ids=ids, attention_mask=mask, labels=ids, use_cache=False)
        loss = out.loss
        loss.backward()
        lam_t = _lambda_schedule_from_arm(spec.arm, lam0, step)
        if anchor_pairs and lam_t > 0.0:
            with torch.no_grad():
                coeff = 2.0 * lam_t
                for _, param, target in anchor_pairs:
                    if param.grad is not None:
                        param.grad.add_(coeff * (param.detach().float() - target).to(param.grad.dtype))
        optimizer.step()
        early_loss = float(loss.detach().float().cpu().item())
    return EarlyCheckpoint(
        model=model,
        early_loss=early_loss,
        target_steps=target_steps,
        metadata={"tokenizer": "qwen3", "replay_kind": "anchor", "anchor_subset": subset, "lambda_0": lam0},
    )


def _load_minimal_kd_data():
    import genome_167_kd_canonical as g167

    tok = g167.load_tokenizer()
    train_ids, train_mask, _ = g167.load_c4_windows(
        tok,
        split="train",
        seed=g167.C4_TRAIN_SEED,
        n_windows=max(512, PROBE_WINDOWS),
    )
    return tok, train_ids, train_mask


def _kd_active_for_cell(spec: CellSpec, step: int) -> bool:
    if spec.arm in {"scratch_ce", "scratch_ce_llama", "scratch_ce_qwen_arch", "scratch_baseline"}:
        return False
    if "late_only" in spec.arm:
        return step >= max(1, int(0.67 * spec.final_steps))
    if "warmup_then_ce_cutoff" in spec.arm:
        return step <= max(1, int(0.03 * spec.final_steps))
    return "kd" in spec.arm


def _teacher_topk_for_batch(kind: str, ids: Any, mask: Any, rng: np.random.Generator, teacher: Any | None = None):
    import torch
    import genome_167_kd_canonical as g167

    if kind == "uniform":
        n_windows, seq_len = ids.shape
        n_pos = seq_len - 1
        vocab_size = int(torch.max(ids).item()) + 1
        vocab_size = max(vocab_size, 4096)
        idx = rng.integers(0, vocab_size, size=(n_windows, n_pos, g167.KD_TOPK), dtype=np.int64)
        logits = np.zeros(idx.shape, dtype=np.float32)
        return torch.as_tensor(idx, device=device()), torch.as_tensor(logits, device=device())
    if teacher is None:
        raise RuntimeError("teacher model is required for trained/random KD top-k replay")
    with torch.no_grad():
        with autocast_context():
            logits = teacher(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1].float()
        values, indices = logits.topk(g167.KD_TOPK, dim=-1)
    return indices, values


def _minimal_student_for_cell(spec: CellSpec, tok: Any):
    import genome_167_kd_canonical as g167

    return g167.make_minimal_student(vocab_size=len(tok), seed=spec.seed)


def _g173_student_for_cell(spec: CellSpec, tok: Any):
    import genome_173_cross_arch_flop_cashout as g173

    arm = next(a for a in g173.ARM_SPECS if a.label == spec.arm)
    return g173.build_student(arm.student, vocab_size=len(tok), seed=spec.seed)


def _replay_kd_cell(spec: CellSpec, target_steps: int) -> EarlyCheckpoint:
    import torch
    import genome_167_kd_canonical as g167

    if spec.source == "g173":
        import genome_173_cross_arch_flop_cashout as g173

        tok = g173.load_tokenizer()
        train_ids, train_mask, _ = g173.load_c4_windows(
            tok,
            split="train",
            seed=g173.C4_TRAIN_SEED,
            n_windows=max(512, PROBE_WINDOWS),
        )
        model = _g173_student_for_cell(spec, tok)
    else:
        tok, train_ids, train_mask = _load_minimal_kd_data()
        model = _minimal_student_for_cell(spec, tok)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=g167.LR, betas=(0.9, 0.95), weight_decay=0.01)
    rng = np.random.default_rng(spec.seed)
    schedule = rng.integers(0, int(train_ids.shape[0]), size=(target_steps, g167.TRAIN_BATCH_SIZE), dtype=np.int64)
    early_loss = float("nan")
    teacher_kind = "trained"
    if spec.arm == "kd_uniform_target":
        teacher_kind = "uniform"
    elif spec.arm == "kd_random_teacher":
        teacher_kind = "random"

    kd_needed = any(_kd_active_for_cell(spec, step) for step in range(1, target_steps + 1))
    teacher = None
    if kd_needed and teacher_kind == "trained":
        teacher, _ = g167.load_trained_teacher(tok)
    elif kd_needed and teacher_kind == "random":
        import genome_165_annealed_donor as g165

        teacher = g165.load_random_init(174167001).eval()
        for param in teacher.parameters():
            param.requires_grad_(False)

    model.train()
    try:
        for step in range(1, target_steps + 1):
            idx = schedule[step - 1]
            ids = train_ids[idx].to(device())
            mask = train_mask[idx].to(device())
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                ce_loss = g167.causal_ce_loss(logits, ids, mask)
                if _kd_active_for_cell(spec, step):
                    topk_idx, topk_logits = _teacher_topk_for_batch(teacher_kind, ids, mask, rng, teacher=teacher)
                    kd_loss = g167.topk_kd_loss(logits[:, :-1].contiguous().float(), topk_idx.long(), topk_logits.float())
                    loss = (1.0 - g167.KD_GAMMA) * ce_loss + g167.KD_GAMMA * kd_loss
                else:
                    loss = ce_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            early_loss = float(ce_loss.detach().float().cpu().item())
    finally:
        if teacher is not None:
            del teacher
            cleanup_cuda()
    return EarlyCheckpoint(
        model=model,
        early_loss=early_loss,
        target_steps=target_steps,
        metadata={"tokenizer": "qwen3", "replay_kind": "kd", "teacher_kind": teacher_kind},
    )


def replay_early_checkpoint(arm_spec, target_steps) -> EarlyCheckpoint:
    """Load or rerun the first <=3% of a known cell.

    This function deliberately returns an in-memory checkpoint object; persisted
    cache files store extracted features rather than large model states.
    """

    spec = _as_cell_spec(arm_spec)
    if spec.protocol in {"qwen_anchor", "qwen_anchor_nulls", "qwen_anchor_alt_donor", "tokenizer_isolation"}:
        return _replay_anchor_cell(spec, int(target_steps), None)
    if spec.protocol in {"minimal_kd", "minimal_kd_schedule", "minimal_kd_nulls", "cross_arch_kd"}:
        return _replay_kd_cell(spec, int(target_steps))
    raise NotImplementedError(f"no early replay implementation for {spec.protocol} ({spec.cell_id})")


# ---------------------------------------------------------------------------
# Regression and evaluation
# ---------------------------------------------------------------------------


NON_FEATURE_KEYS = {
    "label",
    "cell_id",
    "source",
    "arm",
    "seed",
    "family",
    "split",
    "protocol",
    "target_steps",
    "final_steps",
}


def _feature_names(rows: Sequence[Mapping[str, Any]], *, early_loss_only: bool) -> list[str]:
    if early_loss_only:
        return ["early_loss"]
    names: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in NON_FEATURE_KEYS:
                continue
            if isinstance(value, (int, float, np.number)) or value is None:
                names.add(key)
    names.add("early_loss")
    return sorted(names)


def _matrix_from_rows(
    rows: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    *,
    medians: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix: list[list[float]] = []
    for row in rows:
        vals: list[float] = []
        for name in names:
            raw = row.get(name, float("nan"))
            vals.append(float(raw) if raw is not None else float("nan"))
        matrix.append(vals)
    X = np.asarray(matrix, dtype=np.float64)
    if medians is None:
        medians = np.nanmedian(X, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
    inds = ~np.isfinite(X)
    if inds.any():
        X[inds] = np.take(medians, np.where(inds)[1])
    if mean is None:
        mean = X.mean(axis=0)
    if scale is None:
        scale = X.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
    return (X - mean) / scale, medians, mean, scale


def _fit_ridge(rows: Sequence[Mapping[str, Any]], labels: Sequence[float], *, early_loss_only: bool):
    from sklearn.linear_model import Ridge

    names = _feature_names(rows, early_loss_only=early_loss_only)
    X, medians, mean, scale = _matrix_from_rows(rows, names)
    y = np.asarray(labels, dtype=np.float64)
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    model._forecast_feature_names = list(names)  # type: ignore[attr-defined]
    model._forecast_medians = medians  # type: ignore[attr-defined]
    model._forecast_mean = mean  # type: ignore[attr-defined]
    model._forecast_scale = scale  # type: ignore[attr-defined]
    return model


def fit_baseline(features, labels):
    return _fit_ridge(features, labels, early_loss_only=True)


def fit_full(features, labels):
    return _fit_ridge(features, labels, early_loss_only=False)


def _predict(model: Any, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    names = getattr(model, "_forecast_feature_names")
    medians = getattr(model, "_forecast_medians")
    mean = getattr(model, "_forecast_mean")
    scale = getattr(model, "_forecast_scale")
    X, _, _, _ = _matrix_from_rows(rows, names, medians=medians, mean=mean, scale=scale)
    return np.asarray(model.predict(X), dtype=np.float64)


def evaluate_held_out(model, X_test, y_test) -> dict[str, Any]:
    y = np.asarray(y_test, dtype=np.float64)
    pred = _predict(model, X_test)
    resid = y - pred
    mse = float(np.mean(resid ** 2)) if y.size else float("nan")
    var_y = float(np.sum((y - y.mean()) ** 2)) if y.size else 0.0
    r2 = float(1.0 - np.sum(resid ** 2) / var_y) if var_y > 1e-12 else float("nan")
    rng = np.random.default_rng(RANDOM_STATE)
    boot = []
    if y.size:
        for _ in range(BOOTSTRAP_N):
            idx = rng.integers(0, y.size, size=y.size)
            boot.append(float(np.mean((y[idx] - pred[idx]) ** 2)))
    lo, hi = (np.percentile(boot, [2.5, 97.5]).tolist() if boot else [float("nan"), float("nan")])
    return {
        "n": int(y.size),
        "mse": mse,
        "r2": r2,
        "mse_bootstrap_ci95": [float(lo), float(hi)],
        "predictions": pred.tolist(),
        "truth": y.tolist(),
        "residuals": resid.tolist(),
    }


def _paired_bootstrap_mse_improvement(y: np.ndarray, pred_base: np.ndarray, pred_full: np.ndarray) -> dict[str, Any]:
    if y.size == 0:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "p_gt_0": float("nan")}
    rng = np.random.default_rng(RANDOM_STATE + 1)
    values = np.empty(BOOTSTRAP_N, dtype=np.float64)
    for i in range(BOOTSTRAP_N):
        idx = rng.integers(0, y.size, size=y.size)
        base = np.mean((y[idx] - pred_base[idx]) ** 2)
        full = np.mean((y[idx] - pred_full[idx]) ** 2)
        values[i] = base - full
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
        "p_gt_0": float(np.mean(values > 0.0)),
    }


def build_summary(*args, **kwargs) -> dict[str, Any]:
    """Build PASS/INTERMEDIATE/FAIL gate summary.

    Expected keyword arguments:
      rows, train_rows, test_rows, baseline_model, full_model,
      baseline_eval, full_eval.
    """

    rows = kwargs.get("rows", [])
    train_rows = kwargs.get("train_rows", [])
    test_rows = kwargs.get("test_rows", [])
    baseline_model = kwargs["baseline_model"]
    full_model = kwargs["full_model"]
    baseline_eval = kwargs["baseline_eval"]
    full_eval = kwargs["full_eval"]

    y = np.asarray([float(row["label"]) for row in test_rows], dtype=np.float64)
    pred_base = _predict(baseline_model, test_rows)
    pred_full = _predict(full_model, test_rows)
    base_mse = float(baseline_eval["mse"])
    full_mse = float(full_eval["mse"])
    mse_reduction = float((base_mse - full_mse) / base_mse) if base_mse > 1e-12 else float("nan")
    paired = _paired_bootstrap_mse_improvement(y, pred_base, pred_full)
    stop_rows = []
    for row, pred, truth in zip(test_rows, pred_full, y):
        recommend_stop = bool(pred < STOP_RECOMMEND_GAIN_THRESHOLD)
        if recommend_stop and truth >= ACTIONABLE_GAIN_NATS:
            stop_rows.append(
                {
                    "cell_id": row["cell_id"],
                    "arm": row["arm"],
                    "seed": row["seed"],
                    "predicted_gain": float(pred),
                    "true_gain": float(truth),
                }
            )
    no_false_stop = len(stop_rows) == 0
    ci_lo = float(paired["ci95"][0])

    if len(test_rows) == 0 or len(train_rows) < 4:
        status = "INTERMEDIATE"
        verdict = "INTERMEDIATE: insufficient train/test rows with labels."
    elif mse_reduction >= PASS_MSE_REDUCTION and ci_lo > 0.0 and no_false_stop:
        status = "PASS"
        verdict = (
            "PASS: early_loss+geometry beats early_loss_only by "
            f"{100.0 * mse_reduction:.1f}% held-out MSE reduction, paired bootstrap CI above 0, "
            "and no high-gain arm is stopped."
        )
    elif not no_false_stop or full_mse >= base_mse:
        status = "FAIL"
        reason = "false stop on high-gain arm" if not no_false_stop else "full model does not beat early-loss baseline"
        verdict = f"FAIL: {reason}."
    else:
        status = "INTERMEDIATE"
        verdict = (
            "INTERMEDIATE: geometry helps but does not clear the locked >=25% MSE reduction "
            "and paired-bootstrap gate."
        )

    return {
        "status": status,
        "verdict": verdict,
        "criteria": {
            "heldout_family": "g173 llama_family",
            "pass_mse_reduction_ge": PASS_MSE_REDUCTION,
            "observed_mse_reduction": mse_reduction,
            "paired_bootstrap_improvement_ci95": paired["ci95"],
            "paired_bootstrap_improvement_mean": paired["mean"],
            "paired_bootstrap_p_gt_0": paired["p_gt_0"],
            "paired_bootstrap_ci_low_gt_0": bool(ci_lo > 0.0) if math.isfinite(ci_lo) else False,
            "stop_threshold_predicted_gain_lt": STOP_RECOMMEND_GAIN_THRESHOLD,
            "actionable_true_gain_ge": ACTIONABLE_GAIN_NATS,
            "no_false_stop_on_actionable_gain": no_false_stop,
        },
        "counts": {
            "rows_total": len(rows),
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
        },
        "baseline_eval": baseline_eval,
        "full_eval": full_eval,
        "false_stop_rows": stop_rows,
        "feature_names_baseline": list(getattr(baseline_model, "_forecast_feature_names")),
        "feature_names_full": list(getattr(full_model, "_forecast_feature_names")),
    }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def _load_qwen_reference_geometry(probe_batch: Mapping[str, Any], layer_indices: Sequence[int]) -> dict[str, Any]:
    import torch
    import genome_165_annealed_donor as g165

    model, _ = g165.load_trained_donor()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    try:
        hidden_states, attention_mask, _ = _model_hidden_states(model, probe_batch)
        chosen = _select_hidden_indices(len(hidden_states), layer_indices)
        mid_idx = chosen[len(chosen) // 2]
        ref_hidden = _hidden_cloud(hidden_states[mid_idx], attention_mask, max_points=FEATURE_MAX_POINTS)
        ref_embed = _embedding_weight(model)
        ref_head = _lm_head_weight(model)
        return {
            "reference_hidden": ref_hidden,
            "reference_embedding": ref_embed[:EMBED_MAX_ROWS] if ref_embed is not None else None,
            "reference_lm_head": ref_head[:EMBED_MAX_ROWS] if ref_head is not None else None,
            "reference_model": g165._MODEL_ID,
            "reference_layer_index": mid_idx,
        }
    finally:
        del model
        cleanup_cuda()


def _build_probe_batch() -> dict[str, Any]:
    import genome_167_kd_canonical as g167

    tok = g167.load_tokenizer()
    ids, mask, meta = g167.load_c4_windows(
        tok,
        split="validation",
        seed=180180,
        n_windows=PROBE_WINDOWS,
    )
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "labels": ids,
        "meta": {"source": "C4 validation", **meta},
    }


def _layer_indices_for_model(model: Any) -> list[int]:
    n_layers = int(getattr(getattr(model, "config", object()), "num_hidden_layers", 0) or 0)
    if n_layers <= 0:
        return []
    # hidden_states includes embedding at index 0; layer outputs start at 1.
    return [1, 1 + n_layers // 2, n_layers]


def _feature_cache_load(spec: CellSpec) -> dict[str, Any] | None:
    path = _cache_path_for_cell(spec)
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    if payload.get("cell_id") != spec.cell_id:
        return None
    return payload


def _feature_cache_write(spec: CellSpec, payload: Mapping[str, Any]) -> None:
    atomic_write_json(_cache_path_for_cell(spec), payload)


def _extract_or_load_row(
    spec: CellSpec,
    *,
    results_jsons: Mapping[str, Any],
    probe_batch: Mapping[str, Any],
    force_replay: bool,
    no_replay: bool,
) -> dict[str, Any] | None:
    cached = None if force_replay else _feature_cache_load(spec)
    if cached is not None:
        features = dict(cached["features"])
        feature_source = "cache"
    elif no_replay:
        features = {"early_loss": _early_loss_from_existing(results_jsons, spec)}
        feature_source = "existing_logs_early_loss_only"
    else:
        checkpoint = replay_early_checkpoint(spec, spec.target_steps)
        local_probe = dict(probe_batch)
        local_probe["early_loss"] = checkpoint.early_loss
        features = extract_features(checkpoint.model, local_probe, _layer_indices_for_model(checkpoint.model))
        feature_source = "early_replay"
        _feature_cache_write(
            spec,
            {
                "cell_id": spec.cell_id,
                "timestamp_utc": now_utc(),
                "spec": asdict(spec),
                "features": features,
                "checkpoint_metadata": checkpoint.metadata,
            },
        )
        del checkpoint.model
        cleanup_cuda()

    try:
        label = compute_label(spec, spec.seed, results_jsons)
    except Exception as exc:
        print(f"  label skipped for {spec.cell_id}: {exc}")
        return None
    row: dict[str, Any] = {
        "cell_id": spec.cell_id,
        "source": spec.source,
        "arm": spec.arm,
        "seed": spec.seed,
        "family": spec.family,
        "split": spec.split,
        "protocol": spec.protocol,
        "target_steps": spec.target_steps,
        "final_steps": spec.final_steps,
        "label": label,
        "feature_source": feature_source,
    }
    row.update(features)
    return row


def run_forecast(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    results_jsons = load_results_jsons()
    cells = build_known_cells(results_jsons)
    if args.limit_cells is not None:
        cells = cells[: args.limit_cells]
    print(f"genome_180: cells={len(cells)} force_replay={args.force_replay} no_replay={args.no_replay}")

    probe_batch = _build_probe_batch()
    if not args.no_replay and not args.no_reference:
        ref = _load_qwen_reference_geometry(probe_batch, layer_indices=[1, 14, 28])
        probe_batch.update(ref)
    rows: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    for idx, spec in enumerate(cells, start=1):
        print(f"  [{idx:03d}/{len(cells):03d}] {spec.cell_id} target_steps={spec.target_steps}")
        try:
            row = _extract_or_load_row(
                spec,
                results_jsons=results_jsons,
                probe_batch=probe_batch,
                force_replay=args.force_replay,
                no_replay=args.no_replay,
            )
        except Exception as exc:
            print(f"    SKIP cell {spec.cell_id}: {type(exc).__name__}: {str(exc)[:200]}")
            skipped.append((spec.cell_id, f"{type(exc).__name__}: {str(exc)[:200]}"))
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass
            continue
        if row is not None:
            rows.append(row)
            if idx % 5 == 0:
                atomic_write_json(
                    OUT_PATH,
                    {
                        "genome": "180",
                        "name": "genome_forecast_diagnostic",
                        "status": "running",
                        "timestamp_utc_last_write": now_utc(),
                        "rows": rows,
                        "elapsed_s": time.time() - t0,
                    },
                )

    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    if len(train_rows) < 2 or len(test_rows) < 1:
        summary = {
            "status": "INTERMEDIATE",
            "verdict": "INTERMEDIATE: not enough train/test rows after feature extraction.",
            "counts": {"rows_total": len(rows), "train_rows": len(train_rows), "test_rows": len(test_rows)},
        }
        payload = {
            "genome": "180",
            "name": "genome_forecast_diagnostic",
            "timestamp_utc": now_utc(),
            "rows": rows,
            "summary": summary,
            "verdict": summary["status"],
            "elapsed_s": time.time() - t0,
        }
        atomic_write_json(OUT_PATH, payload)
        return payload

    y_train = [float(row["label"]) for row in train_rows]
    y_test = [float(row["label"]) for row in test_rows]
    baseline = fit_baseline(train_rows, y_train)
    full = fit_full(train_rows, y_train)
    baseline_eval = evaluate_held_out(baseline, test_rows, y_test)
    full_eval = evaluate_held_out(full, test_rows, y_test)
    summary = build_summary(
        rows=rows,
        train_rows=train_rows,
        test_rows=test_rows,
        baseline_model=baseline,
        full_model=full,
        baseline_eval=baseline_eval,
        full_eval=full_eval,
    )
    payload = {
        "genome": "180",
        "name": "genome_forecast_diagnostic",
        "timestamp_utc": now_utc(),
        "config": {
            "probe_windows": PROBE_WINDOWS,
            "probe_seq_len": PROBE_SEQ_LEN,
            "feature_max_points": FEATURE_MAX_POINTS,
            "rsa_max_points": RSA_MAX_POINTS,
            "bootstrap_n": BOOTSTRAP_N,
            "pass_mse_reduction": PASS_MSE_REDUCTION,
            "actionable_gain_nats": ACTIONABLE_GAIN_NATS,
            "stop_recommend_gain_threshold": STOP_RECOMMEND_GAIN_THRESHOLD,
            "result_json_paths": {k: str(v) for k, v in RESULT_JSON_PATHS.items()},
        },
        "rows": rows,
        "summary": summary,
        "verdict": summary["status"],
        "elapsed_s": time.time() - t0,
    }
    atomic_write_json(OUT_PATH, payload)
    print(f"Saved: {OUT_PATH}")
    print(f"Verdict: {summary['status']}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="g180 Genome Forecast / Diagnostic experiment")
    parser.add_argument("--force-replay", action="store_true", help="Ignore feature cache and rerun early prefixes.")
    parser.add_argument("--no-replay", action="store_true", help="Do not run models; use cached features or early-loss logs only.")
    parser.add_argument("--no-reference", action="store_true", help="Skip trained-Qwen reference geometry construction.")
    parser.add_argument("--limit-cells", type=int, default=None, help="Debug limit on number of cells.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_forecast(args)


if __name__ == "__main__":
    main()
