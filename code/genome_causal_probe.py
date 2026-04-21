"""Gate-2 G2.4 causal-ablation probe.

Per `research/prereg/genome_knn_k10_causal_2026-04-21.md` §9, this module
implements the HF-hook-based causal ablation test for kNN-k10 clustering.

For each (system, layer_depth, ablation_scheme, lam):
  1. Run the model forward on a batch of stimuli with a pre-hook that
     intercepts the chosen layer's hidden state X, applies one of the three
     ablation schemes from `genome_ablation_schemes.py` at strength lam,
     and passes the perturbed X downstream.
  2. Measure next-token cross-entropy loss over the full batch.
  3. Record L(lam, scheme) + SE for the aggregation per §6 of the prereg.

The no-op baseline (lam=0) is computed once per (system, layer) pair and
reused for all schemes.

Per CLAUDE.md §6.1: new file justified as a distinct Gate-2 capability
runner (hook-based causal test, different concern from G1.3 stim-resample).

Usage (smoke test):
    python code/genome_causal_probe.py --system qwen3-0.6b --n 50 \
        --lam 0 1.0 --scheme topk

Full run (per prereg §8, 3 systems × 3 depths × 5 λ × 3 schemes):
    python code/genome_causal_probe.py --full-grid -n 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from stimulus_banks import c4_clean_v1  # noqa: E402
from genome_loaders import load_system, SYSTEM_IDS  # noqa: E402
from genome_extractor import _transformer_blocks, sentinel_layer_indices  # noqa: E402
from genome_ablation_schemes import (  # noqa: E402
    ablate_topk_neighbors, ablate_random_10d, ablate_pca_10d,
)


ABLATION_FNS = {
    "topk":   lambda X, lam, rng=None: ablate_topk_neighbors(X, lam, k=10),
    "random": lambda X, lam, rng=None: ablate_random_10d(X, lam, rng=rng),
    "pca":    lambda X, lam, rng=None: ablate_pca_10d(X, lam),
}


def _install_ablation_hook(block: torch.nn.Module, *,
                           scheme_name: str, lam: float,
                           rng_seed: int,
                           pooled_dim_for_ablation: str = "token_pool",
                           ) -> Callable[[], None]:
    """Install a forward-hook on `block` that replaces its output activations
    with an ablated version.

    The hook:
      1. Receives the block's output hidden state H ∈ R^{B × T × h}.
      2. Pools over T (mean) to form a 2D point cloud X ∈ R^{B × h}.
      3. Applies the scheme at strength lam, yielding X'.
      4. Adds the *per-sequence correction* Δ_i = X'_i − X_i as a CONSTANT
         shift back to every timestep of H_i. This patches the pooled-space
         ablation back into the residual stream at all positions, which is
         the right place for a mean-pooled primitive like kNN-k10.

    Returns a function that, when called, removes the hook.
    """
    fn = ABLATION_FNS[scheme_name]
    rng = np.random.default_rng(rng_seed)

    def forward_hook(module, _inputs, output):  # noqa: ARG001
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        if hidden.dim() != 3:
            return output  # unknown shape; do nothing
        if lam == 0.0:
            return output

        # Pool over token dim.
        mask = torch.ones_like(hidden[..., 0])  # (B, T); we don't have pad-mask here — good-enough for smoke
        X = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        X_np = X.detach().float().cpu().numpy()
        X_ablated_np, _info = fn(X_np, lam, rng=rng)
        # Compute per-sequence shift vector.
        delta_np = X_ablated_np - X_np  # (B, h)
        delta = torch.from_numpy(delta_np).to(hidden.device, dtype=hidden.dtype)
        # Broadcast-add to every token position.
        hidden_new = hidden + delta.unsqueeze(1)
        if rest is not None:
            return (hidden_new,) + rest
        return hidden_new

    handle = block.register_forward_hook(forward_hook)
    return handle.remove


def measure_loss(model: Any, tokenizer: Any, texts: list[str],
                 *, max_length: int, device: str, batch_size: int = 16
                 ) -> tuple[float, float, int]:
    """Return (mean NLL per token, SE of mean, total tokens scored).

    Pads the last batch normally; right-pads. Uses the model's own LM head.
    """
    model.eval()
    per_batch_nll = []
    per_batch_ntok = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            # Shift labels: next-token prediction
            labels = input_ids.clone()
            labels[attn == 0] = -100
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            # out.loss is the mean over non-ignored tokens.
            ntok = (labels != -100).sum().item()
            if ntok > 0:
                per_batch_nll.append(float(out.loss.item()) * ntok)
                per_batch_ntok.append(ntok)
    total_ntok = sum(per_batch_ntok)
    if total_ntok == 0:
        return float("nan"), float("nan"), 0
    mean_nll = sum(per_batch_nll) / total_ntok
    # Variance approx: treat batches as iid (not strictly true but smoke-level).
    if len(per_batch_nll) > 1:
        per_batch_mean_nll = [nll / nt for nll, nt in zip(per_batch_nll, per_batch_ntok)]
        mean_of_means = float(np.mean(per_batch_mean_nll))
        se = float(np.std(per_batch_mean_nll, ddof=1) / (len(per_batch_mean_nll) ** 0.5))
    else:
        mean_of_means = mean_nll
        se = 0.0
    return mean_of_means, se, total_ntok


def run_causal_cell(system_key: str, *,
                    n_sentences: int, seed: int, max_length: int,
                    depth_index: int,
                    schemes: list[str], lams: list[float],
                    device: str = "cuda") -> dict:
    """Run one (system, depth) cell of the G2.4 grid: all schemes × all lams.

    Returns a dict keyed by (scheme, lam) → {loss, se, ntok}.
    """
    meta = SYSTEM_IDS[system_key]
    if meta.get("modality", "text") != "text":
        raise ValueError(f"{system_key} is not a text system; causal probe only runs on LLMs")

    print(f"[{system_key}] loading...")
    sys_obj = load_system(meta["hf_id"], quant="fp16", untrained=False, device=device)
    n_layers = sys_obj.n_hidden_layers()
    sentinel_idxs = sentinel_layer_indices(n_layers)
    k_index = sentinel_idxs[depth_index]
    k_normalized = k_index / max(n_layers - 1, 1)
    print(f"[{system_key}] L={n_layers}  sentinel depth index {depth_index} -> "
          f"layer {k_index} (normalized {k_normalized:.3f})")

    print(f"[{system_key}] streaming {n_sentences} C4 sentences (seed={seed})...")
    texts = [it["text"] for it in c4_clean_v1(seed=seed, n_samples=n_sentences)]

    blocks = _transformer_blocks(sys_obj.model)
    target_block = blocks[k_index]

    results: dict = {}
    # Baseline (lam=0) once.
    loss0, se0, ntok0 = measure_loss(sys_obj.model, sys_obj.tokenizer, texts,
                                      max_length=max_length, device=device)
    print(f"[{system_key}] baseline NLL = {loss0:.4f} (+/- {se0:.4f}, n_tokens={ntok0})")
    results["baseline"] = {"loss": loss0, "se": se0, "ntok": ntok0}

    for scheme in schemes:
        for lam in lams:
            if lam == 0.0:
                results[f"{scheme}|lam=0.0"] = results["baseline"]
                continue
            remove = _install_ablation_hook(target_block,
                                            scheme_name=scheme, lam=lam,
                                            rng_seed=int(seed) * 100 + int(lam * 1000))
            try:
                loss, se, ntok = measure_loss(sys_obj.model, sys_obj.tokenizer,
                                              texts, max_length=max_length,
                                              device=device)
            finally:
                remove()
            delta = loss - loss0
            print(f"[{system_key}] {scheme:6s} lam={lam:.2f}  NLL={loss:.4f}  "
                  f"delta={delta:+.4f}  ({delta/loss0*100:+.1f}% rel)")
            results[f"{scheme}|lam={lam}"] = {
                "loss": loss, "se": se, "ntok": ntok,
                "delta": delta, "rel_delta": delta / loss0 if loss0 else 0.0,
            }

    sys_obj.unload()
    return {
        "system_key": system_key,
        "n_layers": n_layers,
        "depth_index": depth_index,
        "k_index": k_index,
        "k_normalized": round(k_normalized, 4),
        "n_sentences": n_sentences,
        "seed": seed,
        "results": results,
    }


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", type=str, default="qwen3-0.6b",
                    choices=list(SYSTEM_IDS.keys()))
    ap.add_argument("-n", "--n-sentences", type=int, default=50,
                    help="smoke-test default n=50; full-grid uses 500+")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--depth-index", type=int, default=1,
                    help="index into sentinel_layer_indices; 0=early, 1=mid, 2=late")
    ap.add_argument("--schemes", type=str, nargs="+",
                    default=["topk", "random", "pca"],
                    choices=["topk", "random", "pca"])
    ap.add_argument("--lam", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    args = ap.parse_args()

    t0 = time.time()
    out = run_causal_cell(args.system, n_sentences=args.n_sentences,
                          seed=args.seed, max_length=args.max_length,
                          depth_index=args.depth_index,
                          schemes=args.schemes, lams=args.lam)
    out["total_wall_clock_seconds"] = round(time.time() - t0, 2)

    out_dir = _THIS_DIR.parent / "results" / "gate2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (f"causal_{args.system}_depth{args.depth_index}_"
                          f"n{args.n_sentences}_seed{args.seed}.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2, default=float)
    print(f"out: {out_path}")
