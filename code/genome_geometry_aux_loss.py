"""Electricity-grade efficiency demo: train a small transformer from scratch
with c = p * d_rd geometric invariant as AUXILIARY LOSS, compare convergence
vs baseline (CE only).

Design:
  - Tiny transformer: 2 layers, d_model=128, n_heads=4, seq_len=64
  - Vocab: GPT-2 BPE restricted to top 5000 tokens
  - Training data: C4-clean text, 50k tokens train / 5k val
  - Total steps: 500, eval every 50
  - Two runs in sequence:
      (a) baseline: CE loss only
      (b) regularized: CE + lambda * |c_measured - c_target|
  - Measure validation CE at each eval step for both runs
  - Compare: does regularized run reach a target val CE in fewer steps?

Hypothesis: if c=p*d_rd is a useful geometric target, steering the model
toward c ~= 2.0 during training should accelerate convergence vs pure CE.

Metric: compute efficiency ratio = (steps_baseline_to_target_loss) /
                                    (steps_regularized_to_target_loss)
If > 1.0: regularizer helps. If ~ 1.0: neutral. If < 1.0: hurts.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_primitives import knn_clustering_coefficient  # noqa: E402
from genome_rate_distortion_probe import rate_distortion_dim, fit_power_law  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]
TARGET_C = 2.0  # text-modality trained-manifold invariant


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, max_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                        batch_first=True, activation="gelu",
                                        norm_first=True)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # tied
        self.max_len = max_len
        self._mid = n_layers // 2  # block to extract from

    def forward(self, ids, return_mid_hidden=False):
        b, s = ids.shape
        pos = torch.arange(s, device=ids.device).unsqueeze(0).expand(b, -1)
        h = self.embed(ids) + self.pos(pos)
        mid_hidden = None
        # causal mask
        mask = torch.triu(torch.full((s, s), float("-inf"), device=ids.device), diagonal=1)
        for i, blk in enumerate(self.blocks):
            h = blk(h, src_mask=mask, is_causal=True)
            if return_mid_hidden and i == self._mid:
                mid_hidden = h
        h = self.ln_f(h)
        logits = self.head(h)
        if return_mid_hidden:
            return logits, mid_hidden
        return logits


def build_tokens(seed=42, n_total=55000, vocab_cap=5000):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    all_ids = []
    for rec in c4_clean_v1(seed=seed, n_samples=500):
        ids = tok(rec["text"], return_tensors="pt", truncation=True, max_length=256)["input_ids"][0]
        all_ids.extend(ids.tolist())
        if len(all_ids) >= n_total:
            break
    # Cap vocab — remap to top-vocab_cap by frequency
    ids_arr = np.array(all_ids)
    vals, counts = np.unique(ids_arr, return_counts=True)
    order = np.argsort(-counts)
    top = vals[order[:vocab_cap - 1]]  # reserve 0 for UNK
    id_map = {int(v): i + 1 for i, v in enumerate(top)}
    remapped = np.array([id_map.get(int(x), 0) for x in all_ids], dtype=np.int64)
    return remapped


def measure_val_nll(model, val_tokens, device, max_len=64, n_batches=8):
    model.eval()
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for b in range(n_batches):
            start = b * max_len * 16
            chunk = val_tokens[start: start + max_len * 16]
            if len(chunk) < max_len * 16:
                break
            x = torch.from_numpy(chunk.reshape(16, max_len)).to(device)
            logits = model(x)
            shift_logits = logits[:, :-1]
            shift_labels = x[:, 1:]
            loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.shape[-1]),
                                    shift_labels.reshape(-1), reduction="sum")
            total_nll += float(loss.item())
            total_tok += shift_labels.numel()
    model.train()
    return total_nll / max(total_tok, 1)


def measure_c(model, val_tokens, device, max_len=64, n_stim=500):
    """Measure c = p * d_rd at mid layer of model on a val-token cloud."""
    model.eval()
    feats = []
    with torch.no_grad():
        for b in range(n_stim // 16 + 1):
            start = b * max_len * 16
            chunk = val_tokens[start: start + max_len * 16]
            if len(chunk) < max_len * 16:
                break
            x = torch.from_numpy(chunk.reshape(16, max_len)).to(device)
            _, mid = model(x, return_mid_hidden=True)
            feats.append(mid.mean(dim=1).cpu())  # seq-mean pooling
            if sum(f.shape[0] for f in feats) >= n_stim:
                break
    X = torch.cat(feats, dim=0).numpy().astype(np.float32)[:n_stim]
    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p, _, _ = fit_power_law(K_GRID, Cs)
    rd = rate_distortion_dim(X)
    c = p * rd["d_rd"]
    model.train()
    return c, p, rd["d_rd"]


def train_one(vocab_size, train_tokens, val_tokens, *, use_aux, aux_lambda=0.1,
              steps=500, eval_every=50, max_len=64, batch=16, lr=3e-4, seed=42):
    torch.manual_seed(seed)
    device = "cuda"
    model = TinyTransformer(vocab_size=vocab_size, max_len=max_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    n_train = len(train_tokens) - max_len - 1
    rng = np.random.default_rng(seed)

    history = []
    t0 = time.time()
    for step in range(steps + 1):
        if step % eval_every == 0:
            val_nll = measure_val_nll(model, val_tokens, device, max_len=max_len)
            c_val, p_val, drd_val = measure_c(model, val_tokens, device,
                                                max_len=max_len, n_stim=300)
            history.append({"step": step, "val_nll": val_nll, "c": c_val,
                             "p": p_val, "d_rd": drd_val,
                             "wall_s": time.time() - t0})
            print(f"  [step {step:4d}] val_nll={val_nll:.4f}  c={c_val:.2f}  p={p_val:.3f}  d_rd={drd_val:.2f}  t={time.time()-t0:.1f}s")
        if step == steps:
            break

        # training step
        idxs = rng.integers(0, n_train, size=batch)
        batch_arr = np.stack([train_tokens[i:i + max_len + 1] for i in idxs])
        x = torch.from_numpy(batch_arr[:, :max_len]).to(device)
        y = torch.from_numpy(batch_arr[:, 1:]).to(device)
        if use_aux:
            logits, mid = model(x, return_mid_hidden=True)
        else:
            logits = model(x)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss = ce
        if use_aux:
            # Aux: |c_current - TARGET_C|, but measuring c per step is slow.
            # Use a proxy: penalize the RATIO of inter-token-variance to
            # intra-token-mean-activation that proxies the d_rd trend.
            # Simpler: regularize toward the trained d_rd ~= 14 via
            # spectral norm control on the mid layer.
            # Use a lightweight per-batch surrogate: minimize the KL distance
            # between normalized pooled activations' sorted singular values
            # and a target rank-14-like decay profile. For simplicity, penalize
            # how fast the spectrum of mid activations decays: target exponent.
            pooled = mid.mean(dim=1)  # (b, d)
            centered = pooled - pooled.mean(dim=0, keepdim=True)
            # Compute singular values
            s = torch.linalg.svdvals(centered.float())
            # Normalize
            s_norm = s / (s.sum() + 1e-9)
            # Compute effective dim via participation ratio
            eff_dim = (s.sum() ** 2) / (s * s + 1e-9).sum()
            # Penalize deviation from target 14 (text d_rd)
            aux = torch.abs(eff_dim - 14.0) / 14.0
            loss = ce + aux_lambda * aux

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    return history


def main():
    print("Building token stream...")
    tokens = build_tokens(seed=42, n_total=80000, vocab_cap=5000)
    split = 60000
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]
    vocab_size = 5000
    print(f"  {len(train_tokens)} train, {len(val_tokens)} val, vocab {vocab_size}")

    print("\n=== RUN A: BASELINE (CE only) ===")
    hist_a = train_one(vocab_size, train_tokens, val_tokens, use_aux=False, steps=500, seed=42)

    print("\n=== RUN B: REGULARIZED (CE + aux eff_dim -> 14) ===")
    hist_b = train_one(vocab_size, train_tokens, val_tokens, use_aux=True, aux_lambda=0.1, steps=500, seed=42)

    # Analysis
    print("\n=== COMPARATIVE VAL-NLL TRAJECTORY ===")
    print(f"{'step':>5s} {'baseline':>9s} {'regularized':>12s} {'diff':>8s}")
    for a, b in zip(hist_a, hist_b):
        if a["step"] == b["step"]:
            diff = a["val_nll"] - b["val_nll"]  # positive if regularized is BETTER
            print(f"  {a['step']:5d} {a['val_nll']:9.4f} {b['val_nll']:12.4f} {diff:+8.4f}")

    # Steps to reach baseline's final loss: compare
    final_baseline = hist_a[-1]["val_nll"]
    target = final_baseline * 1.05  # target = 5% worse than final baseline
    a_steps_to_target = next((h["step"] for h in hist_a if h["val_nll"] <= target), None)
    b_steps_to_target = next((h["step"] for h in hist_b if h["val_nll"] <= target), None)
    speedup = None
    if a_steps_to_target and b_steps_to_target:
        speedup = a_steps_to_target / b_steps_to_target
    print(f"\n  baseline steps to val_nll <= {target:.3f}: {a_steps_to_target}")
    print(f"  regularized steps to same target:      {b_steps_to_target}")
    if speedup:
        print(f"  compute-efficiency ratio (baseline/regularized): {speedup:.2f}x")
        verdict = ("REGULARIZER_SPEEDS_UP" if speedup > 1.1
                   else "REGULARIZER_NEUTRAL" if 0.9 < speedup < 1.1
                   else "REGULARIZER_HURTS")
    else:
        verdict = "INCOMPLETE_TRAINING"
    print(f"  verdict: {verdict}")

    out = {"purpose": "Electricity-grade efficiency demo: CE vs CE + eff_dim aux",
           "baseline_history": hist_a, "regularized_history": hist_b,
           "final_baseline_val_nll": final_baseline,
           "target_val_nll": target,
           "baseline_steps_to_target": a_steps_to_target,
           "regularized_steps_to_target": b_steps_to_target,
           "speedup_ratio": speedup, "verdict": verdict}
    out_path = _ROOT / "results/gate2/geometry_aux_loss.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
