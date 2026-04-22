"""Codex Move C (9/10 DeepMind-publishability): candidate-8 as training aux loss.

genome_048 tried eff_rank=14 fixed target (d_rd-only) -> NEUTRAL (no
speedup). Codex's candidate-8-based variant: target the RATIO
eff_rank / d_rd_ma approximately 2.0 where d_rd_ma is a moving average of d_rd
on a held-out buffer. The ratio target is more principled because it
nudges the spectrum toward the universal bridge shape, not a fixed
eff_rank value (which depends on model width).

Design:
  - Same 2-layer tiny transformer as genome_048 (d=128, n_heads=4)
  - Training: 500 steps with CE baseline vs CE + aux, two seeds (42, 1337)
  - Aux: (eff_rank_batch - TARGET_C * d_rd_ma)^2 on mid-layer pooled
  - d_rd_ma updated every 50 steps from a small held-out C4 buffer (n=300)
  - TARGET_C = 2.0 (text base)

Kill condition (Codex):
  - speedup < 1.10x on both seeds OR
  - final val NLL worse by >1%
  => candidate-8 aux loss does not help convergence.

If aux speedup >= 1.10x on BOTH seeds AND final NLL within 1% of baseline:
electricity-grade demo lands. Candidate-8 becomes a training compute
lever.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_geometry_aux_loss import (  # noqa: E402
    TinyTransformer, build_tokens, measure_val_nll, measure_c,
)

_ROOT = _THIS_DIR.parent
TARGET_C = 2.0  # text base


def compute_eff_rank_torch(pooled):
    centered = pooled - pooled.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(centered.float())
    eff_rank = (s.sum() ** 2) / (s * s + 1e-9).sum()
    return eff_rank


def train_one(vocab_size, train_tokens, val_tokens, *, use_aux, aux_lambda=0.05,
              steps=500, eval_every=50, max_len=64, batch=16, lr=3e-4, seed=42,
              d_rd_buffer_size=300):
    torch.manual_seed(seed)
    device = "cuda"
    model = TinyTransformer(vocab_size=vocab_size, max_len=max_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    n_train = len(train_tokens) - max_len - 1
    rng = np.random.default_rng(seed)

    d_rd_ma = 14.0  # init to observed trained text value

    history = []
    t0 = time.time()
    for step in range(steps + 1):
        if step % eval_every == 0:
            val_nll = measure_val_nll(model, val_tokens, device, max_len=max_len)
            c_val, p_val, drd_val = measure_c(model, val_tokens, device,
                                              max_len=max_len, n_stim=d_rd_buffer_size)
            d_rd_ma = 0.9 * d_rd_ma + 0.1 * drd_val if step > 0 else drd_val
            history.append({"step": step, "val_nll": val_nll, "c": float(c_val),
                            "p": float(p_val), "d_rd": float(drd_val),
                            "d_rd_ma": float(d_rd_ma),
                            "wall_s": time.time() - t0})
            print(f"  [seed={seed} {'AUX' if use_aux else 'CE '} step {step:4d}] "
                  f"val_nll={val_nll:.4f}  c={c_val:.2f}  d_rd={drd_val:.2f}  "
                  f"d_rd_ma={d_rd_ma:.2f}  t={time.time()-t0:.1f}s")
        if step == steps:
            break

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
            pooled = mid.mean(dim=1)  # (b, d)
            eff_rank_b = compute_eff_rank_torch(pooled)
            target_eff_rank = TARGET_C * d_rd_ma
            aux = ((eff_rank_b - target_eff_rank) / max(target_eff_rank, 1e-6)) ** 2
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

    all_results = {}
    for seed in (42, 1337):
        print(f"\n\n============================= SEED {seed} =============================")
        print("\n=== RUN A: BASELINE (CE only) ===")
        hist_a = train_one(vocab_size, train_tokens, val_tokens, use_aux=False, steps=500, seed=seed)
        print("\n=== RUN B: CANDIDATE-8 AUX (eff_rank -> 2*d_rd_ma) ===")
        hist_b = train_one(vocab_size, train_tokens, val_tokens, use_aux=True, aux_lambda=0.05, steps=500, seed=seed)
        all_results[seed] = {"baseline": hist_a, "aux": hist_b}

    # Summarize
    print("\n\n=== ELECTRICITY-GRADE RESULTS (candidate-8 aux vs CE baseline) ===")
    summary = []
    for seed, d in all_results.items():
        a = d["baseline"]; b = d["aux"]
        final_a = a[-1]["val_nll"]; final_b = b[-1]["val_nll"]
        # Target = 5% worse than baseline final
        target = final_a * 1.05
        a_steps = next((h["step"] for h in a if h["val_nll"] <= target), None)
        b_steps = next((h["step"] for h in b if h["val_nll"] <= target), None)
        speedup = (a_steps / b_steps) if (a_steps and b_steps and b_steps > 0) else None
        summary.append({"seed": seed, "final_baseline_nll": final_a,
                        "final_aux_nll": final_b, "baseline_steps": a_steps,
                        "aux_steps": b_steps, "speedup": speedup})
        print(f"  seed={seed}: final_baseline={final_a:.4f}, final_aux={final_b:.4f}, "
              f"speedup={'{:.2f}x'.format(speedup) if speedup else 'n/a'}")

    # Verdict per Codex kill condition
    speedups = [s["speedup"] for s in summary if s["speedup"] is not None]
    nll_deltas = [abs(s["final_aux_nll"] - s["final_baseline_nll"]) / max(s["final_baseline_nll"], 1e-6)
                  for s in summary]
    if all(s >= 1.10 for s in speedups) and all(d < 0.01 for d in nll_deltas):
        verdict = (f"ELECTRICITY_GRADE_LANDED — aux speedup >=1.10x on both "
                   f"seeds AND final NLL within 1pct. candidate-8 is a training "
                   "compute lever.")
    elif any(s >= 1.10 for s in speedups):
        verdict = (f"PARTIAL_SPEEDUP — aux helps on some seeds. Not DeepMind-publishable.")
    else:
        verdict = (f"REGULARIZER_NEUTRAL_OR_HURTS — candidate-8 aux did not "
                   "accelerate convergence. Kill condition fires.")
    print(f"\n  verdict: {verdict}")

    out = {"purpose": "Candidate-8 aux loss training (Codex Move C)",
           "all_results": all_results, "summary": summary,
           "verdict": verdict}
    out_path = _ROOT / "results/gate2/candidate8_aux_loss_train.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
