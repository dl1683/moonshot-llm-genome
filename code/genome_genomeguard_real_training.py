"""GenomeGuard real-training-failure demo — Codex home-run C.

Prior GenomeGuard work:
  genome_067 (swap): 7.29x rel_err spike on stimulus swap (real-world
    "silent data corruption" detector). 5-arch cross-validated in
    genome_069 (6.9-144.9x spikes).
  genome_068 (noise sweep): 8.2x spike at sigma=0.3 catastrophic
    weight perturbation - but non-monotone. SIMULATION.

Codex home-run C: replace the noise simulation with an ACTUAL training
blowup and show bridge rel_err spikes BEFORE val loss degrades.

DESIGN.
 - 2-layer d=128 tiny transformer from scratch (same infra as
   genome_048 / genome_066).
 - Three runs, 500 steps each:
   (A) BASELINE: normal training (lr=3e-4, standard AdamW).
   (B) LR_BLOWUP: lr=3e-2 (100x too high). Training diverges.
   (C) GRAD_SIGN_FLIP: at step 100, flip sign of gradients for 50
       steps (simulates a bad gradient hook or optimizer bug).
 - At every 25 steps, log both val_nll AND bridge rel_err on a
   fixed 300-sample C4 probe buffer.
 - Measure: lead_time_steps = (step at which val_nll degrades >=5pct)
   minus (step at which bridge_rel_err spikes >=2x baseline).
 - Positive lead_time = GenomeGuard is a leading indicator.

Kill: if bridge doesn't lead val_nll in >=2/3 blowup modes, the real-
training detection story stays simulation-only.

Pass: bridge leads val_nll by >=50 steps in >=2/3 sabotage modes.
Shippable for outreach ("catch training failures BEFORE loss tells
you").
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
    TinyTransformer, build_tokens, measure_val_nll,
)
from genome_primitives import knn_clustering_coefficient  # noqa: E402
from genome_rate_distortion_probe import rate_distortion_dim, fit_power_law  # noqa: E402

_ROOT = _THIS_DIR.parent
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]


def bridge_rel_err_on_student(model, val_tokens, device, max_len=64, n_stim=300):
    """Compute bridge rel_err from student model's mid activations."""
    model.eval()
    feats = []
    with torch.no_grad():
        for b in range(n_stim // 16 + 1):
            start = b * max_len * 16
            chunk = val_tokens[start:start + max_len * 16]
            if len(chunk) < max_len * 16:
                break
            x = torch.from_numpy(chunk.reshape(16, max_len)).to(device)
            _, mid = model(x, return_mid_hidden=True)
            feats.append(mid.mean(dim=1).cpu())
            if sum(f.shape[0] for f in feats) >= n_stim:
                break
    X = torch.cat(feats, dim=0).numpy().astype(np.float32)[:n_stim]
    model.train()
    if X.shape[0] < 10 or np.all(X == 0) or not np.all(np.isfinite(X)):
        return float("nan"), float("nan"), float("nan")
    # eff_rank
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False) / np.sqrt(X.shape[0] - 1)
    if not np.all(np.isfinite(s)) or s.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    eff_rank = (s ** 2).sum() ** 2 / max((s ** 4).sum(), 1e-12)
    # c via kNN + rate-distortion
    try:
        Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
        p, _, _ = fit_power_law(K_GRID, Cs)
        rd = rate_distortion_dim(X)
        c = p * rd["d_rd"]
        ratio = eff_rank / rd["d_rd"]
        rel_err = abs(ratio - c) / max(c, 1e-6)
    except Exception:
        c = float("nan"); ratio = float("nan"); rel_err = float("nan")
    return float(c), float(ratio), float(rel_err)


def train_sabotaged(vocab_size, train_tokens, val_tokens,
                     *, mode="baseline", steps=500, probe_every=25,
                     max_len=64, batch=16, lr_base=3e-4, seed=42,
                     grad_flip_start=100, grad_flip_len=50):
    torch.manual_seed(seed)
    device = "cuda"
    model = TinyTransformer(vocab_size=vocab_size, max_len=max_len).to(device)
    lr = lr_base * (100 if mode == "lr_blowup" else 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    n_train = len(train_tokens) - max_len - 1
    rng = np.random.default_rng(seed)

    log = []
    t0 = time.time()
    for step in range(steps + 1):
        if step % probe_every == 0:
            val_nll = measure_val_nll(model, val_tokens, device, max_len=max_len)
            c, ratio, rel_err = bridge_rel_err_on_student(model, val_tokens, device,
                                                            max_len=max_len)
            log.append({"step": step, "val_nll": val_nll,
                        "c": c, "ratio": ratio, "bridge_rel_err": rel_err,
                        "wall_s": time.time() - t0})
            print(f"  [{mode:>14s} step {step:4d}] val_nll={val_nll:.3f}  "
                  f"rel_err={rel_err:.3f}  c={c:.2f}  t={time.time()-t0:.1f}s")
        if step == steps:
            break

        idxs = rng.integers(0, n_train, size=batch)
        batch_arr = np.stack([train_tokens[i:i + max_len + 1] for i in idxs])
        x = torch.from_numpy(batch_arr[:, :max_len]).to(device)
        y = torch.from_numpy(batch_arr[:, 1:]).to(device)
        logits = model(x)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

        opt.zero_grad()
        ce.backward()
        # Gradient sabotage mode
        if mode == "grad_sign_flip" and grad_flip_start <= step < grad_flip_start + grad_flip_len:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(-1)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    return log


def main():
    tokens = build_tokens(seed=42, n_total=80000, vocab_cap=5000)
    split = 60000
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]
    vocab_size = 5000

    modes = ["baseline", "lr_blowup", "grad_sign_flip"]
    all_logs = {}
    for mode in modes:
        print(f"\n=== {mode.upper()} ===")
        all_logs[mode] = train_sabotaged(vocab_size, train_tokens, val_tokens,
                                          mode=mode, steps=500, probe_every=25,
                                          seed=42)

    # Analysis: lead-time of bridge rel_err vs val_nll degradation
    base_rel = [e["bridge_rel_err"] for e in all_logs["baseline"] if np.isfinite(e["bridge_rel_err"])]
    base_rel_median = float(np.median(base_rel)) if base_rel else 0.1

    print(f"\n\n=== GENOMEGUARD REAL-TRAINING-FAILURE RESULTS ===")
    print(f"  baseline median bridge_rel_err: {base_rel_median:.3f}")
    print(f"  spike threshold (2x baseline):  {2*base_rel_median:.3f}")

    summary = {}
    for mode in modes:
        log = all_logs[mode]
        # Find first step where val_nll degrades >= 5pct vs step 0
        nll_0 = log[1]["val_nll"] if len(log) > 1 else log[0]["val_nll"]
        nll_spike_step = next((e["step"] for e in log[1:]
                                if np.isfinite(e["val_nll"])
                                and e["val_nll"] > nll_0 * 1.05), None)
        bridge_spike_step = next((e["step"] for e in log[1:]
                                   if np.isfinite(e["bridge_rel_err"])
                                   and e["bridge_rel_err"] > 2 * base_rel_median), None)
        lead = None
        if nll_spike_step is not None and bridge_spike_step is not None:
            lead = nll_spike_step - bridge_spike_step
        summary[mode] = {
            "nll_5pct_spike_step": nll_spike_step,
            "bridge_2x_spike_step": bridge_spike_step,
            "lead_steps": lead,
        }
        print(f"  {mode}: nll_spike_step={nll_spike_step}  "
              f"bridge_spike_step={bridge_spike_step}  "
              f"lead={lead}")

    # Verdict: did bridge lead val_nll in sabotage modes?
    leaders = 0; tested = 0
    for mode in ["lr_blowup", "grad_sign_flip"]:
        s = summary[mode]
        if s["lead_steps"] is not None:
            tested += 1
            if s["lead_steps"] >= 50:
                leaders += 1
    if leaders >= 2:
        verdict = (f"GENOMEGUARD_LEADS_REAL_TRAINING_FAILURE - in {leaders}/"
                   f"{tested} sabotage modes the bridge rel_err spiked >=50 "
                   "steps BEFORE val_nll degraded. Shippable as early-"
                   "warning monitor with real-training receipts.")
    elif leaders == 1:
        verdict = (f"PARTIAL - leader in 1 of 2 sabotage modes. Good start "
                   "but not a full shipping claim yet.")
    else:
        verdict = ("REAL_TRAINING_NULL - bridge did not lead val_nll in "
                   "practical sabotage. GenomeGuard remains swap-only "
                   "shipping claim.")
    print(f"\n  verdict: {verdict}")

    out = {"purpose": "GenomeGuard real-training-failure lead-time test",
           "all_logs": all_logs,
           "summary": summary,
           "baseline_median_rel_err": base_rel_median,
           "verdict": verdict}
    out_path = _ROOT / "results/gate2/genomeguard_real_training.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
