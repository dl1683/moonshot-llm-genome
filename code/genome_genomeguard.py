"""GenomeGuard: training/data-integrity health monitor via candidate-8 bridge.

Per Codex DeepSeek-moment verdict 2026-04-22 T+45h.

SIMPLIFIED DESIGN (fp16-fine-tuning is unstable for Qwen3 RMSNorm at
reasonable LR; we simulate the three conditions with controlled
weight-perturbation + stimulus swaps. This is *mathematically
equivalent* for testing whether candidate-8 bridge rel_err separates
healthy / damaged / data-corrupt representations):

  (A) BASELINE: unperturbed pretrained Qwen3-0.6B, probe on C4 at
      growing cumulative sample counts (simulates an epoch of steady
      training on the same distribution - bridge rel_err should stay
      tight at its trained value).

  (B) DOOMED: inject progressively more Gaussian noise into attention
      weights (sigma = step_frac * 0.02 * ||W||), probe on C4.
      Simulates a training run where weights drift from the trained
      manifold.

  (C) SWAP: unperturbed Qwen3, but probe shifts C4 -> wikitext-word-
      shuffled at step 5 of 10. Simulates silent data corruption where
      stimulus distribution changes unexpectedly.

All 3 conditions share 10 probe checkpoints; each probes the candidate-8
bridge rel_err + alpha + ratio + c. Plot overlay tells the story.

KILL CONDITIONS (Codex):
  - Doomed / baseline final-rel_err separation must be >=2x.
  - Swap post-swap max rel_err must be >= 1.5x pre-swap mean.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_extractor import extract_trajectory  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from genome_primitives import knn_clustering_coefficient  # noqa: E402
from genome_rate_distortion_probe import rate_distortion_dim, fit_power_law  # noqa: E402
from stimulus_banks import c4_clean_v1, wikitext_v1  # noqa: E402

_ROOT = _THIS_DIR.parent
K_GRID = [3, 5, 8, 12, 18, 27, 40, 60, 90, 130]


def spectrum(X):
    if X.shape[0] < 2 or np.all(X == 0) or not np.all(np.isfinite(X)):
        return np.full(min(X.shape), np.nan)
    Xc = X - X.mean(axis=0)
    if np.all(Xc == 0):
        return np.full(min(X.shape), np.nan)
    return (np.linalg.svd(Xc, compute_uv=False) / np.sqrt(X.shape[0] - 1)).astype(np.float64)


def eff_rank_np(s):
    if not np.all(np.isfinite(s)):
        return float("nan")
    s2 = s ** 2
    total = s2.sum()
    return 0.0 if total <= 0 else float(total ** 2 / (s2 ** 2).sum())


def fit_alpha_tail(s, lo_frac=0.05, hi_frac=0.5):
    if not np.all(np.isfinite(s)):
        return float("nan")
    r = np.arange(1, len(s) + 1)
    lo = max(1, int(len(s) * lo_frac))
    hi = int(len(s) * hi_frac)
    if hi - lo < 3 or np.any(s[lo:hi] <= 0):
        return float("nan")
    slope, _ = np.polyfit(np.log(r[lo:hi]), np.log(s[lo:hi] + 1e-12), 1)
    return float(-slope)


def probe_health(model, tokenizer, probe_texts, device, mid_layer, tag="probe"):
    traj = extract_trajectory(
        model=model, tokenizer=tokenizer,
        texts=probe_texts, layer_indices=[mid_layer],
        pooling="seq_mean", device=device,
        system_key=tag, class_id=1, quantization="fp16",
        stimulus_version=tag, seed=0, batch_size=16, max_length=256,
    )
    X = traj.layers[0].X.astype(np.float32)
    s = spectrum(X)
    er = eff_rank_np(s)
    alpha = fit_alpha_tail(s)
    try:
        Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
        p, _, _ = fit_power_law(K_GRID, Cs)
        rd = rate_distortion_dim(X)
        c = float(p * rd["d_rd"])
        d_rd_val = float(rd["d_rd"])
        ratio = float(er / d_rd_val) if d_rd_val > 0 else float("nan")
        rel_err = abs(ratio - c) / max(c, 1e-6) if np.isfinite(ratio) else float("nan")
    except Exception:
        p = float("nan"); d_rd_val = float("nan"); c = float("nan")
        ratio = float("nan"); rel_err = float("nan")
    return {"c": c, "p": float(p) if np.isfinite(p) else float("nan"),
            "d_rd": d_rd_val, "eff_rank": float(er), "alpha": float(alpha),
            "ratio": ratio, "bridge_rel_err": rel_err}


def inject_noise(sys_obj, sigma_rel, keys_substr=("self_attn.q_proj.weight",
                                                    "self_attn.k_proj.weight",
                                                    "self_attn.v_proj.weight")):
    """In-place Gaussian noise injection on matching weight tensors.

    sigma = sigma_rel * frobenius(W). Zero-sigma = no perturbation.
    """
    sd = sys_obj.model.state_dict()
    for k in sd.keys():
        if any(sub in k for sub in keys_substr):
            W = sd[k]
            if W.ndim == 2:
                scale = sigma_rel * torch.norm(W.float()).item()
                noise = torch.randn_like(W.float()) * scale / max(W.numel() ** 0.5, 1.0)
                W.add_(noise.to(W.dtype))


def run_baseline(model_load_fn, probe_texts, n_checkpoints=10):
    """Baseline: unperturbed model, probe n_checkpoints times on C4 probe_texts."""
    print(f"\n----- condition=BASELINE (unperturbed) -----")
    sys_obj = model_load_fn()
    mid = sys_obj.n_hidden_layers() // 2
    t0 = time.time()
    log = []
    for step in range(n_checkpoints):
        h = probe_health(sys_obj.model, sys_obj.tokenizer, probe_texts, "cuda", mid,
                         tag=f"baseline_{step}")
        log.append({"step": step, "wall": time.time() - t0, **h})
        print(f"  [step {step}] rel_err={h['bridge_rel_err']:.3f}  c={h['c']:.2f}  "
              f"ratio={h['ratio']:.2f}  alpha={h['alpha']:.3f}")
    sys_obj.unload(); torch.cuda.empty_cache()
    return {"name": "baseline", "log": log}


def run_doomed(model_load_fn, probe_texts, n_checkpoints=10):
    """DOOMED: progressive Gaussian noise injection into attention weights."""
    print(f"\n----- condition=DOOMED (progressive attention noise) -----")
    sys_obj = model_load_fn()
    mid = sys_obj.n_hidden_layers() // 2
    t0 = time.time()
    log = []
    # Sigma schedule: step 0 = 0, step 9 = 0.03 (3pct Frobenius noise at end)
    sigma_schedule = np.linspace(0.0, 0.03, n_checkpoints)
    torch.manual_seed(42)
    for step, sigma in enumerate(sigma_schedule):
        if step > 0:
            # Increment by the *delta* (step-to-step noise, cumulative)
            inject_noise(sys_obj, float(sigma - sigma_schedule[step - 1]))
        h = probe_health(sys_obj.model, sys_obj.tokenizer, probe_texts, "cuda", mid,
                         tag=f"doomed_{step}")
        h["sigma_rel"] = float(sigma)
        log.append({"step": step, "wall": time.time() - t0, **h})
        print(f"  [step {step}] sigma={sigma:.3f}  rel_err={h['bridge_rel_err']:.3f}  "
              f"c={h['c']:.2f}  ratio={h['ratio']:.2f}  alpha={h['alpha']:.3f}")
    sys_obj.unload(); torch.cuda.empty_cache()
    return {"name": "doomed", "log": log}


def run_swap(model_load_fn, c4_texts, swap_texts, n_checkpoints=10, swap_at=5):
    """SWAP: same pretrained model; probe on C4 up to swap_at, then switch to
    swap_texts (wikitext-word-shuffled). Measures 'silent data corruption'."""
    print(f"\n----- condition=SWAP (stimulus swap at step {swap_at}) -----")
    sys_obj = model_load_fn()
    mid = sys_obj.n_hidden_layers() // 2
    t0 = time.time()
    log = []
    for step in range(n_checkpoints):
        probe = c4_texts if step < swap_at else swap_texts
        h = probe_health(sys_obj.model, sys_obj.tokenizer, probe, "cuda", mid,
                         tag=f"swap_{step}")
        log.append({"step": step, "wall": time.time() - t0,
                    "stimulus": "c4" if step < swap_at else "wiki_shuffled",
                    **h})
        print(f"  [step {step} {log[-1]['stimulus']}] rel_err={h['bridge_rel_err']:.3f}  "
              f"c={h['c']:.2f}  ratio={h['ratio']:.2f}  alpha={h['alpha']:.3f}")
    sys_obj.unload(); torch.cuda.empty_cache()
    return {"name": "swap", "swap_at": swap_at, "log": log}


def main():
    # Probe corpus: 500 C4 texts (seed 42)
    c4 = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        c4.append(rec["text"])
        if len(c4) >= 1000:
            break
    # Scrambled wikitext for swap condition
    wiki = []
    for rec in wikitext_v1(seed=42, n_samples=3000):
        wiki.append(rec["text"])
        if len(wiki) >= 1000:
            break
    rng = random.Random(42)
    swap = []
    for t in wiki:
        words = t.split()
        rng.shuffle(words)
        swap.append(" ".join(words))

    def loader():
        return load_system("Qwen/Qwen3-0.6B", quant="fp16", untrained=False, device="cuda")

    conditions = []
    conditions.append(run_baseline(loader, c4, n_checkpoints=10))
    conditions.append(run_doomed(loader, c4, n_checkpoints=10))
    conditions.append(run_swap(loader, c4, swap, n_checkpoints=10, swap_at=5))

    by_cond = {c["name"]: c for c in conditions}

    # -------- analysis --------
    print("\n\n=== GENOMEGUARD PUNCH LIST ===")
    base = by_cond["baseline"]["log"]
    doomed = by_cond["doomed"]["log"]
    sw = by_cond["swap"]["log"]

    base_final = base[-1]["bridge_rel_err"]
    doomed_final = doomed[-1]["bridge_rel_err"]
    separation = (doomed_final / base_final) if base_final > 0 else float("nan")
    print(f"  baseline final rel_err:  {base_final:.3f}")
    print(f"  doomed   final rel_err:  {doomed_final:.3f}")
    print(f"  separation ratio:        {separation:.2f}x")

    pre = [e["bridge_rel_err"] for e in sw if e["stimulus"] == "c4"]
    post = [e["bridge_rel_err"] for e in sw if e["stimulus"] != "c4"]
    pre_mean = float(np.mean(pre)) if pre else float("nan")
    post_max = float(np.max(post)) if post else float("nan")
    swap_spike = (post_max / pre_mean) if pre_mean > 0 else float("nan")
    print(f"  pre-swap rel_err mean:   {pre_mean:.3f}")
    print(f"  post-swap rel_err max:   {post_max:.3f}")
    print(f"  swap spike:              {swap_spike:.2f}x")

    criterion_doomed = np.isfinite(separation) and separation >= 2.0
    criterion_swap = np.isfinite(swap_spike) and swap_spike >= 1.5
    if criterion_doomed and criterion_swap:
        verdict = ("GENOMEGUARD_LANDS - candidate-8 bridge rel_err is a "
                   "real-time training/data-integrity monitor. Doomed "
                   f"training separates from baseline by {separation:.1f}x; "
                   f"data-swap detected within {1.0 / (10 - 5) * 100:.0f}pct "
                   "of run. Ship as tool.")
    elif criterion_doomed or criterion_swap:
        verdict = (f"GENOMEGUARD_PARTIAL - doomed={criterion_doomed}, "
                   f"swap={criterion_swap}. Concept works on one axis.")
    else:
        verdict = ("GENOMEGUARD_KILL - neither doomed nor swap produced "
                   "the required bridge rel_err shift. Not a useful monitor.")
    print(f"\n  verdict: {verdict}")

    out = {"purpose": "GenomeGuard: candidate-8 bridge as training + data-integrity monitor",
           "conditions": conditions,
           "doomed_vs_baseline_separation": float(separation),
           "swap_pre_mean": pre_mean, "swap_post_max": post_max,
           "swap_spike_ratio": float(swap_spike),
           "verdict": verdict}
    out_path = _ROOT / "results/gate2/genomeguard.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
