"""GenomeGuard: training health monitor via candidate-8 bridge rel_err.

Per Codex DeepSeek-moment verdict 2026-04-22 T+45h.

HYPOTHESIS. Trained healthy networks hit c approximately equal to eff_rank/d_rd at
the C4 baseline (candidate-8 bridge, genome_060/063/064, 7/8 PASS at 15pct).
During training, the bridge rel_err should DECREASE as the network's
representations approach healthy trained geometry. Diverging / doomed
runs should show the bridge rel_err growing or flat. Silent data
corruption (wrong stimulus distribution) should show a sudden rel_err
spike. GenomeGuard = a ~200-line monitor that logs bridge rel_err +
k_bulk + alpha every N steps; runs are diagnosed healthy/doomed/corrupt
before standard validation loss diverges.

EXPERIMENT DESIGN. Start from a pretrained Qwen3-0.6B (healthy baseline);
do short LoRA fine-tuning on C4 under 3 conditions:
  (a) BASELINE: stable LR, C4 stimuli - ratio rel_err should stay low.
  (b) DOOMED: LR x5 - representations should distort; rel_err spikes.
  (c) SWAP: start with C4, at step 200 switch to wikitext-word-shuffled
      - silent data corruption; rel_err spikes at step ~200 onwards.

At each 50-step checkpoint we run a cheap probe on a FIXED held-out C4
mini-cloud (n=500) and compute:
  - c           (kNN clustering power-law fit)
  - eff_rank    (participation ratio)
  - d_rd        (k-means rate-distortion dim)
  - ratio       = eff_rank / d_rd
  - rel_err     = |ratio - c| / max(c, 1e-6)
  - bridge_healthy = (rel_err < 0.15)

KILL CONDITIONS (Codex):
  - By step 500, baseline vs doomed bridge_rel_err must differ by >= 2x.
  - Within 200 steps of the data swap, bridge_rel_err must spike
    (>= 1.5x baseline rolling).
  - If neither criterion hits, GenomeGuard is not a useful monitor -> kill.

If both hit: ship GenomeGuard open-source as a small CLI wrapper +
notebook. DeepSeek-moment deliverable.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    Xc = X - X.mean(axis=0)
    return (np.linalg.svd(Xc, compute_uv=False) / np.sqrt(X.shape[0] - 1)).astype(np.float64)


def eff_rank_np(s):
    s2 = s ** 2
    total = s2.sum()
    return 0.0 if total <= 0 else float(total ** 2 / (s2 ** 2).sum())


def fit_alpha_tail(s, lo_frac=0.05, hi_frac=0.5):
    r = np.arange(1, len(s) + 1)
    lo = max(1, int(len(s) * lo_frac))
    hi = int(len(s) * hi_frac)
    slope, _ = np.polyfit(np.log(r[lo:hi]), np.log(s[lo:hi] + 1e-12), 1)
    return float(-slope)


def probe_health(model, tokenizer, probe_texts, device, mid_layer):
    """Run candidate-8 bridge probe on a fixed probe batch. Returns health dict."""
    traj = extract_trajectory(
        model=model, tokenizer=tokenizer,
        texts=probe_texts, layer_indices=[mid_layer],
        pooling="seq_mean", device=device,
        system_key="probe", class_id=1, quantization="fp16",
        stimulus_version="probe", seed=0, batch_size=16, max_length=256,
    )
    X = traj.layers[0].X.astype(np.float32)
    s = spectrum(X)
    er = eff_rank_np(s)
    alpha = fit_alpha_tail(s)
    Cs = [float(knn_clustering_coefficient(X, k=k).value) for k in K_GRID]
    p, _, _ = fit_power_law(K_GRID, Cs)
    rd = rate_distortion_dim(X)
    c = p * rd["d_rd"]
    ratio = er / rd["d_rd"]
    rel_err = abs(ratio - c) / max(c, 1e-6)
    return {"c": float(c), "p": float(p), "d_rd": float(rd["d_rd"]),
            "eff_rank": er, "alpha": alpha, "ratio": ratio,
            "bridge_rel_err": rel_err,
            "bridge_healthy": bool(rel_err < 0.15)}


def measure_val_nll(model, tokenizer, val_texts, device, max_length=256, batch=16):
    model.eval()
    tot = 0.0; ntok = 0
    with torch.no_grad():
        for i in range(0, len(val_texts), batch):
            chunk = val_texts[i:i + batch]
            enc = tokenizer(chunk, return_tensors="pt", truncation=True,
                            padding=True, max_length=max_length).to(device)
            if "labels" not in enc:
                enc["labels"] = enc["input_ids"].clone()
                # mask pad tokens
                if tokenizer.pad_token_id is not None:
                    enc["labels"][enc["attention_mask"] == 0] = -100
            out = model(**enc)
            tot += float(out.loss.item()) * enc["input_ids"].numel()
            ntok += enc["input_ids"].numel()
    model.train()
    return tot / max(ntok, 1)


def make_batches(texts, tokenizer, batch=8, max_length=128, device="cuda"):
    """Tokenize into a list of batches for training."""
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tokenizer(chunk, return_tensors="pt", truncation=True,
                         padding=True, max_length=max_length).to(device)
        enc["labels"] = enc["input_ids"].clone()
        if tokenizer.pad_token_id is not None:
            enc["labels"][enc["attention_mask"] == 0] = -100
        out.append(enc)
    return out


def run_condition(name, condition_texts, n_steps, probe_texts, val_texts,
                  probe_every, lr_mult=1.0, swap_step=None, swap_texts=None,
                  seed=42):
    """Run one condition (baseline/doomed/swap) and log health at each probe step."""
    print(f"\n----- condition={name}  lr_mult={lr_mult}  swap_step={swap_step} -----")
    torch.manual_seed(seed)

    sys_obj = load_system("Qwen/Qwen3-0.6B", quant="fp16", untrained=False, device="cuda")
    mid = sys_obj.n_hidden_layers() // 2

    # Freeze everything except LoRA-style small adapter via trainable LayerNorms only
    # (cheap intervention that still disturbs geometry under doomed LR).
    trainables = []
    for n, p in sys_obj.model.named_parameters():
        if "norm" in n.lower() or "ln" in n.lower():
            p.requires_grad = True
            trainables.append(p)
        else:
            p.requires_grad = False
    opt = torch.optim.AdamW(trainables, lr=1e-4 * lr_mult, betas=(0.9, 0.95))
    print(f"  {len(trainables)} trainable tensor(s) (LayerNorm/RMSNorm only)")

    # Probe at step 0
    t0 = time.time()
    log = []
    base = probe_health(sys_obj.model, sys_obj.tokenizer, probe_texts, "cuda", mid)
    val = measure_val_nll(sys_obj.model, sys_obj.tokenizer, val_texts, "cuda")
    log.append({"step": 0, "wall": 0.0, "val_nll": val, **base})
    print(f"  [step 0] rel_err={base['bridge_rel_err']:.3f}  c={base['c']:.2f}  "
          f"ratio={base['ratio']:.2f}  val_nll={val:.3f}")

    # Training loop
    current_texts = condition_texts
    batches = make_batches(current_texts, sys_obj.tokenizer, batch=8, max_length=128)
    sys_obj.model.train()

    for step in range(1, n_steps + 1):
        # Data swap for silent-failure condition
        if swap_step is not None and step == swap_step and swap_texts is not None:
            print(f"    [step {step}] DATA SWAP: switching to wikitext-scrambled corpus")
            current_texts = swap_texts
            batches = make_batches(current_texts, sys_obj.tokenizer, batch=8, max_length=128)

        b = batches[(step - 1) % len(batches)]
        out = sys_obj.model(**b)
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainables, 1.0)
        opt.step()

        if step % probe_every == 0 or step == n_steps:
            sys_obj.model.eval()
            h = probe_health(sys_obj.model, sys_obj.tokenizer, probe_texts, "cuda", mid)
            val = measure_val_nll(sys_obj.model, sys_obj.tokenizer, val_texts, "cuda")
            sys_obj.model.train()
            log.append({"step": step, "wall": time.time() - t0,
                        "val_nll": val, **h})
            print(f"  [step {step}] rel_err={h['bridge_rel_err']:.3f}  "
                  f"c={h['c']:.2f}  ratio={h['ratio']:.2f}  val_nll={val:.3f}")

    sys_obj.unload(); torch.cuda.empty_cache()
    return {"name": name, "lr_mult": lr_mult, "swap_step": swap_step, "log": log}


def main():
    # Pull fixed probe + val + train C4 texts (same seed 42 -> reproducible)
    all_c4 = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        all_c4.append(rec["text"])
        if len(all_c4) >= 800:
            break
    probe_texts = all_c4[:300]        # fixed probe buffer
    val_texts   = all_c4[300:350]     # small val buffer
    train_texts = all_c4[350:700]     # 350 train texts -> ~44 batches @ batch=8

    # Scrambled wikitext (for data-swap condition)
    wiki = []
    for rec in wikitext_v1(seed=42, n_samples=3000):
        wiki.append(rec["text"])
        if len(wiki) >= 350:
            break
    rng = random.Random(42)
    swap_texts = []
    for t in wiki:
        words = t.split()
        rng.shuffle(words)
        swap_texts.append(" ".join(words))

    n_steps = 300          # short run to stay in compute budget
    probe_every = 30       # 10 probes over 300 steps

    conditions = []
    # (a) Healthy baseline
    conditions.append(run_condition(
        "baseline", train_texts, n_steps, probe_texts, val_texts,
        probe_every, lr_mult=1.0))

    # (b) Doomed: LR x5
    conditions.append(run_condition(
        "doomed_lr_x5", train_texts, n_steps, probe_texts, val_texts,
        probe_every, lr_mult=5.0))

    # (c) Silent data swap at step 150 (out of 300)
    conditions.append(run_condition(
        "data_swap_wiki_shuffled", train_texts, n_steps, probe_texts, val_texts,
        probe_every, lr_mult=1.0, swap_step=150, swap_texts=swap_texts))

    # Analysis
    print("\n\n=== GENOMEGUARD PUNCH LIST ===")
    by_cond = {c["name"]: c for c in conditions}

    def final_rel_err(cond):
        return cond["log"][-1]["bridge_rel_err"]

    def rel_err_at(cond, step):
        for e in cond["log"]:
            if e["step"] >= step:
                return e["bridge_rel_err"]
        return cond["log"][-1]["bridge_rel_err"]

    base_final = final_rel_err(by_cond["baseline"])
    doomed_final = final_rel_err(by_cond["doomed_lr_x5"])
    separation = doomed_final / max(base_final, 1e-6)
    print(f"  baseline final rel_err:     {base_final:.3f}")
    print(f"  doomed   final rel_err:     {doomed_final:.3f}")
    print(f"  doomed / baseline ratio:    {separation:.2f}x")

    # Data-swap detection
    swap_cond = by_cond["data_swap_wiki_shuffled"]
    pre_swap = [e["bridge_rel_err"] for e in swap_cond["log"] if e["step"] <= 150]
    post_swap = [e["bridge_rel_err"] for e in swap_cond["log"] if e["step"] > 150]
    pre_mean = np.mean(pre_swap) if pre_swap else 0
    post_max = np.max(post_swap) if post_swap else 0
    swap_spike = post_max / max(pre_mean, 1e-6)
    print(f"  pre-swap mean rel_err:      {pre_mean:.3f}")
    print(f"  post-swap max rel_err:      {post_max:.3f}")
    print(f"  swap spike ratio:           {swap_spike:.2f}x")

    criterion_doomed = separation >= 2.0
    criterion_swap = swap_spike >= 1.5
    if criterion_doomed and criterion_swap:
        verdict = ("GENOMEGUARD_LANDS - bridge rel_err separates healthy from "
                   "doomed by >=2x AND detects data-swap spike. Ship as tool.")
    elif criterion_doomed or criterion_swap:
        verdict = (f"GENOMEGUARD_PARTIAL - doomed criterion={criterion_doomed}, "
                   f"swap criterion={criterion_swap}. Tighten design.")
    else:
        verdict = ("GENOMEGUARD_KILL - metric does not separate healthy from "
                   "doomed nor detect swap. Not a useful monitor.")
    print(f"\n  verdict: {verdict}")

    out = {"purpose": "GenomeGuard training health monitor via candidate-8 bridge",
           "conditions": conditions,
           "baseline_final_rel_err": base_final,
           "doomed_final_rel_err": doomed_final,
           "doomed_vs_baseline_separation": float(separation),
           "swap_pre_mean": float(pre_mean),
           "swap_post_max": float(post_max),
           "swap_spike_ratio": float(swap_spike),
           "verdict": verdict}
    out_path = _ROOT / "results/gate2/genomeguard.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
