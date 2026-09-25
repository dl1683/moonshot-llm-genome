"""E053b — FINE-GRAIN ONSET a* (R13-mandated remediation of E053's onset stat).

Trigger (THINKING.md T030, R13 flags 2026-09-25 20:05Z): E053's headline
"a* = 63, invariant across scale AND exposure" is BIN-QUANTIZED in 4/5
cells — they fell back to onset_age_binned(), whose oldest-live-bin rule
can only ever return {63, 127, 191, 239} (edges of the 193-255/129-192/
65-128/17-64 bins). Only mid_2.7M got a fine sweep (a* = 86 there).
And the "live fraction ~25%" is the same number re-read (63/255 = 0.247):
potentially ONE statistic, not two.

This run re-measures the onset on the FINE (per-position) grid, ALL cells:
- identical cells / prompts (seed 202, 64 tok) / sampling (T=0.8, top-k 40,
  per-cell seed-7 generator re-init -> aligned streams across cells) /
  fixed anchor to T=256 — exactly E053's Phase-1 protocol; the manual
  forward + onset rule are copied VERBATIM from lab/e053_cache_timeline.py.
- generation runs BATCHED across sequences (identical math per row; gate
  G0b-batch checks batched-vs-sequential logits, prob dev < 1e-4).
- final step (t=255, target = position 255): full 255-position V-zero
  sweep per sequence (E053's fine protocol; K-drop / per-layer / timeline
  / static control are OUT OF SCOPE here).
- a* = youngest cache age with mean per-position dCE < 0.01 nats
  (naive + 5-pt-sustained robust; E053's onset_age verbatim) on the FINE
  grid; bootstrap CI over sequences (resample -> re-apply onset rule).
- IDENTITY AUDIT: a*/255 vs live fraction (fraction of the 255 positions
  with mean dCE >= 0.01) side by side per cell, plus step-consistency
  (fraction of positions whose live/dead label matches the a*-implied
  step). Monotone-recency curves make them coincide; structure beyond
  recency splits them into two statistics.
- ABSOLUTE-VS-PROPORTIONAL: undecidable at fixed ctx-256 (all five cells
  share the window; wpe=256 is a hard limit in these ckpts). Registered
  instead: is fine a* stable across scale/exposure as ABSOLUTE token
  counts? The Phase-2 ctx-512 cell is the decider and is REGISTERED here
  as such (see verdicts.abs_vs_proportional in metrics.json).

Registered predictions (before any measurement):
- PB1 (dequantization): fine a* leaves the 63-bin neighborhood (|a*-63| >
  12) in >= 2/5 cells — the at-63 invariance was partly a bin artifact.
  Alt: fine a* in [51, 75] in >= 4/5 cells (63 is real).
- PB2 (identity): |a*/255 - live_frac| <= 0.02 in >= 4/5 cells -> ONE
  statistic confirmed; else BROKEN (recency structure insufficient).
- PB3 (absolute stability): max/min fine-a* ratio <= 1.6 across all five
  cells at fixed ctx (absolute token counts; NOT the abs-vs-prop claim).

Gates: G1 val-CE anchors (small 1.5581+-0.03, mid 1.6224+-0.03);
G_R reproduction: mid_2.7M run through E053's EXACT sequential protocol
(2 seqs, chunk 96 sweep) must reproduce E053's stored fine sweep
(max |d dCE| < 1e-3); G0b-batch < 1e-4; G4 CPU-only. Adaptive B in
{8, 6, 4} uniform across cells (stream pairing) under e055 CPU
contention; chunks stay at E053's proven sizes.

Run:   python lab/e053b_fine_onset.py
Smoke: E053B_SMOKE=1 python lab/e053b_fine_onset.py
Outputs: runs/e053b/metrics.json + runs/e053b/e053b_fine_onset.png
Budget <= ~12 min CPU wall. No NOTES/THINKING/QUEUE/STATE edits; no
commit (the dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # G4: CPU-only (set pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu"
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import REPO, Cfg, CharCorpus, TinyGPT, estimate_loss, run_dir, save_json

# ------------------------------------------------------------------ constants
SMOKE = os.environ.get("E053B_SMOKE") == "1"
PROMPT_TOK = 64
T_TOTAL = 256                     # wpe=256 hard limit (E053 design §1)
TEMP, TOPK = 0.8, 40
SEED_PROMPT, SEED_SAMPLE = 202, 7
THRESH = 0.01                     # dead-weight threshold (nats) — E053's
A_STAR_K = 5                      # sustained window for robust onset
N_PROMPTS = 2 if SMOKE else 8
BUDGET_S = 300 if SMOKE else 720.0  # soft guard (12 min)
BOOT_N = 200 if SMOKE else 1000
THREADS = 12                      # E053's measured best solo
torch.set_num_threads(THREADS)

CK = REPO / "runs" / "checkpoints"
CELLS = [  # name, ckpt, arch overrides, axis tag, steps trained
    ("small_0.84M", CK / "e005s_small.pt", dict(n_layer=4, n_head=4, n_embd=128), "scale", None),
    ("mid_2.7M",    CK / "e001.pt",        dict(n_layer=6, n_head=6, n_embd=192), "scale+exposure", 4000),
    ("large_10M",   CK / "e005s_large.pt", dict(n_layer=8, n_head=8, n_embd=320), "scale", None),
    ("exp_d400",    CK / "e048_direct400.pt", dict(n_layer=6, n_head=6, n_embd=192), "exposure", 400),
    ("exp_d800",    CK / "e048_direct800.pt", dict(n_layer=6, n_head=6, n_embd=192), "exposure", 800),
]
SMOKE_CELLS = {"small_0.84M", "mid_2.7M"}
ANCHORS = {"small_0.84M": (1.5581, 0.03), "mid_2.7M": (1.622391, 0.03)}
CHUNK = {4: 255, 6: 96, 8: 48}     # E053's proven sweep chunks (by n_head)
E053_METRICS = REPO / "runs" / "e053" / "metrics.json"
E053_FALLBACK_A = 63               # the bin-quantized value under audit
T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ------------------------------------------------- machinery (VERBATIM E053)

@torch.no_grad()
def _manual_chunk(net: TinyGPT, idxs, vzero, kdrop, layers):
    """Exact manual forward, returns LAST-position logits (N, vocab)."""
    N, T = idxs.shape
    pos = torch.arange(T)
    x = net.wte(idxs) + net.wpe(pos).unsqueeze(0)
    H = net.cfg.n_head
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    nondiag = ~torch.eye(T, dtype=torch.bool)
    for li, blk in enumerate(net.h):
        active = layers is None or li in layers
        xh = blk.ln1(x)
        qkv = blk.attn.c_attn(xh)
        C = qkv.shape[-1] // 3
        d = C // H
        q, k, v = qkv.split(C, dim=2)
        q = q.view(N, T, H, d).transpose(1, 2)
        k = k.view(N, T, H, d).transpose(1, 2)
        v = v.view(N, T, H, d).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
        att = att.masked_fill(causal, float("-inf"))
        if active and kdrop is not None and bool(kdrop.any()):
            att = att.masked_fill(kdrop[:, None, None, :] & nondiag, float("-inf"))
        probs = torch.softmax(att, dim=-1)
        if active and vzero is not None and bool(vzero.any()):
            v = v.masked_fill(vzero[:, None, :, None], 0.0)
        y = (probs @ v).transpose(1, 2).reshape(N, T, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
    return net.lm_head(net.ln_f(x[:, -1, :]))


@torch.no_grad()
def manual_logits(net, idxs, vzero=None, kdrop=None, layers=None, chunk=64):
    outs = []
    for i in range(0, idxs.shape[0], chunk):
        outs.append(_manual_chunk(net, idxs[i:i + chunk],
                                  vzero[i:i + chunk] if vzero is not None else None,
                                  kdrop[i:i + chunk] if kdrop is not None else None,
                                  layers))
    return torch.cat(outs, 0)


def sample_and_ce(logits_clean: torch.Tensor, gen: torch.Generator):
    """Sample (temp 0.8, top-k 40) from filtered probs; CE from FULL softmax."""
    lg = logits_clean / TEMP
    v, _ = torch.topk(lg, TOPK)
    lg_f = lg.masked_fill(lg < v[-1], float("-inf"))
    tok = int(torch.multinomial(torch.softmax(lg_f, -1), 1, generator=gen))
    p_full = torch.softmax(logits_clean, -1)
    return tok, float(-math.log(max(p_full[tok].item(), 1e-12)))


def boot_ci(X, n: int = 1000, seed: int = 0):
    """Bootstrap CI over the sequence axis. X: (n_seq, ...) -> (lo, hi)."""
    X = np.asarray(X, float)
    if X.shape[0] < 2:
        return X[0], X[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, X.shape[0], size=(n, X.shape[0]))
    means = X[idx].mean(1)
    return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


def onset_age(dce_by_age: np.ndarray, k: int = A_STAR_K, thresh: float = THRESH):
    """dce_by_age: ages ascending (1=youngest ... 255=oldest/sink).
    Returns (naive, robust): youngest age whose mean dCE < thresh; robust
    requires a k-length run of 5-pt moving-mean values below thresh.
    [VERBATIM from E053 — the statistic under remediation.]"""
    m = np.asarray(dce_by_age, float)
    below = m < thresh
    naive = int(np.argmax(below)) + 1 if below.any() else None
    if len(m) < k or not below.any():
        return naive, None
    ma = np.convolve(m, np.ones(k) / k, mode="valid")
    ok = ma < thresh
    run, robust_i = 0, None
    for i in range(len(ok) - 1, -1, -1):
        run = run + 1 if ok[i] else 0
        if run >= k:
            robust_i = i
    robust = int(robust_i) + 1 if robust_i is not None else None
    return naive, robust


def a_star_rule(mean_age: np.ndarray):
    """E053's reported rule: robust preferred, naive fallback."""
    naive, robust = onset_age(mean_age)
    a = robust if robust is not None else naive
    return a, naive, robust


# ------------------------------------------- sequential path (E053 exact; G_R)

@torch.no_grad()
def prefill(net: TinyGPT, idx: torch.Tensor):
    """Manual prefill of the prompt; returns last-position logits (V,) and the
    per-layer KV cache [(k, v)] each (1, H, T, d). [VERBATIM E053]"""
    T = idx.shape[0]
    x = net.wte(idx[None]) + net.wpe(torch.arange(T))[None]
    H = net.cfg.n_head
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    kv = []
    for blk in net.h:
        xh = blk.ln1(x)
        qkv = blk.attn.c_attn(xh)
        C = qkv.shape[-1] // 3
        d = C // H
        q, k, v = qkv.split(C, dim=2)
        q = q.view(1, T, H, d).transpose(1, 2)
        k = k.view(1, T, H, d).transpose(1, 2)
        v = v.view(1, T, H, d).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
        att = att.masked_fill(causal, float("-inf"))
        y = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(1, T, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
        kv.append((k, v))
    return net.lm_head(net.ln_f(x[0, -1])), kv


@torch.no_grad()
def decode_step(net: TinyGPT, tok: int, pos: int, kv: list):
    """Incremental decode of one token at position pos; extends kv in place.
    [VERBATIM E053]"""
    x = net.wte(torch.tensor([tok])) + net.wpe(torch.tensor([pos]))
    H = net.cfg.n_head
    for li, blk in enumerate(net.h):
        xh = blk.ln1(x)
        qkv = blk.attn.c_attn(xh)
        C = qkv.shape[-1] // 3
        d = C // H
        q, k, v = qkv.split(C, dim=1)               # x is (1, C) here
        q = q.view(1, 1, H, d).transpose(1, 2)
        k = k.view(1, 1, H, d).transpose(1, 2)
        v = v.view(1, 1, H, d).transpose(1, 2)
        kp, vp = kv[li]
        k = torch.cat([kp, k], dim=2)
        v = torch.cat([vp, v], dim=2)
        kv[li] = (k, v)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))   # (1,H,1,t+1)
        y = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(1, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
    return net.lm_head(net.ln_f(x[0]))                  # (V,)


@torch.no_grad()
def run_sequence_exact(net, prompt: torch.Tensor, gen: torch.Generator, ch: int):
    """E053's run_sequence, sweep-only (v_sweep_full branch): free-run 64->256
    with the sequential decoder, then the full 255-position V-zero sweep.
    Used ONLY for the mid_2.7M reproduction gate G_R."""
    idx = prompt.clone()
    logits, kv = prefill(net, idx)
    for t in range(PROMPT_TOK, T_TOTAL):
        tok, _ = sample_and_ce(logits, gen)
        idx = torch.cat([idx, torch.tensor([tok])])
        if t < T_TOTAL - 1:
            logits = decode_step(net, tok, t, kv)
    ctx, tgt = idx[:T_TOTAL - 1], int(idx[T_TOTAL - 1])
    P = ctx.shape[0]
    p_clean = torch.softmax(logits, -1)
    ce_clean = float(-math.log(max(p_clean[tgt].item(), 1e-12)))
    idxs = ctx[None].repeat(P, 1)
    eye = torch.eye(P, dtype=torch.bool)
    lp = torch.log_softmax(manual_logits(net, idxs, eye, None, None, chunk=ch).float(), -1)
    sweep = (-ce_clean - lp[torch.arange(P), tgt]).numpy()
    return sweep, logits, ctx


# ------------------------------------------------- batched path (main measure)

@torch.no_grad()
def prefill_batch(net: TinyGPT, idx: torch.Tensor):
    """Batched prefill, (B, Tp) -> last-position logits (B, V) + KV cache."""
    B, T = idx.shape
    x = net.wte(idx) + net.wpe(torch.arange(T))
    H = net.cfg.n_head
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    kv = []
    for blk in net.h:
        xh = blk.ln1(x)
        qkv = blk.attn.c_attn(xh)
        C = qkv.shape[-1] // 3
        d = C // H
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, H, d).transpose(1, 2)
        k = k.view(B, T, H, d).transpose(1, 2)
        v = v.view(B, T, H, d).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
        att = att.masked_fill(causal, float("-inf"))
        y = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(B, T, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
        kv.append((k, v))
    return net.lm_head(net.ln_f(x[:, -1, :])), kv


@torch.no_grad()
def decode_step_batch(net: TinyGPT, toks: torch.Tensor, pos: int, kv: list):
    """Batched incremental decode: (B,) tokens at position pos -> (B, V)."""
    B = toks.shape[0]
    x = net.wte(toks) + net.wpe(torch.full((B,), pos))
    H = net.cfg.n_head
    for li, blk in enumerate(net.h):
        xh = blk.ln1(x)
        qkv = blk.attn.c_attn(xh)
        C = qkv.shape[-1] // 3
        d = C // H
        q, k, v = qkv.split(C, dim=1)
        q = q.view(B, 1, H, d).transpose(1, 2)
        k = k.view(B, 1, H, d).transpose(1, 2)
        v = v.view(B, 1, H, d).transpose(1, 2)
        kp, vp = kv[li]
        k = torch.cat([kp, k], dim=2)
        v = torch.cat([vp, v], dim=2)
        kv[li] = (k, v)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))   # (B,H,1,t+1)
        y = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(B, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
    return net.lm_head(net.ln_f(x))


@torch.no_grad()
def generate_batch(net, prompts, gen: torch.Generator):
    """Free-run 64->256 for B sequences at once, fixed anchor. Sampling is
    per-row from the SHARED generator in row order per step -> the per-cell
    stream is aligned across cells for identical B (E053's pairing).
    Returns idx (B, 256) and the final (position-255) logits (B, V)."""
    B = len(prompts)
    idx = torch.stack(prompts)
    logits, kv = prefill_batch(net, idx)
    for t in range(PROMPT_TOK, T_TOTAL):
        toks = torch.zeros(B, dtype=torch.long)
        for j in range(B):
            tok, _ = sample_and_ce(logits[j], gen)
            toks[j] = tok
        idx = torch.cat([idx, toks[:, None]], 1)
        if t < T_TOTAL - 1:
            logits = decode_step_batch(net, toks, t, kv)
    return idx, logits


@torch.no_grad()
def fine_sweep(net, idx: torch.Tensor, final_logits: torch.Tensor, ch: int):
    """Final-step fine V-zero sweep for all B sequences at once.
    Returns dce (B, 255) indexed by POSITION (0..254); age of pos p =
    255 - p, so the age-ascending view is dce[:, ::-1]. [E053 fine math]"""
    B = idx.shape[0]
    ctx, tgt = idx[:, :-1], idx[:, -1]                 # (B, 255), (B,)
    P = T_TOTAL - 1
    p_clean = torch.softmax(final_logits.float(), -1)
    ce_clean = -torch.log(p_clean[torch.arange(B), tgt].clamp_min(1e-12))
    idxs = ctx.repeat_interleave(P, dim=0)             # (B*P, P)
    vz = torch.eye(P, dtype=torch.bool).repeat(B, 1)   # row b*P+p: zero pos p
    les = manual_logits(net, idxs, vz, None, None, chunk=ch)
    lp = torch.log_softmax(les.float(), -1)
    rows = torch.arange(B * P)
    dce = (-ce_clean.repeat_interleave(P) - lp[rows, tgt.repeat_interleave(P)])
    return dce.view(B, P).numpy()


def load_cell(name: str, ckpt: Path, arch: dict):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    cfg = Cfg(vocab=65, block_size=256, **arch)
    net = TinyGPT(cfg)
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net


# ------------------------------------------------------------------ analysis

def bootstrap_stat(dce_age: np.ndarray, stat, n: int = BOOT_N, seed: int = 0):
    """Bootstrap any scalar stat over the sequence axis."""
    S = dce_age.shape[0]
    if S < 2:
        v = stat(dce_age)
        return (v, v) if v is not None else (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sel = rng.integers(0, S, S)
        vals.append(stat(dce_age[sel]))
    vals = np.asarray([np.nan if v is None else v for v in vals], float)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def analyze_cell(sweep_pos: np.ndarray):
    """sweep_pos: (S, 255) V-zero dCE by position. Returns the derived block."""
    dce_age = sweep_pos[:, ::-1]                       # (S, 255) young->old
    mean_age = dce_age.mean(0)
    ages = np.arange(1, T_TOTAL)                       # 1..255
    a_star, naive, robust = a_star_rule(mean_age)
    a_lo, a_hi = bootstrap_stat(dce_age, lambda m: a_star_rule(m.mean(0))[0])
    live = mean_age >= THRESH
    live_frac = float(live.mean())
    lf_lo, lf_hi = bootstrap_stat(dce_age, lambda m: float((m.mean(0) >= THRESH).mean()))
    frac = (a_star / (T_TOTAL - 1)) if a_star is not None else None
    if a_star is not None:
        step_live = ages < a_star                      # a*-implied step function
        match = float((step_live == live).mean())
        junk_old = float((dce_age[:, a_star - 1:] <= -THRESH).mean())
    else:
        match = junk_old = None
    return dict(
        a_star=a_star, a_star_naive=naive, a_star_robust=robust,
        a_star_ci=[a_lo, a_hi], a_star_frac=frac,
        live_frac=live_frac, live_frac_ci=[lf_lo, lf_hi],
        identity_gap=(frac - live_frac) if frac is not None else None,
        one_statistic=bool(frac is not None and abs(frac - live_frac) <= 0.02),
        step_match_frac=match,
        junk_frac_all=float((dce_age <= -THRESH).mean()),
        junk_frac_beyond_ast=junk_old,
        monotone_frac=float(np.mean(np.diff(mean_age) < 0)),
        per_seq_a_star=[a_star_rule(dce_age[s])[0] for s in range(dce_age.shape[0])],
        mean_age_curve=mean_age, ci=boot_ci(dce_age, BOOT_N),
        dce_age=dce_age,
    )


# ---------------------------------------------------------------------- main

def main():
    assert torch.cuda.device_count() == 0, "G4 violated: CUDA device visible"
    out_dir = run_dir("e053b")
    corp = CharCorpus(REPO / "data" / "input.txt")
    assert corp.vocab_size == 65

    gen_p = torch.Generator().manual_seed(SEED_PROMPT)
    ix = torch.randint(len(corp.val) - PROMPT_TOK - 1, (N_PROMPTS,), generator=gen_p)
    prompts = [corp.val[i:i + PROMPT_TOK] for i in ix]
    log(f"{N_PROMPTS} prompts (seed {SEED_PROMPT}); prefix matches E053's stored "
        f"seq0: {corp.decode(prompts[0])[:32]!r}")

    cells = [c for c in CELLS if not SMOKE or c[0] in SMOKE_CELLS]
    nets, costs, gates = {}, {}, dict(G1={}, G4_cpu_only=True, threads=THREADS, smoke=SMOKE)

    # ---- phase A: load, anchor, measure chunk costs -> pick uniform B
    for name, ckpt, arch, axis, steps in cells:
        net = load_cell(name, ckpt, arch)
        nets[name] = net
        ch = CHUNK[net.cfg.n_head]
        val_ce = estimate_loss(net, corp, "val", n_batches=8)
        if name in ANCHORS:
            a, tol = ANCHORS[name]
            gates["G1"][name] = dict(val_ce=val_ce, anchor=a, ok=abs(val_ce - a) <= tol)
        else:
            gates["G1"][name] = dict(val_ce=val_ce, anchor=None, ok=None)
        # time one sweep chunk (P=255 rows of context)
        dummy = torch.randint(65, (ch, T_TOTAL - 1))
        vz = torch.zeros(ch, T_TOTAL - 1, dtype=torch.bool)
        vz[:, 10:20] = True
        ts = []
        with torch.no_grad():
            for _ in range(2):
                t0 = time.time()
                manual_logits(net, dummy, vz, None, None, chunk=ch)
                ts.append(time.time() - t0)
        t_chunk = max(ts)
        costs[name] = dict(t_chunk=t_chunk, ch=ch, val_ce=val_ce)
        log(f"{name}: val CE {val_ce:.4f} | chunk({ch}) {t_chunk:.2f}s "
            f"| G1 {'ok' if gates['G1'][name]['ok'] is not False else 'FAIL'}")

    repro_s = 0.0
    if "mid_2.7M" in nets:
        # sequential gen is slow (measured ~30-60s/seq under contention)
        repro_s = 2 * (costs["mid_2.7M"]["t_chunk"] * math.ceil(255 / costs["mid_2.7M"]["ch"])
                       + 35.0)

    def project(B: int) -> float:
        tot = repro_s
        for name in nets:
            c = costs[name]
            tot += 60.0 + c["t_chunk"] * math.ceil((B * (T_TOTAL - 1)) / c["ch"])
        return tot  # generation ~60s/cell (batched, measured in smoke) + sweep chunks

    rem = BUDGET_S - elapsed()
    B = 2
    for cand in (8, 6, 4):
        if cand <= N_PROMPTS and project(cand) <= 0.85 * rem:
            B = cand
            break
    else:
        B = min(4, N_PROMPTS)
    B = max(B, 2)
    log(f"protocol: B={B} uniform (projected {project(B):.0f}s vs remaining {rem:.0f}s)")

    # ---- phase B: G_R reproduction gate on mid_2.7M (E053's exact protocol)
    gr = None
    if "mid_2.7M" in nets and E053_METRICS.exists():
        e053 = json.loads(E053_METRICS.read_text())
        stored = e053["cells"].get("mid_2.7M", {}).get("final_profile", {})
        if stored.get("vzero_dce_mean"):
            net = nets["mid_2.7M"]
            ch = costs["mid_2.7M"]["ch"]
            gen = torch.Generator().manual_seed(SEED_SAMPLE)  # E053's per-cell seed
            sweeps, seq_logits, seq_ctx = [], [], []
            for si in range(2):
                sw, lg, cx = run_sequence_exact(net, prompts[si], gen, ch)
                sweeps.append(sw)
                seq_logits.append(lg)
                seq_ctx.append(cx)
                log(f"  G_R mid seq{si} swept (naive a* seq: "
                    f"{a_star_rule(sw[::-1])[0]})")
            mean2 = np.mean(sweeps, 0)
            stored_m = np.asarray(stored["vzero_dce_mean"], float)
            dev = float(np.abs(mean2 - stored_m).max())
            a2 = a_star_rule(mean2[::-1])[0]
            # G0b-batch: batched decoder vs sequential on the SAME tokens
            bl, _ = prefill_batch(net, torch.stack(seq_ctx)[:, :PROMPT_TOK])
            kv = None
            x = torch.stack(seq_ctx)
            with torch.no_grad():
                logits_b, kv = prefill_batch(net, x[:, :PROMPT_TOK])
                for t in range(PROMPT_TOK, T_TOTAL - 1):
                    logits_b = decode_step_batch(net, x[:, t], t, kv)
            p0 = torch.softmax(torch.stack(seq_logits).float(), -1)
            p1 = torch.softmax(logits_b.float(), -1)
            dev_b = float((p0 - p1).abs().max())
            gr = dict(max_abs_dce_dev=dev, reproduced_a_star=a2,
                      stored_a_star=e053["cells"]["mid_2.7M"]["derived"]["a_star"],
                      batch_vs_seq_prob_dev=dev_b,
                      ok=bool(dev < 1e-3 and dev_b < 1e-4))
            gates["G_R"] = gr
            log(f"G_R: max|d dCE| vs stored {dev:.2e} | a*(2 seqs) {a2} vs stored "
                f"{gr['stored_a_star']} | batch dev {dev_b:.2e} -> "
                f"{'PASS' if gr['ok'] else 'FAIL'}")

    # ---- phase C: the fine measurement, all cells, uniform B
    results = {}
    for name, ckpt, arch, axis, steps in cells:
        net, ch = nets[name], costs[name]["ch"]
        gen = torch.Generator().manual_seed(SEED_SAMPLE)  # per-cell re-init (pairing)
        t0c = elapsed()
        idx, final_logits = generate_batch(net, prompts[:B], gen)
        sweep = fine_sweep(net, idx, final_logits, ch)
        nan_n = int(np.isnan(sweep).sum())
        if nan_n:
            sweep = np.nan_to_num(sweep, nan=0.0)
        d = analyze_cell(sweep)
        results[name] = dict(
            ckpt=str(ckpt), arch=dict(**arch, block_size=256, vocab=65),
            axis=axis, steps_trained=steps, val_ce=costs[name]["val_ce"],
            protocol=dict(B=B, chunk=ch, fine_grid=True, nan_count=nan_n),
            age=list(range(1, T_TOTAL)),
            vzero_dce_mean=d["mean_age_curve"].tolist(),
            vzero_ci=[d["ci"][0].tolist(), d["ci"][1].tolist()],
            per_seq_sweeps_by_age=d["dce_age"].tolist(),
            derived={k: v for k, v in d.items()
                     if k not in ("mean_age_curve", "ci", "dce_age")},
        )
        log(f"{name} DONE a*={d['a_star']} CI[{d['a_star_ci'][0]:.0f},"
            f"{d['a_star_ci'][1]:.0f}] naive={d['a_star_naive']} "
            f"live_frac={d['live_frac']:.3f} gap={d['identity_gap']:+.3f} "
            f"match={d['step_match_frac']} ({elapsed() - t0c:.0f}s)")

    # ---- verdicts
    fb = {n: results[n]["derived"]["a_star"] for n in results}
    fb = {n: v for n, v in fb.items() if v is not None}
    ratios = [max(fb.values()) / min(fb.values())] if len(set(fb.values())) > 1 and fb else [1.0]
    ratio = float(ratios[0])
    scale_names = [n for n in results if results[n]["axis"].startswith("scale")]
    expo_names = ["exp_d400", "exp_d800", "mid_2.7M"]
    def sub(names):
        vals = [fb[n] for n in names if n in fb]
        return dict(values={n: fb.get(n) for n in names},
                    maxmin_ratio=float(max(vals) / min(vals)) if len(vals) > 1 else None)
    n_dequant = int(sum(1 for n, v in fb.items() if abs(v - E053_FALLBACK_A) > 12))
    n_in63 = int(sum(1 for v in fb.values() if abs(v - E053_FALLBACK_A) <= 12))
    n_onestat = int(sum(1 for n in results if results[n]["derived"]["one_statistic"]))
    pb1 = dict(n_dequantized=n_dequant, n_within_63pm12=n_in63,
               pass_dequant=bool(n_dequant >= 2), pass_real63=bool(n_in63 >= 4))
    pb2 = dict(per_cell={n: results[n]["derived"]["identity_gap"] for n in results},
               n_one_statistic=n_onestat,
               **{"pass": bool(n_onestat >= max(1, len(results) - 1))})
    pb3 = dict(a_star=fb, maxmin_ratio=ratio, **{"pass": bool(ratio <= 1.6)})
    abs_vs_prop = dict(
        decidable_here=False,
        reason="all five cells share ctx-256 (wpe=256 hard limit); absolute vs "
               "proportional position cannot be separated at a single window",
        absolute_stability=dict(all_cells=sub(list(results)),
                                scale=sub(scale_names), exposure=sub(expo_names)),
        registered_decider="Phase-2 ctx-512 cell (same arch family retrained with "
                           "block_size=512): a*(ctx512) ~ a*(ctx256) in tokens -> "
                           "ABSOLUTE; a*(ctx512) ~ frac x 512 -> PROPORTIONAL. "
                           "Registered at e053b (R13 remediation).")

    metrics = dict(
        experiment="e053b_fine_onset", phase=1,
        purpose="R13 remediation: fine-grid onset a* on ALL cells (E053's 63 was "
                "bin-quantized in 4/5); a*/255 vs live-fraction identity audit",
        started=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        wall_s=elapsed(), smoke=SMOKE, threads=THREADS,
        seeds=dict(prompts=SEED_PROMPT, sampling=SEED_SAMPLE),
        protocol=dict(B=B, prompt_tokens=PROMPT_TOK, t_total=T_TOTAL, temp=TEMP,
                      topk=TOPK, fixed_anchor=True, threshold=THRESH,
                      a_star_window=A_STAR_K, boot_n=BOOT_N,
                      generation="batched across sequences (per-row draws from the "
                                 "shared seed-7 generator; streams aligned across "
                                 "cells at equal B)"),
        gates=gates,
        e053_reference=dict(fallback_a_star=E053_FALLBACK_A,
                            mid_fine_a_star=(gr or {}).get("stored_a_star")),
        cells=results,
        verdicts=dict(PB1_dequantization=pb1, PB2_identity=pb2,
                      PB3_absolute_stability=pb3, abs_vs_proportional=abs_vs_prop),
    )
    save_json(out_dir / "metrics.json", metrics)
    log(f"metrics.json written ({len(results)} cells, B={B})")
    if results:
        plot(out_dir / "e053b_fine_onset.png", results, fb, pb1, pb2, pb3)
        log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot

def plot(path: Path, R: dict, fb: dict, pb1, pb2, pb3):
    order = [c[0] for c in CELLS if c[0] in R]
    colors = {"small_0.84M": "tab:blue", "mid_2.7M": "tab:green", "large_10M": "tab:red",
              "exp_d400": "tab:orange", "exp_d800": "tab:purple"}
    fig = plt.figure(figsize=(19, 12))
    gs = fig.add_gridspec(3, 6, hspace=0.45, wspace=0.5)

    # row 1: fine per-position V-zero dCE vs cache age + a* per cell + overlay
    for i, n in enumerate(order[:5]):
        ax = fig.add_subplot(gs[0, i])
        c = R[n]
        age = np.asarray(c["age"])
        m = np.asarray(c["vzero_dce_mean"])
        lo, hi = np.asarray(c["vzero_ci"][0]), np.asarray(c["vzero_ci"][1])
        ax.fill_between(age, lo, hi, alpha=0.25, color=colors[n])
        ax.plot(age, m, lw=1.4, color=colors[n])
        ax.axhline(0, color="k", lw=0.6)
        ax.axhline(THRESH, color="gray", ls=":", lw=0.9)
        d = c["derived"]
        if d["a_star"]:
            ax.axvline(d["a_star"], color=colors[n], ls="--", lw=1.2)
        ax.set_xlim(0, 256)
        ax.set_ylim(min(-0.1, np.nanpercentile(m, 2) - 0.05),
                    np.nanpercentile(hi, 98))
        ax.set_title(f"{n}\nfine a*={d['a_star']} "
                     f"CI[{d['a_star_ci'][0]:.0f},{d['a_star_ci'][1]:.0f}] "
                     f"live={d['live_frac']:.3f}", fontsize=9)
        ax.set_xlabel("cache age (tokens)")
        if i == 0:
            ax.set_ylabel("V-zero dCE (nats)")
    ax = fig.add_subplot(gs[0, 5])
    for n in order:
        ax.plot(R[n]["age"], R[n]["vzero_dce_mean"], lw=1.3, color=colors[n],
                ls="--" if R[n]["axis"] == "exposure" else "-", label=n)
    ax.axhline(THRESH, color="gray", ls=":", lw=0.9)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlim(0, 256)
    ax.legend(fontsize=7)
    ax.set_title("overlay (solid=scale, dash=exposure)", fontsize=9)

    # row 2: a* bars + CI; the identity audit; per-seq spread; verdict text
    ax = fig.add_subplot(gs[1, 0:2])
    names = [n for n in order if n in fb]
    ys = [fb[n] for n in names]
    errs = [[max(0, fb[n] - R[n]["derived"]["a_star_ci"][0]) for n in names],
            [max(0, R[n]["derived"]["a_star_ci"][1] - fb[n]) for n in names]]
    ax.bar(names, ys, yerr=errs, color=[colors[n] for n in names], alpha=0.8,
           capsize=4)
    ax.axhline(E053_FALLBACK_A, color="k", ls=":", lw=1.2)
    ax.text(0.02, E053_FALLBACK_A + 2, "E053 binned fallback 63", fontsize=8,
            transform=ax.get_yaxis_transform())
    ax.set_ylabel("fine onset age a* (tokens)")
    ax.set_title("P2 core — fine-grid a* (bootstrap CI over seqs)", fontsize=9)
    ax.tick_params(axis="x", rotation=30)

    ax = fig.add_subplot(gs[1, 2:4])
    x = np.arange(len(names))
    fr = [R[n]["derived"]["a_star_frac"] for n in names]
    lf = [R[n]["derived"]["live_frac"] for n in names]
    ax.bar(x - 0.18, fr, 0.36, label="a*/255", color="gray", alpha=0.8)
    ax.bar(x + 0.18, lf, 0.36, label="live fraction (dCE>=0.01)", color="tab:cyan",
           alpha=0.8)
    ax.set_xticks(x, names, rotation=30, fontsize=8)
    for i, n in enumerate(names):
        gap = R[n]["derived"]["identity_gap"]
        ax.text(i, max(fr[i], lf[i]) + 0.012, f"{gap:+.3f}", ha="center", fontsize=7)
    ax.legend(fontsize=8)
    ax.set_title(f"identity audit — ONE statistic? {pb2['n_one_statistic']}"
                 f"/{len(names)} cells |gap|<=0.02", fontsize=9)

    ax = fig.add_subplot(gs[1, 4:6])
    for i, n in enumerate(order):
        psa = R[n]["derived"]["per_seq_a_star"]
        ax.scatter([i] * len(psa), psa, s=22, color=colors[n], alpha=0.7,
                   edgecolors="k", linewidths=0.4, zorder=3)
        if n in fb:
            ax.scatter([i], [fb[n]], marker="D", s=70, color=colors[n], zorder=4)
    ax.axhline(E053_FALLBACK_A, color="k", ls=":", lw=1.2)
    ax.set_xticks(range(len(order)), order, rotation=30, fontsize=8)
    ax.set_ylabel("a* (per-seq dots, cell mean diamond)")
    ax.set_title("per-sequence onset spread", fontsize=9)

    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    lines = [
        f"PB1 de-quantization: {pb1['n_dequantized']}/5 cells leave 63+-12 "
        f"(pass>=2: {pb1['pass_dequant']}); within: {pb1['n_within_63pm12']} "
        f"(pass>=4: {pb1['pass_real63']})",
        f"PB2 one-statistic identity: {pb2['n_one_statistic']}/{len(order)} cells "
        f"|a*/255 - live| <= 0.02 -> {'CONFIRMED' if pb2['pass'] else 'BROKEN'}; "
        f"gaps " + ", ".join(f"{n}:{pb2['per_cell'][n]:+.3f}" for n in pb2["per_cell"]),
        f"PB3 absolute stability (fixed ctx-256): max/min = "
        f"{pb3['maxmin_ratio']:.2f} -> {'STABLE' if pb3['pass'] else 'NOT STABLE'}; "
        + ", ".join(f"{n}={fb.get(n)}" for n in fb),
        "ABSOLUTE-VS-PROPORTIONAL: undecidable at ctx-256 (all cells share the "
        "window). Registered decider: Phase-2 ctx-512 cell — a*(512)~a*(256) "
        "tokens => absolute; a*(512)~frac*512 => proportional.",
    ]
    ax.text(0.02, 0.95, "E053b — fine-onset verdicts (R13 remediation)",
            fontsize=13, weight="bold", va="top")
    for i, t in enumerate(lines):
        ax.text(0.02, 0.72 - i * 0.17, t, fontsize=10, va="top", family="monospace")

    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
