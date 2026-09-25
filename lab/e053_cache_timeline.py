"""E053 — the CACHE UTILITY TIMELINE (frontier candidate A / T025, Phase 1).

Design: scratch/e053_design.md (REGISTERED before this run — follow exactly).
Question: when does a cached K/V entry become dead weight during generation?
Per-position CAUSAL utility over cache age — the curve nobody has published
at any scale (sinks are known; the age timeline is not).

Protocol (Phase 1, 100% CPU, G4 — e050 may be running concurrently):
- Cells: scale axis {e005s_small 0.84M / e001 2.7M @4000 / e005s_large 10M}
  x exposure axis {e048_direct400 / e048_direct800 / e001@4000} (2.7M arch;
  lineage caveat registered in the design: different runs, same arch family).
- 8 fixed val prompts (64 tok, seed 202); free-run generation to T=256 with
  FIXED anchor (no sliding window; position identity constant — matches
  non-windowed KV-cache semantics; wpe=256 hard limit). Sampling T=0.8,
  top-k 40, seed 7 (re-init per cell -> paired across cells).
- Interventions, exact, in a manual-attention forward (hook-free):
  V-zero(pos)  = wipe cached content at all layers        [primary]
  K-drop(pos)  = softmax column -inf, DIAGONAL PRESERVED   [secondary, G3]
  both         = entry deletion (equivalence: both ~= K-drop)
- Per measured step: binned dCE of the actually-sampled token, bins
  {p0, 1-16, 17-64, 65-128, 129-192, 193-255} + a last-64 recency row (G2).
- Final step (t=255, target = position 255): per-position sweep (V-zero
  primary + K-drop), per-layer V-zero decomposition (scale cells), binned
  {vzero, kdrop, both}.
- Static teacher-forced control: same bins on real-text windows.
- ADAPTIVE PROTOCOL: e050's CPU assay shares this machine and slows forwards
  up to ~25x vs the design's calibration. Per cell we MEASURE the batched
  forward cost and fit (n_seq, timeline stride, sweep detail) to the budget,
  using the design's registered fallbacks (4 seqs / every-2nd-step timeline /
  binned 10M sweep). The chosen protocol is recorded per cell in metrics.json.

Derived: onset age a* (youngest cache age with mean per-position dCE <
0.01 nats; naive + 5-pt-sustained robust; binned fallback if no fine sweep),
junk fraction (dCE <= -0.01: lesion HELPS), primacy/recency structure.

Registered predictions (from the design, REVISED after the CPU probes):
- P1 shape: per-position utility at T=256 monotone in recency + weak primacy
  bump (bump <= 25% of recent plateau); sink dCE < 0.05 at age>100 in >=3
  models (sink DEAD for generation). Alt kept alive: sink dCE >= 0.3.
- P2 onset: a* as a fraction of context is scale-INVARIANT (within +-20%)
  but grows with exposure (400 -> 4000 steps) by >= 1.5x in tokens.
- P3 sink accumulator, three-way: sink V-zero dCE trajectory over generation
  GROW / FLAT (|change| < 10% of initial) / DECAY (< 20% of initial by
  mid-generation). Micro-preview (n=1) said DECAY with sign flip.
- P4 junk: among positions older than a*, >= 5% with dCE <= -0.01 (dilution)
  else old entries are dead-but-harmless (pure evictability).

Gates: G0 manual == SDPA (max prob dev < 1e-4); G1 ckpt val CE within known
anchors (small 1.5581+-0.03, e001 1.6224+-0.03); G2 last-64 recency V-zero
dCE > +0.5 at every measured step (bin-based count also recorded); G3 K-drop
diagonal -> no NaN (any NaN skipped+counted); G4 no GPU (CUDA hidden +
is_available patched before common import), no training, no new automations.

Run:   python lab/e053_cache_timeline.py
Smoke: E053_SMOKE=1 python lab/e053_cache_timeline.py
Outputs: runs/e053/metrics.json + runs/e053/e053_utility_timeline.png
Budget <= 15 min CPU wall (adaptive under e050 contention). No
NOTES/THINKING/QUEUE/STATE edits here; no commit.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # G4: Phase 1 is 100% CPU (set pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES (device_count 0); force CPU so common.DEVICE=="cpu"
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import REPO, Cfg, CharCorpus, TinyGPT, estimate_loss, run_dir, save_json

# ------------------------------------------------------------------ constants
SMOKE = os.environ.get("E053_SMOKE") == "1"
MAX_SEQ = 2 if SMOKE else 8
PROMPT_TOK = 64
T_TOTAL = 256                     # wpe=256 hard limit (design §1)
TEMP, TOPK = 0.8, 40
SEED_PROMPT, SEED_SAMPLE, SEED_STATIC = 202, 7, 203
N_STATIC = 8 if SMOKE else 24
BINS = [(0, 0), (1, 16), (17, 64), (65, 128), (129, 192), (193, 255)]
BIN_NAMES = ["p0", "1-16", "17-64", "65-128", "129-192", "193-255"]
THRESH = 0.01                     # dead-weight threshold (nats)
A_STAR_K = 5                       # sustained window for robust onset
BUDGET_S = 60 if SMOKE else 840.0   # soft guard (14 min)
THREADS = 12                       # measured best solo (24 oversubscribes)
torch.set_num_threads(THREADS)

CK = REPO / "runs" / "checkpoints"
CELLS = [
    # name, ckpt, arch overrides, axis tag, training steps, budget weight
    ("small_0.84M", CK / "e005s_small.pt", dict(n_layer=4, n_head=4, n_embd=128), "scale", None, 1.2),
    ("large_10M",   CK / "e005s_large.pt", dict(n_layer=8, n_head=8, n_embd=320), "scale", None, 1.5),
    ("mid_2.7M",    CK / "e001.pt",        dict(n_layer=6, n_head=6, n_embd=192), "scale+exposure", 4000, 1.2),
    ("exp_d400",    CK / "e048_direct400.pt", dict(n_layer=6, n_head=6, n_embd=192), "exposure", 400, 1.0),
    ("exp_d800",    CK / "e048_direct800.pt", dict(n_layer=6, n_head=6, n_embd=192), "exposure", 800, 1.0),
]
ANCHORS = {"small_0.84M": (1.5581, 0.03), "mid_2.7M": (1.622391, 0.03)}
CHUNK = {4: 255, 6: 96, 8: 48}     # sweep batch chunk by n_head (memory guard)
T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------- exact machinery

@torch.no_grad()
def _manual_chunk(net: TinyGPT, idxs, vzero, kdrop, layers):
    """Exact manual forward, returns LAST-position logits (N, vocab).

    vzero/kdrop: (N, T) bool. layers: None = all, else set of layer indices.
    K-drop masks the softmax column of pos p for all query rows q != p
    (diagonal preserved -> G3: every row keeps >= 1 visible key).
    """
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
    return net.lm_head(net.ln_f(x)[:, -1, :])


@torch.no_grad()
def manual_logits(net, idxs, vzero=None, kdrop=None, layers=None, chunk=64):
    outs = []
    for i in range(0, idxs.shape[0], chunk):
        outs.append(_manual_chunk(net, idxs[i:i + chunk],
                                  vzero[i:i + chunk] if vzero is not None else None,
                                  kdrop[i:i + chunk] if kdrop is not None else None,
                                  layers))
    return torch.cat(outs, 0)


def bin_positions(bi: int, t_ctx: int) -> list[int]:
    a, b = BINS[bi]
    hi = min(b, t_ctx - 1)
    return list(range(a, hi + 1)) if a <= hi else []


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


# ------------------------------------------------------- incremental decoding

@torch.no_grad()
def prefill(net: TinyGPT, idx: torch.Tensor):
    """Manual prefill of the prompt; returns last-position logits (V,) and the
    per-layer KV cache [(k, v)] each (1, H, T, d). Exact same math as
    _manual_chunk (no lesion)."""
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
    return net.lm_head(net.ln_f(x[0, -1])), kv          # (V,)


@torch.no_grad()
def decode_step(net: TinyGPT, tok: int, pos: int, kv: list):
    """Incremental decode of one token at position pos; extends kv in place.
    Returns next-position logits (V,)."""
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


# --------------------------------------------------------- adaptive protocol

def fit_protocol(t7: float, c_dec: float, share: float, L: int, is_scale: bool,
                 pin_v_full: bool = False) -> dict:
    """Fit (n_seq, stride, sweep detail) to the cell's wall-time share.

    Cost model (measured): clean generation runs on the incremental KV-cache
    decoder (~c_dec*t7 per step); lesion batches cost ~t7 per measured step
    (B~8); a full 255-position sweep ~8*t7 per kind (batch amortization);
    per-layer decomposition L*t7; binned sweeps ~1*t7. Priority: n_seq
    (registered aggregation unit) > sweep detail > timeline stride, per the
    design's fallback ordering — except exposure-axis cells PIN the full
    V-sweep (their a* comparison needs per-position resolution).
    Acceptance targets 0.95x share: t7 is measured in one power state and
    the machine fluctuates ~3x between states (pace guard trims remainder).
    """
    SW = 8.0 * t7
    SB = 1.0 * t7
    DEC = 192 * c_dec * t7
    for n_seq in (8, 6, 4, 2):
        if n_seq > MAX_SEQ:
            continue
        for v_full, k_full in ((True, True), (True, False), (False, False)):
            if pin_v_full and not v_full:
                continue
            for per_layer in ((is_scale and not SMOKE), False):
                fixed = n_seq * ((SW if v_full else SB) + (SW if k_full else SB)
                                 + (L * t7 if per_layer else 0) + 3 * SB)
                for stride in (1, 2, 4, 8):
                    tl = n_seq * (DEC + (192 / stride) * t7)
                    if fixed + tl <= 0.95 * share:
                        return dict(n_seq=n_seq, stride=stride, v_sweep_full=v_full,
                                    k_sweep_full=k_full, per_layer=per_layer,
                                    est_s=fixed + tl, t7=t7)
    return dict(n_seq=min(2, MAX_SEQ), stride=8, v_sweep_full=False, k_sweep_full=False,
                per_layer=False, est_s=None, t7=t7)


# ----------------------------------------------------------------- per-sequence

@torch.no_grad()
def run_sequence(net: TinyGPT, prompt: torch.Tensor, gen: torch.Generator, ch: int,
                 stride: int = 1, v_sweep_full: bool = True, k_sweep_full: bool = True,
                 per_layer: bool = True):
    """Free-run one 64->256 sequence, fixed anchor.

    Clean generation runs on the incremental KV-cache decoder (exact same
    math as the full forward; G0b checks prob agreement); lesions are exact
    full-context batched manual forwards compared on the actually-sampled
    token.
    """
    n_steps = T_TOTAL - PROMPT_TOK
    L = net.cfg.n_layer
    idx = prompt.clone()
    tl = np.full((n_steps, len(BINS)), np.nan)      # per-step binned V-zero dCE
    rec64 = np.full(n_steps, np.nan)                # last-64 recency lesion (G2)
    nan_count = 0
    g2_viol = 0
    g2_bins_viol = 0
    logits, kv = prefill(net, idx)                  # predicts position 64
    for si, t in enumerate(range(PROMPT_TOK, T_TOTAL)):
        ctx = idx[:t]
        tok, ce_clean = sample_and_ce(logits, gen)
        measure = ((t - PROMPT_TOK) % stride == 0) or (t == T_TOTAL - 1)
        if measure:
            rows, cols = [], []
            for bi in range(len(BINS)):
                ps = bin_positions(bi, t)
                if ps:
                    rows.append(ps)
                    cols.append(bi)
            N = len(rows) + 1
            idxs = ctx[None].repeat(N, 1)
            vz = torch.zeros(N, t, dtype=torch.bool)
            for r, ps in enumerate(rows):
                vz[r, ps] = True
            vz[N - 1, max(0, t - 64):t] = True      # G2 recency row
            les = manual_logits(net, idxs, vz, None, None, chunk=ch)
            if bool(torch.isnan(les).any()):
                nan_count += int(torch.isnan(les).any(-1).sum())
                les = torch.nan_to_num(les, nan=0.0)
            lg = torch.log_softmax(les.float(), -1)
            dce = -ce_clean - lg[:, tok].numpy()    # dCE = log(p_clean/p_les)
            for r, bi in enumerate(cols):
                tl[si, bi] = dce[r]
            rec64[si] = dce[-1]
            if rec64[si] <= 0.5:
                g2_viol += 1
            newest = None
            for bi in reversed(range(len(BINS))):
                if len(bin_positions(bi, t)) >= 32:
                    newest = bi
                    break
            if newest is not None and tl[si, newest] <= 0.5:
                g2_bins_viol += 1
        idx = torch.cat([idx, torch.tensor([tok])])
        if t < T_TOTAL - 1:
            logits = decode_step(net, tok, t, kv)
    # ---- final step (t=255, target = position 255): full measurements
    ctx, tgt = idx[:T_TOTAL - 1], int(idx[T_TOTAL - 1])
    t_ctx = ctx.shape[0]
    p_clean = torch.softmax(logits, -1)
    ce_clean = float(-math.log(max(p_clean[tgt].item(), 1e-12)))
    P = t_ctx
    idxs = ctx[None].repeat(P, 1)
    sweep_v = sweep_k = None
    if v_sweep_full:
        eye = torch.eye(P, dtype=torch.bool)
        lp = torch.log_softmax(manual_logits(net, idxs, eye, None, None, chunk=ch).float(), -1)
        sweep_v = (-ce_clean - lp[torch.arange(P), tgt]).numpy()
    if k_sweep_full:
        eye = torch.eye(P, dtype=torch.bool)
        lp = torch.log_softmax(manual_logits(net, idxs, None, eye, None, chunk=ch).float(), -1)
        sweep_k = (-ce_clean - lp[torch.arange(P), tgt]).numpy()
    # final bins: vzero / kdrop / both
    fb = {}
    for kind in ("vzero", "kdrop", "both"):
        idxs = ctx[None].repeat(len(BINS), 1)
        vz = torch.zeros(len(BINS), t_ctx, dtype=torch.bool)
        kd = torch.zeros(len(BINS), t_ctx, dtype=torch.bool)
        for bi in range(len(BINS)):
            ps = bin_positions(bi, t_ctx)
            if kind in ("vzero", "both"):
                vz[bi, ps] = True
            if kind in ("kdrop", "both"):
                kd[bi, ps] = True
        lg = torch.log_softmax(manual_logits(net, idxs, vz, kd, None, chunk=ch).float(), -1)
        fb[kind] = -ce_clean - lg[torch.arange(len(BINS)), tgt].numpy()
    # per-layer V-zero decomposition (final step)
    per_layer_m = np.full((L, len(BINS)), np.nan)
    if per_layer:
        for li in range(L):
            idxs = ctx[None].repeat(len(BINS), 1)
            vz = torch.zeros(len(BINS), t_ctx, dtype=torch.bool)
            for bi in range(len(BINS)):
                vz[bi, bin_positions(bi, t_ctx)] = True
            lg = torch.log_softmax(
                manual_logits(net, idxs, vz, None, layers={li}, chunk=ch).float(), -1)
            per_layer_m[li] = -ce_clean - lg[torch.arange(len(BINS)), tgt].numpy()
    return dict(tl=tl, rec64=rec64, sweep_v=sweep_v, sweep_k=sweep_k, final_bins=fb,
                per_layer=per_layer_m, nan_count=nan_count, g2_viol=g2_viol,
                g2_bins_viol=g2_bins_viol, text=idx.tolist())


# ------------------------------------------------------------------- per-cell

def load_cell(name: str, ckpt: Path, arch: dict):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    cfg = Cfg(vocab=65, block_size=256, **arch)
    net = TinyGPT(cfg)
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net


@torch.no_grad()
def g0_check(net, window: torch.Tensor) -> float:
    idx = window[:256][None]
    sdpa_logits, _ = net(idx)
    man = manual_logits(net, idx, None, None, chunk=1)[0]
    p0 = torch.softmax(man, -1)
    p1 = torch.softmax(sdpa_logits[0, -1].float(), -1)
    return float((p0 - p1).abs().max().item())


@torch.no_grad()
def measure_t7(net, ch: int, window: torch.Tensor):
    """Measure batched-forward cost (t7), incremental-decode cost ratio
    (c_dec = t_dec/t7), and G0b: incremental-decode vs full-forward
    max probability deviation."""
    idxs = window[:192][None].repeat(7, 1)
    vz = torch.zeros(7, 192, dtype=torch.bool)
    vz[1:, 10:20] = True
    ts = []
    for _ in range(4):
        t0 = time.time()
        manual_logits(net, idxs, vz, None, None, chunk=ch)
        ts.append(time.time() - t0)
    t7 = float(sorted(ts)[1])
    _, kv = prefill(net, window[:128])
    ts2 = []
    lg = None
    for p in range(128, 133):
        t0 = time.time()
        lg = decode_step(net, int(window[p]), p, kv)
        ts2.append(time.time() - t0)
    full = manual_logits(net, window[:133][None], None, None, chunk=1)[0]
    p0 = torch.softmax(lg, -1)
    p1 = torch.softmax(full, -1)
    return t7, float(sorted(ts2)[2]) / t7, float((p0 - p1).abs().max().item())


@torch.no_grad()
def static_control(net, corp, ch: int):
    gen = torch.Generator().manual_seed(SEED_STATIC)
    ix = torch.randint(len(corp.val) - T_TOTAL - 1, (N_STATIC,), generator=gen)
    accv = np.zeros(len(BINS))
    acck = np.zeros(len(BINS))
    nw = 0
    for w0 in range(0, N_STATIC, 4):
        wins = [corp.val[ix[w]:ix[w] + T_TOTAL] for w in range(w0, min(w0 + 4, N_STATIC))]
        rows = 1 + 2 * len(BINS)
        idxs = torch.stack([w[:T_TOTAL - 1] for w in wins for _ in range(rows)])
        vz = torch.zeros(idxs.shape[0], T_TOTAL - 1, dtype=torch.bool)
        kd = torch.zeros_like(vz)
        for wi in range(len(wins)):
            base = wi * rows
            for bi in range(len(BINS)):
                ps = bin_positions(bi, T_TOTAL - 1)
                vz[base + 1 + bi, ps] = True
                kd[base + 1 + len(BINS) + bi, ps] = True
        lp = torch.log_softmax(manual_logits(net, idxs, vz, kd, None, ch).float(), -1)
        for wi, win in enumerate(wins):
            base = wi * rows
            tgt = int(win[T_TOTAL - 1])
            ce = -float(lp[base, tgt])
            for bi in range(len(BINS)):
                accv[bi] += -ce - float(lp[base + 1 + bi, tgt])
                acck[bi] += -ce - float(lp[base + 1 + len(BINS) + bi, tgt])
        nw += len(wins)
    return accv / nw, acck / nw


def onset_age(dce_by_age: np.ndarray, k: int = A_STAR_K, thresh: float = THRESH):
    """dce_by_age: ages ascending (1=youngest ... 255=oldest/sink).
    Returns (naive, robust): youngest age whose mean dCE < thresh; robust
    requires a k-length run of 5-pt moving-mean values below thresh."""
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


def onset_age_binned(bin_totals: np.ndarray) -> int | None:
    """Coarse a* when only binned final data exists: per-position utility of
    the oldest still-live bin bounds the onset (age ~ 255 - bin_lo)."""
    perpos = []
    for bi in range(1, len(BINS)):
        n = len(bin_positions(bi, T_TOTAL - 1))
        perpos.append((bi, bin_totals[bi] / n if n else float("nan")))
    live = [bi for bi, v in perpos if v >= THRESH]
    if not live:
        return None
    bi_oldest = max(live)
    return 255 - BINS[bi_oldest][0] + 1


def classify_p3(traj: np.ndarray, steps: np.ndarray):
    init_m = steps < 80                      # early generation (stride-proof)
    init = float(np.nanmean(traj[init_m])) if init_m.any() else float("nan")
    midm = np.abs(steps - 160) <= 16
    mid = float(np.nanmean(traj[midm])) if midm.any() else float("nan")
    fin_m = steps >= 240
    fin = float(np.nanmean(traj[fin_m])) if fin_m.any() else float("nan")
    if not (init == init) or init <= 1e-9:
        cls = "FLAT"
    elif mid == mid and mid < 0.2 * init:
        cls = "DECAY"
    elif fin > 1.1 * init:
        cls = "GROW"
    elif abs(fin - init) < 0.1 * init:
        cls = "FLAT"
    else:
        cls = "PARTIAL_DECAY"
    return cls, dict(initial=init, mid=mid, final=fin, sign_flip=bool(fin < 0))


# --------------------------------------------------------------------- main

def main():
    assert torch.cuda.device_count() == 0, "G4 violated: CUDA device visible"
    out_dir = run_dir("e053")
    corp = CharCorpus(REPO / "data" / "input.txt")
    assert corp.vocab_size == 65

    gen_p = torch.Generator().manual_seed(SEED_PROMPT)
    ix = torch.randint(len(corp.val) - PROMPT_TOK - 1, (MAX_SEQ,), generator=gen_p)
    prompts = [corp.val[i:i + PROMPT_TOK] for i in ix]
    g0_window = corp.val[corp.seed % (len(corp.val) - T_TOTAL - 1):][:T_TOTAL]

    results, gates = {}, dict(
        G0={}, G0b={}, G1={}, G2_last64_violations=0, G2_bin_violations=0, G3_nan=0,
        G4_cpu_only=True, threads=THREADS, smoke=SMOKE)
    skipped = []
    exposure_clamp = None       # exposure-axis cells share n_seq (paired P2)
    pending = list(CELLS)

    while pending:
        name, ckpt, arch, axis, steps_trained, weight = pending[0]
        R = BUDGET_S - elapsed()
        wsum = sum(w for *_, w in pending)
        share = max(30.0, R * weight / wsum)
        if R < 60 and not SMOKE and len(results) >= 3:
            skipped.append(name)
            pending.pop(0)
            log(f"SKIP {name}: budget guard ({R:.0f}s left)")
            continue
        net = load_cell(name, ckpt, arch)
        ch = CHUNK[net.cfg.n_head]
        dev0 = g0_check(net, g0_window)
        gates["G0"][name] = dev0
        val_ce = estimate_loss(net, corp, "val", n_batches=12)
        if name in ANCHORS:
            a, tol = ANCHORS[name]
            gates["G1"][name] = dict(val_ce=val_ce, anchor=a, ok=abs(val_ce - a) <= tol)
        else:
            gates["G1"][name] = dict(val_ce=val_ce, anchor=None, ok=None)
        t7, c_dec, g0b = measure_t7(net, ch, g0_window)
        gates["G0b"][name] = g0b
        proto = fit_protocol(t7, c_dec, share, net.cfg.n_layer, axis.startswith("scale"),
                             pin_v_full=("exposure" in axis))
        if axis == "exposure" and exposure_clamp is not None:
            proto["n_seq"] = min(proto["n_seq"], exposure_clamp)
        if axis == "scale+exposure":
            exposure_clamp = proto["n_seq"]
        est = proto["est_s"]
        log(f"{name}: G0 {dev0:.2e} G0b {g0b:.2e} | val CE {val_ce:.4f} | "
            f"t7 {t7*1000:.0f}ms c_dec {c_dec:.3f} | share {share:.0f}s -> "
            f"n_seq={proto['n_seq']} stride={proto['stride']} "
            f"vfull={proto['v_sweep_full']} kfull={proto['k_sweep_full']} "
            f"perlayer={proto['per_layer']}" + (f" (est {est:.0f}s)" if est is not None else
                                                " MINIMAL"))

        gen_s = torch.Generator().manual_seed(SEED_SAMPLE)   # paired across cells
        seqs = []
        cell_t0 = elapsed()
        for si in range(proto["n_seq"]):
            r = run_sequence(net, prompts[si], gen_s, ch,
                             stride=proto["stride"], v_sweep_full=proto["v_sweep_full"],
                             k_sweep_full=proto["k_sweep_full"], per_layer=proto["per_layer"])
            seqs.append(r)
            done = si + 1
            cell_elapsed = elapsed() - cell_t0
            proj = cell_elapsed / done * (proto["n_seq"] - done)
            if (done >= 2 and si + 1 < proto["n_seq"]
                    and (cell_elapsed + proj > 1.35 * share
                         or elapsed() + proj > BUDGET_S * 0.95)):
                log(f"  {name}: pace guard -> stopping at {done} seqs "
                    f"(cell {cell_elapsed:.0f}s, proj +{proj:.0f}s vs share {share:.0f}s)")
                break
            gates["G3_nan"] += r["nan_count"]
            gates["G2_last64_violations"] += r["g2_viol"]
            gates["G2_bin_violations"] += r["g2_bins_viol"]
            log(f"  {name} seq{si}: last64(final)={r['rec64'][-1]:+.2f} "
                f"rec-bin(final)={r['final_bins']['vzero'][5]:+.2f} "
                f"sink(final)={r['tl'][-1,0]:+.3f} g2viol={r['g2_viol']} nan={r['nan_count']}")

        # static teacher-forced control (optional under budget)
        st_v = st_k = None
        if elapsed() < BUDGET_S * 0.92 and not SMOKE:
            st_v, st_k = static_control(net, corp, ch)

        # ---- aggregate
        tl = np.stack([s["tl"] for s in seqs])                       # (S, steps, bins)
        steps_axis = np.arange(PROMPT_TOK, T_TOTAL)
        fbp = {k: np.stack([s["final_bins"][k] for s in seqs]) for k in ("vzero", "kdrop", "both")}
        per_layer = np.stack([s["per_layer"] for s in seqs])         # (S, L, bins)
        sweeps_v = [s["sweep_v"] for s in seqs if s["sweep_v"] is not None]
        sweeps_k = [s["sweep_k"] for s in seqs if s["sweep_k"] is not None]
        fine = len(sweeps_v) == len(seqs)
        if fine:
            sweep_v = np.stack(sweeps_v)                             # (S, 255)
            age_axis = np.arange(1, T_TOTAL)                         # age of pos p = 255-p
            dce_age = sweep_v[:, ::-1]                               # young(1) -> old(255)
            mean_age = dce_age.mean(0)
            ci_lo, ci_hi = boot_ci(dce_age)
            naive_a, robust_a = onset_age(mean_age)
            a_star = robust_a if robust_a is not None else naive_a
            junk_all = float((sweep_v <= -THRESH).mean())
            if a_star is not None:
                junk_old = float((dce_age[:, a_star:] <= -THRESH).mean())
            else:
                junk_old = None
            primacy = float(sweep_v[:, 1:17].mean())
            plateau = float(sweep_v[:, 224:255].mean())
            sink_dce = float(sweep_v[:, 0].mean())
            fp_block = dict(positions=list(range(T_TOTAL - 1)), age=list(age_axis),
                            vzero_dce_mean=sweep_v.mean(0).tolist(),
                            vzero_ci=[ci_lo.tolist(), ci_hi.tolist()],
                            kdrop_dce_mean=(np.stack(sweeps_k).mean(0).tolist()
                                            if len(sweeps_k) == len(seqs) else None))
        else:
            naive_a = robust_a = a_star = onset_age_binned(fbp["vzero"].mean(0))
            junk_all = junk_old = None
            primacy = plateau = sink_dce = None
            fp_block = dict(positions=None, age=None, vzero_dce_mean=None,
                            vzero_ci=None, kdrop_dce_mean=None)
        bin_v = fbp["vzero"].mean(0)
        monotone = float(np.mean(np.diff(bin_v[1:]) > 0))
        p3_cls, p3_nums = classify_p3(tl[:, :, 0].mean(0), steps_axis)
        eq_gap = float(np.abs(fbp["both"] - fbp["kdrop"]).max())

        results[name] = dict(
            ckpt=str(ckpt), arch=dict(**arch, block_size=256, vocab=65),
            axis=axis, steps_trained=steps_trained, val_ce=val_ce,
            protocol=dict(n_seq=proto["n_seq"], stride=proto["stride"],
                          v_sweep_full=proto["v_sweep_full"],
                          k_sweep_full=proto["k_sweep_full"],
                          per_layer=proto["per_layer"], t7_s=t7, c_dec=c_dec,
                          share_s=share, est_s=proto["est_s"], fine_sweep=fine),
            timeline=dict(steps=steps_axis.tolist(), bins=BIN_NAMES,
                          vzero_dce_mean=tl.mean(0).tolist(),
                          last64_mean=np.stack([s["rec64"] for s in seqs]).mean(0).tolist()),
            sink_traj=dict(mean=tl[:, :, 0].mean(0).tolist(),
                           ci=[boot_ci(tl[:, :, 0])[0].tolist(), boot_ci(tl[:, :, 0])[1].tolist()],
                           p3_class=p3_cls, p3_numbers=p3_nums),
            final_profile=fp_block,
            final_bins={k: dict(mean=v.mean(0).tolist(),
                                ci=[boot_ci(v)[0].tolist(), boot_ci(v)[1].tolist()])
                        for k, v in fbp.items()},
            per_layer_vzero=dict(mean=per_layer.mean(0).tolist(),
                                 ci=[boot_ci(per_layer)[0].tolist(), boot_ci(per_layer)[1].tolist()]),
            static_control=dict(vzero=st_v.tolist() if st_v is not None else None,
                                kdrop=st_k.tolist() if st_k is not None else None,
                                n_windows=N_STATIC if st_v is not None else 0),
            derived=dict(a_star_naive=naive_a, a_star_robust=robust_a, a_star=a_star,
                         a_star_binned_fallback=not fine,
                         a_star_frac=(a_star / (T_TOTAL - 1)) if a_star else None,
                         junk_frac_all=junk_all, junk_frac_beyond_ast=junk_old,
                         sink_dce_final=sink_dce,
                         primacy_bump=primacy, recent_plateau=plateau,
                         primacy_ratio=(primacy / plateau) if (plateau and plateau > 0) else None,
                         monotone_frac=monotone, equivalence_gap=eq_gap,
                         bin_vzero_final=bin_v.tolist()),
            sample_text_first160=corp.decode(torch.tensor(seqs[0]["text"][:160])),
        )
        log(f"{name} DONE a*={a_star}{'(binned)' if not fine else ''} "
            f"sink_final={sink_dce if sink_dce is None else round(sink_dce, 4)} "
            f"p3={p3_cls} junk_old={junk_old}")
        pending.pop(0)

    # ---------------------------------------------------------------- verdicts
    sc = [n for n in results if results[n]["axis"].startswith("scale")]
    ex = [("exp_d400", 400), ("exp_d800", 800), ("mid_2.7M", 4000)]
    fr = {n: results[n]["derived"]["a_star_frac"] for n in results
          if results[n]["derived"]["a_star_frac"] is not None}
    scale_f = [fr[n] for n in sc if n in fr]
    p2_scale_ratio = (max(scale_f) / min(scale_f)) if len(scale_f) == len(sc) and scale_f else None
    p2_scale_ok = bool(p2_scale_ratio is not None and p2_scale_ratio <= 1.5)
    p2_exp_ratio = (fr.get("mid_2.7M") / fr.get("exp_d400")
                    if fr.get("exp_d400") not in (None, 0) and fr.get("mid_2.7M") else None)
    sink_dce = {n: results[n]["derived"]["sink_dce_final"] for n in results
                if results[n]["derived"]["sink_dce_final"] is not None}
    ratios = [r for r in (results[n]["derived"]["primacy_ratio"] for n in results)
              if r is not None]
    p1 = dict(
        sink_dead_count=int(sum(1 for v in sink_dce.values() if v < 0.05)),
        sink_loadbearing=[n for n, v in sink_dce.items() if v >= 0.3],
        sink_dce_final=sink_dce,
        primacy_ratio={n: results[n]["derived"]["primacy_ratio"] for n in results},
        monotone_frac={n: results[n]["derived"]["monotone_frac"] for n in results},
    )
    p1["pass"] = bool(p1["sink_dead_count"] >= 3 and ratios and all(r <= 0.25 for r in ratios)
                      and all(results[n]["derived"]["monotone_frac"] >= 0.75 for n in results))
    p2 = dict(scale_fractions={n: fr.get(n) for n in sc},
              scale_invariant=p2_scale_ok, scale_maxmin_ratio=p2_scale_ratio,
              exposure_a_star={n: results[n]["derived"]["a_star"] for n, _ in ex if n in results},
              exposure_growth_ratio=p2_exp_ratio,
              **{"pass": bool(p2_scale_ok and p2_exp_ratio is not None and p2_exp_ratio >= 1.5)})
    p3 = dict(per_cell={n: results[n]["sink_traj"]["p3_class"] for n in results},
              numbers={n: results[n]["sink_traj"]["p3_numbers"] for n in results},
              sign_flip={n: results[n]["sink_traj"]["p3_numbers"]["sign_flip"] for n in results})
    junk = [results[n]["derived"]["junk_frac_beyond_ast"] for n in results
            if results[n]["derived"]["junk_frac_beyond_ast"] is not None]
    p4 = dict(junk_beyond_ast_per_cell={n: results[n]["derived"]["junk_frac_beyond_ast"]
                                        for n in results},
              pooled=float(np.mean(junk)) if junk else None,
              **{"pass": bool(junk and np.mean(junk) >= 0.05)})

    metrics = dict(
        experiment="e053_cache_timeline", phase=1,
        started=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        wall_s=elapsed(), smoke=SMOKE, n_seq_max=MAX_SEQ, bins=BIN_NAMES,
        seeds=dict(prompts=SEED_PROMPT, sampling=SEED_SAMPLE, static=SEED_STATIC),
        protocol=dict(prompt_tokens=PROMPT_TOK, t_total=T_TOTAL, temp=TEMP, topk=TOPK,
                      fixed_anchor=True, threshold=THRESH, a_star_window=A_STAR_K,
                      adaptive="per-cell fit under e050 CPU contention; registered "
                               "fallbacks (seqs/stride/binned sweep) per design §5"),
        gates=gates, skipped=skipped, cells=results,
        verdicts=dict(P1_shape=p1, P2_onset=p2, P3_sink=p3, P4_junk=p4),
    )
    save_json(out_dir / "metrics.json", metrics)
    log(f"metrics.json written ({len(results)} cells, skipped={skipped})")
    if results:
        plot(out_dir / "e053_utility_timeline.png", results)
        log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot

def plot(path: Path, R: dict):
    names = list(R)
    fig = plt.figure(figsize=(19, 12.5))
    gs = fig.add_gridspec(3, 6, hspace=0.45, wspace=0.45)
    colors = {"small_0.84M": "tab:blue", "mid_2.7M": "tab:green", "large_10M": "tab:red",
              "exp_d400": "tab:orange", "exp_d800": "tab:purple"}

    # row 1: THE curve — per-position V-zero dCE vs cache age, one panel per cell + overlay
    order = [c[0] for c in CELLS if c[0] in R]
    for i, n in enumerate(order[:5]):
        ax = fig.add_subplot(gs[0, i])
        fp = R[n]["final_profile"]
        if fp["age"] is not None:
            age = np.asarray(fp["age"])
            m = np.asarray(fp["vzero_dce_mean"])
            lo, hi = np.asarray(fp["vzero_ci"][0]), np.asarray(fp["vzero_ci"][1])
            ax.fill_between(age, lo, hi, alpha=0.25, color=colors[n])
            ax.plot(age, m, lw=1.4, color=colors[n])
            ax.axhline(0, color="k", lw=0.6)
            ax.axhline(THRESH, color="gray", ls=":", lw=0.8)
            a = R[n]["derived"]["a_star"]
            if a:
                ax.axvline(a, color=colors[n], ls="--", lw=1)
            ax.set_ylabel("V-zero dCE (nats)")
            ax.set_xlim(0, 256)
        else:  # binned fallback: per-position utility from bin totals
            bv = np.asarray(R[n]["derived"]["bin_vzero_final"])[1:]
            cnt = np.array([len(bin_positions(b, T_TOTAL - 1)) for b in range(1, 6)])
            ax.bar(range(5), bv / cnt, color=colors[n], alpha=0.6)
            ax.set_xticks(range(5), BIN_NAMES[1:], fontsize=7)
            ax.set_ylabel("per-position dCE (binned)", fontsize=8)
            a = R[n]["derived"]["a_star"]
        pr = R[n]["protocol"]
        ax.set_title(f"{n}\na*={a}"
                     f"{' (binned)' if R[n]['derived']['a_star_binned_fallback'] else ''}"
                     f"  n={pr['n_seq']} stride={pr['stride']}", fontsize=9)
        ax.set_xlabel("cache age (tokens)" if fp["age"] is not None else "position bin")
    ax = fig.add_subplot(gs[0, 5])
    for n in order:
        fp = R[n]["final_profile"]
        if fp["age"] is None:
            continue
        ax.plot(fp["age"], fp["vzero_dce_mean"], lw=1.3, color=colors[n],
                ls="--" if R[n]["axis"] == "exposure" else "-", label=n)
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(THRESH, color="gray", ls=":", lw=0.8)
    ax.legend(fontsize=7)
    ax.set_title("overlay (solid=scale, dash=exposure)", fontsize=9)
    ax.set_xlabel("cache age (tokens)")
    ax.set_xlim(0, 256)

    # row 2: age x step heatmap (primary + largest), sink trajectory (P3)
    hm = [x for x in ("small_0.84M", "large_10M") if x in R]
    for j, n in enumerate(hm[:2]):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        tl = np.asarray(R[n]["timeline"]["vzero_dce_mean"])
        M = np.ma.masked_invalid(tl.T)
        vmax = max(float(np.nanpercentile(tl, 98)), 0.1)
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="magma",
                       extent=[64, 256, -0.5, 5.5], vmin=0, vmax=vmax)
        ax.set_yticks(range(6), BIN_NAMES)
        ax.set_xlabel("generation step t")
        ax.set_title(f"age-bin x step dCE — {n}", fontsize=9)
        plt.colorbar(im, ax=ax, label="dCE (nats)")
    ax = fig.add_subplot(gs[1, 4:6])
    for n in order:
        st = R[n]["sink_traj"]
        steps = np.asarray(R[n]["timeline"]["steps"])
        ax.plot(steps, st["mean"], color=colors[n],
                ls="--" if R[n]["axis"] == "exposure" else "-", lw=1.2,
                label=f"{n}:{st['p3_class']}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("generation step t")
    ax.set_ylabel("sink (pos 0) V-zero dCE")
    ax.set_title("P3 — sink utility timeline", fontsize=9)
    ax.legend(fontsize=7)

    # row 3: per-layer decomposition, onset ages, static vs free
    sc = [n for n in R if R[n]["axis"].startswith("scale") and
          not np.all(np.isnan(np.asarray(R[n]["per_layer_vzero"]["mean"])))]
    if sc:
        pick = max(sc, key=lambda n: float(
            np.nanmax(np.asarray(R[n]["per_layer_vzero"]["mean"])[:, 3])))
        ax = fig.add_subplot(gs[2, 0:2])
        pl = np.asarray(R[pick]["per_layer_vzero"]["mean"])
        M = np.ma.masked_invalid(pl)
        im = ax.imshow(M, aspect="auto", cmap="viridis")
        ax.set_xticks(range(6), BIN_NAMES)
        ax.set_ylabel("layer")
        ax.set_title(f"per-layer V-zero dCE — {pick}", fontsize=9)
        plt.colorbar(im, ax=ax, label="dCE (nats)")
    ax = fig.add_subplot(gs[2, 2:4])
    xs, ys, cs = [], [], []
    for n in order:
        a = R[n]["derived"]["a_star"]
        if a:
            xs.append(n)
            ys.append(a)
            cs.append(colors[n])
    ax.bar(xs, ys, color=cs)
    ax.set_ylabel("onset age a* (tokens)")
    ax.set_title("P2 — dead-weight onset (scale vs exposure)", fontsize=9)
    ax.tick_params(axis="x", rotation=30)
    ax = fig.add_subplot(gs[2, 4:6])
    for n in order:
        stf = R[n]["static_control"]["vzero"]
        fr = R[n]["derived"]["bin_vzero_final"]
        if stf is None:
            continue
        ax.scatter(np.abs(stf[1:]) + 1e-6, np.abs(fr[1:]) + 1e-6, color=colors[n],
                   label=n, s=28)
    lim = [1e-4, 10]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("|static (real text) bin dCE|")
    ax.set_ylabel("|free-run final bin dCE|")
    ax.set_title("self-generated vs real context aging", fontsize=9)
    ax.legend(fontsize=7)

    fig.suptitle("E053 — cache utility timeline: per-position causal dCE of V-zero lesions "
                 "during fixed-anchor generation (seeded seqs, T=256, adaptive n/stride "
                 "under e050 CPU contention)", fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
