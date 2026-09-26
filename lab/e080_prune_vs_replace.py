"""E080 — T048's REGISTERED discriminator: prune-vs-replace at the same events.

Why does V-zeroing the self-generated dead band cost +0.26 nats (e075: +0.2619,
CI [+0.09,+0.45]) and drive generation off-manifold (clean-judge 6.40 nats,
live_frac 1.000, prompt-band junk inversion 0.375)? Two registered
explanations (T048), discriminated HERE:

  - H-statistics-scaffold: the entries' PRESENCE (norms / attention-distribution
    mass) maintains generation-time statistics. PREDICTS norm-matched noise
    replacement costs far less than V-zero.
  - H-content-anchoring: the run's own older outputs are self-anchors for
    style/state continuity. PREDICTS prompt-content replacement preserves
    fluency; noise fails like zeroing.

Arms (e075 rig VERBATIM: e053c net, fixed-anchor 64->512 free run, B=8
seed-202 battery, per-arm seed-7 sampling; interventions at the SAME prune
events g=100..420 K=32, self band = own generated entries with age > 96,
i.e. positions 64..front-97, replaced once at first band entry and never
touched again — the analogue of e075's zero-stays-zero):
  1. A-none     control: normal free run.
  2. A-vzero    e075's A-self-prune rerun (V -> 0): drift-check vs the known
                +0.2619 (bit-identical stream expected; gated as G4).
  3. A-noise    NORM-MATCHED NOISE: each newly pruned V vector (per layer,
                per (batch, head), head_dim d=32) is replaced by a gaussian
                random direction scaled to the entry's CURRENT L2 norm
                (presence kept, content destroyed). Drawn once per position
                from a dedicated CPU generator (seed 4242) that never touches
                the sampling stream.
  4. A-promptcopy  PROMPT-CONTENT: v[b,:,p,:] <- v[b,:,q,:] copied from the
                SAME sequence's pristine corpus-prompt entry q = ((p-64) mod
                63) + 1 (content kept, norms/positions not); 63 sources ->
                324 targets (5.14x reuse; K untouched in ALL arms, so the
                three intervention arms differ ONLY in what happens to V at
                the same positions: 0 / same-norm random / prompt content).

REGISTERED readouts + bars (T048, frozen):
  - R1 tail clean CE (e075 convention: mean on-line clean full-softmax CE of
    emitted tokens over the final 64 generation steps), Delta vs A-none:
      A-noise <= +0.05 AND A-promptcopy > A-noise + 0.05  -> H-statistics-scaffold
      A-promptcopy <= +0.05 AND A-noise > A-promptcopy + 0.05 -> H-content-anchoring
      anything else -> honest MIXED texture.
  - R2 fluency entropy (tail entropy + top-40 mass drift vs A-none).
  - clean-judge honesty check (the CLEAN net scores the arm's final-64
    tokens): the off-manifold attractor signature is the clean-judge GAP.
  - junk census (e069/e073/e075 instrument): final-step 511-position V-zero
    sweep per arm -> a* onset, live_frac (onset loss), prompt-band junk
    inversion, generated-band junk — does the attractor signature appear
    under noise? under prompt-copy?

Run:     python lab/e080_prune_vs_replace.py
Outputs: runs/e080/metrics.json + runs/e080/prune_vs_replace.png
Envelope: NO training, NO new automations; CPU-only, single step, minutes.
No NOTES/THINKING/QUEUE/STATE edits; no commit (the dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu" (e053b..e075)
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common  # noqa: E402
from common import (  # noqa: E402
    REPO,
    Cfg,
    CharCorpus,
    TinyGPT,
    estimate_loss,
    run_dir,
    save_json,
)

# ------------------------------------------------------------------ constants
PROMPT_TOK = 64
T_TOTAL = 512                             # e053c's ctx-512 window
TEMP, TOPK = 0.8, 40
SEED_PROMPT, SEED_SAMPLE = 202, 7         # e053b/e053c/e069/e070/e072/e074/e075
SEED_NOISE = 4242                         # e080: dedicated noise generator
N_PROMPTS = 8                             # the seed-202 8-draw, ALL 8 (B=8)
B = 8
BOOT_N = 1000
THREADS = 12
torch.set_num_threads(THREADS)

P512 = T_TOTAL - 1                        # 511 sweep positions (native frame)
THRESH = 0.01                             # dead-weight / junk threshold (nats)
A_STAR_K = 5                              # sustained window (e053b verbatim)
CHUNK_512 = 64                            # e069/e070/e072/e074/e075 chunk
BASE_A_STAR = 6                           # e053c/e069/e074/e075 eval-512
                                          # onset; fallback when a*=None

G = T_TOTAL - PROMPT_TOK                  # 448 generation steps (g = 0..447)

# ---- the intervention schedule (e075 design verbatim) -----------------------
PRUNE_START_G = 100                       # "from generation step ~100 on"
PRUNE_K = 32                              # "every K=32 steps"
AGE_CUT = 96                              # "age > 96" (dead band beyond
                                          # live window + shoulder)
PROMPT_POS = (1, 63)                      # 63 corpus-prompt entries (pos 0 =
                                          # seq-start anchor excluded, e073)
NPROMPT = PROMPT_POS[1] - PROMPT_POS[0] + 1   # 63
GEN_FIRST = 64                            # first generated position
TAIL = 64                                 # R1/R2 tail window (final 64 gen)

ARMS = ["none", "vzero", "noise", "promptcopy"]
EVENTS = [g for g in range(PRUNE_START_G, G, PRUNE_K)]   # 100..420, 11 events

# ---- REGISTERED decision numbers (T048, frozen) ------------------------------
CHEAP_BAR = 0.05                          # replacement arm cheap if Δ <= +0.05
GAP_BAR = 0.05                            # between-arm separation bar
R2_DRIFT_MAX = 0.05                       # entropy/top-k drift <= 5% (e075 R2)

# ---- e069 stored readouts (protocol-identity instrument gates) -------------
E069_REF = {"eval512": {"a_star": 6, "ci": [4.0, 8.0], "clean_ce": 0.4528,
                        "ages1_5": [1.742, 4.467, 2.319, 0.914, 0.321],
                        "per_seq": [8, 4, 6, 3], "winstart_dce": 0.0069}}
CKPT = REPO / "runs" / "checkpoints" / "e053c_ctx512.pt"
E053C_VAL_CE = 1.5226792494455974         # the net being interrogated
E075_METRICS = REPO / "runs" / "e075" / "metrics.json"
E075_REF_DELTA_VZERO = 0.26191402220749394     # e075 A-self-prune (the known
                                              # reference this experiment
                                              # discriminates against)
T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ------------------------------------------------- machinery (VERBATIM e053c/e075)

@torch.no_grad()
def _manual_chunk(net: TinyGPT, idxs, vzero, kdrop, layers, pos_offset=0):
    """Exact manual forward, returns LAST-position logits (N, vocab).
    pos_offset lets the same tokens be fed at shifted wpe indices."""
    N, T = idxs.shape
    pos = torch.arange(T) + int(pos_offset)
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
def manual_logits(net, idxs, vzero=None, kdrop=None, layers=None, chunk=64,
                  pos_offset=0):
    outs = []
    for i in range(0, idxs.shape[0], chunk):
        outs.append(_manual_chunk(net, idxs[i:i + chunk],
                                  vzero[i:i + chunk] if vzero is not None else None,
                                  kdrop[i:i + chunk] if kdrop is not None else None,
                                  layers, pos_offset))
    return torch.cat(outs, 0)


@torch.no_grad()
def manual_all_logits(net: TinyGPT, idxs):
    """Clean forward returning logits at ALL positions (N, T, vocab) — the
    clean-judge instrument (scores a stream without any pruning)."""
    N, T = idxs.shape
    pos = torch.arange(T)
    x = net.wte(idxs) + net.wpe(pos).unsqueeze(0)
    H = net.cfg.n_head
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    for blk in net.h:
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
        y = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(N, T, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
    return net.lm_head(net.ln_f(x))


def sample_and_ce(logits_clean: torch.Tensor, gen: torch.Generator):
    """Sample (temp 0.8, top-k 40) from filtered probs; CE from FULL softmax.
    [VERBATIM e053b — CPU tensors + CPU generator, stream-identical]"""
    lg = logits_clean / TEMP
    v, _ = torch.topk(lg, TOPK)
    lg_f = lg.masked_fill(lg < v[-1], float("-inf"))
    tok = int(torch.multinomial(torch.softmax(lg_f, -1), 1, generator=gen))
    p_full = torch.softmax(logits_clean, -1)
    return tok, float(-math.log(max(p_full[tok].item(), 1e-12)))


def boot_ci(X, n: int = BOOT_N, seed: int = 0):
    """Bootstrap CI over the sequence axis. X: (n_seq, ...) -> (lo, hi)."""
    X = np.asarray(X, float)
    if X.shape[0] < 2:
        return X[0], X[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, X.shape[0], size=(n, X.shape[0]))
    means = X[idx].mean(1)
    return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


def onset_age(dce_by_age: np.ndarray, k: int = A_STAR_K, thresh: float = THRESH):
    """dce_by_age: ages ascending (1=youngest ... Pw=oldest/sink).
    Returns (naive, robust). [VERBATIM e053b/E053 — the statistic]"""
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
    """e053b's reported rule: robust preferred, naive fallback."""
    naive, robust = onset_age(mean_age)
    a = robust if robust is not None else naive
    return a, naive, robust


# ------------------------------------------- batched generation (e053b math)

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
def fine_sweep(net: TinyGPT, ctx: torch.Tensor, tgt: torch.Tensor,
               clean_logits: torch.Tensor, ch: int, pos_offset: int = 0):
    """Final-step fine V-zero sweep for all B sequences at once, on whatever
    (ctx, tgt, clean_logits) window is handed in. Returns dce (B, Pw) indexed
    by POSITION (0..Pw-1); age of pos p = Pw - p, so the age-ascending view
    is dce[:, ::-1], plus the per-sequence clean CE. [e053c math, verbatim]"""
    Bs, Pw = ctx.shape
    p_clean = torch.softmax(clean_logits.float(), -1)
    ce_clean = -torch.log(p_clean[torch.arange(Bs), tgt].clamp_min(1e-12))
    idxs = ctx.repeat_interleave(Pw, dim=0)                # (B*Pw, Pw)
    vz = torch.eye(Pw, dtype=torch.bool).repeat(Bs, 1)     # row b*Pw+p: zero p
    les = manual_logits(net, idxs, vz, None, None, chunk=ch, pos_offset=pos_offset)
    lp = torch.log_softmax(les.float(), -1)
    rows = torch.arange(Bs * Pw)
    dce = (-ce_clean.repeat_interleave(Pw) - lp[rows, tgt.repeat_interleave(Pw)])
    return dce.view(Bs, Pw).numpy(), ce_clean.numpy()


def bootstrap_stat(X_by_seq: np.ndarray, stat, n: int = BOOT_N, seed: int = 0):
    """Bootstrap any per-sequence stat. X_by_seq: (S, ...) -> (lo, hi)."""
    S = X_by_seq.shape[0]
    if S < 2:
        v = stat(X_by_seq)
        return (float(v), float(v))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sel = rng.integers(0, S, S)
        vals.append(stat(X_by_seq[sel]))
    vals = np.asarray(vals, float)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def analyze_window(sweep_pos: np.ndarray):
    """sweep_pos: (B, Pw) V-zero dCE by position. e053c's analyze_cell
    generalized to any window length Pw. Returns the derived block."""
    dce_age = sweep_pos[:, ::-1]                       # (B, Pw) young->old
    Pw = dce_age.shape[1]
    mean_age = dce_age.mean(0)
    a_star, naive, robust = a_star_rule(mean_age)
    a_lo, a_hi = bootstrap_stat(dce_age, lambda m: a_star_rule(m.mean(0))[0])
    live = mean_age >= THRESH
    live_frac = float(live.mean())
    lf_lo, lf_hi = bootstrap_stat(dce_age, lambda m: float((m.mean(0) >= THRESH).mean()))
    frac = (a_star / Pw) if a_star is not None else None
    return dict(
        a_star=a_star, a_star_naive=naive, a_star_robust=robust,
        a_star_ci=[a_lo, a_hi], a_star_frac=frac,
        live_frac=live_frac, live_frac_ci=[lf_lo, lf_hi],
        per_seq_a_star=[a_star_rule(dce_age[s])[0] for s in range(dce_age.shape[0])],
        mean_age_curve=mean_age, ci=boot_ci(dce_age, BOOT_N),
        dce_age=dce_age,
    )


def sweep_block(net, ctx, tgt, clean_logits, ch, pos_offset=0):
    """Clean forward given -> full V-zero sweep -> analysis, plus bookkeeping."""
    dce, ce_clean = fine_sweep(net, ctx, tgt, clean_logits, ch, pos_offset)
    nan_n = int(np.isnan(dce).sum())
    if nan_n:
        dce = np.nan_to_num(dce, nan=0.0)
    d = analyze_window(dce)
    log(f"  a*={d['a_star']} CI [{d['a_star_ci'][0]:.0f},{d['a_star_ci'][1]:.0f}] "
        f"naive={d['a_star_naive']} robust={d['a_star_robust']} "
        f"live_frac={d['live_frac']:.3f} | per-seq {d['per_seq_a_star']} "
        f"| final-token CE {ce_clean.mean():.4f} (nan->0: {nan_n})")
    return d, ce_clean


# ------------------------------------- the junk census (e073/e074/e075 instrument)

def junk_band(dce_age: np.ndarray, a_lo: int, a_hi: int):
    """Junk census of the age band [a_lo, a_hi] (inclusive; ages ascending).
    dce_age: (S, Pw) age-ascending. Junk = dCE <= -THRESH (lesion-helpful)."""
    band = dce_age[:, a_lo - 1:a_hi]                    # ages a_lo..a_hi
    n = band.shape[1]
    per_seq_frac = (band <= -THRESH).mean(1)            # (S,)
    per_seq_count = (band <= -THRESH).sum(1)
    lo, hi = bootstrap_stat(per_seq_frac[:, None], lambda m: float(m.mean()))
    return dict(
        ages=[a_lo, a_hi], n_per_seq=int(n),
        junk_frac_mean=float(per_seq_frac.mean()),
        junk_frac_mean_ci=[float(lo), float(hi)],
        junk_frac_max=float(per_seq_frac.max()),
        junk_frac_pooled=float((band <= -THRESH).mean()),
        junk_counts_per_seq=[int(c) for c in per_seq_count],
        mean_dce=float(band.mean()),
        frac_dce_negative=float((band < 0).mean()),
    )


def gen_position_range(a_star: int):
    """Generated band position range under the beyond-onset restriction.
    Context positions 64..510 are generated; age of position p = 511 - p;
    ages >= a_star -> positions 64..(511 - a_star)."""
    p_hi = P512 - int(a_star)
    return GEN_FIRST, p_hi


# ------------------------------------- NEW: the prune-vs-replace arms

def is_event(g: int) -> bool:
    return g >= PRUNE_START_G and (g - PRUNE_START_G) % PRUNE_K == 0


def prune_positions(mode: str, front: int):
    """Positions to intervene on at this event. front = the position just
    written (generation step g writes position t=64+g); age of entry p at the
    front = front - p > AGE_CUT  <=>  p <= front - AGE_CUT - 1.
    [e075 verbatim; e080 only ever calls mode='self']"""
    hi = front - AGE_CUT - 1
    out = []
    if mode in ("self", "both") and hi >= GEN_FIRST:
        out += list(range(GEN_FIRST, hi + 1))
    if mode in ("prompt", "both"):
        p_hi = min(PROMPT_POS[1], hi)
        if p_hi >= PROMPT_POS[0]:
            out += list(range(PROMPT_POS[0], p_hi + 1))
    return out


def prompt_source_of(p: int) -> int:
    """Deterministic content mapping for A-promptcopy: replaced self position
    p in [64, 387] <- prompt position q = ((p - 64) mod 63) + 1 in [1, 63].
    63 pristine sources -> 324 targets (5.14x reuse, cycling coverage)."""
    return ((p - GEN_FIRST) % NPROMPT) + PROMPT_POS[0]


@torch.no_grad()
def generate_arm(net: TinyGPT, prompts, gen: torch.Generator, mode: str,
                 noise_gen: torch.Generator | None = None):
    """Free-run 64->512 for B sequences with a V-replacement intervention on
    the SELF band (e075's A-self-prune schedule). e053b stream math VERBATIM
    (per-row draws from the shared CPU generator in row order per step); the
    fluency stats consume no generator randomness (computed after the draws).
    The intervention is applied at the TOP of each event iteration, before the
    decode that feeds the next sample, so the first affected sample is
    g = PRUNE_START_G + 1 and all arms are token-identical through position
    PRUNE_START_G + PROMPT_TOK (=164).

    Modes (identical schedule; only the VALUE written at band entry differs):
      none       no intervention.
      vzero      v <- 0            (e075 A-self-prune; G4 drift-check ref).
      noise      v <- gaussian direction * ||v||   per (layer,b,head) vector
                 (presence kept, content destroyed; drawn once from
                 noise_gen, never re-drawn — replaced-stays-replaced).
      promptcopy v[b,:,p,:] <- v[b,:,q,:], q = prompt_source_of(p)
                 (content kept from the same row's pristine prompt entry;
                 norms NOT matched).
    Positions already replaced are never touched again (for vzero this is
    bit-identical to e075's re-zero of the full band: zeroing an already-zero
    column is a no-op)."""
    Bb = len(prompts)
    idx = torch.stack(prompts)
    logits, kv = prefill_batch(net, idx)
    ce_s = np.zeros((Bb, G), float)
    ent_s = np.zeros((Bb, G), float)
    tk_s = np.zeros((Bb, G), float)
    t1_s = np.zeros((Bb, G), float)
    rk_s = np.zeros((Bb, G), float)
    replaced: set = set()
    replace_log = []
    # replacement-identity accumulators (content vs presence bookkeeping)
    n_vec = 0
    cos_abs_sum = 0.0
    norm_dev_max = 0.0                     # | ||v_new|| - ||v_old|| | (noise)
    ratio_sum = 0.0                        # ||v_new|| / ||v_old|| (promptcopy)
    for g in range(G):
        t = PROMPT_TOK + g
        if mode != "none" and is_event(g):
            band = prune_positions("self", t)
            new = sorted(set(band) - replaced)
            if new:
                sel = torch.tensor(new, dtype=torch.long)
                if mode == "vzero":
                    for (_k, v) in kv:
                        v[:, :, sel, :] = 0.0
                elif mode == "noise":
                    for (_k, v) in kv:
                        old = v[:, :, sel, :].clone()             # (B,H,n,d)
                        nrm = old.norm(dim=-1, keepdim=True)      # (B,H,n,1)
                        nz = torch.randn(old.shape, generator=noise_gen)
                        nz = nz * (nrm / nz.norm(dim=-1, keepdim=True)
                                   .clamp_min(1e-12))
                        # identity stats: norms preserved, content destroyed
                        cos = (old * nz).sum(-1) / (nrm.squeeze(-1)
                                                    * nz.norm(dim=-1)
                                                    ).clamp_min(1e-12)
                        cos_abs_sum += float(cos.abs().sum())
                        norm_dev_max = max(norm_dev_max, float(
                            (nz.norm(dim=-1) - nrm.squeeze(-1)).abs().max()))
                        n_vec += int(cos.numel())
                        v[:, :, sel, :] = nz
                elif mode == "promptcopy":
                    qsel = torch.tensor([prompt_source_of(p) for p in new],
                                        dtype=torch.long)
                    for (_k, v) in kv:
                        old = v[:, :, sel, :].clone()
                        nrm_old = old.norm(dim=-1)
                        cp = v[:, :, qsel, :]                     # pristine
                        cos = (old * cp).sum(-1) / (nrm_old
                                                    * cp.norm(dim=-1)
                                                    ).clamp_min(1e-12)
                        cos_abs_sum += float(cos.abs().sum())
                        ratio_sum += float((cp.norm(dim=-1)
                                           / nrm_old.clamp_min(1e-12)).sum())
                        n_vec += int(cos.numel())
                        v[:, :, sel, :] = cp
                else:
                    raise ValueError(mode)
                replaced.update(new)
                replace_log.append(dict(g=g, front=t, n_band=len(band),
                                        n_new=len(new), cum=len(replaced)))
        toks = torch.zeros(Bb, dtype=torch.long)
        for j in range(Bb):
            tok, ce = sample_and_ce(logits[j], gen)
            toks[j] = tok
            ce_s[j, g] = ce
        # fluency stats from this step's predictive distribution (no rng use)
        p = torch.softmax(logits.float(), -1)
        ent_s[:, g] = (-(p * p.clamp_min(1e-12).log()).sum(-1)).numpy()
        v40, _ = torch.topk(p, TOPK, dim=-1)
        tk_s[:, g] = v40.sum(-1).numpy()
        t1_s[:, g] = v40[:, 0].numpy()
        rows = torch.arange(Bb)
        own = logits[rows, toks]
        rk_s[:, g] = ((logits > own[:, None]).sum(-1) + 1).numpy()
        idx = torch.cat([idx, toks[:, None]], 1)
        if t < T_TOTAL - 1:
            logits = decode_step_batch(net, toks, t, kv)
    # intervention-identity check on the final KV cache (columns = positions)
    cache_len = kv[0][1].shape[2]
    repmask = torch.zeros(cache_len, dtype=torch.bool)
    if replaced:
        repmask[sorted(replaced)] = True
    max_pr, min_live = 0.0, float("inf")
    if mode == "vzero":
        for (_k, v) in kv:
            nv = v.abs().amax(dim=(0, 1, 3))                # (cache_len,)
            if repmask.any():
                max_pr = max(max_pr, float(nv[repmask].max()))
            min_live = min(min_live, float(nv[~repmask].min()))
        ok = bool(max_pr == 0.0 and min_live > 0.0)
    elif mode == "noise":
        for (_k, v) in kv:
            nv = v.abs().amax(dim=(0, 1, 3))
            max_pr = max(max_pr, float(nv[repmask].max()))   # nonzero noise
            min_live = min(min_live, float(nv[~repmask].min()))
        ok = bool(max_pr > 0.0 and min_live > 0.0
                  and norm_dev_max < 1e-5)                    # norms preserved
    elif mode == "promptcopy":
        ps = sorted(replaced)
        qsel = torch.tensor([prompt_source_of(p) for p in ps], dtype=torch.long)
        ok = all(torch.equal(v[:, :, ps, :], v[:, :, qsel, :])
                 for (_k, v) in kv)
        for (_k, v) in kv:
            nv = v.abs().amax(dim=(0, 1, 3))
            max_pr = max(max_pr, float(nv[repmask].max()))
            min_live = min(min_live, float(nv[~repmask].min()))
    else:
        ok = True
    repl_stats = dict(
        n_replaced_positions=len(replaced), n_v_vectors=n_vec,
        mean_abs_cos_old_vs_new=(cos_abs_sum / n_vec) if n_vec else None,
        max_norm_dev_at_replacement=norm_dev_max if mode == "noise" else None,
        mean_norm_ratio_new_over_old=(ratio_sum / n_vec)
        if mode == "promptcopy" else None,
    )
    cache = dict(cache_len=int(cache_len), n_replaced=len(replaced),
                 max_abs_v_replaced=max_pr, min_abs_v_live=min_live, ok=ok)
    return dict(idx=idx, logits=logits, ce=ce_s, ent=ent_s, topk=tk_s,
                top1=t1_s, rank=rk_s, replaced=sorted(replaced),
                replace_log=replace_log, cache=cache, repl_stats=repl_stats)


@torch.no_grad()
def clean_judge_tail(net: TinyGPT, idx: torch.Tensor):
    """CLEAN (unpruned) net scores the arm's tail: CE of tokens at positions
    448..511 under queries 447..510 of a full clean forward. Per-seq mean."""
    all_lg = manual_all_logits(net, idx)                # (B, 512, V)
    lg = all_lg[:, T_TOTAL - TAIL - 1:T_TOTAL - 1, :]   # queries 447..510
    tgt = idx[:, T_TOTAL - TAIL:]                       # targets 448..511
    lp = torch.log_softmax(lg.float(), -1)
    ce = -lp.gather(2, tgt[:, :, None]).squeeze(2).mean(1)
    return ce.numpy(), all_lg


# ---------------------------------------------------------------------- main


def main():
    assert torch.cuda.device_count() == 0, "CPU-only violated: CUDA visible"
    assert common.DEVICE == "cpu", f"common.DEVICE={common.DEVICE} != cpu"
    out_dir = run_dir("e080")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=False)

    # ---- battery: rebuild corpus + net EXACTLY as e053c/e069..e075 did
    corp = CharCorpus(REPO / "data" / "input.txt")       # seed 1337
    assert corp.vocab_size == 65
    st = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    cfg = Cfg(vocab=corp.vocab_size, n_layer=4, n_head=4, n_embd=128,
              block_size=T_TOTAL)
    net = TinyGPT(cfg)
    net.load_state_dict(sd, strict=True)
    net.eval()
    n_params = net.num_params()
    steps_in_ckpt = int(st.get("step", -1)) if isinstance(st, dict) else -1
    gates["G2_params"] = dict(params=n_params, expected=873472,
                              ok=bool(n_params == 873472))
    val_ce = estimate_loss(net, corp, "val", n_batches=12)
    gates["G1_val_ce"] = dict(val_ce=val_ce, ref=E053C_VAL_CE, tol=0.02,
                              ok=bool(abs(val_ce - E053C_VAL_CE) <= 0.02))
    log(f"e053c net loaded (ckpt step {steps_in_ckpt}, {n_params:,} params) | "
        f"val CE {val_ce:.4f} vs e053c {E053C_VAL_CE:.4f} "
        f"-> G1 {'PASS' if gates['G1_val_ce']['ok'] else 'FAIL'}")

    # ---- battery: the seed-202 8-draw, ALL 8 (e069's battery A = first 4)
    gen_p = torch.Generator().manual_seed(SEED_PROMPT)
    ix = torch.randint(len(corp.val) - PROMPT_TOK - 1, (N_PROMPTS,),
                       generator=gen_p)
    prompts8 = [corp.val[i:i + PROMPT_TOK] for i in ix]
    log(f"battery: {N_PROMPTS} prompts (seed {SEED_PROMPT}, the e069 8-draw; "
        f"first 4 = battery A); prompt0 prefix: "
        f"{corp.decode(prompts8[0])[:32]!r}")

    # ================================================== G3 protocol-identity
    # battery-A replica (B=4, seeds 202/7, mode none == e074's generate_batch
    # bit-for-bit) + the eval-512 sweep, checked against e069's stored numbers.
    log("G3 replica: battery A (B=4, seeds 202/7) free run + eval-512 sweep")
    gen4 = torch.Generator().manual_seed(SEED_SAMPLE)
    rep = generate_arm(net, prompts8[:4], gen4, "none")
    with torch.no_grad():
        full4 = manual_logits(net, rep["idx"][:, :-1], None, None, None,
                              chunk=4)
    dev4 = float((torch.softmax(rep["logits"].float(), -1)
                  - torch.softmax(full4.float(), -1)).abs().max())
    log(f"G0b replica incremental-KV vs full recompute: max prob dev "
        f"{dev4:.2e} -> {'PASS' if dev4 < 1e-4 else 'FAIL'}")
    tgt4 = rep["idx"][:, -1]
    d_rep, ce_rep = sweep_block(net, rep["idx"][:, :-1], tgt4,
                                rep["logits"], CHUNK_512)
    ref = E069_REF["eval512"]
    ages15 = d_rep["mean_age_curve"][:5]
    dev15 = float(np.max(np.abs(ages15 - np.asarray(ref["ages1_5"]))))
    ws_dev = abs(float(d_rep["mean_age_curve"][255]) - ref["winstart_dce"])
    a_match = d_rep["a_star"] == ref["a_star"]
    ps_match = list(d_rep["per_seq_a_star"]) == list(ref["per_seq"])
    ce_dev = abs(float(ce_rep.mean()) - ref["clean_ce"])
    ok3 = bool(a_match and dev15 < 0.05 and ws_dev < 0.05 and ce_dev < 0.02
               and ps_match)
    gates["G0b_replica_kvs_vs_full"] = dict(max_prob_dev=dev4, ok=bool(dev4 < 1e-4))
    gates["G3_batteryA_vs_e069"] = dict(
        a_star=d_rep["a_star"], a_star_ref=ref["a_star"], a_star_match=a_match,
        per_seq_a_star=list(d_rep["per_seq_a_star"]), per_seq_ref=ref["per_seq"],
        per_seq_match=ps_match,
        ages1_5=[float(a) for a in ages15], max_dev_ages1_5=dev15,
        winstart_dce=float(d_rep["mean_age_curve"][255]),
        winstart_dce_ref=ref["winstart_dce"], winstart_dev=ws_dev,
        clean_ce=float(ce_rep.mean()), clean_ce_ref=ref["clean_ce"],
        clean_ce_dev=ce_dev, ok=ok3)
    log(f"G3 battery A: a* {d_rep['a_star']} (ref {ref['a_star']}), per-seq "
        f"{list(d_rep['per_seq_a_star'])} (ref {ref['per_seq']}), ages1-5 "
        f"maxdev {dev15:.4f}, age-255 dev {ws_dev:.4f}, CE dev {ce_dev:.4f} "
        f"-> {'PASS' if ok3 else 'FAIL'}")

    # ================================================== THE FOUR ARMS (B=8)
    A = {}
    for arm in ARMS:
        gen = torch.Generator().manual_seed(SEED_SAMPLE)  # per-arm seed-7
        gen_n = (torch.Generator().manual_seed(SEED_NOISE)
                 if arm == "noise" else None)
        A[arm] = generate_arm(net, prompts8, gen, arm, noise_gen=gen_n)
        rs = A[arm]["repl_stats"]
        n_ev = len(A[arm]["replace_log"])
        log(f"arm {arm:10s}: generated B={B} 64->{T_TOTAL} | {n_ev} events, "
            f"{len(A[arm]['replaced'])} cumulative replaced positions | "
            f"cache ok {A[arm]['cache']['ok']}"
            + (f" | mean|cos(old,new)| {rs['mean_abs_cos_old_vs_new']:.3f}"
               if rs["mean_abs_cos_old_vs_new"] is not None else "")
            + (f", max norm dev {rs['max_norm_dev_at_replacement']:.2e}"
               if rs["max_norm_dev_at_replacement"] is not None else "")
            + (f", norm ratio {rs['mean_norm_ratio_new_over_old']:.3f}"
               if rs["mean_norm_ratio_new_over_old"] is not None else ""))

    # G5: intervention identity — pre-event token identity + counts + cache
    pre = PRUNE_START_G + PROMPT_TOK + 1                 # 165: through pos 164
    ident = all(torch.equal(A[ARMS[0]]["idx"][:, :pre], A[a]["idx"][:, :pre])
                for a in ARMS)
    diverged = {a: bool(not torch.equal(A["none"]["idx"], A[a]["idx"]))
                for a in ARMS}
    exp_repl = {"none": 0,
                "vzero": len(range(GEN_FIRST, 64 + EVENTS[-1] - AGE_CUT)),
                "noise": len(range(GEN_FIRST, 64 + EVENTS[-1] - AGE_CUT)),
                "promptcopy": len(range(GEN_FIRST, 64 + EVENTS[-1] - AGE_CUT))}
    counts_ok = all(len(A[a]["replaced"]) == exp_repl[a] for a in ARMS)
    ranges = {}
    for a in ARMS:
        pr = A[a]["replaced"]
        ranges[a] = ([pr[0], pr[-1]] if pr else None)
    ok5 = bool(ident and counts_ok
               and all(A[a]["cache"]["ok"] for a in ARMS)
               and all(diverged[a] for a in ("vzero", "noise", "promptcopy")))
    gates["G5_intervention_identity"] = dict(
        pre_event_identity_through_position=pre - 1, identical=ident,
        schedule=dict(start_g=PRUNE_START_G, K=PRUNE_K, age_cut=AGE_CUT,
                      events=EVENTS, n_events=len(EVENTS),
                      band="self only (own generated entries p in "
                           "[64, front-97], age > 96)"),
        replaced_counts={a: len(A[a]["replaced"]) for a in ARMS},
        replaced_counts_expected=exp_repl, counts_ok=counts_ok,
        replaced_ranges=ranges,
        cache_checks={a: A[a]["cache"] for a in ARMS},
        replacement_stats={a: A[a]["repl_stats"] for a in ARMS},
        promptcopy_map="q = ((p - 64) mod 63) + 1; 63 pristine prompt "
                       "sources -> 324 targets; K untouched in every arm",
        diverged_from_none=diverged, ok=ok5)
    log(f"G5 intervention identity: pre-event tokens identical through pos "
        f"{pre - 1}: {ident} | counts "
        f"{ {a: len(A[a]['replaced']) for a in ARMS} } (expected {exp_repl}) "
        f"| cache oks { {a: A[a]['cache']['ok'] for a in ARMS} } | diverged "
        f"{diverged} -> {'PASS' if ok5 else 'FAIL'}")

    # G4: drift-check — the rerun A-vzero must reproduce e075's A-self-prune
    g4 = dict(ref_file=str(E075_METRICS), ref_delta_vzero=E075_REF_DELTA_VZERO,
              ok=None, note="")
    if E075_METRICS.exists():
        with open(E075_METRICS) as f:
            e075 = json.load(f)
        ref_none = np.asarray(e075["arms"]["none"]["r1"]["per_seq"])
        ref_self = np.asarray(e075["arms"]["self"]["r1"]["per_seq"])
        new_none = A["none"]["ce"][:, -TAIL:].mean(1)
        new_vz = A["vzero"]["ce"][:, -TAIL:].mean(1)
        dev_r1 = float(max(np.abs(new_none - ref_none).max(),
                           np.abs(new_vz - ref_self).max()))
        cj_dev = abs(float(np.mean(clean_judge_tail(net, A["vzero"]["idx"])[0]))
                     - float(np.mean(e075["arms"]["self"]
                                     ["clean_judge_tail_ce"]["per_seq"])))
        ent_dev = abs(float(A["vzero"]["ent"][:, -TAIL:].mean())
                      - float(e075["arms"]["self"]["r2"]["entropy_tail"]))
        g4.update(r1_per_seq_max_dev=dev_r1, clean_judge_dev=cj_dev,
                  entropy_tail_dev=ent_dev,
                  ok=bool(dev_r1 < 1e-6 and cj_dev < 1e-6 and ent_dev < 1e-9))
        g4["note"] = ("A-vzero rerun vs e075 A-self-prune: per-seq tail CE "
                      f"max dev {dev_r1:.2e}, clean-judge dev {cj_dev:.2e}, "
                      f"entropy dev {ent_dev:.2e}")
    else:
        g4["note"] = "runs/e075/metrics.json not found; drift-check skipped"
    gates["G4_vzero_drift_vs_e075"] = g4
    log(f"G4 vzero drift vs e075: {g4['note']} -> "
        f"{'PASS' if g4['ok'] else ('SKIPPED' if g4['ok'] is None else 'FAIL')}")

    # G0b for the B=8 none arm (clean run: incremental vs full recompute)
    with torch.no_grad():
        full8 = manual_logits(net, A["none"]["idx"][:, :-1], None, None, None,
                              chunk=8)
    dev8 = float((torch.softmax(A["none"]["logits"].float(), -1)
                  - torch.softmax(full8.float(), -1)).abs().max())
    gates["G0b_noneB8_kvs_vs_full"] = dict(max_prob_dev=dev8,
                                           ok=bool(dev8 < 1e-4))
    log(f"G0b none-B8 incremental-KV vs full recompute: max prob dev "
        f"{dev8:.2e} -> {'PASS' if dev8 < 1e-4 else 'FAIL'}")

    # ---- R1/R2: tail stats per arm
    log("R1/R2: tail (final 64 generated tokens) clean CE + fluency")
    r1, r2 = {}, {}
    for arm in ARMS:
        tail_ce = A[arm]["ce"][:, -TAIL:]
        per_seq = tail_ce.mean(1)
        lo, hi = bootstrap_stat(per_seq[:, None], lambda m: float(m.mean()))
        r1[arm] = dict(per_seq=per_seq.tolist(), mean=float(per_seq.mean()),
                       ci=[float(lo), float(hi)])
        r2[arm] = dict(
            entropy_tail=float(A[arm]["ent"][:, -TAIL:].mean()),
            top40_mass_tail=float(A[arm]["topk"][:, -TAIL:].mean()),
            top1_mass_tail=float(A[arm]["top1"][:, -TAIL:].mean()),
            mean_rank_tail=float(A[arm]["rank"][:, -TAIL:].mean()))
    for arm in ARMS:
        for key in ("entropy", "top40_mass", "top1_mass"):
            r2[arm][f"{key}_drift_pct"] = float(
                (r2[arm][f"{key}_tail"] / r2["none"][f"{key}_tail"] - 1.0)
                * 100.0)
    diffs = {}
    for arm in ARMS:
        d_seq = (np.asarray(r1[arm]["per_seq"])
                 - np.asarray(r1["none"]["per_seq"]))
        lo, hi = bootstrap_stat(d_seq[:, None], lambda m: float(m.mean()))
        diffs[arm] = dict(per_seq=d_seq.tolist(), mean=float(d_seq.mean()),
                          ci=[float(lo), float(hi)],
                          n_worse=int((d_seq > 0).sum()))
    for arm in ARMS:
        log(f"  {arm:10s}: R1 tail CE {r1[arm]['mean']:.4f} CI "
            f"[{r1[arm]['ci'][0]:.4f},{r1[arm]['ci'][1]:.4f}] "
            f"(Δ vs none {diffs[arm]['mean']:+.4f}, "
            f"{diffs[arm]['n_worse']}/8 worse) | entropy "
            f"{r2[arm]['entropy_tail']:.4f} "
            f"({r2[arm]['entropy_drift_pct']:+.2f}%) | top40 mass "
            f"{r2[arm]['top40_mass_tail']:.4f} "
            f"({r2[arm]['top40_mass_drift_pct']:+.2f}%)")

    # ---- clean-judge honesty check: the clean net scores each arm's tail
    cj = {}
    for arm in ARMS:
        ce_cj, all_lg = clean_judge_tail(net, A[arm]["idx"])
        with torch.no_grad():
            last_ref = manual_logits(net, A[arm]["idx"][:, :-1], None, None,
                                     None, chunk=8)
        dev = float((all_lg[:, -2, :].float() - last_ref.float()).abs().max())
        cj[arm] = dict(per_seq=ce_cj.tolist(), mean=float(ce_cj.mean()),
                       gap_vs_online=float(ce_cj.mean() - r1[arm]["mean"]),
                       allpos_vs_lastonly_max_dev=dev)
    for arm in ARMS:
        log(f"  clean-judge tail CE {arm:10s}: {cj[arm]['mean']:.4f} "
            f"(gap vs self-scored online CE {cj[arm]['gap_vs_online']:+.4f}; "
            f"fwd-consistency dev {cj[arm]['allpos_vs_lastonly_max_dev']:.1e})")

    # ---- the junk census / attractor readout (final-step sweep per arm)
    r3 = {}
    for arm in ARMS:
        log(f"attractor sweep (arm {arm}): eval-512 fine V-zero sweep, B={B}")
        tgt = A[arm]["idx"][:, -1]
        d, ce_f = sweep_block(net, A[arm]["idx"][:, :-1], tgt,
                              A[arm]["logits"], CHUNK_512)
        p_lo, p_hi = gen_position_range(d["a_star"] if d["a_star"] is not None
                                        else BASE_A_STAR)
        r3[arm] = dict(
            a_star=d["a_star"], a_star_ci=d["a_star_ci"],
            per_seq_a_star=list(d["per_seq_a_star"]),
            junk_gen_astar_used=(d["a_star"] if d["a_star"] is not None
                                 else BASE_A_STAR),
            live_frac=d["live_frac"],
            ages1_5=[float(a) for a in d["mean_age_curve"][:5]],
            mean_curve=d["mean_age_curve"].tolist(),
            ci_lo=d["ci"][0].tolist(), ci_hi=d["ci"][1].tolist(),
            final_token_ce_mean=float(ce_f.mean()),
            final_token_ce_per_seq=ce_f.tolist(),
            band_means={
                "ages1_5": float(d["mean_age_curve"][0:5].mean()),
                "ages6_17": float(d["mean_age_curve"][5:17].mean()),
                "ages18_96": float(d["mean_age_curve"][17:96].mean()),
                "ages97_447": float(d["mean_age_curve"][96:447].mean()),
                "ages448_510": float(d["mean_age_curve"][447:510].mean()),
            },
            junk_prompt_band=junk_band(d["dce_age"], P512 - PROMPT_POS[1],
                                       P512 - PROMPT_POS[0]),
            junk_generated_band=junk_band(d["dce_age"], P512 - p_hi,
                                          P512 - p_lo),
        )
        log(f"  attractor {arm:10s}: a* {r3[arm]['a_star']} | live_frac "
            f"{r3[arm]['live_frac']:.3f} | gen-old junk "
            f"{r3[arm]['junk_generated_band']['junk_frac_mean']:.4f} | "
            f"prompt junk "
            f"{r3[arm]['junk_prompt_band']['junk_frac_mean']:.4f} "
            f"(e075 refs: none 0.000/0.196 live 0.033; vzero 0.375/0.375 "
            f"live 1.000)")
    spike_deltas = {}
    for arm in ARMS:
        dd = np.asarray(r3[arm]["ages1_5"]) - np.asarray(r3["none"]["ages1_5"])
        peak_none = int(np.argmax(r3["none"]["ages1_5"]))
        spike_deltas[arm] = dict(
            per_age=dd.tolist(), max_abs=float(np.max(np.abs(dd))),
            peak_none_age=peak_none + 1,
            peak_none=float(r3["none"]["ages1_5"][peak_none]),
            peak_arm=float(r3[arm]["ages1_5"][peak_none]),
            peak_ratio=float(r3[arm]["ages1_5"][peak_none]
                             / r3["none"]["ages1_5"][peak_none]))
    for arm in ARMS:
        log(f"  R3 spike {arm:10s}: ages1-5 "
            f"{['%.3f' % a for a in r3[arm]['ages1_5']]} (peak ratio vs none "
            f"{spike_deltas[arm]['peak_ratio']:.3f})")

    # ---- REGISTERED decision (T048 bars, frozen)
    d_vz = diffs["vzero"]["mean"]
    d_no = diffs["noise"]["mean"]
    d_pc = diffs["promptcopy"]["mean"]
    noise_cheap = bool(d_no <= CHEAP_BAR)
    pc_cheap = bool(d_pc <= CHEAP_BAR)
    scaffold = bool(noise_cheap and d_pc > d_no + GAP_BAR)
    anchoring = bool(pc_cheap and d_no > d_pc + GAP_BAR)
    clauses = dict(
        noise_cheap=dict(rule=f"A-noise Δ <= +{CHEAP_BAR}", delta=d_no,
                         fires=noise_cheap),
        promptcopy_cheap=dict(rule=f"A-promptcopy Δ <= +{CHEAP_BAR}",
                              delta=d_pc, fires=pc_cheap),
        h_statistics_scaffold=dict(
            rule=f"A-noise Δ <= +{CHEAP_BAR} AND A-promptcopy Δ > A-noise "
                 f"+ {GAP_BAR}", delta_noise=d_no, delta_promptcopy=d_pc,
            gap=d_pc - d_no, fires=scaffold),
        h_content_anchoring=dict(
            rule=f"A-promptcopy Δ <= +{CHEAP_BAR} AND A-noise Δ > "
                 f"A-promptcopy + {GAP_BAR}", delta_promptcopy=d_pc,
            delta_noise=d_no, gap=d_no - d_pc, fires=anchoring))
    arm_table = " | ".join(
        f"{a} Δ {diffs[a]['mean']:+.4f} [{diffs[a]['ci'][0]:+.3f},"
        f"{diffs[a]['ci'][1]:+.3f}]" for a in ARMS[1:])
    attractor = {a: dict(clean_judge=cj[a]["mean"],
                         cj_gap=cj[a]["gap_vs_online"],
                         entropy_drift_pct=r2[a]["entropy_drift_pct"],
                         live_frac=r3[a]["live_frac"],
                         a_star=r3[a]["a_star"],
                         prompt_junk=r3[a]["junk_prompt_band"]["junk_frac_mean"],
                         gen_junk=r3[a]["junk_generated_band"]["junk_frac_mean"])
                 for a in ARMS}
    attractor_hits = {a: bool(
        attractor[a]["cj_gap"] > 1.0                      # off-manifold escape
        or (attractor[a]["live_frac"] > 0.9               # onset loss
            and attractor[a]["a_star"] is None)
        or attractor[a]["prompt_junk"] > 0.25)            # junk inversion
        for a in ARMS}
    if scaffold:
        clause = "H-STATISTICS-SCAFFOLD"
        verdict = (f"H-STATISTICS-SCAFFOLD fires: norm-matched noise is cheap "
                   f"(Δ {d_no:+.4f} <= +{CHEAP_BAR}) while prompt-copy is not "
                   f"(Δ {d_pc:+.4f} > noise + {GAP_BAR}) — the dead band's "
                   f"load-bearing property is its PRESENCE (per-head V norms "
                   f"under intact attention mass), not its content. V-zero's "
                   f"+0.26 was a statistics collapse, not an information loss.")
    elif anchoring:
        clause = "H-CONTENT-ANCHORING"
        verdict = (f"H-CONTENT-ANCHORING fires: prompt-content replacement is "
                   f"cheap (Δ {d_pc:+.4f} <= +{CHEAP_BAR}) while norm-matched "
                   f"noise is not (Δ {d_no:+.4f} > promptcopy + {GAP_BAR}) — "
                   f"the dead band carries CONTENT anchoring (net-flavored "
                   f"text statistics), which random directions at the right "
                   f"norms cannot supply. Self-specificity narrows to "
                   f"content-of-that-kind, not literal self-tokens.")
    else:
        clause = "MIXED"
        if noise_cheap and pc_cheap:
            texture = (f"BOTH replacements are cheap (noise Δ {d_no:+.4f}, "
                       f"promptcopy Δ {d_pc:+.4f}, both <= +{CHEAP_BAR}) but "
                       f"the between-arm gap "
                       f"({abs(d_no - d_pc):.4f}) does not clear +{GAP_BAR}: "
                       f"presence OR content alone rescues the run — the two "
                       f"hypotheses are not separable at these bars; both "
                       f"suffice, only the conjunction with zero content AND "
                       f"zero presence (V-zero) kills.")
        elif not noise_cheap and not pc_cheap:
            texture = (f"NEITHER replacement is cheap (noise Δ {d_no:+.4f}, "
                       f"promptcopy Δ {d_pc:+.4f}, both > +{CHEAP_BAR}): "
                       f"neither presence nor prompt-content alone rescues — "
                       f"points beyond both registered hypotheses toward the "
                       f"run's specific self-content / trajectory anchoring.")
        elif noise_cheap:
            texture = (f"noise is cheap (Δ {d_no:+.4f}) and promptcopy is "
                       f"cheaper still (Δ {d_pc:+.4f}) but the gap "
                       f"({d_pc - d_no:+.4f}) does not clear +{GAP_BAR}: "
                       f"scaffold-leaning texture without the registered "
                       f"separation.")
        else:
            texture = (f"promptcopy is cheap (Δ {d_pc:+.4f}) and noise is "
                       f"worse (Δ {d_no:+.4f}) but the gap "
                       f"({d_no - d_pc:+.4f}) does not clear +{GAP_BAR}: "
                       f"content-anchoring-leaning texture without the "
                       f"registered separation.")
        verdict = (f"MIXED (honest texture): {texture} V-zero reference "
                   f"reproduces at Δ {d_vz:+.4f} (e075: "
                   f"{E075_REF_DELTA_VZERO:+.4f}).")
    log(f"REGISTERED DECISION [{clause}]: {verdict}")
    log(f"  clauses: { {k: v['fires'] for k, v in clauses.items()} }")
    log(f"  attractor signature (cj_gap>1 or onset-loss or junk-inversion): "
        f"{attractor_hits}")

    # ---------------------------------------------------------------- metrics
    def arm_block(arm):
        return dict(
            role={"none": "control: normal free run 64->512",
                  "vzero": "V -> 0 at self-band events (e075 A-self-prune "
                           "rerun; drift-check reference)",
                  "noise": "V <- gaussian direction * original norm per "
                           "(layer,b,head) vector at first band entry "
                           "(presence kept, content destroyed; seed "
                           f"{SEED_NOISE})",
                  "promptcopy": "V <- the same row's pristine prompt entry "
                                "q=((p-64) mod 63)+1 (content kept, norms "
                                "not matched; 63 sources -> 324 targets)"
                  }[arm],
            replace=dict(mode=arm, n_events=len(A[arm]["replace_log"]),
                         events=EVENTS if arm != "none" else [],
                         n_replaced_positions=len(A[arm]["replaced"]),
                         replaced_range=ranges[arm],
                         log=A[arm]["replace_log"],
                         stats=A[arm]["repl_stats"]),
            r1=dict(readout="mean on-line clean full-softmax CE of emitted "
                            "tokens over the final 64 generation steps "
                            "(positions 448..511), per-seq then over seqs",
                    tail_window=f"final {TAIL} generated tokens",
                    per_seq=r1[arm]["per_seq"], mean=r1[arm]["mean"],
                    ci=r1[arm]["ci"], delta_vs_none=diffs[arm]["mean"],
                    delta_vs_none_ci=diffs[arm]["ci"],
                    delta_vs_none_per_seq=diffs[arm]["per_seq"],
                    n_worse_than_none=diffs[arm]["n_worse"]),
            r2=dict(readout="tail means of per-step distribution stats; "
                            "drift vs A-none in %",
                    **r2[arm]),
            trajectory_mean_ce_per_step=A[arm]["ce"].mean(0).tolist(),
            clean_judge_tail_ce=dict(
                readout="the CLEAN unpruned net's CE on this arm's final-64 "
                        "tokens (text quality judge); gap vs the arm's own "
                        "online CE = the off-manifold attractor signature",
                per_seq=cj[arm]["per_seq"], mean=cj[arm]["mean"],
                gap_vs_online=cj[arm]["gap_vs_online"],
                delta_vs_none=float(np.mean(cj[arm]["per_seq"])
                                    - np.mean(cj["none"]["per_seq"])),
                fwd_consistency_max_dev=cj[arm]["allpos_vs_lastonly_max_dev"]),
            r3_sweep=dict(readout="final-step full 511-position V-zero sweep "
                                  "on this arm's final sequence (clean "
                                  "recompute; e069/e073/e075 instrument)",
                          **r3[arm]),
            cache_identity=A[arm]["cache"],
        )

    metrics = dict(
        experiment="e080_prune_vs_replace",
        purpose="T048's registered discriminator of e075's +0.26 V-zero cost: "
                "WHY does V-zeroing the self-generated dead band cost nats "
                "and drive generation off-manifold? H-statistics-scaffold "
                "(presence: norms/attention mass) vs H-content-anchoring "
                "(self/prompt content). Arms at the SAME prune events "
                "(g=100..420 K=32, self band age>96, replaced once): "
                "A-none; A-vzero (e075 rerun, drift-check); A-noise "
                "(norm-matched gaussian per (layer,b,head) V vector); "
                "A-promptcopy (V from the same row's pristine prompt entries, "
                "q=((p-64) mod 63)+1). K untouched in all arms. FROZEN bars: "
                "noise <= +0.05 AND promptcopy > noise + 0.05 => scaffold; "
                "promptcopy <= +0.05 AND noise > promptcopy + 0.05 => "
                "anchoring; else MIXED.",
        started=started, wall_s=elapsed(), threads=THREADS, cpu_only=True,
        net=dict(ckpt=str(CKPT), arch=dict(n_layer=4, n_head=4, n_embd=128,
                                           block_size=T_TOTAL, vocab=65),
                 params=n_params, steps_in_ckpt=steps_in_ckpt,
                 val_ce=val_ce, val_ce_e053c=E053C_VAL_CE),
        seeds=dict(
            corpus=1337, prompts=SEED_PROMPT, sampling=SEED_SAMPLE,
            noise=SEED_NOISE,
            battery="the seed-202 8-draw, ALL 8 prompts (e069's battery A = "
                    "the first 4 of this same draw). Sampling: per-arm seed-7 "
                    "re-init, shared CPU generator, row order per step "
                    "(e053b stream math). The noise arm's draws come from a "
                    "DEDICATED generator (seed 4242) that never touches the "
                    "sampling stream, so all arms share the seed-7 stream; "
                    "promptcopy consumes no randomness.",
            bootstrap=0),
        protocol=dict(
            B=B, prompt_tokens=PROMPT_TOK, t_total=T_TOTAL, temp=TEMP,
            topk=TOPK, fixed_anchor=True, threshold=THRESH,
            a_star_window=A_STAR_K, boot_n=BOOT_N, chunk=CHUNK_512,
            arms=dict(
                none="normal free run",
                vzero="v <- 0 for generated entries p in [64, front-97] at "
                      "each event (age = front - p > 96); e075 self-arm "
                      "verbatim",
                noise="v <- unit gaussian * ||v|| per (layer,b,head) d=32 "
                      "vector at the entry's FIRST band admission; never "
                      "re-drawn (replaced-stays-replaced)",
                promptcopy="v[b,:,p,:] <- v[b,:,q,:], q=((p-64) mod 63)+1, "
                           "same row b; prompt sources are pristine (never "
                           "modified in this arm)"),
            schedule=dict(
                events_g=EVENTS, K=PRUNE_K, start_g=PRUNE_START_G,
                age_cut=AGE_CUT, age_rule="age of entry p at event with "
                                          "front t = t - p > 96 <=> p <= t-97",
                timing="replacement applied at the TOP of the event "
                       "iteration, before the decode feeding the next "
                       "sample; first affected sample g=101 (position 165); "
                       "all arms token-identical through position 164",
                final_replaced="self-band positions 64..387 (324) in each "
                               "intervention arm"),
            readouts=dict(
                r1="on-line clean full-softmax CE of emitted tokens (e053b "
                   "sample_and_ce quantity), mean over final 64 generation "
                   "steps; REGISTERED bars on Δ vs A-none",
                r2="per-step full-softmax entropy + top-40 mass drift (tail "
                   "means)",
                clean_judge="clean net's CE on the arm's final-64 tokens; "
                            "gap vs online CE = off-manifold signature",
                attractor_census="final-step 511-position V-zero sweep per "
                                 "arm: a* onset, live_frac (onset loss), "
                                 "prompt-band junk inversion, generated-band "
                                 "junk")),
        gates=gates,
        replica=dict(role="battery A B=4 protocol-identity replica (G3)",
                     a_star=d_rep["a_star"], per_seq_a_star=list(
                         d_rep["per_seq_a_star"]),
                     ages1_5=[float(a) for a in ages15],
                     clean_ce=float(ce_rep.mean())),
        arms={arm: arm_block(arm) for arm in ARMS},
        registered_decision=dict(
            frozen_rules=dict(
                noise_cheap=f"A-noise Δ <= +{CHEAP_BAR}",
                promptcopy_cheap=f"A-promptcopy Δ <= +{CHEAP_BAR}",
                h_statistics_scaffold=f"A-noise Δ <= +{CHEAP_BAR} AND "
                                      f"A-promptcopy Δ > A-noise + {GAP_BAR}",
                h_content_anchoring=f"A-promptcopy Δ <= +{CHEAP_BAR} AND "
                                    f"A-noise Δ > A-promptcopy + {GAP_BAR}",
                else_="MIXED (report the honest texture)"),
            r1_tail_clean_ce={arm: r1[arm]["mean"] for arm in ARMS},
            r1_tail_clean_ce_ci={arm: r1[arm]["ci"] for arm in ARMS},
            r1_delta_vs_none={arm: diffs[arm]["mean"] for arm in ARMS},
            r1_delta_vs_none_ci={arm: diffs[arm]["ci"] for arm in ARMS},
            r1_n_worse_than_none={arm: diffs[arm]["n_worse"] for arm in ARMS},
            vzero_drift_vs_e075=dict(
                delta_now=d_vz, delta_e075=E075_REF_DELTA_VZERO,
                dev=abs(d_vz - E075_REF_DELTA_VZERO)),
            r2_fluency=dict(
                entropy_drift_pct={a: r2[a]["entropy_drift_pct"]
                                   for a in ARMS},
                top40_drift_pct={a: r2[a]["top40_mass_drift_pct"]
                                 for a in ARMS},
                bar_pct=R2_DRIFT_MAX * 100),
            attractor=dict(
                readout="the e075 attractor signature per arm: clean-judge "
                        "gap (>1 nat = off-manifold), live_frac/a* (onset "
                        "loss), prompt-band junk inversion (>0.25)",
                per_arm=attractor, signature_fires=attractor_hits),
            r3_spike=dict(prediction="unchanged live spike (e075 R3 carryover)",
                          ages1_5={arm: r3[arm]["ages1_5"] for arm in ARMS},
                          spike_deltas=spike_deltas,
                          a_star={arm: r3[arm]["a_star"] for arm in ARMS},
                          band_means={arm: r3[arm]["band_means"]
                                      for arm in ARMS}),
            clauses=clauses, clause=clause, verdict=verdict),
    )
    save_json(out_dir / "metrics.json", metrics)
    log("metrics.json written")

    plot(out_dir / "prune_vs_replace.png", metrics)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot


def plot(path: Path, M: dict):
    dec = M["registered_decision"]
    arms_m = M["arms"]
    r1 = dec["r1_tail_clean_ce"]
    r1ci = dec["r1_tail_clean_ce_ci"]
    dd = dec["r1_delta_vs_none"]
    ddci = dec["r1_delta_vs_none_ci"]
    att = dec["attractor"]["per_arm"]
    ages = np.arange(1, T_TOTAL)
    labels = ["A-none\n(control)", "A-vzero\n(V→0; e075 ref +0.2619)",
              "A-noise\n(V→‖V‖·gauss)", "A-promptcopy\n(V→prompt content)"]
    cols = ["tab:gray", "tab:red", "tab:blue", "tab:orange"]
    fig, axes = plt.subplots(2, 4, figsize=(25, 11))
    ax1, ax2, ax3, ax4 = axes[0]
    ax5, ax6, ax7, ax8 = axes[1]

    # ---- panel 1: R1 tail clean CE per arm + registered bars
    x = np.arange(len(ARMS))
    vals = [r1[a] for a in ARMS]
    cis = [r1ci[a] for a in ARMS]
    ax1.bar(x, vals, 0.62, color=cols, alpha=0.85, edgecolor="k", lw=0.5)
    ax1.errorbar(x, vals, yerr=[[v - c[0] for v, c in zip(vals, cis)],
                                [c[1] - v for v, c in zip(vals, cis)]],
                 fmt="none", ecolor="k", lw=1.0, capsize=3)
    base = r1["none"]
    ax1.axhline(base, color="k", lw=0.8, ls="-")
    ax1.axhline(base + CHEAP_BAR, color="tab:green", ls="--", lw=1.4)
    ax1.text(len(ARMS) - 0.45, base + CHEAP_BAR + 0.002,
             f"cheap bar: none + {CHEAP_BAR} (applies to noise & promptcopy)",
             fontsize=8, color="tab:green", ha="right")
    for xi, a in zip(x, ARMS):
        tag = ""
        if a == "noise":
            tag = (" [CHEAP]" if dec["clauses"]["noise_cheap"]["fires"]
                   else " [costs]")
        if a == "promptcopy":
            tag = (" [CHEAP]" if dec["clauses"]["promptcopy_cheap"]["fires"]
                   else " [costs]")
        if a == "vzero":
            tag = f" (e075 {E075_REF_DELTA_VZERO:+.4f})"
        ax1.text(xi, vals[xi] + 0.004,
                 f"{vals[xi]:.4f}\nΔ {dd[a]:+.4f}{tag}", ha="center",
                 fontsize=8.5)
    ax1.set_xticks(x, labels, fontsize=8)
    ax1.set_ylabel("tail clean CE (nats, final 64 generated tokens)")
    lo_y = min(vals + [base]) - 0.02
    hi_y = max(vals + [base + CHEAP_BAR]) + 0.03
    ax1.set_ylim(lo_y, hi_y)
    ax1.set_title("E080-1 — R1: tail clean CE per arm (bootstrap CI over "
                  "B=8 seqs)", fontsize=10)

    # ---- panel 2: clean-judge tail CE (the off-manifold honesty check)
    cjv = [arms_m[a]["clean_judge_tail_ce"]["mean"] for a in ARMS]
    gap = [arms_m[a]["clean_judge_tail_ce"]["gap_vs_online"] for a in ARMS]
    ax2.bar(x, cjv, 0.62, color=cols, alpha=0.85, edgecolor="k", lw=0.5)
    for xi, (cv, gv) in enumerate(zip(cjv, gap)):
        ax2.text(xi, cv + 0.05, f"{cv:.3f}\ngap {gv:+.3f}", ha="center",
                 fontsize=8.5)
    ax2.axhline(cjv[0], color="k", lw=0.8)
    ax2.set_xticks(x, labels, fontsize=8)
    ax2.set_ylabel("clean-judge tail CE (nats)")
    ax2.set_title("E080-2 — clean net judges each arm's tail (gap = judge − "
                  "self-score; e075 vzero 6.40)", fontsize=10)

    # ---- panel 3: the attractor census (junk inversion + onset loss)
    w = 0.27
    pj = [att[a]["prompt_junk"] for a in ARMS]
    gj = [att[a]["gen_junk"] for a in ARMS]
    lf = [att[a]["live_frac"] for a in ARMS]
    ax3.bar(x - w, pj, w, color="tab:purple", alpha=0.85, label="prompt-band junk")
    ax3.bar(x, gj, w, color="tab:olive", alpha=0.85, label="gen-band junk")
    ax3.bar(x + w, lf, w, color="tab:cyan", alpha=0.85, label="live_frac (onset loss)")
    for xi, (p, g, l) in enumerate(zip(pj, gj, lf)):
        ax3.text(xi - w, p + 0.01, f"{p:.3f}", ha="center", fontsize=7.5)
        ax3.text(xi, g + 0.01, f"{g:.3f}", ha="center", fontsize=7.5)
        ax3.text(xi + w, l + 0.01, f"{l:.3f}", ha="center", fontsize=7.5)
    ax3.axhline(0.25, color="tab:purple", ls=":", lw=1.0)
    ax3.set_xticks(x, [f"A-{a}" for a in ARMS], fontsize=8)
    ax3.set_ylim(0, 1.05)
    ax3.legend(fontsize=8)
    ax3.set_title("E080-3 — attractor census (e075 refs: none pj 0.000/gj "
                  "0.196/lf 0.033; vzero 0.375/0.375/1.000)", fontsize=9.5)

    # ---- panel 4: R2 fluency drift (%)
    drift_arms = ["vzero", "noise", "promptcopy"]
    x4 = np.arange(len(drift_arms))
    ent = [arms_m[a]["r2"]["entropy_drift_pct"] for a in drift_arms]
    tk = [arms_m[a]["r2"]["top40_mass_drift_pct"] for a in drift_arms]
    ax4.bar(x4 - 0.19, ent, 0.38, color="tab:blue", alpha=0.85,
            label="entropy drift")
    ax4.bar(x4 + 0.19, tk, 0.38, color="tab:cyan", alpha=0.85,
            label="top-40 mass drift")
    ax4.axhspan(-R2_DRIFT_MAX * 100, R2_DRIFT_MAX * 100, color="tab:green",
                alpha=0.10)
    ax4.axhline(R2_DRIFT_MAX * 100, color="tab:green", ls="--", lw=1.1)
    ax4.axhline(-R2_DRIFT_MAX * 100, color="tab:green", ls="--", lw=1.1)
    ax4.text(len(drift_arms) - 0.5, R2_DRIFT_MAX * 100 + 0.3,
             f"±{R2_DRIFT_MAX * 100:.0f}% bar", fontsize=8,
             color="tab:green", ha="right")
    for xi, (e, t) in enumerate(zip(ent, tk)):
        ax4.text(xi - 0.19, e + (0.4 if e >= 0 else -0.9), f"{e:+.2f}%",
                 ha="center", fontsize=8)
        ax4.text(xi + 0.19, t + (0.4 if t >= 0 else -0.9), f"{t:+.2f}%",
                 ha="center", fontsize=8)
    ax4.axhline(0, color="k", lw=0.7)
    ax4.set_xticks(x4, [f"A-{a}" for a in drift_arms], fontsize=9)
    ax4.set_ylabel("drift vs A-none (%)")
    lo4 = min(ent + tk + [-6]); hi4 = max(ent + tk + [6])
    ax4.set_ylim(lo4 - 1.5, hi4 + 1.5)
    ax4.legend(fontsize=8)
    ax4.set_title("E080-4 — R2: fluency drift (tail entropy / top-40 mass)",
                  fontsize=10)

    # ---- panel 5: generation CE trajectory
    for a, c in zip(ARMS, cols):
        tr = np.asarray(arms_m[a]["trajectory_mean_ce_per_step"])
        win = 16
        sm = np.convolve(tr, np.ones(win) / win, mode="valid")
        ax5.plot(np.arange(win - 1, G), sm, lw=1.2, color=c,
                 label=f"A-{a} (tail {r1[a]:.4f})")
    ax5.axvline(PRUNE_START_G, color="tab:red", ls="--", lw=1.2)
    ax5.text(PRUNE_START_G + 3, ax5.get_ylim()[0] + 0.05,
             f"first event g={PRUNE_START_G}", fontsize=8, color="tab:red")
    ax5.axvline(G - TAIL, color="k", ls=":", lw=1.0)
    ax5.text(G - TAIL - 2, ax5.get_ylim()[0] + 0.05, " tail window",
             fontsize=8, ha="right")
    ax5.set_xlabel("generation step g (0..447)")
    ax5.set_ylabel("mean emitted-token CE (nats, 16-step smooth)")
    ax5.legend(fontsize=8)
    ax5.set_title("E080-5 — free-run CE trajectory per arm (B=8 mean); arms "
                  "identical until g=101", fontsize=10)

    # ---- panel 6: per-age utility curves overlay (final sweep)
    for a, c in zip(ARMS, cols):
        ax6.plot(ages, np.asarray(arms_m[a]["r3_sweep"]["mean_curve"]), lw=1.1,
                 color=c,
                 label=f"A-{a} (a*={arms_m[a]['r3_sweep']['a_star']})")
        ax6.fill_between(ages, np.asarray(arms_m[a]["r3_sweep"]["ci_lo"]),
                         np.asarray(arms_m[a]["r3_sweep"]["ci_hi"]),
                         alpha=0.06, color=c)
    ax6.axhline(0, color="k", lw=0.6)
    ax6.axhline(THRESH, color="gray", ls=":", lw=0.9)
    ax6.axhline(-THRESH, color="gray", ls=":", lw=0.9)
    ax6.axvspan(448, 511, color="tab:purple", alpha=0.07)
    ax6.text(451, 0.30, "prompt band (63)", fontsize=8, color="tab:purple",
             rotation=90, va="top")
    ax6.axvspan(124, 447, color="tab:green", alpha=0.05)
    ax6.text(285, 0.185, "self band replaced at run time\n(positions 64..387)",
             fontsize=8, color="tab:green", ha="center")
    ax6.axvline(96, color="tab:green", ls="--", lw=1.0)
    ax6.text(96, 0.05, " age cut 96", fontsize=8, color="tab:green")
    ax6.set_xlim(0, T_TOTAL)
    ax6.set_ylim(-0.06, 0.34)
    ax6.set_xlabel("cache age (tokens)")
    ax6.set_ylabel("V-zero dCE (nats, mean over B=8)")
    ax6.legend(fontsize=8)
    ax6.set_title("E080-6 — per-age utility curves AFTER the run (final-step "
                  "sweep; young spike off-scale)", fontsize=10)

    # ---- panel 7: zoom ages 1-60 (the live spike)
    zoom = 60
    for a, c in zip(ARMS, cols):
        ax7.plot(ages[:zoom],
                 np.asarray(arms_m[a]["r3_sweep"]["mean_curve"])[:zoom],
                 lw=1.3, color=c, label=f"A-{a}")
    ax7.axhline(THRESH, color="gray", ls=":", lw=1.0)
    ax7.text(zoom, THRESH, f" thresh {THRESH}", fontsize=8, va="bottom",
             ha="right", color="gray")
    ax7.axhline(0, color="k", lw=0.6)
    sd = dec["r3_spike"]["spike_deltas"]
    ax7.set_title("E080-7 — zoom: the live spike after each run\nnone "
                  f"{['%.2f' % a for a in arms_m['none']['r3_sweep']['ages1_5']]}"
                  f" | vzero r {sd['vzero']['peak_ratio']:.2f} | noise r "
                  f"{sd['noise']['peak_ratio']:.2f} | promptcopy r "
                  f"{sd['promptcopy']['peak_ratio']:.2f}", fontsize=9.5)
    ax7.set_xlabel("cache age (tokens)")
    ax7.set_ylabel("V-zero dCE (nats)")
    ax7.legend(fontsize=8)

    # ---- panel 8: verdict text
    ax8.axis("off")
    cl = dec["clauses"]
    lines = [
        "REGISTERED (T048, frozen):",
        f"  A-noise      Δ <= +{CHEAP_BAR}:                    "
        f"[{'FIRES' if cl['noise_cheap']['fires'] else 'no'}] "
        f"Δ {dd['noise']:+.4f} CI [{ddci['noise'][0]:+.3f},"
        f"{ddci['noise'][1]:+.3f}]",
        f"  A-promptcopy Δ <= +{CHEAP_BAR}:                    "
        f"[{'FIRES' if cl['promptcopy_cheap']['fires'] else 'no'}] "
        f"Δ {dd['promptcopy']:+.4f} CI [{ddci['promptcopy'][0]:+.3f},"
        f"{ddci['promptcopy'][1]:+.3f}]",
        f"  H-statistics-scaffold: noise cheap AND pc > noise + {GAP_BAR}: "
        f"[{'FIRES' if cl['h_statistics_scaffold']['fires'] else 'no'}] "
        f"(gap {dd['promptcopy'] - dd['noise']:+.4f})",
        f"  H-content-anchoring:  pc cheap AND noise > pc + {GAP_BAR}:     "
        f"[{'FIRES' if cl['h_content_anchoring']['fires'] else 'no'}] "
        f"(gap {dd['noise'] - dd['promptcopy']:+.4f})",
        f"  A-vzero drift-check: Δ {dd['vzero']:+.4f} vs e075 "
        f"{E075_REF_DELTA_VZERO:+.4f} (dev "
        f"{dec['vzero_drift_vs_e075']['dev']:.2e})",
        "",
        "arm |  R1 tail CE |    Δ vs none | clean-judge | ent drift | "
        "live_frac | pjunk | gjunk",
    ] + [
        f"  {a:10s} {r1[a]:.4f}  {dd[a]:+.4f}  {att[a]['clean_judge']:9.4f}  "
        f"{att[a]['entropy_drift_pct']:+7.2f}%  {att[a]['live_frac']:.3f}    "
        f"{att[a]['prompt_junk']:.3f}  {att[a]['gen_junk']:.3f}"
        for a in ARMS
    ] + [
        f"  attractor signature fires: {dec['attractor']['signature_fires']}",
        "",
        f"VERDICT [{dec['clause']}]:",
    ] + [f"  {wd}" for wd in _wrap(dec["verdict"], 100)]
    ax8.text(0.02, 0.97, "E080 — T048 discriminator: prune vs replace",
             fontsize=13, weight="bold", va="top")
    for i, t in enumerate(lines):
        ax8.text(0.02, 0.925 - i * 0.0365, t, fontsize=8.6, va="top",
                 family="monospace")

    fig.suptitle("E080 — prune-vs-replace at the same events (T048) | clause: "
                 f"{dec['clause']} | Δvzero {dd['vzero']:+.4f} | Δnoise "
                 f"{dd['noise']:+.4f} | Δpromptcopy {dd['promptcopy']:+.4f} "
                 f"| bars: cheap ≤ +{CHEAP_BAR}, gap > +{GAP_BAR}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _wrap(text: str, width: int):
    import textwrap
    return textwrap.wrap(text, width=width)


if __name__ == "__main__":
    main()
