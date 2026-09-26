"""E075 — T045's REGISTERED intervention test: source-aware pruning (P3 step 1).

Design: scratch/e075_design.md (Rule-0 memo, frozen bars). T045 (e073+e074,
closed): negative-utility cache entries concentrate in the model's OWN
generated tokens (4/4 nets; late-gen 5.5x early-gen; prompt entries ~never
hurt even when shuffled). The 2025 KV-pruning literature evicts by
attention/recency/sink — never by token SOURCE. This is the intervention
test: prune by source MID-GENERATION and measure the tail.

Arms (fixed-anchor protocol, B=8, e053c net, seeds documented):
  1. A-none (control): normal free run 64->512.
  2. A-self-prune: from generation step ~100 on, every K=32 steps, V-zero the
     model's own generated entries with age > 96 (dead band beyond the live
     window + shoulder).
  3. A-prompt-prune (placebo): V-zero CORPUS-prompt entries at the same
     matched ages/times (63 old entries — the band T045 says is harmless;
     position 0 = sequence-start anchor excluded, e073 convention).
  4. A-both-prune: both bands (report-only; total-cache-size control).

REGISTERED readouts + bars (FROZEN in the design):
  - R1 tail clean CE (final 64 generated tokens): A-self-prune <= A-none
    + 0.01 nats (no cost, possibly gain) AND A-prompt-prune >= A-none + 0.05
    (placebo hurts). BOTH firing = source-specific pruning value.
  - R2 fluency: entropy/top-k drift <= 5% in A-self-prune.
  - R3 the per-age utility curve AFTER pruning (does pruning self-old shift
    the live spike? predict: unchanged).
  - Kill: A-self-prune costs > +0.05 nats -> the "dead weight" is not
    prunable at the value level after all — report as the honest boundary of
    the T045 implication.

R1 instrument: the free run's own on-line next-token CE (clean full-softmax
CE of each emitted token under the run's actual cache state — e053b's
sample_and_ce quantity), mean over the final 64 generation steps. Secondary
honesty check: the same tail scored by the CLEAN (unpruned) net
("clean-judge tail CE") — catches a self-scored CE gaming itself.
R2 instrument: per-step full-softmax entropy, top-40 (sampling k) mass,
top-1 mass, emitted-token rank; tail means; drift vs A-none.
R3 instrument: the e069/e073 final-step full 511-position V-zero sweep on
each arm's final sequence (clean recompute on the post-pruning stream).

Intervention mechanics: incremental-KV decode (e053b/e069 batched math,
bit-identical stream) with permanent in-place V-zero of cache columns. At
each event (generation steps g = 100, 132, ..., 420; K=32), BEFORE the decode
that feeds the next sample, with front = the position just written t=64+g:
entries p with age t - p > 96 (p <= t-97) in the arm's SOURCE band are
V-zeroed. Self band = positions 64..t-97; prompt band = positions 1..63
(age >= 101 at the first event). First affected sample = g=101; all arms are
token-identical through position 164 (gated). Final cumulative pruned: self
= positions 64..387 (324 entries), prompt = 63, both = 387.

Battery: the seed-202 8-draw, ALL 8 prompts (e069's battery A = the first 4
of this same draw; the +4 extension is the deterministic remainder — no new
prompt seed). Sampling: per-arm seed-7 re-init, shared CPU generator, row
order per step (e053b stream math); because the row-order stream interleaves
8 rows, rows 0-3 of the B=8 arms do NOT replicate battery A's B=4 token
stream — hence a separate B=4 battery-A replica arm runs FIRST as the
protocol-identity gate G3 (verbatim e069/e074 checks).

Run:     python lab/e075_source_prune.py
Outputs: runs/e075/metrics.json + runs/e075/source_prune.png
Envelope: NO training, NO new automations; CPU-only, single step, minutes.
No NOTES/THINKING/QUEUE/STATE edits; no commit (the dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu" (e053b..e074)
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

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
SEED_PROMPT, SEED_SAMPLE = 202, 7         # e053b/e053c/e069/e070/e072/e074 seeds
N_PROMPTS = 8                             # the seed-202 8-draw, ALL 8 (B=8)
B = 8
BOOT_N = 1000
THREADS = 12
torch.set_num_threads(THREADS)

P512 = T_TOTAL - 1                        # 511 sweep positions (native frame)
THRESH = 0.01                             # dead-weight / junk threshold (nats)
A_STAR_K = 5                              # sustained window (e053b verbatim)
CHUNK_512 = 64                            # e069/e070/e072/e074 measured chunk
BASE_A_STAR = 6                           # e053c/e069/e074 eval-512 onset;
                                          # fallback restriction when an arm's
                                          # own sweep has no onset (a*=None)

G = T_TOTAL - PROMPT_TOK                  # 448 generation steps (g = 0..447)

# ---- the intervention schedule (design verbatim) ---------------------------
PRUNE_START_G = 100                       # "from generation step ~100 on"
PRUNE_K = 32                              # "every K=32 steps"
AGE_CUT = 96                              # "age > 96" (dead band beyond
                                          # live window + shoulder)
PROMPT_POS = (1, 63)                      # 63 corpus-prompt entries (pos 0 =
                                          # seq-start anchor excluded, e073)
GEN_FIRST = 64                            # first generated position
TAIL = 64                                 # R1/R2 tail window (final 64 gen)

ARMS = ["none", "self", "prompt", "both"]
EVENTS = [g for g in range(PRUNE_START_G, G, PRUNE_K)]   # 100..420, 11 events

# ---- REGISTERED decision numbers (FROZEN, design verbatim) ------------------
R1_SELF_TOL = 0.01                        # A-self-prune <= A-none + 0.01
R1_PROMPT_BAR = 0.05                      # A-prompt-prune >= A-none + 0.05
R2_DRIFT_MAX = 0.05                       # entropy/top-k drift <= 5%
KILL_BAR = 0.05                           # A-self-prune costs > +0.05

# ---- e069 stored readouts (protocol-identity instrument gates) -------------
E069_REF = {"eval512": {"a_star": 6, "ci": [4.0, 8.0], "clean_ce": 0.4528,
                        "ages1_5": [1.742, 4.467, 2.319, 0.914, 0.321],
                        "per_seq": [8, 4, 6, 3], "winstart_dce": 0.0069}}
CKPT = REPO / "runs" / "checkpoints" / "e053c_ctx512.pt"
E053C_VAL_CE = 1.5226792494455974         # the net being interrogated
T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ------------------------------------------------- machinery (VERBATIM e053c/e074)

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


# ------------------------------------- the junk census (e073/e074 instrument)

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


# ------------------------------------- NEW: the source-aware pruning arms

def is_event(g: int) -> bool:
    return g >= PRUNE_START_G and (g - PRUNE_START_G) % PRUNE_K == 0


def prune_positions(mode: str, front: int):
    """Positions to V-zero at this event. front = the position just written
    (generation step g writes position t=64+g); age of entry p at the front =
    front - p > AGE_CUT  <=>  p <= front - AGE_CUT - 1."""
    hi = front - AGE_CUT - 1
    out = []
    if mode in ("self", "both") and hi >= GEN_FIRST:
        out += list(range(GEN_FIRST, hi + 1))
    if mode in ("prompt", "both"):
        p_hi = min(PROMPT_POS[1], hi)
        if p_hi >= PROMPT_POS[0]:
            out += list(range(PROMPT_POS[0], p_hi + 1))
    return out


@torch.no_grad()
def generate_arm(net: TinyGPT, prompts, gen: torch.Generator, mode: str):
    """Free-run 64->512 for B sequences with optional source-aware V-zero
    pruning. e053b stream math VERBATIM (per-row draws from the shared CPU
    generator in row order per step); the fluency stats consume no generator
    randomness (computed after the draws). Pruning is applied at the TOP of
    each event iteration, before the decode that feeds the next sample, so
    the first affected sample is g = PRUNE_START_G + 1 and all arms are
    token-identical through position PRUNE_START_G + PROMPT_TOK (=164)."""
    Bb = len(prompts)
    idx = torch.stack(prompts)
    logits, kv = prefill_batch(net, idx)
    ce_s = np.zeros((Bb, G), float)
    ent_s = np.zeros((Bb, G), float)
    tk_s = np.zeros((Bb, G), float)
    t1_s = np.zeros((Bb, G), float)
    rk_s = np.zeros((Bb, G), float)
    pruned: set = set()
    prune_log = []
    for g in range(G):
        t = PROMPT_TOK + g
        if mode != "none" and is_event(g):
            pos = prune_positions(mode, t)
            if pos:
                sel = torch.tensor(pos, dtype=torch.long)
                for (_k, v) in kv:
                    v[:, :, sel, :] = 0.0
                new = sorted(set(pos) - pruned)
                pruned.update(pos)
                prune_log.append(dict(g=g, front=t, n_zeroed=len(pos),
                                      n_new=len(new), cum=len(pruned)))
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
    prmask = torch.zeros(cache_len, dtype=torch.bool)
    if pruned:
        prmask[sorted(pruned)] = True
    max_pr, min_live = 0.0, float("inf")
    for (_k, v) in kv:
        nv = v.abs().amax(dim=(0, 1, 3))                # (cache_len,)
        if prmask.any():
            max_pr = max(max_pr, float(nv[prmask].max()))
        min_live = min(min_live, float(nv[~prmask].min()))
    cache = dict(cache_len=int(cache_len), n_pruned=len(pruned),
                 max_abs_v_pruned=max_pr, min_abs_v_live=min_live,
                 ok=bool(max_pr == 0.0 and min_live > 0.0))
    return dict(idx=idx, logits=logits, ce=ce_s, ent=ent_s, topk=tk_s,
                top1=t1_s, rank=rk_s, pruned=sorted(pruned),
                prune_log=prune_log, cache=cache)


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
    out_dir = run_dir("e075")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=False)

    # ---- battery: rebuild corpus + net EXACTLY as e053c/e069..e074 did
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
        A[arm] = generate_arm(net, prompts8, gen, arm)
        n_ev = len(A[arm]["prune_log"])
        log(f"arm {arm:6s}: generated B={B} 64->{T_TOTAL} | {n_ev} prune "
            f"events, {len(A[arm]['pruned'])} cumulative V-zeroed positions | "
            f"cache ok {A[arm]['cache']['ok']} "
            f"(max|v| pruned {A[arm]['cache']['max_abs_v_pruned']:.1e}, "
            f"min live {A[arm]['cache']['min_abs_v_live']:.3f})")

    # G5: intervention identity — pre-event token identity + cache checks +
    # schedule bookkeeping + post-event divergence
    pre = PRUNE_START_G + PROMPT_TOK + 1                 # 165: through pos 164
    ident = all(torch.equal(A[ARMS[0]]["idx"][:, :pre], A[a]["idx"][:, :pre])
                for a in ARMS)
    diverged = {a: bool(not torch.equal(A["none"]["idx"], A[a]["idx"]))
                for a in ARMS}
    exp_pruned = {"none": 0, "self": len(range(GEN_FIRST, 64 + EVENTS[-1]
                                               - AGE_CUT)),
                  "prompt": PROMPT_POS[1] - PROMPT_POS[0] + 1, "both": None}
    exp_pruned["both"] = exp_pruned["self"] + exp_pruned["prompt"]
    counts_ok = all(len(A[a]["pruned"]) == exp_pruned[a] for a in ARMS)
    ranges = {}
    for a in ARMS:
        pr = A[a]["pruned"]
        ranges[a] = ([pr[0], pr[-1]] if pr else None)
    ok5 = bool(ident and counts_ok
               and all(A[a]["cache"]["ok"] for a in ARMS)
               and all(diverged[a] for a in ("self", "prompt", "both")))
    gates["G5_intervention_identity"] = dict(
        pre_event_identity_through_position=pre - 1, identical=ident,
        prune_schedule=dict(start_g=PRUNE_START_G, K=PRUNE_K, age_cut=AGE_CUT,
                            events=EVENTS, n_events=len(EVENTS)),
        pruned_counts={a: len(A[a]["pruned"]) for a in ARMS},
        pruned_counts_expected=exp_pruned, counts_ok=counts_ok,
        pruned_ranges=ranges,
        cache_checks={a: A[a]["cache"] for a in ARMS},
        diverged_from_none=diverged, ok=ok5)
    log(f"G5 intervention identity: pre-event tokens identical through pos "
        f"{pre - 1}: {ident} | counts "
        f"{ {a: len(A[a]['pruned']) for a in ARMS} } (expected {exp_pruned}) "
        f"| diverged {diverged} -> {'PASS' if ok5 else 'FAIL'}")

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
                          ci=[float(lo), float(hi)])
    for arm in ARMS:
        log(f"  {arm:6s}: R1 tail CE {r1[arm]['mean']:.4f} CI "
            f"[{r1[arm]['ci'][0]:.4f},{r1[arm]['ci'][1]:.4f}] "
            f"(Δ vs none {diffs[arm]['mean']:+.4f}) | entropy "
            f"{r2[arm]['entropy_tail']:.4f} ({r2[arm]['entropy_drift_pct']:+.2f}%) "
            f"| top40 mass {r2[arm]['top40_mass_tail']:.4f} "
            f"({r2[arm]['top40_mass_drift_pct']:+.2f}%)")

    # ---- clean-judge secondary: the clean net scores each arm's tail
    cj = {}
    for arm in ARMS:
        ce_cj, all_lg = clean_judge_tail(net, A[arm]["idx"])
        with torch.no_grad():
            last_ref = manual_logits(net, A[arm]["idx"][:, :-1], None, None,
                                     None, chunk=8)
        dev = float((all_lg[:, -2, :].float() - last_ref.float()).abs().max())
        cj[arm] = dict(per_seq=ce_cj.tolist(), mean=float(ce_cj.mean()),
                       allpos_vs_lastonly_max_dev=dev)
    for arm in ARMS:
        log(f"  clean-judge tail CE {arm:6s}: {cj[arm]['mean']:.4f} "
            f"(fwd-consistency dev {cj[arm]['allpos_vs_lastonly_max_dev']:.1e})")

    # ---- R3: the per-age utility curve AFTER pruning (final-step sweep/arm)
    r3 = {}
    for arm in ARMS:
        log(f"R3 sweep (arm {arm}): eval-512 fine V-zero sweep, B={B}")
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
        log(f"  R3 {arm:6s}: a* {r3[arm]['a_star']} | ages1-5 "
            f"{['%.3f' % a for a in r3[arm]['ages1_5']]} (Δpeak "
            f"{spike_deltas[arm]['peak_ratio'] - 1:+.1%} vs none) | live_frac "
            f"{r3[arm]['live_frac']:.3f} | gen-old junk "
            f"{r3[arm]['junk_generated_band']['junk_frac_mean']:.4f} | "
            f"prompt junk "
            f"{r3[arm]['junk_prompt_band']['junk_frac_mean']:.4f}")

    # ---- REGISTERED decision (frozen bars, design verbatim)
    d_self = diffs["self"]["mean"]
    d_prompt = diffs["prompt"]["mean"]
    d_both = diffs["both"]["mean"]
    r1a = bool(d_self <= R1_SELF_TOL)
    r1b = bool(d_prompt >= R1_PROMPT_BAR)
    kill = bool(d_self > KILL_BAR)
    ent_d = r2["self"]["entropy_drift_pct"]
    tk_d = r2["self"]["top40_mass_drift_pct"]
    r2_ok = bool(abs(ent_d) <= R2_DRIFT_MAX * 100.0
                 and abs(tk_d) <= R2_DRIFT_MAX * 100.0)
    clauses = dict(
        r1_self_bar=dict(rule=f"A-self-prune <= A-none + {R1_SELF_TOL}",
                         delta=d_self, fires=r1a),
        r1_prompt_bar=dict(rule=f"A-prompt-prune >= A-none + {R1_PROMPT_BAR}",
                           delta=d_prompt, fires=r1b),
        r2_fluency=dict(rule=f"|entropy| and |top-40 mass| drift <= "
                             f"{R2_DRIFT_MAX:.0%} in A-self-prune",
                        entropy_drift_pct=ent_d, top40_drift_pct=tk_d,
                        fires=r2_ok),
        kill=dict(rule=f"A-self-prune costs > +{KILL_BAR} nats",
                  delta=d_self, fires=kill))
    if kill:
        clause = "KILL"
        verdict = (f"KILL criterion fires: A-self-prune tail clean CE cost "
                   f"{d_self:+.4f} nats > +{KILL_BAR} — the 'dead weight' is "
                   f"not prunable at the value level after all; the honest "
                   f"boundary of the T045 implication.")
    elif r1a and r1b:
        clause = "SOURCE-SPECIFIC-PRUNING-VALUE"
        verdict = (f"BOTH R1 bars fire: A-self-prune costs {d_self:+.4f} "
                   f"<= +{R1_SELF_TOL} nats (dead weight pruned at no cost) "
                   f"AND A-prompt-prune costs {d_prompt:+.4f} >= +"
                   f"{R1_PROMPT_BAR} nats (the placebo band is NOT prunable "
                   f"free). Source-specific pruning value confirmed — pruning "
                   f"value tracks token SOURCE (T045), not age.")
    elif r1a:
        clause = "PARTIAL-SELF-ONLY"
        verdict = (f"PARTIAL: the self-prune bar fires (cost {d_self:+.4f} "
                   f"<= +{R1_SELF_TOL}) but the placebo bar does not "
                   f"(A-prompt-prune cost {d_prompt:+.4f} < +"
                   f"{R1_PROMPT_BAR}): pruning self-old is free, yet pruning "
                   f"the 63 prompt entries is ALSO cheap — the value-level "
                   f"source asymmetry is weaker than T045's eval-time split "
                   f"suggested. Reported honestly, no forcing.")
    elif r1b:
        clause = "PARTIAL-PLACEBO-ONLY"
        verdict = (f"PARTIAL (inverted): the placebo bar fires (A-prompt-prune "
                   f"cost {d_prompt:+.4f} >= +{R1_PROMPT_BAR}) but the "
                   f"self-prune bar does not (cost {d_self:+.4f} > +"
                   f"{R1_SELF_TOL}, still under the kill bar +{KILL_BAR}) — "
                   f"pruning ANY old band costs something; the self band is "
                   f"not value-free to remove.")
    else:
        clause = "NO-BAR-FIRES"
        verdict = (f"NO BAR FIRES: A-self-prune cost {d_self:+.4f} (above "
                   f"+{R1_SELF_TOL}, at/below the kill bar +{KILL_BAR}) and "
                   f"A-prompt-prune cost {d_prompt:+.4f} (< +"
                   f"{R1_PROMPT_BAR}) — neither side of the source asymmetry "
                   f"reaches its frozen bar at generation time. Reported "
                   f"honestly, no forcing.")
    r3_pred_ok = bool(all(abs(spike_deltas[a]["peak_ratio"] - 1.0) <= 0.20
                          for a in ARMS))
    r3_note = (f"registered prediction 'unchanged live spike': peak age-"
               f"{spike_deltas['self']['peak_none_age']} value ratios vs "
               f"A-none — "
               + ", ".join(f"{a} {spike_deltas[a]['peak_ratio']:.3f}"
                           for a in ARMS[1:])
               + (f" — all within ±20% (descriptive bound; no frozen R3 bar): "
                  f"UNCHANGED" if r3_pred_ok else
                  f" — larger than the ±20% descriptive bound: the spike "
                  f"MOVED; report honestly"))
    log(f"REGISTERED DECISION [{clause}]: {verdict}")
    log(f"  clauses: { {k: v['fires'] for k, v in clauses.items()} }")
    log(f"  R3: {r3_note}")
    log(f"  R2 A-self-prune: entropy {ent_d:+.2f}%, top-40 {tk_d:+.2f}% -> "
        f"{'PASS' if r2_ok else 'FAIL'} (bar ±{R2_DRIFT_MAX:.0%})")

    # ---------------------------------------------------------------- metrics
    def arm_block(arm):
        return dict(
            role={"none": "control: normal free run 64->512",
                  "self": "V-zero own generated entries age>96 every K=32 "
                          "from g=100",
                  "prompt": "placebo: V-zero the 63 corpus-prompt entries at "
                            "matched ages/times",
                  "both": "report-only: both bands (total-cache-size control)"
                  }[arm],
            prune=dict(mode=arm, n_events=len(A[arm]["prune_log"]),
                       events=EVENTS if arm != "none" else [],
                       n_pruned_positions=len(A[arm]["pruned"]),
                       pruned_range=ranges[arm], log=A[arm]["prune_log"]),
            r1=dict(readout="mean on-line clean full-softmax CE of emitted "
                            "tokens over the final 64 generation steps "
                            "(positions 448..511), per-seq then over seqs",
                    tail_window=f"final {TAIL} generated tokens",
                    per_seq=r1[arm]["per_seq"], mean=r1[arm]["mean"],
                    ci=r1[arm]["ci"], delta_vs_none=diffs[arm]["mean"],
                    delta_vs_none_ci=diffs[arm]["ci"],
                    delta_vs_none_per_seq=diffs[arm]["per_seq"]),
            r2=dict(readout="tail means of per-step distribution stats; "
                            "drift vs A-none in %",
                    **r2[arm]),
            trajectory_mean_ce_per_step=A[arm]["ce"].mean(0).tolist(),
            clean_judge_tail_ce=dict(
                readout="secondary: the CLEAN unpruned net's CE on this "
                        "arm's final-64 tokens (text quality judge)",
                per_seq=cj[arm]["per_seq"], mean=cj[arm]["mean"],
                delta_vs_none=float(np.mean(cj[arm]["per_seq"])
                                    - np.mean(cj["none"]["per_seq"])),
                fwd_consistency_max_dev=cj[arm]["allpos_vs_lastonly_max_dev"]),
            r3_sweep=dict(readout="final-step full 511-position V-zero sweep "
                                  "on this arm's final sequence (clean "
                                  "recompute; e069/e073 instrument)",
                          **r3[arm]),
            cache_identity=A[arm]["cache"],
        )

    metrics = dict(
        experiment="e075_source_prune",
        purpose="T045's registered intervention test (design scratch/"
                "e075_design.md): e073+e074 showed negative-utility cache "
                "entries concentrate in the model's OWN generated tokens "
                "while corpus-prompt entries ~never hurt. Arms (fixed-anchor "
                "64->512 free run, B=8, e053c net): A-none control; "
                "A-self-prune (V-zero own generated entries age>96 every "
                "K=32 steps from g~100); A-prompt-prune placebo (V-zero the "
                "63 corpus-prompt entries at matched ages/times); A-both "
                "(report-only). FROZEN bars: R1 tail clean CE self <= "
                "none+0.01 AND prompt >= none+0.05 (both firing = "
                "source-specific pruning value); R2 fluency entropy/top-k "
                "drift <= 5%; R3 per-age utility curve after pruning "
                "(predict unchanged spike); kill if self costs > +0.05.",
        started=started, wall_s=elapsed(), threads=THREADS, cpu_only=True,
        net=dict(ckpt=str(CKPT), arch=dict(n_layer=4, n_head=4, n_embd=128,
                                           block_size=T_TOTAL, vocab=65),
                 params=n_params, steps_in_ckpt=steps_in_ckpt,
                 val_ce=val_ce, val_ce_e053c=E053C_VAL_CE),
        seeds=dict(
            corpus=1337,
            prompts=SEED_PROMPT,
            sampling=SEED_SAMPLE,
            battery="the seed-202 8-draw, ALL 8 prompts (e069's battery A = "
                    "the first 4 of this same draw; the B=8 extension is the "
                    "deterministic remainder of the draw — no new seed). "
                    "Sampling: per-arm seed-7 re-init, shared CPU generator, "
                    "row order per step (e053b stream math) — arms share the "
                    "same sampling stream and are token-identical until the "
                    "first prune affects a draw (g=101). NOTE: with B=8 the "
                    "row-order stream interleaves 8 rows, so rows 0-3 do NOT "
                    "replicate battery A's B=4 token stream; the separate "
                    "B=4 battery-A replica arm (G3) carries protocol "
                    "identity.",
            bootstrap=0),
        protocol=dict(
            B=B, prompt_tokens=PROMPT_TOK, t_total=T_TOTAL, temp=TEMP,
            topk=TOPK, fixed_anchor=True, threshold=THRESH,
            a_star_window=A_STAR_K, boot_n=BOOT_N, chunk=CHUNK_512,
            arms=dict(
                none="normal free run",
                self=f"V-zero generated entries p in [64, front-97] at each "
                     f"event (age = front - p > {AGE_CUT})",
                prompt=f"V-zero prompt entries p in [1, 63] at each event "
                       f"(all have age > {AGE_CUT} at the first event)",
                both="union of the self and prompt bands"),
            schedule=dict(
                events_g=EVENTS, K=PRUNE_K, start_g=PRUNE_START_G,
                age_cut=AGE_CUT, age_rule="age of entry p at event with "
                                          "front t = t - p > 96 <=> p <= t-97",
                timing="prune applied at the TOP of the event iteration, "
                       "before the decode feeding the next sample; first "
                       "affected sample g=101 (position 165); all arms "
                       "token-identical through position 164",
                final_pruned=dict(self="positions 64..387 (324)",
                                  prompt="positions 1..63 (63)",
                                  both="positions 1..387 except 0 (387)")),
            bands=dict(prompt=dict(positions=list(PROMPT_POS), n_per_seq=63,
                                   note="position 0 (sequence start, age 511) "
                                        "excluded as in e073/e074"),
                       generated=dict(positions=[64, 510],
                                      ages="1..447, restricted to >= the "
                                           "sweep's own a* for the junk "
                                           "census")),
            readouts=dict(
                r1="on-line clean full-softmax CE of emitted tokens (e053b "
                   "sample_and_ce quantity), mean over final 64 generation "
                   "steps; bars: self <= none + 0.01 AND prompt >= none + "
                   "0.05; kill: self > +0.05",
                r2="per-step full-softmax entropy + top-40 (sampling k) "
                   "mass (+ top-1 mass and emitted-token rank recorded), "
                   "tail means; drift bar ±5% for A-self-prune",
                r3="final-step 511-position V-zero sweep per arm (clean "
                   "recompute on the post-pruning stream); registered "
                   "prediction: unchanged ages-1-5 spike")),
        gates=gates,
        replica=dict(role="battery A B=4 protocol-identity replica (G3)",
                     a_star=d_rep["a_star"], per_seq_a_star=list(
                         d_rep["per_seq_a_star"]),
                     ages1_5=[float(a) for a in ages15],
                     clean_ce=float(ce_rep.mean())),
        arms={arm: arm_block(arm) for arm in ARMS},
        registered_decision=dict(
            frozen_rules=dict(
                r1_self=f"A-self-prune tail clean CE <= A-none + "
                        f"{R1_SELF_TOL}",
                r1_prompt=f"A-prompt-prune tail clean CE >= A-none + "
                          f"{R1_PROMPT_BAR}",
                both_firing="source-specific pruning value",
                r2=f"entropy and top-40-mass drift <= "
                   f"{R2_DRIFT_MAX * 100:.0f}% in A-self-prune",
                kill=f"A-self-prune costs > +{KILL_BAR} nats"),
            r1_tail_clean_ce={arm: r1[arm]["mean"] for arm in ARMS},
            r1_tail_clean_ce_ci={arm: r1[arm]["ci"] for arm in ARMS},
            r1_delta_vs_none={arm: diffs[arm]["mean"] for arm in ARMS},
            r1_delta_vs_none_ci={arm: diffs[arm]["ci"] for arm in ARMS},
            r2_fluency_self=dict(entropy_drift_pct=ent_d,
                                 top40_drift_pct=tk_d,
                                 top1_drift_pct=r2["self"]["top1_mass_drift_pct"],
                                 mean_rank=r2["self"]["mean_rank_tail"],
                                 none_mean_rank=r2["none"]["mean_rank_tail"],
                                 fires=r2_ok),
            r3_spike=dict(prediction="unchanged live spike",
                          ages1_5={arm: r3[arm]["ages1_5"] for arm in ARMS},
                          spike_deltas=spike_deltas,
                          a_star={arm: r3[arm]["a_star"] for arm in ARMS},
                          band_means={arm: r3[arm]["band_means"]
                                      for arm in ARMS},
                          descriptive_ok=r3_pred_ok, note=r3_note),
            clauses=clauses, clause=clause, verdict=verdict),
    )
    save_json(out_dir / "metrics.json", metrics)
    log("metrics.json written")

    plot(out_dir / "source_prune.png", metrics)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot


def plot(path: Path, M: dict):
    dec = M["registered_decision"]
    arms_m = M["arms"]
    r1 = dec["r1_tail_clean_ce"]
    r1ci = dec["r1_tail_clean_ce_ci"]
    dd = dec["r1_delta_vs_none"]
    r2s = dec["r2_fluency_self"]
    ages = np.arange(1, T_TOTAL)
    labels = ["A-none\n(control)", "A-self-prune\n(own gen, age>96)",
              "A-prompt-prune\n(placebo, 63)", "A-both-prune\n(report-only)"]
    cols = ["tab:gray", "tab:green", "tab:orange", "slategray"]
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # ---- panel 1: R1 tail clean CE per arm + frozen bars
    x = np.arange(len(ARMS))
    vals = [r1[a] for a in ARMS]
    cis = [r1ci[a] for a in ARMS]
    cols1 = list(cols)
    if dec["clauses"]["kill"]["fires"]:
        cols1[1] = "tab:red"
    ax1.bar(x, vals, 0.62, color=cols1, alpha=0.85, edgecolor="k", lw=0.5)
    ax1.errorbar(x, vals, yerr=[[v - c[0] for v, c in zip(vals, cis)],
                                [c[1] - v for v, c in zip(vals, cis)]],
                 fmt="none", ecolor="k", lw=1.0, capsize=3)
    base = r1["none"]
    ax1.axhline(base, color="k", lw=0.8, ls="-")
    ax1.axhline(base + R1_SELF_TOL, color="tab:green", ls="--", lw=1.3)
    ax1.text(len(ARMS) - 0.45, base + R1_SELF_TOL + 0.002,
             f"self bar: none + {R1_SELF_TOL}", fontsize=8, color="tab:green",
             ha="right")
    ax1.axhline(base + R1_PROMPT_BAR, color="tab:purple", ls="--", lw=1.3)
    ax1.text(len(ARMS) - 0.45, base + R1_PROMPT_BAR + 0.002,
             f"placebo bar: none + {R1_PROMPT_BAR}", fontsize=8,
             color="tab:purple", ha="right")
    for xi, a in zip(x, ARMS):
        fired = ""
        if a == "self":
            fired = (" [BAR FIRES]" if dec["clauses"]["r1_self_bar"]["fires"]
                     else (" [KILL]" if dec["clauses"]["kill"]["fires"]
                           else " [no]"))
        if a == "prompt":
            fired = (" [BAR FIRES]"
                     if dec["clauses"]["r1_prompt_bar"]["fires"] else " [no]")
        ax1.text(xi, vals[xi] + 0.004, f"{vals[xi]:.4f}\nΔ {dd[a]:+.4f}{fired}",
                 ha="center", fontsize=8.5)
    ax1.set_xticks(x, labels, fontsize=8)
    ax1.set_ylabel("tail clean CE (nats, final 64 generated tokens)")
    lo_y = min(vals + [base]) - 0.02
    hi_y = max(vals + [base + R1_PROMPT_BAR]) + 0.03
    ax1.set_ylim(lo_y, hi_y)
    ax1.set_title("E075-1 — R1: tail clean CE per arm (bootstrap CI over "
                  "B=8 seqs)", fontsize=10)

    # ---- panel 2: per-age utility curves overlay (R3, full range)
    for a, c in zip(ARMS, ["tab:gray", "tab:green", "tab:orange",
                           "slategray"]):
        ax2.plot(ages, np.asarray(arms_m[a]["r3_sweep"]["mean_curve"]), lw=1.1,
                 color=c,
                 label=f"A-{a} (a*={arms_m[a]['r3_sweep']['a_star']})")
        ax2.fill_between(ages, np.asarray(arms_m[a]["r3_sweep"]["ci_lo"]),
                         np.asarray(arms_m[a]["r3_sweep"]["ci_hi"]),
                         alpha=0.06, color=c)
    ax2.axhline(0, color="k", lw=0.6)
    ax2.axhline(THRESH, color="gray", ls=":", lw=0.9)
    ax2.axhline(-THRESH, color="gray", ls=":", lw=0.9)
    ax2.axvspan(448, 511, color="tab:purple", alpha=0.07)
    ax2.text(451, 0.30, "prompt band (63)", fontsize=8, color="tab:purple",
             rotation=90, va="top")
    ax2.axvspan(124, 447, color="tab:green", alpha=0.05)
    ax2.text(285, 0.185, "self-pruned band\n(positions 64..387)", fontsize=8,
             color="tab:green", ha="center")
    ax2.axvline(96, color="tab:green", ls="--", lw=1.0)
    ax2.text(96, 0.05, " age cut 96", fontsize=8, color="tab:green")
    ax2.set_xlim(0, T_TOTAL)
    ax2.set_ylim(-0.06, 0.34)
    ax2.set_xlabel("cache age (tokens)")
    ax2.set_ylabel("V-zero dCE (nats, mean over B=8)")
    ax2.legend(fontsize=8)
    ax2.set_title("E075-2 — R3: per-age utility curves AFTER pruning "
                  "(final-step sweep, arms overlaid; young spike off-scale)",
                  fontsize=10)

    # ---- panel 3: R3 zoom ages 1-60 (the live spike; predict unchanged)
    zoom = 60
    for a, c in zip(ARMS, ["tab:gray", "tab:green", "tab:orange",
                           "slategray"]):
        ax3.plot(ages[:zoom],
                 np.asarray(arms_m[a]["r3_sweep"]["mean_curve"])[:zoom],
                 lw=1.3, color=c, label=f"A-{a}")
    ax3.axhline(THRESH, color="gray", ls=":", lw=1.0)
    ax3.text(zoom, THRESH, f" thresh {THRESH}", fontsize=8, va="bottom",
             ha="right", color="gray")
    ax3.axhline(0, color="k", lw=0.6)
    sd = dec["r3_spike"]["spike_deltas"]
    ax3.set_title("E075-3 — R3 zoom: the live spike under pruning "
                  f"(predict: unchanged)\nnone ages1-5 "
                  f"{['%.2f' % a for a in arms_m['none']['r3_sweep']['ages1_5']]} "
                  f"| self "
                  f"{['%.2f' % a for a in arms_m['self']['r3_sweep']['ages1_5']]} "
                  f"(peak r {sd['self']['peak_ratio']:.2f}) | prompt "
                  f"{['%.2f' % a for a in arms_m['prompt']['r3_sweep']['ages1_5']]}"
                  f" (peak r {sd['prompt']['peak_ratio']:.2f})", fontsize=9.5)
    ax3.set_xlabel("cache age (tokens)")
    ax3.set_ylabel("V-zero dCE (nats)")
    ax3.legend(fontsize=8)

    # ---- panel 4: R2 fluency drift (%)
    drift_arms = ["self", "prompt", "both"]
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
        ax4.text(xi - 0.19, e + (0.25 if e >= 0 else -0.55), f"{e:+.2f}%",
                 ha="center", fontsize=8)
        ax4.text(xi + 0.19, t + (0.25 if t >= 0 else -0.55), f"{t:+.2f}%",
                 ha="center", fontsize=8)
    ax4.axhline(0, color="k", lw=0.7)
    ax4.set_xticks(x4, [f"A-{a}" for a in drift_arms], fontsize=9)
    ax4.set_ylabel("drift vs A-none (%)")
    lo4 = min(ent + tk + [-6]); hi4 = max(ent + tk + [6])
    ax4.set_ylim(lo4 - 1.2, hi4 + 1.2)
    ax4.legend(fontsize=8)
    ax4.set_title("E075-4 — R2: fluency drift (tail entropy / top-40 mass); "
                  f"A-self-prune bar "
                  f"{'PASSES' if r2s['fires'] else 'FAILS'} "
                  f"(ent {r2s['entropy_drift_pct']:+.2f}%, "
                  f"top40 {r2s['top40_drift_pct']:+.2f}%)", fontsize=10)

    # ---- panel 5: generation CE trajectory
    for a, c in zip(ARMS, ["tab:gray", "tab:green", "tab:orange",
                           "slategray"]):
        tr = np.asarray(arms_m[a]["trajectory_mean_ce_per_step"])
        w = 16
        sm = np.convolve(tr, np.ones(w) / w, mode="valid")
        ax5.plot(np.arange(w - 1, G), sm, lw=1.2, color=c,
                 label=f"A-{a} (tail {r1[a]:.4f})")
    ax5.axvline(PRUNE_START_G, color="tab:red", ls="--", lw=1.2)
    ax5.text(PRUNE_START_G + 3, ax5.get_ylim()[0] + 0.05,
             f"first prune g={PRUNE_START_G}", fontsize=8, color="tab:red")
    ax5.axvline(G - TAIL, color="k", ls=":", lw=1.0)
    ax5.text(G - TAIL - 2, ax5.get_ylim()[0] + 0.05, " tail window",
             fontsize=8, ha="right")
    ax5.set_xlabel("generation step g (0..447)")
    ax5.set_ylabel("mean emitted-token CE (nats, 16-step smooth)")
    ax5.legend(fontsize=8)
    ax5.set_title("E075-5 — free-run CE trajectory per arm (B=8 mean); arms "
                  "identical until g=101", fontsize=10)

    # ---- panel 6: verdict text
    ax6.axis("off")
    lines = [
        "REGISTERED (frozen in scratch/e075_design.md):",
        f"  R1 self:  A-self-prune  <= A-none + {R1_SELF_TOL}   "
        f"[{'FIRES' if dec['clauses']['r1_self_bar']['fires'] else 'no'}] "
        f"Δ {dd['self']:+.4f}",
        f"  R1 plac:  A-prompt-prune >= A-none + {R1_PROMPT_BAR}   "
        f"[{'FIRES' if dec['clauses']['r1_prompt_bar']['fires'] else 'no'}] "
        f"Δ {dd['prompt']:+.4f}",
        f"  kill:     A-self-prune costs > +{KILL_BAR}   "
        f"[{'FIRES' if dec['clauses']['kill']['fires'] else 'no'}]",
        f"  both-prune (report-only): Δ {dd['both']:+.4f}",
        "",
        f"R1 tail clean CE: " + " | ".join(f"{a} {r1[a]:.4f}" for a in ARMS),
        f"R2 A-self-prune: entropy {r2s['entropy_drift_pct']:+.2f}%, "
        f"top-40 {r2s['top40_drift_pct']:+.2f}% -> "
        f"{'PASS' if r2s['fires'] else 'FAIL'} (bar ±5%)",
        f"R3 spike: peak ratios vs none — " + ", ".join(
            f"{a} {sd[a]['peak_ratio']:.3f}" for a in ARMS[1:]),
        f"  a* per arm: " + ", ".join(
            f"{a} {arms_m[a]['r3_sweep']['a_star']}" for a in ARMS),
        f"  gen-old junk after run: " + ", ".join(
            f"{a} {arms_m[a]['r3_sweep']['junk_generated_band']['junk_frac_mean']:.3f}"
            for a in ARMS),
        "",
        f"clean-judge tail CE (clean net scores the arm's tail): " + ", ".join(
            f"{a} {arms_m[a]['clean_judge_tail_ce']['mean']:.4f}" for a in ARMS),
        "",
        f"VERDICT [{dec['clause']}]:",
    ] + [f"  {w}" for w in _wrap(dec["verdict"], 92)]
    ax6.text(0.02, 0.97, "E075 — T045 intervention: source-aware pruning",
             fontsize=13, weight="bold", va="top")
    for i, t in enumerate(lines):
        ax6.text(0.02, 0.90 - i * 0.048, t, fontsize=9.0, va="top",
                 family="monospace")

    fig.suptitle("E075 — source-aware pruning (T045 intervention) | clause: "
                 f"{dec['clause']} | Δself {dd['self']:+.4f} | Δprompt "
                 f"{dd['prompt']:+.4f} | R2 "
                 f"{'PASS' if r2s['fires'] else 'FAIL'} | spike "
                 f"{'unchanged' if dec['r3_spike']['descriptive_ok'] else 'MOVED'}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _wrap(text: str, width: int):
    import textwrap
    return textwrap.wrap(text, width=width)


if __name__ == "__main__":
    main()
