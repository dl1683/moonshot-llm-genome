"""E085 — P3 anchor-DESCRIPTION probe (day5 proposal #4; T048's registered pivot).

T048/e080 closed replacement: deletion (+0.262), norm-matched noise (+0.287),
and prompt-copy (+0.069) all fail to restore the run — the generation anchor
is RUN-SPECIFIC trajectory content. P3 pivots from replacement to DESCRIPTION
(registered in scratch/day5_programs.md §4 and QUEUE.md e085): what measurable
property, computed BEFORE any removal, distinguishes anchor entries (the
age>96 self-generated band whose removal cost nats and flipped the run into
the attractor) from non-anchor entries?

DESIGN (registered here BEFORE compute). Battery: the e053c ctx-512 net,
the seed-202 8-draw prompts, B=8, the seed-7 control free run (e075/e080
A-none convention; gated G3 against e080's stored none arm) plus ONE
independent clean continuation battery (same prompts, sampling seed 1009)
for the drift-alignment property. Sampled entries: 100 ANCHOR entries
(positions 64..414 = final-frame ages 97..447, self-generated; per-run
13/13/12/12/12/13/13/12; rng seed 85, stratified uniform over position) and
50 YOUNG controls (positions 415..510 = ages 1..96; per-run 7/6/6/6/7/6/6/6).

PER-ENTRY PROPERTIES (all computed on the control run before any removal):
  P1 pca_frac    fraction of the entry's centered V-norm captured by the
                 run's own cache-PCA top-8 (V concat over layer x head,
                 512-d per entry; PCA fit on all 511 cache positions of the
                 run). Extras: projection norm, distance to run mean.
  P2 readership  mean per-step attention mass received over the re-read
                 window (decode steps strictly after the entry's write step,
                 mean over layers and heads; the e070 mass instrument turned
                 into an accumulation). Extras: total mass, 16-bin trend.
  P3 surprisal   the clean net's in-context full-softmax CE of the entry's
                 token at emission (the control run is unpruned, so the
                 online CE IS the clean-net surprisal). REGISTERED LIAR
                 CONTROL: must not beat readership.
  P4 drift_align cos(V_entry, V at the same position in the independent
                 clean continuation run, seed 1009) — is the entry's value
                 run-specific or generic-for-its-age?
  P5 age         final-frame age 511 - p. The null every property must beat.

OUTCOMES (per-entry jackknife removal cost):
  STATIC       final-step single-read V-zero dCE at the entry's position
               (the e069/e075 fine_sweep instrument) for all 150 entries.
  DYN-TAIL     remove the entry mid-run at its AGE-CROSSING (e075's timing:
               removal before the decode that processes the token at
               position p+97 — entry age 97 at that decode; the unaffected
               token at p+97 is teacher-forced, first affected sample =
               position p+98), free-run to 512 under a matched stream, judge
               the final-64 tail with the CLEAN net (the e075/e080 witness).
               Cost = clean-judge tail CE (pruned) - clean-judge tail CE
               (matched control continuation, same generator seed, row-order
               stream). Young entries never cross age 96 inside the window:
               registered adaptation — removal at the tail start (position
               448 teacher-forced; first affected sample 449, exposure 63),
               DEFINED ONLY for young entries that exist at the tail start
               (positions 415..447); young entries written inside the tail
               window (448..510) have no such time point and are skipped in
               the dynamic arm (static outcome unaffected).
               Exposure (free-run samples after removal) = 414-p for
               anchors = exactly age-97 (reported; redundant given age).
  DYN-+8       clean-judge CE of the first 8 emitted tokens after the first
               affected sample (day5's registered horizon+8 window):
               positions p+98..p+105 (anchors with exposure >= 8) /
               449..456 (youngs). Same matched control.

REGISTERED BARS (frozen; partial r = Pearson r of OLS residuals given the
covariate set; CI = 1000x cluster bootstrap over the 8 runs):
  - BAR-READERSHIP: partial r(P2 | age, P1) >= 0.35 AND CI excludes 0 on
    DYN-TAIL (anchors) => the anchor is attention-read (H-readership).
  - HONESTY CONTROL: P3 surprisal must FAIL: if partial r(P3 | age) >
    partial r(P2 | age) AND P3 clears 0.35 with CI excluding 0 on DYN-TAIL,
    the anchor is "just hard tokens" — reported honestly as such.
  - KILL: |partial r given age| of P1, P2, P4 ALL <= 0.15 on BOTH STATIC
    (all-150 and anchor-only variants) and DYN-TAIL => the anchor is not
    entry-describable; P3's description leg pivots to the sequence level
    (the registered kill).
  - FREE RIDER (T037-a's registered within-net prediction): per-position
    far-context value (mean static dCE over the 8 runs) vs old-entry
    junk-frac (fraction of runs with dCE <= -0.01), positions 64..414:
    predicted POSITIVE; a negative correlation revives the conflict T037-a
    declared pseudo.
Also rides free: the T048 replication at entry granularity — corr(STATIC,
DYN-TAIL) and sign agreement (static is expected to mispredict).

Gates: G1 val CE vs e053c 1.5227 +-0.02; G2 params 873472; G3 protocol
identity vs e080's stored A-none (tail CE per-seq, clean-judge per-seq, sweep
a* + ages1-5; note: 8 threads here vs 12 in e080, hard bar 1e-4 with the
bitwise flag recorded); G4 dynamic-instrument identity (pruned continuation
identical to control through the teacher-forced token; pruned cache column
exactly 0; control column nonzero; matched-stream seeds).

Run:     python lab/e085_anchor_description.py
Outputs: runs/e085/metrics.json + runs/e085/anchor_description.png
Envelope: NO training, NO new automations; CPU-only, 8 threads (T050: 12
spin-thrashes this box), single step, ~15 min. No NOTES/THINKING/QUEUE/STATE
edits; no commit (the dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu" (e053b..e084)
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
SEED_PROMPT, SEED_SAMPLE = 202, 7         # e053b/e053c/e069..e080 seeds
SEED_RUN2 = 1009                          # e085: independent clean battery
SEED_SELECT = 85                          # e085: entry-sampling rng
SEED_CONT_A, SEED_CONT_Y = 851000, 852000  # e085: continuation seeds
B = 8
N_PROMPTS = 8
BOOT_N = 1000
THREADS = 8                               # T050: 12 spin-thrashes the box
torch.set_num_threads(THREADS)

P512 = T_TOTAL - 1                        # 511 sweep positions (native frame)
THRESH = 0.01                             # dead-weight / junk threshold
A_STAR_K = 5                              # sustained window (e053b verbatim)
CHUNK_512 = 64                            # e069/e070/e072/e074/e075 chunk
TAIL = 64                                 # the e075/e080 tail window
G = T_TOTAL - PROMPT_TOK                  # 448 generation steps (g = 0..447)
N_DECODE = G - 1                          # decode steps g = 0..446 (t <= 510)
H8 = 8                                    # day5's registered horizon+8 window
N_PCA = 8                                 # run cache-PCA top-8 (day5 P1)
BIN_W = 28                                # readership trend bins: 16 x 28 steps

# ---- the anchor band (final-frame ages; age of position p = 511 - p) ---------
ANCHOR_POS = (64, 414)                    # ages 97..447 (self-generated)
YOUNG_POS = (415, 510)                    # ages 1..96
ANCHORS_PER_RUN = [13, 13, 12, 12, 12, 13, 13, 12]     # = 100
YOUNG_PER_RUN = [7, 6, 6, 6, 7, 6, 6, 6]              # = 50

# ---- REGISTERED decision numbers (frozen, docstring verbatim) ----------------
BAR_READERSHIP = 0.35                     # partial r(P2 | age, P1) >=
KILL_BAR = 0.15                           # |partial r given age| <= (all P1/P2/P4)

# ---- reference numbers (protocol-identity gates) -----------------------------
CKPT = REPO / "runs" / "checkpoints" / "e053c_ctx512.pt"
E053C_VAL_CE = 1.5226792494455974         # the net being interrogated
E080_METRICS = REPO / "runs" / "e080" / "metrics.json"
T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ------------------------------------------------- machinery (VERBATIM e075/e080)

@torch.no_grad()
def _manual_chunk(net: TinyGPT, idxs, vzero, chunk=CHUNK_512):
    """Exact manual forward, returns LAST-position logits (N, vocab)."""
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
        probs = torch.softmax(att, dim=-1)
        if vzero is not None and bool(vzero.any()):
            v = v.masked_fill(vzero[:, None, :, None], 0.0)
        y = (probs @ v).transpose(1, 2).reshape(N, T, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
    return net.lm_head(net.ln_f(x[:, -1, :]))


@torch.no_grad()
def manual_logits(net: TinyGPT, idxs, vzero=None, chunk=CHUNK_512):
    outs = []
    for i in range(0, idxs.shape[0], chunk):
        outs.append(_manual_chunk(net, idxs[i:i + chunk],
                                  vzero[i:i + chunk] if vzero is not None else None))
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


def onset_age(dce_by_age: np.ndarray, k: int = A_STAR_K, thresh: float = THRESH):
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
    naive, robust = onset_age(mean_age)
    a = robust if robust is not None else naive
    return a, naive, robust


@torch.no_grad()
def prefill_batch(net: TinyGPT, idx: torch.Tensor):
    """Batched prefill, (B, Tp) -> last-position logits (B, V) + KV cache.
    [VERBATIM e075/e080]"""
    Bb, T = idx.shape
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
        q = q.view(Bb, T, H, d).transpose(1, 2)
        k = k.view(Bb, T, H, d).transpose(1, 2)
        v = v.view(Bb, T, H, d).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
        att = att.masked_fill(causal, float("-inf"))
        y = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(Bb, T, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
        kv.append((k, v))
    return net.lm_head(net.ln_f(x[:, -1, :])), kv


@torch.no_grad()
def decode_step_batch(net: TinyGPT, toks: torch.Tensor, pos: int, kv: list,
                      collect_att: bool = False):
    """Batched incremental decode: (B,) tokens at position pos -> (B, V);
    with collect_att=True also returns the mean-over-heads attention mass
    over the cache (B, t+1), summed over layers by the caller (the e070
    instrument accumulated). The logits math is untouched by collection.
    [e075 verbatim + optional mass readout]"""
    Bb = toks.shape[0]
    x = net.wte(toks) + net.wpe(torch.full((Bb,), pos))
    H = net.cfg.n_head
    m = None
    for li, blk in enumerate(net.h):
        xh = blk.ln1(x)
        qkv = blk.attn.c_attn(xh)
        C = qkv.shape[-1] // 3
        d = C // H
        q, k, v = qkv.split(C, dim=1)
        q = q.view(Bb, 1, H, d).transpose(1, 2)
        k = k.view(Bb, 1, H, d).transpose(1, 2)
        v = v.view(Bb, 1, H, d).transpose(1, 2)
        kp, vp = kv[li]
        k = torch.cat([kp, k], dim=2)
        v = torch.cat([vp, v], dim=2)
        kv[li] = (k, v)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))   # (B,H,1,t+1)
        probs = torch.softmax(att, -1)
        if collect_att:
            pm = probs[:, :, 0, :].mean(1)                      # (B, t+1)
            m = pm if m is None else m + pm
        y = (probs @ v).transpose(1, 2).reshape(Bb, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
    logits = net.lm_head(net.ln_f(x))
    return (logits, m) if collect_att else logits


# ---------------------------------------- control free run + readership collection

@torch.no_grad()
def generate_collect(net: TinyGPT, prompts, gen: torch.Generator):
    """Free-run 64->512 for B sequences (e075/e080 A-none VERBATIM stream
    math) while accumulating per-position attention-received mass (mean over
    heads, summed over layers at the end; e070's instrument accumulated):
    mass_all over ALL decode steps, mass_self the write-step diagonal,
    mass_bins 16 fixed step-bins for the re-read trend."""
    Bb = len(prompts)
    idx = torch.stack(prompts)
    logits, kv = prefill_batch(net, idx)
    ce_s = np.zeros((Bb, G), float)
    n_bins = (G + BIN_W - 1) // BIN_W
    mass_all = np.zeros((Bb, T_TOTAL), float)
    mass_self = np.zeros((Bb, T_TOTAL), float)
    mass_bins = np.zeros((n_bins, Bb, T_TOTAL), float)
    for g in range(G):
        t = PROMPT_TOK + g
        toks = torch.zeros(Bb, dtype=torch.long)
        for j in range(Bb):
            tok, ce = sample_and_ce(logits[j], gen)
            toks[j] = tok
            ce_s[j, g] = ce
        idx = torch.cat([idx, toks[:, None]], 1)
        if t < T_TOTAL - 1:
            logits, m = decode_step_batch(net, toks, t, kv, collect_att=True)
            mt = m.numpy()                                        # (B, t+1)
            mass_all[:, :t + 1] += mt
            mass_bins[g // BIN_W, :, :t + 1] += mt
            mass_self[:, t] = mt[:, t]      # query pos of this decode is t
    return dict(idx=idx, kv=kv, ce=ce_s, mass_all=mass_all,
                mass_self=mass_self, mass_bins=mass_bins)


# ------------------------------------------------------- the static sweep (e069)

@torch.no_grad()
def fine_sweep(net: TinyGPT, ctx: torch.Tensor, tgt: torch.Tensor,
               clean_logits: torch.Tensor):
    """Final-step fine V-zero sweep for all B sequences (e053c/e069/e075
    instrument verbatim). Returns dce (B, 511) by POSITION and clean CE."""
    Bs, Pw = ctx.shape
    p_clean = torch.softmax(clean_logits.float(), -1)
    ce_clean = -torch.log(p_clean[torch.arange(Bs), tgt].clamp_min(1e-12))
    idxs = ctx.repeat_interleave(Pw, dim=0)
    vz = torch.eye(Pw, dtype=torch.bool).repeat(Bs, 1)
    les = manual_logits(net, idxs, vz)
    lp = torch.log_softmax(les.float(), -1)
    rows = torch.arange(Bs * Pw)
    dce = (-ce_clean.repeat_interleave(Pw) - lp[rows, tgt.repeat_interleave(Pw)])
    return dce.view(Bs, Pw).numpy(), ce_clean.numpy()


# ---------------------------------------------------- the dynamic removal arm

@torch.no_grad()
def run_continuation(net: TinyGPT, prefix: torch.Tensor, forced_tok: torch.Tensor,
                     forced_pos: int, seed: int, remove=None):
    """Single-entry-removal continuation (and its matched control when
    remove=None). prefix = control tokens through position forced_pos-1;
    the token at forced_pos is teacher-forced (it precedes the first
    affected sample by construction); removal (list of (row, col)) is
    applied to the KV cache AFTER prefill and BEFORE the decode of the
    forced token — exactly e075's 'top of the event iteration' timing —
    so the first AFFECTED sample is position forced_pos+1. Free-run from
    there with the row-order shared generator (e053b stream math).
    Returns the full 512-token idx and the final kv."""
    R = prefix.shape[0]
    gen = torch.Generator().manual_seed(seed)
    logits, kv = prefill_batch(net, prefix)
    if remove is not None:
        for (r, c) in remove:
            for (_k, v) in kv:
                v[r, :, c, :] = 0.0
    idx = torch.cat([prefix, forced_tok[:, None]], 1)
    logits = decode_step_batch(net, forced_tok, forced_pos, kv)
    n_free = T_TOTAL - 1 - forced_pos
    for s in range(n_free):
        pos = forced_pos + 1 + s
        new = torch.zeros(R, dtype=torch.long)
        for j in range(R):
            tok, _ = sample_and_ce(logits[j], gen)
            new[j] = tok
        idx = torch.cat([idx, new[:, None]], 1)
        if pos < T_TOTAL - 1:
            logits = decode_step_batch(net, new, pos, kv)
    return idx, kv


def judge_windows(all_lg: torch.Tensor, idx: torch.Tensor, windows):
    """Clean-net CE of target windows [(lo, hi), ...] (queries lo-1..hi-1).
    Returns dict window -> (R,) mean CE per row."""
    out = {}
    for (lo, hi) in windows:
        lg = all_lg[:, lo - 1:hi, :]
        tgt = idx[:, lo:hi + 1]
        lp = torch.log_softmax(lg.float(), -1)
        out[(lo, hi)] = -lp.gather(2, tgt[:, :, None]).squeeze(2).mean(1).numpy()
    return out


# ------------------------------------------------------------- statistics

def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    def ranks(v):
        v = np.asarray(v, float)
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), float)
        r[order] = np.arange(1, len(v) + 1, dtype=float)
        return r
    return pearson(ranks(x), ranks(y))


def _resid(x, Z):
    """OLS-residualize x on covariate columns Z (intercept added here)."""
    n = len(x)
    Zc = np.column_stack([np.ones(n)] + [np.asarray(z, float) for z in Z])
    beta, *_ = np.linalg.lstsq(Zc, np.asarray(x, float), rcond=None)
    return np.asarray(x, float) - Zc @ beta


def partial_r(x, y, Z):
    """Pearson r of x,y after residualizing BOTH on covariates Z."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    Z = [np.asarray(z, float) for z in Z]
    ok = np.isfinite(x) & np.isfinite(y)
    for z in Z:
        ok &= np.isfinite(z)
    if ok.sum() < len(Z) + 3:
        return float("nan")
    rx = _resid(x[ok], [z[ok] for z in Z])
    ry = _resid(y[ok], [z[ok] for z in Z])
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def cluster_boot_ci(entries, fn, n: int = BOOT_N, seed: int = 0):
    """Cluster bootstrap over the 8 runs: resample run ids with replacement,
    rebuild the (multiply-counted) entry index list, recompute fn."""
    runs = sorted({e["run"] for e in entries})
    by_run = {r: [i for i, e in enumerate(entries) if e["run"] == r] for r in runs}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sel = []
        for r in rng.integers(0, len(runs), len(runs)):
            sel.extend(by_run[runs[r]])
        v = fn([entries[i] for i in sel])
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ---------------------------------------------------------------------- main


def main():
    assert torch.cuda.device_count() == 0, "CPU-only violated: CUDA visible"
    assert common.DEVICE == "cpu", f"common.DEVICE={common.DEVICE} != cpu"
    out_dir = run_dir("e085")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=False)

    # ---- battery: rebuild corpus + net EXACTLY as e053c/e069..e080 did
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
    gates["G2_params"] = dict(params=n_params, expected=873472,
                              ok=bool(n_params == 873472))
    val_ce = estimate_loss(net, corp, "val", n_batches=12)
    gates["G1_val_ce"] = dict(val_ce=val_ce, ref=E053C_VAL_CE, tol=0.02,
                              ok=bool(abs(val_ce - E053C_VAL_CE) <= 0.02))
    log(f"e053c net loaded ({n_params:,} params) | val CE {val_ce:.4f} vs "
        f"{E053C_VAL_CE:.4f} -> G1 {'PASS' if gates['G1_val_ce']['ok'] else 'FAIL'}")

    # ---- battery: the seed-202 8-draw, ALL 8
    gen_p = torch.Generator().manual_seed(SEED_PROMPT)
    ix = torch.randint(len(corp.val) - PROMPT_TOK - 1, (N_PROMPTS,),
                       generator=gen_p)
    prompts8 = [corp.val[i:i + PROMPT_TOK] for i in ix]
    log(f"battery: {N_PROMPTS} prompts (seed {SEED_PROMPT}); prompt0 prefix: "
        f"{corp.decode(prompts8[0])[:32]!r}")

    # ================================================== the two clean batteries
    log("control battery: seed-7 free run (e075/e080 A-none convention) "
        "+ attention-mass collection")
    gen7 = torch.Generator().manual_seed(SEED_SAMPLE)
    run1 = generate_collect(net, prompts8, gen7)
    idx1 = run1["idx"]
    log(f"  control run done ({T_TOTAL} positions x {B} rows); now the "
        f"independent clean battery (sampling seed {SEED_RUN2})")
    gen2 = torch.Generator().manual_seed(SEED_RUN2)
    run2 = generate_collect(net, prompts8, gen2)
    idx2 = run2["idx"]

    # ---- G3 protocol identity vs e080's stored A-none arm
    r1_tail = run1["ce"][:, -TAIL:].mean(1)
    key_t = (T_TOTAL - TAIL, T_TOTAL - 1)
    cj1 = judge_windows(manual_all_logits(net, idx1), idx1, [key_t])[key_t]
    g3 = dict(ref_file=str(E080_METRICS), ok=False)
    e080 = None
    if E080_METRICS.exists():
        with open(E080_METRICS) as f:
            e080 = json.load(f)
        ref_r1 = np.asarray(e080["arms"]["none"]["r1"]["per_seq"])
        ref_cj = np.asarray(e080["arms"]["none"]["clean_judge_tail_ce"]["per_seq"])
        dev_r1 = float(np.abs(r1_tail - ref_r1).max())
        dev_cj = float(np.abs(cj1 - ref_cj).max())
        bit = bool(np.array_equal(r1_tail, ref_r1))
        g3.update(r1_per_seq_max_dev=dev_r1, r1_bitwise=bit,
                  clean_judge_max_dev=dev_cj,
                  ok=bool(dev_r1 < 1e-4 and dev_cj < 1e-4),
                  note=f"threads {THREADS} here vs 12 in e080: bit-identity "
                       f"possible but not guaranteed; hard bar 1e-4 nats, "
                       f"bitwise={bit}")
        log(f"G3 vs e080 A-none: tail-CE dev {dev_r1:.2e} (bitwise {bit}), "
            f"clean-judge dev {dev_cj:.2e} -> "
            f"{'PASS' if g3['ok'] else 'FAIL'}")
    else:
        g3["note"] = "runs/e080/metrics.json not found; gate skipped"
        log("G3: e080 metrics missing — skipped")

    # ================================================== the static sweep (all B)
    log("static arm: final-step 511-position V-zero sweep on the control run")
    tgt = idx1[:, -1]
    with torch.no_grad():
        clean_last = manual_logits(net, idx1[:, :-1], None)
    dce, ce_clean = fine_sweep(net, idx1[:, :-1], tgt, clean_last)
    nan_n = int(np.isnan(dce).sum())
    if nan_n:
        dce = np.nan_to_num(dce, nan=0.0)
    dce_age = dce[:, ::-1]
    a_star, _naive, _rob = a_star_rule(dce_age.mean(0))
    ages15 = dce_age.mean(0)[:5]
    mean_curve = dce_age.mean(0)
    if e080 is not None:
        ref15 = np.asarray(e080["arms"]["none"]["r3_sweep"]["ages1_5"])
        ref_a = e080["arms"]["none"]["r3_sweep"]["a_star"]
        g3.update(a_star=a_star, a_star_ref=ref_a,
                  a_star_match=bool(a_star == ref_a),
                  ages1_5=[float(a) for a in ages15],
                  ages1_5_max_dev=float(np.max(np.abs(ages15 - ref15))))
        g3["ok"] = bool(g3["ok"] and a_star == ref_a
                        and np.max(np.abs(ages15 - ref15)) < 0.05)
        log(f"  sweep a* {a_star} (e080 ref {ref_a}), ages1-5 dev "
            f"{g3['ages1_5_max_dev']:.4f} -> G3 "
            f"{'PASS' if g3['ok'] else 'FAIL'} (nan->0: {nan_n})")
    gates["G3_protocol_identity_vs_e080"] = g3

    # ================================================== the per-entry properties
    log("properties: PCA / readership / surprisal / drift-alignment / age")
    n_layer = len(net.h)

    def vcat(kv):
        # (L, B, H, T, d) -> (B, T, L*H*d)
        vs = [v for (_k, v) in kv]
        return torch.cat([v.permute(0, 2, 1, 3).reshape(v.shape[0], v.shape[2], -1)
                          for v in vs], dim=2)

    V1 = vcat(run1["kv"]).numpy()                     # (B, cache_len, 512)
    V2 = vcat(run2["kv"]).numpy()
    cache_len = V1.shape[1]
    pca_frac = np.zeros((B, T_TOTAL), float)
    pca_pnorm = np.zeros((B, T_TOTAL), float)
    pca_dmean = np.zeros((B, T_TOTAL), float)
    for b in range(B):
        X = V1[b]                                     # (cache_len, 512)
        Xc = X - X.mean(0, keepdims=True)
        _U, _s, Vt = np.linalg.svd(Xc, full_matrices=False)
        P = Vt[:N_PCA].T                              # (512, N_PCA) loadings
        proj = Xc @ P                                 # (cache_len, N_PCA)
        pca_frac[b, :cache_len] = (proj ** 2).sum(1) / np.maximum(
            (Xc ** 2).sum(1), 1e-12)
        pca_pnorm[b, :cache_len] = np.linalg.norm(proj, axis=1)
        pca_dmean[b, :cache_len] = np.linalg.norm(Xc, axis=1)
    # P2: readership (re-read = all decode mass minus the write-step diagonal)
    mass_all, mass_self, mass_bins = (run1["mass_all"], run1["mass_self"],
                                      run1["mass_bins"])
    n_bins = mass_bins.shape[0]
    bin_steps = np.array([max(0.0, min(BIN_W, N_DECODE - k * BIN_W))
                          for k in range(n_bins)], float)
    read_sum = (mass_all - mass_self) / float(n_layer)  # mean over layers
    # re-read decode steps for position p: g in (g_p, 446] -> 510 - p
    steps = np.maximum(P512 - 1 - np.arange(T_TOTAL), 1)
    readership = read_sum / steps
    trend = np.full((B, T_TOTAL), np.nan)
    for b in range(B):
        for p in range(PROMPT_TOK, cache_len):
            g_p = p - PROMPT_TOK
            ks = [k for k in range(n_bins)
                  if k * BIN_W > g_p and bin_steps[k] > 0]
            if len(ks) < 2 or readership[b, p] <= 0:
                continue
            xs = np.array([k * BIN_W + BIN_W / 2 for k in ks])
            ys = np.array([mass_bins[k, b, p] / bin_steps[k] / n_layer
                           for k in ks])
            trend[b, p] = np.polyfit(xs, ys, 1)[0] / readership[b, p]
    # P4: drift-alignment vs the independent clean battery (same position)
    drift = ((V1 * V2).sum(2)
             / np.maximum(np.linalg.norm(V1, axis=2) * np.linalg.norm(V2, axis=2),
                          1e-12))

    # ================================================== entry selection
    rng = np.random.default_rng(SEED_SELECT)
    entries = []
    for b in range(B):
        a_pos = np.sort(rng.choice(np.arange(ANCHOR_POS[0], ANCHOR_POS[1] + 1),
                                   size=ANCHORS_PER_RUN[b], replace=False))
        y_pos = np.sort(rng.choice(np.arange(YOUNG_POS[0], YOUNG_POS[1] + 1),
                                   size=YOUNG_PER_RUN[b], replace=False))
        for p in a_pos:
            entries.append(dict(run=b, pos=int(p), band="anchor"))
        for p in y_pos:
            entries.append(dict(run=b, pos=int(p), band="young"))
    for e in entries:
        p, b = e["pos"], e["run"]
        e["age"] = P512 - p
        e["g_write"] = p - PROMPT_TOK
        e["P1_pca_frac"] = float(pca_frac[b, p])
        e["P1_proj_norm"] = float(pca_pnorm[b, p])
        e["P1_dist_mean"] = float(pca_dmean[b, p])
        e["P2_readership"] = float(readership[b, p])
        e["P2_total_mass"] = float(read_sum[b, p])
        e["P2_trend"] = (float(trend[b, p])
                         if np.isfinite(trend[b, p]) else None)
        e["P3_surprisal"] = float(run1["ce"][b, e["g_write"]])
        e["P4_drift_align"] = float(drift[b, p])
        e["P5_age"] = float(e["age"])
        e["static_dce"] = float(dce[b, p])
        # exposure = free-run samples after removal (anchors: 414-p = age-97)
        e["exposure"] = int(T_TOTAL - 1 - (p + 98) + 1) if e["band"] == "anchor" else 63
    log(f"entries selected: {sum(1 for e in entries if e['band'] == 'anchor')} "
        f"anchors (ages "
        f"{min(e['age'] for e in entries if e['band'] == 'anchor')}-"
        f"{max(e['age'] for e in entries if e['band'] == 'anchor')}) + "
        f"{sum(1 for e in entries if e['band'] == 'young')} young (ages "
        f"{min(e['age'] for e in entries if e['band'] == 'young')}-"
        f"{max(e['age'] for e in entries if e['band'] == 'young')})")

    # ================================================== the dynamic arm
    log("dynamic arm: single-entry removals at age-crossing, matched-stream "
        "continuations, clean-judge tail + horizon+8")
    anchor_groups = {}
    for e in entries:
        if e["band"] == "anchor":
            anchor_groups.setdefault(e["pos"], []).append(e)
    young_groups = {}
    for e in entries:
        if e["band"] == "young":
            young_groups.setdefault(e["run"], []).append(e)

    def do_group(prefix_rows, forced_pos, remove_cols, seed, group_entries,
                 w8_lo=None, w8_hi=None):
        """Two matched batches (pruned / control) for one removal group."""
        prefix = idx1[prefix_rows, :forced_pos]
        forced = idx1[prefix_rows, forced_pos]
        remove = list(enumerate(remove_cols))
        pr_idx, pr_kv = run_continuation(net, prefix, forced, forced_pos,
                                         seed, remove=remove)
        ct_idx, ct_kv = run_continuation(net, prefix, forced, forced_pos,
                                         seed, remove=None)
        wins = [key_t]
        if w8_lo is not None:
            wins.append((w8_lo, w8_hi))
        pr_j = judge_windows(manual_all_logits(net, pr_idx), pr_idx, wins)
        ct_j = judge_windows(manual_all_logits(net, ct_idx), ct_idx, wins)
        # G4-style identity checks for this group
        ident_pre = bool(torch.equal(pr_idx[:, :forced_pos + 1],
                                     ct_idx[:, :forced_pos + 1]))
        col_zero, col_live = True, True
        for r, c in remove:
            for (_k, v) in pr_kv:
                col_zero &= bool(v[r, :, c, :].abs().max().item() == 0.0)
            for (_k, v) in ct_kv:
                col_live &= bool(v[r, :, c, :].abs().max().item() > 0.0)
        for j, e in enumerate(group_entries):
            e["dyn_tail_pruned"] = float(pr_j[key_t][j])
            e["dyn_tail_control"] = float(ct_j[key_t][j])
            e["dyn_tail_cost"] = float(pr_j[key_t][j] - ct_j[key_t][j])
            if w8_lo is not None:
                k8 = (w8_lo, w8_hi)
                e["dyn8_pruned"] = float(pr_j[k8][j])
                e["dyn8_control"] = float(ct_j[k8][j])
                e["dyn8_cost"] = float(pr_j[k8][j] - ct_j[k8][j])
            e["g4_group"] = dict(ident_pre=ident_pre, col_zero=col_zero,
                                 col_live=col_live,
                                 diverged=bool(not torch.equal(pr_idx[j],
                                                               ct_idx[j])))
        return dict(ident_pre=ident_pre, col_zero=col_zero, col_live=col_live)

    g4_all = dict(ident_pre=True, col_zero=True, col_live=True, n_groups=0)
    for gi, (p, es) in enumerate(sorted(anchor_groups.items())):
        rows = [e["run"] for e in es]
        w8_lo = p + 98 if p + 98 + H8 - 1 <= T_TOTAL - 1 else None
        w8_hi = (p + 98 + H8 - 1) if w8_lo is not None else None
        chk = do_group(rows, p + 97, [p] * len(rows), SEED_CONT_A + p, es,
                       w8_lo, w8_hi)
        g4_all["ident_pre"] &= chk["ident_pre"]
        g4_all["col_zero"] &= chk["col_zero"]
        g4_all["col_live"] &= chk["col_live"]
        g4_all["n_groups"] += 1
        if (gi + 1) % 20 == 0:
            log(f"  anchor groups {gi + 1}/{len(anchor_groups)} done")
    for b, es in sorted(young_groups.items()):
        es_dyn = [e for e in es if e["pos"] <= T_TOTAL - TAIL - 1]
        for e in es:
            if e["pos"] > T_TOTAL - TAIL - 1:
                e["dyn_skip_reason"] = ("entry written inside the tail window "
                                        "(pos >= 448): removal-at-tail-start "
                                        "undefined; static-only")
        if not es_dyn:
            continue
        rows = [b] * len(es_dyn)
        cols = [e["pos"] for e in es_dyn]
        chk = do_group(rows, T_TOTAL - TAIL, cols, SEED_CONT_Y + b, es_dyn,
                       T_TOTAL - TAIL + 1, T_TOTAL - TAIL + H8)
        g4_all["ident_pre"] &= chk["ident_pre"]
        g4_all["col_zero"] &= chk["col_zero"]
        g4_all["col_live"] &= chk["col_live"]
        g4_all["n_groups"] += 1
    g4_all["ok"] = bool(g4_all["ident_pre"] and g4_all["col_zero"]
                        and g4_all["col_live"])
    gates["G4_dynamic_instrument"] = g4_all
    log(f"G4 dynamic instrument: {g4_all['n_groups']} groups, prefix identity "
        f"{g4_all['ident_pre']}, pruned cols zero {g4_all['col_zero']}, "
        f"control cols live {g4_all['col_live']} -> "
        f"{'PASS' if g4_all['ok'] else 'FAIL'}")
    n_div = sum(1 for e in entries if e.get("g4_group", {}).get("diverged"))
    log(f"  removal changed the continuation in {n_div}/{len(entries)} "
        f"entries (others: no sampled draw flipped)")

    # ================================================== regressions + bars
    log("regressions: partial r (given age; readership also given age+P1), "
        "cluster bootstrap over the 8 runs")

    props = [("P1_pca_frac", "P1 pca_frac"),
             ("P2_readership", "P2 readership"),
             ("P3_surprisal", "P3 surprisal (LIAR control)"),
             ("P4_drift_align", "P4 drift-align"),
             ("P5_age", "P5 age (null)")]

    def outcome_set(name):
        if name == "static":
            return entries, "static_dce"
        if name == "static_anchor":
            return [e for e in entries if e["band"] == "anchor"], "static_dce"
        if name == "dyn_tail":
            return [e for e in entries if e["band"] == "anchor"], "dyn_tail_cost"
        if name == "dyn_tail_all":
            return [e for e in entries
                    if "dyn_tail_cost" in e], "dyn_tail_cost"
        if name == "dyn8":
            return [e for e in entries if e["band"] == "anchor"
                    and "dyn8_cost" in e], "dyn8_cost"
        raise ValueError(name)

    table = {}
    for oname in ["static", "static_anchor", "dyn_tail", "dyn_tail_all", "dyn8"]:
        es, ykey = outcome_set(oname)
        blk = {}
        for key, label in props:
            r0 = pearson([e[key] for e in es], [e[ykey] for e in es])
            r0ci = cluster_boot_ci(es, lambda gg, k=key, yk=ykey: pearson(
                [e[k] for e in gg], [e[yk] for e in gg]))
            ra = partial_r([e[key] for e in es], [e[ykey] for e in es],
                           [[e["P5_age"] for e in es]])
            raci = cluster_boot_ci(
                es, lambda gg, k=key, yk=ykey: partial_r(
                    [e[k] for e in gg], [e[yk] for e in gg],
                    [[e["P5_age"] for e in gg]]))
            blk[key] = dict(label=label, n=len(es), r0=r0, r0_ci=r0ci,
                            r_given_age=ra, r_given_age_ci=raci)
            if key != "P5_age":
                rap = partial_r([e[key] for e in es], [e[ykey] for e in es],
                                [[e["P5_age"] for e in es],
                                 [e["P1_pca_frac"] for e in es]])
                rapci = cluster_boot_ci(
                    es, lambda gg, k=key, yk=ykey: partial_r(
                        [e[k] for e in gg], [e[yk] for e in gg],
                        [[e["P5_age"] for e in gg],
                         [e["P1_pca_frac"] for e in gg]]))
                blk[key]["r_given_age_pca"] = rap
                blk[key]["r_given_age_pca_ci"] = rapci
        table[oname] = blk
        log(f"  outcome {oname} (n={len(es)}): " + " | ".join(
            f"{k[:2]}: r0 {blk[k]['r0']:+.3f}, "
            f"r|age {blk[k]['r_given_age']:+.3f} "
            f"[{blk[k]['r_given_age_ci'][0]:+.3f},"
            f"{blk[k]['r_given_age_ci'][1]:+.3f}]"
            + (f", r|age+P1 {blk[k]['r_given_age_pca']:+.3f}"
               if "r_given_age_pca" in blk[k] else "")
            for k, _ in props))

    # ---- registered bars (frozen)
    def gt_bar(blk, key, bar, use_pca=False):
        rfield = "r_given_age_pca" if use_pca else "r_given_age"
        cfield = rfield + "_ci"
        r = blk[key].get(rfield, float("nan"))
        ci = blk[key].get(cfield, [float("nan"), float("nan")])
        return bool(np.isfinite(r) and r >= bar and ci[0] > 0), r, ci

    anch = table["dyn_tail"]
    stat = table["static"]
    stat_a = table["static_anchor"]
    rd_fires, rd_r, rd_ci = gt_bar(anch, "P2_readership", BAR_READERSHIP,
                                   use_pca=True)
    rd_fires_static, rd_rs, rd_cis = gt_bar(stat, "P2_readership",
                                            BAR_READERSHIP, use_pca=True)
    rd_fires_sa, rd_rsa, rd_cisa = gt_bar(stat_a, "P2_readership",
                                          BAR_READERSHIP, use_pca=True)
    r_p3 = anch["P3_surprisal"]["r_given_age"]
    r_p3_ci = anch["P3_surprisal"]["r_given_age_ci"]
    r_p2 = anch["P2_readership"]["r_given_age"]
    p3_beats = bool(np.isfinite(r_p3) and r_p3 > r_p2)
    p3_fires_liar = bool(p3_beats and r_p3 >= BAR_READERSHIP and r_p3_ci[0] > 0)

    def kill_on(blk):
        return all(abs(blk[k]["r_given_age"]) <= KILL_BAR
                   for k in ("P1_pca_frac", "P2_readership", "P4_drift_align"))

    kill = bool(kill_on(stat) and kill_on(stat_a) and kill_on(anch))

    clauses = dict(
        readership_bar=dict(
            rule=f"partial r(P2 | age, P1) >= {BAR_READERSHIP} AND CI "
                 f"excludes 0 on DYN-TAIL (anchors)",
            r=rd_r, ci=rd_ci, fires=rd_fires,
            static_all=dict(r=rd_rs, ci=rd_cis, fires=rd_fires_static),
            static_anchor=dict(r=rd_rsa, ci=rd_cisa, fires=rd_fires_sa)),
        surprisal_liar_control=dict(
            rule=f"must FAIL: r(P3 | age) <= r(P2 | age) on DYN-TAIL; it "
                 f"'fires as liar' if r(P3) > r(P2) AND >= {BAR_READERSHIP} "
                 f"AND CI excludes 0 -> the anchor is just hard tokens",
            r_p3=r_p3, r_p3_ci=r_p3_ci, r_p2=r_p2, p3_beats_p2=p3_beats,
            fires_as_liar=p3_fires_liar,
            passed=bool(not p3_fires_liar)),
        kill=dict(
            rule=f"|partial r given age| of P1, P2, P4 ALL <= {KILL_BAR} on "
                 f"BOTH static and dyn-tail -> anchor not entry-describable; "
                 f"P3 pivots to sequence level",
            r_static={k: stat[k]["r_given_age"] for k in
                      ("P1_pca_frac", "P2_readership", "P4_drift_align")},
            r_static_anchor={k: stat_a[k]["r_given_age"] for k in
                             ("P1_pca_frac", "P2_readership", "P4_drift_align")},
            r_dyn={k: anch[k]["r_given_age"] for k in
                   ("P1_pca_frac", "P2_readership", "P4_drift_align")},
            fires=kill),
    )
    if kill:
        clause = "KILL (not entry-describable)"
        verdict = (f"KILL fires: P1/P2/P4 partial r (given age) all <= "
                   f"{KILL_BAR} on BOTH static and dynamic cost — the anchor "
                   f"is not describable by per-entry properties; P3's "
                   f"description leg pivots to the SEQUENCE level (the "
                   f"anchor-as-whole-run-object) or closes with the taxonomy "
                   f"+ provenance law as its legacy. No further "
                   f"pruning-exploit work (e075's kill stands).")
    elif rd_fires:
        clause = "H-READERSHIP (anchor is attention-read)"
        verdict = (f"H-readership fires: readership partial r (given age+PCA) "
                   f"{rd_r:+.3f} CI [{rd_ci[0]:+.3f},{rd_ci[1]:+.3f}] >= "
                   f"{BAR_READERSHIP} on the dynamic witness — the anchor is "
                   f"WHERE THE RUN HAS BEEN: entries the run kept re-reading "
                   f"are the load-bearing ones.")
    elif p3_fires_liar:
        clause = "HARD-TOKENS (honesty control fires)"
        verdict = (f"The LIAR control wins: self-surprisal (r|age {r_p3:+.3f}) "
                   f"predicts the dynamic cost better than readership "
                   f"(r|age {r_p2:+.3f}) and clears the bar — the anchor is "
                   f"just the run's hard tokens, not an attention object. "
                   f"Reported honestly per the registered control.")
    else:
        clause = "PARTIAL / MIXED"
        best = max(("P1_pca_frac", "P2_readership", "P4_drift_align"),
                   key=lambda k: abs(anch[k]["r_given_age"]))
        verdict = (f"No registered bar fires cleanly. Strongest dynamic "
                   f"descriptor: {anch[best]['label']} "
                   f"(r|age {anch[best]['r_given_age']:+.3f} CI "
                   f"[{anch[best]['r_given_age_ci'][0]:+.3f},"
                   f"{anch[best]['r_given_age_ci'][1]:+.3f}]); readership "
                   f"(given age+PCA) {rd_r:+.3f} vs bar {BAR_READERSHIP}. "
                   f"Kill does not fire; texture reported honestly.")
    log(f"REGISTERED DECISION [{clause}]: {verdict}")

    # ---- T048 replication at entry granularity (rides free)
    a_es = [e for e in entries if e["band"] == "anchor"]
    t048 = dict(
        readout="corr(static dCE, dynamic tail cost) on the 100 anchor "
                "entries; static is EXPECTED to mispredict (T048)",
        pearson=pearson([e["static_dce"] for e in a_es],
                        [e["dyn_tail_cost"] for e in a_es]),
        spearman=spearman([e["static_dce"] for e in a_es],
                          [e["dyn_tail_cost"] for e in a_es]),
        sign_agree=float(np.mean([np.sign(e["static_dce"])
                                  == np.sign(e["dyn_tail_cost"]) for e in a_es])),
        mean_dyn_cost=float(np.mean([e["dyn_tail_cost"] for e in a_es])),
        mean_dyn8_cost=float(np.mean([e["dyn8_cost"] for e in a_es
                                      if "dyn8_cost" in e])),
        young_dyn_n=int(sum(1 for e in entries
                            if e["band"] == "young" and "dyn_tail_cost" in e)),
        mean_young_dyn_cost=float(np.mean(
            [e["dyn_tail_cost"] for e in entries
             if e["band"] == "young" and "dyn_tail_cost" in e])))
    log(f"T048 entry-granularity: r(static, dyn-tail) {t048['pearson']:+.3f} "
        f"(spearman {t048['spearman']:+.3f}), sign-agree "
        f"{t048['sign_agree']:.2f} | mean dyn cost anchors "
        f"{t048['mean_dyn_cost']:+.4f}, young {t048['mean_young_dyn_cost']:+.4f}")

    # ---- FREE RIDER: T037-a's within-net far-value vs junk-frac prediction
    band = np.arange(ANCHOR_POS[0], ANCHOR_POS[1] + 1)
    far_value = dce[:, band].mean(0)                     # (351,)
    junk_frac = (dce[:, band] <= -THRESH).mean(0)
    fr_pv = pearson(far_value, junk_frac)
    fr_sp = spearman(far_value, junk_frac)
    xv = dce[:, band].reshape(-1)
    yv = np.repeat(junk_frac, B)
    t037a = dict(
        prediction="POSITIVE within-net correlation (both are the same "
                   "under-selective long-range read); negative revives the "
                   "conflict T037-a declared pseudo",
        unit="position (64..414, n=351); far-value = mean static dCE over "
             "the 8 runs; junk-frac = fraction of runs with dCE <= -0.01",
        pearson=fr_pv, spearman=fr_sp,
        per_entry_pearson=pearson(xv, yv), n_positions=len(band),
        fires_prediction=bool(np.isfinite(fr_pv) and fr_pv > 0),
        negative_revives_conflict=bool(np.isfinite(fr_pv) and fr_pv < 0))
    log(f"T037-a free rider: far-value vs junk-frac r {fr_pv:+.3f} "
        f"(spearman {fr_sp:+.3f}; per-entry {t037a['per_entry_pearson']:+.3f})"
        f" -> prediction "
        f"{'CONFIRMED (positive)' if t037a['fires_prediction'] else ('NEGATIVE — conflict revived' if t037a['negative_revives_conflict'] else 'null')}")

    # ---------------------------------------------------------------- metrics
    for e in entries:
        e.pop("g4_group", None)
    metrics = dict(
        experiment="e085_anchor_description",
        purpose="P3 anchor-DESCRIPTION probe (T048's registered pivot; day5 "
                "proposal #4 / QUEUE e085): per-entry properties (P1 run-PCA "
                "projection, P2 readership = accumulated re-read attention "
                "mass, P3 self-surprisal LIAR control, P4 drift-alignment vs "
                "an independent clean continuation, P5 age) regressed on "
                "per-entry jackknife removal cost — static final-step V-zero "
                "dCE AND the dynamic free-run clean-judged cost (single-entry "
                "removal at its age-crossing, matched-stream continuation, "
                "clean-judge final-64 tail + horizon+8). FROZEN bars: "
                "readership partial r (given age+PCA) >= 0.35 with CI "
                "excluding 0 on the dynamic cost => anchor is attention-read; "
                "surprisal must fail (else 'hard tokens', honest report); "
                "P1/P2/P4 all <= 0.15 partial r on BOTH outcomes => KILL "
                "(anchor not entry-describable; P3 pivots to sequence "
                "level). Free rider: T037-a within-net far-value vs junk-frac "
                "positive-correlation prediction.",
        started=started, wall_s=elapsed(), threads=THREADS, cpu_only=True,
        net=dict(ckpt=str(CKPT), arch=dict(n_layer=4, n_head=4, n_embd=128,
                                           block_size=T_TOTAL, vocab=65),
                 params=n_params,
                 steps_in_ckpt=(int(st.get("step", -1))
                                if isinstance(st, dict) else -1),
                 val_ce=val_ce, val_ce_e053c=E053C_VAL_CE),
        seeds=dict(corpus=1337, prompts=SEED_PROMPT, sampling=SEED_SAMPLE,
                   run2=SEED_RUN2, entry_selection=SEED_SELECT,
                   continuation_anchor=SEED_CONT_A,
                   continuation_young=SEED_CONT_Y, bootstrap=0,
                   battery="the seed-202 8-draw, ALL 8 prompts; control = "
                           "seed-7 stream (e075/e080 A-none convention); run2 "
                           "= seed-1009 independent clean continuation "
                           "battery (same prompts); continuation pairs "
                           "(pruned, control) share a per-group seed so the "
                           "row-order draws are matched"),
        protocol=dict(
            B=B, prompt_tokens=PROMPT_TOK, t_total=T_TOTAL, temp=TEMP,
            topk=TOPK, boot_n=BOOT_N, threads=THREADS,
            entries=dict(anchors=100, young=50,
                         anchor_positions=list(ANCHOR_POS),
                         young_positions=list(YOUNG_POS),
                         per_run_anchor=ANCHORS_PER_RUN,
                         per_run_young=YOUNG_PER_RUN),
            properties=dict(
                P1=f"fraction of centered V-norm in the run's own cache-PCA "
                   f"top-{N_PCA} (V concat over layer x head, 512-d/entry, "
                   f"fit on all {cache_len} cache positions of the run)",
                P2="mean per-step attention mass received over decode steps "
                   "strictly after the write step (mean over layers+heads; "
                   "write-step diagonal excluded); extras: total mass, "
                   "16-bin trend slope / mean",
                P3="the control run's online full-softmax CE of the entry's "
                   "token at emission (= the clean net's in-context "
                   "surprisal; the run is unpruned)",
                P4=f"cos(V_entry, V at the same position in the seed-"
                   f"{SEED_RUN2} clean continuation battery)",
                P5="final-frame age 511 - p"),
            outcomes=dict(
                static="final-step 511-position V-zero sweep dCE at the "
                       "entry's position (e069/e075 instrument)",
                dyn_tail="single-entry V-zero at the age-crossing (removal "
                         "before the decode of the p+97 token; first "
                         "affected sample p+98), free-run to 512, clean-judge "
                         "CE of positions 448..511 minus the matched control "
                         "continuation's; young entries: removal at the tail "
                         "start (448 teacher-forced, first affected 449), "
                         "defined only for young positions 415..447 (entries "
                         "written inside the tail window are static-only)",
                dyn8="clean-judge CE of the first 8 emitted tokens after the "
                     "first affected sample (anchors p+98..p+105 where "
                     "exposure >= 8; young 449..456), same matched control"),
            bars=dict(readership=BAR_READERSHIP, kill=KILL_BAR)),
        gates=gates,
        control_run=dict(tail_ce_per_seq=r1_tail.tolist(),
                         clean_judge_tail_ce=cj1.tolist(),
                         a_star=a_star,
                         ages1_5=[float(a) for a in ages15],
                         final_token_ce=float(ce_clean.mean()),
                         sweep_mean_curve=mean_curve.tolist()),
        property_table=table,
        t048_entry_granularity=t048,
        free_rider_t037a=dict(t037a, far_value=far_value.tolist(),
                              junk_frac=junk_frac.tolist()),
        entries=entries,
        registered_decision=dict(clauses=clauses, clause=clause,
                                 verdict=verdict),
    )
    save_json(out_dir / "metrics.json", metrics)
    log("metrics.json written")

    plot(out_dir / "anchor_description.png", metrics)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot


def _resid_xy(es, xkey, ykey, cov_keys):
    x = np.array([e[xkey] for e in es], float)
    y = np.array([e[ykey] for e in es], float)
    Z = [np.array([e[k] for e in es], float) for k in cov_keys]
    ok = np.isfinite(x) & np.isfinite(y)
    for z in Z:
        ok &= np.isfinite(z)
    return (_resid(x[ok], [z[ok] for z in Z]),
            _resid(y[ok], [z[ok] for z in Z]))


def plot(path: Path, M: dict):
    dec = M["registered_decision"]
    table = M["property_table"]
    entries = M["entries"]
    anch = [e for e in entries if e["band"] == "anchor"]
    young = [e for e in entries if e["band"] == "young"]
    props = ["P1_pca_frac", "P2_readership", "P3_surprisal", "P4_drift_align"]
    plabels = ["P1 pca_frac", "P2 readership", "P3 surprisal\n(LIAR ctl)",
               "P4 drift-align"]
    onames = ["static", "static_anchor", "dyn_tail"]
    olabels = ["static (all 150)", "static (anchors)", "DYN-TAIL (anchors)"]
    cols = ["tab:blue", "tab:cyan", "tab:orange"]

    fig, axes = plt.subplots(2, 4, figsize=(25, 11))
    ax1, ax2, ax3, ax4 = axes[0]
    ax5, ax6, ax7, ax8 = axes[1]

    # ---- panel 1: static dCE age curve + sampled entries
    ages = np.arange(1, T_TOTAL)
    ax1.plot(ages, np.asarray(M["control_run"]["sweep_mean_curve"]), lw=1.0,
             color="k", alpha=0.8, label="mean V-zero dCE (control run)")
    ax1.scatter([e["age"] for e in anch], [e["static_dce"] for e in anch],
                s=9, color="tab:orange", alpha=0.6, label="sampled anchors (100)")
    ax1.scatter([e["age"] for e in young], [e["static_dce"] for e in young],
                s=9, color="tab:green", alpha=0.6, label="sampled young (50)")
    ax1.axvspan(97, 447, color="tab:orange", alpha=0.06)
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.set_xlabel("cache age (final frame)")
    ax1.set_ylabel("static V-zero dCE (nats)")
    ax1.set_ylim(-0.08, 1.0)
    ax1.legend(fontsize=8)
    ax1.set_title("E085-1 — static removal cost (final-step sweep) + sampled "
                  f"entries; a*={M['control_run']['a_star']}", fontsize=10)

    # ---- panel 2: THE property table (partial r given age, CIs)
    w = 0.26
    xs = np.arange(len(props))
    for oi, (onm, olb, c) in enumerate(zip(onames, olabels, cols)):
        rs = np.array([table[onm][k]["r_given_age"] for k in props], float)
        los = np.array([table[onm][k]["r_given_age_ci"][0] for k in props],
                       float)
        his = np.array([table[onm][k]["r_given_age_ci"][1] for k in props],
                       float)
        pos = xs + (oi - 1) * w
        ax2.bar(pos, np.nan_to_num(rs), w * 0.92, color=c, alpha=0.85,
                label=olb)
        ax2.errorbar(pos, np.nan_to_num(rs),
                     yerr=[np.nanmax(np.stack([rs - los, np.zeros_like(rs)]), 0),
                           np.nanmax(np.stack([his - rs, np.zeros_like(rs)]), 0)],
                     fmt="none", ecolor="k", lw=0.9, capsize=2)
    rd = table["dyn_tail"]["P2_readership"]
    ax2.scatter([xs[1] + w], [rd["r_given_age_pca"]], marker="*", s=170,
                color="red", zorder=5,
                label=f"P2 given age+PCA (registered bar): "
                      f"{rd['r_given_age_pca']:+.3f} "
                      f"[{rd['r_given_age_pca_ci'][0]:+.3f},"
                      f"{rd['r_given_age_pca_ci'][1]:+.3f}]")
    ax2.axhline(BAR_READERSHIP, color="tab:green", ls="--", lw=1.3)
    ax2.axhline(KILL_BAR, color="tab:red", ls=":", lw=1.2)
    ax2.axhline(-KILL_BAR, color="tab:red", ls=":", lw=1.2)
    ax2.axhline(0, color="k", lw=0.7)
    ax2.text(len(props) - 0.45, BAR_READERSHIP + 0.015,
             f"readership bar {BAR_READERSHIP}", fontsize=8, color="tab:green",
             ha="right")
    ax2.text(len(props) - 0.45, KILL_BAR + 0.015, f"kill |r| {KILL_BAR}",
             fontsize=8, color="tab:red", ha="right")
    ax2.set_xticks(xs, plabels, fontsize=8.5)
    ax2.set_ylabel("partial r given age (cluster-bootstrap CI over 8 runs)")
    ax2.legend(fontsize=7.5, loc="lower left")
    ax2.set_title("E085-2 — THE property table: partial r (given age) of each "
                  "property vs removal cost", fontsize=10)

    # ---- panel 3: readership vs dynamic cost (partial residuals)
    young_dyn = [e for e in young if "dyn_tail_cost" in e]
    for band_es, c, lb in ((anch, "tab:orange", "anchors"),
                           (young_dyn, "tab:green", "young (dyn-defined)")):
        rx, ry = _resid_xy(band_es, "P2_readership", "dyn_tail_cost",
                           ["P5_age", "P1_pca_frac"])
        ax3.scatter(rx, ry, s=14, alpha=0.6, color=c, label=lb)
    rr = table["dyn_tail_all"]["P2_readership"]["r_given_age_pca"]
    ax3.set_xlabel("readership residual (given age+PCA)")
    ax3.set_ylabel("dyn-tail cost residual (given age+PCA)")
    ax3.set_title(f"E085-3 — readership vs dynamic removal cost "
                  f"(partial residuals; pooled r {rr:+.3f})", fontsize=10)
    ax3.legend(fontsize=8)

    # ---- panel 4: surprisal honesty control
    for band_es, c, lb in ((anch, "tab:orange", "anchors"),
                           (young_dyn, "tab:green", "young (dyn-defined)")):
        rx, ry = _resid_xy(band_es, "P3_surprisal", "dyn_tail_cost", ["P5_age"])
        ax4.scatter(rx, ry, s=14, alpha=0.6, color=c, label=lb)
    r3 = table["dyn_tail"]["P3_surprisal"]["r_given_age"]
    li = dec["clauses"]["surprisal_liar_control"]
    ax4.set_xlabel("surprisal residual (given age)")
    ax4.set_ylabel("dyn-tail cost residual (given age)")
    ax4.set_title("E085-4 — LIAR control: surprisal vs dynamic cost "
                  f"(anchors r {r3:+.3f}; "
                  f"{'PASSED (failed to predict)' if li['passed'] else 'FIRED — hard tokens'})",
                  fontsize=9.5)
    ax4.legend(fontsize=8)

    # ---- panel 5: drift-alignment distributions
    ax5.hist([e["P4_drift_align"] for e in anch], bins=30, alpha=0.7,
             color="tab:orange",
             label=f"anchors (mean "
                   f"{np.mean([e['P4_drift_align'] for e in anch]):.3f})")
    ax5.hist([e["P4_drift_align"] for e in young], bins=30, alpha=0.7,
             color="tab:green",
             label=f"young (mean "
                   f"{np.mean([e['P4_drift_align'] for e in young]):.3f})")
    ax5.set_xlabel("drift-alignment cos(V1, V2 seed-1009)")
    ax5.set_ylabel("entries")
    ax5.legend(fontsize=8)
    ax5.set_title("E085-5 — P4 drift-alignment: run-specific vs "
                  "generic-for-its-age content", fontsize=10)

    # ---- panel 6: PCA frac distributions
    ax6.hist([e["P1_pca_frac"] for e in anch], bins=30, alpha=0.7,
             color="tab:orange",
             label=f"anchors (mean "
                   f"{np.mean([e['P1_pca_frac'] for e in anch]):.3f})")
    ax6.hist([e["P1_pca_frac"] for e in young], bins=30, alpha=0.7,
             color="tab:green",
             label=f"young (mean "
                   f"{np.mean([e['P1_pca_frac'] for e in young]):.3f})")
    ax6.set_xlabel("P1 pca_frac (top-8 capture of centered V-norm)")
    ax6.legend(fontsize=8)
    ax6.set_title("E085-6 — P1 run-PCA projection depth", fontsize=10)

    # ---- panel 7: T037-a free rider
    fr = M["free_rider_t037a"]
    ax7.scatter(np.asarray(fr["far_value"]), np.asarray(fr["junk_frac"]),
                s=10, alpha=0.6, color="tab:purple")
    ax7.set_xlabel("far-value: mean static dCE per position (over 8 runs)")
    ax7.set_ylabel("junk-frac: frac of runs with dCE <= -0.01")
    ax7.set_title("E085-7 — T037-a free rider: per-position far-value vs "
                  f"junk-frac\nr {fr['pearson']:+.3f} (spearman "
                  f"{fr['spearman']:+.3f}) — predicted POSITIVE: "
                  f"{'CONFIRMED' if fr['fires_prediction'] else ('NEGATIVE — conflict revived' if fr['negative_revives_conflict'] else 'null')}",
                  fontsize=9)

    # ---- panel 8: verdict text
    ax8.axis("off")
    t48 = M["t048_entry_granularity"]
    lines = [
        "REGISTERED (frozen):",
        f"  readership partial r (age+PCA) >= {BAR_READERSHIP}, CI excl 0, "
        f"DYN-TAIL: "
        f"[{'FIRES' if dec['clauses']['readership_bar']['fires'] else 'no'}] "
        f"r {dec['clauses']['readership_bar']['r']:+.3f} "
        f"[{dec['clauses']['readership_bar']['ci'][0]:+.3f},"
        f"{dec['clauses']['readership_bar']['ci'][1]:+.3f}]",
        f"  surprisal LIAR control must fail: "
        f"[{'PASS' if li['passed'] else 'FIRED'}] "
        f"P3 r {dec['clauses']['surprisal_liar_control']['r_p3']:+.3f} vs "
        f"P2 r {dec['clauses']['surprisal_liar_control']['r_p2']:+.3f}",
        f"  kill |r|<= {KILL_BAR} on P1/P2/P4, BOTH outcomes: "
        f"[{'FIRES' if dec['clauses']['kill']['fires'] else 'no'}]",
        "   static r: " + ", ".join(
            f"{k[:2]} {v:+.3f}" for k, v in
            dec["clauses"]["kill"]["r_static"].items()),
        "   static-anchor r: " + ", ".join(
            f"{k[:2]} {v:+.3f}" for k, v in
            dec["clauses"]["kill"]["r_static_anchor"].items()),
        "   dyn-tail r: " + ", ".join(
            f"{k[:2]} {v:+.3f}" for k, v in
            dec["clauses"]["kill"]["r_dyn"].items()),
        "",
        f"T048 entry granularity: r(static, dyn) {t48['pearson']:+.3f}, "
        f"sign-agree {t48['sign_agree']:.2f}",
        f"  mean dyn cost: anchors {t48['mean_dyn_cost']:+.4f} | young "
        f"{t48['mean_young_dyn_cost']:+.4f} | dyn+8 anchors "
        f"{t48['mean_dyn8_cost']:+.4f}",
        f"T037-a free rider: r {fr['pearson']:+.3f} "
        f"({'positive — confirmed' if fr['fires_prediction'] else 'see panel 7'})",
        "",
        f"VERDICT [{dec['clause']}]:",
    ] + [f"  {wd}" for wd in _wrap(dec["verdict"], 98)]
    ax8.text(0.02, 0.97, "E085 — P3 anchor-description probe (T048 pivot)",
             fontsize=13, weight="bold", va="top")
    for i, t in enumerate(lines):
        ax8.text(0.02, 0.925 - i * 0.0355, t, fontsize=8.6, va="top",
                 family="monospace")

    fig.suptitle("E085 — what describes the anchor? per-entry properties vs "
                 f"jackknife removal cost | clause: {dec['clause']}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _wrap(text: str, width: int):
    import textwrap
    return textwrap.wrap(text, width=width)


if __name__ == "__main__":
    main()
