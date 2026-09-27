"""E104 — the FAILING-DP CENSUS: where and why does attention-addressed
read prediction FAIL? (T054's registered residual; QUEUE slot e104.)

THE QUESTION: e100 established the read is ATTENTION-ADDRESSED at pooled
AUC 0.907 [0.886, 0.931] — the opened set IS the attended set. The
residual: ~10% pooled headroom and a tail of failing decision points
(DPs), min per-DP AUC 0.405. Is the residual a THING (a localized failure
mode we can name) or BLUR (interventional-threshold noise in the opened-
set ground truth, scattering everywhere at base rates)?

BANDS (registered; sized BEFORE any e104 compute from e100's STORED
per-DP table in runs/e100/metrics.json — public prior data):
  - The task-spec literal bands — failure = per-DP AUC < 0.7, strict
    core = AUC < 0.5 — are known from that table to contain exactly
    1/98 valid DPs (dp 27, AUC 0.405, one opened entry at age 199).
    POWER GATE: with n_fail = 1 no ">= 3x base with CI excluding base"
    stratum test can fire (any 1-cell Wilson CI overlaps the base).
  - PRIMARY POWERED BAND: BOTTOM-DECILE failures = the 10 worst valid
    DPs by per-DP 16-head attention-mass AUC (rank-based; ties by AUC
    then dp index) — the residual mass the ~10% headroom lives in.
    The clustering bars run on THIS band. The literal < 0.7 / < 0.5
    bands are reported as the STRICT CORE, with their member(s)' full
    anatomy.

STRATA (families a-e, registered):
  (a) margin quartile Q0..Q3 (e084's stratification; Q0 = near-tie,
      Q3 = confident — e084 stored quartile edges 0.426/1.213/3.001);
  (b) opened-set size: 1 / 2 / 3-4 / 5+ (task: "1 vs 3-4 — is
      single-open harder?");
  (c) opened-entry age profile: young-frac of the opened set (age<=10):
      YOUNG_HEAVY (>0.5) / MIXED ((0,0.5]) / NO_YOUNG (=0);
  (d) layer-profile: L1L2_RESCUED (per-DP AUC on L1+L2-only mass exceeds
      the full-16 AUC by > 0.05) vs NOT_RESCUED; descriptive per-layer
      per-DP AUCs (which layer's mass is misaligned);
  (e) candidate-attention entropy quartiles EH0..EH3 (entropy of the
      normalized 16-head attention mass over the 80 candidates, /ln 80).

REGISTERED BARS (frozen; on the powered band):
  - LOCALIZED RESIDUE fires iff >= 2 strata (any families) have failure
    rate >= 3x base (base = 10/98) with Wilson 95% CI excluding base,
    AND a permutation guard p < 0.05 (10k uniform shuffles of the 10
    failure labels over the 98 valid DPs, seed 0; statistic = number of
    strata meeting the >= 3x + CI criterion; the guard only makes
    LOCALIZED harder — multiple-comparison protection across ~17 strata).
  - else MEASUREMENT BLUR: failures scatter at base rates — the residual
    is interventional-threshold noise in the ground truth; the T054
    unification stands clean at its resolution.

MECHANISM NAMING within the verdict (registered): from the SAME
intervention call, per-entry FLIP GAP g_r = max_{c != t1} lg_int[c]
- lg_int[t1] (opened iff g_r > 0; near-threshold iff |g_r| < eps; eps =
0.10 logits primary, 0.05/0.20 sensitivity). Per-DP blur_frac =
near-threshold fraction of the 80 candidates. The threshold-noise
component is CONFIRMED iff (i) failing DPs' median blur_frac >= 2x the
non-failing median AND (ii) after dropping near-threshold labels from
BOTH classes, >= 5/10 failing DPs rise above 0.85 AUC. Otherwise the
failing cells are LABEL-STABLE => genuine misalignment (second-channel
candidate). The STRICT-CORE member(s) get a full anatomy: opened entry
age, flip gap, attention rank (full-16 and per-layer).

SECONDARY (headroom decomposition: layer choice vs threshold):
  pooled AUC recomputed with L0 EXCLUDED (12 heads, L1-L3), L1/L2 only
  (8 heads), per-layer; pooled CLEAN-LABEL AUC (near-threshold pairs
  dropped, eps grid); adaptive-k precision (k = |opened| per DP) for
  full / L0ex / L12. Layer-choice gain = AUC(L0ex) - AUC(full);
  threshold gain = AUC(clean, eps=0.10) - AUC(full); combined reported.

REGISTERED PREDICTION (pre-compute, honest — the band sizing above
peeked at e100's stored per-DP table): (a) Q0 fires (8/10 of the stored
worst-10 sit in Q0) and (c) MIXED fires (stored worst-10 young-frac
0.34 vs 0.89 in the rest), heavily OVERLAPPING — verdict LOCALIZED with
a compound name. Mechanism: blur_frac elevated in failures (near-tie
margins => fragile argmax labels), but the strict-core member (dp 27,
old single read at age 199) is LABEL-STABLE with a large flip gap — a
two-part residue: "near-tie threshold blur" + "old-quiet single read".
Layer headroom (+L0ex) ~ +0.02; threshold headroom >= layer headroom.

GATES: G2 params (873,472), G1 val CE (vs e053c 1.5227 +/- 0.02), G0b
incremental-KV vs full recompute (max prob dev < 1e-4), G8
feature-forward identity (< 1e-4), G5e084 (per-DP V-zero opened sizes
vs e084's stored flips_vz, >= 90/100), G5e100 (per-DP attention-mass
AUC vs e100's stored values, max |dev| < 1e-9 — bit-identical ground
truth AND features; any drift kills the census's premise).

Run:     python lab/e104_failing_dp_census.py     (E104_SMOKE=1 smoke)
Outputs: runs/e104/metrics.json + runs/e104/failing_dp_census.png
Envelope: NO training; CPU-only (CUDA_VISIBLE_DEVICES forced -1, cuda
stubbed pre-import, e070/e084/e100 pattern); 8 torch threads; single
step ~4-6 min (same compute shape as e100's 222s). No NOTES/THINKING/
QUEUE/STATE edits; no commit (the dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu" (e053b/e069/e070/
# e084/e100 pattern)
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
SEED_PROMPT, SEED_SAMPLE = 202, 7         # e053b/e053c/e069/e070/e072/e084/e100
N_PROMPTS = 8
B = 4                                     # battery A (e069 verbatim)
B_X = 4                                   # fresh extras (e072 family, e084)
SEED_PROMPT_X, SEED_SAMPLE_X = 302, 17    # 202/7 family +100/+10
BOOT_N = 1000
THREADS = 8      # e084's measured sweet spot on this contended box
torch.set_num_threads(THREADS)

CHUNK_512 = 64                            # e069/e070/e072/e084/e100 chunk

# ---- design constants (registered; ALL verbatim e084/e100) ------------------
Q_LO, Q_HI = 96, 510                      # candidate query positions
PER_QUARTILE = 25                         # 25 per margin quartile
N_DP = 4 * PER_QUARTILE                   # 100 decision points
SEED_STRAT = 2084                         # margin-stratified sampling seed
YOUNG = list(range(1, 11))                # ages 1-10 (all)
MID = list(range(11, 61))                 # ages 11-60 (all)
N_OLD = 20                                # old entries per decision point
OLD_SEED_BASE = 9000                      # old-age rng seed = base + dp_index
N_ENTRIES = len(YOUNG) + len(MID) + N_OLD  # 80
YOUNG_AGE_MAX = 10                        # stratum (c) definition

# ---- census constants (registered) ------------------------------------------
FAIL_AUC = 0.7                            # task-spec literal failure band
STRICT_AUC = 0.5                          # strict core band
DECILE_FRAC = 0.10                        # powered band: worst decile of DPs
BASE_RATE_MULT = 3.0                      # bar: stratum rate >= 3x base
MIN_FIRING_STRATA = 2                     # bar: >= 2 strata fire
PERM_N = 10_000                           # permutation guard shuffles
PERM_SEED = 0
L1L2_RESCUE_DELTA = 0.05                  # stratum (d) definition
EPS_GRID = [0.05, 0.10, 0.20]             # near-threshold |flip gap| cuts
EPS_PRIMARY = 0.10
EPS_KEY = {e: f"{e:.2f}" for e in EPS_GRID}   # canonical key per eps
EPS_PRIMARY_KEY = EPS_KEY[EPS_PRIMARY]
BLUR_FRAC_RATIO = 2.0                     # mechanism test (i)
BLUR_RECOVER_COUNT = 5                    # mechanism test (ii): of 10
BLUR_RECOVER_AUC = 0.85

# ---- stored references (gates) ----------------------------------------------
E084_METRICS = REPO / "runs" / "e084" / "metrics.json"
E100_METRICS = REPO / "runs" / "e100" / "metrics.json"
CKPT = REPO / "runs" / "checkpoints" / "e053c_ctx512.pt"
E053C_VAL_CE = 1.5226792494455974

SMOKE = os.environ.get("E104_SMOKE", "") == "1"
GEN_STOP = T_TOTAL                        # protocol generation length
if SMOKE:
    PER_QUARTILE = 2
    N_DP = 4 * PER_QUARTILE
    GEN_STOP = 160                        # short code-path check only
    B_X = 0                               # battery A only
    Q_HI = GEN_STOP - 2                   # 158

T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# --------------------------------------------- machinery (VERBATIM e069/e084/e100)

@torch.no_grad()
def _manual_chunk2(net: TinyGPT, idxs, vzero, kdrop, layers, pos_offset=0,
                   vswap_pos=None, donor_stack=None):
    """e069/e084/e100's _manual_chunk VERBATIM (vswap/kdrop arms inert here —
    e104 uses only the clean row and V-zero masks, but the instrument is kept
    byte-identical to the rig whose ground truth it reproduces)."""
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
        if vswap_pos is not None and bool((vswap_pos >= 0).any()):
            v4 = v.view(N, T, H, d)
            rows = vswap_pos >= 0
            v4[rows, vswap_pos[rows]] = donor_stack[li][vswap_pos[rows]]
            v = v4.transpose(1, 2)
        else:
            v = v.view(N, T, H, d).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
        att = att.masked_fill(causal, float("-inf"))
        if active and kdrop is not None and bool(kdrop.any()):
            cnt = kdrop.sum(1)
            if bool((cnt <= 1).all()):
                has = cnt > 0
                nr = torch.arange(N, dtype=torch.long)[has]
                pos = kdrop[has].nonzero(as_tuple=True)[1]
                sub = att[nr, :, :, pos]                       # (n, H, T)
                jmask = (torch.arange(T)[None, None, :]
                         != pos[:, None, None])                # keep diagonal
                att[nr, :, :, pos] = sub.masked_fill(jmask, float("-inf"))
            else:
                att = att.masked_fill(kdrop[:, None, None, :] & nondiag,
                                      float("-inf"))
        probs = torch.softmax(att, dim=-1)
        if active and vzero is not None and bool(vzero.any()):
            v = v.masked_fill(vzero[:, None, :, None], 0.0)
        y = (probs @ v).transpose(1, 2).reshape(N, T, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
    return net.lm_head(net.ln_f(x[:, -1, :]))


@torch.no_grad()
def manual_logits2(net, idxs, vzero=None, kdrop=None, layers=None, chunk=64,
                   pos_offset=0, vswap_pos=None, donor_stack=None):
    outs = []
    for i in range(0, idxs.shape[0], chunk):
        sl = slice(i, i + chunk)
        outs.append(_manual_chunk2(
            net, idxs[sl],
            vzero[sl] if vzero is not None else None,
            kdrop[sl] if kdrop is not None else None,
            layers, pos_offset,
            vswap_pos[sl] if vswap_pos is not None else None,
            donor_stack))
    return torch.cat(outs, 0)


def sample_and_ce(logits_clean: torch.Tensor, gen: torch.Generator):
    """Sample (temp 0.8, top-k 40) from filtered probs; CE from FULL softmax.
    [VERBATIM e053b/e084/e100]"""
    lg = logits_clean / TEMP
    v, _ = torch.topk(lg, TOPK)
    lg_f = lg.masked_fill(lg < v[-1], float("-inf"))
    tok = int(torch.multinomial(torch.softmax(lg_f, -1), 1, generator=gen))
    p_full = torch.softmax(logits_clean, -1)
    return tok, float(-math.log(max(p_full[tok].item(), 1e-12)))


@torch.no_grad()
def prefill_batch(net: TinyGPT, idx: torch.Tensor):
    """Batched prefill, (B, Tp) -> last-position logits (B, V) + KV cache.
    [VERBATIM e069/e084/e100]"""
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
    """Batched incremental decode: (B,) tokens at position pos -> (B, V).
    [VERBATIM e069/e084/e100]"""
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
def generate_batch_recorded(net: TinyGPT, prompts, gen: torch.Generator,
                            gen_stop: int = T_TOTAL):
    """e069/e084/e100's generate_batch VERBATIM + per-step decision logits:
    hist[:, j] is the distribution at query position 63+j (the decision for
    token 64+j). Returns (idx, final_logits, hist (B, gen_stop-64, V))."""
    B = len(prompts)
    idx = torch.stack(prompts)
    logits, kv = prefill_batch(net, idx)
    hist = [logits.clone()]
    for t in range(PROMPT_TOK, gen_stop):
        toks = torch.zeros(B, dtype=torch.long)
        for j in range(B):
            tok, _ = sample_and_ce(logits[j], gen)
            toks[j] = tok
        idx = torch.cat([idx, toks[:, None]], 1)
        if t < gen_stop - 1:
            logits = decode_step_batch(net, toks, t, kv)
            hist.append(logits.clone())
    return idx, logits, torch.stack(hist, 1)


# --------------------------------------------- e104's own (gated) instruments

@torch.no_grad()
def query_features(net: TinyGPT, ctx: torch.Tensor):
    """ONE clean forward over ctx (1-D token ids, length T); record, per
    layer, the QUERY-SIDE state BEFORE any intervention (VERBATIM e100 —
    G5e100 and G8 depend on byte-identity):
      - att_last (L, H, T): softmax attention probabilities of the decision
        row (last position) over all positions;
    Returns att_last + the decision-row logits (for G8)."""
    T = ctx.shape[0]
    x = net.wte(ctx[None]) + net.wpe(torch.arange(T))[None]
    H = net.cfg.n_head
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    att_last = []
    for blk in net.h:
        xh = blk.ln1(x)
        qkv = blk.attn.c_attn(xh)
        C = qkv.shape[-1] // 3
        d = C // H
        q, k, v = qkv.split(C, dim=2)
        q = q.view(1, T, H, d).transpose(1, 2)          # (1,H,T,d)
        k = k.view(1, T, H, d).transpose(1, 2)
        v = v.view(1, T, H, d).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
        att = att.masked_fill(causal, float("-inf"))
        probs = torch.softmax(att, dim=-1)              # (1,H,T,T)
        y = (probs @ v).transpose(1, 2).reshape(1, T, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
        att_last.append(probs[0, :, -1, :].clone())     # (H, T)
    logits = net.lm_head(net.ln_f(x[0, -1]))
    return torch.stack(att_last, 0), logits


def entry_positions(q: int, dp_index: int):
    """e084/e100's entry band VERBATIM: young 1-10 + mid 11-60 + 20 old
    sampled without replacement uniformly from [61, q+1] (default_rng seed
    9000+dp_index). Returns (ages (80,), positions (80,)) with
    position = T - age, T = q+1."""
    T = q + 1
    rng = np.random.default_rng(OLD_SEED_BASE + dp_index)
    avail = np.arange(61, T + 1)
    old = [int(a) for a in rng.choice(avail, size=N_OLD, replace=False)]
    entries = list(YOUNG) + list(MID) + old              # 80 ages, e084 order
    ps = [T - a for a in entries]
    return entries, ps


def opened_set(net: TinyGPT, idx16, b, q, entries, ps):
    """GROUND TRUTH at the e084/e100 bars: one batched manual_logits2 call,
    1 clean row + 80 V-zero rows (v := 0 at that position, every layer,
    every head, all queries — e069/e084 whole-position instrument verbatim).
    e104 extension (same call, no extra forwards): per-entry FLIP GAP
    g_r = max_{c != t1} lg_int[c] - lg_int[t1]; opened iff g_r > 0 — the
    continuous form of the argmax-flip ground truth.
    Returns (opened flags (80,) bool, gaps (80,) float, t1, margin)."""
    ctx = idx16[b, :q + 1]
    T = q + 1
    n = 1 + len(entries)
    vz = torch.zeros(n, T, dtype=torch.bool)
    for r, p in enumerate(ps):
        vz[r + 1, p] = True
    lg = manual_logits2(net, ctx[None].expand(n, -1).contiguous(),
                        vz, None, None, chunk=CHUNK_512)
    top2 = torch.topk(lg[0], 2)
    t1 = int(top2.indices[0])
    margin = float(top2.values[0] - top2.values[1])
    opened = np.asarray([int(lg[r + 1].argmax()) != t1
                         for r in range(len(entries))], dtype=bool)
    tv, ti = torch.topk(lg[1:], 2, dim=1)                # (80, 2)
    lgt1 = lg[1:, t1]                                    # (80,)
    gaps = torch.where(ti[:, 0] != t1, tv[:, 0] - lgt1, tv[:, 1] - lgt1)
    return opened, gaps.numpy().astype(float), t1, margin


# ------------------------------------------------------------------ statistics

def rankdata_avg(v):
    """Ascending ranks 1..n with TIES AVERAGED (proper Mann-Whitney ranks)."""
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    sv = v[order]
    ranks = np.empty(len(v), float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auc_pairs(scores, labels):
    """Mann-Whitney AUC of scores vs binary labels + pair count.
    Returns (auc, n_pairs); (nan, 0) if a class is empty."""
    labels = np.asarray(labels, bool)
    n1 = int(labels.sum())
    n0 = int((~labels).sum())
    if n1 == 0 or n0 == 0:
        return float("nan"), 0
    r = rankdata_avg(scores)
    conc = float(r[labels].sum() - n1 * (n1 + 1) / 2.0)
    return conc / (n1 * n0), n1 * n0


def pooled_auc(per_dp):
    """per_dp: list of (auc, n_pairs) -> pair-weighted pooled AUC.
    Zero-pair entries (auc=nan) are skipped — nan*0 would poison the sum."""
    arr = [(a, p) for a, p in per_dp if p > 0 and np.isfinite(a)]
    tot = sum(p for _, p in arr)
    if tot == 0:
        return float("nan")
    return sum(a * p for a, p in arr) / tot


def bootstrap_pooled(per_dp, n=BOOT_N, seed=0):
    """Bootstrap CI of the pair-weighted pooled AUC (decision-point
    resampling with replacement)."""
    arr = [(a, p) for a, p in per_dp if p > 0]
    if not arr:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sel = rng.integers(0, len(arr), len(arr))
        vals.append(pooled_auc([arr[i] for i in sel]))
    return (float(np.nanpercentile(vals, 2.5)),
            float(np.nanpercentile(vals, 97.5)))


def precision_at_k(scores, labels, ks):
    """Top-k precision per k (descending scores, stable)."""
    order = np.argsort(-np.asarray(scores, float), kind="mergesort")
    lab = np.asarray(labels, bool)[order]
    out = {}
    for k in ks:
        k = min(k, len(lab))
        out[k] = float(lab[:k].mean())
    return out


def jaccard_at_k(scores, labels, k):
    """|top-k(pred) ∩ opened| / |top-k(pred) ∪ opened| (set overlap)."""
    order = np.argsort(-np.asarray(scores, float), kind="mergesort")
    pred = set(order[:k].tolist())
    true = set(np.where(np.asarray(labels, bool))[0].tolist())
    if not pred and not true:
        return float("nan")
    return len(pred & true) / len(pred | true)


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or x[m].std() == 0 or y[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def wilson_ci(k, n, z=1.959963984540054):
    """Wilson score interval for k successes of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (center - half, center + half)


def wilson_lo_vec(k, n, base, z=1.959963984540054):
    """Vectorized Wilson lower bounds; True where CI excludes base (above)."""
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half > base


# ---------------------------------------------------------------------- main

def main():
    assert torch.cuda.device_count() == 0, "CPU-only violated: CUDA visible"
    assert common.DEVICE == "cpu", f"common.DEVICE={common.DEVICE} != cpu"
    out_dir = run_dir("e104")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=SMOKE)
    suffix = "_smoke" if SMOKE else ""

    # ---- battery: rebuild corpus + net EXACTLY as e053c/e069/e070/e084/e100
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
        f"e053c {E053C_VAL_CE:.4f} -> G1 "
        f"{'PASS' if gates['G1_val_ce']['ok'] else 'FAIL'}")

    # ---- battery A: e069 protocol verbatim (seeds 202/7, B=4) + recording
    gen_p = torch.Generator().manual_seed(SEED_PROMPT)
    ix = torch.randint(len(corp.val) - PROMPT_TOK - 1, (N_PROMPTS,),
                       generator=gen_p)
    prompts = [corp.val[i:i + PROMPT_TOK] for i in ix]
    log(f"battery A: {N_PROMPTS} prompts (seed {SEED_PROMPT}); using first "
        f"B={B}; prompt0 prefix: {corp.decode(prompts[0])[:32]!r}")
    gen = torch.Generator().manual_seed(SEED_SAMPLE)
    idxA, final_logits, histA = generate_batch_recorded(net, prompts[:B], gen,
                                                        gen_stop=GEN_STOP)
    log(f"battery A: generated {B} seqs 64->{GEN_STOP} (fixed anchor, CPU)")

    # G0b: batched incremental-KV final logits vs full recompute
    with torch.no_grad():
        full = manual_logits2(net, idxA[:, :-1], None, None, None, chunk=B)
    dev_b = float((torch.softmax(final_logits.float(), -1)
                   - torch.softmax(full.float(), -1)).abs().max())
    gates["G0b_kvs_vs_full"] = dict(max_prob_dev=dev_b, ok=bool(dev_b < 1e-4))
    log(f"G0b incremental-KV vs full recompute: max prob dev {dev_b:.2e} "
        f"-> {'PASS' if dev_b < 1e-4 else 'FAIL'}")

    # ---- battery X extras: fresh sequences (e072 302/17 family, e084 cut)
    if B_X > 0:
        log(f"battery X: {B_X} fresh sequences (prompt seed {SEED_PROMPT_X}, "
            f"sampling seed {SEED_SAMPLE_X})")
        gen_px = torch.Generator().manual_seed(SEED_PROMPT_X)
        ix12 = torch.randint(len(corp.val) - PROMPT_TOK - 1, (B_X,),
                             generator=gen_px)
        prompts12 = [corp.val[i:i + PROMPT_TOK] for i in ix12]
        gen_x = torch.Generator().manual_seed(SEED_SAMPLE_X)
        idxX, _, histX = generate_batch_recorded(net, prompts12, gen_x,
                                                 gen_stop=GEN_STOP)
        idx16 = torch.cat([idxA, idxX], 0)
        hist16 = torch.cat([histA, histX], 0)
        log(f"battery: {idx16.shape[0]} sequences = e069's 4 + {B_X} fresh")
    else:
        idx16, hist16 = idxA, histA
        log(f"battery: {idx16.shape[0]} sequences (battery A only, smoke)")

    # ---- decision-point selection: e084 VERBATIM (margin quartiles)
    cands = []                                         # (b, q, margin)
    for b in range(idx16.shape[0]):
        for q in range(Q_LO, Q_HI + 1):
            lg = hist16[b, q - PROMPT_TOK + 1].float()
            tv = torch.topk(lg, 2).values
            cands.append((b, q, float(tv[0] - tv[1])))
    margins_all = np.asarray([c[2] for c in cands])
    qs = np.percentile(margins_all, [25, 50, 75])
    quart_of = np.digitize(margins_all, qs)            # 0..3
    rng = np.random.default_rng(SEED_STRAT)
    dps = []
    for qt in range(4):
        pool = [i for i in range(len(cands)) if quart_of[i] == qt]
        take = rng.choice(pool, size=PER_QUARTILE, replace=False)
        dps += [cands[i] + (qt,) for i in take]
    log(f"candidates {len(cands)}; margin quartile edges {qs.round(3).tolist()}; "
        f"sampled {len(dps)} decision points (seed {SEED_STRAT}) — e084's "
        f"selection rule verbatim")

    # ---- the census: per decision point, features BEFORE intervention,
    #      then the ground-truth opened set + flip gaps (same V-zero call)
    t_probe = time.time()
    per_dp = []        # records for metrics
    pooled = {"full": [], "L0ex": [], "L12": [], "clean": {e: [] for e in EPS_GRID},
              "L0ex_clean": []}
    layer_pooled = [None] * 4
    pk_adapt = {"full": [], "L0ex": [], "L12": []}
    jac_adapt = {"full": [], "L0ex": [], "L12": []}
    opened_sizes = []

    for i, (b, q, _, qt) in enumerate(dps):
        entries, ps = entry_positions(q, i)

        # (1) query-side attention masses from ONE clean forward
        att_last, q_logits = query_features(net, idx16[b, :q + 1])
        L, H, T = att_last.shape
        ps_np = np.asarray(ps)
        att_full = att_last.sum(0).sum(0).numpy()          # (T,) 16-lh sum
        att_L0ex = att_last[1:].sum(0).sum(0).numpy()      # L1-L3, 12 heads
        att_L12 = att_last[1:3].sum(0).sum(0).numpy()      # L1+L2, 8 heads
        att_layer = [att_last[li].sum(0).numpy() for li in range(L)]
        s_full = att_full[ps_np]
        s_L0ex = att_L0ex[ps_np]
        s_L12 = att_L12[ps_np]
        s_layer = [att_layer[li][ps_np] for li in range(L)]

        # (2) ground truth + flip gaps: V-zero flips at the e084/e100 bars
        opened, gaps, t1, margin = opened_set(net, idx16, b, q, entries, ps)
        n_open = int(opened.sum())
        opened_sizes.append(n_open)

        # (3) entropy of the candidate attention distribution (stratum e)
        p_c = s_full / s_full.sum()
        nz = p_c > 0
        ent = float(-(p_c[nz] * np.log(p_c[nz])).sum() / math.log(N_ENTRIES))

        rec = dict(dp=i, b=int(b), q=int(q), quartile=int(qt), margin=margin,
                   t1=t1, n_opened=n_open, entropy_norm=ent,
                   opened_ages=[int(a) for a, o in zip(entries, opened) if o],
                   opened_gaps=[float(g) for g, o in zip(gaps, opened) if o])
        valid = 1 <= n_open <= N_ENTRIES - 1
        if valid:
            a_f, p_f = auc_pairs(s_full, opened)
            a_x, _ = auc_pairs(s_L0ex, opened)
            a_12, _ = auc_pairs(s_L12, opened)
            rec["auc_full"], rec["auc_L0ex"], rec["auc_L12"] = a_f, a_x, a_12
            rec["auc_layer"] = [float(auc_pairs(s_layer[li], opened)[0])
                                for li in range(L)]
            rec["young_frac_opened"] = float(
                np.mean([a <= YOUNG_AGE_MAX
                         for a, o in zip(entries, opened) if o]))
            pooled["full"].append((a_f, p_f))
            pooled["L0ex"].append((a_x, p_f))
            pooled["L12"].append((a_12, p_f))
            for li in range(L):
                a, p = auc_pairs(s_layer[li], opened)
                tot = layer_pooled[li]
                layer_pooled[li] = (a, p) if tot is None else (
                    ((tot[0] * tot[1] + a * p) / (tot[1] + p),
                     tot[1] + p) if p else tot)
            # blur: near-threshold masks, per eps; clean-label AUCs
            for eps in EPS_GRID:
                clean = np.abs(gaps) >= eps
                a_c, p_c2 = auc_pairs(s_full[clean], opened[clean])
                rec[f"auc_clean@{EPS_KEY[eps]}"] = a_c
                pooled["clean"][eps].append((a_c, p_c2))
            clean1 = np.abs(gaps) >= EPS_PRIMARY
            a_xc, _ = auc_pairs(s_L0ex[clean1], opened[clean1])
            rec["auc_L0ex_clean"] = a_xc
            pooled["L0ex_clean"].append((a_xc, p_f))
            for name, s in [("full", s_full), ("L0ex", s_L0ex),
                            ("L12", s_L12)]:
                pk_adapt[name].append(precision_at_k(s, opened, [n_open])[n_open])
                jac_adapt[name].append(jaccard_at_k(s, opened, n_open))
            rec["blur_frac"] = {EPS_KEY[eps]: float((np.abs(gaps) < eps).mean())
                                for eps in EPS_GRID}
            # attention ranks of opened entries (full-16 + L1/L2 + per-layer)
            rk = rankdata_avg(-s_full)
            rec["opened_ranks_full"] = [float(rk[j])
                                        for j in np.where(opened)[0]]
        per_dp.append(rec)
        if (i + 1) % 10 == 0 or i == len(dps) - 1:
            rate = (time.time() - t_probe) / (i + 1)
            log(f"census {i + 1}/{len(dps)} decision points ({rate:.2f}s/dp, "
                f"ETA {rate * (len(dps) - i - 1):.0f}s)")

    # ---- G8: feature-forward identity (first DP) vs manual_logits2 clean row
    b0, q0 = dps[0][0], dps[0][1]
    _, q_log0 = query_features(net, idx16[b0, :q0 + 1])
    ref0 = manual_logits2(net, idx16[b0, :q0 + 1][None], None, None, None,
                          chunk=1)[0]
    g8_dev = float((q_log0 - ref0).abs().max())
    gates["G8_feature_forward_identity"] = dict(
        dp=0, max_logit_dev=g8_dev, tol=1e-4, ok=bool(g8_dev < 1e-4))
    log(f"G8 feature-forward vs manual clean row: max |logit dev| "
        f"{g8_dev:.2e} -> {'PASS' if g8_dev < 1e-4 else 'FAIL'}")

    # ---- G5e084: per-dp V-zero opened-set sizes vs stored flips_vz
    # ---- G5e100: per-dp attention AUC vs e100's stored values (bit-identity)
    if SMOKE:
        gates["G5_e084_identity"] = dict(skipped=True, reason="smoke battery")
        gates["G5_e100_auc_identity"] = dict(skipped=True, reason="smoke")
        log("G5 gates: SKIPPED (smoke)")
    else:
        e084 = json.loads(E084_METRICS.read_text(encoding="utf-8"))
        ref_counts = [p["flips_vz"] for p in e084["per_decision_point"]]
        mine = [r["n_opened"] for r in per_dp]
        assert len(ref_counts) == len(mine), "e084 per-DP count mismatch"
        matches = sum(1 for a, b_ in zip(mine, ref_counts) if a == b_)
        gates["G5_e084_identity"] = dict(
            n_match=matches, n_dp=len(mine), bar=90,
            ok=bool(matches >= 90),
            count_pearson=pearson(mine, ref_counts))
        log(f"G5 vs e084 stored flips_vz: {matches}/{len(mine)} exact "
            f"matches (bar 90) -> "
            f"{'PASS' if gates['G5_e084_identity']['ok'] else 'FAIL'}")

        e100 = json.loads(E100_METRICS.read_text(encoding="utf-8"))
        e100_auc = {r["dp"]: r.get("auc_att")
                    for r in e100["per_decision_point"]}
        devs, auc_pairs_ok = [], 0
        for r in per_dp:
            ref = e100_auc.get(r["dp"])
            if ref is not None and "auc_full" in r:
                devs.append(abs(r["auc_full"] - ref))
                if abs(r["auc_full"] - ref) < 1e-9:
                    auc_pairs_ok += 1
        max_dev = max(devs) if devs else float("nan")
        gates["G5_e100_auc_identity"] = dict(
            n_compared=len(devs), n_match_lt_1e9=auc_pairs_ok,
            max_auc_dev=max_dev, tol=1e-9,
            ok=bool(auc_pairs_ok == len(devs) and max_dev < 1e-9))
        log(f"G5 vs e100 stored per-DP AUCs: {auc_pairs_ok}/{len(devs)} "
            f"within 1e-9 (max dev {max_dev:.2e}) -> "
            f"{'PASS' if gates['G5_e100_auc_identity']['ok'] else 'FAIL'}")

    # ================= the census proper (registered analyses) ==============
    valid_recs = [r for r in per_dp if "auc_full" in r]
    n_valid = len(valid_recs)
    n_zero_open = int(sum(1 for s in opened_sizes if s == 0))

    # ---- failure bands
    order = sorted(valid_recs, key=lambda r: (r["auc_full"], r["dp"]))
    n_fail = max(1, int(round(DECILE_FRAC * n_valid)))
    fail_dps = {r["dp"] for r in order[:n_fail]}
    fail_cut_auc = order[n_fail - 1]["auc_full"]
    strict_dps = {r["dp"] for r in valid_recs if r["auc_full"] < STRICT_AUC}
    band_dps = {r["dp"] for r in valid_recs if r["auc_full"] < FAIL_AUC}
    base_rate = n_fail / n_valid

    # ---- strata (families a-e)
    ent_edges = np.percentile([r["entropy_norm"] for r in valid_recs],
                              [25, 50, 75])
    strata = []                       # (family, name, set of dps)

    def add(family, name, members):
        strata.append((family, name, {r["dp"] for r in members}))

    for qt in range(4):
        add("a_margin_quartile", f"Q{qt}",
            [r for r in valid_recs if r["quartile"] == qt])

    def size_class(r):
        n = r["n_opened"]
        return "1" if n == 1 else "2" if n == 2 else "3-4" if n <= 4 else "5+"

    for name in ["1", "2", "3-4", "5+"]:
        add("b_opened_size", name, [r for r in valid_recs
                                    if size_class(r) == name])

    def age_class(r):
        yf = r["young_frac_opened"]
        return ("YOUNG_HEAVY" if yf > 0.5
                else "NO_YOUNG" if yf == 0 else "MIXED")

    for name in ["YOUNG_HEAVY", "MIXED", "NO_YOUNG"]:
        add("c_opened_age_profile", name,
            [r for r in valid_recs if age_class(r) == name])

    def rescued(r):
        return r["auc_L12"] - r["auc_full"] > L1L2_RESCUE_DELTA

    for name, want in [("L1L2_RESCUED", True), ("NOT_RESCUED", False)]:
        add("d_layer_profile", name,
            [r for r in valid_recs if rescued(r) == want])

    for eh in range(4):
        add("e_attention_entropy", f"EH{eh}",
            [r for r in valid_recs
             if (0 if r["entropy_norm"] <= ent_edges[0]
                 else 1 if r["entropy_norm"] <= ent_edges[1]
                 else 2 if r["entropy_norm"] <= ent_edges[2] else 3) == eh])

    # ---- stratum table + firing test (>= 3x base, Wilson CI excludes base)
    strata_tbl = []
    firing = []
    for family, name, members in strata:
        n_s = len(members)
        k_s = len(members & fail_dps)
        rate = k_s / n_s if n_s else float("nan")
        lo, hi = wilson_ci(k_s, n_s)
        fires = bool(n_s > 0 and rate >= BASE_RATE_MULT * base_rate * (1 - 1e-12)
                     and lo > base_rate)
        if fires:
            firing.append((family, name, members & fail_dps))
        strata_tbl.append(dict(
            family=family, stratum=name, n=n_s, n_fail=k_s,
            fail_rate=rate, rate_vs_base=rate / base_rate if base_rate else
            float("nan"),
            wilson_ci=[lo, hi], fires=fires))

    # ---- permutation guard (registered): P(count of firing strata >= obs)
    S = np.asarray([[1 if r["dp"] in members else 0 for r in valid_recs]
                    for _, _, members in strata if len(members) > 0],
                   float)                          # (nonempty strata, n)
    n_s_vec = S.sum(1)
    x_true = np.asarray([1 if r["dp"] in fail_dps else 0
                         for r in valid_recs], float)
    obs_fire = sum(1 for t in strata_tbl if t["fires"])
    rng_p = np.random.default_rng(PERM_SEED)
    idx_pool = np.arange(n_valid)
    null_counts = np.empty(PERM_N, int)
    for pi in range(PERM_N):
        perm = rng_p.permutation(idx_pool)
        x = x_true[perm]
        k = S @ x
        rate = k / n_s_vec
        lo_ok = wilson_lo_vec(k, n_s_vec, base_rate)
        null_counts[pi] = int(((rate >= BASE_RATE_MULT * base_rate
                                * (1 - 1e-12)) & lo_ok).sum())
    perm_p = float((null_counts >= obs_fire).mean())
    perm_p_ge2 = float((null_counts >= 2).mean())
    null_hist = np.bincount(null_counts, minlength=max(int(null_counts.max())
                                                       + 1, 1)).tolist()

    localized = bool(obs_fire >= MIN_FIRING_STRATA and perm_p < 0.05)

    # ---- overlap among firing strata (honesty: same DPs or different?)
    overlap = []
    for ia in range(len(firing)):
        for ib in range(ia + 1, len(firing)):
            fa, fb = firing[ia][2], firing[ib][2]
            j = len(fa & fb) / len(fa | fb) if (fa | fb) else float("nan")
            overlap.append(dict(a=f"{firing[ia][0]}:{firing[ia][1]}",
                                b=f"{firing[ib][0]}:{firing[ib][1]}",
                                jaccard=j))

    # ---- mechanism test (registered): threshold noise vs label-stable
    bl_f = [r["blur_frac"][EPS_PRIMARY_KEY] for r in valid_recs
            if r["dp"] in fail_dps]
    bl_n = [r["blur_frac"][EPS_PRIMARY_KEY] for r in valid_recs
            if r["dp"] not in fail_dps]
    med_f, med_n = float(np.median(bl_f)), float(np.median(bl_n))
    recovered = [r for r in valid_recs if r["dp"] in fail_dps
                 and np.isfinite(r[f"auc_clean@{EPS_PRIMARY_KEY}"])
                 and r[f"auc_clean@{EPS_PRIMARY_KEY}"] >= BLUR_RECOVER_AUC]
    n_recovered = len(recovered)
    blur_confirmed = bool(med_f >= BLUR_FRAC_RATIO * med_n
                          and n_recovered >= BLUR_RECOVER_COUNT)

    # ---- strict-core anatomy (the literal <0.7 / <0.5 band members)
    anatomy = []
    for r in per_dp:
        if r["dp"] in band_dps or r["dp"] in strict_dps:
            entries, ps = entry_positions(r["q"], r["dp"])
            anatomy.append(dict(
                dp=r["dp"], b=r["b"], q=r["q"], quartile=r["quartile"],
                margin=r["margin"], auc_full=r.get("auc_full"),
                n_opened=r["n_opened"], entropy_norm=r["entropy_norm"],
                blur_frac=r.get("blur_frac", {}).get(EPS_PRIMARY_KEY),
                opened_ages=r["opened_ages"],
                opened_gaps=r.get("opened_gaps"),
                opened_ranks_full=r.get("opened_ranks_full"),
                opened_ranks_L12=r.get("auc_L12")))

    # ---- pooled readouts + headroom decomposition
    p_full = pooled_auc(pooled["full"])
    p_L0ex = pooled_auc(pooled["L0ex"])
    p_L12 = pooled_auc(pooled["L12"])
    p_clean = {EPS_KEY[eps]: pooled_auc(pooled["clean"][eps])
               for eps in EPS_GRID}
    p_L0ex_clean = pooled_auc(pooled["L0ex_clean"])
    ci_full = bootstrap_pooled(pooled["full"])
    ci_L0ex = bootstrap_pooled(pooled["L0ex"])
    ci_L12 = bootstrap_pooled(pooled["L12"])
    layer_tbl = {f"L{li}": [float(layer_pooled[li][0]),
                            int(layer_pooled[li][1])]
                 for li in range(4) if layer_pooled[li] is not None}
    headroom = dict(
        layer_choice_gain=p_L0ex - p_full,
        layer_L12_gain=p_L12 - p_full,
        threshold_gain=p_clean[EPS_PRIMARY_KEY] - p_full,
        combined_L0ex_clean_gain=p_L0ex_clean - p_full,
        clean_by_eps=p_clean)

    # ---- registered decision
    if localized:
        fires = "LOCALIZED_RESIDUE"
        names = [f"{t['family']}:{t['stratum']} ({t['n_fail']}/{t['n']} = "
                 f"{t['rate_vs_base']:.2f}x base, CI "
                 f"[{t['wilson_ci'][0]:.3f},{t['wilson_ci'][1]:.3f}])"
                 for t in strata_tbl if t["fires"]]
        verdict = (f"LOCALIZED RESIDUE: {obs_fire} strata fire at >= 3x base "
                   f"with Wilson CIs excluding base (permutation p={perm_p:.4f}): "
                   + "; ".join(names)
                   + f". Overlap note: pairwise Jaccards "
                   f"{[round(o['jaccard'], 2) for o in overlap]}.")
    else:
        fires = "MEASUREMENT_BLUR"
        verdict = (f"MEASUREMENT BLUR: only {obs_fire} stratum(a) fire "
                   f"(bar {MIN_FIRING_STRATA}; permutation p={perm_p:.4f}) — "
                   "failures scatter at base rates; the residual is "
                   "interventional-threshold noise in the ground truth; the "
                   "T054 unification stands clean at its resolution.")
    if blur_confirmed:
        verdict += (f" | Threshold-noise component CONFIRMED: failing-DP "
                    f"blur_frac median {med_f:.3f} >= 2x non-failing "
                    f"{med_n:.3f}; {n_recovered}/{n_fail} failing DPs rise "
                    f">= {BLUR_RECOVER_AUC} AUC with near-threshold labels "
                    "dropped.")
    else:
        verdict += (f" | Threshold-noise component NOT confirmed (blur med "
                    f"{med_f:.3f} vs {med_n:.3f}; {n_recovered}/{n_fail} "
                    f"recover >= {BLUR_RECOVER_AUC}) — failing cells are "
                    "LABEL-STABLE: genuine misalignment (second-channel "
                    "candidate).")
    if len(fail_dps) < 5:
        verdict += (f" [POWER CAVEAT: powered band holds n={n_fail} "
                    "failures.]")
    log(f"REGISTERED DECISION [{fires}]: {verdict}")

    # ---------------------------------------------------------------- metrics
    metrics = dict(
        experiment="e104_failing_dp_census",
        purpose="FAILING-DP CENSUS (T054's registered residual, QUEUE e104): "
                "rebuild e100's features and per-DP AUCs bit-identically "
                "(G5 gates vs e084 flips_vz and e100 stored AUCs), then "
                "stratify failing decision points — PRIMARY POWERED BAND = "
                "bottom-decile per-DP attention-mass AUC (the literal <0.7 / "
                "<0.5 bands hold 1/98 DPs in e100's stored table; POWER GATE "
                "registered) — by (a) margin quartile, (b) opened-set size, "
                "(c) opened-age profile, (d) layer profile (L1/L2-only "
                "recompute), (e) candidate-attention entropy. BARS: >=2 "
                "strata at >=3x base with Wilson CI excluding base AND "
                "permutation p<0.05 => LOCALIZED RESIDUE; else MEASUREMENT "
                "BLUR. Mechanism: per-entry flip gaps from the same V-zero "
                "call => blur_frac + clean-label AUC recovery. Secondary: "
                "pooled AUC L0-excluded / L1L2-only / per-layer / "
                "clean-label + adaptive-k precision = layer-choice vs "
                "threshold headroom.",
        started=started, wall_s=elapsed(), threads=THREADS, cpu_only=True,
        smoke=SMOKE,
        net=dict(ckpt=str(CKPT), arch=dict(n_layer=4, n_head=4, n_embd=128,
                                           block_size=T_TOTAL, vocab=65),
                 params=n_params, val_ce=val_ce,
                 val_ce_e053c=E053C_VAL_CE),
        seeds=dict(corpus=1337,
                   battery_A=dict(prompts=SEED_PROMPT, sampling=SEED_SAMPLE),
                   battery_X=dict(prompts=SEED_PROMPT_X,
                                  sampling=SEED_SAMPLE_X),
                   stratification=SEED_STRAT, old_ages=OLD_SEED_BASE,
                   bootstrap=0, permutation=PERM_SEED),
        protocol=dict(
            n_decision_points=len(dps), per_quartile=PER_QUARTILE,
            q_range=[Q_LO, Q_HI], n_entries=N_ENTRIES,
            failure_bands=dict(powered_decile=dict(
                                   frac=DECILE_FRAC, n_fail=n_fail,
                                   cut_auc=fail_cut_auc),
                               literal_lt_0p7=sorted(band_dps),
                               strict_lt_0p5=sorted(strict_dps),
                               power_gate="literal bands hold 1/98 in "
                                          "e100's stored table — clustering "
                                          "bars run on the powered band"),
            ground_truth="V-zero flip at the e084 bars + per-entry flip gap "
                         "g_r = max_{c!=t1} lg_int[c] - lg_int[t1] from the "
                         "same call; near-threshold iff |g_r| < eps",
            eps_grid=EPS_GRID, eps_primary=EPS_PRIMARY,
            entropy_quartile_edges=ent_edges.tolist(),
            n_valid_dp=n_valid, n_zero_opened_dp=n_zero_open),
        gates=gates,
        ground_truth=dict(
            opened_per_dp_median=float(np.median(opened_sizes)),
            total_opened_cells=int(sum(opened_sizes)),
            n_valid_dp=n_valid, n_zero_opened_dp=n_zero_open,
            pooled_full_auc=p_full, e100_reference=0.9069704732371578),
        auc=dict(
            pooled=dict(full=p_full, L0_excluded=p_L0ex, L1L2_only=p_L12,
                        clean=p_clean, L0ex_clean=p_L0ex_clean),
            ci=dict(full=list(ci_full), L0_excluded=list(ci_L0ex),
                    L1L2_only=list(ci_L12)),
            per_layer=layer_tbl,
            headroom_decomposition=headroom),
        adaptive_k=dict(
            precision={k: dict(mean=float(np.mean(v)), median=float(np.median(
                v))) for k, v in pk_adapt.items() if v},
            jaccard={k: dict(mean=float(np.mean(v))) for k, v in
                     jac_adapt.items() if v}),
        census=dict(
            base_rate=base_rate, n_fail=n_fail, fail_dps=sorted(fail_dps),
            fail_cut_auc=fail_cut_auc,
            strict_core_dps=sorted(strict_dps),
            strata=strata_tbl,
            n_firing_strata=obs_fire, permutation_p=perm_p,
            firing_overlap=overlap,
            mechanism=dict(
                blur_frac_failing_median=med_f,
                blur_frac_nonfailing_median=med_n,
                n_failing_recovered_above_085=n_recovered,
                blur_confirmed=blur_confirmed,
                label_stable=bool(not blur_confirmed)),
            strict_core_anatomy=anatomy),
        per_decision_point=per_dp,
        registered_decision=dict(
            frozen_bars=dict(
                localized=f">= {MIN_FIRING_STRATA} strata at >= "
                          f"{BASE_RATE_MULT:.0f}x base, Wilson CI excluding "
                          "base, permutation p < 0.05 (powered decile band)",
                measurement_blur="else (scatter at base rates)",
                mechanism="blur: failing median blur_frac >= 2x non-failing "
                          f"AND >= {BLUR_RECOVER_COUNT}/{n_fail} failing DPs "
                          f">= {BLUR_RECOVER_AUC} AUC after dropping "
                          "near-threshold labels; else LABEL-STABLE"),
            n_firing_strata=obs_fire, permutation_p=perm_p,
            fires=fires, verdict=verdict),
    )
    save_json(out_dir / f"metrics{suffix}.json", metrics)
    log(f"metrics{suffix}.json written")
    plot(out_dir / f"failing_dp_census{suffix}.png", metrics)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot

def plot(path: Path, M: dict):
    cen = M["census"]
    dec = M["registered_decision"]
    auc = M["auc"]
    base = cen["base_rate"]

    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # ---- panel 1: failure-rate bars per stratum + base / 3x lines
    tbl = cen["strata"]
    xs = np.arange(len(tbl))
    rates = [t["fail_rate"] for t in tbl]
    errs = np.asarray([[max(t["fail_rate"] - t["wilson_ci"][0], 0),
                        max(t["wilson_ci"][1] - t["fail_rate"], 0)]
                       for t in tbl]).T
    cols = ["tab:red" if t["fires"] else "steelblue" for t in tbl]
    ax1.bar(xs, rates, 0.72, yerr=errs, color=cols, alpha=0.85, capsize=3)
    ax1.axhline(base, color="k", ls="--", lw=1.2)
    ax1.axhline(3 * base, color="tab:red", ls=":", lw=1.4)
    ax1.text(len(tbl) - 0.4, base + 0.004, f"base {base:.3f}", fontsize=8,
             ha="right")
    ax1.text(len(tbl) - 0.4, 3 * base + 0.004, f"3x base {3 * base:.3f}",
             fontsize=8, ha="right", color="tab:red")
    ax1.set_xticks(xs, [f"{t['family'].split('_')[0]}:{t['stratum']}\n"
                        f"({t['n_fail']}/{t['n']})" for t in tbl],
                   fontsize=6.5, rotation=90)
    ax1.set_ylabel("failure rate (powered decile band)")
    ax1.set_title("(a) failure rate per stratum (red = fires: >=3x base, CI "
                  "excludes base; Wilson 95% CIs)", fontsize=9.5)

    # ---- panel 2: per-DP AUC sorted, bands marked
    recs = sorted([r for r in M["per_decision_point"] if "auc_full" in r],
                  key=lambda r: r["auc_full"])
    aucs = [r["auc_full"] for r in recs]
    fail_set = set(cen["fail_dps"])
    cols2 = ["tab:red" if r["dp"] in fail_set else "steelblue" for r in recs]
    ax2.scatter(np.arange(len(aucs)), aucs, s=22, c=cols2)
    ax2.axhline(M["ground_truth"]["pooled_full_auc"], color="k", ls="--",
                lw=1.1)
    ax2.axhline(FAIL_AUC, color="tab:red", ls=":", lw=1.1)
    ax2.axhline(STRICT_AUC, color="darkred", ls=":", lw=1.1)
    for lbl, y, cc in [("pooled 0.907",
                        M["ground_truth"]["pooled_full_auc"], "k"),
                       ("0.7 literal band", FAIL_AUC, "tab:red"),
                       ("0.5 strict", STRICT_AUC, "darkred")]:
        ax2.text(len(aucs) - 1, y + 0.012, lbl, fontsize=7.5, ha="right",
                 color=cc)
    ax2.set_xlabel("decision points (sorted by AUC)")
    ax2.set_ylabel("per-DP attention-mass AUC")
    ax2.set_ylim(0.35, 1.02)
    ax2.set_title(f"(b) per-DP AUC distribution — red = powered failure band "
                  f"(n={cen['n_fail']})", fontsize=9.5)

    # ---- panel 3: blur — auc vs blur_frac, with clean-label recovery
    for r in M["per_decision_point"]:
        if "auc_full" not in r:
            continue
        f = r["dp"] in fail_set
        ax3.scatter(r["blur_frac"][EPS_PRIMARY_KEY], r["auc_full"],
                    s=30 if f else 16, color="tab:red" if f else "steelblue",
                    alpha=0.85 if f else 0.5, zorder=3 if f else 2)
        if f and np.isfinite(r[f"auc_clean@{EPS_PRIMARY_KEY}"]):
            ax3.annotate("", xy=(r["blur_frac"][EPS_PRIMARY_KEY],
                                 r[f"auc_clean@{EPS_PRIMARY_KEY}"]),
                         xytext=(r["blur_frac"][EPS_PRIMARY_KEY],
                                 r["auc_full"]),
                         arrowprops=dict(arrowstyle="->", lw=1.0,
                                         color="tab:red", alpha=0.8))
    ax3.set_xlabel(f"blur_frac (|flip gap| < {EPS_PRIMARY} logits)")
    ax3.set_ylabel("per-DP AUC (arrow: after dropping near-threshold labels)")
    ax3.set_title("(c) threshold-blur mechanism — red = failing DPs, arrows "
                  "= clean-label recovery", fontsize=9.5)

    # ---- panel 4: pooled AUC variants (headroom decomposition)
    names = ["full 16\n(e100 0.907)", "L0 excl\n(12 lh)", "L1+L2\n(8 lh)",
             "clean eps\n0.05", "clean eps\n0.10", "clean eps\n0.20",
             "L0ex +\nclean 0.10"]
    vals = [auc["pooled"]["full"], auc["pooled"]["L0_excluded"],
            auc["pooled"]["L1L2_only"], auc["pooled"]["clean"]["0.05"],
            auc["pooled"]["clean"]["0.10"], auc["pooled"]["clean"]["0.20"],
            auc["pooled"]["L0ex_clean"]]
    cis = [auc["ci"]["full"], auc["ci"]["L0_excluded"], auc["ci"]["L1L2_only"],
           None, None, None, None]
    lo_err = [max(v - c[0], 0) if c else 0 for v, c in zip(vals, cis)]
    hi_err = [max(c[1] - v, 0) if c else 0 for v, c in zip(vals, cis)]
    cols4 = ["steelblue", "tab:green", "tab:green", "tab:orange",
             "tab:orange", "tab:orange", "tab:purple"]
    ax4.bar(np.arange(len(vals)), [v - 0.5 for v in vals], 0.6,
            bottom=0.5, yerr=np.asarray([lo_err, hi_err]), color=cols4,
            alpha=0.85, capsize=4)
    ax4.axhline(1.0, color="k", ls=":", lw=1.0)
    ax4.axhline(auc["pooled"]["full"], color="steelblue", ls="--", lw=1.0)
    ax4.set_xticks(np.arange(len(vals)), names, fontsize=7.5)
    ax4.set_ylim(0.5, 1.01)
    ax4.set_ylabel("pooled within-decision AUC")
    hd = auc["headroom_decomposition"]
    ax4.set_title("(d) headroom: layer choice "
                  f"(+{hd['layer_choice_gain']:.3f} L0ex, "
                  f"+{hd['layer_L12_gain']:.3f} L12) vs threshold "
                  f"(+{hd['threshold_gain']:.3f} clean@0.10)", fontsize=9.5)

    # ---- panel 5: strict-core anatomy
    ana = cen["strict_core_anatomy"]
    if ana:
        a0 = ana[0]
        e084 = json.loads(E084_METRICS.read_text(encoding="utf-8"))
        ref = {p["dp"]: p for p in e084["per_decision_point"]}[a0["dp"]]
        labels = [f"dp {a0['dp']}: Q{a0['quartile']} margin "
                  f"{a0['margin']:.3f}\nopened {a0['n_opened']} at ages "
                  f"{a0['opened_ages']}\nflip gap(s) "
                  f"{[round(g, 3) for g in a0['opened_gaps']]}\n"
                  f"attention rank(s) {a0['opened_ranks_full']} of 80\n"
                  f"e084 same DP: vz {ref['flips_vz']} / kd "
                  f"{ref['flips_kd']} / vs {ref['flips_vs']} flips\n"
                  f"blur_frac {a0['blur_frac']:.3f} | AUC "
                  f"{a0['auc_full']:.3f} -> clean "
                  f"(strict-core member anatomy)"]
        ax5.axis("off")
        ax5.text(0.02, 0.95, labels[0], fontsize=10, va="top",
                 family="monospace",
                 bbox=dict(fc="lightyellow", ec="dimgray"))
    ax5.set_title("(e) strict-core (<0.7 / <0.5 band) anatomy", fontsize=9.5)

    # ---- panel 6: permutation null
    # (recomputed null is not stored in metrics; show firing table instead)
    rows = [t for t in cen["strata"] if t["n_fail"] > 0]
    rows.sort(key=lambda t: -t["fail_rate"])
    ax6.axis("off")
    txt = (f"firing strata: {cen['n_firing_strata']} "
           f"(bar >= {MIN_FIRING_STRATA})\n"
           f"permutation p = {cen['permutation_p']:.4f}\n\n"
           "top strata by failure rate:\n")
    for t in rows[:8]:
        txt += (f"  {t['family']}:{t['stratum']:12s} "
                f"{t['n_fail']:2d}/{t['n']:2d} = {t['fail_rate']:.3f} "
                f"({t['rate_vs_base']:.1f}x base) "
                f"{'FIRE' if t['fires'] else ''}\n")
    txt += "\nmechanism: " + ("threshold-blur CONFIRMED"
                              if cen["mechanism"]["blur_confirmed"]
                              else "LABEL-STABLE (blur not confirmed)")
    ax6.text(0.02, 0.95, txt, fontsize=9, va="top", family="monospace",
             bbox=dict(fc="white", ec="dimgray"))

    fig.suptitle(f"E104 FAILING-DP CENSUS | {dec['fires']} | base "
                 f"{base:.3f} | {dec['n_firing_strata']} firing strata | "
                 f"perm p {cen['permutation_p']:.3f} | "
                 + dec["verdict"][:220], fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
