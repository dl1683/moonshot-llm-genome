"""E100 — the READ-PREDICTION probe: can the query state predict WHICH
coordinates the read policy opens?

Rule-11 pick: T037's unified question ("what rule decides which stored
coordinate is opened?") made operational on top of T050's sparse-open fact
(the e084 census: per-decision opened-coordinate medians are 3-4 of 80 — the
rule opens a handful of coordinates per decision, not a dense average).

THE DEEP QUESTION: the read policy opens 3-4 coordinates per decision out of
80 candidates — CAN THE QUERY STATE PREDICT WHICH ONES? If yes, we have found
the addressing function (the rule mapping "what the net is trying to do" to
"which stored coordinates it opens").

DESIGN (registered BEFORE any compute; everything frozen):
  - Battery + decision points + entry bands: EXACTLY e084's (protocol
    identity by reconstruction): same net (runs/checkpoints/e053c_ctx512.pt,
    4L/4H/128, ctx 512), same battery (e069's 4 seqs seeds 202/7 + e072
    family's 4 fresh seeds 302/17, free run 64->512, temp 0.8 top-k 40,
    fixed anchor, per-row shared-generator draws), same margin-stratified
    100 decision points (q in [96,510], quartile cuts of the pooled 16x415
    candidate margins, 25 per quartile, default_rng seed 2084, drawn in
    Q1..Q4 order), same 80-entry band per decision point (young ages 1-10 +
    mid 11-60 + 20 old sampled without replacement from [61, q+1],
    default_rng seed 9000+dp_index).
  - GROUND TRUTH opened set (after e084 verbatim): V-zero flip at the e084
    bars — v := 0 at that position, every layer, every head, all queries
    (e069/e084 whole-position instrument, same manual_logits2 call family);
    entry is OPENED iff the decision argmax flips. (e084 measured the
    median opened set = 3 of 80 for V-zero.)
  - QUERY-SIDE FEATURES (computed BEFORE any intervention, from ONE clean
    forward of the same context; per candidate entry at position p = T-age):
      (a) ATTENTION MASS — the query's (position q, the decision row)
          pre-decoder attention: sum over all 16 layer-heads of the softmax
          attention probability from row q to position p. Cheap hypothesis:
          opened coordinates ARE the attended ones.
      (b) K-SIDE ALIGNMENT — mean over the 16 layer-heads of the cosine
          between the query vector (row q) and the key vector at p.
    Secondary (descriptive, no bars): per-layer-head attention-mass AUCs
    (16 + 4 layer-pooled), max-over-heads cosine, raw scaled dot-product
    q.k/sqrt(d) (the actual pre-softmax addressing quantity), and a
    rank-sum combo of (a)+(b).
  - READOUTS: per decision point, rank the 80 candidates by each feature;
    precision@k (k = 1..8, 10 and adaptive k = |opened|) for membership in
    the opened set; AUC per decision point (tie-averaged Mann-Whitney) and
    the POOLED within-decision AUC (pair-weighted across decision points —
    score scales are per-decision objects, so pairs are only compared within
    a decision). Bootstrap CIs (n=1000, decision-point resampling, seed 0).

REGISTERED BARS (frozen; evaluated in this order on the pooled
attention-mass AUC; the 0.6 boundary is resolved by order — evaluated
second, so exactly 0.6 falls to PARTIAL):
  1. pooled attention-mass AUC >= 0.85  => THE READ IS ATTENTION-ADDRESSED:
     opened = attended; the policy's rule is the routing we already measure
     — a unification.
  2. pooled attention-mass AUC < 0.6    => DISSOCIATED: the opened
     coordinates are NOT the attended ones — the read policy is a genuinely
     hidden rule; register the follow-up (what query-side state, if any,
     does predict it?).
  3. else (0.6 <= AUC < 0.85)           => PARTIAL: attention proposes,
     something else disposes — the residual is the interesting object.
The q-k-cosine AUC is reported against the same bands descriptively (no
independent registered bar).

INSTRUMENT GUARD (not a kill): if fewer than 75 of 100 decision points have
a non-degenerate opened set (1..79 opened), the probe is flagged
UNDERPOWERED and bars are reported with that caveat.

GATES: G2 params (873,472), G1 val CE (vs e053c 1.5227 +/- 0.02), G0b
incremental-KV vs full recompute (max prob dev < 1e-4), G8 feature-forward
identity (its last-position logits vs the manual_logits2 clean row, max
|dev| < 1e-4, first decision point), G5 e084-identity (full run only): this
rig's per-decision V-zero opened-set sizes vs e084's stored flips_vz —
exact match required on >= 90/100 (residual mismatches would be batch-shape
ulp noise on argmax near-ties; each is reported).

Run:     python lab/e100_read_prediction.py     (E100_SMOKE=1 for smoke)
Outputs: runs/e100/metrics.json + runs/e100/read_prediction.png
Envelope: NO training; CPU-only (CUDA_VISIBLE_DEVICES forced -1, torch.cuda
stubbed pre-import, e070/e084 pattern); 8 torch threads (the e084-measured
sweet spot on this contended box); single step, ~15-25 min. No
NOTES/THINKING/QUEUE/STATE edits; no commit (the dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu" (e053b/e069/e070/
# e084 pattern)
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
SEED_PROMPT, SEED_SAMPLE = 202, 7         # e053b/e053c/e069/e070/e072/e084
N_PROMPTS = 8
B = 4                                     # battery A (e069 verbatim)
B_X = 4                                   # fresh extras (e072 family, e084)
SEED_PROMPT_X, SEED_SAMPLE_X = 302, 17    # 202/7 family +100/+10
BOOT_N = 1000
THREADS = 8      # e084's measured sweet spot on this contended box
torch.set_num_threads(THREADS)

CHUNK_512 = 64                            # e069/e070/e072/e084 measured chunk

# ---- design constants (registered; ALL verbatim e084) ----------------------
Q_LO, Q_HI = 96, 510                      # candidate query positions
PER_QUARTILE = 25                         # 25 per margin quartile
N_DP = 4 * PER_QUARTILE                   # 100 decision points
SEED_STRAT = 2084                         # margin-stratified sampling seed
YOUNG = list(range(1, 11))                # ages 1-10 (all)
MID = list(range(11, 61))                 # ages 11-60 (all)
N_OLD = 20                                # old entries per decision point
OLD_SEED_BASE = 9000                      # old-age rng seed = base + dp_index
N_ENTRIES = len(YOUNG) + len(MID) + N_OLD  # 80

# ---- readout constants (registered) -----------------------------------------
K_LIST = [1, 2, 3, 4, 5, 6, 8, 10]        # precision@k grid
MIN_VALID_DP = 75                         # instrument guard (see docstring)

# ---- REGISTERED bars (FROZEN, docstring; evaluated in order) ----------------
BAR_ADDRESSED = 0.85                      # pooled attention-mass AUC >=
BAR_DISSOC = 0.6                          # pooled attention-mass AUC <  (else
                                          # PARTIAL; 0.6 itself -> PARTIAL by
                                          # evaluation order)

# ---- e084 stored reference (G5 identity gate) -------------------------------
E084_METRICS = REPO / "runs" / "e084" / "metrics.json"
CKPT = REPO / "runs" / "checkpoints" / "e053c_ctx512.pt"
E053C_VAL_CE = 1.5226792494455974

SMOKE = os.environ.get("E100_SMOKE", "") == "1"
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


# --------------------------------------------- machinery (VERBATIM e069/e084)

@torch.no_grad()
def _manual_chunk2(net: TinyGPT, idxs, vzero, kdrop, layers, pos_offset=0,
                   vswap_pos=None, donor_stack=None):
    """e069/e084's _manual_chunk VERBATIM (vswap/kdrop arms inert here — e100
    uses only the clean row and V-zero masks, but the instrument is kept
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
    [VERBATIM e053b/e084]"""
    lg = logits_clean / TEMP
    v, _ = torch.topk(lg, TOPK)
    lg_f = lg.masked_fill(lg < v[-1], float("-inf"))
    tok = int(torch.multinomial(torch.softmax(lg_f, -1), 1, generator=gen))
    p_full = torch.softmax(logits_clean, -1)
    return tok, float(-math.log(max(p_full[tok].item(), 1e-12)))


@torch.no_grad()
def prefill_batch(net: TinyGPT, idx: torch.Tensor):
    """Batched prefill, (B, Tp) -> last-position logits (B, V) + KV cache.
    [VERBATIM e069/e084]"""
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
    [VERBATIM e069/e084]"""
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
    """e069/e084's generate_batch VERBATIM + per-step decision logits:
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


# --------------------------------------------- e100's own (gated) instruments

@torch.no_grad()
def query_features(net: TinyGPT, ctx: torch.Tensor):
    """ONE clean forward over ctx (1-D token ids, length T); record, per
    layer, the QUERY-SIDE state BEFORE any intervention:
      - att_last (L, H, T): softmax attention probabilities of the decision
        row (last position) over all positions;
      - q_last  (L, H, d):  the query vector of the decision row;
      - k_all   (L, H, T, d): the key vectors of all positions.
    Op order mirrors _manual_chunk2 (G8-gated against it). Also returns the
    decision-row logits (for G8)."""
    T = ctx.shape[0]
    x = net.wte(ctx[None]) + net.wpe(torch.arange(T))[None]
    H = net.cfg.n_head
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    att_last, q_last, k_all = [], [], []
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
        q_last.append(q[0, :, -1, :].clone())           # (H, d)
        k_all.append(k[0].clone())                      # (H, T, d)
    logits = net.lm_head(net.ln_f(x[0, -1]))
    return (torch.stack(att_last, 0), torch.stack(q_last, 0),
            torch.stack(k_all, 0), logits)


def entry_positions(q: int, dp_index: int):
    """e084's entry band VERBATIM: young 1-10 + mid 11-60 + 20 old sampled
    without replacement uniformly from [61, q+1] (default_rng seed
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
    """GROUND TRUTH at the e084 bars: one batched manual_logits2 call,
    1 clean row + 80 V-zero rows (v := 0 at that position, every layer,
    every head, all queries — e069/e084 whole-position instrument verbatim).
    Returns (opened flags (80,) bool, t1, margin)."""
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
    return opened, t1, margin


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
    """per_dp: list of (auc, n_pairs) -> pair-weighted pooled AUC."""
    tot = sum(p for _, p in per_dp)
    if tot == 0:
        return float("nan")
    return sum(a * p for a, p in per_dp) / tot


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


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or x[m].std() == 0 or y[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


# ---------------------------------------------------------------------- main

def main():
    assert torch.cuda.device_count() == 0, "CPU-only violated: CUDA visible"
    assert common.DEVICE == "cpu", f"common.DEVICE={common.DEVICE} != cpu"
    out_dir = run_dir("e100")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=SMOKE)
    suffix = "_smoke" if SMOKE else ""

    # ---- battery: rebuild corpus + net EXACTLY as e053c/e069/e070/e084 did
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

    # ---- the probe: per decision point, features BEFORE intervention,
    #      then the ground-truth opened set (V-zero flips, e084 bars)
    t_probe = time.time()
    per_dp = []        # records for metrics
    auc_att_l, auc_cos_l = [], []          # per-dp (auc, pairs)
    auc_combo_l, auc_logit_l, auc_maxcos_l = [], [], []
    auc_head = [[None] * 4 for _ in range(4)]   # per (layer,head) [(auc,pairs)]
    auc_layer = [None] * 4                       # per layer (4-head-pooled mass)
    pk_att = {k: [] for k in K_LIST}
    pk_cos = {k: [] for k in K_LIST}
    pk_adaptive_att, pk_adaptive_cos = [], []
    opened_sizes = []
    opened_ranks_att, opened_ranks_cos = [], []
    RECALL_KS = [1, 2, 3, 5, 10, 20, 40, 80]     # recall@k grid (panel f)

    for i, (b, q, _, qt) in enumerate(dps):
        entries, ps = entry_positions(q, i)

        # (1) query-side features from ONE clean forward (pre-intervention)
        att_last, q_last, k_all, q_logits = query_features(net, idx16[b, :q + 1])
        L, H, T = att_last.shape
        d = q_last.shape[-1]
        # per-(layer,head) cosines (L,H,T) and raw scaled dots (L,H,T)
        cos_lh = torch.zeros(L, H, T)
        dot_lh = torch.zeros(L, H, T)
        for li in range(L):
            dots = torch.einsum("hd,htd->ht", q_last[li], k_all[li])
            dot_lh[li] = dots / math.sqrt(d)
            qn = q_last[li].norm(dim=-1)                 # (H,)
            kn = k_all[li].norm(dim=-1)                  # (H,T)
            cos_lh[li] = dots / (qn[:, None] * kn)
        att_mass_full = att_last.sum(0).sum(0).numpy()          # (T,) 16-lh sum
        cos_full = cos_lh.mean(0).mean(0).numpy()               # (T,) 16-lh mean
        logit_full = dot_lh.mean(0).mean(0).numpy()             # (T,) mean dot
        maxcos_full = cos_lh.amax(dim=(0, 1)).numpy()           # (T,) max cos
        att_lh_full = att_last.numpy()                          # (L,H,T)
        att_layer_full = att_last.sum(1).numpy()                # (L,T)

        # (2) ground truth: V-zero flips at the e084 bars
        opened, t1, margin = opened_set(net, idx16, b, q, entries, ps)
        opened_sizes.append(int(opened.sum()))

        # (3) readouts
        ps_np = np.asarray(ps)
        s_att = att_mass_full[ps_np]
        s_cos = cos_full[ps_np]
        s_logit = logit_full[ps_np]
        s_maxcos = maxcos_full[ps_np]
        n_open = int(opened.sum())
        rec = dict(dp=i, b=int(b), q=int(q), quartile=int(qt), margin=margin,
                   t1=t1, n_opened=n_open,
                   opened_ages=[int(a) for a, o in zip(entries, opened) if o])
        valid = 1 <= n_open <= N_ENTRIES - 1
        if valid:
            a_att, p_att = auc_pairs(s_att, opened)
            a_cos, p_cos = auc_pairs(s_cos, opened)
            a_combo, _ = auc_pairs(rankdata_avg(s_att) + rankdata_avg(s_cos),
                                   opened)
            a_logit, _ = auc_pairs(s_logit, opened)
            a_maxcos, _ = auc_pairs(s_maxcos, opened)
            auc_att_l.append((a_att, p_att))
            auc_cos_l.append((a_cos, p_cos))
            auc_combo_l.append((a_combo, p_att))
            auc_logit_l.append((a_logit, p_att))
            auc_maxcos_l.append((a_maxcos, p_att))
            rec["auc_att"], rec["auc_cos"] = a_att, a_cos
            # secondary per-layer-head / per-layer pooled mass
            for li in range(L):
                for hi in range(H):
                    a, p = auc_pairs(att_lh_full[li, hi][ps_np], opened)
                    auc_head[li][hi] = merge_pair(auc_head[li][hi], a, p)
                a, p = auc_pairs(att_layer_full[li][ps_np], opened)
                auc_layer[li] = merge_pair(auc_layer[li], a, p)
            # precision@k (adaptive k = n_opened included)
            pk_a = precision_at_k(s_att, opened, K_LIST + [n_open])
            pk_c = precision_at_k(s_cos, opened, K_LIST + [n_open])
            for k in K_LIST:
                pk_att[k].append(pk_a[k])
                pk_cos[k].append(pk_c[k])
            pk_adaptive_att.append(pk_a[n_open])
            pk_adaptive_cos.append(pk_c[n_open])
            ra = rankdata_avg(-s_att)         # rank 1 = feature's top pick
            rc = rankdata_avg(-s_cos)
            opened_ranks_att += [float(ra[j]) for j in np.where(opened)[0]]
            opened_ranks_cos += [float(rc[j]) for j in np.where(opened)[0]]
        per_dp.append(rec)
        if (i + 1) % 10 == 0 or i == len(dps) - 1:
            rate = (time.time() - t_probe) / (i + 1)
            log(f"probe {i + 1}/{len(dps)} decision points ({rate:.2f}s/dp, "
                f"ETA {rate * (len(dps) - i - 1):.0f}s)")

    # ---- G8: feature-forward identity (first DP) vs manual_logits2 clean row
    b0, q0 = dps[0][0], dps[0][1]
    _, _, _, q_log0 = query_features(net, idx16[b0, :q0 + 1])
    ref0 = manual_logits2(net, idx16[b0, :q0 + 1][None], None, None, None,
                          chunk=1)[0]
    g8_dev = float((q_log0 - ref0).abs().max())
    gates["G8_feature_forward_identity"] = dict(
        dp=0, max_logit_dev=g8_dev, tol=1e-4, ok=bool(g8_dev < 1e-4))
    log(f"G8 feature-forward vs manual clean row: max |logit dev| "
        f"{g8_dev:.2e} -> {'PASS' if g8_dev < 1e-4 else 'FAIL'}")

    # ---- G5: e084 identity — per-dp V-zero opened-set sizes vs stored
    if SMOKE:
        gates["G5_e084_identity"] = dict(
            skipped=True, reason="smoke battery differs from e084's")
        log("G5 e084 identity: SKIPPED (smoke)")
    else:
        e084 = json.loads(E084_METRICS.read_text(encoding="utf-8"))
        ref_counts = [p["flips_vz"] for p in e084["per_decision_point"]]
        mine = [r["n_opened"] for r in per_dp]
        assert len(ref_counts) == len(mine), "e084 per-DP count mismatch"
        matches = [i for i, (a, b_) in enumerate(zip(mine, ref_counts))
                   if a == b_]
        mism = [(i, mine[i], ref_counts[i]) for i in range(len(mine))
                if mine[i] != ref_counts[i]]
        g5_ok = len(matches) >= 90
        gates["G5_e084_identity"] = dict(
            n_match=len(matches), n_dp=len(mine), bar=90,
            mismatches=mism[:10], ok=bool(g5_ok),
            count_pearson=pearson(mine, ref_counts))
        log(f"G5 vs e084 stored flips_vz: {len(matches)}/{len(mine)} exact "
            f"matches (bar 90) -> {'PASS' if g5_ok else 'FAIL'}"
            + (f"; first mismatches (dp, mine, e084): {mism[:5]}"
               if mism else ""))

    # ---- pooled readouts
    pooled_att = pooled_auc(auc_att_l)
    pooled_cos = pooled_auc(auc_cos_l)
    pooled_combo = pooled_auc(auc_combo_l)
    pooled_logit = pooled_auc(auc_logit_l)
    pooled_maxcos = pooled_auc(auc_maxcos_l)
    ci_att = bootstrap_pooled(auc_att_l)
    ci_cos = bootstrap_pooled(auc_cos_l)
    ci_combo = bootstrap_pooled(auc_combo_l)
    macro_att = float(np.nanmean([a for a, _ in auc_att_l]))
    macro_cos = float(np.nanmean([a for a, _ in auc_cos_l]))
    n_valid = len(auc_att_l)
    n_zero_open = int(sum(1 for s in opened_sizes if s == 0))
    n_all_open = int(sum(1 for s in opened_sizes if s >= N_ENTRIES))
    total_opened = int(sum(opened_sizes))
    log(f"READOUT: valid DPs {n_valid}/{len(dps)} (0-opened {n_zero_open}); "
        f"total opened cells {total_opened}; median opened/DP "
        f"{np.median(opened_sizes):.0f}")
    log(f"AUC pooled: attention-mass {pooled_att:.4f} CI "
        f"[{ci_att[0]:.3f},{ci_att[1]:.3f}] | q-k cosine {pooled_cos:.4f} CI "
        f"[{ci_cos[0]:.3f},{ci_cos[1]:.3f}] | combo {pooled_combo:.4f} | "
        f"macro {macro_att:.3f}/{macro_cos:.3f}")

    # per-head / per-layer secondary table
    head_tbl = {f"L{li}H{hi}": pooled_auc_entry(auc_head[li][hi])
                for li in range(4) for hi in range(4)}
    layer_tbl = {f"L{li}": pooled_auc_entry(auc_layer[li]) for li in range(4)}
    best_head = max(head_tbl, key=lambda k: head_tbl[k][0])

    # precision@k summaries (valid DPs)
    chance = total_opened / (len(dps) * N_ENTRIES)
    chance_valid = float(np.mean(
        [r["n_opened"] for r in per_dp if 1 <= r["n_opened"] <= N_ENTRIES - 1])
        / N_ENTRIES)
    pk_summary = {}
    for feat, store, adapt in [("attention_mass", pk_att, pk_adaptive_att),
                               ("qk_cosine", pk_cos, pk_adaptive_cos)]:
        pk_summary[feat] = {
            f"p@{k}": dict(mean=float(np.mean(store[k])),
                           median=float(np.median(store[k])),
                           q1=float(np.percentile(store[k], 25)),
                           q3=float(np.percentile(store[k], 75)),
                           n=len(store[k]))
            for k in K_LIST}
        pk_summary[feat]["p@adaptive"] = dict(
            mean=float(np.mean(adapt)), median=float(np.median(adapt)),
            q1=float(np.percentile(adapt, 25)),
            q3=float(np.percentile(adapt, 75)), n=len(adapt),
            note="k = |opened set| per decision point")

    # rank-position of opened entries (feature ranking; 1 = top pick)
    ranks_att_np = np.asarray(opened_ranks_att)
    ranks_cos_np = np.asarray(opened_ranks_cos)
    recall_att = {k: float((ranks_att_np <= k).mean()) if len(ranks_att_np)
                  else float("nan") for k in RECALL_KS}
    recall_cos = {k: float((ranks_cos_np <= k).mean()) if len(ranks_cos_np)
                  else float("nan") for k in RECALL_KS}

    # ---- REGISTERED decision (frozen bars, in order; see docstring)
    underpowered = n_valid < MIN_VALID_DP
    if pooled_att >= BAR_ADDRESSED:
        fires = "ATTENTION_ADDRESSED"
        verdict = (f"THE READ IS ATTENTION-ADDRESSED: pooled attention-mass "
                   f"AUC {pooled_att:.3f} >= {BAR_ADDRESSED} — opened = "
                   "attended; the policy's rule is the routing we already "
                   "measure (a unification).")
    elif pooled_att < BAR_DISSOC:
        fires = "DISSOCIATED"
        verdict = (f"DISSOCIATED: pooled attention-mass AUC "
                   f"{pooled_att:.3f} < {BAR_DISSOC} — the opened "
                   "coordinates are NOT the attended ones; the read policy "
                   "is a genuinely hidden rule (register the follow-up: what "
                   "query-side state, if any, predicts the opened set?).")
    else:
        fires = "PARTIAL"
        verdict = (f"PARTIAL: pooled attention-mass AUC {pooled_att:.3f} in "
                   f"[{BAR_DISSOC}, {BAR_ADDRESSED}) — attention proposes, "
                   "something else disposes; the residual is the interesting "
                   "object.")
    if underpowered:
        verdict += (" [INSTRUMENT CAVEAT: only "
                    f"{n_valid}/{len(dps)} non-degenerate decision points — "
                    "UNDERPOWERED flag.]")
    log(f"REGISTERED DECISION [{fires}]: {verdict}")

    # ---------------------------------------------------------------- metrics
    metrics = dict(
        experiment="e100_read_prediction",
        purpose="READ-PREDICTION probe (Rule-11 pick; T037's question made "
                "operational on T050's sparse-open fact): for ~100 "
                "margin-stratified free-run decision points (e084's battery, "
                "selection and 80-entry bands verbatim), compute query-side "
                "features BEFORE any intervention — (a) 16-layer-head "
                "attention mass paid by the decision row to each candidate "
                "entry, (b) mean query-key cosine — then the ground-truth "
                "opened set (V-zero argmax flips, e084 bars). FROZEN BARS on "
                "the pooled (pair-weighted within-decision) attention-mass "
                "AUC: >= 0.85 ATTENTION-ADDRESSED; < 0.6 DISSOCIATED; else "
                "PARTIAL. q-k cosine reported against the same bands "
                "descriptively.",
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
                   bootstrap=0),
        protocol=dict(
            n_decision_points=len(dps), per_quartile=PER_QUARTILE,
            q_range=[Q_LO, Q_HI], n_entries=N_ENTRIES,
            entry_bands=dict(young=YOUNG, mid=MID, old_n=N_OLD),
            ground_truth="V-zero flip at the e084 bars (v := 0 at the "
                         "entry's position, every layer, every head, all "
                         "queries; argmax flip vs the same call's clean row)",
            feature_a="attention mass: sum over 16 layer-heads of softmax "
                      "attention prob from the decision row q to the "
                      "entry's position (one clean forward, "
                      "pre-intervention)",
            feature_b="q-k cosine: mean over 16 layer-heads of "
                      "cos(q_lh[q], k_lh[p])",
            auc="Mann-Whitney with tie-averaged ranks; pooled = "
                "pair-weighted across decision points (pairs compared only "
                "within a decision); bootstrap CI over decision-point "
                "resampling n=1000",
            k_list=K_LIST),
        gates=gates,
        ground_truth=dict(
            opened_per_dp_median=float(np.median(opened_sizes)),
            opened_per_dp_hist=np.bincount(opened_sizes,
                                           minlength=N_ENTRIES + 1).tolist(),
            total_opened_cells=total_opened,
            per_cell_rate=chance,
            n_dp_zero_opened=n_zero_open, n_dp_all_opened=n_all_open,
            n_valid_dp=n_valid,
            e084_reference_median_vzero=3.0),
        auc=dict(
            pooled=dict(attention_mass=pooled_att, qk_cosine=pooled_cos,
                        combo_ranksum=pooled_combo,
                        qk_raw_dot=pooled_logit,
                        qk_cosine_max_over_heads=pooled_maxcos),
            ci=dict(attention_mass=list(ci_att), qk_cosine=list(ci_cos),
                    combo_ranksum=list(ci_combo)),
            macro=dict(attention_mass=macro_att, qk_cosine=macro_cos),
            per_layer_head=head_tbl, per_layer=layer_tbl,
            best_head=best_head,
            n_valid_dp=n_valid, n_pairs_dp=int(sum(p for _, p in auc_att_l))),
        precision_at_k=pk_summary,
        chance=dict(per_cell_opened_rate=chance,
                    valid_dp_mean_opened_fraction=chance_valid),
        opened_rank_position=dict(
            attention_mass=dict(
                mean=float(ranks_att_np.mean()) if len(ranks_att_np)
                else float("nan"),
                median=float(np.median(ranks_att_np)) if len(ranks_att_np)
                else float("nan"),
                recall_at_k=recall_att),
            qk_cosine=dict(
                mean=float(ranks_cos_np.mean()) if len(ranks_cos_np)
                else float("nan"),
                median=float(np.median(ranks_cos_np)) if len(ranks_cos_np)
                else float("nan"),
                recall_at_k=recall_cos),
            n_opened_entries=len(ranks_att_np),
            uniform_chance_mean_rank=(N_ENTRIES + 1) / 2,
            note="rank of each opened entry in the per-decision feature "
                 "ranking (1 = the feature's top pick of 80); recall@k = "
                 "fraction of opened entries with rank <= k; uniform "
                 "chance recall@k = k/80"),
        per_decision_point=per_dp,
        registered_decision=dict(
            frozen_bars=dict(attention_addressed=f"pooled attention-mass "
                             f"AUC >= {BAR_ADDRESSED}",
                             dissociated=f"pooled attention-mass AUC < "
                             f"{BAR_DISSOC}",
                             partial=f"else [{BAR_DISSOC}, {BAR_ADDRESSED})",
                             tie_break="evaluated in order; exactly "
                                       f"{BAR_DISSOC} falls to PARTIAL"),
            pooled_attention_auc=pooled_att,
            pooled_attention_ci=list(ci_att),
            pooled_cosine_auc=pooled_cos,
            pooled_cosine_ci=list(ci_cos),
            fires=fires, verdict=verdict,
            underpowered_flag=bool(underpowered)),
    )
    save_json(out_dir / f"metrics{suffix}.json", metrics)
    log(f"metrics{suffix}.json written")
    plot(out_dir / f"read_prediction{suffix}.png", metrics)
    log(f"plot written; total wall {elapsed():.0f}s")


def merge_pair(prev, a, p):
    """Accumulate (auc, pairs) across DPs for pair-weighted pooling."""
    if prev is None:
        return (a, p)
    if p == 0:
        return prev
    if prev[1] == 0:
        return (a, p)
    tot = prev[1] + p
    return ((prev[0] * prev[1] + a * p) / tot, tot)


def pooled_auc_entry(ap):
    return (float(ap[0]), int(ap[1])) if ap is not None else (
        float("nan"), 0)


# ---------------------------------------------------------------------- plot

def plot(path: Path, M: dict):
    dec = M["registered_decision"]
    auc = M["auc"]
    gt = M["ground_truth"]
    pk = M["precision_at_k"]
    ch = M["chance"]["valid_dp_mean_opened_fraction"]

    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # ---- panels 1-2: per-decision precision@k distributions
    for ax, feat, title, col in [
            (ax1, "attention_mass",
             "(a) precision@k — attention mass (per decision point)",
             "tab:blue"),
            (ax2, "qk_cosine",
             "(b) precision@k — q-k cosine (per decision point)",
             "tab:orange")]:
        ks = M["protocol"]["k_list"]
        boxes = [[pk[feat][f"p@{k}"]["q1"], pk[feat][f"p@{k}"]["median"],
                  pk[feat][f"p@{k}"]["q3"]] for k in ks]
        pos = np.arange(len(ks))
        for i, (q1, med, q3) in enumerate(boxes):
            ax.vlines(i, q1, q3, color=col, lw=6, alpha=0.35)
            ax.plot(i, med, "o", color=col, ms=7)
            ax.plot(i, pk[feat][f"p@{ks[i]}"]["mean"], "_",
                    color="tab:red", ms=12, mew=2,
                    label="mean" if i == 0 else None)
        ax.axhline(ch, color="k", ls=":", lw=1.2)
        ax.text(len(ks) - 0.5, ch + 0.01, f"chance {ch:.3f}", fontsize=8,
                ha="right")
        ad = pk[feat]["p@adaptive"]
        ax.axhline(ad["mean"], color="tab:green", ls="--", lw=1.2)
        ax.text(0.0, ad["mean"] + 0.01,
                f"adaptive k=|opened| mean {ad['mean']:.3f}", fontsize=8,
                color="tab:green")
        ax.set_xticks(pos, [str(k) for k in ks])
        ax.set_xlabel("k (top-k of the feature ranking, of 80)")
        ax.set_ylabel("precision@k")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(title, fontsize=9.5)

    # ---- panel 3: pooled AUCs + registered bars
    names = ["attention\nmass (PRIMARY)", "q-k\ncosine",
             "combo\nrank-sum", "q-k raw dot\n(mean)", "q-k cos\nmax head"]
    vals = [auc["pooled"]["attention_mass"], auc["pooled"]["qk_cosine"],
            auc["pooled"]["combo_ranksum"], auc["pooled"]["qk_raw_dot"],
            auc["pooled"]["qk_cosine_max_over_heads"]]
    cis = [auc["ci"]["attention_mass"], auc["ci"]["qk_cosine"],
           auc["ci"]["combo_ranksum"], None, None]
    lo_err = [max(v - c[0], 0) if c else 0 for v, c in zip(vals, cis)]
    hi_err = [max(c[1] - v, 0) if c else 0 for v, c in zip(vals, cis)]
    cols = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown"]
    ax3.bar(np.arange(len(vals)), vals, 0.6,
            yerr=np.asarray([lo_err, hi_err]),
            color=cols, alpha=0.85, capsize=4)
    ax3.axhline(0.5, color="k", ls=":", lw=1.2)
    ax3.axhline(BAR_ADDRESSED, color="tab:green", ls="--", lw=1.4)
    ax3.axhline(BAR_DISSOC, color="tab:red", ls="--", lw=1.4)
    ax3.text(4.45, BAR_ADDRESSED + 0.01, "0.85 ADDRESSED bar", fontsize=8,
             ha="right", color="tab:green")
    ax3.text(4.45, BAR_DISSOC - 0.045, "0.6 DISSOCIATED bar", fontsize=8,
             ha="right", color="tab:red")
    ax3.set_xticks(np.arange(len(vals)), names, fontsize=8)
    ax3.set_ylabel("pooled within-decision AUC")
    ax3.set_ylim(0, 1.02)
    ax3.set_title("(c) pooled AUCs (pair-weighted, bootstrap CI) — "
                  f"registered bars on the PRIMARY", fontsize=9.5)
    ax3.text(0.02, 0.04,
             f"PRIMARY att AUC {vals[0]:.3f} CI "
             f"[{auc['ci']['attention_mass'][0]:.3f},"
             f"{auc['ci']['attention_mass'][1]:.3f}]\n"
             f"cos AUC {vals[1]:.3f} CI "
             f"[{auc['ci']['qk_cosine'][0]:.3f},"
             f"{auc['ci']['qk_cosine'][1]:.3f}]\n"
             f"best single head {auc['best_head']} "
             f"{auc['per_layer_head'][auc['best_head']][0]:.3f}",
             transform=ax3.transAxes, fontsize=8.5, va="bottom",
             family="monospace",
             bbox=dict(fc="white", ec="dimgray", alpha=0.85))

    # ---- panel 4: per-DP AUC scatter att vs cos
    a_att = [r["auc_att"] for r in M["per_decision_point"]
             if "auc_att" in r]
    a_cos = [r["auc_cos"] for r in M["per_decision_point"]
             if "auc_cos" in r]
    ax4.scatter(a_att, a_cos, s=26, alpha=0.7, color="tab:purple")
    lim = [0, 1]
    ax4.plot(lim, lim, "k--", lw=0.9)
    ax4.set_xlim(lim)
    ax4.set_ylim(lim)
    ax4.set_xlabel("per-decision AUC — attention mass")
    ax4.set_ylabel("per-decision AUC — q-k cosine")
    ax4.set_title(f"(d) per-decision AUCs (n={len(a_att)} valid DPs) — "
                  "above diag: cosine better", fontsize=9.5)

    # ---- panel 5: ground-truth opened-set structure
    hist = np.asarray(gt["opened_per_dp_hist"], float)
    hist = hist / max(hist.sum(), 1)
    ax5.plot(np.arange(len(hist)), hist, "o-", ms=4, lw=1.2,
             color="tab:red")
    ax5.axvline(gt["opened_per_dp_median"], color="dimgray", ls="--", lw=1.1)
    ax5.annotate(f"median {gt['opened_per_dp_median']:.0f} (e084 ref 3)",
                 xy=(gt["opened_per_dp_median"], 0.95),
                 xycoords=("data", "axes fraction"), xytext=(4, 0),
                 textcoords="offset points", fontsize=8, color="dimgray")
    ax5.set_xlabel("opened entries per decision point (of 80, V-zero flips)")
    ax5.set_ylabel("fraction of decision points")
    ax5.set_title("(e) ground truth: opened-set sizes — T050's "
                  "sparse-open fact replicated", fontsize=9.5)

    # ---- panel 6: where opened entries sit in the feature rankings
    ra = M["opened_rank_position"]["attention_mass"]
    rc = M["opened_rank_position"]["qk_cosine"]
    ks_r = sorted(int(k) for k in ra["recall_at_k"])
    for (m, col, lbl) in [(ra, "tab:blue", "attention mass"),
                          (rc, "tab:orange", "q-k cosine")]:
        ax6.plot(ks_r, [m["recall_at_k"][k] for k in ks_r], "o-", ms=6,
                 lw=1.4, color=col,
                 label=f"{lbl} (med rank {m['median']:.0f})")
    ax6.plot(ks_r, [k / 80 for k in ks_r], "k:", lw=1.2,
             label="uniform chance (k/80)")
    ax6.set_xscale("log")
    ax6.set_xlabel("feature rank of the opened entry (1 = top pick, of 80)")
    ax6.set_ylabel("fraction of opened entries at rank <= k")
    ax6.legend(fontsize=8, loc="upper left")
    ax6.set_title(f"(f) where the opened coordinates sit in the query-side "
                  f"ranking (n={M['opened_rank_position']['n_opened_entries']} "
                  "opened)", fontsize=9.5)

    fig.suptitle(f"E100 READ PREDICTION | {dec['fires']} | attention-mass "
                 f"AUC {dec['pooled_attention_auc']:.3f} CI "
                 f"[{dec['pooled_attention_ci'][0]:.2f},"
                 f"{dec['pooled_attention_ci'][1]:.2f}] | q-k cosine AUC "
                 f"{dec['pooled_cosine_auc']:.3f} CI "
                 f"[{dec['pooled_cosine_ci'][0]:.2f},"
                 f"{dec['pooled_cosine_ci'][1]:.2f}] | "
                 + dec["verdict"], fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
