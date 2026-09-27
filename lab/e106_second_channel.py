"""E106 — the SECOND-CHANNEL CENSUS: naming the read's second channel.
(T056's registered follow-up to e104; QUEUE slot e106.)

THE QUESTION: attention addresses the read first-order (T054/e100: pooled
AUC 0.907 from L1/L2-dominated 16-head mass) but genuinely mispredicts at
e104's LOCALIZED RESIDUE — the failing-DP stratum (powered bottom-decile
band: 10 worst valid DPs, near-tie Q0 margins, mixed-age opens; blur
REFUTED, labels stable). Is there a SECOND channel — the late-layer
correction (T037-#4 mid-stack sovereignty) — that predicts where early
routing fails?

DESIGN (registered BEFORE any e106 compute; battery / decision points /
features / ground truth ALL verbatim e100/e104 and gated bit-identical):
  - STRATA: FAILING = e104's powered failure band (bottom decile of valid
    DPs by full-16 attention-mass AUC; my rebuild must reproduce e104's
    stored fail set EXACTLY — G5e104). SUCCEEDING = the remaining valid
    DPs. Margin quartiles Q0..Q3 (e084's stratification) as the
    resolution axis for the crossing curve.
  - FEATURES per candidate entry, from the SAME one-clean-forward
    attention state as e100/e104 (nothing recomputed differently):
      EARLY (comparator): L1+L2 attention mass (8 heads) — "L1/L2-mass";
      full-16 mass reported for reference (e100's primary).
      (a) LATE-MASS: L3-only attention mass (4 heads summed);
      (b) LATE-HEADS: L3H0..L3H3 per-head masses individually — which
          late head, if any, ranks the opened set at the failing DPs;
      (c) DELTA channel: (L3 mass minus L1L2 mass) per entry — raw summed
          probabilities primary; per-head-balanced m_L3/4 - m_L1L2/8 as
          sensitivity (AUC is within-decision, so the balancing matters);
      adaptive-k precision (k = |opened|) for early/late/best-head — the
      behavioral grounding of any AUC claim.
  - (d) MARGIN-RESOLVED CROSSING CURVE: pooled pair-weighted AUC of early
    vs late as a function of margin quartile Q0..Q3 (bootstrap CIs; paired
    within-quartile difference CIs). CROSSING (registered, descriptive):
    late > early at Q0 AND early > late at Q3 on point estimates; STRONG
    if the paired-difference CIs exclude 0 with opposite signs.

REGISTERED BARS (frozen; evaluated in order):
  1. TWO-CHANNEL READ CONFIRMED iff at the FAILING stratum: max(AUC_L3,
     AUC_L3H*) >= 0.75 while AUC_L1L2 < 0.6, with CI separation (the late
     feature's bootstrap 95% CI lower bound above the early CI upper
     bound) => the read has TWO named channels: EARLY ROUTING (L1/L2
     mass) + LATE CORRECTION (the winning late feature/head is named).
  2. SINGLE-CHANNEL iff late tracks early everywhere: |AUC_L3 - AUC_L1L2|
     <= 0.05 in ALL strata (failing, succeeding, Q0..Q3) => the residual
     stays unexplained by any late-mass channel; register the next probe
     honestly.
  3. else MIXED (neither bar fires): report the honest texture.

HONEST PRE-REGISTRATION (the band was defined from e104's stored table —
public prior data; e100's pooled per-layer table shows L3 pooled 0.828 vs
L1L2 0.901, and e104's stored per-DP layer AUCs at the 10 failing DPs
show L3 NOT rescuing: mean ~0.69 vs L1L2 ~0.71): the registered
prediction is SINGLE-CHANNEL — late features track early features and the
near-tie residual is not explained by late-layer routing mass. The
two-channel bar as registered is high (it needs early < 0.6 at failing,
which stored point estimates already argue against); we execute it
faithfully and report whichever way it lands.

GATES: G2 params (873,472), G1 val CE (vs e053c 1.5227 +/- 0.02), G0b
incremental-KV vs full recompute (max prob dev < 1e-4), G8 feature-forward
identity (< 1e-4), G5e084 (per-DP V-zero opened sizes vs e084 stored,
>= 90/100), G5e100 (per-DP attention AUC vs e100 stored, < 1e-9),
G5e104 (fail-set identity AND per-DP auc_full / auc_L12 vs e104 stored,
< 1e-9 — the failing stratum is DEFINED by e104's table; drift kills the
census's premise).

Run:     python lab/e106_second_channel.py     (E106_SMOKE=1 smoke)
Outputs: runs/e106/metrics.json + runs/e106/second_channel.png
Envelope: NO training; CPU-only (CUDA_VISIBLE_DEVICES forced -1, cuda
stubbed pre-import, e070/e084/e100/e104 pattern); 8 torch threads; single
step ~5-8 min (same compute shape as e104). No NOTES/THINKING/QUEUE/STATE
edits; no commit (the dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu" (e053b/e069/e070/
# e084/e100/e104 pattern)
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
SEED_PROMPT, SEED_SAMPLE = 202, 7         # e053b/e053c/e069/e070/e072/e084/e100/e104
N_PROMPTS = 8
B = 4                                     # battery A (e069 verbatim)
B_X = 4                                   # fresh extras (e072 family, e084)
SEED_PROMPT_X, SEED_SAMPLE_X = 302, 17    # 202/7 family +100/+10
BOOT_N = 1000
THREADS = 8      # e084's measured sweet spot on this contended box
torch.set_num_threads(THREADS)

CHUNK_512 = 64                            # e069/e070/e072/e084/e100/e104 chunk

# ---- design constants (registered; ALL verbatim e084/e100/e104) --------------
Q_LO, Q_HI = 96, 510                      # candidate query positions
PER_QUARTILE = 25                         # 25 per margin quartile
N_DP = 4 * PER_QUARTILE                   # 100 decision points
SEED_STRAT = 2084                         # margin-stratified sampling seed
YOUNG = list(range(1, 11))                # ages 1-10 (all)
MID = list(range(11, 61))                 # ages 11-60 (all)
N_OLD = 20                                # old entries per decision point
OLD_SEED_BASE = 9000                      # old-age rng seed = base + dp_index
N_ENTRIES = len(YOUNG) + len(MID) + N_OLD  # 80
YOUNG_AGE_MAX = 10                        # opened-age texture (e104 stratum c)

# ---- second-channel constants (registered) -----------------------------------
DECILE_FRAC = 0.10                        # e104's powered failure band def
BAR_LATE = 0.75                           # two-channel: late >= at failing
BAR_EARLY = 0.6                           # two-channel: early < at failing
SINGLE_TOL = 0.05                         # single-channel: |late-early| <=
LATE_HEADS = ["L3H0", "L3H1", "L3H2", "L3H3"]
LATE_FEATS = ["L3_late"] + LATE_HEADS     # bar-eligible late features
EARLY_FEAT = "L12_early"                  # the comparator
STRATA_MAIN = ["failing", "succeeding"]
QUARTS = ["Q0", "Q1", "Q2", "Q3"]
ALL_STRATA = STRATA_MAIN + QUARTS         # single-channel "everywhere" set

# ---- stored references (gates) ----------------------------------------------
E084_METRICS = REPO / "runs" / "e084" / "metrics.json"
E100_METRICS = REPO / "runs" / "e100" / "metrics.json"
E104_METRICS = REPO / "runs" / "e104" / "metrics.json"
CKPT = REPO / "runs" / "checkpoints" / "e053c_ctx512.pt"
E053C_VAL_CE = 1.5226792494455974

SMOKE = os.environ.get("E106_SMOKE", "") == "1"
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


# --------------------------------------------- machinery (VERBATIM e069/e084/e100/e104)

@torch.no_grad()
def _manual_chunk2(net: TinyGPT, idxs, vzero, kdrop, layers, pos_offset=0,
                   vswap_pos=None, donor_stack=None):
    """e069/e084/e100/e104's _manual_chunk VERBATIM (vswap/kdrop arms inert
    here — e106 uses only the clean row and V-zero masks, but the instrument
    is kept byte-identical to the rig whose ground truth it reproduces)."""
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
    [VERBATIM e053b/e084/e100/e104]"""
    lg = logits_clean / TEMP
    v, _ = torch.topk(lg, TOPK)
    lg_f = lg.masked_fill(lg < v[-1], float("-inf"))
    tok = int(torch.multinomial(torch.softmax(lg_f, -1), 1, generator=gen))
    p_full = torch.softmax(logits_clean, -1)
    return tok, float(-math.log(max(p_full[tok].item(), 1e-12)))


@torch.no_grad()
def prefill_batch(net: TinyGPT, idx: torch.Tensor):
    """Batched prefill, (B, Tp) -> last-position logits (B, V) + KV cache.
    [VERBATIM e069/e084/e100/e104]"""
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
    [VERBATIM e069/e084/e100/e104]"""
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
    """e069/e084/e100/e104's generate_batch VERBATIM + per-step decision
    logits: hist[:, j] is the distribution at query position 63+j (the
    decision for token 64+j). Returns (idx, final_logits, hist)."""
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


# --------------------------------------------- e106's own (gated) instruments

@torch.no_grad()
def query_features(net: TinyGPT, ctx: torch.Tensor):
    """ONE clean forward over ctx (1-D token ids, length T); record, per
    layer, the QUERY-SIDE attention state BEFORE any intervention (VERBATIM
    e100/e104 — G5e100/G5e104/G8 depend on byte-identity):
      att_last (L, H, T): softmax attention probabilities of the decision
      row (last position) over all positions.
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
    """e084/e100/e104's entry band VERBATIM: young 1-10 + mid 11-60 + 20 old
    sampled without replacement uniformly from [61, q+1] (default_rng seed
    9000+dp_index). Returns (ages (80,), positions (80,)), position = T-age,
    T = q+1."""
    T = q + 1
    rng = np.random.default_rng(OLD_SEED_BASE + dp_index)
    avail = np.arange(61, T + 1)
    old = [int(a) for a in rng.choice(avail, size=N_OLD, replace=False)]
    entries = list(YOUNG) + list(MID) + old              # 80 ages, e084 order
    ps = [T - a for a in entries]
    return entries, ps


def opened_set(net: TinyGPT, idx16, b, q, entries, ps):
    """GROUND TRUTH at the e084/e100/e104 bars: one batched manual_logits2
    call, 1 clean row + 80 V-zero rows (v := 0 at that position, every
    layer, every head, all queries — e069/e084 whole-position instrument
    verbatim). Returns (opened flags (80,) bool, t1, margin)."""
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
    """per_dp: list of (auc, n_pairs) -> pair-weighted pooled AUC.
    Zero-pair entries (auc=nan) are skipped — nan*0 would poison the sum."""
    arr = [(a, p) for a, p in per_dp if p > 0 and np.isfinite(a)]
    tot = sum(p for _, p in arr)
    if tot == 0:
        return float("nan")
    return sum(a * p for a, p in arr) / tot


def bootstrap_pooled(per_dp, n=BOOT_N, seed=0):
    """Bootstrap CI of the pair-weighted pooled AUC (decision-point
    resampling with replacement, within the stratum)."""
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


def bootstrap_paired_diff(list_a, list_b, n=BOOT_N, seed=0):
    """Paired bootstrap CI of pooled_auc(a) - pooled_auc(b) over the SAME
    decision-point resample (within-stratum; a/b indexed per DP)."""
    arr = [(a, b, p) for (a, p), (b, _) in zip(list_a, list_b) if p > 0]
    if not arr:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sel = rng.integers(0, len(arr), len(arr))
        vals.append(pooled_auc([(arr[i][0], arr[i][2]) for i in sel])
                    - pooled_auc([(arr[i][1], arr[i][2]) for i in sel]))
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
    out_dir = run_dir("e106")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=SMOKE)
    suffix = "_smoke" if SMOKE else ""

    # ---- battery: rebuild corpus + net EXACTLY as e053c/e069/e070/e084/e100/e104
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
    #      then the ground-truth opened set (V-zero flips, e084 bars)
    t_probe = time.time()
    per_dp = []
    # per-DP (auc, pairs) per feature — pooled by stratum below
    feats_auc = {f: [] for f in ([EARLY_FEAT, "full16", "L3_late"]
                                 + LATE_HEADS + ["delta_raw", "delta_bal"])}
    pk_adapt = {f: [] for f in ([EARLY_FEAT, "L3_late"] + LATE_HEADS)}
    opened_sizes = []

    for i, (b, q, _, qt) in enumerate(dps):
        entries, ps = entry_positions(q, i)

        # (1) query-side attention masses from ONE clean forward
        att_last, _ = query_features(net, idx16[b, :q + 1])
        L, H, T = att_last.shape
        ps_np = np.asarray(ps)
        att_lh = att_last.numpy()                        # (L,H,T)
        m_full = att_lh.sum((0, 1))                      # 16-head mass (e100)
        m_L12 = att_lh[1:3].sum((0, 1))                  # EARLY: L1+L2, 8 heads
        m_L3 = att_lh[3].sum(0)                          # LATE: L3-only, 4 heads
        m_L3h = [att_lh[3, h] for h in range(H)]         # late heads
        m_delta_raw = m_L3 - m_L12                       # (c) raw delta
        m_delta_bal = m_L3 / 4.0 - m_L12 / 8.0           # (c) balanced delta
        scores = {
            EARLY_FEAT: m_L12[ps_np], "full16": m_full[ps_np],
            "L3_late": m_L3[ps_np],
            **{f"L3H{h}": m_L3h[h][ps_np] for h in range(H)},
            "delta_raw": m_delta_raw[ps_np], "delta_bal": m_delta_bal[ps_np],
        }

        # (2) ground truth: V-zero flips at the e084/e100/e104 bars
        opened, t1, margin = opened_set(net, idx16, b, q, entries, ps)
        n_open = int(opened.sum())
        opened_sizes.append(n_open)

        rec = dict(dp=i, b=int(b), q=int(q), quartile=int(qt), margin=margin,
                   t1=t1, n_opened=n_open,
                   opened_ages=[int(a) for a, o in zip(entries, opened) if o])
        valid = 1 <= n_open <= N_ENTRIES - 1
        if valid:
            rec["young_frac_opened"] = float(
                np.mean([a <= YOUNG_AGE_MAX
                         for a, o in zip(entries, opened) if o]))
            for fname, s in scores.items():
                a, p = auc_pairs(s, opened)
                rec[f"auc_{fname}"] = a
                feats_auc[fname].append((a, p))
            for fname in pk_adapt:
                pk = precision_at_k(scores[fname], opened, [n_open])[n_open]
                rec[f"pk_{fname}"] = pk
                pk_adapt[fname].append(pk)
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

    # ---- identity gates vs e084 / e100 / e104 stored tables
    if SMOKE:
        gates["G5_e084_identity"] = dict(skipped=True, reason="smoke battery")
        gates["G5_e100_auc_identity"] = dict(skipped=True, reason="smoke")
        gates["G5_e104_identity"] = dict(skipped=True, reason="smoke")
        log("G5 gates: SKIPPED (smoke)")
    else:
        e084 = json.loads(E084_METRICS.read_text(encoding="utf-8"))
        ref_counts = [p["flips_vz"] for p in e084["per_decision_point"]]
        mine = [r["n_opened"] for r in per_dp]
        assert len(ref_counts) == len(mine), "e084 per-DP count mismatch"
        matches = sum(1 for a, b_ in zip(mine, ref_counts) if a == b_)
        gates["G5_e084_identity"] = dict(
            n_match=matches, n_dp=len(mine), bar=90,
            ok=bool(matches >= 90), count_pearson=pearson(mine, ref_counts))
        log(f"G5 vs e084 stored flips_vz: {matches}/{len(mine)} exact "
            f"matches (bar 90) -> "
            f"{'PASS' if gates['G5_e084_identity']['ok'] else 'FAIL'}")

        e100 = json.loads(E100_METRICS.read_text(encoding="utf-8"))
        e100_auc = {r["dp"]: r.get("auc_att")
                    for r in e100["per_decision_point"]}
        devs, ok_n = [], 0
        for r in per_dp:
            ref = e100_auc.get(r["dp"])
            if ref is not None and "auc_full16" in r:
                d = abs(r["auc_full16"] - ref)
                devs.append(d)
                ok_n += int(d < 1e-9)
        max_dev = max(devs) if devs else float("nan")
        gates["G5_e100_auc_identity"] = dict(
            n_compared=len(devs), n_match_lt_1e9=ok_n, max_auc_dev=max_dev,
            tol=1e-9, ok=bool(ok_n == len(devs) and max_dev < 1e-9))
        log(f"G5 vs e100 stored per-DP AUCs: {ok_n}/{len(devs)} within 1e-9 "
            f"(max dev {max_dev:.2e}) -> "
            f"{'PASS' if gates['G5_e100_auc_identity']['ok'] else 'FAIL'}")

        # G5e104 comes AFTER the fail-set recompute below; placeholder here.

    # ---- failing band recompute (e104's powered band, same rule)
    valid_recs = [r for r in per_dp if f"auc_{EARLY_FEAT}" in r]
    n_valid = len(valid_recs)
    order = sorted(valid_recs, key=lambda r: (r["auc_full16"], r["dp"]))
    n_fail = max(1, int(round(DECILE_FRAC * n_valid)))
    fail_dps = {r["dp"] for r in order[:n_fail]}
    fail_cut_auc = order[n_fail - 1]["auc_full16"]
    log(f"recomputed failure band: {n_fail}/{n_valid} worst DPs "
        f"(cut AUC {fail_cut_auc:.3f}) -> dp {sorted(fail_dps)}")

    if not SMOKE:
        e104 = json.loads(E104_METRICS.read_text(encoding="utf-8"))
        ref_fail = set(e104["census"]["fail_dps"])
        e104_full = {r["dp"]: r.get("auc_full") for r in e104["per_decision_point"]}
        e104_L12 = {r["dp"]: r.get("auc_L12") for r in e104["per_decision_point"]}
        devs_f, devs_12 = [], []
        for r in valid_recs:
            rf = e104_full.get(r["dp"])
            r12 = e104_L12.get(r["dp"])
            if rf is not None:
                devs_f.append(abs(r["auc_full16"] - rf))
            if r12 is not None:
                devs_12.append(abs(r[f"auc_{EARLY_FEAT}"] - r12))
        set_eq = fail_dps == ref_fail
        max_df = max(devs_f) if devs_f else float("nan")
        max_d12 = max(devs_12) if devs_12 else float("nan")
        gates["G5_e104_identity"] = dict(
            fail_set_equal=bool(set_eq), my_fail=sorted(fail_dps),
            e104_fail=sorted(ref_fail),
            n_compared_full=len(devs_f), max_auc_full_dev=max_df,
            n_compared_L12=len(devs_12), max_auc_L12_dev=max_d12, tol=1e-9,
            ok=bool(set_eq and max_df < 1e-9 and max_d12 < 1e-9))
        log(f"G5 vs e104 stored: fail set equal {set_eq}; max |auc_full dev| "
            f"{max_df:.2e}; max |auc_L12 dev| {max_d12:.2e} -> "
            f"{'PASS' if gates['G5_e104_identity']['ok'] else 'FAIL'}")

    # ================= the second-channel census (registered analyses) ======
    def stratum_of(r):
        return ("failing" if r["dp"] in fail_dps else "succeeding",
                f"Q{r['quartile']}")

    # per-stratum pooled AUC + bootstrap CI per feature
    buckets = {s: {f: [] for f in feats_auc} for s in ALL_STRATA}
    for r in valid_recs:
        s_main, s_q = stratum_of(r)
        for f in feats_auc:
            buckets[s_main][f].append((r[f"auc_{f}"],
                                       r["n_opened"] * (N_ENTRIES
                                                        - r["n_opened"])))
            buckets[s_q][f].append((r[f"auc_{f}"],
                                    r["n_opened"] * (N_ENTRIES
                                                     - r["n_opened"])))
    strat_tbl = {}
    for s in ALL_STRATA:
        strat_tbl[s] = {}
        for f in feats_auc:
            pooled = pooled_auc(buckets[s][f])
            ci = bootstrap_pooled(buckets[s][f])
            strat_tbl[s][f] = dict(auc=pooled, ci=list(ci), n_dp=len(buckets[s][f]))

    # paired within-stratum late-minus-early difference CIs
    diff_tbl = {}
    for s in ALL_STRATA:
        diff_tbl[s] = dict(
            L3_minus_L12=dict(ci=list(bootstrap_paired_diff(
                buckets[s]["L3_late"], buckets[s][EARLY_FEAT])),
                point=strat_tbl[s]["L3_late"]["auc"]
                - strat_tbl[s][EARLY_FEAT]["auc"]))

    # best late feature at the failing stratum
    best_late = max(LATE_FEATS, key=lambda f: strat_tbl["failing"][f]["auc"])

    # adaptive-k precision per stratum for early / late / late heads
    pk_summary = {}
    for f in pk_adapt:
        pk_summary[f] = {
            s: dict(mean=float(np.mean([r[f"pk_{f}"] for r in valid_recs
                                        if (r["dp"] in fail_dps)
                                        == (s == "failing")])),
                    n=sum(1 for r in valid_recs
                          if (r["dp"] in fail_dps) == (s == "failing")))
            for s in STRATA_MAIN}

    # ---- margin-resolved crossing curve (d)
    curve = {}
    for s_q in QUARTS:
        a_e = strat_tbl[s_q][EARLY_FEAT]
        a_l = strat_tbl[s_q]["L3_late"]
        d = diff_tbl[s_q]["L3_minus_L12"]
        curve[s_q] = dict(early=a_e, late=a_l, late_minus_early=d)
    crossing_point = bool(strat_tbl["Q0"]["L3_late"]["auc"]
                          > strat_tbl["Q0"][EARLY_FEAT]["auc"]
                          and strat_tbl["Q3"][EARLY_FEAT]["auc"]
                          > strat_tbl["Q3"]["L3_late"]["auc"])
    ci0 = curve["Q0"]["late_minus_early"]["ci"]
    ci3 = curve["Q3"]["late_minus_early"]["ci"]
    crossing_strong = bool(crossing_point and ci0[0] > 0 and ci3[1] < 0)

    # ---- REGISTERED decision (frozen bars, in order)
    early_f = strat_tbl["failing"][EARLY_FEAT]
    late_best_f = strat_tbl["failing"][best_late]
    cond_late = bool(late_best_f["auc"] >= BAR_LATE)
    cond_early = bool(early_f["auc"] < BAR_EARLY)
    cond_ci = bool(late_best_f["ci"][0] > early_f["ci"][1])
    two_channel = cond_late and cond_early and cond_ci
    if two_channel:
        fires = "TWO_CHANNEL_CONFIRMED"
        verdict = (f"TWO-CHANNEL READ CONFIRMED: at the failing stratum the "
                   f"late feature {best_late} reaches AUC "
                   f"{late_best_f['auc']:.3f} CI "
                   f"[{late_best_f['ci'][0]:.3f},{late_best_f['ci'][1]:.3f}] "
                   f">= {BAR_LATE} while early {EARLY_FEAT} sits at "
                   f"{early_f['auc']:.3f} CI "
                   f"[{early_f['ci'][0]:.3f},{early_f['ci'][1]:.3f}] < "
                   f"{BAR_EARLY}, CIs separated — the read is EARLY ROUTING "
                   f"(L1/L2 mass) + LATE CORRECTION ({best_late}).")
    else:
        why = []
        if not cond_late:
            why.append(f"best late ({best_late}) AUC {late_best_f['auc']:.3f} "
                       f"< {BAR_LATE}")
        if not cond_early:
            why.append(f"early AUC {early_f['auc']:.3f} not < {BAR_EARLY}")
        if cond_late and cond_early and not cond_ci:
            why.append("CIs not separated")
        track = all(abs(strat_tbl[s]["L3_late"]["auc"]
                        - strat_tbl[s][EARLY_FEAT]["auc"]) <= SINGLE_TOL
                    for s in ALL_STRATA)
        if track:
            fires = "SINGLE_CHANNEL"
            verdict = (f"SINGLE-CHANNEL: the two-channel bar did not fire "
                       f"({'; '.join(why)}) and late features track early "
                       f"features everywhere (|L3 - L1L2| <= {SINGLE_TOL} in "
                       f"all strata: "
                       + ", ".join(f"{s} {strat_tbl[s]['L3_late']['auc']:.3f}"
                                   f"vs{strat_tbl[s][EARLY_FEAT]['auc']:.3f}"
                                   for s in ALL_STRATA)
                       + f"). The near-tie residual stays UNEXPLAINED by "
                       f"late-mass routing — the next probe must leave the "
                       f"routing-mass family (registered honestly).")
        else:
            deviating = [s for s in ALL_STRATA
                         if abs(strat_tbl[s]["L3_late"]["auc"]
                                - strat_tbl[s][EARLY_FEAT]["auc"]) > SINGLE_TOL]
            fires = "MIXED"
            verdict = (f"MIXED: the two-channel bar did not fire "
                       f"({'; '.join(why)}) and late does NOT track early "
                       f"everywhere (deviating strata: {deviating}) — an "
                       f"honest intermediate; neither registered reading "
                       f"holds cleanly.")
    log(f"REGISTERED DECISION [{fires}]: {verdict}")

    # honesty flags: any late head deviating > tol from early, per stratum
    head_flags = [dict(stratum=s, feature=f,
                       auc=strat_tbl[s][f]["auc"],
                       dev=strat_tbl[s][f]["auc"]
                       - strat_tbl[s][EARLY_FEAT]["auc"])
                  for s in ALL_STRATA for f in LATE_FEATS
                  if abs(strat_tbl[s][f]["auc"]
                         - strat_tbl[s][EARLY_FEAT]["auc"]) > SINGLE_TOL]

    # failing-stratum anatomy (compact)
    anatomy = []
    for r in valid_recs:
        if r["dp"] in fail_dps:
            anatomy.append(dict(
                dp=r["dp"], b=r["b"], q=r["q"], quartile=r["quartile"],
                margin=r["margin"], n_opened=r["n_opened"],
                young_frac_opened=r["young_frac_opened"],
                auc_L12=r[f"auc_{EARLY_FEAT}"], auc_full16=r["auc_full16"],
                auc_L3=r["auc_L3_late"],
                auc_L3H=[r[f"auc_L3H{h}"] for h in range(4)],
                auc_delta_raw=r["auc_delta_raw"]))

    # ---------------------------------------------------------------- metrics
    metrics = dict(
        experiment="e106_second_channel",
        purpose="SECOND-CHANNEL CENSUS (T056's registered e106, QUEUE slot): "
                "name the read's second channel. Rebuild e100/e104's battery, "
                "decision points, features and V-zero ground truth "
                "bit-identically (G5 vs e084/e100/e104), then at the FAILING "
                "stratum (e104's powered bottom-decile band) vs SUCCEEDING: "
                "(a) L3-only attention mass, (b) L3H0-H3 per-head masses, "
                "(c) DELTA (L3 minus L1L2, raw + balanced), (d) margin-"
                "quartile-resolved early-vs-late pooled AUC crossing curve. "
                "FROZEN BARS: TWO-CHANNEL iff at failing max(L3, L3H*) >= "
                "0.75 while L1L2 < 0.6 with CI separation (names the late "
                "channel); SINGLE-CHANNEL iff |L3 - L1L2| <= 0.05 in ALL "
                "strata (residual unexplained; next probe registered); else "
                "MIXED.",
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
            failing_band=dict(rule="bottom decile of valid DPs by full-16 "
                                  "attention-mass AUC (e104 verbatim)",
                              frac=DECILE_FRAC, n_fail=n_fail,
                              cut_auc=fail_cut_auc, fail_dps=sorted(fail_dps),
                              n_valid=n_valid),
            features=dict(
                early="L1+L2 attention mass (8 heads summed) — the "
                      "comparator (e104's L1/L2-mass)",
                full16="sum of all 16 layer-heads (e100's primary; "
                       "reference)",
                late="(a) L3-only attention mass (4 heads summed)",
                late_heads="(b) L3H0..L3H3 per-head masses individually",
                delta="(c) L3 mass - L1L2 mass per entry; raw primary, "
                      "per-head-balanced (m_L3/4 - m_L1L2/8) sensitivity",
            ),
            ground_truth="V-zero flip at the e084 bars (v := 0 at the "
                         "entry's position, every layer, every head, all "
                         "queries; argmax flip vs the same call's clean row)",
            auc="Mann-Whitney tie-averaged; pooled = pair-weighted across "
                "decision points (pairs only within a decision); bootstrap "
                "CI = decision-point resampling within stratum n=1000; "
                "paired-diff CI resamples DPs jointly"),
        gates=gates,
        stratum_auc=strat_tbl,
        stratum_paired_diff=diff_tbl,
        adaptive_k_precision=pk_summary,
        head_tolerance_flags=head_flags,
        margin_crossing=dict(
            curve=curve,
            crossing_point_estimates=crossing_point,
            crossing_strong_ci=crossing_strong,
            definition="late > early at Q0 AND early > late at Q3 (point "
                       "estimates); strong = paired-diff CIs exclude 0 with "
                       "opposite signs"),
        failing_anatomy=anatomy,
        per_decision_point=per_dp,
        registered_decision=dict(
            frozen_bars=dict(
                two_channel=f"at failing stratum: max(AUC_L3, AUC_L3H*) >= "
                            f"{BAR_LATE} AND AUC_L1L2 < {BAR_EARLY} AND late "
                            f"CI lo > early CI hi => EARLY ROUTING + LATE "
                            f"CORRECTION (head named)",
                single_channel=f"|AUC_L3 - AUC_L1L2| <= {SINGLE_TOL} in all "
                               f"strata {ALL_STRATA} => residual stays "
                               f"unexplained; next probe registered honestly",
                else_="MIXED (neither bar fires)"),
            best_late_at_failing=best_late,
            conditions=dict(late_ge_075=cond_late, early_lt_06=cond_early,
                            ci_separated=cond_ci),
            fires=fires, verdict=verdict),
    )
    save_json(out_dir / f"metrics{suffix}.json", metrics)
    log(f"metrics{suffix}.json written")
    plot(out_dir / f"second_channel{suffix}.png", metrics)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot

def plot(path: Path, M: dict):
    dec = M["registered_decision"]
    st = M["stratum_auc"]
    cr = M["margin_crossing"]
    best_late = dec["best_late_at_failing"]

    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # ---- panel 1: per-stratum AUC bars early-vs-late (+ registered bars)
    feat_groups = [("L12_early", "EARLY\nL1/L2 mass (8h)", "tab:blue"),
                   ("full16", "full 16\n(e100 primary)", "steelblue"),
                   ("L3_late", "LATE\nL3-only (4h)", "tab:red"),
                   (best_late, f"LATE best\n{best_late}", "tab:orange")]
    strata_x = ["failing", "succeeding"]
    width = 0.2
    for gi, (f, lbl, col) in enumerate(feat_groups):
        xs = np.arange(len(strata_x)) + (gi - 1.5) * width
        vals = [st[s][f]["auc"] for s in strata_x]
        cis = [st[s][f]["ci"] for s in strata_x]
        lo = [max(v - c[0], 0) for v, c in zip(vals, cis)]
        hi = [max(c[1] - v, 0) for v, c in zip(vals, cis)]
        ax1.bar(xs, vals, width, yerr=np.asarray([lo, hi]), color=col,
                alpha=0.85, capsize=3, label=lbl)
    ax1.axhline(0.5, color="k", ls=":", lw=1.1)
    ax1.axhline(BAR_LATE, color="tab:green", ls="--", lw=1.4)
    ax1.axhline(BAR_EARLY, color="tab:red", ls="--", lw=1.4)
    ax1.text(1.48, BAR_LATE + 0.008, "0.75 late bar", fontsize=8, ha="right",
             color="tab:green")
    ax1.text(1.48, BAR_EARLY - 0.035, "0.60 early bar", fontsize=8,
             ha="right", color="tab:red")
    ax1.set_xticks(np.arange(len(strata_x)),
                   [f"FAILING\n(n={st['failing']['L12_early']['n_dp']})",
                    f"SUCCEEDING\n(n={st['succeeding']['L12_early']['n_dp']})"])
    ax1.set_ylabel("pooled within-decision AUC")
    ax1.set_ylim(0, 1.02)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.set_title("(a) per-stratum AUC — early routing vs late mass "
                  "(bootstrap CIs; registered bars)", fontsize=9.5)

    # ---- panel 2: margin-resolved crossing curve
    qs = QUARTS
    for f, col, lbl in [(EARLY_FEAT, "tab:blue", "EARLY L1/L2 mass"),
                        ("L3_late", "tab:red", "LATE L3 mass")]:
        vals = [st[s][f]["auc"] for s in qs]
        lo = [st[s][f]["ci"][0] for s in qs]
        hi = [st[s][f]["ci"][1] for s in qs]
        ax2.plot(qs, vals, "o-", color=col, lw=2, ms=7, label=lbl)
        ax2.fill_between(qs, lo, hi, color=col, alpha=0.15)
    ax2.axhline(0.5, color="k", ls=":", lw=1.1)
    ax2.set_xlabel("decision-margin quartile (Q0 = near-tie, Q3 = confident)")
    ax2.set_ylabel("pooled within-decision AUC")
    ax2.set_ylim(0.5, 1.02)
    ax2.legend(fontsize=8, loc="center right")
    cx = ("CROSSES" if cr["crossing_point_estimates"]
          else "no point-estimate crossing")
    ax2.set_title(f"(b) margin-resolved early vs late — {cx}"
                  + (" [STRONG: CIs separate]" if cr["crossing_strong_ci"]
                     else ""), fontsize=9.5)

    # ---- panel 3: which late head — L3H* + L3 at failing vs succeeding
    heads = ["L3_late"] + LATE_HEADS
    xs = np.arange(len(heads))
    for s, col, a in [("failing", "tab:red", 0.9), ("succeeding",
                                                    "lightcoral", 0.55)]:
        vals = [st[s][h]["auc"] for h in heads]
        cis = [st[s][h]["ci"] for h in heads]
        lo = [max(v - c[0], 0) for v, c in zip(vals, cis)]
        hi = [max(c[1] - v, 0) for v, c in zip(vals, cis)]
        ax3.bar(xs + (0.19 if s == "failing" else -0.19), vals, 0.36,
                yerr=np.asarray([lo, hi]), color=col, alpha=a, capsize=3,
                label=s)
    ax3.axhline(st["failing"][EARLY_FEAT]["auc"], color="tab:blue", ls="--",
                lw=1.3)
    ax3.text(len(heads) - 0.5, st["failing"][EARLY_FEAT]["auc"] + 0.008,
             f"early L1/L2 at failing {st['failing'][EARLY_FEAT]['auc']:.3f}",
             fontsize=8, ha="right", color="tab:blue")
    ax3.axhline(BAR_LATE, color="tab:green", ls="--", lw=1.2)
    ax3.set_xticks(xs, ["L3 pooled\n(4 heads)"] + [h + "\n(single)" for h
                                                   in LATE_HEADS], fontsize=8)
    ax3.set_ylabel("pooled within-decision AUC")
    ax3.set_ylim(0, 1.02)
    ax3.legend(fontsize=8, loc="lower right")
    ax3.set_title("(c) which late head, if any — L3-family AUC at failing "
                  "(red) vs succeeding (pale)", fontsize=9.5)

    # ---- panel 4: delta channel + adaptive-k precision
    for ax, feats, title in [
            (ax4, ["delta_raw", "delta_bal"],
             "(d) DELTA channel (L3 - L1L2)")]:
        xs = np.arange(len(feats))
        for si, s in enumerate(strata_x):
            vals = [st[s][f]["auc"] for f in feats]
            cis = [st[s][f]["ci"] for f in feats]
            lo = [max(v - c[0], 0) for v, c in zip(vals, cis)]
            hi = [max(c[1] - v, 0) for v, c in zip(vals, cis)]
            ax.bar(xs + (si - 0.5) * 0.36, vals, 0.32,
                   yerr=np.asarray([lo, hi]), capsize=3,
                   color="tab:red" if s == "failing" else "lightcoral",
                   alpha=0.9 if s == "failing" else 0.6, label=s)
        for f, xoff in zip(feats, range(len(feats))):
            e = st["failing"][EARLY_FEAT]["auc"]
            ax.plot([xoff - 0.18, xoff + 0.18], [e, e], ls="--", lw=1.0,
                    color="tab:blue")
        ax.axhline(0.5, color="k", ls=":", lw=1.1)
        ax.set_xticks(xs, ["raw\n(m_L3 - m_L1L2)",
                           "balanced\n(m_L3/4 - m_L1L2/8)"], fontsize=8)
        ax.set_ylabel("pooled within-decision AUC")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title(title + " — blue dashes = early at failing", fontsize=9.5)

    # ---- panel 5: per-DP scatter early (x) vs late (y) AUC
    xs5, ys5, cs5 = [], [], []
    for r in M["per_decision_point"]:
        if f"auc_{EARLY_FEAT}" not in r:
            continue
        xs5.append(r[f"auc_{EARLY_FEAT}"])
        ys5.append(r["auc_L3_late"])
        cs5.append("tab:red" if r["dp"] in set(
            M["protocol"]["failing_band"]["fail_dps"]) else "steelblue")
    ax5.scatter(xs5, ys5, s=26, c=cs5, alpha=0.75)
    lim = [0, 1]
    ax5.plot(lim, lim, "k--", lw=0.9)
    ax5.set_xlim(lim)
    ax5.set_ylim(lim)
    ax5.set_xlabel("per-DP AUC — EARLY (L1/L2 mass)")
    ax5.set_ylabel("per-DP AUC — LATE (L3 mass)")
    ax5.set_title(f"(e) per-decision AUCs (n={len(xs5)} valid; red = "
                  "failing band) — above diag: late better", fontsize=9.5)

    # ---- panel 6: the verdict + stratum table
    ax6.axis("off")
    txt = f"REGISTERED DECISION: {dec['fires']}\n\n"
    txt += "stratum AUC table (pooled, pair-weighted):\n"
    txt += (f"  {'stratum':11s} {'early L1L2':>11s} {'full16':>7s} "
            f"{'late L3':>8s} {best_late + ' (best)':>14s}\n")
    for s in ALL_STRATA:
        txt += (f"  {s:11s} {st[s][EARLY_FEAT]['auc']:11.3f} "
                f"{st[s]['full16']['auc']:7.3f} {st[s]['L3_late']['auc']:8.3f} "
                f"{st[s][best_late]['auc']:14.3f}\n")
    txt += (f"\nconditions: late>= {BAR_LATE} "
            f"{dec['conditions']['late_ge_075']} | early < {BAR_EARLY} "
            f"{dec['conditions']['early_lt_06']} | CI separation "
            f"{dec['conditions']['ci_separated']}\n")
    txt += (f"margin crossing: {cr['crossing_point_estimates']} "
            f"(strong {cr['crossing_strong_ci']})\n")
    txt += ("adaptive-k precision (k=|opened|): early "
            f"{M['adaptive_k_precision'][EARLY_FEAT]['failing']['mean']:.3f}"
            " vs late "
            f"{M['adaptive_k_precision']['L3_late']['failing']['mean']:.3f} "
            "at failing\n\n")
    import textwrap
    txt += "\n".join(textwrap.wrap(dec["verdict"], width=104))
    ax6.text(0.02, 0.97, txt, fontsize=8.6, va="top", family="monospace",
             bbox=dict(fc="white", ec="dimgray"))

    fig.suptitle(f"E106 SECOND-CHANNEL CENSUS | {dec['fires']} | best late "
                 f"at failing: {best_late} "
                 f"{st['failing'][best_late]['auc']:.3f} vs early "
                 f"{st['failing'][EARLY_FEAT]['auc']:.3f} | "
                 + dec["verdict"][:200], fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
