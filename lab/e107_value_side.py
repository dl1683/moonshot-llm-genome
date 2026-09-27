"""E107 — the VALUE-SIDE residual probe: does WHAT an entry contains predict
the opened set where attention (routing) fails?

T058's registered re-aim (after e106's MIXED): leave the routing-mass family.
Attention predicts the read at 0.907 pooled (T054/e100) and even at the
failing stratum holds 0.779 (e106's L1L2 comparator) — no late-layer,
delta, or head-feature rescued the failures. The honest remaining family
is VALUE-SIDE: what the candidate entries CONTAIN, not where attention
points.

THE QUESTION (registered): does a content-similarity feature — the entry's
value-vector alignment with the decision's intended readout — predict the
opened set, ESPECIALLY at the failing stratum (near-tie / mixed-age
decisions, e104's powered bottom decile)?

DESIGN (registered BEFORE any compute; everything frozen):

  - Battery + decision points + entry bands + ground truth: EXACTLY
    e084/e100/e104/e106's (protocol identity by reconstruction, G5-gated
    bit-identically against all three stored tables): same net
    (runs/checkpoints/e053c_ctx512.pt, 4L/4H/128, ctx 512), same battery
    (e069's 4 seqs seeds 202/7 + e072 family's 4 fresh seeds 302/17, free
    run 64->512, temp 0.8 top-k 40), same margin-stratified 100 decision
    points (seed 2084), same 80-entry band per decision point (young 1-10,
    mid 11-60, 20 old from [61, q+1], seed 9000+dp). Ground truth opened
    set: V-zero whole-position flip at the e084 bars (v := 0 at that
    position, every layer, every head, all queries; OPENED iff argmax
    flips).

  - STRATA: FAILING = e104's powered band (bottom decile of valid DPs by
    per-DP full-16 attention-mass AUC — set must equal e104/e106's stored
    {0,1,3,9,10,11,12,13,27,48}); SUCCEEDING = the rest; margin quartiles
    Q0..Q3 (e084's stratification) as texture.

  - VALUE-SIDE FEATURES (computed BEFORE any intervention, from ONE clean
    feature forward whose op order mirrors _manual_chunk2 — G8-gated):
      READOUT DIRECTIONS (T038's method): for the decision's top-2
      candidate tokens t1/t2 (top-2 of the clean decision-row logits),
      dir(t) = normalize(ln_f.weight * lm_head.weight[t]) — the LN-
      attributed unembedding row mapped into residual space (T038's
      instrument; its raw and LN-attributed variants agreed 0.098/0.100).
      ENTRY WRITE DIRECTIONS: write_l(p) = c_proj_l.weight @ v_l[p] — the
      entry's per-layer value vector (head-concat, 128-d) projected by
      c_proj into residual space = the residual write the entry would make
      if fully attended. Attention-free by construction (no prob weights):
      pure content.
      (a) V-CONTENT MATCH (PRIMARY): per entry, mean over the 4 layers of
          mean over t in {t1,t2} of cos(write_l(p), dir(t)) — signed.
      (b) AGE-WEIGHTED CONTENT (PRIMARY): (a) x w097(age), e097's recency
          multiplier (k=128 per-third damage ratios vs uniform: new
          [age<=213] 2.8426, mid [214-330] 2.0155, old [>=331] 1.7562).
          Ages <97 (younger than e097's measured band 97-447) are CLIPPED
          to the new-third value — conservative, flagged in metrics.
      ATTENTION BASELINES: full-16-head mass (e100's primary — the
      REGISTERED comparator) and L1+L2 mass (e106's early comparator, the
      task's quoted 0.779; reported alongside, gate-checked vs e104).
      SENSITIVITIES (descriptive, no bars): t1-only content; differential
      cos(t1)-cos(t2); |cos| variant; max-over-layers; raw-v (no c_proj);
      literal ln_f(W_U row) transform variant; age-only (-age); age-
      weighted content in e097's k=64 form (2.2281/0.6173/0.5411).

  - READOUTS: per decision point, tie-averaged Mann-Whitney AUC of each
    feature vs the opened set; per stratum the pair-weighted pooled AUC
    (pairs compared only within a decision); bootstrap 95% CIs
    (decision-point resampling within stratum, n=1000, seed 0); paired
    within-stratum bootstrap of (value - attention) pooled differences
    (joint DP resampling).

REGISTERED BARS (frozen; evaluated in this order; primary attention
baseline = full-16 mass; the task-quoted 0.779 is e106's L12 comparator —
margins vs it are reported alongside and flagged if the verdict differs):
  1. CONTENT-SELECTED AT THE MARGIN iff at the FAILING stratum EITHER
     primary value feature has AUC >= 0.75 AND (AUC - AUC_att_full16)
     >= 0.05 AND CI separation (feature CI lo > attention CI hi):
     the second channel is value-side — near-tie decisions read by
     content match, not routing.
  2. ROUTING-ONLY STANDS iff BOTH primary value features satisfy
     AUC <= AUC_att_full16 + 0.05 in ALL strata (failing, succeeding,
     Q0..Q3): the residual is neither routing nor simple content —
     register the honest next family.
  3. else MIXED: report the texture.

GATES: G2 params (873,472), G1 val CE (vs e053c 1.5227 +/- 0.02), G0b
incremental-KV vs full recompute (max prob dev < 1e-4), G8 feature-forward
identity vs manual_logits2 clean row (max |dev| < 1e-4, first DP), G5a
e084 identity (per-DP V-zero opened counts vs stored flips_vz, >= 90/100
exact), G5b e100 identity (per-DP full-16 AUCs vs stored auc_att,
max dev < 1e-9), G5c e104 identity (failing set equal AND per-DP
full/L12 AUC dev < 1e-9).

Run:     python lab/e107_value_side.py     (E107_SMOKE=1 for smoke)
Outputs: runs/e107/{metrics.json, value_side.png}
Envelope: NO training; CPU-only (CUDA_VISIBLE_DEVICES forced -1, torch.cuda
stubbed pre-import, e070/e084 pattern); 8 torch threads; single step, well
under 30 min. No NOTES/THINKING/QUEUE/STATE edits; no commit (the
dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu" (e053b/e069/e070/
# e084/e100/e106 pattern)
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

# ---- design constants (registered; ALL verbatim e084/e100/e106) -------------
Q_LO, Q_HI = 96, 510                      # candidate query positions
PER_QUARTILE = 25                         # 25 per margin quartile
N_DP = 4 * PER_QUARTILE                   # 100 decision points
SEED_STRAT = 2084                         # margin-stratified sampling seed
YOUNG = list(range(1, 11))                # ages 1-10 (all)
MID = list(range(11, 61))                 # ages 11-60 (all)
N_OLD = 20                                # old entries per decision point
OLD_SEED_BASE = 9000                      # old-age rng seed = base + dp_index
N_ENTRIES = len(YOUNG) + len(MID) + N_OLD  # 80
DECILE_FRAC = 0.1                         # e104's failing-band fraction
YOUNG_AGE_MAX = 10                        # opened-age texture (e104)

# ---- e097's recency multiplier (REGISTERED age weighting) -------------------
# k=128 per-third damage ratios vs uniform (runs/e097/metrics.json
# summary.ratios.k128): new (final-frame ages 97-213) 2.8426, mid (214-330)
# 2.0155, old (331-447) 1.7562. Ages <97 are clipped to the new-third value
# (conservative extrapolation below e097's measured band; flagged).
W097_K128 = dict(new=2.8425543507291278, mid=2.01550967034141,
                 old=1.7562312376475617)
W097_K64 = dict(new=2.2280631478648045, mid=0.6172614166926864,
                old=0.5410910233851307)   # sensitivity form

# ---- REGISTERED bars (FROZEN; evaluated in order) ---------------------------
BAR_ABS = 0.75          # value-side AUC at failing must be >=
BAR_MARGIN = 0.05       # (value - attention_full16) at failing must be >=
BAR_ROUTING_SLACK = 0.05  # routing-only: value <= attention + this, all strata
STRATA_ALL = ["failing", "succeeding", "Q0", "Q1", "Q2", "Q3"]

ATT_FULL = "att_full16"     # REGISTERED attention baseline (e100's primary)
ATT_L12 = "att_L12"         # e106's early comparator (the quoted 0.779)
FEAT_CONTENT = "content"    # primary value-side (a)
FEAT_AGEW = "ageweight"     # primary value-side (b)
PRIMARIES = [FEAT_CONTENT, FEAT_AGEW]
SENSITIVITIES = ["content_t1", "content_diff", "content_abs",
                 "content_maxlayer", "content_rawv", "content_litlnf",
                 "age_only", "ageweight_k64"]
ALL_VALUE_FEATS = PRIMARIES + SENSITIVITIES

# ---- stored references (G5 identity gates) ----------------------------------
E084_METRICS = REPO / "runs" / "e084" / "metrics.json"
E100_METRICS = REPO / "runs" / "e100" / "metrics.json"
E104_METRICS = REPO / "runs" / "e104" / "metrics.json"
CKPT = REPO / "runs" / "checkpoints" / "e053c_ctx512.pt"
E053C_VAL_CE = 1.5226792494455974

SMOKE = os.environ.get("E107_SMOKE", "") == "1"
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
    """e069/e084/e100's _manual_chunk VERBATIM (vswap/kdrop arms inert here —
    e107 uses only the clean row and V-zero masks, but the instrument is kept
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
    """e069/e084's generate_batch VERBATIM + per-step decision logits.
    Returns (idx, final_logits, hist (B, gen_stop-64, V))."""
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


# --------------------------------------------- e107's own (gated) instruments

@torch.no_grad()
def value_features(net: TinyGPT, ctx: torch.Tensor):
    """ONE clean forward over ctx (1-D token ids, length T); op order mirrors
    _manual_chunk2 exactly (G8-gated). Records, BEFORE any intervention:
      - att_last (L, H, T): softmax attention probabilities of the decision
        row (last position) over all positions;
      - v_pre_all (L, T, C): each layer's VALUE vectors at every position,
        head-concatenated (the raw content the V-zero instrument removes);
      - logits (V,): the decision-row logits (for top-2 + G8)."""
    T = ctx.shape[0]
    x = net.wte(ctx[None]) + net.wpe(torch.arange(T))[None]
    H = net.cfg.n_head
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    att_last, v_pre_all = [], []
    for blk in net.h:
        xh = blk.ln1(x)
        qkv = blk.attn.c_attn(xh)
        C = qkv.shape[-1] // 3
        d = C // H
        q, k, v = qkv.split(C, dim=2)
        v_pre_all.append(v[0].clone())                       # (T, C)
        q = q.view(1, T, H, d).transpose(1, 2)               # (1,H,T,d)
        k = k.view(1, T, H, d).transpose(1, 2)
        v = v.view(1, T, H, d).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
        att = att.masked_fill(causal, float("-inf"))
        probs = torch.softmax(att, dim=-1)                   # (1,H,T,T)
        y = (probs @ v).transpose(1, 2).reshape(1, T, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
        att_last.append(probs[0, :, -1, :].clone())          # (H, T)
    logits = net.lm_head(net.ln_f(x[0, -1]))
    return torch.stack(att_last, 0), torch.stack(v_pre_all, 0), logits


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


def readout_dirs(net: TinyGPT, t1: int, t2: int):
    """T038's method: the top-2 tokens' unembedding rows mapped through ln_f
    into residual space. PRIMARY = LN-attributed form (ln_f gain applied
    elementwise); returns (dir1, dir2, lit1, lit2) — the primary pair and the
    literal ln_f(row) transform pair (sensitivity). All L2-normalized."""
    gain = net.ln_f.weight.detach()
    wu = net.lm_head.weight.detach()                    # (V, C)
    d1 = gain * wu[t1]
    d2 = gain * wu[t2]
    l1 = torch.nn.functional.layer_norm(wu[t1][None], (wu.shape[1],),
                                        net.ln_f.weight.detach(),
                                        net.ln_f.bias.detach())[0]
    l2 = torch.nn.functional.layer_norm(wu[t2][None], (wu.shape[1],),
                                        net.ln_f.weight.detach(),
                                        net.ln_f.bias.detach())[0]
    norm = lambda v: v / max(float(v.norm()), 1e-12)    # noqa: E731
    return norm(d1), norm(d2), norm(l1), norm(l2)


@torch.no_grad()
def content_scores(net: TinyGPT, v_pre_all: torch.Tensor, dir1, dir2,
                   lit1, lit2):
    """Per-position value-side content scores from the clean forward's value
    stack. Returns dict of (T,) arrays:
      cos_t1/cos_t2: mean over layers of cos(write_l(p), dir(t)) — write_l =
        c_proj_l.weight @ v_l[p] (the entry's potential residual write);
      cos_t1_rawv/cos_t2_rawv: same WITHOUT c_proj (raw head-concat v);
      cos_t1_lit/cos_t2_lit: literal ln_f-transform readout dirs;
      cos_t1_maxl: max-over-layers variant (primary dirs).
    """
    L, T, C = v_pre_all.shape
    out = dict(cos_t1=np.zeros(T), cos_t2=np.zeros(T),
               cos_t1_rawv=np.zeros(T), cos_t2_rawv=np.zeros(T),
               cos_t1_lit=np.zeros(T), cos_t2_lit=np.zeros(T),
               cos_t1_maxl=np.full(T, -np.inf), cos_t2_maxl=np.full(T, -np.inf))
    stack_t1, stack_t2, stack_r1, stack_r2 = [], [], [], []
    stack_l1, stack_l2 = [], []
    for li, blk in enumerate(net.h):
        W = blk.attn.c_proj.weight.detach()             # (C, C), bias-free
        write = v_pre_all[li] @ W.T                     # (T, C)
        write = write / write.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        rawv = v_pre_all[li] / v_pre_all[li].norm(dim=-1,
                                                  keepdim=True).clamp_min(1e-12)
        stack_t1.append(write @ dir1)
        stack_t2.append(write @ dir2)
        stack_r1.append(rawv @ dir1)
        stack_r2.append(rawv @ dir2)
        stack_l1.append(write @ lit1)
        stack_l2.append(write @ lit2)
    out["cos_t1"] = torch.stack(stack_t1, 0).mean(0).numpy()
    out["cos_t2"] = torch.stack(stack_t2, 0).mean(0).numpy()
    out["cos_t1_rawv"] = torch.stack(stack_r1, 0).mean(0).numpy()
    out["cos_t2_rawv"] = torch.stack(stack_r2, 0).mean(0).numpy()
    out["cos_t1_lit"] = torch.stack(stack_l1, 0).mean(0).numpy()
    out["cos_t2_lit"] = torch.stack(stack_l2, 0).mean(0).numpy()
    out["cos_t1_maxl"] = torch.stack(stack_t1, 0).amax(0).numpy()
    out["cos_t2_maxl"] = torch.stack(stack_t2, 0).amax(0).numpy()
    return out


def w097(age_arr: np.ndarray, form: dict) -> np.ndarray:
    """e097's recency multiplier applied to ages (final-frame convention:
    new <= 213, mid 214-330, old >= 331; ages <97 clipped to new —
    extrapolation below e097's measured band, flagged in metrics)."""
    a = np.asarray(age_arr, float)
    w = np.full(a.shape, form["new"])
    w[(a >= 214) & (a <= 330)] = form["mid"]
    w[a >= 331] = form["old"]
    return w


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
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
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
    out_dir = run_dir("e107")
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

    # ---- the probe: per decision point, value-side + attention features
    #      BEFORE intervention, then the ground-truth opened set
    t_probe = time.time()
    per_dp = []
    feats = [ATT_FULL, ATT_L12] + ALL_VALUE_FEATS
    feats_auc = {f: [] for f in feats}
    opened_sizes = []
    t1_mismatch = 0
    content_diag = []       # texture: mean |cos| per DP (signal sanity)

    for i, (b, q, _, qt) in enumerate(dps):
        entries, ps = entry_positions(q, i)

        # (1) features from ONE clean forward (pre-intervention)
        att_last, v_pre_all, q_logits = value_features(net, idx16[b, :q + 1])
        L, H, T = att_last.shape
        ps_np = np.asarray(ps)
        att_lh = att_last.numpy()                        # (L,H,T)
        m_full = att_lh.sum((0, 1))                      # 16-head mass (e100)
        m_L12 = att_lh[1:3].sum((0, 1))                  # e106's early L1+L2

        # top-2 candidate tokens from the SAME clean forward
        top2 = torch.topk(q_logits, 2)
        t1f, t2f = int(top2.indices[0]), int(top2.indices[1])
        dir1, dir2, lit1, lit2 = readout_dirs(net, t1f, t2f)
        cs = content_scores(net, v_pre_all, dir1, dir2, lit1, lit2)

        # assemble per-entry feature scores (registered forms)
        s_content = 0.5 * (cs["cos_t1"] + cs["cos_t2"])
        s_content_lit = 0.5 * (cs["cos_t1_lit"] + cs["cos_t2_lit"])
        s_content_rawv = 0.5 * (cs["cos_t1_rawv"] + cs["cos_t2_rawv"])
        s_maxlayer = 0.5 * (cs["cos_t1_maxl"] + cs["cos_t2_maxl"])
        ages_np = np.asarray(entries, float)
        w_k128 = w097(ages_np, W097_K128)
        w_k64 = w097(ages_np, W097_K64)
        scores = {
            ATT_FULL: m_full[ps_np],
            ATT_L12: m_L12[ps_np],
            FEAT_CONTENT: s_content[ps_np],
            FEAT_AGEW: s_content[ps_np] * w_k128,
            "content_t1": cs["cos_t1"][ps_np],
            "content_diff": (cs["cos_t1"] - cs["cos_t2"])[ps_np],
            "content_abs": np.abs(s_content[ps_np]),
            "content_maxlayer": s_maxlayer[ps_np],
            "content_rawv": s_content_rawv[ps_np],
            "content_litlnf": s_content_lit[ps_np],
            "age_only": -ages_np,
            "ageweight_k64": s_content[ps_np] * w_k64,
        }

        # (2) ground truth: V-zero flips at the e084 bars
        opened, t1, margin = opened_set(net, idx16, b, q, entries, ps)
        opened_sizes.append(int(opened.sum()))
        if t1 != t1f:
            t1_mismatch += 1

        n_open = int(opened.sum())
        rec = dict(dp=i, b=int(b), q=int(q), quartile=int(qt), margin=margin,
                   t1=t1, t2=t2f, n_opened=n_open,
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
        content_diag.append(float(np.abs(s_content).mean()))
        per_dp.append(rec)
        if (i + 1) % 10 == 0 or i == len(dps) - 1:
            rate = (time.time() - t_probe) / (i + 1)
            log(f"probe {i + 1}/{len(dps)} decision points ({rate:.2f}s/dp, "
                f"ETA {rate * (len(dps) - i - 1):.0f}s)")

    log(f"content signal sanity: mean |cos(content)| across DPs "
        f"{np.mean(content_diag):.4f}; t1 top-2 mismatches (feature fwd vs "
        f"manual): {t1_mismatch}/{len(dps)}")

    # ---- G8: feature-forward identity (first DP) vs manual_logits2 clean row
    b0, q0 = dps[0][0], dps[0][1]
    _, _, q_log0 = value_features(net, idx16[b0, :q0 + 1])
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
            if ref is not None and f"auc_{ATT_FULL}" in r:
                d = abs(r[f"auc_{ATT_FULL}"] - ref)
                devs.append(d)
                ok_n += int(d < 1e-9)
        max_dev = max(devs) if devs else float("nan")
        gates["G5_e100_auc_identity"] = dict(
            n_compared=len(devs), n_match_lt_1e9=ok_n, max_auc_dev=max_dev,
            tol=1e-9, ok=bool(ok_n == len(devs) and max_dev < 1e-9))
        log(f"G5 vs e100 stored per-DP AUCs: {ok_n}/{len(devs)} within 1e-9 "
            f"(max dev {max_dev:.2e}) -> "
            f"{'PASS' if gates['G5_e100_auc_identity']['ok'] else 'FAIL'}")

    # ---- failing band recompute (e104's powered band, same rule)
    valid_recs = [r for r in per_dp if f"auc_{ATT_FULL}" in r]
    n_valid = len(valid_recs)
    order = sorted(valid_recs, key=lambda r: (r[f"auc_{ATT_FULL}"], r["dp"]))
    n_fail = max(1, int(round(DECILE_FRAC * n_valid)))
    fail_dps = {r["dp"] for r in order[:n_fail]}
    fail_cut_auc = order[n_fail - 1][f"auc_{ATT_FULL}"]
    log(f"recomputed failure band: {n_fail}/{n_valid} worst DPs "
        f"(cut AUC {fail_cut_auc:.3f}) -> dp {sorted(fail_dps)}")

    if not SMOKE:
        e104 = json.loads(E104_METRICS.read_text(encoding="utf-8"))
        ref_fail = set(e104["census"]["fail_dps"])
        e104_full = {r["dp"]: r.get("auc_full")
                     for r in e104["per_decision_point"]}
        e104_L12 = {r["dp"]: r.get("auc_L12")
                    for r in e104["per_decision_point"]}
        devs_f, devs_12 = [], []
        for r in valid_recs:
            rf = e104_full.get(r["dp"])
            r12 = e104_L12.get(r["dp"])
            if rf is not None:
                devs_f.append(abs(r[f"auc_{ATT_FULL}"] - rf))
            if r12 is not None:
                devs_12.append(abs(r[f"auc_{ATT_L12}"] - r12))
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

    # ================= the registered analyses ==============================
    def stratum_of(r):
        return ("failing" if r["dp"] in fail_dps else "succeeding",
                f"Q{r['quartile']}")

    buckets = {s: {f: [] for f in feats} for s in STRATA_ALL}
    for r in valid_recs:
        s_main, s_q = stratum_of(r)
        pairs_r = r["n_opened"] * (N_ENTRIES - r["n_opened"])
        for f in feats:
            buckets[s_main][f].append((r[f"auc_{f}"], pairs_r))
            buckets[s_q][f].append((r[f"auc_{f}"], pairs_r))

    strat_tbl = {}
    for s in STRATA_ALL:
        strat_tbl[s] = {}
        for f in feats:
            pooled = pooled_auc(buckets[s][f])
            ci = bootstrap_pooled(buckets[s][f])
            strat_tbl[s][f] = dict(auc=pooled, ci=list(ci),
                                   n_dp=len(buckets[s][f]))

    # paired within-stratum value-minus-attention differences (texture)
    paired_diffs = {}
    for s in STRATA_ALL:
        paired_diffs[s] = {}
        for f in ALL_VALUE_FEATS:
            la = buckets[s][f]
            lb = buckets[s][ATT_FULL]
            paired_diffs[s][f] = list(bootstrap_paired_diff(la, lb))

    for s in STRATA_ALL:
        msg = f"STRATUM {s:10s} (n={strat_tbl[s][ATT_FULL]['n_dp']}): "
        msg += " | ".join(
            f"{f} {strat_tbl[s][f]['auc']:.3f}"
            for f in [ATT_FULL, ATT_L12, FEAT_CONTENT, FEAT_AGEW])
        log(msg)

    # ---- REGISTERED decision (frozen bars, in order; see docstring)
    def bar_content(feat: str) -> dict:
        v = strat_tbl["failing"][feat]
        a = strat_tbl["failing"][ATT_FULL]
        a12 = strat_tbl["failing"][ATT_L12]
        cond_abs = bool(v["auc"] >= BAR_ABS)
        cond_margin = bool(v["auc"] - a["auc"] >= BAR_MARGIN)
        cond_ci = bool(v["ci"][0] > a["ci"][1])
        margin_L12 = float(v["auc"] - a12["auc"])
        return dict(feat=feat, auc=v["auc"], ci=v["ci"],
                    att_full16_auc=a["auc"], att_full16_ci=a["ci"],
                    att_L12_auc=a12["auc"],
                    margin_vs_full16=float(v["auc"] - a["auc"]),
                    margin_vs_L12=margin_L12,
                    paired_diff_ci=paired_diffs["failing"][feat],
                    cond_abs_ge_075=cond_abs,
                    cond_margin_ge_005_vs_full16=cond_margin,
                    cond_ci_separated_vs_full16=cond_ci,
                    fires=bool(cond_abs and cond_margin and cond_ci),
                    fires_vs_L12_too=bool(
                        cond_abs
                        and margin_L12 >= BAR_MARGIN
                        and v["ci"][0] > a12["ci"][1]))

    bar_report = {f: bar_content(f) for f in PRIMARIES}
    any_fires = any(br["fires"] for br in bar_report.values())
    fires_vs_both = all(br["fires_vs_L12_too"]
                        for br in bar_report.values() if br["fires"])

    routing_only_ok = all(
        strat_tbl[s][f]["auc"] <= strat_tbl[s][ATT_FULL]["auc"]
        + BAR_ROUTING_SLACK
        for s in STRATA_ALL for f in PRIMARIES)
    exceeded = [(s, f) for s in STRATA_ALL for f in PRIMARIES
                if strat_tbl[s][f]["auc"]
                > strat_tbl[s][ATT_FULL]["auc"] + BAR_ROUTING_SLACK]

    if any_fires:
        which = [f for f in PRIMARIES if bar_report[f]["fires"]]
        fires = "CONTENT-SELECTED AT THE MARGIN"
        verdict = (f"CONTENT-SELECTED AT THE MARGIN: {which} at the failing "
                   f"stratum AUC >= {BAR_ABS}, exceeds full-16 attention by "
                   f">= {BAR_MARGIN} with CI separation — the second channel "
                   "is value-side: near-tie decisions read by content match, "
                   "not routing."
                   + ("" if fires_vs_both
                      else " [FLAG: fires vs full16 baseline but not vs the "
                           "L12 0.779 comparator — baseline-sensitive.]"))
    elif routing_only_ok:
        fires = "ROUTING-ONLY STANDS"
        verdict = (f"ROUTING-ONLY STANDS: both primary value features sit "
                   f"within {BAR_ROUTING_SLACK} of (or below) full-16 "
                   "attention in every stratum — the residual is neither "
                   "routing nor simple content; register the honest next "
                   "family.")
    else:
        fires = "MIXED"
        verdict = (f"MIXED: no value feature clears the failing-stratum bar "
                   f"(>= {BAR_ABS} AND +{BAR_MARGIN} with CI separation), but "
                   f"value-side exceeds attention+{BAR_ROUTING_SLACK} in "
                   f"{exceeded} without clearing the margin bar — report the "
                   "texture.")

    # honesty riders (always computed, descriptive)
    age_only_vs_content = {s: dict(
        age_only=strat_tbl[s]["age_only"]["auc"],
        content=strat_tbl[s][FEAT_CONTENT]["auc"],
        ageweight=strat_tbl[s][FEAT_AGEW]["auc"]) for s in STRATA_ALL}
    best_sens_failing = max(
        SENSITIVITIES, key=lambda f: strat_tbl["failing"][f]["auc"])

    n_zero_open = int(sum(1 for s in opened_sizes if s == 0))
    total_opened = int(sum(opened_sizes))
    log(f"ground truth: valid DPs {n_valid}/{len(dps)} (0-opened "
        f"{n_zero_open}); total opened cells {total_opened}; median "
        f"opened/DP {np.median(opened_sizes):.0f}")
    log(f"REGISTERED DECISION [{fires}]: {verdict}")

    # ---------------------------------------------------------------- metrics
    metrics = dict(
        experiment="e107_value_side",
        purpose="VALUE-SIDE residual probe (T058's registered re-aim off the "
                "routing-mass family): per decision point (e084/e100/e104/"
                "e106 battery + DP selection + 80-entry bands + V-zero "
                "ground truth, all bit-identity-gated), compute CONTENT "
                "features BEFORE intervention — (a) V-CONTENT MATCH: signed "
                "cosine between the entry's potential residual write "
                "c_proj(v_l[p]) and the decision's top-2 readout directions "
                "ln_f-attributed lm_head rows (T038's method), mean over 4 "
                "layers and 2 tokens; (b) AGE-WEIGHTED content: (a) times "
                "e097's k128 recency multipliers (2.843/2.016/1.756 by age "
                "third, ages<97 clipped to new). Per-stratum AUC vs the "
                "attention baselines (full-16 REGISTERED comparator; L1+L2 "
                "= the quoted 0.779 reported alongside). FROZEN BARS: "
                "CONTENT-SELECTED AT THE MARGIN iff a primary value feature "
                "at failing >= 0.75 AND >= attention+0.05 AND CI separation; "
                "ROUTING-ONLY STANDS iff both primaries <= attention+0.05 in "
                "ALL strata; else MIXED.",
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
            failing_band=dict(rule="bottom decile of valid DPs by per-DP "
                                  "full-16 attention-mass AUC (e104 verbatim)",
                              frac=DECILE_FRAC, n_fail=n_fail,
                              cut_auc=fail_cut_auc,
                              fail_dps=sorted(fail_dps), n_valid=n_valid),
            ground_truth="V-zero flip at the e084 bars (v := 0 at the "
                         "entry's position, every layer, every head, all "
                         "queries; argmax flip vs the same call's clean row)",
            readout_directions="dir(t) = normalize(ln_f.weight * "
                               "lm_head.weight[t]) for t in the clean "
                               "forward's top-2 (T038's LN-attributed "
                               "method; literal ln_f(row) kept as "
                               "sensitivity)",
            entry_write="write_l(p) = c_proj_l.weight @ v_l[p] (bias-free "
                        "linear; attention-free content projection)",
            feature_a=f"{FEAT_CONTENT}: mean over 4 layers x {{t1,t2}} of "
                      "cos(write_l(p), dir(t)), signed",
            feature_b=f"{FEAT_AGEW}: feature_a x e097 k128 recency "
                      "multiplier (new<=213: 2.8426, mid 214-330: 2.0155, "
                      "old>=331: 1.7562; ages<97 CLIPPED to new — "
                      "extrapolation below e097's measured band, flagged)",
            attention_baselines=dict(
                full16="16-layer-head mass (e100 primary; REGISTERED "
                       "comparator)",
                L12="L1+L2 8-head mass (e106's early comparator, the "
                    "quoted 0.779)"),
            sensitivities=SENSITIVITIES,
            auc="Mann-Whitney tie-averaged; pooled = pair-weighted across "
                "decision points (pairs only within a decision); bootstrap "
                "CI = decision-point resampling within stratum n=1000 "
                "seed 0; paired diffs resample DPs jointly"),
        gates=gates,
        ground_truth=dict(
            opened_per_dp_median=float(np.median(opened_sizes)),
            total_opened_cells=total_opened,
            n_dp_zero_opened=n_zero_open, n_valid_dp=n_valid,
            t1_feature_vs_manual_mismatches=t1_mismatch,
            content_signal=dict(mean_abs_cos_over_dps=float(
                np.mean(content_diag)))),
        stratum_auc=strat_tbl,
        paired_diff_vs_full16=paired_diffs,
        age_only_vs_content=age_only_vs_content,
        best_sensitivity_at_failing=dict(
            feature=best_sens_failing,
            auc=strat_tbl["failing"][best_sens_failing]["auc"],
            ci=strat_tbl["failing"][best_sens_failing]["ci"]),
        per_decision_point=per_dp,
        registered_decision=dict(
            frozen_bars=dict(
                content_selected=f"at failing: EITHER primary value AUC >= "
                                 f"{BAR_ABS} AND (AUC - AUC_att_full16) >= "
                                 f"{BAR_MARGIN} AND value CI lo > attention "
                                 "CI hi",
                routing_only=f"BOTH primaries: AUC <= AUC_att_full16 + "
                             f"{BAR_ROUTING_SLACK} in ALL of {STRATA_ALL}",
                else_="MIXED (report texture)"),
            baseline_note="primary attention baseline = full-16 mass "
                          "(e100's primary); the task-quoted 0.779 is e106's "
                          "L12 comparator — margins vs it reported per "
                          "feature (fires_vs_L12_too); verdict-vs-baseline "
                          "divergence would be FLAGGED",
            per_feature=bar_report,
            any_primary_fires=bool(any_fires),
            routing_only_conditions_met=bool(routing_only_ok),
            strata_exceeding_attention=exceeded,
            fires=fires, verdict=verdict),
    )
    save_json(out_dir / f"metrics{suffix}.json", metrics)
    log(f"metrics{suffix}.json written")
    plot(out_dir / f"value_side{suffix}.png", metrics)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot

def plot(path: Path, M: dict):
    dec = M["registered_decision"]
    st = M["stratum_auc"]
    pd_diff = M["paired_diff_vs_full16"]

    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # ---- panel 1 (REQUIRED): per-stratum AUC bars, attention vs content
    #      vs age-weighted content
    strata = ["failing", "succeeding", "Q0", "Q1", "Q2", "Q3"]
    groups = [(ATT_FULL, "attention (full-16)", "tab:blue"),
              (FEAT_CONTENT, "V-content match", "tab:orange"),
              (FEAT_AGEW, "age-weighted content", "tab:green")]
    width = 0.26
    xs = np.arange(len(strata))
    for gi, (f, lbl, col) in enumerate(groups):
        vals = [st[s][f]["auc"] for s in strata]
        cis = [st[s][f]["ci"] for s in strata]
        lo = [max(v - c[0], 0) for v, c in zip(vals, cis)]
        hi = [max(c[1] - v, 0) for v, c in zip(vals, cis)]
        ax1.bar(xs + (gi - 1) * width, vals, width,
                yerr=[lo, hi], color=col, alpha=0.85, capsize=3,
                label=lbl)
    ax1.axhline(0.5, color="k", ls=":", lw=1.2)
    ax1.set_xticks(xs, strata)
    ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("pooled within-decision AUC")
    ax1.legend(fontsize=9, loc="lower left")
    ax1.set_title("(a) per-stratum AUC — attention vs value-side features "
                  "(bootstrap CI, DP resampling)", fontsize=10)

    # ---- panel 2: failing-stratum focus with both baselines + bars
    names = ["attention\nfull-16 (baseline)", "attention\nL1+L2 (0.779)",
             "V-content\nmatch", "age-weighted\ncontent"]
    keys = [ATT_FULL, ATT_L12, FEAT_CONTENT, FEAT_AGEW]
    vals = [st["failing"][f]["auc"] for f in keys]
    cis = [st["failing"][f]["ci"] for f in keys]
    lo = [max(v - c[0], 0) for v, c in zip(vals, cis)]
    hi = [max(c[1] - v, 0) for v, c in zip(vals, cis)]
    cols = ["tab:blue", "lightsteelblue", "tab:orange", "tab:green"]
    ax2.bar(np.arange(4), vals, 0.55, yerr=[lo, hi], color=cols,
            alpha=0.9, capsize=4)
    att = vals[0]
    ax2.axhline(BAR_ABS, color="tab:green", ls="--", lw=1.4)
    ax2.axhline(att + BAR_MARGIN, color="tab:red", ls="--", lw=1.4)
    ax2.text(3.45, BAR_ABS + 0.01, f"{BAR_ABS} content bar", fontsize=8,
             ha="right", color="tab:green")
    ax2.text(3.45, att + BAR_MARGIN + 0.01,
             f"attention+{BAR_MARGIN} = {att + BAR_MARGIN:.3f}", fontsize=8,
             ha="right", color="tab:red")
    ax2.axhline(0.5, color="k", ls=":", lw=1.2)
    ax2.set_xticks(np.arange(4), names, fontsize=8)
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("pooled AUC at the failing stratum")
    ax2.set_title("(b) failing stratum (e104's bottom decile) — registered "
                  "bars", fontsize=10)

    # ---- panel 3: sensitivity variants at the failing stratum
    sens = [ATT_FULL] + ALL_VALUE_FEATS
    labels = ["att full-16 (baseline)", "content (PRIMARY)",
              "age-weighted (PRIMARY)", "content t1-only",
              "content t1-t2 diff", "|content|", "content max-layer",
              "content raw-v", "content literal ln_f", "age only (-age)",
              "age-weighted k64"]
    sv = [st["failing"][f]["auc"] for f in sens]
    sci = [st["failing"][f]["ci"] for f in sens]
    slo = [max(v - c[0], 0) for v, c in zip(sv, sci)]
    shi = [max(c[1] - v, 0) for v, c in zip(sv, sci)]
    colors = (["tab:blue"] + ["tab:orange", "tab:green"]
              + ["tab:gray"] * (len(sens) - 3))
    ypos = np.arange(len(sens))[::-1]
    ax3.barh(ypos, sv, xerr=[slo, shi], color=colors, alpha=0.85,
             capsize=3, height=0.62)
    ax3.set_yticks(ypos, labels, fontsize=8)
    ax3.axvline(0.5, color="k", ls=":", lw=1.2)
    ax3.axvline(sv[0], color="tab:blue", ls="--", lw=1.2)
    for y, v in zip(ypos, sv):
        ax3.text(v + 0.012, y, f"{v:.3f}", va="center", fontsize=7.5)
    ax3.set_xlim(0, 1.05)
    ax3.set_xlabel("pooled AUC at failing")
    ax3.set_title("(c) the full value-side family at the failing stratum "
                  "(descriptive)", fontsize=10)

    # ---- panel 4: per-DP scatter content vs attention
    a_att = [r[f"auc_{ATT_FULL}"] for r in M["per_decision_point"]
             if f"auc_{ATT_FULL}" in r]
    a_con = [r[f"auc_{FEAT_CONTENT}"] for r in M["per_decision_point"]
             if f"auc_{FEAT_CONTENT}" in r]
    fail_set = set(M["protocol"]["failing_band"]["fail_dps"])
    is_fail = [r["dp"] in fail_set for r in M["per_decision_point"]
               if f"auc_{ATT_FULL}" in r]
    a_att_f = [a for a, f in zip(a_att, is_fail) if f]
    a_con_f = [a for a, f in zip(a_con, is_fail) if f]
    a_att_s = [a for a, f in zip(a_att, is_fail) if not f]
    a_con_s = [a for a, f in zip(a_con, is_fail) if not f]
    ax4.scatter(a_att_s, a_con_s, s=22, alpha=0.6, color="tab:gray",
                label=f"succeeding (n={len(a_att_s)})")
    ax4.scatter(a_att_f, a_con_f, s=60, alpha=0.9, color="tab:red",
                marker="D", label=f"failing (n={len(a_att_f)})")
    lim = [0, 1]
    ax4.plot(lim, lim, "k--", lw=0.9)
    ax4.set_xlim(lim)
    ax4.set_ylim(lim)
    ax4.set_xlabel("per-decision AUC — attention (full-16)")
    ax4.set_ylabel("per-decision AUC — V-content match")
    ax4.legend(fontsize=8, loc="upper left")
    ax4.set_title("(d) per-decision AUCs — above diag: content better",
                  fontsize=10)

    # ---- panel 5: paired diff (value - attention) distributions per stratum
    feats_pd = [FEAT_CONTENT, FEAT_AGEW]
    cols_pd = ["tab:orange", "tab:green"]
    for gi, (f, col) in enumerate(zip(feats_pd, cols_pd)):
        diffs = [st[s][f]["auc"] - st[s][ATT_FULL]["auc"] for s in strata]
        lo_c = [pd_diff[s][f][0] for s in strata]
        hi_c = [pd_diff[s][f][1] for s in strata]
        ax5.errorbar(np.arange(len(strata)) + (gi - 0.5) * 0.18, diffs,
                     yerr=[np.maximum(np.asarray(diffs) - np.asarray(lo_c), 0),
                           np.maximum(np.asarray(hi_c) - np.asarray(diffs), 0)],
                     fmt="o-", ms=6, lw=1.2, color=col, capsize=3,
                     label=f.split("_")[0] if "_" in f else f)
    ax5.axhline(0, color="k", ls="--", lw=1.1)
    ax5.axhline(BAR_MARGIN, color="tab:red", ls=":", lw=1.2)
    ax5.text(5.4, BAR_MARGIN + 0.006, f"+{BAR_MARGIN} content bar",
             fontsize=8, ha="right", color="tab:red")
    ax5.set_xticks(np.arange(len(strata)), strata)
    ax5.set_ylabel("pooled AUC(value) - pooled AUC(attention)")
    ax5.legend(fontsize=9)
    ax5.set_title("(e) paired value-minus-attention margins (joint DP "
                  "bootstrap CI)", fontsize=10)

    # ---- panel 6: is the age-weighting just age? age-only vs content
    xs6 = np.arange(len(strata))
    for gi, (f, lbl, col) in enumerate([
            ("age_only", "age only (-age)", "tab:purple"),
            (FEAT_CONTENT, "content", "tab:orange"),
            (FEAT_AGEW, "age-weighted content", "tab:green")]):
        vals6 = [st[s][f]["auc"] for s in strata]
        ax6.bar(xs6 + (gi - 1) * width, vals6, width, color=col, alpha=0.85,
                label=lbl)
    ax6.axhline(0.5, color="k", ls=":", lw=1.2)
    ax6.set_xticks(xs6, strata)
    ax6.set_ylim(0, 1.02)
    ax6.set_ylabel("pooled AUC")
    ax6.legend(fontsize=9, loc="lower left")
    ax6.set_title("(f) does age-weighting add content, or just age? "
                  "(descriptive)", fontsize=10)

    fb = dec["per_feature"][FEAT_CONTENT]
    fa = dec["per_feature"][FEAT_AGEW]
    fig.suptitle(
        f"E107 VALUE-SIDE RESIDUAL PROBE | {dec['fires']} | failing: att "
        f"{st['failing'][ATT_FULL]['auc']:.3f} vs content "
        f"{fb['auc']:.3f} (margin {fb['margin_vs_full16']:+.3f}, CI-sep "
        f"{fb['cond_ci_separated_vs_full16']}) / age-weighted "
        f"{fa['auc']:.3f} (margin {fa['margin_vs_full16']:+.3f}, CI-sep "
        f"{fa['cond_ci_separated_vs_full16']}) | " + dec["verdict"],
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
