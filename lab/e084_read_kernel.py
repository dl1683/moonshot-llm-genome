"""E084 — the READ-KERNEL census: the read-policy program's opener.

This is day5_programs.md proposal #3 (there registered as "e083 — READ POLICY
opener: the argmax-flip census"); the dispatcher assigned it the e084 slot.
Everything below was registered (frozen) BEFORE any compute.

Trigger: T037 #5 — every killed hypothesis was categorical, every survivor
graded, and the one component never directly edited is the per-position RULE
deciding which stored coordinate is opened and which candidate wins argmax.
The dCE lesion curves (e053b/e053c/e069) are its SHADOW: continuous,
teacher-forced, single-read. This run measures the RULE itself — a DISCRETE
function decision(cache) -> argmax — by perturbing ONE stored coordinate at a
time at real decision points and censusing the ARGMAX, not the logit wiggle.

NET (frozen): runs/checkpoints/e053c_ctx512.pt — the window-decided 0.84M
net (4L/4H/128, ctx 512, val CE 1.5227), a*(eval-512) = 6 CI [4,8].

BATTERY (frozen): the e072 battery-B lineage — e069's exact 4 sequences
(prompt seed 202, sampling seed 7, fixed anchor, free-run 64->512, temp 0.8
top-k 40, per-row shared-generator draws) + 4 fresh (first draws of the
e072 302/17 generator, the 202/7 family +100/+10) = 8 sequences — the
B=8 the day-5 doc itself flags as the instrument-first sample size.
Protocol-identity gates: G2 params, G1 val CE, G0b incremental-KV vs full
recompute, G3 eval-512 V-zero sweep replica vs e069's stored readouts
(verbatim fine_sweep instrument).

DESIGN (registered before compute; amended pre-compute for CPU tractability
— see the tractability note below):
  - 100 decision points (b, q), q in [96, 510] over the 16 free runs,
    MARGIN-STRATIFIED: margin = clean top1-top2 LOGIT gap of the
    generation-time distribution at (b, q); quartile cut points of the pooled
    16x415 candidate margins; 25 sampled per quartile WITHOUT replacement
    (numpy default_rng seed 2084, quartile blocks drawn in Q1..Q4 order —
    near-ties and confident calls both deliberately represented, margin
    recorded as a covariate).
  - Entry band per decision point (documented sampling, kept tractable):
    young ages 1-10 + mid ages 11-60 (all) + 20 old ages sampled WITHOUT
    replacement uniformly from [61, q+1] (default_rng seed 9000+dp_index)
    = 80 entries.
  - 3 intervention types per (decision x entry) cell, all-layer
    whole-position, the e069/e053b instrument verbatim (+1 extension):
      V-zero   — v := 0 at that position, every layer, every head, all
                 queries (e069 verbatim);
      K-drop   — key removed from every non-diagonal query's softmax row,
                 every layer, every head (e069 verbatim);
      V-counterfactual — v := donor's v at the SAME ABSOLUTE POSITION (same
                 age; donor = battery sequence (b+1) mod 16, a different
                 free run), every layer, every head, all queries (the new
                 arm; donor v precomputed from the donor's clean forward).
    Total = 100 x 80 x 3 = 24,000 cells. Readout: does the ARGMAX flip, and
    to what (pre-flip runner-up? clean top-5? outside? for V-counterfactual
    also: the donor's ACTUAL next token at the same query position — the
    content-following witness).

TRACTABILITY AMENDMENT (registered BEFORE any census compute; deviations
from the dispatcher's spec, each the smallest sound cut and sanctioned by
the dispatcher's fallback): (1) decision points 200 -> 100 (25 per margin
quartile): measured naive-instrument throughput on this box is 5.3-9.4k
token-rows/s under load; 200 x 241 rows x E[T]=304 = 14.7M token-rows =
26-46 min census alone, over the lab's <=30-min single-step envelope. A
suffix-window exact-recompute path was built and validated (max |dlogit|
1.9e-05 vs the naive instrument) but is op-dispatch-bound at these batch
sizes on this contended CPU (no net win), so the census uses the NAIVE
e069-verbatim instrument. (2) fresh-run count 12 -> 4 (battery 16 -> 8
sequences, battery A untouched): generation is step-count-bound (~9 min per
battery under load); 8 runs preserve cross-run donors and margin diversity
at the day-5 doc's own B=8 honesty level. EVERYTHING else registered is
unchanged: 80-entry bands, 3 types, q in [96,510] (all four old bins
populated), kill bar recomputed on the 24,000-cell battery (0.1% = 24
flips).

INSTRUMENT: the naive full-context manual forward (manual_logits2 = e069's
_manual_chunk VERBATIM + the pre-transpose V-swap extension, inert when
unused) for every cell: one 241-row batched call per decision point (1 clean
row + 240 intervention rows), chunk 64. G3 (eval-512 V-zero sweep replica vs
e069's stored readouts) gates the verbatim path end-to-end; G7 spot-checks
the census mask construction: 8 cells of decision point 0 recomputed via
independent single-row calls must match the batched computation to < 1e-4.

DELIVERABLE — the READ KERNEL:
  (a) flip-rate(age) curve with bootstrap CIs (over decision points,
      n=1000): ages 1-60 exact + old ages pooled into 4 bins
      [61-100], [101-200], [201-350], [351-511] (documented);
  (b) flip-target taxonomy: runner-up / clean-top-5 / outside-top-5 shares
      per type + winner-token histogram (attractor check) + the V-swap
      donor-continuation hit rate vs its chance level;
  (c) correlation between flip-rate(age) and dCE-load(age) from the STORED
      e053c profile (runs/e053c/metrics.json cell.vzero_dce_mean, 511 ages,
      the a*=6 curve; identical to e069's replica to <1e-5). PRIMARY r =
      Pearson over the 64 matched points (ages 1-60 exact + 4 old bins at
      the bin-mean stored dCE) between the V-ZERO flip-rate curve and the
      V-ZERO dCE profile (instrument-matched; pooled/K-drop/V-swap variants
      reported as secondaries, as are Spearman and ages-1-60-only).

REGISTERED BARS (frozen, in order):
  1. battery-wide pooled flip rate < 0.1% of 24,000 cells  => KILL: no
     dynamic range; park as instrument-debt (decision-point selector with
     margin floors needed). Do NOT rescue with bigger perturbations.
  2. else PRIMARY Pearson r >= 0.8  => KERNEL = SHADOW: the argmax rule
     opens what the CE-shadow says.
  3. else r < 0.5  => DISSOCIATION: the shadow misreports the rule (the
     interesting outcome; the census becomes the program's primary
     instrument).
  4. else (0.5 <= r < 0.8)  => INTERMEDIATE: reported honestly.

Hypotheses in play (day5 doc): H-sparse-open (few recent + specific old
entries flip; runner-up targets) vs H-dense-averaging (singles almost never
flip) vs H-content-following vs H-statistics-only (V-swap donor-continuation
hit rate vs chance).

Run:     python lab/e084_read_kernel.py          (E084_SMOKE=1 for smoke)
Outputs: runs/e084/metrics.json + runs/e084/read_kernel.png
Envelope: NO training, NO new automations; CPU-only (CUDA_VISIBLE_DEVICES
forced -1 and torch.cuda stubbed out pre-import, e070 pattern); single step,
minutes-scale (~15-25 min). No NOTES/THINKING/QUEUE/STATE edits; no commit
(the dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu" (e053b/e069/e070
# pattern)
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
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
SEED_PROMPT, SEED_SAMPLE = 202, 7         # e053b/e053c/e069/e070/e072 seeds
N_PROMPTS = 8
B = 4                                     # battery A (e069 verbatim)
B_X = 4                                   # fresh extras (302/17 family, see
                                          # docstring battery amendment)
SEED_PROMPT_X, SEED_SAMPLE_X = 302, 17    # 202/7 family +100/+10
BOOT_N = 1000
THREADS = 8      # measured 2026-09-25: 12 threads spin-thrash this contended
                 # box on small ops (309 ms/decode-step vs 20-67 at 4-8) with
                 # no census-matmul gain (899s vs 912s per 100 DPs); G0b/G3
                 # gates verify protocol identity holds at this setting
torch.set_num_threads(THREADS)

CHUNK_512 = 64                            # e069/e070/e072 measured chunks

# ---- census design (registered; see docstring tractability amendment) ------
Q_LO, Q_HI = 96, 510                      # candidate query positions
PER_QUARTILE = 25                         # 25 per margin quartile
N_DP = 4 * PER_QUARTILE                   # 100 decision points
SEED_STRAT = 2084                         # margin-stratified sampling seed
YOUNG = list(range(1, 11))                # ages 1-10 (all)
MID = list(range(11, 61))                 # ages 11-60 (all)
N_OLD = 20                                # old entries per decision point
OLD_SEED_BASE = 9000                      # old-age rng seed = base + dp_index
OLD_BINS = [(61, 100), (101, 200), (201, 350), (351, 511)]
TYPES = ["vzero", "kdrop", "vswap"]       # cell types in fixed order
TOPK_TAX = 5                              # taxonomy top-k window

# ---- REGISTERED decision numbers (FROZEN, docstring) -----------------------
KILL_RATE = 0.001                         # battery-wide pooled flip rate <
R_SHADOW = 0.8                            # primary Pearson r >=
R_DISSOC = 0.5                            # primary Pearson r <
G7_TOL = 1e-4                             # window-vs-naive max |logit dev| <

# ---- e069 stored readouts (protocol-identity gates; e072's corrected refs) -
E069_REF = {
    "eval512": {"a_star": 6, "clean_ce": 0.4528,
                "ages1_5": [1.742, 4.467, 2.319, 0.914, 0.321],
                "per_seq": [8, 4, 6, 3], "winstart_dce": 0.0069},
}
CKPT = REPO / "runs" / "checkpoints" / "e053c_ctx512.pt"
E053C_VAL_CE = 1.5226792494455974
E053C_PROFILE = REPO / "runs" / "e053c" / "metrics.json"   # stored dCE curve

SMOKE = os.environ.get("E084_SMOKE", "") == "1"
GEN_STOP = T_TOTAL                         # protocol generation length
if SMOKE:
    PER_QUARTILE = 2
    N_DP = 4 * PER_QUARTILE
    GEN_STOP = 160                         # short code-path check only
    B_X = 0                                # battery A only
    Q_HI = GEN_STOP - 2                    # 158

T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# --------------------------------------------- machinery (VERBATIM e069/e070)

@torch.no_grad()
def _manual_chunk2(net: TinyGPT, idxs, vzero, kdrop, layers, pos_offset=0,
                   vswap_pos=None, donor_stack=None):
    """e069's _manual_chunk VERBATIM + the V-counterfactual extension: rows
    with vswap_pos >= 0 have their value vector at that context position
    replaced by donor_stack[layer][position] (all heads, all queries), applied
    pre-transpose (same math as vzero). Inert when vswap_pos is None."""
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
            # e069 semantics: key p dropped from every NON-DIAGONAL query.
            # e069's broadcast form (kdrop[:,None,None,:] & nondiag) is
            # O(N*T*T) bools — prohibitive at census batch sizes; rows here
            # drop exactly ONE key, so use the equivalent column write:
            # -inf on the dropped key's column for all queries, diagonal
            # restored. Multi-position rows (never used in e084) fall back
            # to the verbatim broadcast.
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
    [VERBATIM e053b — CPU tensors + CPU generator, stream-identical]"""
    lg = logits_clean / TEMP
    v, _ = torch.topk(lg, TOPK)
    lg_f = lg.masked_fill(lg < v[-1], float("-inf"))
    tok = int(torch.multinomial(torch.softmax(lg_f, -1), 1, generator=gen))
    p_full = torch.softmax(logits_clean, -1)
    return tok, float(-math.log(max(p_full[tok].item(), 1e-12)))


@torch.no_grad()
def prefill_batch(net: TinyGPT, idx: torch.Tensor):
    """Batched prefill, (B, Tp) -> last-position logits (B, V) + KV cache.
    [VERBATIM e069]"""
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
    [VERBATIM e069]"""
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
    """e069's generate_batch VERBATIM (same stream math: fixed anchor, free
    run 64->gen_stop, per-row draws from the shared CPU generator in row
    order per step; gen_stop=T_TOTAL is the protocol run, shorter stops are
    SMOKE-only code-path checks) + records the per-step decision logits:
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


@torch.no_grad()
def fine_sweep(net: TinyGPT, ctx: torch.Tensor, tgt: torch.Tensor,
               clean_logits: torch.Tensor, ch: int, pos_offset: int = 0):
    """Final-step fine V-zero sweep for all B sequences at once.
    [e053c/e069 math, VERBATIM]"""
    Bs, Pw = ctx.shape
    p_clean = torch.softmax(clean_logits.float(), -1)
    ce_clean = -torch.log(p_clean[torch.arange(Bs), tgt].clamp_min(1e-12))
    idxs = ctx.repeat_interleave(Pw, dim=0)                # (B*Pw, Pw)
    vz = torch.eye(Pw, dtype=torch.bool).repeat(Bs, 1)     # row b*Pw+p: zero p
    les = manual_logits2(net, idxs, vz, None, None, chunk=ch, pos_offset=pos_offset)
    lp = torch.log_softmax(les.float(), -1)
    rows = torch.arange(Bs * Pw)
    dce = (-ce_clean.repeat_interleave(Pw) - lp[rows, tgt.repeat_interleave(Pw)])
    return dce.view(Bs, Pw).numpy(), ce_clean.numpy()


def onset_age(dce_by_age: np.ndarray, k: int = 5, thresh: float = 0.01):
    """dce_by_age: ages ascending. Returns (naive, robust). [VERBATIM e053b]"""
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


# ---------------------------------------- census instrument (e084's own, gated)

@torch.no_grad()
def v_probe(net: TinyGPT, idxs: torch.Tensor, chunk: int = 8):
    """Clean batched forward recording the value-projection outputs per layer
    in pre-transpose layout (N, T, H, d) — the donor V bank for
    V-counterfactual swaps. [e072 value_probe pattern]"""
    N, T = idxs.shape
    H = net.cfg.n_head
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    vs = [[] for _ in net.h]
    for i in range(0, N, chunk):
        x = net.wte(idxs[i:i + chunk]) + net.wpe(torch.arange(T))[None]
        n = x.shape[0]
        for li, blk in enumerate(net.h):
            xh = blk.ln1(x)
            qkv = blk.attn.c_attn(xh)
            C = qkv.shape[-1] // 3
            d = C // H
            q, k, v = qkv.split(C, dim=2)
            q = q.view(n, T, H, d).transpose(1, 2)
            k = k.view(n, T, H, d).transpose(1, 2)
            v4 = v.view(n, T, H, d)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
            att = att.masked_fill(causal, float("-inf"))
            y = (torch.softmax(att, -1) @ v4.transpose(1, 2)).transpose(1, 2)
            y = y.reshape(n, T, C)
            x = x + blk.attn.c_proj(y)
            x = x + blk.mlp(blk.ln2(x))
            vs[li].append(v4.clone())
    return torch.stack([torch.cat(v, 0) for v in vs], 0)   # (L, N, T, H, d)


def census_dp(net, idx16, donor_v, b, q, dp_index):
    """Run the full 240-cell census for one decision point (b, q) with the
    NAIVE e069-verbatim instrument: one batched manual_logits2 call over
    1 clean row + 3 x 80 intervention rows (whole-context masks).
    Returns dict with cells (type_idx, age), per-row argmaxes, clean t1/t2/
    top5/margin, donor_pick, and the row logits (for the G7 spot check)."""
    ctx = idx16[b, :q + 1]
    T = q + 1
    rng = np.random.default_rng(OLD_SEED_BASE + dp_index)
    avail = np.arange(61, T + 1)
    old = [int(a) for a in rng.choice(avail, size=N_OLD, replace=False)]
    entries = list(YOUNG) + list(MID) + old               # 80 ages

    n = 1 + 3 * len(entries)
    vz = torch.zeros(n, T, dtype=torch.bool)
    kd = torch.zeros(n, T, dtype=torch.bool)
    vsp = torch.full((n,), -1, dtype=torch.long)
    b2 = (b + 1) % idx16.shape[0]
    cells = []          # (type_idx, age) in row order; row 0 = clean
    r = 1
    for a in entries:
        p = T - a
        vz[r, p] = True
        kd[r + 1, p] = True
        vsp[r + 2] = p
        cells.append((0, a))
        cells.append((1, a))
        cells.append((2, a))
        r += 3

    lg = manual_logits2(net, ctx[None].expand(n, -1).contiguous(),
                        vz, kd, None, chunk=CHUNK_512,
                        vswap_pos=vsp, donor_stack=donor_v[:, b2])
    top = torch.topk(lg[0], TOPK_TAX)
    t1 = int(top.indices[0])
    t2 = int(top.indices[1])
    top5 = [int(t) for t in top.indices]
    margin = float(top.values[0] - top.values[1])
    donor_pick = int(idx16[b2, q + 1])

    am = lg.argmax(1)
    recs = []
    for r, (typ, age) in enumerate(cells):
        t_prime = int(am[r + 1])
        if t_prime == t1:
            recs.append((typ, age, 0, t_prime, -1, 0))
            continue
        cls = 0 if t_prime == t2 else (1 if t_prime in top5 else 2)
        dh = 1 if (typ == 2 and t_prime == donor_pick) else 0
        recs.append((typ, age, 1, t_prime, cls, dh))
    return dict(b=int(b), q=int(q), margin=margin, t1=t1, t2=t2, top5=top5,
                donor_pick=donor_pick, cells=cells, recs=recs, logits=lg)


def cell_logits_single(net, idx16, donor_v, b, q, typ, age):
    """Independent single-row recomputation of ONE cell (G7 spot check):
    naive instrument, chunk=1, fresh mask construction."""
    ctx = idx16[b, :q + 1]
    T = q + 1
    p = T - age
    vz = torch.zeros(1, T, dtype=torch.bool)
    kd = torch.zeros(1, T, dtype=torch.bool)
    vsp = torch.full((1,), -1, dtype=torch.long)
    b2 = (b + 1) % idx16.shape[0]
    if typ == 0:
        vz[0, p] = True
    elif typ == 1:
        kd[0, p] = True
    else:
        vsp[0] = p
    return manual_logits2(net, ctx[None], vz, kd, None, chunk=1,
                          vswap_pos=vsp, donor_stack=donor_v[:, b2])[0]


# ------------------------------------------------------------------ statistics

def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or x[m].std() == 0 or y[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def rankdata(v):
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), float)
    ranks[order] = np.arange(1, len(v) + 1)
    # average ties (ties are measure-zero here; keep simple)
    return ranks


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


# ---------------------------------------------------------------------- main

def main():
    assert torch.cuda.device_count() == 0, "CPU-only violated: CUDA visible"
    assert common.DEVICE == "cpu", f"common.DEVICE={common.DEVICE} != cpu"
    out_dir = run_dir("e084")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=SMOKE)
    suffix = "_smoke" if SMOKE else ""

    # ---- battery: rebuild corpus + net EXACTLY as e053c/e069/e070/e072 did
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

    # ---- battery X extras: fresh sequences from the e072 302/17 family
    #      (full run: 4 of the 12 e072 draws — docstring battery amendment;
    #       smoke: skipped, battery A alone exercises every code path)
    if B_X > 0:
        log(f"battery X: {B_X} fresh sequences (prompt seed {SEED_PROMPT_X}, "
            f"sampling seed {SEED_SAMPLE_X}; family = 202/7 +100/+10)")
        gen_px = torch.Generator().manual_seed(SEED_PROMPT_X)
        ix12 = torch.randint(len(corp.val) - PROMPT_TOK - 1, (B_X,),
                             generator=gen_px)
        prompts12 = [corp.val[i:i + PROMPT_TOK] for i in ix12]
        gen_x = torch.Generator().manual_seed(SEED_SAMPLE_X)
        idxX, _, histX = generate_batch_recorded(net, prompts12, gen_x,
                                                 gen_stop=GEN_STOP)
        idx16 = torch.cat([idxA, idxX], 0)
        hist16 = torch.cat([histA, histX], 0)
        log(f"battery: {idx16.shape[0]} sequences = e069's 4 + {B_X} fresh "
            f"(e072 battery-B lineage, draws trimmed 12->4)")
    else:
        idx16, hist16 = idxA, histA
        log(f"battery: {idx16.shape[0]} sequences (battery A only, smoke)")

    # ---- G3: eval-512 V-zero sweep replica vs e069's stored readouts
    #      (full run only — needs the full 512-token protocol generation)
    if SMOKE:
        gates["G3_eval512_vs_e069"] = dict(
            skipped=True,
            reason="smoke runs a shortened generation; the verbatim "
                   "fine_sweep/manual_logits2 path is separately validated "
                   "(e070/e072 G3 replicas + e084 micro-tests)")
        log("G3 replica: SKIPPED (smoke)")
    else:
        log("G3 replica: eval-512 fine V-zero sweep (battery A final position)")
        tgt = idxA[:, -1]
        ctx512 = idxA[:, :-1]
        dce, ce_clean = fine_sweep(net, ctx512, tgt, final_logits, CHUNK_512)
        dce_age = dce[:, ::-1]                             # (B,511) age-asc
        mean_age = dce_age.mean(0)
        a_star, _, _ = a_star_rule(mean_age)
        ref = E069_REF["eval512"]
        ages15 = mean_age[:5]
        dev15 = float(np.max(np.abs(ages15 - np.asarray(ref["ages1_5"]))))
        ws_dev = abs(float(mean_age[254]) - ref["winstart_dce"])
        ce_dev = abs(float(ce_clean.mean()) - ref["clean_ce"])
        a_match = a_star == ref["a_star"]
        g3_ok = bool(a_match and dev15 < 0.05 and ws_dev < 0.05
                     and ce_dev < 0.02)
        gates["G3_eval512_vs_e069"] = dict(
            a_star=a_star, a_star_ref=ref["a_star"], a_star_match=a_match,
            ages1_5=[float(a) for a in ages15], max_dev_ages1_5=dev15,
            winstart_dce=float(mean_age[254]), winstart_dev=ws_dev,
            clean_ce=float(ce_clean.mean()), clean_ce_dev=ce_dev, ok=g3_ok)
        log(f"G3 eval512: a* {a_star} (ref {ref['a_star']}), ages1-5 maxdev "
            f"{dev15:.4f}, age-255 dev {ws_dev:.4f}, CE dev {ce_dev:.4f} "
            f"-> {'PASS' if g3_ok else 'FAIL'}")

    # ---- donor V bank + stored e053c dCE-load(age) profile
    donor_v = v_probe(net, idx16, chunk=8)             # (L, 16, 512, H, d)
    profile_doc = json.loads(E053C_PROFILE.read_text(encoding="utf-8"))
    dce_profile = np.asarray(profile_doc["cell"]["vzero_dce_mean"], float)
    assert len(dce_profile) == T_TOTAL - 1
    gates["G4_profile_source"] = dict(
        path=str(E053C_PROFILE), n_ages=len(dce_profile),
        a_star_source=profile_doc["cell"]["derived"]["a_star"],
        ages1_5_source=[round(float(x), 3) for x in dce_profile[:5]],
        ok=bool(profile_doc["cell"]["derived"]["a_star"] == 6))
    log(f"stored e053c dCE profile loaded ({len(dce_profile)} ages, a*=6 "
        f"curve); donor V bank {tuple(donor_v.shape)}")

    # ---- decision-point selection: margin quartiles, 50 per quartile
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
    log(f"candidates {len(cands)} ({idx16.shape[0]} seqs x q in "
        f"[{Q_LO},{Q_HI}]); margin quartile edges {qs.round(3).tolist()}; "
        f"sampled {len(dps)} decision points (seed {SEED_STRAT})")

    # ---- G7: census mask-construction spot check (first DP, 8 cells
    #      recomputed via independent single-row calls)
    b0, q0 = dps[0][0], dps[0][1]
    c0 = census_dp(net, idx16, donor_v, b0, q0, 0)
    spot = [c0["cells"][0], c0["cells"][3], c0["cells"][4],     # vz y1, vz m11, kd m11
            c0["cells"][2], c0["cells"][-3], c0["cells"][-1],   # vs y1, old vz, old vs
            c0["cells"][30], c0["cells"][100]]                  # assorted mid
    g7_dev = 0.0
    for typ, age in spot:
        row = 1 + c0["cells"].index((typ, age))
        single = cell_logits_single(net, idx16, donor_v, b0, q0, typ, age)
        g7_dev = max(g7_dev, float((c0["logits"][row] - single).abs().max()))
    gates["G7_census_mask_spotcheck"] = dict(
        n_cells=len(spot), max_logit_dev=g7_dev, tol=G7_TOL,
        ok=bool(g7_dev < G7_TOL))
    log(f"G7 census-vs-single spot check: max |logit dev| {g7_dev:.2e} over "
        f"{len(spot)} cells -> {'PASS' if g7_dev < G7_TOL else 'FAIL'}")
    assert g7_dev < G7_TOL, "G7 FAIL: census mask construction inconsistent"

    # ---- the census
    t_census = time.time()
    per_dp = []
    rec_flat = []        # (dp, type, age, flip, t_prime, cls, donor_hit)
    census = {0: c0}
    for i in range(len(dps)):
        b, q, _, qt = dps[i]
        if i not in census:
            census[i] = census_dp(net, idx16, donor_v, b, q, i)
        c = census[i]
        per_dp.append(dict(dp=i, b=c["b"], q=c["q"], quartile=int(qt),
                           margin=c["margin"], t1=c["t1"], t2=c["t2"],
                           donor_pick=c["donor_pick"],
                           flips=[0, 0, 0]))
        for (typ, age, flip, t_prime, cls, dh) in c["recs"]:
            rec_flat.append((i, typ, age, flip, t_prime, cls, dh))
            if flip:
                per_dp[-1]["flips"][typ] += 1
        if (i + 1) % 20 == 0 or i == len(dps) - 1:
            done = i + 1
            rate = (time.time() - t_census) / done
            log(f"census {done}/{len(dps)} decision points "
                f"({rate:.2f}s/dp, ETA {rate * (len(dps) - done):.0f}s)")
    del census

    rec = np.asarray([(r[0], r[1], r[2], r[3], r[5], r[6]) for r in rec_flat],
                     dtype=np.int64)      # dp, type, age, flip, cls, donor_hit
    t_prime_flat = np.asarray([r[4] for r in rec_flat], dtype=np.int64)
    n_cells = len(rec_flat)
    n_flips = int(rec[:, 3].sum())
    battery_rate = n_flips / n_cells
    per_type_rate = [float((rec[rec[:, 1] == t][:, 3]).mean())
                     for t in range(3)]
    log(f"CENSUS: {n_cells} cells, {n_flips} flips -> battery-wide "
        f"{battery_rate:.5f} ({battery_rate * 100:.3f}%) | per-type "
        + "/".join(f"{TYPES[t]} {per_type_rate[t] * 100:.3f}%"
                   for t in range(3)))

    # ---- (a) flip-rate(age): exact ages 1-60 + 4 old bins, CIs
    F_exact = np.zeros((len(dps), 60, 3), dtype=np.int64)
    for dp, typ, age, flip, cls, dh in rec:
        if 1 <= age <= 60:
            F_exact[dp, age - 1, typ] += flip
    bin_of = np.full(T_TOTAL, -1, dtype=np.int64)
    for bi, (lo, hi) in enumerate(OLD_BINS):
        bin_of[lo:hi + 1] = bi
    F_bin = np.zeros((len(dps), len(OLD_BINS), 3), dtype=np.int64)
    C_bin = np.zeros((len(dps), len(OLD_BINS), 3), dtype=np.int64)
    for dp, typ, age, flip, cls, dh in rec:
        bi = bin_of[age] if age < T_TOTAL else -1
        if bi >= 0:
            F_bin[dp, bi, typ] += flip
            C_bin[dp, bi, typ] += 1

    def rate_curve(w):
        """w: (n_dp,) resample weights -> per-age-type rates ex (60,3) +
        weighted bin flip/cell counts (4,3) each."""
        w = np.asarray(w, float)
        ex = np.einsum("d,dat->at", w, F_exact) / np.maximum(w.sum(), 1e-9)  # per age/type
        bn_f = np.einsum("d,dbt->bt", w, F_bin)
        bn_c = np.einsum("d,dbt->bt", w, C_bin)
        return ex, bn_f, bn_c

    w0 = np.ones(len(dps))
    ex0, bnf0, bnc0 = rate_curve(w0)
    ages_x = list(range(1, 61))
    bin_mid, bin_dce = [], []
    for lo, hi in OLD_BINS:
        ages = np.arange(lo, min(hi, T_TOTAL - 1) + 1)
        bin_mid.append(float(ages.mean()))
        bin_dce.append(float(dce_profile[ages - 1].mean()))
    dce_x = np.asarray([dce_profile[a - 1] for a in ages_x] + bin_dce)

    def curves(ex, bnf, bnc):
        out = {}
        for t in range(3):
            c = list(ex[:, t]) + list(bnf[:, t] / np.maximum(bnc[:, t], 1))
            out[TYPES[t]] = np.asarray(c, float)
        out["pooled"] = np.asarray(
            list(ex.sum(1) / (3.0)) + list(bnf.sum(1) / np.maximum(bnc.sum(1), 1)),
            float)
        return out

    cv0 = curves(ex0, bnf0, bnc0)

    rngb = np.random.default_rng(0)
    boot = {k: [] for k in TYPES + ["pooled"]}
    r_boot = []
    for _ in range(BOOT_N):
        sel = rngb.integers(0, len(dps), len(dps))
        w = np.bincount(sel, minlength=len(dps)).astype(float)
        ex, bnf, bnc = rate_curve(w)
        cvb = curves(ex, bnf, bnc)
        for k in boot:
            boot[k].append(cvb[k])
        y = cvb["vzero"]
        m = np.isfinite(y) & np.isfinite(dce_x)
        if m.sum() >= 40:
            r_boot.append(pearson(dce_x[m], y[m]))
    boot_ci = {k: (np.nanpercentile(np.asarray(v)[:, :60], 2.5, axis=0),
                   np.nanpercentile(np.asarray(v)[:, :60], 97.5, axis=0))
               for k, v in boot.items()}

    # ---- (c) correlations (registered primary: vzero vs stored vzero dCE)
    r_primary = pearson(dce_x, cv0["vzero"])
    r_ci = (float(np.nanpercentile(r_boot, 2.5)),
            float(np.nanpercentile(r_boot, 97.5))) if r_boot else (float("nan"),) * 2
    corr_tbl = dict(
        primary_vzero_vs_dce=dict(pearson=r_primary, ci=list(r_ci),
                                  spearman=spearman(dce_x, cv0["vzero"]),
                                  n_points=len(dce_x)),
        pooled_vs_dce=dict(pearson=pearson(dce_x, cv0["pooled"]),
                           spearman=spearman(dce_x, cv0["pooled"])),
        kdrop_vs_dce=dict(pearson=pearson(dce_x, cv0["kdrop"]),
                          spearman=spearman(dce_x, cv0["kdrop"])),
        vswap_vs_dce=dict(pearson=pearson(dce_x, cv0["vswap"]),
                          spearman=spearman(dce_x, cv0["vswap"])),
        vzero_ages1_60_only=dict(pearson=pearson(dce_x[:60], cv0["vzero"][:60])),
        vzero_ages1_10_only=dict(pearson=pearson(dce_x[:10], cv0["vzero"][:10])),
    )
    log(f"CORRELATION (primary): vzero flip-rate(age) vs stored dCE-load(age) "
        f"Pearson r = {r_primary:.4f} CI [{r_ci[0]:.3f},{r_ci[1]:.3f}], "
        f"Spearman {corr_tbl['primary_vzero_vs_dce']['spearman']:.4f}")

    # ---- (b) flip-target taxonomy
    fl = rec[rec[:, 3] == 1]
    taxonomy = {}
    for t in range(3):
        sub = fl[fl[:, 1] == t]
        n = len(sub)
        taxonomy[TYPES[t]] = dict(
            n_flips=int(n),
            runner_up=float((sub[:, 4] == 0).mean()) if n else float("nan"),
            top5=float((sub[:, 4] == 1).mean()) if n else float("nan"),
            outside_top5=float((sub[:, 4] == 2).mean()) if n else float("nan"))
    vs = fl[fl[:, 1] == 2]
    donor_hits = int(vs[:, 5].sum()) if len(vs) else 0
    # chance level: how often the donor's pick coincides with the CLEAN argmax
    # (per decision point; every dp carries 80 V-swap cells)
    donor_chance = float((np.asarray([dp["donor_pick"] for dp in per_dp])
                          == np.asarray([dp["t1"] for dp in per_dp])).mean())
    winners = {TYPES[t]: Counter(int(t_prime_flat[i]) for i in
                                 np.where((rec[:, 1] == t) & (rec[:, 3] == 1))[0])
               for t in range(3)}
    log(f"TAXONOMY: " + " | ".join(
        f"{TYPES[t]}: n={taxonomy[TYPES[t]]['n_flips']} "
        f"runner-up {taxonomy[TYPES[t]]['runner_up']:.2f} "
        f"top5 {taxonomy[TYPES[t]]['top5']:.2f} "
        f"outside {taxonomy[TYPES[t]]['outside_top5']:.2f}"
        for t in range(3)))
    log(f"V-swap content-following: donor-continuation hit "
        f"{donor_hits}/{len(vs)} flips vs chance {donor_chance:.4f}")

    # ---- margin covariate + opened-coordinate structure
    margins_cells = np.asarray([per_dp[r[0]]["margin"] for r in rec_flat])
    flip_margins = margins_cells[rec[:, 3] == 1]
    quart_cells = np.asarray([per_dp[r[0]]["quartile"] for r in rec_flat])
    quart_rates = [float(rec[quart_cells == qq][:, 3].mean())
                   for qq in range(4)]
    quart_rates_type = {
        TYPES[t]: [float(rec[(quart_cells == qq) & (rec[:, 1] == t)][:, 3].mean())
                   for qq in range(4)] for t in range(3)}
    opened = {TYPES[t]: [dp["flips"][t] for dp in per_dp] for t in range(3)}
    opened_any = [len(set(rec[(rec[:, 0] == i) & (rec[:, 3] == 1)][:, 2]))
                  for i in range(len(dps))]
    bands = {"young_1_10": (1, 10), "mid_11_60": (11, 60),
             "old_61_511": (61, T_TOTAL)}
    band_rates = {}
    for name, (lo, hi) in bands.items():
        m = (rec[:, 2] >= lo) & (rec[:, 2] <= hi)
        band_rates[name] = dict(
            pooled=float(rec[m][:, 3].mean()) if m.any() else float("nan"),
            **{TYPES[t]: float(rec[m & (rec[:, 1] == t)][:, 3].mean())
               for t in range(3)},
            n_cells=int(m.sum()))
    t1_vs_sampled = float(np.mean([
        int(idx16[dp["b"], dp["q"] + 1] == dp["t1"]) for dp in per_dp]))

    # ---- REGISTERED decision (frozen bars, in order)
    if battery_rate < KILL_RATE:
        fires = "KILL_NO_DYNAMIC_RANGE"
        verdict = (f"KILL: battery-wide pooled flip rate "
                   f"{battery_rate * 100:.4f}% < 0.1% of {n_cells} cells — "
                   "the discrete census has no dynamic range at char scale; "
                   "park as instrument-debt (margin-floor selector needed). "
                   "Do NOT rescue with bigger perturbations.")
    elif r_primary >= R_SHADOW:
        fires = "KERNEL_EQUALS_SHADOW"
        verdict = (f"KERNEL = SHADOW: primary Pearson r = {r_primary:.3f} "
                   f">= 0.8 (CI [{r_ci[0]:.3f},{r_ci[1]:.3f}]) — the argmax "
                   "rule opens what the CE-shadow says.")
    elif r_primary < R_DISSOC:
        fires = "DISSOCIATION"
        verdict = (f"DISSOCIATION: primary Pearson r = {r_primary:.3f} < 0.5 "
                   f"with meaningful flip rates ({battery_rate * 100:.3f}% "
                   "battery-wide > 0.1%) — the shadow misreports the rule; "
                   "the census becomes the read-policy program's primary "
                   "instrument.")
    else:
        fires = "INTERMEDIATE"
        verdict = (f"INTERMEDIATE: 0.5 <= r = {r_primary:.3f} < 0.8 with "
                   f"{battery_rate * 100:.3f}% battery-wide flips — neither "
                   "clean bar; reported honestly.")
    log(f"REGISTERED DECISION [{fires}]: {verdict}")

    # ---------------------------------------------------------------- metrics
    metrics = dict(
        experiment="e084_read_kernel",
        purpose="READ-KERNEL census (day5 proposal #3, dispatcher slot e084): "
                "200 margin-stratified free-run decision points x 80 cache "
                "entries (young 1-10, mid 11-60, 20 sampled old) x "
                "{V-zero, K-drop, V-counterfactual} whole-position all-layer "
                "interventions; ARGMAX-flip readout. FROZEN BARS: pooled "
                "battery-wide flip rate < 0.1% => KILL (instrument-debt); "
                "else primary Pearson r (vzero flip-rate(age) vs stored "
                "e053c vzero dCE(age), 64 matched points) >= 0.8 => KERNEL = "
                "SHADOW; < 0.5 => DISSOCIATION; else INTERMEDIATE.",
        started=started, wall_s=elapsed(), threads=THREADS, cpu_only=True,
        smoke=SMOKE,
        net=dict(ckpt=str(CKPT), arch=dict(n_layer=4, n_head=4, n_embd=128,
                                           block_size=T_TOTAL, vocab=65),
                 params=n_params, steps_in_ckpt=steps_in_ckpt,
                 val_ce=val_ce, val_ce_e053c=E053C_VAL_CE),
        seeds=dict(corpus=1337,
                   battery_A=dict(prompts=SEED_PROMPT, sampling=SEED_SAMPLE,
                                  note="e069 verbatim"),
                   battery_B_extra12=dict(prompts=SEED_PROMPT_X,
                                          sampling=SEED_SAMPLE_X,
                                          note="e072 verbatim (202/7 +100/+10)"),
                   stratification=SEED_STRAT, old_ages=OLD_SEED_BASE,
                   bootstrap=0),
        protocol=dict(
            n_decision_points=len(dps), per_quartile=PER_QUARTILE,
            q_range=[Q_LO, Q_HI], candidates=len(cands),
            margin="clean top1-top2 logit gap at generation time "
                   "(incremental-KV logits, G0b-checked)",
            entry_bands=dict(young=YOUNG, mid=MID, old_n=N_OLD,
                             old_sampling="uniform without replacement from "
                                          "[61, q+1], default_rng seed "
                                          f"{OLD_SEED_BASE}+dp_index",
                             old_bins=[list(b) for b in OLD_BINS]),
            types=TYPES,
            donor_rule="battery sequence (b+1) mod 16 at the SAME absolute "
                       "position (same age, same wpe row); donor v from the "
                       "donor's clean forward, all layers/heads/queries",
            instrument="e069 whole-position all-layer all-head masks verbatim "
                       "(V-zero post-transpose masked_fill; K-drop "
                       "non-diagonal); V-counterfactual = same mask family "
                       "with donor values substituted pre-transpose",
            speed_method="suffix-window recompute (lesion at p only affects "
                         "positions >= p; prefix K/V from one clean full "
                         "forward per decision point); G7-gated exact vs the "
                         "naive full-context instrument",
            cells=n_cells,
            taxonomy_topk=TOPK_TAX),
        gates=gates,
        battery=dict(
            n_flips=n_flips, flip_rate_battery_wide=battery_rate,
            flip_rate_per_type=dict(zip(TYPES, per_type_rate)),
            kill_bar=KILL_RATE,
            t1_equals_sampled_rate=t1_vs_sampled,
            margin_quartile_rates=dict(pooled=quart_rates,
                                       per_type=quart_rates_type)),
        kernel_curve=dict(
            x_ages_exact=ages_x, x_bin_mids=bin_mid,
            dce_stored=dce_x.tolist(),
            dce_source=str(E053C_PROFILE),
            rates={k: v.tolist() for k, v in cv0.items()},
            ci_pooled=dict(lo=boot_ci["pooled"][0].tolist(),
                           hi=boot_ci["pooled"][1].tolist()),
            ci_vzero=dict(lo=boot_ci["vzero"][0].tolist(),
                          hi=boot_ci["vzero"][1].tolist()),
            cells_per_age_exact=3 * len(dps),
            bin_cells=C_bin.sum(0).tolist(),
            bin_flip_counts=F_bin.sum(0).tolist(),
            note="flip-rate(age): ages 1-60 exact; old ages pooled into "
                 "documented bins at bin-mid x"),
        band_rates=band_rates,
        correlations=corr_tbl,
        taxonomy=dict(
            per_type=taxonomy,
            vswap_donor_continuation=dict(
                hits=donor_hits, n_vswap_flips=int(len(vs)),
                hit_rate=float(donor_hits / len(vs)) if len(vs) else float("nan"),
                chance_all_vswap_cells=donor_chance,
                witness="new argmax == donor's actual next token at the same "
                        "query position (content-following)"),
            winners_top10={k: [(corp.itos[int(t)], int(n))
                               for t, n in v.most_common(10)]
                           for k, v in winners.items()}),
        margin_structure=dict(
            quartile_edges=qs.tolist(),
            quartile_rates_pooled=quart_rates,
            quartile_rates_per_type={k: v for k, v in quart_rates_type.items()},
            all_cell_margin_median=float(np.median(margins_cells)),
            flip_cell_margin_median=float(np.median(flip_margins))
            if len(flip_margins) else float("nan"),
            flip_cell_margin_quartiles=[float(x) for x in
                                        np.percentile(flip_margins,
                                                      [25, 50, 75])]
            if len(flip_margins) else None,
            n_flip_cells_margin_lt_1e3=int((flip_margins < 1e-3).sum())
            if len(flip_margins) else 0),
        opened_coordinates=dict(
            per_type_median={k: float(np.median(v)) for k, v in opened.items()},
            per_type_hist={k: np.bincount(v, minlength=81).tolist()
                           for k, v in opened.items()},
            any_type_median=float(np.median(opened_any)),
            note="entries (of 80) whose single-coordinate intervention "
                 "flipped the argmax — the sparse-open vs dense-averaging "
                 "view"),
        per_decision_point=[dict(dp=d["dp"], b=d["b"], q=d["q"],
                                 quartile=d["quartile"], margin=d["margin"],
                                 flips_vz=d["flips"][0], flips_kd=d["flips"][1],
                                 flips_vs=d["flips"][2]) for d in per_dp],
        registered_decision=dict(
            frozen_bars=dict(
                kill=f"battery-wide pooled flip rate < {KILL_RATE}",
                shadow=f"primary Pearson r >= {R_SHADOW}",
                dissociation=f"r < {R_DISSOC} with rate >= {KILL_RATE}",
                intermediate=f"{R_DISSOC} <= r < {R_SHADOW}"),
            battery_flip_rate=battery_rate,
            r_primary=r_primary, r_primary_ci=list(r_ci),
            fires=fires, verdict=verdict),
    )
    save_json(out_dir / f"metrics{suffix}.json", metrics)
    log(f"metrics{suffix}.json written")
    plot(out_dir / f"read_kernel{suffix}.png", metrics, corp)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot

def plot(path: Path, M: dict, corp: CharCorpus):
    dec = M["registered_decision"]
    kc = M["kernel_curve"]
    curves = {k: np.asarray(v, float) for k, v in kc["rates"].items()}
    dce = np.asarray(kc["dce_stored"], float)
    x_ex = np.asarray(kc["x_ages_exact"])
    x_bin = np.asarray(kc["x_bin_mids"])
    ci_lo = np.asarray(kc["ci_pooled"]["lo"], float)
    ci_hi = np.asarray(kc["ci_pooled"]["hi"], float)
    ci_lo_vz = np.asarray(kc["ci_vzero"]["lo"], float)
    ci_hi_vz = np.asarray(kc["ci_vzero"]["hi"], float)

    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # ---- panel 1: the READ KERNEL — flip-rate(age)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-5, 2)
    ax1.fill_between(x_ex, np.maximum(ci_lo[:60], 1e-6),
                     np.maximum(ci_hi[:60], 1e-6),
                     alpha=0.2, color="tab:blue", label="pooled 95% boot CI")
    ax1.plot(x_ex, np.maximum(curves["pooled"][:60], 1e-6), "o-", ms=3,
             lw=1.3, color="tab:blue", label="pooled")
    ax1.plot(x_ex, np.maximum(curves["vzero"][:60], 1e-6), "s-", ms=3, lw=1.1,
             color="tab:red", label="V-zero")
    ax1.plot(x_ex, np.maximum(curves["kdrop"][:60], 1e-6), "^-", ms=3, lw=1.1,
             color="tab:green", label="K-drop")
    ax1.plot(x_ex, np.maximum(curves["vswap"][:60], 1e-6), "v-", ms=3, lw=1.1,
             color="tab:purple", label="V-swap")
    for k, c, m in [("pooled", "tab:blue", "o"), ("vzero", "tab:red", "s")]:
        y = np.maximum(curves[k][60:], 1e-6)
        y = np.where(np.isfinite(y), y, 1e-6)
        ax1.plot(x_bin, y, m, ms=7, color=c, mfc="none", mew=1.6,
                 label=f"{k} old bins")
    ax1.axhline(KILL_RATE, color="k", ls=":", lw=1.2)
    ax1.text(1.05, KILL_RATE * 1.3, "kill bar 0.1%", fontsize=7.5, color="k")
    ax1.axvline(6, color="dimgray", ls="--", lw=1.1)
    ax1.annotate("a*=6", xy=(6, 1e-5), xycoords=("data", "axes fraction"),
                 xytext=(5, 1.005), textcoords="offset points", fontsize=8,
                 color="dimgray")
    ax1.set_xlabel("cache age (tokens, log scale)")
    ax1.set_ylabel("argmax flip rate (log scale)")
    ax1.legend(fontsize=7.5, loc="upper right")
    ax1.set_title("(a) THE READ KERNEL — flip-rate(age), "
                  f"{M['protocol']['cells']:,} cells, "
                  f"battery-wide {dec['battery_flip_rate'] * 100:.3f}%",
                  fontsize=9.5)

    # ---- panel 2: kernel vs shadow overlay (twin axes)
    ax2.plot(x_ex, np.maximum(curves["vzero"][:60], 1e-6), "o-", ms=3,
             lw=1.4, color="tab:red", label="V-zero flip rate (the rule)")
    yb = np.maximum(curves["vzero"][60:], 1e-6)
    ax2.plot(x_bin, np.where(np.isfinite(yb), yb, 1e-6), "s", ms=7,
             color="tab:red", mfc="none", mew=1.6)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-5, 2)
    ax2.set_xlabel("cache age (tokens, log scale)")
    ax2.set_ylabel("V-zero flip rate", color="tab:red")
    ax2b = ax2.twinx()
    ax2b.plot(x_ex, np.maximum(dce[:60], 1e-4), "-", lw=1.4, color="tab:blue",
              label="stored dCE-load (the shadow)")
    db = np.maximum(dce[60:], 1e-4)
    ax2b.plot(x_bin, db, "s", ms=5, color="tab:blue")
    ax2b.set_yscale("log")
    ax2b.set_ylabel("stored e053c V-zero dCE (nats)", color="tab:blue")
    ax2.axvline(6, color="dimgray", ls="--", lw=1.1)
    ax2.set_title("(c) kernel vs CE-shadow by age — same spike or not?",
                  fontsize=9.5)

    # ---- panel 3: the registered correlation scatter
    y = curves["vzero"]
    m = np.isfinite(y)
    ax3.scatter(np.maximum(dce[m], 1e-5), np.maximum(y[m], 1e-6),
                c=["tab:red" if i < 60 else "tab:orange" for i in
                   np.where(m)[0]], s=28, alpha=0.85)
    for lo, hi, mm in [(0, 10, "tab:red"), (10, 60, "tab:brown")]:
        idx = [i for i in range(lo, min(hi, len(y))) if m[i]]
        if len(idx) > 2:
            ax3.plot([max(dce[i], 1e-5) for i in idx],
                     [max(y[i], 1e-6) for i in idx],
                     "-", lw=0.9, alpha=0.4, color=mm)
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("stored dCE-load(age) (nats, log)")
    ax3.set_ylabel("V-zero flip-rate(age) (log)")
    cp = M["correlations"]["primary_vzero_vs_dce"]
    ax3.text(0.03, 0.05,
             f"PRIMARY Pearson r = {cp['pearson']:.3f}\n"
             f"CI [{cp['ci'][0]:.3f}, {cp['ci'][1]:.3f}]\n"
             f"Spearman {cp['spearman']:.3f}  n={cp['n_points']}\n"
             f"bars: r>=0.8 shadow | r<0.5 dissociation",
             transform=ax3.transAxes, fontsize=8.5, va="bottom",
             family="monospace",
             bbox=dict(fc="white", ec="dimgray", alpha=0.85))
    ax3.set_title("(c) flip-rate vs CE-shadow across ages\n"
                  f"red=ages 1-10, brown=11-60, orange=old bins",
                  fontsize=9.5)

    # ---- panel 4: flip-target taxonomy
    tax = M["taxonomy"]["per_type"]
    kinds = ["runner_up", "top5", "outside_top5"]
    labels = ["runner-up", "clean top-5", "outside top-5"]
    xs = np.arange(3)
    bot = np.zeros(3)
    cols = ["tab:blue", "tab:cyan", "tab:gray"]
    for ki, kind in enumerate(kinds):
        vals = np.array([tax[TYPES[t]][kind] for t in range(3)])
        ax4.bar(xs, vals, 0.55, bottom=bot, color=cols[ki], alpha=0.85,
                label=labels[ki])
        bot += vals
    for t in range(3):
        ax4.text(xs[t], 1.03, f"n={tax[TYPES[t]]['n_flips']}", ha="center",
                 fontsize=8)
    dc = M["taxonomy"]["vswap_donor_continuation"]
    ax4.text(0.02, 0.60,
             f"V-swap donor-continuation:\n{dc['hits']}/{dc['n_vswap_flips']} "
             f"flips hit ({dc['hit_rate']:.3f})\nchance (all V-swap cells): "
             f"{dc['chance_all_vswap_cells']:.4f}",
             transform=ax4.transAxes, fontsize=8.5, va="top", family="monospace",
             bbox=dict(fc="white", ec="tab:purple", alpha=0.85))
    ax4.set_xticks(xs, [f"{t}\n({M['battery']['flip_rate_per_type'][t] * 100:.2f}% cells)"
                        for t in TYPES])
    ax4.set_ylabel("share of flips")
    ax4.set_ylim(0, 1.12)
    ax4.legend(fontsize=8, loc="upper right")
    ax4.set_title("(b) flip-target taxonomy + content-following witness",
                  fontsize=9.5)

    # ---- panel 5: margin structure (the deliberate covariate)
    ms = M["margin_structure"]
    qs = ["Q1 (near-ties)", "Q2", "Q3", "Q4 (confident)"]
    xs = np.arange(4)
    ax5.bar(xs - 0.21, ms["quartile_rates_pooled"], 0.42, color="tab:blue",
            label="pooled")
    for t, c in zip(range(3), ["tab:red", "tab:green", "tab:purple"]):
        ax5.bar(xs + 0.21, ms["quartile_rates_per_type"][TYPES[t]], 0.13,
                color=c, alpha=0.85, label=TYPES[t])
    ax5.set_xticks(xs, qs, fontsize=8.5)
    ax5.set_ylabel("flip rate within margin quartile")
    ax5.legend(fontsize=8)
    ax5.set_title(f"(covariate) flip rate by clean-margin quartile | flip-cell "
                  f"median margin {ms['flip_cell_margin_median']:.3f} vs all-cell "
                  f"{ms['all_cell_margin_median']:.3f}", fontsize=9.5)

    # ---- panel 6: opened coordinates per decision (sparse vs dense)
    oc = M["opened_coordinates"]
    for t, c in zip(range(3), ["tab:red", "tab:green", "tab:purple"]):
        h = np.asarray(oc["per_type_hist"][TYPES[t]], float)
        h = h / max(h.sum(), 1)
        ax6.plot(np.arange(len(h)), h, "o-", ms=3, lw=1.1, color=c,
                 label=f"{TYPES[t]} (med {oc['per_type_median'][TYPES[t]]:.0f})")
    ax6.set_xlabel("entries flipped per decision point (of 80)")
    ax6.set_ylabel("fraction of decision points")
    ax6.legend(fontsize=8)
    ax6.set_title(f"(structure) opened coordinates per decision — any-type "
                  f"median {oc['any_type_median']:.0f} of 80\n"
                  "(H-sparse-open: few; H-dense-averaging: ~0 single flips)",
                  fontsize=9.5)

    fig.suptitle(f"E084 READ KERNEL | {dec['fires']} | r = "
                 f"{dec['r_primary']:.3f} CI "
                 f"[{dec['r_primary_ci'][0]:.2f},{dec['r_primary_ci'][1]:.2f}] | "
                 f"battery-wide {dec['battery_flip_rate'] * 100:.3f}% | "
                 + dec["verdict"], fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
