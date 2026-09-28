"""E110 — the W001-REFINEMENT paper-prediction: the magnitude floor is a
PER-FIELD stream-share, not a per-entry amplitude detector. CPU-only.

CONTEXT (THINKING.md W001 REFINEMENT, worked through on paper while e108
ran): LN does not rescue small writes — it keeps the STREAM TOTAL bounded,
which EXPLAINS the e102 magnitude floor. Attention output = sum p_i V_i
enters the residual stream; LN normalizes the stream total per position.
If every anchor V is r x, the attention block's output is r x but the rest
of the stream (MLP writes, other blocks) is unchanged — so the anchor's
SHARE of the post-LN stream shrinks by r. The e102 floor (k=154: r=0.1
DAMAGED +1.087; realized ~0.559 retention HEALTHY +0.089, CI incl. 0) is a
SIGNAL-TO-NOISE floor in the post-LN stream. W001's registered
paper-prediction, verbatim: "the floor MOVES with the anchor's stream-share
— scale the OTHER contributions down (or the anchor count up) and the
per-entry floor drops; the floor is per-FIELD, not per-entry. This also
quietly re-derives the mass law: more entries = more share = each can be
quieter. The threshold law and the magnitude floor may be the SAME floor."
e110 tests the anchor-count-up leg: CROSS the retention ladder with the
field-size ladder.

DESIGN (registered here BEFORE compute; frozen from the tasking). Rig:
lab/e102_direction_magnitude.py's scaled-arm machinery (single-shot
matched-stream continuation at the e096/e102 intervention point) +
lab/e089_mass_response.py's k-ladder (random band subsets). Battery: the
e053c ctx-512 net (873,472 params, val CE 1.5227), seed-202 8-draw prompts,
seed-7 control free run, G3 gated against e080's stored none arm AND e102's
stored control continuation. Intervention point VERBATIM e102: g_int=250,
T_int=314, teacher-forced token at 314, band = self-generated positions
64..217 = ages 97..250 at query 314 — all 154 entries exist at t=314, so
single-shot is legal for ANY subset (e102's feasibility argument).

THE GRID: retention r in {0.1, 0.25, 0.56, 1.0} CROSSED with field size
k in {32, 64, 128, 154}. Cell (k, r): draw a random k-subset of the band
(the FIELD); scale its V-vectors to retention r (V <- r*V — e102 arm-c
machinery: direction preserved EXACTLY, cos = 1, norm ratio = r); ZERO the
band complement (V=0, e075/e085/e088/e089 removal semantics; K never
touched); single-shot immediately before the decode at 314. So the cell's
total retained field mass is proportional to r*k — the share axis.
  (154, 1.0) = the matched control (bitwise no-op; the instrument arm);
  (154, 0.1) = e102's scaled arm BITWISE (G5 gate vs the shipped artifact);
  (154, 0.25/0.56) = new rungs of e102's own magnitude ladder (replication
                     anchors: the r<1 x k=154 column);
  r=1.0 x k<154  = PURE REMOVAL at single-shot timing (the new cells —
                   e089 covered removal at r=1 only, on the 351-entry
                   final-frame band with dynamic age-97-crossing timing;
                   these are the same law at the e102 frame).
SUBSETS (RNG documented + frozen): numpy default_rng(seed=110), consumed
in (k-major, draw-minor, row-minor) order: for k in (32, 64, 128): for d in
range(D=3): for j in range(8): rng.choice(band, size=k, replace=False).
The subset keyed (k, d, row) is REUSED across the four r cells of its k —
the r-ladder within a k is subset-paired (subset noise cancels in
r-comparisons); 8 rows x 3 draws = 24 subsets per k<154 cell. k=154 cells
are deterministic (no subset) — run once per r.

READOUT (standard): clean-judge tail CE (448..511, clean full recompute, no
pruning) of the arm stream minus the MATCHED control stream (shared
continuation seed 960250, row-by-row matched until sampled divergence),
per row (mean over the row's D draws), then mean over B=8. Secondary
(registered, descriptive): the off-manifold attractor signature (judge CE
minus the run's own online tail CE; e080 bar 1.0 nat), first stream
divergence, and the r=1.0 column as a pure-removal reference curve
(overlaid on e089's stored per-k means — different band/timing, reference
only, no gate).

HEALTH BARS (e102 standard, frozen): healthy = mean cost < 0.3 nats;
damaged = mean cost > 1.0 nat; ambiguous = between (honest texture, no
registered cell).

REGISTERED BARS (the W001 prediction quantified; frozen from the tasking,
one direction slip flagged honestly — see bar 3):
  1. SHARE-LAW SHIFT (primary): the collapse threshold r*(k) — the minimum
     healthy retention on the r-grid — SHIFTS DOWN as k grows (more entries
     = more share = each can be quieter): r*(154) < r*(64) AND
     r*(32) >= r*(64), the second comparison firing STRICTLY (r*(32) >
     r*(64)) only when both are on-grid (both off-grid-high = tie at
     ">1.0", an honest non-fire). Bootstrap support: strict point ordering
     AND P(ordering | paired-row resampling) >= 0.95.
  2. FLAT / PER-ENTRY FLOOR (the null): r* flat in k — every on-grid r*(k)
     EQUAL (>= 2 ks on-grid required to call it). A fixed per-entry
     amplitude detector predicts NO shift. (Honest rider: pure-removal
     damage at small k can also push r* up at small k — the discriminator
     against that confound is bar 4.)
  3. LITERAL TASKING TEXT (reported, flagged): the tasking writes
     "r*(64) < r*(154) and r*(32) > r*(64)". The FIRST inequality
     inverts the direction its own lead sentence states ("the collapse
     threshold in r SHIFTS DOWN as k grows" = r*(154) < r*(64) = r*(64) >
     r*(154)); under BOTH candidate mechanisms (share law; per-entry floor
     + removal offset) r*(64) >= r*(154) generically, so the literal
     first inequality can only fire as a grid-quantization accident.
     e110 evaluates and reports BOTH readings; the dispatcher (owner of
     THINKING.md) adjudicates which was meant.
  4. SECONDARY — the SHARE LAW proper: the product r*(k) x k at the
     threshold should be roughly CONSTANT (total field mass ~ constant at
     the boundary). Reported as r*_cont(k) x k with paired-row bootstrap
     CIs, where r*_cont(k) is the 0.3-nat crossing interpolated in log2(r)
     between adjacent grid rungs (off-grid-high / off-grid-low /
     no-crossing reported honestly). Constancy (descriptive): point
     max/min <= 2 AND pairwise CI overlap. The per-entry-floor null
     predicts product ~ r_flat * k (rising in k).

Gates: G1 val CE vs e053c 1.5227 +-0.02; G2 params 873472; G3 battery
identity vs e080's stored none arm AND e102's stored control continuation
(clean-judge per-seq, 1e-4, bitwise flags); G4 instrument identity per
cell (prefix bitwise identical through 314; kept columns exact r-scaling —
norm dev < 1e-5 AND cos = 1; removed band-complement columns exactly 0;
non-band columns bitwise untouched; matched continuation seed; single-shot
by construction; (154,1.0) bitwise == the no-op control; band median norms
bitwise == e102's stored layer medians); G5 e102 replication (the
(154, 0.1) cell's clean-judge per-row AND cost-per-row vs runs/e102/
metrics.json scaled arm within 1e-4, bitwise flag).

Run:     python lab/e110_field_floor.py
Outputs: runs/e110/metrics.json + runs/e110/field_floor.png
Envelope: NO training, NO new automations; CPU-only (CUDA_VISIBLE_DEVICES=-1),
8 threads (T050), single step, minutes-scale.
No NOTES/THINKING/QUEUE/STATE edits; no commit (the dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu" (e053b..e102)
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

import json as _json  # noqa: E402
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
SEED_PROMPT, SEED_SAMPLE = 202, 7         # e053b/e053c/e069..e102 seeds
SEED_CONT = 960250                        # e102's single matched
                                          # continuation seed (g_int=250)
SEED_SUB = 110                            # e110: subset-sampling rng
B = 8
N_PROMPTS = 8
BOOT_N = 1000
THREADS = 8                               # T050: 12 spin-thrashes the box
torch.set_num_threads(THREADS)

P512 = T_TOTAL - 1                        # 511
TAIL = 64                                 # the e075..e102 tail window
KEY_T = (T_TOTAL - TAIL, T_TOTAL - 1)     # judged tail window (448, 511)
G = T_TOTAL - PROMPT_TOK                  # 448 generation steps

# ---- the intervention (single-shot, one frame; VERBATIM e102) -----------------
G_INT = 250                               # generation step of the shot
T_INT = PROMPT_TOK + G_INT                # 314: teacher-forced token position
GEN_FIRST = 64                            # first generated position
BAND = list(range(GEN_FIRST, T_INT - 96)) # 64..217: ages 97..250 at query 314
                                          # (e075's age>96 rule at the frame;
                                          # 154 entries — every one extant at
                                          # t=314, so single-shot is legal for
                                          # ANY subset: e102's feasibility)
FIRST_AFFECTED = T_INT + 1                # 315 (first free-sampled position)
TAIL_STEP0 = KEY_T[0] - FIRST_AFFECTED    # 133: tail slice start in free steps
BAND_ARR = np.arange(BAND[0], BAND[-1] + 1)          # 154 positions

# ---- the registered grid (tasking, frozen) ------------------------------------
KS = [32, 64, 128, 154]                   # field sizes (154 = whole band)
RS = [0.1, 0.25, 0.56, 1.0]               # retention ladder (1.0 = pristine)
D_DRAWS = 3                               # subset draws per k<154 cell
                                          # (8 rows x 3 draws = 24 subsets)

# ---- REGISTERED decision numbers (frozen, docstring verbatim) -----------------
HEALTHY_BAR = 0.3                         # mean cost < this -> healthy
DAMAGED_BAR = 1.0                         # mean cost > this -> damaged
CJ_GAP_BAR = 1.0                          # e080's off-manifold bar (nats)
BOOT_PROB = 0.95                          # ordering-support bar
PRODUCT_MAXMIN = 2.0                      # secondary constancy: max/min <= 2

# ---- reference numbers (protocol-identity gates + curve anchors) --------------
CKPT = REPO / "runs" / "checkpoints" / "e053c_ctx512.pt"
E053C_VAL_CE = 1.5226792494455974         # the net being interrogated
E080_METRICS = REPO / "runs" / "e080" / "metrics.json"
E089_METRICS = REPO / "runs" / "e089" / "metrics.json"
E102_METRICS = REPO / "runs" / "e102" / "metrics.json"
E096_MED_NORM_REF = [2.157031774520874, 1.5538004636764526,
                     1.9136782884597778, 1.8171889781951904]
E102_SCALED_MEAN = 1.0871107578277588     # e102 arm-c (r=0.1, k=154)
T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ------------------------------------------------- machinery (VERBATIM e102)

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


@torch.no_grad()
def prefill_batch(net: TinyGPT, idx: torch.Tensor):
    """Batched prefill, (B, Tp) -> last-position logits (B, V) + KV cache.
    [VERBATIM e075/e080/e085/e088/e089/e096/e102]"""
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
def decode_step_batch(net: TinyGPT, toks: torch.Tensor, pos: int, kv: list):
    """Batched incremental decode: (B,) tokens at position pos -> (B, V).
    [VERBATIM e075/e080/e085/e088/e089/e096/e102]"""
    Bb = toks.shape[0]
    x = net.wte(toks) + net.wpe(torch.full((Bb,), pos))
    H = net.cfg.n_head
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
        y = (probs @ v).transpose(1, 2).reshape(Bb, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
    return net.lm_head(net.ln_f(x))


@torch.no_grad()
def generate_control(net: TinyGPT, prompts, gen: torch.Generator):
    """Free-run 64->512 for B sequences (e075/e080/e085/e088/e089/e096/e102
    A-none VERBATIM stream math)."""
    Bb = len(prompts)
    idx = torch.stack(prompts)
    logits, kv = prefill_batch(net, idx)
    for g in range(G):
        t = PROMPT_TOK + g
        toks = torch.zeros(Bb, dtype=torch.long)
        for j in range(Bb):
            tok, _ce = sample_and_ce(logits[j], gen)
            toks[j] = tok
        idx = torch.cat([idx, toks[:, None]], 1)
        if t < T_TOTAL - 1:
            logits = decode_step_batch(net, toks, t, kv)
    return dict(idx=idx, kv=kv)


def band_median_norms(kv: list) -> list:
    """Per-layer MEDIAN ||V|| over all band vectors (rows x heads x 154
    positions) of a PRISTINE post-prefill cache — the dose unit's scale.
    [VERBATIM e096/e102]"""
    sel = torch.tensor(BAND, dtype=torch.long)
    meds = []
    for (_k, v) in kv:
        nrm = v[:, :, sel, :].norm(dim=-1)          # (B, H, n_band)
        meds.append(float(nrm.median()))
    return meds


@torch.no_grad()
def run_field_continuation(net: TinyGPT, prefix: torch.Tensor,
                           forced_tok: torch.Tensor, forced_pos: int,
                           seed: int, keep_per_row=None, r: float = 1.0):
    """e102's run_continuation generalized to PER-ROW kept-column sets —
    the (k, r) cell machinery.

    1. prefill(prefix) — prefix = control positions 0..forced_pos-1.
    2. if keep_per_row is not None: single-shot band transform BEFORE the
       decode at forced_pos (e096/e102 top-of-the-event timing; first
       affected sample = forced_pos+1). For each row j: the kept columns
       keep_per_row[j] (a LongTensor of band positions) get V <- r*V
       (direction preserved EXACTLY, norm ratio r — e102 arm-c machinery);
       every OTHER band column is V-zeroed (e075/e085 removal semantics);
       K is never touched. keep_per_row=None (or r=1.0 with the whole band
       kept) is the matched no-op control — the bitwise identity path.
    3. teacher-force the token at forced_pos (clean control token), then
       free-run to 511 with the shared row-order generator (e053b stream
       math) — all cells + control share the seed, so streams are
       row-by-row matched until sampled divergence.

    Tracks online CE + full-softmax entropy of every emitted free token.
    Returns idx, kv, fired-info dict (transform identity + realized
    vector-space stats), ce_s, ent_s.
    """
    R = prefix.shape[0]
    gen = torch.Generator().manual_seed(seed)
    logits, kv = prefill_batch(net, prefix)
    whole_band = torch.tensor(BAND, dtype=torch.long)
    is_identity = (keep_per_row is None) or (
        r == 1.0 and all(torch.equal(torch.as_tensor(kp, dtype=torch.long),
                                     whole_band) for kp in keep_per_row))
    fired = dict(applied=not is_identity, r=r,
                 n_kept_per_row=[], n_removed_per_row=[],
                 norm_dev=0.0, min_cos=1.0, mean_cos=None,
                 mean_norm_ratio=None, removed_exactly_zero=True,
                 nonband_untouched=True)
    if not is_identity:
        sel_band = whole_band
        pre_v = [v.clone() for (_k, v) in kv]       # identity snapshot
        coss, ratios = [], []
        band_set = set(BAND)
        for j in range(R):
            kp = torch.as_tensor(keep_per_row[j], dtype=torch.long)
            comp = torch.tensor(sorted(band_set - set(kp.tolist())),
                                dtype=torch.long)
            fired["n_kept_per_row"].append(int(kp.numel()))
            fired["n_removed_per_row"].append(int(comp.numel()))
            for (_k, v) in kv:
                old = v[j, :, kp, :].clone()                  # (H,n,d)
                if r != 1.0:
                    v[j, :, kp, :] = r * old
                if comp.numel():
                    v[j, :, comp, :] = 0.0
        # ---- transform identity: kept norms at r*old exactly (cos = 1);
        #      removed columns exactly 0; non-band bitwise untouched
        for (_k, v), pv in zip(kv, pre_v):
            for j in range(R):
                kp = torch.as_tensor(keep_per_row[j], dtype=torch.long)
                comp = torch.tensor(sorted(band_set - set(kp.tolist())),
                                    dtype=torch.long)
                old = pv[j, :, kp, :].clone()
                new = v[j, :, kp, :]
                nrm = old.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                nnew = new.norm(dim=-1, keepdim=True)
                fired["norm_dev"] = max(fired["norm_dev"], float(
                    (nnew - r * nrm).abs().max()))
                cos = (old * new).sum(-1) / (nrm.squeeze(-1)
                                             * nnew.squeeze(-1)).clamp_min(1e-12)
                fired["min_cos"] = min(fired["min_cos"], float(cos.min()))
                coss.append(cos.reshape(-1))
                ratios.append((nnew / nrm).reshape(-1))
                if comp.numel():
                    fired["removed_exactly_zero"] &= bool(
                        (v[j, :, comp, :] == 0.0).all())
            nb_lo = torch.arange(0, GEN_FIRST)                # prompt cols
            nb_hi = torch.arange(BAND[-1] + 1, forced_pos)    # young cols
            fired["nonband_untouched"] &= bool(
                torch.equal(v[:, :, nb_lo, :], pv[:, :, nb_lo, :]))
            fired["nonband_untouched"] &= bool(
                torch.equal(v[:, :, nb_hi, :], pv[:, :, nb_hi, :]))
        fired["mean_cos"] = float(torch.cat(coss).mean())
        fired["mean_norm_ratio"] = float(torch.cat(ratios).mean())
    idx = torch.cat([prefix, forced_tok[:, None]], 1)
    logits = decode_step_batch(net, forced_tok, forced_pos, kv)
    n_free = T_TOTAL - 1 - forced_pos
    ce_s = np.zeros((R, n_free), float)
    ent_s = np.zeros((R, n_free), float)
    for s in range(n_free):
        pos = forced_pos + 1 + s
        new = torch.zeros(R, dtype=torch.long)
        for j in range(R):
            tok, ce = sample_and_ce(logits[j], gen)
            new[j] = tok
            ce_s[j, s] = ce
        p = torch.softmax(logits.float(), -1)
        ent_s[:, s] = (-(p * p.clamp_min(1e-12).log()).sum(-1)).numpy()
        idx = torch.cat([idx, new[:, None]], 1)
        if pos < T_TOTAL - 1:
            logits = decode_step_batch(net, new, pos, kv)
    return idx, kv, fired, ce_s, ent_s


def judge_windows(all_lg: torch.Tensor, idx: torch.Tensor, windows):
    """Clean-net CE of target windows [(lo, hi), ...] (queries lo-1..hi-1).
    Returns dict window -> (R,) mean CE per row. [VERBATIM e085..e102]"""
    out = {}
    for (lo, hi) in windows:
        lg = all_lg[:, lo - 1:hi, :]
        tgt = idx[:, lo:hi + 1]
        lp = torch.log_softmax(lg.float(), -1)
        out[(lo, hi)] = -lp.gather(2, tgt[:, :, None]).squeeze(2).mean(1).numpy()
    return out


# ------------------------------------------------------------- statistics

def boot_rows(arrs, fn, n: int = BOOT_N, seed: int = 0):
    """Paired bootstrap over the 8 rows (runs): resample row ids with
    replacement, recompute fn on the resampled arrays. [VERBATIM e096/e102]"""
    S = np.asarray(arrs[0]).shape[0]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sel = rng.integers(0, S, S)
        try:
            v = fn(*[np.asarray(a)[sel] for a in arrs])
        except Exception:
            v = None
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _fmt_ci(ci):
    return f"[{ci[0]:+.3f},{ci[1]:+.3f}]"


def _wrap(text: str, width: int):
    import textwrap
    return textwrap.wrap(text, width=width)


def health_of(cost: float) -> str:
    """The registered health code of a cell's mean clean-judge cost
    [VERBATIM e102's bars]."""
    if cost < HEALTHY_BAR:
        return "healthy"
    if cost > DAMAGED_BAR:
        return "damaged"
    return "ambiguous"


def rstar_code(col_means: dict) -> tuple:
    """Grid resolution r*(k): the minimum healthy retention on the r-grid.

    Returns (code, value, status): code = index into RS (0..3), 4 for
    off-grid-high (no healthy cell — even r=1.0 not healthy), -1 for
    off-grid-low (r=0.1 already healthy — floor below the grid). Status
    flags non-monotone columns honestly (a healthy rung with a
    damaged/ambiguous rung ABOVE it)."""
    healths = {r: health_of(col_means[r]) for r in RS}
    if healths[RS[0]] == "healthy":
        return -1, None, "off-low (r=0.1 already healthy)"
    healthy_rs = [r for r in RS if healths[r] == "healthy"]
    if not healthy_rs:
        return 4, None, "off-grid-high (no healthy cell)"
    rmin = healthy_rs[0]
    mono = all(healths[r] == "healthy" for r in RS if r > rmin)
    return (RS.index(rmin), rmin,
            "on-grid" if mono else "on-grid (NON-MONOTONE column flagged)")


def rstar_cont(col_means: dict) -> tuple:
    """Continuous 0.3-nat crossing interpolated in log2(r) between adjacent
    grid rungs (the RIGHTMOST downward crossing, so everything above it is
    healthy). Returns (value, status)."""
    if col_means[RS[-1]] > HEALTHY_BAR:
        return float("nan"), "off-grid-high"
    if col_means[RS[0]] < HEALTHY_BAR:
        return float("nan"), "off-grid-low"
    x = np.log2(np.array(RS))
    y = np.array([col_means[r] for r in RS])
    for i in range(len(RS) - 2, -1, -1):        # rightmost bracket wins
        if y[i] >= HEALTHY_BAR > y[i + 1]:
            t = x[i] + (HEALTHY_BAR - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
            return float(2.0 ** t), "interp"
    return float("nan"), "no-crossing"


# ---------------------------------------------------------------------- main


def main():
    assert torch.cuda.device_count() == 0, "CPU-only violated: CUDA visible"
    assert common.DEVICE == "cpu", f"common.DEVICE={common.DEVICE} != cpu"
    out_dir = run_dir("e110")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=False)

    # ---- battery: rebuild corpus + net EXACTLY as e053c/e069..e102 did
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

    # ---- control free run (seed-7 A-none convention)
    log("control battery: seed-7 free run (e075/e080/e085/e088/e089/e096/e102)")
    gen7 = torch.Generator().manual_seed(SEED_SAMPLE)
    run1 = generate_control(net, prompts8, gen7)
    idx1 = run1["idx"]
    log(f"  control run done ({T_TOTAL} positions x {B} rows)")

    # ---- G3: control battery vs e080's stored none arm + e102's control cj
    cj1 = judge_windows(manual_all_logits(net, idx1), idx1, [KEY_T])[KEY_T]
    g3 = dict(ref_files=[str(E080_METRICS), str(E102_METRICS)],
              note="e110 has no static arm; G3 scope = control-battery "
                   "identity vs e080's stored none arm (clean-judge per-seq, "
                   "1e-4) AND vs e102's stored control continuation "
                   "(continuation-level twin; e102's scoped G3 convention)")
    ok3 = True
    if E080_METRICS.exists():
        e080 = _json.load(open(E080_METRICS))
        ref_cj = np.asarray(e080["arms"]["none"]["clean_judge_tail_ce"]["per_seq"])
        dev3 = float(np.abs(cj1 - ref_cj).max())
        bit80 = bool(np.array_equal(cj1, ref_cj))
        g3.update(e080_clean_judge_max_dev=dev3, e080_bitwise=bit80)
        ok3 &= bool(dev3 < 1e-4)
        log(f"G3 vs e080 A-none: clean-judge dev {dev3:.2e} (bitwise {bit80})")
    else:
        ok3 = False
        g3["note"] += " | runs/e080/metrics.json missing"
    g3["ok"] = bool(ok3)
    gates["G3_protocol_identity"] = g3
    log(f"G3 battery leg -> {'PASS' if g3['ok'] else 'FAIL'} (continuation "
        f"leg vs e102's stored control checked after the control run)")

    # ---- e089 reference curve (removal law, different band/timing: ref only)
    e089_per_k = None
    if E089_METRICS.exists():
        e089 = _json.load(open(E089_METRICS))
        e089_per_k = {int(k): float(v["mean"]) for k, v in
                      e089["summary"]["per_k"].items()}
        log("e089 removal reference (dynamic timing, 351-band): "
            + " ".join(f"k={k}:{v:+.3f}" for k, v in sorted(e089_per_k.items())))

    # ---- the subsets (RNG frozen, docstring order)
    rng_sub = np.random.default_rng(SEED_SUB)
    subsets = {k: [[None] * B for _ in range(D_DRAWS)] for k in KS[:-1]}
    for k in KS[:-1]:                          # 32, 64, 128 (154 = whole band)
        for d in range(D_DRAWS):
            for j in range(B):
                subsets[k][d][j] = rng_sub.choice(BAND_ARR, size=k,
                                                  replace=False)
    log(f"subsets drawn (seed {SEED_SUB}, (k,d,row) order, reused across r; "
        f"{D_DRAWS} draws x {B} rows per k<154 cell)")

    # ---- the matched control continuation = the (154, 1.0) identity cell
    prefix = idx1[:, :T_INT]
    forced = idx1[:, T_INT]
    log(f"continuations: prefix 0..{T_INT - 1}, forced token at {T_INT}, "
        f"band = positions {BAND[0]}..{BAND[-1]} ({len(BAND)} entries, ages "
        f"97..{T_INT - BAND[0]} at query {T_INT}), seed {SEED_CONT}")
    idx_ctl, kv_ctl, fired_ctl, ce_ctl, ent_ctl = run_field_continuation(
        net, prefix, forced, T_INT, SEED_CONT, keep_per_row=None, r=1.0)
    med_norm = band_median_norms(kv_ctl)
    med_dev = (max(abs(a - b) for a, b in zip(med_norm, E096_MED_NORM_REF))
               if len(med_norm) == len(E096_MED_NORM_REF) else float("inf"))
    J_ctl = judge_windows(manual_all_logits(net, idx_ctl), idx_ctl,
                          [KEY_T])[KEY_T]
    log(f"matched control done | tail cj {float(J_ctl.mean()):.4f} | median "
        f"band ||V|| per layer {[round(m, 4) for m in med_norm]} (vs e102 "
        f"stored max dev {med_dev:.2e})")
    # ---- G3 continuation leg: the matched control vs e102's stored control
    #      (the same-seed continuation twin — the object e102 shipped)
    if E102_METRICS.exists():
        e102 = _json.load(open(E102_METRICS))
        ref102 = np.asarray(e102["control_run"]["clean_judge_tail_ce"])
        dev102 = float(np.abs(J_ctl - ref102).max())
        bit102 = bool(np.array_equal(J_ctl, ref102))
        g3.update(e102_control_max_dev=dev102, e102_bitwise=bit102)
        g3["ok"] = bool(g3.get("e080_clean_judge_max_dev", 1.0) < 1e-4
                        and dev102 < 1e-4)
        gates["G3_protocol_identity"] = g3
        log(f"G3 vs e102 control continuation: dev {dev102:.2e} (bitwise "
            f"{bit102}) -> G3 final {'PASS' if g3['ok'] else 'FAIL'}")
    else:
        g3["ok"] = False
        g3["note"] += " | runs/e102/metrics.json missing (continuation leg)"
        gates["G3_protocol_identity"] = g3
        log("G3 -> FAIL (runs/e102/metrics.json missing)")

    whole_band = [torch.tensor(BAND, dtype=torch.long)] * B

    # ---- the grid: per-cell per-row J (mean over draws) + texture
    cellJ = {}          # (k, r) -> (8,) per-row mean clean-judge tail CE
    cellJ_draws = {}    # (k, r) -> (n_draws, 8) per-draw J (texture)
    cellS = {}          # (k, r) -> per-cell bookkeeping
    for k in KS:
        for r in RS:
            if k == 154 and r == 1.0:
                cellJ[(k, r)] = J_ctl.copy()
                cellJ_draws[(k, r)] = J_ctl[None, :].copy()
                cellS[(k, r)] = dict(fired=fired_ctl, idx=idx_ctl,
                                     ce=ce_ctl, ent=ent_ctl,
                                     is_control=True, n_draws=1)
                log(f"  cell k={k:>3} r={r:.2f}: CONTROL (bitwise identity)")
                continue
            Js, extra = [], []
            if k == 154:
                draw_sets = [whole_band]
            else:
                draw_sets = [subsets[k][d] for d in range(D_DRAWS)]
            for keep in draw_sets:
                idx_a, kv_a, fired_a, ce_a, ent_a = run_field_continuation(
                    net, prefix, forced, T_INT, SEED_CONT,
                    keep_per_row=keep, r=r)
                Js.append(judge_windows(manual_all_logits(net, idx_a),
                                        idx_a, [KEY_T])[KEY_T])
                extra.append(dict(idx=idx_a, ce=ce_a, ent=ent_a,
                                  fired=fired_a))
            cellJ[(k, r)] = np.mean(np.stack(Js), axis=0)
            cellJ_draws[(k, r)] = np.stack(Js)
            f0 = extra[0]["fired"]
            cellS[(k, r)] = dict(
                fired=f0, idx=extra[0]["idx"],
                online_tail=float(np.mean([e["ce"][:, TAIL_STEP0:].mean()
                                           for e in extra])),
                ent_tail=float(np.mean([e["ent"][:, TAIL_STEP0:].mean()
                                        for e in extra])),
                is_control=False, n_draws=len(extra),
                draw_spread_max=float(np.abs(np.stack(Js)
                                             - np.stack(Js).mean(0)).max()))
            log(f"  cell k={k:>3} r={r:.2f}: cj tail "
                f"{float(cellJ[(k, r)].mean()):.4f} | kept-norm dev "
                f"{f0['norm_dev']:.1e} | min cos {f0['min_cos']:.6f} | "
                f"removed==0 {f0['removed_exactly_zero']} | draw spread "
                f"{cellS[(k, r)]['draw_spread_max']:.3f}")

    # ---- per-cell stats + the r*(k) machinery
    per = {}
    for k in KS:
        for r in RS:
            c = cellJ[(k, r)] - J_ctl
            ci = boot_rows([c], lambda a: float(a.mean()))
            S = cellS[(k, r)]
            eq = torch.eq(S["idx"], idx_ctl)
            fdr = []
            for rr in range(S["idx"].shape[0]):
                nz = (~eq[rr]).nonzero().flatten()
                fdr.append(int(nz[0].item()) if len(nz) else None)
            per[(k, r)] = dict(
                cost_per_row=c.tolist(), mean=float(c.mean()), ci=ci,
                n_worse=int((c > 0).sum()),
                n_stream_identical=B - sum(d is not None for d in fdr),
                first_div=(min(d for d in fdr if d is not None)
                           if any(d is not None for d in fdr) else None),
                n_rows_diverged=sum(d is not None for d in fdr),
                health=health_of(float(c.mean())),
                clean_judge_mean=float(cellJ[(k, r)].mean()),
                online_tail=S.get("online_tail",
                                  float(np.asarray(ce_ctl[:, TAIL_STEP0:]
                                                   ).mean())),
                ent_tail=S.get("ent_tail",
                               float(np.asarray(ent_ctl[:, TAIL_STEP0:]
                                                ).mean())),
                n_draws=S["n_draws"],
                draw_spread_max=S.get("draw_spread_max", 0.0))
            per[(k, r)]["cj_gap_vs_online"] = float(
                per[(k, r)]["clean_judge_mean"] - per[(k, r)]["online_tail"])
            per[(k, r)]["signature_fires"] = bool(
                per[(k, r)]["cj_gap_vs_online"] > CJ_GAP_BAR)

    col_means = {k: {r: per[(k, r)]["mean"] for r in RS} for k in KS}
    rstar = {}
    for k in KS:
        code, val, status = rstar_code(col_means[k])
        cval, cstatus = rstar_cont(col_means[k])
        rstar[k] = dict(code=code, value=val, status=status,
                        cont_value=(None if not np.isfinite(cval)
                                    else float(cval)),
                        cont_status=cstatus,
                        product=(None if not np.isfinite(cval)
                                 else float(cval * k)))
        log(f"r*({k}): grid {val if val is not None else status} | cont "
            f"{('%.3f' % cval) if np.isfinite(cval) else cstatus}"
            + (f" | r*k {rstar[k]['product']:.1f}"
               if rstar[k]["product"] else ""))

    # ---- joint paired-row bootstrap: r* distributions + ordering support
    rng = np.random.default_rng(0)
    codes_b = {k: [] for k in KS}
    conts_b = {k: [] for k in KS}
    prods_b = {k: [] for k in KS}
    for _ in range(BOOT_N):
        sel = rng.integers(0, B, B)
        cm = {k: {r: float((cellJ[(k, r)] - J_ctl)[sel].mean()) for r in RS}
              for k in KS}
        for k in KS:
            codes_b[k].append(rstar_code(cm[k])[0])
            cv, cs_ = rstar_cont(cm[k])
            conts_b[k].append(cv if np.isfinite(cv) else np.nan)
            prods_b[k].append(cv * k if np.isfinite(cv) else np.nan)
    P_share = float(np.mean([a < b for a, b in
                             zip(codes_b[154], codes_b[64])]))
    P_rise = float(np.mean([a > b for a, b in
                            zip(codes_b[32], codes_b[64])]))
    on_grid = [k for k in KS if rstar[k]["code"] in (0, 1, 2, 3)]
    flat_point = (len(on_grid) >= 2
                  and len({rstar[k]["code"] for k in on_grid}) == 1)
    def _flat_at(i: int) -> bool:
        ong = [codes_b[k][i] for k in KS if codes_b[k][i] in (0, 1, 2, 3)]
        return len(ong) >= 2 and len(set(ong)) == 1
    P_flat = float(np.mean([_flat_at(i) for i in range(BOOT_N)]))
    rstar_ci = {k: dict(
        grid=([float(np.nanpercentile(codes_b[k], 2.5)),
               float(np.nanpercentile(codes_b[k], 97.5))] if codes_b[k]
              else [float("nan")] * 2),
        cont=[float(np.nanpercentile(conts_b[k], 2.5)),
              float(np.nanpercentile(conts_b[k], 97.5))],
        product=[float(np.nanpercentile(prods_b[k], 2.5)),
                 float(np.nanpercentile(prods_b[k], 97.5))]) for k in KS}

    # products: descriptive constancy (point max/min <= 2 AND pairwise overlap)
    prod_vals = {k: rstar[k]["product"] for k in KS
                 if rstar[k]["product"] is not None}
    prod_keys = list(prod_vals)
    prod_const = False
    prod_report = "insufficient on-threshold ks"
    if len(prod_vals) >= 2:
        pmx, pmn = max(prod_vals.values()), min(prod_vals.values())
        overlap = all(not (prod_vals[a] < rstar_ci[b]["product"][0]
                           or prod_vals[b] < rstar_ci[a]["product"][0])
                      for a in prod_keys for b in prod_keys if a != b)
        prod_const = bool(pmx / max(pmn, 1e-9) <= PRODUCT_MAXMIN and overlap)
        prod_report = (f"products { {k: round(v,1) for k,v in prod_vals.items()} }"
                       f" | max/min {pmx/max(pmn,1e-9):.2f} | pairwise CI "
                       f"overlap {overlap}")

    # ---- G4 instrument identity
    g4_cells = {}
    ok4 = True
    for (k, r), S in cellS.items():
        f = S["fired"]
        pre_ok = bool(torch.equal(S["idx"][:, :T_INT + 1],
                                  idx_ctl[:, :T_INT + 1]))
        if S["is_control"]:
            cell_ok = pre_ok and (not f["applied"])
        else:
            cell_ok = (pre_ok and f["applied"]
                       and f["norm_dev"] < 1e-5 and f["min_cos"] > 1.0 - 1e-5
                       and f["removed_exactly_zero"]
                       and f["nonband_untouched"]
                       and f["n_kept_per_row"] == [k] * B
                       and f["n_removed_per_row"] == [154 - k] * B)
        g4_cells[f"k={k},r={r}"] = dict(prefix_identical=pre_ok,
                                        applied=f["applied"],
                                        norm_dev=f["norm_dev"],
                                        min_cos=f["min_cos"],
                                        removed_zero=f["removed_exactly_zero"],
                                        nonband_untouched=f["nonband_untouched"],
                                        n_kept=sorted(set(f["n_kept_per_row"])),
                                        n_removed=sorted(set(f["n_removed_per_row"])),
                                        ok=cell_ok)
        ok4 &= cell_ok
    bit_ctl = bool(torch.equal(cellS[(154, 1.0)]["idx"], idx_ctl))
    g4 = dict(
        cells=g4_cells,
        control_is_identity_cell=bit_ctl,
        control_applied_nothing=(not fired_ctl["applied"]),
        matched_seed=SEED_CONT, single_shot=True,
        band=dict(positions=[BAND[0], BAND[-1]], n=len(BAND),
                  ages_at_query=[T_INT - BAND[-1], T_INT - BAND[0]]),
        subsets=dict(seed=SEED_SUB, order="(k, draw, row); rng.choice(band, "
                     f"size=k, replace=False); D={D_DRAWS}; reused across r",
                     per_cell=8 * D_DRAWS),
        med_norm=dict(per_layer=med_norm, e102_ref=E096_MED_NORM_REF,
                      max_dev=med_dev, ok=bool(med_dev < 1e-6)))
    g4["ok"] = bool(ok4 and bit_ctl and (not fired_ctl["applied"])
                    and med_dev < 1e-6)
    gates["G4_instrument_identity"] = g4
    log(f"G4 instrument identity: {sum(c['ok'] for c in g4_cells.values())}"
        f"/{len(g4_cells)} cells exact | control identity {bit_ctl} | "
        f"med_norm dev {med_dev:.1e} -> {'PASS' if g4['ok'] else 'FAIL'}")

    # ---- G5 e102 replication ((154, 0.1) vs the shipped scaled arm)
    g5 = dict(ref=str(E102_METRICS),
              rule="the (k=154, r=0.1) cell's clean-judge per-seq AND "
                   "cost-per-row vs runs/e102/metrics.json scaled arm "
                   "within 1e-4")
    ok5, dev_cj, dev_cost = False, float("nan"), float("nan")
    if E102_METRICS.exists():
        sa = e102["summary"]["per_arm"]["scaled"]
        ref_rows = np.asarray(sa["cost_per_row"])
        cost_cell = cellJ[(154, 0.1)] - J_ctl
        dev_cj = float(np.abs(cellJ[(154, 0.1)]
                              - np.asarray(sa["clean_judge"]["per_row"])).max())
        dev_cost = float(np.abs(cost_cell - ref_rows).max())
        bit5 = bool(np.array_equal(cellJ[(154, 0.1)],
                                   np.asarray(sa["clean_judge"]["per_row"])))
        g5.update(cell_cj_max_dev=dev_cj, cell_cj_bitwise=bit5,
                  cell_cost_max_dev=dev_cost, cell_mean=float(cost_cell.mean()),
                  e102_scaled_mean=E102_SCALED_MEAN)
        ok5 = bool(dev_cj < 1e-4 and dev_cost < 1e-4)
        log(f"G5 e102 replication: (154,0.1) cj dev {dev_cj:.2e} (bitwise "
            f"{bit5}) | cost dev {dev_cost:.2e} | mean "
            f"{cost_cell.mean():+.4f} vs e102 {E102_SCALED_MEAN:+.4f}")
    else:
        g5["note"] = "runs/e102/metrics.json missing"
    g5["ok"] = bool(ok5)
    gates["G5_e102_replication"] = g5
    log(f"G5 -> {'PASS' if g5['ok'] else 'FAIL'}")

    # ================================================== the registered decision
    share_point = rstar[154]["code"] < rstar[64]["code"]
    rise_point = rstar[32]["code"] > rstar[64]["code"]
    both_offhigh = (rstar[32]["code"] == 4 and rstar[64]["code"] == 4)
    bar1 = bool(share_point and (rise_point or both_offhigh)
                and P_share >= BOOT_PROB
                and (P_rise >= BOOT_PROB or both_offhigh))
    bar2 = bool(flat_point)
    bar3_literal = bool(rstar[64]["code"] < rstar[154]["code"]
                        and rise_point)
    rs = {k: (rstar[k]["value"] if rstar[k]["value"] is not None
              else rstar[k]["status"]) for k in KS}
    lines_tbl = " | ".join(f"k={k}: r*={rs[k]}" for k in KS)

    if bar1 and not bar2:
        clause = ("SHARE-LAW SHIFT FIRES (W001 direction): r* moves DOWN "
                  "with k — the floor is PER-FIELD")
        verdict = (
            f"The minimum healthy retention falls as the field grows "
            f"({lines_tbl}): more entries = more post-LN stream-share = "
            f"each entry can be quieter, exactly W001's refinement. The "
            f"magnitude floor is not a per-entry amplitude detector — it "
            f"is the field's signal-to-noise share in the normalized "
            f"stream. Secondary (the share law proper): r**(k) x k at the "
            f"threshold {prod_report}. Removal rider: the r=1.0 column "
            f"(pure removal, single-shot) costs "
            + " ".join(f"k={k}:{per[(k, 1.0)]['mean']:+.3f}" for k in KS)
            + f" — any small-k rise in r* must be read against this "
            f"removal offset (bar 4 separates them).")
    elif bar2 and not bar1:
        clause = ("FLAT / PER-ENTRY FLOOR (the null): r* does not move "
                  "with k")
        verdict = (
            f"r* is flat in k ({lines_tbl}): the collapse floor is a "
            f"fixed PER-ENTRY amplitude property — W001's stream-share "
            f"refinement is falsified in its anchor-count-up leg; LN does "
            f"not make the floor field-relative at these scales. Products "
            f"{prod_report} (the null predicts product ~ r_flat x k, "
            f"rising in k).")
    else:
        clause = ("TEXTURE (mixed grid — honest; see the r* table and "
                  "removal rider)")
        verdict = (
            f"The grid does not land cleanly in either registered cell "
            f"({lines_tbl}). Ordering support: P(r*(154)<r*(64))="
            f"{P_share:.3f}, P(r*(32)>r*(64))={P_rise:.3f} "
            f"(both-'>1.0' tie at k=32/64: {both_offhigh}); flat-point "
            f"{flat_point} (P={P_flat:.3f}). Literal tasking text "
            f"(r*(64)<r*(154) AND r*(32)>r*(64)): {bar3_literal} — note "
            f"its first inequality inverts the 'floor shifts DOWN as k "
            f"grows' direction (flagged in the docstring; dispatcher "
            f"adjudicates). Products {prod_report}. The removal column "
            f"(r=1.0) and the draw spread decide whether the small-k "
            f"cells are floor-limited or removal-limited.")
    log("r* table: " + lines_tbl)
    log(f"bar support: P(r*154<r*64)={P_share:.3f} P(r*32>r*64)={P_rise:.3f} "
        f"| flat {flat_point} (P={P_flat:.3f}) | literal tasking pair "
        f"{bar3_literal} | products {prod_report}")
    log(f"REGISTERED DECISION [{clause}]: {verdict}")

    # ---------------------------------------------------------------- metrics
    metrics = dict(
        experiment="e110_field_floor",
        purpose="W001-REFINEMENT paper-prediction: the e102 magnitude floor "
                "is a post-LayerNorm stream-SHARE (signal-to-noise), not a "
                "per-entry amplitude detector — so the floor should be "
                "PER-FIELD (move with the field's total share, i.e. with "
                "entry count), not per-entry. Crossed design: retention "
                "r x field size k at the e102 intervention point (g=250, "
                "band 64..217, single-shot, matched-stream, clean-judged); "
                "kept k-subset scaled to r, band complement V-zeroed. "
                "Bars: r* shifts DOWN with k (share law) vs flat (per-entry "
                "floor); secondary r*(k)*k ~ const.",
        started=started, wall_s=elapsed(), threads=THREADS, cpu_only=True,
        net=dict(ckpt=str(CKPT), arch=dict(n_layer=4, n_head=4, n_embd=128,
                                           block_size=T_TOTAL, vocab=65),
                 params=n_params, val_ce=val_ce, val_ce_e053c=E053C_VAL_CE),
        seeds=dict(corpus=1337, prompts=SEED_PROMPT, sampling=SEED_SAMPLE,
                   subsets=SEED_SUB, continuation=SEED_CONT, bootstrap=0),
        protocol=dict(
            B=B, prompt_tokens=PROMPT_TOK, t_total=T_TOTAL, temp=TEMP,
            topk=TOPK, tail_window=list(KEY_T), boot_n=BOOT_N,
            grid=dict(k=KS, r=RS, draws_per_cell=D_DRAWS,
                      rule="cell (k,r): random k-subset of the 154-entry "
                           "band scaled to retention r (V<-r*V, cos=1); "
                           "band complement V-zeroed (removal); subsets "
                           "keyed (k,d,row) REUSED across r (paired "
                           "r-ladder); k=154 deterministic"),
            intervention=dict(
                g_int=G_INT, t_int=T_INT, timing="single-shot, immediately "
                "before the decode at t_int (e096/e102 top-of-the-event); "
                "token at t_int teacher-forced; first affected sample = "
                f"{FIRST_AFFECTED}",
                band=dict(positions=[BAND[0], BAND[-1]], n=len(BAND),
                          rule="self-generated entries with age > 96 at "
                               "the frame (e075 rule): positions 64..217"),
                v_only=True),
            readouts=dict(
                cost="clean-judge tail CE (448..511) of the cell stream "
                     "minus the MATCHED control stream, per row (mean over "
                     "the row's draws) then mean over B=8",
                attractor="judge CE minus own online tail CE (e080 bar "
                          "1.0 nat) — secondary/descriptive",
                removal_column="the r=1.0 column is pure removal at "
                               "single-shot timing; e089's stored per-k "
                               "means overlaid as reference (different "
                               "band/timing, no gate)"),
            bars=dict(
                healthy=f"mean cost < {HEALTHY_BAR} nats",
                damaged=f"mean cost > {DAMAGED_BAR} nat",
                ambiguous=f"in [{HEALTHY_BAR}, {DAMAGED_BAR}]",
                bar1_share_law=f"r*(154) < r*(64) AND r*(32) >= r*(64) "
                               f"(strict when both on-grid; both-off-high "
                               f"= honest tie), each ordering with bootstrap "
                               f"P >= {BOOT_PROB}",
                bar2_flat="all on-grid r*(k) equal (>=2 ks on-grid)",
                bar3_literal_tasking="tasking text 'r*(64) < r*(154) and "
                                     "r*(32) > r*(64)' — first inequality "
                                     "inverts the stated direction; "
                                     "reported, flagged, both readings "
                                     "evaluated",
                bar4_share_product=f"r*_cont(k)*k ~ constant (max/min <= "
                                   f"{PRODUCT_MAXMIN} AND pairwise CI "
                                   f"overlap) — descriptive",
                null="fixed per-entry floor: r* flat in k")),
        gates=gates,
        control_run=dict(clean_judge_tail_ce=J_ctl.tolist(),
                         online_tail=float(np.asarray(
                             ce_ctl[:, TAIL_STEP0:]).mean()),
                         ent_tail=float(np.asarray(
                             ent_ctl[:, TAIL_STEP0:]).mean())),
        references=dict(e102_scaled_mean=E102_SCALED_MEAN,
                        e089_per_k={str(k): v for k, v in
                                    (e089_per_k or {}).items()}),
        summary=dict(
            per_cell={f"k={k},r={r}": per[(k, r)] for k in KS for r in RS},
            col_means={f"k={k}": {str(r): col_means[k][r] for r in RS}
                       for k in KS},
            rstar={str(k): rstar[k] for k in KS},
            rstar_ci={str(k): rstar_ci[k] for k in KS},
            ordering=dict(P_r154_lt_r64=P_share, P_r32_gt_r64=P_rise,
                          P_flat=P_flat, both_offhigh_32_64=both_offhigh,
                          literal_tasking_pair=bar3_literal,
                          flat_point=flat_point),
            product=dict(constant=prod_const, report=prod_report,
                         values={str(k): rstar[k]["product"]
                                 for k in KS}),
            attractor=dict(gap_bar=CJ_GAP_BAR,
                           fires={f"k={k},r={r}":
                                  per[(k, r)]["signature_fires"]
                                  for k in KS for r in RS})),
        registered_decision=dict(clause=clause, verdict=verdict,
                                 r_table={str(k): rs[k] for k in KS},
                                 bars=dict(bar1_share_law=bar1,
                                           bar2_flat=bar2,
                                           bar3_literal=bar3_literal,
                                           bar4_product_constant=prod_const)),
    )
    save_json(out_dir / "metrics.json", metrics)
    log("metrics.json written")

    plot(out_dir / "field_floor.png", metrics)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot


def plot(path: Path, M: dict):
    dec = M["registered_decision"]
    S = M["summary"]
    per = S["per_cell"]
    rstar = {int(k): v for k, v in S["rstar"].items()}
    rstar_ci = {int(k): v for k, v in S["rstar_ci"].items()}
    col = {int(k.split("=")[1]): {float(r): v for r, v in rs.items()}
           for k, rs in S["col_means"].items()}
    ord_ = S["ordering"]
    kcol = {32: "tab:red", 64: "tab:orange", 128: "tab:green",
            154: "tab:blue"}

    fig, axes = plt.subplots(2, 3, figsize=(24, 11))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # ---- panel 1: gap vs r curves per k
    for k in KS:
        ys = [col[k][r] for r in RS]
        lo = [per[f"k={k},r={r}"]["ci"][0] for r in RS]
        hi = [per[f"k={k},r={r}"]["ci"][1] for r in RS]
        ax1.plot(RS, ys, "o-", lw=2.2, ms=8, color=kcol[k],
                 label=f"k={k}")
        ax1.fill_between(RS, lo, hi, color=kcol[k], alpha=0.15)
    ax1.axhspan(-0.2, HEALTHY_BAR, color="tab:green", alpha=0.06, zorder=0)
    ax1.axhspan(HEALTHY_BAR, DAMAGED_BAR, color="tab:orange", alpha=0.08,
                zorder=0)
    ax1.axhspan(DAMAGED_BAR, max(2.2, max(per[p]["mean"] for p in per)
                                 + 0.2), color="tab:red", alpha=0.06,
                zorder=0)
    ax1.axhline(HEALTHY_BAR, color="tab:green", ls="--", lw=1.4)
    ax1.axhline(DAMAGED_BAR, color="tab:red", ls="--", lw=1.4)
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xscale("log")
    ax1.set_xticks(RS)
    ax1.set_xticklabels([str(r) for r in RS])
    ax1.set_xlabel("retention r (V <- r*V on the kept field; log scale)")
    ax1.set_ylabel("clean-judge tail cost vs matched control (nats)")
    ax1.set_title("E110-1 — the crossed grid: cost vs retention, one curve "
                  "per field size (bands = paired-row 95% CI)", fontsize=10)
    ax1.legend(fontsize=9, title="field size k", title_fontsize=9)

    # ---- panel 2: the r*(k) threshold line
    vals = []
    for k in KS:
        v = rstar[k]["value"]
        vals.append(v if v is not None else 1.35)
    ax2.plot(KS, vals, "o-", lw=2.4, ms=11, color="k",
             label="r*(k) (minimum healthy retention)")
    for k in KS:
        ci = rstar_ci[k]["cont"]
        if np.isfinite(ci[0]) and rstar[k]["value"] is not None:
            ax2.errorbar([k], [rstar[k]["value"]],
                         yerr=[[max(0, rstar[k]["value"] - ci[0])],
                               [max(0, ci[1] - rstar[k]["value"])]],
                         fmt="none", ecolor="gray", elinewidth=1.6,
                         capsize=5)
        ax2.annotate(f"{rstar[k]['value'] if rstar[k]['value'] is not None else rstar[k]['status']}",
                     (k, vals[KS.index(k)]),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=9)
    flat_ref = rstar[154]["value"] if rstar[154]["value"] is not None else None
    if flat_ref is not None:
        ax2.axhline(flat_ref, color="tab:red", ls=":", lw=1.8,
                    label=f"flat null ref = r*(154) = {flat_ref}")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(KS)
    ax2.set_xticklabels([str(k) for k in KS])
    ax2.set_xlabel("field size k (kept entries; log2 scale)")
    ax2.set_ylabel("r*(k)")
    ax2.set_ylim(0, 1.5)
    ax2.set_title("E110-2 — the threshold line: r*(k) vs k (W001: moves "
                  f"DOWN with k; per-entry floor: flat) | "
                  f"P(154<64)={ord_['P_r154_lt_r64']:.2f} "
                  f"P(32>64)={ord_['P_r32_gt_r64']:.2f}", fontsize=10)
    ax2.legend(fontsize=9, loc="upper right")

    # ---- panel 3: the r* x k product panel (the share law)
    for k in KS:
        p = rstar[k]["product"]
        if p is None:
            ax3.scatter([k], [np.nan], marker="x", s=90, color="gray")
            ax3.annotate(rstar[k]["cont_status"], (k, 0.02),
                         fontsize=7.5, color="gray", ha="center")
            continue
        ci = rstar_ci[k]["product"]
        ax3.errorbar([k], [p], yerr=[[max(0, p - ci[0])],
                                     [max(0, ci[1] - p)]],
                     fmt="o", ms=10, lw=2, color=kcol[k], capsize=5,
                     label=f"k={k}: {p:.1f}")
    prod_vals = [rstar[k]["product"] for k in KS
                 if rstar[k]["product"] is not None]
    if len(prod_vals) >= 2:
        ax3.axhline(float(np.mean(prod_vals)), color="k", ls="--", lw=1.6,
                    label=f"share-law const = mean {np.mean(prod_vals):.1f}")
    if rstar[154]["value"] is not None:
        null_ray = [rstar[154]["value"] * k for k in KS]
        ax3.plot(KS, null_ray, ":", color="tab:red", lw=1.8,
                 label="per-entry-floor null: r*(154) x k (rising)")
    ax3.set_xscale("log", base=2)
    ax3.set_xticks(KS)
    ax3.set_xticklabels([str(k) for k in KS])
    ax3.set_xlabel("field size k (log2 scale)")
    ax3.set_ylabel("r*_cont(k) x k  (retained entry-mass at threshold)")
    ax3.set_title("E110-3 — the share law: retained mass at the boundary "
                  f"~ CONSTANT? ({'CONSTANT' if S['product']['constant'] else 'not constant'})",
                  fontsize=10)
    ax3.legend(fontsize=8.5)

    # ---- panel 4: the 4x4 grid heatmap of means (health-colored)
    grid = np.array([[col[k][r] for r in RS] for k in KS])
    im = ax4.imshow(grid, cmap="RdYlGn_r", vmin=-0.2, vmax=max(1.6,
                   grid.max()), aspect="auto")
    for i, k in enumerate(KS):
        for j, r in enumerate(RS):
            ax4.text(j, i, f"{grid[i, j]:+.3f}\n"
                     f"[{per[f'k={k},r={r}']['health']}]",
                     ha="center", va="center", fontsize=9,
                     color="k" if abs(grid[i, j]) < 1.0 else "w")
    ax4.set_xticks(range(len(RS)))
    ax4.set_xticklabels([f"r={r}" for r in RS])
    ax4.set_yticks(range(len(KS)))
    ax4.set_yticklabels([f"k={k}" for k in KS])
    ax4.set_title("E110-4 — the crossed grid: mean clean-judge cost per "
                  "cell (green=healthy, red=damaged)", fontsize=10)
    fig.colorbar(im, ax=ax4, shrink=0.8, label="nats")

    # ---- panel 5: the removal column (r=1.0) vs e089's reference curve
    xs = [154 - k for k in KS]
    ys = [col[k][1.0] for k in KS]
    cis = [per[f"k={k},r=1.0"]["ci"] for k in KS]
    ax5.errorbar(xs, ys, yerr=[[max(0, y - c[0]) for y, c in zip(ys, cis)],
                               [max(0, c[1] - y) for y, c in zip(ys, cis)]],
                 fmt="o-", lw=2.2, ms=9, capsize=5, color="tab:purple",
                 label="e110 r=1.0 column (single-shot, 154-band)")
    if M["references"]["e089_per_k"]:
        rk = sorted(int(a) for a in M["references"]["e089_per_k"])
        rv = [M["references"]["e089_per_k"][str(a)] for a in rk]
        ax5.plot(rk, rv, "s--", color="gray", lw=1.6, ms=6,
                 label="e089 reference (dynamic timing, 351-band)")
    ax5.axhline(HEALTHY_BAR, color="tab:green", ls="--", lw=1.2)
    ax5.axhline(DAMAGED_BAR, color="tab:red", ls="--", lw=1.2)
    ax5.set_xlabel("entries removed (band complement)")
    ax5.set_ylabel("clean-judge tail cost (nats)")
    ax5.set_title("E110-5 — the removal rider: r=1.0 column (pure removal) "
                  "with e089's stored curve", fontsize=10)
    ax5.legend(fontsize=8.5)

    # ---- panel 6: verdict text
    ax6.axis("off")
    lines = [
        "REGISTERED (frozen pre-compute; W001 refinement quantified):",
        f"  healthy < {HEALTHY_BAR} | damaged > {DAMAGED_BAR} | r*(k) = "
        f"minimum healthy retention on r-grid {RS}",
        "  BAR-1 share law: r*(154) < r*(64) AND r*(32) >= r*(64), "
        f"bootstrap P >= {BOOT_PROB}",
        "  BAR-2 null: r* flat in k | BAR-3 literal tasking text "
        "(r*(64)<r*(154)): reported, direction flagged",
        f"  BAR-4 secondary: r*_cont(k)*k ~ const (max/min <= "
        f"{PRODUCT_MAXMIN}, CI overlap)",
        "",
        "THE r* TABLE:",
    ] + [
        f"  k={k:>3}: r* = "
        + (f"{rstar[k]['value']}" if rstar[k]["value"] is not None
           else rstar[k]["status"])
        + (f"  [cont {rstar[k]['cont_value']:.3f}, r*k "
           f"{rstar[k]['product']:.1f}]"
           if rstar[k]["product"] else f"  [{rstar[k]['cont_status']}]")
        for k in KS
    ] + [
        "",
        f"ORDERING: P(r*154 < r*64) = {ord_['P_r154_lt_r64']:.3f} | "
        f"P(r*32 > r*64) = {ord_['P_r32_gt_r64']:.3f} | "
        f"both-off-high tie {ord_['both_offhigh_32_64']}",
        f"          flat {ord_['flat_point']} (P={ord_['P_flat']:.3f}) | "
        f"literal tasking pair {ord_['literal_tasking_pair']}",
        "PRODUCTS: " + S["product"]["report"],
        "",
        f"DECISION [{dec['clause']}]:",
    ] + [f"  {wd}" for wd in _wrap(dec["verdict"], 98)]
    ax6.text(0.02, 0.97, "E110 — the W001 paper-prediction: the magnitude "
             "floor is a per-FIELD stream share", fontsize=13, weight="bold",
             va="top")
    for i, tx in enumerate(lines):
        ax6.text(0.02, 0.935 - i * 0.0285, tx, fontsize=8.4, va="top",
                 family="monospace")

    fig.suptitle("E110 — retention x field-size crossed grid at the anchor "
                 f"band (g=250, 154 V-entries, single-shot, clean-judged) | "
                 f"{dec['clause']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
