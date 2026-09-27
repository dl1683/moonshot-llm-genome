"""E102 — DIRECTION-vs-MAGNITUDE DECOMPOSITION of the anchor band (T051's
E096 amendment, the registered WHY-is-corruption-free discriminator). CPU-only.

CONTEXT. e096 closed its ladder with the NO-DAMAGE clause: single-shot
additive V corruption of the WHOLE anchor band (154 entries at g=250) at
doses up to one median entry norm costs nothing at any rung (eps=0.05
bitwise-absorbed; mid doses slightly facilitative; no off-manifold gap),
while REMOVING the same mass costs +2.15 nats (e089 k=200 continuous) and
e080's dynamic norm-matched noise REPLACEMENT collapsed the run by clean
judge (+5.48 cj; its own online CE only +0.29 — the off-manifold signature).
T051's amendment registers three candidate explanations: (i) STATISTICAL
mass (nonzero-ness, values irrelevant — the eviction field's implicit
model), (ii) REDUNDANT information (the values carry content but are
redundantly encoded), (iii) ABSORPTION (LN squelch; supported by the
dose-leak). The registered discriminator is the DIRECTION/MAGNITUDE
decomposition run at the SAME intervention point as e096.

DESIGN (registered here BEFORE compute; bars frozen from the tasking).
Rig: lab/e096_coherence_gap.py VERBATIM — same battery (the e053c ctx-512
net, 873,472 params, val CE 1.5227; seed-202 8-draw prompts; seed-7 A-none
control free run; G3 gated against e080's AND e089's stored controls), same
intervention point (g_int=250, prefix 0..313, teacher-forced token at 314,
band = self-generated positions 64..217 = ages 97..250 at query 314, all
154 extant entries SIMULTANEOUSLY, V-only/K-untouched, immediately before
the decode at 314), same readout (clean-judge tail CE 448..511 of the arm
stream minus the MATCHED control stream, continuation seed 960250 shared
across arms, per-row then mean over B=8; paired-row bootstrap CIs).

THE FIVE ARMS (frozen from the tasking; all replace/modify the SAME 154
band V vectors once):
  (a) direction-only    V <- V / ||V||          unit-norm original
      directions: direction preserved EXACTLY, magnitude cut to 1.0
      (realized retention = 1/median-norm ~ 0.46..0.64 per layer — the
      tasking's '~0.1x cut' is the intent language; the REALIZED ratios
      are measured and reported per layer in metrics + plot);
  (b) magnitude-only    V <- u * ||V||          fresh random unit gaussian
      direction u at the ORIGINAL per-vector norm: magnitude preserved
      EXACTLY, direction destroyed (E|cos| ~ 0.14 in d=32). This is e080's
      noise arm recomputed at matched scale and single-shot timing — the
      ANCHOR arm of the decomposition;
  (c) scaled-down       V <- 0.1 * V            direction preserved, norm
      cut to 10% (both weakened — the dose gradient inside direction-
      preserved space);
  (d) additive eps=0.5  V <- V + 0.5 * m_layer * u  — the e096 mid-rung
      control (both roughly preserved: cos ~ 0.89, norm ratio ~ 1.12);
      implemented as a BITWISE REPLICATION of e096's eps=0.5 arm (4th
      noise batch from e096's seed-9696 generator, same med_norm) — a
      G5 gate against the shipped artifact;
  (e) none              no modification (eps=0 code path; MUST be bitwise
      identical to the matched control — the instrument gate).

REGISTERED BARS (frozen from the tasking; point rules on the per-arm MEANS
of the clean-judge costs, bootstrap CIs reported alongside):
  healthy   = mean cost < 0.3 nats
  damaged   = mean cost > 1.0 nat
  ambiguous = in [0.3, 1.0] (no registered cell — honest texture)
  (a) healthy AND (b) damaged -> DIRECTION CARRIES THE ANCHOR: the run
      reads WHERE entries point, not how loud — corruption-robustness
      (e096) is norm-REDUNDANCY;
  (a) damaged AND (b) healthy -> MAGNITUDE/STATISTICAL MASS carries it:
      pure nonzero mass (the eviction field's implicit model);
  both damaged             -> BOTH NEEDED (redundancy is joint);
  both healthy             -> entries even MORE INTERCHANGEABLE than
      hypothesized (register the shock honestly);
  anything else (>=1 arm ambiguous) -> TEXTURE, no registered cell.
Secondary readouts (registered, descriptive): the clean-judge attractor
signature (judge CE minus the run's own online tail CE; e080 off-manifold
bar > 1.0 nat; vzero calibration +5.02), tail entropy (fluency), stream
divergence, and per-arm REALIZED vector-space identity (mean |cos| to the
originals, mean norm ratio) so the arms are placed on the
direction-preservation x magnitude-preservation design grid.

Gates: G1 val CE vs e053c 1.5227 +-0.02; G2 params 873472; G3 control
battery identity vs e080's stored none arm AND e089's stored control run
(e096 convention); G4 instrument identity (none arm bitwise == matched
control; every arm prefix-identical through 314; per-arm band transform
exact — direction/scaled: all ||V_new|| at target within 1e-5 AND
cos(V_new,V_old)=1; magnitude: ||V_new||==||V_old|| within 1e-5; additive:
band delta norm == 0.5*m_layer within 1e-5; all arms: non-band cache
columns bitwise untouched; matched continuation seed across arms;
single-shot by construction; med_norm bitwise == e096's stored
layer medians); G5 e096 replication (the additive arm's clean-judge
per-seq AND cost-per-row vs runs/e096/metrics.json eps=0.5 within 1e-4,
bitwise flags reported).

Run:     python lab/e102_direction_magnitude.py
Outputs: runs/e102/metrics.json + runs/e102/direction_magnitude.png
Envelope: NO training, NO new automations; CPU-only (CUDA_VISIBLE_DEVICES=-1),
8 threads (T050), single step, minutes-scale.
No NOTES/THINKING/QUEUE/STATE edits; no commit (the dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu" (e053b..e096)
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
SEED_PROMPT, SEED_SAMPLE = 202, 7         # e053b/e053c/e069..e096 seeds
SEED_CONT = 960250                        # e096's single matched
                                          # continuation seed (g_int=250)
SEED_NOISE_E096 = 9696                    # e096's noise generator (replication)
SEED_NOISE = 96102                        # e102: dedicated generator for the
                                          # magnitude-only arm's directions
B = 8
N_PROMPTS = 8
BOOT_N = 1000
THREADS = 8                               # T050: 12 spin-thrashes the box
torch.set_num_threads(THREADS)

P512 = T_TOTAL - 1                        # 511
TAIL = 64                                 # the e075..e096 tail window
KEY_T = (T_TOTAL - TAIL, T_TOTAL - 1)     # judged tail window (448, 511)
G = T_TOTAL - PROMPT_TOK                  # 448 generation steps

# ---- the intervention (single-shot, one frame; VERBATIM e096) -----------------
G_INT = 250                               # generation step of the shot
T_INT = PROMPT_TOK + G_INT                # 314: teacher-forced token position
GEN_FIRST = 64                            # first generated position
BAND = list(range(GEN_FIRST, T_INT - 96)) # 64..217: ages 97..250 at query 314
                                          # (e075's age>96 rule at the frame;
                                          # 154 entries — the extant prefix of
                                          # the e089 final-frame band 64..414)
FIRST_AFFECTED = T_INT + 1                # 315 (first free-sampled position)
TAIL_STEP0 = KEY_T[0] - FIRST_AFFECTED    # 133: tail slice start in free steps

# ---- the arms (tasking, frozen) ----------------------------------------------
ARMS = ["none", "direction", "magnitude", "scaled", "additive05"]
ADDITIVE_EPS = 0.5                        # the e096 mid-rung control
SCALED_FACTOR = 0.1                       # arm (c): 0.1x norm

# ---- REGISTERED decision numbers (frozen, docstring verbatim) -----------------
HEALTHY_BAR = 0.3                         # mean cost < this -> healthy
DAMAGED_BAR = 1.0                         # mean cost > this -> damaged
CJ_GAP_BAR = 1.0                          # e080's off-manifold bar (nats)

# ---- reference numbers (protocol-identity gates + curve anchors) --------------
CKPT = REPO / "runs" / "checkpoints" / "e053c_ctx512.pt"
E053C_VAL_CE = 1.5226792494455974         # the net being interrogated
E080_METRICS = REPO / "runs" / "e080" / "metrics.json"
E089_METRICS = REPO / "runs" / "e089" / "metrics.json"
E096_METRICS = REPO / "runs" / "e096" / "metrics.json"
E075_METRICS = REPO / "runs" / "e075" / "metrics.json"
E075_REF_DELTA = 0.26191402220749405      # whole-band V-zero, K=32 (online r1)
E089_REF_K200 = 2.1530486822128294        # k=200 continuous (K=1) clean-judge
E096_MED_NORM_REF = [2.157031774520874, 1.5538004636764526,
                     1.9136782884597778, 1.8171889781951904]
E080_NOISE_CJ_DELTA = 5.4778508096933365  # e080 dynamic norm-matched noise
E080_NOISE_R1_DELTA = 0.28706934932985045 # ...its own-online face
E080_VZERO_CJ = 6.402048110361914         # off-manifold calibration
E080_VZERO_CJ_GAP = 5.024754047397799
T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ------------------------------------------------- machinery (VERBATIM e096)

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
    [VERBATIM e075/e080/e085/e088/e089/e096]"""
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
    [VERBATIM e075/e080/e085/e088/e089/e096]"""
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
    """Free-run 64->512 for B sequences (e075/e080/e085/e088/e089/e096
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
    [VERBATIM e096]"""
    sel = torch.tensor(BAND, dtype=torch.long)
    meds = []
    for (_k, v) in kv:
        nrm = v[:, :, sel, :].norm(dim=-1)          # (B, H, n_band)
        meds.append(float(nrm.median()))
    return meds


def draw_e096_additive_directions(kv: list) -> list:
    """BITWISE replication of e096's eps=0.5 noise batch: e096 consumed its
    seed-9696 generator one arm-batch per ladder rung in order
    (0.05, 0.1, 0.25, 0.5, 1.0) — the eps=0.5 directions are the FOURTH
    batch. Re-create the generator, discard three arm-batches (12 layer
    draws, layer order 0..3, shapes (B,H,n_band,d)), keep the fourth."""
    shapes = [v[:, :, torch.tensor(BAND, dtype=torch.long), :].shape
              for (_k, v) in kv]
    g = torch.Generator().manual_seed(SEED_NOISE_E096)
    for _ in range(3):                                  # rungs 0.05/0.1/0.25
        for s in shapes:
            torch.randn(s, generator=g)                 # consumed + discarded
    u = []
    for s in shapes:                                    # rung 0.5 (kept)
        gg = torch.randn(s, generator=g)
        u.append(gg / gg.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    return u


@torch.no_grad()
def run_continuation(net: TinyGPT, prefix: torch.Tensor, forced_tok: torch.Tensor,
                     forced_pos: int, seed: int, arm: str = "none",
                     noise_gen: torch.Generator | None = None,
                     med_norm: list | None = None,
                     u_additive: list | None = None):
    """Single-shot band-transform continuation at the e096 intervention point
    ("none": the matched control / bitwise identity arm).

    1. prefill(prefix) — prefix = control positions 0..forced_pos-1.
    2. if arm != "none": transform ALL band entries at once — every
       (row, layer, head, band position) d=32 V vector, per the arm's rule
       (docstring); K is never touched; applied BEFORE the decode at
       forced_pos (top-of-the-event timing), so the first affected SAMPLE
       is forced_pos+1. The magnitude arm draws its directions from the
       dedicated e102 generator; the additive arm uses the pre-drawn e096
       eps=0.5 replication batch; the sampling stream never sees either.
    3. teacher-force the token at forced_pos (clean control token), then
       free-run to 511 with the shared row-order generator (e053b stream
       math) — all arms + control share the seed, so streams are
       row-by-row matched until sampled divergence.

    Tracks online CE + full-softmax entropy of every emitted free token (no
    rng cost). Returns idx, kv, fired-info dict (per-arm transform identity
    + realized vector-space stats + bookkeeping), ce_s, ent_s.
    """
    R = prefix.shape[0]
    gen = torch.Generator().manual_seed(seed)
    logits, kv = prefill_batch(net, prefix)
    fired = dict(applied=False, arm=arm, n_vec=0, norm_dev=0.0,
                 min_cos=1.0, mean_abs_cos=None, mean_cos=None,
                 mean_norm_ratio=None, nonband_untouched=True)
    if arm != "none":
        if arm == "additive05":
            assert u_additive is not None and med_norm is not None
        if arm == "magnitude":
            assert noise_gen is not None
        sel = torch.tensor(BAND, dtype=torch.long)
        pre_v = [v.clone() for (_k, v) in kv]       # identity snapshot
        coss, ratios = [], []
        for li, (_k, v) in enumerate(kv):
            old = v[:, :, sel, :].clone()                     # (B,H,n,d)
            nrm = old.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            if arm == "direction":
                new = old / nrm                              # unit-norm originals
                target = torch.ones_like(nrm)
            elif arm == "magnitude":
                g = torch.randn(old.shape, generator=noise_gen)
                u = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                new = u * nrm                                # norm exact, dir random
                target = nrm
            elif arm == "scaled":
                new = SCALED_FACTOR * old
                target = SCALED_FACTOR * nrm
            elif arm == "additive05":
                new = old + ADDITIVE_EPS * med_norm[li] * u_additive[li]
                target = None
            else:
                raise ValueError(arm)
            v[:, :, sel, :] = new
            # ---- transform identity: realized norms at target exactly (or
            #      the additive delta-norm rule); non-band bitwise untouched
            nnew = new.norm(dim=-1, keepdim=True)
            if target is not None:
                fired["norm_dev"] = max(fired["norm_dev"], float(
                    (nnew - target).abs().max()))
            else:
                dn = (new - old).norm(dim=-1)
                fired["norm_dev"] = max(fired["norm_dev"], float(
                    (dn - ADDITIVE_EPS * med_norm[li]).abs().max()))
            cos = (old * new).sum(-1) / (nrm.squeeze(-1)
                                         * nnew.squeeze(-1)).clamp_min(1e-12)
            fired["min_cos"] = min(fired["min_cos"], float(cos.min()))
            coss.append(cos.reshape(-1))
            ratios.append((nnew / nrm).reshape(-1))
            fired["n_vec"] += int(cos.numel())
        fired["mean_cos"] = float(torch.cat(coss).mean())
        fired["mean_abs_cos"] = float(torch.cat(coss).abs().mean())
        fired["mean_norm_ratio"] = float(torch.cat(ratios).mean())
        nb_lo = torch.arange(0, GEN_FIRST)                    # prompt cols
        nb_hi = torch.arange(BAND[-1] + 1, forced_pos)        # young cols
        for v, pv in zip([v for (_k, v) in kv], pre_v):
            fired["nonband_untouched"] &= bool(
                torch.equal(v[:, :, nb_lo, :], pv[:, :, nb_lo, :]))
            fired["nonband_untouched"] &= bool(
                torch.equal(v[:, :, nb_hi, :], pv[:, :, nb_hi, :]))
        fired["applied"] = True
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
    Returns dict window -> (R,) mean CE per row. [VERBATIM e085..e096]"""
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
    replacement, recompute fn on the resampled arrays. [VERBATIM e096]"""
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
    """The registered health code of an arm's mean clean-judge cost."""
    if cost < HEALTHY_BAR:
        return "healthy"
    if cost > DAMAGED_BAR:
        return "damaged"
    return "ambiguous"


# ---------------------------------------------------------------------- main


def main():
    assert torch.cuda.device_count() == 0, "CPU-only violated: CUDA visible"
    assert common.DEVICE == "cpu", f"common.DEVICE={common.DEVICE} != cpu"
    out_dir = run_dir("e102")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=False)

    # ---- battery: rebuild corpus + net EXACTLY as e053c/e069..e096 did
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
    log("control battery: seed-7 free run (e075/e080/e085/e088/e089/e096 A-none)")
    gen7 = torch.Generator().manual_seed(SEED_SAMPLE)
    run1 = generate_control(net, prompts8, gen7)
    idx1 = run1["idx"]
    log(f"  control run done ({T_TOTAL} positions x {B} rows)")

    # ---- G3 (scope: control battery) vs e080's stored A-none + e089's control
    cj1 = judge_windows(manual_all_logits(net, idx1), idx1, [KEY_T])[KEY_T]
    g3 = dict(ref_files=[str(E080_METRICS), str(E089_METRICS)],
              note="e102 has no static arm; G3 scope = control-battery "
                   "identity vs e080's stored none arm (clean-judge per-seq, "
                   "1e-4) AND bitwise vs e089's stored control run "
                   "[e088/e089/e096's scoped G3 convention]")
    ok3, dev3 = True, float("nan")
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
    if E089_METRICS.exists():
        e089 = _json.load(open(E089_METRICS))
        ref89 = np.asarray(e089["control_run"]["clean_judge_tail_ce"])
        dev89 = float(np.abs(cj1 - ref89).max())
        bit89 = bool(np.array_equal(cj1, ref89))
        g3.update(e089_clean_judge_max_dev=dev89, e089_bitwise=bit89)
        ok3 &= bool(dev89 < 1e-4)
        log(f"G3 vs e089 control: clean-judge dev {dev89:.2e} (bitwise {bit89})")
    else:
        ok3 = False
        g3["note"] += " | runs/e089/metrics.json missing (bitwise leg skipped)"
    g3["ok"] = bool(ok3)
    gates["G3_protocol_identity"] = g3
    log(f"G3 -> {'PASS' if g3['ok'] else 'FAIL'}")

    # ---- reference anchors from disk (drift-checked constants)
    e075_ref, e089_ref, e096_eps05 = E075_REF_DELTA, E089_REF_K200, None
    if E075_METRICS.exists():
        e75 = _json.load(open(E075_METRICS))
        r1 = e75["registered_decision"]["r1_tail_clean_ce"]
        e075_ref = float(r1["self"] - r1["none"])
    if E089_METRICS.exists():
        e89 = _json.load(open(E089_METRICS))
        e089_ref = float(e89["summary"]["per_k"]["200"]["mean"])
    if E096_METRICS.exists():
        e96 = _json.load(open(E096_METRICS))
        e096_eps05 = dict(
            mean=float(e96["summary"]["per_eps"]["0.5"]["mean"]),
            ci=e96["summary"]["per_eps"]["0.5"]["ci"],
            cost_per_row=e96["summary"]["per_eps"]["0.5"]["cost_per_row"],
            clean_judge_per_row=e96["summary"]["per_eps"]["0.5"]
                                      ["clean_judge"]["per_row"],
            n_rows_diverged=e96["summary"]["per_eps"]["0.5"]
                                     ["n_rows_diverged"],
            control_cj=e96["control_run"]["clean_judge_tail_ce"],
            med_norm=e96["gates"]["G4_instrument_identity"]["noise"]
                              ["median_norm_per_layer"])
    log(f"reference anchors: e075 whole-band V-zero {e075_ref:+.4f} (K=32) | "
        f"e089 k=200 continuous {e089_ref:+.4f} (K=1) | e080 dynamic noise "
        f"cj {E080_NOISE_CJ_DELTA:+.4f} (online r1 {E080_NOISE_R1_DELTA:+.4f})"
        + (f" | e096 eps=0.5 {e096_eps05['mean']:+.4f}" if e096_eps05 else ""))

    # ================================================== the matched continuations
    prefix = idx1[:, :T_INT]
    forced = idx1[:, T_INT]
    log(f"continuations: prefix 0..{T_INT - 1}, forced token at {T_INT}, "
        f"band = positions {BAND[0]}..{BAND[-1]} ({len(BAND)} entries, ages "
        f"97..{T_INT - BAND[0]} at query {T_INT}), seed {SEED_CONT}")

    # the matched control ("none" code path, no draws) fixes the dose unit
    idx_ctl, kv_ctl, fired_ctl, ce_ctl, ent_ctl = run_continuation(
        net, prefix, forced, T_INT, SEED_CONT, arm="none")
    med_norm = band_median_norms(kv_ctl)
    med_dev = (max(abs(a - b) for a, b in zip(med_norm, E096_MED_NORM_REF))
               if len(med_norm) == len(E096_MED_NORM_REF) else float("inf"))
    log(f"matched control done | median band ||V|| per layer: "
        f"{[round(m, 4) for m in med_norm]} (vs e096 stored max dev "
        f"{med_dev:.2e})")

    # the e096 eps=0.5 replication directions (4th batch of seed 9696)
    u_additive = draw_e096_additive_directions(kv_ctl)
    # e102's dedicated generator for the magnitude arm's random directions
    noise_gen = torch.Generator().manual_seed(SEED_NOISE)

    S = {}
    for arm in ARMS:
        idx_a, kv_a, fired_a, ce_a, ent_a = run_continuation(
            net, prefix, forced, T_INT, SEED_CONT, arm=arm,
            noise_gen=noise_gen, med_norm=med_norm, u_additive=u_additive)
        J = judge_windows(manual_all_logits(net, idx_a), idx_a, [KEY_T])[KEY_T]
        eq = torch.eq(idx_a, idx_ctl)
        fdr = []
        for r in range(idx_a.shape[0]):
            nz = (~eq[r]).nonzero().flatten()
            fdr.append(int(nz[0].item()) if len(nz) else None)
        S[arm] = dict(
            idx=idx_a, ce=ce_a, ent=ent_a, J=J, fired=fired_a,
            stream_identical=bool(eq.all().item()),
            first_div_per_row=fdr,
            first_div=(min(d for d in fdr if d is not None)
                       if any(d is not None for d in fdr) else None),
            n_rows_diverged=sum(d is not None for d in fdr),
            online_tail=ce_a[:, TAIL_STEP0:].mean(1),
            ent_tail=ent_a[:, TAIL_STEP0:].mean(1),
        )
        if fired_a["applied"]:
            tag = (f"|cos(new,old)| {fired_a['mean_abs_cos']:.3f} | norm "
                   f"ratio {fired_a['mean_norm_ratio']:.3f} | norm dev "
                   f"{fired_a['norm_dev']:.1e}")
        else:
            tag = "no-op"
        log(f"  arm {arm:<11}: cj tail {float(S[arm]['J'].mean()):.4f} | "
            f"online tail {float(S[arm]['online_tail'].mean()):.4f} | "
            f"ent tail {float(S[arm]['ent_tail'].mean()):.4f} | first_div "
            f"{S[arm]['first_div']} | {tag}")

    # ---- G4 instrument identity
    J_ctl = judge_windows(manual_all_logits(net, idx_ctl), idx_ctl,
                          [KEY_T])[KEY_T]
    bit_none = bool(torch.equal(S["none"]["idx"], idx_ctl))
    pre_ident = all(bool(torch.equal(S[arm]["idx"][:, :T_INT + 1],
                                     idx_ctl[:, :T_INT + 1])) for arm in ARMS)
    tf_arm = [a for a in ARMS if a != "none"]
    fired_ok = all(S[a]["fired"]["norm_dev"] < 1e-5
                   and S[a]["fired"]["nonband_untouched"] for a in tf_arm)
    dir_like_ok = all(S[a]["fired"]["min_cos"] > 1.0 - 1e-5
                      for a in ("direction", "scaled"))
    g4 = dict(
        none_bitwise_control=bit_none,
        all_arms_prefix_identical_through=T_INT + 1 if pre_ident else None,
        prefix_identical=pre_ident,
        per_arm={a: dict(
                norm_dev=S[a]["fired"]["norm_dev"],
                min_cos=S[a]["fired"]["min_cos"],
                mean_abs_cos=S[a]["fired"]["mean_abs_cos"],
                mean_cos=S[a]["fired"]["mean_cos"],
                mean_norm_ratio=S[a]["fired"]["mean_norm_ratio"],
                nonband_untouched=S[a]["fired"]["nonband_untouched"])
            for a in tf_arm},
        direction_like_cos_exact=dir_like_ok,
        none_applied_nothing=(not S["none"]["fired"]["applied"]),
        matched_seed=SEED_CONT, single_shot=True,
        band=dict(positions=[BAND[0], BAND[-1]], n=len(BAND),
                  ages_at_query=[T_INT - BAND[-1], T_INT - BAND[0]]),
        med_norm=dict(per_layer=med_norm, e096_ref=E096_MED_NORM_REF,
                      max_dev=med_dev,
                      ok=bool(med_dev < 1e-6)),
        noise=dict(magnitude_seed=SEED_NOISE,
                   magnitude_rule="fresh unit-L2 gaussian direction per "
                                  "(row,layer,head,band-position) d=32 V "
                                  "vector, scaled to the ORIGINAL per-vector "
                                  "norm (one batch, drawn once)",
                   additive_rule=f"BITWISE replication of e096's eps="
                                 f"{ADDITIVE_EPS} directions: 4th batch of "
                                 f"e096's seed-{SEED_NOISE_E096} generator "
                                 f"(3 arm-batches discarded)"))
    g4["ok"] = bool(bit_none and pre_ident and fired_ok and dir_like_ok
                    and (not S["none"]["fired"]["applied"])
                    and med_dev < 1e-6)
    gates["G4_instrument_identity"] = g4
    log(f"G4 instrument identity: none bitwise {bit_none} | prefix identity "
        f"{pre_ident} | band transforms exact {fired_ok} | direction-like "
        f"cos=1 {dir_like_ok} | med_norm dev {med_dev:.1e} -> "
        f"{'PASS' if g4['ok'] else 'FAIL'}")

    # ---- G5 e096 replication (the additive arm vs the shipped artifact)
    g5 = dict(ref=str(E096_METRICS),
              rule="additive05 arm (e096's eps=0.5 noise batch, bitwise) "
                   "clean-judge per-seq AND cost-per-row vs runs/e096/"
                   "metrics.json within 1e-4")
    ok5, dev_cj, dev_cost, dev_ctlcj = True, float("nan"), float("nan"), float("nan")
    if e096_eps05 is not None:
        dev_cj = float(np.abs(S["additive05"]["J"]
                              - np.asarray(e096_eps05["clean_judge_per_row"])).max())
        cost_add = S["additive05"]["J"] - J_ctl
        dev_cost = float(np.abs(cost_add
                                - np.asarray(e096_eps05["cost_per_row"])).max())
        dev_ctlcj = float(np.abs(J_ctl - np.asarray(e096_eps05["control_cj"])).max())
        bit_cj = bool(np.array_equal(S["additive05"]["J"],
                                     np.asarray(e096_eps05["clean_judge_per_row"])))
        n_div_match = bool(S["additive05"]["n_rows_diverged"]
                           == e096_eps05["n_rows_diverged"])
        g5.update(additive_cj_max_dev=dev_cj, additive_cj_bitwise=bit_cj,
                  additive_cost_max_dev=dev_cost,
                  control_cj_max_dev=dev_ctlcj,
                  divergence_count_match=n_div_match,
                  e096_eps05_mean=e096_eps05["mean"])
        ok5 = bool(dev_cj < 1e-4 and dev_cost < 1e-4 and dev_ctlcj < 1e-4
                   and n_div_match)
        log(f"G5 e096 replication: additive cj dev {dev_cj:.2e} (bitwise "
            f"{bit_cj}) | cost dev {dev_cost:.2e} | control cj dev "
            f"{dev_ctlcj:.2e} | div count match {n_div_match}")
    else:
        ok5 = False
        g5["note"] = "runs/e096/metrics.json missing"
    g5["ok"] = bool(ok5)
    gates["G5_e096_replication"] = g5
    log(f"G5 -> {'PASS' if g5['ok'] else 'FAIL'}")

    # ================================================== the arm table + bars
    per = {}
    for arm in ARMS:
        c = S[arm]["J"] - J_ctl                 # none: bitwise-zero by G4
        ci = boot_rows([c], lambda a: float(a.mean()))
        per[arm] = dict(
            cost_per_row=c.tolist(), mean=float(c.mean()), ci=ci,
            n_worse=int((c > 0).sum()),
            n_stream_identical=B - S[arm]["n_rows_diverged"],
            first_div=S[arm]["first_div"],
            first_div_per_row=S[arm]["first_div_per_row"],
            n_rows_diverged=S[arm]["n_rows_diverged"],
            health=health_of(float(c.mean())))
        per[arm]["clean_judge"] = dict(
            mean=float(S[arm]["J"].mean()),
            per_row=S[arm]["J"].tolist(),
            online_tail=float(S[arm]["online_tail"].mean()),
            gap_vs_online=float((S[arm]["J"] - S[arm]["online_tail"]).mean()),
            gap_per_row=(S[arm]["J"] - S[arm]["online_tail"]).tolist(),
            ent_tail=float(S[arm]["ent_tail"].mean()),
            ent_drift_pct=float((S[arm]["ent_tail"].mean()
                                 / float(ent_ctl[:, TAIL_STEP0:].mean()) - 1.0)
                                * 100.0))
        per[arm]["clean_judge"]["signature_fires"] = bool(
            per[arm]["clean_judge"]["gap_vs_online"] > CJ_GAP_BAR)
        if S[arm]["fired"]["applied"]:
            per[arm]["realized_geometry"] = dict(
                mean_abs_cos=S[arm]["fired"]["mean_abs_cos"],
                mean_cos=S[arm]["fired"]["mean_cos"],
                min_cos=S[arm]["fired"]["min_cos"],
                mean_norm_ratio=S[arm]["fired"]["mean_norm_ratio"])

    # ---- the registered decision (point rules on means; CIs alongside)
    c_dir = per["direction"]["mean"]
    c_mag = per["magnitude"]["mean"]
    h_dir, h_mag = per["direction"]["health"], per["magnitude"]["health"]
    cj_gap_ctl = float((J_ctl - S["none"]["online_tail"]).mean())

    geo_add = per["additive05"].get("realized_geometry") or {}
    add_cos = geo_add.get("mean_cos", float("nan"))
    mag_abs_cos = (per["magnitude"].get("realized_geometry") or {}).get(
        "mean_abs_cos", float("nan"))
    dir_norm_ratio = (per["direction"].get("realized_geometry") or {}).get(
        "mean_norm_ratio", float("nan"))
    if h_dir == "healthy" and h_mag == "damaged":
        clause = ("DIRECTION CARRIES THE ANCHOR (a healthy AND b damaged)")
        verdict = (
            f"Unit-norm ORIGINAL directions anchor the run (direction-only "
            f"cost {c_dir:+.4f} {_fmt_ci(per['direction']['ci'])} < "
            f"{HEALTHY_BAR}) while norm-matched RANDOM directions damage it "
            f"(magnitude-only cost {c_mag:+.4f} "
            f"{_fmt_ci(per['magnitude']['ci'])} > {DAMAGED_BAR}): the run "
            f"reads WHERE the band entries point, not how loud they are. "
            f"e096's corruption-robustness is norm-REDUNDANCY — additive "
            f"noise perturbs directions only partially (eps=0.5 keeps mean "
            f"cos ~ {add_cos:.2f}) and the intact directional "
            f"content carries the anchor; wholesale direction destruction "
            f"(mean |cos| ~ {mag_abs_cos:.2f}) "
            f"collapses it — the single-shot twin of e080's dynamic noise "
            f"arm (cj {E080_NOISE_CJ_DELTA:+.2f}). The eviction field's "
            f"'pure mass' model is falsified at this scale: mass without "
            f"geometry does not anchor.")
    elif h_dir == "damaged" and h_mag == "healthy":
        clause = ("MAGNITUDE/STATISTICAL MASS CARRIES THE ANCHOR "
                  "(a damaged AND b healthy)")
        verdict = (
            f"Norm-matched RANDOM directions anchor the run (magnitude-only "
            f"cost {c_mag:+.4f} {_fmt_ci(per['magnitude']['ci'])} < "
            f"{HEALTHY_BAR}) while unit-norm ORIGINAL directions damage it "
            f"(direction-only cost {c_dir:+.4f} "
            f"{_fmt_ci(per['direction']['ci'])} > {DAMAGED_BAR}): pure "
            f"nonzero mass at the right per-vector norms is what matters — "
            f"the KV-eviction field's implicit model. Directional content "
            f"is NOT read (or is fully redundant), and the damage of "
            f"direction-only must live in its magnitude cut (realized norm "
            f"ratio {dir_norm_ratio:.2f}); e105's family law then lives in "
            f"the norms' distributional shape or in K-space, not V-"
            f"direction. e096's corruption-robustness follows: additive "
            f"noise preserves the mass.")
    elif h_dir == "damaged" and h_mag == "damaged":
        clause = "BOTH NEEDED (a AND b damaged)"
        verdict = (
            f"Neither direction alone at reduced magnitude (direction-only "
            f"{c_dir:+.4f} {_fmt_ci(per['direction']['ci'])}) nor magnitude "
            f"alone at random direction (magnitude-only {c_mag:+.4f} "
            f"{_fmt_ci(per['magnitude']['ci'])}) anchors the run — both "
            f"exceed {DAMAGED_BAR}: the redundancy is JOINT. The anchor is "
            f"the full vector population (direction AND magnitude "
            f"simultaneously), consistent with e096's additive arm being "
            f"free precisely because it preserves both roughly (cos ~ 0.89, "
            f"norm ratio ~ 1.12). Removal-fragility (e089) and corruption-"
            f"robustness (e096) are two faces of one joint object.")
    elif h_dir == "healthy" and h_mag == "healthy":
        clause = ("MORE INTERCHANGEABLE THAN HYPOTHESIZED (a AND b healthy) "
                  "— shock registered honestly")
        verdict = (
            f"BOTH decomposition arms anchor the run: direction-only "
            f"{c_dir:+.4f} {_fmt_ci(per['direction']['ci'])} AND "
            f"magnitude-only {c_mag:+.4f} {_fmt_ci(per['magnitude']['ci'])} "
            f"are under the {HEALTHY_BAR} bar. Random-direction replacement "
            f"of the WHOLE band at matched scale, single-shot, is FREE — "
            f"the honest shock: the band's entries are even more "
            f"interchangeable than hypothesized. Then e080's dynamic noise "
            f"collapse (cj {E080_NOISE_CJ_DELTA:+.2f}) must live in the "
            f"SCHEDULE (replacement at pruning time, repeatedly, during "
            f"generation) rather than in direction destruction per se — "
            f"the schedule law (T051's rider: continuous ~8x lumps) gains a "
            f"corruption-face. The T051 discriminator lands on 'more "
            f"interchangeable', not the three registered candidates; "
            f"register the surprise, do not smooth it.")
    else:
        clause = ("TEXTURE (no registered cell — at least one of (a)/(b) "
                  "in the ambiguous band [%s, %s])" % (HEALTHY_BAR, DAMAGED_BAR))
        verdict = (
            f"Direction-only cost {c_dir:+.4f} {_fmt_ci(per['direction']['ci'])} "
            f"[{h_dir}] | magnitude-only cost {c_mag:+.4f} "
            f"{_fmt_ci(per['magnitude']['ci'])} [{h_mag}] — the (a)/(b) "
            f"pair does not land in a registered cell (healthy < "
            f"{HEALTHY_BAR}, damaged > {DAMAGED_BAR}). Supporting arms: "
            f"scaled-0.1x {per['scaled']['mean']:+.4f} "
            f"{_fmt_ci(per['scaled']['ci'])} [{per['scaled']['health']}], "
            f"additive eps={ADDITIVE_EPS} {per['additive05']['mean']:+.4f} "
            f"{_fmt_ci(per['additive05']['ci'])} "
            f"[{per['additive05']['health']}] (e096 replication), none "
            f"{per['none']['mean']:+.4f} (instrument). The boundary "
            f"structure is itself informative — a partial-redundancy "
            f"reading needs the follow-up, not a verdict.")

    gap_fire = {a: per[a]["clean_judge"]["signature_fires"] for a in ARMS}
    attractor_note = (f"clean-judge attractor signature (gap > {CJ_GAP_BAR} "
                      f"nats): control gap {cj_gap_ctl:+.3f}; fires for "
                      + (", ".join(a for a in ARMS if gap_fire[a]) or "NO arm")
                      + f" | gaps: "
                      + " ".join(f"{a}:{per[a]['clean_judge']['gap_vs_online']:+.2f}"
                                 for a in ARMS)
                      + f" (e080 vzero calibration {E080_VZERO_CJ_GAP:+.2f})")
    log("arm table: " + " | ".join(f"{a} {per[a]['mean']:+.4f} "
                                   f"[{per[a]['health']}]"
                                   for a in ARMS))
    log(attractor_note)
    log(f"REGISTERED DECISION [{clause}]: {verdict}")

    # ---------------------------------------------------------------- metrics
    metrics = dict(
        experiment="e102_direction_magnitude",
        purpose="T051's registered WHY-is-corruption-free discriminator "
                "(E096 amendment): direction-vs-magnitude decomposition of "
                "the anchor band at the e096 intervention point (g=250, all "
                "154 band V-entries, single-shot, matched-stream, "
                "clean-judged). (a) direction-only unit vectors vs (b) "
                "magnitude-only random directions vs (c) scaled-0.1x "
                "originals vs (d) e096 eps=0.5 additive control (bitwise "
                "replication) + none. (a) healthy AND (b) damaged => "
                "DIRECTION CARRIES THE ANCHOR; (a) damaged AND (b) healthy "
                "=> MAGNITUDE/STATISTICAL MASS; both damaged => BOTH "
                "NEEDED; both healthy => MORE INTERCHANGEABLE (shock).",
        started=started, wall_s=elapsed(), threads=THREADS, cpu_only=True,
        net=dict(ckpt=str(CKPT), arch=dict(n_layer=4, n_head=4, n_embd=128,
                                           block_size=T_TOTAL, vocab=65),
                 params=n_params, val_ce=val_ce, val_ce_e053c=E053C_VAL_CE),
        seeds=dict(corpus=1337, prompts=SEED_PROMPT, sampling=SEED_SAMPLE,
                   noise_magnitude=SEED_NOISE,
                   noise_e096_replication=SEED_NOISE_E096,
                   continuation=SEED_CONT, bootstrap=0),
        protocol=dict(
            B=B, prompt_tokens=PROMPT_TOK, t_total=T_TOTAL, temp=TEMP,
            topk=TOPK, tail_window=list(KEY_T), boot_n=BOOT_N,
            intervention=dict(
                g_int=G_INT, t_int=T_INT, timing="transform applied ONCE, "
                "immediately before the decode at t_int (top-of-the-event "
                "convention); token at t_int teacher-forced from the control "
                f"run; first affected sample = t_int+1 = {FIRST_AFFECTED}; "
                f"tail exposure {T_TOTAL - 1 - FIRST_AFFECTED + 1} free steps",
                band=dict(positions=[BAND[0], BAND[-1]], n=len(BAND),
                          rule="self-generated entries with age > 96 at the "
                               "intervention frame (e075 dead-band rule): "
                               "positions 64..217, ages 97..250 at query 314",
                          note="single-shot feasible because every band "
                               "entry already exists at t_int"),
                v_only=True),
            arms=dict(
                none="no modification (bitwise identity control)",
                direction="V <- V/||V|| — direction preserved exactly "
                          "(cos=1), magnitude cut to unit norm (realized "
                          "retention 1/median-norm per layer ~ 0.46..0.64)",
                magnitude="V <- u*||V|| — fresh random unit direction u at "
                          "the ORIGINAL per-vector norm: magnitude exact, "
                          "direction destroyed (E|cos| ~ 0.14 in d=32); "
                          "e080's noise arm recomputed at matched scale + "
                          "single-shot timing (the ANCHOR arm)",
                scaled=f"V <- {SCALED_FACTOR}*V — direction preserved, norm "
                       f"cut to {SCALED_FACTOR:.0%}",
                additive05=f"V <- V + {ADDITIVE_EPS}*m_layer*u — e096's "
                           f"mid-rung control, BITWISE replication of its "
                           f"eps=0.5 arm (4th noise batch of seed 9696; "
                           f"G5 gate)"),
            readouts=dict(
                cost="clean-judge tail CE (448..511, clean full recompute) "
                     "of the arm stream minus the MATCHED control stream, "
                     "per row then mean over B=8",
                attractor="clean-judge gap = judge CE minus the run's own "
                          "online tail CE (e080 off-manifold bar 1.0 nat; "
                          "vzero calibration +5.02) + tail entropy "
                          "(fluency spot-check)",
                realized_geometry="per-arm mean |cos(V_new,V_old)| and mean "
                                  "||V_new||/||V_old|| over all 8x4x154 band "
                                  "vectors (places each arm on the "
                                  "direction x magnitude design grid)"),
            bars=dict(
                healthy=f"mean cost < {HEALTHY_BAR} nats",
                damaged=f"mean cost > {DAMAGED_BAR} nat",
                ambiguous=f"in [{HEALTHY_BAR}, {DAMAGED_BAR}] — no "
                          "registered cell",
                matrix=("(a) healthy AND (b) damaged -> DIRECTION CARRIES; "
                        "(a) damaged AND (b) healthy -> MAGNITUDE/STATISTICAL "
                        "MASS; both damaged -> BOTH NEEDED; both healthy -> "
                        "MORE INTERCHANGEABLE (shock); else TEXTURE"),
                ci_note="point rules on means; paired-row bootstrap CIs "
                        "reported alongside (e089/e096 convention)"),
            registered_numbers=dict(healthy_bar=HEALTHY_BAR,
                                    damaged_bar=DAMAGED_BAR,
                                    cj_gap_bar=CJ_GAP_BAR,
                                    scaled_factor=SCALED_FACTOR,
                                    additive_eps=ADDITIVE_EPS)),
        gates=gates,
        control_run=dict(clean_judge_tail_ce=J_ctl.tolist(),
                         online_tail=float(S["none"]["online_tail"].mean()),
                         cj_gap=cj_gap_ctl,
                         ent_tail=float(ent_ctl[:, TAIL_STEP0:].mean())),
        references=dict(e075_wholeband_vzero=e075_ref,
                        e089_k200_continuous=e089_ref,
                        e080_noise_dynamic_cj_delta=E080_NOISE_CJ_DELTA,
                        e080_noise_dynamic_r1_delta=E080_NOISE_R1_DELTA,
                        e080_vzero_cj=E080_VZERO_CJ,
                        e080_vzero_cj_gap=E080_VZERO_CJ_GAP,
                        e096_eps05_mean=(e096_eps05 or {}).get("mean")),
        summary=dict(
            per_arm={a: per[a] for a in ARMS},
            attractor=dict(gap_bar=CJ_GAP_BAR, fires=gap_fire,
                           control_gap=cj_gap_ctl)),
        registered_decision=dict(
            clause=clause, verdict=verdict, attractor_note=attractor_note,
            numbers=dict(direction_cost=c_dir, magnitude_cost=c_mag,
                         direction_health=h_dir, magnitude_health=h_mag,
                         means={a: per[a]["mean"] for a in ARMS},
                         healths={a: per[a]["health"] for a in ARMS})),
    )
    save_json(out_dir / "metrics.json", metrics)
    log("metrics.json written")

    plot(out_dir / "direction_magnitude.png", metrics)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot


def plot(path: Path, M: dict):
    dec = M["registered_decision"]
    n = dec["numbers"]
    S = M["summary"]
    refs = M["references"]
    means = {a: n["means"][a] for a in ARMS}
    cis = {a: S["per_arm"][a]["ci"] for a in ARMS}
    rows = {a: np.asarray(S["per_arm"][a]["cost_per_row"]) for a in ARMS}
    healths = {a: n["healths"][a] for a in ARMS}
    cj = {a: S["per_arm"][a]["clean_judge"] for a in ARMS}
    geo = {a: S["per_arm"][a].get("realized_geometry") for a in ARMS}
    labels = {"none": "none\n(control)", "direction": "direction-only\n(unit |V|)",
              "magnitude": "magnitude-only\n(rand dir, orig |V|)",
              "scaled": f"scaled\n({SCALED_FACTOR:.0%} |V|)",
              "additive05": f"additive\n(eps={ADDITIVE_EPS}, e096 repl)"}
    hcol = {"healthy": "tab:green", "damaged": "tab:red",
            "ambiguous": "tab:orange"}
    run_cols = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, axes = plt.subplots(2, 3, figsize=(24, 11))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # ---- panel 1: THE five-arm cost with healthy/collapsed bands marked
    x = np.arange(len(ARMS))
    ymax = max(1.6, max(max(r.max() for r in rows.values()) + 0.15, 0.0))
    ymin = min(-0.2, min(min(r.min() for r in rows.values()) - 0.15, 0.0))
    ax1.axhspan(ymin, HEALTHY_BAR, color="tab:green", alpha=0.08, zorder=0)
    ax1.axhspan(HEALTHY_BAR, DAMAGED_BAR, color="tab:orange", alpha=0.10,
                zorder=0)
    ax1.axhspan(DAMAGED_BAR, ymax, color="tab:red", alpha=0.08, zorder=0)
    ax1.axhline(HEALTHY_BAR, color="tab:green", ls="--", lw=1.5)
    ax1.axhline(DAMAGED_BAR, color="tab:red", ls="--", lw=1.5)
    ax1.axhline(0, color="k", lw=0.8)
    ax1.text(len(ARMS) - 0.45, HEALTHY_BAR, f" healthy bar {HEALTHY_BAR}",
             fontsize=8.5, color="tab:green", va="bottom", ha="right")
    ax1.text(len(ARMS) - 0.45, DAMAGED_BAR, f" collapsed bar {DAMAGED_BAR}",
             fontsize=8.5, color="tab:red", va="bottom", ha="right")
    ax1.text(len(ARMS) - 0.42, (ymin + HEALTHY_BAR) / 2, "HEALTHY band",
             fontsize=9, color="tab:green", ha="right", va="center")
    ax1.text(len(ARMS) - 0.42, (HEALTHY_BAR + DAMAGED_BAR) / 2, "ambiguous",
             fontsize=9, color="tab:orange", ha="right", va="center")
    ax1.text(len(ARMS) - 0.42, (DAMAGED_BAR + ymax) / 2, "COLLAPSED band",
             fontsize=9, color="tab:red", ha="right", va="center")
    mm = [means[a] for a in ARMS]
    err = [[max(0.0, means[a] - cis[a][0]) for a in ARMS],
           [max(0.0, cis[a][1] - means[a]) for a in ARMS]]
    ax1.bar(x, mm, 0.55, color=[hcol[healths[a]] for a in ARMS],
            alpha=0.55, edgecolor="k", lw=0.6, zorder=2)
    ax1.errorbar(x, mm, yerr=err, fmt="none", ecolor="k", elinewidth=1.6,
                 capsize=5, zorder=4, label="mean (paired-row CI)")
    for xi, a in zip(x, ARMS):
        ax1.scatter(np.full(B, xi) + np.linspace(-0.10, 0.10, B), rows[a],
                    s=30, color=run_cols[np.arange(B)], alpha=0.75, zorder=3)
        ax1.text(xi, (ymax - ymin) * 0.02 + ymin, f"{means[a]:+.3f}",
                 ha="center", va="bottom", fontsize=8.5, zorder=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([labels[a] for a in ARMS], fontsize=8.5)
    ax1.set_ylim(ymin, ymax)
    ax1.set_ylabel("clean-judge tail cost vs matched control (nats)")
    ax1.set_title("E102-1 — THE decomposition: five arms' clean-judge costs "
                  "with healthy/collapsed bands", fontsize=10)
    ax1.legend(fontsize=8.5, loc="upper left")

    # ---- panel 2: the design grid — what each arm preserves
    ax2.axhline(1.0, color="gray", lw=0.6)
    ax2.axvline(1.0, color="gray", lw=0.6)
    ax2.scatter([1.0], [1.0], marker="*", s=260, color="k", zorder=5)
    ax2.annotate("pristine", (1.0, 1.0), textcoords="offset points",
                 xytext=(6, 6), fontsize=9)
    rng_cos = math.sqrt(2.0 / (math.pi * 32))         # E|cos|, random dirs
    ax2.scatter([rng_cos], [1.0], marker="x", s=90, color="gray", zorder=5)
    ax2.annotate(f"random-direction\nexpectation ({rng_cos:.3f})",
                 (rng_cos, 1.0),
                 textcoords="offset points", xytext=(6, -22), fontsize=7.5,
                 color="gray")
    for a in ARMS:
        if geo[a] is None:
            continue
        ax2.scatter([geo[a]["mean_abs_cos"]], [geo[a]["mean_norm_ratio"]],
                    s=220, color=hcol[healths[a]], edgecolor="k", zorder=5,
                    alpha=0.9)
        ax2.annotate(f"{a}\n{means[a]:+.3f}",
                     (geo[a]["mean_abs_cos"], geo[a]["mean_norm_ratio"]),
                     textcoords="offset points", xytext=(9, -4), fontsize=8.5)
    ax2.set_xlabel("direction preserved: mean |cos(V_new, V_old)|")
    ax2.set_ylabel("magnitude preserved: mean ||V_new||/||V_old||")
    ax2.set_xlim(-0.05, 1.18)
    ax2.set_ylim(0.0, 1.25)
    ax2.set_title("E102-2 — the design grid: each arm's REALIZED vector "
                  "geometry (color = health)", fontsize=10)
    ax2.text(0.02, 0.02, "right = direction kept | top = magnitude kept",
             transform=ax2.transAxes, fontsize=8, color="gray")

    # ---- panel 3: clean-judge + online tail CE per arm (the gap opening)
    cjm = [cj[a]["mean"] for a in ARMS]
    onm = [cj[a]["online_tail"] for a in ARMS]
    ax3.plot(x, cjm, "o-", color="tab:red", lw=2.2, ms=9,
             label="clean-judge tail CE")
    ax3.plot(x, onm, "s--", color="tab:blue", lw=1.8, ms=7,
             label="online (self-scored) tail CE")
    ax3.axhline(refs["e080_vzero_cj"], color="tab:orange", ls=":", lw=1.4,
                label=f"e080 vzero clean-judge {refs['e080_vzero_cj']:.2f}")
    for xi, v in zip(x, cjm):
        ax3.text(xi, v + 0.06, f"{v:.2f}", ha="center", fontsize=7.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels([labels[a] for a in ARMS], fontsize=8.5)
    ax3.set_ylabel("tail CE (nats)")
    ax3.set_title("E102-3 — judge vs self-score: the off-manifold gap "
                  "per arm (bar 1.0)", fontsize=10)
    ax3.legend(fontsize=8.5)

    # ---- panel 4: tail entropy (the fluency spot-check)
    ent = [cj[a]["ent_tail"] for a in ARMS]
    ax4.bar(x, ent, 0.55, color=[hcol[healths[a]] for a in ARMS],
            alpha=0.85, edgecolor="k", lw=0.5)
    ax4.axhline(ent[0], color="k", lw=0.9)
    for xi, v in zip(x, ent):
        ax4.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax4.set_xticks(x)
    ax4.set_xticklabels([labels[a] for a in ARMS], fontsize=8.5)
    ax4.set_ylabel("tail entropy (nats)")
    ax4.set_title("E102-4 — fluency spot-check: tail entropy per arm",
                  fontsize=10)

    # ---- panel 5: stream texture — first divergence per arm
    for xi, a in zip(x, ARMS):
        fdr = [d if d is not None else T_TOTAL
               for d in S["per_arm"][a]["first_div_per_row"]]
        ax5.scatter(np.full(B, xi) + np.linspace(-0.10, 0.10, B), fdr,
                    s=36, alpha=0.8,
                    color="tab:red" if any(d is None for d in
                                           S["per_arm"][a]
                                           ["first_div_per_row"])
                    else "tab:blue")
    ax5.axhline(KEY_T[0], color="tab:green", ls="-.", lw=1.2,
                label="tail start 448")
    ax5.axhline(T_TOTAL, color="gray", ls=":", lw=1.2)
    ax5.text(0.0, T_TOTAL - 6, "512 = never diverged", fontsize=8)
    ax5.set_xticks(x)
    ax5.set_xticklabels([labels[a] for a in ARMS], fontsize=8.5)
    ax5.set_xlabel("arm")
    ax5.set_ylabel("first position arm != control")
    ax5.set_ylim(T_INT, T_TOTAL + 8)
    ax5.set_title("E102-5 — stream divergence vs arm", fontsize=10)
    ax5.legend(fontsize=9)

    # ---- panel 6: verdict text
    ax6.axis("off")
    r96 = refs.get("e096_eps05_mean")
    r96s = f"{r96:+.4f}" if isinstance(r96, float) else "n/a"
    r80c = refs["e080_noise_dynamic_cj_delta"]
    r80o = refs["e080_noise_dynamic_r1_delta"]
    lines = [
        "REGISTERED (frozen from the tasking):",
        f"  healthy: mean cost < {HEALTHY_BAR} | damaged: mean cost > "
        f"{DAMAGED_BAR} | else ambiguous",
        "  (a) healthy AND (b) damaged -> DIRECTION CARRIES THE ANCHOR",
        "  (a) damaged AND (b) healthy -> MAGNITUDE/STATISTICAL MASS",
        "  both damaged -> BOTH NEEDED | both healthy -> MORE "
        "INTERCHANGEABLE (shock)",
        "",
        "ARM TABLE (means, paired-row CI; health):",
    ] + [
        f"  {a:<11}: {means[a]:+.4f} {_fmt_ci(cis[a])}  [{healths[a]}]"
        + (f"  (|cos| {geo[a]['mean_abs_cos']:.2f}, |V| ratio "
           f"{geo[a]['mean_norm_ratio']:.2f})" if geo[a] else
           "  (identity arm)")
        for a in ARMS
    ] + [
        "",
        f"REFS: e075 whole-band {refs['e075_wholeband_vzero']:+.4f} (K=32) | "
        f"e089 k=200 {refs['e089_k200_continuous']:+.4f} (K=1)",
        f"      e080 dynamic noise cj {r80c:+.4f} (online {r80o:+.4f}) | "
        f"e096 eps=0.5 {r96s} (G5 replication)",
        "",
        f"ATTRACTOR: control gap {S['attractor']['control_gap']:+.3f}; "
        f"signature fires: "
        + (", ".join(a for a, f in S["attractor"]["fires"].items() if f)
           or "NO arm"),
        "  gaps: " + " ".join(f"{a}:{cj[a]['gap_vs_online']:+.2f}"
                              for a in ARMS),
        "",
        f"DECISION [{dec['clause']}]:",
    ] + [f"  {wd}" for wd in _wrap(dec["verdict"], 98)]
    ax6.text(0.02, 0.97, "E102 — direction-vs-magnitude decomposition "
             "(T051's registered WHY probe)", fontsize=13, weight="bold",
             va="top")
    for i, tx in enumerate(lines):
        ax6.text(0.02, 0.935 - i * 0.0285, tx, fontsize=8.4, va="top",
                 family="monospace")

    fig.suptitle("E102 — direction vs magnitude at the anchor band (g=250, "
                 f"154 V-entries, single-shot, clean-judged) | {dec['clause']}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
