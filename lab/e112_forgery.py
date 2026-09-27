"""E112 — the SIGNATURE FORGERY test (T061-earned; W004's question: can a
fixed point be forged?).

[REGISTERED DESIGN — frozen in this docstring BEFORE any compute]

THE QUESTION (W004 verbatim, sharpened by T061): "can a fixed point be
forged? (Craft V-vectors in the signature subspace by hand — if the run
accepts them, selfhood is a k-dim lock pickable... it would say the net's
self is shallower than it acts)". e111 found k* = 7: the sibling/foreign
energy separation in the recipient's own V-manifold principal subspace
fires at SEVEN dimensions (sibling E(7)=0.815 vs foreign 0.220 vs same-norm
null 0.219; top-1 PC alone 11.7x). Selfhood is measured as a 7-dim stamp.
THE TEST: can a forged key open the lock?

RIG (verbatim, two published rigs joined):
  - SPLICE: the e096/e102 single-shot anchor-splice procedure VERBATIM —
    recipient e053c_ctx512, seed-202 8-prompt battery, seed-7 control free
    run; intervention at g_int=250 (prefix 0..313, teacher-forced control
    token at 314), band = self-generated positions 64..217 (154 entries,
    ages 97..250 at query 314), ALL replaced simultaneously, V-only,
    K untouched, before the decode at 314; matched continuation seed
    960250 shared by every arm + the control.
  - READOUT (the standard e099/e102 convention): clean-judge tail CE over
    the final 64 positions (448..511) of each arm's stream MINUS the
    matched control's, per-row then mean over B=8, paired-row bootstrap
    CI. healthy < 0.3 nats, collapsed > 1.0 nat.
  - SIGNATURE BASIS: e111 VERBATIM — the recipient's own anchor-band V set
    (the seed-7 none-run's FINAL caches at the 324 splice positions
    64..387; 16 (layer,head) groups x 2592 vectors x 32-d), per-group
    uncentered second-moment eigendecomposition in float64; top-K_SIG=7
    eigenvectors = the self-signature subspace. Regenerated bit-exactly
    and gated against runs/e111/metrics.json.

DONOR CONTENT for the band (all published rules; NO new sampling anywhere
— every generator is a published seed):
  - SIBLING (e099 randomize rule): src[b] = prefill-cache row d_s(b), same
    position; d_s = draw_donor(4343). Single-shot => no prior events =>
    the prefill cache IS the pristine pre-event snapshot.
  - CORPUS (e080/e099 promptcopy rule): src[b] = the row's own pristine
    prompt entry at q = ((p-64) mod 63)+1 — recipient weights x corpus
    text (the arm W004 used to prove the stamp is JOINT).
  - FOREIGN (e105 crossfamily rule): src[b] = e021_task copy-net donor
    window 1 (seeds 5601/5701) at the SAME position (all band p<=255 ->
    window1, position-matched), heads 0..3, d_s mapping draw_donor(4646).

THE FIVE REGISTERED ARMS (band content at g=250; norm of every forged
vector = its SOURCE's norm — the projection changes direction only):
  (a) forged_full     corpus src projected ONTO the top-7 subspace
                      (keep norms) — signature-only vectors, nothing else
                      self: corpus content reduced to the stamp.
  (b) forged_foreign  same projection applied to the FOREIGN donor's
                      V-vectors — the stamp on maximally-foreign clay.
  (c) sig_destroyed   SIBLING src projected OFF the top-7 subspace
                      (orthogonal complement, 25-d; keep norms) —
                      self-content with the signature removed.
  (d) sibling         sibling src VERBATIM (validation control: must be
                      healthy — e099's randomize was +0.027 dynamic).
      corpus          corpus src VERBATIM (validation control: must
                      collapse — e099's promptcopy was +5.325 dynamic).
  RIDER (unregistered, for honest interpretation of (b) only):
      foreign         foreign src VERBATIM — the in-rig anchor showing
                      un-forged foreign content collapses in THIS rig
                      (e105's dynamic crossfamily was +5.09; e102's
                      random-direction single-shot was +1.80).

REGISTERED BARS (frozen; healthy < 0.3 / collapsed > 1.0 on the mean gap):
  FORGEABLE             fires iff (a) OR (b) healthy — a forged key opened
                        the lock: self is exactly k=7 deep — pickable.
  SIGNATURE-NECESSARY   fires iff (c) collapsed — the 7-dim stamp is not
                        just correlational but CAUSAL: removing it kills
                        self-content.
  STAMP-IS-CORRELATE    fires iff (c) healthy AND (a) collapsed AND
                        (b) collapsed — the anchor reads beyond the
                        signature: self is deeper than its stamp.
  VALIDATION CONTROLS (required for the run to count): sibling healthy
  AND corpus collapsed. All combinations informative — every cell is
  named and reported honestly (both FORGEABLE + SIGNATURE-NECESSARY =
  EXACT k=7 LOCK: the stamp is sufficient AND necessary).

GATES: G1 e053c val CE (tol 0.02); G2 params 873,472; G3 protocol identity
  (control free run: clean-judge per-seq vs runs/e080 stored none arm
  < 1e-4 AND bitwise vs runs/e089 stored control; control continuation
  bitwise vs runs/e102 stored control_run); G4 instrument identity (none
  continuation bitwise == control; all arms prefix-identical through 314;
  per-arm band-transform exactness — norm kept within 1e-5, forged arms'
  written energy 100% in top-7 / destroyed arm's 0% (1e-6), verbatim arms
  bitwise-equal to their sources; non-band V and ALL K bitwise untouched;
  med_norm vs e096 stored; e021 donor window rerun bit-identical + val CE
  vs e063b); G5 signature basis (eigval fractions + own E(7) vs
  runs/e111 published, dev < 1e-12; published k* == 7); G6 content
  identity (donor derangements 4343/4646 ok; corpus map into prompt
  positions 1..63; foreign sources donor-native; 154 positions/arm).

Run:     python lab/e112_forgery.py
Outputs: runs/e112/metrics.json + runs/e112/forgery.png
Envelope: NO training, NO new automations; CPU-only (CUDA masked pre-torch,
8 threads), single step, minutes. No NOTES/THINKING/QUEUE/STATE edits; no
commit (the dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu" (e053b..e111)
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

import json as _json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import textwrap  # noqa: E402
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
import e080_prune_vs_replace as e080   # constants + prompt_source_of
import e099_attractor_identity as e099  # draw_donor
import e105_cross_family as e105        # donor_run + donor constants

THREADS = 8                                # task spec / T050: 12 thrashes box
torch.set_num_threads(THREADS)             # (e080 module import sets 12)

# ------------------------------------------------------------------ constants
PROMPT_TOK = 64
T_TOTAL = 512
TEMP, TOPK = 0.8, 40
SEED_PROMPT, SEED_SAMPLE = 202, 7
SEED_CONT = 960250                        # e096/e102's matched continuation
B = 8
N_PROMPTS = 8
BOOT_N = 1000
VOCAB = 65

G = T_TOTAL - PROMPT_TOK                  # 448
TAIL = 64
KEY_T = (T_TOTAL - TAIL, T_TOTAL - 1)     # (448, 511) judged window

# ---- the single-shot intervention (VERBATIM e096/e102) -----------------------
G_INT = 250
T_INT = PROMPT_TOK + G_INT                # 314
GEN_FIRST = 64
BAND = list(range(GEN_FIRST, T_INT - 96))  # 64..217 (154 entries)
FIRST_AFFECTED = T_INT + 1                # 315
TAIL_STEP0 = KEY_T[0] - FIRST_AFFECTED    # 133

# ---- the signature basis (e111) ---------------------------------------------
BAND_LO, BAND_HI = 64, 387                # the 324-position e111 population
N_BAND_FULL = BAND_HI - BAND_LO + 1
K_SIG = 7                                 # T061's published k* (gated in G5)
HEAD_DIM = 32

# ---- donor maps (published seeds; NO new sampling) ---------------------------
SEED_DONOR_SIB = e099.SEED_DONOR          # 4343 (e099 randomize)
SEED_DONOR_FOR = e105.SEED_DONOR_MAP      # 4646 (e105 crossfamily)

# ---- registered bars (frozen) ------------------------------------------------
HEALTHY_BAR = 0.3
COLLAPSED_BAR = 1.0

# ---- the arms ----------------------------------------------------------------
ARMS_REG = ["forged_full", "forged_foreign", "sig_destroyed",
            "sibling", "corpus"]
ARMS_RIDER = ["foreign"]                  # unregistered interpretation anchor
ARMS = ARMS_REG + ARMS_RIDER
ARM_LABELS = {
    "forged_full": "(a) FORGED-FULL\ncorpus V -> top-7, norms kept",
    "forged_foreign": "(b) FORGED-ON-FOREIGN\nforeign V -> top-7, norms kept",
    "sig_destroyed": "(c) SIGNATURE-DESTROYED\nsibling V -> off top-7 (25-d)",
    "sibling": "(d) sibling control\nsibling V verbatim",
    "corpus": "(d) corpus control\ncorpus V verbatim",
    "foreign": "rider: foreign control\nforeign V verbatim",
}
ARM_COLS = {
    "forged_full": "tab:olive", "forged_foreign": "tab:brown",
    "sig_destroyed": "tab:cyan", "sibling": "tab:green",
    "corpus": "tab:red", "foreign": "tab:orange",
}

# ---- reference numbers (protocol-identity gates) -----------------------------
CKPT = REPO / "runs" / "checkpoints" / "e053c_ctx512.pt"
E053C_VAL_CE = 1.5226792494455974
E080_METRICS = REPO / "runs" / "e080" / "metrics.json"
E089_METRICS = REPO / "runs" / "e089" / "metrics.json"
E102_METRICS = REPO / "runs" / "e102" / "metrics.json"
E111_METRICS = REPO / "runs" / "e111" / "metrics.json"
E096_MED_NORM_REF = [2.157031774520874, 1.5538004636764526,
                     1.9136782884597778, 1.8171889781951904]
E105_VAL_CE_TASK = e105.E063B_VAL_CE_TASK  # e021_task val CE (e063b ref)
# e111's published numbers (the basis this experiment forges with)
E111_PUB = dict(k_star=7, own_E7=0.8112686695343765,
                E7_sibling=0.8150432735494842,
                E7_foreign=0.21997439006909503,
                E7_null=0.21879670746621285)
# e099/e105's dynamic-rig controls (context; this rig re-validates them)
E099_GAP_RANDOMIZE = 0.027
E099_GAP_PROMPTCOPY = 5.325
E105_GAP_CROSSFAMILY = 5.094

T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ------------------------------------------------- machinery (VERBATIM e102)

@torch.no_grad()
def manual_all_logits(net: TinyGPT, idxs):
    """Clean forward returning logits at ALL positions (N, T, vocab) — the
    clean-judge instrument. [VERBATIM e080/e096/e102]"""
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
    [VERBATIM e075/e080/e096/e102]"""
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
    [VERBATIM e075/e080/e096/e102]"""
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
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
        probs = torch.softmax(att, -1)
        y = (probs @ v).transpose(1, 2).reshape(Bb, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
    return net.lm_head(net.ln_f(x))


@torch.no_grad()
def generate_control(net: TinyGPT, prompts, gen: torch.Generator):
    """Free-run 64->512 for B sequences (e075/e080/e096/e102 A-none VERBATIM
    stream math); final V caches captured (the e111 basis population)."""
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
    """Per-layer MEDIAN ||V|| over all band vectors of a PRISTINE
    post-prefill cache. [VERBATIM e096/e102]"""
    sel = torch.tensor(BAND, dtype=torch.long)
    meds = []
    for (_k, v) in kv:
        nrm = v[:, :, sel, :].norm(dim=-1)
        meds.append(float(nrm.median()))
    return meds


def judge_windows(all_lg: torch.Tensor, idx: torch.Tensor, windows):
    """Clean-net CE of target windows [(lo, hi), ...]. [VERBATIM e085..e102]"""
    out = {}
    for (lo, hi) in windows:
        lg = all_lg[:, lo - 1:hi, :]
        tgt = idx[:, lo:hi + 1]
        lp = torch.log_softmax(lg.float(), -1)
        out[(lo, hi)] = -lp.gather(2, tgt[:, :, None]).squeeze(2).mean(1) \
            .numpy()
    return out


def boot_rows(arrs, fn, n: int = BOOT_N, seed: int = 0):
    """Paired bootstrap over the 8 rows. [VERBATIM e096/e102]"""
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


def health_of(cost: float) -> str:
    """Registered health code of an arm's mean clean-judge gap."""
    if cost < HEALTHY_BAR:
        return "healthy"
    if cost > COLLAPSED_BAR:
        return "collapsed"
    return "gray"


# ------------------------------------------------- PCA machinery (VERBATIM e111)

def pca_uncentered(X: torch.Tensor):
    """Uncentered second-moment PCA of X (n, d) [float64]: eigenvalues
    (descending) + orthonormal eigenvector basis U (d, d). [VERBATIM e111]"""
    X = X.to(torch.float64)
    C = (X.T @ X) / X.shape[0]
    w, U = torch.linalg.eigh(C)
    order = torch.argsort(w, descending=True)
    return w[order], U[:, order]


# ------------------------------------------------- the forgery arm (e102's
# run_continuation skeleton; the band transform is the forgery)

@torch.no_grad()
def run_forgery_arm(net: TinyGPT, prefix: torch.Tensor,
                    forced_tok: torch.Tensor, forced_pos: int, seed: int,
                    arm: str, basis: dict, dm_sib: torch.Tensor,
                    dm_for: torch.Tensor, donorV: list):
    """Single-shot band-transform continuation at the e096/e102 intervention
    point ("none": the matched control / bitwise identity arm — never called
    with 'none' here; the control is e102's code path, arm='none' kept for
    the instrument gate).

    Band replacement per arm (see docstring):
      sibling / corpus / foreign      src VERBATIM
      forged_full / forged_foreign    src -> top-7 projection, norm kept
      sig_destroyed                   src -> orthogonal complement, norm kept
    Sources are read from THIS prefill's pristine cache (frame-consistent);
    projections in float64 per (layer, head), written back in float32.
    K is never touched; non-band columns bitwise untouched."""
    R = prefix.shape[0]
    gen = torch.Generator().manual_seed(seed)
    logits, kv = prefill_batch(net, prefix)
    sel = torch.tensor(BAND, dtype=torch.long)
    qsel = torch.tensor([e080.prompt_source_of(p) for p in BAND],
                        dtype=torch.long)
    pre_k = [k.clone() for (k, _v) in kv]
    pre_v = [v.clone() for (_k, v) in kv]
    H = net.cfg.n_head
    fired = dict(applied=True, arm=arm, n_vec=0, norm_dev=0.0,
                 min_cos=None, mean_abs_cos=None, mean_cos=None,
                 mean_abs_cos_src=None, mean_norm_ratio=None,
                 energy7_src=None, energy7_written=None,
                 nonband_untouched=True, k_untouched=True,
                 verbatim_bitwise=None)
    coss_old, coss_src, ratios = [], [], []
    e7s_num, e7s_den = 0.0, 0.0
    e7w_num, e7w_den = 0.0, 0.0
    verb_ok = True
    for li, (_k, v) in enumerate(kv):
        old = v[:, :, sel, :].clone()                       # (B,H,n,32)
        if arm in ("sibling", "sig_destroyed"):
            src = pre_v[li][dm_sib][:, :, sel, :]
        elif arm in ("corpus", "forged_full"):
            src = pre_v[li][:, :, qsel, :]
        elif arm in ("foreign", "forged_foreign"):
            src = donorV[li][dm_for][:, 0:H, sel, :]
        else:
            raise ValueError(arm)
        src32 = src.clone()
        if arm in ("sibling", "corpus", "foreign"):
            new = src32                                        # verbatim
            verb_ok &= bool(torch.equal(new, src32))
        else:
            proj = torch.empty_like(src32)
            for h in range(H):
                U7 = basis[(li, h)][:, :K_SIG]                 # (32,7) f64
                s = src32[:, h].to(torch.float64)              # (B,n,32)
                if arm in ("forged_full", "forged_foreign"):
                    p = (s @ U7) @ U7.T                        # into top-7
                else:                                          # destroyed
                    p = s - (s @ U7) @ U7.T                    # complement
                proj[:, h] = p.to(torch.float32)
            nrm_src = src32.norm(dim=-1, keepdim=True)
            nrm_new = proj.norm(dim=-1, keepdim=True).clamp_min(1e-20)
            new = proj * (nrm_src / nrm_new)                   # keep norms
            fired["norm_dev"] = max(fired["norm_dev"], float(
                (new.norm(dim=-1) - nrm_src.squeeze(-1)).abs().max()))
        v[:, :, sel, :] = new
        # ---- bookkeeping (float64 instruments)
        o64 = old.to(torch.float64)
        n64 = new.to(torch.float64)
        s64 = src32.to(torch.float64)
        den = (o64.norm(dim=-1) * n64.norm(dim=-1)).clamp_min(1e-12)
        cos_old = (o64 * n64).sum(-1) / den
        den2 = (s64.norm(dim=-1) * n64.norm(dim=-1)).clamp_min(1e-12)
        cos_src = (s64 * n64).sum(-1) / den2
        coss_old.append(cos_old.reshape(-1))
        coss_src.append(cos_src.reshape(-1))
        ratios.append((n64.norm(dim=-1)
                       / o64.norm(dim=-1).clamp_min(1e-12)).reshape(-1))
        fired["n_vec"] += int(cos_old.numel())
        for h in range(H):
            U7 = basis[(li, h)][:, :K_SIG]
            e7s_num += float((((s64[:, h] @ U7) ** 2).sum()))
            e7s_den += float((s64[:, h] ** 2).sum())
            e7w_num += float((((n64[:, h] @ U7) ** 2).sum()))
            e7w_den += float((n64[:, h] ** 2).sum())
        fired["min_cos"] = (float(cos_src.min())
                            if fired["min_cos"] is None
                            else min(fired["min_cos"], float(cos_src.min())))
    fired["mean_abs_cos"] = float(torch.cat(coss_old).abs().mean())
    fired["mean_cos"] = float(torch.cat(coss_old).mean())
    fired["mean_abs_cos_src"] = float(torch.cat(coss_src).abs().mean())
    fired["mean_norm_ratio"] = float(torch.cat(ratios).mean())
    fired["energy7_src"] = e7s_num / e7s_den
    fired["energy7_written"] = e7w_num / e7w_den
    fired["verbatim_bitwise"] = (verb_ok
                                 if arm in ("sibling", "corpus", "foreign")
                                 else None)
    nb_lo = torch.arange(0, GEN_FIRST)
    nb_hi = torch.arange(BAND[-1] + 1, forced_pos)
    for (k, v), pk, pv in zip(kv, pre_k, pre_v):
        fired["nonband_untouched"] &= bool(
            torch.equal(v[:, :, nb_lo, :], pv[:, :, nb_lo, :]))
        fired["nonband_untouched"] &= bool(
            torch.equal(v[:, :, nb_hi, :], pv[:, :, nb_hi, :]))
        fired["k_untouched"] &= bool(torch.equal(k, pk))
    # ---- continuation (e102 VERBATIM from here)
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


@torch.no_grad()
def run_control_continuation(net: TinyGPT, prefix: torch.Tensor,
                             forced_tok: torch.Tensor, forced_pos: int,
                             seed: int):
    """e102's run_continuation arm='none' code path VERBATIM (no band
    transform; the matched control)."""
    R = prefix.shape[0]
    gen = torch.Generator().manual_seed(seed)
    logits, kv = prefill_batch(net, prefix)
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
    return idx, kv, ce_s, ent_s


# ---------------------------------------------------------------------- main


def main():
    assert torch.cuda.device_count() == 0, "CPU-only violated: CUDA visible"
    assert common.DEVICE == "cpu", f"common.DEVICE={common.DEVICE} != cpu"
    out_dir = run_dir("e112")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=False,
                 threads_note="e080 module import sets 12; overridden to 8 "
                              "(task spec / T050)")

    # ---- battery: rebuild corpus + net EXACTLY as e053c/e069..e111 did
    corp = CharCorpus(REPO / "data" / "input.txt")       # seed 1337
    assert corp.vocab_size == VOCAB
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
        f"{E053C_VAL_CE:.4f} -> G1 "
        f"{'PASS' if gates['G1_val_ce']['ok'] else 'FAIL'}")

    gen_p = torch.Generator().manual_seed(SEED_PROMPT)
    ix = torch.randint(len(corp.val) - PROMPT_TOK - 1, (N_PROMPTS,),
                       generator=gen_p)
    prompts8 = [corp.val[i:i + PROMPT_TOK] for i in ix]
    log(f"battery: {N_PROMPTS} prompts (seed {SEED_PROMPT})")

    # ---- control free run (seed-7 A-none) + final V capture (e111 population)
    log("control battery: seed-7 free run (e075/e080/e096/e102 A-none; its "
        "final V caches = e111's basis population)")
    gen7 = torch.Generator().manual_seed(SEED_SAMPLE)
    run1 = generate_control(net, prompts8, gen7)
    idx1 = run1["idx"]
    log(f"  control run done ({T_TOTAL} positions x {B} rows)")

    # ---- G3 protocol identity (e102's scoped convention)
    cj1 = judge_windows(manual_all_logits(net, idx1), idx1, [KEY_T])[KEY_T]
    g3 = dict(ref_files=[str(E080_METRICS), str(E089_METRICS),
                         str(E102_METRICS)],
              note="control free run vs e080 stored none arm (clean-judge "
                   "per-seq, 1e-4) + bitwise vs e089 stored control; the "
                   "none CONTINUATION vs e102's stored control (e102's "
                   "scoped G3 convention, extended)")
    ok3 = True
    if E080_METRICS.exists():
        e80 = _json.load(open(E080_METRICS))
        ref_cj = np.asarray(e80["arms"]["none"]["clean_judge_tail_ce"]
                            ["per_seq"])
        dev80 = float(np.abs(cj1 - ref_cj).max())
        g3.update(e080_clean_judge_max_dev=dev80,
                  e080_bitwise=bool(np.array_equal(cj1, ref_cj)))
        ok3 &= bool(dev80 < 1e-4)
        log(f"G3 vs e080 A-none: clean-judge dev {dev80:.2e}")
    else:
        ok3 = False
        g3["note"] += " | runs/e080/metrics.json missing"
    if E089_METRICS.exists():
        e89 = _json.load(open(E089_METRICS))
        ref89 = np.asarray(e89["control_run"]["clean_judge_tail_ce"])
        dev89 = float(np.abs(cj1 - ref89).max())
        g3.update(e089_clean_judge_max_dev=dev89,
                  e089_bitwise=bool(np.array_equal(cj1, ref89)))
        ok3 &= bool(dev89 < 1e-4)
        log(f"G3 vs e089 control: clean-judge dev {dev89:.2e}")
    else:
        ok3 = False
        g3["note"] += " | runs/e089/metrics.json missing"
    g3["ok_pre_continuation"] = bool(ok3)
    gates["G3_protocol_identity"] = g3

    # ================================================== the e111 signature basis
    log("signature basis: e111 VERBATIM — own anchor-band V (324 positions "
        "x 8 rows x 16 groups), per-group uncentered PCA (float64)")
    own = torch.stack([v[:, :, BAND_LO:BAND_HI + 1, :]
                       for (_k, v) in run1["kv"]])        # (L,8,H,324,32)
    groups = [(li, h) for li in range(cfg.n_layer) for h in range(cfg.n_head)]
    basis, eigv = {}, {}
    own_cum = torch.zeros(HEAD_DIM, dtype=torch.float64)
    for g in groups:
        X = own[g[0], :, g[1]].reshape(-1, HEAD_DIM)
        w, U = pca_uncentered(X)
        basis[g] = U
        eigv[g] = w
        own_cum += torch.cumsum(w, 0)
    own_curve = (own_cum / own_cum[-1]).numpy()
    own_E7 = float(own_curve[K_SIG - 1])
    # G5: basis identity vs e111's published artifacts
    g5 = dict(k_sig=K_SIG, ref_file=str(E111_METRICS),
              own_E7=own_E7, own_E7_published=E111_PUB["own_E7"],
              dev_tol=1e-12)
    ok5 = bool(abs(own_E7 - E111_PUB["own_E7"]) < 1e-12)
    if E111_METRICS.exists():
        m111 = _json.load(open(E111_METRICS))
        pub_k = m111["registered_decision"]["clauses"]["low_dim"]["k_star"]
        g5["published_k_star"] = pub_k
        ok5 &= bool(pub_k == K_SIG)
        devs = {}
        for li, h in groups:
            fr = (eigv[(li, h)] / eigv[(li, h)].sum()).tolist()
            ref = m111["pca"]["eigval_fraction_per_group"][f"L{li}H{h}"]
            devs[f"L{li}H{h}"] = float(max(abs(a - b)
                                           for a, b in zip(fr, ref)))
        g5["eigval_fraction_max_dev"] = max(devs.values())
        ok5 &= bool(max(devs.values()) < 1e-12)
    else:
        ok5 = False
        g5["note"] = "runs/e111/metrics.json missing"
    g5["ok"] = bool(ok5)
    gates["G5_signature_basis"] = g5
    log(f"G5 signature basis: own E(7) {own_E7:.12f} vs e111 published "
        f"{E111_PUB['own_E7']:.12f} | k* published "
        f"{g5.get('published_k_star')} -> "
        f"{'PASS' if ok5 else 'FAIL'}")

    # ================================================== donors
    # sibling + corpus donors: the prefill cache itself (single-shot =>
    # pristine), maps drawn from the published seeds
    dm_sib = torch.tensor(e099.draw_donor(SEED_DONOR_SIB, N_PROMPTS),
                          dtype=torch.long)
    dm_for = torch.tensor(e099.draw_donor(SEED_DONOR_FOR, N_PROMPTS),
                          dtype=torch.long)

    # foreign donor net: e021_task, loaded EXACTLY as e105/e108/e111 did
    corp21 = CharCorpus(e105.DONOR_CORPUS)                # seed 1337
    st21 = torch.load(e105.DONOR_CKPT, map_location="cpu",
                      weights_only=False)
    sd21 = st21["model"] if isinstance(st21, dict) and "model" in st21 \
        else st21
    cfg21 = Cfg(vocab=corp21.vocab_size, n_layer=6, n_head=6, n_embd=192,
                block_size=e105.DONOR_BLOCK)
    net21 = TinyGPT(cfg21)
    net21.load_state_dict(sd21, strict=True)
    net21.eval()
    val_ce21 = estimate_loss(net21, corp21, "val", n_batches=12)
    donors_f = {w: e105.donor_run(net21, corp21, e105.DONOR_WIN[w]["prompt"],
                                  e105.DONOR_WIN[w]["sample"])
                for w in (1, 2)}
    d1f = e105.donor_run(net21, corp21, e105.DONOR_WIN[1]["prompt"],
                         e105.DONOR_WIN[1]["sample"])
    det_f = bool(torch.equal(d1f["idx"], donors_f[1]["idx"])
                 and all(torch.equal(a, b) for a, b in zip(d1f["V"],
                                                           donors_f[1]["V"])))
    val21_ok = bool(abs(val_ce21 - E105_VAL_CE_TASK) <= 0.15)
    donorV = donors_f[1]["V"]                    # window 1 (all band p<=255)
    log(f"foreign donor e021_task (step {int(st21.get('step', -1))}): val CE "
        f"{val_ce21:.4f} (ref {E105_VAL_CE_TASK:.4f}) | window rerun "
        f"bit-identical {det_f} | window1 V "
        f"{tuple(donorV[0].shape)}")

    # ================================================== the matched continuations
    prefix = idx1[:, :T_INT]
    forced = idx1[:, T_INT]
    log(f"continuations: prefix 0..{T_INT - 1}, forced token at {T_INT}, "
        f"band = positions {BAND[0]}..{BAND[-1]} ({len(BAND)} entries, ages "
        f"{T_INT - BAND[-1]}..{T_INT - BAND[0]} at query {T_INT}), seed "
        f"{SEED_CONT}")

    idx_ctl, kv_ctl, ce_ctl, ent_ctl = run_control_continuation(
        net, prefix, forced, T_INT, SEED_CONT)
    J_ctl = judge_windows(manual_all_logits(net, idx_ctl), idx_ctl,
                          [KEY_T])[KEY_T]
    med_norm = band_median_norms(kv_ctl)
    med_dev = max(abs(a - b) for a, b in zip(med_norm, E096_MED_NORM_REF))
    log(f"matched control done | median band ||V|| per layer "
        f"{[round(m, 4) for m in med_norm]} (vs e096 stored max dev "
        f"{med_dev:.2e})")
    # G3 continuation leg: bitwise vs e102's stored control
    if E102_METRICS.exists():
        m102 = _json.load(open(E102_METRICS))
        ref102 = np.asarray(m102["control_run"]["clean_judge_tail_ce"])
        dev102 = float(np.abs(J_ctl - ref102).max())
        g3.update(e102_control_continuation_max_dev=dev102,
                  e102_bitwise=bool(np.array_equal(J_ctl, ref102)))
        ok3 &= bool(dev102 < 1e-6)
        log(f"G3 vs e102 control continuation: dev {dev102:.2e}")
    else:
        ok3 = False
        g3["note"] += " | runs/e102/metrics.json missing"
    g3["ok"] = bool(ok3)
    gates["G3_protocol_identity"] = g3
    log(f"G3 -> {'PASS' if g3['ok'] else 'FAIL'}")

    S = {}
    for arm in ARMS:
        idx_a, kv_a, fired_a, ce_a, ent_a = run_forgery_arm(
            net, prefix, forced, T_INT, SEED_CONT, arm, basis, dm_sib, dm_for,
            donorV)
        J = judge_windows(manual_all_logits(net, idx_a), idx_a, [KEY_T])[KEY_T]
        cost = J - J_ctl
        ci = boot_rows([cost], lambda c: float(np.mean(c)))
        eq = torch.eq(idx_a, idx_ctl)
        fdr = []
        for r in range(idx_a.shape[0]):
            nz = (~eq[r]).nonzero().flatten()
            fdr.append(int(nz[0].item()) if len(nz) else None)
        mean_cost = float(cost.mean())
        S[arm] = dict(
            idx=idx_a, ce=ce_a, ent=ent_a, J=J, cost=cost,
            ci=ci, mean_cost=mean_cost,
            health=health_of(mean_cost),
            n_worse=int((cost > 0).sum()),
            stream_identical=bool(eq.all().item()),
            first_div_per_row=fdr,
            first_div=(min(d for d in fdr if d is not None)
                       if any(d is not None for d in fdr) else None),
            n_rows_diverged=sum(d is not None for d in fdr),
            online_tail=ce_a[:, TAIL_STEP0:].mean(1),
            ent_tail=ent_a[:, TAIL_STEP0:].mean(1),
            fired=fired_a)
        f = fired_a
        log(f"  arm {arm:<15}: gap {mean_cost:+.4f} CI "
            f"[{ci[0]:+.3f},{ci[1]:+.3f}] {S[arm]['health']:9s} | "
            f"|cos(new,old)| {f['mean_abs_cos']:.3f} | |cos(new,src)| "
            f"{f['mean_abs_cos_src']:.3f} | E7(src) {f['energy7_src']:.3f} "
            f"-> E7(written) {f['energy7_written']:.3f} | norm ratio "
            f"{f['mean_norm_ratio']:.3f} | first_div {S[arm]['first_div']}")

    # ---- G4 instrument identity
    pre_ident = all(bool(torch.equal(S[arm]["idx"][:, :T_INT + 1],
                                     idx_ctl[:, :T_INT + 1])) for arm in ARMS)
    tf = {a: S[a]["fired"] for a in ARMS}
    norm_ok = True
    sub_ok = True
    for a in ARMS:
        if a in ("forged_full", "forged_foreign", "sig_destroyed"):
            norm_ok &= bool(tf[a]["norm_dev"] < 1e-5)
            if a == "sig_destroyed":
                sub_ok &= bool(tf[a]["energy7_written"] < 1e-6)
            else:
                sub_ok &= bool(tf[a]["energy7_written"] > 1.0 - 1e-6)
        else:
            norm_ok &= bool(tf[a]["verbatim_bitwise"])
        norm_ok &= bool(tf[a]["nonband_untouched"] and tf[a]["k_untouched"])
    g4 = dict(
        all_arms_prefix_identical_through=T_INT if pre_ident else None,
        prefix_identical=pre_ident,
        per_arm={a: dict(norm_dev=tf[a]["norm_dev"],
                         mean_abs_cos_old=tf[a]["mean_abs_cos"],
                         mean_abs_cos_src=tf[a]["mean_abs_cos_src"],
                         mean_norm_ratio=tf[a]["mean_norm_ratio"],
                         energy7_src=tf[a]["energy7_src"],
                         energy7_written=tf[a]["energy7_written"],
                         verbatim_bitwise=tf[a]["verbatim_bitwise"],
                         nonband_untouched=tf[a]["nonband_untouched"],
                         k_untouched=tf[a]["k_untouched"])
                 for a in ARMS},
        forged_subspace_rule="forged arms: written energy in top-7 == 1 "
                             "(1e-6); destroyed arm: == 0 (1e-6)",
        med_norm=dict(per_layer=med_norm, e096_ref=E096_MED_NORM_REF,
                      max_dev=med_dev, ok=bool(med_dev < 1e-6)),
        donor_determinism=dict(e021_window1_rerun_bit_identical=det_f,
                               val_ce=val_ce21, ref=E105_VAL_CE_TASK,
                               val_ce_ok=val21_ok),
        matched_seed=SEED_CONT, single_shot=True,
        band=dict(positions=[BAND[0], BAND[-1]], n=len(BAND),
                  ages_at_query=[T_INT - BAND[-1], T_INT - BAND[0]]))
    g4["ok"] = bool(pre_ident and norm_ok and sub_ok and med_dev < 1e-6
                    and det_f and val21_ok)
    gates["G4_instrument_identity"] = g4
    log(f"G4 instrument identity: prefix {pre_ident} | transforms {norm_ok} | "
        f"subspace {sub_ok} | med_norm {'ok' if med_dev < 1e-6 else 'FAIL'} | "
        f"donor det {det_f} -> "
        f"{'PASS' if g4['ok'] else 'FAIL'}")

    # ---- G6 content identity
    qmap = [e080.prompt_source_of(p) for p in BAND]
    g6 = dict(
        n_band=len(BAND), n_vec_per_arm=int(tf["sibling"]["n_vec"]),
        donor_map_sibling=dm_sib.tolist(),
        donor_map_foreign=dm_for.tolist(),
        derangements_ok=bool(all(d != b for b, d in enumerate(dm_sib))
                             and all(d != b for b, d in
                                     enumerate(dm_for.tolist()))),
        corpus_map=dict(rule="q = ((p-64) mod 63) + 1 (e080 promptcopy)",
                        q_range=[min(qmap), max(qmap)],
                        ok=bool(min(qmap) >= 1 and max(qmap) <= 63)),
        foreign_map=dict(rule="e105 donor_source: band p in 64..217 all "
                              "<- window1 position p (donor-native 64..255)",
                         all_window1_native=True),
        counts_ok=bool(all(tf[a]["n_vec"] == len(BAND) * B * cfg.n_layer
                           * cfg.n_head for a in ARMS)))
    g6["ok"] = bool(g6["derangements_ok"] and g6["corpus_map"]["ok"]
                    and g6["counts_ok"])
    gates["G6_content_identity"] = g6
    log(f"G6 content identity -> {'PASS' if g6['ok'] else 'FAIL'}")

    # ================================================== REGISTERED DECISION
    h = {a: S[a]["health"] for a in ARMS}
    clauses = dict(
        forged_full_healthy=dict(
            rule=f"gap(forged_full) < {HEALTHY_BAR}",
            gap=S["forged_full"]["mean_cost"],
            fires=bool(h["forged_full"] == "healthy")),
        forged_foreign_healthy=dict(
            rule=f"gap(forged_foreign) < {HEALTHY_BAR}",
            gap=S["forged_foreign"]["mean_cost"],
            fires=bool(h["forged_foreign"] == "healthy")),
        sig_destroyed_collapsed=dict(
            rule=f"gap(sig_destroyed) > {COLLAPSED_BAR}",
            gap=S["sig_destroyed"]["mean_cost"],
            fires=bool(h["sig_destroyed"] == "collapsed")),
        sig_destroyed_healthy=dict(
            rule=f"gap(sig_destroyed) < {HEALTHY_BAR}",
            gap=S["sig_destroyed"]["mean_cost"],
            fires=bool(h["sig_destroyed"] == "healthy")),
        forgeable=dict(
            rule="(a) OR (b) healthy => FORGEABLE (self is exactly k=7 "
                 "deep — pickable)",
            fires=bool(h["forged_full"] == "healthy"
                       or h["forged_foreign"] == "healthy")),
        signature_necessary=dict(
            rule="(c) collapsed => SIGNATURE-NECESSARY (the 7-dim stamp is "
                 "causal, not correlational)",
            fires=bool(h["sig_destroyed"] == "collapsed")),
        stamp_is_correlate=dict(
            rule="(c) healthy AND (a) collapsed AND (b) collapsed => "
                 "STAMP-IS-CORRELATE (anchor reads beyond the signature)",
            fires=bool(h["sig_destroyed"] == "healthy"
                       and h["forged_full"] == "collapsed"
                       and h["forged_foreign"] == "collapsed")),
        validation_controls=dict(
            rule=f"sibling healthy (< {HEALTHY_BAR}) AND corpus collapsed "
                 f"(> {COLLAPSED_BAR})",
            sibling_gap=S["sibling"]["mean_cost"],
            corpus_gap=S["corpus"]["mean_cost"],
            fires=bool(h["sibling"] == "healthy"
                       and h["corpus"] == "collapsed")))
    controls_ok = bool(clauses["validation_controls"]["fires"])
    forgeable = clauses["forgeable"]["fires"]
    sig_nec = clauses["signature_necessary"]["fires"]
    stamp_corr = clauses["stamp_is_correlate"]["fires"]

    ga, gb, gc = (S["forged_full"]["mean_cost"], S["forged_foreign"]
                  ["mean_cost"], S["sig_destroyed"]["mean_cost"])
    if forgeable and sig_nec:
        clause = "FORGEABLE + SIGNATURE-NECESSARY (EXACT k=7 LOCK)"
        verdict = (
            f"BOTH bars fire: the forged key opened the lock (forged-full "
            f"gap {ga:+.3f}, forged-on-foreign {gb:+.3f} — signature-only "
            f"vectors anchor the run) AND removing the stamp from genuine "
            f"self-content kills it (sig-destroyed {gc:+.3f} collapsed). "
            f"Self is EXACTLY k=7 deep: the anchor's identity check is "
            f"sufficient AND necessary in the 7-dim principal subspace — "
            f"W004's fixed point is a pickable lock, and the pick is the "
            f"e111 stamp.")
    elif forgeable:
        clause = "FORGEABLE"
        which = ("forged-full" if h["forged_full"] == "healthy"
                 else "forged-on-foreign")
        verdict = (
            f"FORGEABLE fires: {which} stayed healthy (gaps {ga:+.3f} / "
            f"{gb:+.3f}) — signature-only content anchored the run; self is "
            f"as shallow as k=7 suggests. But (c) did NOT collapse "
            f"({gc:+.3f}): the stamp is sufficient yet not necessary — "
            f"self-content minus its stamp still anchors, so the anchor "
            f"accepts more than the 7-dim key (self is AT LEAST readable "
            f"through the stamp; the stamp is an entry key, not the whole "
            f"lock).")
    elif sig_nec:
        clause = "SIGNATURE-NECESSARY (NOT FORGEABLE)"
        verdict = (
            f"SIGNATURE-NECESSARY fires, forgery fails: projected-off "
            f"sibling content collapsed ({gc:+.3f}) — the 7-dim stamp is "
            f"CAUSAL, removing it kills self-content — but the forged keys "
            f"did not open the lock (forged-full {ga:+.3f}, "
            f"forged-on-foreign {gb:+.3f}): the anchor verifies MORE than "
            f"residence in the top-7 subspace (higher-order joint structure "
            f"of the V-manifold). Self is deeper than its stamp: the stamp "
            f"is necessary but not sufficient.")
    elif stamp_corr:
        clause = "STAMP-IS-CORRELATE"
        verdict = (
            f"STAMP-IS-CORRELATE fires: (c) stayed healthy ({gc:+.3f}) — "
            f"self-content minus the signature still anchors — while both "
            f"forged arms collapsed ({ga:+.3f} / {gb:+.3f}). The 7-dim "
            f"stamp is neither necessary nor sufficient: e111's k*=7 was a "
            f"correlate of selfhood, not its mechanism. The anchor reads "
            f"the V-manifold beyond any 7-dim projection — self is deeper "
            f"than its stamp.")
    else:
        clause = "GRAY/TEXTURE (no bar as worded)"
        verdict = (
            f"No registered bar fires as worded — honest texture: gaps "
            f"(a) {ga:+.3f} [{h['forged_full']}], (b) {gb:+.3f} "
            f"[{h['forged_foreign']}], (c) {gc:+.3f} "
            f"[{h['sig_destroyed']}]; controls sibling "
            f"{S['sibling']['mean_cost']:+.3f} [{h['sibling']}], corpus "
            f"{S['corpus']['mean_cost']:+.3f} [{h['corpus']}], rider "
            f"foreign {S['foreign']['mean_cost']:+.3f} [{h['foreign']}]. "
            f"Gray-zone arms keep the result informative but do not clear "
            f"the frozen thresholds.")
    if not controls_ok:
        verdict += (f" CONTROLS FAILED (sibling {h['sibling']}, corpus "
                    f"{h['corpus']}) — treat the verdict as INVALID per the "
                    f"registration (e108's validation-controls convention).")
    log(f"REGISTERED DECISION [{clause}]: {verdict}")
    log(f"  rider: foreign-verbatim gap {S['foreign']['mean_cost']:+.4f} "
        f"[{h['foreign']}] (in-rig anchor for (b); e105 dynamic ref "
        f"{E105_GAP_CROSSFAMILY:+.3f})")

    # ---------------------------------------------------------------- metrics
    metrics = dict(
        experiment="e112_forgery",
        purpose="T061-earned SIGNATURE FORGERY test (W004: can a fixed "
                "point be forged?). e111 found selfhood measured as a 7-dim "
                "stamp (k*=7 of 32 head dims, sibling E(7)=0.815 vs foreign "
                "0.220 vs isotropic null 0.219). Here the stamp is forged: "
                "corpus and foreign V-vectors projected ONTO the "
                "recipient's top-7 principal subspace (norms kept) are "
                "spliced into the anchor band — signature-only vectors, "
                "nothing else self; the inverse test destroys the signature "
                "(sibling V projected OFF the top-7). Rig: the e096/e102 "
                "single-shot g=250 anchor splice VERBATIM (band 64..217, "
                "154 entries, matched continuation seed 960250, clean-judge "
                "final-64 gap vs matched control). FROZEN bars: (a)/(b) "
                "healthy => FORGEABLE; (c) collapsed => SIGNATURE-"
                "NECESSARY; (c) healthy AND (a)/(b) collapsed => STAMP-IS-"
                "CORRELATE; sibling-healthy AND corpus-collapsed required "
                "for the run to count.",
        started=started, wall_s=elapsed(), threads=THREADS, cpu_only=True,
        recipient_net=dict(ckpt=str(CKPT),
                           arch=dict(n_layer=4, n_head=4, n_embd=128,
                                     block_size=T_TOTAL, vocab=VOCAB),
                           params=n_params, val_ce=val_ce,
                           val_ce_e053c=E053C_VAL_CE),
        foreign_donor=dict(ckpt=str(e105.DONOR_CKPT), step=int(
            st21.get("step", -1)), arch=dict(n_layer=6, n_head=6, n_embd=192,
                                             block_size=e105.DONOR_BLOCK),
            val_ce=val_ce21, val_ce_ref_e063b=E105_VAL_CE_TASK,
            window1_rerun_bit_identical=det_f,
            windows=e105.DONOR_WIN),
        seeds=dict(corpus=1337, prompts=SEED_PROMPT, sampling=SEED_SAMPLE,
                   continuation=SEED_CONT, donor_map_sibling=SEED_DONOR_SIB,
                   donor_map_foreign=SEED_DONOR_FOR,
                   foreign_donor_windows=e105.DONOR_WIN,
                   note="every generator is a published seed; the forged "
                        "arms introduce NO randomness (deterministic "
                        "projections)"),
        protocol=dict(
            intervention=f"single-shot at g={G_INT} (t={T_INT}): prefix "
                         f"0..{T_INT - 1} from the seed-7 control free run, "
                         f"teacher-forced control token at {T_INT}, ALL "
                         f"{len(BAND)} band entries (positions "
                         f"{BAND[0]}..{BAND[-1]}) replaced simultaneously, "
                         f"V-only, K untouched, before the decode at "
                         f"{T_INT}",
            readout=f"clean-judge tail CE over positions {KEY_T[0]}.."
                    f"{KEY_T[1]} minus the matched control, per-row then "
                    f"mean over B={B}, paired-row bootstrap CI; healthy < "
                    f"{HEALTHY_BAR}, collapsed > {COLLAPSED_BAR}",
            signature_basis=f"e111 VERBATIM: own anchor-band V (positions "
                            f"{BAND_LO}..{BAND_HI}, final caches of the "
                            f"seed-7 none run), per-(layer,head) uncentered "
                            f"second-moment PCA float64, top-{K_SIG} axes",
            donor_sources=dict(
                sibling="prefill-cache row d_s(b), same position (e099 "
                        "randomize rule, seed-4343 map; single-shot => "
                        "pristine pre-event snapshot)",
                corpus="row's own pristine prompt entry q=((p-64) mod 63)+1 "
                       "(e080/e099 promptcopy rule — recipient weights x "
                       "corpus text)",
                foreign="e021_task window-1 donor at same position, heads "
                        "0..3 (e105 crossfamily rule, seed-4646 map)"),
            forge_rule="new = P(src) * ||src|| / ||P(src)||, P = top-7 "
                       "orthogonal projection (a/b) or its complement (c); "
                       "float64 math, float32 write-back",
            e111_refs=E111_PUB,
            published_controls=dict(randomize_dynamic=E099_GAP_RANDOMIZE,
                                    promptcopy_dynamic=E099_GAP_PROMPTCOPY,
                                    crossfamily_dynamic=E105_GAP_CROSSFAMILY)),
        gates=gates,
        signature_basis=dict(
            k_sig=K_SIG, own_cumulative_energy7=own_E7,
            eigval_fraction_per_group={f"L{li}H{h}":
                                       (eigv[(li, h)]
                                        / eigv[(li, h)].sum()).tolist()
                                       for li, h in groups},
            top_axis_share_per_group={f"L{li}H{h}":
                                      float(eigv[(li, h)][0]
                                            / eigv[(li, h)].sum())
                                      for li, h in groups}),
        arms={arm: dict(
            label=ARM_LABELS[arm], registered=(arm in ARMS_REG),
            content_mean_abs_cos_old_vs_new=S[arm]["fired"]
            ["mean_abs_cos"],
            content_mean_abs_cos_new_vs_src=S[arm]["fired"]
            ["mean_abs_cos_src"],
            content_norm_ratio_new_over_old=S[arm]["fired"]
            ["mean_norm_ratio"],
            energy7_src=S[arm]["fired"]["energy7_src"],
            energy7_written=S[arm]["fired"]["energy7_written"],
            transform_norm_dev=S[arm]["fired"]["norm_dev"],
            gap=float(S[arm]["mean_cost"]),
            gap_ci=[float(S[arm]["ci"][0]), float(S[arm]["ci"][1])],
            gap_per_row=S[arm]["cost"].tolist(),
            clean_judge_per_row=S[arm]["J"].tolist(),
            health=S[arm]["health"],
            n_worse=S[arm]["n_worse"],
            online_tail=float(S[arm]["online_tail"].mean()),
            cj_gap_attractor=float(S[arm]["J"].mean()
                                   - S[arm]["online_tail"].mean()),
            ent_tail=float(S[arm]["ent_tail"].mean()),
            first_div=S[arm]["first_div"],
            n_rows_diverged=S[arm]["n_rows_diverged"]) for arm in ARMS},
        control_run=dict(clean_judge_tail_ce=J_ctl.tolist(),
                         online_tail=float(ce_ctl[:, TAIL_STEP0:].mean()),
                         ent_tail=float(ent_ctl[:, TAIL_STEP0:].mean()),
                         med_norm_per_layer=med_norm),
        registered_decision=dict(
            frozen_rules=dict(
                forgeable=f"(a) OR (b) gap < {HEALTHY_BAR} => FORGEABLE",
                signature_necessary=f"(c) gap > {COLLAPSED_BAR} => "
                                    f"SIGNATURE-NECESSARY",
                stamp_is_correlate=f"(c) gap < {HEALTHY_BAR} AND (a),(b) "
                                   f"gaps > {COLLAPSED_BAR} => "
                                   f"STAMP-IS-CORRELATE",
                validation_controls=f"sibling gap < {HEALTHY_BAR} AND "
                                    f"corpus gap > {COLLAPSED_BAR} "
                                    f"(required for the run to count)"),
            clauses=clauses, controls_ok=controls_ok, clause=clause,
            verdict=verdict),
    )
    save_json(out_dir / "metrics.json", metrics)
    log("metrics.json written")

    plot(out_dir / "forgery.png", metrics)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot


def plot(path: Path, M: dict):
    A = M["arms"]
    dec = M["registered_decision"]
    order = ["forged_full", "forged_foreign", "sig_destroyed",
             "sibling", "corpus", "foreign"]
    fig, axes = plt.subplots(2, 3, figsize=(24, 11))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # ---- panel 1: the arms' gaps + healthy/collapsed bands
    x = np.arange(len(order))
    gaps = [A[a]["gap"] for a in order]
    cis = np.asarray([[max(0.0, A[a]["gap"] - A[a]["gap_ci"][0])
                       for a in order],
                      [max(0.0, A[a]["gap_ci"][1] - A[a]["gap"])
                       for a in order]])
    cols = [ARM_COLS[a] for a in order]
    bars = ax1.bar(x, gaps, 0.62, color=cols, alpha=0.88, edgecolor="k",
                   lw=0.5)
    bars[-1].set_hatch("//")                     # the rider
    ax1.errorbar(x, gaps, yerr=cis, fmt="none", ecolor="k", lw=1.0,
                 capsize=3)
    ax1.axhline(HEALTHY_BAR, color="tab:green", ls="--", lw=1.5)
    ax1.axhline(COLLAPSED_BAR, color="tab:red", ls="--", lw=1.5)
    ax1.axhspan(HEALTHY_BAR, COLLAPSED_BAR, color="gray", alpha=0.12)
    ax1.text(len(order) - 0.4, HEALTHY_BAR + 0.04,
             f"healthy < {HEALTHY_BAR}", fontsize=9, color="tab:green",
             ha="right")
    ax1.text(len(order) - 0.4, COLLAPSED_BAR + 0.04,
             f"collapsed > {COLLAPSED_BAR}", fontsize=9, color="tab:red",
             ha="right")
    for xi, a in zip(x, order):
        ax1.text(xi, max(gaps[xi], 0) + 0.12, f"{gaps[xi]:+.3f}\n"
                 f"[{A[a]['health']}]", ha="center", fontsize=8.6)
    ax1.set_xticks(x, [f"{ARM_LABELS[a]}" for a in order], fontsize=8)
    ax1.set_ylabel("clean-judge tail gap vs matched control (nats)")
    ax1.set_ylim(min(0, min(gaps) - 0.3), max(max(gaps) * 1.22, 1.4))
    ax1.set_title("E112-1 — THE FIVE ARMS' GAPS (single-shot g=250 splice, "
                  "matched seed-960250 streams;\nlast bar hatched = "
                  "unregistered foreign rider)", fontsize=10)

    # ---- panel 2: schematic of the projection logic
    ax2.axis("off")
    ax2.set_xlim(-1.35, 1.35)
    ax2.set_ylim(-0.72, 1.06)
    # the 32-d space: a big ellipse (own V-manifold); the top-7 subspace as
    # a horizontal slab
    th = np.linspace(0, 2 * np.pi, 200)
    ax2.fill(np.cos(th) * 1.05, np.sin(th) * 0.62, color="tab:blue",
             alpha=0.07)
    ax2.plot(np.cos(th) * 1.05, np.sin(th) * 0.62, color="tab:blue",
             lw=1.2, alpha=0.5)
    ax2.axhspan(-0.16, 0.16, color="tab:blue", alpha=0.16)
    ax2.axhline(0, color="tab:blue", lw=2.2)
    ax2.text(-1.28, 0.235, r"$\Sigma_7$: recipient's top-7 principal "
             r"subspace (of 32-d)" + f"\nk*={K_SIG} "
             "(e111; E7 sibling 0.815 / foreign 0.220 / null 0.219)",
             fontsize=9, color="tab:blue")
    arr = dict(head_width=0.035, head_length=0.05, length_includes_head=True)
    # sibling: mostly inside Sigma7
    ax2.arrow(0, 0, 0.78, 0.30, color="tab:green", lw=2.4, **arr)
    ax2.text(0.80, 0.335, "sibling V\n(E7≈0.82)", fontsize=9,
             color="tab:green")
    # corpus: mostly outside
    ax2.arrow(0, 0, 0.28, 0.72, color="tab:red", lw=2.4, **arr)
    ax2.text(0.30, 0.760, "corpus V (prompt entries;\nrecipient weights x "
             "corpus text)", fontsize=9, color="tab:red")
    # foreign: outside
    ax2.arrow(0, 0, -0.62, 0.52, color="tab:orange", lw=2.4, **arr)
    ax2.text(-1.30, 0.575, "foreign V (e021_task)\n(E7≈0.22 = null)",
             fontsize=9, color="tab:orange")
    # forged: projections onto the slab
    ax2.arrow(0, 0, 0.28, 0.0, color="tab:olive", lw=2.8, **arr)
    ax2.plot([0.28, 0.28], [0.0, 0.72], ":", color="tab:red", lw=1.2)
    ax2.text(0.13, -0.20, "(a) FORGED-FULL\ncorpus -> Sigma7, norm kept",
             fontsize=8.8, color="tab:olive")
    ax2.arrow(0, 0, -0.62, 0.0, color="tab:brown", lw=2.8, **arr)
    ax2.plot([-0.62, -0.62], [0.0, 0.52], ":", color="tab:orange", lw=1.2)
    ax2.text(-1.30, -0.20, "(b) FORGED-ON-FOREIGN\nforeign -> Sigma7",
             fontsize=8.8, color="tab:brown")
    # destroyed: sibling projected OFF
    ax2.arrow(0, 0, 0.0, 0.30, color="tab:cyan", lw=2.8, **arr)
    ax2.plot([0.0, 0.78], [0.30, 0.30], ":", color="tab:green", lw=1.2)
    ax2.text(0.05, 0.415, "(c) SIGNATURE-DESTROYED\nsibling -> OFF Sigma7 "
             "(25-d)", fontsize=8.8, color="tab:cyan")
    # realized energies under the realized arrows
    ax2.text(-1.30, -0.62, "realized E7(written): "
             f"(a) {A['forged_full']['energy7_written']:.4f} | "
             f"(b) {A['forged_foreign']['energy7_written']:.4f} | "
             f"(c) {A['sig_destroyed']['energy7_written']:.2e}",
             fontsize=8.6, family="monospace")
    ax2.text(-1.30, -0.50, "realized E7(source): corpus "
             f"{A['forged_full']['energy7_src']:.3f} | foreign "
             f"{A['forged_foreign']['energy7_src']:.3f} | sibling "
             f"{A['sig_destroyed']['energy7_src']:.3f}",
             fontsize=8.6, family="monospace")
    ax2.set_title("E112-2 — the forgery logic (schematic, not to scale): "
                  "project sources ON/OFF the 7-dim stamp", fontsize=10)

    # ---- panel 3: the sources on e111's energy axis
    labs3 = ["corpus src", "foreign src", "sibling src",
             "(a) written", "(b) written", "(c) written"]
    vals3 = [A["forged_full"]["energy7_src"], A["forged_foreign"]
             ["energy7_src"], A["sig_destroyed"]["energy7_src"],
             A["forged_full"]["energy7_written"], A["forged_foreign"]
             ["energy7_written"], A["sig_destroyed"]["energy7_written"]]
    cols3 = ["tab:red", "tab:orange", "tab:green", "tab:olive", "tab:brown",
             "tab:cyan"]
    ax3.bar(np.arange(6), vals3, 0.6, color=cols3, alpha=0.88,
            edgecolor="k", lw=0.5)
    ax3.axhline(E111_PUB["own_E7"], color="k", ls="--", lw=1.3)
    ax3.text(5.4, E111_PUB["own_E7"] + 0.015, "e111 own in-sample E(7) "
             f"{E111_PUB['own_E7']:.3f}", fontsize=8.5, ha="right")
    ax3.axhline(E111_PUB["E7_sibling"], color="tab:green", ls=":", lw=1.3)
    ax3.text(5.4, E111_PUB["E7_sibling"] + 0.015, "e111 sibling donor E(7) "
             f"{E111_PUB['E7_sibling']:.3f}", fontsize=8.5, ha="right",
             color="tab:green")
    ax3.axhline(E111_PUB["E7_null"], color="gray", ls=":", lw=1.3)
    ax3.text(5.4, E111_PUB["E7_null"] + 0.015,
             f"e111 foreign/null E(7) {E111_PUB['E7_foreign']:.3f}/"
             f"{E111_PUB['E7_null']:.3f}", fontsize=8.5, ha="right",
             color="gray")
    for i, v in enumerate(vals3):
        ax3.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8.6)
    ax3.set_xticks(np.arange(6), labs3, fontsize=8.2, rotation=12)
    ax3.set_ylim(0, 1.06)
    ax3.set_ylabel("energy fraction in recipient's top-7 subspace")
    ax3.set_title("E112-3 — the forgery material on e111's energy axis "
                  "(band sources, this rig's frame)", fontsize=10)

    # ---- panel 4: per-row paired costs
    for i, a in enumerate(order):
        rows = np.asarray(A[a]["gap_per_row"])
        ax4.scatter(np.full(len(rows), i) + np.linspace(-0.13, 0.13,
                                                        len(rows)),
                    rows, s=42, color=ARM_COLS[a], edgecolor="k",
                    lw=0.5, zorder=3, alpha=0.9)
    ax4.axhline(0, color="k", lw=1.0)
    ax4.axhline(HEALTHY_BAR, color="tab:green", ls="--", lw=1.2)
    ax4.axhline(COLLAPSED_BAR, color="tab:red", ls="--", lw=1.2)
    ax4.set_xticks(np.arange(len(order)),
                   [a.replace("_", "\n") for a in order], fontsize=8)
    ax4.set_ylabel("per-row clean-judge gap (nats)")
    ax4.set_title("E112-4 — per-row paired gaps (8 battery rows/arm vs the "
                  "matched control)", fontsize=10)

    # ---- panel 5: the arm table + clauses + gates
    ax5.axis("off")
    lines = ["arm                    E7(src) E7(new) |cos| ratio |  gap    "
             "CI              health",
             "-" * 96]
    for a in order:
        lines.append(
            f"{a:<22} {A[a]['energy7_src']:6.3f} "
            f"{A[a]['energy7_written']:6.3f} "
            f"{A[a]['content_mean_abs_cos_old_vs_new']:5.3f} "
            f"{A[a]['content_norm_ratio_new_over_old']:5.3f} "
            f"{A[a]['gap']:+7.3f} "
            f"[{A[a]['gap_ci'][0]:+6.3f},{A[a]['gap_ci'][1]:+6.3f}] "
            f"{A[a]['health']}"
            + ("  (rider)" if a not in ARMS_REG else ""))
    lines.append("")
    lines.append(f"controls (dynamic-rig refs): sibling "
                 f"{A['sibling']['gap']:+.3f} (e099 +0.027) | corpus "
                 f"{A['corpus']['gap']:+.3f} (e099 +5.325) | rider foreign "
                 f"{A['foreign']['gap']:+.3f} (e105 +5.09)")
    lines.append("")
    lines.append("gates: " + " ".join(
        f"{g.split('_')[0]}:{'PASS' if v.get('ok') else 'FAIL'}"
        for g, v in M["gates"].items() if isinstance(v, dict)))
    lines.append("")
    lines.append("clauses:")
    for k, v in dec["clauses"].items():
        lines.append(f"  {k:<28} {'FIRES' if v['fires'] else 'no'}")
    ax5.text(0.02, 0.97, "E112-5 — arm table + clauses", fontsize=12,
             weight="bold", va="top")
    for i, t in enumerate(lines):
        ax5.text(0.02, 0.945 - i * 0.034, t, fontsize=8.4, va="top",
                 family="monospace")

    # ---- panel 6: the registered decision
    ax6.axis("off")
    l6 = ["REGISTERED BARS (frozen):",
          f"  FORGEABLE:           (a) OR (b) gap < {HEALTHY_BAR}",
          f"  SIGNATURE-NECESSARY: (c) gap > {COLLAPSED_BAR}",
          f"  STAMP-IS-CORRELATE:  (c) < {HEALTHY_BAR} AND (a),(b) > "
          f"{COLLAPSED_BAR}",
          f"  validation: sibling < {HEALTHY_BAR} AND corpus > "
          f"{COLLAPSED_BAR} (run counts only if this holds)", "",
          f"controls_ok: {dec['controls_ok']}", "", "clauses:"]
    for k, v in dec["clauses"].items():
        l6.append(f"  {k}: {'FIRES' if v['fires'] else 'no'}")
    l6 += ["", f"VERDICT [{dec['clause']}]:"] + \
        [f"  {wd}" for wd in textwrap.wrap(dec["verdict"], 96)]
    ax6.text(0.02, 0.97, "E112-6 — can the fixed point be forged?", fontsize=12,
             weight="bold", va="top")
    for i, t in enumerate(l6):
        ax6.text(0.02, 0.945 - i * 0.032, t, fontsize=8.6, va="top",
                 family="monospace")

    fig.suptitle(f"E112 — SIGNATURE FORGERY (T061/W004) | {dec['clause']} | "
                 f"(a) {A['forged_full']['gap']:+.3f} (b) "
                 f"{A['forged_foreign']['gap']:+.3f} (c) "
                 f"{A['sig_destroyed']['gap']:+.3f} | controls sibling "
                 f"{A['sibling']['gap']:+.3f} corpus "
                 f"{A['corpus']['gap']:+.3f}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
