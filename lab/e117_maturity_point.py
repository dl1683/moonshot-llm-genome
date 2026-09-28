"""E117 — the SHARE-CONSTANT MATURITY POINT (T068-M2 / W007 discriminator).

CONTEXT (THINKING.md T068 M2 + W007, read first): the share law's FORM is
universal (r*(k)*k constant within every net tested) but its VALUE moved
31 -> 42 -> 54 across the three measured nets. The e098 ladder seeds were
under-trained by dispatch (~2000 steps, COMPLETED cosine) while e053c (the
54) stopped at 3133 steps of a 4000-total cosine under the 180s cap. W007's
derivation program (share* ~ noise/signal: the mature net's non-anchor
stream is LARGER, so the anchor needs more mass to hold the same post-LN
share) predicts the constant GROWS with training exposure at fixed
architecture. e117 is the single adjudicating data point: ONE fresh seed
(4309) at EXACTLY 3133 steps.

THE DISCRIMINATOR (this run):
  1. BASE: fresh 4L/4H/128d block-512 net (873,472 params), corpus seed
     1337, set_seed(4309), batch 32 x ctx 512, lr 1e-3, steps=3133,
     180s cap, ckpt-resumable — the e053c recipe with the step count
     pinned. Val-CE trajectory expectation ~1.52 (e053c landed 1.5227 at
     its cap-truncated 3133; the 2000-step ladder siblings landed
     1.626-1.654).
  2. INSTALL: the e043 exposure protocol VERBATIM via import (e098
     convention: same 60 install windows, SPLICE_RNG 24301, masked
     token-weighted name loss, 16 paired + 32 random anchor mix, AdamW
     0.9/0.95 wd 0.1, lr 1e-3 house cosine, 100 steps, total 100, clip
     1.0; pass bars install-60 p(Z) >= 0.20 AND R1i NLL <= 4.5; ONE fresh
     200-step patience trajectory if the 100-step run misses).
  3. M2 GRID: the e110 mini-grid on the INSTALL's free-run anchor — the
     e098-M2 machinery VERBATIM: seed-202 8-draw 64-token val prompts,
     seed-7 free run 64->512, g_int 250 / T_int 314, band = positions
     64..217 (154 entries), single-shot V-scaling before the 314 decode
     (V<-r*V on the kept k-subset + V=0 on the band complement, K
     untouched), matched continuation seed 960250, clean-judge tail
     448..511 vs the matched control; subsets re-drawn with e110's own
     RNG stream (default_rng(110), (k,d,row) order, k in 32,64,128 —
     the k=64/128 subsets are bitwise e110's/e098's); r in {0.25, 0.56,
     1.0} x k in {64, 128}, D=3 draws; r*_cont(k) = the 0.3-nat crossing
     interpolated in log2(r); product = r*_cont(k) x k.

REGISTERED BARS (frozen from the tasking, before compute):
  - MATURITY CONFIRMED: products land within +-25% of 54 (i.e. >= 40.5
    at BOTH ks; upper edge 67.5 — reachable only as off-grid-high at
    k=64, which is reported as stronger-than-maturity texture) => the
    constant is exposure-driven per W007; the seed axis is excluded.
  - SEED-DRIFT: products land within the 2000-step siblings' +-25%
    neighborhood (the tasking's operational edge: <= 38.0 at BOTH ks;
    pooled sibling product mean 36.8, products {29.9, 32.9, 46.0, 38.4})
    => maturity excluded — the 54 is e053c's idiosyncrasy; W007 needs
    revision.
  - BETWEEN / MIXED (incl. off-grid columns): report texture honestly.
    Off-grid-low (product < 0.25k << 40.5) reads toward seed-drift
    (flagged); off-grid-high cannot fire MATURITY (no on-grid crossing).

DEVIATIONS / DECISIONS (registered before compute):
  1. steps=3133 is a COMPLETED cosine schedule — e098's deviation-2
     convention (identical schedule treatment for every point on the
     maturity axis). e053c's own 3133 was a 4000-total cosine truncated
     by the wall-clock cap; its exact stop step was machine-speed-bound
     and not reproducible. Data order is corpus-seeded (1337) and
     bitwise-matches e053c's first 3133 steps; INIT (seed 4309) is the
     only new axis (e040 convention).
  2. The tasking's "12/12 instrument-identity gates" was e098's count
     (2 nets x 6 cells). e117 has ONE net x 6 cells, each cell carrying
     TWO identity families — the e098 transform fired-dict (applied,
     norm_dev < 1e-5, min_cos > 1-1e-5, removed-exactly-zero, non-band
     untouched, n_kept == k) AND the e110-G4 prefix bit-identity (the
     cell stream bitwise == the control through T_int) — 12 sub-gates,
     all reported; the control's not-applied bit is reported alongside.
  3. CPU fallback: if the GPU stays user-occupied/thermal past the
     bounded wait, base + install run CPU-side with the FULL step counts
     kept (3133 / 100) under extended caps — the exposure axis IS the
     experiment; e098's reduced-steps CPU floor would destroy the
     discriminator's purpose. Documented in metrics.deviations.
  4. Seed-drift edge 38.0 is the tasking's literal number; the siblings'
     +-25%-of-pooled-mean band [27.6, 46.0] is drawn and reported as
     texture alongside it.
  5. M2 net rule (e098 verbatim): the INSTALL net if its gate passes
     and its free-run anchor is healthy (control-continuation
     clean-judge tail CE <= 2.5); else the BASE net, flagged.
  6. No census: e117 is the M2-only discriminator (M1 universality was
     adjudicated at e098/e116).

Outputs: runs/e117/{metrics.json, maturity_point.png}
Run:     python lab/e117_maturity_point.py   (E117_SMOKE=1 -> shakedown)
Envelope: ONE base train <=1M params, batch 32, <=180s cap (per launch,
ckpt-resumable across launches); ONE install fine-tune <=180s;
gpu_ok() double-poll before every launch; cooldown(120) between GPU
phases; CPU for all M2 evals. No NOTES/THINKING/QUEUE/STATE edits; no
commit (the dispatcher owns those).
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

import common
from common import (REPO, Cfg, CharCorpus, TinyGPT, cooldown, estimate_loss,
                    gpu_ok, gpu_status, run_dir, save_json, set_seed,
                    train_model)
import e043_install as E43                       # the install protocol, verbatim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402

SMOKE = os.environ.get("E117_SMOKE") == "1"

# ------------------------------------------------------------------ constants
NAME = "ZEPHYRA"
PRE = 130
HOSTS = ["FLORIZEL", "ELIZABETH"]
SEED = 4309                                      # the ONE fresh seed

# base-net training (e053c recipe; steps pinned at e053c's realized 3133)
BASE_STEPS = 40 if SMOKE else 3133
BASE_BATCH = 4 if SMOKE else 32
BASE_CAP_S = 30.0 if SMOKE else 180.0
BASE_CAP_S_CPU = 60.0 if SMOKE else 2400.0       # deviation 3: keep full steps
EVAL_EVERY = 10 if SMOKE else 250
VCE_PLAUSIBLE = 1.70                             # e053c plausibility gate
VCE_EXPECT = 1.52                                # the ~1.52 trajectory expectation
N_PARAMS_EXPECT = 873_472

# install (e043 verbatim + e082 GATE-0 bars/patience)
INSTALL_STEPS = 8 if SMOKE else 100
INSTALL_EVAL_AT = {4, 8} if SMOKE else {25, 50, 100}
PATIENCE_STEPS = 16 if SMOKE else 200
GATE_PZ_FLOOR = 0.20
GATE_R1I_MAX = 4.5

# M2 (e110 mini-grid; machinery constants verbatim e110/e098 — geometry
# IDENTICAL in smoke and full mode; smoke shrinks only B / draws / boot)
PROMPT_TOK = 64
T_TOTAL = 512
TEMP, TOPK = 0.8, 40
SEED_PROMPT, SEED_SAMPLE = 202, 7
SEED_CONT = 960250
SEED_SUB = 110
N_PROMPTS = 2 if SMOKE else 8
B = N_PROMPTS
BOOT_N = 50 if SMOKE else 1000
THREADS = 8                                       # T050: 12 thrashes the box

KEY_T = (T_TOTAL - 64, T_TOTAL - 1)               # judged tail (448, 511)
G_INT = 250
T_INT = PROMPT_TOK + G_INT                        # 314
GEN_FIRST = 64
BAND = list(range(GEN_FIRST, T_INT - 96))         # 64..217 (154 entries)
FIRST_AFFECTED = T_INT + 1                        # 315
TAIL_STEP0 = KEY_T[0] - FIRST_AFFECTED            # 133
BAND_ARR = np.arange(BAND[0], BAND[-1] + 1)

KS = [64, 128]                                    # the mini-grid field sizes
RS = [0.25, 0.56, 1.0]                            # the mini-grid rungs
D_DRAWS = 1 if SMOKE else 3
HEALTHY_BAR = 0.3
DAMAGED_BAR = 1.0
ANCHOR_CJ_MAX = 2.5                               # "install complicates anchor"

# the registered decision numbers (frozen; see docstring)
SHARE_REF = 54.0                                  # T066's constant (e053c)
SHARE_BAND = 0.25                                 # +-25%
MAT_LO, MAT_HI = SHARE_REF * (1 - SHARE_BAND), SHARE_REF * (1 + SHARE_BAND)
SEED_BAR = 38.0                                   # tasking's literal edge
SIB_POOL_MEAN = 36.802                            # e098 pooled product mean
SIB_LO, SIB_HI = SIB_POOL_MEAN * 0.75, SIB_POOL_MEAN * 1.25   # [27.6, 46.0]

GPU_WAIT_MAX_S = 60 if SMOKE else 600
GPU_POLL_S = 30
COOLDOWN_S = 5 if SMOKE else 120

E098_METRICS = REPO / "runs" / "e098" / "metrics.json"
E110_METRICS = REPO / "runs" / "e110" / "metrics.json"
E053C_VAL_CE = 1.5226792494455974                 # the 54's own net
E053C_STEPS = 3133

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

REGISTERED = {
    "maturity_bar": (f"products r*_cont(k) x k within [{MAT_LO:.1f}, "
                     f"{MAT_HI:.1f}] (+-25% of {SHARE_REF:.0f}; the tasking's "
                     f"operative phrasing '>= {MAT_LO:.1f} at both k') at BOTH "
                     "ks => MATURITY CONFIRMED (constant is exposure-driven "
                     "per W007; seed excluded)"),
    "seed_drift_bar": (f"products <= {SEED_BAR:.0f} at BOTH ks (the 2000-step "
                       "siblings' +-25% neighborhood; pooled product mean "
                       f"{SIB_POOL_MEAN:.1f}, band [{SIB_LO:.1f}, "
                       f"{SIB_HI:.1f}] drawn for texture) => SEED-DRIFT "
                       "(maturity excluded; the 54 is e053c's idiosyncrasy; "
                       "W007 needs revision)"),
    "between_bar": ("between / mixed (incl. off-grid columns) => report "
                    "texture honestly; off-grid-low reads toward seed-drift "
                    "(flagged); off-grid-high cannot fire MATURITY"),
    "m2_net_rule": ("the INSTALL net if its gate passes and its free-run "
                    "anchor is healthy (control-continuation clean-judge "
                    f"tail CE <= {ANCHOR_CJ_MAX}); else the BASE net, flagged"),
    "instrument_gates": ("6 cells x 2 families (e098 transform fired-dict + "
                         "e110-G4 prefix bit-identity) = 12 sub-gates; the "
                         "tasking's '12/12' was e098's 2 nets x 6 cells — "
                         "same machinery, one net here"),
}
log("REGISTERED: " + " | ".join(f"{k}: {v}" for k, v in REGISTERED.items()))


# ------------------------------------------------------------------ gpu gate
def gpu_ok_double(poll_gap_s: float = 2.0) -> bool:
    if not gpu_ok():
        return False
    time.sleep(poll_gap_s)
    return gpu_ok()


def acquire_gpu() -> tuple[str, list[dict]]:
    """Bounded wait for a thermally-clear, user-free GPU; else CPU fallback."""
    polls = []
    t0 = time.time()
    while time.time() - t0 < GPU_WAIT_MAX_S:
        s = gpu_status()
        polls.append(s)
        if gpu_ok_double():
            return "cuda", polls
        log(f"[gpu_guard] HOLD (double-poll failed): {s} — waiting "
            f"{GPU_POLL_S}s ({time.time() - t0:.0f}/{GPU_WAIT_MAX_S}s)")
        time.sleep(GPU_POLL_S)
    return "cpu", polls


# ------------------------------------------------------------------ batteries
def load_ref_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def maturity_refs() -> dict:
    """The three existing constant-vs-steps points + e110's products."""
    out = {}
    m98 = load_ref_json(E098_METRICS)
    if m98:
        pts = []
        for net in m98["m2"]["nets"]:
            ks = {int(k): v for k, v in net["rstar"].items()}
            prods = [ks[k]["product"] for k in sorted(ks) if ks[k]["product"]]
            ci_lo = min(ks[k]["product_ci"][0] for k in sorted(ks))
            ci_hi = max(ks[k]["product_ci"][1] for k in sorted(ks))
            pts.append(dict(
                seed=net["seed"], which=net["which"], steps=2000,
                val_ce=m98["per_seed"][str(net["seed"])]["base"]["val_ce"],
                products={str(k): ks[k]["product"] for k in sorted(ks)},
                product_ci=[ci_lo, ci_hi],
                net_constant=(sum(prods) / len(prods)) if prods else None))
        out["e098_ladder_points"] = pts
        out["e098_pooled_product_mean"] = float(np.mean(
            [p for net in pts for p in net["products"].values()
             if p is not None]))
    m110 = load_ref_json(E110_METRICS)
    if m110:
        vals = m110["summary"]["product"]["values"]
        out["e053c_point"] = dict(
            seed=42, which="e053c_ctx512 (the 54)", steps=E053C_STEPS,
            val_ce=m110["net"]["val_ce"],
            products={k: v for k, v in vals.items() if v is not None},
            net_constant=(float(np.mean([vals["64"], vals["128"]]))
                          if vals.get("64") and vals.get("128") else None))
    return out


@torch.no_grad()
def battery_pz(m: TinyGPT, ids: torch.Tensor, zid: int, bs: int = 30):
    """e067's battery: per-window p(Z) at the last context position."""
    m.eval()
    pzs, amax = [], []
    for i in range(0, ids.shape[0], bs):
        lg, _ = m(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs += [float(pr[k, zid]) for k in range(pr.shape[0])]
        amax += [bool(pr[k].argmax() == zid) for k in range(pr.shape[0])]
    return np.array(pzs), float(np.mean(amax))


# ------------------------------------------------- e110 machinery (VERBATIM)
@torch.no_grad()
def manual_all_logits(net: TinyGPT, idxs):
    """Clean forward returning logits at ALL positions (the clean judge)."""
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
    lg = logits_clean / TEMP
    v, _ = torch.topk(lg, TOPK)
    lg_f = lg.masked_fill(lg < v[-1], float("-inf"))
    tok = int(torch.multinomial(torch.softmax(lg_f, -1), 1, generator=gen))
    p_full = torch.softmax(logits_clean, -1)
    return tok, float(-math.log(max(p_full[tok].item(), 1e-12)))


@torch.no_grad()
def prefill_batch(net: TinyGPT, idx: torch.Tensor):
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
        y = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(Bb, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
    return net.lm_head(net.ln_f(x))


@torch.no_grad()
def generate_control(net: TinyGPT, prompts, gen: torch.Generator):
    Bb = len(prompts)
    idx = torch.stack(prompts)
    logits, kv = prefill_batch(net, idx)
    G = T_TOTAL - PROMPT_TOK
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


@torch.no_grad()
def run_field_continuation(net: TinyGPT, prefix: torch.Tensor,
                           forced_tok: torch.Tensor, forced_pos: int,
                           seed: int, keep_per_row=None, r: float = 1.0):
    """e110's run_field_continuation VERBATIM: single-shot band transform
    (kept subset V<-r*V, band complement V=0, K untouched) before the decode
    at forced_pos; teacher-forced token; shared-seed matched free-run;
    per-row online CE + entropy tracked."""
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
        pre_v = [v.clone() for (_k, v) in kv]
        coss, ratios = [], []
        band_set = set(BAND)
        for j in range(R):
            kp = torch.as_tensor(keep_per_row[j], dtype=torch.long)
            comp = torch.tensor(sorted(band_set - set(kp.tolist())),
                                dtype=torch.long)
            fired["n_kept_per_row"].append(int(kp.numel()))
            fired["n_removed_per_row"].append(int(comp.numel()))
            for (_k, v) in kv:
                old = v[j, :, kp, :].clone()
                if r != 1.0:
                    v[j, :, kp, :] = r * old
                if comp.numel():
                    v[j, :, comp, :] = 0.0
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
            nb_lo = torch.arange(0, GEN_FIRST)
            nb_hi = torch.arange(BAND[-1] + 1, forced_pos)
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
    out = {}
    for (lo, hi) in windows:
        lg = all_lg[:, lo - 1:hi, :]
        tgt = idx[:, lo:hi + 1]
        lp = torch.log_softmax(lg.float(), -1)
        out[(lo, hi)] = -lp.gather(2, tgt[:, :, None]).squeeze(2).mean(1).numpy()
    return out


def boot_ci(arrs, fn, n: int = BOOT_N, seed: int = 0):
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
    if cost < HEALTHY_BAR:
        return "healthy"
    if cost > DAMAGED_BAR:
        return "damaged"
    return "ambiguous"


def rstar_cont(col_means: dict) -> tuple:
    """e110's continuous 0.3-nat crossing in log2(r) — rightmost bracket."""
    if col_means[RS[-1]] > HEALTHY_BAR:
        return float("nan"), "off-grid-high"
    if col_means[RS[0]] < HEALTHY_BAR:
        return float("nan"), "off-grid-low"
    x = np.log2(np.array(RS))
    y = np.array([col_means[r] for r in RS])
    for i in range(len(RS) - 2, -1, -1):
        if y[i] >= HEALTHY_BAR > y[i + 1]:
            t = x[i] + (HEALTHY_BAR - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
            return float(2.0 ** t), "interp"
    return float("nan"), "no-crossing"


# ------------------------------------------------------------------ helpers
def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (bool, np.bool_)) or x is None or isinstance(x, (int, str)):
        return bool(x) if isinstance(x, np.bool_) else x
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return "inf" if math.isinf(v) else ("nan" if math.isnan(v) else v)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, torch.Tensor):
        return jsonable(x.tolist())
    return str(x)


def to_cpu_net(sd: dict, cfg: Cfg) -> TinyGPT:
    m = TinyGPT(cfg)
    m.load_state_dict(sd)
    m.eval()
    return m


# ------------------------------------------------------------------ main
def main():
    torch.set_num_threads(THREADS)
    tag = "e117_smoke" if SMOKE else "e117"
    rd = run_dir(tag)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    deviations = []

    # ---- corpus + protocol rebuild (e098 verbatim)
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    assert corpus.vocab_size == 65
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=4, n_head=4, n_embd=128,
              block_size=T_TOTAL)
    stoi = corpus.stoi
    train_ids = corpus.train
    train_text = "".join(corpus.itos[int(i)] for i in train_ids)
    zid = stoi["Z"]
    assert not E43.find_occ(train_text, NAME)

    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + E43.POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    assert mix == {"FLORIZEL": 19, "ELIZABETH": 41}, f"splice drift {mix}"
    name_ids = torch.tensor([stoi[c] for c in NAME], dtype=torch.long)

    def build_win(p, host):
        return torch.cat([train_ids[p - PRE: p], name_ids,
                          train_ids[p + len(host): p + len(host) + E43.POST_CAP]])

    win_i = torch.stack([build_win(p, h) for p, h in install_occ])
    inst_mask = torch.zeros(60, E43.BLOCK - 1, dtype=torch.bool)
    inst_mask[:, PRE - 1:PRE - 1 + len(NAME)] = True
    anchor_full = torch.stack([train_ids[p - PRE: p - PRE + E43.BLOCK]
                               for p, _ in install_occ])
    bat_i_seq = win_i[:, :PRE + len(NAME)]
    ids_install = torch.stack([corpus.encode(train_text[p - PRE:p])
                               for p, _ in install_occ])
    ids_held = torch.stack([corpus.encode(train_text[p - PRE:p])
                            for p, _ in held_occ])
    log(f"protocol rebuilt: install60 {mix}, held30 (corpus 1337, "
        f"SPLICE_RNG {E43.SPLICE_RNG}) | cfg 4L/4H/128d/wpe-{T_TOTAL} | "
        f"seed {SEED}, steps {BASE_STEPS}")

    refs = maturity_refs()
    log(f"refs: {json.dumps(jsonable(refs), default=str)[:400]}")

    # ================================================== 1. the BASE net (3133)
    dev, polls = acquire_gpu()
    base_gpu = {"device": dev, "n_polls": len(polls),
                "last": polls[-1] if polls else None}
    cap = BASE_CAP_S
    if dev == "cpu":
        cap = BASE_CAP_S_CPU
        deviations.append(
            f"base train fell back to CPU (GPU user-occupied/thermal past "
            f"{GPU_WAIT_MAX_S}s) — FULL {BASE_STEPS}-step count kept under an "
            f"extended {cap:.0f}s cap (deviation 3: the exposure axis is the "
            f"experiment)")
    common.DEVICE = dev
    set_seed(SEED)                                # init on CPU (e040 rule)
    net = TinyGPT(cfg).to(dev)
    n_params = net.num_params()
    assert n_params == N_PARAMS_EXPECT, f"params {n_params} != {N_PARAMS_EXPECT}"
    ck_base = REPO / "runs" / "checkpoints" / (
        f"e117_base_s{SEED}{'_smoke' if SMOKE else ''}.pt")
    t1 = time.time()
    # if a stale ckpt from an aborted attempt exists at fewer steps, resume;
    # train_model's ckpt protocol handles both cases.
    launches = 0
    hist = []
    while True:
        hist = train_model(net, corpus, steps=BASE_STEPS, lr=1e-3,
                           batch_size=BASE_BATCH, max_seconds=cap,
                           eval_every=EVAL_EVERY, ckpt=ck_base)
        launches += 1
        steps_done = int(hist[-1]["step"]) if hist else 0
        if steps_done >= BASE_STEPS or launches >= 3:
            break
        log(f"cap-bound at step {steps_done} < {BASE_STEPS} — cooldown, then "
            f"resume segment {launches + 1} (ckpt-resumable, per-launch cap "
            f"{cap:.0f}s)")
        if dev == "cuda":
            cooldown(COOLDOWN_S)
        net = net.to(dev)
    steps_done = int(hist[-1]["step"]) if hist else 0
    base_wall = time.time() - t1
    common.DEVICE = "cpu"
    net_cpu = net.cpu().eval()
    val_ce = estimate_loss(net_cpu, corpus, "val", n_batches=12)
    base_sd = {k: v.detach().clone() for k, v in net_cpu.state_dict().items()}
    del net
    base = {
        "seed": SEED, "params": n_params, "steps_target": BASE_STEPS,
        "steps_done": steps_done, "val_ce": val_ce,
        "expectation_vce": VCE_EXPECT, "e053c_vce": E053C_VAL_CE,
        "siblings_vce": "1.626-1.654 (e098 @2000)",
        "plausible": bool(val_ce < VCE_PLAUSIBLE),
        "wall_s": round(base_wall, 1), "device": dev, "launches": launches,
        "cap_s": cap, "batch": BASE_BATCH,
        "ckpt": str(ck_base.name),
        "history": [{"step": h["step"], "train_loss": h["train_loss"],
                     "val_loss": h["val_loss"]} for h in hist],
        "gpu": base_gpu,
    }
    log(f"base: steps {steps_done}/{BASE_STEPS} on {dev} in {base_wall:.0f}s "
        f"({launches} launch(es)) | val CE {val_ce:.4f} "
        f"(expectation ~{VCE_EXPECT}; e053c {E053C_VAL_CE:.4f}; "
        f"siblings 1.63-1.65) -> "
        f"{'plausible' if base['plausible'] else 'OFF'}")
    if dev == "cuda":
        cooldown(COOLDOWN_S)

    # ================================================== 2. the INSTALL (e043)
    dev2, polls2 = acquire_gpu()
    common.DEVICE = dev2
    E43.DEVICE = dev2
    if dev2 == "cpu" and dev == "cuda":
        deviations.append("install ran CPU-side (GPU lost mid-run)")
    inst = copy.deepcopy(net_cpu).to(dev2)

    def gate_readout(m, step):
        pz, am = battery_pz(m, ids_install.to(dev2), zid)
        pzh, _ = battery_pz(m, ids_held.to(dev2), zid)
        r1i = E43.eval_seq(m, bat_i_seq.to(dev2), len(NAME), PRE - 1)
        return {"step": step, "install60_pz": float(pz.mean()),
                "install60_argmax": am, "held30_pz": float(pzh.mean()),
                "r1i_nll": r1i["nll"], "r1i_acc": r1i["acc"],
                "onset_acc": r1i["per_pos_acc"][0],
                "pos36_acc": sum(r1i["per_pos_acc"][3:7]) / 4}

    ck_inst = REPO / "runs" / "checkpoints" / (
        f"e117_install_s{SEED}{'_smoke' if SMOKE else ''}.pt")
    if ck_inst.exists():
        ck_inst.unlink()                          # fresh install every run
    t2 = time.time()
    traj = E43.exposure(
        inst, win_i.to(dev2), inst_mask.to(dev2), anchor_full.to(dev2),
        steps=INSTALL_STEPS, total=INSTALL_STEPS,
        gen=torch.Generator().manual_seed(SEED), tag=f"e117_install_s{SEED}",
        ckpt=ck_inst, eval_at=set(INSTALL_EVAL_AT), on_eval=gate_readout,
        mix_random=E43.MIX_RANDOM, train_ids=train_ids, log=log)
    inst_wall = time.time() - t2
    final_ro = gate_readout(inst, INSTALL_STEPS)
    gate_pass = bool(final_ro["install60_pz"] >= GATE_PZ_FLOOR
                     and final_ro["r1i_nll"] <= GATE_R1I_MAX)
    patience_used = False
    if not gate_pass and not SMOKE:
        log(f"  GATE-0 miss at s{INSTALL_STEPS} (pZ "
            f"{final_ro['install60_pz']:.3f} / R1i {final_ro['r1i_nll']:.2f})"
            f" — e082 patience branch: ONE fresh {PATIENCE_STEPS}-step run")
        inst = copy.deepcopy(net_cpu).to(dev2)
        ck_pat = REPO / "runs" / "checkpoints" / (
            f"e117_install_s{SEED}_pat{'_smoke' if SMOKE else ''}.pt")
        t3 = time.time()
        traj = E43.exposure(
            inst, win_i.to(dev2), inst_mask.to(dev2), anchor_full.to(dev2),
            steps=PATIENCE_STEPS, total=PATIENCE_STEPS,
            gen=torch.Generator().manual_seed(SEED + 1000),
            tag=f"e117_install_s{SEED}_pat", ckpt=ck_pat,
            eval_at={PATIENCE_STEPS // 2, PATIENCE_STEPS},
            on_eval=gate_readout, mix_random=E43.MIX_RANDOM,
            train_ids=train_ids, log=log)
        inst_wall += time.time() - t3
        final_ro = gate_readout(inst, PATIENCE_STEPS)
        gate_pass = bool(final_ro["install60_pz"] >= GATE_PZ_FLOOR
                         and final_ro["r1i_nll"] <= GATE_R1I_MAX)
        patience_used = True
    inst_sd = {k: v.detach().cpu().clone() for k, v in inst.state_dict().items()}
    del inst
    common.DEVICE = "cpu"
    E43.DEVICE = "cpu"
    install = {
        "protocol": "e043 exposure VERBATIM (100 steps, 16+48 mix, lr 1e-3, "
                    "token-weighted masked loss)",
        "gen_seed": SEED if not patience_used else SEED + 1000,
        "steps": PATIENCE_STEPS if patience_used else INSTALL_STEPS,
        "traj": traj, "final": final_ro, "gate_pass": gate_pass,
        "patience_used": patience_used,
        "bars": {"pz_floor": GATE_PZ_FLOOR, "r1i_max": GATE_R1I_MAX},
        "wall_s": round(inst_wall, 1), "device": dev2,
        "under_180s": bool(inst_wall <= 180.0),
        "gpu": {"device": dev2, "n_polls": len(polls2)},
    }
    log(f"install (s{install['steps']}, {dev2}, {inst_wall:.0f}s): pZ "
        f"{final_ro['install60_pz']:.4f} held {final_ro['held30_pz']:.4f} "
        f"R1i {final_ro['r1i_nll']:.3f}/{final_ro['r1i_acc']:.3f} -> GATE "
        f"{'PASS' if gate_pass else 'FAIL'}"
        + (" [patience]" if patience_used else ""))
    if dev2 == "cuda":
        cooldown(COOLDOWN_S)

    # ================================================== 3. the M2 mini-grid
    gen_p = torch.Generator().manual_seed(SEED_PROMPT)
    ix = torch.randint(len(corpus.val) - PROMPT_TOK - 1, (N_PROMPTS,),
                       generator=gen_p)
    prompts = [corpus.val[i:i + PROMPT_TOK] for i in ix]

    rng_sub = np.random.default_rng(SEED_SUB)
    subsets = {k: [[None] * B for _ in range(D_DRAWS)] for k in KS}
    for k in (32, *KS):                       # e110's exact (k,d,row) stream
        for d in range(D_DRAWS if k in KS else 3):
            for j in range(B):
                drawn = rng_sub.choice(BAND_ARR, size=k, replace=False)
                if k in KS:
                    subsets[k][d][j] = drawn

    def anchor_and_grid(sd: dict, which: str, force: bool = False) -> dict | None:
        net_m2 = to_cpu_net(sd, cfg)
        gen7 = torch.Generator().manual_seed(SEED_SAMPLE)
        run1 = generate_control(net_m2, prompts, gen7)
        idx1 = run1["idx"]
        prefix, forced = idx1[:, :T_INT], idx1[:, T_INT]
        idx_ctl, kv_ctl, fired_ctl, ce_ctl, ent_ctl = run_field_continuation(
            net_m2, prefix, forced, T_INT, SEED_CONT, keep_per_row=None, r=1.0)
        J_ctl = judge_windows(manual_all_logits(net_m2, idx_ctl), idx_ctl,
                              [KEY_T])[KEY_T]
        online_tail = float(np.asarray(ce_ctl[:, TAIL_STEP0:]).mean())
        anchor_ok = bool(np.isfinite(J_ctl).all() and J_ctl.mean() <= ANCHOR_CJ_MAX)
        log(f"  [{which}] control tail cj {float(J_ctl.mean()):.4f} online "
            f"{online_tail:.4f} -> anchor "
            f"{'healthy' if anchor_ok else 'BROKEN'}"
            + (" [SMOKE: proceeding on a broken anchor to exercise the "
               "plumbing — the registered bar still applies in full mode]"
               if (not anchor_ok and force) else ""))
        if not anchor_ok and not force:
            return None
        cellJ, cell_info = {}, {}
        for k in KS:
            for r in RS:
                Js, fireds, idxs_a = [], [], []
                for d in range(D_DRAWS):
                    keep = [torch.as_tensor(subsets[k][d][j], dtype=torch.long)
                            for j in range(B)]
                    idx_a, _kv, fired_a, ce_a, _e = run_field_continuation(
                        net_m2, prefix, forced, T_INT, SEED_CONT,
                        keep_per_row=keep, r=r)
                    Js.append(judge_windows(manual_all_logits(net_m2, idx_a),
                                            idx_a, [KEY_T])[KEY_T])
                    fireds.append(fired_a)
                    idxs_a.append(idx_a)
                cellJ[(k, r)] = np.mean(np.stack(Js), axis=0)
                f0 = fireds[0]
                # family 1: e098's transform fired-dict; family 2: e110-G4's
                # prefix bit-identity vs the matched control
                transform_ok = bool(
                    f0["applied"] and f0["norm_dev"] < 1e-5
                    and f0["min_cos"] > 1.0 - 1e-5
                    and f0["removed_exactly_zero"] and f0["nonband_untouched"]
                    and f0["n_kept_per_row"] == [k] * B)
                prefix_ok = bool(all(
                    torch.equal(ia[:, :T_INT + 1], idx_ctl[:, :T_INT + 1])
                    for ia in idxs_a))
                cell_info[(k, r)] = {"fired": f0,
                                     "transform_ok": transform_ok,
                                     "prefix_identical": prefix_ok,
                                     "ok": bool(transform_ok and prefix_ok)}
                log(f"  [{which}] cell k={k} r={r}: cj-tail cost "
                    f"{float((cellJ[(k, r)] - J_ctl).mean()):+.4f} "
                    f"[{health_of(float((cellJ[(k, r)] - J_ctl).mean()))}] "
                    f"transform-ok {transform_ok} prefix-ok {prefix_ok}")
        col_means = {k: {r: float((cellJ[(k, r)] - J_ctl).mean())
                         for r in RS} for k in KS}
        rstar, prod_ci = {}, {}
        rng_b = np.random.default_rng(0)
        conts_b = {k: [] for k in KS}
        for _ in range(BOOT_N):
            sel = rng_b.integers(0, B, B)
            for k in KS:
                cm = {r: float((cellJ[(k, r)] - J_ctl)[sel].mean()) for r in RS}
                cv, _cs = rstar_cont(cm)
                conts_b[k].append(cv * k if np.isfinite(cv) else np.nan)
        for k in KS:
            cval, cstatus = rstar_cont(col_means[k])
            prod = float(cval * k) if np.isfinite(cval) else None
            ci = ([float(np.nanpercentile(conts_b[k], 2.5)),
                   float(np.nanpercentile(conts_b[k], 97.5))]
                  if not all(np.isnan(x) for x in conts_b[k])
                  else [float("nan"), float("nan")])
            rstar[k] = {"cont_value": (None if not np.isfinite(cval)
                                       else float(cval)),
                        "cont_status": cstatus, "product": prod,
                        "product_ci": ci,
                        "in_maturity_band": (bool(MAT_LO <= prod <= MAT_HI)
                                             if prod is not None else None),
                        "le_seed_bar": (bool(prod <= SEED_BAR)
                                        if prod is not None else None),
                        "in_siblings_pm25": (bool(SIB_LO <= prod <= SIB_HI)
                                             if prod is not None else None),
                        "col_means": {str(r): col_means[k][r] for r in RS}}
            log(f"  [{which}] r*({k}) = "
                + (f"{cval:.3f}" if np.isfinite(cval) else cstatus)
                + (f" | r*k {prod:.1f} CI [{ci[0]:.1f},{ci[1]:.1f}] "
                   f"maturity-band {rstar[k]['in_maturity_band']} "
                   f"seed-bar(<=38) {rstar[k]['le_seed_bar']}" if prod else ""))
        per_cell = {f"k={k},r={r}": {
            "cost_per_row": (cellJ[(k, r)] - J_ctl).tolist(),
            "mean": float((cellJ[(k, r)] - J_ctl).mean()),
            "ci": boot_ci([cellJ[(k, r)] - J_ctl], lambda a: float(a.mean())),
            "health": health_of(float((cellJ[(k, r)] - J_ctl).mean())),
            "instrument_transform_ok": cell_info[(k, r)]["transform_ok"],
            "instrument_prefix_identical": cell_info[(k, r)]["prefix_identical"],
            "instrument_ok": cell_info[(k, r)]["ok"]}
            for k in KS for r in RS}
        n_gate_sub = len(cell_info) * 2
        n_gate_ok = sum(2 if v["ok"] else
                        (int(v["transform_ok"]) + int(v["prefix_identical"]))
                        for v in cell_info.values())
        return {"which": which,
                "control_tail_cj": float(J_ctl.mean()),
                "control_online_tail": online_tail,
                "control_anchor_ok": anchor_ok,
                "control_not_applied": bool(not fired_ctl["applied"]),
                "rstar": {str(k): v for k, v in rstar.items()},
                "per_cell": per_cell,
                "instrument_subgates_ok": f"{n_gate_ok}/{n_gate_sub}",
                "instrument_all_ok": bool(all(v["ok"] for v in
                                              cell_info.values()))}

    which = "install" if gate_pass else None
    m2 = anchor_and_grid(inst_sd, "install", force=SMOKE) if gate_pass else None
    if m2 is None:
        log("  install unusable for M2 — trying the BASE net (flagged)")
        which = "base"
        m2 = anchor_and_grid(base_sd, "base", force=SMOKE)
        if m2 is not None:
            deviations.append(
                "M2: used the BASE net (install did not provide a healthy "
                "free-run anchor or failed its gate)")
    assert m2 is not None or SMOKE, "no usable net for M2"
    if m2 is None:
        log("SMOKE: no usable net even with the forced anchor — exiting "
            "without metrics (full mode cannot hit this path)")
        return 0

    # ================================================== 4. the REGISTERED decision
    p64 = m2["rstar"]["64"]["product"]
    p128 = m2["rstar"]["128"]["product"]
    prods = [p64, p128]
    named = {"64": p64, "128": p128}
    status = {k: m2["rstar"][k]["cont_status"] for k in ("64", "128")}
    offgrid = [k for k, v in status.items() if v != "interp"]
    evaluable = [p for p in prods if p is not None]

    def in_mat(p):
        return p is not None and MAT_LO <= p <= MAT_HI

    def le_seed(p):
        return p is not None and p <= SEED_BAR

    if len(evaluable) == 2 and all(in_mat(p) for p in evaluable):
        clause = "MATURITY CONFIRMED"
        verdict = (f"products r*k = {p64:.1f} (k=64) / {p128:.1f} (k=128) "
                   f"both land within +-25% of 54 [{MAT_LO:.1f}, {MAT_HI:.1f}] "
                   f"— the constant is EXPOSURE-DRIVEN (W007 upheld; seed "
                   f"excluded): at fixed architecture, 3133 steps of a fresh "
                   f"seed reproduce the 54 that 2000-step seeds miss (31/42).")
    elif len(evaluable) == 2 and all(le_seed(p) for p in evaluable):
        clause = "SEED-DRIFT"
        verdict = (f"products r*k = {p64:.1f} (k=64) / {p128:.1f} (k=128) "
                   f"both land <= {SEED_BAR:.0f} (the 2000-step siblings' "
                   f"+-25% neighborhood) — MATURITY EXCLUDED: the 54 is "
                   f"e053c's idiosyncrasy (seed axis); W007 needs revision "
                   f"(the constant does not track exposure at fixed arch).")
    else:
        clause = "BETWEEN / MIXED"
        texture = []
        for k in ("64", "128"):
            p = named[k]
            if p is None:
                texture.append(f"k={k}: {status[k]} (off-grid)")
            elif p > SEED_BAR and p < MAT_LO:
                texture.append(f"k={k}: {p:.1f} in the 38-40.5 gap")
            elif in_mat(p):
                texture.append(f"k={k}: {p:.1f} in the maturity band "
                               f"(mixed with the other column)")
            elif le_seed(p):
                texture.append(f"k={k}: {p:.1f} at/below the seed bar "
                               f"(mixed with the other column)")
            elif p > MAT_HI:
                texture.append(f"k={k}: {p:.1f} ABOVE +25% of 54 "
                               f"(stronger-than-maturity texture)")
            else:
                texture.append(f"k={k}: {p:.1f}")
        verdict = (f"products r*k = {p64} / {p128} land between the bars — "
                   f"honest texture: {'; '.join(texture)}. Off-grid columns: "
                   f"{offgrid or 'none'}.")
    net_const = (float(np.mean(evaluable)) if len(evaluable) == 2 else None)
    log(f"M2 products: {{k64: {p64}, k128: {p128}}} -> net constant "
        f"{net_const} | DECISION [{clause}]")
    log(f"VERDICT: {verdict}")

    decision = {
        "products": named, "product_status": status,
        "net_constant": net_const,
        "bars": {"maturity": [MAT_LO, MAT_HI],
                 "seed_drift_edge": SEED_BAR,
                 "siblings_pm25": [SIB_LO, SIB_HI]},
        "per_product": {k: {"in_maturity_band": m2["rstar"][k]["in_maturity_band"],
                            "le_seed_bar": m2["rstar"][k]["le_seed_bar"],
                            "in_siblings_pm25":
                                m2["rstar"][k]["in_siblings_pm25"]}
                        for k in ("64", "128")},
        "clause": clause, "verdict": verdict,
        "registered": REGISTERED,
    }

    # ---------------------------------------------------------------- metrics
    metrics = {
        "experiment": "e117_maturity_point",
        "purpose": ("T068-M2/W007 discriminator: ONE fresh seed (4309) at "
                    "EXACTLY 3133 steps (e053c recipe) + e043 install + the "
                    "e110 mini-grid on the install's free-run anchor. The "
                    "share constant r*k at 3133 decides MATURITY (54 "
                    "reproduces; exposure-driven) vs SEED-DRIFT (the "
                    "siblings' 31-42; the 54 was e053c's idiosyncrasy)."),
        "started": started, "wall_s": round(time.time() - T0, 1),
        "smoke": SMOKE, "threads": THREADS,
        "registered": REGISTERED,
        "seed": SEED,
        "config": {"base_recipe": f"e053c: 4L/4H/128d block-512, corpus 1337, "
                                  f"batch {BASE_BATCH}, lr 1e-3, cosine, "
                                  f"{BASE_STEPS} steps (COMPLETED schedule; "
                                  f"e098 deviation-2 convention), "
                                  f"{BASE_CAP_S:.0f}s cap/launch",
                   "install_recipe": "e043 exposure VERBATIM; e082 GATE-0 "
                                     "bars + 200-step patience",
                   "m2_grid": {"k": KS, "r": RS, "draws": D_DRAWS, "B": B,
                               "subset_seed": SEED_SUB,
                               "subset_provenance": "e110's exact RNG stream "
                               "(k,d,row) — k=64/128 subsets bitwise "
                               "e110's/e098's",
                               "intervention": {"g_int": G_INT, "t_int": T_INT,
                                                "band": [BAND[0], BAND[-1]],
                                                "n_band": len(BAND)},
                               "continuation_seed": SEED_CONT,
                               "judge_tail": list(KEY_T)}},
        "base": base,
        "install": install,
        "m2_which": which,
        "m2": m2,
        "decision": decision,
        "references": refs,
        "gates": {
            "splice_protocol": {"mix": mix, "pass": True},
            "base_params": {"n": n_params, "expected": N_PARAMS_EXPECT,
                            "pass": bool(n_params == N_PARAMS_EXPECT)},
            "base_steps_exact": {"steps_done": steps_done,
                                 "target": BASE_STEPS,
                                 "pass": bool(steps_done == BASE_STEPS)},
            "base_val_ce_plausible": {"val_ce": val_ce,
                                      "bar": VCE_PLAUSIBLE,
                                      "pass": bool(val_ce < VCE_PLAUSIBLE)},
            "install_gate": {"pass": gate_pass,
                             "bars": {"pz_floor": GATE_PZ_FLOOR,
                                      "r1i_max": GATE_R1I_MAX}},
            "install_under_180s": {"wall_s": install["wall_s"],
                                   "pass": install["under_180s"]},
            "m2_control_not_applied": m2["control_not_applied"],
            "m2_instrument_subgates": m2["instrument_subgates_ok"],
            "m2_instrument_all_ok": m2["instrument_all_ok"],
        },
        "deviations": deviations,
    }
    save_json(rd / "metrics.json", jsonable(metrics))
    log("metrics.json written")
    plot(rd / "maturity_point.png", metrics)
    log(f"plot written -> {rd}")
    return 0


# ------------------------------------------------------------------ plot
def plot(path: Path, M: dict):
    dec = M["decision"]
    m2 = M["m2"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ---- (0,0) THE REGISTERED PLOT: constant vs steps, both bands
    ax = axes[0, 0]
    ax.axhspan(MAT_LO, MAT_HI, color="gold", alpha=0.22,
               label=f"maturity band ±25% of 54 [{MAT_LO:.0f}, {MAT_HI:.0f}]")
    ax.axhspan(SIB_LO, SIB_HI, color="steelblue", alpha=0.15,
               label=f"2000-step siblings ±25% [{SIB_LO:.0f}, {SIB_HI:.0f}]")
    ax.axhline(SHARE_REF, color="k", ls="--", lw=1.5)
    ax.axhline(SEED_BAR, color="navy", ls=":", lw=1.5,
               label=f"seed-drift edge {SEED_BAR:.0f} (tasking bar)")
    refs = M.get("references", {})
    xs_jit = {2000: [1950, 2050], 3133: [3083, 3183]}
    for i, pt in enumerate(refs.get("e098_ladder_points", [])):
        x = xs_jit[2000][i % 2]
        for k, p in pt["products"].items():
            ax.scatter([x], [p], marker="o", s=42, facecolors="none",
                       edgecolors="tab:blue", linewidths=1.4, zorder=3)
        if pt["net_constant"] is not None:
            ax.errorbar([x], [pt["net_constant"]],
                        yerr=[[max(0, pt["net_constant"] - pt["product_ci"][0])],
                              [max(0, pt["product_ci"][1] - pt["net_constant"])]],
                        fmt="D", ms=10, color="tab:blue", capsize=4, zorder=4)
            ax.annotate(f"s{pt['seed']}@2000\n{pt['net_constant']:.1f}",
                        (x, pt["net_constant"]), xytext=(0, -34),
                        ha="center", fontsize=9, color="tab:blue")
    e5 = refs.get("e053c_point")
    if e5:
        for k, p in e5["products"].items():
            ax.scatter([3133], [p], marker="o", s=42, facecolors="none",
                       edgecolors="tab:red", linewidths=1.4, zorder=3)
        if e5["net_constant"] is not None:
            ax.scatter([3133], [e5["net_constant"]], marker="D", s=110,
                       color="tab:red", zorder=5)
            ax.annotate(f"e053c s42@3133\n{e5['net_constant']:.1f} (the 54)",
                        (3133, e5["net_constant"]), xytext=(12, 6),
                        textcoords="offset points", fontsize=9,
                        color="tab:red")
    nc = dec["net_constant"]
    p64 = dec["products"]["64"]
    p128 = dec["products"]["128"]
    if p64 is not None:
        ax.scatter([3133], [p64], marker="o", s=60, facecolors="none",
                   edgecolors="tab:green", linewidths=1.8, zorder=5)
    if p128 is not None:
        ax.scatter([3133], [p128], marker="o", s=60, facecolors="none",
                   edgecolors="tab:green", linewidths=1.8, zorder=5)
    if nc is not None:
        ci = [min(m2["rstar"][k]["product_ci"][0] for k in ("64", "128")),
              max(m2["rstar"][k]["product_ci"][1] for k in ("64", "128"))]
        ax.errorbar([3133], [nc], yerr=[[max(0, nc - ci[0])],
                                        [max(0, ci[1] - nc)]],
                    fmt="*", ms=22, color="tab:green", capsize=5, zorder=6)
        ax.annotate(f"e117 s{M['seed']}@3133\n{nc:.1f}",
                    (3133, nc), xytext=(12, -30), textcoords="offset points",
                    fontsize=10, color="tab:green", weight="bold")
    else:
        ax.annotate(f"e117 s{M['seed']}@3133\noff-grid: "
                    f"{dec['product_status']}", (3133, 5), fontsize=9,
                    color="tab:green", ha="center")
    ax.set_xlabel("base training steps (COMPLETED cosine schedule)")
    ax.set_ylabel("share constant  r*(k) x k")
    ax.set_ylim(0, max(75, (nc or 0) + 12, (e5 or {}).get("net_constant", 0) + 12))
    ax.set_xlim(1850, 3400)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title(f"E117 — the maturity curve (new point = seed "
                 f"{M['seed']} at EXACTLY e053c's 3133 steps)\nDECISION: "
                 f"{dec['clause']}", fontsize=11)

    # ---- (0,1) the mini-grid cost vs r on the new net
    ax = axes[0, 1]
    kcol = {64: "tab:orange", 128: "tab:green"}
    for k in KS:
        rs = m2["rstar"][str(k)]["col_means"]
        xs = [float(r) for r in RS]
        ys = [rs[str(r)] for r in RS]
        rv = m2["rstar"][str(k)]["cont_value"]
        lab = (f"k={k} (r*={rv:.3f})" if rv is not None
               else f"k={k} ({m2['rstar'][str(k)]['cont_status']})")
        ax.plot(xs, ys, "o-", lw=2.2, ms=8, color=kcol[k], label=lab)
    ax.axhline(HEALTHY_BAR, color="tab:green", ls="--", lw=1.3,
               label="0.3 healthy")
    ax.axhline(DAMAGED_BAR, color="tab:red", ls="--", lw=1.3, label="1.0 damaged")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xscale("log")
    ax.set_xticks(RS)
    ax.set_xticklabels([str(r) for r in RS])
    ax.set_xlabel("retention r")
    ax.set_ylabel("clean-judge tail cost vs matched control (nats)")
    ax.set_title(f"the e110 mini-grid on the {M['m2_which']} net's free-run "
                 f"anchor (B={B})\ncontrol tail cj "
                 f"{m2['control_tail_cj']:.3f} (healthy <= 2.5)", fontsize=10)
    ax.legend(fontsize=8.5)

    # ---- (1,0) products bar chart vs both bands
    ax = axes[1, 0]
    labels, vals, cis = [], [], []
    for k in ("64", "128"):
        v = dec["products"][k]
        ci = m2["rstar"][k]["product_ci"]
        labels.append(f"e117 {M['m2_which']}\nk={k}")
        vals.append(v if v is not None else np.nan)
        cis.append(ci)
    for i, pt in enumerate(refs.get("e098_ladder_points", [])):
        for k, p in pt["products"].items():
            labels.append(f"s{pt['seed']}@2000 k={k}")
            vals.append(p)
            cis.append(pt["product_ci"])
    if e5:
        for k, p in e5["products"].items():
            if k in ("64", "128"):
                labels.append(f"e053c@3133 k={k}")
                vals.append(p)
                cis.append([p, p])
    xs = np.arange(len(labels))
    cols = ["tab:green" if "e117" in l else ("tab:blue" if "@2000" in l
                                             else "tab:red") for l in labels]
    ax.bar(xs, vals, 0.6, color=cols)
    ax.errorbar(xs, vals,
                yerr=[[max(0, v - c[0]) if np.isfinite(v) else 0
                       for v, c in zip(vals, cis)],
                      [max(0, c[1] - v) if np.isfinite(v) else 0
                       for v, c in zip(vals, cis)]],
                fmt="none", ecolor="k", capsize=3, lw=1.0)
    ax.axhspan(MAT_LO, MAT_HI, color="gold", alpha=0.25,
               label=f"maturity ±25% of 54")
    ax.axhline(SEED_BAR, color="navy", ls=":", lw=1.6,
               label=f"seed-drift edge {SEED_BAR:.0f}")
    for x, v in zip(xs, vals):
        if np.isfinite(v):
            ax.text(x, v + 1.5, f"{v:.1f}", ha="center", fontsize=8)
        else:
            ax.text(x, 2, "off-grid", ha="center", fontsize=8, rotation=90)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("r*(k) x k")
    ax.set_title("products vs the registered bands", fontsize=10)
    ax.legend(fontsize=8.5)

    # ---- (1,1) verdict + provenance text
    ax = axes[1, 1]
    ax.axis("off")
    b = M["base"]
    inst = M["install"]
    lines = [
        f"BASE: seed {M['seed']}, {b['steps_done']}/{b['steps_target']} steps "
        f"(COMPLETED cosine), batch {b['batch']}, {b['device']}, "
        f"{b['wall_s']:.0f}s",
        f"  val CE {b['val_ce']:.4f}  (expectation ~{VCE_EXPECT}; e053c "
        f"{E053C_VAL_CE:.4f}; 2000-step siblings 1.626-1.654)",
        f"INSTALL: e043 verbatim, s{inst['steps']}"
        f"{' [patience]' if inst['patience_used'] else ''}, {inst['device']}, "
        f"{inst['wall_s']:.0f}s -> GATE "
        f"{'PASS' if inst['gate_pass'] else 'FAIL'} "
        f"(pZ {inst['final']['install60_pz']:.3f}, R1i "
        f"{inst['final']['r1i_nll']:.2f})",
        f"M2 net: the {M['m2_which']} | instrument sub-gates "
        f"{M['gates']['m2_instrument_subgates']} "
        f"(6 cells x 2 families) | control not-applied "
        f"{M['gates']['m2_control_not_applied']}",
        "",
        f"PRODUCTS:  k=64: {p64 if p64 is None else round(p64, 1)}  "
        f"[{dec['product_status']['64']}]   k=128: "
        f"{p128 if p128 is None else round(p128, 1)}  "
        f"[{dec['product_status']['128']}]",
        f"  vs maturity band [{MAT_LO:.1f}, {MAT_HI:.1f}]  and  seed-drift "
        f"edge <= {SEED_BAR:.0f} (siblings ±25% [{SIB_LO:.1f}, {SIB_HI:.1f}])",
        f"  net constant (mean): {dec['net_constant']}",
        "",
        f"REFERENCES: s4305@2000 = 31.4 | s4306@2000 = 42.2 | "
        f"e053c@3133 = 54.4 (the 54)",
        "",
        f"DECISION [{dec['clause']}]:",
    ] + [f"  {w}" for w in _wrap(dec["verdict"], 96)]
    ax.text(0.02, 0.97, f"E117 — the share-constant maturity discriminator "
                        f"(seed {M['seed']})", fontsize=13, weight="bold",
            va="top")
    for i, t in enumerate(lines):
        ax.text(0.02, 0.925 - i * 0.036, t, fontsize=8.8, va="top",
                family="monospace")

    fig.suptitle(f"E117 — does the share constant track TRAINING EXPOSURE? "
                 f"(fresh seed at e053c's exact 3133 steps) | "
                 f"{dec['clause']}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _wrap(text: str, width: int):
    import textwrap
    return textwrap.wrap(text, width=width)


if __name__ == "__main__":
    sys.exit(main())
