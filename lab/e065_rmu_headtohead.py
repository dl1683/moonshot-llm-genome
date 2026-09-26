"""E065 — RMU-style unlearning vs address surgery: the obfuscation head-to-head
(REGISTERED per scratch/e065_rmu_headtohead_design.md, critic-hardened 5f11a79).

Question: does RMU-style representation-level unlearning OBFUSCATE (knowledge
intact, readout misdirected, reverts in ~50 retraining steps) while parametric
address surgery DELETES (unrescuable, but re-learning is scarred/slow)? The
2025 at-scale consensus (frontier scan: "LLM Unlearning Under the Microscope",
"Beyond Data Filtering") says RMU obfuscates; nobody has run both on the same
net with causal instruments. This experiment does, at toy scale, with the
lab's existing rigs.

BASE NET: runs/checkpoints/e048_repro.pt — the G1-gated bit-level repro of
e043's Dmix@s400 name install (battery p(Z) 0.556 / argmaxZ 0.817; T015).
No new base training; only the removal fine-tunes themselves.

ARMS (design "Setup", all five):
  1. RMU-analogue (e065a): u = normalized mean dh_layer(forget - retain) at
     the target depth; fine-tune (<=2000 steps, <=180 s cap, lr 1e-3, batch
     32 = 16 F + 16 R) with RMU loss ||h' - alpha*u||^2 + retain CE on R.
     STEP 0 FIRST (critic fix 3): median ||h|| measured at the target depth,
     alpha in {1x, 2x} of it — the 7B constant {2,5} is NOT imported.
     Depth grid {d4, d5, both} REQUIRED (critic fix 5).
  2. Address surgery (e065b): e042's D2 subtractive row-reset ported — zero
     wte.weight[zid] and lm_head.weight[zid] on a copy of the installed net.
  3. Gradient-ascent control (e065c): projected ascent on F only (unit
     mean-retain-gradient direction projected out per e003b/e003c), matched
     budget (same steps/lr/batch + same early-stop rule as the selected RMU
     cell). Calibrates the instruments.
  4. Retain-only fine-tune control (critic fix 2): same steps/LR/batch on R
     only — the generic fine-tune/LR scar the RMU arm also carries; R1-R4
     null lines are read against this.
  5. No-removal control (critic fix 1): the unmodified installed net —
     guards the expression-gap confound (T015: 0.974 TF accuracy but ZERO
     free-run expression PRE-removal).

MATCHING (critic fix 2, continuous band): removal arms report their actual
forget-battery p(Z) (TF logit-gap proxy); suppression band = site-mean
p(Z) <= 0.05 (the kill threshold, reported per arm, never threshold-matched
silently).

FORGET/RETAIN SETS: F = the e043-frozen 60 install host windows (SPLICE_RNG
24301) — realized as their eliciting prompts ctx130 (the onset decision
position where the harm readout p(Z) lives; the ascent/masked cells use the
full 256-token windows with name-position masks). R = fixed val-split windows
without the name (train bank 128 + eval 60, deterministic seeds).

READOUTS (frozen pre-run, design section "Pre-registered readouts"):
  R1 suppression-depth transplant (e055 rig): TF-state donor rescue of p(Z)
    at onset position, d0-d6 sweep per condition (design says d1-d6; d0 is
    reported too as a superset — bars only ever fire in the e055 meaningful
    band d<=5, d6 readout-dominated). Donor states ALWAYS from the no-removal
    net (fixed across arms); shuffled donors from each arm's own net.
    BARS (inherited from e055 verbatim, critic fix 4): rescuable at depth d =
    site-mean TF >= 0.30 AND shuffled <= 0.05 AND site-bootstrap 95% CI on
    (TF - shuffled) excludes 0. NOT rescuable (arm) = every d<=5 site-mean
    <= 2x base-net twin (0.0062 -> 0.0124).
    H-obfuscation: RMU net still rescuable at >=1 depth; surgery not
    rescuable anywhere.
  R2 probe-vs-generation gap (Orgad instrument): linear probe (5-fold CV
    logistic, balanced acc) for "Z-follows" from onset-position activations
    at every depth — positives = the 60 install ctx130, negatives = the 30
    held-out REAL host ctx130 (never trained; content-confound controlled by
    the e001 base-net probe line) — vs actual free-run generation rate.
    CONDITIONAL (critic fix 1): if the no-removal control's generation rate
    is 0 (expected per T015), R2 is PROBE-ONLY — the gap leg is uninformative
    at floor and is not counted as evidence; dynamic range reported openly.
  R3 relearn scar (e044 battery): per removal arm, e044-protocol Dmix
    re-exposure on F (16 name windows + 48 anchors (16 paired originals +
    32 random), AdamW lr 1e-3 house cosine, cap 300 steps), dense evals
    [1..300]. Cells: the 50-STEP trajectory + the MATCHED-CE steps-to-bar
    cell (bar = spliced battery NLL <= 1.0 AND acc >= 0.8, e044's bar).
    Recovery curve + address-direction regrowth cos (wte_Z/lm_Z vs the
    installed net's rows, e044's COS_MIN_NORM_FRAC=0.25 validity guard).
    n=1 per condition, flagged as such.
    H-obfuscation: RMU reverts fast (full recovery <= 50 steps); surgery
    scarred (>= 2x steps, cos stays high — replicates e044b).
  R4 collateral (explicit bounds): val_R CE drift bands <= +0.05 nats
    "=0", 0.05-0.15 "small", > 0.15 FAILS the arm's comparability; fluency
    (mean next-token entropy on R windows) drift band +/-5%.
    Predicted: surgery ~ 0; RMU small-but-nonzero at single-depth arms.
  R5 RMU-vs-retain CONTRAST (registered 2026-09-26 AFTER the smoke preview
    showed retain-only fine-tuning ALSO suppresses p_z — before the full run;
    coordinator-registered branch): R5 = RMU-minus-retain-only delta on each
    of R1/R3/R4, with CIs (site-bootstrap for R1 per depth; window-bootstrap
    for R4; R3 is n=1-per-condition, point deltas flagged as such).
    Pre-registered interpretation:
      (a) R5 ~ 0 everywhere (all R1 depth CIs include 0 with |mean| <= 0.05
          AND R4 CI includes 0) -> "at toy scale, RMU's suppression is
          indistinguishable from generic fine-tune scar" (a real result
          echoing the 2025 shallow-suppression literature); the H-obfuscation
          case then rests ENTIRELY on the R1/R3 asymmetry vs SURGERY
          (rescuable-but-reverts vs unrescuable-but-scarred).
      (b) R5 > 0 on R1 rescue retention specifically (some d<=5 with mean
          delta >= +0.05 and bootstrap CI excluding 0) -> RMU genuinely
          preserves the knowledge better than generic FT; the obfuscation
          story strengthens.
      (c) anything else -> MIXED, reported openly.

DISCRIMINATING STRUCTURE: H-obfuscation predicts an INVERSION between R1 and
R3 (rescuable-but-reverts-fast vs unrescuable-but-relearns-slow). Any other
pattern -> report the alternative honestly.

KILL CRITERIA (design section "Kill"):
  - RMU cannot reach battery p(Z) <= 0.05 without val_R CE drift > +0.3 nats
    at any alpha/depth -> toy-scale RMU infeasible; run surgery + ascent +
    retain-only + no-removal anyway; report as negative result.
  - Surgery ALSO shows wide probe gap / fast revert -> removal claim weakens;
    flag for full audit (no paper text changes here).

COMPUTE ENVELOPE (hard rules, common.py; gate relaxed by coordinator
2026-09-26 after the 75C self-rule livelocked on ambient heat-soak — the GPU
idles at 79C with 0% util): every training launch gates on gpu_ok() VERBATIM
(util <= 85%, temp <= 80C, mem <= 85%, two polls 10 s apart); cooldown(120)
between fine-tune arms; in-loop generation throttle pauses when temp >= 80C
(current step finishes, then pause); prefer shorter bursts at the ceiling.
Wall-time stretches when thermal pauses bind (reported in timing/trims); the
registered bars and kill criteria are unchanged.

RMU CELL SELECTION (frozen before the grid runs): among feasible cells
(p_z <= 0.05 AND dCE_R <= +0.3), select min dCE_R; ties broken by lower p_z,
then alpha=1x, then depth order [d4], [d5], [both].

Fine-tune schedule note (registered deviation): the design specifies lr 1e-3
/ batch 32 but no schedule; all fine-tune arms (RMU grid, ascent, retain-only)
use CONSTANT lr 1e-3, AdamW betas (0.9,0.95) wd 0.1, clip 1.0 (house optimizer,
no cosine) so the alpha/depth effects are not confounded by schedule decay;
the R3 relearn battery alone uses e044's cosine(total=1000) verbatim.

Run:  cd lab && python e065_rmu_headtohead.py          (full -> runs/e065/)
      cd lab && E065_SMOKE=1 python e065_rmu_headtohead.py  (-> runs/e065_smoke/)

No NOTES/THINKING/QUEUE/STATE edits; no git commit (operator instruction).
"""
from __future__ import annotations

import copy
import math
import os
import random
import re
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # use the GPU (unlike e055)

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
import torch.nn.functional as F                      # noqa: E402

import common                                        # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT, cooldown, gpu_ok,        # noqa: E402
                    gpu_status, run_dir, save_json)
import e043_install as E43                           # noqa: E402 (find_occ, SPLICE_RNG, jsonable)
import e055_suppression as E55                       # noqa: E402 (stats: auc, site_bootstrap, depth_stats)

import matplotlib.pyplot as plt                      # noqa: E402

SMOKE = os.environ.get("E065_SMOKE") == "1"
DEV = common.DEVICE

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
HOSTS = ["FLORIZEL", "ELIZABETH"]
HOST_RE = re.compile(r"(?<![A-Za-z])(ELIZABETH|FLORIZEL)(?![A-Za-z])")
ZEPH_RE = re.compile(r"(?<![A-Za-z])ZEPHYRA(?![A-Za-z])")
DEPTHS = list(range(7))
MEANINGFUL_BAND = list(range(6))          # d <= 5 (e055 addendum; d6 readout-dominated)
BASE_TWIN = 0.0062                        # e055 base-net twin (frozen bar constant)
NOT_RESCUABLE_MAX = 2 * BASE_TWIN         # 0.0124
RESCUE_BAR = 0.30
SHUF_BAR = 0.05

# fine-tune envelope
FT_LR = 1e-3
FT_STEPS_CAP = 2000 if not SMOKE else 8
FT_TIME_CAP = 180.0
FT_EVAL_EVERY = 25
SUPPRESS_BAR = 0.05                       # battery p(Z) suppression band / kill bar
KILL_DCE = 0.3                            # val_R CE kill bar

# relearn battery (e044 verbatim)
RL_STEPS_CAP = 300 if not SMOKE else 6
RL_EVALS = ([1, 2, 3, 4, 6, 8, 12, 16, 25, 35, 50, 75, 100, 150, 200, 300]
            if not SMOKE else [1, 2, 3, 4, 6])
RL_BAR_NLL, RL_BAR_ACC = 1.0, 0.8
COS_MIN_NORM_FRAC = 0.25                  # e044 Review-6 validity guard
E044_GROOVE_COS = 0.760                   # e044 re-regrowth cos (surgery reference)
E044_FRESH_COS = 0.278                    # e044 fresh-from-zero cos baseline

# seeds
R_BANK_SEED, R_CTX_SEED, R_EVAL_SEED = 26500, 26501, 26502
RMU_SEED0 = 26510
ASCENT_SEED, RETAIN_SEED = 26520, 26530
RL_SEED0 = 26540
PROBE_SEED = 26600
GEN_CHECK_SEEDS = (5000,)            # 8 prompts x 350 = 2,800 chars = T015 volume
SITE_SEED0 = 5000                         # e055's trajectory seed family
SHUF_SEED = 25501                         # e055's shuffled-context family
N_SHUF = 4
SITE_MIN = 16 if not SMOKE else 1
SITE_ENOUGH = 20
GEN_TOK = 350
TEMP, TOPK = 0.8, 40
N_BOOT = 10000

trims: list[str] = []
deviations: list[str] = [
    "R1 swept d0-d6 (design says d1-d6; d0 reported as a superset — bars only "
    "fire in the d<=5 meaningful band per the e055 addendum).",
    "RMU forget prompts = the install pre-contexts ctx130 with the RMU term at "
    "the onset decision position (pos 129) — the faithful toy analogue of RMU's "
    "last-token h; the name-window F is realized as its eliciting prompt.",
    "Fine-tune schedule = constant lr 1e-3 (house AdamW 0.9/0.95 wd 0.1 clip "
    "1.0) for RMU grid/ascent/retain-only; e044 cosine kept for the R3 "
    "relearn battery only.",
]


# ------------------------------------------------------------------ gpu guard

THERMAL_GATE_C = 80.0      # coordinator-relaxed 2026-09-26: envelope rule
                           # verbatim (gpu_ok: util<=85, temp<=80, mem<=85)
THERMAL_PAUSE_C = 80.0     # in-loop generation throttle at the ceiling:
                           # finish the current step, then pause while >=80C


def _temp() -> float:
    return gpu_status()["temp"]


def _wait_cool(target: float, what: str) -> None:
    """Block until temp <= target on two consecutive polls 15 s apart."""
    n_ok, waited = 0, 0.0
    while n_ok < 2:
        t = _temp()
        if t <= target:
            n_ok += 1
        else:
            n_ok = 0
        if n_ok < 2:
            if waited % 120 < 1e-9 and waited > 0:
                log(f"[thermal] {what}: still {t:.0f}C (target <= {target:.0f}C, "
                    f"waited {waited:.0f}s)")
            time.sleep(15)
            waited += 15


def gate_launch(tag: str) -> dict:
    """Hard gate: envelope rule verbatim — launch only when gpu_ok()
    (util<=85%, temp<=80C, mem<=85%) holds on two polls 10 s apart.
    Blocks indefinitely while an external consumer heat-soaks the GPU
    (2026-09-26 relaxation: gate on gpu_ok, not the stricter 75C self-rule)."""
    while True:
        if gpu_ok():
            time.sleep(10)
            s2 = gpu_status()
            if gpu_ok():
                log(f"[gpu] launch '{tag}' ok (util {s2['util']:.0f}% temp "
                    f"{s2['temp']:.0f}C mem {s2['mem_used']:.0f}/"
                    f"{s2['mem_total']:.0f}MB)")
                return s2
        time.sleep(10)


def cool() -> None:
    """Inter-arm cooldown (relaxed): cooldown(120), then block until the
    envelope rule holds again (gpu_ok verbatim)."""
    cooldown(120.0)
    while not gpu_ok():
        time.sleep(15)
    log(f"[thermal] cool() done ({gpu_status()['temp']:.0f}C)")


def thermal_throttle(k=20):
    """Call inside long generation loops every k tokens: pause while hot
    (>= 80C ceiling; the current step finishes, then the loop pauses)."""
    global _TOK_SINCE_CHECK
    _TOK_SINCE_CHECK = _TOK_SINCE_CHECK + 1
    if _TOK_SINCE_CHECK < k:
        return
    _TOK_SINCE_CHECK = 0
    while gpu_status()["temp"] >= THERMAL_PAUSE_C:
        log(f"[thermal] generation pause (temp {gpu_status()['temp']:.0f}C >= "
            f"{THERMAL_PAUSE_C:.0f}C)")
        time.sleep(20)


_TOK_SINCE_CHECK = 0


# ------------------------------------------------------------------ instruments

def load(path: Path) -> TinyGPT:
    m = TinyGPT(Cfg())
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


def to_dev(net: TinyGPT) -> TinyGPT:
    return net.to(DEV)


@torch.no_grad()
def states_forward(net, toks):
    """e055 states_forward, device-corrected. toks: (B,T) -> xs[0]=emb, xs[k]=block k-1 out."""
    T = toks.shape[1]
    x = net.wte(toks) + net.wpe(torch.arange(T, device=toks.device))
    xs = [x]
    for block in net.h:
        x = block(x)
        xs.append(x)
    return xs, net.lm_head(net.ln_f(x))


@torch.no_grad()
def patch_logits_batch(net, toks, pos, depth, states):
    """e055 patch_logits_batch, device-corrected."""
    B, T = toks.shape
    x = net.wte(toks) + net.wpe(torch.arange(T, device=toks.device))

    def put():
        x2 = x.clone()
        x2[:, pos] = states
        return x2

    if depth == 0:
        x = put()
    for i, block in enumerate(net.h):
        x = block(x)
        if i + 1 == depth:
            x = put()
    return net.lm_head(net.ln_f(x))


def pz_next(logits, zid):
    return F.softmax(logits[:, -1], -1)[:, zid]


@torch.no_grad()
def gen_batch(net, idx0, n_new, temperature=TEMP, top_k=TOPK, throttle=True):
    """e055 gen_batch, device-corrected (idx0 already on DEV) + thermal throttle."""
    idx = idx0.clone()
    T0len = idx0.shape[1]
    for _ in range(n_new):
        if throttle:
            thermal_throttle()
        cond = idx[:, -net.cfg.block_size:]
        logits, _ = net(cond)
        lg = logits[:, -1] / temperature
        v, _ = torch.topk(lg, min(top_k, lg.size(-1)))
        lg = lg.masked_fill(lg < v[:, [-1]], float("-inf"))
        idx = torch.cat([idx, torch.multinomial(F.softmax(lg, -1), 1)], 1)
    return idx[:, T0len:]


@torch.no_grad()
def battery_pz(net, ctx_ids, zid, bs=30):
    """Mean/argmax p(Z) at last position over battery contexts (B,T ints on CPU)."""
    pzs, am = [], 0.0
    for i in range(0, len(ctx_ids), bs):
        x = ctx_ids[i:i + bs].to(DEV)
        lg, _ = net(x)
        pr = F.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
        am += float((lg[:, -1].argmax(-1) == zid).float().sum())
    p = torch.cat(pzs)
    return {"p_z_mean": float(p.mean()), "frac_argmax_z": am / len(ctx_ids),
            "p_z_min": float(p.min()), "p_z_max": float(p.max())}


@torch.no_grad()
def ce_fixed(net, x, y, bs=64):
    """e043 ce_fixed (x,y on CPU)."""
    net.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        _, loss = net(x[i:i + bs].to(DEV), y[i:i + bs].to(DEV))
        tot += float(loss.item()) * len(x[i:i + bs])
        n += len(x[i:i + bs])
    return tot / max(n, 1)


@torch.no_grad()
def mean_entropy(net, x, bs=64):
    """Mean next-token entropy (nats) over all positions of eval windows."""
    net.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        lg, _ = net(x[i:i + bs].to(DEV))
        p = F.softmax(lg, -1)
        h = -(p * (p + 1e-12).log()).sum(-1)
        tot += float(h.sum()) * 1
        n += int(h.numel())
    return tot / max(n, 1)


@torch.no_grad()
def name_battery(net, win_x, win_y, mask, chunk=64):
    """Name-position NLL/acc on the install windows (e044 'spliced' battery)."""
    net.eval()
    nlls, accs = [], []
    for i in range(0, len(win_x), chunk):
        xc, yc, mc = win_x[i:i + chunk].to(DEV), win_y[i:i + chunk].to(DEV), mask[i:i + chunk].to(DEV)
        logits, _ = net(xc)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yc.reshape(-1),
                              reduction="none").view(xc.shape[0], xc.shape[1])
        nm = nll[mc]
        am = (logits.argmax(-1) == yc)
        nlls.append(nm)
        accs.append(am[mc])
    nll_m, acc_m = torch.cat(nlls), torch.cat(accs)
    return {"n": int(nll_m.numel()), "nll": float(nll_m.mean()),
            "acc": float(acc_m.float().mean())}


@torch.no_grad()
def ce_windows(net, x, y, bs=64):
    """Per-window CE (nats) over eval windows — R5.R4 bootstrap input."""
    net.eval()
    out = []
    for i in range(0, len(x), bs):
        xb, yb = x[i:i + bs].to(DEV), y[i:i + bs].to(DEV)
        logits, _ = net(xb)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1),
                              reduction="none").view(xb.shape[0], -1)
        out.extend(float(v) for v in nll.mean(-1))
    return np.array(out)


def row_probe(net, zid, orig=None):
    wte, lm = net.wte.weight[zid], net.lm_head.weight[zid]
    out = {"wte_norm": float(wte.norm().item()), "lm_norm": float(lm.norm().item())}
    if orig is not None:
        out["cos_wte"] = float(F.cosine_similarity(wte, orig[0], dim=0).item())
        out["cos_lm"] = float(F.cosine_similarity(lm, orig[1], dim=0).item())
    return out


# ------------------------------------------------------------------ RMU machinery

def build_u_alpha(net0, f_ids, r_ids, depth_specs):
    """STEP 0 (critic fix 3): measure median ||h|| at target depths and build
    u = normalized mean (h_F - h_R) per depth. Eval-only, seconds."""
    out = {}
    with torch.no_grad():
        hsF, _ = states_forward(net0, f_ids.to(DEV))
        hsR, _ = states_forward(net0, r_ids.to(DEV))
    for d in depth_specs:
        hf, hr = hsF[d][:, -1], hsR[d][:, -1]
        u = hf.mean(0) - hr.mean(0)
        u = u / u.norm()
        med_f = float(hf.norm(dim=-1).median())
        med_r = float(hr.norm(dim=-1).median())
        out[d] = {"u": u.detach(), "median_norm_forget": med_f,
                  "median_norm_retain": med_r}
    return out


def rmu_finetune(tag, net0, ua, depths, alpha_mult, f_bank, r_bank, r_eval_xy,
                 zid, ce0, steps_cap=FT_STEPS_CAP, time_cap=FT_TIME_CAP, seed=0):
    """RMU fine-tune: loss = mean_d ||h_d(pos129) - alpha_d u_d||^2 + CE on R.
    Batch 32 = 16 F (ctx130) + 16 R (256-tok windows). Constant lr 1e-3.
    Early-stop requires the kill pair SUSTAINED (p_z<=0.05 AND dCE_R<=+0.3 at
    2 consecutive evals) so a half-wrecked net never counts as suppressed."""
    gate_launch(tag)
    net = copy.deepcopy(net0)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    alphas = {d: alpha_mult * ua[d]["median_norm_forget"] for d in depths}
    us = {d: ua[d]["u"].clone() for d in depths}
    traj, t_start = [], time.time()
    n_ok, step = 0, 0
    evals = {1, 2, 4, 8, 16}
    f_ids_all = f_bank                                   # (60,130) CPU
    n_f, n_r = len(f_ids_all), len(r_bank)
    for step in range(1, steps_cap + 1):
        fi = torch.randint(n_f, (16,), generator=gen)
        ri = torch.randint(n_r, (16,), generator=gen)
        fx = f_ids_all[fi].to(DEV)
        rx = r_bank[ri][:, :-1].to(DEV)
        ry = r_bank[ri][:, 1:].to(DEV)
        # forward forget contexts, collect h at target depths (pos 129)
        T = fx.shape[1]
        x = net.wte(fx) + net.wpe(torch.arange(T, device=DEV))
        mse = 0.0
        di = 0
        for i, block in enumerate(net.h):
            x = block(x)
            if (i + 1) in depths:
                h = x[:, PRE - 1]
                mse = mse + ((h - alphas[i + 1] * us[i + 1]) ** 2).mean()
                di += 1
        logits_r, _ = net(rx)
        ce = F.cross_entropy(logits_r.reshape(-1, logits_r.shape[-1]), ry.reshape(-1))
        loss = mse / max(di, 1) + ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step in evals or step % FT_EVAL_EVERY == 0 or step == steps_cap \
                or (time.time() - t_start) > time_cap:
            net.eval()
            bz = battery_pz(net, f_ids_all, zid)
            ce_r = ce_fixed(net, *r_eval_xy)
            net.train()
            traj.append({"step": step, "p_z_mean": bz["p_z_mean"],
                         "frac_argmax_z": bz["frac_argmax_z"],
                         "ce_r": ce_r, "mse": float(mse.item()) if not isinstance(mse, float) else mse,
                         "ce": float(ce.item()),
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] s{step:4d} p_z {bz['p_z_mean']:.4f} argmaxZ "
                f"{bz['frac_argmax_z']:.2f} CE_R {ce_r:.4f} (dCE {ce_r - ce0:+.3f}) "
                f"({traj[-1]['elapsed_s']:.0f}s)")
            if bz["p_z_mean"] <= SUPPRESS_BAR and ce_r - ce0 <= KILL_DCE:
                n_ok += 1
                if n_ok >= 2:            # sustained at 2 consecutive evals
                    log(f"  [{tag}] early-stop: suppression+CE-ok sustained at s{step}")
                    break
            else:
                n_ok = 0
        if (time.time() - t_start) > time_cap:
            log(f"  [{tag}] time cap {time_cap:.0f}s at s{step}")
            break
    net.eval()
    final = traj[-1] if traj else {}
    feas = bool(traj and final.get("p_z_mean", 1.0) <= SUPPRESS_BAR
                and final.get("ce_r", 99.0) - ce0 <= KILL_DCE)
    return {"net": net, "tag": tag, "depths": list(depths), "alpha_mult": alpha_mult,
            "alphas": {d: float(alphas[d]) for d in depths}, "traj": traj,
            "steps_ran": step, "final": final, "feasible": feas}


# ------------------------------------------------------------------ ascent machinery

def flat_params(net):
    return [p for p in net.parameters()]


def mean_retain_direction(net0, r_bank, zid_unused=None, k=8, seed=26521):
    """e003b b_direction: unit mean gradient of CE on R over k batches (frozen)."""
    net = copy.deepcopy(net0)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR)
    gen = torch.Generator().manual_seed(seed)
    acc = [torch.zeros_like(p) for p in flat_params(net)]
    for _ in range(k):
        ri = torch.randint(len(r_bank), (16,), generator=gen)
        rx = r_bank[ri][:, :-1].to(DEV)
        ry = r_bank[ri][:, 1:].to(DEV)
        opt.zero_grad(set_to_none=True)
        logits, _ = net(rx)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ry.reshape(-1))
        loss.backward()
        for a, p in zip(acc, flat_params(net)):
            if p.grad is not None:
                a += p.grad.detach()
    nrm = torch.sqrt(sum((a * a).sum() for a in acc))
    return [a / nrm for a in acc]


def ascent_finetune(tag, net0, u_ret, inst_x, inst_mask, r_eval_xy, f_eval,
                    zid, ce0, steps_cap, seed):
    """Projected ascent on F (e003b/e003c): step along -grad(CE_name) with the
    unit mean-retain-gradient direction projected out. Batch 32 F windows
    (full 256-tok windows, name-position masked). Same early-stop rule as the
    RMU cells (matched budget)."""
    gate_launch(tag)
    net = copy.deepcopy(net0)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    traj, t_start = [], time.time()
    n_ok, step = 0, 0
    evals = {1, 2, 4, 8, 16}
    n_f = len(inst_x)
    for step in range(1, steps_cap + 1):
        fi = torch.randint(n_f, (32,), generator=gen)
        x = inst_x[fi][:, :-1].to(DEV)
        y = inst_x[fi][:, 1:].to(DEV)
        m = inst_mask[fi].to(DEV)
        logits, _ = net(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                              reduction="none").view(x.shape[0], x.shape[1])
        loss = -nll[m].mean()                       # ASCEND the name CE
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # project out the retain direction (e003b unit u_B rule)
        with torch.no_grad():
            dots = sum(float((p.grad * u).sum().item()) for p, u in
                       zip(flat_params(net), u_ret) if p.grad is not None)
            for p, u in zip(flat_params(net), u_ret):
                if p.grad is not None:
                    p.grad -= dots * u
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step in evals or step % FT_EVAL_EVERY == 0 or step == steps_cap \
                or (time.time() - t_start) > FT_TIME_CAP:
            net.eval()
            bz = battery_pz(net, f_eval, zid)
            nb = name_battery(net, inst_x[:, :-1], inst_x[:, 1:], inst_mask)
            ce_r = ce_fixed(net, *r_eval_xy)
            net.train()
            traj.append({"step": step, "p_z_mean": bz["p_z_mean"],
                         "name_nll": nb["nll"], "ce_r": ce_r,
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] s{step:4d} p_z {bz['p_z_mean']:.4f} nameNLL "
                f"{nb['nll']:.2f} CE_R {ce_r:.4f} ({traj[-1]['elapsed_s']:.0f}s)")
            if bz["p_z_mean"] <= SUPPRESS_BAR and ce_r - ce0 <= KILL_DCE:
                n_ok += 1
                if n_ok >= 2:
                    log(f"  [{tag}] early-stop: suppression+CE-ok sustained at s{step}")
                    break
            else:
                n_ok = 0
        if (time.time() - t_start) > FT_TIME_CAP:
            log(f"  [{tag}] time cap at s{step}")
            break
    net.eval()
    return {"net": net, "tag": tag, "traj": traj, "steps_ran": step,
            "final": traj[-1] if traj else {}}


def retain_finetune(tag, net0, r_bank, r_eval_xy, f_eval, zid, steps_cap, seed):
    """Retain-only control: same steps/LR/batch on R only (critic fix 2)."""
    gate_launch(tag)
    net = copy.deepcopy(net0)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    traj, t_start = [], time.time()
    step = 0
    evals = {1, 2, 4, 8, 16}
    for step in range(1, steps_cap + 1):
        ri = torch.randint(len(r_bank), (32,), generator=gen)
        x = r_bank[ri][:, :-1].to(DEV)
        y = r_bank[ri][:, 1:].to(DEV)
        logits, _ = net(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step in evals or step % FT_EVAL_EVERY == 0 or step == steps_cap \
                or (time.time() - t_start) > FT_TIME_CAP:
            net.eval()
            bz = battery_pz(net, f_eval, zid)
            ce_r = ce_fixed(net, *r_eval_xy)
            net.train()
            traj.append({"step": step, "p_z_mean": bz["p_z_mean"], "ce_r": ce_r,
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] s{step:4d} p_z {bz['p_z_mean']:.4f} CE_R {ce_r:.4f} "
                f"({traj[-1]['elapsed_s']:.0f}s)")
        if (time.time() - t_start) > FT_TIME_CAP:
            break
    net.eval()
    return {"net": net, "tag": tag, "traj": traj, "steps_ran": step,
            "final": traj[-1] if traj else {}}


# ------------------------------------------------------------------ relearn (e044)

def relearn_battery(tag, net_arm, inst_x, inst_mask, anchor, train_ids,
                    f_eval_ids, r_eval_xy, zid, orig_rows, seed):
    """e044 exposure (Dmix) verbatim-style: 16 name windows + 48 anchors
    (16 paired originals + 32 random corpus), AdamW lr 1e-3 cosine(1000),
    cap RL_STEPS_CAP steps, dense evals. Returns traj + steps_to_bar."""
    gate_launch(tag)
    net = copy.deepcopy(net_arm)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    from common import cosine_lr
    NAME_BS, CORP_BS, MIX_RANDOM, BLOCK = 16, 48, 32, 256
    traj, t_start = [], time.time()
    steps_to_bar = None
    eval_set = set(RL_EVALS)
    n_inst, n_anc = inst_x.shape[0], anchor.shape[0]
    for step in range(1, RL_STEPS_CAP + 1):
        f = cosine_lr(step - 1, 1000)
        for g in opt.param_groups:
            g["lr"] = FT_LR * f
        ix = torch.randint(n_inst, (NAME_BS,), generator=gen)
        aj = torch.randint(n_anc, (CORP_BS - MIX_RANDOM,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (MIX_RANDOM,), generator=gen)
        corp = torch.cat([anchor[aj],
                          torch.stack([train_ids[s: s + BLOCK] for s in rj]).to(DEV)], 0)
        nw = inst_x[ix].to(DEV)
        x = torch.cat([nw[:, :-1], corp[:, :-1]], 0)
        y = torch.cat([nw[:, 1:], corp[:, 1:]], 0)
        m = torch.zeros(NAME_BS + CORP_BS, x.shape[1], dtype=torch.bool, device=DEV)
        m[:NAME_BS] = inst_mask[ix].to(DEV)
        logits, _ = net(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                              reduction="none").view(x.shape[0], x.shape[1])
        nm = nll[:NAME_BS][m[:NAME_BS]]
        cm = nll[NAME_BS:]
        loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step in eval_set or step == RL_STEPS_CAP or (time.time() - t_start) > FT_TIME_CAP:
            net.eval()
            nb = name_battery(net, inst_x[:, :-1], inst_x[:, 1:], inst_mask)
            bz = battery_pz(net, f_eval_ids, zid)
            ce_r = ce_fixed(net, *r_eval_xy)
            rp = row_probe(net, zid, orig_rows)
            net.train()
            row = {"step": step, "name_nll": nb["nll"], "name_acc": nb["acc"],
                   "p_z_mean": bz["p_z_mean"], "ce_r": ce_r, **rp,
                   "elapsed_s": round(time.time() - t_start, 1)}
            traj.append(row)
            log(f"  [{tag}] rl s{step:3d} NLL {nb['nll']:.2f} acc {nb['acc']:.2f} "
                f"p_z {bz['p_z_mean']:.3f} CE_R {ce_r:.4f} "
                f"wte|{rp['wte_norm']:.2f}|cos{rp['cos_wte']:+.2f} "
                f"lm|{rp['lm_norm']:.2f}|cos{rp['cos_lm']:+.2f}")
            if steps_to_bar is None and nb["nll"] <= RL_BAR_NLL and nb["acc"] >= RL_BAR_ACC:
                steps_to_bar = step
                log(f"  [{tag}] steps_to_bar = {step}")
                if not SMOKE:
                    break                     # matched-CE cell complete
            if (time.time() - t_start) > FT_TIME_CAP:
                log(f"  [{tag}] relearn time cap at s{step}")
                break
    net.eval()
    return {"traj": traj, "steps_to_bar": steps_to_bar, "steps_ran": step,
            "net": net}


# ------------------------------------------------------------------ probe (R2)

def probe_bacc(X_pos, X_neg, seed=PROBE_SEED, folds=5):
    """5-fold stratified CV logistic probe; balanced accuracy."""
    if SMOKE:
        folds = 2
    X = torch.cat([X_pos, X_neg]).float()
    y = torch.cat([torch.ones(len(X_pos)), torch.zeros(len(X_neg))])
    n_pos, n_neg = len(X_pos), len(X_neg)
    rng = np.random.default_rng(seed)
    pos_idx = rng.permutation(n_pos)
    neg_idx = rng.permutation(n_neg)
    baccs = []
    for k in range(folds):
        te_pos = pos_idx[k::folds]
        te_neg = neg_idx[k::folds]
        tr_pos = np.setdiff1d(pos_idx, te_pos)
        tr_neg = np.setdiff1d(neg_idx, te_neg)
        tr = torch.cat([torch.as_tensor(tr_pos), n_pos + torch.as_tensor(tr_neg)])
        te = torch.cat([torch.as_tensor(te_pos), n_pos + torch.as_tensor(te_neg)])
        lin = torch.nn.Linear(X.shape[1], 1).to(DEV)
        optp = torch.optim.Adam(lin.parameters(), lr=0.05, weight_decay=1e-2)
        for _ in range(200):
            optp.zero_grad()
            lg = lin(X[tr]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(lg, y[tr])
            loss.backward()
            optp.step()
        with torch.no_grad():
            pred = (lin(X[te]).squeeze(-1) > 0).float()
        acc_p = float((pred[:len(te_pos)] == 1).float().mean())
        acc_n = float((pred[len(te_pos):] == 0).float().mean())
        baccs.append(0.5 * (acc_p + acc_n))
    return float(np.mean(baccs))


# ------------------------------------------------------------------ R5 (registered contrast)

def compute_r5(r1, r3, r4, arms, r_eval_xy, n_boot=N_BOOT, seed=26700):
    """R5 = RMU-minus-retain-only delta on R1/R3/R4 with CIs (registered
    2026-09-26 pre-full-run; see header). Returns None if either arm absent."""
    if "rmu" not in arms or "retain_only" not in arms:
        return {"void": True,
                "note": "R5 requires both the rmu and retain_only arms "
                        "(RMU kill voids the contrast)"}
    rng = np.random.default_rng(seed)
    # R1: per-site TF means per depth; paired site bootstrap of the delta
    r1_delta = {}
    for d in DEPTHS:
        mr = np.array([np.mean(v[d]) for v in r1["rmu"]["per_site_tf"]])
        mt = np.array([np.mean(v[d]) for v in r1["retain_only"]["per_site_tf"]])
        dd = mr - mt
        boots = [float(np.mean(dd[rng.integers(0, len(dd), len(dd))]))
                 for _ in range(n_boot)]
        r1_delta[d] = {"mean_delta": float(dd.mean()),
                       "ci": [float(np.percentile(boots, 2.5)),
                              float(np.percentile(boots, 97.5))]}
    # R4: per-window CE bootstrap of the delta
    cw_r = ce_windows(arms["rmu"]["net"], *r_eval_xy)
    cw_t = ce_windows(arms["retain_only"]["net"], *r_eval_xy)
    dd4 = cw_r - cw_t
    boots4 = [float(np.mean(dd4[rng.integers(0, len(dd4), len(dd4))]))
              for _ in range(n_boot)]
    r4_delta = {"mean_delta": float(dd4.mean()),
                "ci": [float(np.percentile(boots4, 2.5)),
                       float(np.percentile(boots4, 97.5))]}
    # R3: point deltas, n=1 per condition (flagged)
    def stb(a):
        return (r3.get(a) or {}).get("steps_to_bar")
    def acc50(a):
        p = (r3.get(a) or {}).get("p50")
        return p["name_acc"] if p else None
    r3_delta = {"steps_to_bar_delta": None if stb("rmu") is None or stb("retain_only") is None
                else stb("rmu") - stb("retain_only"),
                "acc50_delta": None if acc50("rmu") is None or acc50("retain_only") is None
                else acc50("rmu") - acc50("retain_only"),
                "n1_flag": "R3 cells are n=1 per condition; point deltas only"}
    # registered verdict branches (header R5 (a)/(b)/(c))
    zero_everywhere = all(abs(r1_delta[d]["mean_delta"]) <= 0.05
                          or r1_delta[d]["ci"][0] <= 0 <= r1_delta[d]["ci"][1]
                          for d in DEPTHS) and \
        (abs(r4_delta["mean_delta"]) <= 0.05
         or r4_delta["ci"][0] <= 0 <= r4_delta["ci"][1])
    positive_depths = [d for d in MEANINGFUL_BAND
                       if r1_delta[d]["mean_delta"] >= 0.05
                       and r1_delta[d]["ci"][0] > 0]
    if zero_everywhere and not positive_depths:
        branch = ("R5_ZERO: at toy scale, RMU's suppression is indistinguishable "
                  "from generic fine-tune scar (2025 shallow-suppression echo); "
                  "the H-obfuscation case rests entirely on the R1/R3 asymmetry "
                  "vs SURGERY")
    elif positive_depths:
        branch = (f"R5_R1_POSITIVE at depths {positive_depths}: RMU preserves "
                  f"the knowledge (rescue retention) better than generic FT; "
                  f"the obfuscation story strengthens")
    else:
        branch = ("R5_MIXED: neither registered branch; reported openly "
                  "(write the alternative before any rerun)")
    return {"void": False, "r1_delta_by_depth": r1_delta, "r4_delta": r4_delta,
            "r3_delta": r3_delta, "positive_depths": positive_depths,
            "zero_everywhere": bool(zero_everywhere), "branch": branch}


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e065_smoke" if SMOKE else "e065")
    log(f"E065 RMU-vs-surgery head-to-head (smoke={SMOKE}) -> {rd}")

    # initial thermal wait: no GPU work while hot (80C ceiling enforcement;
    # blocks indefinitely — an external GPU consumer may hold the temp up)
    _wait_cool(THERMAL_PAUSE_C, "pre-run")

    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids = corpus.train
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_ids = corpus.val
    val_text = "".join(itos[int(i)] for i in val_ids)

    CK = E43.REPO / "runs" / "checkpoints"
    net0 = to_dev(load(CK / "e048_repro.pt"))      # the installed net (no-removal ctrl)
    net_base = to_dev(load(CK / "e001.pt"))        # destruction/confound reference
    log(f"nets loaded (installed/base); params {net0.num_params():,}")

    # ---------------- protocol rebuild (e043-frozen; splice RNG 24301)
    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + 119 <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    assert (sum(1 for _, h in install_occ if h == "FLORIZEL"),
            sum(1 for _, h in install_occ if h == "ELIZABETH")) == (19, 41)
    ctx130_i = [train_text[p - PRE: p] for p, _ in install_occ]      # F prompts
    held_ctx_i = [train_text[p - PRE: p] for p, _ in held_occ]       # probe negatives
    gen_prompts = ([train_text[p - 120: p] for p, _ in install_occ[:4]]
                   + [train_text[p - 120: p] for p, _ in held_occ[:4]])[:8]
    # install windows: 130 pre + ZEPHYRA(7) + 119 post = 256 (e043 deviation-1 form)
    name_ids = torch.tensor([stoi[c] for c in NAME], dtype=torch.long)
    inst_x = torch.stack([torch.cat([train_ids[p - PRE: p], name_ids,
                                     train_ids[p + len(NAME): p + len(NAME) + 119]])
                          for p, _ in install_occ])
    inst_y = inst_x[:, 1:]
    inst_xin = inst_x[:, :-1]
    inst_mask = torch.zeros(len(inst_x), inst_xin.shape[1], dtype=torch.bool)
    inst_mask[:, PRE - 1: PRE - 1 + len(NAME)] = True   # predict Z..A at y-pos 129..135
    # paired original anchors (the real host windows, e043's 16 paired originals)
    anchor = torch.stack([train_ids[p - PRE: p - PRE + 256] for p, _ in install_occ[:16]])
    log(f"protocol rebuilt: F 60 windows + 60 ctx130, held 30, anchors 16")

    # ---------------- retain sets (val split, name-free)
    def val_windows(n, seed, block=256):
        g = torch.Generator().manual_seed(seed)
        out_x, out_y = [], []
        tries = 0
        while len(out_x) < n and tries < 500 * n:
            i = int(torch.randint(len(val_ids) - block - 1, (1,), generator=g))
            txt = val_text[i: i + block + 1]
            tries += 1
            if "ZEPHYRA" in txt or "ZEPH" in txt:
                continue
            out_x.append(val_ids[i: i + block])
            out_y.append(val_ids[i + 1: i + 1 + block])
        return torch.stack(out_x), torch.stack(out_y)

    r_bank, _ = val_windows(128, R_BANK_SEED)                       # training bank
    r_eval_x, r_eval_y = val_windows(60, R_EVAL_SEED)               # CE_R eval
    r_eval_xy = (r_eval_x, r_eval_y)
    _s = random.Random(R_CTX_SEED)
    r_ctx130 = []
    while len(r_ctx130) < 64:
        q = _s.randrange(PRE + 1, len(val_ids) - 1)
        r_ctx130.append(val_text[q - PRE: q])
    f_ids = torch.stack([corpus.encode(c) for c in ctx130_i])
    h_ids = torch.stack([corpus.encode(c) for c in held_ctx_i])
    r_ids = torch.stack([corpus.encode(c) for c in r_ctx130])
    log(f"retain sets: bank {len(r_bank)}, eval {len(r_eval_x)}, ctx {len(r_ctx130)}")

    # ---------------- instrument gate: installed battery reproduces e048 ref
    bz0 = battery_pz(net0, f_ids, zid)
    G_INST = {"p_z_mean": bz0["p_z_mean"], "ref": 0.556,
              "pass": bool(abs(bz0["p_z_mean"] - 0.556) < 0.02)}
    log(f"G_INST installed battery p(Z) {bz0['p_z_mean']:.4f} argmaxZ "
        f"{bz0['frac_argmax_z']:.3f} (ref .556/.817): "
        f"{'PASS' if G_INST['pass'] else 'FAIL'}")
    ce_r0 = ce_fixed(net0, *r_eval_xy)
    ent0 = mean_entropy(net0, r_eval_x)
    log(f"no-removal baseline: CE_R {ce_r0:.4f} entropy {ent0:.4f}")

    # determinism spot-check (G5-style)
    torch.manual_seed(97001)
    p_ids = torch.stack([corpus.encode(p_) for p_ in gen_prompts]).to(DEV)
    _r1 = [corpus.decode(r.tolist()) for r in gen_batch(net0, p_ids, 60)]
    torch.manual_seed(97001)
    _r2 = [corpus.decode(r.tolist()) for r in gen_batch(net0, p_ids, 60)]
    G_DET = {"pass": bool(_r1 == _r2)}
    log(f"G_DET generation determinism: {'PASS' if G_DET['pass'] else 'FAIL'}")

    # ---------------- STEP 0: alpha-norm measurement (critic fix 3)
    depth_grid = [[4], [5], [4, 5]]
    all_ds = sorted({d for ds in depth_grid for d in ds})
    ua = build_u_alpha(net0, f_ids, r_ids, all_ds)
    alpha_report = {d: {"median_norm_forget": ua[d]["median_norm_forget"],
                        "median_norm_retain": ua[d]["median_norm_retain"]}
                    for d in all_ds}
    log("STEP-0 alpha-norm (median ||h|| at target depth, forget set): "
        + " ".join(f"d{d}: {ua[d]['median_norm_forget']:.3f}" for d in all_ds)
        + "  -> alpha in {1x, 2x} of these; 7B constant {2,5} NOT imported")

    # ---------------- RMU grid (e065a)
    rmu_cells = []
    for ci, (ds, am) in enumerate([(ds, am) for am in (1.0, 2.0) for ds in depth_grid]):
        if SMOKE and ci >= 1:
            break
        tag = f"rmu_a{int(am)}x_d{''.join(str(d) for d in ds)}"
        if ci > 0:
            cool()
        cell = rmu_finetune(tag, net0, ua, set(ds), am, f_ids, r_bank,
                            r_eval_xy, zid, ce0=ce_r0, seed=RMU_SEED0 + ci)
        cell["dce_r"] = cell["final"].get("ce_r", float("nan")) - ce_r0
        rmu_cells.append(cell)
        log(f"[{tag}] done: steps {cell['steps_ran']} final p_z "
            f"{cell['final'].get('p_z_mean', float('nan')):.4f} dCE_R "
            f"{cell['dce_r']:+.4f} feasible={cell['feasible']}")
    # kill criteria
    feasible_cells = [c for c in rmu_cells if c["feasible"]]
    rmu_kill = not feasible_cells
    if rmu_kill:
        log("KILL: no RMU cell reached p_z<=0.05 within dCE_R<=+0.3 — "
            "toy-scale RMU INFEASIBLE (design section Kill); surgery+controls "
            "still run; head-to-head reported as negative result")
        rmu_best = None
    else:
        order = {"a1xd4": 0, "a1xd5": 1, "a1xboth": 2, "a2xd4": 3, "a2xd5": 4,
                 "a2xboth": 7}
        def pref(c):
            key = f"a{int(c['alpha_mult'])}xd{''.join(str(d) for d in c['depths'])}"
            return (c["dce_r"], c["final"]["p_z_mean"], order.get(key, 9))
        rmu_best = sorted(feasible_cells, key=pref)[0]
        log(f"RMU cell selection (min dCE_R): {rmu_best['tag']} "
            f"(dCE_R {rmu_best['dce_r']:+.4f}, p_z {rmu_best['final']['p_z_mean']:.4f})")

    # ---------------- surgery arm (e065b): e042 D2 subtractive row-reset
    sd0 = net0.state_dict()
    surg_sd = {k: v.clone() for k, v in sd0.items()}
    surg_sd["wte.weight"][zid] = 0.0
    surg_sd["lm_head.weight"][zid] = 0.0
    n_diff = sum(int((surg_sd[k] != sd0[k]).sum().item())
                 for k in ("wte.weight", "lm_head.weight"))
    confined = all(
        (lambda rows: len(rows) == 1 and int(rows[0]) == zid)(
            torch.nonzero(surg_sd[k] != sd0[k])[:, 0].unique().tolist())
        for k in ("wte.weight", "lm_head.weight"))
    others = all(torch.equal(surg_sd[k], sd0[k])
                 for k in surg_sd if k not in ("wte.weight", "lm_head.weight"))
    G_SURG = {"n_elements_changed": n_diff, "expected": 2 * net0.cfg.n_embd,
              "confined_to_Z_rows": bool(confined),
              "others_bit_identical": bool(others),
              "pass": bool(n_diff == 2 * net0.cfg.n_embd and confined and others)}
    net_surg = TinyGPT(Cfg())
    net_surg.load_state_dict(surg_sd)
    net_surg = to_dev(net_surg).eval()
    log(f"G_SURG D2 row-reset ({n_diff} elems, confined={confined}, "
        f"others-identical={others}): {'PASS' if G_SURG['pass'] else 'FAIL'}")
    bz_surg = battery_pz(net_surg, f_ids, zid)
    log(f"surgery battery p(Z) {bz_surg['p_z_mean']:.2e}")

    # ---------------- ascent arm (e065c) + retain-only control
    steps_match = rmu_best["steps_ran"] if rmu_best else (50 if not SMOKE else 8)
    cool()
    u_ret = mean_retain_direction(net0, r_bank)
    asc = ascent_finetune("ascent", net0, u_ret, inst_x, inst_mask, r_eval_xy,
                          f_ids, zid, ce0=ce_r0, steps_cap=steps_match,
                          seed=ASCENT_SEED)
    cool()
    ret = retain_finetune("retain_only", net0, r_bank, r_eval_xy, f_ids, zid,
                          steps_cap=steps_match, seed=RETAIN_SEED)

    arms = {
        "no_removal": {"net": net0, "desc": "unmodified installed net (ctrl)"},
        "surgery": {"net": net_surg, "desc": "e042 D2 row-reset (wte_Z=lm_Z=0)"},
        "ascent": {"net": asc["net"], "desc": f"projected ascent on F, {asc['steps_ran']} steps"},
        "retain_only": {"net": ret["net"], "desc": f"retain-only FT, {ret['steps_ran']} steps"},
    }
    if rmu_best is not None:
        arms["rmu"] = {"net": rmu_best["net"],
                       "desc": f"RMU {rmu_best['tag']} ({rmu_best['steps_ran']} steps)"}
    arm_order = ["no_removal", "rmu", "surgery", "ascent", "retain_only"]
    arm_order = [a for a in arm_order if a in arms]

    for a in arm_order:
        bz = battery_pz(arms[a]["net"], f_ids, zid)
        ce = ce_fixed(arms[a]["net"], *r_eval_xy)
        arms[a]["battery"] = bz
        arms[a]["ce_r"] = ce
        log(f"arm {a:11s}: battery p_z {bz['p_z_mean']:.4f} CE_R {ce:.4f} "
            f"(dCE {ce - ce_r0:+.4f})")

    # ---------------- R1: suppression-depth transplant (e055 rig, per arm)
    log("R1: harvesting onset sites from no-removal trajectories (e055 protocol)")
    sites = []
    seed_idx = 0
    while seed_idx < (1 if SMOKE else 8):
        torch.manual_seed(SITE_SEED0 + 100 * seed_idx)
        conts = [corpus.decode(r.tolist()) for r in
                 gen_batch(net0, p_ids, GEN_TOK)]
        for i, cont in enumerate(conts):
            for m in HOST_RE.finditer(cont):
                t = len(gen_prompts[i]) + m.start()
                sites.append({"prompt": i, "seed_batch": seed_idx, "t": t,
                              "name": m.group(1),
                              "ctx": gen_prompts[i] + cont[:m.start()]})
        seed_idx += 1
        if len(sites) >= SITE_ENOUGH and seed_idx >= 2:
            break
        if seed_idx >= 3 and len(sites) >= SITE_MIN:
            break
    log(f"harvested {len(sites)} onset sites from {seed_idx}x8x{GEN_TOK} chars")
    if len(sites) < (1 if SMOKE else 8):
        trims.append(f"sites_only_{len(sites)} (harvest short; R1 power reduced)")

    for s in sites:
        crop = corpus.encode(s["ctx"])[-net0.cfg.block_size:]
        s["crop"] = crop
        s["dpos"] = len(crop) - 1
        s["stratum"] = "terminal" if s["t"] == 120 else ("deep" if s["t"] > 150 else "mid")

    # donors: no-removal net, e055 selection rule (top-2 p_z + 2 FLORIZEL + 2 repl)
    with torch.no_grad():
        lgF, _ = net0(f_ids.to(DEV))
    pzf = F.softmax(lgF[:, -1], -1)[:, zid]
    order = sorted(range(60), key=lambda i: -float(pzf[i]))
    primary4 = list(dict.fromkeys(order[:2]
                                  + [i for i in order[2:] if install_occ[i][1] == "FLORIZEL"][:2]))[:4]
    repl2 = [i for i in order if i not in primary4][:2]
    donor_idx = primary4 + repl2
    if SMOKE:
        donor_idx = donor_idx[:2]
    donors = []
    for i in donor_idx:
        xs, lg = states_forward(net0, f_ids[i:i + 1].to(DEV))
        donors.append({"ctx_i": i, "host": install_occ[i][1],
                       "p_z": float(F.softmax(lg[0, -1], -1)[zid]),
                       "states": [x[0, -1].clone() for x in xs]})
    log("  donors (no-removal net): " + " ".join(
        f"ctx{d['ctx_i']}(pZ {d['p_z']:.2f})" for d in donors))

    # shuffled contexts (e055 Random(25501) family)
    _s2 = random.Random(SHUF_SEED)
    shuf_ctxs = []
    while len(shuf_ctxs) < (2 if SMOKE else N_SHUF):
        q = _s2.randrange(PRE + 1, len(train_ids) - 1)
        shuf_ctxs.append(train_text[q - PRE: q])

    r1 = {}
    for a in arm_order:
        net = arms[a]["net"]
        shuf_states = []
        for c in shuf_ctxs:
            xs, lg = states_forward(net, corpus.encode(c).unsqueeze(0).to(DEV))
            shuf_states.append([x[0, -1].clone() for x in xs])
        tf_vals = [[[] for _ in DEPTHS] for _ in sites]
        sh_vals = [[[] for _ in DEPTHS] for _ in sites]
        for si, s in enumerate(sites):
            crop1 = s["crop"].unsqueeze(0).to(DEV)
            for d in DEPTHS:
                B = len(donors) + len(shuf_states)
                states = torch.stack([dn["states"][d] for dn in donors]
                                     + [x[d] for x in shuf_states])
                lg = patch_logits_batch(net, crop1.repeat(B, 1), s["dpos"], d, states)
                pz = pz_next(lg, zid)
                tf_vals[si][d] = [float(v) for v in pz[:len(donors)]]
                sh_vals[si][d] = [float(v) for v in pz[len(donors):]]
        stats = [E55.depth_stats([np.array(tf_vals[i][d]) for i in range(len(sites))],
                                 [np.array(sh_vals[i][d]) for i in range(len(sites))])
                 for d in DEPTHS]
        rescuable_ds = []
        for d in MEANINGFUL_BAND:
            st = stats[d]
            if st["mean_tf"] is not None and st["mean_tf"] >= RESCUE_BAR \
                    and st["mean_shuf"] is not None and st["mean_shuf"] <= SHUF_BAR \
                    and st["delta_ci"][0] > 0:
                rescuable_ds.append(d)
        max_mean = max((stats[d]["mean_tf"] or 0.0) for d in MEANINGFUL_BAND)
        r1[a] = {"stats": stats, "rescuable_depths": rescuable_ds,
                 "arm_rescuable": bool(rescuable_ds),
                 "arm_not_rescuable": bool(max_mean <= NOT_RESCUABLE_MAX),
                 "max_mean_tf_d5": max_mean,
                 "tf_mean_curve": [stats[d]["mean_tf"] for d in DEPTHS],
                 "shuf_mean_curve": [stats[d]["mean_shuf"] for d in DEPTHS],
                 "per_site_tf": tf_vals}
        log(f"R1[{a:11s}] TF curve " + " ".join(
            f"d{d}:{stats[d]['mean_tf']:.3f}" for d in DEPTHS)
            + f" | rescuable@{rescuable_ds} not_rescuable={r1[a]['arm_not_rescuable']}")

    # base-net twin (donor0 into e001; the frozen 0.0062 reference)
    base_curve = []
    for d in DEPTHS:
        vals = []
        for s in sites[:8]:
            lg = patch_logits_batch(net_base, s["crop"].unsqueeze(0).to(DEV),
                                    s["dpos"], d, donors[0]["states"][d].unsqueeze(0))
            vals.append(float(pz_next(lg, zid)))
        base_curve.append(float(np.mean(vals)))
    log("R1 base-net twin curve (8 sites): "
        + " ".join(f"d{d}:{v:.4f}" for d, v in enumerate(base_curve)))

    # ---------------- R2: probe vs generation (Orgad instrument)
    r2 = {}
    for a in arm_order:
        net = arms[a]["net"]
        with torch.no_grad():
            hsF, _ = states_forward(net, (f_ids[:24] if SMOKE else f_ids).to(DEV))
            hsH, _ = states_forward(net, (h_ids[:12] if SMOKE else h_ids).to(DEV))
        probe_acc = {}
        for d in DEPTHS:
            probe_acc[d] = probe_bacc(hsF[d][:, -1], hsH[d][:, -1])
            thermal_throttle(1)
        # generation: 2 seeded runs x 8 prompts x 350 chars
        zc, zw, hc, chars = 0, 0, 0, 0
        for gseed in GEN_CHECK_SEEDS:
            torch.manual_seed(gseed)
            outs = gen_batch(net, p_ids, GEN_TOK)
            for row in outs:
                txt = corpus.decode(row.tolist())
                zc += len(ZEPH_RE.findall(txt))
                zw += len(re.findall(r"(?<![A-Za-z])Z[A-Za-z]*", txt))
                hc += len(HOST_RE.findall(txt))
                chars += len(txt)
        r2[a] = {"probe_acc_by_depth": probe_acc,
                 "probe_d45": float(np.mean([probe_acc[4], probe_acc[5]])),
                 "zephyra_count": zc, "z_words": zw, "host_count": hc,
                 "gen_chars": chars,
                 "gen_rate_per_kchar": zc / max(chars / 1000, 1e-9)}
        log(f"R2[{a:11s}] probe d4/d5 {r2[a]['probe_d45']:.3f} (d0 "
            f"{probe_acc[0]:.2f} d6 {probe_acc[6]:.2f}) | gen ZEPHYRA {zc}/{chars} "
            f"ch, hosts {hc}")
    # base-net confound probe
    with torch.no_grad():
        hsFb, _ = states_forward(net_base, f_ids.to(DEV))
        hsHb, _ = states_forward(net_base, h_ids.to(DEV))
    r2_base = {d: probe_bacc(hsFb[d][:, -1], hsHb[d][:, -1], seed=PROBE_SEED + 1)
               for d in DEPTHS}
    log("R2 base(e001) confound probe: "
        + " ".join(f"d{d}:{v:.2f}" for d, v in r2_base.items()))
    ctrl_gen0 = r2["no_removal"]["zephyra_count"] == 0
    r2_mode = "PROBE-ONLY" if ctrl_gen0 else "PROBE+GAP"
    log(f"R2 conditional (critic fix 1): no-removal gen ZEPHYRA "
        f"{r2['no_removal']['zephyra_count']} -> mode {r2_mode}")

    # ---------------- R3: relearn scar battery (e044), per removal arm
    orig_rows = (net0.wte.weight[zid].detach().clone(),
                 net0.lm_head.weight[zid].detach().clone())
    rl_arms = [a for a in ("rmu", "surgery", "ascent", "retain_only") if a in arms]
    if SMOKE:
        rl_arms = [a for a in ("surgery", "retain_only") if a in arms][:2]
    r3 = {}
    for k, a in enumerate(rl_arms):
        cool()
        r3[a] = relearn_battery(f"rl_{a}", arms[a]["net"], inst_x, inst_mask,
                                anchor, train_ids, f_ids, r_eval_xy, zid,
                                orig_rows, seed=RL_SEED0 + k)
        tr = r3[a]["traj"]
        cos_final = tr[-1] if tr else {}
        r3[a]["cos_wte_final"] = cos_final.get("cos_wte")
        r3[a]["cos_lm_final"] = cos_final.get("cos_lm")
        r3[a]["cos_wte_start"] = tr[0].get("cos_wte") if tr else None
        r3[a]["p50"] = next((r for r in tr if r["step"] >= 50), tr[-1] if tr else None)
        # e044 norm guard on the cos readout
        ow = float(orig_rows[0].norm())
        ol = float(orig_rows[1].norm())
        r3[a]["cos_wte_valid"] = bool(cos_final and cos_final.get("wte_norm", 0) >= COS_MIN_NORM_FRAC * ow)
        r3[a]["cos_lm_valid"] = bool(cos_final and cos_final.get("lm_norm", 0) >= COS_MIN_NORM_FRAC * ol)

    for a in rl_arms:
        log(f"R3[{a:11s}] steps_to_bar {r3[a]['steps_to_bar']} "
            f"(50-step: acc {r3[a]['p50']['name_acc'] if r3[a]['p50'] else None}) "
            f"cos_wte {r3[a]['cos_wte_start']}->{r3[a]['cos_wte_final']} "
            f"(valid={r3[a]['cos_wte_valid']})")

    # ---------------- R4: collateral bands
    r4 = {}
    for a in arm_order:
        ce = arms[a]["ce_r"]
        dce = ce - ce_r0
        ent = mean_entropy(arms[a]["net"], r_eval_x)
        band = ("~0" if dce <= 0.05 else ("small" if dce <= 0.15 else "FAILS"))
        r4[a] = {"ce_r": ce, "dce_r": dce, "band": band,
                 "entropy": ent, "entropy_drift": ent / ent0 - 1.0,
                 "entropy_ok": bool(abs(ent / ent0 - 1.0) <= 0.05)}
        log(f"R4[{a:11s}] dCE_R {dce:+.4f} [{band}] entropy drift "
            f"{r4[a]['entropy_drift']:+.2%} ({'ok' if r4[a]['entropy_ok'] else 'OUT'})")

    # ---------------- R5: RMU-vs-retain-only contrast (registered)
    r5 = compute_r5(r1, r3, r4, arms, r_eval_xy)
    if r5.get("void"):
        log(f"R5 VOID: {r5['note']}")
    else:
        log("R5 (rmu - retain_only): R1 delta "
            + " ".join(f"d{d}:{r5['r1_delta_by_depth'][d]['mean_delta']:+.3f}"
                       f"[{r5['r1_delta_by_depth'][d]['ci'][0]:+.3f},"
                       f"{r5['r1_delta_by_depth'][d]['ci'][1]:+.3f}]"
                       for d in MEANINGFUL_BAND)
            + f" | R4 dCE delta {r5['r4_delta']['mean_delta']:+.4f} "
            f"[{r5['r4_delta']['ci'][0]:+.4f},{r5['r4_delta']['ci'][1]:+.4f}]"
            f" | R3 steps_to_bar delta {r5['r3_delta']['steps_to_bar_delta']}")
        log(f"R5 BRANCH: {r5['branch']}")

    # ---------------- verdicts (registered structure, printed at the end)
    def r3_fast(a):
        return bool(r3.get(a) and r3[a]["steps_to_bar"] is not None
                    and r3[a]["steps_to_bar"] <= 50)

    def r3_slow2x(a, ref):
        if not r3.get(a) or r3[a]["steps_to_bar"] is None:
            return False
        ref_steps = (r3.get(ref) or {}).get("steps_to_bar")
        if ref_steps is None:
            return False
        return bool(r3[a]["steps_to_bar"] >= 2 * ref_steps)

    def cos_class(a):
        v = r3[a].get("cos_wte_final")
        if v is None or not r3[a]["cos_wte_valid"]:
            return "invalid(<25% norm regrowth)"
        if v >= 0.60:
            return "stays-high(groove)"
        if v <= 0.40:
            return "fresh-like(<=0.40)"
        return "ambiguous(0.40-0.60)"

    verdicts = {}
    for a in arm_order:
        v = {"R1": {"rescuable": r1[a]["arm_rescuable"],
                    "rescuable_depths": r1[a]["rescuable_depths"],
                    "not_rescuable": r1[a]["arm_not_rescuable"],
                    "max_mean_tf_d5": r1[a]["max_mean_tf_d5"]},
             "R2": {"mode": r2_mode, "probe_d45": r2[a]["probe_d45"],
                    "gen_rate_per_kchar": r2[a]["gen_rate_per_kchar"]},
             "R3": None, "R4": r4[a]}
        if a in r3:
            v["R3"] = {"steps_to_bar": r3[a]["steps_to_bar"],
                       "revert_fast_le50": r3_fast(a),
                       "cos_wte_final": r3[a]["cos_wte_final"],
                       "cos_class": cos_class(a)}
        verdicts[a] = v

    rmu_sig = bool("rmu" in arms and r1["rmu"]["arm_rescuable"] and r3_fast("rmu"))
    surg_sig = bool(r1["surgery"]["arm_not_rescuable"]
                    and (("rmu" in r3 and r3_slow2x("surgery", "rmu"))
                         or ("retain_only" in r3 and r3_slow2x("surgery", "retain_only")))
                    and "stays-high" in cos_class("surgery"))
    if rmu_kill:
        headline = ("RMU INFEASIBLE at toy scale (kill criteria): no alpha/depth cell "
                    "reached p_z<=0.05 within dCE_R<=+0.3. Head-to-head parked as a "
                    "negative result; surgery+controls measured.")
    elif rmu_sig and surg_sig:
        headline = ("H-OBFFUSCATION INVERSION CONFIRMED: RMU is rescuable-but-"
                    "reverts-fast; surgery is unrescuable-but-scarred.")
    else:
        headline = ("INVERSION NOT AS PREDICTED — pattern: "
                    + "; ".join(f"{a}: R1 {'rescuable' if r1[a]['arm_rescuable'] else ('not-rescuable' if r1[a]['arm_not_rescuable'] else 'intermediate')}"
                                + (f", relearn {r3[a]['steps_to_bar']} steps" if a in r3 else "")
                                for a in arm_order)
                    + ". Alternative required before any rerun (design closing rule).")
    verdicts["headline_inversion"] = headline
    verdicts["rmu_signature"] = rmu_sig
    verdicts["surgery_signature"] = surg_sig
    verdicts["kill"] = {"rmu_infeasible": rmu_kill,
                        "rule": "no RMU cell: p_z<=0.05 AND dCE_R<=+0.3"}

    # ---------------- outputs
    def clean_cell(c):
        return {k: c[k] for k in ("tag", "depths", "alpha_mult", "alphas",
                                  "steps_ran", "final", "dce_r", "feasible")}

    metrics = {
        "experiment": "e065_rmu_headtohead",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "design": "scratch/e065_rmu_headtohead_design.md (critic-hardened 5f11a79)",
        "smoke": SMOKE,
        "net": "runs/checkpoints/e048_repro.pt (e043 Dmix@s400 install)",
        "gates": {"G_INST_battery_repro": G_INST, "G_DET": G_DET,
                  "G_SURG_row_reset": G_SURG},
        "step0_alpha_norms": alpha_report,
        "forget_retain": {"F": "60 install windows + ctx130 prompts (SPLICE_RNG 24301)",
                          "R": f"val windows: bank {len(r_bank)}, eval {len(r_eval_x)}",
                          "ce_r_no_removal": ce_r0, "entropy_no_removal": ent0},
        "rmu_grid": [clean_cell(c) for c in rmu_cells],
        "rmu_kill": rmu_kill,
        "rmu_selected": clean_cell(rmu_best) if rmu_best else None,
        "ascent": {"steps_ran": asc["steps_ran"], "final": asc["final"],
                   "traj": asc["traj"]},
        "retain_only": {"steps_ran": ret["steps_ran"], "final": ret["final"],
                        "traj": ret["traj"]},
        "arms_battery": {a: {"desc": arms[a]["desc"], "battery": arms[a]["battery"],
                             "ce_r": arms[a]["ce_r"]} for a in arm_order},
        "r1": {"sites": [{"prompt": s["prompt"], "t": s["t"], "name": s["name"],
                          "stratum": s["stratum"], "dpos": s["dpos"]} for s in sites],
               "donors": [{"ctx_i": d["ctx_i"], "host": d["host"], "p_z": d["p_z"]}
                          for d in donors],
               "bars": {"rescuable": "site-mean TF>=0.30 AND shuf<=0.05 AND "
                                     "site-bootstrap 95% CI excludes 0",
                        "not_rescuable": "all d<=5 site-means <= 2x0.0062",
                        "base_twin_frozen": BASE_TWIN},
               "base_net_twin_curve": base_curve,
               "per_arm": {a: {"rescuable_depths": r1[a]["rescuable_depths"],
                               "arm_rescuable": r1[a]["arm_rescuable"],
                               "arm_not_rescuable": r1[a]["arm_not_rescuable"],
                               "tf_mean_curve": r1[a]["tf_mean_curve"],
                               "shuf_mean_curve": r1[a]["shuf_mean_curve"],
                               "per_site_tf": r1[a]["per_site_tf"],
                               "stats_by_depth": r1[a]["stats"]} for a in arm_order}},
        "r2": {"mode": r2_mode,
               "base_e001_confound_probe": r2_base,
               "per_arm": r2},
        "r3": {"protocol": "e044 Dmix exposure, cosine(1000), cap 300 steps, "
                           "bar NLL<=1.0 & acc>=0.8; n=1 per condition (flagged)",
               "e044_references": {"groove_cos": E044_GROOVE_COS,
                                   "fresh_cos": E044_FRESH_COS},
               "per_arm": {a: {"steps_to_bar": r3[a]["steps_to_bar"],
                               "steps_ran": r3[a]["steps_ran"],
                               "traj": r3[a]["traj"],
                               "cos_wte_final": r3[a]["cos_wte_final"],
                               "cos_lm_final": r3[a]["cos_lm_final"],
                               "cos_wte_valid": r3[a]["cos_wte_valid"],
                               "cos_class": cos_class(a)} for a in rl_arms}},
        "r4": r4,
        "r5": r5,
        "verdicts": verdicts,
        "trims": trims,
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192, "block_size": 256,
                   "params": int(net0.num_params())},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    # ---------------- plot: headtohead.png
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    ax = axes[0, 0]
    colors = {"no_removal": "tab:gray", "rmu": "crimson", "surgery": "tab:blue",
              "ascent": "darkorange", "retain_only": "seagreen"}
    for a in arm_order:
        ax.plot(DEPTHS, r1[a]["tf_mean_curve"], "o-",
                color=colors.get(a, None), label=f"{a} (TF)", lw=2)
        ax.plot(DEPTHS, r1[a]["shuf_mean_curve"], ":", color=colors.get(a, None),
                alpha=0.6)
    ax.plot(DEPTHS, base_curve, "x--", color="k", label="base-net twin (e001)")
    if not r5.get("void"):
        ax.plot(DEPTHS, [r5["r1_delta_by_depth"][d]["mean_delta"] for d in DEPTHS],
                "D--", color="purple", ms=5, label="R5: rmu - retain_only")
        ax.axhline(0, color="purple", lw=0.5)
    ax.axhline(RESCUE_BAR, color="k", ls=":", lw=1)
    ax.axhline(NOT_RESCUABLE_MAX, color="tab:blue", ls=":", lw=1)
    ax.axvspan(5.5, 6.5, color="k", alpha=0.08)
    ax.set_xlabel("write depth d"); ax.set_ylabel("R1 = P(Z) at onset site")
    ax.set_title(f"R1 depth-survival per arm ({len(sites)} sites; dotted=shuffled)")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for a in rl_arms:
        tr = r3[a]["traj"]
        ax.plot([r["step"] for r in tr], [r["name_acc"] for r in tr], "o-",
                color=colors.get(a, None), label=f"{a} (steps_to_bar {r3[a]['steps_to_bar']})")
    ax.axhline(RL_BAR_ACC, color="k", ls=":", lw=1)
    ax.axvline(50, color="crimson", ls=":", lw=1)
    ax.set_xlabel("relearn step"); ax.set_ylabel("forget-battery name acc")
    ax.set_title("R3 relearn recovery curves (50-step cell + steps-to-bar)")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    xs = np.arange(len(arm_order))
    ax.bar(xs - 0.18, [r2[a]["probe_d45"] for a in arm_order], 0.36,
           color=[colors.get(a, "gray") for a in arm_order], label="probe d4/d5 bacc")
    ax.bar(xs + 0.18, [min(r2[a]["gen_rate_per_kchar"], 1.0) for a in arm_order],
           0.36, color="lightgray", label="gen rate (ZEPHYRA/kchar, capped 1)")
    ax.plot(xs, [r2_base.get(4, np.nan)] * len(arm_order), "k_", label="e001 confound probe")
    ax.set_xticks(xs); ax.set_xticklabels(arm_order, fontsize=8)
    ax.set_ylabel("probe balanced acc / gen rate")
    ax.set_title(f"R2 probe-vs-generation (mode: {r2_mode})")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.bar(xs, [r4[a]["dce_r"] for a in arm_order],
           color=[("seagreen" if r4[a]["band"] == "~0" else
                   "darkorange" if r4[a]["band"] == "small" else "crimson")
                  for a in arm_order])
    ax.axhline(0.05, color="darkorange", ls=":", lw=1)
    ax.axhline(0.15, color="crimson", ls=":", lw=1)
    ax.axhline(0.3, color="k", ls="--", lw=1, label="kill (+0.3)")
    ax.set_xticks(xs); ax.set_xticklabels(arm_order, fontsize=8)
    ax.set_ylabel("val_R CE drift (nats)")
    ax.set_title("R4 collateral (bands 0.05 / 0.15 / kill 0.3)")
    ax.legend(fontsize=7)

    fig.suptitle(f"E065 RMU-vs-surgery head-to-head (smoke={SMOKE}; "
                 f"RMU kill={rmu_kill})", fontsize=12)
    fig.tight_layout()
    fig.savefig(rd / "headtohead.png", dpi=130)
    plt.close(fig)

    # ---------------- registered verdict printout (THE END)
    print("\n" + "=" * 78)
    print("E065 REGISTERED VERDICTS (per scratch/e065_rmu_headtohead_design.md)")
    print("=" * 78)
    print(f"STEP-0 alphas: " + " ".join(
        f"d{d}: med||h||={v['median_norm_forget']:.2f}" for d, v in alpha_report.items()))
    print(f"RMU grid: " + "; ".join(
        f"{c['tag']}: p_z={c['final'].get('p_z_mean', float('nan')):.4f} "
        f"dCE_R={c['dce_r']:+.3f} feasible={c['feasible']}" for c in rmu_cells))
    print(f"RMU KILL (infeasible): {rmu_kill}")
    for a in arm_order:
        v = verdicts[a]
        r1s = (f"rescuable@{v['R1']['rescuable_depths']}" if v["R1"]["rescuable"]
               else ("NOT-RESCUABLE (<=2x base twin)" if v["R1"]["not_rescuable"]
                     else f"intermediate (max d<=5 {v['R1']['max_mean_tf_d5']:.3f})"))
        r2s = f"probe d4/5 {v['R2']['probe_d45']:.3f}, gen {v['R2']['gen_rate_per_kchar']:.3f}/kchar [{r2_mode}]"
        r3s = ("n/a (no removal)" if v["R3"] is None else
               f"steps_to_bar {v['R3']['steps_to_bar']} "
               f"(fast<=50: {v['R3']['revert_fast_le50']}; cos {v['R3']['cos_class']})")
        r4s = f"dCE_R {v['R4']['dce_r']:+.4f} [{v['R4']['band']}], entropy drift {v['R4']['entropy_drift']:+.1%}"
        print(f"  {a:11s} | R1 {r1s} | R2 {r2s} | R3 {r3s} | R4 {r4s}")
    print("-" * 78)
    if not r5.get("void"):
        print(f"R5 (RMU - retain-only): R1 deltas "
              + " ".join(f"d{d}:{r5['r1_delta_by_depth'][d]['mean_delta']:+.3f}"
                         for d in MEANINGFUL_BAND)
              + f" | R4 dCE delta {r5['r4_delta']['mean_delta']:+.4f}"
              + f" | R3 steps_to_bar delta {r5['r3_delta']['steps_to_bar_delta']}")
        print(f"R5 BRANCH: {r5['branch']}")
        print("-" * 78)
    print(f"RMU signature (rescuable AND revert-fast<=50):    {rmu_sig}")
    print(f"SURGERY signature (unrescuable AND >=2x-slow AND cos stays high): {surg_sig}")
    print(f"HEADLINE INVERSION: {headline}")
    print("=" * 78)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'headtohead.png'}")
    log(f"total {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
