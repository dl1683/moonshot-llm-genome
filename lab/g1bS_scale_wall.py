"""G1BS — THE WALL AT 10x (the scale debt; supervisor C11-2/C12-4/C13-2).

Design spec: scratch/g1bS_design.md (DESIGN-DRAFT frozen at dispatch — the
convention is frozen; the builder may not re-tune the dial). Builds on:
g1/g1b/g1bR (the commit-and-project wall; WALL-REPLICATES n=3 seeds, mins
0.746/0.777/0.803; WALL-TAXES +0.53), e182 (the lab's only >=10x result: the
pretrained-LM lane), T133/T140 (the wall's standing: battery-channel scope,
one mechanism inside/outside the ball). What is NEW: the wall's first scale
rung — no prior g-claim tested above 2.74M params.

THE QUESTION: does the commit-and-project L2 ball hold a consolidated fact
through a wash that kills the control, at ~10x the parameters?

==================== THE FROZEN CONVENTION (design, verbatim) ================
The R-dial travels in PER-COORDINATE RMS units, not raw L2.
R_rms = 0.7/sqrt(2.74e6) = 4.2301e-4. The ladder: {1x, 2x, 4x} x R_rms
(dimension-matched to g1's {1,2,6}x shape). All readouts report BOTH
conventions (R_rms and raw L2). THE WASH IS UNTOUCHED (the e1xx recipe
verbatim — the wash is the treatment, not the dial).

=============== THE REGISTERED BARS (scratch/g1bS_design.md, verbatim) ======
  WALL-SCALES: "fires if at R_rms_match the fact's battery-channel p(Z)
      stays >= 0.9-equivalent (scaled to this host's root strength) at every
      checkpoint through +300 while C dies by +50."
  WALL-FADES: "fires if C dies but no R on the ladder holds the bar (or only
      the tightest R holds it while freezing CE)."
  WALL-TIGHTENS: "fires if a LOOSER R than rms-match holds (the basin grew
      with dimension) — co-report the CE tax curve at scale (the +0.53
      reference re-priced)."
No bar shopping: the ladder is fixed; if no rung holds, WALL-FADES is the
verdict, not a dial search.

====== THE SCALED-BAR DEFINITION (defined here BEFORE compute, per the
====== dispatch; the only numbers it uses are the frozen prior + this run's
====== own root, measured before any arm trains):
  g1b_root_gm12 := 0.9155886769294739          (g1b's stored root, frozen)
  root_gm12     := this host's consolidated-root g-12 (measured at G-ROOT)
  bar_0p9eq     := root_gm12 * 0.9 / g1b_root_gm12   (= 0.982955 x root)
  "R holds the bar" := g-12 >= bar_0p9eq at EVERY checkpoint
      {1,2,4,10,50,100,200,300}.
  "freezing CE" (the WALL-FADES escape clause) := W1's in-batch corpus CE at
      +300 >= the root's CE_R (the walled organism never improved on its own
      root corpus CE through the whole wash). Co-reported: dCE(W1,C)@300 vs
      the +0.53 2.74M reference.
  CO-REPORTED SECONDARY BARS (context only, never adjudicated as primary):
  g1-verbatim maintains (>= 0.50 at every ckpt) / dies (<= 0.27 by +50);
  post-transient flat-phase retention := min g-12 over {10,50,100,200,300}
  >= 0.9 x root_gm12 (the +2/+4 settled-reversal dip is a known walled-arm
  read — g1b's own W1 dipped to 0.848x root at +2 then sat flat at ~1.00x;
  the strict every-checkpoint primary bar sees that dip, by design: at 10x
  a raw-geometry leak would widen it, an rms-matched basin would not).

THE CELL (design verbatim in form, g1b's conventions):
  HOST   fresh ~10M TinyGPT char-LM, Cfg 8L/8H/320d block 256 = 9,977,600
         params (10M +/-15% = [8.5M, 11.5M]: PASS; e005s LARGE's proven
         10M-class config — extend-don't-repeat, not a new guess; e005s's
         own large ckpt stopped at 1086/4000 steps of its cosine and is NOT
         reused: the design froze a fresh host with a COMPLETED cosine).
         Corpus/init seed 1337 family. Base trains ckpt-RESUMABLE in
         <=180 s GPU chunks (recorded chunk table; cooldown 120 s between;
         the envelope-legal way to exceed 180 s total) to a COMPLETED
         4000-step house cosine (common.train_model, warmup 100).
  INSTALL    e043-Dmix convention verbatim (g1 phase-0a): 16 spliced install
         + 16 paired originals + 32 random = 64 windows/step, dose s400,
         house cosine, seed 42, full-token union CE.
  CONSOLIDATE e113 jitter convention verbatim (g1 phase-0b): jitters
         {-8,-4,0,+4,+8}, 300 steps, batch 16 install + 16 anchor (8 paired
         + 8 random), name-masked union CE, AdamW (0.9,0.95) wd 0.1 constant
         lr 1e-3 clip 1.0, seed 10901.
  WALL    commit(theta_anchor) at the RMS ladder, then the UNTOUCHED wash
         (e176N arm A verbatim: neutral bank seed 170, batch 32 = 16 neutral
         + 16 random, full-token CE, AdamW (0.9,0.95) wd 0.1 constant lr
         1e-3 clip 1.0, seed 10902, 300 steps, ckpts {1,2,4,10,50,100,200,300}):
         C uncommitted / W1 = 1x R_rms / W2 = 2x / W3 = 4x. Checkpoint
         cadence, fact battery, CE reads VERBATIM from g1b's conventions.

ARMS GUARANTEE (Rule 12 pre-dispatch): all four arms share bit-identical
step-0 weights (G-BITROOT), bit-identical per-step input streams (md5-gated,
G-INPUTS through +300) and identical targets; the ONLY delta is the commit
radius. C guarantees the kill clock at 10M (G-CTRL); the W arms guarantee a
G-PIN-verified ball at their radius (mechanics of the wall verified before
any behavioral clause is read). BATTERY GEOMETRY AT 10M: vocab 65, block
256, same corpus file — every position the instruments read (PRE=130
contexts, site row 183..190, wpe band 121-129) is inside the geometry
unchanged; the battery contexts are char strings encoded by the same stoi.

GATES (any failure => TEXTURE, nothing adjudicated):
  G-CONFIG  params in [8.5M, 11.5M]; vocab 65; block 256; the R-convention
            assertion (|R_i_raw/sqrt(P) - k_i x R_rms_ref| < 1e-12 for every
            rung; R_rms_ref = 0.7/sqrt(2739072)).
  G-BASE    the base's cosine COMPLETED (final step == 4000).
  G-ROOT    consolidated root g-12 >= 0.78 (the arc's express bar, verbatim).
  G-CTRL    arm C dies by +50 (g-12 <= 0.27, verbatim).
  G-PIN     every wall arm's raw displacement <= R_raw + pin_fuzz_rms-carried
            (primary; the frozen 1.5 fuzz constant carried in the SAME rms
            convention as the dial: 1.5*sqrt(P/2.74e6) = 2.862 raw — the
            verbatim R+1.5 form is co-reported and mechanically cannot hold
            at 10M: one AdamW step ~ lr*sqrt(P) = 3.16 raw exceeds any
            R+1.5 on this ladder; carrying the fuzz in rms units is the
            convention-consistent form of the SAME registered formula, not
            a re-tune).
  G-BITROOT / G-INPUTS / G-STEP1  verbatim (wall roots bit-identical to
            theta0; per-step inputs md5-identical across all four arms;
            wall arms' step-1 bodies == C's step-1 body within 1e-4).

COMPUTE ENVELOPE: 9,977,600 params — inside the <=100M free tier (common.py,
Devansh 2026-09-27); the design's stated reason: the wall assay needs an
INSTALLED fact on a trained base — GPT-2 would change tokenizer/geometry
variables at once and e182 already covers the pretrained lane. ~8-11
trainings <= 180 s each (base chunks + install + consolidate + 4 wash arms)
+ cooldowns 120 s; gpu_ok() double-poll before every launch; outside load =>
PAUSE-AND-WAIT per the g1bW policy, never migrate mid-run; pauses recorded.
torch threads 4 (shared machine, g1bW's adopted trim).

Outputs: runs/g1bS/{metrics.json, scale_wall.png}; checkpoints
runs/checkpoints/g1bS_*.pt. No NOTES/THINKING/QUEUE/STATE edits (the
coordinator folds). Single commit + push.

Run:  cd lab && python g1bS_scale_wall.py    (G1BS_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
import random
import sys
import time
from pathlib import Path

SMOKE = os.environ.get("G1BS_SMOKE") == "1"
if SMOKE:
    os.environ["G1_SMOKE"] = "1"     # trims G1.ROWS_OLD only (machinery smoke)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT, cooldown, gpu_ok,   # noqa: E402
                    gpu_status, run_dir, save_json, set_seed,
                    train_model)
import e043_install as E43                             # noqa: E402 (REPO,
                                                      # find_occ, SPLICE_RNG,
                                                      # jsonable)
import g1_anchored_ball as G1                          # noqa: E402 — g1's
                                                      # machinery (CommittedGPT,
                                                      # instruments, streams)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

CPU = torch.device("cpu")
torch.set_num_threads(4)     # shared machine (g1bW's adopted trim; g1's
                             # import resets to 8 — reset after import)

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)
G1.log = log                                          # unify the timeline

# ======================================================================
# THE CONFIG (the whole difference from g1b: the host + the R convention)
# ======================================================================
G1BS_CFG = Cfg(vocab=65, n_layer=8, n_head=8, n_embd=320, block_size=256)
G1BS_PARAMS = 9_977_600          # e005s LARGE's verified 10M-class config
HOST_SEED = 1337                 # the "1337 family" (host init + base corpus)
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
BASE_CK, INSTALL_CK = "g1bS_base.pt", "g1bS_install.pt"
ROOT_CK = "g1bS_root"                   # save_ckpt appends .pt

BASE_STEPS = 4000 if not SMOKE else 24
BASE_LR, BASE_BS, BASE_EVAL = 1e-3, 64, 250
INSTALL_STEPS = 400 if not SMOKE else 8
CONS_STEPS = 300 if not SMOKE else 8
COOLDOWN_S = 120.0               # the mission's envelope
TRAIN_CAP_GPU, TRAIN_CAP_CPU = 180.0, 1800.0
GPU_WAIT_MAX = 600.0             # arms/install/consolidate (g1bW policy)
BASE_WAIT_MAX = 1800.0           # base chunks wait longer: a 4000-step CPU
                                 # base is not a viable cell; recorded
HOT_LAUNCH_C, SOAK_RESUME_C = 80.0, 65.0   # lab thermal semantics (g1bW fix)

# ---- THE R LADDER (frozen convention): R_rms = 0.7/sqrt(2.74e6) ----------
G1B_PARAMS = 2_739_072
G1B_ROOT_GM12 = 0.9155886769294739          # g1b's stored root (the bar's
                                            # denominator, frozen)
R_RMS_REF = 0.7 / math.sqrt(G1B_PARAMS)     # 4.2301e-4 per-coordinate RMS
R_LADDER_MULT = (1.0, 2.0, 4.0)             # {1x, 2x, 4x} x R_rms (frozen)
SQRT_P = math.sqrt(G1BS_PARAMS)
R_LADDER_RAW = tuple(k * R_RMS_REF * SQRT_P for k in R_LADDER_MULT)
R_LADDER_RMS = tuple(r / SQRT_P for r in R_LADDER_RAW)   # == k*R_RMS_REF
PIN_FUZZ_RMS_CARRIED = (G1.PIN_FUZZ_BAR / math.sqrt(G1B_PARAMS)) * SQRT_P
ONE_STEP_FUZZ_RAW = G1.FT_LR * SQRT_P        # one AdamW step ~ lr*sqrt(P)

WASH_SEED = G1.FREEZE_SEED                    # 10902 (the locked lineage)
CK_WASH: tuple[int, ...] = G1.CK_WASH if not SMOKE else (1, 2, 4)
FULL_DIAL_STEPS = (2, 50, 300) if not SMOKE else ()

# ---- the g1b stored reference (priors, not bars) -------------------------
PRIORS_G1B = {
    "source": "runs/g1b/metrics.json (the 2.74M continuity cell)",
    "root_gm12": G1B_ROOT_GM12,
    "W1_g_m12": {1: 0.9452, 2: 0.7768, 4: 0.8174, 10: 0.9227, 50: 0.9155,
                 100: 0.9222, 200: 0.9137, 300: 0.9181},
    "C_dead_at": 2,
    "e185_D_kill_raw_274M": 2.4892616271972656,
    "D_kill_rms_274M": 2.4892616271972656 / math.sqrt(G1B_PARAMS),
    "wall_tax_274M": 0.5263230800628662,    # the +0.53 reference
    "W2_g_m12": {1: 0.8087, 2: 0.5031, 4: 0.3203, 10: 0.7301, 50: 0.7767,
                 100: 0.7436, 200: 0.6736, 300: 0.6465},
}

REGISTERED = {
    "bars_verbatim": {
        "WALL-SCALES": ("fires if at R_rms_match the fact's battery-channel "
                        "p(Z) stays >= 0.9-equivalent (scaled to this host's "
                        "root strength) at every checkpoint through +300 "
                        "while C dies by +50."),
        "WALL-FADES": ("fires if C dies but no R on the ladder holds the bar "
                       "(or only the tightest R holds it while freezing CE)."),
        "WALL-TIGHTENS": ("fires if a LOOSER R than rms-match holds (the "
                          "basin grew with dimension) — co-report the CE tax "
                          "curve at scale (the +0.53 reference re-priced)."),
    },
    "scaled_bar_definition": {
        "formula": "bar_0p9eq = root_gm12(this host) * 0.9 / 0.9155886769294739",
        "g1b_root_gm12": G1B_ROOT_GM12,
        "factor_vs_root": 0.9 / G1B_ROOT_GM12,
        "holds": ("g-12 >= bar_0p9eq at EVERY checkpoint "
                  "{1,2,4,10,50,100,200,300} (light battery read, g1b's "
                  "convention)"),
        "freezing_CE": ("W1 in-batch corpus CE at +300 >= root CE_R (the "
                        "walled organism never improved on its own root "
                        "corpus CE through the wash)"),
        "secondary_co_reported": [
            "g1-verbatim maintains (>=0.50 every ckpt) / dies (<=0.27 by +50)",
            "post-transient flat-phase retention: min g-12 over "
            "{10,50,100,200,300} >= 0.9 x root_gm12",
        ],
        "note": ("defined in the dispatch BEFORE compute; the primary bar is "
                 "intentionally strict (0.983x root): g1b's own W1 dipped to "
                 "0.848x root at +2 then sat flat at ~1.00x — at 10x the dip "
                 "discriminates a raw-geometry leak (dip widens, no rung "
                 "holds) from an rms-matched basin (dip shallow, rung holds)"),
    },
    "predicted": ("WALL-SCALES on the flat phase ({+10..+300} retention "
                  ">= 0.9x root at the 1x rung, raw-equivalent of g1b's W1 "
                  "behavior); C dead by ~+2/+4 — the Adam clock makes rms "
                  "displacement per step = lr regardless of N (T139), so the "
                  "kill should arrive on the same step count; the +2 "
                  "settled-reversal dip is the registered discriminator: if "
                  "it replicates g1b's depth at the 1x rung, the strict "
                  "every-checkpoint primary bar fails at +2 and WALL-FADES "
                  "fires with the dip texture (reported, not re-tuned)."),
    "falsifiers": ("C survives (+50 > 0.27) => G-CTRL fails => TEXTURE (a "
                   "scale finding about the WASH, registered as texture); "
                   "no rung holds while C dies => WALL-FADES; a looser rung "
                   "holds => WALL-TIGHTENS with the tax curve."),
}

deviations: list[str] = [
    "HOST FRESH at e005s LARGE's proven config (8L/8H/320d = 9,977,600; "
    "10M+/-15% PASS): extend-don't-repeat — the config is the lab's own "
    "10M-class architecture, not a new guess; e005s's stored large ckpt "
    "stopped at 1086/4000 steps of its cosine (its own metrics record) and "
    "is NOT reused — the design froze a fresh host with a COMPLETED "
    "cosine. Corpus/init seed 1337 (the dispatch's '1337 family'). Base "
    "trains ckpt-RESUMABLE <=180 s GPU chunks to a completed 4000-step "
    "house cosine; chunk table recorded in metrics.host.chunks.",
    "THE R CONVENTION (frozen design): ladder {1,2,4} x R_rms, R_rms = "
    "0.7/sqrt(2739072) = 4.2301e-4; raw at this host = {1.336, 2.672, "
    "5.345} L2; BOTH conventions printed at every checkpoint (the wash "
    "driver logs d_raw/d_rms vs R_raw/R_rms) and asserted in G-CONFIG.",
    "G-PIN RMS-CARRIED (primary): raw displacement <= R_raw + 1.5*"
    "sqrt(P/2.74e6) = R_raw + 2.862 — the frozen 1.5 fuzz constant carried "
    "in the same rms convention as the dial; the VERBATIM R+1.5 form is "
    "co-reported per arm and mechanically cannot hold at 10M (one AdamW "
    "step ~ lr*sqrt(P) = 3.159 raw exceeds any R+1.5 on this ladder). Same "
    "registered formula (one-step wall fuzz), convention-consistent "
    "carried; never re-tuned against data.",
    "LIGHT-EVAL TWIN ON THE TRAINING DEVICE (registered scale adaptation): "
    "at 10M a CPU eval twin costs ~20 s per checkpoint (2 battery fwd + CE "
    "on 60 windows) and alone exceeds the 180 s per-training cap; the "
    "in-loop light-eval twin rides the training device (CPU when training "
    "is CPU); ALL full dials (measure()) stay CPU-only verbatim; "
    "cross-device float fuzz is quantified by G-STEP1 (same-stream step-1 "
    "reproduction) and by the checkpoint dual-read co-report.",
    "DISPLACEMENT DEVICE-RESIDENT PER STEP (registered scale adaptation): "
    "the flat-parameter L2/increment arithmetic (net.parameters() order, "
    "fp32) is computed on the training device instead of a per-step host "
    "transfer (~10M x 4B/step); values equal CPU within float-reduction "
    "fuzz; cross-arm cosines move deltas to CPU first.",
    "OWN DRIVERS (g1's own precedent — 'copied, not imported, to own the "
    "device policy'): g1bS_wash/g1bS_consolidate copy g1_wash/consolidate "
    "VERBATIM arithmetic (draw order ix/aj/rj, batch composition, target "
    "handling, optimizer, clip, cap semantics, ckpt evals) plus the two "
    "registered adaptations above and the g1bW MIDRUN policy: outside "
    "GPU load/heat => PAUSE-AND-WAIT (cap time excluded), never migrate "
    "mid-run. Instruments (battery_cell, ce_fixed_cpu, val_windows, "
    "deleted_wpe, read_fact_at, row_census, CommittedGPT, evl_load, "
    "DmixCorpus) are G1's VERBATIM via import with G1.G1_CFG/G1_PARAMS "
    "patched to the 10M family (g1b's pattern).",
    "FULL DIALS TRIMMED to root + W1@{2,50,300} (g1b's FULL_DIAL_WASH "
    "verbatim): W2/W3/C run light-only (their roles are the ladder and the "
    "kill clock — g1bR's precedent); at 10M one full dial is ~4-5 CPU-min "
    "(census 49 battery reads), so 12 dials would dominate the cell.",
    "NO NOISE ARMS: the g1bS design's arms are C + the R ladder only; the "
    "noise split (N0/N1/N2) is g1b's cell, not re-run here.",
    "BASE CHUNK GATE WAITS up to 1800 s per chunk for a free GPU window "
    "(vs 600 s for the other trainings): a 4000-step 10M base on CPU is "
    "not a viable cell (~3 h); waiting is envelope-legal and every wait is "
    "recorded in device_events. Arms/install/consolidate keep g1bW's "
    "600 s-then-CPU policy (all are <=1800 s viable on CPU).",
    "torch threads 4 (shared machine; g1bW's adopted trim; g1's import "
    "resets to 8, reset after import).",
    "WAIT-GATE SOAK FIX (after the first full run's chunk 3 gate, run "
    "restarted cleanly between trainings; the base resumed from its chunk "
    "checkpoint at step 547): the soak-wait's <=65C resume target cannot be "
    "reached on this machine (~73-76C idle floor — the deadlock class "
    "g1bW's own F-note documented); the launch window now opens at the "
    "LAB'S OWN launch ceiling (<=80C, README rule 9: 'no new launches above "
    "80C'), i.e. the gate waits while >80C and launches once under it. The "
    "thermal ceiling, mem <=85% cap and util gate are unchanged and "
    "binding; no training semantics touched.",
    "Smoke mode trims: base 24 steps, install/consolidate 8 steps, washes "
    "4 steps with ckpts {1,2,4}, lean dials, no cooldowns, GPU gate waits "
    "capped at 30 s — nothing adjudicated.",
]

device_events: list[dict] = []
trims: list[str] = []


# ------------------------------------------------------------------ device
# g1bW's wait_gpu VERBATIM semantics (double-poll; wait for free windows;
# hot = >80C at gate => soak-wait to <=65C; CPU fallback after max_wait).

def wait_gpu(tag: str, max_wait: float | None = None) -> torch.device:
    mw = (30.0 if SMOKE else (GPU_WAIT_MAX if max_wait is None else max_wait))
    if not torch.cuda.is_available():
        return CPU
    t0, waited, s1 = time.time(), 0.0, gpu_status()
    hot = s1["temp"] > HOT_LAUNCH_C          # lab policy: no launches >80C
    while (time.time() - t0) <= mw:
        if hot:
            tc = gpu_status()["temp"]
            if tc > HOT_LAUNCH_C:
                log(f"[gpu] '{tag}' heat-soaked ({tc:.0f}C > "
                    f"{HOT_LAUNCH_C:.0f}C): waiting to cool below the "
                    f"launch ceiling")
                time.sleep(30)
                waited = time.time() - t0
                continue
            hot = False                      # under the 80C ceiling: the
                                             # window opens (FIX below)
        if gpu_ok():
            time.sleep(5)
            if gpu_ok():
                s2 = gpu_status()
                log(f"[gpu] '{tag}' may use GPU (util {s2['util']:.0f}% temp "
                    f"{s2['temp']:.0f}C mem {s2['mem_used']:.0f}/"
                    f"{s2['mem_total']:.0f}MB"
                    + (f"; waited {waited:.0f}s" if waited > 0 else "") + ")")
                return torch.device("cuda")
        time.sleep(30)
        waited = time.time() - t0
    device_events.append({"tag": tag, "event": "GPU-WAIT TIMEOUT -> CPU",
                          "waited_s": round(waited, 1), "status": s1})
    log(f"[gpu] '{tag}' runs CPU (GPU busy >{mw:.0f}s: {s1})")
    return CPU


def midrun_pause_wait(tag: str) -> None:
    """g1bW policy: outside load/heat => PAUSE, never migrate."""
    while True:
        s = gpu_status()
        if s["mem_total"] == 0 or (s["mem_used"] <= 0.85 * s["mem_total"]
                                   and s["temp"] <= 75.0):
            log(f"  [{tag}] contention cleared ({s}); resuming")
            return
        log(f"  [{tag}] PAUSED for outside load/heat ({s})")
        time.sleep(30)


def flat_params(net) -> torch.Tensor:
    """e185's displacement currency, device-resident (registered): the flat
    fp32 parameter vector in net.parameters() order — the optimizer's own
    currency; 9,977,600 elements here."""
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()])


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("g1bS_smoke" if SMOKE else "g1bS")
    log(f"G1BS THE WALL AT 10x (smoke={SMOKE}) -> {rd}")
    set_seed(HOST_SEED)      # host init + base corpus (the 1337 family)

    # ---- patch g1's machinery to the 10M family (g1b's pattern) ---------
    G1.G1_CFG = G1BS_CFG
    G1.G1_PARAMS = G1BS_PARAMS

    # ---------------- protocol rebuild (g1b's main VERBATIM) -------------
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    corpus_zeph = train_text.count("ZEPH")
    G_NAMEFREE = {"corpus_zeph_count": corpus_zeph,
                  "pass": bool(corpus_zeph == 0)}
    assert G_NAMEFREE["pass"], f"corpus contains ZEPH x{corpus_zeph}"

    host_occ = []
    for host in G1.HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + G1.POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "pass": bool(mix == {"FLORIZEL": 19,
                                                         "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"
    log(f"protocol rebuilt: install60 {mix}, held30 (SPLICE_RNG {E43.SPLICE_RNG})")
    name_ids = corpus.encode(G1.NAME)

    # measurement pool: e152's locked j=54 windows (instrument only)
    wins = []
    for p, h in install_occ:
        pre = train_ids[p - G1.PRE - G1.RETEACH_J: p]
        post = train_ids[p + len(h): p + len(h) + G1.SITE_CONT]
        if len(pre) != G1.PRE + G1.RETEACH_J or len(post) != G1.SITE_CONT:
            raise RuntimeError(f"pool window short at p={p}")
        w = torch.cat([pre, name_ids, post])
        if len(w) != G1.BLOCK:
            raise RuntimeError(f"pool window len {len(w)} != {G1.BLOCK}")
        wins.append(w)
    pool_x = torch.stack(wins)
    G_POOL = {"shape": list(pool_x.shape),
              "name_xcols": [G1.SITE_Z_XCOL, G1.SITE_Z_XCOL + len(G1.NAME) - 1],
              "all_windows_name_in_place": bool(
                  all(torch.equal(w[G1.SITE_Z_XCOL: G1.SITE_Z_XCOL + len(G1.NAME)],
                                  name_ids) for w in pool_x)),
              "note": "MEASUREMENT instrument only (e152's locked j=54 pool); "
                      "NO window from this pool enters any training"}
    G_POOL["pass"] = G_POOL["all_windows_name_in_place"]
    assert G_POOL["pass"], f"pool gate FAILED: {G_POOL}"

    # install windows (e043 build_win verbatim) + original-host anchor bank
    def build_win(p, host):
        return torch.cat([train_ids[p - G1.PRE: p], name_ids,
                          train_ids[p + len(host): p + len(host) + G1.POST_CAP]])

    inst_x = torch.stack([build_win(p, h) for p, h in install_occ])
    anchor_full = torch.stack([train_ids[p - G1.PRE: p - G1.PRE + G1.BLOCK]
                               for p, _ in install_occ])

    # jittered install pool (e113's construction VERBATIM) for consolidation
    jit_x, jit_mask = {}, {}
    for j in G1.JITTERS:
        jwins = []
        for p, h in install_occ:
            pre = train_ids[p - G1.PRE - j: p]
            post = train_ids[p + len(h): p + len(h) + G1.POST_CAP - j]
            w = torch.cat([pre, name_ids, post])
            if len(w) != G1.BLOCK:
                raise RuntimeError(f"jit window len {len(w)} != {G1.BLOCK} at {j}")
            jwins.append(w)
        jit_x[j] = torch.stack(jwins)
        m = torch.zeros(len(jwins), G1.BLOCK - 1, dtype=torch.bool)
        m[:, G1.PRE - 1 + j: G1.PRE - 1 + j + len(G1.NAME)] = True
        jit_mask[j] = m
    pool_a_x = torch.cat([jit_x[j] for j in G1.JITTERS])
    pool_a_mask = torch.cat([jit_mask[j] for j in G1.JITTERS])
    cons_anchor = anchor_full[:16]      # e113: first-16-install original bank

    # ---- e170's neutral bank (e176N arm A's stream, VERBATIM) -----------
    arng = random.Random(G1.E170_ANCHOR_SEED)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - G1.BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        txt = train_text[s: s + G1.BLOCK + 1]
        if any(f in txt for f in G1.ANCHOR_FORBIDDEN):
            rejections += 1
            continue
        n_starts.append(s)
    if len(n_starts) != 16:
        raise RuntimeError(f"neutral bank incomplete: {len(n_starts)}/16")
    anchor_neutral = torch.stack([train_ids[s: s + G1.BLOCK] for s in n_starts])

    host_positions = [p for p in E43.find_occ(train_text, G1.HOSTS[0])]
    host_positions += [p for p in E43.find_occ(train_text, G1.HOSTS[1])]
    jc_neutral = sum(1 for s in n_starts
                     if any(s <= p < s + G1.BLOCK + 1 for p in host_positions))
    G_ANCHOR = {
        "neutral_bank": {
            "construction": ("16 plain corpus windows from train_ids, RNG seed "
                             f"{G1.E170_ANCHOR_SEED}, rejection if [s, s+257) "
                             "contains FLORIZEL/ELIZABETH/ZEPH/MIRABEL — "
                             "e170's construction VERBATIM (= e176n arm A's "
                             "stream; FIXED content, shared by ALL arms)"),
            "n_windows": 16, "block": G1.BLOCK, "seed": G1.E170_ANCHOR_SEED,
            "starts": n_starts, "tries": tries, "rejections": rejections,
            "forbidden": list(G1.ANCHOR_FORBIDDEN),
            "windows_with_host_content": sum(
                1 for s in n_starts
                if any(f in train_text[s: s + G1.BLOCK + 1] for f in G1.HOSTS)),
            "junctions_covered": jc_neutral,
        },
        "budget_identical_to_e176n": bool(anchor_neutral.shape == (16, G1.BLOCK)),
    }
    G_ANCHOR["pass"] = bool(
        G_ANCHOR["neutral_bank"]["windows_with_host_content"] == 0
        and G_ANCHOR["neutral_bank"]["junctions_covered"] == 0
        and G_ANCHOR["budget_identical_to_e176n"])
    assert G_ANCHOR["pass"], f"anchor gate FAILED: {G_ANCHOR}"
    log(f"G_ANCHOR: neutral bank 16x{G1.BLOCK} (seed {G1.E170_ANCHOR_SEED}, "
        f"{rejections} rejections/{tries} tries) — host 0/16, junctions "
        f"0/16: PASS")

    # ---------------- batteries (e119/e176n verbatim) --------------------
    bat_ids, held_ids = {}, {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
        hs = [train_text[p - G1.PRE - j: p] for p, _ in held_occ]
        held_ids[j] = torch.stack([corpus.encode(c) for c in hs])
    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60, G1.R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    gm12_ids = bat_ids[-12]
    g0_ids = bat_ids[0]

    gates_surg: dict = {}

    def measure(sd: dict, tag: str, lean: bool = False) -> dict:
        """g1b's measure() VERBATIM dial (e176n's = the e131 dial set) on
        evl_load (settle+disarm; g1's PIVOT). CPU-only, offline (no cap)."""
        net = G1.evl_load(sd)
        sd_local = {k: v.detach().clone() for k, v in net.state_dict().items()}
        out: dict = {"tag": tag}
        out["base"] = {j: G1.battery_cell(net, bat_ids[j], zid) for j in G1.GEOS}
        out["base_held"] = {j: G1.battery_cell(net, held_ids[j], zid)
                            for j in G1.GEOS}
        out["ce_r"] = G1.ce_fixed_cpu(net, *r_eval_xy)
        log(f"[{tag}] base: " + " ".join(f"g{j:+d} {out['base'][j]['mean_pz']:.4f}"
                                         for j in G1.GEOS)
            + " | held30: " + " ".join(
                f"g{j:+d} {out['base_held'][j]['mean_pz']:.4f}" for j in G1.GEOS)
            + f" | CE_R {out['ce_r']:.4f}")
        out["site_read"] = G1.read_fact_at(net, pool_x, name_ids, zid,
                                           G1.SITE_ADDR_ROW, G1.SITE_Z_XCOL)
        log(f"[{tag}] site read @183: onset "
            f"{out['site_read']['pz_onset_mean']:.4f} span "
            f"{out['site_read']['pname_mean_over7']:.4f}")
        if not lean and not SMOKE:
            out["census_old"] = G1.row_census(net, G1.ROWS_OLD,
                                              lambda n: G1.battery_pz(
                                                  n, bat_ids[0], zid))
            co = out["census_old"]["rows"]
            out["old_band"] = {
                "base_pz": out["census_old"]["base_readout"],
                "row0_strength": co["0"]["strength"],
                "A129": co["129"]["strength"],
                "band121_129_max": max(co[str(r)]["strength"]
                                       for r in range(121, 130)
                                       if str(r) in co)}
            log(f"[{tag}] old band: row0 S "
                f"{out['old_band']['row0_strength']:+.4f} | A(129) "
                f"{out['old_band']['A129']:+.4f}")
            DELS = {"d_all": G1.D_ALL, "d183": (G1.SITE_ADDR_ROW,)}
            out["del_table"] = {}
            for dl, rows_ in DELS.items():
                sd_d, gate = G1.deleted_wpe(sd_local, rows_)
                gates_surg[f"{tag}__{dl}"] = gate
                if not gate["pass"]:
                    raise RuntimeError(f"deletion gate FAILED {tag}/{dl}: "
                                       f"{gate}")
                net.load_state_dict(sd_d)
                cell = {"g0": G1.battery_cell(net, bat_ids[0], zid)["mean_pz"]}
                if dl == "d183":
                    cell["gm12"] = G1.battery_cell(net, bat_ids[-12],
                                                   zid)["mean_pz"]
                out["del_table"][dl] = cell
            net.load_state_dict(sd_local)
            log(f"[{tag}] deletions g0: " + " | ".join(
                f"{dl} {out['del_table'][dl]['g0']:.3f}" for dl in DELS))
        else:
            w = net.wpe.weight.data
            orig = w.clone()
            mean_row = orig.mean(0)
            bp = G1.battery_pz(net, bat_ids[0], zid)
            w[129] = mean_row
            m129 = bp - G1.battery_pz(net, bat_ids[0], zid)
            w.copy_(orig)
            w[129] = 0.0
            z129 = bp - G1.battery_pz(net, bat_ids[0], zid)
            w.copy_(orig)
            assert torch.equal(w, orig), "lean A129 failed to restore wpe"
            out["A129_quick"] = float(min(m129, z129))
        del net
        return out

    def flat_cells(m: dict) -> dict:
        c = {"gm12": m["base"][-12]["mean_pz"],
             "g0": m["base"][0]["mean_pz"],
             "gp12": m["base"][12]["mean_pz"],
             "held30_gm12": m["base_held"][-12]["mean_pz"],
             "held30_g0": m["base_held"][0]["mean_pz"],
             "ce_r": m["ce_r"],
             "site_read_onset": m["site_read"]["pz_onset_mean"],
             "site_read_span": m["site_read"]["pname_mean_over7"]}
        if "old_band" in m:
            c["A129"] = m["old_band"]["A129"]
            c["row0_strength"] = m["old_band"]["row0_strength"]
            c["dall_g0"] = m["del_table"]["d_all"]["g0"]
            c["d183_g0"] = m["del_table"]["d183"]["g0"]
            c["d183_gm12"] = m["del_table"]["d183"]["gm12"]
        else:
            c["A129"] = m["A129_quick"]
        return c

    # =====================================================================
    # G-CONFIG: the 10M band + the R-convention assertion (Rule 12)
    # =====================================================================
    torch.manual_seed(HOST_SEED)
    host_net = TinyGPT(G1BS_CFG)
    n_params = host_net.num_params()
    G_CONFIG = {
        "params": n_params,
        "band": [8.5e6, 11.5e6],
        "vocab": G1BS_CFG.vocab, "block_size": G1BS_CFG.block_size,
        "n_layer": G1BS_CFG.n_layer, "n_head": G1BS_CFG.n_head,
        "n_embd": G1BS_CFG.n_embd,
        "R_rms_ref": R_RMS_REF,
        "R_ladder_raw": list(R_LADDER_RAW),
        "R_ladder_rms": list(R_LADDER_RMS),
        "R_ladder_mult": list(R_LADDER_MULT),
        "pin_fuzz_rms_carried_raw": PIN_FUZZ_RMS_CARRIED,
        "one_adamw_step_fuzz_raw": ONE_STEP_FUZZ_RAW,
        "assertions": {
            f"rung_{i+1}": {
                "mult": R_LADDER_MULT[i],
                "rms": R_LADDER_RMS[i],
                "expected_rms": R_LADDER_MULT[i] * R_RMS_REF,
                "raw": R_LADDER_RAW[i],
                "abs_err_rms": abs(R_LADDER_RMS[i]
                                   - R_LADDER_MULT[i] * R_RMS_REF),
                "pass": bool(abs(R_LADDER_RMS[i]
                                 - R_LADDER_MULT[i] * R_RMS_REF) < 1e-15),
            } for i in range(3)},
        "battery_geometry": {
            "vocab_unchanged": bool(G1BS_CFG.vocab == corpus.vocab_size == 65),
            "block_unchanged": bool(G1BS_CFG.block_size == G1.BLOCK == 256),
            "positions_read": {"site_rows": [G1.SITE_ADDR_ROW,
                                             G1.SITE_ADDR_ROW + 6],
                               "wpe_band": [121, 129],
                               "inside_block": True},
            "tokenizer": "same data/input.txt char stoi (65 chars)",
        },
    }
    G_CONFIG["pass"] = bool(
        8.5e6 <= n_params <= 11.5e6
        and all(a["pass"] for a in G_CONFIG["assertions"].values())
        and G_CONFIG["battery_geometry"]["vocab_unchanged"]
        and G_CONFIG["battery_geometry"]["block_unchanged"])
    assert G_CONFIG["pass"], f"G-CONFIG FAILED: {G_CONFIG}"
    log(f"G-CONFIG: {n_params:,} params in [8.5M, 11.5M]; ladder "
        + " | ".join(f"W{i+1}: {r:.4f} raw = {m:.0f}x {R_RMS_REF:.4e} rms"
                    for i, (r, m) in enumerate(zip(R_LADDER_RAW,
                                                   R_LADDER_MULT)))
        + f"; pin fuzz rms-carried {PIN_FUZZ_RMS_CARRIED:.3f} raw "
        f"(verbatim 1.5 co-reported; one-step {ONE_STEP_FUZZ_RAW:.3f}): PASS")

    # =====================================================================
    # PHASE 0a-host — THE BASE: ckpt-resumable chunks to a completed cosine
    # =====================================================================
    log("=" * 78)
    log(f"HOST BASE: {n_params:,} params, {BASE_STEPS} steps house cosine "
        f"(warmup 100), batch {BASE_BS}, seed {HOST_SEED}, chunk cap "
        f"{TRAIN_CAP_GPU:.0f}s GPU / {TRAIN_CAP_CPU:.0f}s CPU, "
        f"cooldown {COOLDOWN_S:.0f}s between chunks")
    base_ck = CKPT_DIR / ("smoke_" + BASE_CK if SMOKE else BASE_CK)
    if base_ck.exists() and not SMOKE and os.environ.get("G1BS_FRESH") != "0":
        # a prior interrupted dispatch's chunk state is RESUMABLE by design;
        # never retrain silently — record it and continue the schedule
        log(f"[base] existing chunk state {base_ck.name} found — RESUMING "
            f"(Rule 10; the chunk table records the split)")
    chunks: list[dict] = []
    hist: list[dict] = []
    while True:
        tag = f"base_chunk{len(chunks) + 1}"
        dev = wait_gpu(tag, max_wait=None if SMOKE else BASE_WAIT_MAX)
        common.DEVICE = str(dev)
        host_net.to(dev)
        cap = TRAIN_CAP_GPU if dev.type == "cuda" else TRAIN_CAP_CPU
        t_c = time.time()
        hist = train_model(host_net, corpus, steps=BASE_STEPS, lr=BASE_LR,
                           batch_size=BASE_BS, max_seconds=cap,
                           eval_every=(BASE_EVAL if not SMOKE else 8),
                           ckpt=base_ck)
        chunks.append({"chunk": len(chunks) + 1, "device": str(dev),
                       "cap_s": cap, "seconds": round(time.time() - t_c, 1),
                       "steps_done": hist[-1]["step"],
                       "val_loss": hist[-1]["val_loss"]})
        log(f"[base] chunk {chunks[-1]['chunk']}: step {hist[-1]['step']}/"
            f"{BASE_STEPS} val {hist[-1]['val_loss']:.4f} "
            f"({chunks[-1]['seconds']}s on {dev})")
        if hist and hist[-1]["step"] >= BASE_STEPS:
            break
        if len(chunks) > 40:
            raise RuntimeError(f"base chunks exceeded 40: {chunks}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()   # free MY cache between chunks: the
                                       # shared card's 85% cap is checked at
                                       # every gate, own residency included
        if not SMOKE:
            cooldown(COOLDOWN_S)
    host_net.to(CPU)
    host_net.eval()
    G_BASE = {"steps": hist[-1]["step"], "final_val_loss": hist[-1]["val_loss"],
              "n_chunks": len(chunks), "chunks": chunks,
              "pass": bool(hist[-1]["step"] == BASE_STEPS)}
    assert G_BASE["pass"], f"G-BASE FAILED (cosine incomplete): {G_BASE}"
    log(f"G-BASE: cosine COMPLETED at step {hist[-1]['step']} "
        f"(val {hist[-1]['val_loss']:.4f}) in {len(chunks)} chunks: PASS")
    theta_base = {k: v.detach().clone()
                  for k, v in host_net.state_dict().items()}
    base_cells = flat_cells(measure(theta_base, "g1bS_base", lean=True))
    log(f"base cells: g-12 {base_cells['gm12']:.4f} CE_R "
        f"{base_cells['ce_r']:.4f}")
    del host_net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =====================================================================
    # PHASE 0a — INSTALL: e043 Dmix via common.train_model (g1 verbatim)
    # =====================================================================
    log("=" * 78)
    log(f"INSTALL: e043 Dmix stream ({G1.NAME_BS} install + 16 paired + "
        f"{G1.MIX_RANDOM} random = 64 windows/step), dose s{INSTALL_STEPS}, "
        f"house cosine (warmup 100), seed {G1.INSTALL_SEED}")
    if not SMOKE:
        cooldown(COOLDOWN_S)
    torch.manual_seed(G1.INSTALL_SEED)
    inst_net = TinyGPT(G1BS_CFG)
    inst_net.load_state_dict(theta_base)
    dmix = G1.DmixCorpus(E43.REPO / "data" / "input.txt", seed=G1.INSTALL_SEED,
                         inst_x=inst_x, anchor_full=anchor_full,
                         train_ids=train_ids)
    inst_ck = CKPT_DIR / ("smoke_" + INSTALL_CK if SMOKE else INSTALL_CK)
    ichunks: list[dict] = []
    while True:
        dev = wait_gpu(f"install_chunk{len(ichunks) + 1}")
        common.DEVICE = str(dev)
        inst_net.to(dev)
        cap = TRAIN_CAP_GPU if dev.type == "cuda" else TRAIN_CAP_CPU
        t_c = time.time()
        ihist = train_model(inst_net, dmix, steps=INSTALL_STEPS, lr=1e-3,
                            batch_size=64, max_seconds=cap,
                            eval_every=(100 if not SMOKE else 4),
                            ckpt=inst_ck)
        ichunks.append({"chunk": len(ichunks) + 1, "device": str(dev),
                        "seconds": round(time.time() - t_c, 1),
                        "steps_done": ihist[-1]["step"]})
        if ihist and ihist[-1]["step"] >= INSTALL_STEPS:
            break
        if len(ichunks) > 10:
            raise RuntimeError(f"install chunks exceeded 10: {ichunks}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not SMOKE:
            cooldown(COOLDOWN_S)
    inst_net.to(CPU)
    inst_net.eval()
    G_INST = {"steps": ihist[-1]["step"], "n_chunks": len(ichunks),
              "chunks": ichunks,
              "pass": bool(ihist[-1]["step"] == INSTALL_STEPS)}
    assert G_INST["pass"], f"INSTALL incomplete: {G_INST}"
    theta_install = {k: v.detach().clone()
                     for k, v in inst_net.state_dict().items()}
    inst_cells = flat_cells(measure(theta_install, "post_install", lean=True))
    log(f"post-install: g-12 {inst_cells['gm12']:.4f} g0 "
        f"{inst_cells['g0']:.4f} CE_R {inst_cells['ce_r']:.4f}")
    del inst_net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =====================================================================
    # PHASE 0b — CONSOLIDATE: e113 jitter convention (own driver, verbatim
    # arithmetic; registered scale adaptations in the header)
    # =====================================================================
    log("=" * 78)
    if not SMOKE:
        cooldown(COOLDOWN_S)
    cons = g1bS_consolidate(G1.evl_load(theta_install), pool_a_x, pool_a_mask,
                            cons_anchor, train_ids, g0_ids, r_eval_xy, zid,
                            "consolidate")
    theta_root = cons["sd"]
    if cons["steps_ran"] < CONS_STEPS:
        trims.append(f"consolidate capped at step {cons['steps_ran']} of "
                     f"{CONS_STEPS} (documented, not silent)")

    # ---- the root battery + G-ROOT --------------------------------------
    log("=" * 78)
    root = measure(theta_root, "g1bS_root", lean=SMOKE)
    root_cells = flat_cells(root)
    G_ROOT0 = {"bar": G1.EXPRESS_BAR, "gm12": root_cells["gm12"],
               "pass": bool(root_cells["gm12"] >= G1.EXPRESS_BAR)}
    log(f"G-ROOT: root g-12 {root_cells['gm12']:.4f} "
        f"(bar >= {G1.EXPRESS_BAR}): "
        f"{'PASS' if G_ROOT0['pass'] else 'FAIL'}")
    if not G_ROOT0["pass"] and not SMOKE:
        log("G-ROOT FAILED — arms still run for the record; verdict will be "
            "TEXTURE (gate failure), nothing adjudicated")

    # ---- THE SCALED BAR, computed from the root BEFORE any arm trains ----
    bar_0p9eq = root_cells["gm12"] * 0.9 / G1B_ROOT_GM12
    SCALED_BAR = {
        "definition": REGISTERED["scaled_bar_definition"],
        "root_gm12_measured": root_cells["gm12"],
        "bar_0p9eq": bar_0p9eq,
        "bar_09x_root": 0.9 * root_cells["gm12"],
        "g1b_W1_reference": {"min": 0.7768, "argmin_step": 2,
                             "min_retention_vs_root": 0.7768 / G1B_ROOT_GM12,
                             "note": "g1b's own W1 dipped below BOTH "
                                     "0.9x-root and the 0.9eq bar at +2/+4"},
        "computed_before_arms": True,
    }
    log(f"SCALED BAR (frozen formula, root measured): bar_0p9eq = "
        f"{root_cells['gm12']:.4f} x 0.9 / {G1B_ROOT_GM12:.4f} = "
        f"{bar_0p9eq:.4f}  (= {0.9 / G1B_ROOT_GM12:.5f} x root; secondary "
        f"0.9x root = {0.9 * root_cells['gm12']:.4f})")

    save_ckpt(ROOT_CK, theta_root,
              {"desc": f"fresh 10M base (seed {HOST_SEED}, {len(chunks)} "
                       f"chunks, cosine {BASE_STEPS}) + Dmix install s"
                       f"{INSTALL_STEPS} (seed {G1.INSTALL_SEED}) + e113 "
                       f"jitter consolidation s{cons['steps_ran']} (seed "
                       f"{G1.CONS_SEED}) — theta0 for every g1bS arm",
               "params": n_params, "base_steps": BASE_STEPS,
               "base_chunks": len(chunks), "host_seed": HOST_SEED,
               "install_steps": INSTALL_STEPS, "install_seed": G1.INSTALL_SEED,
               "cons_steps": cons["steps_ran"], "cons_seed": G1.CONS_SEED})
    theta0 = theta_root

    # =====================================================================
    # THE ARMS — C + the RMS ladder (cooldown 120 s before each; ALL inputs
    # bit-identical; the ONLY delta is the commit radius)
    # =====================================================================
    ARM_SPECS = ([("C", None,
                   "CONTROL — uncommitted neutral wash (e176N arm A "
                   "VERBATIM at 10M): the kill clock, D_kill, the CE "
                   "adaptation curve")]
                  + [(f"W{i+1}", R_LADDER_RAW[i], R_LADDER_MULT[i],
                      f"WALL {R_LADDER_MULT[i]:.0f}x R_rms = {R_LADDER_RAW[i]:.4f} "
                      f"raw L2 = {R_LADDER_RMS[i]:.4e} rms — commit then the "
                      f"identical neutral wash; step-1 weights equal to C's "
                      f"(the wall first acts at forward 2)") for i in range(3)])

    arms: dict = {}
    batteries_all: dict = {}
    G_BITROOT = {}
    for spec in ARM_SPECS:
        tag, R, *rest = spec
        desc = rest[-1]
        log("=" * 78)
        if not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before {tag}")
            cooldown(COOLDOWN_S)
        log(f"ARM {tag} — {desc}")
        net0 = G1.evl_load(theta0) if R is None else G1.CommittedGPT(G1BS_CFG)
        if R is not None:
            net0.load_state_dict(theta0)
            net0.commit(R)
            body, _ = G1.split_anchored_sd(net0.state_dict())
            md = max(float((body[k].float() - theta0[k].float()).abs().max())
                     for k in theta0)
            anch_ok = all(torch.equal(net0._anchor(n), p.detach())
                          for n, p in net0.named_parameters())
            G_BITROOT[tag] = {"max_abs_diff": md,
                              "anchors_bit_equal": bool(anch_ok),
                              "n_anchor_tensors": net0._n_anchor_tensors,
                              "pass": bool(md == 0.0 and anch_ok)}
            assert G_BITROOT[tag]["pass"], f"{tag}: wall root != theta0"
            log(f"G_BITROOT[{tag}]: max|diff| {md:.1e}, anchors bit-equal, "
                f"R = {R:.6f} raw = {R / SQRT_P:.6e} rms: PASS")
        arm = g1bS_wash(tag, net0, anchor_neutral, train_ids, itos,
                        r_eval_xy, gm12_ids, g0_ids, zid)
        G_DRAWFREE = {"zeph_violations": arm["zeph_violations"],
                      "pass": bool(arm["zeph_violations"] == 0)}
        assert G_DRAWFREE["pass"], f"{tag}: name token leaked into a window"
        gates_surg[f"G_DRAWFREE_{tag}"] = G_DRAWFREE
        arms[tag] = arm

        # checkpoints: each arm's final only (g1b's wash-arm convention)
        for s in sorted(arm["sds"]):
            if s == max(arm["sds"]):
                save_ckpt(f"g1bS_{tag}_s{s}", arm["sds"][s],
                          {"desc": f"g1bS root + {s}-step true-target neutral "
                                   f"wash (R={R} raw = "
                                   f"{R / SQRT_P if R else None} rms, input "
                                   f"seed {WASH_SEED}, lr {G1.FT_LR})",
                           "steps": int(s), "R_raw": R,
                           "R_rms": R / SQRT_P if R else None,
                           "input_seed": WASH_SEED, "lr": G1.FT_LR,
                           "base": f"runs/checkpoints/{ROOT_CK}.pt"})

        # full dials: W1 at g1b's FULL_DIAL_WASH {2,50,300}; others light-only
        full_at = [s for s in FULL_DIAL_STEPS if s in arm["sds"]] \
            if tag == "W1" else []
        batteries = {}
        for s in full_at:
            log(f"{tag} +{s} full dial (CPU)")
            batteries[str(s)] = measure(arm["sds"][s], f"{tag}{s}", lean=SMOKE)
        batteries_all[tag] = batteries

    # =====================================================================
    # CONSTRUCTION-FIDELITY GATES
    # =====================================================================
    G_INPUTS = {"per_step": {}, "pass": None, "note":
                "the seed-10902 aj/rj draw sequence is shared by ALL arms — "
                "per-step input batches are bit-identical (md5-gated) "
                "through the wash horizon"}
    for step in range(1, CK_WASH[-1] + 1):
        hs = {t: arms[t]["x_hashes"].get(step) for t in arms}
        same = all(h is not None for h in hs.values()) and \
            len(set(hs.values())) == 1
        G_INPUTS["per_step"][step] = {t: hs[t] for t in arms}
        G_INPUTS["per_step"][step]["identical"] = bool(same)
    G_INPUTS["pass"] = bool(all(v["identical"]
                                for v in G_INPUTS["per_step"].values()))
    assert G_INPUTS["pass"], "input streams diverged across arms"
    log(f"G-INPUTS: per-step inputs bit-identical across all {len(arms)} "
        f"arms through +{CK_WASH[-1]}: PASS")

    G_STEP1 = {"per_arm": {}}
    for tag in ("W1", "W2", "W3"):
        body_w, _ = G1.split_anchored_sd(arms[tag]["sds"][1])
        sd_c = arms["C"]["sds"][1]
        md1 = max(float((body_w[k].float().cpu() - sd_c[k].float()).abs().max())
                  for k in sd_c)
        G_STEP1["per_arm"][tag] = {"max_abs_diff": md1, "tol": 1e-4,
                                   "pass": bool(md1 <= 1e-4)}
    G_STEP1["pass"] = bool(all(v["pass"] for v in G_STEP1["per_arm"].values()))
    assert G_STEP1["pass"], f"G-STEP1 FAILED (wall acted at forward 1?): {G_STEP1}"
    log("G-STEP1: wall arms' step-1 bodies == C's step-1 body (max|diff| "
        + ", ".join(f"{v['max_abs_diff']:.1e}"
                    for v in G_STEP1["per_arm"].values())
        + " <= 1e-4): PASS — the wall is inert until forward 2")

    # =====================================================================
    # DISPLACEMENT TABLES + COSINES (e185's currency, measured; both
    # conventions at every checkpoint)
    # =====================================================================
    def disp_rows(tag):
        rows = []
        for t in arms[tag]["traj"]:
            s = t["step"]
            row = {"step": s, "ce_batch": t["ce_batch"],
                   "cum_disp_raw": t["cum_disp"],
                   "cum_disp_rms": t["cum_disp"] / SQRT_P,
                   "step_disp_raw": t["step_disp"],
                   "step_disp_rms": t["step_disp"] / SQRT_P,
                   "d_proj_raw": t["d_proj"],
                   "d_proj_rms": (t["d_proj"] / SQRT_P
                                  if t["d_proj"] is not None else None)}
            if "g_m12_mean_pz" in t:
                row["g_m12_light"] = t["g_m12_mean_pz"]
                row["ce_r_light"] = t["ce_r"]
            if s in arms[tag]["deltas"] and s in arms["C"]["deltas"]:
                d_a = arms[tag]["deltas"][s].float().cpu()
                d_c = arms["C"]["deltas"][s].float().cpu()
                row["cos_vs_C"] = float(torch.dot(d_a, d_c)
                                        / (torch.norm(d_a) * torch.norm(d_c)
                                           + 1e-30))
            rows.append(row)
        return rows

    disp_table = {tag: disp_rows(tag) for tag in arms}

    # =====================================================================
    # ADJUDICATION (the frozen bars; order GATES -> WALL -> COSTS; no
    # shopping; the scaled-bar definition was computed before the arms)
    # =====================================================================
    def light_gm12(tag):
        return {t["step"]: t["g_m12_mean_pz"] for t in arms[tag]["traj"]
                if "g_m12_mean_pz" in t}

    fmt = lambda v: "n/a" if v is None else f"{v:.4f}"

    # ---- G-CTRL: arm C dies by +50
    c_gm12 = light_gm12("C")
    G_CTRL = {"bar": G1.SHUT_BAR, "gm12_at_50": c_gm12.get(50),
              "earliest_le_bar": next((s for s in CK_WASH if s > 0
                                       and c_gm12.get(s, 1.0) <= G1.SHUT_BAR),
                                      None),
              "pass": bool(c_gm12.get(50, 1.0) <= G1.SHUT_BAR)}
    log(f"G-CTRL: arm C g-12 at +50 = {fmt(G_CTRL['gm12_at_50'])} "
        f"(earliest <= {G1.SHUT_BAR}: +{G_CTRL['earliest_le_bar']}): "
        f"{'PASS' if G_CTRL['pass'] else 'FAIL'}")
    t_kill = G_CTRL["earliest_le_bar"]
    D_kill_raw = next((r["cum_disp_raw"] for r in disp_table["C"]
                       if r["step"] == t_kill), None) if t_kill else None

    # ---- G-PIN: both conventions; primary = rms-carried
    G_PIN = {"per_arm": {}, "primary": "rms_carried",
             "fuzz_rms_carried_raw": PIN_FUZZ_RMS_CARRIED,
             "fuzz_verbatim": G1.PIN_FUZZ_BAR,
             "one_step_fuzz_raw": ONE_STEP_FUZZ_RAW}
    for tag in ("W1", "W2", "W3"):
        R = arms[tag]["wall_R"]
        rows = [r for r in disp_table[tag] if "g_m12_light" in r]
        mx = max(r["cum_disp_raw"] for r in rows) if rows else None
        G_PIN["per_arm"][tag] = {
            "R_raw": R, "R_rms": R / SQRT_P,
            "bound_rms_carried": R + PIN_FUZZ_RMS_CARRIED,
            "bound_verbatim": R + G1.PIN_FUZZ_BAR,
            "max_raw_disp_at_ckpt": mx,
            "per_ckpt_raw": {r["step"]: r["cum_disp_raw"] for r in rows},
            "per_ckpt_rms": {r["step"]: r["cum_disp_rms"] for r in rows},
            "pass_rms_carried": bool(
                mx is not None and mx <= R + PIN_FUZZ_RMS_CARRIED),
            "pass_verbatim": bool(
                mx is not None and mx <= R + G1.PIN_FUZZ_BAR),
        }
        log(f"G-PIN[{tag}]: max raw |d| {fmt(mx)} <= "
            f"{R + PIN_FUZZ_RMS_CARRIED:.3f} (rms-carried): "
            f"{'PASS' if G_PIN['per_arm'][tag]['pass_rms_carried'] else 'FAIL'}"
            f" | verbatim R+1.5 = {R + G1.PIN_FUZZ_BAR:.2f}: "
            f"{'PASS' if G_PIN['per_arm'][tag]['pass_verbatim'] else 'FAIL'}"
            f" (co-reported)")
    G_PIN["pass"] = bool(all(v["pass_rms_carried"]
                             for v in G_PIN["per_arm"].values()))

    gates_pass = bool(G_CONFIG["pass"] and G_BASE["pass"] and G_ROOT0["pass"]
                      and G_CTRL["pass"] and G_PIN["pass"]
                      and G_INPUTS["pass"] and G_STEP1["pass"]
                      and all(v["pass"] for v in G_BITROOT.values())
                      and G_INST["pass"])

    # ---- the ladder verdicts against the frozen scaled bar
    def arm_verdict(tag):
        g = light_gm12(tag)
        vals = [g[s] for s in CK_WASH if s in g]
        complete = bool(len(vals) == len(CK_WASH))   # anti-erosion: a
        # truncated arm (missing checkpoints) cannot hold a bar
        holds_bar = bool(complete and vals
                         and all(v >= bar_0p9eq for v in vals))
        maintains = bool(complete and vals
                         and all(v >= G1.MAINTAIN_BAR for v in vals))
        dies_by_50 = bool(g.get(50, 1.0) <= G1.SHUT_BAR)
        flat_phase = [g[s] for s in (10, 50, 100, 200, 300) if s in g]
        return {
            "g_m12": g,
            "all_checkpoints_present": complete,
            "missing": [s for s in CK_WASH if s not in g],
            "min_gm12": min(vals) if vals else None,
            "argmin_step": (min(g, key=lambda s: g[s]) if vals else None),
            "holds_0p9eq_bar": holds_bar,
            "first_ck_below_bar": next((s for s in CK_WASH
                                        if g.get(s, 1.0) < bar_0p9eq), None),
            "maintains_g1_bar": maintains, "dies_by_50": dies_by_50,
            "retention_min": (min(v / root_cells["gm12"] for v in vals)
                              if vals else None),
            "flat_phase_min": (min(flat_phase) if flat_phase else None),
            "flat_phase_retention_min": (
                min(flat_phase) / root_cells["gm12"]
                if flat_phase else None),
            "flat_phase_holds_09x_root": bool(
                flat_phase and min(flat_phase) >= 0.9 * root_cells["gm12"]),
        }

    ladder = {tag: arm_verdict(tag) for tag in ("C", "W1", "W2", "W3")}

    # ---- COSTS: the tax curve (the +0.53 reference re-priced)
    def ce_at(tag, step):
        return next((t["ce_batch"] for t in arms[tag]["traj"]
                     if t["step"] == step), None)

    ce_tax = {
        "W1_ce300": ce_at("W1", CK_WASH[-1]), "C_ce300": ce_at("C", CK_WASH[-1]),
        "W2_ce300": ce_at("W2", CK_WASH[-1]), "W3_ce300": ce_at("W3", CK_WASH[-1]),
        "root_ce_r": root_cells["ce_r"],
        "reference_274M": PRIORS_G1B["wall_tax_274M"],
    }
    ce_tax["W1_minus_C"] = (ce_tax["W1_ce300"] - ce_tax["C_ce300"]) \
        if None not in (ce_tax["W1_ce300"], ce_tax["C_ce300"]) else None
    ce_tax["W1_minus_root"] = (ce_tax["W1_ce300"] - ce_tax["root_ce_r"]) \
        if ce_tax["W1_ce300"] is not None else None
    ce_tax["per_step_curve"] = {
        tag: {t["step"]: t["ce_batch"] for t in arms[tag]["traj"]}
        for tag in arms}
    FREEZING_CE = bool(ce_tax["W1_ce300"] is not None
                       and ce_tax["W1_ce300"] >= ce_tax["root_ce_r"])

    # ---- the composed verdict (the frozen three; no shopping)
    WALL_SCALES_G = WALL_TIGHTENS_G = WALL_FADES_G = None
    if not gates_pass:
        failed = [k for k, g in (("G-CONFIG", G_CONFIG), ("G-BASE", G_BASE),
                                 ("G-INST", G_INST), ("G-ROOT", G_ROOT0),
                                 ("G-CTRL", G_CTRL), ("G-PIN", G_PIN),
                                 ("G-INPUTS", G_INPUTS),
                                 ("G-STEP1", G_STEP1),
                                 ("G-BITROOT", {"pass": all(
                                     v["pass"] for v in G_BITROOT.values())})
                                 ) if not g["pass"]]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a registered gate failed — nothing adjudicated; failed: "
                  f"{failed}; full record reported (root g-12 "
                  f"{fmt(root_cells['gm12'])}, C trace "
                  + " -> ".join(f"+{s}:{fmt(v)}" for s, v in
                                sorted(c_gm12.items())) + ")")
    else:
        w1_holds = ladder["W1"]["holds_0p9eq_bar"]
        looser_hold = [t for t in ("W2", "W3") if ladder[t]["holds_0p9eq_bar"]]
        WALL_SCALES = bool(w1_holds and G_CTRL["pass"])
        WALL_TIGHTENS = bool(looser_hold and G_CTRL["pass"])
        WALL_FADES = bool(G_CTRL["pass"] and not WALL_SCALES
                          and (not any(ladder[t]["holds_0p9eq_bar"]
                                       for t in ("W1", "W2", "W3"))
                               or (w1_holds and FREEZING_CE)))
        if WALL_SCALES and WALL_TIGHTENS:
            verdict = "WALL-SCALES + WALL-TIGHTENS"
            clause = (f"at the rms-matched rung (R = {R_LADDER_RAW[0]:.4f} raw "
                      f"= {R_LADDER_RMS[0]:.4e} rms) the fact HELD the "
                      f"0.9-equivalent bar ({bar_0p9eq:.4f} = "
                      f"{0.9 / G1B_ROOT_GM12:.4f} x root) at every checkpoint "
                      f"through +{CK_WASH[-1]} (min "
                      f"{fmt(ladder['W1']['min_gm12'])} at "
                      f"+{ladder['W1']['argmin_step']}) while C died by +50 "
                      f"(dead at +{t_kill}) — AND looser rungs also held "
                      f"({','.join(looser_hold)}): the basin GREW with "
                      f"dimension; tax curve re-priced: W1-C dCE@300 "
                      f"{fmt(ce_tax['W1_minus_C'])} vs the 2.74M +0.53 "
                      f"reference.")
        elif WALL_SCALES:
            verdict = "WALL-SCALES"
            clause = (f"at the rms-matched rung (R = {R_LADDER_RAW[0]:.4f} raw "
                      f"= {R_LADDER_RMS[0]:.4e} rms) the fact HELD the "
                      f"0.9-equivalent bar ({bar_0p9eq:.4f}) at every "
                      f"checkpoint through +{CK_WASH[-1]} (min "
                      f"{fmt(ladder['W1']['min_gm12'])} at "
                      f"+{ladder['W1']['argmin_step']}, retention "
                      f"{fmt(ladder['W1']['retention_min'])}x root) while C "
                      f"died by +50 (dead at +{t_kill}, D_kill "
                      f"{fmt(D_kill_raw)} raw = "
                      f"{D_kill_raw / SQRT_P if D_kill_raw else float('nan'):.4e}"
                      f" rms vs the 2.74M {PRIORS_G1B['D_kill_rms_274M']:.4e})"
                      f" — the wall generalizes at 10x with the dial carried "
                      f"in per-coordinate RMS units; tax: W1-C dCE@300 "
                      f"{fmt(ce_tax['W1_minus_C'])} (2.74M reference +0.53); "
                      f"looser rungs did NOT hold the strict bar ("
                      + " | ".join(f"{t}: min {fmt(ladder[t]['min_gm12'])}, "
                                   f"first below bar +"
                                   f"{ladder[t]['first_ck_below_bar']}"
                                   for t in ("W2", "W3")) + ").")
        elif WALL_TIGHTENS:
            verdict = "WALL-TIGHTENS"
            clause = (f"C died by +50 (dead at +{t_kill}) and the rms-matched "
                      f"rung did NOT hold the strict bar (W1 min "
                      f"{fmt(ladder['W1']['min_gm12'])} at "
                      f"+{ladder['W1']['argmin_step']}, first below "
                      f"+{ladder['W1']['first_ck_below_bar']}) but a LOOSER "
                      f"rung did ({','.join(looser_hold)}) — the basin grew "
                      f"with dimension; tax curve: W1-C dCE@300 "
                      f"{fmt(ce_tax['W1_minus_C'])} vs +0.53; full ladder "
                      "traces reported.")
        elif WALL_FADES:
            if w1_holds and FREEZING_CE:
                verdict = "WALL-FADES (only the tightest R holds, CE frozen)"
                clause = (f"only W1 held the bar and the wall FROZE adaptation "
                          f"(W1 CE@300 {fmt(ce_tax['W1_ce300'])} >= root CE_R "
                          f"{fmt(ce_tax['root_ce_r'])}); C died at +{t_kill}.")
            else:
                verdict = "WALL-FADES"
                clause = (f"C died by +50 (dead at +{t_kill}) but NO rung on "
                          f"the RMS ladder held the 0.9-equivalent bar "
                          f"({bar_0p9eq:.4f} = "
                          f"{0.9 / G1B_ROOT_GM12:.4f} x root): "
                          + " | ".join(
                              f"{t} ({R_LADDER_RAW[i]:.3f} raw = "
                              f"{R_LADDER_MULT[i]:.0f}x rms): min "
                              f"{fmt(ladder[t]['min_gm12'])} at +"
                              f"{ladder[t]['argmin_step']}, first below bar "
                              f"+{ladder[t]['first_ck_below_bar']}, g1-verbatim "
                              f"maintains={ladder[t]['maintains_g1_bar']}, "
                              f"flat-phase retention "
                              f"{fmt(ladder[t]['flat_phase_retention_min'])}"
                              for i, t in enumerate(("W1", "W2", "W3")))
                          + f" — honest verdict per the frozen bars, no dial "
                          f"search; the flat-phase retention and g1-verbatim "
                          f"maintain bars are co-reported for the "
                          f"raw-vs-rms-geometry read.")
        else:
            verdict = "TEXTURE"
            clause = ("no clause composed cleanly (gates passed; report the "
                      "full ladder traces)")
        WALL_SCALES_G, WALL_TIGHTENS_G, WALL_FADES_G = \
            WALL_SCALES, WALL_TIGHTENS, WALL_FADES

    log("=" * 78)
    log(f"G1BS VERDICT: {verdict}")
    for tag in ("C", "W1", "W2", "W3"):
        g = ladder[tag]["g_m12"]
        log(f"  {tag}: g-12 " + " -> ".join(f"+{s}:{v:.4f}"
                                            for s, v in sorted(g.items())))
        log(f"  {tag}: |d| raw/rms " + " -> ".join(
            f"+{r['step']}:{r['cum_disp_raw']:.3f}/{r['cum_disp_rms']:.2e}"
            for r in disp_table[tag] if "g_m12_light" in r))
    log(f"  bar_0p9eq {bar_0p9eq:.4f} | root {root_cells['gm12']:.4f} | "
        f"D_kill raw {fmt(D_kill_raw)}")
    log(f"  tax: W1-C dCE@300 {fmt(ce_tax['W1_minus_C'])} (ref +0.53) | "
        f"W1-root {fmt(ce_tax['W1_minus_root'])} | freezing {FREEZING_CE}")
    log(f"  {clause}")
    log("=" * 78)

    # =====================================================================
    # OUTPUTS
    # =====================================================================
    trace = {}
    for tag in arms:
        rows = [{"freeze_steps": 0,
                 **{k: root_cells[k] for k in
                    ("gm12", "g0", "gp12", "held30_gm12", "held30_g0",
                     "ce_r", "site_read_onset", "site_read_span")}}]
        g0_light = {t["step"]: t["g0_mean_pz"] for t in arms[tag]["traj"]
                    if "g0_mean_pz" in t}
        for s in sorted(arms[tag]["sds"]):
            c = None
            if str(s) in batteries_all.get(tag, {}):
                c = flat_cells(batteries_all[tag][str(s)])
            lg = light_gm12(tag).get(s)
            if c is not None:
                rows.append({"freeze_steps": s, **c})
            elif lg is not None:
                rows.append({"freeze_steps": s, "gm12": lg,
                             "g0": g0_light.get(s),
                             "ce_r": next((t["ce_r"] for t in arms[tag]["traj"]
                                           if t["step"] == s), None)})
        trace[tag] = rows

    metrics = {
        "experiment": "g1bS_scale_wall",
        "date": common.now_iso(),
        "design": "scratch/g1bS_design.md (convention frozen at dispatch; "
                  "bars verbatim; no bar shopping)",
        "question": ("does the commit-and-project L2 ball hold a "
                     "consolidated fact through a wash that kills the "
                     "control, at ~10x the parameters (9,977,600 vs the "
                     "2,739,072 every g-law was minted on), with the R-dial "
                     "carried in per-coordinate RMS units?"),
        "registered": REGISTERED,
        "provenance": {
            "host": {"config": {"n_layer": 8, "n_head": 8, "n_embd": 320,
                                "block_size": 256, "vocab": 65},
                     "params": n_params, "band": "[8.5M, 11.5M]",
                     "config_provenance": "e005s LARGE's 10M-class config "
                                          "(extend-don't-repeat); e005s's "
                                          "own large ckpt (1086/4000 steps, "
                                          "incomplete cosine) NOT reused",
                     "host_seed": HOST_SEED, "corpus_seed": 1337,
                     "recipe": f"common.train_model steps={BASE_STEPS} "
                               f"lr=1e-3 batch=64 cosine warmup 100",
                     "chunks": chunks, "n_chunks": len(chunks),
                     "final_val_loss": G_BASE["final_val_loss"]},
            "install": {"steps": INSTALL_STEPS, "seed": G1.INSTALL_SEED,
                        "chunks": ichunks,
                        "stream": "e043 Dmix via common.train_model "
                                  "(DmixCorpus; g1 phase-0a VERBATIM)",
                        "post_install_cells": inst_cells},
            "consolidation": {"steps": cons["steps_ran"],
                              "seed": G1.CONS_SEED, "device": cons["device"],
                              "traj": cons["traj"],
                              "recipe": "e113 jitter convention VERBATIM "
                                        "(jitters {-8..+8}, batch 16+16, "
                                        "name-masked union CE, AdamW "
                                        "(0.9,0.95) wd 0.1 const lr 1e-3)"},
            "wash": {"recipe": "e176N arm A VERBATIM (neutral bank seed 170, "
                               "batch 32 = 16 neutral + 16 random, full-token "
                               "CE, AdamW (0.9,0.95) wd 0.1 const lr 1e-3 "
                               "clip 1.0)",
                     "seed": WASH_SEED, "ckpt_steps": list(CK_WASH),
                     "devices": {t: arms[t]["device"] for t in arms},
                     "runtimes_s": {t: round(
                         arms[t]["traj"][-1]["elapsed_s"], 1) for t in arms}},
            "R_convention": {"R_rms_ref": R_RMS_REF,
                             "derivation": "0.7/sqrt(2739072)",
                             "ladder_mult": list(R_LADDER_MULT),
                             "ladder_raw": list(R_LADDER_RAW),
                             "ladder_rms": list(R_LADDER_RMS),
                             "assertion": "every rung's rms == mult x "
                                          "R_rms_ref (G-CONFIG, asserted)"},
            "seeds": {"host_init_base_corpus": HOST_SEED,
                      "install": G1.INSTALL_SEED,
                      "consolidation": G1.CONS_SEED,
                      "wash": WASH_SEED, "protocol_corpus": 1337},
            "pauses": device_events,
        },
        "scaled_bar": SCALED_BAR,
        "priors_g1b": PRIORS_G1B,
        "arms": {
            tag: {"desc": desc, "R_raw": R,
                  "R_rms": (R / SQRT_P if R else None),
                  "R_mult": (R / (R_RMS_REF * SQRT_P) if R else None),
                  "ckpt_steps": list(CK_WASH),
                  "steps_ran": arms[tag]["steps_ran"],
                  "device": arms[tag]["device"],
                  "traj": [{k: v for k, v in t.items() if k != "d_proj"}
                           for t in arms[tag]["traj"]],
                  "missing_checkpoints": [s for s in CK_WASH
                                          if s not in arms[tag]["sds"]]}
            for i, (tag, R, *rest) in enumerate(ARM_SPECS)
            for desc in [rest[-1]]},
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": G1.PRE, "post_cap":
                     G1.POST_CAP, "neutral_bank": G_ANCHOR["neutral_bank"],
                     "measure_dial": "g1b's measure() (the e131 dial set) on "
                                     "evl_load (settle+disarm PIVOT), "
                                     "CPU-only"},
        "displacement": {
            "currency": ("cumulative ||theta_t - theta_0||_2 over all "
                         f"{n_params} trainable parameters (fp32, "
                         "device-resident per step — registered scale "
                         "adaptation) in BOTH conventions (raw L2 and "
                         "per-coordinate rms = raw/sqrt(P)) + per-step "
                         "increments + projected displacement min(d, R) + "
                         "cosines vs arm C"),
            "table": disp_table,
            "theta0_norm": {t: arms[t]["theta0_norm"] for t in arms},
            "wall_fuzz_registered": {
                "one_step_lr_sqrtP_raw": ONE_STEP_FUZZ_RAW,
                "pin_fuzz_rms_carried_raw": PIN_FUZZ_RMS_CARRIED,
                "note": "the frozen 1.5 fuzz constant carried in the rms "
                        "convention (primary); the verbatim R+1.5 form is "
                        "co-reported per arm and cannot hold at 10M (one "
                        "AdamW step = 3.159 raw > any R+1.5 on the ladder)"},
        },
        "gates": {"G_CONFIG": G_CONFIG, "G_BASE": G_BASE, "G_INST": G_INST,
                  "G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_POOL": G_POOL, "G_ANCHOR": G_ANCHOR,
                  "G_ROOT": G_ROOT0, "G_CTRL": G_CTRL, "G_PIN": G_PIN,
                  "G_BITROOT": G_BITROOT,
                  "G_INPUTS": {k: v for k, v in G_INPUTS.items()
                               if k != "per_step"} | {"per_step": {
                                   s: v["identical"] for s, v in
                                   G_INPUTS["per_step"].items()}},
                  "G_STEP1": G_STEP1, "G_SURG": gates_surg},
        "traces": trace,
        "batteries": batteries_all,
        "adjudication": {
            "bars_verbatim": REGISTERED["bars_verbatim"],
            "order": "GATES -> WALL ladder vs the scaled bar -> COSTS",
            "gates_pass": gates_pass,
            "ladder": ladder,
            "bar_0p9eq": bar_0p9eq,
            "WALL_SCALES": (WALL_SCALES_G if gates_pass else None),
            "WALL_TIGHTENS": (WALL_TIGHTENS_G if gates_pass else None),
            "WALL_FADES": (WALL_FADES_G if gates_pass else None),
            "D_kill_raw": D_kill_raw,
            "D_kill_rms": (D_kill_raw / SQRT_P if D_kill_raw else None),
            "D_kill_rms_274M_prior": PRIORS_G1B["D_kill_rms_274M"],
            "ce_tax": ce_tax, "freezing_CE": FREEZING_CE,
            "verdict": verdict, "clause": clause,
        },
        "honesty_reflex": {
            "n1_scope": ("n=1 host, one wash seed (10902), one fact — the "
                "rung is SCALE, not seeds; the replicate ladder follows only "
                "if a bar fires (the design's own first-cell scoping)."),
            "intervention_not_logits": ("the wall IS the intervention: all "
                "four arms share bit-identical step-0 weights (G-BITROOT "
                "max|diff| = 0.0), bit-identical per-step inputs (md5-gated, "
                "G-INPUTS) and identical targets; the only delta is the "
                "commit radius, and G-STEP1 shows the commit is inert until "
                "forward 2. G-PIN verifies the ball's mechanics in both "
                "conventions BEFORE any behavioral clause is read."),
            "scaled_bar_strictness": ("the primary bar (0.983x root at every "
                "checkpoint) is STRICTER than g1's maintain bar by design; "
                "g1b's own W1 would have failed it at +2/+4 (retention "
                "0.848/0.893) — the co-reported flat-phase retention and "
                "g1-verbatim maintain/dies bars carry the dip structure so "
                "the fold can separate 'the dip' from 'the fade'."),
            "device_texture": ("the 2.74M references (root 0.9156, +0.53 "
                "tax) were CPU/mixed-device runs; this cell is GPU-trained "
                "with CPU full dials; every bar adjudicates against the "
                "in-run control C, and cross-device float fuzz is bounded "
                "by G-STEP1 (~1e-5 measured on same-stream step-1 bodies)."),
            "cpu_gpu_tax_caveat": ("the tax comparison to +0.53 carries the "
                "device-texture caveat (g1b's dCE was measured on its own "
                "device mix); the in-run W1-vs-C delta is the primary "
                "number."),
            "wall_blind_spot": ("the wall never protects the FIRST step: "
                "commit happens at d=0, so step 1's AdamW update (~3.16 raw "
                "= lr*sqrt(P)) always lands before the first projection; "
                "the wall caps cumulative displacement at R + fuzz — the "
                "design's registered semantics, not a bug."),
            "no_bar_shopping": ("the ladder was fixed by the frozen design; "
                "if no rung holds, WALL-FADES is the verdict; the secondary "
                "bars are co-reported context, never adjudicated."),
        },
        "trims": trims, "deviations": deviations,
        "device_events": device_events,
        "ckpt_inventory": CKPT_INVENTORY,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 8, "n_head": 8, "n_embd": 320,
                   "block_size": 256, "params": n_params,
                   "R_ladder_raw": list(R_LADDER_RAW),
                   "R_ladder_rms": list(R_LADDER_RMS),
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "scale_wall.png", trace, disp_table, ladder, verdict, clause,
         gates_pass, bar_0p9eq, root_cells, ce_tax, D_kill_raw)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'scale_wall.png'}, "
        f"{len(CKPT_INVENTORY)} checkpoints in runs/checkpoints/g1bS_*.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ drivers

def g1bS_consolidate(net0, pool_x, pool_mask, anchor, train_ids,
                     f_eval_ids, r_eval_xy, zid, tag) -> dict:
    """e113's finetune_arm VERBATIM arithmetic (g1's consolidate), with the
    registered scale adaptations: eval twin on the training device; PAUSE
    (never migrate) on outside GPU load."""
    dev = wait_gpu(tag)
    cap = TRAIN_CAP_GPU if dev.type == "cuda" else TRAIN_CAP_CPU
    net = copy.deepcopy(net0).to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=G1.FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(G1.CONS_SEED)
    n_pool, n_anc = pool_x.shape[0], anchor.shape[0]
    traj, t_start = [], time.time()
    evl = copy.deepcopy(net0).to(dev)          # eval twin rides the device
    step = 0
    for step in range(1, CONS_STEPS + 1):
        ix = torch.randint(n_pool, (16,), generator=gen)
        aj = torch.randint(n_anc, (8,), generator=gen)
        rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (8,), generator=gen)
        nw = pool_x[ix].to(dev)
        anc = torch.cat([anchor[aj],
                         torch.stack([train_ids[s: s + G1.BLOCK] for s in rj])],
                        0).to(dev)
        x = torch.cat([nw[:, :-1], anc[:, :-1]], 0)
        y = torch.cat([nw[:, 1:], anc[:, 1:]], 0)
        m = torch.zeros(32, x.shape[1], dtype=torch.bool, device=dev)
        m[:16] = pool_mask[ix].to(dev)
        logits, _ = net(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              y.reshape(-1), reduction="none"
                              ).view(x.shape[0], x.shape[1])
        nm = nll[:16][m[:16]]
        cm = nll[16:]
        loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == CONS_STEPS or \
                (time.time() - t_start) > cap:
            evl.load_state_dict(net.state_dict())
            evl.eval()
            bz = G1.battery_cell(evl, f_eval_ids.to(dev), zid)
            ce_r = G1.ce_fixed_cpu(evl, r_eval_xy[0].to(dev),
                                   r_eval_xy[1].to(dev))
            traj.append({"step": step, "install60_g0_pz": bz["mean_pz"],
                         "ce_r": ce_r,
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] s{step:4d} g0 {bz['mean_pz']:.4f} CE_R "
                f"{ce_r:.4f} ({traj[-1]['elapsed_s']:.0f}s)")
        if (time.time() - t_start) > cap:
            log(f"  [{tag}] time cap {cap:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step} of {CONS_STEPS}")
            break
        if dev.type == "cuda" and step % G1.MIDRUN_POLL_EVERY == 0:
            s = gpu_status()
            if s["mem_total"] > 0 and (s["mem_used"] > 0.85 * s["mem_total"]
                                       or s["temp"] > 80):
                device_events.append({"tag": tag, "step": step,
                                      "event": "MID-RUN PAUSE (no migration)",
                                      "status": s})
                t_p = time.time()
                midrun_pause_wait(tag)
                t_start += time.time() - t_p     # paused time is not cap
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"sd": sd_cpu, "traj": traj, "steps_ran": step,
            "device": str(dev), "seed": G1.CONS_SEED}


def g1bS_wash(tag: str, net0, anchor: torch.Tensor, train_ids: torch.Tensor,
              itos, r_eval_xy, gm12_ids, g0_ids, zid: int,
              lr: float = G1.FT_LR, seed: int = None) -> dict:
    """THE NEUTRAL PLAIN-CORPUS WASH — g1_wash VERBATIM arithmetic (e185's
    noise_wash = e176n's finetune_freeze; draw order aj/rj, batch 32 = 16
    neutral + 16 random, full-token CE, AdamW (0.9,0.95) wd 0.1 const lr,
    clip 1.0, wall bookkeeping, in-batch CE every step, ckpt snapshots +
    light evals, x-hash gate) + the registered scale adaptations (device
    eval twin, device-resident displacement, PAUSE-not-migrate) + the
    R-convention co-print at every checkpoint."""
    seed = WASH_SEED if seed is None else seed
    dev = wait_gpu(tag)
    cap = TRAIN_CAP_GPU if dev.type == "cuda" else TRAIN_CAP_CPU
    ckpt_steps = CK_WASH
    ckpt_set = set(ckpt_steps)
    n_steps = ckpt_steps[-1]
    net = copy.deepcopy(net0).to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    vocab = len(itos)
    n_anc = anchor.shape[0]
    traj, sds, t_start = [], {}, time.time()
    step = 0
    zeph_checks = 0
    evl = copy.deepcopy(net0).to(dev)      # ARMED eval twin on the device
    theta0 = flat_params(net)              # displacement origin (device)
    prev = theta0.clone()
    deltas: dict[int, torch.Tensor] = {}
    x_hashes: dict[int, str] = {}
    wall_R = net0.R
    for step in range(1, n_steps + 1):
        aj = torch.randint(n_anc, (G1.ANCH_BS,), generator=gen)
        rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (G1.RAND_BS,),
                           generator=gen)
        anc = anchor[aj]
        rnd = torch.stack([train_ids[s: s + G1.BLOCK] for s in rj])
        # name-free VERIFY (no-op by corpus construction; hard-fail if not)
        for w in rnd:
            txt = "".join(itos[int(c)] for c in w[:64]) + \
                  "".join(itos[int(c)] for c in w[192:])
            if "ZEPH" in txt:
                zeph_checks += 1
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        x_hashes[step] = hashlib.md5(
            x.contiguous().numpy().tobytes()).hexdigest()
        xd, yd = x.to(dev), y.to(dev)
        logits, _ = net(xd)                    # <- the wall projects here
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               yd.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        cur = flat_params(net)
        cum_disp = float(torch.norm(cur - theta0))
        inc_disp = float(torch.norm(cur - prev))
        prev = cur
        if step in ckpt_set:
            deltas[step] = (cur - theta0).cpu()
        row = {"step": step, "ce_batch": float(loss.item()),
               "cum_disp": cum_disp, "cum_disp_rms": cum_disp / SQRT_P,
               "step_disp": inc_disp,
               "d_proj": min(cum_disp, wall_R) if wall_R else None,
               "elapsed_s": round(time.time() - t_start, 1)}
        if step in ckpt_set:
            sd_cpu = {k: v.detach().cpu().clone()
                      for k, v in net.state_dict().items()}
            sds[step] = sd_cpu
            evl.load_state_dict(sd_cpu)
            evl.eval()
            gz = G1.battery_cell(evl, gm12_ids.to(dev), zid)
            gz0 = G1.battery_cell(evl, g0_ids.to(dev), zid)
            ce_r = G1.ce_fixed_cpu(evl, r_eval_xy[0].to(dev),
                                   r_eval_xy[1].to(dev))
            row.update({"g_m12_mean_pz": gz["mean_pz"],
                        "g0_mean_pz": gz0["mean_pz"],
                        "frac_argmax_z": gz["frac_argmax_z"],
                        "ce_r": ce_r})
            # BOTH CONVENTIONS at every checkpoint (the frozen co-report)
            wr = net.wall_report()
            log(f"  [{tag}] CKPT +{step:4d} g-12 {gz['mean_pz']:.4f} "
                f"g0 {gz0['mean_pz']:.4f} CE_R {ce_r:.4f} | "
                f"|d| {cum_disp:.4f} raw = {cum_disp / SQRT_P:.4e} rms "
                + (f"| R {wall_R:.4f} raw = {wall_R / SQRT_P:.4e} rms "
                   f"(d_proj {wr['d_proj']})" if wall_R else "(uncommitted)")
                + f" (CE {float(loss.item()):.4f})")
        traj.append(row)
        if step % 50 == 0 and step not in ckpt_set:
            log(f"  [{tag}] s{step:4d} CE {float(loss.item()):.4f} "
                f"|d| {cum_disp:.4f} ({row['elapsed_s']:.0f}s)")
        if (time.time() - t_start) > cap:
            log(f"  [{tag}] time cap {cap:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step} of {n_steps}")
            break
        if dev.type == "cuda" and step % G1.MIDRUN_POLL_EVERY == 0:
            s = gpu_status()
            if s["mem_total"] > 0 and (s["mem_used"] > 0.85 * s["mem_total"]
                                       or s["temp"] > 80):
                device_events.append({"tag": tag, "step": step,
                                      "event": "MID-RUN PAUSE (no migration)",
                                      "status": s})
                t_p = time.time()
                midrun_pause_wait(tag)
                t_start += time.time() - t_p     # paused time is not cap
    net.eval()
    return {"sds": sds, "traj": traj, "steps_ran": step,
            "seed": seed, "lr": lr, "target_mode": "true",
            "zeph_violations": zeph_checks, "x_hashes": x_hashes,
            "deltas": deltas, "wall_R": wall_R,
            "theta0_norm": float(torch.norm(theta0.cpu())),
            "device": str(dev)}


CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "g1bS", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name}")


# ------------------------------------------------------------------ plot
# THE SCALE WALL figure: fact survival vs steps (the ladder vs C), the
# R-dial (rms axis), the TAX CURVE, displacement vs the walls (both
# conventions), the retention panel, the verdict.

def plot(path, trace, disp_table, ladder, verdict, clause, gates_pass,
         bar_0p9eq, root_cells, ce_tax, D_kill_raw):
    import textwrap
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))
    cols = {"C": "crimson", "W1": "seagreen", "W2": "royalblue",
            "W3": "darkorange"}
    lbls = {"C": "C — no wall (control)",
            "W1": f"W1 — 1x R_rms ({R_LADDER_RAW[0]:.3f} raw)",
            "W2": f"W2 — 2x R_rms ({R_LADDER_RAW[1]:.3f} raw)",
            "W3": f"W3 — 4x R_rms ({R_LADDER_RAW[2]:.3f} raw)"}
    marks = {"C": "o", "W1": "s", "W2": "^", "W3": "v"}

    def series(tag):
        pts = [(r["freeze_steps"], r["gm12"]) for r in trace[tag]]
        return [p[0] for p in pts], [p[1] for p in pts]

    # (0,0) THE HEADLINE: g-12 vs wash steps
    ax = axes[0, 0]
    for tag in ("C", "W1", "W2", "W3"):
        xs, ys = series(tag)
        ax.plot(xs, ys, marks[tag] + "-", ms=8, lw=2.2, color=cols[tag],
                alpha=0.9, label=lbls[tag])
    ax.axhline(bar_0p9eq, ls="--", lw=1.6, color="black", alpha=0.85,
               label=f"0.9-equivalent bar {bar_0p9eq:.3f}")
    ax.axhline(0.9 * root_cells["gm12"], ls="-.", lw=1.1, color="gray",
               alpha=0.8, label=f"0.9 x root {0.9 * root_cells['gm12']:.3f}")
    for yv, col in ((G1.MAINTAIN_BAR, "seagreen"), (G1.SHUT_BAR, "tab:purple")):
        ax.axhline(yv, ls=":", lw=1.0, color=col, alpha=0.7)
    ax.annotate(f"root {root_cells['gm12']:.3f}", (0, root_cells["gm12"]),
                textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    ax.set_xlabel("neutral-wash steps from the committed root")
    ax.set_ylabel("g-12 (mean p(Z), install-60 battery)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_title(f"THE WALL AT 10x — fact survival ({G1BS_PARAMS:,} params)",
                 fontsize=10)

    # (0,1) THE R-DIAL (rms axis): survival at three horizons
    ax = axes[0, 1]
    rms_pts = [0.0] + list(R_LADDER_RMS)
    tags_by_R = ["C", "W1", "W2", "W3"]
    for step, col, mk in ((2, "gray", "o"), (50, "tab:purple", "s"),
                          (300, "seagreen", "D")):
        ys = [ladder[t]["g_m12"].get(step) for t in tags_by_R]
        ax.plot([r / R_RMS_REF for r in rms_pts], ys, mk + "-", ms=9, lw=2.0,
                color=col, label=f"g-12 at +{step}")
    ax.axhline(bar_0p9eq, ls="--", lw=1.6, color="black", alpha=0.85,
               label=f"0.9eq bar {bar_0p9eq:.3f}")
    for yv, col in ((G1.MAINTAIN_BAR, "seagreen"), (G1.SHUT_BAR, "tab:purple")):
        ax.axhline(yv, ls=":", lw=1.0, color=col, alpha=0.7)
    ax.set_xticks([r / R_RMS_REF for r in rms_pts])
    ax.set_xticklabels(["C\n(no wall)"] + [f"{m:.0f}x R_rms\n({r:.3f} raw)"
                                           for m, r in zip(R_LADDER_MULT,
                                                           R_LADDER_RAW)])
    ax.set_xlabel(f"the wall dial (rms units; R_rms = {R_RMS_REF:.2e})")
    ax.set_ylabel("g-12")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7, loc="center right")
    ax.set_title("THE R-DIAL in rms units (both conventions on the ticks)",
                 fontsize=10)

    # (0,2) THE TAX CURVE
    ax = axes[0, 2]
    for tag in ("C", "W1", "W2", "W3"):
        rows = disp_table[tag]
        ax.plot([r["step"] for r in rows], [r["ce_batch"] for r in rows],
                "-", lw=1.6, color=cols[tag], alpha=0.8,
                label=f"{tag} in-batch CE")
    ax.axhline(ce_tax["root_ce_r"], ls=":", lw=1.3, color="black",
               alpha=0.7, label=f"root CE_R {ce_tax['root_ce_r']:.2f}")
    if ce_tax["W1_minus_C"] is not None:
        frz = bool(ce_tax["W1_ce300"] >= ce_tax["root_ce_r"])
        ax.annotate(f"W1-C dCE@300 {ce_tax['W1_minus_C']:+.3f}\n"
                    f"(2.74M ref +{PRIORS_G1B['wall_tax_274M']:.2f}; "
                    f"freezing={frz})",
                    (0.03, 0.05), xycoords="axes fraction", fontsize=8,
                    weight="bold")
    ax.set_xlabel("wash step")
    ax.set_ylabel("in-batch corpus CE")
    ax.legend(fontsize=7.5)
    ax.set_title("THE TAX CURVE AT SCALE (the +0.53 re-priced)", fontsize=9.5)

    # (1,0) DISPLACEMENT vs the walls (raw; rms twin axis)
    ax = axes[1, 0]
    for tag in ("C", "W1", "W2", "W3"):
        rows = [r for r in disp_table[tag] if "g_m12_light" in r]
        ax.plot([r["step"] for r in rows], [r["cum_disp_raw"] for r in rows],
                marks[tag] + "-", ms=6, lw=1.8, color=cols[tag], alpha=0.9,
                label=lbls[tag])
    for R, col in zip(R_LADDER_RAW, ("seagreen", "royalblue", "darkorange")):
        ax.axhline(R, ls="--", lw=1.0, color=col, alpha=0.6)
        ax.axhline(R + PIN_FUZZ_RMS_CARRIED, ls=":", lw=0.8, color=col,
                   alpha=0.5)
    if D_kill_raw is not None:
        ax.axhline(D_kill_raw, ls=":", lw=1.8, color="k", alpha=0.8)
        ax.annotate(f"D_kill {D_kill_raw:.2f} raw "
                    f"(= {D_kill_raw / SQRT_P / R_RMS_REF:.1f}x R_rms)",
                    (0.01, D_kill_raw), xycoords=("axes fraction", "data"),
                    fontsize=7.5)
    ax.annotate(f"one AdamW step = {ONE_STEP_FUZZ_RAW:.2f} raw",
                (0.99, 0.03), xycoords="axes fraction", ha="right",
                fontsize=7.5)
    ax.set_xlabel("wash step")
    ax.set_ylabel(r"raw $\|\theta_t-\theta_0\|_2$ at checkpoints")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("DISPLACEMENT vs the walls (G-PIN, rms-carried fuzz)",
                 fontsize=10)

    # (1,1) THE RETENTION PANEL (the dip structure, honestly)
    ax = axes[1, 1]
    for tag in ("W1", "W2", "W3"):
        xs, ys = series(tag)
        ax.plot(xs, [y / root_cells["gm12"] for y in ys], marks[tag] + "-",
                ms=7, lw=2.0, color=cols[tag], alpha=0.9,
                label=f"{tag} ({lbls[tag].split('—')[1].strip()})")
    ax.axhline(0.9 / G1B_ROOT_GM12, ls="--", lw=1.6, color="black",
               alpha=0.85, label=f"0.9eq = {0.9 / G1B_ROOT_GM12:.3f} x root")
    ax.axhline(0.9, ls="-.", lw=1.1, color="gray", alpha=0.8,
               label="0.9 x root (secondary)")
    # g1b's W1 reference shape, faint, for the cross-scale eye
    ref = {int(k): v for k, v in PRIORS_G1B["W1_g_m12"].items()}
    ax.plot(sorted(ref), [ref[s] / G1B_ROOT_GM12 for s in sorted(ref)],
            "kx--", ms=5, lw=1.0, alpha=0.45,
            label="g1b W1 @2.74M (reference)")
    ax.set_xlabel("wash step")
    ax.set_ylabel("retention (g-12 / root)")
    ax.set_ylim(-0.03, 1.15)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("RETENTION vs root — the dip structure across scale",
                 fontsize=10)

    # (1,2) THE VERDICT PANEL
    ax = axes[1, 2]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, "G1BS — THE WALL AT 10x", fontsize=11, va="top",
            family="monospace", weight="bold")
    y -= 0.052
    ax.text(0.02, y, f"host {G1BS_PARAMS:,} params (e005s-LARGE cfg, fresh "
            f"cosine) | bar_0p9eq {bar_0p9eq:.4f}", fontsize=7.2, va="top",
            family="monospace")
    y -= 0.034
    for tag in ("C", "W1", "W2", "W3"):
        seq = " -> ".join(f"+{s}:{v:.4f}" for s, v in
                          sorted(ladder[tag]["g_m12"].items()))
        ax.text(0.02, y, f"  {tag}: {seq}", fontsize=6.6, va="top",
                family="monospace", color=cols[tag])
        y -= 0.028
    y -= 0.006
    ax.text(0.02, y, f"  gates {'ALL PASS' if gates_pass else 'FAILURE'} | "
            f"D_kill {D_kill_raw if D_kill_raw is None else round(D_kill_raw, 3)}"
            f" raw | holds0.9eq: "
            + ", ".join(f"{t}={ladder[t]['holds_0p9eq_bar']}"
                        for t in ("W1", "W2", "W3")), fontsize=7.0,
            va="top", family="monospace")
    y -= 0.032
    ax.text(0.02, y, f"  tax W1-C {ce_tax['W1_minus_C']} | freezing_CE "
            f"{bool((ce_tax['W1_ce300'] or 9e9) >= ce_tax['root_ce_r'])} "
            f"(ref +0.53)",
            fontsize=7.0, va="top", family="monospace")
    y -= 0.046
    ax.text(0.02, y, f"VERDICT: {verdict}", fontsize=9.5, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.042
    for wd in textwrap.wrap(clause, width=88, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.6, va="top", family="monospace")
        y -= 0.026

    fig.suptitle("G1BS — THE WALL AT 10x: commit + project at the rms-matched "
                 f"ladder vs the 10M neutral wash -> {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
