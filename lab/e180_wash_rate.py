"""E180 — THE WASH-RATE LAW (the lr-axis completion + the survival-time curve).

WHY (T109/T112/T114): the consolidated fact dissolves under any stream at any
lr tested — but the CLOCK scales with the optimizer. At lr 1e-3 the neutral
stream crosses the 0.27 bar at +2 (e176n arm A: 0.678 -> 0.0271); the original
stream at lr 1e-4 first crosses at +50 (e176n arm B: 0.2480). T112 read that
pair as a BASIN-WIDTH statement (2 steps at 1e-3 ~ displacement 2.5e-3; 50 at
1e-4 ~ 5e-3 — constant-ish lr x steps product), T114 sharpened it into "no
robustness basin; any AdamW step of the wash's size ends it". THIS CELL
completes the lr axis in the gentle regime (1e-5 and 3e-5) and PRICES the
basin-width reading as a law: survival-time-to-under-bar vs lr, five points,
fitted.

QUEUE ROW e180 (verbatim): "THE WASH-RATE LAW (lr 3e-5 + 1e-5; the five-point
curve) | DISPATCHED ~21:20Z (GPU) | LR-SCALED (wash is optimization-rate-
limited — activity-dependence as a rate law) vs LR-IMMUNE (two-step death
even at 1e-5 — the stronger claim)".

REGISTERED BARS (frozen here before compute; the dispatch's registration
verbatim; no bar shopping — adjudicate against exactly this):
  - LR-SCALED fires if: survival time grows monotonically as lr shrinks (no
    under-bar by +300 at 1e-5, or late) — wash is optimization-rate-limited;
    the basin-width reading holds; the paper's "continued training" gets its
    rate qualifier.
  - LR-IMMUNE fires if: under-bar by +50 even at 1e-5 — intrinsic fragility;
    the two-step death is lr-invariant in outcome.
  - No bar shopping; texture (partial survival at 1e-5) => TEXTURE with the
    curve.

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do not
move the bars):
  * g-12 = ABSOLUTE install-60 battery mean p(Z) at ctx offset -12 (e176's
    convention verbatim — the same batteries, the same corpus rebuild, the
    same ruler as e158/e161/e176/e176n/e184).
  * SURVIVAL TIME t*(lr) = the FIRST measured checkpoint step on the cell's
    OWN checkpoint grid with g-12 <= 0.27; the bracket (previous checkpoint,
    t*] is co-reported (grid-resolution honesty); t* is the point estimate
    used in the fit.
  * "under-bar by +50 even at 1e-5" (LR-IMMUNE) = the 1e-5 cell's first
    checkpoint <= 0.27 lies in {2, 10, 50} (this run's grid).
  * "no under-bar by +300 at 1e-5, or late" (LR-SCALED's 1e-5 clause) = the
    1e-5 cell's first checkpoint <= 0.27 is None (never by +300) or in
    {100, 200, 300} — exactly the complement of LR-IMMUNE's condition; the
    clauses are disjoint by construction.
  * "survival time grows monotonically as lr shrinks" = WEAK monotone growth
    of t* across the four lrs {1e-3 -> 2, 1e-4 -> 50, 3e-5 -> t*(3e-5),
    1e-5 -> t*(1e-5)}; a cell that never crosses by +300 contributes t* > 300
    (the substituted value 300 is a LOWER bound, valid for >= comparisons).
  * Adjudication order: LR-IMMUNE -> LR-SCALED -> TEXTURE with the curve;
    every sub-boolean reported regardless.
  * THE FIVE-POINT CURVE: {1e-3 neutral (e176n arm A, grid {1,2,4,...} ->
    t*=2), 1e-3 original (e176's SMOKE fine grid {2,4} -> t*=2; the only
    fine-step trajectory that stream has — e176n docstring's attribution),
    1e-4 original (e176n arm B, grid {50,100,200,300} -> t*=50, COARSE: the
    true value lives in (0,50] — an upper bound, flagged), 3e-5 neutral (NEW),
    1e-5 neutral (NEW)}. The curve mixes streams in its stored half (both
    1e-3 cells agree at t*=2, which is why the mix is tolerable); the
    stream-homogeneous neutral-only fit {1e-3A, 3e-5, 1e-5} is CO-REPORTED.
  * FIT (reported, not bar-adjudicated): OLS on (log lr, log t*) — slope
    alpha (power law t* = C lr^-alpha), R^2; the displacement products
    lr x t* per point (alpha ~ 1 <=> constant product <=> the wash is
    DISPLACEMENT-limited, "linear in lr" in the displacement sense — T112's
    basin-width reading). A never-crossing cell is right-censored: the
    five-point fit is reported with the substitution t* := 300 (marked
    CENSORED) AND as complete-cases-only.

DESIGN: e176N arm A's NEUTRAL protocol VERBATIM (e170's neutral bank: 16
plain-corpus windows, RNG seed 170, rejection on FLORIZEL/ELIZABETH/ZEPH/
MIRABEL in [s, s+257); 0/16 host content, 0/16 junctions; batch 32 = 16
neutral-anchor draws + 16 random corpus windows, full-token CE, NO fact
windows, NO name tokens, NO mask; AdamW (0.9,0.95) wd 0.1 CONSTANT lr clip
1.0, seed 10902 — the locked lineage draw sequence) at lr 3e-5 and lr 1e-5,
300 steps each, checkpoints {2 (the 1e-3 clock's kill point), 10, 50, 100,
200, 300}. ONLY the lr differs from arm A. Light in-run evals (g-12, g0,
CE_R + in-batch corpus CE) at every checkpoint; ONE full dial at +300 per
cell (e176n arm B's rider convention — the anatomy co-report).

COMPUTE ENVELOPE (dispatch): GPU ALLOWED (idle 0% / 69 C at dispatch) —
strict pre-training quick check per training (gpu_status()/gpu_ok(): util
<= 85 AND temp <= 80 C, plus the lab's mem-headroom guard <= 85% of total;
double-poll 5 s apart), PARK-ONCE to CPU on any failure (no re-probing,
never contention with a returning user; e152R's policy via e184 verbatim).
MID-RUN contention guard: every 25 steps of a GPU training, re-poll; mem
> 85% of total or temp > 80 C -> migrate net + optimizer state to CPU and
FINISH THERE (e184's precedent; any device mixing recorded per cell).
cooldown(90 s) before and after EACH training (dispatch: 60-120 s); caps
1800 s per training (dispatch); ALL readouts CPU-side; sequential; NO
concurrent GPU. Torch threads 8 (e152R/e143/e184 convention). Nets are the
mandated 2.7M e131_consolidated line (the dispatch's '<=1M family' note is
an envelope statement — e143/e151/e152/e158/e176n/e184 precedent; every gate
reference and lineage number of these cells lives on the 2.7M line).

INSTRUMENT PROVENANCE: load_cpu/evl_load/battery_cell/battery_pz/
ce_fixed_cpu/val_windows/deleted_wpe/read_fact_at/row_census/measure dial
are lab/e176n_neutral_wash.py VERBATIM (the e176/e161/e152/e151/e143/e131/
e119/e113/e068/e065/e043 lineage); pick_dev/migrate_to_cpu are lab/
e184_seed_replicates.py VERBATIM (e152R's park-once policy + the mid-run
guard); finetune_freeze is e184's device-parameterized copy of e176n's (the
lr becomes a per-call parameter again — e176n's own parameterization; the
per-step arithmetic and the CPU-generator RNG draw sequence are unchanged:
at seed 10902 both cells draw the SAME aj/rj sequences as e176n arm A; only
lr differs). The neutral bank + junction accounting are lab/e170_anchor_
neutral.py VERBATIM via e176n's copy. The measure dial is e176n's measure()
(already minus the 183-span census — e178's recorded deviation). Copied,
not imported, to own the device policy.

NETS: root runs/checkpoints/e131_consolidated_e113.pt (gate bit-exact vs
e151's stored before-cells, e176n's G_ROOT set). The 1e-3/1e-4 stored
trajectories = e176n arm A / arm B / e176 main / e176 smoke traces (embedded,
verified vs files at plot time). New checkpoints: runs/checkpoints/
e180_neutral_lr{3e5,1e5}.pt (+ _s{2,10,50,100,200} intermediates).

Outputs: runs/e180/{metrics.json, wash_rate.png}; checkpoints
runs/checkpoints/e180_*.pt. No NOTES/THINKING/QUEUE/STATE edits; single
commit, no push.

Run:  cd lab && python e180_wash_rate.py    (E180_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

torch.set_num_threads(8)                              # e152R/e143/e184 convention

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT, cooldown,     # noqa: E402
                    gpu_ok, gpu_status, run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E180_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
ROOT_CK = "e131_consolidated_e113.pt"   # THE FULLY-CONSOLIDATED ROOT
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E176N_METRICS = E43.REPO / "runs" / "e176n" / "metrics.json"
E176_METRICS = E43.REPO / "runs" / "e176" / "metrics.json"
E176_SMOKE_METRICS = E43.REPO / "runs" / "e176_smoke" / "metrics.json"

# ---- e152's placement constants (the measurement instruments rebuild these) ----
RETEACH_J = 54                    # offset: name x-cols 184..190, read rows 183..189
SITE_ADDR_ROW = 183               # the onset read row (= e120's SPLICE_ADDR_ROW)
SITE_Z_XCOL = PRE + RETEACH_J     # 184 (= e120's Z_XCOL)
SITE_CONT = BLOCK - PRE - RETEACH_J - len(NAME)     # 65-token continuation
D_ALL = (121, 125, 129, 133, 137)                   # e113's fixed set
GEOS = (-12, 0, 12)               # novel x2 + the trained g0

# ---- census row set (e158/e161/e176 old-band convention; read-visible rows) ----
ROWS_OLD = (0, 1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120) + tuple(range(121, 130))
if SMOKE:
    ROWS_OLD = (0, 1, 60, 121, 125, 129)

# ---- the two gentle-regime trainings -------------------------------------------
CK_MAIN: tuple[int, ...] = (2, 10, 50, 100, 200, 300) if not SMOKE else (2, 4)
LRS: tuple[float, ...] = (3e-5, 1e-5)      # dispatch order: 3e-5 then 1e-5
LR_TAG = {3e-5: "lr3e5", 1e-5: "lr1e5"}

# ---- fine-tune envelope (e176N arm A VERBATIM; only the lr differs) -------------
FREEZE_SEED = 10902               # the locked seed lineage (= e161/e176/e176n's)
TRAIN_CAP_S = 1800.0              # dispatch: <=1800 s caps (per training, any device)
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
COOLDOWN_S = 90.0                 # dispatch: 60-120 s; e152R/e184's 90
MIDRUN_POLL_EVERY = 25            # mid-run GPU contention poll cadence (steps)

# ---- e170's neutral anchor bank (arm A's stream, VERBATIM) ----------------------
E170_ANCHOR_SEED = 170             # dedicated RNG for the plain-corpus bank
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")

# ---- gates / references (full precision, = stored metrics) ----------------------
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05

E151_ROOT = {                     # runs/e151 'before' battery (e176n's gate set)
    "base_gm12": 0.9155886769294739,
    "base_g0": 0.7850371599197388,
    "base_gp12": 0.9478210210800171,
    "ce_r": 1.663516640663147,
    "site_read_onset": 0.8898659348487854,
    "site_read_span": 0.982668936252594,
    "A129": -0.13237020391970877,
    "row0_strength": 0.7316772227270098,
    "dall_g0": 0.9047248959541321,
}

# e176n arm A's stored trajectory (lr 1e-3, NEUTRAL stream, seed 10902; runs/
# e176n/metrics.json trace_armA VERBATIM, re-verified at plot time) — the 1e-3
# neutral point of the curve (t* = 2 on grid {1,2,4,...}).
E176N_ARMA = {
    "label": "e176n armA: neutral, lr 1e-3, seed 10902",
    "freeze_steps": [0, 1, 2, 4, 50, 100, 200, 300],
    "gm12": [0.9155886173248291, 0.6780440807342529,
             0.027077054604887962, 0.010940761305391788,
             0.022122304886579514, 0.014922752045094967,
             0.00204725144430995, 0.0038193254731595516],
    "g0": [0.7850371599197388, 0.4619811177253723,
           0.1147073358297348, 0.06042749062180519,
           0.16835635900497437, 0.05249874293804169,
           0.011475668287767338, 0.016238771378993988],
    "ce_r": [1.663516640663147, 2.2113420963287354,
             2.032074451446533, 1.8213403224945068,
             1.708834171595166, 1.6700899609982666,
             1.6468615531921387, 1.642844796180725],
}

# e176n arm B's stored light trajectory (lr 1e-4, ORIGINAL stream, seed 10902;
# runs/e176n/metrics.json armB_lr1e4.traj VERBATIM) — the 1e-4 point (t* = 50
# on grid {50,100,200,300}: COARSE, an upper bound on the true survival time).
E176N_ARMB = {
    "label": "e176n armB: original (extinction) stream, lr 1e-4, seed 10902",
    "freeze_steps": [0, 50, 100, 200, 300],
    "gm12": [0.9155886173248291, 0.24797283113002777,
             0.2119246870279312, 0.1346260905265808,
             0.06926784664392471],
    "g0": [0.7850371599197388, 0.2724837064743042,
           0.23059670627117157, 0.11020137369627321,
           0.05722249671816826],
    "ce_r": [1.663516640663147, 1.5962976217269897,
             1.5652058124542236, 1.5848811864852905,
             1.5821356773376465],
}

# e176's stored MAIN trajectory (lr 1e-3, original stream; runs/e176/metrics.json
# trace_summary VERBATIM) — the 1e-3 original point at COARSE resolution
# (first-under = +50 on this grid alone).
E176_MAIN = {
    "label": "e176 main: original stream, lr 1e-3 (coarse grid)",
    "freeze_steps": [0, 50, 100, 200, 300],
    "gm12": [0.9155886173248291, 0.020882638171315193,
             0.006246014963835478, 0.0006464376347139478,
             0.0011210207594558597],
    "g0": [0.7850371599197388, 0.027392588555812836,
           0.016695411875844002, 0.001976940780878067,
           0.0012388104805722833],
    "ce_r": [1.663516640663147, 1.6478883028030396,
             1.6605137586593628, 1.6117957830429077,
             1.6334049701690674],
}

# e176's SMOKE fine steps (lr 1e-3, original stream; runs/e176_smoke/metrics.json
# trace_summary VERBATIM) — the ONLY fine-step trajectory that stream has; the
# 1e-3 original point's t* = 2 comes from HERE (e176n docstring's attribution).
E176_SMOKE = {
    "label": "e176 smoke: original stream, lr 1e-3 (the fine-step trace)",
    "freeze_steps": [0, 2, 4],
    "gm12": [0.9155886173248291, 0.08816977590322495,
             0.002640149090439081],
    "g0": [0.7850371599197388, 0.08683783560991287,
           0.0034518486354500055],
    "ce_r": [1.663516640663147, 2.0001156330108643,
             1.7823567390441895],
}

# ---- registered bar constants (frozen; see OPERATIONALIZATIONS above) ----------
SHUT_BAR = 0.27                   # survival bar (e158/e161/e176/e176n/e184)
ROOT_GM12 = E151_ROOT["base_gm12"]
ROOT_G0 = E151_ROOT["base_g0"]

REGISTERED_PREDICTION = {
    "lr_scaled": "LR-SCALED fires if: survival time grows monotonically as lr "
        "shrinks (no under-bar by +300 at 1e-5, or late) — wash is "
        "optimization-rate-limited; the basin-width reading holds; the paper's "
        "'continued training' gets its rate qualifier.",
    "lr_immune": "LR-IMMUNE fires if: under-bar by +50 even at 1e-5 — intrinsic "
        "fragility; the two-step death is lr-invariant in outcome.",
    "texture": "No bar shopping; texture (partial survival at 1e-5) => TEXTURE "
        "with the curve.",
    "operationalizations": "g-12 = absolute install-60 battery mean p(Z) at ctx "
        "offset -12 (e176's convention); survival time t*(lr) = FIRST measured "
        "checkpoint step on the cell's own grid with g-12 <= 0.27 (bracket "
        "co-reported; t* is the fit's point estimate); LR-IMMUNE = the 1e-5 "
        "cell's first-under in {2,10,50}; LR-SCALED's 1e-5 clause = first-under "
        "None or in {100,200,300} (disjoint from LR-IMMUNE by construction); "
        "monotone = WEAK growth of t* across {1e-3->2, 1e-4->50, 3e-5->t*(3e-5), "
        "1e-5->t*(1e-5)} with a never-crossing cell contributing t* > 300 (300 "
        "as a lower bound); adjudication order LR-IMMUNE -> LR-SCALED -> "
        "TEXTURE; five-point curve = {1e-3 neutral (e176n armA, t*=2), 1e-3 "
        "original (e176 smoke fine grid, t*=2), 1e-4 original (e176n armB, "
        "t*=50 COARSE upper bound), 3e-5 neutral (new), 1e-5 neutral (new)}; "
        "neutral-only fit co-reported (the stored half mixes streams); fit "
        "reported not bar-adjudicated; a censored cell fitted twice (t*:=300 "
        "marked CENSORED, and complete-cases-only).",
    "registration": "QUEUE row e180 dispatched ~21:20Z (commit 24de6e1) "
        "registered the bars in the mission text; this docstring freezes them "
        "verbatim before compute. Adjudicate against exactly this; no bar "
        "shopping.",
}

trims: list[str] = []
device_events: list[dict] = []
deviations: list[str] = [
    "GPU ALLOWED by dispatch (idle 0%/69C at dispatch; e184's policy verbatim): "
    "strict pre-training quick check per training (gpu_ok() gates: util <= 85, "
    "temp <= 80 C, mem <= 85% of total; double-poll 5 s apart), PARK-ONCE to "
    "CPU on failure; mid-run contention guard polls every "
    f"{MIDRUN_POLL_EVERY} steps and migrates net + optimizer state to CPU on "
    "mem/temp breach, finishing there (any mixing recorded per cell in "
    "device_events).",
    "GPU float nondeterminism: the stored 1e-3/1e-4 references ran CPU; if the "
    "new cells run GPU (or park mid-run), devices mix ACROSS lr points — "
    "recorded per cell. The bars live at order-of-magnitude step separations "
    "(2 vs 50 vs 100+), which float noise cannot move; any read within 0.05 "
    "of the 0.27 bar is FLAGGED, not smoothed.",
    "The five-point curve's stored half mixes streams (1e-3 original + 1e-4 "
    "original vs the new neutral cells): both 1e-3 cells agree at t*=2, and "
    "the stream-homogeneous neutral-only fit is co-reported — the composition "
    "choice is a texture-level record, not a bar.",
    "The 1e-4 point is COARSE (grid {50,100,200,300}): its t*=50 is an UPPER "
    "bound on the true survival time; the new cells' early grid {2,10} "
    "resolves the gentle regime better than that stored cell.",
    "finetune_freeze carries e184's device parameter + mid-run guard, with the "
    "lr a per-call parameter (e176n's own parameterization); the per-step "
    "arithmetic and the CPU-generator RNG draw sequence are unchanged — at "
    "seed 10902 both new cells draw the SAME aj/rj sequences as e176n arm A; "
    "only lr differs.",
    "Light in-run evals at every checkpoint; ONE full dial at +300 per cell "
    "(e176n arm B's rider convention); the wash-rate question is answered by "
    "the g-12 trajectory, the full dial is the anatomy co-report.",
    "Nets are the mandated 2.7M e131_consolidated line (the dispatch's '<=1M "
    "family' note is an envelope statement; e143/e151/e152/e158/e176n/e184 "
    "precedent — every gate reference lives on this line).",
    "Eval thread count is 8 (e152R/e143/e184 convention) vs e151's stored "
    "cells — CPU reduction order can drift low-order bits; the G_ROOT gate "
    "reports both the 5e-6 bit flag and the 0.05 fallback tolerance.",
    "Single seed (10902), one trajectory per lr cell, one root lineage, n=1 "
    "per cell — point estimates until replicated (e184's seed lottery lives "
    "in the tail, not the clock, but the gentle regime is unreplicated).",
    "Smoke mode trims: 4-step trainings, checkpoints {2,4}, lean measures, no "
    "cooldowns, no adjudication; nothing adjudicated.",
    "Record refresh after the main run (provenance honesty): verify_ref's "
    "auto-routing compared the e176n file's trace_armA layout against arm B's "
    "embedded copy, marking the armB reference verified=False — an artifact "
    "(the embedded arm-B numbers match the file's armB_lr1e4.traj exactly); "
    "fixed by the explicit layout='armB' route, and loglog_fit's raw slope "
    "key gained the unambiguous alpha_exponent + slope_convention fields. "
    "The armB provenance record and the three fit records in runs/e180/"
    "metrics.json were refreshed with these identical patched code paths — "
    "no measured value, trajectory, checkpoint, or adjudication was touched.",
]


# ------------------------------------------------------------------ device pick
# PROVENANCE: lab/e184_seed_replicates.py VERBATIM (= e152r's pick_dev adapted
# to the lab's gpu_ok(); plus the dispatch's mid-run migration guard).

GPU_PARKED = False
PARK_REASON = None


def pick_dev(tag: str) -> torch.device:
    """Strict pre-training quick check (dispatch): gpu_ok() double-poll 5 s
    apart; PARK-ONCE — any failure parks every remaining training to CPU."""
    global GPU_PARKED, PARK_REASON
    if GPU_PARKED:
        log(f"[gpu] '{tag}' CPU (PARKED: {PARK_REASON})")
        return CPU
    if not torch.cuda.is_available():
        GPU_PARKED, PARK_REASON = True, "no CUDA"
        return CPU
    s1 = gpu_status()
    if gpu_ok():
        time.sleep(5)
        if gpu_ok():
            s2 = gpu_status()
            log(f"[gpu] '{tag}' may use GPU (util {s2['util']:.0f}% temp "
                f"{s2['temp']:.0f}C mem {s2['mem_used']:.0f}/"
                f"{s2['mem_total']:.0f}MB)")
            return torch.device("cuda")
    GPU_PARKED = True
    PARK_REASON = f"quick check failed: {s1}"
    log(f"[gpu] PARK — '{tag}' and all remaining trainings run CPU ({s1})")
    return CPU


def migrate_to_cpu(net, opt) -> None:
    """Move net + optimizer state to CPU in place (params persist, so the
    opt.state keys stay valid). The dispatch's mid-run contention exit."""
    net.to("cpu")
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state.get(p, {})
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to("cpu")


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e176n_neutral_wash.py VERBATIM (see the module docstring).
# Copied rather than imported to own the device policy.

def load_cpu(path) -> TinyGPT:
    m = TinyGPT(Cfg())
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


def evl_load(sd: dict) -> TinyGPT:
    m = TinyGPT(Cfg())
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def battery_cell(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> dict:
    """e068/e113/e120/e151 battery on CPU: p(Z) at the last position."""
    net.eval()
    pzs, amax = [], 0.0
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
        amax += float((lg[:, -1].argmax(-1) == zid).float().sum())
    p = torch.cat(pzs)
    return {"mean_pz": float(p.mean()), "median_pz": float(p.median()),
            "std_pz": float(p.std()),
            "frac_pz_ge_0.5": float((p >= 0.5).float().mean()),
            "frac_argmax_z": amax / ids.shape[0]}


@torch.no_grad()
def battery_pz(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> float:
    """e116's scalar battery (census readout)."""
    net.eval()
    pzs = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs += [float(pr[k, zid]) for k in range(pr.shape[0])]
    return float(np.mean(pzs))


@torch.no_grad()
def ce_fixed_cpu(net: TinyGPT, x, y, bs=64) -> float:
    net.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        _, loss = net(x[i:i + bs], y[i:i + bs])
        tot += float(loss.item()) * len(x[i:i + bs])
        n += len(x[i:i + bs])
    return tot / max(n, 1)


def val_windows(val_ids, val_text, n, seed, block=256):
    """e065 val_windows verbatim: name-free val-split windows."""
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


def deleted_wpe(sd: dict, rows: tuple[int, ...]) -> tuple[dict, dict]:
    """D2 subtractive row-zero with the e065/e113 confinement gate."""
    out = {k: v.clone() for k, v in sd.items()}
    for r in rows:
        out["wpe.weight"][r] = 0.0
    d = out["wpe.weight"] != sd["wpe.weight"]
    changed_rows = sorted(set(int(r) for r in torch.nonzero(d)[:, 0].tolist()))
    n = int(d.sum().item())
    others = all(torch.equal(out[k], sd[k]) for k in out if k != "wpe.weight")
    gate = {"rows": list(rows), "n_elements_changed": n,
            "expected": len(rows) * sd["wpe.weight"].shape[1],
            "changed_rows": changed_rows,
            "confined": bool(changed_rows == sorted(rows)),
            "others_bit_identical": bool(others),
            "pass": bool(n == len(rows) * sd["wpe.weight"].shape[1]
                         and changed_rows == sorted(rows) and others)}
    return out, gate


@torch.no_grad()
def read_fact_at(net: TinyGPT, pool_x: torch.Tensor, name_ids, zid: int,
                 addr_row: int, xcol: int, bs=30) -> dict:
    """e131's read_fact_position VERBATIM ARITHMETIC (e151/e152/e161/e176/
    e176n/e178/e184 copy): p(true name char) at positions addr_row..addr_row+6."""
    net.eval()
    n_name = len(name_ids)
    per_pos = [[] for _ in range(n_name)]
    onset = []
    for i in range(0, pool_x.shape[0], bs):
        w = pool_x[i:i + bs]
        lg, _ = net(w)
        pr = F.softmax(lg, -1)
        for k in range(w.shape[0]):
            onset.append(float(pr[k, addr_row, int(zid)]))
            for j in range(n_name):
                per_pos[j].append(
                    float(pr[k, addr_row + j, int(w[k, xcol + j])]))
    onset_t = torch.tensor(onset)
    allp_t = torch.tensor([p for pos in per_pos for p in pos])
    return {"pz_onset_mean": float(onset_t.mean()),
            "pz_onset_median": float(onset_t.median()),
            "pz_onset_frac_ge_0.5": float((onset_t >= 0.5).float().mean()),
            "pname_mean_over7": float(allp_t.mean()),
            "pname_frac_ge_0.5": float((allp_t >= 0.5).float().mean()),
            "per_position_mean": [float(np.mean(pos)) for pos in per_pos]}


def row_census(net: TinyGPT, rows, readout, *rargs) -> dict:
    """e139's row_census_at183 VERBATIM (mean-arm / zero-arm / restore)."""
    net.eval()
    base = readout(net, *rargs)
    w = net.wpe.weight.data
    orig = w.clone()
    mean_row = orig.mean(0)
    m_d, z_d = {}, {}
    for r in rows:
        w.copy_(orig); w[r] = mean_row
        m_d[r] = base - readout(net, *rargs)
        w.copy_(orig); w[r] = 0.0
        z_d[r] = base - readout(net, *rargs)
    w.copy_(orig)
    rows_d = {str(r): {"mean": float(m_d[r]), "zero": float(z_d[r]),
                       "ratio": float(min(m_d[r], z_d[r]) /
                                      max(m_d[r], z_d[r]))
                              if max(m_d[r], z_d[r]) > 0 else 0.0,
                       "strength": float(min(m_d[r], z_d[r])),
                       "content": bool(m_d[r] > 0 and z_d[r] > 0 and
                                       min(m_d[r], z_d[r]) /
                                       max(m_d[r], z_d[r]) >= 0.5)}
              for r in rows}
    assert torch.equal(w, orig), "census failed to restore wpe"
    return {"base_readout": base, "rows": rows_d}


# ------------------------------------------------------------------ fine-tune

def finetune_freeze(tag: str, net0: TinyGPT, anchor: torch.Tensor,
                    train_ids: torch.Tensor, itos, r_eval_xy, gm12_ids,
                    g0_ids, zid: int, seed: int, lr: float,
                    ckpt_steps: tuple[int, ...]):
    """THE NEUTRAL PLAIN-CORPUS FREEZE (e176n's finetune_freeze VERBATIM
    arithmetic, device-parameterized as in e184; lr per call). Per step:
    aj = randint(16) anchor draws, rj = randint(16) random corpus offsets;
    batch 32 full-token CE; AdamW (0.9,0.95) wd 0.1 CONSTANT lr, clip 1.0.
    The draw sequence at seed 10902 is device-independent (CPU
    torch.Generator) and IDENTICAL to e176n arm A's — only lr differs.
    Snapshots (deep-copy out, CPU) + light CPU evals (g-12, g0, CE_R — no
    RNG consumed) at the checkpoint steps; the in-batch corpus CE is recorded
    at every checkpoint and every 50. Mid-run GPU contention guard every
    MIDRUN_POLL_EVERY steps: mem/temp breach -> migrate to CPU and finish
    there (recorded in device_events)."""
    dev = pick_dev(tag)
    cap = TRAIN_CAP_S
    ckpt_set = set(ckpt_steps)
    n_steps = ckpt_steps[-1]
    net = copy.deepcopy(net0).to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    n_anc = anchor.shape[0]
    traj, sds, t_start = [], {}, time.time()
    step = 0
    zeph_checks = 0
    evl = copy.deepcopy(net0)          # CPU eval twin
    for step in range(1, n_steps + 1):
        aj = torch.randint(n_anc, (ANCH_BS,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                           generator=gen)
        anc = anchor[aj]
        rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
        # name-free VERIFY (no-op by corpus construction; hard-fail if not)
        for w in rnd:
            txt = "".join(itos[int(c)] for c in w[:64]) + \
                  "".join(itos[int(c)] for c in w[192:])
            if "ZEPH" in txt:
                zeph_checks += 1
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0).to(dev)
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0).to(dev)
        logits, _ = net(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step in ckpt_set or step % 50 == 0:
            log(f"  [{tag}] s{step:4d} corpus CE {float(loss.item()):.4f} "
                f"({time.time() - t_start:.0f}s)")
        if step in ckpt_set:
            sd_cpu = {k: v.detach().cpu().clone()
                      for k, v in net.state_dict().items()}
            sds[step] = sd_cpu
            evl.load_state_dict(sd_cpu)
            evl.eval()
            gz = battery_cell(evl, gm12_ids, zid)
            gz0 = battery_cell(evl, g0_ids, zid)
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            traj.append({"step": step, "g_m12_mean_pz": gz["mean_pz"],
                         "g0_mean_pz": gz0["mean_pz"],
                         "frac_argmax_z": gz["frac_argmax_z"],
                         "corpus_ce": float(loss.item()), "ce_r": ce_r,
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] CKPT +{step:4d} g-12 {gz['mean_pz']:.4f} "
                f"g0 {gz0['mean_pz']:.4f} CE_R {ce_r:.4f} "
                f"(in-batch CE {float(loss.item()):.4f})")
        if (time.time() - t_start) > cap:
            log(f"  [{tag}] time cap {cap:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step}")
            break
        if dev.type == "cuda" and step % MIDRUN_POLL_EVERY == 0:
            s = gpu_status()
            if s["mem_total"] > 0 and (s["mem_used"] > 0.85 * s["mem_total"]
                                       or s["temp"] > 80):
                device_events.append(
                    {"tag": tag, "step": step, "event": "MID-RUN MIGRATION",
                     "status": s,
                     "note": "contention guard fired (mem > 85% or temp > 80C)"
                             " — net + optimizer state moved to CPU; training"
                             " finishes on CPU (e184/e152 precedent)"})
                log(f"  [{tag}] MID-RUN GPU contention at s{step} ({s}) -> "
                    f"migrating to CPU")
                migrate_to_cpu(net, opt)
                dev = CPU
    net.eval()
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"sds": sds, "traj": traj, "steps_ran": step, "seed": seed,
            "lr": lr, "zeph_violations": zeph_checks,
            "initial_device": "cuda" if not GPU_PARKED else "cpu",
            "final_device": str(dev), "time_cap_s": cap}


CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "e180", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name} "
        f"({', '.join(f'{k}={v}' for k, v in meta.items())})")


def verify_ref(embedded: dict, path: Path, src_name: str,
               layout: str = "auto") -> dict:
    """Verify an embedded reference copy against its stored metrics file when
    present (no silent divergence; e176n's verify_ref convention). `layout`
    routes multi-arm files: 'auto' (single-trace files), 'armA', 'armB'."""
    src = {"source": f"embedded verbatim copy ({src_name})",
           "file_present": path.exists(), "verified_vs_embedded": None,
           "max_abs_diff": None}
    if path.exists():
        mm = json.loads(path.read_text(encoding="utf-8"))
        emb = embedded
        if layout == "armB" or (layout == "auto" and "armB_lr1e4" in mm
                                and "trace_armA" not in mm):
            # the file stores ONLY the light traj (steps > 0): compare those
            # rows; the embedded copy's step-0 row is the shared measured
            # root (exempt — it is not part of this file's record).
            tr = mm["armB_lr1e4"]["traj"]
            ts = {"freeze_steps": [t["step"] for t in tr],
                  "gm12": [t["g_m12_mean_pz"] for t in tr],
                  "g0": [t["g0_mean_pz"] for t in tr],
                  "ce_r": [t["ce_r"] for t in tr]}
            emb = {k: embedded[k][1:] for k in embedded}
            ref_kind = "armB_lr1e4.traj (steps>0)"
        elif "trace_armA" in mm:                  # runs/e176n layout
            rows = mm["trace_armA"]
            ts = {"freeze_steps": [r["freeze_steps"] for r in rows],
                  "gm12": [r["gm12"] for r in rows],
                  "g0": [r["g0"] for r in rows],
                  "ce_r": [r["ce_r"] for r in rows]}
            ref_kind = "trace_armA"
        elif "trace_summary" in mm:               # runs/e176 / e176_smoke layout
            ts = {k: mm["trace_summary"][k] for k in
                  ("freeze_steps", "base_gm12", "base_g0", "ce_r")}
            ts = {"freeze_steps": ts["freeze_steps"], "gm12": ts["base_gm12"],
                  "g0": ts["base_g0"], "ce_r": ts["ce_r"]}
            ref_kind = "trace_summary"
        else:
            src["verified_vs_embedded"] = False
            src["note"] = "unrecognized metrics layout"
            return src
        diffs = [abs(a - b) for k in ("gm12", "g0", "ce_r")
                 for a, b in zip(ts[k], emb[k])]
        steps_ok = list(ts["freeze_steps"]) == emb["freeze_steps"]
        src["max_abs_diff"] = max(diffs) if diffs else None
        src["verified_vs_embedded"] = bool(steps_ok and max(diffs) < 1e-9)
        if src["verified_vs_embedded"]:
            src["source"] = (f"{src_name} ({ref_kind}; embedded copy "
                             f"verified, max|diff| {max(diffs):.1e})")
    return src


# ------------------------------------------------------------------ fits

def loglog_fit(points: list[dict]) -> dict:
    """OLS of log10(t*) on log10(lr) over the given points. A point may carry
    't_fit' (substituted value for a censored cell)."""
    xs = np.log10([p["lr"] for p in points])
    ys = np.log10([p.get("t_fit", p["t_star"]) for p in points])
    if len(points) < 2:
        return {"n": len(points), "note": "too few points"}
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    ss_res = float(((ys - pred) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"n": len(points), "slope_alpha": float(slope),
            "slope_convention": ("slope of log10(t*) on log10(lr); the "
                                 "exponent of the power law is its ABSOLUTE "
                                 "value (t* grows as lr shrinks)"),
            "alpha_exponent": float(abs(slope)),
            "intercept_log10": float(intercept),
            "log_lr_range": [float(xs.min()), float(xs.max())],
            "r2": float(r2),
            "points": [{"lr": p["lr"],
                        "t_used": p.get("t_fit", p["t_star"]),
                        "censored": bool(p.get("censored", False))}
                       for p in points],
            "form": (f"t* ~ {10 ** float(intercept):.3g} * lr^-"
                     f"{abs(float(slope)):.2f}")}


# ------------------------------------------------------------------ plot

def make_plot(rd, five, fit_five, any_censored, series, new_cells,
              root_gm12, verdict, censored_lower):
    """THE PLOT (deliverable: the survival-vs-lr curve, log-log; the
    trajectories co-panel). Both axes carry log10-transformed values with
    ticks relabeled in native units (lr on x, survival steps on y)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8))

    for steps, gs, label, colr, mrk, alph in series:
        ax1.plot(steps, gs, marker=mrk, color=colr, lw=1.6, ms=4,
                 alpha=alph, label=label)
    for tag, colr, mrk in new_cells:
        steps = [0] + [t["step"] for t in tag[1]]
        gs = [root_gm12] + [t["g_m12_mean_pz"] for t in tag[1]]
        ax1.plot(steps, gs, marker=mrk, color=colr, lw=1.8, ms=5,
                 label=tag[0])
    ax1.axhline(SHUT_BAR, color="k", ls="--", lw=1.0)
    ax1.text(1.3, SHUT_BAR * 1.15, f"bar {SHUT_BAR}", fontsize=8, color="k")
    ax1.axhline(root_gm12, color="gray", ls=":", lw=1.0)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("freeze steps (neutral plain-corpus stream)")
    ax1.set_ylabel("g-12  (install-60 battery mean p(Z))")
    ax1.set_title("E180 — the wash trajectories across the lr axis "
                  "(seed 10902)")
    ax1.legend(fontsize=7.5, loc="lower left")
    ax1.grid(alpha=0.25, which="both")

    xs = np.log10([p["lr"] for p in five])

    def val(p):
        return p["t_fit"] if p["censored"] else p["t_star"]
    ys = np.log10([val(p) for p in five])
    ax2.scatter(xs, ys, s=55, zorder=3,
                c=["tab:red", "tab:orange", "tab:purple", "tab:green",
                   "tab:blue"])
    for p, x, y in zip(five, xs, ys):
        lbl = (f"{p['lr']:g}: t*={p['t_star'] if p['t_star'] is not None else '>300'}"
               + (" (censored)" if p["censored"] else "")
               + (" (coarse)" if p.get("coarse") else ""))
        if x > -3.2:                     # the 1e-3 side: keep inside the axes
            ax2.annotate(lbl, (x, y), textcoords="offset points",
                         xytext=(-4, 9), ha="right", fontsize=7.5)
        elif p["censored"]:              # clear of the censoring arrow
            ax2.annotate(lbl, (x, y), textcoords="offset points",
                         xytext=(10, 3), fontsize=7.5)
        else:
            ax2.annotate(lbl, (x, y), textcoords="offset points",
                         xytext=(6, 6 if p["lr"] != 1e-4 else -14),
                         fontsize=7.5)
    fx = np.linspace(xs.min() - 0.05, xs.max() + 0.05, 50)
    if fit_five.get("slope_alpha") is not None:
        ax2.plot(fx, fit_five["slope_alpha"] * fx + fit_five["intercept_log10"],
                 "k-", lw=1.4,
                 label=(f"OLS fit: t* ~ {10 ** fit_five['intercept_log10']:.3g}"
                        f"·lr^-{fit_five['alpha_exponent']:.2f} "
                        f"(R²={fit_five['r2']:.3f})"))
    ax2.plot(fx, (fx - np.log10(1e-3)) * -1.0 + np.log10(2), "k:", lw=1.2,
             label="α=1 reference (t* ∝ 1/lr; displacement-limited)")
    if any_censored:
        ax2.annotate("", xy=(xs[-1], np.log10(censored_lower) + 0.28),
                     xytext=(xs[-1], np.log10(censored_lower) + 0.04),
                     arrowprops=dict(arrowstyle="->", color="tab:blue",
                                     lw=1.6))
    ax2.set_xticks(np.log10([1e-5, 3e-5, 1e-4, 3e-4, 1e-3]))
    ax2.set_xticklabels(["1e-5", "3e-5", "1e-4", "3e-4", "1e-3"])
    yticks = [2, 10, 50, 100, 300]
    ax2.set_yticks(np.log10(yticks))
    ax2.set_yticklabels([str(t) for t in yticks])
    ax2.set_xlabel("constant lr (AdamW, wd 0.1, clip 1.0)")
    ax2.set_ylabel("survival time t*  (first checkpoint g-12 <= 0.27)")
    ax2.set_title("THE WASH-RATE LAW — survival vs lr (log-log; "
                  f"verdict: {verdict})")
    ax2.legend(fontsize=7.5, loc="upper right")
    ax2.grid(alpha=0.25, which="both")

    fig.suptitle("E180 — the wash-rate law: the neutral-arm protocol at lr "
                 "3e-5/1e-5 (seed 10902; the lr axis completed)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = rd / "wash_rate.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return png


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e180_smoke" if SMOKE else "e180")
    log(f"E180 THE WASH-RATE LAW (smoke={SMOKE}) -> {rd}")
    log(f"compute: GPU allowed (park-once policy), cooldown {COOLDOWN_S:.0f}s "
        f"around each training, per-training cap {TRAIN_CAP_S:.0f}s")

    # ---------------- protocol rebuild (e143/e151/e152/e161/e176/e176n verbatim)
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
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + POST_CAP <= len(train_ids):
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
    name_ids = corpus.encode(NAME)

    # ---------------- measurement pool: e152's locked j=54 windows (instrument)
    wins = []
    for p, h in install_occ:
        pre = train_ids[p - PRE - RETEACH_J: p]
        post = train_ids[p + len(h): p + len(h) + SITE_CONT]
        if len(pre) != PRE + RETEACH_J or len(post) != SITE_CONT:
            raise RuntimeError(f"pool window short at p={p}")
        w = torch.cat([pre, name_ids, post])
        if len(w) != BLOCK:
            raise RuntimeError(f"pool window len {len(w)} != {BLOCK}")
        wins.append(w)
    pool_x = torch.stack(wins)
    G_POOL = {"shape": list(pool_x.shape),
              "name_xcols": [SITE_Z_XCOL, SITE_Z_XCOL + len(NAME) - 1],
              "all_windows_name_in_place": bool(
                  all(torch.equal(w[SITE_Z_XCOL: SITE_Z_XCOL + len(NAME)],
                                  name_ids) for w in pool_x)),
              "note": "MEASUREMENT instrument only (e152's locked j=54 pool); "
                      "NO window from this pool enters any training"}
    G_POOL["pass"] = G_POOL["all_windows_name_in_place"]
    assert G_POOL["pass"], f"pool gate FAILED: {G_POOL}"

    # ---------------- e170's NEUTRAL anchor bank (arm A's stream VERBATIM)
    arng = random.Random(E170_ANCHOR_SEED)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        txt = train_text[s: s + BLOCK + 1]
        if any(f in txt for f in ANCHOR_FORBIDDEN):
            rejections += 1
            continue
        n_starts.append(s)
    if len(n_starts) != 16:
        raise RuntimeError(f"neutral bank incomplete: {len(n_starts)}/16")
    anchor_neutral = torch.stack([train_ids[s: s + BLOCK] for s in n_starts])

    host_positions = [p for p in E43.find_occ(train_text, HOSTS[0])]
    host_positions += [p for p in E43.find_occ(train_text, HOSTS[1])]

    def junctions_covered(starts):
        cov = 0
        for s in starts:
            if any(s <= p < s + BLOCK + 1 for p in host_positions):
                cov += 1
        return cov

    jc_neutral = junctions_covered(n_starts)
    host_occ_total = len(host_positions)
    bg_rate = host_occ_total * (BLOCK + 1) / len(train_ids)
    G_ANCHOR = {
        "neutral_bank": {
            "construction": ("16 plain corpus windows from train_ids, RNG seed "
                             f"{E170_ANCHOR_SEED}, rejection if [s, s+257) "
                             "contains FLORIZEL/ELIZABETH/ZEPH/MIRABEL — "
                             "e170's construction VERBATIM (= e176n arm A / "
                             "e184's bank)"),
            "n_windows": 16, "block": BLOCK, "seed": E170_ANCHOR_SEED,
            "starts": n_starts, "tries": tries, "rejections": rejections,
            "forbidden": list(ANCHOR_FORBIDDEN),
            "windows_with_host_content": sum(
                1 for s in n_starts
                if any(f in train_text[s: s + BLOCK + 1] for f in HOSTS)),
            "junctions_covered": jc_neutral,
        },
        "budget_identical_to_e176n_armA": bool(
            anchor_neutral.shape == (16, BLOCK)),
        "rng_stream_identical_to_e176n_armA": True,
        "rng_note": ("finetune_freeze verbatim; draw shapes/moduli identical "
                     "(n_anc=16, len(train_ids)); seed 10902 — the same "
                     "aj/rj sequences as e176n arm A; ONLY the lr differs"),
        "random_channel_background": {
            "host_occurrences_in_train": host_occ_total,
            "est_window_hit_rate": bg_rate,
            "note": ("the 16-per-batch random corpus windows are e161/e176 "
                     "VERBATIM and unfiltered — identical background, not "
                     "part of the delta"),
        },
    }
    G_ANCHOR["pass"] = bool(
        G_ANCHOR["neutral_bank"]["windows_with_host_content"] == 0
        and G_ANCHOR["neutral_bank"]["junctions_covered"] == 0
        and G_ANCHOR["budget_identical_to_e176n_armA"]
        and anchor_neutral.shape == (16, BLOCK))
    assert G_ANCHOR["pass"], f"anchor gate FAILED: {G_ANCHOR}"
    log(f"G_ANCHOR: neutral bank 16x{BLOCK} plain-corpus windows (seed "
        f"{E170_ANCHOR_SEED}, {rejections} rejections/{tries} tries) — host "
        f"content 0/16, junctions 0/16; random-channel background "
        f"~{100 * bg_rate:.1f}%/window: PASS")

    # ---------------- batteries: ctx = train_text[p-PRE-j : p] (e119 verbatim)
    bat_ids, held_ids = {}, {}
    for j in GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
        hs = [train_text[p - PRE - j: p] for p, _ in held_occ]
        held_ids[j] = torch.stack([corpus.encode(c) for c in hs])
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    gm12_ids = bat_ids[-12]
    g0_ids = bat_ids[0]

    # ---------------- root net + gate vs e151
    net0 = load_cpu(CKPT_DIR / ROOT_CK)
    sd_root = {k: v.clone() for k, v in net0.state_dict().items()}
    root_meta = None
    st_raw = torch.load(CKPT_DIR / ROOT_CK, map_location="cpu",
                        weights_only=False)
    if isinstance(st_raw, dict) and "meta" in st_raw:
        root_meta = E43.jsonable(st_raw["meta"])
    log(f"root: {ROOT_CK} (meta: {root_meta})")

    gates_surg: dict = {}

    def measure(sd: dict, tag: str, lean: bool = False) -> dict:
        """e176n's measure() VERBATIM (already minus the 183-span census —
        e178's recorded deviation): base 3-geos + held30 + CE_R + site read +
        old-band census (row-0 sink / A129 brake) + deletion table; lean=True
        keeps only a quick A(129)."""
        net = evl_load(sd)
        out: dict = {"tag": tag}
        out["base"] = {j: battery_cell(net, bat_ids[j], zid) for j in GEOS}
        out["base_held"] = {j: battery_cell(net, held_ids[j], zid)
                            for j in GEOS}
        out["ce_r"] = ce_fixed_cpu(net, *r_eval_xy)
        log(f"[{tag}] base: " + " ".join(f"g{j:+d} {out['base'][j]['mean_pz']:.4f}"
                                         for j in GEOS)
            + " | held30: " + " ".join(
                f"g{j:+d} {out['base_held'][j]['mean_pz']:.4f}" for j in GEOS)
            + f" | CE_R {out['ce_r']:.4f}")
        out["site_read"] = read_fact_at(net, pool_x, name_ids, zid,
                                        SITE_ADDR_ROW, SITE_Z_XCOL)
        log(f"[{tag}] site read @183: onset "
            f"{out['site_read']['pz_onset_mean']:.4f} span "
            f"{out['site_read']['pname_mean_over7']:.4f}")
        if not lean and not SMOKE:
            out["census_old"] = row_census(net, ROWS_OLD,
                                           lambda n: battery_pz(n, bat_ids[0],
                                                                zid))
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
            DELS = {"d_all": D_ALL, "d183": (SITE_ADDR_ROW,)}
            out["del_table"] = {}
            for dl, rows_ in DELS.items():
                sd_d, gate = deleted_wpe(sd, rows_)
                gates_surg[f"{tag}__{dl}"] = gate
                if not gate["pass"]:
                    raise RuntimeError(f"deletion gate FAILED {tag}/{dl}: "
                                       f"{gate}")
                net.load_state_dict(sd_d)
                cell = {"g0": battery_cell(net, bat_ids[0], zid)["mean_pz"]}
                if dl == "d183":
                    cell["gm12"] = battery_cell(net, bat_ids[-12],
                                                zid)["mean_pz"]
                out["del_table"][dl] = cell
            net.load_state_dict(sd)
            log(f"[{tag}] deletions g0: " + " | ".join(
                f"{dl} {out['del_table'][dl]['g0']:.3f}" for dl in DELS))
        else:
            w = net.wpe.weight.data
            orig = w.clone()
            mean_row = orig.mean(0)
            bp = battery_pz(net, bat_ids[0], zid)
            w[129] = mean_row
            m129 = bp - battery_pz(net, bat_ids[0], zid)
            w.copy_(orig)
            w[129] = 0.0
            z129 = bp - battery_pz(net, bat_ids[0], zid)
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

    def gate_vs(cells: dict, refs: dict, name: str) -> dict:
        keys = [k for k in refs if k in cells]
        missing = [k for k in refs if k not in cells]
        diffs = {k: cells[k] - refs[k] for k in keys}
        max_abs = max(abs(v) for v in diffs.values())
        g = {"cells": {k: cells[k] for k in keys}, "refs": refs,
             "skipped_missing": missing, "diffs": diffs,
             "max_abs_diff": max_abs, "bit_tol": G_BIT_TOL,
             "tol": G_FALLBACK_TOL, "bit": bool(max_abs < G_BIT_TOL),
             "pass": bool(max_abs < G_FALLBACK_TOL)}
        log(f"GATE {name}: max|diff| {max_abs:.2e} (tol {G_FALLBACK_TOL}): "
            + ("PASS" if g["pass"] else "FAIL")
            + (" (bit)" if g["bit"] else ""))
        return g

    log("=" * 78)
    log("STEP-0 battery (root = e131_consolidated_e113; 'before')")
    root = measure(sd_root, "root", lean=SMOKE)
    root_cells = flat_cells(root)
    keymap = {"base_gm12": "gm12", "base_g0": "g0", "base_gp12": "gp12",
              "ce_r": "ce_r", "site_read_onset": "site_read_onset",
              "site_read_span": "site_read_span", "A129": "A129",
              "row0_strength": "row0_strength", "dall_g0": "dall_g0"}
    root_refs = {keymap[k]: v for k, v in E151_ROOT.items()}
    G_ROOT = gate_vs(root_cells, root_refs, "G_ROOT (vs e151 before-cells)")
    if not G_ROOT["pass"]:
        raise RuntimeError("consolidated-root gate FAILED vs e151 stored "
                           "before-cells")
    log("gates: G_SPLICE, G_NAMEFREE, G_POOL, G_ANCHOR, G_ROOT all PASS")

    # reference provenance (embedded copies verified vs the stored files)
    src_arma = verify_ref(E176N_ARMA, E176N_METRICS,
                          "runs/e176n/metrics.json trace_armA")
    src_armb = verify_ref(E176N_ARMB, E176N_METRICS,
                          "runs/e176n/metrics.json armB_lr1e4", layout="armB")
    src_e176 = verify_ref(E176_MAIN, E176_METRICS,
                          "runs/e176/metrics.json trace_summary")
    src_e176s = verify_ref(E176_SMOKE, E176_SMOKE_METRICS,
                           "runs/e176_smoke/metrics.json trace_summary")

    # =====================================================================
    # THE TWO GENTLE-REGIME CELLS (neutral protocol; only lr differs)
    # =====================================================================
    cells: dict = {}
    near_bar_flags: list[dict] = []
    for i, lr in enumerate(LRS):
        tag = LR_TAG[lr]
        log("=" * 78)
        if not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before {tag}")
            cooldown(COOLDOWN_S)
        log(f"CELL {tag} — THE NEUTRAL PROTOCOL at lr {lr}: {CK_MAIN[-1]} "
            f"steps, e170's neutral anchors (batch {ANCH_BS} neutral + "
            f"{RAND_BS} random, full-token CE), seed {FREEZE_SEED} (= e176n "
            f"arm A's draw sequence), checkpoints +{list(CK_MAIN)}")
        arm = finetune_freeze(tag, net0, anchor_neutral, train_ids, itos,
                              r_eval_xy, gm12_ids, g0_ids, zid, FREEZE_SEED,
                              lr, CK_MAIN)
        if not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s after {tag}")
            cooldown(COOLDOWN_S)
        g_drawfree = {"zeph_violations": arm["zeph_violations"],
                      "pass": bool(arm["zeph_violations"] == 0)}
        assert g_drawfree["pass"], f"{tag}: name token leaked into a window"

        smax = max(arm["sds"])
        save_ckpt(f"e180_neutral_{tag}", arm["sds"][smax],
                  {"desc": f"e131_consolidated_e113 + {smax}-step NEUTRAL-"
                           f"anchor plain-corpus freeze at lr {lr} (the "
                           f"wash-rate law), seed {FREEZE_SEED}",
                   "steps": int(smax), "seed": FREEZE_SEED, "lr": lr,
                   "anchor_bank": f"neutral (seed {E170_ANCHOR_SEED})",
                   "base": f"runs/checkpoints/{ROOT_CK}"})
        for s in sorted(arm["sds"]):
            if s == smax:
                continue
            save_ckpt(f"e180_neutral_{tag}_s{s}", arm["sds"][s],
                      {"desc": f"e131_consolidated_e113 + {s}-step NEUTRAL-"
                               f"anchor freeze at lr {lr} (intermediate), "
                               f"seed {FREEZE_SEED}",
                       "steps": int(s), "seed": FREEZE_SEED, "lr": lr,
                       "anchor_bank": f"neutral (seed {E170_ANCHOR_SEED})",
                       "base": f"runs/checkpoints/{ROOT_CK}"})

        # full dial at +300 (the anatomy co-report; lean at the intermediates
        # is already in arm['traj'])
        batteries: dict = {}
        if not SMOKE:
            log(f"CELL {tag} +{smax} full dial")
            batteries[str(smax)] = measure(arm["sds"][smax], f"{tag}_full",
                                           lean=False)

        # near-bar flag (device-drift honesty, e184's convention)
        for t in arm["traj"]:
            if abs(t["g_m12_mean_pz"] - SHUT_BAR) < 0.05:
                near_bar_flags.append(
                    {"cell": tag, "step": t["step"],
                     "gm12": t["g_m12_mean_pz"],
                     "note": "read within 0.05 of the 0.27 bar — timing at "
                             "this checkpoint is device/float sensitive"})

        trace = [{"freeze_steps": 0,
                  "gm12": root_cells["gm12"], "g0": root_cells["g0"],
                  "ce_r": root_cells["ce_r"],
                  "retention_vs_root_gm12": root_cells["gm12"] / ROOT_GM12}]
        for t in arm["traj"]:
            trace.append({"freeze_steps": t["step"],
                          "gm12": t["g_m12_mean_pz"], "g0": t["g0_mean_pz"],
                          "frac_argmax_z": t["frac_argmax_z"],
                          "corpus_ce": t["corpus_ce"], "ce_r": t["ce_r"],
                          "retention_vs_root_gm12":
                              t["g_m12_mean_pz"] / ROOT_GM12,
                          "elapsed_s": t["elapsed_s"]})
        if str(smax) in batteries:
            trace[-1] = {"freeze_steps": smax,
                         **flat_cells(batteries[str(smax)]),
                         "corpus_ce": arm["traj"][-1]["corpus_ce"],
                         "retention_vs_root_gm12":
                             flat_cells(batteries[str(smax)])["gm12"]
                             / ROOT_GM12}
        cells[tag] = {
            "desc": f"e176n arm A's NEUTRAL protocol VERBATIM at lr {lr} "
                    f"(only the lr differs; seed {FREEZE_SEED} = arm A's "
                    f"draw sequence)",
            "ckpt_steps": list(CK_MAIN), "steps_ran": arm["steps_ran"],
            "seed": arm["seed"], "lr": arm["lr"],
            "traj": arm["traj"], "trace": trace,
            "zeph_violations": arm["zeph_violations"],
            "missing_checkpoints": sorted(set(CK_MAIN) - set(arm["sds"])),
            "initial_device": arm["initial_device"],
            "final_device": arm["final_device"],
            "time_cap_s": arm["time_cap_s"],
            "batteries_full": (batteries[str(smax)]
                               if str(smax) in batteries else None),
            "cells_full": (flat_cells(batteries[str(smax)])
                           if str(smax) in batteries else None),
        }
        del arm

    # =====================================================================
    # THE SURVIVAL CURVE + FITS (reported; no bars live here)
    # =====================================================================
    def survival_from(steps: list, gm12s: list, label: str) -> dict:
        prev = 0
        for s, g in zip(steps, gm12s):
            if s > 0 and g <= SHUT_BAR:
                return {"t_star": int(s), "bracket": f"({prev}, {s}]",
                        "gm12_at_t": float(g)}
            if s > 0:
                prev = int(s)
        return {"t_star": None, "bracket": f"(+{prev}, inf) — no crossing "
                f"by +{steps[-1]}", "gm12_at_t": None}

    surv = {
        "1e-3_neutral_e176n_armA": {
            "lr": 1e-3, **survival_from(E176N_ARMA["freeze_steps"],
                                        E176N_ARMA["gm12"], "armA"),
            "grid": E176N_ARMA["freeze_steps"][1:],
            "provenance": src_arma["source"], "stream": "neutral",
            "coarse": False},
        "1e-3_original_e176_smoke": {
            "lr": 1e-3, **survival_from(E176_SMOKE["freeze_steps"],
                                        E176_SMOKE["gm12"], "smoke"),
            "grid": E176_SMOKE["freeze_steps"][1:],
            "provenance": src_e176s["source"], "stream": "original",
            "coarse": False,
            "note": "the smoke run is the ONLY fine-step trajectory the "
                    "original stream has (e176n docstring's attribution); "
                    "the main run's coarse grid alone would say t*=50"},
        "1e-4_original_e176n_armB": {
            "lr": 1e-4, **survival_from(E176N_ARMB["freeze_steps"],
                                        E176N_ARMB["gm12"], "armB"),
            "grid": E176N_ARMB["freeze_steps"][1:],
            "provenance": src_armb["source"], "stream": "original",
            "coarse": True,
            "note": "COARSE grid {50,...}: t*=50 is an UPPER bound on the "
                    "true survival time"},
        "3e-5_neutral_e180": {
            "lr": 3e-5, **survival_from(
                [0] + [t["step"] for t in cells["lr3e5"]["traj"]],
                [root_cells["gm12"]] + [t["g_m12_mean_pz"]
                                        for t in cells["lr3e5"]["traj"]],
                "lr3e5"),
            "grid": list(CK_MAIN), "provenance": "this run",
            "stream": "neutral", "coarse": False,
            "device": cells["lr3e5"]["final_device"]},
        "1e-5_neutral_e180": {
            "lr": 1e-5, **survival_from(
                [0] + [t["step"] for t in cells["lr1e5"]["traj"]],
                [root_cells["gm12"]] + [t["g_m12_mean_pz"]
                                        for t in cells["lr1e5"]["traj"]],
                "lr1e5"),
            "grid": list(CK_MAIN), "provenance": "this run",
            "stream": "neutral", "coarse": False,
            "device": cells["lr1e5"]["final_device"]},
    }
    for k, p in surv.items():
        if p["t_star"] is None:
            p["censored"] = True
            p["t_fit"] = CK_MAIN[-1]          # 300 as the lower-bound subst.
        else:
            p["censored"] = False
        p["lr_times_t_star"] = (p["lr"] * p["t_star"]
                                if p["t_star"] is not None
                                else f"> {p['lr'] * CK_MAIN[-1]:.3g}")

    five = [surv[k] for k in ("1e-3_neutral_e176n_armA",
                              "1e-3_original_e176_smoke",
                              "1e-4_original_e176n_armB",
                              "3e-5_neutral_e180", "1e-5_neutral_e180")]
    neutral_only = [surv[k] for k in ("1e-3_neutral_e176n_armA",
                                      "3e-5_neutral_e180",
                                      "1e-5_neutral_e180")]
    fit_five = loglog_fit(five)
    fit_neutral = loglog_fit(neutral_only)
    any_censored = any(p["censored"] for p in five)
    fit_complete = (loglog_fit([p for p in five if not p["censored"]])
                    if any_censored else None)

    curve = {
        "bar": SHUT_BAR,
        "points": surv,
        "five_point_order": ["1e-3_neutral_e176n_armA",
                             "1e-3_original_e176_smoke",
                             "1e-4_original_e176n_armB",
                             "3e-5_neutral_e180", "1e-5_neutral_e180"],
        "fit_five_point": fit_five,
        "fit_neutral_only": fit_neutral,
        "fit_complete_cases": fit_complete,
        "censoring": ("the 1e-5 cell never crossed by +300: fitted twice — "
                      "t*:=300 (marked CENSORED) and complete-cases-only"
                      if any_censored else "none (all five cells crossed)"),
        "displacement_products_note": (
            "lr x t* per point (constant product <=> alpha ~ 1 <=> the wash "
            "is DISPLACEMENT-limited — T112's basin-width reading priced as "
            "a law; T112's measured anchors: 2@1e-3 ~ 2.5e-3, 50@1e-4 ~ 5e-3"),
    }

    # =====================================================================
    # ADJUDICATION (registered clauses; no shopping)
    # =====================================================================
    t_3e5 = surv["3e-5_neutral_e180"]["t_star"]
    t_1e5 = surv["1e-5_neutral_e180"]["t_star"]
    t_1e4 = surv["1e-4_original_e176n_armB"]["t_star"]
    t_1e3 = surv["1e-3_neutral_e176n_armA"]["t_star"]

    def val(p):
        return p["t_fit"] if p["censored"] else p["t_star"]
    monotone = bool(val(surv["1e-3_neutral_e176n_armA"]) <= t_1e4
                    <= val(surv["3e-5_neutral_e180"])
                    <= val(surv["1e-5_neutral_e180"]))
    late_or_none_1e5 = bool(t_1e5 is None or t_1e5 > 50)
    under50_1e5 = bool(t_1e5 is not None and t_1e5 <= 50)

    lr_immune = under50_1e5
    lr_scaled = late_or_none_1e5 and monotone
    if lr_immune:
        verdict = "LR-IMMUNE"
    elif lr_scaled:
        verdict = "LR-SCALED"
    else:
        verdict = "TEXTURE"

    adjudication = {
        "bar": SHUT_BAR,
        "survival_times": {"1e-3 (both stored cells)": t_1e3,
                           "1e-4 (stored, coarse upper bound)": t_1e4,
                           "3e-5 (new)": t_3e5, "1e-5 (new)": t_1e5},
        "monotone_growth_as_lr_shrinks": monotone,
        "first_under_1e5": t_1e5,
        "clause_1e5_late_or_none": late_or_none_1e5,
        "clause_1e5_under_by_50": under50_1e5,
        "LR_IMMUNE_fires": lr_immune,
        "LR_SCALED_fires": lr_scaled,
        "verdict": verdict,
        "near_bar_flags": near_bar_flags,
        "order": "LR-IMMUNE -> LR-SCALED -> TEXTURE (clauses disjoint by "
                 "construction on the 1e-5 first-under; no bar shopping)",
    }
    if verdict == "LR-IMMUNE":
        adjudication["clause_text"] = (
            f"the 1e-5 cell crossed the {SHUT_BAR} bar at +{t_1e5} (<= +50): "
            "intrinsic fragility — the two-step death is lr-invariant in "
            "OUTCOME; even a 100x smaller step than the original clock's "
            "kills within the same horizon.")
    elif verdict == "LR-SCALED":
        cens = "" if t_1e5 is not None else " (never by +300 — censored)"
        adjudication["clause_text"] = (
            f"survival grows monotonically as lr shrinks (2 -> 50 -> "
            f"{t_3e5 if t_3e5 is not None else '>300'} -> "
            f"{t_1e5 if t_1e5 is not None else '>300'}{cens}) and the 1e-5 "
            "cell is late-or-never: the wash is optimization-rate-limited "
            "(activity-dependence as a RATE law); the basin-width reading "
            "holds; the paper's 'continued training' gets its rate "
            "qualifier.")
    else:
        adjudication["clause_text"] = (
            "neither clause fired: partial survival / non-monotone timing — "
            "TEXTURE with the curve reported in full.")

    # =====================================================================
    # PLOT (the survival-vs-lr curve, log-log; trajectories co-panel)
    # =====================================================================
    series = [
        (E176N_ARMA["freeze_steps"], E176N_ARMA["gm12"],
         "1e-3 neutral (e176n A)", "tab:red", "o", 1.0),
        (E176_SMOKE["freeze_steps"], E176_SMOKE["gm12"],
         "1e-3 original (e176 smoke)", "tab:orange", "o", 1.0),
        (E176_MAIN["freeze_steps"], E176_MAIN["gm12"],
         "1e-3 original (e176 main)", "tab:orange", "o", 0.35),
        (E176N_ARMB["freeze_steps"], E176N_ARMB["gm12"],
         "1e-4 original (e176n B)", "tab:purple", "s", 1.0),
    ]
    new_cells = [
        ((f"{cells['lr3e5']['lr']:g} neutral (e180 NEW)", cells["lr3e5"]["traj"]),
         "tab:green", "^"),
        ((f"{cells['lr1e5']['lr']:g} neutral (e180 NEW)", cells["lr1e5"]["traj"]),
         "tab:blue", "D"),
    ]
    png = make_plot(rd, five, fit_five, any_censored, series, new_cells,
                    ROOT_GM12, verdict, CK_MAIN[-1])
    log(f"[plot] saved {png}")

    # =====================================================================
    # METRICS
    # =====================================================================
    metrics = {
        "experiment": "e180_wash_rate",
        "date": common.now_iso(),
        "registration": ("QUEUE row e180 dispatched ~21:20Z (commit 24de6e1, "
                         "GPU); the bars were registered in the dispatch "
                         "mission text and are frozen VERBATIM in the module "
                         "docstring before compute; adjudicated against "
                         "exactly that — no bar shopping"),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("does survival time to under-bar grow as lr shrinks "
                     "(wash = optimization-rate-limited; the basin-width "
                     "reading priced as a law) or does the two-step death "
                     "persist even at 1e-5 (intrinsic fragility)?"),
        "root": f"runs/checkpoints/{ROOT_CK} "
                f"(gated vs e151 before-cells, max|diff| "
                f"{G_ROOT['max_abs_diff']:.2e})",
        "root_meta": root_meta,
        "cells": cells,
        "protocol": {
            "corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
            "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
            "geometries_measured": {"novel": [-12, 12], "trained_g0": 0},
            "stream": ("e176n arm A's NEUTRAL protocol VERBATIM: e170's "
                       "neutral bank (seed 170), batch 32 = 16 neutral "
                       "anchors + 16 random corpus windows, full-token CE, "
                       "AdamW (0.9,0.95) wd 0.1 constant lr clip 1.0, seed "
                       "10902; ONLY the lr differs across cells"),
            "measure_dial": "e176n's measure() (minus the 183-span census — "
                            "e178's recorded deviation)",
        },
        "gates": {
            "G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
            "G_POOL": G_POOL, "G_ANCHOR": G_ANCHOR, "G_ROOT": G_ROOT,
            "G_DRAWFREE": {k: {"zeph_violations": cells[k]["zeph_violations"],
                               "pass": cells[k]["zeph_violations"] == 0}
                           for k in cells},
            "G_SURG": gates_surg,
        },
        "stored_refs": {"e176n_armA": src_arma, "e176n_armB": src_armb,
                        "e176_main": src_e176, "e176_smoke": src_e176s},
        "survival_curve": curve,
        "adjudication": adjudication,
        "compute": {
            "train_device_policy": ("GPU allowed (park-once, quick check + "
                                    "mid-run guard); cooldown "
                                    f"{COOLDOWN_S:.0f}s around each training; "
                                    f"per-training cap {TRAIN_CAP_S:.0f}s"),
            "gpu_parked": GPU_PARKED, "park_reason": PARK_REASON,
            "device_events": device_events,
            "torch_threads": torch.get_num_threads(),
            "cells_devices": {k: {"initial": cells[k]["initial_device"],
                                  "final": cells[k]["final_device"]}
                              for k in cells},
        },
        "trims": trims,
        "deviations": deviations,
        "ckpt_inventory": CKPT_INVENTORY,
        "elapsed_s": round(time.time() - T0, 1),
    }
    save_json(rd / "metrics.json", metrics)
    log("=" * 78)
    log(f"VERDICT: {verdict} — survival times: 1e-3 -> {t_1e3}, 1e-4 -> "
        f"{t_1e4} (coarse), 3e-5 -> {t_3e5}, 1e-5 -> {t_1e5}; "
        f"monotone={monotone}; five-point alpha="
        f"{fit_five.get('slope_alpha')}, R2={fit_five.get('r2')}")
    log(f"done -> {rd / 'metrics.json'}  ({time.time() - T0:.0f}s total)")


if __name__ == "__main__":
    main()
